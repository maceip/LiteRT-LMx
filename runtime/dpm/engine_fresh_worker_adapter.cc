// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/dpm/engine_fresh_worker_adapter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_agent_replay_runtime.h"
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_projection_replay_runtime.h"
#include "runtime/dpm/dpm_replay_executor.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/engine_fresh_worker_contract.h"
#include "runtime/dpm/fresh_worker_process.h"
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/exact_litert_decode.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

struct DecodedWorkerInvocation {
  DPMCanonicalReplayRequest replay_request;
  std::optional<DPMProjectionExecutionRequest> projection;
  std::optional<DPMAgentExecutionRequest> agent;
  std::optional<DPMAgentDeltaExecutionRequest> agent_delta;
  uint32_t max_output_tokens = 0;
};

struct CanonicalExactEvidence {
  std::string token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> logit_frames;
};

bool IsValidUtf8(absl::string_view text) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t lead = bytes[index++];
    if (lead <= 0x7f) continue;

    size_t continuation_count = 0;
    uint8_t minimum_second = 0x80;
    uint8_t maximum_second = 0xbf;
    if (lead >= 0xc2 && lead <= 0xdf) {
      continuation_count = 1;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      continuation_count = 2;
      if (lead == 0xe0) minimum_second = 0xa0;
      if (lead == 0xed) maximum_second = 0x9f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      continuation_count = 3;
      if (lead == 0xf0) minimum_second = 0x90;
      if (lead == 0xf4) maximum_second = 0x8f;
    } else {
      return false;
    }
    if (continuation_count > text.size() - index ||
        bytes[index] < minimum_second || bytes[index] > maximum_second) {
      return false;
    }
    ++index;
    for (size_t offset = 1; offset < continuation_count; ++offset, ++index) {
      if (bytes[index] < 0x80 || bytes[index] > 0xbf) return false;
    }
  }
  return true;
}

absl::Status ValidateFixedEngineSettings(const EngineSettings& settings) {
  if (settings.IsBenchmarkEnabled() ||
      settings.GetBenchmarkParams().has_value()) {
    return absl::UnimplementedError(
        "Exact fresh workers do not admit benchmark-controlled execution.");
  }
  if (settings.GetVisionExecutorSettings().has_value() ||
      settings.GetAudioExecutorSettings().has_value()) {
    return absl::UnimplementedError(
        "Exact fresh workers require a fixed text-only Engine.");
  }
  const LlmExecutorSettings& executor =
      settings.GetMainExecutorSettings();
  if (executor.GetBackend() != Backend::CPU &&
      executor.GetBackend() != Backend::GPU) {
    return absl::UnimplementedError(
        "Exact fresh workers admit only the compiled LiteRT CPU profile or a "
        "fully evidenced compiled LiteRT Metal profile.");
  }
  if (executor.GetSamplerBackend() != Backend::UNSPECIFIED &&
      executor.GetSamplerBackend() != Backend::CPU) {
    return absl::UnimplementedError(
        "Exact fresh workers require the CPU sampler backend.");
  }
  if (executor.GetLoraRank() != 0) {
    return absl::UnimplementedError(
        "Exact fresh workers do not admit LoRA-enabled executors.");
  }
  if (executor.GetModelAssets().fake_weights_mode() !=
      FakeWeightsMode::FAKE_WEIGHTS_NONE) {
    return absl::UnimplementedError(
        "Exact fresh workers do not admit fake-weight execution.");
  }
  if (executor.GetAdvancedSettings().has_value()) {
    const AdvancedSettings& advanced = *executor.GetAdvancedSettings();
    if (advanced.enable_speculative_decoding) {
      return absl::UnimplementedError(
          "Exact fresh workers do not admit MTP or speculative decode.");
    }
    if (advanced.is_benchmark) {
      return absl::UnimplementedError(
          "Exact fresh workers do not admit executor benchmark paths.");
    }
    if (advanced.enable_profiling ||
        advanced.num_logits_to_print_after_decode != 0) {
      return absl::UnimplementedError(
          "Exact fresh workers do not admit profiling or logits debug "
          "callbacks.");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateResolvedSessionConfig(
    const SessionConfig& config, DPMReplayStage stage,
    uint32_t max_output_tokens) {
  const SessionConfig::MemoryStrategy expected_memory_strategy =
      stage == DPMReplayStage::kProjection
          ? SessionConfig::MemoryStrategy::kStatelessDeterministicProjection
          : SessionConfig::MemoryStrategy::kStateful;
  if (config.GetMemoryStrategy() != expected_memory_strategy ||
      config.GetMaxOutputTokens() !=
          static_cast<int>(max_output_tokens) ||
      config.GetNumOutputCandidates() != 1 ||
      config.GetSamplerBackend() != Backend::CPU ||
      config.GetApplyPromptTemplateInSession() ||
      config.UseExternalSampler()) {
    return absl::FailedPreconditionError(
        "Resolved exact-worker session differs from the fixed stage "
        "contract.");
  }
  if (config.AudioModalityEnabled() || config.VisionModalityEnabled() ||
      config.GetAudioEmbeddingsCallback() != nullptr ||
      config.GetScopedLoraFile() != nullptr ||
      config.GetAudioScopedLoraFile() != nullptr) {
    return absl::UnimplementedError(
        "Resolved exact-worker session contains unsupported modality, "
        "embedding callback, or LoRA state.");
  }
  if (config.GetSuppressTokensConfig().enabled()) {
    return absl::UnimplementedError(
        "Exact fresh workers do not admit token suppression or logits "
        "modification.");
  }
  const proto::SamplerParameters& sampler = config.GetSamplerParams();
  if (sampler.type() != proto::SamplerParameters::GREEDY ||
      sampler.backend() != proto::SamplerParameters::CPU ||
      sampler.k() != 0 || sampler.p() != 0.0f ||
      sampler.temperature() != 0.0f || sampler.has_seed()) {
    return absl::FailedPreconditionError(
        "Resolved exact-worker sampler is not canonical CPU GREEDY with "
        "stable min-index ties.");
  }
  return absl::OkStatus();
}

absl::StatusOr<DecodedWorkerInvocation> DecodeCanonicalInvocation(
    const FreshWorkerRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(request));
  ABSL_ASSIGN_OR_RETURN(
      DPMCanonicalReplayRequest replay_request,
      DecodeDPMCanonicalReplayRequest(request.request_payload));
  ABSL_ASSIGN_OR_RETURN(const std::string canonical_replay_request,
                        EncodeDPMCanonicalReplayRequest(replay_request));
  if (canonical_replay_request != request.request_payload) {
    return absl::DataLossError(
        "Fresh worker request does not contain a canonical replay "
        "envelope.");
  }

  DecodedWorkerInvocation invocation;
  invocation.replay_request = replay_request;
  switch (replay_request.stage) {
    case DPMReplayStage::kProjection: {
      if (replay_request.request_contract_version !=
          kDPMProjectionReplayContractVersion) {
        return absl::FailedPreconditionError(
            "Fresh worker projection contract version is unsupported.");
      }
      ABSL_ASSIGN_OR_RETURN(
          DPMProjectionExecutionRequest projection,
          DecodeDPMProjectionExecutionRequest(
              replay_request.canonical_payload));
      ABSL_ASSIGN_OR_RETURN(
          const std::string canonical_projection,
          EncodeDPMProjectionExecutionRequest(projection));
      if (canonical_projection != replay_request.canonical_payload) {
        return absl::DataLossError(
            "Fresh worker projection request is not canonically encoded.");
      }
      if (!IsValidUtf8(projection.canonical_prompt)) {
        return absl::InvalidArgumentError(
            "Fresh worker projection prompt is not valid UTF-8.");
      }
      invocation.max_output_tokens =
          projection.projection_config.max_output_tokens;
      invocation.projection = std::move(projection);
      if (request.execution_plan.prefill_mode !=
              FreshWorkerPrefillMode::kFullCanonicalPrefill ||
          request.execution_plan.capture_producing_capsule) {
        return absl::UnimplementedError(
            "Exact projection workers do not restore or capture stateful "
            "session capsules.");
      }
      break;
    }

    case DPMReplayStage::kAgentDecision: {
      if (replay_request.request_contract_version !=
          kDPMAgentReplayContractVersion) {
        return absl::FailedPreconditionError(
            "Fresh worker agent-decision contract version is unsupported.");
      }
      ABSL_ASSIGN_OR_RETURN(
          DPMAgentExecutionRequest agent,
          DecodeDPMAgentExecutionRequest(replay_request.canonical_payload));
      ABSL_ASSIGN_OR_RETURN(const std::string canonical_agent,
                            EncodeDPMAgentExecutionRequest(agent));
      if (canonical_agent != replay_request.canonical_payload) {
        return absl::DataLossError(
            "Fresh worker agent request is not canonically encoded.");
      }
      invocation.max_output_tokens = agent.max_output_tokens;
      invocation.agent = std::move(agent);
      if (request.execution_plan.prefill_mode ==
          FreshWorkerPrefillMode::kOwnPositionCapsuleDelta) {
        ABSL_ASSIGN_OR_RETURN(
            DPMAgentDeltaExecutionRequest delta,
            DecodeDPMAgentDeltaExecutionRequest(
                request.execution_plan.canonical_execution_payload));
        ABSL_ASSIGN_OR_RETURN(
            const std::string canonical_delta,
            EncodeDPMAgentDeltaExecutionRequest(delta));
        if (canonical_delta !=
            request.execution_plan.canonical_execution_payload) {
          return absl::DataLossError(
              "Fresh worker agent delta is not canonically encoded.");
        }
        ABSL_RETURN_IF_ERROR(ValidateDPMAgentDeltaExecutionBinding(
            *invocation.agent, request.execution_plan, delta));
        invocation.agent_delta = std::move(delta);
      }
      break;
    }
  }
  if (invocation.max_output_tokens == 0 ||
      invocation.max_output_tokens > kMaximumDPMGenerationTokens ||
      replay_request.max_output_tokens != invocation.max_output_tokens ||
      invocation.projection.has_value() == invocation.agent.has_value()) {
    return absl::InvalidArgumentError(
        "Fresh worker stage request is incomplete or its authenticated "
        "outer token limit differs from the canonical stage payload.");
  }
  return invocation;
}

absl::Status ValidateFreshSession(
    const Engine& engine, const Engine::Session& session,
    const SessionConfig& resolved_config,
    const ExactLiteRtProfile& profile) {
  ABSL_RETURN_IF_ERROR(ValidateResolvedSessionConfig(
      session.GetSessionConfig(),
      resolved_config.GetMemoryStrategy() ==
              SessionConfig::MemoryStrategy::kStatelessDeterministicProjection
          ? DPMReplayStage::kProjection
          : DPMReplayStage::kAgentDecision,
      static_cast<uint32_t>(resolved_config.GetMaxOutputTokens())));
  ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity session_identity,
                        session.GetSessionHandoffIdentity());
  if (session_identity != profile.session_identity) {
    return absl::FailedPreconditionError(
        "Fresh exact session identity differs from the Engine-derived "
        "profile.");
  }
  ABSL_ASSIGN_OR_RETURN(const int current_step, session.GetCurrentStep());
  if (current_step != 0) {
    return absl::FailedPreconditionError(
        "Exact worker Engine did not create a step-zero session.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<std::vector<int>> token_history,
      session.GetExactProcessedTokenHistory());
  if (token_history.size() != 1 || !token_history.front().empty()) {
    return absl::FailedPreconditionError(
        "Exact worker Engine did not create an empty single-candidate "
        "session.");
  }
  const int vocabulary_size = engine.GetTokenizer().GetVocabSize();
  if (vocabulary_size <= 0 ||
      static_cast<uint32_t>(vocabulary_size) !=
          profile.logits_frame.vocabulary_size) {
    return absl::FailedPreconditionError(
        "Fresh exact session tokenizer vocabulary differs from the loaded "
        "logits contract.");
  }
  return absl::OkStatus();
}

absl::Status PrefillProjection(
    Engine::Session* session,
    const DPMProjectionExecutionRequest& projection) {
  if (session == nullptr || projection.canonical_prompt.empty() ||
      !IsValidUtf8(projection.canonical_prompt)) {
    return absl::InvalidArgumentError(
        "Exact projection requires a live session and canonical UTF-8 "
        "prompt.");
  }
  std::vector<InputData> contents;
  contents.emplace_back(InputText(std::string(projection.canonical_prompt)));
  return session->RunPrefill(contents);
}

absl::Status PrefillAgent(Engine* engine, Engine::Session* session,
                          const std::vector<
                              DPMAgentGenerationRequest::PrefillChunk>&
                              canonical_chunks) {
  if (engine == nullptr || session == nullptr || canonical_chunks.empty()) {
    return absl::InvalidArgumentError(
        "Exact agent execution requires one Engine session and canonical "
        "prefill input.");
  }
  const int vocabulary_size = engine->GetTokenizer().GetVocabSize();
  if (vocabulary_size <= 0) {
    return absl::FailedPreconditionError(
        "Exact agent worker tokenizer has no measurable vocabulary.");
  }
  for (const DPMAgentGenerationRequest::PrefillChunk& chunk :
       canonical_chunks) {
    std::vector<InputData> contents;
    switch (chunk.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text:
        if (chunk.text.empty() || !chunk.token_ids.empty() ||
            !IsValidUtf8(chunk.text)) {
          return absl::InvalidArgumentError(
              "Exact agent text chunk is empty, mixed, or not UTF-8.");
        }
        contents.emplace_back(InputText(std::string(chunk.text)));
        break;

      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds: {
        if (!chunk.text.empty() || chunk.token_ids.empty()) {
          return absl::InvalidArgumentError(
              "Exact agent token chunk is empty or mixed with text.");
        }
        for (int token_id : chunk.token_ids) {
          if (token_id < 0 || token_id >= vocabulary_size) {
            return absl::InvalidArgumentError(
                "Exact agent token chunk contains an out-of-range ID.");
          }
        }
        ABSL_ASSIGN_OR_RETURN(
            auto token_buffer,
            support::Tokenizer::TokenIdsToTensorBuffer(chunk.token_ids));
        contents.emplace_back(InputText(std::move(token_buffer)));
        break;
      }

      default:
        return absl::InvalidArgumentError(
            "Exact agent request contains an unknown chunk encoding.");
    }
    // Keeping one prefill call per canonical chunk preserves the exact
    // text/token boundary and prevents historical decision IDs from being
    // re-tokenized. The same helper is used for complete logical input and
    // for the validated post-checkpoint suffix.
    ABSL_RETURN_IF_ERROR(session->RunPrefill(contents));
  }
  return absl::OkStatus();
}

absl::Status ValidateImportedSession(
    const Engine::Session& session, const SessionConfig& resolved_config,
    const ExactLiteRtProfile& profile) {
  ABSL_RETURN_IF_ERROR(ValidateResolvedSessionConfig(
      session.GetSessionConfig(), DPMReplayStage::kAgentDecision,
      static_cast<uint32_t>(resolved_config.GetMaxOutputTokens())));
  ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity session_identity,
                        session.GetSessionHandoffIdentity());
  if (session_identity != profile.session_identity) {
    return absl::FailedPreconditionError(
        "Imported exact session identity differs from the worker-derived "
        "profile.");
  }
  ABSL_ASSIGN_OR_RETURN(const int current_step, session.GetCurrentStep());
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<std::vector<int>> token_history,
      session.GetExactProcessedTokenHistory());
  if (current_step <= 0 || token_history.size() != 1 ||
      token_history.front().empty()) {
    return absl::FailedPreconditionError(
        "Imported own-position session has no producing continuation "
        "boundary.");
  }
  return absl::OkStatus();
}

absl::StatusOr<FreshWorkerLogitElementType> ConvertElementType(
    ExactLiteRtLogitsElementType element_type) {
  switch (element_type) {
    case ExactLiteRtLogitsElementType::kFloat16:
      return FreshWorkerLogitElementType::kFloat16;
    case ExactLiteRtLogitsElementType::kFloat32:
      return FreshWorkerLogitElementType::kFloat32;
    case ExactLiteRtLogitsElementType::kUnsupported:
      break;
  }
  return absl::UnimplementedError(
      "Exact worker produced an unsupported logits element type.");
}

absl::StatusOr<CanonicalExactEvidence> ConvertExactEvidence(
    const ExactLiteRtDecodeEvidence& evidence,
    const ExactLiteRtProfile& profile, uint32_t max_output_tokens) {
  if (evidence.logits_frame_contract != profile.logits_frame ||
      evidence.sampled_token_ids.empty() ||
      evidence.sampled_token_ids.size() > max_output_tokens ||
      evidence.sampled_token_ids.size() != evidence.logits_frames.size()) {
    return absl::DataLossError(
        "Exact decode evidence differs from the Engine-derived logits "
        "contract or token limit.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const FreshWorkerLogitElementType element_type,
      ConvertElementType(evidence.logits_frame_contract.element_type));
  const uint32_t element_byte_width =
      element_type == FreshWorkerLogitElementType::kFloat16 ? 2 : 4;

  CanonicalExactEvidence canonical;
  canonical.logit_frames.reserve(evidence.logits_frames.size());
  for (size_t index = 0; index < evidence.logits_frames.size(); ++index) {
    const ExactLiteRtLogitsFrameEvidence& frame =
        evidence.logits_frames[index];
    if (frame.frame_index != index ||
        frame.contract != evidence.logits_frame_contract ||
        frame.sampled_token_id != evidence.sampled_token_ids[index] ||
        frame.sampled_token_id < 0 ||
        static_cast<uint32_t>(frame.sampled_token_id) >=
            evidence.logits_frame_contract.vocabulary_size) {
      return absl::DataLossError(
          "Exact decode logits frames are not ordered and paired with every "
          "sampled token.");
    }
    FreshWorkerLogitFrameEvidence worker_frame{
        .element_type = element_type,
        .element_byte_width = element_byte_width,
        .batch_size = evidence.logits_frame_contract.batch_size,
        .sequence_size = evidence.logits_frame_contract.sequence_size,
        .vocabulary_size = evidence.logits_frame_contract.vocabulary_size,
        .byte_count = evidence.logits_frame_contract.byte_count,
        .sha256 = frame.sha256,
    };
    ABSL_RETURN_IF_ERROR(
        ValidateFreshWorkerLogitFrameEvidence(worker_frame));
    canonical.logit_frames.push_back(worker_frame);
  }
  ABSL_ASSIGN_OR_RETURN(
      canonical.token_bytes,
      EncodeFreshWorkerTokenIds(evidence.sampled_token_ids));
  return canonical;
}

absl::Status ValidateExactResponses(const Responses& responses) {
  if (responses.GetTaskState() != TaskState::kDone &&
      responses.GetTaskState() != TaskState::kMaxNumTokensReached) {
    return absl::InternalError(
        "Exact worker decode returned a non-success terminal state.");
  }
  if (responses.GetTexts().size() != 1 ||
      !IsValidUtf8(responses.GetTexts().front())) {
    return absl::DataLossError(
        "Exact worker decode did not return one UTF-8 visible candidate.");
  }
  return absl::OkStatus();
}

absl::StatusOr<FreshWorkerExecutionOutput> BuildStageOutput(
    const DecodedWorkerInvocation& invocation,
    const ExactLiteRtDecodeResult& decoded,
    const ExactLiteRtProfile& profile) {
  ABSL_RETURN_IF_ERROR(ValidateExactResponses(decoded.responses));
  ABSL_ASSIGN_OR_RETURN(
      CanonicalExactEvidence evidence,
      ConvertExactEvidence(decoded.evidence, profile,
                           invocation.max_output_tokens));

  FreshWorkerExecutionOutput output;
  output.token_bytes = evidence.token_bytes;
  output.logit_frames = std::move(evidence.logit_frames);
  switch (invocation.replay_request.stage) {
    case DPMReplayStage::kProjection:
      if (!invocation.projection.has_value() ||
          invocation.agent.has_value()) {
        return absl::InternalError(
            "Decoded exact projection invocation lost its stage payload.");
      }
      // Exactly one model call has already completed. Invalid JSON is a hard
      // failure: there is no extraction, correction prompt, or repair call.
      ABSL_ASSIGN_OR_RETURN(
          output.canonical_output,
          CanonicalizeDPMProjectionOutput(
              decoded.responses.GetTexts().front(),
              invocation.projection->source_event_count,
              invocation.projection->projection_config));
      break;

    case DPMReplayStage::kAgentDecision: {
      if (!invocation.agent.has_value() ||
          invocation.projection.has_value()) {
        return absl::InternalError(
            "Decoded exact agent invocation lost its stage payload.");
      }
      // Visible text is allowed to be empty when the first sampled token is a
      // stop/EOS token. DPMTOK01 still records that sampled token and the
      // corresponding full logits frame; re-tokenizing visible text is never
      // used as evidence.
      DPMAgentDecisionEnvelope envelope{
          .decision_output = decoded.responses.GetTexts().front(),
          .canonical_token_bytes = output.token_bytes,
      };
      ABSL_ASSIGN_OR_RETURN(output.canonical_output,
                            EncodeDPMAgentDecisionEnvelope(envelope));
      break;
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerExecutionOutput(output));
  return output;
}

absl::StatusOr<FreshWorkerDerivedExecution> ExecuteEngineFreshWorkerRequest(
    EngineSettings fixed_engine_settings,
    const FreshWorkerExecutionContext& context) {
  const FreshWorkerRequest& request = context.request;
  // RunFreshWorkerOnce authenticates the envelope before reaching here. Do
  // all canonical parsing before model load so malformed stage input cannot
  // select artifacts or allocate an inference runtime.
  ABSL_ASSIGN_OR_RETURN(const DecodedWorkerInvocation invocation,
                        DecodeCanonicalInvocation(request));
  const bool restore_requested =
      request.execution_plan.prefill_mode ==
      FreshWorkerPrefillMode::kOwnPositionCapsuleDelta;
  const bool capture_requested =
      request.execution_plan.capture_producing_capsule;
  if (restore_requested != invocation.agent_delta.has_value() ||
      restore_requested != (context.restore_source != nullptr) ||
      restore_requested != (context.restore_options != nullptr) ||
      capture_requested != (context.producing_sink != nullptr) ||
      capture_requested != (context.producing_options != nullptr)) {
    return absl::FailedPreconditionError(
        "Exact Engine worker context disagrees with its authenticated "
        "execution plan.");
  }
  if ((restore_requested || capture_requested) &&
      invocation.replay_request.stage != DPMReplayStage::kAgentDecision) {
    return absl::UnimplementedError(
        "Only exact agent-decision workers admit session capsules.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFixedEngineSettings(fixed_engine_settings));

  // This explicit factory kind forbids fallback to legacy/TFLite engines. The
  // request cannot supply a factory kind, Engine, catalog, callback, backend,
  // model path, or input-hint-dependent setting.
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Engine> engine,
      EngineFactory::Create(
          EngineFactory::EngineType::kAdvancedLiteRTCompiledModel,
          std::move(fixed_engine_settings), /*input_prompt_as_hint=*/""));
  if (engine == nullptr) {
    return absl::InternalError(
        "Advanced LiteRT factory returned a null exact-worker Engine.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFixedEngineSettings(engine->GetEngineSettings()));

  ABSL_ASSIGN_OR_RETURN(
      SessionConfig session_config,
      MakeEngineFreshWorkerSessionConfig(
          invocation.replay_request.stage, invocation.max_output_tokens));
  ABSL_RETURN_IF_ERROR(
      session_config.MaybeUpdateAndValidate(engine->GetEngineSettings()));
  ABSL_RETURN_IF_ERROR(ValidateResolvedSessionConfig(
      session_config, invocation.replay_request.stage,
      invocation.max_output_tokens));

  ExactLiteRtProfileAssertion profile_assertion;
  profile_assertion.expected_profile_id = request.exact_profile_hash;
  ABSL_ASSIGN_OR_RETURN(
      const ExactLiteRtProfile profile_before,
      engine->ResolveExactLiteRtProfile(session_config, profile_assertion));
  if (profile_before.profile_id != request.exact_profile_hash) {
    return absl::FailedPreconditionError(
        "Worker-derived exact profile does not match the authenticated "
        "request.");
  }

  // Exactly one session is created. An own-position capsule is imported only
  // into this step-zero target; there is no parent Session and no worker-local
  // fallback to full prefill.
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
                        engine->CreateSession(session_config));
  if (session == nullptr) {
    return absl::InternalError(
        "Advanced LiteRT Engine returned a null exact-worker session.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFreshSession(*engine, *session, session_config,
                           profile_before));

  if (restore_requested) {
    ABSL_RETURN_IF_ERROR(session->ImportHandoffFrom(
        *context.restore_source, *context.restore_options));
    ABSL_RETURN_IF_ERROR(
        ValidateImportedSession(*session, session_config, profile_before));
    ABSL_ASSIGN_OR_RETURN(
        const ExactLiteRtProfile profile_after_import,
        engine->ResolveExactLiteRtProfile(session->GetSessionConfig(),
                                          profile_assertion));
    if (profile_after_import != profile_before) {
      return absl::FailedPreconditionError(
          "Worker-derived ExactLiteRtProfile changed during capsule "
          "import.");
    }
  }

  switch (invocation.replay_request.stage) {
    case DPMReplayStage::kProjection:
      ABSL_RETURN_IF_ERROR(
          PrefillProjection(session.get(), *invocation.projection));
      break;
    case DPMReplayStage::kAgentDecision:
      ABSL_RETURN_IF_ERROR(PrefillAgent(
          engine.get(), session.get(),
          restore_requested
              ? invocation.agent_delta->canonical_delta_prefill_chunks
              : invocation.agent->full_canonical_prefill_chunks));
      break;
  }

  // This is the only decode call in the adapter. The API captures each full
  // logits tensor before sampling and records the sampled stop/EOS ID before
  // ordinary response filtering.
  ABSL_ASSIGN_OR_RETURN(
      const ExactLiteRtDecodeResult exact_decode,
      session->RunExactDecode(static_cast<int>(
          invocation.max_output_tokens)));

  // Rehash/rederive after inference, while the producing session and loaded
  // Engine are still alive. A changed artifact, runtime profile, session
  // config, logits contract, or platform identity invalidates the result.
  ABSL_ASSIGN_OR_RETURN(
      const ExactLiteRtProfile profile_after,
      engine->ResolveExactLiteRtProfile(session->GetSessionConfig(),
                                        profile_assertion));
  if (profile_after != profile_before ||
      profile_after.profile_id != request.exact_profile_hash) {
    return absl::FailedPreconditionError(
        "Worker-derived ExactLiteRtProfile changed during execution.");
  }
  ABSL_RETURN_IF_ERROR(ValidateResolvedSessionConfig(
      session->GetSessionConfig(), invocation.replay_request.stage,
      invocation.max_output_tokens));
  ABSL_ASSIGN_OR_RETURN(
      const SessionHandoffIdentity producing_session_identity,
      session->GetSessionHandoffIdentity());
  if (producing_session_identity != profile_after.session_identity) {
    return absl::FailedPreconditionError(
        "Producing session identity changed or differs from the final "
        "Engine-derived exact profile.");
  }
  ABSL_ASSIGN_OR_RETURN(
      FreshWorkerExecutionOutput output,
      BuildStageOutput(invocation, exact_decode, profile_after));
  const Hash256 finalized_output_evidence_hash =
      ComputeFreshWorkerOutputEvidenceHash(
          output.canonical_output, output.token_bytes, output.logit_frames);

  // Canonical output, exact token bytes, and every ordered logits hash are
  // finalized before exporting the producing session. A failed export fails
  // the whole request and cannot turn these bytes into an uncaptured success.
  if (capture_requested) {
    ABSL_ASSIGN_OR_RETURN(const int post_decode_step,
                          session->GetCurrentStep());
    ABSL_ASSIGN_OR_RETURN(
        const std::vector<std::vector<int>> post_decode_history,
        session->GetExactProcessedTokenHistory());
    ABSL_RETURN_IF_ERROR(session->ExportHandoffTo(
        *context.producing_options, context.producing_sink));
    ABSL_ASSIGN_OR_RETURN(const int post_export_step,
                          session->GetCurrentStep());
    ABSL_ASSIGN_OR_RETURN(
        const std::vector<std::vector<int>> post_export_history,
        session->GetExactProcessedTokenHistory());
    if (post_export_step != post_decode_step ||
        post_export_history != post_decode_history ||
        ComputeFreshWorkerOutputEvidenceHash(
            output.canonical_output, output.token_bytes,
            output.logit_frames) != finalized_output_evidence_hash) {
      return absl::FailedPreconditionError(
          "Producing-session export changed exact continuation or output "
          "evidence.");
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      const ExactLiteRtProfile profile_final,
      engine->ResolveExactLiteRtProfile(session->GetSessionConfig(),
                                        profile_assertion));
  if (profile_final != profile_before) {
    return absl::FailedPreconditionError(
        "Worker-derived ExactLiteRtProfile changed during producing-session "
        "capture.");
  }
  ABSL_RETURN_IF_ERROR(ValidateResolvedSessionConfig(
      session->GetSessionConfig(), invocation.replay_request.stage,
      invocation.max_output_tokens));
  ABSL_ASSIGN_OR_RETURN(
      const SessionHandoffIdentity final_session_identity,
      session->GetSessionHandoffIdentity());
  if (final_session_identity != profile_final.session_identity ||
      final_session_identity != producing_session_identity) {
    return absl::FailedPreconditionError(
        "Producing session identity changed after exact output or capsule "
        "capture.");
  }

  return FreshWorkerDerivedExecution{
      .derived_exact_profile_hash = profile_final.profile_id,
      .replay_isolation = FreshWorkerReplayIsolation::kEmptyCatalogs,
      .output = std::move(output),
      .restored_checkpoint_id =
          request.execution_plan.restore_checkpoint_id,
      .exported_producing_capsule = capture_requested,
      .producing_session_identity = final_session_identity,
  };
}

}  // namespace

absl::Status RunEngineFreshWorkerOnce(
    EngineSettings fixed_engine_settings) {
  // RunFreshWorkerOnce is synchronous and invokes its callback exactly once.
  // Keep an independent guard here so this concrete adapter cannot silently
  // become a reusable in-process Engine service if the generic boundary ever
  // changes.
  bool consumed = false;
  return RunFreshWorkerOnce(
      [&fixed_engine_settings, &consumed](
          const FreshWorkerExecutionContext& context)
          -> absl::StatusOr<FreshWorkerDerivedExecution> {
        if (consumed) {
          return absl::FailedPreconditionError(
              "Concrete Engine fresh-worker adapter is one-shot.");
        }
        consumed = true;
        return ExecuteEngineFreshWorkerRequest(
            std::move(fixed_engine_settings), context);
      });
}

}  // namespace litert::lm
