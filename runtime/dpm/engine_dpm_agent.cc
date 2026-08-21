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

#include "runtime/dpm/engine_dpm_agent.h"

#include <algorithm>
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
#include "runtime/dpm/dpm_engine.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/byte_stream.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

absl::Status ValidateDPMEngineSettings(const EngineSettings& settings) {
#ifdef LITERT_LM_DEBUGGER_ENABLED
  return absl::UnimplementedError(
      "DPM agent generation does not support debugger-installed graph "
      "callbacks.");
#endif
  if (settings.IsBenchmarkEnabled() ||
      settings.GetVisionExecutorSettings().has_value() ||
      settings.GetAudioExecutorSettings().has_value()) {
    return absl::UnimplementedError(
        "DPM agent generation requires the ordinary text-only, "
        "non-benchmark Engine path.");
  }
  const LlmExecutorSettings& executor = settings.GetMainExecutorSettings();
  if (executor.GetLoraRank() != 0) {
    return absl::UnimplementedError(
        "DPM agent generation does not support a LoRA-enabled executor.");
  }
  if (executor.GetAdvancedSettings().has_value() &&
      executor.GetAdvancedSettings()->enable_speculative_decoding) {
    return absl::UnimplementedError(
        "DPM agent generation does not support MTP or speculative decode.");
  }
  if (executor.GetAdvancedSettings().has_value() &&
      (executor.GetAdvancedSettings()->is_benchmark ||
       executor.GetAdvancedSettings()->enable_profiling ||
       executor.GetAdvancedSettings()->num_logits_to_print_after_decode !=
           0)) {
    return absl::UnimplementedError(
        "DPM agent generation does not support benchmark, profiling, or "
        "logits debug execution paths.");
  }
  return absl::OkStatus();
}

absl::Status ValidateDPMConfig(const SessionConfig& config) {
  if (config.GetMemoryStrategy() !=
      SessionConfig::MemoryStrategy::kStateful) {
    return absl::FailedPreconditionError(
        "DPM agent sessions must remain stateful so capsule restore and delta "
        "prefill cannot be erased by projection reset policy.");
  }
  if (config.AudioModalityEnabled() || config.VisionModalityEnabled() ||
      config.GetAudioEmbeddingsCallback() != nullptr) {
    return absl::InvalidArgumentError(
        "DPM agent sessions must be text-only.");
  }
  if (config.GetScopedLoraFile() != nullptr ||
      config.GetAudioScopedLoraFile() != nullptr) {
    return absl::UnimplementedError(
        "DPM agent session handoff does not support LoRA state.");
  }
  if (config.UseExternalSampler()) {
    return absl::UnimplementedError(
        "DPM agent session handoff does not support an external sampler.");
  }
  if (config.GetNumOutputCandidates() != 1) {
    return absl::FailedPreconditionError(
        "DPM agent runtime did not resolve exactly one output candidate.");
  }
  if (config.GetSuppressTokensConfig().enabled()) {
    return absl::UnimplementedError(
        "DPM agent generation does not support inherited or requested token "
        "suppression.");
  }
  if (config.GetSamplerBackend() != Backend::CPU) {
    return absl::FailedPreconditionError(
        "DPM agent runtime did not resolve the CPU sampler backend.");
  }
  const proto::SamplerParameters& sampler = config.GetSamplerParams();
  if (sampler.type() != proto::SamplerParameters::GREEDY) {
    return absl::UnimplementedError(
        "DPM agent session handoff requires GREEDY sampling.");
  }
  if (sampler.backend() != proto::SamplerParameters::CPU) {
    return absl::FailedPreconditionError(
        "DPM agent runtime did not resolve the explicit CPU sampler "
        "profile.");
  }
  if (sampler.k() != 0 || sampler.p() != 0.0f ||
      sampler.temperature() != 0.0f || sampler.has_seed()) {
    return absl::FailedPreconditionError(
        "DPM agent runtime did not canonicalize GREEDY-irrelevant sampler "
        "fields.");
  }
  if (config.GetApplyPromptTemplateInSession()) {
    return absl::FailedPreconditionError(
        "DPM agent runtime did not disable session prompt templating.");
  }
  return absl::OkStatus();
}

absl::Status ValidateFreshExactHistory(
    const std::vector<std::vector<int>>& history) {
  if (history.size() != 1) {
    return absl::FailedPreconditionError(
        "DPM agent session does not expose exactly one token history.");
  }
  if (!history[0].empty()) {
    return absl::FailedPreconditionError(
        "DPM agent Engine created a session with non-fresh token history.");
  }
  return absl::OkStatus();
}

bool IsValidUtf8(const std::string& text) {
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

}  // namespace

absl::StatusOr<std::unique_ptr<EngineDPMAgentRuntime>>
EngineDPMAgentRuntime::Create(
    Engine* engine, SessionConfig session_config,
    std::optional<SessionHandoffIdentity> expected_identity) {
  return CreateInternal(engine, std::move(session_config), std::nullopt,
                        expected_identity);
}

absl::StatusOr<std::unique_ptr<EngineDPMAgentRuntime>>
EngineDPMAgentRuntime::Create(
    Engine* engine, SessionConfig session_config,
    CapsuleRestoreAdmissionBinding capsule_restore_admission,
    std::optional<SessionHandoffIdentity> expected_identity) {
  return CreateInternal(
      engine, std::move(session_config),
      std::optional<CapsuleRestoreAdmissionBinding>(
          std::move(capsule_restore_admission)),
      expected_identity);
}

absl::StatusOr<std::unique_ptr<EngineDPMAgentRuntime>>
EngineDPMAgentRuntime::CreateInternal(
    Engine* engine, SessionConfig session_config,
    std::optional<CapsuleRestoreAdmissionBinding>
        capsule_restore_admission,
    std::optional<SessionHandoffIdentity> expected_identity) {
  if (engine == nullptr) {
    return absl::InvalidArgumentError(
        "DPM agent runtime requires a loaded Engine.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateDPMEngineSettings(engine->GetEngineSettings()));
  if (session_config.GetMemoryStrategy() ==
      SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
    return absl::InvalidArgumentError(
        "DPM agent runtime rejects a stateless projection memory strategy; "
        "agent sessions require stateful capsule restore and delta prefill.");
  }
  if (session_config.GetMemoryStrategy() !=
      SessionConfig::MemoryStrategy::kStateful) {
    return absl::InvalidArgumentError(
        "DPM agent runtime received an unknown session memory strategy.");
  }
  if (session_config.AudioModalityEnabled() ||
      session_config.VisionModalityEnabled() ||
      session_config.GetAudioEmbeddingsCallback() != nullptr) {
    return absl::InvalidArgumentError(
        "DPM agent runtime cannot discard requested non-text modalities.");
  }
  if (session_config.GetScopedLoraFile() != nullptr ||
      session_config.GetAudioScopedLoraFile() != nullptr) {
    return absl::UnimplementedError(
        "DPM agent runtime cannot discard requested LoRA state.");
  }
  if (session_config.UseExternalSampler()) {
    return absl::UnimplementedError(
        "DPM agent runtime cannot discard a requested external sampler.");
  }
  if (session_config.GetNumOutputCandidates() != 1) {
    return absl::InvalidArgumentError(
        "DPM agent runtime requires exactly one requested output candidate.");
  }
  if (session_config.GetSamplerBackend() != Backend::UNSPECIFIED &&
      session_config.GetSamplerBackend() != Backend::CPU) {
    return absl::InvalidArgumentError(
        "DPM agent runtime cannot discard a requested non-CPU sampler "
        "backend.");
  }

  const proto::SamplerParameters& requested_sampler =
      session_config.GetSamplerParams();
  if (requested_sampler.type() !=
          proto::SamplerParameters::TYPE_UNSPECIFIED &&
      requested_sampler.type() != proto::SamplerParameters::GREEDY) {
    return absl::InvalidArgumentError(
        "DPM agent runtime cannot discard a requested non-GREEDY sampler.");
  }
  if (requested_sampler.backend() !=
          proto::SamplerParameters::UNSPECIFIED &&
      requested_sampler.backend() != proto::SamplerParameters::CPU) {
    return absl::InvalidArgumentError(
        "DPM agent runtime cannot discard a requested non-CPU proto sampler "
        "backend.");
  }

  // These are product invariants, not caller identity inputs. Resolve only
  // after pinning them so the Engine hashes the exact profile it will create.
  session_config.SetNumOutputCandidates(1);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.SetApplyPromptTemplateInSession(false);
  session_config.SetMemoryStrategy(SessionConfig::MemoryStrategy::kStateful);
  proto::SamplerParameters& sampler =
      session_config.GetMutableSamplerParams();
  sampler.set_type(proto::SamplerParameters::GREEDY);
  sampler.set_k(0);
  sampler.set_p(0.0f);
  sampler.set_temperature(0.0f);
  sampler.clear_seed();
  sampler.set_backend(proto::SamplerParameters::CPU);
  ABSL_RETURN_IF_ERROR(
      session_config.MaybeUpdateAndValidate(engine->GetEngineSettings()));
  ABSL_RETURN_IF_ERROR(ValidateDPMConfig(session_config));

  ABSL_ASSIGN_OR_RETURN(
      SessionHandoffIdentity authoritative_identity,
      engine->ResolveSessionHandoffIdentity(session_config));
  if (expected_identity.has_value() &&
      *expected_identity != authoritative_identity) {
    return absl::FailedPreconditionError(
        "DPM agent identity assertion does not match the loaded Engine.");
  }

  std::optional<ExactLiteRtProfile> capsule_restore_profile;
  std::optional<Hash256> capsule_restore_admission_record_id;
  std::optional<SessionHandoffCapability> session_handoff_capability;
  std::optional<CapsuleRestoreOperationalCoverage>
      capsule_restore_operational_coverage;
  if (capsule_restore_admission.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        AuthenticatedCapsuleRestoreAdmission admission,
        ResolveAuthenticatedCapsuleRestoreAdmission(
            engine, session_config, *capsule_restore_admission));
    if (admission.profile.session_identity != authoritative_identity ||
        admission.capability.session_identity != authoritative_identity ||
        admission.record.capability != admission.capability) {
      return absl::FailedPreconditionError(
          "CapsuleRestore admission does not match the live parent runtime's "
          "resolved session identity.");
    }
    capsule_restore_profile = std::move(admission.profile);
    capsule_restore_admission_record_id = admission.record.record_id;
    session_handoff_capability = std::move(admission.capability);
    capsule_restore_operational_coverage =
        std::move(admission.operational_coverage);
  }

  auto runtime = std::unique_ptr<EngineDPMAgentRuntime>(
      new EngineDPMAgentRuntime(
          engine, std::move(session_config), authoritative_identity,
          std::move(capsule_restore_admission),
          std::move(capsule_restore_profile),
          capsule_restore_admission_record_id,
          std::move(session_handoff_capability),
          std::move(capsule_restore_operational_coverage)));
  ABSL_RETURN_IF_ERROR(runtime->ValidateRuntimeSupport());
  if (runtime->capsule_restore_admission_.has_value()) {
    ABSL_RETURN_IF_ERROR(runtime->ValidateSessionHandoffSupport());
  }
  return runtime;
}

absl::Status EngineDPMAgentRuntime::ValidateRuntimeSupport() const {
  if (engine_ == nullptr) {
    return absl::FailedPreconditionError(
        "DPM agent runtime has lost its loaded Engine.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateDPMEngineSettings(engine_->GetEngineSettings()));
  ABSL_RETURN_IF_ERROR(ValidateDPMConfig(resolved_session_config_));
  ABSL_ASSIGN_OR_RETURN(
      SessionHandoffIdentity current_identity,
      engine_->ResolveSessionHandoffIdentity(resolved_session_config_));
  if (current_identity != session_handoff_identity_) {
    return absl::FailedPreconditionError(
        "Loaded Engine identity changed after DPM agent admission.");
  }
  const bool has_binding = capsule_restore_admission_.has_value();
  if (has_binding != capsule_restore_profile_.has_value() ||
      has_binding != capsule_restore_admission_record_id_.has_value() ||
      has_binding != session_handoff_capability_.has_value() ||
      has_binding != capsule_restore_operational_coverage_.has_value()) {
    return absl::InternalError(
        "DPM agent CapsuleRestore admission binding is internally "
        "inconsistent.");
  }
  if (has_binding) {
    ABSL_ASSIGN_OR_RETURN(
        const AuthenticatedCapsuleRestoreAdmission current,
        ResolveCurrentCapsuleRestoreAdmission());
    (void)current;
  }
  return absl::OkStatus();
}

absl::Status EngineDPMAgentRuntime::ValidateSupport() const {
  return ValidateRuntimeSupport();
}

absl::Status EngineDPMAgentRuntime::ValidateSessionHandoffSupport() const {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeSupport());

  // CapsuleRestore is a separate capability from generation and
  // CanonicalWinnerReplay. Exercise the authenticated fresh-state
  // export/import surface only when the product configuration enables
  // checkpoint restore or capture. This probe does not establish
  // own-position continuation equality; CapsuleRestore admission remains
  // responsible for that stronger claim.
  ABSL_RETURN_IF_ERROR(ProbeSessionHandoffSupport());
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission after,
      ResolveCurrentCapsuleRestoreAdmission());
  if (after.record.record_id !=
          *capsule_restore_admission_record_id_ ||
      after.profile != *capsule_restore_profile_ ||
      after.capability != *session_handoff_capability_ ||
      after.operational_coverage !=
          *capsule_restore_operational_coverage_) {
    return absl::AbortedError(
        "CapsuleRestore admission changed during live-runtime support "
        "validation.");
  }
  return absl::OkStatus();
}

absl::StatusOr<AuthenticatedCapsuleRestoreAdmission>
EngineDPMAgentRuntime::ResolveCurrentCapsuleRestoreAdmission() const {
  const bool has_binding = capsule_restore_admission_.has_value();
  if (has_binding != capsule_restore_profile_.has_value() ||
      has_binding != capsule_restore_admission_record_id_.has_value() ||
      has_binding != session_handoff_capability_.has_value() ||
      has_binding != capsule_restore_operational_coverage_.has_value()) {
    return absl::InternalError(
        "DPM agent CapsuleRestore authority is internally inconsistent.");
  }
  if (!has_binding) {
    return absl::FailedPreconditionError(
        "DPM agent runtime was constructed without CapsuleRestore "
        "admission.");
  }
  ABSL_ASSIGN_OR_RETURN(
      AuthenticatedCapsuleRestoreAdmission current,
      ResolveAuthenticatedCapsuleRestoreAdmission(
          engine_, resolved_session_config_, *capsule_restore_admission_));
  if (current.record.record_id !=
          *capsule_restore_admission_record_id_ ||
      current.profile != *capsule_restore_profile_ ||
      current.capability != *session_handoff_capability_ ||
      current.operational_coverage !=
          *capsule_restore_operational_coverage_ ||
      current.record.capability != current.capability ||
      current.capability.session_identity != session_handoff_identity_) {
    return absl::AbortedError(
        "Authenticated CapsuleRestore admission, profile, or capability "
        "changed after live-runtime construction.");
  }
  return current;
}

absl::StatusOr<std::optional<Hash256>>
EngineDPMAgentRuntime::GetCapsuleRestoreAdmissionRecordId() const {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeSupport());
  if (!capsule_restore_admission_.has_value()) {
    return std::optional<Hash256>();
  }
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  return std::optional<Hash256>(current.record.record_id);
}

absl::StatusOr<std::optional<Hash256>>
EngineDPMAgentRuntime::GetSessionHandoffCapabilityId() const {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeSupport());
  if (!capsule_restore_admission_.has_value()) {
    return std::optional<Hash256>();
  }
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  return std::optional<Hash256>(current.capability.capability_id);
}

absl::StatusOr<std::optional<SessionHandoffCapability>>
EngineDPMAgentRuntime::GetSessionHandoffCapability() const {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeSupport());
  if (!capsule_restore_admission_.has_value()) {
    return std::optional<SessionHandoffCapability>();
  }
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  return std::optional<SessionHandoffCapability>(current.capability);
}

absl::StatusOr<std::optional<CapsuleRestoreOperationalCoverage>>
EngineDPMAgentRuntime::GetCapsuleRestoreOperationalCoverage() const {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeSupport());
  if (!capsule_restore_admission_.has_value()) {
    return std::optional<CapsuleRestoreOperationalCoverage>();
  }
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreOperationalCoverage(
      current.operational_coverage));
  return std::optional<CapsuleRestoreOperationalCoverage>(
      current.operational_coverage);
}

absl::Status EngineDPMAgentRuntime::ValidateSession(
    const Engine::Session& session) const {
  ABSL_RETURN_IF_ERROR(ValidateDPMConfig(session.GetSessionConfig()));
  ABSL_ASSIGN_OR_RETURN(SessionHandoffIdentity session_identity,
                        session.GetSessionHandoffIdentity());
  if (session_identity != session_handoff_identity_) {
    return absl::FailedPreconditionError(
        "DPM agent session identity differs from its loaded runtime.");
  }
  return absl::OkStatus();
}

absl::Status EngineDPMAgentRuntime::ProbeSessionHandoffSupport() const {
  if (engine_ == nullptr) {
    return absl::FailedPreconditionError(
        "DPM agent runtime has lost its loaded Engine.");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Engine::Session> source,
      engine_->CreateSession(resolved_session_config_));
  if (source == nullptr) {
    return absl::InternalError(
        "DPM agent Engine returned a null handoff-probe source session.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSession(*source));

  // This key authenticates only an in-memory, fresh-state capability probe. It
  // is deliberately not configurable and is never used for a durable capsule.
  SessionHandoffOptions options;
  options.key_id = "litert-lm-dpm-probe-v1";
  options.authentication_key = "0123456789abcdef0123456789abcdef";
  options.expected_identity = session_handoff_identity_;

  std::string envelope;
  StringByteSink sink(&envelope);
  ABSL_RETURN_IF_ERROR(source->ExportHandoffTo(options, &sink));
  if (envelope.empty()) {
    return absl::DataLossError(
        "DPM agent handoff capability probe produced an empty envelope.");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Engine::Session> target,
      engine_->CreateSession(resolved_session_config_));
  if (target == nullptr) {
    return absl::InternalError(
        "DPM agent Engine returned a null handoff-probe target session.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSession(*target));
  StringByteSource source_bytes(envelope);
  ABSL_RETURN_IF_ERROR(target->ImportHandoffFrom(source_bytes, options));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::vector<int>> restored_history,
      target->GetExactProcessedTokenHistory());
  return ValidateFreshExactHistory(restored_history);
}

absl::StatusOr<std::unique_ptr<Engine::Session>>
EngineDPMAgentRuntime::CreateSession() {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeSupport());
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Engine::Session> session,
      engine_->CreateSession(resolved_session_config_));
  if (session == nullptr) {
    return absl::InternalError("DPM agent Engine returned a null session.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSession(*session));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::vector<int>> fresh_history,
      session->GetExactProcessedTokenHistory());
  ABSL_RETURN_IF_ERROR(ValidateFreshExactHistory(fresh_history));
  if (capsule_restore_admission_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        const AuthenticatedCapsuleRestoreAdmission current,
        ResolveCurrentCapsuleRestoreAdmission());
    (void)current;
  }
  return session;
}

absl::StatusOr<DPMAgentGenerationOutcome> EngineDPMAgentRuntime::Generate(
    Engine::Session* session, const DPMAgentGenerationRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeSupport());
  if (session == nullptr) {
    return absl::InvalidArgumentError(
        "DPM agent generation requires a live producing session.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSession(*session));
  if (request.max_output_tokens <= 0) {
    return absl::InvalidArgumentError(
        "DPM agent max output tokens must be positive.");
  }
  if (request.canonical_prefill_chunks.empty()) {
    return absl::InvalidArgumentError(
        "DPM agent generation requires canonical prefill chunks.");
  }

  const int vocabulary_size = engine_->GetTokenizer().GetVocabSize();
  if (vocabulary_size <= 0) {
    return absl::FailedPreconditionError(
        "DPM agent tokenizer has no measurable vocabulary.");
  }
  for (const DPMAgentGenerationRequest::PrefillChunk& chunk :
       request.canonical_prefill_chunks) {
    std::vector<InputData> contents;
    contents.reserve(1);
    switch (chunk.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text:
        if (!chunk.token_ids.empty()) {
          return absl::InvalidArgumentError(
              "UTF-8 DPM prefill chunk also contains token IDs.");
        }
        if (chunk.text.empty()) {
          return absl::InvalidArgumentError(
              "UTF-8 DPM prefill chunk must not be empty.");
        }
        if (!IsValidUtf8(chunk.text)) {
          return absl::InvalidArgumentError(
              "UTF-8 DPM prefill chunk contains invalid UTF-8 bytes.");
        }
        contents.emplace_back(InputText(std::string(chunk.text)));
        break;

      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds: {
        if (!chunk.text.empty()) {
          return absl::InvalidArgumentError(
              "Token-ID DPM prefill chunk also contains UTF-8 bytes.");
        }
        if (chunk.token_ids.empty()) {
          return absl::InvalidArgumentError(
              "Token-ID DPM prefill chunk must not be empty.");
        }
        for (int token_id : chunk.token_ids) {
          if (token_id < 0 || token_id >= vocabulary_size) {
            return absl::InvalidArgumentError(
                "Token-ID DPM prefill chunk contains an out-of-range ID.");
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
            "DPM prefill chunk has an unknown encoding.");
    }
    // A distinct call per chunk preserves the canonical input/decision
    // boundary and prevents historical decision IDs from being re-tokenized.
    ABSL_RETURN_IF_ERROR(session->RunPrefill(contents));
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::vector<int>> history_before,
      session->GetExactProcessedTokenHistory());
  if (history_before.size() != 1) {
    return absl::FailedPreconditionError(
        "DPM agent pre-decode history has multiple candidates.");
  }
  for (int token_id : history_before[0]) {
    if (token_id < 0 || token_id >= vocabulary_size) {
      return absl::DataLossError(
          "DPM agent pre-decode history contains an out-of-range token ID.");
    }
  }

  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  decode_config.SetMaxOutputTokens(request.max_output_tokens);
  ABSL_ASSIGN_OR_RETURN(Responses responses,
                        session->RunDecode(decode_config));
  if (responses.GetTaskState() != TaskState::kDone &&
      responses.GetTaskState() != TaskState::kMaxNumTokensReached) {
    return absl::InternalError(
        "DPM agent decode returned a non-success terminal state.");
  }
  if (responses.GetTexts().size() != 1) {
    return absl::DataLossError(
        "DPM agent decode did not return exactly one text candidate.");
  }
  if (!IsValidUtf8(responses.GetTexts()[0])) {
    return absl::DataLossError(
        "DPM agent decode returned invalid UTF-8 decision bytes.");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::vector<int>> history_after,
      session->GetExactProcessedTokenHistory());
  if (history_after.size() != 1 ||
      history_after[0].size() < history_before[0].size() ||
      !std::equal(history_before[0].begin(), history_before[0].end(),
                  history_after[0].begin())) {
    return absl::DataLossError(
        "DPM agent decode did not extend the exact executor token history.");
  }
  for (int token_id : history_after[0]) {
    if (token_id < 0 || token_id >= vocabulary_size) {
      return absl::DataLossError(
          "DPM agent post-decode history contains an out-of-range token ID.");
    }
  }
  std::vector<int> decision_token_ids(
      history_after[0].begin() + history_before[0].size(),
      history_after[0].end());
  if (decision_token_ids.empty()) {
    return absl::DataLossError(
        "DPM agent decode produced no exact executor token delta.");
  }
  if (decision_token_ids.size() >
      static_cast<size_t>(request.max_output_tokens)) {
    return absl::DataLossError(
        "DPM agent decode exceeded its exact output-token limit.");
  }

  // The session is deliberately left alive with this exact post-decode state
  // so DPMEngine can capture the producing capsule before releasing it.
  return DPMAgentGenerationOutcome{
      .decision_output = responses.GetTexts()[0],
      .decision_token_ids = std::move(decision_token_ids),
  };
}

}  // namespace litert::lm
