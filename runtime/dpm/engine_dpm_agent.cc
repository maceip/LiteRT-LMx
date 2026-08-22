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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/capsule_restore_evidence.h"
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_prepared_prefill_runtime.h"
#include "runtime/dpm/fresh_worker_process.h"
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
       executor.GetAdvancedSettings()->num_logits_to_print_after_decode != 0)) {
    return absl::UnimplementedError(
        "DPM agent generation does not support benchmark, profiling, or "
        "logits debug execution paths.");
  }
  return absl::OkStatus();
}

absl::Status ValidateDPMConfig(const SessionConfig& config) {
  if (config.GetMemoryStrategy() != SessionConfig::MemoryStrategy::kStateful) {
    return absl::FailedPreconditionError(
        "DPM agent sessions must remain stateful so capsule restore and delta "
        "prefill cannot be erased by projection reset policy.");
  }
  if (config.AudioModalityEnabled() || config.VisionModalityEnabled() ||
      config.GetAudioEmbeddingsCallback() != nullptr) {
    return absl::InvalidArgumentError("DPM agent sessions must be text-only.");
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

bool HasSameStateWitnessAdmissionAuthority(
    const AuthenticatedCapsuleRestoreAdmission& lhs,
    const AuthenticatedCapsuleRestoreAdmission& rhs) {
  return lhs.record.record_id == rhs.record.record_id &&
         lhs.profile == rhs.profile && lhs.capability == rhs.capability &&
         lhs.operational_coverage == rhs.operational_coverage &&
         lhs.record.operational_coverage == lhs.operational_coverage &&
         rhs.record.operational_coverage == rhs.operational_coverage;
}

CapsuleRestoreAuthority MakeCapsuleRestoreAuthority(
    const AuthenticatedCapsuleRestoreAdmission& admission) {
  return CapsuleRestoreAuthority{
      .capability = admission.capability,
      .admission_record_id = admission.record.record_id,
      .coverage_id = admission.operational_coverage.coverage_id,
      .qualification_evidence_hash =
          admission.operational_coverage.qualification_evidence_hash,
  };
}

absl::StatusOr<std::vector<CapsuleCanonicalPrefillChunk>>
ConvertCapsuleCanonicalPrefillChunks(
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& chunks) {
  std::vector<CapsuleCanonicalPrefillChunk> converted;
  converted.reserve(chunks.size());
  for (const DPMAgentGenerationRequest::PrefillChunk& chunk : chunks) {
    CapsuleCanonicalPrefillChunk canonical;
    switch (chunk.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text:
        if (!chunk.token_ids.empty()) {
          return absl::InvalidArgumentError(
              "CapsuleRestore text chunk also contains exact token IDs.");
        }
        canonical.encoding = CapsuleCanonicalPrefillChunk::Encoding::kUtf8Text;
        canonical.utf8_text = chunk.text;
        break;
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds:
        if (!chunk.text.empty()) {
          return absl::InvalidArgumentError(
              "CapsuleRestore token chunk also contains UTF-8 text.");
        }
        canonical.encoding =
            CapsuleCanonicalPrefillChunk::Encoding::kExactTokenIds;
        canonical.token_ids.reserve(chunk.token_ids.size());
        for (int token_id : chunk.token_ids) {
          if (token_id < 0 || static_cast<int64_t>(token_id) >
                  (std::numeric_limits<int32_t>::max)()) {
            return absl::InvalidArgumentError(
                "CapsuleRestore chunk contains a non-int32 token ID.");
          }
          canonical.token_ids.push_back(static_cast<int32_t>(token_id));
        }
        break;
      default:
        return absl::InvalidArgumentError(
            "CapsuleRestore chunk has an unknown encoding.");
    }
    converted.push_back(std::move(canonical));
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCanonicalPrefillChunks(converted));
  return converted;
}

absl::Status ValidateCapsuleRestoreOperationInputs(
    const DPMAgentCapsuleRestoreOperation& operation,
    const Hash256& restore_checkpoint_id,
    const SessionContinuationStateWitness& restored_state_witness,
    const Hash256& logical_agent_request_hash, int max_output_tokens,
    const AuthenticatedCapsuleRestoreAdmission& admission) {
  if (operation.format_version !=
      DPMAgentCapsuleRestoreOperation::kFormatVersion) {
    return absl::FailedPreconditionError(
        "CapsuleRestore operation version is unsupported.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleCaptureEvidence(operation.source_capture_evidence));
  ABSL_RETURN_IF_ERROR(
      ValidateSessionContinuationStateWitness(restored_state_witness));
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffReauthenticationEvidence(
      operation.durable_to_transient_reauthentication));

  const CapsuleRestoreAuthority expected_authority =
      MakeCapsuleRestoreAuthority(admission);
  const CapsuleCaptureEvidence& source = operation.source_capture_evidence;
  const CapsuleRestoreStateWitnessOperationalDomain& domain =
      admission.operational_coverage.operational_domain;
  if (operation.current_authority != expected_authority ||
      source.plan.authority != expected_authority) {
    return absl::FailedPreconditionError(
        "CapsuleRestore operation or source capture differs from the freshly "
        "reauthenticated loaded-Engine authority.");
  }
  if (restore_checkpoint_id == Hash256{} ||
      source.checkpoint_id != restore_checkpoint_id ||
      source.plan.checkpoint_authentication_key_id !=
          domain.checkpoint_authentication_key_id ||
      source.checkpoint_authentication_key_id !=
          domain.checkpoint_authentication_key_id) {
    return absl::FailedPreconditionError(
        "CapsuleRestore operation names another checkpoint or durable "
        "authentication domain.");
  }

  const SessionHandoffReauthenticationEvidence& reauthentication =
      operation.durable_to_transient_reauthentication;
  if (reauthentication.session_identity !=
          expected_authority.capability.session_identity ||
      reauthentication.source_envelope_hash !=
          source.checkpoint_envelope_hash ||
      reauthentication.source_envelope_size !=
          source.checkpoint_envelope_size ||
      reauthentication.source_key_id !=
          source.checkpoint_authentication_key_id ||
      reauthentication.destination_envelope_hash !=
          restored_state_witness.envelope_hash ||
      reauthentication.destination_envelope_size !=
          restored_state_witness.envelope_size ||
      reauthentication.destination_key_id != restored_state_witness.key_id ||
      reauthentication.destination_key_id !=
          kFreshWorkerTransientRestoreKeyId ||
      reauthentication.purpose !=
          kFreshWorkerDurableRestoreToTransientReauthenticationPurpose ||
      reauthentication.capsule_codec_contract_hash !=
          expected_authority.capability.capsule_codec_contract_hash ||
      reauthentication.canonical_continuation_state_hash !=
          source.transient_to_durable_reauthentication
              .canonical_continuation_state_hash) {
    return absl::FailedPreconditionError(
        "CapsuleRestore operation does not bind the durable source endpoint "
        "to the transient envelope imported by the live target.");
  }
  if (restored_state_witness.session_identity !=
          expected_authority.capability.session_identity ||
      restored_state_witness.phase != SessionHandoffPhase::kDecoded ||
      !restored_state_witness.ran_decode ||
      restored_state_witness.current_step <= 0 ||
      static_cast<uint64_t>(restored_state_witness.current_step) !=
          source.plan.capture_end_step ||
      restored_state_witness.processed_history_token_bytes_hash !=
          source.checkpoint_history_token_bytes_hash) {
    return absl::FailedPreconditionError(
        "CapsuleRestore transient target witness differs from the source "
        "capture's decoded own-position state.");
  }

  const CapsuleDPMRestoreTarget& target = operation.target_state;
  const CapsuleDPMCheckpointState& checkpoint = source.plan.checkpoint_state;
  if (target.log_id.empty() || target.source_event_count == 0 ||
      target.prospective_response_event_index != target.source_event_count ||
      target.source_prefix_hash == Hash256{} ||
      target.projection_request_hash == Hash256{} ||
      target.projection_manifest_hash == Hash256{} ||
      target.correction_digest == Hash256{} ||
      target.agent_transcript_prefix_hash == Hash256{} ||
      target.logical_agent_request_hash != logical_agent_request_hash ||
      target.log_id != checkpoint.log_id ||
      target.correction_digest != checkpoint.correction_digest ||
      target.source_event_count <= checkpoint.response_event_index ||
      target.source_prefix_hash == checkpoint.source_prefix_hash ||
      checkpoint.response_event_index ==
          (std::numeric_limits<uint64_t>::max)()) {
    return absl::FailedPreconditionError(
        "CapsuleRestore target does not describe the authoritative current "
        "same-epoch logical request after the source checkpoint.");
  }
  if (max_output_tokens <= 0 ||
      static_cast<uint64_t>(max_output_tokens) > domain.maximum_output_tokens) {
    return absl::FailedPreconditionError(
        "CapsuleRestore output limit is outside the current authenticated "
        "authenticated operational domain.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCapsuleRestorePreparedShape(
    const std::vector<CapsuleCanonicalPrefillChunk>& chunks,
    const DPMPreparedPrefillPlan& prepared,
    const AuthenticatedCapsuleRestoreAdmission& admission,
    int max_output_tokens) {
  const CapsuleRestoreStateWitnessOperationalDomain& domain =
      admission.operational_coverage.operational_domain;
  uint64_t text_bytes = 0;
  uint64_t token_ids = 0;
  uint32_t encoding_mask = 0;
  for (const CapsuleCanonicalPrefillChunk& chunk : chunks) {
    switch (chunk.encoding) {
      case CapsuleCanonicalPrefillChunk::Encoding::kUtf8Text:
        if (chunk.utf8_text.size() >
            domain.maximum_prefill_text_bytes - text_bytes) {
          return absl::FailedPreconditionError(
              "CapsuleRestore delta text exceeds its admitted domain.");
        }
        text_bytes += chunk.utf8_text.size();
        encoding_mask |= CapsuleRestoreStateWitnessEncodingBit(
            CapsuleRestoreStateWitnessEncoding::kUtf8Text);
        break;
      case CapsuleCanonicalPrefillChunk::Encoding::kExactTokenIds:
        if (chunk.token_ids.size() >
            domain.maximum_prefill_token_ids - token_ids) {
          return absl::FailedPreconditionError(
              "CapsuleRestore delta token input exceeds its admitted "
              "domain.");
        }
        token_ids += chunk.token_ids.size();
        encoding_mask |= CapsuleRestoreStateWitnessEncodingBit(
            CapsuleRestoreStateWitnessEncoding::kExactTokenIds);
        break;
    }
  }
  if (chunks.size() < domain.minimum_prefill_chunks ||
      chunks.size() > domain.maximum_prefill_chunks || encoding_mask == 0 ||
      (encoding_mask & ~domain.admitted_encoding_mask) != 0 ||
      prepared.start_step < domain.minimum_checkpoint_step ||
      prepared.start_step > domain.maximum_checkpoint_step ||
      prepared.end_step <= prepared.start_step) {
    return absl::FailedPreconditionError(
        "CapsuleRestore prepared delta is outside its authenticated chunk, "
        "encoding, checkpoint, or position domain.");
  }
  const uint64_t delta_positions = prepared.end_step - prepared.start_step;
  const uint64_t output_tokens = static_cast<uint64_t>(max_output_tokens);
  if (delta_positions < domain.minimum_delta_positions ||
      delta_positions > domain.maximum_delta_positions ||
      prepared.end_step > domain.maximum_context_positions ||
      output_tokens > domain.maximum_context_positions ||
      prepared.end_step > domain.maximum_context_positions - output_tokens ||
      prepared.start_step > (std::numeric_limits<uint32_t>::max)() ||
      prepared.end_step > (std::numeric_limits<uint32_t>::max)()) {
    return absl::FailedPreconditionError(
        "CapsuleRestore prepared delta or requested decode exceeds its "
        "authenticated shape or context bound.");
  }
  return absl::OkStatus();
}

absl::StatusOr<CapsuleRestoreEvidence> BuildAndValidateCapsuleRestoreEvidence(
    const DPMAgentCapsuleRestoreOperation& operation,
    const Hash256& restore_checkpoint_id,
    const SessionContinuationStateWitness& restored_state_witness,
    const std::vector<CapsuleCanonicalPrefillChunk>& canonical_chunks,
    DPMPreparedPrefillPlan prepared_prefill_plan, int max_output_tokens) {
  const CapsuleCaptureEvidence& source = operation.source_capture_evidence;

  CapsulePrefillPlan capsule_prefill;
  capsule_prefill.mode = CapsulePrefillMode::kOwnPositionCapsuleDelta;
  capsule_prefill.event_range_start =
      source.plan.checkpoint_state.response_event_index + 1;
  capsule_prefill.event_range_end = operation.target_state.source_event_count;
  capsule_prefill.start_step =
      static_cast<uint32_t>(prepared_prefill_plan.start_step);
  capsule_prefill.end_step =
      static_cast<uint32_t>(prepared_prefill_plan.end_step);
  capsule_prefill.canonical_chunks = canonical_chunks;
  ABSL_ASSIGN_OR_RETURN(
      capsule_prefill.canonical_delta_chunks_hash,
      ComputeCapsuleCanonicalDeltaChunksHash(canonical_chunks));
  capsule_prefill.prepared_plan = std::move(prepared_prefill_plan);
  ABSL_RETURN_IF_ERROR(ValidateCapsulePrefillPlan(capsule_prefill));

  CapsuleRestorePlan restore_plan;
  restore_plan.authority = operation.current_authority;
  restore_plan.source_capture_plan_hash = source.plan.plan_hash;
  restore_plan.source_capture_evidence_id = source.evidence_id;
  restore_plan.checkpoint_id = restore_checkpoint_id;
  restore_plan.checkpoint_state = source.plan.checkpoint_state;
  restore_plan.checkpoint_envelope_hash = source.checkpoint_envelope_hash;
  restore_plan.checkpoint_envelope_size = source.checkpoint_envelope_size;
  restore_plan.checkpoint_authentication_key_id =
      source.checkpoint_authentication_key_id;
  restore_plan.checkpoint_step = source.plan.capture_end_step;
  restore_plan.checkpoint_history_token_bytes_hash =
      source.checkpoint_history_token_bytes_hash;
  restore_plan.target_state = operation.target_state;
  restore_plan.prefill = std::move(capsule_prefill);
  restore_plan.maximum_output_tokens = static_cast<uint32_t>(max_output_tokens);
  ABSL_ASSIGN_OR_RETURN(restore_plan.plan_hash,
                        ComputeCapsuleRestorePlanHash(restore_plan));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestorePlan(restore_plan));

  CapsuleRestoreEvidence restore_evidence;
  restore_evidence.plan = std::move(restore_plan);
  restore_evidence.durable_to_transient_reauthentication =
      operation.durable_to_transient_reauthentication;
  restore_evidence.target_post_import = restored_state_witness;
  ABSL_ASSIGN_OR_RETURN(restore_evidence.evidence_id,
                        ComputeCapsuleRestoreEvidenceId(restore_evidence));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidence(restore_evidence));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreEvidenceForSourceCapture(restore_evidence, source));
  return restore_evidence;
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
  return CreateInternal(engine, std::move(session_config),
      std::optional<CapsuleRestoreAdmissionBinding>(
          std::move(capsule_restore_admission)),
      expected_identity);
}

absl::StatusOr<std::unique_ptr<EngineDPMAgentRuntime>>
EngineDPMAgentRuntime::CreateInternal(
    Engine* engine, SessionConfig session_config,
    std::optional<CapsuleRestoreAdmissionBinding> capsule_restore_admission,
    std::optional<SessionHandoffIdentity> expected_identity) {
  if (engine == nullptr) {
    return absl::InvalidArgumentError(
        "DPM agent runtime requires a loaded Engine.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMEngineSettings(engine->GetEngineSettings()));
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
  if (requested_sampler.type() != proto::SamplerParameters::TYPE_UNSPECIFIED &&
      requested_sampler.type() != proto::SamplerParameters::GREEDY) {
    return absl::InvalidArgumentError(
        "DPM agent runtime cannot discard a requested non-GREEDY sampler.");
  }
  if (requested_sampler.backend() != proto::SamplerParameters::UNSPECIFIED &&
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
  proto::SamplerParameters& sampler = session_config.GetMutableSamplerParams();
  sampler.set_type(proto::SamplerParameters::GREEDY);
  sampler.set_k(0);
  sampler.set_p(0.0f);
  sampler.set_temperature(0.0f);
  sampler.clear_seed();
  sampler.set_backend(proto::SamplerParameters::CPU);
  ABSL_RETURN_IF_ERROR(
      session_config.MaybeUpdateAndValidate(engine->GetEngineSettings()));
  ABSL_RETURN_IF_ERROR(ValidateDPMConfig(session_config));

  ABSL_ASSIGN_OR_RETURN(SessionHandoffIdentity authoritative_identity,
      engine->ResolveSessionHandoffIdentity(session_config));
  if (expected_identity.has_value() &&
      *expected_identity != authoritative_identity) {
    return absl::FailedPreconditionError(
        "DPM agent identity assertion does not match the loaded Engine.");
  }

  std::optional<AuthenticatedCapsuleRestoreAdmission> initial_admission;
  if (capsule_restore_admission.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        AuthenticatedCapsuleRestoreAdmission admission,
        ResolveAuthenticatedCapsuleRestoreAdmission(
            engine, session_config, *capsule_restore_admission));
    if (admission.profile.session_identity != authoritative_identity ||
        admission.capability.session_identity != authoritative_identity ||
        admission.operational_coverage.runtime_derived_session_identity !=
            authoritative_identity ||
        admission.operational_coverage.runtime_derived_profile !=
            admission.profile ||
        admission.operational_coverage.runtime_derived_capability !=
            admission.capability ||
        admission.record.operational_coverage !=
            admission.operational_coverage) {
      return absl::FailedPreconditionError(
          "CapsuleRestore admission does not match the live parent "
          "runtime's Engine-derived authority.");
    }
    initial_admission = std::move(admission);
  }

  auto runtime =
      std::unique_ptr<EngineDPMAgentRuntime>(new EngineDPMAgentRuntime(
          engine, std::move(session_config), authoritative_identity,
          std::move(capsule_restore_admission), std::move(initial_admission)));
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
  ABSL_RETURN_IF_ERROR(ValidateDPMEngineSettings(engine_->GetEngineSettings()));
  ABSL_RETURN_IF_ERROR(ValidateDPMConfig(resolved_session_config_));
  ABSL_ASSIGN_OR_RETURN(
      const SessionHandoffIdentity current_identity,
      engine_->ResolveSessionHandoffIdentity(resolved_session_config_));
  if (current_identity != session_handoff_identity_) {
    return absl::FailedPreconditionError(
        "Loaded Engine identity changed after DPM agent admission.");
  }
  if (capsule_restore_admission_.has_value() !=
      initial_capsule_restore_admission_.has_value()) {
    return absl::InternalError(
        "DPM agent CapsuleRestore authority is internally inconsistent.");
  }
  if (capsule_restore_admission_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(const AuthenticatedCapsuleRestoreAdmission current,
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
  if (!capsule_restore_admission_.has_value()) {
    return absl::FailedPreconditionError(
        "DPM agent runtime was constructed without CapsuleRestore "
        "admission.");
  }
  ABSL_ASSIGN_OR_RETURN(const AuthenticatedCapsuleRestoreAdmission before,
                        ResolveCurrentCapsuleRestoreAdmission());
  ABSL_RETURN_IF_ERROR(ProbeSessionHandoffSupport());
  ABSL_ASSIGN_OR_RETURN(const AuthenticatedCapsuleRestoreAdmission after,
        ResolveCurrentCapsuleRestoreAdmission());
  if (!HasSameStateWitnessAdmissionAuthority(before, after)) {
      return absl::AbortedError(
        "CapsuleRestore authority changed during live-runtime support "
          "validation.");
    }
  return absl::OkStatus();
}

absl::StatusOr<AuthenticatedCapsuleRestoreAdmission>
EngineDPMAgentRuntime::ResolveCurrentCapsuleRestoreAdmission() const {
  if (capsule_restore_admission_.has_value() !=
      initial_capsule_restore_admission_.has_value()) {
    return absl::InternalError(
        "DPM agent CapsuleRestore authority is internally inconsistent.");
  }
  if (!capsule_restore_admission_.has_value()) {
    return absl::FailedPreconditionError(
        "DPM agent runtime was constructed without CapsuleRestore "
        "admission.");
  }
  ABSL_ASSIGN_OR_RETURN(
      AuthenticatedCapsuleRestoreAdmission current,
      ResolveAuthenticatedCapsuleRestoreAdmission(
          engine_, resolved_session_config_, *capsule_restore_admission_));
  if (!HasSameStateWitnessAdmissionAuthority(
          current, *initial_capsule_restore_admission_) ||
      current.profile.session_identity != session_handoff_identity_ ||
      current.capability.session_identity != session_handoff_identity_ ||
      current.operational_coverage.runtime_derived_session_identity !=
          session_handoff_identity_) {
    return absl::AbortedError(
        "Authenticated CapsuleRestore authority changed after live-runtime "
        "construction.");
  }
  return current;
}

absl::StatusOr<std::optional<AuthenticatedCapsuleRestoreAdmission>>
EngineDPMAgentRuntime::GetAuthenticatedCapsuleRestoreAdmission() const {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeSupport());
  if (!capsule_restore_admission_.has_value()) {
    return std::optional<AuthenticatedCapsuleRestoreAdmission>();
  }
  ABSL_ASSIGN_OR_RETURN(AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  return std::optional<AuthenticatedCapsuleRestoreAdmission>(
      std::move(current));
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

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> source,
      engine_->CreateSession(resolved_session_config_));
  if (source == nullptr) {
    return absl::InternalError(
        "DPM agent Engine returned a null handoff-probe source session.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSession(*source));

  // This key authenticates only an in-memory, fresh-state capability probe. It
  // is deliberately not configurable and is never used for a durable capsule.
  SessionHandoffOptions options;
  options.key_id = "litert-lm-dpm-probe";
  options.authentication_key = "0123456789abcdef0123456789abcdef";
  options.expected_identity = session_handoff_identity_;

  std::string envelope;
  StringByteSink sink(&envelope);
  ABSL_RETURN_IF_ERROR(source->ExportHandoffTo(options, &sink));
  if (envelope.empty()) {
    return absl::DataLossError(
        "DPM agent handoff capability probe produced an empty envelope.");
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> target,
      engine_->CreateSession(resolved_session_config_));
  if (target == nullptr) {
    return absl::InternalError(
        "DPM agent Engine returned a null handoff-probe target session.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSession(*target));
  StringByteSource source_bytes(envelope);
  ABSL_RETURN_IF_ERROR(target->ImportHandoffFrom(source_bytes, options));
  ABSL_ASSIGN_OR_RETURN(std::vector<std::vector<int>> restored_history,
      target->GetExactProcessedTokenHistory());
  return ValidateFreshExactHistory(restored_history);
}

absl::StatusOr<std::unique_ptr<Engine::Session>>
EngineDPMAgentRuntime::CreateSession() {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeSupport());
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
      engine_->CreateSession(resolved_session_config_));
  if (session == nullptr) {
    return absl::InternalError("DPM agent Engine returned a null session.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSession(*session));
  ABSL_ASSIGN_OR_RETURN(std::vector<std::vector<int>> fresh_history,
      session->GetExactProcessedTokenHistory());
  ABSL_RETURN_IF_ERROR(ValidateFreshExactHistory(fresh_history));
  if (capsule_restore_admission_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(const AuthenticatedCapsuleRestoreAdmission current,
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
  if (request.logical_agent_request_hash == Hash256{}) {
    return absl::InvalidArgumentError(
        "DPM agent generation requires its logical request hash.");
  }
  if (request.restore_checkpoint_id.has_value() !=
      request.restored_state_witness.has_value()) {
    return absl::InvalidArgumentError(
        "DPM agent restore requires both a checkpoint and live import "
        "witness.");
  }

  const bool is_restore = request.restore_checkpoint_id.has_value();
  const bool has_authority = capsule_restore_admission_.has_value();
  const bool has_operation = request.capsule_restore_operation.has_value();
  if (has_operation != (has_authority && is_restore)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore operation evidence is required exactly for a "
        "restore-shaped request on a capsule-enabled runtime and is "
        "forbidden for fresh or generation-only requests.");
  }

  std::optional<AuthenticatedCapsuleRestoreAdmission> operation_authority;
  if (has_authority) {
    ABSL_ASSIGN_OR_RETURN(operation_authority,
                          ResolveCurrentCapsuleRestoreAdmission());
  }

  std::optional<std::vector<CapsuleCanonicalPrefillChunk>>
      capsule_canonical_chunks;
  if (is_restore) {
    if (!operation_authority.has_value()) {
      return absl::FailedPreconditionError(
          "DPM agent restore-shaped generation requires an authenticated "
          "CapsuleRestore admission binding.");
    }
    ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreOperationInputs(
        *request.capsule_restore_operation, *request.restore_checkpoint_id,
        *request.restored_state_witness, request.logical_agent_request_hash,
        request.max_output_tokens, *operation_authority));
    ABSL_ASSIGN_OR_RETURN(
        capsule_canonical_chunks,
        ConvertCapsuleCanonicalPrefillChunks(request.canonical_prefill_chunks));
  }

  DPMPreparedPrefillStart start;
  if (request.restore_checkpoint_id.has_value()) {
    start.kind = DPMPreparedPrefillStartKind::kOwnPositionRestore;
    start.restore_checkpoint_id = request.restore_checkpoint_id;
    start.restored_state_witness = request.restored_state_witness;
  }
  ABSL_ASSIGN_OR_RETURN(DPMPreparedPrefillPlan prepared_prefill_plan,
      PrepareDPMEnginePrefillPlan(
          engine_, session, request.canonical_prefill_chunks,
          request.logical_agent_request_hash, start));

  std::optional<CapsuleRestoreEvidence> capsule_restore_evidence;
  if (operation_authority.has_value() && is_restore) {
    ABSL_RETURN_IF_ERROR(ValidateCapsuleRestorePreparedShape(
        *capsule_canonical_chunks, prepared_prefill_plan, *operation_authority,
        request.max_output_tokens));
    ABSL_ASSIGN_OR_RETURN(
        capsule_restore_evidence,
        BuildAndValidateCapsuleRestoreEvidence(
            *request.capsule_restore_operation, *request.restore_checkpoint_id,
            *request.restored_state_witness, *capsule_canonical_chunks,
            prepared_prefill_plan, request.max_output_tokens));
    ABSL_ASSIGN_OR_RETURN(
        const AuthenticatedCapsuleRestoreAdmission before_execute_authority,
        ResolveCurrentCapsuleRestoreAdmission());
    if (!HasSameStateWitnessAdmissionAuthority(before_execute_authority,
                                               *operation_authority) ||
        MakeCapsuleRestoreAuthority(before_execute_authority) !=
            request.capsule_restore_operation->current_authority) {
      return absl::AbortedError(
          "CapsuleRestore authority changed while preparing the "
          "operation gate.");
    }
  }

  // Every capture/authority/endpoint/target/shape join above is
  // complete before the first model-visible prefill call executes.
  ABSL_RETURN_IF_ERROR(ExecuteDPMEnginePrefillPlan(
      session, prepared_prefill_plan, request.restored_state_witness));

  const int vocabulary_size = engine_->GetTokenizer().GetVocabSize();
  if (vocabulary_size <= 0) {
    return absl::FailedPreconditionError(
        "DPM agent tokenizer has no measurable vocabulary.");
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<std::vector<int>> history_before,
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
  ABSL_ASSIGN_OR_RETURN(Responses responses, session->RunDecode(decode_config));
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

  ABSL_ASSIGN_OR_RETURN(std::vector<std::vector<int>> history_after,
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

  if (operation_authority.has_value()) {
    ABSL_ASSIGN_OR_RETURN(const AuthenticatedCapsuleRestoreAdmission current,
        ResolveCurrentCapsuleRestoreAdmission());
    if (!HasSameStateWitnessAdmissionAuthority(current, *operation_authority) ||
        (capsule_restore_evidence.has_value() &&
         MakeCapsuleRestoreAuthority(current) !=
             request.capsule_restore_operation->current_authority)) {
      return absl::AbortedError(
          "CapsuleRestore authority changed during DPM "
          "generation.");
    }
  }

  // The session is deliberately left alive with this exact post-decode state
  // so DPMEngine can capture the producing capsule before releasing it.
  return DPMAgentGenerationOutcome{
      .decision_output = responses.GetTexts()[0],
      .decision_token_ids = std::move(decision_token_ids),
      .prepared_prefill_plan = std::move(prepared_prefill_plan),
      .capsule_restore_evidence = std::move(capsule_restore_evidence),
  };
}

}  // namespace litert::lm
