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

#include "runtime/dpm/session_checkpoint.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kDescriptorMagic = {'D', 'P', 'M', 'C',
                                                  'K', 'P', '0', '1'};
constexpr uint64_t kMaximumCheckpointCapsuleBytes =
    uint64_t{8} * 1024 * 1024 * 1024;

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
}

void AppendU32(uint32_t value, std::string* output) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendU64(uint64_t value, std::string* output) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendI64(int64_t value, std::string* output) {
  AppendU64(static_cast<uint64_t>(value), output);
}

absl::Status AppendString(absl::string_view value, std::string* output) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError(
        "DPM checkpoint descriptor string is too large.");
  }
  AppendU32(static_cast<uint32_t>(value.size()), output);
  output->append(value.data(), value.size());
  return absl::OkStatus();
}

void AppendHash(const Hash256& value, std::string* output) {
  output->append(reinterpret_cast<const char*>(value.bytes.data()),
                 value.bytes.size());
}

bool HasZeroOrMissingHash(const std::optional<Hash256>& hash) {
  return !hash.has_value() || IsZeroHash(*hash);
}

bool HasNoExactRegenerationFields(
    const DPMSessionCheckpointDescriptor& descriptor) {
  return !descriptor.exact_profile_id.has_value() &&
         IsZeroHash(descriptor.exact_profile_admission_record_id) &&
         IsZeroHash(descriptor.exact_request_execution_evidence_id) &&
         descriptor.worker_prefill_mode ==
             DPMCheckpointWorkerPrefillMode::kNone &&
         IsZeroHash(descriptor.execution_plan_hash) &&
         IsZeroHash(descriptor.exact_output_evidence_hash) &&
         !descriptor.worker_provenance.has_value();
}

absl::Status ValidateCommonDescriptorFields(
    const DPMSessionCheckpointDescriptor& descriptor) {
  if (descriptor.log_id.empty()) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint requires a log id.");
  }
  if (descriptor.stage != DPMSessionCheckpointStage::kAgentDecision) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint has an unsupported capture stage.");
  }
  if (descriptor.response_event_index != descriptor.source_event_count) {
    return absl::InvalidArgumentError(
        "DPM checkpoint response index must immediately follow its source "
        "prefix.");
  }
  if (descriptor.key_id.empty() || descriptor.key_id.size() > 1024) {
    return absl::InvalidArgumentError(
        "DPM checkpoint authentication key id must contain 1 to 1024 "
        "bytes.");
  }
  if (descriptor.created_unix_micros <= 0) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint requires a positive creation timestamp.");
  }
  if (descriptor.envelope_size == 0 || IsZeroHash(descriptor.envelope_hash)) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint requires a non-empty envelope.");
  }
  if (IsZeroHash(descriptor.source_prefix_hash) ||
      IsZeroHash(descriptor.projection_request_hash) ||
      IsZeroHash(descriptor.projection_manifest_hash) ||
      IsZeroHash(descriptor.correction_digest) ||
      IsZeroHash(descriptor.agent_request_hash) ||
      IsZeroHash(descriptor.agent_transcript_hash) ||
      IsZeroHash(descriptor.session_identity.model_artifact_hash) ||
      IsZeroHash(descriptor.session_identity.runtime_artifact_hash) ||
      IsZeroHash(descriptor.session_identity.inference_profile_hash)) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint is missing a required identity hash.");
  }
  return absl::OkStatus();
}

absl::Status ValidateExactWorkerProvenance(
    const DPMExactWorkerCheckpointProvenance& provenance,
    const DPMSessionCheckpointDescriptor& descriptor) {
  if (provenance.run_index != 0 || IsZeroHash(provenance.execution_plan_hash) ||
      IsZeroHash(provenance.request_envelope_hash) ||
      IsZeroHash(provenance.result_envelope_hash) ||
      provenance.transient_envelope_size == 0 ||
      provenance.transient_envelope_size > kMaximumCheckpointCapsuleBytes ||
      IsZeroHash(provenance.transient_envelope_hash) ||
      IsZeroHash(provenance.output_evidence_hash)) {
    return absl::InvalidArgumentError(
        "DPM exact checkpoint worker provenance is incomplete, oversized, "
        "or not run zero.");
  }
  if (provenance.execution_plan_hash != descriptor.execution_plan_hash ||
      provenance.output_evidence_hash !=
          descriptor.exact_output_evidence_hash) {
    return absl::DataLossError(
        "DPM exact checkpoint worker provenance differs from its physical "
        "plan or exact output evidence.");
  }
  return absl::OkStatus();
}

absl::Status ValidateDescriptorFields(
    const DPMSessionCheckpointDescriptor& descriptor) {
  if (descriptor.format_version !=
      DPMSessionCheckpointDescriptor::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Unsupported DPM session checkpoint descriptor format.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCommonDescriptorFields(descriptor));
  if (descriptor.envelope_size > kMaximumCheckpointCapsuleBytes) {
    return absl::ResourceExhaustedError(
        "DPM checkpoint durable capsule exceeds its size limit.");
  }
  if (descriptor.restored_from_checkpoint_id.has_value() &&
      IsZeroHash(*descriptor.restored_from_checkpoint_id)) {
    return absl::InvalidArgumentError(
        "DPM checkpoint restored-from descriptor ID must be nonzero.");
  }
  if (IsZeroHash(descriptor.capsule_restore_capability_id) ||
      IsZeroHash(descriptor.capsule_restore_admission_record_id) ||
      IsZeroHash(descriptor.capsule_restore_coverage_id)) {
    return absl::InvalidArgumentError(
        "DPM checkpoints require complete CapsuleRestore capability, "
        "admission, and operational-coverage provenance.");
  }
  if (IsZeroHash(descriptor.capsule_capture_plan_hash)) {
    return absl::InvalidArgumentError(
        "DPM checkpoint is missing its capsule capture-plan hash.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateDPMPreparedPrefillWorkBinding(descriptor.prepared_prefill_work));
  if (descriptor.capsule_capture_plan_hash ==
          descriptor.prepared_prefill_work.plan_id ||
      descriptor.capsule_capture_plan_hash ==
          descriptor.prepared_prefill_work.canonical_source_chunks_hash ||
      descriptor.capsule_capture_plan_hash ==
          descriptor.prepared_prefill_work.resolved_token_plan_hash ||
      descriptor.capsule_capture_plan_hash ==
          descriptor.prepared_prefill_work.shape_schedule_hash) {
    return absl::InvalidArgumentError(
        "DPM checkpoint substituted prepared-work evidence for its "
        "capture-plan hash.");
  }
  const bool restored = descriptor.restored_from_checkpoint_id.has_value();
  if ((restored && descriptor.prepared_prefill_work.start_kind !=
                       DPMPreparedPrefillStartKind::kOwnPositionRestore) ||
      (!restored && descriptor.prepared_prefill_work.start_kind !=
                        DPMPreparedPrefillStartKind::kFreshSession)) {
    return absl::InvalidArgumentError(
        "DPM checkpoint prepared work disagrees with its restored-parent "
        "provenance.");
  }

  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(descriptor.replay_mode));
  switch (descriptor.capture_origin) {
    case DPMCheckpointCaptureOrigin::kLiveParentSession:
    case DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker:
      break;
    case DPMCheckpointCaptureOrigin::kNone:
    default:
      return absl::InvalidArgumentError(
          "A publishable DPM checkpoint must identify its capture origin.");
  }

  switch (descriptor.replay_mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      if (descriptor.capture_origin !=
              DPMCheckpointCaptureOrigin::kLiveParentSession ||
          !HasNoExactRegenerationFields(descriptor)) {
        return absl::InvalidArgumentError(
            "WinnerReplay checkpoints require a live-parent capture and "
            "cannot carry exact-worker provenance.");
      }
      return absl::OkStatus();

    case DPMReplayMode::kExactRegeneration:
      if (descriptor.capture_origin !=
              DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker ||
          HasZeroOrMissingHash(descriptor.exact_profile_id) ||
          IsZeroHash(descriptor.exact_profile_admission_record_id) ||
          IsZeroHash(descriptor.exact_request_execution_evidence_id) ||
          IsZeroHash(descriptor.execution_plan_hash) ||
          IsZeroHash(descriptor.exact_output_evidence_hash) ||
          !descriptor.worker_provenance.has_value()) {
        return absl::InvalidArgumentError(
            "ExactRegeneration checkpoints require complete profile, "
            "admission, request, plan, output, and authenticated-worker "
            "provenance.");
      }
      switch (descriptor.worker_prefill_mode) {
        case DPMCheckpointWorkerPrefillMode::kFullCanonicalPrefill:
          if (restored) {
            return absl::InvalidArgumentError(
                "A full-prefill exact checkpoint cannot claim a restored "
                "checkpoint.");
          }
          break;
        case DPMCheckpointWorkerPrefillMode::kOwnPositionCapsuleDelta:
          if (!restored) {
            return absl::InvalidArgumentError(
                "An own-position exact checkpoint requires its restored "
                "checkpoint ID.");
          }
          break;
        case DPMCheckpointWorkerPrefillMode::kNone:
        default:
          return absl::InvalidArgumentError(
              "ExactRegeneration checkpoint has no admitted worker prefill "
              "mode.");
      }
      return ValidateExactWorkerProvenance(*descriptor.worker_provenance,
                                           descriptor);
  }
  return absl::InternalError("Validated DPM replay mode lost its known value.");
}

absl::Status AppendOptionalHash(const std::optional<Hash256>& value,
                                std::string* output) {
  output->push_back(value.has_value() ? '\1' : '\0');
  if (value.has_value()) AppendHash(*value, output);
  return absl::OkStatus();
}

void AppendPreparedPrefillWorkBinding(
    const DPMPreparedPrefillWorkBinding& binding, std::string* output) {
  AppendU32(static_cast<uint32_t>(binding.start_kind), output);
  AppendHash(binding.plan_id, output);
  AppendHash(binding.canonical_source_chunks_hash, output);
  AppendHash(binding.resolved_token_plan_hash, output);
  AppendHash(binding.shape_schedule_hash, output);
}

absl::StatusOr<std::string> EncodeDescriptorForHash(
    const DPMSessionCheckpointDescriptor& descriptor) {
  ABSL_RETURN_IF_ERROR(ValidateDescriptorFields(descriptor));
  std::string bytes;
  bytes.reserve(1096 + descriptor.log_id.size() + descriptor.key_id.size());
  bytes.append(kDescriptorMagic.data(), kDescriptorMagic.size());
  AppendU32(descriptor.format_version, &bytes);
  ABSL_RETURN_IF_ERROR(AppendString(descriptor.log_id, &bytes));
  bytes.push_back(static_cast<char>(descriptor.stage));
  AppendU64(descriptor.source_event_count, &bytes);
  AppendHash(descriptor.source_prefix_hash, &bytes);
  AppendU64(descriptor.response_event_index, &bytes);
  AppendHash(descriptor.projection_request_hash, &bytes);
  AppendHash(descriptor.projection_manifest_hash, &bytes);
  AppendHash(descriptor.correction_digest, &bytes);
  AppendHash(descriptor.agent_request_hash, &bytes);
  AppendHash(descriptor.agent_transcript_hash, &bytes);
  AppendHash(descriptor.session_identity.model_artifact_hash, &bytes);
  AppendHash(descriptor.session_identity.runtime_artifact_hash, &bytes);
  AppendHash(descriptor.session_identity.inference_profile_hash, &bytes);
  ABSL_RETURN_IF_ERROR(AppendString(descriptor.key_id, &bytes));
  AppendHash(descriptor.envelope_hash, &bytes);
  AppendU64(descriptor.envelope_size, &bytes);
  AppendI64(descriptor.created_unix_micros, &bytes);

  bytes.push_back(static_cast<char>(descriptor.replay_mode));
  bytes.push_back(static_cast<char>(descriptor.capture_origin));
  ABSL_RETURN_IF_ERROR(
      AppendOptionalHash(descriptor.restored_from_checkpoint_id, &bytes));
  ABSL_RETURN_IF_ERROR(AppendOptionalHash(descriptor.exact_profile_id, &bytes));
  AppendHash(descriptor.exact_profile_admission_record_id, &bytes);
  AppendHash(descriptor.capsule_restore_capability_id, &bytes);
  AppendHash(descriptor.capsule_restore_admission_record_id, &bytes);
  AppendHash(descriptor.capsule_restore_coverage_id, &bytes);
  AppendHash(descriptor.exact_request_execution_evidence_id, &bytes);
  bytes.push_back(static_cast<char>(descriptor.worker_prefill_mode));
  AppendHash(descriptor.execution_plan_hash, &bytes);
  AppendHash(descriptor.exact_output_evidence_hash, &bytes);
  bytes.push_back(descriptor.worker_provenance.has_value() ? '\1' : '\0');
  if (descriptor.worker_provenance.has_value()) {
    const DPMExactWorkerCheckpointProvenance& provenance =
        *descriptor.worker_provenance;
    AppendU32(provenance.run_index, &bytes);
    AppendHash(provenance.execution_plan_hash, &bytes);
    AppendHash(provenance.request_envelope_hash, &bytes);
    AppendHash(provenance.result_envelope_hash, &bytes);
    AppendU64(provenance.transient_envelope_size, &bytes);
    AppendHash(provenance.transient_envelope_hash, &bytes);
    AppendHash(provenance.output_evidence_hash, &bytes);
  }

  AppendHash(descriptor.capsule_capture_plan_hash, &bytes);
  AppendPreparedPrefillWorkBinding(descriptor.prepared_prefill_work, &bytes);
  return bytes;
}

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

}  // namespace

absl::Status ValidateDPMPreparedPrefillWorkBinding(
    const DPMPreparedPrefillWorkBinding& binding) {
  switch (binding.start_kind) {
    case DPMPreparedPrefillStartKind::kFreshSession:
    case DPMPreparedPrefillStartKind::kOwnPositionRestore:
      break;
    default:
      return absl::InvalidArgumentError(
          "DPM prepared-prefill work binding has an unsupported start kind.");
  }
  if (IsZeroHash(binding.plan_id) ||
      IsZeroHash(binding.canonical_source_chunks_hash) ||
      IsZeroHash(binding.resolved_token_plan_hash) ||
      IsZeroHash(binding.shape_schedule_hash)) {
    return absl::InvalidArgumentError(
        "DPM prepared-prefill work binding is incomplete.");
  }
  if (binding.plan_id == binding.canonical_source_chunks_hash ||
      binding.plan_id == binding.resolved_token_plan_hash ||
      binding.plan_id == binding.shape_schedule_hash ||
      binding.canonical_source_chunks_hash ==
          binding.resolved_token_plan_hash ||
      binding.canonical_source_chunks_hash == binding.shape_schedule_hash ||
      binding.resolved_token_plan_hash == binding.shape_schedule_hash) {
    return absl::InvalidArgumentError(
        "DPM prepared-prefill work binding substitutes one hash domain for "
        "another.");
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> ComputeDPMSessionCheckpointId(
    const DPMSessionCheckpointDescriptor& descriptor) {
  ABSL_ASSIGN_OR_RETURN(std::string canonical,
                        EncodeDescriptorForHash(descriptor));
  return Sha256(canonical);
}

absl::Status ValidateDPMSessionCheckpointArtifact(
    const DPMSessionCheckpointArtifact& artifact) {
  ABSL_RETURN_IF_ERROR(ValidateDescriptorFields(artifact.descriptor));
  if (artifact.authenticated_envelope.size() !=
      artifact.descriptor.envelope_size) {
    return absl::DataLossError(
        "DPM session checkpoint envelope size does not match descriptor.");
  }
  if (Sha256(artifact.authenticated_envelope) !=
      artifact.descriptor.envelope_hash) {
    return absl::DataLossError(
        "DPM session checkpoint envelope hash does not match descriptor.");
  }
  ABSL_ASSIGN_OR_RETURN(Hash256 expected_id,
                        ComputeDPMSessionCheckpointId(artifact.descriptor));
  if (expected_id != artifact.descriptor.descriptor_id) {
    return absl::DataLossError(
        "DPM session checkpoint descriptor id is not canonical.");
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
