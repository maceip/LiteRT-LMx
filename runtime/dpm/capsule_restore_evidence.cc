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

#include "runtime/dpm/capsule_restore_evidence.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/fresh_worker_process.h"
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kFullPrefillChunksDomain =
    "LITERT_LMX_CAPSULE_FULL_PREFILL_CHUNKS_SHA256";
constexpr absl::string_view kDeltaChunksDomain =
    "LITERT_LMX_CAPSULE_DELTA_CHUNKS_SHA256";
constexpr absl::string_view kCapturePlanDomain =
    "LITERT_LMX_CAPSULE_CAPTURE_PLAN_SHA256";
constexpr absl::string_view kRestorePlanDomain =
    "LITERT_LMX_CAPSULE_RESTORE_PLAN_SHA256";
constexpr absl::string_view kCaptureEvidenceContractDomain =
    "LITERT_LMX_CAPSULE_CAPTURE_EXPLICIT_REAUTH_CONTRACT_SHA256";
constexpr absl::string_view kRestoreEvidenceContractDomain =
    "LITERT_LMX_CAPSULE_RESTORE_EXPLICIT_REAUTH_CONTRACT_SHA256";
constexpr absl::string_view kCaptureEvidenceDomain =
    "LITERT_LMX_CAPSULE_CAPTURE_EVIDENCE_SHA256";
constexpr absl::string_view kRestoreEvidenceDomain =
    "LITERT_LMX_CAPSULE_RESTORE_EVIDENCE_SHA256";

bool IsZeroHash(const Hash256& hash) {
  uint8_t combined = 0;
  for (uint8_t byte : hash.bytes) combined |= byte;
  return combined == 0;
}

bool IsCompleteIdentity(const SessionHandoffIdentity& identity) {
  return !IsZeroHash(identity.model_artifact_hash) &&
         !IsZeroHash(identity.runtime_artifact_hash) &&
         !IsZeroHash(identity.inference_profile_hash);
}

bool HasControlByte(absl::string_view text) {
  for (unsigned char byte : text) {
    if (byte < 0x20 || byte == 0x7f) return true;
  }
  return false;
}

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

void AppendI32(int32_t value, std::string* output) {
  AppendU32(static_cast<uint32_t>(value), output);
}

void AppendHash(const Hash256& hash, std::string* output) {
  output->append(reinterpret_cast<const char*>(hash.bytes.data()),
                 hash.bytes.size());
}

void AppendString(absl::string_view value, std::string* output) {
  AppendU64(value.size(), output);
  output->append(value.data(), value.size());
}

void AppendOptionalHash(const std::optional<Hash256>& value,
                        std::string* output) {
  AppendU32(value.has_value() ? 1 : 0, output);
  if (value.has_value()) AppendHash(*value, output);
}

void AppendOptionalU64(const std::optional<uint64_t>& value,
                       std::string* output) {
  AppendU32(value.has_value() ? 1 : 0, output);
  if (value.has_value()) AppendU64(*value, output);
}

void AppendIdentity(const SessionHandoffIdentity& identity,
                    std::string* output) {
  AppendHash(identity.model_artifact_hash, output);
  AppendHash(identity.runtime_artifact_hash, output);
  AppendHash(identity.inference_profile_hash, output);
}

void AppendCapability(const SessionHandoffCapability& capability,
                      std::string* output) {
  AppendU32(capability.version, output);
  AppendHash(capability.capability_id, output);
  AppendIdentity(capability.session_identity, output);
  AppendHash(capability.exact_profile_id, output);
  AppendU32(static_cast<uint32_t>(capability.backend), output);
  AppendHash(capability.complete_state_inventory_hash, output);
  AppendHash(capability.capsule_codec_contract_hash, output);
}

void AppendCanonicalChunks(
    const std::vector<CapsuleCanonicalPrefillChunk>& chunks,
    std::string* output) {
  AppendU32(static_cast<uint32_t>(chunks.size()), output);
  for (const CapsuleCanonicalPrefillChunk& chunk : chunks) {
    AppendU32(static_cast<uint32_t>(chunk.encoding), output);
    switch (chunk.encoding) {
      case CapsuleCanonicalPrefillChunk::Encoding::kUtf8Text:
        AppendString(chunk.utf8_text, output);
        break;
      case CapsuleCanonicalPrefillChunk::Encoding::kExactTokenIds:
        AppendU32(static_cast<uint32_t>(chunk.token_ids.size()), output);
        for (int32_t token_id : chunk.token_ids) {
          AppendI32(token_id, output);
        }
        break;
    }
  }
}

void AppendPreparedPrefillPlan(const DPMPreparedPrefillPlan& plan,
                               std::string* output) {
  AppendU32(plan.format_version, output);
  AppendHash(plan.plan_id, output);
  AppendIdentity(plan.session_identity, output);
  AppendHash(plan.logical_agent_request_hash, output);
  AppendHash(plan.canonical_source_chunks_hash, output);
  AppendU32(static_cast<uint32_t>(plan.start_kind), output);
  AppendOptionalHash(plan.restore_checkpoint_id, output);
  AppendOptionalHash(plan.start_state_witness_id, output);
  AppendU64(plan.start_step, output);
  AppendU64(plan.start_history_token_count, output);
  AppendHash(plan.start_history_token_bytes_hash, output);
  AppendU32(plan.batch_size, output);
  AppendU32(plan.vocabulary_size, output);
  AppendU64(plan.end_step, output);
  AppendU32(static_cast<uint32_t>(plan.calls.size()), output);
  for (const DPMPreparedPrefillCall& call : plan.calls) {
    AppendU32(static_cast<uint32_t>(call.source_encoding), output);
    AppendHash(call.source_chunk_hash, output);
    AppendU64(call.start_token_position, output);
    AppendU64(call.end_token_position, output);
    AppendU32(static_cast<uint32_t>(call.token_segments.size()), output);
    for (const DPMPreparedPrefillSegment& segment : call.token_segments) {
      AppendU32(static_cast<uint32_t>(segment.token_ids.size()), output);
      for (int32_t token_id : segment.token_ids) {
        AppendI32(token_id, output);
      }
    }
  }
  AppendHash(plan.resolved_token_plan_hash, output);
  AppendHash(plan.shape_schedule_hash, output);
}

void AppendAuthority(const CapsuleRestoreAuthority& authority,
                     std::string* output) {
  AppendCapability(authority.capability, output);
  AppendHash(authority.admission_record_id, output);
  AppendHash(authority.coverage_id, output);
  AppendHash(authority.qualification_evidence_hash, output);
}

void AppendCheckpointState(const CapsuleDPMCheckpointState& state,
                           std::string* output) {
  AppendString(state.log_id, output);
  AppendU64(state.source_event_count, output);
  AppendHash(state.source_prefix_hash, output);
  AppendU64(state.response_event_index, output);
  AppendHash(state.projection_request_hash, output);
  AppendHash(state.projection_manifest_hash, output);
  AppendHash(state.correction_digest, output);
  AppendHash(state.agent_transcript_hash, output);
  AppendHash(state.logical_agent_request_hash, output);
}

void AppendRestoreTarget(const CapsuleDPMRestoreTarget& state,
                         std::string* output) {
  AppendString(state.log_id, output);
  AppendU64(state.source_event_count, output);
  AppendHash(state.source_prefix_hash, output);
  AppendU64(state.prospective_response_event_index, output);
  AppendHash(state.projection_request_hash, output);
  AppendHash(state.projection_manifest_hash, output);
  AppendHash(state.correction_digest, output);
  AppendHash(state.agent_transcript_prefix_hash, output);
  AppendHash(state.logical_agent_request_hash, output);
}

void AppendPrefillPlan(const CapsulePrefillPlan& plan, std::string* output) {
  AppendU32(static_cast<uint32_t>(plan.mode), output);
  AppendU64(plan.event_range_start, output);
  AppendU64(plan.event_range_end, output);
  AppendU32(plan.start_step, output);
  AppendU32(plan.end_step, output);
  AppendCanonicalChunks(plan.canonical_chunks, output);
  AppendHash(plan.canonical_full_prefill_chunks_hash, output);
  AppendHash(plan.canonical_delta_chunks_hash, output);
  AppendPreparedPrefillPlan(plan.prepared_plan, output);
}

std::string EncodeCapturePlanFields(const CapsuleCapturePlan& plan) {
  std::string encoded;
  AppendU32(plan.format_version, &encoded);
  AppendAuthority(plan.authority, &encoded);
  AppendU32(static_cast<uint32_t>(plan.capture_basis), &encoded);
  AppendCheckpointState(plan.checkpoint_state, &encoded);
  AppendHash(plan.agent_transcript_prefix_hash, &encoded);
  AppendPrefillPlan(plan.prefill, &encoded);
  AppendHash(plan.producing_output_evidence_hash, &encoded);
  AppendU32(plan.generated_token_count, &encoded);
  AppendU32(plan.capture_end_step, &encoded);
  AppendString(plan.checkpoint_authentication_key_id, &encoded);
  AppendOptionalHash(plan.parent_checkpoint_id, &encoded);
  AppendOptionalU64(plan.parent_response_event_index, &encoded);
  AppendOptionalHash(plan.parent_restore_evidence_id, &encoded);
  return encoded;
}

std::string EncodeRestorePlanFields(const CapsuleRestorePlan& plan) {
  std::string encoded;
  AppendU32(plan.format_version, &encoded);
  AppendAuthority(plan.authority, &encoded);
  AppendHash(plan.source_capture_plan_hash, &encoded);
  AppendHash(plan.source_capture_evidence_id, &encoded);
  AppendHash(plan.checkpoint_id, &encoded);
  AppendCheckpointState(plan.checkpoint_state, &encoded);
  AppendHash(plan.checkpoint_envelope_hash, &encoded);
  AppendU64(plan.checkpoint_envelope_size, &encoded);
  AppendString(plan.checkpoint_authentication_key_id, &encoded);
  AppendU32(plan.checkpoint_step, &encoded);
  AppendHash(plan.checkpoint_history_token_bytes_hash, &encoded);
  AppendRestoreTarget(plan.target_state, &encoded);
  AppendPrefillPlan(plan.prefill, &encoded);
  AppendU32(plan.maximum_output_tokens, &encoded);
  return encoded;
}

void AppendWitness(const SessionContinuationStateWitness& witness,
                   std::string* output) {
  AppendU32(witness.format_version, output);
  AppendHash(witness.witness_id, output);
  AppendIdentity(witness.session_identity, output);
  AppendU32(static_cast<uint32_t>(witness.phase), output);
  AppendI32(witness.current_step, output);
  AppendU32(witness.ran_decode ? 1 : 0, output);
  AppendHash(witness.processed_history_token_bytes_hash, output);
  AppendHash(witness.envelope_hash, output);
  AppendU64(witness.envelope_size, output);
  AppendString(witness.key_id, output);
}

void AppendReauthenticationEvidence(
    const SessionHandoffReauthenticationEvidence& evidence,
    std::string* output) {
  AppendU32(evidence.format_version, output);
  AppendHash(evidence.evidence_id, output);
  AppendIdentity(evidence.session_identity, output);
  AppendHash(evidence.canonical_continuation_state_hash, output);
  AppendHash(evidence.source_envelope_hash, output);
  AppendU64(evidence.source_envelope_size, output);
  AppendString(evidence.source_key_id, output);
  AppendHash(evidence.destination_envelope_hash, output);
  AppendU64(evidence.destination_envelope_size, output);
  AppendString(evidence.destination_key_id, output);
  AppendHash(evidence.capsule_codec_contract_hash, output);
  AppendHash(evidence.reauthentication_contract_hash, output);
  AppendString(evidence.purpose, output);
}

std::string EncodeRestoreEvidenceFields(
    const CapsuleRestoreEvidence& evidence) {
  std::string encoded;
  AppendU32(evidence.format_version, &encoded);
  AppendHash(evidence.plan.plan_hash, &encoded);
  encoded.append(EncodeRestorePlanFields(evidence.plan));
  AppendReauthenticationEvidence(evidence.durable_to_transient_reauthentication,
                                 &encoded);
  AppendWitness(evidence.target_post_import, &encoded);
  return encoded;
}

std::string EncodeCaptureEvidenceFields(
    const CapsuleCaptureEvidence& evidence) {
  std::string encoded;
  AppendU32(evidence.format_version, &encoded);
  AppendHash(evidence.plan.plan_hash, &encoded);
  encoded.append(EncodeCapturePlanFields(evidence.plan));
  AppendHash(evidence.checkpoint_id, &encoded);
  AppendHash(evidence.checkpoint_envelope_hash, &encoded);
  AppendU64(evidence.checkpoint_envelope_size, &encoded);
  AppendString(evidence.checkpoint_authentication_key_id, &encoded);
  AppendHash(evidence.checkpoint_history_token_bytes_hash, &encoded);
  AppendWitness(evidence.producer_before_export, &encoded);
  AppendWitness(evidence.producer_after_export, &encoded);
  AppendWitness(evidence.fresh_import_target, &encoded);
  AppendReauthenticationEvidence(evidence.transient_to_durable_reauthentication,
                                 &encoded);
  AppendU32(evidence.parent_restore_evidence.has_value() ? 1 : 0, &encoded);
  if (evidence.parent_restore_evidence.has_value()) {
    AppendHash(evidence.parent_restore_evidence->evidence_id, &encoded);
    encoded.append(
        EncodeRestoreEvidenceFields(*evidence.parent_restore_evidence));
  }
  return encoded;
}

absl::StatusOr<Hash256> HashCanonical(absl::string_view domain,
                                      absl::string_view canonical) {
  Sha256Hasher hasher;
  hasher.Update(domain);
  hasher.Update(canonical);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "CapsuleRestore evidence canonical hashing produced a zero digest.");
  }
  return result;
}

absl::StatusOr<Hash256> ComputeEmptyHistoryHash() {
  ABSL_ASSIGN_OR_RETURN(const std::string empty_history,
                        EncodeFreshWorkerTokenIds(std::vector<int32_t>()));
  Sha256Hasher hasher;
  const Hash256 result = hasher.OneShot(empty_history);
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "CapsuleRestore empty DPMTOK01 history produced a zero digest.");
  }
  return result;
}

absl::Status ValidatePublicKeyId(absl::string_view key_id) {
  if (key_id.empty() || key_id.size() > kMaximumCapsuleEvidenceKeyIdBytes ||
      !IsValidUtf8(key_id) || HasControlByte(key_id)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore evidence authentication key ID is invalid.");
  }
  return absl::OkStatus();
}

absl::Status ValidateLogId(absl::string_view log_id) {
  if (log_id.empty() || log_id.size() > kMaximumCapsuleEvidenceLogIdBytes ||
      !IsValidUtf8(log_id) || HasControlByte(log_id)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore evidence log ID must be bounded UTF-8 without "
        "control bytes.");
  }
  return absl::OkStatus();
}

absl::Status ValidateAuthority(const CapsuleRestoreAuthority& authority) {
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffCapability(authority.capability));
  if (!IsCompleteIdentity(authority.capability.session_identity) ||
      IsZeroHash(authority.admission_record_id) ||
      IsZeroHash(authority.coverage_id) ||
      IsZeroHash(authority.qualification_evidence_hash)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore requires the complete capability, admission, "
        "coverage, and qualification binding.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCheckpointState(const CapsuleDPMCheckpointState& state) {
  ABSL_RETURN_IF_ERROR(ValidateLogId(state.log_id));
  if (state.source_event_count == 0 ||
      state.response_event_index != state.source_event_count ||
      IsZeroHash(state.source_prefix_hash) ||
      IsZeroHash(state.projection_request_hash) ||
      IsZeroHash(state.projection_manifest_hash) ||
      IsZeroHash(state.correction_digest) ||
      IsZeroHash(state.agent_transcript_hash) ||
      IsZeroHash(state.logical_agent_request_hash)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore checkpoint state has incomplete log, projection, "
        "correction, transcript, or logical-request provenance.");
  }
  return absl::OkStatus();
}

absl::Status ValidateRestoreTarget(const CapsuleDPMRestoreTarget& state) {
  ABSL_RETURN_IF_ERROR(ValidateLogId(state.log_id));
  if (state.source_event_count == 0 ||
      state.prospective_response_event_index != state.source_event_count ||
      IsZeroHash(state.source_prefix_hash) ||
      IsZeroHash(state.projection_request_hash) ||
      IsZeroHash(state.projection_manifest_hash) ||
      IsZeroHash(state.correction_digest) ||
      IsZeroHash(state.agent_transcript_prefix_hash) ||
      IsZeroHash(state.logical_agent_request_hash)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore target has incomplete log, projection, correction, "
        "transcript-prefix, or logical-request provenance.");
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> ComputePreparedSourceChunkHash(
    const CapsuleCanonicalPrefillChunk& chunk) {
  switch (chunk.encoding) {
    case CapsuleCanonicalPrefillChunk::Encoding::kUtf8Text:
      return ComputeDPMPreparedPrefillUtf8SourceChunkHash(chunk.utf8_text);
    case CapsuleCanonicalPrefillChunk::Encoding::kExactTokenIds:
      return ComputeDPMPreparedPrefillExactTokenSourceChunkHash(
          chunk.token_ids);
    default:
      return absl::InvalidArgumentError(
          "CapsuleRestore canonical chunk encoding is unsupported.");
  }
}

absl::Status ValidatePreparedPlanAgainstCanonicalChunks(
    const DPMPreparedPrefillPlan& prepared,
    const std::vector<CapsuleCanonicalPrefillChunk>& chunks) {
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(prepared));
  if (prepared.calls.size() != chunks.size()) {
    return absl::FailedPreconditionError(
        "Prepared DPM prefill did not preserve one physical call per "
        "canonical source chunk.");
  }
  for (size_t index = 0; index < chunks.size(); ++index) {
    const CapsuleCanonicalPrefillChunk& chunk = chunks[index];
    const DPMPreparedPrefillCall& call = prepared.calls[index];
    ABSL_ASSIGN_OR_RETURN(const Hash256 expected_source_hash,
                          ComputePreparedSourceChunkHash(chunk));
    const DPMPreparedPrefillSourceEncoding expected_encoding =
        chunk.encoding == CapsuleCanonicalPrefillChunk::Encoding::kUtf8Text
            ? DPMPreparedPrefillSourceEncoding::kUtf8Text
            : DPMPreparedPrefillSourceEncoding::kExactTokenIds;
    const bool may_have_bos_segment =
        index == 0 &&
        prepared.start_kind == DPMPreparedPrefillStartKind::kFreshSession;
    if (call.source_encoding != expected_encoding ||
        call.source_chunk_hash != expected_source_hash ||
        call.token_segments.empty() || call.token_segments.size() > 2 ||
        (call.token_segments.size() == 2 && !may_have_bos_segment) ||
        (call.token_segments.size() == 2 &&
         call.token_segments.front().token_ids.size() != 1)) {
      return absl::FailedPreconditionError(
          "Prepared DPM prefill changed a source chunk or hid an unsupported "
          "call/BOS/token-buffer segment boundary.");
    }
    if (chunk.encoding ==
            CapsuleCanonicalPrefillChunk::Encoding::kExactTokenIds &&
        call.token_segments.back().token_ids != chunk.token_ids) {
      return absl::FailedPreconditionError(
          "Prepared DPM prefill changed the exact token-buffer segment for a "
          "canonical token chunk.");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateDecodedWitnessState(
    const SessionContinuationStateWitness& witness,
    const SessionHandoffIdentity& expected_identity, uint32_t expected_step,
    const Hash256& expected_history_hash) {
  ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(witness));
  if (witness.session_identity != expected_identity ||
      witness.phase != SessionHandoffPhase::kDecoded || !witness.ran_decode ||
      witness.current_step < 0 ||
      static_cast<uint32_t>(witness.current_step) != expected_step ||
      witness.processed_history_token_bytes_hash != expected_history_hash) {
    return absl::FailedPreconditionError(
        "CapsuleRestore decoded state witness does not match the exact "
        "identity, phase, own position, decode state, or history.");
  }
  return absl::OkStatus();
}

bool ContinuationStateProjectionMatches(
    const SessionContinuationStateWitness& left,
    const SessionContinuationStateWitness& right) {
  return left.session_identity == right.session_identity &&
         left.phase == right.phase && left.current_step == right.current_step &&
         left.ran_decode == right.ran_decode &&
         left.processed_history_token_bytes_hash ==
             right.processed_history_token_bytes_hash;
}

bool RestoreTargetMatchesCapture(const CapsuleDPMRestoreTarget& target,
                                 const CapsuleCapturePlan& capture) {
  return target.log_id == capture.checkpoint_state.log_id &&
         target.source_event_count ==
             capture.checkpoint_state.source_event_count &&
         target.source_prefix_hash ==
             capture.checkpoint_state.source_prefix_hash &&
         target.prospective_response_event_index ==
             capture.checkpoint_state.response_event_index &&
         target.projection_request_hash ==
             capture.checkpoint_state.projection_request_hash &&
         target.projection_manifest_hash ==
             capture.checkpoint_state.projection_manifest_hash &&
         target.correction_digest ==
             capture.checkpoint_state.correction_digest &&
         target.agent_transcript_prefix_hash ==
             capture.agent_transcript_prefix_hash &&
         target.logical_agent_request_hash ==
             capture.checkpoint_state.logical_agent_request_hash;
}

absl::Status ValidateCapturePlanFields(const CapsuleCapturePlan& plan,
                                       bool require_plan_hash);
absl::Status ValidateRestorePlanFields(const CapsuleRestorePlan& plan,
                                       bool require_plan_hash);

}  // namespace

absl::Status ValidateCapsuleCanonicalPrefillChunks(
    const std::vector<CapsuleCanonicalPrefillChunk>& chunks) {
  if (chunks.empty() || chunks.size() > kMaximumCapsuleEvidencePrefillChunks) {
    return absl::InvalidArgumentError(
        "CapsuleRestore canonical prefill requires a bounded nonempty chunk "
        "sequence.");
  }
  uint64_t text_bytes = 0;
  uint64_t token_ids = 0;
  for (const CapsuleCanonicalPrefillChunk& chunk : chunks) {
    switch (chunk.encoding) {
      case CapsuleCanonicalPrefillChunk::Encoding::kUtf8Text:
        if (chunk.utf8_text.empty() || !chunk.token_ids.empty() ||
            !IsValidUtf8(chunk.utf8_text)) {
          return absl::InvalidArgumentError(
              "CapsuleRestore canonical text chunk is empty, mixed, or not "
              "UTF-8.");
        }
        if (chunk.utf8_text.size() >
            kMaximumCapsuleEvidencePrefillTextBytes - text_bytes) {
          return absl::ResourceExhaustedError(
              "CapsuleRestore canonical prefill text exceeds its limit.");
        }
        text_bytes += chunk.utf8_text.size();
        break;
      case CapsuleCanonicalPrefillChunk::Encoding::kExactTokenIds:
        if (!chunk.utf8_text.empty() || chunk.token_ids.empty()) {
          return absl::InvalidArgumentError(
              "CapsuleRestore exact-token chunk is empty or mixed with "
              "text.");
        }
        if (chunk.token_ids.size() >
            kMaximumCapsuleEvidenceTokenIds - token_ids) {
          return absl::ResourceExhaustedError(
              "CapsuleRestore canonical token IDs exceed their limit.");
        }
        for (int32_t token_id : chunk.token_ids) {
          if (token_id < 0) {
            return absl::InvalidArgumentError(
                "CapsuleRestore canonical chunk contains a negative token "
                "ID.");
          }
        }
        token_ids += chunk.token_ids.size();
        break;
      default:
        return absl::InvalidArgumentError(
            "CapsuleRestore canonical chunk encoding is unsupported.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> ComputeCapsuleCanonicalFullPrefillChunksHash(
    const std::vector<CapsuleCanonicalPrefillChunk>& chunks) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCanonicalPrefillChunks(chunks));
  std::string canonical;
  AppendU32(kCapsuleRestoreEvidenceFormatVersion, &canonical);
  AppendCanonicalChunks(chunks, &canonical);
  return HashCanonical(kFullPrefillChunksDomain, canonical);
}

absl::StatusOr<Hash256> ComputeCapsuleCanonicalDeltaChunksHash(
    const std::vector<CapsuleCanonicalPrefillChunk>& chunks) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCanonicalPrefillChunks(chunks));
  std::string canonical;
  AppendU32(kCapsuleRestoreEvidenceFormatVersion, &canonical);
  AppendCanonicalChunks(chunks, &canonical);
  return HashCanonical(kDeltaChunksDomain, canonical);
}

absl::Status ValidateCapsulePrefillPlan(const CapsulePrefillPlan& plan) {
  if (plan.event_range_start >= plan.event_range_end ||
      plan.start_step >= plan.end_step ||
      plan.end_step > kMaximumCapsuleEvidenceTokenIds) {
    return absl::InvalidArgumentError(
        "CapsuleRestore prefill has an invalid event or own-position range.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleCanonicalPrefillChunks(plan.canonical_chunks));
  ABSL_RETURN_IF_ERROR(ValidatePreparedPlanAgainstCanonicalChunks(
      plan.prepared_plan, plan.canonical_chunks));
  if (plan.prepared_plan.start_step != plan.start_step ||
      plan.prepared_plan.end_step != plan.end_step ||
      plan.prepared_plan.start_history_token_count != plan.start_step) {
    return absl::FailedPreconditionError(
        "CapsuleRestore prefill and runtime-prepared plan name different exact "
        "own-position ranges or starting history counts.");
  }

  switch (plan.mode) {
    case CapsulePrefillMode::kFullCanonicalPrefill: {
      ABSL_ASSIGN_OR_RETURN(
          const Hash256 expected_chunks_hash,
          ComputeCapsuleCanonicalFullPrefillChunksHash(plan.canonical_chunks));
      ABSL_ASSIGN_OR_RETURN(const Hash256 empty_history_hash,
                            ComputeEmptyHistoryHash());
      if (plan.canonical_full_prefill_chunks_hash != expected_chunks_hash ||
          !IsZeroHash(plan.canonical_delta_chunks_hash) ||
          plan.prepared_plan.start_kind !=
              DPMPreparedPrefillStartKind::kFreshSession ||
          plan.prepared_plan.restore_checkpoint_id.has_value() ||
          plan.prepared_plan.start_state_witness_id.has_value() ||
          plan.prepared_plan.start_history_token_bytes_hash !=
              empty_history_hash) {
        return absl::DataLossError(
            "CapsuleRestore full-prefill plan has noncanonical or mixed chunk "
            "hashes or restore-only prepared state.");
      }
      break;
    }
    case CapsulePrefillMode::kOwnPositionCapsuleDelta: {
      ABSL_ASSIGN_OR_RETURN(
          const Hash256 expected_chunks_hash,
          ComputeCapsuleCanonicalDeltaChunksHash(plan.canonical_chunks));
      if (plan.canonical_delta_chunks_hash != expected_chunks_hash ||
          !IsZeroHash(plan.canonical_full_prefill_chunks_hash) ||
          plan.prepared_plan.start_kind !=
              DPMPreparedPrefillStartKind::kOwnPositionRestore ||
          !plan.prepared_plan.restore_checkpoint_id.has_value() ||
          !plan.prepared_plan.start_state_witness_id.has_value()) {
        return absl::DataLossError(
            "CapsuleRestore delta plan has noncanonical or mixed chunk "
            "hashes or lacks its prepared own-position state.");
      }
      break;
    }
    default:
      return absl::UnimplementedError(
          "CapsuleRestore prefill mode is unsupported; off-position grafting "
          "is not admitted.");
  }
  return absl::OkStatus();
}

namespace {

absl::Status ValidateCapturePlanFields(const CapsuleCapturePlan& plan,
                                       bool require_plan_hash) {
  if (plan.format_version != CapsuleCapturePlan::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Capsule capture plan version is unsupported.");
  }
  ABSL_RETURN_IF_ERROR(ValidateAuthority(plan.authority));
  ABSL_RETURN_IF_ERROR(ValidateCheckpointState(plan.checkpoint_state));
  ABSL_RETURN_IF_ERROR(ValidateCapsulePrefillPlan(plan.prefill));
  ABSL_RETURN_IF_ERROR(
      ValidatePublicKeyId(plan.checkpoint_authentication_key_id));
  if (IsZeroHash(plan.agent_transcript_prefix_hash) ||
      IsZeroHash(plan.producing_output_evidence_hash) ||
      plan.generated_token_count == 0 ||
      plan.generated_token_count > kMaximumCapsuleEvidenceGenerationTokens ||
      plan.capture_end_step > kMaximumCapsuleEvidenceTokenIds ||
      plan.prefill.end_step >
          (std::numeric_limits<uint32_t>::max)() - plan.generated_token_count ||
      plan.capture_end_step !=
          plan.prefill.end_step + plan.generated_token_count ||
      plan.prefill.event_range_end !=
          plan.checkpoint_state.source_event_count ||
      plan.prefill.prepared_plan.session_identity !=
          plan.authority.capability.session_identity ||
      plan.prefill.prepared_plan.logical_agent_request_hash !=
          plan.checkpoint_state.logical_agent_request_hash ||
      (require_plan_hash && IsZeroHash(plan.plan_hash))) {
    return absl::InvalidArgumentError(
        "Capsule capture plan has incomplete transcript/output provenance or "
        "inconsistent event, token, or step boundaries.");
  }

  const bool has_parent_checkpoint = plan.parent_checkpoint_id.has_value();
  const bool has_parent_response = plan.parent_response_event_index.has_value();
  const bool has_parent_restore = plan.parent_restore_evidence_id.has_value();
  if (has_parent_checkpoint != has_parent_response ||
      has_parent_checkpoint != has_parent_restore ||
      (has_parent_checkpoint &&
       (IsZeroHash(*plan.parent_checkpoint_id) ||
        IsZeroHash(*plan.parent_restore_evidence_id)))) {
    return absl::InvalidArgumentError(
        "Capsule capture parent checkpoint, position, and restore evidence "
        "must be present together and nonzero.");
  }

  switch (plan.capture_basis) {
    case CapsuleCaptureBasis::kRootFreshSession:
      if (plan.prefill.mode != CapsulePrefillMode::kFullCanonicalPrefill ||
          plan.prefill.start_step != 0 || has_parent_checkpoint) {
        return absl::FailedPreconditionError(
            "A root capsule capture requires a fresh-session full prefill of "
            "the current correction epoch and must not name parent restore "
            "evidence.");
      }
      break;
    case CapsuleCaptureBasis::kVerifiedParentRestore:
      if (plan.prefill.mode != CapsulePrefillMode::kOwnPositionCapsuleDelta ||
          plan.prefill.start_step == 0 || !has_parent_checkpoint ||
          *plan.parent_response_event_index ==
              (std::numeric_limits<uint64_t>::max)() ||
          *plan.parent_response_event_index + 1 !=
              plan.prefill.event_range_start ||
          *plan.parent_response_event_index >=
              plan.checkpoint_state.response_event_index ||
          plan.prefill.prepared_plan.restore_checkpoint_id !=
              plan.parent_checkpoint_id) {
        return absl::FailedPreconditionError(
            "A verified-parent capsule capture requires an exact own-position "
            "delta beginning immediately after the parent response.");
      }
      break;
    default:
      return absl::UnimplementedError("Capsule capture basis is unsupported.");
  }
  return absl::OkStatus();
}

absl::Status ValidateRestorePlanFields(const CapsuleRestorePlan& plan,
                                       bool require_plan_hash) {
  if (plan.format_version != CapsuleRestorePlan::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Capsule restore plan version is unsupported.");
  }
  ABSL_RETURN_IF_ERROR(ValidateAuthority(plan.authority));
  ABSL_RETURN_IF_ERROR(ValidateCheckpointState(plan.checkpoint_state));
  ABSL_RETURN_IF_ERROR(ValidateRestoreTarget(plan.target_state));
  ABSL_RETURN_IF_ERROR(ValidateCapsulePrefillPlan(plan.prefill));
  ABSL_RETURN_IF_ERROR(
      ValidatePublicKeyId(plan.checkpoint_authentication_key_id));

  if (IsZeroHash(plan.source_capture_plan_hash) ||
      IsZeroHash(plan.source_capture_evidence_id) ||
      IsZeroHash(plan.checkpoint_id) ||
      IsZeroHash(plan.checkpoint_envelope_hash) ||
      plan.checkpoint_envelope_size == 0 ||
      plan.checkpoint_envelope_size > kMaximumCapsuleEvidenceEnvelopeBytes ||
      plan.checkpoint_step == 0 ||
      plan.checkpoint_step > kMaximumCapsuleEvidenceTokenIds ||
      IsZeroHash(plan.checkpoint_history_token_bytes_hash) ||
      plan.maximum_output_tokens == 0 ||
      plan.maximum_output_tokens > kMaximumCapsuleEvidenceGenerationTokens ||
      plan.prefill.mode != CapsulePrefillMode::kOwnPositionCapsuleDelta ||
      plan.prefill.start_step != plan.checkpoint_step ||
      plan.prefill.prepared_plan.session_identity !=
          plan.authority.capability.session_identity ||
      plan.prefill.prepared_plan.logical_agent_request_hash !=
          plan.target_state.logical_agent_request_hash ||
      plan.prefill.prepared_plan.restore_checkpoint_id !=
          std::optional<Hash256>(plan.checkpoint_id) ||
      plan.prefill.prepared_plan.start_history_token_bytes_hash !=
          plan.checkpoint_history_token_bytes_hash ||
      plan.checkpoint_state.log_id != plan.target_state.log_id ||
      plan.checkpoint_state.correction_digest !=
          plan.target_state.correction_digest ||
      plan.target_state.source_event_count <=
          plan.checkpoint_state.response_event_index ||
      plan.target_state.source_prefix_hash ==
          plan.checkpoint_state.source_prefix_hash ||
      plan.checkpoint_state.response_event_index ==
          (std::numeric_limits<uint64_t>::max)() ||
      plan.prefill.event_range_start !=
          plan.checkpoint_state.response_event_index + 1 ||
      plan.prefill.event_range_end != plan.target_state.source_event_count ||
      (require_plan_hash && IsZeroHash(plan.plan_hash))) {
    return absl::FailedPreconditionError(
        "Capsule restore plan is incomplete or is not an exact same-epoch "
        "own-position checkpoint-plus-delta plan.");
  }
  return absl::OkStatus();
}

absl::Status ValidateReauthentication(
    const SessionHandoffReauthenticationEvidence& reauthentication,
    const CapsuleRestoreAuthority& authority,
    absl::string_view expected_purpose) {
  ABSL_RETURN_IF_ERROR(
      ValidateSessionHandoffReauthenticationEvidence(reauthentication));
  ABSL_RETURN_IF_ERROR(ValidatePublicKeyId(reauthentication.source_key_id));
  ABSL_RETURN_IF_ERROR(
      ValidatePublicKeyId(reauthentication.destination_key_id));
  if (reauthentication.session_identity !=
          authority.capability.session_identity ||
      reauthentication.capsule_codec_contract_hash !=
          authority.capability.capsule_codec_contract_hash ||
      reauthentication.capsule_codec_contract_hash !=
          GetSessionHandoffCapsuleCodecContractHash() ||
      reauthentication.reauthentication_contract_hash !=
          GetSessionHandoffReauthenticationEvidenceContractHash() ||
      reauthentication.purpose != expected_purpose) {
    return absl::FailedPreconditionError(
        "CapsuleRestore reauthentication evidence belongs to another "
        "runtime, capsule/state-commitment contract, or product direction.");
  }
  return absl::OkStatus();
}

bool WitnessEnvelopeMatchesReauthenticationSource(
    const SessionContinuationStateWitness& witness,
    const SessionHandoffReauthenticationEvidence& reauthentication) {
  return witness.envelope_hash == reauthentication.source_envelope_hash &&
         witness.envelope_size == reauthentication.source_envelope_size &&
         witness.key_id == reauthentication.source_key_id;
}

bool WitnessEnvelopeMatchesReauthenticationDestination(
    const SessionContinuationStateWitness& witness,
    const SessionHandoffReauthenticationEvidence& reauthentication) {
  return witness.envelope_hash == reauthentication.destination_envelope_hash &&
         witness.envelope_size == reauthentication.destination_envelope_size &&
         witness.key_id == reauthentication.destination_key_id;
}

absl::Status ValidateRestoreEvidenceFields(
    const CapsuleRestoreEvidence& evidence, bool require_evidence_id) {
  if (evidence.format_version != CapsuleRestoreEvidence::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Capsule restore evidence version is unsupported.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestorePlan(evidence.plan));
  const SessionHandoffReauthenticationEvidence& reauthentication =
      evidence.durable_to_transient_reauthentication;
  ABSL_RETURN_IF_ERROR(ValidateReauthentication(
      reauthentication, evidence.plan.authority,
      kFreshWorkerDurableRestoreToTransientReauthenticationPurpose));
  ABSL_RETURN_IF_ERROR(ValidateDecodedWitnessState(
      evidence.target_post_import,
      evidence.plan.authority.capability.session_identity,
      evidence.plan.checkpoint_step,
      evidence.plan.checkpoint_history_token_bytes_hash));
  if (reauthentication.source_envelope_hash !=
          evidence.plan.checkpoint_envelope_hash ||
      reauthentication.source_envelope_size !=
          evidence.plan.checkpoint_envelope_size ||
      reauthentication.source_key_id !=
          evidence.plan.checkpoint_authentication_key_id ||
      reauthentication.destination_key_id !=
          kFreshWorkerTransientRestoreKeyId ||
      !WitnessEnvelopeMatchesReauthenticationDestination(
          evidence.target_post_import, reauthentication)) {
    return absl::FailedPreconditionError(
        "Capsule restore evidence does not join the exact durable "
        "checkpoint source to the transient envelope imported by its live "
        "target.");
  }
  if (evidence.plan.prefill.prepared_plan.start_state_witness_id !=
      std::optional<Hash256>(evidence.target_post_import.witness_id)) {
    return absl::FailedPreconditionError(
        "Capsule restore prepared prefill is not bound to the actual "
        "transient post-import target witness.");
  }
  if (require_evidence_id && IsZeroHash(evidence.evidence_id)) {
    return absl::InvalidArgumentError("Capsule restore evidence ID is absent.");
  }
  return absl::OkStatus();
}

absl::Status ValidateParentRestoreLink(
    const CapsuleCaptureEvidence& child,
    const CapsuleRestoreEvidence& parent_restore) {
  const CapsuleCapturePlan& capture = child.plan;
  const CapsuleRestorePlan& restore = parent_restore.plan;
  if (!capture.parent_checkpoint_id.has_value() ||
      !capture.parent_response_event_index.has_value() ||
      !capture.parent_restore_evidence_id.has_value() ||
      *capture.parent_checkpoint_id != restore.checkpoint_id ||
      *capture.parent_response_event_index !=
          restore.checkpoint_state.response_event_index ||
      *capture.parent_restore_evidence_id != parent_restore.evidence_id ||
      capture.authority != restore.authority ||
      !RestoreTargetMatchesCapture(restore.target_state, capture) ||
      capture.prefill != restore.prefill ||
      capture.prefill.start_step != restore.checkpoint_step ||
      capture.prefill.prepared_plan.start_state_witness_id !=
          std::optional<Hash256>(
              parent_restore.target_post_import.witness_id) ||
      capture.prefill.prepared_plan.start_history_token_bytes_hash !=
          parent_restore.target_post_import
              .processed_history_token_bytes_hash ||
      capture.generated_token_count > restore.maximum_output_tokens ||
      parent_restore.target_post_import.current_step < 0 ||
      static_cast<uint32_t>(parent_restore.target_post_import.current_step) !=
          capture.prefill.start_step) {
    return absl::FailedPreconditionError(
        "Recursive capsule capture is not bound to the exact verified "
        "parent restore, transient target witness, own position, and delta "
        "plan.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCaptureEvidenceFields(
    const CapsuleCaptureEvidence& evidence, bool require_evidence_id) {
  if (evidence.format_version != CapsuleCaptureEvidence::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Capsule capture evidence version is unsupported.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCapturePlan(evidence.plan));
  ABSL_RETURN_IF_ERROR(
      ValidatePublicKeyId(evidence.checkpoint_authentication_key_id));
  if (IsZeroHash(evidence.checkpoint_id) ||
      IsZeroHash(evidence.checkpoint_envelope_hash) ||
      evidence.checkpoint_envelope_size == 0 ||
      evidence.checkpoint_envelope_size >
          kMaximumCapsuleEvidenceEnvelopeBytes ||
      IsZeroHash(evidence.checkpoint_history_token_bytes_hash) ||
      evidence.checkpoint_authentication_key_id !=
          evidence.plan.checkpoint_authentication_key_id ||
      evidence.checkpoint_authentication_key_id ==
          kFreshWorkerTransientRestoreKeyId ||
      evidence.checkpoint_authentication_key_id ==
          kFreshWorkerTransientProducingKeyId ||
      (require_evidence_id && IsZeroHash(evidence.evidence_id))) {
    return absl::InvalidArgumentError(
        "Capsule capture evidence has incomplete checkpoint, envelope, "
        "key, history, or content-address metadata.");
  }

  const SessionHandoffIdentity& identity =
      evidence.plan.authority.capability.session_identity;
  ABSL_RETURN_IF_ERROR(ValidateDecodedWitnessState(
      evidence.producer_before_export, identity, evidence.plan.capture_end_step,
      evidence.checkpoint_history_token_bytes_hash));
  ABSL_RETURN_IF_ERROR(ValidateDecodedWitnessState(
      evidence.producer_after_export, identity, evidence.plan.capture_end_step,
      evidence.checkpoint_history_token_bytes_hash));
  ABSL_RETURN_IF_ERROR(ValidateDecodedWitnessState(
      evidence.fresh_import_target, identity, evidence.plan.capture_end_step,
      evidence.checkpoint_history_token_bytes_hash));
  if (evidence.producer_before_export != evidence.producer_after_export ||
      evidence.producer_before_export != evidence.fresh_import_target) {
    return absl::DataLossError(
        "Capsule capture transient producer exports and fresh-target "
        "import/re-export witnesses are not canonically equal.");
  }

  const SessionHandoffReauthenticationEvidence& reauthentication =
      evidence.transient_to_durable_reauthentication;
  ABSL_RETURN_IF_ERROR(ValidateReauthentication(
      reauthentication, evidence.plan.authority,
      kFreshWorkerTransientProducingToDurableReauthenticationPurpose));
  if (!WitnessEnvelopeMatchesReauthenticationSource(
          evidence.producer_before_export, reauthentication) ||
      reauthentication.source_key_id != kFreshWorkerTransientProducingKeyId ||
      reauthentication.destination_envelope_hash !=
          evidence.checkpoint_envelope_hash ||
      reauthentication.destination_envelope_size !=
          evidence.checkpoint_envelope_size ||
      reauthentication.destination_key_id !=
          evidence.checkpoint_authentication_key_id) {
    return absl::FailedPreconditionError(
        "Capsule capture evidence does not join the exact transient "
        "producer envelope to the durable checkpoint endpoint.");
  }

  switch (evidence.plan.capture_basis) {
    case CapsuleCaptureBasis::kRootFreshSession:
      if (evidence.parent_restore_evidence.has_value()) {
        return absl::FailedPreconditionError(
            "A root capsule capture must not carry parent restore "
            "evidence.");
      }
      break;
    case CapsuleCaptureBasis::kVerifiedParentRestore:
      if (!evidence.parent_restore_evidence.has_value()) {
        return absl::FailedPreconditionError(
            "A verified-parent capsule capture omitted its parent "
            "restore evidence.");
      }
      ABSL_RETURN_IF_ERROR(
          ValidateCapsuleRestoreEvidence(*evidence.parent_restore_evidence));
      ABSL_RETURN_IF_ERROR(ValidateParentRestoreLink(
          evidence, *evidence.parent_restore_evidence));
      break;
    default:
      return absl::UnimplementedError(
          "Capsule capture evidence basis is unsupported.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Hash256> ComputeCapsuleCapturePlanHash(
    const CapsuleCapturePlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateCapturePlanFields(plan, false));
  return HashCanonical(kCapturePlanDomain, EncodeCapturePlanFields(plan));
}

absl::Status ValidateCapsuleCapturePlan(const CapsuleCapturePlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateCapturePlanFields(plan, true));
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_hash,
                        ComputeCapsuleCapturePlanHash(plan));
  if (plan.plan_hash != expected_hash) {
    return absl::DataLossError("Capsule capture plan hash is not canonical.");
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> ComputeCapsuleRestorePlanHash(
    const CapsuleRestorePlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateRestorePlanFields(plan, false));
  return HashCanonical(kRestorePlanDomain, EncodeRestorePlanFields(plan));
}

absl::Status ValidateCapsuleRestorePlan(const CapsuleRestorePlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateRestorePlanFields(plan, true));
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_hash,
                        ComputeCapsuleRestorePlanHash(plan));
  if (plan.plan_hash != expected_hash) {
    return absl::DataLossError("Capsule restore plan hash is not canonical.");
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> ComputeCapsuleRestoreEvidenceId(
    const CapsuleRestoreEvidence& evidence) {
  ABSL_RETURN_IF_ERROR(ValidateRestoreEvidenceFields(evidence, false));
  return HashCanonical(kRestoreEvidenceDomain,
                       EncodeRestoreEvidenceFields(evidence));
}

absl::Status ValidateCapsuleRestoreEvidence(
    const CapsuleRestoreEvidence& evidence) {
  ABSL_RETURN_IF_ERROR(ValidateRestoreEvidenceFields(evidence, true));
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_id,
                        ComputeCapsuleRestoreEvidenceId(evidence));
  if (evidence.evidence_id != expected_id) {
    return absl::DataLossError("Capsule restore evidence ID is not canonical.");
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> ComputeCapsuleCaptureEvidenceId(
    const CapsuleCaptureEvidence& evidence) {
  ABSL_RETURN_IF_ERROR(ValidateCaptureEvidenceFields(evidence, false));
  return HashCanonical(kCaptureEvidenceDomain,
                       EncodeCaptureEvidenceFields(evidence));
}

absl::Status ValidateCapsuleCaptureEvidence(
    const CapsuleCaptureEvidence& evidence) {
  ABSL_RETURN_IF_ERROR(ValidateCaptureEvidenceFields(evidence, true));
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_id,
                        ComputeCapsuleCaptureEvidenceId(evidence));
  if (evidence.evidence_id != expected_id) {
    return absl::DataLossError("Capsule capture evidence ID is not canonical.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCapsuleRestoreEvidenceForSourceCapture(
    const CapsuleRestoreEvidence& restore_evidence,
    const CapsuleCaptureEvidence& source_capture_evidence) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidence(restore_evidence));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidence(source_capture_evidence));
  const CapsuleRestorePlan& restore = restore_evidence.plan;
  const CapsuleCaptureEvidence& capture = source_capture_evidence;
  if (restore.authority != capture.plan.authority ||
      restore.source_capture_plan_hash != capture.plan.plan_hash ||
      restore.source_capture_evidence_id != capture.evidence_id ||
      restore.checkpoint_id != capture.checkpoint_id ||
      restore.checkpoint_state != capture.plan.checkpoint_state ||
      restore.checkpoint_envelope_hash != capture.checkpoint_envelope_hash ||
      restore.checkpoint_envelope_size != capture.checkpoint_envelope_size ||
      restore.checkpoint_authentication_key_id !=
          capture.checkpoint_authentication_key_id ||
      restore.checkpoint_step != capture.plan.capture_end_step ||
      restore.checkpoint_history_token_bytes_hash !=
          capture.checkpoint_history_token_bytes_hash) {
    return absl::FailedPreconditionError(
        "Capsule restore does not match the complete authorized source "
        "capture, durable checkpoint, own position, and runtime authority.");
  }

  const SessionHandoffReauthenticationEvidence& capture_reauthentication =
      capture.transient_to_durable_reauthentication;
  const SessionHandoffReauthenticationEvidence& restore_reauthentication =
      restore_evidence.durable_to_transient_reauthentication;
  if (capture_reauthentication.destination_envelope_hash !=
          restore_reauthentication.source_envelope_hash ||
      capture_reauthentication.destination_envelope_size !=
          restore_reauthentication.source_envelope_size ||
      capture_reauthentication.destination_key_id !=
          restore_reauthentication.source_key_id ||
      capture_reauthentication.canonical_continuation_state_hash !=
          restore_reauthentication.canonical_continuation_state_hash ||
      capture_reauthentication.session_identity !=
          restore_reauthentication.session_identity ||
      capture_reauthentication.capsule_codec_contract_hash !=
          restore_reauthentication.capsule_codec_contract_hash ||
      capture_reauthentication.reauthentication_contract_hash !=
          restore_reauthentication.reauthentication_contract_hash) {
    return absl::FailedPreconditionError(
        "Capsule restore does not prove the exact durable endpoint and "
        "complete canonical continuation state produced by its source "
        "capture.");
  }
  if (!ContinuationStateProjectionMatches(restore_evidence.target_post_import,
          capture.fresh_import_target)) {
    return absl::FailedPreconditionError(
        "Capsule restore target and source capture disagree on the "
        "continuation identity, decoded phase, own position, decode state, "
        "or processed history.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCapsuleCaptureEvidenceForParentCapture(
    const CapsuleCaptureEvidence& child_capture_evidence,
    const CapsuleCaptureEvidence& parent_capture_evidence) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidence(child_capture_evidence));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidence(parent_capture_evidence));
  if (child_capture_evidence.plan.capture_basis !=
          CapsuleCaptureBasis::kVerifiedParentRestore ||
      !child_capture_evidence.parent_restore_evidence.has_value()) {
    return absl::FailedPreconditionError(
        "Parent-capture validation requires a verified-parent "
        "descendant.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidenceForSourceCapture(
      *child_capture_evidence.parent_restore_evidence,
      parent_capture_evidence));
  return ValidateParentRestoreLink(
      child_capture_evidence, *child_capture_evidence.parent_restore_evidence);
}

Hash256 GetCapsuleRestoreCaptureEvidenceContractHash() {
  std::string canonical;
  const auto append_frame = [&canonical](absl::string_view value) {
    AppendU32(static_cast<uint32_t>(value.size()), &canonical);
    canonical.append(value.data(), value.size());
  };
  AppendU32(kCapsuleRestoreEvidenceFormatVersion, &canonical);
  AppendU64(kMaximumCapsuleEvidenceEnvelopeBytes, &canonical);
  AppendU32(kMaximumCapsuleEvidenceTokenIds, &canonical);
  AppendHash(GetSessionContinuationStateWitnessContractHash(), &canonical);
  AppendHash(GetSessionHandoffCapsuleCodecContractHash(), &canonical);
  AppendHash(GetSessionHandoffReauthenticationEvidenceContractHash(),
             &canonical);
  append_frame(kFreshWorkerTransientProducingToDurableReauthenticationPurpose);
  append_frame(kFreshWorkerTransientProducingKeyId);
  append_frame("LOGICAL_AND_RUNTIME_PREPARED_CAPTURE_PLAN");
  append_frame("ROOT_FULL_PREFILL_CURRENT_CORRECTION_EPOCH");
  append_frame("THREE_EQUAL_TRANSIENT_PRODUCER_WITNESSES");
  append_frame("EXACT_TRANSIENT_SOURCE_TO_DURABLE_DESTINATION_ENDPOINTS");
  append_frame("REWRAP_INVARIANT_COMPLETE_CONTINUATION_STATE");
  append_frame("EDGE_VALIDATED_RECURSIVE_ANCESTRY");
  Sha256Hasher hasher;
  hasher.Update(kCaptureEvidenceContractDomain);
  hasher.Update(canonical);
  return hasher.Finalize();
}

Hash256 GetCapsuleRestoreRestoreEvidenceContractHash() {
  std::string canonical;
  const auto append_frame = [&canonical](absl::string_view value) {
    AppendU32(static_cast<uint32_t>(value.size()), &canonical);
    canonical.append(value.data(), value.size());
  };
  AppendU32(kCapsuleRestoreEvidenceFormatVersion, &canonical);
  AppendU64(kMaximumCapsuleEvidenceEnvelopeBytes, &canonical);
  AppendU32(kMaximumCapsuleEvidenceTokenIds, &canonical);
  AppendHash(GetSessionContinuationStateWitnessContractHash(), &canonical);
  AppendHash(GetSessionHandoffCapsuleCodecContractHash(), &canonical);
  AppendHash(GetSessionHandoffReauthenticationEvidenceContractHash(),
             &canonical);
  append_frame(kFreshWorkerDurableRestoreToTransientReauthenticationPurpose);
  append_frame(kFreshWorkerTransientRestoreKeyId);
  append_frame("LOGICAL_AND_RUNTIME_PREPARED_RESTORE_PLAN");
  append_frame("EXACT_DURABLE_SOURCE_TO_TRANSIENT_DESTINATION_ENDPOINTS");
  append_frame("TRANSIENT_POST_IMPORT_TARGET_WITNESS");
  append_frame("SOURCE_CAPTURE_DURABLE_ENDPOINT_AND_STATE_HASH_EQUALITY");
  append_frame("OFF_POSITION_GRAFT_UNREPRESENTABLE");
  Sha256Hasher hasher;
  hasher.Update(kRestoreEvidenceContractDomain);
  hasher.Update(canonical);
  return hasher.Finalize();
}

}  // namespace litert::lm
