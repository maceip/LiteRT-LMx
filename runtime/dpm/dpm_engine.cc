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

#include "runtime/dpm/dpm_engine.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/core/session_handoff_codec.h"
#include "runtime/dpm/capsule_restore_evidence_binding.h"
#include "runtime/dpm/correction_digest.h"
#include "runtime/dpm/dpm_agent_replay_runtime.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_receipt_validation.h"
#include "runtime/dpm/fresh_worker_process.h"
#include "runtime/platform/hash/hmac_sha256.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/util/byte_stream.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kTranscriptDomain = "DPM_AGENT_TRANSCRIPT";
constexpr absl::string_view kAgentRequestDomain = "DPM_AGENT_REQUEST";
constexpr absl::string_view kWinnerRestoreTransientKeyDomain =
    "LITERT_LMX_DPM_WINNER_RESTORE_TRANSIENT_KEY_HMAC_SHA256";
constexpr absl::string_view kWinnerCaptureTransientKeyDomain =
    "LITERT_LMX_DPM_WINNER_CAPTURE_TRANSIENT_KEY_HMAC_SHA256";
constexpr size_t kMaximumCheckpointKeyIdSize = 1024;

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
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

bool ContainsControlByte(absl::string_view text) {
  for (unsigned char byte : text) {
    if (byte < 0x20 || byte == 0x7f) return true;
  }
  return false;
}

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

void UpdateU64(uint64_t value, Sha256Hasher* hasher) {
  std::array<char, 8> encoded{};
  for (int i = 7; i >= 0; --i) {
    encoded[7 - i] = static_cast<char>((value >> (i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(encoded.data(), encoded.size()));
}

void UpdateU32(uint32_t value, Sha256Hasher* hasher) {
  std::array<char, 4> encoded{};
  for (int i = 3; i >= 0; --i) {
    encoded[3 - i] = static_cast<char>((value >> (i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(encoded.data(), encoded.size()));
}

void UpdateBytesFrame(char tag, absl::string_view bytes, Sha256Hasher* hasher) {
  hasher->Update(absl::string_view(&tag, 1));
  UpdateU64(bytes.size(), hasher);
  hasher->Update(bytes);
}

void UpdateTokenFrame(const std::vector<int>& token_ids, Sha256Hasher* hasher) {
  constexpr char kTag = 'T';
  hasher->Update(absl::string_view(&kTag, 1));
  UpdateU64(token_ids.size(), hasher);
  for (int token_id : token_ids) {
    UpdateU32(static_cast<uint32_t>(static_cast<int32_t>(token_id)), hasher);
  }
}

absl::string_view HashBytes(const Hash256& hash) {
  return absl::string_view(reinterpret_cast<const char*>(hash.bytes.data()),
                           hash.bytes.size());
}

std::string DeriveRequestTransientKey(absl::string_view parent_key,
                                      absl::string_view domain,
    const Hash256& logical_agent_request_hash,
                                      const Hash256& source_checkpoint_id,
                                      absl::string_view operation_id) {
  const Hash256 derived =
      HmacSha256(parent_key, {domain, HashBytes(logical_agent_request_hash),
       HashBytes(source_checkpoint_id), operation_id});
  return std::string(reinterpret_cast<const char*>(derived.bytes.data()),
                     derived.bytes.size());
}

bool SameCapsuleRestoreRuntimeAuthority(
    const AuthenticatedCapsuleRestoreAdmission& left_admission,
    const CapsuleRestoreAuthority& left_authority,
    const AuthenticatedCapsuleRestoreAdmission& right_admission,
    const CapsuleRestoreAuthority& right_authority) {
  return left_authority == right_authority &&
         left_admission.record.record_id == right_admission.record.record_id &&
         left_admission.profile == right_admission.profile &&
         left_admission.capability == right_admission.capability &&
         left_admission.operational_coverage ==
             right_admission.operational_coverage;
}

absl::StatusOr<std::vector<CapsuleCanonicalPrefillChunk>>
ToCapsuleCanonicalPrefillChunks(
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& chunks) {
  std::vector<CapsuleCanonicalPrefillChunk> converted;
  converted.reserve(chunks.size());
  for (const DPMAgentGenerationRequest::PrefillChunk& chunk : chunks) {
    CapsuleCanonicalPrefillChunk converted_chunk;
    switch (chunk.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text:
        if (!chunk.token_ids.empty()) {
          return absl::InvalidArgumentError(
              "DPM text prefill chunk also contains token IDs.");
        }
        converted_chunk.encoding =
            CapsuleCanonicalPrefillChunk::Encoding::kUtf8Text;
        converted_chunk.utf8_text = chunk.text;
        break;
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds:
        if (!chunk.text.empty()) {
          return absl::InvalidArgumentError(
              "DPM token prefill chunk also contains UTF-8 text.");
        }
        converted_chunk.encoding =
            CapsuleCanonicalPrefillChunk::Encoding::kExactTokenIds;
        converted_chunk.token_ids.reserve(chunk.token_ids.size());
        for (int token_id : chunk.token_ids) {
          if (token_id < 0 || static_cast<uint64_t>(token_id) >
                  static_cast<uint64_t>(
                      (std::numeric_limits<int32_t>::max)())) {
            return absl::InvalidArgumentError(
                "DPM prefill contains a token outside the capsule evidence "
                "range.");
          }
          converted_chunk.token_ids.push_back(static_cast<int32_t>(token_id));
        }
        break;
      default:
        return absl::InvalidArgumentError(
            "DPM prefill contains an unknown canonical encoding.");
    }
    converted.push_back(std::move(converted_chunk));
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCanonicalPrefillChunks(converted));
  return converted;
}

absl::StatusOr<CapsulePrefillPlan> BuildCapsulePrefillPlan(
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& chunks,
    const DPMPreparedPrefillPlan& prepared_plan, uint64_t event_range_start,
    uint64_t event_range_end) {
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(prepared_plan));
  if (prepared_plan.start_step > (std::numeric_limits<uint32_t>::max)() ||
      prepared_plan.end_step > (std::numeric_limits<uint32_t>::max)()) {
    return absl::ResourceExhaustedError(
        "DPM prepared prefill positions exceed the capsule evidence domain.");
  }
  CapsulePrefillPlan plan;
  plan.event_range_start = event_range_start;
  plan.event_range_end = event_range_end;
  plan.start_step = static_cast<uint32_t>(prepared_plan.start_step);
  plan.end_step = static_cast<uint32_t>(prepared_plan.end_step);
  ABSL_ASSIGN_OR_RETURN(plan.canonical_chunks,
                        ToCapsuleCanonicalPrefillChunks(chunks));
  plan.prepared_plan = prepared_plan;
  switch (prepared_plan.start_kind) {
    case DPMPreparedPrefillStartKind::kFreshSession: {
      plan.mode = CapsulePrefillMode::kFullCanonicalPrefill;
      ABSL_ASSIGN_OR_RETURN(
          plan.canonical_full_prefill_chunks_hash,
          ComputeCapsuleCanonicalFullPrefillChunksHash(plan.canonical_chunks));
      break;
    }
    case DPMPreparedPrefillStartKind::kOwnPositionRestore: {
      plan.mode = CapsulePrefillMode::kOwnPositionCapsuleDelta;
      ABSL_ASSIGN_OR_RETURN(
          plan.canonical_delta_chunks_hash,
          ComputeCapsuleCanonicalDeltaChunksHash(plan.canonical_chunks));
      break;
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsulePrefillPlan(plan));
  return plan;
}

uint64_t ActiveCorrectionEpochStart(const DPMLogSnapshot& snapshot,
                                    uint64_t event_count) {
  uint64_t start = 0;
  for (const DPMEvent& event : snapshot.events) {
    if (event.index >= event_count) break;
    if (event.kind == DPMEvent::Kind::kCorrection &&
        event.index != (std::numeric_limits<uint64_t>::max)()) {
      start = event.index + 1;
    }
  }
  return start;
}

uint64_t ActiveCorrectionEpochStart(const DPMLogSnapshot& snapshot) {
  return ActiveCorrectionEpochStart(snapshot, snapshot.events.size());
}

absl::StatusOr<CapsuleRestoreEvidence> BuildExactCapsuleRestoreEvidence(
    const DPMSessionCheckpointArtifact& source_artifact,
    const CapsuleCaptureEvidence& source_capture_evidence,
    const CapsuleRestoreAuthority& current_authority,
    const CapsuleDPMRestoreTarget& target_state,
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& delta_chunks,
    const DPMPreparedPrefillPlan& prepared_prefill_plan,
    const SessionHandoffReauthenticationEvidence& reauthentication,
    const SessionContinuationStateWitness& restored_state_witness,
    uint32_t maximum_output_tokens) {
  ABSL_RETURN_IF_ERROR(ValidateDPMSessionCheckpointArtifact(source_artifact));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidence(source_capture_evidence));
  if (source_capture_evidence.checkpoint_id !=
          source_artifact.descriptor.descriptor_id ||
      prepared_prefill_plan.start_kind !=
          DPMPreparedPrefillStartKind::kOwnPositionRestore ||
      prepared_prefill_plan.restore_checkpoint_id !=
          std::optional<Hash256>(source_artifact.descriptor.descriptor_id) ||
      source_artifact.descriptor.response_event_index ==
          (std::numeric_limits<uint64_t>::max)()) {
    return absl::FailedPreconditionError(
        "Exact DPM restore inputs do not describe one own-position source "
        "checkpoint and prepared delta.");
  }
  ABSL_ASSIGN_OR_RETURN(CapsulePrefillPlan prefill,
                        BuildCapsulePrefillPlan(
          delta_chunks, prepared_prefill_plan,
          source_artifact.descriptor.response_event_index + 1,
          target_state.source_event_count));
  CapsuleRestorePlan plan;
  plan.authority = current_authority;
  plan.source_capture_plan_hash = source_capture_evidence.plan.plan_hash;
  plan.source_capture_evidence_id = source_capture_evidence.evidence_id;
  plan.checkpoint_id = source_artifact.descriptor.descriptor_id;
  plan.checkpoint_state = source_capture_evidence.plan.checkpoint_state;
  plan.checkpoint_envelope_hash = source_artifact.descriptor.envelope_hash;
  plan.checkpoint_envelope_size = source_artifact.descriptor.envelope_size;
  plan.checkpoint_authentication_key_id = source_artifact.descriptor.key_id;
  plan.checkpoint_step = source_capture_evidence.plan.capture_end_step;
  plan.checkpoint_history_token_bytes_hash =
      source_capture_evidence.checkpoint_history_token_bytes_hash;
  plan.target_state = target_state;
  plan.prefill = std::move(prefill);
  plan.maximum_output_tokens = maximum_output_tokens;
  ABSL_ASSIGN_OR_RETURN(plan.plan_hash, ComputeCapsuleRestorePlanHash(plan));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestorePlan(plan));

  CapsuleRestoreEvidence evidence;
  evidence.plan = std::move(plan);
  evidence.durable_to_transient_reauthentication = reauthentication;
  evidence.target_post_import = restored_state_witness;
  ABSL_ASSIGN_OR_RETURN(evidence.evidence_id,
                        ComputeCapsuleRestoreEvidenceId(evidence));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidence(evidence));
  return evidence;
}

Hash256 SnapshotDigest(const Sha256Hasher& hasher) {
  Sha256Hasher copy = hasher;
  return copy.Finalize();
}

struct TranscriptCursor {
  std::optional<Hash256> correction_digest;
  Sha256Hasher hasher;
};

void ResetTranscriptHasher(const Hash256& correction_digest,
                           TranscriptCursor* cursor) {
  cursor->correction_digest = correction_digest;
  cursor->hasher = Sha256Hasher();
  cursor->hasher.Update(kTranscriptDomain);
  const absl::string_view digest_bytes(
      reinterpret_cast<const char*>(correction_digest.bytes.data()),
      correction_digest.bytes.size());
  UpdateBytesFrame('C', digest_bytes, &cursor->hasher);
}

Hash256 ComputeLogicalAgentRequestHash(
    const TranscriptCursor& transcript_before_turn,
    absl::string_view canonical_agent_input, uint32_t max_decision_tokens,
    const Hash256& correction_digest, const Hash256& projection_manifest_hash,
    const SessionHandoffIdentity& agent_identity) {
  TranscriptCursor logical_prefix = transcript_before_turn;
  if (!logical_prefix.correction_digest.has_value() ||
      *logical_prefix.correction_digest != correction_digest) {
    ResetTranscriptHasher(correction_digest, &logical_prefix);
  }
  const Hash256 prefix_hash = SnapshotDigest(logical_prefix.hasher);
  Sha256Hasher hasher;
  hasher.Update(kAgentRequestDomain);
  UpdateBytesFrame('C',
                   absl::string_view(reinterpret_cast<const char*>(
                                         correction_digest.bytes.data()),
                                     correction_digest.bytes.size()),
                   &hasher);
  UpdateBytesFrame(
      'T',
      absl::string_view(reinterpret_cast<const char*>(prefix_hash.bytes.data()),
                        prefix_hash.bytes.size()),
      &hasher);
  UpdateBytesFrame('X',
                   absl::string_view(reinterpret_cast<const char*>(
                                         projection_manifest_hash.bytes.data()),
                                     projection_manifest_hash.bytes.size()),
                   &hasher);
  UpdateBytesFrame(
      'M',
      absl::string_view(reinterpret_cast<const char*>(
                            agent_identity.model_artifact_hash.bytes.data()),
                        agent_identity.model_artifact_hash.bytes.size()),
      &hasher);
  UpdateBytesFrame(
      'R',
      absl::string_view(reinterpret_cast<const char*>(
                            agent_identity.runtime_artifact_hash.bytes.data()),
                        agent_identity.runtime_artifact_hash.bytes.size()),
      &hasher);
  UpdateBytesFrame(
      'P',
      absl::string_view(reinterpret_cast<const char*>(
                            agent_identity.inference_profile_hash.bytes.data()),
                        agent_identity.inference_profile_hash.bytes.size()),
      &hasher);
  constexpr char kLimitTag = 'L';
  hasher.Update(absl::string_view(&kLimitTag, 1));
  UpdateU32(max_decision_tokens, &hasher);
  UpdateBytesFrame('I', canonical_agent_input, &hasher);
  return hasher.Finalize();
}

absl::Status ValidateSnapshotShape(const DPMLogSnapshot& snapshot) {
  if (snapshot.log_id.empty() || snapshot.case_id.empty() ||
      snapshot.log_id.size() > kMaximumDPMProjectionIdentityBytes ||
      snapshot.case_id.size() > kMaximumDPMProjectionIdentityBytes) {
    return absl::DataLossError(
        "DPM log snapshot has missing or over-bound log/case identity.");
  }
  if (!IsValidUtf8(snapshot.log_id) || !IsValidUtf8(snapshot.case_id) ||
      ContainsControlByte(snapshot.log_id) ||
      ContainsControlByte(snapshot.case_id)) {
    return absl::DataLossError(
        "DPM log snapshot identities must be UTF-8 without control bytes.");
  }
  if (snapshot.generation != snapshot.events.size()) {
    return absl::DataLossError(
        "DPM log snapshot generation does not match its event count.");
  }
  if (IsZeroHash(snapshot.prefix_hash)) {
    return absl::DataLossError(
        "DPM log snapshot has no cryptographic prefix hash.");
  }
  if (snapshot.prefix_hashes.size() != snapshot.events.size() + 1 ||
      snapshot.prefix_hashes.empty() ||
      snapshot.prefix_hashes.back() != snapshot.prefix_hash) {
    return absl::DataLossError(
        "DPM log snapshot has no complete immutable prefix-proof index.");
  }
  for (const Hash256& prefix : snapshot.prefix_hashes) {
    if (IsZeroHash(prefix)) {
      return absl::DataLossError(
          "DPM log snapshot prefix-proof index contains an empty digest.");
    }
  }
  for (uint64_t i = 0; i < snapshot.events.size(); ++i) {
    if (snapshot.events[i].index != i ||
        snapshot.events[i].case_id != snapshot.case_id) {
      return absl::DataLossError(
          "DPM log snapshot contains non-contiguous or cross-case events.");
    }
    if (snapshot.events[i].operation_id.empty() ||
        snapshot.events[i].operation_id.size() >
            kMaximumDPMEventOperationIdBytes ||
        snapshot.events[i].payload.size() > kMaximumDPMEventPayloadBytes ||
        ContainsControlByte(snapshot.events[i].operation_id) ||
        !IsValidUtf8(snapshot.events[i].operation_id) ||
        !IsValidUtf8(snapshot.events[i].case_id) ||
        !IsValidUtf8(snapshot.events[i].payload)) {
      return absl::DataLossError(
          "DPM log snapshot contains invalid or over-bound event fields.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> SnapshotPrefixHashAt(const DPMLogSnapshot& snapshot,
                                             uint64_t event_count) {
  if (event_count > snapshot.generation ||
      snapshot.prefix_hashes.size() != snapshot.events.size() + 1 ||
      snapshot.prefix_hashes.empty() ||
      snapshot.prefix_hashes.back() != snapshot.prefix_hash) {
    return absl::DataLossError(
        "DPM snapshot cannot authenticate the requested raw-log prefix.");
  }
  const Hash256 prefix =
      snapshot.prefix_hashes[static_cast<size_t>(event_count)];
  if (IsZeroHash(prefix)) {
    return absl::DataLossError(
        "DPM snapshot contains an empty raw-log prefix proof.");
  }
  return prefix;
}

bool IsDecisionInputKind(DPMEvent::Kind kind) {
  return kind == DPMEvent::Kind::kUser || kind == DPMEvent::Kind::kTool ||
         kind == DPMEvent::Kind::kInternal;
}

absl::StatusOr<Hash256> ComputeReceiptAgentReplayRequestHash(
    const DPMLogSnapshot& snapshot, const DPMEvent& response,
    const DPMTurnReceipt& receipt);

absl::Status ValidateInputEvent(const DPMEvent& event,
                                const DPMTurnRequest& request) {
  if (event.kind != request.kind ||
      event.operation_id != request.operation_id ||
      event.case_id != request.case_id || event.payload != request.payload) {
    return absl::AlreadyExistsError(
        "DPM operation id is already bound to different input bytes.");
  }
  if (request.timestamp_us.has_value() &&
      event.timestamp_us != *request.timestamp_us) {
    return absl::AlreadyExistsError(
        "DPM operation id is already bound to a different timestamp.");
  }
  if (event.turn_receipt.has_value()) {
    return absl::DataLossError(
        "DPM input event unexpectedly contains a model-turn receipt.");
  }
  if (event.timestamp_us <= 0) {
    return absl::DataLossError("DPM input event has a non-positive timestamp.");
  }
  return absl::OkStatus();
}

absl::Status ValidateReceiptAndAdvance(const DPMLogSnapshot& snapshot,
                                       const DPMEvent& response,
                                       TranscriptCursor* transcript) {
  if (response.kind != DPMEvent::Kind::kModelTurn ||
      !response.turn_receipt.has_value()) {
    return absl::DataLossError(
        "DPM model event is missing its authoritative turn receipt.");
  }
  const DPMTurnReceipt& receipt = *response.turn_receipt;
  if (receipt.format_version != DPMTurnReceipt::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Unsupported DPM turn receipt version.");
  }
  if (receipt.operation_id.empty() ||
      receipt.operation_id != response.operation_id ||
      receipt.response_event_index != response.index ||
      receipt.decision_output != response.payload) {
    return absl::DataLossError("DPM model event and turn receipt disagree.");
  }
  if (receipt.input_event_index >= response.index ||
      receipt.input_event_index >= snapshot.events.size()) {
    return absl::DataLossError(
        "DPM turn receipt names an invalid input event.");
  }
  const DPMEvent& input = snapshot.events[receipt.input_event_index];
  if (!IsDecisionInputKind(input.kind) ||
      input.operation_id != receipt.operation_id || input.turn_receipt ||
      input.case_id != response.case_id ||
      receipt.response_event_index != receipt.input_event_index + 1 ||
      response.timestamp_us < input.timestamp_us) {
    return absl::DataLossError(
        "DPM turn receipt is not bound to its authoritative input event.");
  }
  if (receipt.canonical_agent_input.empty() ||
      receipt.canonical_agent_input.size() >
          kMaximumDPMCanonicalAgentInputBytes ||
      receipt.projected_memory.empty() ||
      receipt.projected_memory.size() > kMaximumDPMEventPayloadBytes ||
      receipt.decision_token_ids.empty() ||
      IsZeroHash(receipt.agent_session_identity.model_artifact_hash) ||
      IsZeroHash(receipt.agent_session_identity.runtime_artifact_hash) ||
      IsZeroHash(receipt.agent_session_identity.inference_profile_hash) ||
      receipt.max_decision_tokens == 0 ||
      receipt.max_decision_tokens > kMaximumDPMGenerationTokens ||
      IsZeroHash(receipt.agent_request_hash) ||
      IsZeroHash(receipt.agent_transcript_hash)) {
    return absl::DataLossError(
        "DPM turn receipt is missing required reconstruction data.");
  }
  if (!IsValidUtf8(receipt.operation_id) ||
      !IsValidUtf8(receipt.projected_memory) ||
      !IsValidUtf8(receipt.canonical_agent_input) ||
      !IsValidUtf8(receipt.decision_output)) {
    return absl::DataLossError(
        "DPM turn receipt contains non-UTF-8 reconstruction fields.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateDPMProjectionManifest(receipt.projection_manifest));
  if (receipt.projection_manifest.log_id != snapshot.log_id ||
      receipt.projection_manifest.case_id != snapshot.case_id ||
      receipt.projection_manifest.source_event_count != response.index ||
      receipt.projection_manifest.input_event_index !=
          receipt.input_event_index ||
      ComputeCanonicalDPMProjectionOutputHash(receipt.projected_memory) !=
          receipt.projection_manifest.output_hash) {
    return absl::DataLossError(
        "DPM turn receipt projection manifest is not bound to its raw-log "
        "source and projected-memory bytes.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_projection_correction_digest,
      ComputeDPMCorrectionDigestForPrefix(
          snapshot, receipt.projection_manifest.source_event_count));
  if (receipt.projection_manifest.correction_digest !=
      expected_projection_correction_digest) {
    return absl::DataLossError(
        "DPM turn receipt projection correction lineage is not derived from "
        "its authoritative source prefix.");
  }
  if (receipt.projection_manifest.baseline_manifest_hash.has_value()) {
    const uint64_t baseline_response_index =
        receipt.projection_manifest.event_range_start;
    if (baseline_response_index >= response.index ||
        baseline_response_index >= snapshot.events.size()) {
      return absl::DataLossError(
          "DPM projection baseline does not name a prior response event.");
    }
    const DPMEvent& baseline_response =
        snapshot.events[baseline_response_index];
    if (baseline_response.kind != DPMEvent::Kind::kModelTurn ||
        !baseline_response.turn_receipt.has_value()) {
      return absl::DataLossError(
          "DPM projection baseline does not name an authoritative receipt.");
    }
    const DPMProjectionManifest& baseline =
        baseline_response.turn_receipt->projection_manifest;
    ABSL_ASSIGN_OR_RETURN(const Hash256 expected_baseline_correction_digest,
        ComputeDPMCorrectionDigestForPrefix(
            snapshot, baseline.source_event_count));
    if (baseline.source_event_count != baseline_response_index ||
        baseline.manifest_hash !=
            *receipt.projection_manifest.baseline_manifest_hash ||
        baseline.output_hash !=
            *receipt.projection_manifest.baseline_output_hash ||
        baseline.correction_digest != expected_baseline_correction_digest ||
        baseline.config_hash != receipt.projection_manifest.config_hash ||
        baseline.runtime_identity !=
            receipt.projection_manifest.runtime_identity ||
        baseline.replay_mode != receipt.projection_manifest.replay_mode ||
        baseline.exact_profile_id !=
            receipt.projection_manifest.exact_profile_id) {
      return absl::DataLossError(
          "DPM projection baseline hashes do not identify the compatible "
          "prior authoritative receipt.");
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMTurnReceiptReplayEvidence(receipt));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_agent_replay_request_hash,
      ComputeReceiptAgentReplayRequestHash(snapshot, response, receipt));
  if (expected_agent_replay_request_hash != receipt.agent_replay_request_hash) {
    return absl::DataLossError(
        "DPM agent replay request hash is not derived from the "
        "authoritative full correction-epoch transcript.");
  }
  if (receipt.decision_token_ids.size() > receipt.max_decision_tokens) {
    return absl::DataLossError(
        "DPM turn receipt exceeds its committed decision-token limit.");
  }
  for (int token_id : receipt.decision_token_ids) {
    if (token_id < 0 ||
        static_cast<int64_t>(token_id) > std::numeric_limits<int32_t>::max()) {
      return absl::DataLossError(
          "DPM turn receipt contains a non-canonical decision token id.");
    }
  }

  if (ComputeLogicalAgentRequestHash(
          *transcript, receipt.canonical_agent_input,
          receipt.max_decision_tokens,
          receipt.projection_manifest.correction_digest,
          receipt.projection_manifest.manifest_hash,
          receipt.agent_session_identity) != receipt.agent_request_hash) {
    return absl::DataLossError(
        "DPM turn receipt agent request hash is not canonical.");
  }

  if (!transcript->correction_digest.has_value() ||
      *transcript->correction_digest !=
          receipt.projection_manifest.correction_digest) {
    ResetTranscriptHasher(receipt.projection_manifest.correction_digest,
                          transcript);
  }
  UpdateBytesFrame('I', receipt.canonical_agent_input, &transcript->hasher);
  UpdateTokenFrame(receipt.decision_token_ids, &transcript->hasher);
  UpdateBytesFrame('O', receipt.decision_output, &transcript->hasher);
  if (SnapshotDigest(transcript->hasher) != receipt.agent_transcript_hash) {
    return absl::DataLossError(
        "DPM turn receipt transcript hash chain is invalid.");
  }
  return absl::OkStatus();
}

struct OperationRecord {
  const DPMEvent* input = nullptr;
  const DPMEvent* response = nullptr;
  const DPMEvent* correction = nullptr;
  std::optional<Hash256> correction_digest_after;
};

struct AuthoritativeLogIndex {
  std::unordered_map<std::string, OperationRecord> operations;
  const DPMEvent* pending_input = nullptr;
  Hash256 correction_digest;
  std::optional<SessionHandoffIdentity> agent_identity;
  TranscriptCursor transcript;
};

absl::StatusOr<AuthoritativeLogIndex> ValidateAndIndexAuthoritativeLog(
    const DPMLogSnapshot& snapshot) {
  ABSL_RETURN_IF_ERROR(ValidateSnapshotShape(snapshot));
  AuthoritativeLogIndex index;
  ABSL_ASSIGN_OR_RETURN(
      index.correction_digest,
      InitialDPMCorrectionDigest(snapshot.log_id, snapshot.case_id));
  for (const DPMEvent& event : snapshot.events) {
    if (event.operation_id.empty() || event.timestamp_us <= 0) {
      return absl::DataLossError(
          "Authoritative DPM event has no operation id or timestamp.");
    }
    OperationRecord& operation = index.operations[event.operation_id];
    if (event.kind == DPMEvent::Kind::kCorrection) {
      if (event.payload.empty() || event.turn_receipt.has_value() ||
          operation.input != nullptr || operation.response != nullptr ||
          operation.correction != nullptr) {
        return absl::DataLossError(
            "Authoritative DPM correction operation is malformed or "
            "duplicated.");
      }
      operation.correction = &event;
      ABSL_ASSIGN_OR_RETURN(
          index.correction_digest,
          AdvanceDPMCorrectionDigest(index.correction_digest, event));
      operation.correction_digest_after = index.correction_digest;
      continue;
    }
    if (IsDecisionInputKind(event.kind)) {
      if (event.payload.empty() || event.turn_receipt.has_value() ||
          operation.input != nullptr || operation.response != nullptr ||
          operation.correction != nullptr) {
        return absl::DataLossError(
            "Authoritative DPM turn input is malformed or duplicated.");
      }
      operation.input = &event;
      continue;
    }
    if (event.kind != DPMEvent::Kind::kModelTurn) {
      return absl::DataLossError("Authoritative DPM event has unknown kind.");
    }
    if (operation.correction != nullptr || operation.response != nullptr) {
      return absl::DataLossError(
          "Authoritative DPM operation has duplicate response identity.");
    }
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &index.transcript));
    const DPMTurnReceipt& receipt = *event.turn_receipt;
    if (receipt.projection_manifest.correction_digest !=
        index.correction_digest) {
      return absl::DataLossError(
          "DPM turn receipt correction digest is not derived from the raw "
          "log.");
    }
    if (!index.agent_identity.has_value()) {
      index.agent_identity = receipt.agent_session_identity;
    } else if (*index.agent_identity != receipt.agent_session_identity) {
      return absl::DataLossError(
          "A DPM log cannot mix agent model/runtime/profile identities.");
    }
    if (operation.input == nullptr ||
        operation.input->index != receipt.input_event_index ||
        event.index != operation.input->index + 1) {
      return absl::DataLossError(
          "DPM model response does not immediately follow its input.");
    }
    operation.response = &event;
  }

  for (const auto& entry : index.operations) {
    const OperationRecord& operation = entry.second;
    if (operation.correction != nullptr) continue;
    if (operation.input == nullptr) {
      return absl::DataLossError(
          "DPM operation contains a response without an input.");
    }
    if (operation.response != nullptr) continue;
    if (index.pending_input != nullptr ||
        operation.input->index + 1 != snapshot.events.size()) {
      return absl::DataLossError(
          "DPM log contains an interleaved or multiple incomplete turn.");
    }
    index.pending_input = operation.input;
  }
  return index;
}

DPMAgentGenerationRequest::PrefillChunk TextChunk(absl::string_view value) {
  DPMAgentGenerationRequest::PrefillChunk chunk;
  chunk.encoding = DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text;
  chunk.text.assign(value.data(), value.size());
  return chunk;
}

DPMAgentGenerationRequest::PrefillChunk TokenChunk(
    const std::vector<int>& value) {
  DPMAgentGenerationRequest::PrefillChunk chunk;
  chunk.encoding = DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds;
  chunk.token_ids = value;
  return chunk;
}

absl::StatusOr<Hash256> ComputeAgentReplayRequestHash(
    const DPMAgentExecutionRequest& logical_request) {
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMAgentExecutionRequest(logical_request));
  DPMCanonicalReplayRequest replay_request{
      .stage = DPMReplayStage::kAgentDecision,
      .max_output_tokens = logical_request.max_output_tokens,
      .request_contract = std::string(kDPMAgentReplayContract),
      .canonical_payload = std::move(encoded),
  };
  return ComputeDPMCanonicalReplayRequestHash(replay_request);
}

DPMCheckpointWorkerPrefillMode ToCheckpointWorkerPrefillMode(
    FreshWorkerPrefillMode mode) {
  switch (mode) {
    case FreshWorkerPrefillMode::kFullCanonicalPrefill:
      return DPMCheckpointWorkerPrefillMode::kFullCanonicalPrefill;
    case FreshWorkerPrefillMode::kOwnPositionCapsuleDelta:
      return DPMCheckpointWorkerPrefillMode::kOwnPositionCapsuleDelta;
  }
  return DPMCheckpointWorkerPrefillMode::kNone;
}

absl::StatusOr<Hash256> ComputeReceiptAgentReplayRequestHash(
    const DPMLogSnapshot& snapshot, const DPMEvent& response,
    const DPMTurnReceipt& receipt) {
  if (response.index >= snapshot.events.size() ||
      &snapshot.events[response.index] != &response) {
    return absl::DataLossError(
        "DPM agent replay receipt is not located at its response index.");
  }
  std::vector<DPMAgentGenerationRequest::PrefillChunk> full_chunks;
  for (const DPMEvent& prior : snapshot.events) {
    if (prior.index >= response.index) break;
    if (prior.kind != DPMEvent::Kind::kModelTurn ||
        !prior.turn_receipt.has_value()) {
      continue;
    }
    const DPMTurnReceipt& prior_receipt = *prior.turn_receipt;
    if (prior_receipt.projection_manifest.correction_digest !=
        receipt.projection_manifest.correction_digest) {
      full_chunks.clear();
      continue;
    }
    full_chunks.push_back(TextChunk(prior_receipt.canonical_agent_input));
    full_chunks.push_back(TokenChunk(prior_receipt.decision_token_ids));
  }
  full_chunks.push_back(TextChunk(receipt.canonical_agent_input));
  DPMAgentExecutionRequest logical_request{
      .logical_agent_request_hash = receipt.agent_request_hash,
      .correction_digest = receipt.projection_manifest.correction_digest,
      .max_output_tokens = receipt.max_decision_tokens,
      .full_canonical_prefill_chunks = std::move(full_chunks),
  };
  return ComputeAgentReplayRequestHash(logical_request);
}

void AppendTextSection(absl::string_view name, absl::string_view bytes,
                       std::string* output) {
  absl::StrAppend(output, name, " ", bytes.size(), "\n");
  output->append(bytes.data(), bytes.size());
  output->push_back('\n');
}

absl::Status ValidateProjectionOutcome(const DPMProjectionOutcome& projection,
                                       const DPMProjectionRequest& request) {
  if (projection.projected_memory.empty() ||
      projection.projected_memory.size() > kMaximumDPMEventPayloadBytes ||
      !IsValidUtf8(projection.projected_memory)) {
    return absl::FailedPreconditionError(
        "DPM projection provider returned empty, over-bound, or non-UTF-8 "
        "projected memory.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(projection.manifest));
  if (projection.manifest.log_id != request.log.log_id ||
      projection.manifest.case_id != request.log.case_id ||
      projection.manifest.case_id != request.case_id ||
      projection.manifest.source_event_count != request.log.events.size() ||
      projection.manifest.source_prefix_hash != request.log.prefix_hash ||
      projection.manifest.input_event_index != request.input_event_index ||
      ComputeCanonicalDPMProjectionOutputHash(projection.projected_memory) !=
          projection.manifest.output_hash) {
    return absl::FailedPreconditionError(
        "DPM projection provider returned an identity not bound to the "
        "requested raw-log prefix and projected-memory bytes.");
  }
  const auto validate_model_view = [](const std::optional<std::string>& view) {
    return !view.has_value() ||
           (!view->empty() && view->size() <= kMaximumDPMEventPayloadBytes &&
            IsValidUtf8(*view));
  };
  if (!validate_model_view(projection.model_visible_memory) ||
      !validate_model_view(projection.model_visible_input)) {
    return absl::FailedPreconditionError(
        "DPM projection provider returned an empty, over-bound, or non-UTF-8 "
        "model-visible semantic view.");
  }
  return absl::OkStatus();
}

absl::Status ValidateAgentOutcome(const DPMAgentGenerationOutcome& outcome,
                                  uint32_t max_decision_tokens) {
  if (outcome.decision_output.size() > kMaximumDPMEventPayloadBytes ||
      !IsValidUtf8(outcome.decision_output)) {
    return absl::DataLossError(
        "DPM agent returned over-bound or non-UTF-8 decision bytes.");
  }
  if (outcome.decision_token_ids.empty()) {
    return absl::FailedPreconditionError(
        "DPM agent did not return the exact decision token ids.");
  }
  if (outcome.decision_token_ids.size() > max_decision_tokens) {
    return absl::FailedPreconditionError(
        "DPM agent exceeded the authoritative decision-token limit.");
  }
  for (int token_id : outcome.decision_token_ids) {
    if (token_id < 0 ||
        static_cast<int64_t>(token_id) > std::numeric_limits<int32_t>::max()) {
      return absl::DataLossError(
          "DPM agent returned a non-canonical decision token id.");
    }
  }
  if (outcome.prepared_prefill_plan.has_value()) {
    ABSL_RETURN_IF_ERROR(
        ValidateDPMPreparedPrefillPlan(*outcome.prepared_prefill_plan));
  }
  if (outcome.capsule_restore_evidence.has_value()) {
    ABSL_RETURN_IF_ERROR(
        ValidateCapsuleRestoreEvidence(*outcome.capsule_restore_evidence));
    if (!outcome.prepared_prefill_plan.has_value() ||
        !(outcome.capsule_restore_evidence->plan.prefill.prepared_plan ==
          *outcome.prepared_prefill_plan)) {
      return absl::DataLossError(
          "DPM agent restore evidence is detached from its runtime-prepared "
          "prefill plan.");
    }
  }
  return absl::OkStatus();
}

uint64_t CountCommittedTurns(const DPMLogSnapshot& snapshot) {
  uint64_t count = 0;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind == DPMEvent::Kind::kModelTurn && event.turn_receipt) {
      ++count;
    }
  }
  return count;
}

}  // namespace

int64_t SystemDPMClock::NowMicros() const {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

DPMEngine::DPMEngine(DPMEventLog* log,
                     DPMProjectionProvider* projection_provider,
                     DPMAgentReplayRuntime* agent_runtime,
                     DPMSessionCheckpointRepository* checkpoint_repository,
                     DPMEngineConfig config, DPMClock* clock)
    : DPMEngine(log, projection_provider, agent_runtime, checkpoint_repository,
                nullptr, std::move(config), clock) {}

DPMEngine::DPMEngine(DPMEventLog* log,
                     DPMProjectionProvider* projection_provider,
    DPMAgentReplayRuntime* agent_runtime,
    DPMSessionCheckpointRepository* checkpoint_repository,
    CapsuleRestoreEvidenceRepository* evidence_repository,
    DPMEngineConfig config, DPMClock* clock)
    : log_(log),
      projection_provider_(projection_provider),
      agent_runtime_(agent_runtime),
      checkpoint_repository_(checkpoint_repository),
      evidence_repository_(evidence_repository),
      config_(std::move(config)),
      clock_(clock == nullptr ? &system_clock_ : clock) {}

absl::StatusOr<DPMEngine::CapsuleRestoreRuntimeAuthority>
DPMEngine::RequireCurrentCapsuleRestoreAuthority() const {
  if (agent_runtime_ == nullptr) {
    return absl::InvalidArgumentError(
        "CapsuleRestore authority requires an agent runtime.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::optional<AuthenticatedCapsuleRestoreAdmission> admission,
      agent_runtime_->GetAuthenticatedCapsuleRestoreAdmission());
  if (!admission.has_value()) {
    return absl::FailedPreconditionError(
        "DPM capsule restore/capture requires "
        "one current atomic CapsuleRestore "
        "admission from the loaded Engine.");
  }
  ABSL_ASSIGN_OR_RETURN(CapsuleRestoreAuthority authority,
                        BuildCapsuleRestoreAuthority(*admission));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreAuthorityForAdmission(authority, *admission));
  if (authority.capability.session_identity !=
      agent_runtime_->GetSessionHandoffIdentity()) {
    return absl::FailedPreconditionError(
        "Current atomic CapsuleRestore authority belongs to another loaded "
        "agent runtime.");
  }

  ABSL_ASSIGN_OR_RETURN(const DPMStageCapabilities stage,
                        agent_runtime_->GetCapabilities());
  ABSL_RETURN_IF_ERROR(ValidateDPMStageCapabilities(stage));
  if (stage.stage != DPMCapabilityStage::kAgentDecision ||
      stage.replay_mode != agent_runtime_->GetReplayMode() ||
      stage.runtime_identity != authority.capability.session_identity) {
    return absl::DataLossError(
        "CapsuleRestore authority disagrees with the active agent stage.");
  }
  ABSL_ASSIGN_OR_RETURN(const std::optional<Hash256> exact_profile_id,
                        agent_runtime_->GetExactProfileId());
  switch (stage.replay_mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      if (exact_profile_id.has_value() || stage.exact_profile.has_value() ||
          stage.exact_profile_admission_record_id.has_value()) {
        return absl::DataLossError(
            "WinnerReplay CapsuleRestore authority invented an "
            "ExactRegeneration claim.");
      }
      break;
    case DPMReplayMode::kExactRegeneration:
      if (!exact_profile_id.has_value() || !stage.exact_profile.has_value() ||
          *exact_profile_id != stage.exact_profile->profile_id ||
          authority.capability.exact_profile_id !=
              stage.exact_profile->profile_id ||
          authority.capability.backend != stage.exact_profile->backend) {
        return absl::FailedPreconditionError(
            "ExactRegeneration profile and CapsuleRestore capability do not "
            "describe the same Engine-derived runtime.");
      }
      break;
  }
  return CapsuleRestoreRuntimeAuthority{
      .admission = std::move(*admission),
      .authority = std::move(authority),
  };
}

absl::Status DPMEngine::ValidateConfiguration() const {
  if (log_ == nullptr || projection_provider_ == nullptr ||
      agent_runtime_ == nullptr || clock_ == nullptr) {
    return absl::InvalidArgumentError(
        "DPMEngine requires log, projection, agent, and clock dependencies.");
  }
  if (config_.max_decision_tokens <= 0 ||
      static_cast<uint64_t>(config_.max_decision_tokens) >
          kMaximumDPMGenerationTokens) {
    return absl::InvalidArgumentError(
        "DPM decision token limit must be between 1 and 65,536.");
  }
  if (config_.require_checkpoint_at_milestone &&
      config_.checkpoint_interval_turns == 0) {
    return absl::InvalidArgumentError(
        "DPM cannot require a checkpoint at a milestone while automatic "
        "checkpoint milestones are disabled.");
  }
  ABSL_RETURN_IF_ERROR(projection_provider_->ValidateSupport());
  ABSL_RETURN_IF_ERROR(agent_runtime_->ValidateSupport());
  ABSL_RETURN_IF_ERROR(agent_runtime_->ValidateGenerationLimit(
      static_cast<uint32_t>(config_.max_decision_tokens)));
  ABSL_ASSIGN_OR_RETURN(const DPMReplayMode projection_replay_mode,
                        projection_provider_->GetReplayMode());
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(projection_replay_mode));
  const DPMReplayMode agent_replay_mode = agent_runtime_->GetReplayMode();
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(agent_replay_mode));
  if (projection_replay_mode != agent_replay_mode) {
    return absl::FailedPreconditionError(
        "DPM projection and agent stages must use the same named replay "
        "guarantee before any authoritative input is appended.");
  }
  ABSL_ASSIGN_OR_RETURN(const std::optional<Hash256> exact_profile_id,
                        agent_runtime_->GetExactProfileId());
  if ((agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay &&
       exact_profile_id.has_value()) ||
      (agent_replay_mode == DPMReplayMode::kExactRegeneration &&
       (!exact_profile_id.has_value() || IsZeroHash(*exact_profile_id)))) {
    return absl::FailedPreconditionError(
        "DPM agent replay mode and derived exact profile disagree.");
  }
  const SessionHandoffIdentity& identity =
      agent_runtime_->GetSessionHandoffIdentity();
  if (IsZeroHash(identity.model_artifact_hash) ||
      IsZeroHash(identity.runtime_artifact_hash) ||
      IsZeroHash(identity.inference_profile_hash)) {
    return absl::FailedPreconditionError(
        "DPM agent runtime has not resolved its loaded session identity.");
  }
  const bool uses_session_checkpoints = config_.restore_session_checkpoints ||
      config_.checkpoint_interval_turns != 0;
  if (!uses_session_checkpoints) {
    return absl::OkStatus();
  }
  if (checkpoint_repository_ == nullptr || evidence_repository_ == nullptr) {
    return absl::InvalidArgumentError(
        "Enabled DPM checkpoint restore/capture requires checkpoint and "
        "operation-evidence repositories.");
  }
  if (config_.checkpoint_key_id.empty() ||
      config_.checkpoint_key_id.size() > kMaximumCheckpointKeyIdSize) {
    return absl::InvalidArgumentError(
        "DPM checkpoint key id must contain 1 to 1024 bytes.");
  }
  if (config_.checkpoint_authentication_key.size() < 32) {
    return absl::InvalidArgumentError(
        "DPM checkpoint authentication key must contain at least 32 bytes.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerAuthentication(FreshWorkerAuthentication{
          .key_id = config_.operation_evidence_key_id,
          .authentication_key = config_.operation_evidence_authentication_key,
      }));
  if (config_.checkpoint_key_id == config_.operation_evidence_key_id ||
      config_.checkpoint_key_id == kFreshWorkerTransientRestoreKeyId ||
      config_.checkpoint_key_id == kFreshWorkerTransientProducingKeyId ||
      config_.operation_evidence_key_id == kFreshWorkerTransientRestoreKeyId ||
      config_.operation_evidence_key_id ==
          kFreshWorkerTransientProducingKeyId) {
    return absl::FailedPreconditionError(
        "DPM durable checkpoint, operation-evidence, and request-transient "
        "capsule key IDs must be distinct.");
  }
  ABSL_RETURN_IF_ERROR(agent_runtime_->ValidateSessionHandoffSupport());
  ABSL_ASSIGN_OR_RETURN(const CapsuleRestoreRuntimeAuthority authority,
                        RequireCurrentCapsuleRestoreAuthority());
  const CapsuleRestoreStateWitnessOperationalDomain& domain =
      authority.admission.operational_coverage.operational_domain;
  if (domain.checkpoint_authentication_key_id != config_.checkpoint_key_id ||
      domain.operation_evidence_authentication_key_id !=
          config_.operation_evidence_key_id ||
      domain.maximum_output_tokens <
          static_cast<uint32_t>(config_.max_decision_tokens)) {
    return absl::FailedPreconditionError(
        "DPM checkpoint/evidence transport or decision limit is outside the "
        "authenticated CapsuleRestore operational domain.");
  }
  return absl::OkStatus();
}

absl::StatusOr<DPMCapabilities> DPMEngine::GetCapabilities() const {
  ABSL_RETURN_IF_ERROR(ValidateConfiguration());
  ABSL_ASSIGN_OR_RETURN(DPMStageCapabilities projection,
                        projection_provider_->GetCapabilities());
  ABSL_ASSIGN_OR_RETURN(DPMStageCapabilities agent,
                        agent_runtime_->GetCapabilities());
  ABSL_RETURN_IF_ERROR(ValidateDPMStageCapabilities(projection));
  ABSL_RETURN_IF_ERROR(ValidateDPMStageCapabilities(agent));
  if (projection.stage != DPMCapabilityStage::kProjection ||
      agent.stage != DPMCapabilityStage::kAgentDecision ||
      projection.replay_mode != agent.replay_mode ||
      agent.runtime_identity != agent_runtime_->GetSessionHandoffIdentity() ||
      projection.max_output_tokens == 0 ||
      agent.max_output_tokens <
          static_cast<uint32_t>(config_.max_decision_tokens)) {
    return absl::DataLossError(
        "DPM runtime returned inconsistent assembled stage capabilities.");
  }
  const bool exact_mode =
      agent.replay_mode == DPMReplayMode::kExactRegeneration;
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<Hash256> current_agent_exact_profile_id,
      agent_runtime_->GetExactProfileId());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<Hash256> current_agent_exact_admission_id,
      agent_runtime_->GetExactProfileAdmissionRecordId());
  const bool projection_exact_complete =
      projection.exact_profile.has_value() &&
      projection.exact_profile_admission_record_id.has_value() &&
      !IsZeroHash(*projection.exact_profile_admission_record_id);
  const bool agent_exact_complete =
      agent.exact_profile.has_value() &&
      agent.exact_profile_admission_record_id.has_value() &&
      !IsZeroHash(*agent.exact_profile_admission_record_id);
  if (exact_mode != projection_exact_complete ||
      exact_mode != agent_exact_complete ||
      exact_mode != current_agent_exact_profile_id.has_value() ||
      exact_mode != current_agent_exact_admission_id.has_value() ||
      (!exact_mode &&
       (projection.exact_profile.has_value() ||
        projection.exact_profile_admission_record_id.has_value() ||
        agent.exact_profile.has_value() ||
        agent.exact_profile_admission_record_id.has_value()))) {
    return absl::DataLossError(
        "DPM stage capabilities confuse an exact profile candidate with "
        "current ExactRegeneration admission.");
  }
  if (exact_mode) {
    ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(*projection.exact_profile));
    ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(*agent.exact_profile));
    if (*current_agent_exact_profile_id != agent.exact_profile->profile_id ||
        *current_agent_exact_admission_id !=
            *agent.exact_profile_admission_record_id ||
        projection.exact_profile->session_identity !=
            projection.runtime_identity ||
        agent.exact_profile->session_identity != agent.runtime_identity) {
      return absl::DataLossError(
          "DPM exact stage profile identity differs from its loaded runtime "
          "identity.");
    }
  }

  DPMAgentCheckpointCapabilities checkpoints;
  const bool has_checkpoint_key_id = !config_.checkpoint_key_id.empty();
  const bool has_checkpoint_key =
      !config_.checkpoint_authentication_key.empty();
  const bool has_operation_evidence_key_id =
      !config_.operation_evidence_key_id.empty();
  const bool has_operation_evidence_key =
      !config_.operation_evidence_authentication_key.empty();
  if (has_checkpoint_key_id != has_checkpoint_key) {
    return absl::FailedPreconditionError(
        "DPM checkpoint transport has only one authentication credential.");
  }
  if (has_operation_evidence_key_id != has_operation_evidence_key) {
    return absl::FailedPreconditionError(
        "DPM operation-evidence transport has only one authentication "
        "credential.");
  }
  if (has_checkpoint_key_id && checkpoint_repository_ == nullptr) {
    return absl::FailedPreconditionError(
        "DPM checkpoint credentials do not have a checkpoint repository.");
  }
  if (has_operation_evidence_key_id && evidence_repository_ == nullptr) {
    return absl::FailedPreconditionError(
        "DPM operation-evidence credentials do not have an evidence "
        "repository.");
  }
  if (has_checkpoint_key_id &&
      (config_.checkpoint_key_id.size() > kMaximumCheckpointKeyIdSize ||
       config_.checkpoint_authentication_key.size() < 32)) {
    return absl::FailedPreconditionError(
        "DPM checkpoint transport credentials do not satisfy the "
        "authenticated transport contract.");
  }
  if (has_operation_evidence_key_id &&
      (config_.operation_evidence_key_id.size() > kMaximumCheckpointKeyIdSize ||
       config_.operation_evidence_authentication_key.size() < 32 ||
       config_.operation_evidence_authentication_key.size() >
           kMaximumFreshWorkerAuthenticationKeyBytes)) {
    return absl::FailedPreconditionError(
        "DPM operation-evidence credentials do not satisfy the "
        "authenticated transport contract.");
  }
  checkpoints.checkpoint_transport_configured =
      checkpoint_repository_ != nullptr && has_checkpoint_key_id &&
      has_checkpoint_key;
  checkpoints.operation_evidence_transport_configured =
      evidence_repository_ != nullptr && has_operation_evidence_key_id &&
      has_operation_evidence_key;
  checkpoints.restore_enabled = config_.restore_session_checkpoints;
  checkpoints.capture_interval_turns = config_.checkpoint_interval_turns;
  checkpoints.capture_required_at_milestone =
      config_.require_checkpoint_at_milestone;
  if ((checkpoints.restore_enabled ||
       checkpoints.capture_interval_turns != 0) &&
      (!checkpoints.checkpoint_transport_configured ||
       !checkpoints.operation_evidence_transport_configured)) {
    return absl::FailedPreconditionError(
        "DPM checkpoint restore or capture cannot be reported without an "
        "authenticated checkpoint and operation-evidence transport.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::optional<AuthenticatedCapsuleRestoreAdmission>
          state_witness_admission,
      agent_runtime_->GetAuthenticatedCapsuleRestoreAdmission());
  const bool capsule_admitted = state_witness_admission.has_value();
  if (capsule_admitted) {
    ABSL_ASSIGN_OR_RETURN(
        const CapsuleRestoreAuthority authority,
        BuildCapsuleRestoreAuthority(*state_witness_admission));
    ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAuthorityForAdmission(
        authority, *state_witness_admission));
    if (authority.capability.session_identity != agent.runtime_identity) {
      return absl::DataLossError(
          "DPM CapsuleRestore capability belongs to another agent runtime.");
    }
    checkpoints.session_handoff_capability = authority.capability;
    checkpoints.capsule_restore_admission_record_id =
        authority.admission_record_id;
    checkpoints.operational_coverage =
        state_witness_admission->operational_coverage;
    ABSL_ASSIGN_OR_RETURN(
        const CapsuleRestoreRuntimeAuthority current_authority,
        RequireCurrentCapsuleRestoreAuthority());
    if (!SameCapsuleRestoreRuntimeAuthority(
            current_authority.admission, current_authority.authority,
            *state_witness_admission, authority)) {
      return absl::AbortedError(
          "CapsuleRestore authority changed while assembling product "
          "capabilities.");
    }
  }

  DPMCapabilities capabilities;
  capabilities.projection = std::move(projection);
  capabilities.agent = std::move(agent);
  capabilities.checkpoints = std::move(checkpoints);
  capabilities.guarantees = {
      DPMGuaranteeAvailability{
          .guarantee = DPMGuarantee::kLogAuthority,
          .available = true,
      },
      DPMGuaranteeAvailability{
          .guarantee = DPMGuarantee::kCanonicalWinnerReplay,
          .available = !exact_mode,
          .unavailable_reason =
              exact_mode ? "active_mode_is_exact_regeneration" : "",
      },
      DPMGuaranteeAvailability{
          .guarantee = DPMGuarantee::kCapsuleRestore,
          .available =
              capabilities.checkpoints.checkpoint_transport_configured &&
              capabilities.checkpoints
                  .operation_evidence_transport_configured &&
              capsule_admitted,
          .unavailable_reason =
              !capabilities.checkpoints.checkpoint_transport_configured
                  ? "checkpoint_transport_not_configured"
                  : (!capabilities.checkpoints
                           .operation_evidence_transport_configured
                         ? "operation_evidence_transport_not_configured"
                         : (!capsule_admitted ? "capsule_restore_not_admitted"
                         : "")),
      },
      DPMGuaranteeAvailability{
          .guarantee = DPMGuarantee::kExactRegeneration,
          .available = exact_mode,
          .unavailable_reason =
              exact_mode ? "" : "active_mode_is_canonical_winner_replay",
      },
      DPMGuaranteeAvailability{
          .guarantee = DPMGuarantee::kFailClosedCapabilities,
          .available = true,
      },
  };
  const absl::Span<const DPMFeatureRestriction> restrictions =
      GetDPMFeatureRestrictions();
  capabilities.fail_closed_features.assign(restrictions.begin(),
                                            restrictions.end());
  return capabilities;
}

absl::StatusOr<std::optional<DPMTurnResult>> DPMEngine::RecoverCommittedTurn(
    const DPMLogSnapshot& snapshot, const DPMTurnRequest& request) const {
  const DPMEvent* committed = nullptr;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kModelTurn || !event.turn_receipt ||
        event.turn_receipt->operation_id != request.operation_id) {
      continue;
    }
    if (committed != nullptr) {
      return absl::DataLossError(
          "DPM log contains multiple committed turns for one operation id.");
    }
    committed = &event;
  }
  if (committed == nullptr) {
    return std::optional<DPMTurnResult>();
  }

  TranscriptCursor transcript;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &transcript));
    if (event.index == committed->index) break;
  }

  const DPMTurnReceipt& receipt = *committed->turn_receipt;
  ABSL_RETURN_IF_ERROR(
      ValidateInputEvent(snapshot.events[receipt.input_event_index], request));
  if (request.response_timestamp_us.has_value() &&
      committed->timestamp_us != *request.response_timestamp_us) {
    return absl::AlreadyExistsError(
        "DPM operation id is already bound to a different response "
        "timestamp.");
  }
  DPMTurnResult result;
  result.projected_memory = receipt.projected_memory;
  result.decision_output = receipt.decision_output;
  result.decision_token_ids = receipt.decision_token_ids;
  result.input_event_index = receipt.input_event_index;
  result.response_event_index = receipt.response_event_index;
  result.projection_manifest_hash = receipt.projection_manifest.manifest_hash;
  result.projection_replay_mode = receipt.projection_manifest.replay_mode;
  result.projection_execution_evidence_hash =
      receipt.projection_manifest.execution_evidence_hash;
  result.projection_exact_profile_id =
      receipt.projection_manifest.exact_profile_id;
  result.projection_exact_profile_admission_record_id =
      receipt.projection_manifest.exact_profile_admission_record_id;
  result.agent_replay_mode = receipt.agent_replay_mode;
  result.agent_execution_evidence_hash = receipt.agent_execution_evidence_hash;
  result.agent_exact_profile_id = receipt.agent_exact_profile_id;
  result.agent_exact_profile_admission_record_id =
      receipt.agent_exact_profile_admission_record_id;
  result.agent_exact_output_evidence_hash =
      receipt.agent_exact_output_evidence_hash;
  result.agent_exact_logit_frame_count = receipt.agent_exact_logit_frame_count;
  result.agent_reused_canonical_winner = receipt.agent_reused_canonical_winner;
  result.agent_rematerialized_canonical_winner =
      receipt.agent_rematerialized_canonical_winner;
  result.agent_producing_session_matched_output =
      receipt.agent_producing_session_matched_output;
  result.agent_worker_capsule_matched_output =
      receipt.checkpoint_capture_origin ==
          DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker &&
      receipt.agent_exact_worker_checkpoint_provenance.has_value();
  result.session_checkpoint_id = receipt.session_checkpoint_id;
  result.restored_from_session_checkpoint_id =
      receipt.restored_from_session_checkpoint_id;
  result.checkpoint_capture_origin = receipt.checkpoint_capture_origin;
  result.agent_worker_prefill_mode = receipt.agent_worker_prefill_mode;
  result.agent_physical_execution_plan_hash =
      receipt.agent_physical_execution_plan_hash;
  result.agent_capsule_restore_admission_record_id =
      receipt.agent_capsule_restore_admission_record_id;
  result.agent_capsule_restore_capability_id =
      receipt.agent_capsule_restore_capability_id;
  result.agent_capsule_restore_coverage_id =
      receipt.agent_capsule_restore_coverage_id;
  result.agent_exact_worker_checkpoint_provenance =
      receipt.agent_exact_worker_checkpoint_provenance;
  result.agent_prepared_prefill_work = receipt.agent_prepared_prefill_work;
  result.published_checkpoint_capture = receipt.published_checkpoint_capture;
  result.restored_checkpoint_capture_evidence_id =
      receipt.restored_checkpoint_capture_evidence_id;
  result.agent_capsule_restore_evidence_id =
      receipt.agent_capsule_restore_evidence_id;
  result.restored_session_checkpoint =
      receipt.restored_from_session_checkpoint_id.has_value();
  result.recovered_committed_turn = true;
  return std::optional<DPMTurnResult>(std::move(result));
}

absl::Status DPMEngine::ValidateRestoreCandidate(
    const DPMLogSnapshot& current, const DPMProjectionOutcome& projection,
    const RestoreCandidate& candidate,
    const CapsuleRestoreRuntimeAuthority& current_authority) const {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAuthorityForAdmission(
      current_authority.authority, current_authority.admission));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceCheckpointBinding(
      candidate.artifact, candidate.receipt, candidate.capture_evidence,
      current_authority.authority, current_authority.admission));
  const DPMSessionCheckpointDescriptor& descriptor =
      candidate.artifact.descriptor;
  const DPMTurnReceipt& receipt = candidate.receipt;
  const DPMReplayMode active_replay_mode = agent_runtime_->GetReplayMode();

  if (descriptor.format_version !=
          DPMSessionCheckpointDescriptor::kFormatVersion ||
      receipt.format_version != DPMTurnReceipt::kFormatVersion ||
      !receipt.session_checkpoint_id ||
      !receipt.published_checkpoint_capture.has_value() ||
      *receipt.session_checkpoint_id != descriptor.descriptor_id ||
      receipt.published_checkpoint_capture->capture_plan_hash !=
          descriptor.capsule_capture_plan_hash ||
      receipt.published_checkpoint_capture->capture_evidence_id !=
          candidate.capture_evidence.evidence_id ||
      descriptor.log_id != current.log_id ||
      descriptor.log_id != receipt.projection_manifest.log_id ||
      descriptor.stage != DPMSessionCheckpointStage::kAgentDecision ||
      descriptor.response_event_index != receipt.response_event_index ||
      descriptor.source_event_count != receipt.response_event_index ||
      descriptor.source_event_count !=
          receipt.projection_manifest.source_event_count ||
      descriptor.replay_mode != active_replay_mode ||
      receipt.agent_replay_mode != active_replay_mode ||
      descriptor.source_prefix_hash !=
          receipt.projection_manifest.source_prefix_hash ||
      descriptor.response_event_index >= current.events.size() ||
      descriptor.response_event_index >=
          projection.manifest.input_event_index) {
    return absl::FailedPreconditionError(
        "DPM checkpoint descriptor is not attached to this log receipt.");
  }
  const DPMEvent& response = current.events[descriptor.response_event_index];
  if (response.kind != DPMEvent::Kind::kModelTurn || !response.turn_receipt ||
      response.turn_receipt->session_checkpoint_id !=
          receipt.session_checkpoint_id ||
      response.turn_receipt->published_checkpoint_capture !=
          receipt.published_checkpoint_capture ||
      response.turn_receipt->operation_id != receipt.operation_id ||
      response.turn_receipt->agent_request_hash != receipt.agent_request_hash ||
      response.turn_receipt->agent_transcript_hash !=
          receipt.agent_transcript_hash) {
    return absl::FailedPreconditionError(
        "DPM checkpoint receipt is no longer authoritative.");
  }
  if (descriptor.replay_mode != receipt.agent_replay_mode) {
    return absl::FailedPreconditionError(
        "DPM checkpoint replay mode differs from its authoritative receipt.");
  }
  if (descriptor.projection_request_hash !=
          receipt.projection_manifest.request_hash ||
      descriptor.projection_manifest_hash !=
          receipt.projection_manifest.manifest_hash ||
      descriptor.correction_digest !=
          receipt.projection_manifest.correction_digest ||
      descriptor.agent_request_hash != receipt.agent_request_hash ||
      descriptor.agent_transcript_hash != receipt.agent_transcript_hash ||
      descriptor.session_identity != receipt.agent_session_identity ||
      descriptor.correction_digest != projection.manifest.correction_digest ||
      !(descriptor.session_identity ==
        agent_runtime_->GetSessionHandoffIdentity()) ||
      descriptor.key_id != config_.checkpoint_key_id ||
      descriptor.capsule_restore_capability_id !=
          current_authority.authority.capability.capability_id ||
      descriptor.capsule_restore_admission_record_id !=
          current_authority.authority.admission_record_id ||
      descriptor.capsule_restore_coverage_id !=
          current_authority.authority.coverage_id) {
    return absl::FailedPreconditionError(
        "DPM checkpoint artifact is incompatible with the active turn.");
  }

  if (active_replay_mode == DPMReplayMode::kExactRegeneration) {
    ABSL_ASSIGN_OR_RETURN(const std::optional<Hash256> current_profile_id,
                          agent_runtime_->GetExactProfileId());
    ABSL_ASSIGN_OR_RETURN(
        const std::optional<Hash256> current_profile_admission_id,
        agent_runtime_->GetExactProfileAdmissionRecordId());
    if (!current_profile_id.has_value() ||
        !current_profile_admission_id.has_value() ||
        current_authority.authority.capability.exact_profile_id !=
            *current_profile_id ||
        descriptor.exact_profile_id != current_profile_id ||
        receipt.agent_exact_profile_id != current_profile_id ||
        descriptor.exact_profile_admission_record_id !=
            *current_profile_admission_id ||
        receipt.agent_exact_profile_admission_record_id !=
            current_profile_admission_id) {
      return absl::FailedPreconditionError(
          "The current exact consumer does not match its loaded and admitted "
          "CapsuleRestore profile or the source checkpoint's exact "
          "admission.");
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      Hash256 prefix_hash,
      SnapshotPrefixHashAt(current, descriptor.source_event_count));
  if (prefix_hash != descriptor.source_prefix_hash) {
    return absl::FailedPreconditionError(
        "DPM checkpoint raw-log prefix has changed.");
  }

  const CapsuleCapturePlan& capture_plan = candidate.capture_evidence.plan;
  uint64_t expected_prefill_event_range_start = 0;
  switch (capture_plan.capture_basis) {
    case CapsuleCaptureBasis::kRootFreshSession:
      expected_prefill_event_range_start =
          ActiveCorrectionEpochStart(current, descriptor.response_event_index);
      break;
    case CapsuleCaptureBasis::kVerifiedParentRestore: {
      if (!capture_plan.parent_checkpoint_id.has_value() ||
          !capture_plan.parent_response_event_index.has_value() ||
          !capture_plan.parent_restore_evidence_id.has_value()) {
        return absl::FailedPreconditionError(
            "DPM descendant capture omitted its verified parent lineage.");
      }
      const uint64_t parent_response_index =
          *capture_plan.parent_response_event_index;
      if (parent_response_index >= descriptor.response_event_index ||
          parent_response_index >= current.events.size()) {
        return absl::FailedPreconditionError(
            "DPM descendant capture names an invalid parent response.");
      }
      const DPMEvent& parent_response = current.events[parent_response_index];
      if (parent_response.kind != DPMEvent::Kind::kModelTurn ||
          !parent_response.turn_receipt.has_value() ||
          parent_response.turn_receipt->session_checkpoint_id !=
              capture_plan.parent_checkpoint_id ||
          !parent_response.turn_receipt->published_checkpoint_capture
               .has_value() ||
          !candidate.capture_evidence.parent_restore_evidence.has_value() ||
          parent_response.turn_receipt->published_checkpoint_capture
                  ->capture_evidence_id !=
              candidate.capture_evidence.parent_restore_evidence->plan
                  .source_capture_evidence_id ||
          candidate.capture_evidence.parent_restore_evidence->evidence_id !=
              *capture_plan.parent_restore_evidence_id) {
        return absl::FailedPreconditionError(
            "DPM descendant capture parent is not an authoritative raw-log "
            "checkpoint and restore-evidence lineage.");
      }
      expected_prefill_event_range_start = parent_response_index + 1;
      if (ActiveCorrectionEpochStart(current, descriptor.response_event_index) >
          expected_prefill_event_range_start) {
        return absl::FailedPreconditionError(
            "DPM descendant capture crosses a correction boundary.");
      }
      break;
    }
    default:
      return absl::FailedPreconditionError(
          "DPM checkpoint has an unsupported capture basis.");
  }
  if (capture_plan.prefill.event_range_start !=
          expected_prefill_event_range_start ||
      capture_plan.prefill.event_range_end != descriptor.source_event_count) {
    return absl::FailedPreconditionError(
        "DPM checkpoint physical prefill interval is not derived from its "
        "authoritative correction epoch or verified parent.");
  }

  TranscriptCursor transcript;
  bool found = false;
  Hash256 candidate_transcript_hash;
  Hash256 candidate_transcript_prefix_hash;
  std::vector<DPMAgentGenerationRequest::PrefillChunk>
      authoritative_prefill_chunks;
  for (const DPMEvent& event : current.events) {
    if (found && event.kind == DPMEvent::Kind::kCorrection) {
      // Corrections are immutable epoch boundaries. Even a hypothetical hash
      // collision must not permit a descendant capsule to cross one.
      return absl::FailedPreconditionError(
          "DPM correction event invalidates the selected checkpoint "
          "descendant interval.");
    }
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    if (event.index == descriptor.response_event_index) {
      TranscriptCursor prefix = transcript;
      if (!prefix.correction_digest.has_value() ||
          *prefix.correction_digest !=
              receipt.projection_manifest.correction_digest) {
        ResetTranscriptHasher(receipt.projection_manifest.correction_digest,
                              &prefix);
      }
      UpdateBytesFrame('I', receipt.canonical_agent_input, &prefix.hasher);
      candidate_transcript_prefix_hash = SnapshotDigest(prefix.hasher);
      authoritative_prefill_chunks.push_back(
          TextChunk(receipt.canonical_agent_input));
    }
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(current, event, &transcript));
    if (event.index >= expected_prefill_event_range_start &&
        event.index < descriptor.response_event_index) {
      authoritative_prefill_chunks.push_back(
          TextChunk(event.turn_receipt->canonical_agent_input));
      authoritative_prefill_chunks.push_back(
          TokenChunk(event.turn_receipt->decision_token_ids));
    }
    if (event.index == descriptor.response_event_index) {
      found = true;
      candidate_transcript_hash = SnapshotDigest(transcript.hasher);
      continue;
    }
    if (found && event.turn_receipt->projection_manifest.correction_digest !=
                     projection.manifest.correction_digest) {
      return absl::FailedPreconditionError(
          "DPM correction epoch changed after the checkpoint.");
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<CapsuleCanonicalPrefillChunk>
          authoritative_capsule_chunks,
      ToCapsuleCanonicalPrefillChunks(authoritative_prefill_chunks));
  if (!found || candidate_transcript_hash != descriptor.agent_transcript_hash ||
      candidate_transcript_prefix_hash !=
          capture_plan.agent_transcript_prefix_hash ||
      authoritative_capsule_chunks != capture_plan.prefill.canonical_chunks) {
    return absl::FailedPreconditionError(
        "DPM checkpoint transcript or physical prefill chunks are not "
        "derived from the authoritative raw log.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<DPMEngine::RestoreCandidate>>
DPMEngine::FindRestoreCandidate(
    const DPMLogSnapshot& current, const DPMProjectionOutcome& projection,
    const CapsuleRestoreRuntimeAuthority& current_authority) const {
  if (!config_.restore_session_checkpoints ||
      checkpoint_repository_ == nullptr || evidence_repository_ == nullptr) {
    return std::optional<RestoreCandidate>();
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAuthorityForAdmission(
      current_authority.authority, current_authority.admission));
  const DPMReplayMode active_replay_mode = agent_runtime_->GetReplayMode();
  std::optional<Hash256> current_exact_profile_id;
  std::optional<Hash256> current_exact_profile_admission_id;
  if (active_replay_mode == DPMReplayMode::kExactRegeneration) {
    ABSL_ASSIGN_OR_RETURN(current_exact_profile_id,
                          agent_runtime_->GetExactProfileId());
    ABSL_ASSIGN_OR_RETURN(current_exact_profile_admission_id,
                          agent_runtime_->GetExactProfileAdmissionRecordId());
    if (!current_exact_profile_id.has_value() ||
        !current_exact_profile_admission_id.has_value()) {
      return absl::FailedPreconditionError(
          "Exact DPM restore selection lacks the current loaded profile and "
          "admission identities.");
    }
  }
  const FreshWorkerAuthentication evidence_authentication{
      .key_id = config_.operation_evidence_key_id,
      .authentication_key = config_.operation_evidence_authentication_key,
  };
  for (auto it = current.events.rbegin(); it != current.events.rend(); ++it) {
    if (it->kind == DPMEvent::Kind::kCorrection) {
      // The newest correction is the immutable lower bound of the active
      // checkpoint interval. Every older capsule belongs to an invalidated
      // lineage, so do not scan or touch its repository objects.
      break;
    }
    if (it->kind != DPMEvent::Kind::kModelTurn || !it->turn_receipt ||
        !it->turn_receipt->session_checkpoint_id) {
      continue;
    }
    const DPMTurnReceipt& receipt = *it->turn_receipt;
    if (receipt.response_event_index >= projection.manifest.input_event_index ||
        receipt.projection_manifest.correction_digest !=
            projection.manifest.correction_digest ||
        receipt.agent_replay_mode != active_replay_mode ||
        (active_replay_mode == DPMReplayMode::kExactRegeneration &&
         (receipt.agent_exact_profile_id != current_exact_profile_id ||
          receipt.agent_exact_profile_admission_record_id !=
              current_exact_profile_admission_id)) ||
        receipt.agent_session_identity !=
            agent_runtime_->GetSessionHandoffIdentity() ||
        receipt.format_version != DPMTurnReceipt::kFormatVersion ||
        !receipt.published_checkpoint_capture.has_value() ||
        receipt.agent_capsule_restore_capability_id !=
            std::optional<Hash256>(
                current_authority.authority.capability.capability_id) ||
        receipt.agent_capsule_restore_admission_record_id !=
            std::optional<Hash256>(
                current_authority.authority.admission_record_id) ||
        receipt.agent_capsule_restore_coverage_id !=
            std::optional<Hash256>(current_authority.authority.coverage_id)) {
      // Receipt metadata is immutable and authoritative. Filter invalidated
      // epochs and incompatible runtime branches before touching disposable
      // checkpoint storage; an older compatible interval may still exist.
      continue;
    }
    absl::StatusOr<DPMSessionCheckpointArtifact> artifact =
        checkpoint_repository_->Get(*receipt.session_checkpoint_id);
    if (!artifact.ok()) {
      // Checkpoints are authenticated disposable accelerators. Missing or
      // unavailable storage must not prevent reconstruction from the log.
      continue;
    }
    absl::StatusOr<CapsuleCaptureEvidence> capture_evidence =
        evidence_repository_->GetCapture(
            *receipt.session_checkpoint_id,
            receipt.published_checkpoint_capture->capture_evidence_id,
            evidence_authentication);
    if (!capture_evidence.ok()) {
      continue;
    }
    RestoreCandidate candidate{receipt, std::move(*artifact),
                               std::move(*capture_evidence)};
    if (ValidateRestoreCandidate(current, projection, candidate,
                                 current_authority)
            .ok()) {
      return std::optional<RestoreCandidate>(std::move(candidate));
    }
  }
  return std::optional<RestoreCandidate>();
}

absl::StatusOr<std::vector<DPMAgentGenerationRequest::PrefillChunk>>
DPMEngine::BuildFullTranscriptChunks(
    const DPMLogSnapshot& snapshot, absl::string_view current_agent_input,
    const Hash256& current_correction_digest) const {
  std::vector<DPMAgentGenerationRequest::PrefillChunk> chunks;
  TranscriptCursor transcript;
  bool in_current_epoch = false;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &transcript));
    if (event.turn_receipt->projection_manifest.correction_digest !=
        current_correction_digest) {
      chunks.clear();
      in_current_epoch = false;
      continue;
    }
    if (!in_current_epoch) {
      chunks.clear();
      in_current_epoch = true;
    }
    chunks.push_back(TextChunk(event.turn_receipt->canonical_agent_input));
    if (!event.turn_receipt->decision_token_ids.empty()) {
      chunks.push_back(TokenChunk(event.turn_receipt->decision_token_ids));
    }
  }
  chunks.push_back(TextChunk(current_agent_input));
  return chunks;
}

absl::StatusOr<std::vector<DPMAgentGenerationRequest::PrefillChunk>>
DPMEngine::BuildDeltaTranscriptChunks(
    const DPMLogSnapshot& snapshot, uint64_t restored_response_event_index,
    absl::string_view current_agent_input,
    const Hash256& current_correction_digest) const {
  if (restored_response_event_index >= snapshot.events.size() ||
      snapshot.events[restored_response_event_index].kind !=
          DPMEvent::Kind::kModelTurn ||
      !snapshot.events[restored_response_event_index].turn_receipt) {
    return absl::FailedPreconditionError(
        "DPM restore point is not a model-turn event in this log.");
  }
  std::vector<DPMAgentGenerationRequest::PrefillChunk> chunks;
  const Hash256 restored_correction_digest =
      snapshot.events[restored_response_event_index]
          .turn_receipt->projection_manifest.correction_digest;
  if (restored_correction_digest != current_correction_digest) {
    return absl::FailedPreconditionError(
        "DPM restore point is outside the active correction epoch.");
  }
  TranscriptCursor transcript;
  for (const DPMEvent& event : snapshot.events) {
    if (event.index > restored_response_event_index &&
        event.kind == DPMEvent::Kind::kCorrection) {
      return absl::FailedPreconditionError(
          "DPM correction event invalidates the selected checkpoint "
          "descendant interval.");
    }
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &transcript));
    if (event.index <= restored_response_event_index) continue;
    if (event.turn_receipt->projection_manifest.correction_digest !=
        restored_correction_digest) {
      return absl::FailedPreconditionError(
          "DPM correction epoch changed after the selected restore point.");
    }
    chunks.push_back(TextChunk(event.turn_receipt->canonical_agent_input));
    if (!event.turn_receipt->decision_token_ids.empty()) {
      chunks.push_back(TokenChunk(event.turn_receipt->decision_token_ids));
    }
  }
  chunks.push_back(TextChunk(current_agent_input));
  return chunks;
}

absl::StatusOr<std::string> DPMEngine::BuildCanonicalAgentInput(
    const DPMTurnRequest& request, const DPMLogSnapshot& source_snapshot,
    uint64_t input_event_index, const DPMProjectionOutcome& projection) const {
  ABSL_RETURN_IF_ERROR(ValidateSnapshotShape(source_snapshot));
  if (input_event_index >= source_snapshot.events.size()) {
    return absl::InvalidArgumentError(
        "DPM agent input names an event outside its source snapshot.");
  }
  if (request.payload.size() > kMaximumDPMEventPayloadBytes ||
      projection.projected_memory.size() > kMaximumDPMEventPayloadBytes ||
      request.case_id.size() > kMaximumDPMProjectionIdentityBytes) {
    return absl::ResourceExhaustedError(
        "DPM canonical agent input field exceeds its product bound.");
  }
  const absl::string_view model_memory =
      projection.model_visible_memory.has_value()
          ? absl::string_view(*projection.model_visible_memory)
          : absl::string_view(projection.projected_memory);
  const absl::string_view model_input =
      projection.model_visible_input.has_value()
          ? absl::string_view(*projection.model_visible_input)
          : absl::string_view(request.payload);
  std::string input;
  input.reserve(model_input.size() + model_memory.size() + 64);
  input.append("DPM1\nUse MEMORY for INPUT.\n");
  AppendTextSection("MEMORY", model_memory, &input);
  absl::StrAppend(&input, "KIND ", static_cast<int>(request.kind), "\n");
  AppendTextSection("INPUT", model_input, &input);
  if (input.size() > kMaximumDPMCanonicalAgentInputBytes) {
    return absl::ResourceExhaustedError(
        "DPM canonical agent input exceeds its derived product bound.");
  }
  if (!IsValidUtf8(input)) {
    return absl::FailedPreconditionError(
        "DPM projection produced an agent input that is not valid UTF-8.");
  }
  return input;
}

absl::StatusOr<Hash256> DPMEngine::ComputeTranscriptHash(
    const DPMLogSnapshot& snapshot, absl::string_view current_agent_input,
    absl::string_view current_decision,
    const std::vector<int>& current_decision_token_ids,
    const Hash256& current_correction_digest) const {
  TranscriptCursor transcript;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &transcript));
  }
  if (!transcript.correction_digest.has_value() ||
      *transcript.correction_digest != current_correction_digest) {
    ResetTranscriptHasher(current_correction_digest, &transcript);
  }
  UpdateBytesFrame('I', current_agent_input, &transcript.hasher);
  UpdateTokenFrame(current_decision_token_ids, &transcript.hasher);
  UpdateBytesFrame('O', current_decision, &transcript.hasher);
  return transcript.hasher.Finalize();
}

absl::StatusOr<Hash256> DPMEngine::ComputeTranscriptPrefixHash(
    const DPMLogSnapshot& snapshot, absl::string_view current_agent_input,
    const Hash256& current_correction_digest) const {
  TranscriptCursor transcript;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &transcript));
  }
  if (!transcript.correction_digest.has_value() ||
      *transcript.correction_digest != current_correction_digest) {
    ResetTranscriptHasher(current_correction_digest, &transcript);
  }
  UpdateBytesFrame('I', current_agent_input, &transcript.hasher);
  return transcript.hasher.Finalize();
}

absl::StatusOr<DPMEngine::CheckpointCapture> DPMEngine::CaptureProducingSession(
    Engine::Session* session, const DPMLogSnapshot& source_snapshot,
    uint64_t response_event_index, const DPMProjectionOutcome& projection,
    const Hash256& agent_request_hash, const Hash256& transcript_prefix_hash,
    const Hash256& transcript_hash,
    const CapsuleRestoreRuntimeAuthority& capsule_restore_authority,
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>&
        physical_prefill_chunks,
    const DPMPreparedPrefillPlan& prepared_prefill_plan,
    const Hash256& producing_output_evidence_hash,
    uint32_t generated_token_count, uint64_t prefill_event_range_start,
    const std::optional<CapsuleRestoreEvidence>& parent_restore_evidence,
    absl::string_view operation_id, int64_t created_unix_micros) const {
  if (session == nullptr) {
    return absl::InvalidArgumentError(
        "Cannot capture a null DPM producing session.");
  }
  if (operation_id.empty() || generated_token_count == 0 ||
      IsZeroHash(producing_output_evidence_hash)) {
    return absl::InvalidArgumentError(
        "DPM live capture lacks its operation or producing-output evidence.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAuthorityForAdmission(
      capsule_restore_authority.authority,
      capsule_restore_authority.admission));
  ABSL_ASSIGN_OR_RETURN(const CapsuleRestoreRuntimeAuthority authority_before,
      RequireCurrentCapsuleRestoreAuthority());
  if (!SameCapsuleRestoreRuntimeAuthority(
          authority_before.admission, authority_before.authority,
          capsule_restore_authority.admission,
          capsule_restore_authority.authority)) {
    return absl::AbortedError(
        "CapsuleRestore authority changed before live-parent capture.");
  }
  ABSL_ASSIGN_OR_RETURN(SessionHandoffIdentity session_identity,
                        session->GetSessionHandoffIdentity());
  if (session_identity != agent_runtime_->GetSessionHandoffIdentity() ||
      session_identity !=
          capsule_restore_authority.authority.capability.session_identity) {
    return absl::FailedPreconditionError(
        "DPM producing session identity does not match its loaded runtime.");
  }

  if (parent_restore_evidence.has_value()) {
    ABSL_RETURN_IF_ERROR(
        ValidateCapsuleRestoreEvidence(*parent_restore_evidence));
    if (prepared_prefill_plan.start_kind !=
            DPMPreparedPrefillStartKind::kOwnPositionRestore ||
        prepared_prefill_plan.restore_checkpoint_id !=
            std::optional<Hash256>(
                parent_restore_evidence->plan.checkpoint_id) ||
        prefill_event_range_start !=
            parent_restore_evidence->plan.checkpoint_state
                    .response_event_index +
                1 ||
        !(prepared_prefill_plan ==
          parent_restore_evidence->plan.prefill.prepared_plan)) {
      return absl::FailedPreconditionError(
          "DPM descendant capture differs from its verified parent restore "
          "and exact delta plan.");
    }
  } else if (prepared_prefill_plan.start_kind !=
                 DPMPreparedPrefillStartKind::kFreshSession ||
             prepared_prefill_plan.restore_checkpoint_id.has_value() ||
             prefill_event_range_start !=
                 ActiveCorrectionEpochStart(source_snapshot)) {
    return absl::FailedPreconditionError(
        "DPM root capture is not a fresh full prefill of the current "
        "correction epoch.");
  }

  ABSL_ASSIGN_OR_RETURN(
      CapsulePrefillPlan prefill_plan,
      BuildCapsulePrefillPlan(physical_prefill_chunks, prepared_prefill_plan,
                              prefill_event_range_start,
                              source_snapshot.events.size()));
  CapsuleCapturePlan capture_plan;
  capture_plan.authority = capsule_restore_authority.authority;
  capture_plan.capture_basis = parent_restore_evidence.has_value()
                                   ? CapsuleCaptureBasis::kVerifiedParentRestore
                                   : CapsuleCaptureBasis::kRootFreshSession;
  capture_plan.checkpoint_state = CapsuleDPMCheckpointState{
      .log_id = source_snapshot.log_id,
      .source_event_count = source_snapshot.events.size(),
      .source_prefix_hash = source_snapshot.prefix_hash,
      .response_event_index = response_event_index,
      .projection_request_hash = projection.manifest.request_hash,
      .projection_manifest_hash = projection.manifest.manifest_hash,
      .correction_digest = projection.manifest.correction_digest,
      .agent_transcript_hash = transcript_hash,
      .logical_agent_request_hash = agent_request_hash,
  };
  capture_plan.agent_transcript_prefix_hash = transcript_prefix_hash;
  capture_plan.prefill = std::move(prefill_plan);
  capture_plan.producing_output_evidence_hash = producing_output_evidence_hash;
  capture_plan.generated_token_count = generated_token_count;
  if (capture_plan.prefill.end_step >
      (std::numeric_limits<uint32_t>::max)() - generated_token_count) {
    return absl::ResourceExhaustedError(
        "DPM captured continuation exceeds the evidence step domain.");
  }
  capture_plan.capture_end_step =
      capture_plan.prefill.end_step + generated_token_count;
  capture_plan.checkpoint_authentication_key_id = config_.checkpoint_key_id;
  Hash256 transient_parent_id;
  if (parent_restore_evidence.has_value()) {
    capture_plan.parent_checkpoint_id =
        parent_restore_evidence->plan.checkpoint_id;
    capture_plan.parent_response_event_index =
        parent_restore_evidence->plan.checkpoint_state.response_event_index;
    capture_plan.parent_restore_evidence_id =
        parent_restore_evidence->evidence_id;
    transient_parent_id = parent_restore_evidence->plan.checkpoint_id;
  }
  ABSL_ASSIGN_OR_RETURN(capture_plan.plan_hash,
                        ComputeCapsuleCapturePlanHash(capture_plan));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCapturePlan(capture_plan));

  SessionHandoffOptions transient_options;
  transient_options.key_id = std::string(kFreshWorkerTransientProducingKeyId);
  transient_options.authentication_key = DeriveRequestTransientKey(
      config_.checkpoint_authentication_key, kWinnerCaptureTransientKeyDomain,
      agent_request_hash, transient_parent_id, operation_id);
  transient_options.expected_identity = session_identity;
  SessionHandoffOptions durable_options;
  durable_options.key_id = config_.checkpoint_key_id;
  durable_options.authentication_key = config_.checkpoint_authentication_key;
  durable_options.expected_identity = session_identity;

  std::string transient_envelope;
  StringByteSink transient_sink(&transient_envelope);
  ABSL_ASSIGN_OR_RETURN(
      const SessionContinuationStateWitness producer_before_export,
      session->ExportHandoffToWithWitness(transient_options, &transient_sink));
  std::string repeated_transient_envelope;
  StringByteSink repeated_sink(&repeated_transient_envelope);
  ABSL_ASSIGN_OR_RETURN(
      const SessionContinuationStateWitness producer_after_export,
      session->ExportHandoffToWithWitness(transient_options, &repeated_sink));
  if (transient_envelope != repeated_transient_envelope ||
      producer_before_export != producer_after_export) {
    return absl::DataLossError(
        "DPM producing-session exports are not canonical and observational.");
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> fresh_target,
                        agent_runtime_->CreateSession());
  if (fresh_target == nullptr) {
    return absl::InternalError(
        "DPM capture runtime returned a null fresh import target.");
  }
  ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity fresh_target_identity,
      fresh_target->GetSessionHandoffIdentity());
  if (fresh_target_identity != session_identity) {
    return absl::FailedPreconditionError(
        "DPM capture fresh target belongs to another loaded runtime.");
  }
  StringByteSource transient_import_source(transient_envelope);
  ABSL_ASSIGN_OR_RETURN(
      const SessionContinuationStateWitness fresh_import_target,
      fresh_target->ImportHandoffFromWithWitness(transient_import_source,
                                                 transient_options));
  if (fresh_import_target != producer_before_export) {
    return absl::DataLossError(
        "DPM fresh import did not reproduce the producing continuation.");
  }

  CheckpointCapture capture;
  StringByteSource transient_rewrap_source(transient_envelope);
  StringByteSink durable_sink(&capture.artifact.authenticated_envelope);
  ABSL_ASSIGN_OR_RETURN(
      const SessionHandoffReauthenticationEvidence reauthentication,
      ReauthenticateSessionHandoffToWithEvidence(
          transient_rewrap_source, session_identity, transient_options,
          durable_options,
          kFreshWorkerTransientProducingToDurableReauthenticationPurpose,
          &durable_sink));

  DPMSessionCheckpointDescriptor& descriptor = capture.artifact.descriptor;
  descriptor.log_id = source_snapshot.log_id;
  descriptor.stage = DPMSessionCheckpointStage::kAgentDecision;
  descriptor.source_event_count = source_snapshot.events.size();
  descriptor.source_prefix_hash = source_snapshot.prefix_hash;
  descriptor.response_event_index = response_event_index;
  descriptor.projection_request_hash = projection.manifest.request_hash;
  descriptor.projection_manifest_hash = projection.manifest.manifest_hash;
  descriptor.correction_digest = projection.manifest.correction_digest;
  descriptor.agent_request_hash = agent_request_hash;
  descriptor.agent_transcript_hash = transcript_hash;
  descriptor.session_identity = session_identity;
  descriptor.key_id = durable_options.key_id;
  descriptor.envelope_hash = reauthentication.destination_envelope_hash;
  descriptor.envelope_size = reauthentication.destination_envelope_size;
  descriptor.created_unix_micros = created_unix_micros;
  descriptor.replay_mode = DPMReplayMode::kCanonicalWinnerReplay;
  descriptor.capture_origin = DPMCheckpointCaptureOrigin::kLiveParentSession;
  descriptor.restored_from_checkpoint_id = capture_plan.parent_checkpoint_id;
  descriptor.capsule_restore_capability_id =
      capsule_restore_authority.authority.capability.capability_id;
  descriptor.capsule_restore_admission_record_id =
      capsule_restore_authority.authority.admission_record_id;
  descriptor.capsule_restore_coverage_id =
      capsule_restore_authority.authority.coverage_id;
  descriptor.capsule_capture_plan_hash = capture_plan.plan_hash;
  ABSL_ASSIGN_OR_RETURN(
      descriptor.prepared_prefill_work,
      BuildDPMPreparedPrefillWorkBinding(prepared_prefill_plan));
  ABSL_ASSIGN_OR_RETURN(descriptor.descriptor_id,
                        ComputeDPMSessionCheckpointId(descriptor));
  ABSL_RETURN_IF_ERROR(ValidateDPMSessionCheckpointArtifact(capture.artifact));

  capture.capture_evidence.plan = std::move(capture_plan);
  capture.capture_evidence.checkpoint_id = descriptor.descriptor_id;
  capture.capture_evidence.checkpoint_envelope_hash = descriptor.envelope_hash;
  capture.capture_evidence.checkpoint_envelope_size = descriptor.envelope_size;
  capture.capture_evidence.checkpoint_authentication_key_id = descriptor.key_id;
  capture.capture_evidence.checkpoint_history_token_bytes_hash =
      producer_before_export.processed_history_token_bytes_hash;
  capture.capture_evidence.producer_before_export = producer_before_export;
  capture.capture_evidence.producer_after_export = producer_after_export;
  capture.capture_evidence.fresh_import_target = fresh_import_target;
  capture.capture_evidence.transient_to_durable_reauthentication =
      reauthentication;
  capture.capture_evidence.parent_restore_evidence = parent_restore_evidence;
  ABSL_ASSIGN_OR_RETURN(
      capture.capture_evidence.evidence_id,
      ComputeCapsuleCaptureEvidenceId(capture.capture_evidence));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleCaptureEvidence(capture.capture_evidence));

  ABSL_ASSIGN_OR_RETURN(const CapsuleRestoreRuntimeAuthority authority_after,
      RequireCurrentCapsuleRestoreAuthority());
  if (!SameCapsuleRestoreRuntimeAuthority(
          authority_before.admission, authority_before.authority,
          authority_after.admission, authority_after.authority)) {
    return absl::AbortedError(
        "CapsuleRestore authority changed during live-parent capture.");
  }
  return capture;
}

absl::StatusOr<DPMEngine::CheckpointCapture>
DPMEngine::BuildExactWorkerCheckpointCapture(
    absl::string_view authenticated_envelope,
    const ExactRegenerationDPMAgentPhysicalExecution& physical_execution,
    const DPMLogSnapshot& source_snapshot, uint64_t response_event_index,
    const DPMProjectionOutcome& projection, const Hash256& agent_request_hash,
    const Hash256& transcript_prefix_hash, const Hash256& transcript_hash,
    const CapsuleRestoreRuntimeAuthority& capsule_restore_authority,
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>&
        physical_prefill_chunks,
    uint64_t prefill_event_range_start,
    const std::optional<CapsuleRestoreEvidence>& parent_restore_evidence,
    int64_t created_unix_micros) const {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAuthorityForAdmission(
      capsule_restore_authority.authority,
      capsule_restore_authority.admission));
  ABSL_ASSIGN_OR_RETURN(const CapsuleRestoreRuntimeAuthority authority_before,
      RequireCurrentCapsuleRestoreAuthority());
  if (!SameCapsuleRestoreRuntimeAuthority(
          authority_before.admission, authority_before.authority,
          capsule_restore_authority.admission,
          capsule_restore_authority.authority)) {
    return absl::AbortedError(
        "CapsuleRestore authority changed before exact checkpoint "
        "publication.");
  }
  ABSL_RETURN_IF_ERROR(ValidateExactRegenerationDPMAgentPhysicalExecution(
      physical_execution, static_cast<uint32_t>(config_.max_decision_tokens)));
  const DPMAgentReplayExecution& replay = physical_execution.replay_execution;
  if (replay.mode != DPMReplayMode::kExactRegeneration ||
      !replay.exact_profile_id.has_value() ||
      !replay.exact_profile_admission_record_id.has_value() ||
      !replay.exact_output_evidence_hash.has_value() ||
      IsZeroHash(capsule_restore_authority.authority.admission_record_id) ||
      IsZeroHash(
          capsule_restore_authority.authority.capability.capability_id) ||
      capsule_restore_authority.authority.capability.exact_profile_id !=
          *replay.exact_profile_id ||
      capsule_restore_authority.authority.capability.session_identity !=
          physical_execution.request_evidence.session_identity ||
      !physical_execution.run_zero_transient_producing_capsule_evidence
           .has_value() ||
      !physical_execution.durable_producing_capsule_evidence.has_value() ||
      physical_execution.request_evidence.runs.empty()) {
    return absl::FailedPreconditionError(
        "Exact DPM checkpoint capture lacks admitted run-zero producing "
        "capsule evidence.");
  }
  const FreshWorkerProducingCapsuleEvidence& transient =
      *physical_execution.run_zero_transient_producing_capsule_evidence;
  const FreshWorkerDurableProducingCapsuleEvidence& durable =
      *physical_execution.durable_producing_capsule_evidence;
  const ExactRegenerationRunEvidence& run_zero =
      physical_execution.request_evidence.runs.front();
  if (!replay.prepared_prefill_plan.has_value()) {
    return absl::FailedPreconditionError(
        "Exact DPM checkpoint capture lacks the agreed prepared plan.");
  }
  const DPMPreparedPrefillPlan& prepared_prefill_plan =
      *replay.prepared_prefill_plan;
  if (parent_restore_evidence.has_value()) {
    ABSL_RETURN_IF_ERROR(
        ValidateCapsuleRestoreEvidence(*parent_restore_evidence));
    if (physical_execution.restored_checkpoint_id !=
            std::optional<Hash256>(
                parent_restore_evidence->plan.checkpoint_id) ||
        prepared_prefill_plan.start_kind !=
            DPMPreparedPrefillStartKind::kOwnPositionRestore ||
        prefill_event_range_start !=
            parent_restore_evidence->plan.checkpoint_state
                    .response_event_index +
                1 ||
        !(prepared_prefill_plan ==
          parent_restore_evidence->plan.prefill.prepared_plan)) {
      return absl::FailedPreconditionError(
          "Exact descendant capture differs from its verified parent restore "
          "and agreed delta plan.");
    }
  } else if (physical_execution.restored_checkpoint_id.has_value() ||
             prepared_prefill_plan.start_kind !=
                 DPMPreparedPrefillStartKind::kFreshSession ||
             prefill_event_range_start !=
                 ActiveCorrectionEpochStart(source_snapshot)) {
    return absl::FailedPreconditionError(
        "Exact root capture is not a fresh full prefill of the current "
        "correction epoch.");
  }
  if (physical_execution.capture_run_policy !=
          ExactRegenerationCaptureRunPolicy::kRunZeroOnly ||
      run_zero.run_index != 0 ||
      durable.session_identity != agent_runtime_->GetSessionHandoffIdentity() ||
      transient.session_identity != durable.session_identity ||
      durable.key_id != config_.checkpoint_key_id ||
      durable.envelope_size != authenticated_envelope.size() ||
      durable.envelope_hash != Sha256(authenticated_envelope) ||
      durable.key_id !=
          capsule_restore_authority.admission.operational_coverage
              .operational_domain.checkpoint_authentication_key_id ||
      durable.output_evidence_hash != *replay.exact_output_evidence_hash ||
      transient.output_evidence_hash != *replay.exact_output_evidence_hash) {
    return absl::DataLossError(
        "Exact DPM checkpoint bytes or producing-worker evidence disagree "
        "with the consensus output and loaded runtime.");
  }

  ABSL_ASSIGN_OR_RETURN(
      CapsulePrefillPlan prefill_plan,
      BuildCapsulePrefillPlan(physical_prefill_chunks, prepared_prefill_plan,
                              prefill_event_range_start,
                              source_snapshot.events.size()));
  if (replay.decision_token_ids.size() >
      (std::numeric_limits<uint32_t>::max)()) {
    return absl::ResourceExhaustedError(
        "Exact DPM decision exceeds the capture token-count domain.");
  }
  const uint32_t generated_token_count =
      static_cast<uint32_t>(replay.decision_token_ids.size());
  CapsuleCapturePlan capture_plan;
  capture_plan.authority = capsule_restore_authority.authority;
  capture_plan.capture_basis = parent_restore_evidence.has_value()
                                   ? CapsuleCaptureBasis::kVerifiedParentRestore
                                   : CapsuleCaptureBasis::kRootFreshSession;
  capture_plan.checkpoint_state = CapsuleDPMCheckpointState{
      .log_id = source_snapshot.log_id,
      .source_event_count = source_snapshot.events.size(),
      .source_prefix_hash = source_snapshot.prefix_hash,
      .response_event_index = response_event_index,
      .projection_request_hash = projection.manifest.request_hash,
      .projection_manifest_hash = projection.manifest.manifest_hash,
      .correction_digest = projection.manifest.correction_digest,
      .agent_transcript_hash = transcript_hash,
      .logical_agent_request_hash = agent_request_hash,
  };
  capture_plan.agent_transcript_prefix_hash = transcript_prefix_hash;
  capture_plan.prefill = std::move(prefill_plan);
  capture_plan.producing_output_evidence_hash =
      *replay.exact_output_evidence_hash;
  capture_plan.generated_token_count = generated_token_count;
  if (capture_plan.prefill.end_step >
      (std::numeric_limits<uint32_t>::max)() - generated_token_count) {
    return absl::ResourceExhaustedError(
        "Exact DPM captured continuation exceeds the evidence step domain.");
  }
  capture_plan.capture_end_step =
      capture_plan.prefill.end_step + generated_token_count;
  capture_plan.checkpoint_authentication_key_id = durable.key_id;
  if (parent_restore_evidence.has_value()) {
    capture_plan.parent_checkpoint_id =
        parent_restore_evidence->plan.checkpoint_id;
    capture_plan.parent_response_event_index =
        parent_restore_evidence->plan.checkpoint_state.response_event_index;
    capture_plan.parent_restore_evidence_id =
        parent_restore_evidence->evidence_id;
  }
  ABSL_ASSIGN_OR_RETURN(capture_plan.plan_hash,
                        ComputeCapsuleCapturePlanHash(capture_plan));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCapturePlan(capture_plan));

  CheckpointCapture capture;
  capture.artifact.authenticated_envelope.assign(authenticated_envelope.data(),
                                                 authenticated_envelope.size());
  DPMSessionCheckpointDescriptor& descriptor = capture.artifact.descriptor;
  descriptor.log_id = source_snapshot.log_id;
  descriptor.stage = DPMSessionCheckpointStage::kAgentDecision;
  descriptor.source_event_count = source_snapshot.events.size();
  descriptor.source_prefix_hash = source_snapshot.prefix_hash;
  descriptor.response_event_index = response_event_index;
  descriptor.projection_request_hash = projection.manifest.request_hash;
  descriptor.projection_manifest_hash = projection.manifest.manifest_hash;
  descriptor.correction_digest = projection.manifest.correction_digest;
  descriptor.agent_request_hash = agent_request_hash;
  descriptor.agent_transcript_hash = transcript_hash;
  descriptor.session_identity = durable.session_identity;
  descriptor.key_id = durable.key_id;
  descriptor.envelope_hash = durable.envelope_hash;
  descriptor.envelope_size = durable.envelope_size;
  descriptor.created_unix_micros = created_unix_micros;
  descriptor.replay_mode = DPMReplayMode::kExactRegeneration;
  descriptor.capture_origin =
      DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker;
  descriptor.restored_from_checkpoint_id =
      physical_execution.restored_checkpoint_id;
  descriptor.exact_profile_id = replay.exact_profile_id;
  descriptor.exact_profile_admission_record_id =
      *replay.exact_profile_admission_record_id;
  descriptor.capsule_restore_capability_id =
      capsule_restore_authority.authority.capability.capability_id;
  descriptor.capsule_restore_admission_record_id =
      capsule_restore_authority.authority.admission_record_id;
  descriptor.capsule_restore_coverage_id =
      capsule_restore_authority.authority.coverage_id;
  descriptor.exact_request_execution_evidence_id =
      physical_execution.request_evidence.evidence_id;
  descriptor.worker_prefill_mode =
      ToCheckpointWorkerPrefillMode(physical_execution.prefill_mode);
  descriptor.execution_plan_hash =
      physical_execution.physical_execution_plan_hash;
  descriptor.exact_output_evidence_hash = *replay.exact_output_evidence_hash;
  descriptor.worker_provenance = DPMExactWorkerCheckpointProvenance{
      .run_index = run_zero.run_index,
      .execution_plan_hash = physical_execution.physical_execution_plan_hash,
      .request_envelope_hash = run_zero.request_envelope_hash,
      .result_envelope_hash = run_zero.result_envelope_hash,
      .transient_envelope_size = transient.transient_envelope_size,
      .transient_envelope_hash = transient.transient_envelope_hash,
      .output_evidence_hash = transient.output_evidence_hash,
  };
  descriptor.capsule_capture_plan_hash = capture_plan.plan_hash;
  ABSL_ASSIGN_OR_RETURN(
      descriptor.prepared_prefill_work,
      BuildDPMPreparedPrefillWorkBinding(prepared_prefill_plan));
  ABSL_ASSIGN_OR_RETURN(descriptor.descriptor_id,
                        ComputeDPMSessionCheckpointId(descriptor));
  ABSL_RETURN_IF_ERROR(ValidateDPMSessionCheckpointArtifact(capture.artifact));

  capture.capture_evidence.plan = std::move(capture_plan);
  capture.capture_evidence.checkpoint_id = descriptor.descriptor_id;
  capture.capture_evidence.checkpoint_envelope_hash = descriptor.envelope_hash;
  capture.capture_evidence.checkpoint_envelope_size = descriptor.envelope_size;
  capture.capture_evidence.checkpoint_authentication_key_id = descriptor.key_id;
  capture.capture_evidence.checkpoint_history_token_bytes_hash =
      transient.producer_first_export.processed_history_token_bytes_hash;
  capture.capture_evidence.producer_before_export =
      transient.producer_first_export;
  capture.capture_evidence.producer_after_export =
      transient.producer_second_export;
  capture.capture_evidence.fresh_import_target = transient.fresh_import_target;
  capture.capture_evidence.transient_to_durable_reauthentication =
      durable.reauthentication_evidence;
  capture.capture_evidence.parent_restore_evidence = parent_restore_evidence;
  ABSL_ASSIGN_OR_RETURN(
      capture.capture_evidence.evidence_id,
      ComputeCapsuleCaptureEvidenceId(capture.capture_evidence));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleCaptureEvidence(capture.capture_evidence));

  ABSL_ASSIGN_OR_RETURN(const CapsuleRestoreRuntimeAuthority authority_after,
      RequireCurrentCapsuleRestoreAuthority());
  if (!SameCapsuleRestoreRuntimeAuthority(
          authority_before.admission, authority_before.authority,
          authority_after.admission, authority_after.authority)) {
    return absl::AbortedError(
        "CapsuleRestore authority changed while binding exact checkpoint "
        "provenance.");
  }
  return capture;
}

absl::StatusOr<DPMTurnResult> DPMEngine::RunTurn(
    const DPMTurnRequest& request) {
  absl::MutexLock turn_lock(turn_mutex_);
  // A committed operation is recoverable from the authoritative log without
  // an inference runtime, projector, clock, or checkpoint repository. Keep
  // that idempotent recovery path ahead of all inference-support preflight.
  if (log_ == nullptr) {
    return absl::InvalidArgumentError(
        "DPM RunTurn requires an authoritative event log.");
  }
  if (request.operation_id.empty() || request.case_id.empty() ||
      request.payload.empty()) {
    return absl::InvalidArgumentError(
        "DPM turn requires non-empty operation id, case id, and payload.");
  }
  if (!IsValidUtf8(request.operation_id) || !IsValidUtf8(request.case_id) ||
      !IsValidUtf8(request.payload)) {
    return absl::InvalidArgumentError(
        "DPM operation id, case id, and input payload must be valid UTF-8.");
  }
  if (request.operation_id.size() > kMaximumDPMEventOperationIdBytes ||
      request.case_id.size() > kMaximumDPMProjectionIdentityBytes ||
      request.payload.size() > kMaximumDPMEventPayloadBytes ||
      ContainsControlByte(request.operation_id) ||
      ContainsControlByte(request.case_id)) {
    return absl::InvalidArgumentError(
        "DPM operation/case identities or input payload exceed product "
        "bounds, or an identity contains a control byte.");
  }
  if (!IsDecisionInputKind(request.kind)) {
    return absl::InvalidArgumentError(
        "DPM RunTurn accepts only user, tool, or internal input events; "
        "corrections use AppendCorrection.");
  }
  if ((request.timestamp_us.has_value() && *request.timestamp_us <= 0) ||
      (request.response_timestamp_us.has_value() &&
       *request.response_timestamp_us <= 0)) {
    return absl::InvalidArgumentError(
        "DPM turn timestamps must be positive when supplied.");
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<DPMEventLogOperationLease> operation,
                        log_->AcquireOperationLease());
  if (operation == nullptr) {
    return absl::InternalError(
        "DPM event log returned a null operation lease.");
  }
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot initial, log_->Snapshot());
  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex initial_index,
                        ValidateAndIndexAuthoritativeLog(initial));
  if (initial.case_id != request.case_id) {
    return absl::FailedPreconditionError(
        "DPM turn case id does not match the immutable log case.");
  }
  ABSL_ASSIGN_OR_RETURN(Hash256 initial_prefix,
                        SnapshotPrefixHashAt(initial, initial.generation));
  if (initial_prefix != initial.prefix_hash) {
    return absl::AbortedError(
        "DPM log changed while acquiring its authoritative snapshot.");
  }

  std::optional<uint64_t> existing_input_index;
  auto current_operation = initial_index.operations.find(request.operation_id);
  if (current_operation != initial_index.operations.end()) {
    const OperationRecord& record = current_operation->second;
    if (record.correction != nullptr) {
      return absl::AlreadyExistsError(
          "DPM operation id is already bound to a correction.");
    }
    if (record.response != nullptr) {
      ABSL_ASSIGN_OR_RETURN(std::optional<DPMTurnResult> recovered,
                            RecoverCommittedTurn(initial, request));
      if (!recovered.has_value()) {
        return absl::DataLossError(
            "Indexed committed DPM operation could not be recovered.");
      }
      return std::move(*recovered);
    }
    if (record.input == nullptr) {
      return absl::DataLossError(
          "DPM operation index contains neither input nor response.");
    }
    ABSL_RETURN_IF_ERROR(ValidateInputEvent(*record.input, request));
    existing_input_index = record.input->index;
  }
  if (initial_index.pending_input != nullptr &&
      (!existing_input_index.has_value() ||
       initial_index.pending_input->index != *existing_input_index)) {
    return absl::FailedPreconditionError(
        "Another DPM operation has an incomplete tail turn that must be "
        "retried first.");
  }

  // From this point the operation either needs a new input append or must
  // retry an existing recoverable input. Both paths require the complete
  // projection, agent, and checkpoint capability preflight before any new
  // mutation or inference work.
  ABSL_RETURN_IF_ERROR(ValidateConfiguration());
  const SessionHandoffIdentity& loaded_identity =
      agent_runtime_->GetSessionHandoffIdentity();
  if (initial_index.agent_identity.has_value() &&
      *initial_index.agent_identity != loaded_identity) {
    return absl::FailedPreconditionError(
        "DPM log agent identity does not match the loaded engine profile.");
  }

  DPMLogSnapshot source_snapshot;
  uint64_t input_event_index = 0;
  if (existing_input_index.has_value()) {
    input_event_index = *existing_input_index;
    source_snapshot = std::move(initial);
  } else {
    DPMEvent input;
    input.kind = request.kind;
    input.timestamp_us = request.timestamp_us.value_or(clock_->NowMicros());
    if (input.timestamp_us <= 0) {
      return absl::InvalidArgumentError(
          "DPM input timestamp must be positive.");
    }
    input.operation_id = request.operation_id;
    input.case_id = request.case_id;
    input.payload = request.payload;
    ABSL_ASSIGN_OR_RETURN(
        DPMAppendResult appended,
        log_->AppendIfGeneration(std::move(input), initial.generation));
    source_snapshot = std::move(appended.snapshot);
    input_event_index = appended.event_index;
    if (input_event_index >= source_snapshot.events.size()) {
      return absl::DataLossError(
          "DPM log append did not return its authoritative input event.");
    }
    ABSL_RETURN_IF_ERROR(
        ValidateInputEvent(source_snapshot.events[input_event_index], request));
  }

  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex source_index,
                        ValidateAndIndexAuthoritativeLog(source_snapshot));
  if (source_snapshot.case_id != request.case_id ||
      source_index.pending_input == nullptr ||
      source_index.pending_input->index != input_event_index ||
      input_event_index + 1 != source_snapshot.events.size()) {
    return absl::DataLossError(
        "DPM input append did not produce the sole recoverable tail turn.");
  }
  if (source_index.agent_identity.has_value() &&
      *source_index.agent_identity != loaded_identity) {
    return absl::FailedPreconditionError(
        "DPM log agent identity changed before generation.");
  }
  ABSL_ASSIGN_OR_RETURN(
      Hash256 source_prefix,
      SnapshotPrefixHashAt(source_snapshot, source_snapshot.generation));
  if (source_prefix != source_snapshot.prefix_hash) {
    return absl::AbortedError(
        "DPM log changed after publishing the recoverable input.");
  }

  DPMProjectionRequest projection_request;
  projection_request.log = source_snapshot;
  projection_request.input_event_index = input_event_index;
  projection_request.case_id = request.case_id;
  ABSL_ASSIGN_OR_RETURN(DPMProjectionOutcome projection,
                        projection_provider_->Project(projection_request));
  ABSL_RETURN_IF_ERROR(
      ValidateProjectionOutcome(projection, projection_request));
  if (projection.manifest.correction_digest != source_index.correction_digest) {
    return absl::FailedPreconditionError(
        "DPM projection correction digest is not derived from the raw log.");
  }
  const DPMReplayMode agent_replay_mode = agent_runtime_->GetReplayMode();
  if (projection.manifest.replay_mode != agent_replay_mode) {
    return absl::FailedPreconditionError(
        "DPM projection and agent stages must use the same named replay "
        "guarantee.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string canonical_agent_input,
      BuildCanonicalAgentInput(request, source_snapshot, input_event_index,
                               projection));
  const uint32_t max_decision_tokens =
      static_cast<uint32_t>(config_.max_decision_tokens);
  const Hash256 agent_request_hash = ComputeLogicalAgentRequestHash(
      source_index.transcript, canonical_agent_input, max_decision_tokens,
      projection.manifest.correction_digest, projection.manifest.manifest_hash,
      loaded_identity);

  ABSL_ASSIGN_OR_RETURN(
      std::vector<DPMAgentGenerationRequest::PrefillChunk> full_chunks,
      BuildFullTranscriptChunks(source_snapshot, canonical_agent_input,
                                projection.manifest.correction_digest));
  DPMAgentExecutionRequest logical_generation_request{
      .logical_agent_request_hash = agent_request_hash,
      .correction_digest = projection.manifest.correction_digest,
      .max_output_tokens = max_decision_tokens,
      .full_canonical_prefill_chunks = full_chunks,
  };
  ABSL_RETURN_IF_ERROR(
      ValidateDPMAgentExecutionRequest(logical_generation_request));

  if (source_snapshot.events.size() >= std::numeric_limits<uint64_t>::max()) {
    return absl::ResourceExhaustedError(
        "DPM log cannot address another response event.");
  }
  const uint64_t response_event_index = source_snapshot.events.size();
  if (response_event_index != input_event_index + 1) {
    return absl::DataLossError(
        "DPM response would not immediately follow its authoritative input.");
  }
  const int64_t response_timestamp =
      request.response_timestamp_us.value_or(clock_->NowMicros());
  if (response_timestamp <= 0 ||
      response_timestamp <
          source_snapshot.events[input_event_index].timestamp_us) {
    return absl::InvalidArgumentError(
        "DPM response timestamp must be positive and not precede its input.");
  }
  const uint64_t turn_number = CountCommittedTurns(source_snapshot) + 1;
  const bool capture_milestone =
      config_.checkpoint_interval_turns != 0 &&
      turn_number % config_.checkpoint_interval_turns == 0;

  ABSL_ASSIGN_OR_RETURN(
      const Hash256 transcript_prefix_hash,
      ComputeTranscriptPrefixHash(source_snapshot, canonical_agent_input,
                                  projection.manifest.correction_digest));
  const CapsuleDPMRestoreTarget restore_target{
      .log_id = source_snapshot.log_id,
      .source_event_count = source_snapshot.events.size(),
      .source_prefix_hash = source_snapshot.prefix_hash,
      .prospective_response_event_index = response_event_index,
      .projection_request_hash = projection.manifest.request_hash,
      .projection_manifest_hash = projection.manifest.manifest_hash,
      .correction_digest = projection.manifest.correction_digest,
      .agent_transcript_prefix_hash = transcript_prefix_hash,
      .logical_agent_request_hash = agent_request_hash,
  };

  std::optional<CapsuleRestoreRuntimeAuthority> capsule_restore_authority;
  if (config_.restore_session_checkpoints || capture_milestone) {
    ABSL_ASSIGN_OR_RETURN(capsule_restore_authority,
                          RequireCurrentCapsuleRestoreAuthority());
  }
  std::optional<RestoreCandidate> restore_candidate;
  if (config_.restore_session_checkpoints) {
    ABSL_ASSIGN_OR_RETURN(restore_candidate,
                          FindRestoreCandidate(source_snapshot, projection,
                                               *capsule_restore_authority));
  }
  bool capture_this_turn = capture_milestone;
  auto reauthenticate_capsule_authority = [&]() -> absl::Status {
    if (!capsule_restore_authority.has_value()) {
      return absl::InternalError(
          "DPM capsule operation lost its admitted authority.");
    }
    ABSL_ASSIGN_OR_RETURN(
        const CapsuleRestoreRuntimeAuthority current_authority,
        RequireCurrentCapsuleRestoreAuthority());
    if (!SameCapsuleRestoreRuntimeAuthority(
            current_authority.admission, current_authority.authority,
            capsule_restore_authority->admission,
            capsule_restore_authority->authority)) {
      return absl::AbortedError(
          "CapsuleRestore authority changed during the DPM turn.");
    }
    return absl::OkStatus();
  };

  std::unique_ptr<Engine::Session> session;
  auto validate_created_session =
      [&loaded_identity](Engine::Session* candidate) -> absl::Status {
    if (candidate == nullptr) {
      return absl::InternalError("DPM agent runtime returned a null session.");
    }
    absl::StatusOr<SessionHandoffIdentity> candidate_identity =
        candidate->GetSessionHandoffIdentity();
    if (!candidate_identity.ok()) return candidate_identity.status();
    if (*candidate_identity != loaded_identity) {
      return absl::FailedPreconditionError(
          "DPM session identity does not match its loaded agent runtime.");
    }
    return absl::OkStatus();
  };

  std::optional<Hash256> restored_from_checkpoint_id;
  std::optional<CapsuleRestoreEvidence> successful_restore_evidence;
  std::vector<DPMAgentGenerationRequest::PrefillChunk>
      successful_physical_prefill_chunks;
  uint64_t successful_prefill_event_range_start =
      ActiveCorrectionEpochStart(source_snapshot);
  std::optional<ExactRegenerationDPMAgentPhysicalExecution>
      exact_physical_execution;
  std::string exact_staged_capsule;
  DPMAgentReplayExecution agent_execution;
  struct ExactAttempt {
    ExactRegenerationDPMAgentPhysicalExecution physical_execution;
    std::string staged_capsule;
    std::vector<DPMAgentGenerationRequest::PrefillChunk> physical_chunks;
    uint64_t prefill_event_range_start = 0;
    std::optional<CapsuleRestoreEvidence> restore_evidence;
  };
  std::function<absl::StatusOr<ExactAttempt>(const RestoreCandidate*, bool)>
      run_exact_attempt;
  struct WinnerAttempt {
    std::unique_ptr<Engine::Session> session;
    DPMAgentGenerationRequest generation_request;
    DPMAgentReplayExecution execution;
    std::optional<Hash256> restored_checkpoint_id;
    std::optional<CapsuleRestoreEvidence> restore_evidence;
    std::vector<DPMAgentGenerationRequest::PrefillChunk> physical_chunks;
    uint64_t prefill_event_range_start = 0;
  };
  std::function<absl::StatusOr<WinnerAttempt>(const RestoreCandidate*,
                                              const DPMAgentReplayExecution*)>
      execute_winner_attempt;
  if (agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay) {
    execute_winner_attempt = [&](const RestoreCandidate* selected_checkpoint,
            const DPMAgentReplayExecution* selected_winner)
        -> absl::StatusOr<WinnerAttempt> {
      WinnerAttempt attempt;
      attempt.generation_request.max_output_tokens =
          config_.max_decision_tokens;
      attempt.generation_request.logical_agent_request_hash =
          agent_request_hash;
      attempt.generation_request.canonical_prefill_chunks = full_chunks;
      attempt.physical_chunks = full_chunks;
      attempt.prefill_event_range_start =
          ActiveCorrectionEpochStart(source_snapshot);
      ABSL_ASSIGN_OR_RETURN(attempt.session, agent_runtime_->CreateSession());
      ABSL_RETURN_IF_ERROR(validate_created_session(attempt.session.get()));

      if (selected_checkpoint != nullptr) {
        if (!capsule_restore_authority.has_value()) {
          return absl::InternalError(
              "WinnerReplay restore lost its current "
              "CapsuleRestore authority.");
        }
        ABSL_ASSIGN_OR_RETURN(
            const CapsuleRestoreRuntimeAuthority current_authority,
            RequireCurrentCapsuleRestoreAuthority());
        if (!SameCapsuleRestoreRuntimeAuthority(
                current_authority.admission, current_authority.authority,
                capsule_restore_authority->admission,
                capsule_restore_authority->authority)) {
          return absl::AbortedError(
              "CapsuleRestore authority changed before WinnerReplay "
              "restore.");
        }
        ABSL_RETURN_IF_ERROR(
            ValidateRestoreCandidate(source_snapshot, projection,
                                     *selected_checkpoint, current_authority));
        ABSL_ASSIGN_OR_RETURN(
            std::vector<DPMAgentGenerationRequest::PrefillChunk> delta_chunks,
            BuildDeltaTranscriptChunks(
                source_snapshot,
                selected_checkpoint->artifact.descriptor.response_event_index,
                canonical_agent_input, projection.manifest.correction_digest));

        SessionHandoffOptions durable_options;
        durable_options.key_id = config_.checkpoint_key_id;
        durable_options.authentication_key =
            config_.checkpoint_authentication_key;
        durable_options.expected_identity = loaded_identity;
        SessionHandoffOptions transient_options;
        transient_options.key_id =
            std::string(kFreshWorkerTransientRestoreKeyId);
        transient_options.authentication_key = DeriveRequestTransientKey(
            config_.checkpoint_authentication_key,
            kWinnerRestoreTransientKeyDomain, agent_request_hash,
            selected_checkpoint->artifact.descriptor.descriptor_id,
            request.operation_id);
        transient_options.expected_identity = loaded_identity;

        std::string transient_envelope;
        StringByteSource durable_source(
            selected_checkpoint->artifact.authenticated_envelope);
        StringByteSink transient_sink(&transient_envelope);
        ABSL_ASSIGN_OR_RETURN(
            const SessionHandoffReauthenticationEvidence reauthentication,
            ReauthenticateSessionHandoffToWithEvidence(
                durable_source, loaded_identity, durable_options,
                transient_options,
                kFreshWorkerDurableRestoreToTransientReauthenticationPurpose,
                &transient_sink));
        StringByteSource transient_source(transient_envelope);
        ABSL_ASSIGN_OR_RETURN(
            const SessionContinuationStateWitness import_witness,
            attempt.session->ImportHandoffFromWithWitness(transient_source,
                                                          transient_options));
        ABSL_RETURN_IF_ERROR(
            ValidateSessionContinuationStateWitness(import_witness));
        ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());

        attempt.generation_request.canonical_prefill_chunks = delta_chunks;
        attempt.generation_request.restore_checkpoint_id =
            selected_checkpoint->artifact.descriptor.descriptor_id;
        attempt.generation_request.restored_state_witness = import_witness;
        attempt.generation_request.capsule_restore_operation =
            DPMAgentCapsuleRestoreOperation{
                .source_capture_evidence =
                    selected_checkpoint->capture_evidence,
                .current_authority = capsule_restore_authority->authority,
                .target_state = restore_target,
                .durable_to_transient_reauthentication = reauthentication,
            };
        attempt.restored_checkpoint_id =
            selected_checkpoint->artifact.descriptor.descriptor_id;
        attempt.physical_chunks = std::move(delta_chunks);
        attempt.prefill_event_range_start =
            selected_checkpoint->artifact.descriptor.response_event_index + 1;
      }

      absl::StatusOr<DPMAgentReplayExecution> generated =
          selected_winner == nullptr
              ? agent_runtime_->Generate(attempt.session.get(),
                                         attempt.generation_request,
                    logical_generation_request)
              : agent_runtime_->RematerializeCanonicalWinner(
                    attempt.session.get(), attempt.generation_request,
                    logical_generation_request, *selected_winner);
      if (!generated.ok()) return generated.status();
      ABSL_RETURN_IF_ERROR(
          ValidateDPMAgentReplayExecution(*generated, max_decision_tokens));
      if (selected_checkpoint != nullptr) {
        ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
      }
      attempt.execution = std::move(*generated);

      if (!attempt.execution.producing_session_matches_output) {
        if (selected_winner != nullptr) {
          return absl::DataLossError(
              "WinnerReplay rematerialization did not return a live "
              "byte-equal producer.");
        }
        attempt.session.reset();
        attempt.generation_request = DPMAgentGenerationRequest{};
        attempt.restored_checkpoint_id.reset();
        attempt.restore_evidence.reset();
        attempt.physical_chunks.clear();
        return attempt;
      }
      if (!attempt.execution.prepared_prefill_plan.has_value()) {
        return absl::DataLossError(
            "WinnerReplay live producer omitted its prepared work.");
      }
      if (selected_checkpoint == nullptr) {
        if (attempt.execution.capsule_restore_evidence.has_value()) {
          return absl::DataLossError(
              "Fresh WinnerReplay generation invented restore evidence.");
        }
        return attempt;
      }
      if (!attempt.execution.capsule_restore_evidence.has_value()) {
        return absl::DataLossError(
            "Restored WinnerReplay generation omitted capsule operation "
            "evidence.");
      }
      ABSL_ASSIGN_OR_RETURN(
          const CapsuleRestoreRuntimeAuthority current_authority,
          RequireCurrentCapsuleRestoreAuthority());
      if (!SameCapsuleRestoreRuntimeAuthority(
              current_authority.admission, current_authority.authority,
              capsule_restore_authority->admission,
              capsule_restore_authority->authority)) {
        return absl::AbortedError(
            "CapsuleRestore authority changed after WinnerReplay "
            "generation.");
      }
      ABSL_RETURN_IF_ERROR(
          ValidateCapsuleRestoreEvidenceSourceCheckpointBinding(
              *attempt.execution.capsule_restore_evidence,
              selected_checkpoint->capture_evidence,
              selected_checkpoint->artifact, selected_checkpoint->receipt,
              current_authority.authority, current_authority.admission));
      attempt.restore_evidence = attempt.execution.capsule_restore_evidence;
      return attempt;
    };

    const RestoreCandidate* selected_candidate =
        restore_candidate.has_value() ? &*restore_candidate : nullptr;
    absl::StatusOr<WinnerAttempt> winner_attempt =
        execute_winner_attempt(selected_candidate, nullptr);
    if (!winner_attempt.ok() &&
        winner_attempt.status().code() == absl::StatusCode::kAborted) {
      return winner_attempt.status();
    }
    if (!winner_attempt.ok() && selected_candidate != nullptr) {
      winner_attempt = execute_winner_attempt(nullptr, nullptr);
    }
    if (!winner_attempt.ok()) return winner_attempt.status();

    if (!winner_attempt->execution.producing_session_matches_output &&
        capture_this_turn) {
      const DPMAgentReplayExecution selected_winner = winner_attempt->execution;
      absl::StatusOr<WinnerAttempt> rematerialized =
          execute_winner_attempt(selected_candidate, &selected_winner);
      if (!rematerialized.ok() &&
          rematerialized.status().code() == absl::StatusCode::kAborted) {
        return rematerialized.status();
      }
      if (!rematerialized.ok() && selected_candidate != nullptr) {
        rematerialized = execute_winner_attempt(nullptr, &selected_winner);
      }
      if (rematerialized.ok()) {
        winner_attempt = std::move(rematerialized);
      } else if (config_.require_checkpoint_at_milestone) {
        return rematerialized.status();
      } else {
        capture_this_turn = false;
      }
    }

    session = std::move(winner_attempt->session);
    agent_execution = std::move(winner_attempt->execution);
    restored_from_checkpoint_id = winner_attempt->restored_checkpoint_id;
    successful_restore_evidence = std::move(winner_attempt->restore_evidence);
    successful_physical_prefill_chunks =
        std::move(winner_attempt->physical_chunks);
    successful_prefill_event_range_start =
        winner_attempt->prefill_event_range_start;
  } else {
    ABSL_ASSIGN_OR_RETURN(
        const Hash256 replay_request_hash,
        ComputeAgentReplayRequestHash(logical_generation_request));

    run_exact_attempt =
        [&, replay_request_hash](
            const RestoreCandidate* selected_checkpoint,
            bool capture_producing_capsule) -> absl::StatusOr<ExactAttempt> {
      ExactAttempt attempt;
      attempt.physical_chunks = full_chunks;
      attempt.prefill_event_range_start =
          ActiveCorrectionEpochStart(source_snapshot);
      FreshWorkerExecutionPlan execution_plan;
      execution_plan.logical_replay_request_hash = replay_request_hash;
      if (selected_checkpoint != nullptr) {
        if (!capsule_restore_authority.has_value()) {
          return absl::InternalError(
              "Exact restore lost its current CapsuleRestore authority.");
        }
        ABSL_ASSIGN_OR_RETURN(
            const CapsuleRestoreRuntimeAuthority current_authority,
            RequireCurrentCapsuleRestoreAuthority());
        if (!SameCapsuleRestoreRuntimeAuthority(
                current_authority.admission, current_authority.authority,
                capsule_restore_authority->admission,
                capsule_restore_authority->authority)) {
          return absl::AbortedError(
              "CapsuleRestore authority changed before exact restore.");
        }
        ABSL_RETURN_IF_ERROR(
            ValidateRestoreCandidate(source_snapshot, projection,
                                     *selected_checkpoint, current_authority));
        const DPMSessionCheckpointDescriptor& descriptor =
            selected_checkpoint->artifact.descriptor;
        ABSL_ASSIGN_OR_RETURN(
            std::vector<DPMAgentGenerationRequest::PrefillChunk> delta_chunks,
            BuildDeltaTranscriptChunks(
                source_snapshot, descriptor.response_event_index,
                canonical_agent_input, projection.manifest.correction_digest));
        DPMAgentDeltaExecutionRequest delta_request{
            .logical_agent_request_hash = agent_request_hash,
            .correction_digest = projection.manifest.correction_digest,
            .restore_checkpoint_id = descriptor.descriptor_id,
            .restored_response_event_index = descriptor.response_event_index,
            .restored_agent_transcript_hash = descriptor.agent_transcript_hash,
            .max_output_tokens = max_decision_tokens,
            .canonical_delta_prefill_chunks = std::move(delta_chunks),
        };
        attempt.physical_chunks = delta_request.canonical_delta_prefill_chunks;
        attempt.prefill_event_range_start = descriptor.response_event_index + 1;
        ABSL_ASSIGN_OR_RETURN(
            execution_plan.canonical_execution_payload,
            EncodeDPMAgentDeltaExecutionRequest(delta_request));
        execution_plan.prefill_mode =
            FreshWorkerPrefillMode::kOwnPositionCapsuleDelta;
        execution_plan.restore_checkpoint_id = descriptor.descriptor_id;
        execution_plan.session_identity = descriptor.session_identity;
        execution_plan.restore_durable_envelope_hash = descriptor.envelope_hash;
        execution_plan.restore_durable_envelope_size = descriptor.envelope_size;
      }
      if (capture_producing_capsule) {
        if (!capsule_restore_authority.has_value()) {
          return absl::InternalError(
              "Exact capture lost its current CapsuleRestore authority.");
        }
      }
      execution_plan.capture_producing_capsule = capture_producing_capsule;
      ABSL_ASSIGN_OR_RETURN(
          execution_plan.plan_hash,
          ComputeFreshWorkerExecutionPlanHash(execution_plan));

      SessionHandoffOptions restore_options;
      SessionHandoffOptions capture_options;
      std::optional<StringByteSource> restore_source;
      std::optional<StringByteSink> capture_sink;
      ExactRegenerationExecutionInput physical_input{
          .execution_plan = execution_plan,
      };
      if (selected_checkpoint != nullptr) {
        restore_options.key_id = config_.checkpoint_key_id;
        restore_options.authentication_key =
            config_.checkpoint_authentication_key;
        restore_options.expected_identity = loaded_identity;
        restore_source.emplace(
            selected_checkpoint->artifact.authenticated_envelope);
        physical_input.durable_restore_source = &*restore_source;
        physical_input.durable_restore_options = &restore_options;
      }
      if (capture_producing_capsule) {
        capture_options.key_id = config_.checkpoint_key_id;
        capture_options.authentication_key =
            config_.checkpoint_authentication_key;
        capture_options.expected_identity = loaded_identity;
        capture_sink.emplace(&attempt.staged_capsule);
        physical_input.staging_capture_destination = &*capture_sink;
        physical_input.staging_capture_options = &capture_options;
      }

      const bool transfers_capsule =
          selected_checkpoint != nullptr || capture_producing_capsule;
      std::optional<CapsuleRestoreRuntimeAuthority> authority_before_execution;
      if (transfers_capsule) {
        if (!capsule_restore_authority.has_value()) {
          return absl::InternalError(
              "Exact capsule execution lost its CapsuleRestore authority.");
        }
        absl::StatusOr<CapsuleRestoreRuntimeAuthority> current_authority =
            RequireCurrentCapsuleRestoreAuthority();
        if (!current_authority.ok()) {
          return absl::AbortedError(current_authority.status().message());
        }
        authority_before_execution = std::move(*current_authority);
        if (!SameCapsuleRestoreRuntimeAuthority(
                authority_before_execution->admission,
                authority_before_execution->authority,
                capsule_restore_authority->admission,
                capsule_restore_authority->authority)) {
          return absl::AbortedError(
              "CapsuleRestore authority changed before exact worker use.");
        }
      }

      ABSL_ASSIGN_OR_RETURN(ExactRegenerationDPMAgentPhysicalExecution physical,
                            agent_runtime_->GeneratePhysicalExact(
                                logical_generation_request, physical_input));
      if (transfers_capsule) {
        absl::StatusOr<CapsuleRestoreRuntimeAuthority> current_authority =
            RequireCurrentCapsuleRestoreAuthority();
        if (!current_authority.ok()) {
          return absl::AbortedError(current_authority.status().message());
        }
        const CapsuleRestoreRuntimeAuthority authority_after_execution =
            std::move(*current_authority);
        if (!SameCapsuleRestoreRuntimeAuthority(
                authority_after_execution.admission,
                authority_after_execution.authority,
                authority_before_execution->admission,
                authority_before_execution->authority)) {
          return absl::AbortedError(
              "CapsuleRestore authority changed during exact worker use.");
        }
      }
      ABSL_RETURN_IF_ERROR(ValidateExactRegenerationDPMAgentPhysicalExecution(
              physical, max_decision_tokens));
      if (physical.physical_execution_plan_hash != execution_plan.plan_hash ||
          physical.restored_checkpoint_id !=
              execution_plan.restore_checkpoint_id) {
        return absl::DataLossError(
            "Exact DPM worker execution changed its parent-selected physical "
            "checkpoint plan.");
      }
      if (selected_checkpoint != nullptr) {
        const DPMSessionCheckpointDescriptor& source_descriptor =
            selected_checkpoint->artifact.descriptor;
        if (physical.replay_execution.exact_profile_id !=
                source_descriptor.exact_profile_id ||
            physical.replay_execution.exact_profile_admission_record_id !=
                std::optional<Hash256>(
                    source_descriptor.exact_profile_admission_record_id)) {
          return absl::DataLossError(
              "Exact DPM restored worker changed the source checkpoint's "
              "current profile or admission lineage.");
        }
      }
      if (selected_checkpoint != nullptr) {
        if (!physical.run_zero_restore_reauthentication_evidence.has_value() ||
            !physical.run_zero_restored_state_witness.has_value() ||
            !physical.replay_execution.prepared_prefill_plan.has_value()) {
          return absl::DataLossError(
              "Exact restored execution omitted its run-zero rewrap, "
              "witness, or prepared plan.");
        }
        ABSL_ASSIGN_OR_RETURN(
            CapsuleRestoreEvidence restore_evidence,
            BuildExactCapsuleRestoreEvidence(
                selected_checkpoint->artifact,
                selected_checkpoint->capture_evidence,
                capsule_restore_authority->authority, restore_target,
                attempt.physical_chunks,
                *physical.replay_execution.prepared_prefill_plan,
                *physical.run_zero_restore_reauthentication_evidence,
                *physical.run_zero_restored_state_witness,
                max_decision_tokens));
        ABSL_ASSIGN_OR_RETURN(
            const CapsuleRestoreRuntimeAuthority current_authority,
            RequireCurrentCapsuleRestoreAuthority());
        if (!SameCapsuleRestoreRuntimeAuthority(
                current_authority.admission, current_authority.authority,
                capsule_restore_authority->admission,
                capsule_restore_authority->authority)) {
          return absl::AbortedError(
              "CapsuleRestore authority changed while binding exact "
              "restore evidence.");
        }
        ABSL_RETURN_IF_ERROR(
            ValidateCapsuleRestoreEvidenceSourceCheckpointBinding(
                restore_evidence, selected_checkpoint->capture_evidence,
                selected_checkpoint->artifact, selected_checkpoint->receipt,
                current_authority.authority, current_authority.admission));
        attempt.restore_evidence = std::move(restore_evidence);
      }
      attempt.physical_execution = std::move(physical);
      return attempt;
    };

    absl::StatusOr<ExactAttempt> generated_exact = run_exact_attempt(
            restore_candidate.has_value() ? &*restore_candidate : nullptr,
            capture_this_turn);
    if (!generated_exact.ok() &&
        generated_exact.status().code() == absl::StatusCode::kAborted) {
      return generated_exact.status();
    }
    if (!generated_exact.ok() && restore_candidate.has_value()) {
      generated_exact = run_exact_attempt(nullptr, capture_this_turn);
    }
    if (!generated_exact.ok() &&
        generated_exact.status().code() == absl::StatusCode::kAborted) {
      return generated_exact.status();
    }
    if (!generated_exact.ok() && capture_this_turn &&
        !config_.require_checkpoint_at_milestone) {
      generated_exact = run_exact_attempt(nullptr, false);
      capture_this_turn = false;
    }
    if (!generated_exact.ok()) return generated_exact.status();
    exact_staged_capsule = std::move(generated_exact->staged_capsule);
    successful_physical_prefill_chunks =
        std::move(generated_exact->physical_chunks);
    successful_prefill_event_range_start =
        generated_exact->prefill_event_range_start;
    successful_restore_evidence = std::move(generated_exact->restore_evidence);
    exact_physical_execution = std::move(generated_exact->physical_execution);
    agent_execution = exact_physical_execution->replay_execution;
    restored_from_checkpoint_id =
        exact_physical_execution->restored_checkpoint_id;
  }

  struct MaterializedAgentDecision {
    DPMAgentGenerationOutcome outcome;
    std::optional<Hash256> exact_output_evidence_hash;
    uint32_t exact_logit_frame_count = 0;
    Hash256 transcript_hash;
  };
  auto materialize_agent_decision =
      [&](const DPMAgentReplayExecution& execution)
      -> absl::StatusOr<MaterializedAgentDecision> {
    ABSL_RETURN_IF_ERROR(
        ValidateDPMAgentReplayExecution(execution, max_decision_tokens));
    if (execution.mode != agent_replay_mode) {
      return absl::DataLossError(
          "DPM agent execution returned evidence for another replay mode.");
    }
    MaterializedAgentDecision material;
    material.outcome.decision_output = execution.decision_output;
    material.outcome.decision_token_ids = execution.decision_token_ids;
    material.outcome.prepared_prefill_plan = execution.prepared_prefill_plan;
    material.outcome.capsule_restore_evidence =
        execution.capsule_restore_evidence;
    ABSL_RETURN_IF_ERROR(
        ValidateAgentOutcome(material.outcome, max_decision_tokens));
    if (execution.mode == DPMReplayMode::kExactRegeneration) {
      if (execution.exact_logit_frames.size() >
          std::numeric_limits<uint32_t>::max()) {
        return absl::ResourceExhaustedError(
            "Exact agent logits evidence exceeds the receipt count domain.");
      }
      DPMAgentDecisionEnvelope decision_envelope{
          .decision_output = execution.decision_output,
          .canonical_token_bytes = execution.exact_token_bytes,
      };
      ABSL_ASSIGN_OR_RETURN(const std::string canonical_decision_envelope,
          EncodeDPMAgentDecisionEnvelope(decision_envelope));
      material.exact_output_evidence_hash =
          ComputeFreshWorkerOutputEvidenceHash(canonical_decision_envelope,
                                               execution.exact_token_bytes,
              execution.exact_logit_frames);
      if (!execution.exact_output_evidence_hash.has_value() ||
          *execution.exact_output_evidence_hash !=
              *material.exact_output_evidence_hash) {
        return absl::DataLossError(
            "Exact agent runtime output-evidence commitment disagrees with "
            "the canonical decision, token, and ordered logits evidence.");
      }
      material.exact_logit_frame_count =
          static_cast<uint32_t>(execution.exact_logit_frames.size());
    }
    ABSL_ASSIGN_OR_RETURN(
        material.transcript_hash,
        ComputeTranscriptHash(source_snapshot, canonical_agent_input,
                              material.outcome.decision_output,
                              material.outcome.decision_token_ids,
                              projection.manifest.correction_digest));
    return material;
  };
  ABSL_ASSIGN_OR_RETURN(MaterializedAgentDecision materialized_decision,
                        materialize_agent_decision(agent_execution));

  auto adopt_exact_attempt = [&](ExactAttempt attempt) -> absl::Status {
    exact_staged_capsule = std::move(attempt.staged_capsule);
    successful_physical_prefill_chunks = std::move(attempt.physical_chunks);
    successful_prefill_event_range_start = attempt.prefill_event_range_start;
    successful_restore_evidence = std::move(attempt.restore_evidence);
    exact_physical_execution = std::move(attempt.physical_execution);
    agent_execution = exact_physical_execution->replay_execution;
    restored_from_checkpoint_id =
        exact_physical_execution->restored_checkpoint_id;
    ABSL_ASSIGN_OR_RETURN(materialized_decision,
        materialize_agent_decision(agent_execution));
    return absl::OkStatus();
  };
  auto adopt_winner_attempt = [&](WinnerAttempt attempt) -> absl::Status {
    session = std::move(attempt.session);
    agent_execution = std::move(attempt.execution);
    restored_from_checkpoint_id = attempt.restored_checkpoint_id;
    successful_restore_evidence = std::move(attempt.restore_evidence);
    successful_physical_prefill_chunks = std::move(attempt.physical_chunks);
    successful_prefill_event_range_start = attempt.prefill_event_range_start;
    ABSL_ASSIGN_OR_RETURN(materialized_decision,
        materialize_agent_decision(agent_execution));
    return absl::OkStatus();
  };
  auto rerun_full_log = [&](bool request_capture) -> absl::Status {
    successful_restore_evidence.reset();
    restored_from_checkpoint_id.reset();
    if (agent_replay_mode == DPMReplayMode::kExactRegeneration) {
      ABSL_ASSIGN_OR_RETURN(ExactAttempt attempt,
                            run_exact_attempt(nullptr, request_capture));
      return adopt_exact_attempt(std::move(attempt));
    }
    ABSL_ASSIGN_OR_RETURN(WinnerAttempt attempt,
                          execute_winner_attempt(nullptr, nullptr));
    if (request_capture &&
        !attempt.execution.producing_session_matches_output) {
      const DPMAgentReplayExecution selected_winner = attempt.execution;
      ABSL_ASSIGN_OR_RETURN(attempt,
          execute_winner_attempt(nullptr, &selected_winner));
    }
    return adopt_winner_attempt(std::move(attempt));
  };

  auto build_checkpoint_capture =
      [&]() -> absl::StatusOr<std::optional<CheckpointCapture>> {
    if (!capture_this_turn) {
      return std::optional<CheckpointCapture>();
    }
    if (!capsule_restore_authority.has_value()) {
      return absl::InternalError(
          "DPM checkpoint capture lost its current CapsuleRestore authority.");
    }
    if (agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay) {
      if (!agent_execution.producing_session_matches_output ||
          session == nullptr ||
          !agent_execution.prepared_prefill_plan.has_value()) {
        return absl::FailedPreconditionError(
            "WinnerReplay capture requires the live session and prepared "
            "plan that produced the selected bytes.");
      }
      ABSL_ASSIGN_OR_RETURN(
          CheckpointCapture capture,
          CaptureProducingSession(
              session.get(), source_snapshot, response_event_index, projection,
              agent_request_hash, transcript_prefix_hash,
              materialized_decision.transcript_hash, *capsule_restore_authority,
              successful_physical_prefill_chunks,
              *agent_execution.prepared_prefill_plan,
              agent_execution.execution_evidence_hash,
              static_cast<uint32_t>(agent_execution.decision_token_ids.size()),
              successful_prefill_event_range_start, successful_restore_evidence,
              request.operation_id, response_timestamp));
      return std::optional<CheckpointCapture>(std::move(capture));
    }
    if (!exact_physical_execution.has_value() ||
        exact_physical_execution->capture_run_policy !=
            ExactRegenerationCaptureRunPolicy::kRunZeroOnly) {
      return absl::FailedPreconditionError(
          "Exact DPM capture lacks an authenticated run-zero capsule.");
    }
    ABSL_ASSIGN_OR_RETURN(
        CheckpointCapture capture,
        BuildExactWorkerCheckpointCapture(
            exact_staged_capsule, *exact_physical_execution, source_snapshot,
            response_event_index, projection, agent_request_hash,
            transcript_prefix_hash, materialized_decision.transcript_hash,
            *capsule_restore_authority, successful_physical_prefill_chunks,
            successful_prefill_event_range_start, successful_restore_evidence,
            response_timestamp));
    return std::optional<CheckpointCapture>(std::move(capture));
  };

  std::optional<CheckpointCapture> checkpoint_capture;
  DPMTurnReceipt receipt;
  enum class DerivativePublicationStage {
    kNone,
    kRestoreEvidence,
    kCheckpoint,
    kCaptureEvidence,
  };
  const FreshWorkerAuthentication operation_evidence_authentication{
      .key_id = config_.operation_evidence_key_id,
      .authentication_key = config_.operation_evidence_authentication_key,
  };

  bool finalized_derivatives = false;
  for (int finalization_round = 0; finalization_round < 4;
       ++finalization_round) {
    checkpoint_capture.reset();
    absl::StatusOr<std::optional<CheckpointCapture>> built_capture =
        build_checkpoint_capture();
    if (!built_capture.ok()) {
      if (built_capture.status().code() == absl::StatusCode::kAborted) {
        return built_capture.status();
      }
      if (config_.require_checkpoint_at_milestone) {
        return built_capture.status();
      }
      if (agent_replay_mode == DPMReplayMode::kExactRegeneration) {
        ABSL_RETURN_IF_ERROR(rerun_full_log(false));
      }
      capture_this_turn = false;
      continue;
    }
    checkpoint_capture = std::move(*built_capture);

    receipt = DPMTurnReceipt{};
    receipt.operation_id = request.operation_id;
    receipt.input_event_index = input_event_index;
    receipt.response_event_index = response_event_index;
    receipt.projection_manifest = projection.manifest;
    receipt.agent_session_identity = loaded_identity;
    receipt.max_decision_tokens = max_decision_tokens;
    receipt.agent_request_hash = agent_request_hash;
    receipt.agent_replay_mode = agent_execution.mode;
    receipt.agent_replay_request_hash = agent_execution.replay_request_hash;
    receipt.agent_execution_evidence_hash =
        agent_execution.execution_evidence_hash;
    receipt.agent_exact_profile_id = agent_execution.exact_profile_id;
    receipt.agent_exact_profile_admission_record_id =
        agent_execution.exact_profile_admission_record_id;
    receipt.agent_exact_output_evidence_hash =
        materialized_decision.exact_output_evidence_hash;
    receipt.agent_exact_logit_frame_count =
        materialized_decision.exact_logit_frame_count;
    receipt.agent_reused_canonical_winner =
        agent_execution.reused_canonical_winner;
    receipt.agent_rematerialized_canonical_winner =
        agent_execution.rematerialized_canonical_winner;
    receipt.agent_producing_session_matched_output =
        agent_execution.producing_session_matches_output;
    receipt.projected_memory = projection.projected_memory;
    receipt.canonical_agent_input = canonical_agent_input;
    receipt.decision_output = materialized_decision.outcome.decision_output;
    receipt.decision_token_ids =
        materialized_decision.outcome.decision_token_ids;
    receipt.agent_transcript_hash = materialized_decision.transcript_hash;
    if (agent_execution.prepared_prefill_plan.has_value()) {
      ABSL_ASSIGN_OR_RETURN(receipt.agent_prepared_prefill_work,
          BuildDPMPreparedPrefillWorkBinding(
              *agent_execution.prepared_prefill_plan));
    }
    if (successful_restore_evidence.has_value()) {
      if (!restore_candidate.has_value() ||
          restored_from_checkpoint_id !=
              std::optional<Hash256>(
                  restore_candidate->artifact.descriptor.descriptor_id) ||
          successful_restore_evidence->plan.checkpoint_id !=
              *restored_from_checkpoint_id) {
        return absl::DataLossError(
            "DPM accepted restore evidence without its selected source "
            "checkpoint.");
      }
      receipt.restored_from_session_checkpoint_id = restored_from_checkpoint_id;
      receipt.restored_checkpoint_capture_evidence_id =
          restore_candidate->capture_evidence.evidence_id;
      receipt.agent_capsule_restore_evidence_id =
          successful_restore_evidence->evidence_id;
    } else if (restored_from_checkpoint_id.has_value()) {
      return absl::DataLossError(
          "DPM restored execution lacks operation evidence.");
    }
    if (checkpoint_capture.has_value()) {
      receipt.session_checkpoint_id =
          checkpoint_capture->artifact.descriptor.descriptor_id;
      receipt.checkpoint_capture_origin =
          checkpoint_capture->artifact.descriptor.capture_origin;
      receipt.published_checkpoint_capture =
          DPMCheckpointCaptureEvidenceBinding{
              .capture_plan_hash =
                  checkpoint_capture->capture_evidence.plan.plan_hash,
              .capture_evidence_id =
                  checkpoint_capture->capture_evidence.evidence_id,
          };
    }
    const bool receipt_uses_capsule = successful_restore_evidence.has_value() ||
        checkpoint_capture.has_value();
    if (receipt_uses_capsule) {
      if (!capsule_restore_authority.has_value()) {
        return absl::InternalError(
            "DPM capsule use lost its current CapsuleRestore authority.");
      }
      ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
      receipt.agent_capsule_restore_capability_id =
          capsule_restore_authority->authority.capability.capability_id;
      receipt.agent_capsule_restore_admission_record_id =
          capsule_restore_authority->authority.admission_record_id;
      receipt.agent_capsule_restore_coverage_id =
          capsule_restore_authority->authority.coverage_id;
    }
    if (exact_physical_execution.has_value()) {
      receipt.agent_worker_prefill_mode =
          ToCheckpointWorkerPrefillMode(exact_physical_execution->prefill_mode);
      receipt.agent_physical_execution_plan_hash =
          exact_physical_execution->physical_execution_plan_hash;
      if (checkpoint_capture.has_value()) {
        if (!checkpoint_capture->artifact.descriptor.worker_provenance
                 .has_value()) {
          return absl::DataLossError(
              "Exact DPM checkpoint lost run-zero provenance.");
        }
        receipt.agent_exact_worker_checkpoint_provenance =
            checkpoint_capture->artifact.descriptor.worker_provenance;
      }
    }
    ABSL_RETURN_IF_ERROR(ValidateDPMTurnReceiptReplayEvidence(receipt));

    if (receipt_uses_capsule) {
      ABSL_ASSIGN_OR_RETURN(
          const CapsuleRestoreRuntimeAuthority current_authority,
          RequireCurrentCapsuleRestoreAuthority());
      if (!SameCapsuleRestoreRuntimeAuthority(
              current_authority.admission, current_authority.authority,
              capsule_restore_authority->admission,
              capsule_restore_authority->authority)) {
        return absl::AbortedError(
            "CapsuleRestore authority changed before derivative binding.");
      }
      if (successful_restore_evidence.has_value()) {
        ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidenceTurnBinding(
            *successful_restore_evidence, restore_candidate->capture_evidence,
            restore_candidate->artifact, restore_candidate->receipt, receipt,
            current_authority.authority, current_authority.admission));
      }
      if (checkpoint_capture.has_value()) {
        ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceCheckpointBinding(
                checkpoint_capture->artifact, receipt,
            checkpoint_capture->capture_evidence, current_authority.authority,
                current_authority.admission));
        if (successful_restore_evidence.has_value()) {
          ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceForParentCapture(
                  checkpoint_capture->capture_evidence,
                  restore_candidate->capture_evidence));
        }
      }
    }

    DerivativePublicationStage failed_stage = DerivativePublicationStage::kNone;
    absl::Status publication_status = absl::OkStatus();
    if (successful_restore_evidence.has_value()) {
      ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
      publication_status = evidence_repository_->PutRestoreIfAbsent(
          *successful_restore_evidence, operation_evidence_authentication);
      ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
      if (!publication_status.ok()) {
        failed_stage = DerivativePublicationStage::kRestoreEvidence;
      }
    }
    if (publication_status.ok() && checkpoint_capture.has_value()) {
      ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
      publication_status =
          checkpoint_repository_->Put(checkpoint_capture->artifact);
      ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
      if (!publication_status.ok()) {
        failed_stage = DerivativePublicationStage::kCheckpoint;
      }
    }
    if (publication_status.ok() && checkpoint_capture.has_value()) {
      ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
      publication_status = evidence_repository_->PutCaptureIfAbsent(
          checkpoint_capture->capture_evidence,
          operation_evidence_authentication);
      ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
      if (!publication_status.ok()) {
        failed_stage = DerivativePublicationStage::kCaptureEvidence;
      }
    }
    if (publication_status.ok()) {
      finalized_derivatives = true;
      break;
    }
    if (publication_status.code() == absl::StatusCode::kAborted) {
      return publication_status;
    }

    const bool capture_publication_failed =
        failed_stage == DerivativePublicationStage::kCheckpoint ||
        failed_stage == DerivativePublicationStage::kCaptureEvidence;
    if (capture_publication_failed &&
        agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay &&
        !config_.require_checkpoint_at_milestone) {
      checkpoint_capture.reset();
      capture_this_turn = false;
      continue;
    }
    if (finalization_round == 3) return publication_status;
    const bool retry_capture =
        capture_milestone && config_.require_checkpoint_at_milestone;
    ABSL_RETURN_IF_ERROR(rerun_full_log(retry_capture));
    capture_this_turn = retry_capture;
  }
  if (!finalized_derivatives) {
    return absl::InternalError(
        "DPM derivative finalization exhausted its bounded fallback path.");
  }

  const bool receipt_uses_capsule =
      receipt.restored_from_session_checkpoint_id.has_value() ||
      receipt.session_checkpoint_id.has_value();
  if (receipt_uses_capsule) {
    ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
  }

  DPMAgentGenerationOutcome& agent_outcome = materialized_decision.outcome;

  DPMEvent response;
  response.kind = DPMEvent::Kind::kModelTurn;
  response.timestamp_us = response_timestamp;
  response.operation_id = request.operation_id;
  response.case_id = request.case_id;
  response.payload = agent_outcome.decision_output;
  response.turn_receipt = std::move(receipt);
  ABSL_ASSIGN_OR_RETURN(DPMAppendResult committed,
                        log_->AppendIfGeneration(std::move(response),
                                                 source_snapshot.generation));
  if (receipt_uses_capsule) {
    ABSL_RETURN_IF_ERROR(reauthenticate_capsule_authority());
  }
  if (committed.event_index != response_event_index ||
      committed.event_index >= committed.snapshot.events.size()) {
    return absl::DataLossError(
        "DPM log violated contiguous response-index publication.");
  }
  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex committed_index,
                        ValidateAndIndexAuthoritativeLog(committed.snapshot));
  auto committed_operation =
      committed_index.operations.find(request.operation_id);
  if (committed.snapshot.case_id != request.case_id ||
      committed_index.pending_input != nullptr ||
      committed_operation == committed_index.operations.end() ||
      committed_operation->second.response == nullptr ||
      committed_operation->second.response->index != response_event_index) {
    return absl::DataLossError(
        "DPM response commit did not close exactly one authoritative turn.");
  }
  ABSL_ASSIGN_OR_RETURN(
      Hash256 committed_prefix,
      SnapshotPrefixHashAt(committed.snapshot, committed.snapshot.generation));
  if (committed_prefix != committed.snapshot.prefix_hash) {
    return absl::AbortedError(
        "DPM log changed after committing the authoritative response.");
  }
  const DPMTurnReceipt& committed_receipt =
      *committed_operation->second.response->turn_receipt;

  DPMTurnResult result;
  result.projected_memory = committed_receipt.projected_memory;
  result.decision_output = committed_receipt.decision_output;
  result.decision_token_ids = committed_receipt.decision_token_ids;
  result.input_event_index = committed_receipt.input_event_index;
  result.response_event_index = committed.event_index;
  result.projection_manifest_hash =
      committed_receipt.projection_manifest.manifest_hash;
  result.projection_replay_mode =
      committed_receipt.projection_manifest.replay_mode;
  result.projection_execution_evidence_hash =
      committed_receipt.projection_manifest.execution_evidence_hash;
  result.projection_exact_profile_id =
      committed_receipt.projection_manifest.exact_profile_id;
  result.projection_exact_profile_admission_record_id =
      committed_receipt.projection_manifest.exact_profile_admission_record_id;
  result.agent_replay_mode = committed_receipt.agent_replay_mode;
  result.agent_execution_evidence_hash =
      committed_receipt.agent_execution_evidence_hash;
  result.agent_exact_profile_id = committed_receipt.agent_exact_profile_id;
  result.agent_exact_profile_admission_record_id =
      committed_receipt.agent_exact_profile_admission_record_id;
  result.agent_exact_output_evidence_hash =
      committed_receipt.agent_exact_output_evidence_hash;
  result.agent_exact_logit_frame_count =
      committed_receipt.agent_exact_logit_frame_count;
  result.agent_reused_canonical_winner =
      committed_receipt.agent_reused_canonical_winner;
  result.agent_rematerialized_canonical_winner =
      committed_receipt.agent_rematerialized_canonical_winner;
  result.agent_producing_session_matched_output =
      committed_receipt.agent_producing_session_matched_output;
  result.agent_worker_capsule_matched_output =
      committed_receipt.checkpoint_capture_origin ==
          DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker &&
      committed_receipt.agent_exact_worker_checkpoint_provenance.has_value();
  result.session_checkpoint_id = committed_receipt.session_checkpoint_id;
  result.restored_from_session_checkpoint_id =
      committed_receipt.restored_from_session_checkpoint_id;
  result.checkpoint_capture_origin =
      committed_receipt.checkpoint_capture_origin;
  result.agent_worker_prefill_mode =
      committed_receipt.agent_worker_prefill_mode;
  result.agent_physical_execution_plan_hash =
      committed_receipt.agent_physical_execution_plan_hash;
  result.agent_capsule_restore_admission_record_id =
      committed_receipt.agent_capsule_restore_admission_record_id;
  result.agent_capsule_restore_capability_id =
      committed_receipt.agent_capsule_restore_capability_id;
  result.agent_capsule_restore_coverage_id =
      committed_receipt.agent_capsule_restore_coverage_id;
  result.agent_exact_worker_checkpoint_provenance =
      committed_receipt.agent_exact_worker_checkpoint_provenance;
  result.agent_prepared_prefill_work =
      committed_receipt.agent_prepared_prefill_work;
  result.published_checkpoint_capture =
      committed_receipt.published_checkpoint_capture;
  result.restored_checkpoint_capture_evidence_id =
      committed_receipt.restored_checkpoint_capture_evidence_id;
  result.agent_capsule_restore_evidence_id =
      committed_receipt.agent_capsule_restore_evidence_id;
  result.restored_session_checkpoint =
      committed_receipt.restored_from_session_checkpoint_id.has_value();
  return result;
}

absl::StatusOr<DPMCorrectionResult> DPMEngine::AppendCorrection(
    const DPMCorrectionRequest& request) {
  absl::MutexLock turn_lock(turn_mutex_);
  if (log_ == nullptr || clock_ == nullptr) {
    return absl::InvalidArgumentError(
        "DPM correction ingestion requires log and clock dependencies.");
  }
  if (request.operation_id.empty() || request.case_id.empty() ||
      request.canonical_payload.empty()) {
    return absl::InvalidArgumentError(
        "DPM correction requires non-empty operation id, case id, and "
        "canonical payload.");
  }
  if (!IsValidUtf8(request.operation_id) || !IsValidUtf8(request.case_id) ||
      !IsValidUtf8(request.canonical_payload)) {
    return absl::InvalidArgumentError(
        "DPM correction operation id, case id, and payload must be valid "
        "UTF-8.");
  }
  if (request.operation_id.size() > kMaximumDPMEventOperationIdBytes ||
      request.case_id.size() > kMaximumDPMProjectionIdentityBytes ||
      request.canonical_payload.size() > kMaximumDPMEventPayloadBytes ||
      ContainsControlByte(request.operation_id) ||
      ContainsControlByte(request.case_id)) {
    return absl::InvalidArgumentError(
        "DPM correction operation/case identities or payload exceed product "
        "bounds, or an identity contains a control byte.");
  }
  if (request.timestamp_us.has_value() && *request.timestamp_us <= 0) {
    return absl::InvalidArgumentError(
        "DPM correction timestamp must be positive when supplied.");
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<DPMEventLogOperationLease> operation,
                        log_->AcquireOperationLease());
  if (operation == nullptr) {
    return absl::InternalError(
        "DPM event log returned a null operation lease.");
  }
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot snapshot, log_->Snapshot());
  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex index,
                        ValidateAndIndexAuthoritativeLog(snapshot));
  if (snapshot.case_id != request.case_id) {
    return absl::FailedPreconditionError(
        "DPM correction case id does not match the immutable log case.");
  }
  ABSL_ASSIGN_OR_RETURN(Hash256 prefix,
                        SnapshotPrefixHashAt(snapshot, snapshot.generation));
  if (prefix != snapshot.prefix_hash) {
    return absl::AbortedError(
        "DPM log changed while acquiring its correction snapshot.");
  }

  auto existing = index.operations.find(request.operation_id);
  if (existing != index.operations.end()) {
    const OperationRecord& record = existing->second;
    if (record.correction == nullptr || record.input != nullptr ||
        record.response != nullptr ||
        !record.correction_digest_after.has_value() ||
        record.correction->case_id != request.case_id ||
        record.correction->payload != request.canonical_payload) {
      return absl::AlreadyExistsError(
          "DPM operation id is already bound to different authoritative "
          "bytes.");
    }
    if (request.timestamp_us.has_value() &&
        record.correction->timestamp_us != *request.timestamp_us) {
      return absl::AlreadyExistsError(
          "DPM correction operation id is already bound to a different "
          "timestamp.");
    }
    DPMCorrectionResult result;
    result.correction_event_index = record.correction->index;
    result.correction_digest = *record.correction_digest_after;
    result.recovered_existing_correction = true;
    return result;
  }
  if (index.pending_input != nullptr) {
    return absl::FailedPreconditionError(
        "DPM correction cannot cross an incomplete tail turn; retry that "
        "operation first.");
  }

  DPMEvent correction;
  correction.kind = DPMEvent::Kind::kCorrection;
  correction.timestamp_us = request.timestamp_us.value_or(clock_->NowMicros());
  if (correction.timestamp_us <= 0) {
    return absl::InvalidArgumentError(
        "DPM correction timestamp must be positive.");
  }
  correction.operation_id = request.operation_id;
  correction.case_id = request.case_id;
  correction.payload = request.canonical_payload;
  ABSL_ASSIGN_OR_RETURN(
      DPMAppendResult appended,
      log_->AppendIfGeneration(std::move(correction), snapshot.generation));
  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex appended_index,
                        ValidateAndIndexAuthoritativeLog(appended.snapshot));
  auto committed = appended_index.operations.find(request.operation_id);
  if (appended.event_index >= appended.snapshot.events.size() ||
      appended.snapshot.case_id != request.case_id ||
      appended_index.pending_input != nullptr ||
      committed == appended_index.operations.end() ||
      committed->second.correction == nullptr ||
      !committed->second.correction_digest_after.has_value() ||
      committed->second.correction->index != appended.event_index) {
    return absl::DataLossError(
        "DPM correction append was not published as one authoritative event.");
  }

  DPMCorrectionResult result;
  result.correction_event_index = appended.event_index;
  result.correction_digest = *committed->second.correction_digest_after;
  return result;
}

}  // namespace litert::lm
