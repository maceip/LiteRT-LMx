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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/dpm_prepared_prefill_plan.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/engine/session_handoff_capability.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Coverage V2 records exact physical work rather than promoting a coverage ID
// to a profile-wide authorization. The complete, currently reauthenticated
// capability/admission/qualification binding below is required at every plan
// and evidence boundary. The limits keep canonical hashing bounded even though
// text and exact token sequences remain self-contained.
inline constexpr uint32_t kCapsuleRestoreEvidenceV2FormatVersion = 2;
inline constexpr std::size_t kMaximumCapsuleEvidenceLogIdBytes = 16 * 1024;
inline constexpr std::size_t kMaximumCapsuleEvidenceKeyIdBytes = 1024;
inline constexpr uint32_t kMaximumCapsuleEvidencePrefillChunks = 65'536;
inline constexpr uint64_t kMaximumCapsuleEvidencePrefillTextBytes =
    uint64_t{16} * 1024 * 1024;
inline constexpr uint32_t kMaximumCapsuleEvidenceTokenIds = 1'000'000;
inline constexpr uint32_t kMaximumCapsuleEvidenceShapeInvocations = 1'000'000;
inline constexpr uint64_t kMaximumCapsuleEvidenceEnvelopeBytes =
    uint64_t{8} * 1024 * 1024 * 1024;
inline constexpr uint32_t kMaximumCapsuleEvidenceGenerationTokens = 1'000'000;

enum class CapsulePrefillModeV2 : uint32_t {
  kFullCanonicalPrefill = 1,
  kOwnPositionCapsuleDelta = 2,
};

// A root capture starts from a genuinely fresh session. A verified-parent
// capture starts only after the complete parent restore evidence below has been
// validated against its source capture. No third/off-position basis exists.
enum class CapsuleCaptureBasisV2 : uint32_t {
  kRootFreshSession = 1,
  kVerifiedParentRestore = 2,
};

// Exact model-visible chunk boundaries. Text stays as canonical UTF-8 bytes;
// exact token IDs are never detokenized and re-tokenized. Exactly one payload
// representation is populated.
struct CapsuleCanonicalPrefillChunkV2 {
  enum class Encoding : uint32_t {
    kUtf8Text = 1,
    kExactTokenIds = 2,
  };

  Encoding encoding = Encoding::kUtf8Text;
  std::string utf8_text;
  std::vector<int32_t> token_ids;

  bool operator==(const CapsuleCanonicalPrefillChunkV2& other) const {
    return encoding == other.encoding && utf8_text == other.utf8_text &&
           token_ids == other.token_ids;
  }
};

// A plan never treats `coverage_id` as authority by itself. The complete
// Engine-derived capability is canonical and must agree with the currently
// reauthenticated admission record and qualification specification named here.
// Repository authentication remains an integration boundary; this value does
// not turn hashes supplied by an application into runtime facts.
struct CapsuleRestoreAuthorityV2 {
  SessionHandoffCapability capability;
  Hash256 admission_record_id;
  Hash256 coverage_id;
  Hash256 qualification_spec_hash;

  bool operator==(const CapsuleRestoreAuthorityV2& other) const {
    return capability == other.capability &&
           admission_record_id == other.admission_record_id &&
           coverage_id == other.coverage_id &&
           qualification_spec_hash == other.qualification_spec_hash;
  }
  bool operator!=(const CapsuleRestoreAuthorityV2& other) const {
    return !(*this == other);
  }
};

// Completed decision boundary to which a captured capsule is attached. The
// authoritative raw prefix is the input-bearing prefix [0,
// source_event_count); its not-yet-appended response occupies exactly
// `response_event_index == source_event_count`.
struct CapsuleDPMCheckpointStateV2 {
  std::string log_id;
  uint64_t source_event_count = 0;
  Hash256 source_prefix_hash;
  uint64_t response_event_index = 0;
  Hash256 projection_request_hash;
  Hash256 projection_manifest_hash;
  Hash256 correction_digest;
  Hash256 agent_transcript_hash;
  Hash256 logical_agent_request_hash;

  bool operator==(const CapsuleDPMCheckpointStateV2& other) const {
    return log_id == other.log_id &&
           source_event_count == other.source_event_count &&
           source_prefix_hash == other.source_prefix_hash &&
           response_event_index == other.response_event_index &&
           projection_request_hash == other.projection_request_hash &&
           projection_manifest_hash == other.projection_manifest_hash &&
           correction_digest == other.correction_digest &&
           agent_transcript_hash == other.agent_transcript_hash &&
           logical_agent_request_hash == other.logical_agent_request_hash;
  }
  bool operator!=(const CapsuleDPMCheckpointStateV2& other) const {
    return !(*this == other);
  }
};

// Pending target decision for an operational restore. The transcript-prefix
// hash commits the canonical history plus current input before the new decoded
// response is appended; the completed descendant transcript is committed by a
// later CapsuleCapturePlanV2.
struct CapsuleDPMRestoreTargetV2 {
  std::string log_id;
  uint64_t source_event_count = 0;
  Hash256 source_prefix_hash;
  uint64_t prospective_response_event_index = 0;
  Hash256 projection_request_hash;
  Hash256 projection_manifest_hash;
  Hash256 correction_digest;
  Hash256 agent_transcript_prefix_hash;
  Hash256 logical_agent_request_hash;

  bool operator==(const CapsuleDPMRestoreTargetV2& other) const {
    return log_id == other.log_id &&
           source_event_count == other.source_event_count &&
           source_prefix_hash == other.source_prefix_hash &&
           prospective_response_event_index ==
               other.prospective_response_event_index &&
           projection_request_hash == other.projection_request_hash &&
           projection_manifest_hash == other.projection_manifest_hash &&
           correction_digest == other.correction_digest &&
           agent_transcript_prefix_hash ==
               other.agent_transcript_prefix_hash &&
           logical_agent_request_hash == other.logical_agent_request_hash;
  }
  bool operator!=(const CapsuleDPMRestoreTargetV2& other) const {
    return !(*this == other);
  }
};

// Complete physical prefill selection. The event range is half-open and names
// the exact raw-log interval from which the canonical chunks were derived. The
// full and delta hashes are mutually exclusive and domain-separated. The
// runtime-derived prepared plan retains every physical call, every 1-2 exact
// token-buffer segment boundary (including a separate fresh-session BOS
// segment), exact call positions, vocabulary, starting-history commitment,
// resolved-token hash, shape-schedule hash, and per-run witness-bound plan ID.
// Validators recompute all prepared-plan hashes and join each call back to the
// corresponding canonical source chunk; no flattened or opaque digest can hide
// BOS insertion, buffer segmentation, or a changed physical call schedule.
struct CapsulePrefillPlanV2 {
  CapsulePrefillModeV2 mode = CapsulePrefillModeV2::kFullCanonicalPrefill;
  uint64_t event_range_start = 0;
  uint64_t event_range_end = 0;
  uint32_t start_step = 0;
  uint32_t end_step = 0;
  std::vector<CapsuleCanonicalPrefillChunkV2> canonical_chunks;
  Hash256 canonical_full_prefill_chunks_hash;
  Hash256 canonical_delta_chunks_hash;
  DPMPreparedPrefillPlan prepared_plan;

  bool operator==(const CapsulePrefillPlanV2& other) const {
    return mode == other.mode && event_range_start == other.event_range_start &&
           event_range_end == other.event_range_end &&
           start_step == other.start_step && end_step == other.end_step &&
           canonical_chunks == other.canonical_chunks &&
           canonical_full_prefill_chunks_hash ==
               other.canonical_full_prefill_chunks_hash &&
           canonical_delta_chunks_hash ==
               other.canonical_delta_chunks_hash &&
           prepared_plan == other.prepared_plan;
  }
  bool operator!=(const CapsulePrefillPlanV2& other) const {
    return !(*this == other);
  }
};

// Content-addressed plan for publishing one descendant checkpoint. A root plan
// contains a full prefill and no parent references. A verified-parent plan
// contains an own-position delta and names the exact restore evidence that put
// the producer at `prefill.start_step`. The new checkpoint ID is intentionally
// absent because it depends on the envelope produced by this plan.
struct CapsuleCapturePlanV2 {
  static constexpr uint32_t kFormatVersion =
      kCapsuleRestoreEvidenceV2FormatVersion;

  uint32_t format_version = kFormatVersion;
  Hash256 plan_hash;
  CapsuleRestoreAuthorityV2 authority;
  CapsuleCaptureBasisV2 capture_basis =
      CapsuleCaptureBasisV2::kRootFreshSession;
  CapsuleDPMCheckpointStateV2 checkpoint_state;
  // Canonical transcript immediately before the producing decode. This makes
  // the restore-target-to-descendant join exact; checkpoint_state separately
  // commits the transcript after the decoded output was appended.
  Hash256 agent_transcript_prefix_hash;
  CapsulePrefillPlanV2 prefill;
  Hash256 producing_output_evidence_hash;
  uint32_t generated_token_count = 0;
  uint32_t capture_end_step = 0;
  std::string checkpoint_authentication_key_id;
  std::optional<Hash256> parent_checkpoint_id;
  std::optional<uint64_t> parent_response_event_index;
  std::optional<Hash256> parent_restore_evidence_id;
};

// Content-addressed operational plan for one own-position restore followed by
// an exact delta. It binds both the source checkpoint state and the later raw
// log prefix/request. Import occurs at `checkpoint_step`; delta prefill advances
// exactly to `prefill.end_step`. Off-position grafting and full-prefill restore
// modes are intentionally unrepresentable by a valid plan.
struct CapsuleRestorePlanV2 {
  static constexpr uint32_t kFormatVersion =
      kCapsuleRestoreEvidenceV2FormatVersion;

  uint32_t format_version = kFormatVersion;
  Hash256 plan_hash;
  CapsuleRestoreAuthorityV2 authority;
  Hash256 source_capture_plan_hash;
  Hash256 source_capture_evidence_id;
  Hash256 checkpoint_id;
  CapsuleDPMCheckpointStateV2 checkpoint_state;
  Hash256 checkpoint_envelope_hash;
  uint64_t checkpoint_envelope_size = 0;
  std::string checkpoint_authentication_key_id;
  uint32_t checkpoint_step = 0;
  Hash256 checkpoint_history_token_bytes_hash;
  CapsuleDPMRestoreTargetV2 target_state;
  CapsulePrefillPlanV2 prefill;
  uint32_t maximum_output_tokens = 0;
};

// Actual target observation immediately after import and a canonical re-export,
// before any delta token is applied. This evidence alone is not authorization:
// ValidateCapsuleRestoreEvidenceV2ForSourceCapture must also match the complete
// source capture artifact and current authority.
struct CapsuleRestoreEvidenceV2 {
  static constexpr uint32_t kFormatVersion =
      kCapsuleRestoreEvidenceV2FormatVersion;

  uint32_t format_version = kFormatVersion;
  Hash256 evidence_id;
  CapsuleRestorePlanV2 plan;
  SessionContinuationStateWitness target_post_import;
};

// Evidence for a newly captured decoded session. Export must be observational:
// producer-before and producer-after witnesses are identical. A separate fresh
// session then imports and re-exports the envelope, producing the same canonical
// decoded witness. Recursive capture additionally carries the exact parent
// restore evidence used to establish its own-position starting state.
struct CapsuleCaptureEvidenceV2 {
  static constexpr uint32_t kFormatVersion =
      kCapsuleRestoreEvidenceV2FormatVersion;

  uint32_t format_version = kFormatVersion;
  Hash256 evidence_id;
  CapsuleCapturePlanV2 plan;
  Hash256 checkpoint_id;
  Hash256 checkpoint_envelope_hash;
  uint64_t checkpoint_envelope_size = 0;
  std::string checkpoint_authentication_key_id;
  Hash256 checkpoint_history_token_bytes_hash;
  SessionContinuationStateWitness producer_before_export;
  SessionContinuationStateWitness producer_after_export;
  SessionContinuationStateWitness fresh_import_target;
  std::optional<CapsuleRestoreEvidenceV2> parent_restore_evidence;
};

absl::Status ValidateCapsuleCanonicalPrefillChunksV2(
    const std::vector<CapsuleCanonicalPrefillChunkV2>& chunks);
absl::StatusOr<Hash256> ComputeCapsuleCanonicalFullPrefillChunksHashV2(
    const std::vector<CapsuleCanonicalPrefillChunkV2>& chunks);
absl::StatusOr<Hash256> ComputeCapsuleCanonicalDeltaChunksHashV2(
    const std::vector<CapsuleCanonicalPrefillChunkV2>& chunks);

absl::Status ValidateCapsulePrefillPlanV2(
    const CapsulePrefillPlanV2& plan);

absl::StatusOr<Hash256> ComputeCapsuleCapturePlanV2Hash(
    const CapsuleCapturePlanV2& plan);
absl::Status ValidateCapsuleCapturePlanV2(
    const CapsuleCapturePlanV2& plan);

absl::StatusOr<Hash256> ComputeCapsuleRestorePlanV2Hash(
    const CapsuleRestorePlanV2& plan);
absl::Status ValidateCapsuleRestorePlanV2(
    const CapsuleRestorePlanV2& plan);

absl::StatusOr<Hash256> ComputeCapsuleRestoreEvidenceV2Id(
    const CapsuleRestoreEvidenceV2& evidence);
absl::Status ValidateCapsuleRestoreEvidenceV2(
    const CapsuleRestoreEvidenceV2& evidence);

absl::StatusOr<Hash256> ComputeCapsuleCaptureEvidenceV2Id(
    const CapsuleCaptureEvidenceV2& evidence);
absl::Status ValidateCapsuleCaptureEvidenceV2(
    const CapsuleCaptureEvidenceV2& evidence);

// Required authorization join for an operational restore. It proves that the
// source capture, checkpoint, envelope, exact own-position witness, and full
// authority all match the restore plan. A matching coverage_id without this
// join is rejected.
absl::Status ValidateCapsuleRestoreEvidenceV2ForSourceCapture(
    const CapsuleRestoreEvidenceV2& restore_evidence,
    const CapsuleCaptureEvidenceV2& source_capture_evidence);

// Required authorization join for a verified-parent descendant capture. It
// validates the embedded parent restore against the supplied parent capture and
// then proves that the child's delta plan starts from that restored own-position
// state. Root captures are rejected by this function.
absl::Status ValidateCapsuleCaptureEvidenceV2ForParentCapture(
    const CapsuleCaptureEvidenceV2& child_capture_evidence,
    const CapsuleCaptureEvidenceV2& parent_capture_evidence);

// Runtime-owned schema identities bound by Coverage V2 admission. These do
// not authenticate an operation; they prevent an admission record for another
// capture/restore evidence contract from being applied to these validators.
Hash256 GetCapsuleRestoreCaptureEvidenceV2ContractHash();
Hash256 GetCapsuleRestoreRestoreEvidenceV2ContractHash();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_H_
