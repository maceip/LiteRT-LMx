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

// Capsule evidence records exact physical work rather than promoting a
// coverage ID to profile-wide authorization. The complete, currently
// reauthenticated capability/admission/qualification binding below is required
// at every plan and evidence boundary. This is the sole development format;
// earlier unshipped formats are intentionally unsupported.
inline constexpr uint32_t kCapsuleRestoreEvidenceFormatVersion = 1;
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

enum class CapsulePrefillMode : uint32_t {
  kFullCanonicalPrefill = 1,
  kOwnPositionCapsuleDelta = 2,
};

// A root capture starts from a genuinely fresh session and fully prefills the
// current correction epoch. Its raw-log interval therefore need not begin at
// event zero; a higher-level authority check must bind a nonzero start to the
// most recent correction boundary. A verified-parent capture starts only after
// the complete parent restore evidence below has been validated against its
// source capture. No third/off-position basis exists.
enum class CapsuleCaptureBasis : uint32_t {
  kRootFreshSession = 1,
  kVerifiedParentRestore = 2,
};

// Exact model-visible chunk boundaries. Text stays as canonical UTF-8 bytes;
// exact token IDs are never detokenized and re-tokenized. Exactly one payload
// representation is populated.
struct CapsuleCanonicalPrefillChunk {
  enum class Encoding : uint32_t {
    kUtf8Text = 1,
    kExactTokenIds = 2,
  };

  Encoding encoding = Encoding::kUtf8Text;
  std::string utf8_text;
  std::vector<int32_t> token_ids;

  bool operator==(const CapsuleCanonicalPrefillChunk& other) const {
    return encoding == other.encoding && utf8_text == other.utf8_text &&
           token_ids == other.token_ids;
  }
};

// A plan never treats `coverage_id` as authority by itself. The complete
// Engine-derived capability is canonical and must agree with the currently
// reauthenticated admission record and qualification specification named here.
// Repository authentication remains an integration boundary; this value does
// not turn hashes supplied by an application into runtime facts.
struct CapsuleRestoreAuthority {
  SessionHandoffCapability capability;
  Hash256 admission_record_id;
  Hash256 coverage_id;
  Hash256 qualification_evidence_hash;

  bool operator==(const CapsuleRestoreAuthority& other) const {
    return capability == other.capability &&
           admission_record_id == other.admission_record_id &&
           coverage_id == other.coverage_id &&
           qualification_evidence_hash == other.qualification_evidence_hash;
  }
  bool operator!=(const CapsuleRestoreAuthority& other) const {
    return !(*this == other);
  }
};

// Completed decision boundary to which a captured capsule is attached. The
// authoritative raw prefix is the input-bearing prefix [0,
// source_event_count); its not-yet-appended response occupies exactly
// `response_event_index == source_event_count`.
struct CapsuleDPMCheckpointState {
  std::string log_id;
  uint64_t source_event_count = 0;
  Hash256 source_prefix_hash;
  uint64_t response_event_index = 0;
  Hash256 projection_request_hash;
  Hash256 projection_manifest_hash;
  Hash256 correction_digest;
  Hash256 agent_transcript_hash;
  Hash256 logical_agent_request_hash;

  bool operator==(const CapsuleDPMCheckpointState& other) const {
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
  bool operator!=(const CapsuleDPMCheckpointState& other) const {
    return !(*this == other);
  }
};

// Pending target decision for an operational restore. The transcript-prefix
// hash commits the canonical history plus current input before the new decoded
// response is appended; the completed descendant transcript is committed by a
// later CapsuleCapturePlan.
struct CapsuleDPMRestoreTarget {
  std::string log_id;
  uint64_t source_event_count = 0;
  Hash256 source_prefix_hash;
  uint64_t prospective_response_event_index = 0;
  Hash256 projection_request_hash;
  Hash256 projection_manifest_hash;
  Hash256 correction_digest;
  Hash256 agent_transcript_prefix_hash;
  Hash256 logical_agent_request_hash;

  bool operator==(const CapsuleDPMRestoreTarget& other) const {
    return log_id == other.log_id &&
           source_event_count == other.source_event_count &&
           source_prefix_hash == other.source_prefix_hash &&
           prospective_response_event_index ==
               other.prospective_response_event_index &&
           projection_request_hash == other.projection_request_hash &&
           projection_manifest_hash == other.projection_manifest_hash &&
           correction_digest == other.correction_digest &&
           agent_transcript_prefix_hash == other.agent_transcript_prefix_hash &&
           logical_agent_request_hash == other.logical_agent_request_hash;
  }
  bool operator!=(const CapsuleDPMRestoreTarget& other) const {
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
struct CapsulePrefillPlan {
  CapsulePrefillMode mode = CapsulePrefillMode::kFullCanonicalPrefill;
  uint64_t event_range_start = 0;
  uint64_t event_range_end = 0;
  uint32_t start_step = 0;
  uint32_t end_step = 0;
  std::vector<CapsuleCanonicalPrefillChunk> canonical_chunks;
  Hash256 canonical_full_prefill_chunks_hash;
  Hash256 canonical_delta_chunks_hash;
  DPMPreparedPrefillPlan prepared_plan;

  bool operator==(const CapsulePrefillPlan& other) const {
    return mode == other.mode && event_range_start == other.event_range_start &&
           event_range_end == other.event_range_end &&
           start_step == other.start_step && end_step == other.end_step &&
           canonical_chunks == other.canonical_chunks &&
           canonical_full_prefill_chunks_hash ==
               other.canonical_full_prefill_chunks_hash &&
           canonical_delta_chunks_hash == other.canonical_delta_chunks_hash &&
           prepared_plan == other.prepared_plan;
  }
  bool operator!=(const CapsulePrefillPlan& other) const {
    return !(*this == other);
  }
};

// Content-addressed plan for publishing one descendant checkpoint. A root plan
// contains a fresh-session full prefill of the nonempty current correction
// epoch and no parent references. Its event range ends at
// `checkpoint_state.source_event_count`, but may start after event zero when
// the authoritative log has a correction boundary. A verified-parent plan
// contains an own-position delta and names the exact restore evidence that put
// the producer at `prefill.start_step`. The new checkpoint ID is intentionally
// absent because it depends on the envelope produced by this plan.
struct CapsuleCapturePlan {
  static constexpr uint32_t kFormatVersion =
      kCapsuleRestoreEvidenceFormatVersion;

  uint32_t format_version = kFormatVersion;
  Hash256 plan_hash;
  CapsuleRestoreAuthority authority;
  CapsuleCaptureBasis capture_basis = CapsuleCaptureBasis::kRootFreshSession;
  CapsuleDPMCheckpointState checkpoint_state;
  // Canonical transcript immediately before the producing decode. This makes
  // the restore-target-to-descendant join exact; checkpoint_state separately
  // commits the transcript after the decoded output was appended.
  Hash256 agent_transcript_prefix_hash;
  CapsulePrefillPlan prefill;
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
// log prefix/request. Import occurs at `checkpoint_step`; delta prefill
// advances exactly to `prefill.end_step`. Off-position grafting and
// full-prefill restore modes are intentionally unrepresentable by a valid plan.
struct CapsuleRestorePlan {
  static constexpr uint32_t kFormatVersion =
      kCapsuleRestoreEvidenceFormatVersion;

  uint32_t format_version = kFormatVersion;
  Hash256 plan_hash;
  CapsuleRestoreAuthority authority;
  Hash256 source_capture_plan_hash;
  Hash256 source_capture_evidence_id;
  Hash256 checkpoint_id;
  CapsuleDPMCheckpointState checkpoint_state;
  Hash256 checkpoint_envelope_hash;
  uint64_t checkpoint_envelope_size = 0;
  std::string checkpoint_authentication_key_id;
  uint32_t checkpoint_step = 0;
  Hash256 checkpoint_history_token_bytes_hash;
  CapsuleDPMRestoreTarget target_state;
  CapsulePrefillPlan prefill;
  uint32_t maximum_output_tokens = 0;
};

absl::Status ValidateCapsuleCanonicalPrefillChunks(
    const std::vector<CapsuleCanonicalPrefillChunk>& chunks);
absl::StatusOr<Hash256> ComputeCapsuleCanonicalFullPrefillChunksHash(
    const std::vector<CapsuleCanonicalPrefillChunk>& chunks);
absl::StatusOr<Hash256> ComputeCapsuleCanonicalDeltaChunksHash(
    const std::vector<CapsuleCanonicalPrefillChunk>& chunks);

absl::Status ValidateCapsulePrefillPlan(const CapsulePrefillPlan& plan);

absl::StatusOr<Hash256> ComputeCapsuleCapturePlanHash(
    const CapsuleCapturePlan& plan);
absl::Status ValidateCapsuleCapturePlan(const CapsuleCapturePlan& plan);

absl::StatusOr<Hash256> ComputeCapsuleRestorePlanHash(
    const CapsuleRestorePlan& plan);
absl::Status ValidateCapsuleRestorePlan(const CapsuleRestorePlan& plan);

// Operational proof for one durable-checkpoint to request-transient rewrap
// followed by import into a fresh target. The target witness describes the
// transient destination actually imported; it must not be relabeled as the
// durable source envelope.
struct CapsuleRestoreEvidence {
  static constexpr uint32_t kFormatVersion =
      kCapsuleRestoreEvidenceFormatVersion;

  uint32_t format_version = kFormatVersion;
  Hash256 evidence_id;
  CapsuleRestorePlan plan;
  SessionHandoffReauthenticationEvidence durable_to_transient_reauthentication;
  SessionContinuationStateWitness target_post_import;
};

// Operational proof for a newly captured decoded session. The three witnesses
// all describe the same worker-transient producer envelope. The explicit
// reauthentication evidence joins that exact transient endpoint to the
// separately keyed durable checkpoint while committing the complete canonical
// continuation state. Recursive capture embeds the restore evidence that
// established the producing session's own-position starting state.
struct CapsuleCaptureEvidence {
  static constexpr uint32_t kFormatVersion =
      kCapsuleRestoreEvidenceFormatVersion;

  uint32_t format_version = kFormatVersion;
  Hash256 evidence_id;
  CapsuleCapturePlan plan;
  Hash256 checkpoint_id;
  Hash256 checkpoint_envelope_hash;
  uint64_t checkpoint_envelope_size = 0;
  std::string checkpoint_authentication_key_id;
  Hash256 checkpoint_history_token_bytes_hash;
  SessionContinuationStateWitness producer_before_export;
  SessionContinuationStateWitness producer_after_export;
  SessionContinuationStateWitness fresh_import_target;
  SessionHandoffReauthenticationEvidence transient_to_durable_reauthentication;
  std::optional<CapsuleRestoreEvidence> parent_restore_evidence;
};

absl::StatusOr<Hash256> ComputeCapsuleRestoreEvidenceId(
    const CapsuleRestoreEvidence& evidence);
absl::Status ValidateCapsuleRestoreEvidence(
    const CapsuleRestoreEvidence& evidence);

absl::StatusOr<Hash256> ComputeCapsuleCaptureEvidenceId(
    const CapsuleCaptureEvidence& evidence);
absl::Status ValidateCapsuleCaptureEvidence(
    const CapsuleCaptureEvidence& evidence);

// Proves the complete endpoint chain
// capture-transient -> durable checkpoint -> restore-transient. The two
// transient envelopes and witnesses remain intentionally distinct; the exact
// durable endpoint and the rewrap-invariant canonical continuation-state hash
// must agree.
absl::Status ValidateCapsuleRestoreEvidenceForSourceCapture(
    const CapsuleRestoreEvidence& restore_evidence,
    const CapsuleCaptureEvidence& source_capture_evidence);

absl::Status ValidateCapsuleCaptureEvidenceForParentCapture(
    const CapsuleCaptureEvidence& child_capture_evidence,
    const CapsuleCaptureEvidence& parent_capture_evidence);

Hash256 GetCapsuleRestoreCaptureEvidenceContractHash();
Hash256 GetCapsuleRestoreRestoreEvidenceContractHash();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_H_
