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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_ADMISSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_ADMISSION_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/engine/session_handoff_capability.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Product limits for the exact qualification request itself. These limits are
// independent of the potentially much larger LRTSESS1 checkpoint envelope,
// which is authenticated and imported directly by Session.
inline constexpr uint32_t kMaximumCapsuleRestorePrefillChunks = 65'536;
inline constexpr uint64_t kMaximumCapsuleRestorePrefillTextBytes =
    uint64_t{16} * 1024 * 1024;
inline constexpr uint32_t kMaximumCapsuleRestorePrefillTokenIds = 1'000'000;
inline constexpr uint64_t kMaximumCapsuleRestoreAdmissionEnvelopeBytes =
    kMaximumFreshWorkerEnvelopeBytes;
inline constexpr uint64_t kMaximumCapsuleRestoreCheckpointBytes =
    uint64_t{8} * 1024 * 1024 * 1024;
inline constexpr uint32_t
    kMaximumCapsuleRestoreStateWitnessQualificationCases = 4096;
inline constexpr uint32_t
    kMaximumCapsuleRestoreStateWitnessQualificationTrials = 64;

// One exact prefill boundary. Text is retained as canonical UTF-8 bytes;
// token IDs are retained as explicit nonnegative int32 values and are never
// detokenized or re-tokenized. Exactly one representation must be populated.
struct CapsuleRestorePrefillChunk {
  enum class Encoding : uint32_t {
    kUtf8Text = 1,
    kExactTokenIds = 2,
  };

  Encoding encoding = Encoding::kUtf8Text;
  std::string utf8_text;
  std::vector<int32_t> token_ids;

  bool operator==(const CapsuleRestorePrefillChunk& other) const {
    return encoding == other.encoding && utf8_text == other.utf8_text &&
           token_ids == other.token_ids;
  }
  bool operator!=(const CapsuleRestorePrefillChunk& other) const {
    return !(*this == other);
  }
};

// A concrete, single-profile qualification request. Assertions can only
// reject runtime-derived profile/capability evidence. They do not enter the
// canonical request hash and cannot supply any authoritative identity.
struct CapsuleRestoreQualificationSpec {
  SessionConfig session_config = SessionConfig::CreateDefault();
  ExactLiteRtProfileAssertion exact_profile_assertion;
  SessionHandoffCapabilityAssertion capability_assertion;

  // The source is brought to a real post-decode producing boundary before its
  // capsule is captured. Every chunk below is issued as exactly one prefill
  // call, preserving text/token and chunk boundaries.
  std::vector<CapsuleRestorePrefillChunk> checkpoint_prefix_chunks;
  uint32_t checkpoint_output_tokens = 0;

  // After own-position restore, these chunks are applied identically to the
  // live source and restored target before continuation equality is measured.
  std::vector<CapsuleRestorePrefillChunk> delta_chunks;
  uint32_t continuation_output_tokens = 0;
};

// Canonical evidence for one continuation decode. DPMTOK01 token bytes and
// ordered full-vocabulary logits frames remain separate from visible text.
struct CapsuleRestoreContinuationEvidence {
  std::string visible_output;
  std::string token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> logit_frames;
  Hash256 output_evidence_hash;

  bool operator==(const CapsuleRestoreContinuationEvidence& other) const {
    return visible_output == other.visible_output &&
           token_bytes == other.token_bytes &&
           logit_frames == other.logit_frames &&
           output_evidence_hash == other.output_evidence_hash;
  }
  bool operator!=(const CapsuleRestoreContinuationEvidence& other) const {
    return !(*this == other);
  }
};

// Durable evidence that this one complete Engine-derived capability passed the
// named concrete own-position restore qualification. Product policy may admit
// operational capsules only for this same derived profile, capability, codec,
// and qualification specification. The record is not an operational
// checkpoint and does not authenticate arbitrary capsule bytes: every DPM
// checkpoint still requires its own LRTSESS1 authentication, Engine identity,
// raw-log provenance, own-position delta plan, and current record
// reauthentication. A different profile, capability, codec, or qualification
// specification requires a different admission record.
struct CapsuleRestoreAdmissionRecord {
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;
  Hash256 record_id;
  SessionHandoffCapability capability;
  Hash256 qualification_spec_hash;

  uint64_t checkpoint_step = 0;
  std::string checkpoint_history_token_bytes;
  Hash256 checkpoint_envelope_hash;
  uint64_t checkpoint_envelope_size = 0;
  std::string checkpoint_authentication_key_id;

  CapsuleRestoreContinuationEvidence live_continuation;
  CapsuleRestoreContinuationEvidence restored_continuation;

  int64_t qualified_unix_micros = 0;
  std::string record_authentication_key_id;
};

// Operational scope certified by one CapsuleRestore admission. Version 1 is
// intentionally an exact-workload contract: it authorizes only the precise
// checkpoint-producing request and the precise own-position continuation
// request used by qualification. It is not a range, profile-wide lease, or a
// claim about arbitrary checkpoint steps. Later coverage versions may describe
// a rigorously qualified bounded domain without changing the durable DPM
// descriptor/receipt fields, which bind only coverage_id.
enum class CapsuleRestoreCoverageKind : uint32_t {
  kExactWorkload = 1,
  kStateWitnessedOwnPosition = 2,
};

struct CapsuleRestoreOperationalCoverage {
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;
  CapsuleRestoreCoverageKind kind =
      CapsuleRestoreCoverageKind::kExactWorkload;
  Hash256 coverage_id;
  Hash256 capsule_restore_capability_id;
  Hash256 capsule_restore_admission_record_id;
  Hash256 qualification_spec_hash;

  // Canonical hashes include every ordered text/token chunk boundary and the
  // corresponding generation limit. Matching bytes with different chunking
  // are deliberately different workloads.
  Hash256 checkpoint_capture_workload_hash;
  Hash256 restore_continuation_workload_hash;

  // These observations bind the only checkpoint state qualified by Coverage
  // V1. The exact token history remains private to the admission record; its
  // canonical DPMTOK01 digest is sufficient for live-parent capture checks.
  uint64_t checkpoint_step = 0;
  Hash256 checkpoint_history_token_bytes_hash;
  uint64_t checkpoint_envelope_size = 0;
  std::string checkpoint_authentication_key_id;
  uint32_t checkpoint_output_tokens = 0;
  uint32_t continuation_output_tokens = 0;

  bool operator==(const CapsuleRestoreOperationalCoverage& other) const {
    return format_version == other.format_version && kind == other.kind &&
           coverage_id == other.coverage_id &&
           capsule_restore_capability_id ==
               other.capsule_restore_capability_id &&
           capsule_restore_admission_record_id ==
               other.capsule_restore_admission_record_id &&
           qualification_spec_hash == other.qualification_spec_hash &&
           checkpoint_capture_workload_hash ==
               other.checkpoint_capture_workload_hash &&
           restore_continuation_workload_hash ==
               other.restore_continuation_workload_hash &&
           checkpoint_step == other.checkpoint_step &&
           checkpoint_history_token_bytes_hash ==
               other.checkpoint_history_token_bytes_hash &&
           checkpoint_envelope_size == other.checkpoint_envelope_size &&
           checkpoint_authentication_key_id ==
               other.checkpoint_authentication_key_id &&
           checkpoint_output_tokens == other.checkpoint_output_tokens &&
           continuation_output_tokens ==
               other.continuation_output_tokens;
  }
  bool operator!=(const CapsuleRestoreOperationalCoverage& other) const {
    return !(*this == other);
  }
};

// Coverage V2 is a parallel protocol, not a widened interpretation of
// CapsuleRestoreOperationalCoverage. V1 continues to authorize one exact
// workload. V2 qualifies a bounded execution-shape domain, while every actual
// operation remains unauthorized until it presents a future
// SessionContinuationStateWitness and authenticated capture/restore evidence.
// In particular, no finite qualification prompt corpus authorizes other
// content.
enum class CapsuleRestoreStateWitnessCapturePhase : uint32_t {
  // The source has completed at least one decode, and the capsule is restored
  // at the exact token position at which it was captured.
  kDecodedOwnPosition = 1,
};

enum class CapsuleRestoreStateWitnessContentAuthority : uint32_t {
  // Qualification evidence establishes only the named implementation and
  // shape classes. Content authority is always per operation.
  kPerOperationStateWitnessAndEvidenceOnly = 1,
};

enum class CapsuleRestoreStateWitnessOperationEvidence : uint32_t {
  kSessionContinuationStateWitness = uint32_t{1} << 0,
  kCaptureEvidence = uint32_t{1} << 1,
  kRestoreEvidence = uint32_t{1} << 2,
};

constexpr uint32_t CapsuleRestoreStateWitnessOperationEvidenceBit(
    CapsuleRestoreStateWitnessOperationEvidence evidence) {
  return static_cast<uint32_t>(evidence);
}

constexpr uint32_t
CapsuleRestoreStateWitnessRequiredOperationEvidenceMask() {
  return CapsuleRestoreStateWitnessOperationEvidenceBit(
             CapsuleRestoreStateWitnessOperationEvidence::
                 kSessionContinuationStateWitness) |
         CapsuleRestoreStateWitnessOperationEvidenceBit(
             CapsuleRestoreStateWitnessOperationEvidence::kCaptureEvidence) |
         CapsuleRestoreStateWitnessOperationEvidenceBit(
             CapsuleRestoreStateWitnessOperationEvidence::kRestoreEvidence);
}

enum class CapsuleRestoreStateWitnessEncoding : uint32_t {
  kUtf8Text = uint32_t{1} << 0,
  kExactTokenIds = uint32_t{1} << 1,
};

constexpr uint32_t CapsuleRestoreStateWitnessEncodingBit(
    CapsuleRestoreStateWitnessEncoding encoding) {
  return static_cast<uint32_t>(encoding);
}

constexpr uint32_t CapsuleRestoreStateWitnessAllowedEncodingMask() {
  return CapsuleRestoreStateWitnessEncodingBit(
             CapsuleRestoreStateWitnessEncoding::kUtf8Text) |
         CapsuleRestoreStateWitnessEncodingBit(
             CapsuleRestoreStateWitnessEncoding::kExactTokenIds);
}

enum class CapsuleRestoreStateWitnessQualificationCaseKind : uint32_t {
  // A stateful source is prefetched from event zero, decoded, captured, and
  // restored at its own position.
  kFullPrefillCaptureAndOwnPositionRestore = uint32_t{1} << 0,
  // The producing source was itself restored from an admitted ancestor,
  // delta-prefetched, decoded, captured, and restored at its own position.
  kRestoredAncestorDeltaCaptureAndOwnPositionRestore = uint32_t{1} << 1,
};

constexpr uint32_t CapsuleRestoreStateWitnessQualificationCaseKindBit(
    CapsuleRestoreStateWitnessQualificationCaseKind kind) {
  return static_cast<uint32_t>(kind);
}

constexpr uint32_t
CapsuleRestoreStateWitnessRequiredQualificationCaseKindMask() {
  return CapsuleRestoreStateWitnessQualificationCaseKindBit(
             CapsuleRestoreStateWitnessQualificationCaseKind::
                 kFullPrefillCaptureAndOwnPositionRestore) |
         CapsuleRestoreStateWitnessQualificationCaseKindBit(
             CapsuleRestoreStateWitnessQualificationCaseKind::
                 kRestoredAncestorDeltaCaptureAndOwnPositionRestore);
}

// Product-owned qualification policy. The authenticated qualifier chooses the
// bounded counts; validation never accepts fewer than two independent trials,
// a weaker authorization mode, missing operation evidence, or missing
// full-prefill/descendant pathways.
struct CapsuleRestoreStateWitnessQualificationPolicy {
  static constexpr uint32_t kFormatVersion = 2;

  uint32_t format_version = kFormatVersion;
  CapsuleRestoreStateWitnessContentAuthority content_authority =
      CapsuleRestoreStateWitnessContentAuthority::
          kPerOperationStateWitnessAndEvidenceOnly;
  uint32_t required_operation_evidence_mask =
      CapsuleRestoreStateWitnessRequiredOperationEvidenceMask();
  uint32_t required_qualification_case_kind_mask =
      CapsuleRestoreStateWitnessRequiredQualificationCaseKindMask();
  uint32_t minimum_independent_trials_per_shape_class_and_kind = 2;
  uint32_t maximum_qualified_shape_classes = 1;
  Hash256 qualification_verifier_contract_hash;

  bool operator==(
      const CapsuleRestoreStateWitnessQualificationPolicy& other) const {
    return format_version == other.format_version &&
           content_authority == other.content_authority &&
           required_operation_evidence_mask ==
               other.required_operation_evidence_mask &&
           required_qualification_case_kind_mask ==
               other.required_qualification_case_kind_mask &&
           minimum_independent_trials_per_shape_class_and_kind ==
               other.minimum_independent_trials_per_shape_class_and_kind &&
           maximum_qualified_shape_classes ==
               other.maximum_qualified_shape_classes &&
           qualification_verifier_contract_hash ==
               other.qualification_verifier_contract_hash;
  }
  bool operator!=(
      const CapsuleRestoreStateWitnessQualificationPolicy& other) const {
    return !(*this == other);
  }
};

// A bounded operational domain for state-witnessed own-position restore.
// Every opaque contract hash names product/runtime code; it is authenticated
// qualification input, never a caller assertion. Version 2 is CPU-only until
// a later version can bind complete backend-native state and policy evidence.
struct CapsuleRestoreStateWitnessOperationalDomain {
  static constexpr uint32_t kFormatVersion = 2;

  uint32_t format_version = kFormatVersion;
  CapsuleRestoreCoverageKind kind =
      CapsuleRestoreCoverageKind::kStateWitnessedOwnPosition;
  CapsuleRestoreStateWitnessCapturePhase capture_phase =
      CapsuleRestoreStateWitnessCapturePhase::kDecodedOwnPosition;
  ExactLiteRtBackend admitted_backend = ExactLiteRtBackend::kCpu;

  Hash256 resolved_session_config_hash;
  // Coordinates with the future SessionContinuationStateWitness type without
  // defining or weakening that session-owned evidence here.
  Hash256 session_continuation_state_witness_contract_hash;
  Hash256 capture_evidence_contract_hash;
  Hash256 restore_evidence_contract_hash;
  Hash256 deterministic_prefill_plan_contract_hash;
  Hash256 execution_shape_class_contract_hash;
  Hash256 restricted_feature_contract_hash;

  uint64_t maximum_context_positions = 0;
  uint64_t minimum_checkpoint_step = 0;
  uint64_t maximum_checkpoint_step = 0;
  uint64_t minimum_delta_positions = 0;
  uint64_t maximum_delta_positions = 0;
  uint32_t minimum_prefill_chunks = 0;
  uint32_t maximum_prefill_chunks = 0;
  uint64_t maximum_prefill_text_bytes = 0;
  uint32_t maximum_prefill_token_ids = 0;
  uint32_t maximum_output_tokens = 0;
  uint32_t admitted_encoding_mask = 0;

  // Actual capsules and per-operation evidence must authenticate under these
  // non-secret key IDs. Secret key material is never serialized here.
  std::string checkpoint_authentication_key_id;
  std::string operation_evidence_authentication_key_id;

  CapsuleRestoreStateWitnessQualificationPolicy qualification_policy;

  bool operator==(
      const CapsuleRestoreStateWitnessOperationalDomain& other) const {
    return format_version == other.format_version && kind == other.kind &&
           capture_phase == other.capture_phase &&
           admitted_backend == other.admitted_backend &&
           resolved_session_config_hash ==
               other.resolved_session_config_hash &&
           session_continuation_state_witness_contract_hash ==
               other.session_continuation_state_witness_contract_hash &&
           capture_evidence_contract_hash ==
               other.capture_evidence_contract_hash &&
           restore_evidence_contract_hash ==
               other.restore_evidence_contract_hash &&
           deterministic_prefill_plan_contract_hash ==
               other.deterministic_prefill_plan_contract_hash &&
           execution_shape_class_contract_hash ==
               other.execution_shape_class_contract_hash &&
           restricted_feature_contract_hash ==
               other.restricted_feature_contract_hash &&
           maximum_context_positions == other.maximum_context_positions &&
           minimum_checkpoint_step == other.minimum_checkpoint_step &&
           maximum_checkpoint_step == other.maximum_checkpoint_step &&
           minimum_delta_positions == other.minimum_delta_positions &&
           maximum_delta_positions == other.maximum_delta_positions &&
           minimum_prefill_chunks == other.minimum_prefill_chunks &&
           maximum_prefill_chunks == other.maximum_prefill_chunks &&
           maximum_prefill_text_bytes ==
               other.maximum_prefill_text_bytes &&
           maximum_prefill_token_ids ==
               other.maximum_prefill_token_ids &&
           maximum_output_tokens == other.maximum_output_tokens &&
           admitted_encoding_mask == other.admitted_encoding_mask &&
           checkpoint_authentication_key_id ==
               other.checkpoint_authentication_key_id &&
           operation_evidence_authentication_key_id ==
               other.operation_evidence_authentication_key_id &&
           qualification_policy == other.qualification_policy;
  }
  bool operator!=(
      const CapsuleRestoreStateWitnessOperationalDomain& other) const {
    return !(*this == other);
  }
};

// One authenticated mechanism-qualification observation. `shape_class_hash`
// is produced by the domain's deterministic shape-class contract. These
// observations never authorize their prompts or any other content; they only
// qualify the implementation pathway, and V2 still requires fresh operation
// evidence for every real capture and restore.
struct CapsuleRestoreStateWitnessQualificationCaseEvidence {
  static constexpr uint32_t kFormatVersion = 2;

  uint32_t format_version = kFormatVersion;
  Hash256 qualification_case_id;
  CapsuleRestoreStateWitnessQualificationCaseKind kind =
      CapsuleRestoreStateWitnessQualificationCaseKind::
          kFullPrefillCaptureAndOwnPositionRestore;
  Hash256 shape_class_hash;
  Hash256 trial_identity_hash;
  Hash256 source_session_instance_hash;
  Hash256 target_session_instance_hash;

  uint64_t checkpoint_step = 0;
  uint64_t delta_positions = 0;
  uint32_t prefill_chunk_count = 0;
  uint64_t prefill_text_bytes = 0;
  uint32_t prefill_token_ids = 0;
  uint32_t output_tokens = 0;
  uint32_t observed_encoding_mask = 0;

  Hash256 producer_state_witness_hash;
  Hash256 restored_state_witness_hash;
  Hash256 capture_evidence_hash;
  Hash256 restore_evidence_hash;
  Hash256 live_continuation_output_evidence_hash;
  Hash256 restored_continuation_output_evidence_hash;
  Hash256 verifier_certification_hash;

  bool operator==(
      const CapsuleRestoreStateWitnessQualificationCaseEvidence& other) const {
    return format_version == other.format_version &&
           qualification_case_id == other.qualification_case_id &&
           kind == other.kind &&
           shape_class_hash == other.shape_class_hash &&
           trial_identity_hash == other.trial_identity_hash &&
           source_session_instance_hash ==
               other.source_session_instance_hash &&
           target_session_instance_hash ==
               other.target_session_instance_hash &&
           checkpoint_step == other.checkpoint_step &&
           delta_positions == other.delta_positions &&
           prefill_chunk_count == other.prefill_chunk_count &&
           prefill_text_bytes == other.prefill_text_bytes &&
           prefill_token_ids == other.prefill_token_ids &&
           output_tokens == other.output_tokens &&
           observed_encoding_mask == other.observed_encoding_mask &&
           producer_state_witness_hash ==
               other.producer_state_witness_hash &&
           restored_state_witness_hash ==
               other.restored_state_witness_hash &&
           capture_evidence_hash == other.capture_evidence_hash &&
           restore_evidence_hash == other.restore_evidence_hash &&
           live_continuation_output_evidence_hash ==
               other.live_continuation_output_evidence_hash &&
           restored_continuation_output_evidence_hash ==
               other.restored_continuation_output_evidence_hash &&
           verifier_certification_hash ==
               other.verifier_certification_hash;
  }
  bool operator!=(
      const CapsuleRestoreStateWitnessQualificationCaseEvidence& other) const {
    return !(*this == other);
  }
};

// Durable Coverage V2 authority. Its generic `coverage_id` remains suitable
// for existing descriptors and receipts, but its canonical V2 preimage binds
// the complete runtime-derived profile/capability/session evidence, including
// the state inventory and capsule codec hashes carried by `capability`.
struct CapsuleRestoreStateWitnessOperationalCoverage {
  static constexpr uint32_t kFormatVersion = 2;

  uint32_t format_version = kFormatVersion;
  CapsuleRestoreCoverageKind kind =
      CapsuleRestoreCoverageKind::kStateWitnessedOwnPosition;
  Hash256 coverage_id;
  ExactLiteRtProfile runtime_derived_profile;
  SessionHandoffCapability runtime_derived_capability;
  SessionHandoffIdentity runtime_derived_session_identity;
  CapsuleRestoreStateWitnessOperationalDomain operational_domain;
  Hash256 qualification_evidence_hash;

  bool operator==(
      const CapsuleRestoreStateWitnessOperationalCoverage& other) const {
    return format_version == other.format_version && kind == other.kind &&
           coverage_id == other.coverage_id &&
           runtime_derived_profile == other.runtime_derived_profile &&
           runtime_derived_capability == other.runtime_derived_capability &&
           runtime_derived_session_identity ==
               other.runtime_derived_session_identity &&
           operational_domain == other.operational_domain &&
           qualification_evidence_hash ==
               other.qualification_evidence_hash;
  }
  bool operator!=(
      const CapsuleRestoreStateWitnessOperationalCoverage& other) const {
    return !(*this == other);
  }
};

// Authenticated qualification authority for Coverage V2. The cases are
// mechanism evidence and must exactly hash to the coverage's evidence digest.
// Runtime use remains fail-closed until the future operation gate supplies a
// SessionContinuationStateWitness plus capture and restore evidence.
struct CapsuleRestoreStateWitnessAdmissionRecord {
  static constexpr uint32_t kFormatVersion = 2;

  uint32_t format_version = kFormatVersion;
  CapsuleRestoreCoverageKind kind =
      CapsuleRestoreCoverageKind::kStateWitnessedOwnPosition;
  Hash256 record_id;
  CapsuleRestoreStateWitnessOperationalCoverage operational_coverage;
  std::vector<CapsuleRestoreStateWitnessQualificationCaseEvidence>
      qualification_cases;
  int64_t qualified_unix_micros = 0;
  std::string record_authentication_key_id;
};

absl::Status ValidateCapsuleRestoreQualificationSpec(
    const CapsuleRestoreQualificationSpec& spec);

// Hashes the complete resolved profile and capability plus every canonical
// chunk boundary and output limit. Assertion labels are intentionally absent.
absl::StatusOr<Hash256> ComputeCapsuleRestoreQualificationSpecHash(
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const CapsuleRestoreQualificationSpec& spec);

// Composite repository lookup key. Neither a capability nor a request alone
// can select admission evidence for the other.
absl::StatusOr<Hash256> ComputeCapsuleRestoreAdmissionLookupKey(
    const Hash256& capability_id, const Hash256& qualification_spec_hash);

absl::StatusOr<Hash256> ComputeCapsuleRestoreAdmissionRecordId(
    const CapsuleRestoreAdmissionRecord& record);
absl::Status ValidateCapsuleRestoreAdmissionRecord(
    const CapsuleRestoreAdmissionRecord& record);
absl::Status ValidateCapsuleRestoreAdmissionRecordForRuntime(
    const CapsuleRestoreAdmissionRecord& record,
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const CapsuleRestoreQualificationSpec& spec);

// Versioned workload hashes used both to construct Coverage V1 and to match a
// concrete DPM operation before any capsule bytes are imported or exported.
absl::StatusOr<Hash256> ComputeCapsuleRestoreCheckpointWorkloadHash(
    const std::vector<CapsuleRestorePrefillChunk>& chunks,
    uint32_t max_output_tokens);
absl::StatusOr<Hash256> ComputeCapsuleRestoreContinuationWorkloadHash(
    const std::vector<CapsuleRestorePrefillChunk>& chunks,
    uint32_t max_output_tokens);
absl::StatusOr<Hash256> ComputeCapsuleRestoreOperationalCoverageId(
    const CapsuleRestoreOperationalCoverage& coverage);
absl::Status ValidateCapsuleRestoreOperationalCoverage(
    const CapsuleRestoreOperationalCoverage& coverage);
absl::StatusOr<CapsuleRestoreOperationalCoverage>
ComputeCapsuleRestoreOperationalCoverage(
    const CapsuleRestoreAdmissionRecord& record,
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const CapsuleRestoreQualificationSpec& spec);

// Coverage V2 canonicalization and validation. Case IDs, the aggregate
// qualification-evidence hash, coverage ID, and record ID all use disjoint V2
// SHA-256 domains. Callers cannot substitute profile or capability labels:
// ValidateCapsuleRestoreStateWitnessAdmissionRecordForRuntime compares the
// complete authenticated objects with evidence freshly derived by Engine.
absl::Status ValidateCapsuleRestoreStateWitnessQualificationPolicy(
    const CapsuleRestoreStateWitnessQualificationPolicy& policy);
absl::Status ValidateCapsuleRestoreStateWitnessOperationalDomain(
    const CapsuleRestoreStateWitnessOperationalDomain& domain);
absl::StatusOr<Hash256>
ComputeCapsuleRestoreStateWitnessQualificationCaseId(
    const CapsuleRestoreStateWitnessQualificationCaseEvidence& evidence);
absl::Status ValidateCapsuleRestoreStateWitnessQualificationCaseEvidence(
    const CapsuleRestoreStateWitnessQualificationCaseEvidence& evidence,
    const CapsuleRestoreStateWitnessOperationalDomain& domain);
absl::StatusOr<Hash256>
ComputeCapsuleRestoreStateWitnessQualificationEvidenceHash(
    const CapsuleRestoreStateWitnessOperationalDomain& domain,
    const std::vector<CapsuleRestoreStateWitnessQualificationCaseEvidence>&
        qualification_cases);
absl::StatusOr<Hash256>
ComputeCapsuleRestoreStateWitnessOperationalCoverageId(
    const CapsuleRestoreStateWitnessOperationalCoverage& coverage);
absl::Status ValidateCapsuleRestoreStateWitnessOperationalCoverage(
    const CapsuleRestoreStateWitnessOperationalCoverage& coverage);
// Additionally requires every opaque operational contract hash to equal the
// implementation currently linked into this runtime.
absl::Status ValidateCapsuleRestoreStateWitnessOperationalContracts(
    const CapsuleRestoreStateWitnessOperationalCoverage& coverage);
absl::StatusOr<CapsuleRestoreStateWitnessOperationalCoverage>
ComputeCapsuleRestoreStateWitnessOperationalCoverage(
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const CapsuleRestoreStateWitnessOperationalDomain& operational_domain,
    const std::vector<CapsuleRestoreStateWitnessQualificationCaseEvidence>&
        qualification_cases);
absl::StatusOr<Hash256>
ComputeCapsuleRestoreStateWitnessAdmissionRecordId(
    const CapsuleRestoreStateWitnessAdmissionRecord& record);
absl::Status ValidateCapsuleRestoreStateWitnessAdmissionRecord(
    const CapsuleRestoreStateWitnessAdmissionRecord& record);
absl::Status ValidateCapsuleRestoreStateWitnessAdmissionRecordForRuntime(
    const CapsuleRestoreStateWitnessAdmissionRecord& record,
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const Hash256& expected_coverage_id);
absl::StatusOr<Hash256>
ComputeCapsuleRestoreStateWitnessAdmissionLookupKey(
    const Hash256& exact_profile_id, const Hash256& capability_id,
    const Hash256& coverage_id);

// Canonical, bounded admission-record envelope. Its HMAC domain is disjoint
// from LRTSESS1 and every fresh-worker/request/admission protocol domain.
absl::StatusOr<std::string> EncodeCapsuleRestoreAdmissionRecord(
    const CapsuleRestoreAdmissionRecord& record,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<CapsuleRestoreAdmissionRecord>
DecodeCapsuleRestoreAdmissionRecord(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication);

// Coverage V2 uses a distinct DPMCRA02 magic/version and HMAC domain. These
// entry points never dispatch to, or reinterpret, the DPMCRA01 decoder.
absl::StatusOr<std::string>
EncodeCapsuleRestoreStateWitnessAdmissionRecord(
    const CapsuleRestoreStateWitnessAdmissionRecord& record,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<CapsuleRestoreStateWitnessAdmissionRecord>
DecodeCapsuleRestoreStateWitnessAdmissionRecord(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication);

class CapsuleRestoreAdmissionRepository {
 public:
  virtual ~CapsuleRestoreAdmissionRepository() = default;

  // Create-once by capability plus canonical qualification spec. Exact-byte
  // re-publication is idempotent; any conflicting authenticated record fails
  // closed.
  virtual absl::Status PutIfAbsent(
      const CapsuleRestoreAdmissionRecord& record,
      const FreshWorkerAuthentication& authentication) = 0;

  // Every lookup reauthenticates the stored record and revalidates it against
  // the current full runtime-derived profile, capability, and specification.
  virtual absl::StatusOr<CapsuleRestoreAdmissionRecord> Get(
      const ExactLiteRtProfile& runtime_derived_profile,
      const SessionHandoffCapability& runtime_derived_capability,
      const CapsuleRestoreQualificationSpec& spec,
      const FreshWorkerAuthentication& authentication) const = 0;

  // Parallel Coverage V2 store. Defaults remain fail-closed so existing
  // repositories cannot accidentally claim support merely because the base
  // interface gained V2.
  virtual absl::Status PutStateWitnessedOwnPositionIfAbsent(
      const CapsuleRestoreStateWitnessAdmissionRecord&,
      const FreshWorkerAuthentication&) {
    return absl::UnimplementedError(
        "This CapsuleRestore admission repository does not implement "
        "Coverage V2.");
  }

  virtual absl::StatusOr<CapsuleRestoreStateWitnessAdmissionRecord>
  GetStateWitnessedOwnPosition(
      const ExactLiteRtProfile&, const SessionHandoffCapability&,
      const Hash256&, const FreshWorkerAuthentication&) const {
    return absl::UnimplementedError(
        "This CapsuleRestore admission repository does not implement "
        "Coverage V2.");
  }
};

// Immutable authority for the CapsuleRestore guarantee. This binding is
// deliberately independent of replay mode: it can admit own-position capsule
// use for either CanonicalWinnerReplay's live parent session or an
// ExactRegeneration fresh worker. The loaded Engine, not the caller, derives
// the profile, capability, backend, and session identity. Assertions in the
// qualification spec can only reject that runtime-owned evidence.
struct CapsuleRestoreAdmissionBinding {
  const CapsuleRestoreAdmissionRepository* repository = nullptr;
  CapsuleRestoreQualificationSpec qualification_spec;
  FreshWorkerAuthentication record_authentication;
};

// One freshly reauthenticated admission together with the exact runtime
// evidence from which it was selected. Keeping the profile here prevents a
// capability ID from being detached from the concrete SessionConfig that will
// actually create, restore, or capture the session.
struct AuthenticatedCapsuleRestoreAdmission {
  CapsuleRestoreAdmissionRecord record;
  ExactLiteRtProfile profile;
  SessionHandoffCapability capability;
  CapsuleRestoreOperationalCoverage operational_coverage;
};

// Coverage V2 binding. The coverage ID and optional profile/capability fields
// are assertions only: repository lookup and record validation always use the
// complete profile, capability, and session identity freshly derived from the
// loaded Engine and resolved SessionConfig.
struct CapsuleRestoreStateWitnessAdmissionBinding {
  const CapsuleRestoreAdmissionRepository* repository = nullptr;
  Hash256 expected_coverage_id;
  ExactLiteRtProfileAssertion profile_assertion;
  SessionHandoffCapabilityAssertion capability_assertion;
  FreshWorkerAuthentication record_authentication;
};

struct AuthenticatedCapsuleRestoreStateWitnessAdmission {
  CapsuleRestoreStateWitnessAdmissionRecord record;
  ExactLiteRtProfile profile;
  SessionHandoffCapability capability;
  CapsuleRestoreStateWitnessOperationalCoverage operational_coverage;
};

// Resolves `runtime_session_config` and the qualification specification
// independently through the same loaded Engine, requires their complete
// profile/capability/session semantics to agree, and reauthenticates the
// durable admission record. Call this at every support, capability, restore,
// capture, and publication boundary; construction-time success is not a
// permanent admission lease.
absl::StatusOr<AuthenticatedCapsuleRestoreAdmission>
ResolveAuthenticatedCapsuleRestoreAdmission(
    const Engine* engine, const SessionConfig& runtime_session_config,
    const CapsuleRestoreAdmissionBinding& binding);

// Re-resolves all runtime-owned evidence and reauthenticates the create-once
// Coverage V2 record on every operation boundary. No finite qualification case
// is promoted to content authority by this resolver; per-operation witness,
// capture, and restore evidence remain mandatory downstream.
absl::StatusOr<AuthenticatedCapsuleRestoreStateWitnessAdmission>
ResolveAuthenticatedCapsuleRestoreStateWitnessAdmission(
    const Engine* engine, const SessionConfig& runtime_session_config,
    const CapsuleRestoreStateWitnessAdmissionBinding& binding);

class CapsuleRestoreQualifier {
 public:
  explicit CapsuleRestoreQualifier(Engine* authoritative_engine)
      : authoritative_engine_(authoritative_engine) {}

  // Uses exactly one newly-created source and exactly one newly-created
  // target. There is no retry, repair, replay-catalog shortcut, or fallback.
  absl::StatusOr<CapsuleRestoreAdmissionRecord> Qualify(
      const CapsuleRestoreQualificationSpec& spec,
      const SessionHandoffOptions& checkpoint_authentication,
      const FreshWorkerAuthentication& record_authentication) const;

  absl::StatusOr<CapsuleRestoreAdmissionRecord> QualifyAndAdmit(
      const CapsuleRestoreQualificationSpec& spec,
      const SessionHandoffOptions& checkpoint_authentication,
      const FreshWorkerAuthentication& record_authentication,
      CapsuleRestoreAdmissionRepository* repository) const;

 private:
  Engine* authoritative_engine_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_ADMISSION_H_
