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
inline constexpr uint32_t kMaximumCapsuleRestoreStateWitnessQualificationCases =
    4096;
inline constexpr uint32_t
    kMaximumCapsuleRestoreStateWitnessQualificationTrials = 64;

// Capsule restore admits one bounded execution-shape domain. Every operation
// remains unauthorized until it supplies a current session-continuation state
// witness and authenticated capture/restore evidence. Qualification cases
// establish the mechanism, never authority over their prompts or other
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

constexpr uint32_t CapsuleRestoreStateWitnessRequiredOperationEvidenceMask() {
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
  static constexpr uint32_t kFormatVersion = 1;

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
// qualification input, never a caller assertion. The current contract is
// CPU-only because it does not yet bind complete backend-native state.
struct CapsuleRestoreStateWitnessOperationalDomain {
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;
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
    return format_version == other.format_version &&
           capture_phase == other.capture_phase &&
           admitted_backend == other.admitted_backend &&
           resolved_session_config_hash == other.resolved_session_config_hash &&
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
           maximum_prefill_text_bytes == other.maximum_prefill_text_bytes &&
           maximum_prefill_token_ids == other.maximum_prefill_token_ids &&
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
// qualify the implementation pathway. Fresh operation evidence is required
// for every real capture and restore.
struct CapsuleRestoreStateWitnessQualificationCaseEvidence {
  static constexpr uint32_t kFormatVersion = 1;

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
           kind == other.kind && shape_class_hash == other.shape_class_hash &&
           trial_identity_hash == other.trial_identity_hash &&
           source_session_instance_hash == other.source_session_instance_hash &&
           target_session_instance_hash == other.target_session_instance_hash &&
           checkpoint_step == other.checkpoint_step &&
           delta_positions == other.delta_positions &&
           prefill_chunk_count == other.prefill_chunk_count &&
           prefill_text_bytes == other.prefill_text_bytes &&
           prefill_token_ids == other.prefill_token_ids &&
           output_tokens == other.output_tokens &&
           observed_encoding_mask == other.observed_encoding_mask &&
           producer_state_witness_hash == other.producer_state_witness_hash &&
           restored_state_witness_hash == other.restored_state_witness_hash &&
           capture_evidence_hash == other.capture_evidence_hash &&
           restore_evidence_hash == other.restore_evidence_hash &&
           live_continuation_output_evidence_hash ==
               other.live_continuation_output_evidence_hash &&
           restored_continuation_output_evidence_hash ==
               other.restored_continuation_output_evidence_hash &&
           verifier_certification_hash == other.verifier_certification_hash;
  }
  bool operator!=(
      const CapsuleRestoreStateWitnessQualificationCaseEvidence& other) const {
    return !(*this == other);
  }
};

// Durable capsule-restore authority. Its canonical preimage binds the complete
// runtime-derived profile, capability, and session evidence, including the
// state inventory and capsule codec hashes carried by `capability`.
struct CapsuleRestoreOperationalCoverage {
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;
  Hash256 coverage_id;
  ExactLiteRtProfile runtime_derived_profile;
  SessionHandoffCapability runtime_derived_capability;
  SessionHandoffIdentity runtime_derived_session_identity;
  CapsuleRestoreStateWitnessOperationalDomain operational_domain;
  Hash256 qualification_evidence_hash;

  bool operator==(const CapsuleRestoreOperationalCoverage& other) const {
    return format_version == other.format_version &&
           coverage_id == other.coverage_id &&
           runtime_derived_profile == other.runtime_derived_profile &&
           runtime_derived_capability == other.runtime_derived_capability &&
           runtime_derived_session_identity ==
               other.runtime_derived_session_identity &&
           operational_domain == other.operational_domain &&
           qualification_evidence_hash == other.qualification_evidence_hash;
  }
  bool operator!=(const CapsuleRestoreOperationalCoverage& other) const {
    return !(*this == other);
  }
};

// Authenticated qualification authority. The cases are
// mechanism evidence and must exactly hash to the coverage's evidence digest.
// Runtime use remains fail-closed until the operation gate supplies a
// SessionContinuationStateWitness plus authenticated capture/restore evidence.
struct CapsuleRestoreAdmissionRecord {
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;
  Hash256 record_id;
  CapsuleRestoreOperationalCoverage operational_coverage;
  std::vector<CapsuleRestoreStateWitnessQualificationCaseEvidence>
      qualification_cases;
  int64_t qualified_unix_micros = 0;
  std::string record_authentication_key_id;
};

// Callers cannot substitute profile or capability labels: runtime validation
// compares complete authenticated objects with evidence freshly derived by
// Engine. Each identity uses a distinct semantic SHA-256 domain.
absl::Status ValidateCapsuleRestoreStateWitnessQualificationPolicy(
    const CapsuleRestoreStateWitnessQualificationPolicy& policy);
absl::Status ValidateCapsuleRestoreStateWitnessOperationalDomain(
    const CapsuleRestoreStateWitnessOperationalDomain& domain);
absl::StatusOr<Hash256> ComputeCapsuleRestoreStateWitnessQualificationCaseId(
    const CapsuleRestoreStateWitnessQualificationCaseEvidence& evidence);
absl::Status ValidateCapsuleRestoreStateWitnessQualificationCaseEvidence(
    const CapsuleRestoreStateWitnessQualificationCaseEvidence& evidence,
    const CapsuleRestoreStateWitnessOperationalDomain& domain);
absl::StatusOr<Hash256>
ComputeCapsuleRestoreStateWitnessQualificationEvidenceHash(
    const CapsuleRestoreStateWitnessOperationalDomain& domain,
    const std::vector<CapsuleRestoreStateWitnessQualificationCaseEvidence>&
        qualification_cases);
absl::StatusOr<Hash256> ComputeCapsuleRestoreOperationalCoverageId(
    const CapsuleRestoreOperationalCoverage& coverage);
absl::Status ValidateCapsuleRestoreOperationalCoverage(
    const CapsuleRestoreOperationalCoverage& coverage);
// Additionally requires every opaque operational contract hash to equal the
// implementation currently linked into this runtime.
absl::Status ValidateCapsuleRestoreOperationalContracts(
    const CapsuleRestoreOperationalCoverage& coverage);
absl::StatusOr<CapsuleRestoreOperationalCoverage>
ComputeCapsuleRestoreOperationalCoverage(
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const CapsuleRestoreStateWitnessOperationalDomain& operational_domain,
    const std::vector<CapsuleRestoreStateWitnessQualificationCaseEvidence>&
        qualification_cases);
absl::StatusOr<Hash256> ComputeCapsuleRestoreAdmissionRecordId(
    const CapsuleRestoreAdmissionRecord& record);
absl::Status ValidateCapsuleRestoreAdmissionRecord(
    const CapsuleRestoreAdmissionRecord& record);
absl::Status ValidateCapsuleRestoreAdmissionRecordForRuntime(
    const CapsuleRestoreAdmissionRecord& record,
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const Hash256& expected_coverage_id);
absl::StatusOr<Hash256> ComputeCapsuleRestoreAdmissionLookupKey(
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

class CapsuleRestoreAdmissionRepository {
 public:
  virtual ~CapsuleRestoreAdmissionRepository() = default;

  // Create-once by profile, capability, and coverage. Exact-byte
  // re-publication is idempotent; any conflicting authenticated record fails
  // closed.
  virtual absl::Status PutIfAbsent(
      const CapsuleRestoreAdmissionRecord& record,
      const FreshWorkerAuthentication& authentication) = 0;

  // Every lookup reauthenticates the stored record and revalidates it against
  // the current full runtime-derived profile, capability, and coverage.
  virtual absl::StatusOr<CapsuleRestoreAdmissionRecord> Get(
      const ExactLiteRtProfile& runtime_derived_profile,
      const SessionHandoffCapability& runtime_derived_capability,
      const Hash256& coverage_id,
      const FreshWorkerAuthentication& authentication) const = 0;
};

// Immutable authority for the CapsuleRestore guarantee. This binding is
// deliberately independent of replay mode: it can admit own-position capsule
// use for either CanonicalWinnerReplay's live parent session or an
// ExactRegeneration fresh worker. The loaded Engine, not the caller, derives
// the profile, capability, backend, and session identity. Assertions in the
// binding assertions can only reject that runtime-owned evidence.
struct CapsuleRestoreAdmissionBinding {
  const CapsuleRestoreAdmissionRepository* repository = nullptr;
  Hash256 expected_coverage_id;
  ExactLiteRtProfileAssertion profile_assertion;
  SessionHandoffCapabilityAssertion capability_assertion;
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

// Resolves complete runtime-owned evidence and reauthenticates the durable
// create-once record at every operation boundary. Construction-time success is
// not a permanent admission lease, and qualification cases never become
// content authority.
absl::StatusOr<AuthenticatedCapsuleRestoreAdmission>
ResolveAuthenticatedCapsuleRestoreAdmission(
    const Engine* engine, const SessionConfig& runtime_session_config,
    const CapsuleRestoreAdmissionBinding& binding);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_ADMISSION_H_
