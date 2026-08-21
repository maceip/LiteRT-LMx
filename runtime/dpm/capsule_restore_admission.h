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
