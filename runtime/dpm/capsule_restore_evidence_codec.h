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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_CODEC_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_CODEC_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/capsule_restore_evidence.h"
#include "runtime/dpm/fresh_worker_protocol.h"

namespace litert::lm {

// DPMCEV02 is a durable, canonical HMAC-SHA256 envelope for one complete
// Coverage V2 capture-evidence artifact. The bound covers both a descendant's
// exact delta plan and its embedded parent-restore evidence while keeping
// unauthenticated input allocation bounded.
inline constexpr uint32_t
    kAuthenticatedCapsuleCaptureEvidenceV2EnvelopeVersion = 2;
inline constexpr uint64_t
    kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes =
        uint64_t{256} * 1024 * 1024;

// The operation-evidence key is deliberately distinct from the checkpoint
// envelope key named by `evidence`. The secret key is never serialized.
absl::StatusOr<std::string> EncodeAuthenticatedCapsuleCaptureEvidenceV2(
    const CapsuleCaptureEvidenceV2& evidence,
    const FreshWorkerAuthentication& authentication);

// Validates the fixed envelope framing and HMAC before parsing any evidence
// body field. The decoded object is then fully validated and canonically
// re-encoded; alternate encodings and trailing bytes fail closed.
absl::StatusOr<CapsuleCaptureEvidenceV2>
DecodeAuthenticatedCapsuleCaptureEvidenceV2(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication);

// V3 capture and restore records are separate authenticated artifacts. They
// deliberately use distinct framing and HMAC domains from DPMCEV02, so a
// legacy body can never be reinterpreted as explicit reauthentication proof.
inline constexpr uint32_t
    kAuthenticatedCapsuleCaptureEvidenceV3EnvelopeVersion = 3;
inline constexpr uint32_t
    kAuthenticatedCapsuleRestoreEvidenceV3EnvelopeVersion = 3;
inline constexpr uint64_t
    kMaximumAuthenticatedCapsuleCaptureEvidenceV3EnvelopeBytes =
        uint64_t{256} * 1024 * 1024;
inline constexpr uint64_t
    kMaximumAuthenticatedCapsuleRestoreEvidenceV3EnvelopeBytes =
        uint64_t{256} * 1024 * 1024;

absl::StatusOr<std::string> EncodeAuthenticatedCapsuleCaptureEvidenceV3(
    const CapsuleCaptureEvidenceV3& evidence,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<CapsuleCaptureEvidenceV3>
DecodeAuthenticatedCapsuleCaptureEvidenceV3(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication);

absl::StatusOr<std::string> EncodeAuthenticatedCapsuleRestoreEvidenceV3(
    const CapsuleRestoreEvidenceV3& evidence,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<CapsuleRestoreEvidenceV3>
DecodeAuthenticatedCapsuleRestoreEvidenceV3(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication);

// Durable evidence is a disposable authenticated derivative. Implementations
// must use the composite checkpoint/evidence identity and create-once
// publication; callers must already have the exact expected evidence ID from
// an authoritative receipt before Get. Presence in this repository never
// authorizes restore by itself.
class CapsuleRestoreEvidenceRepository {
 public:
  virtual ~CapsuleRestoreEvidenceRepository() = default;

  virtual absl::Status PutIfAbsent(
      const CapsuleCaptureEvidenceV2& evidence,
      const FreshWorkerAuthentication& authentication) = 0;

  virtual absl::StatusOr<CapsuleCaptureEvidenceV2> Get(
      const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
      const FreshWorkerAuthentication& authentication) const = 0;

  // Explicit V3 operations remain named separately from the legacy V2 API.
  // Both record kinds use create-once composite checkpoint/evidence identity;
  // `checkpoint_id` is the captured checkpoint for capture evidence and the
  // consumed source checkpoint for restore evidence.
  virtual absl::Status PutCaptureV3IfAbsent(
      const CapsuleCaptureEvidenceV3& evidence,
      const FreshWorkerAuthentication& authentication) = 0;

  virtual absl::StatusOr<CapsuleCaptureEvidenceV3> GetCaptureV3(
      const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
      const FreshWorkerAuthentication& authentication) const = 0;

  virtual absl::Status PutRestoreV3IfAbsent(
      const CapsuleRestoreEvidenceV3& evidence,
      const FreshWorkerAuthentication& authentication) = 0;

  virtual absl::StatusOr<CapsuleRestoreEvidenceV3> GetRestoreV3(
      const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
      const FreshWorkerAuthentication& authentication) const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_CODEC_H_
