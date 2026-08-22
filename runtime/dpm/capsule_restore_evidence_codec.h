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

// Capture and restore records are separate authenticated artifacts with
// distinct framing and HMAC domains. This is the sole development format;
// earlier unshipped evidence envelopes are intentionally unsupported.
inline constexpr uint32_t
    kAuthenticatedCapsuleCaptureEvidenceEnvelopeFormatVersion = 1;
inline constexpr uint32_t
    kAuthenticatedCapsuleRestoreEvidenceEnvelopeFormatVersion = 1;
inline constexpr uint64_t
    kMaximumAuthenticatedCapsuleCaptureEvidenceEnvelopeBytes =
        uint64_t{256} * 1024 * 1024;
inline constexpr uint64_t
    kMaximumAuthenticatedCapsuleRestoreEvidenceEnvelopeBytes =
        uint64_t{256} * 1024 * 1024;

absl::StatusOr<std::string> EncodeAuthenticatedCapsuleCaptureEvidence(
    const CapsuleCaptureEvidence& evidence,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<CapsuleCaptureEvidence>
DecodeAuthenticatedCapsuleCaptureEvidence(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication);

absl::StatusOr<std::string> EncodeAuthenticatedCapsuleRestoreEvidence(
    const CapsuleRestoreEvidence& evidence,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<CapsuleRestoreEvidence>
DecodeAuthenticatedCapsuleRestoreEvidence(
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

  // Both record kinds use create-once composite checkpoint/evidence identity;
  // `checkpoint_id` is the captured checkpoint for capture evidence and the
  // consumed source checkpoint for restore evidence.
  virtual absl::Status PutCaptureIfAbsent(
      const CapsuleCaptureEvidence& evidence,
      const FreshWorkerAuthentication& authentication) = 0;

  virtual absl::StatusOr<CapsuleCaptureEvidence> GetCapture(
      const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
      const FreshWorkerAuthentication& authentication) const = 0;

  virtual absl::Status PutRestoreIfAbsent(
      const CapsuleRestoreEvidence& evidence,
      const FreshWorkerAuthentication& authentication) = 0;

  virtual absl::StatusOr<CapsuleRestoreEvidence> GetRestore(
      const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
      const FreshWorkerAuthentication& authentication) const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_CODEC_H_
