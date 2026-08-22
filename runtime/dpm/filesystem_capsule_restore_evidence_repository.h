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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_CAPSULE_RESTORE_EVIDENCE_REPOSITORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_CAPSULE_RESTORE_EVIDENCE_REPOSITORY_H_

#include <filesystem>
#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/capsule_restore_evidence_codec.h"

namespace litert::lm {

// Durable owner-only cache of authenticated capture and restore evidence. Every
// record has a create-once composite name containing both checkpoint_id and
// evidence_id. There is intentionally no lookup by checkpoint alone, no
// mutable latest pointer, and no enumeration-as-authority path. A caller must
// already possess the exact expected evidence ID from an authoritative DPM log
// or receipt and must still reauthenticate current runtime admission/capability
// before using a returned artifact.
class FilesystemCapsuleRestoreEvidenceRepository final
    : public CapsuleRestoreEvidenceRepository {
 public:
  static absl::StatusOr<
      std::unique_ptr<FilesystemCapsuleRestoreEvidenceRepository>>
  Create(std::filesystem::path root);

  absl::Status PutCaptureIfAbsent(
      const CapsuleCaptureEvidence& evidence,
      const FreshWorkerAuthentication& authentication) override;

  absl::StatusOr<CapsuleCaptureEvidence> GetCapture(
      const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
      const FreshWorkerAuthentication& authentication) const override;

  absl::Status PutRestoreIfAbsent(
      const CapsuleRestoreEvidence& evidence,
      const FreshWorkerAuthentication& authentication) override;

  absl::StatusOr<CapsuleRestoreEvidence> GetRestore(
      const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
      const FreshWorkerAuthentication& authentication) const override;

 private:
  explicit FilesystemCapsuleRestoreEvidenceRepository(
      std::filesystem::path root);

  absl::Status Initialize();

  std::filesystem::path root_;
  std::filesystem::path product_directory_;
  std::filesystem::path capture_records_directory_;
  std::filesystem::path restore_records_directory_;
  std::filesystem::path lock_path_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_CAPSULE_RESTORE_EVIDENCE_REPOSITORY_H_
