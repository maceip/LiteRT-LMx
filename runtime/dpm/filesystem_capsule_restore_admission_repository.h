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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_CAPSULE_RESTORE_ADMISSION_REPOSITORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_CAPSULE_RESTORE_ADMISSION_REPOSITORY_H_

#include <filesystem>
#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/capsule_restore_admission.h"

namespace litert::lm {

// Durable, owner-only, create-once admission-evidence store. This repository
// never stores or serves LRTSESS1 checkpoint bytes and is not a checkpoint
// repository. Its sole value is an authenticated qualification record keyed
// by the runtime-derived profile, capability, and coverage identity.
class FilesystemCapsuleRestoreAdmissionRepository final
    : public CapsuleRestoreAdmissionRepository {
 public:
  static absl::StatusOr<
      std::unique_ptr<FilesystemCapsuleRestoreAdmissionRepository>>
  Create(std::filesystem::path directory);

  absl::Status PutIfAbsent(
      const CapsuleRestoreAdmissionRecord& record,
      const FreshWorkerAuthentication& authentication) override;

  absl::StatusOr<CapsuleRestoreAdmissionRecord> Get(
      const ExactLiteRtProfile& runtime_derived_profile,
      const SessionHandoffCapability& runtime_derived_capability,
      const Hash256& expected_coverage_id,
      const FreshWorkerAuthentication& authentication) const override;

 private:
  explicit FilesystemCapsuleRestoreAdmissionRepository(
      std::filesystem::path root);
  absl::Status Initialize();

  std::filesystem::path root_;
  std::filesystem::path directory_;
  std::filesystem::path records_directory_;
  std::filesystem::path lock_path_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_CAPSULE_RESTORE_ADMISSION_REPOSITORY_H_
