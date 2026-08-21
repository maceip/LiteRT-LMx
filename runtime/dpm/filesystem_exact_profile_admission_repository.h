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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_EXACT_PROFILE_ADMISSION_REPOSITORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_EXACT_PROFILE_ADMISSION_REPOSITORY_H_

#include <filesystem>
#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/exact_profile_admission.h"

namespace litert::lm {

// Durable create-once store keyed only by the complete derived exact-profile
// digest. Every read authenticates and structurally validates the whole record;
// a missing, truncated, conflicting, or wrong-key record fails closed.
class FilesystemExactProfileAdmissionRepository final
    : public ExactProfileAdmissionRepository {
 public:
  static absl::StatusOr<
      std::unique_ptr<FilesystemExactProfileAdmissionRepository>>
  Create(std::filesystem::path directory);

  absl::Status PutIfAbsent(
      const ExactProfileAdmissionRecord& record,
      const FreshWorkerAuthentication& authentication) override;
  absl::StatusOr<ExactProfileAdmissionRecord> Get(
      const ExactLiteRtProfile& runtime_derived_profile,
      const FreshWorkerAuthentication& authentication) const override;

 private:
  explicit FilesystemExactProfileAdmissionRepository(
      std::filesystem::path directory);
  absl::Status Initialize();

  std::filesystem::path root_;
  std::filesystem::path directory_;
  std::filesystem::path records_directory_;
  std::filesystem::path lock_path_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_EXACT_PROFILE_ADMISSION_REPOSITORY_H_
