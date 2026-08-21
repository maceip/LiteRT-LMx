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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_SESSION_CHECKPOINT_REPOSITORY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_SESSION_CHECKPOINT_REPOSITORY_H_

#include <filesystem>
#include <memory>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/session_checkpoint.h"

namespace litert::lm {

// Content-addressed local repository for disposable authenticated session
// checkpoints. Envelopes and descriptors have separate create-once names. Put
// durably publishes the envelope before publishing the descriptor; Get always
// revalidates both content addresses. There is intentionally no mutable
// "latest" entry: only an ID committed to DPMEventLog grants authority to a
// descriptor.
class FilesystemDPMSessionCheckpointRepository final
    : public DPMSessionCheckpointRepository {
 public:
  static absl::StatusOr<
      std::unique_ptr<FilesystemDPMSessionCheckpointRepository>>
  Create(std::filesystem::path directory);

  absl::Status Put(
      const DPMSessionCheckpointArtifact& artifact) override;
  absl::StatusOr<DPMSessionCheckpointArtifact> Get(
      const Hash256& descriptor_id) const override;

 private:
  explicit FilesystemDPMSessionCheckpointRepository(
      std::filesystem::path directory);

  absl::Status Initialize();

  std::filesystem::path directory_;
  std::filesystem::path envelope_directory_;
  std::filesystem::path descriptor_directory_;
  std::filesystem::path lock_path_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_SESSION_CHECKPOINT_REPOSITORY_H_
