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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_DPM_EVENT_LOG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_DPM_EVENT_LOG_H_

#include <filesystem>
#include <memory>
#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/dpm_event_log.h"

namespace litert::lm {

// Durable single-file implementation of the authoritative DPM event log.
//
// Every record contains a format-tagged canonical event/turn-receipt encoding
// and the previous and resulting SHA-256 prefix digests. All readers perform a
// strict complete scan under a shared interprocess lock. An incomplete frame,
// a non-contiguous index, a broken digest link, or non-canonical receipt bytes
// is reported as data loss; this class never truncates or repairs the log.
class FilesystemDPMEventLog final : public DPMEventLog {
 public:
  // `directory` owns exactly one event log. `log_id` is stored inside the
  // canonical log header together with `case_id`, so opening a directory with
  // either different identity fails closed.
  static absl::StatusOr<std::unique_ptr<FilesystemDPMEventLog>> Create(
      std::filesystem::path directory, std::string log_id, std::string case_id);

  absl::StatusOr<DPMLogSnapshot> Snapshot() const override;
  absl::StatusOr<std::unique_ptr<DPMEventLogOperationLease>>
  AcquireOperationLease() override;
  absl::StatusOr<Hash256> PrefixHash(uint64_t event_count) const override;
  absl::StatusOr<DPMAppendResult> AppendIfGeneration(
      DPMEvent event, uint64_t expected_generation) override;

 private:
  FilesystemDPMEventLog(std::filesystem::path directory, std::string log_id,
                        std::string case_id);

  absl::Status Initialize();

  std::filesystem::path directory_;
  std::filesystem::path log_path_;
  std::filesystem::path lock_path_;
  std::filesystem::path operation_lock_path_;
  std::string log_id_;
  std::string case_id_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FILESYSTEM_DPM_EVENT_LOG_H_
