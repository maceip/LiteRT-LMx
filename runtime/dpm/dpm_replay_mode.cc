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

#include "runtime/dpm/dpm_replay_mode.h"

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

absl::string_view DPMReplayModeToString(DPMReplayMode mode) {
  switch (mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      return "canonical-winner-replay";
    case DPMReplayMode::kExactRegeneration:
      return "exact-regeneration";
  }
  return "unknown";
}

absl::Status ValidateDPMReplayMode(DPMReplayMode mode) {
  switch (mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
    case DPMReplayMode::kExactRegeneration:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError("Unknown DPM replay mode.");
}

absl::string_view DPMReplayStageToString(DPMReplayStage stage) {
  switch (stage) {
    case DPMReplayStage::kProjection:
      return "projection";
    case DPMReplayStage::kAgentDecision:
      return "agent-decision";
  }
  return "unknown";
}

absl::Status ValidateDPMReplayStage(DPMReplayStage stage) {
  switch (stage) {
    case DPMReplayStage::kProjection:
    case DPMReplayStage::kAgentDecision:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError("Unknown DPM replay stage.");
}

}  // namespace litert::lm
