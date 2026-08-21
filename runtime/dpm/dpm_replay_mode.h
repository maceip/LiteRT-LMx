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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_REPLAY_MODE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_REPLAY_MODE_H_

#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// The two replay products intentionally have disjoint authority paths.
//
// kCanonicalWinnerReplay means returning an authenticated output that was
// captured once and selected by a create-once catalog. It can operate without
// an exact-inference admission, but it is replay of a prior winner, not
// independent regeneration.
//
// kExactRegeneration means execution in a fresh worker under a currently
// admitted ExactLiteRtProfile. It must never read from or publish to a
// canonical-winner catalog. Its evidence is the independently generated
// tokens and logits. The enum names the requested contract; admission and
// dispatch code must still enforce it at every concrete dependency boundary.
enum class DPMReplayMode : uint8_t {
  kCanonicalWinnerReplay = 1,
  kExactRegeneration = 2,
};

absl::string_view DPMReplayModeToString(DPMReplayMode mode);
// Validates only that `mode` is a known wire/API value. It is not profile
// admission and does not establish that exact regeneration is available.
absl::Status ValidateDPMReplayMode(DPMReplayMode mode);

// Projection and agent decisions occupy distinct request namespaces even when
// their canonical request bytes happen to be identical.
enum class DPMReplayStage : uint8_t {
  kProjection = 1,
  kAgentDecision = 2,
};

absl::string_view DPMReplayStageToString(DPMReplayStage stage);
absl::Status ValidateDPMReplayStage(DPMReplayStage stage);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_REPLAY_MODE_H_
