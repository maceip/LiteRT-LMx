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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_FRESH_WORKER_CONTRACT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_FRESH_WORKER_CONTRACT_H_

#include <cstdint>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/engine_settings.h"

namespace litert::lm {

// Product-local identifier for the adapter contract a configured worker is
// expected to implement. The parent commits this value, the canonical
// executable image, argv, and the empty environment contract into one worker
// certification digest. The identifier binds configuration under local
// packaging trust; it is not cryptographic proof that arbitrary executable
// bytes implement this contract. Changing the expected contract identifier
// therefore requires a new admission.
inline constexpr absl::string_view kEngineFreshWorkerAdapterContract =
    "litert-lmx-engine-fresh-worker";

// Constructs the sole SessionConfig admitted by both sides of the concrete
// Engine fresh-worker boundary. Parent-side exact-profile derivation and the
// child-owned Engine session must start from this same stage- and limit-bound
// contract; callers cannot supply an identity-bearing SessionConfig.
absl::StatusOr<SessionConfig> MakeEngineFreshWorkerSessionConfig(
    DPMReplayStage stage, uint32_t max_output_tokens);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_FRESH_WORKER_CONTRACT_H_
