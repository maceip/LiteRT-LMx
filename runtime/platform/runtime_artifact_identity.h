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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_RUNTIME_ARTIFACT_IDENTITY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_RUNTIME_ARTIFACT_IDENTITY_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/executor/session_handoff_runtime.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Measures the code images and platform evidence that implement `profile` in
// the current process. It hashes loaded memory, not caller-provided filenames
// or version labels. Unsupported delegate/device classes fail closed.
absl::StatusOr<Hash256> MeasureLoadedRuntimeArtifact(
    const SessionHandoffRuntimeProfile& profile);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_RUNTIME_ARTIFACT_IDENTITY_H_
