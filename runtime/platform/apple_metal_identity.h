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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_APPLE_METAL_IDENTITY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_APPLE_METAL_IDENTITY_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {

// Returns a canonical description of the actual MTLDevice selected by the
// loaded Metal accelerator. `metal_command_queue` must be a command queue for
// exactly that device. Raw object addresses are validation handles only and
// are never included in the returned identity.
//
// This is intentionally macOS-only. Other Apple platforms need their own
// exact-profile admission and platform identity contract.
absl::StatusOr<std::string> DeriveMacOsMetalDeviceIdentity(
    const void* metal_device, const void* metal_command_queue);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_APPLE_METAL_IDENTITY_H_
