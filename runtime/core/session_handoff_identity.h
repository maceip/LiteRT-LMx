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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_HANDOFF_IDENTITY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_HANDOFF_IDENTITY_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/platform/hash/hasher.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// Hashes the fully-resolved session configuration and the concrete runtime
// profile captured by the loaded executor. Unsupported capsule profiles fail
// closed before any session state is exported.
absl::StatusOr<Hash256> DeriveSessionHandoffInferenceProfileHash(
    const EngineSettings& engine_settings,
    const SessionConfig& resolved_session_config,
    absl::string_view canonical_runtime_profile,
    support::TokenizerType tokenizer_type);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_HANDOFF_IDENTITY_H_
