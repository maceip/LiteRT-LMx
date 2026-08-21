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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_EXACT_LITERT_PROFILE_BUILDER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_EXACT_LITERT_PROFILE_BUILDER_H_

#include <cstdint>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/executor/session_handoff_runtime.h"
#include "runtime/platform/hash/hasher.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// Engine-internal evidence captured after model compilation and tokenizer
// loading. Failure to derive it must disable only exact-profile resolution,
// not ordinary inference.
struct LoadedExactLiteRtEvidence {
  Hash256 tokenizer_contract_hash;
  Hash256 litert_model_bytecode_hash;
};

absl::StatusOr<LoadedExactLiteRtEvidence>
DeriveLoadedExactLiteRtEvidence(const Hash256& model_artifact_hash,
                                support::Tokenizer& tokenizer,
                                ModelResources& model_resources);

// Describes which engine-scoped evidence is present without treating a
// configured GPU label as proof of the selected delegate/device.
ExactLiteRtProfileCapability DescribeExactLiteRtProfileCapability(
    Backend configured_backend, uint32_t engine_derived_evidence);

// Constructs a CPU identity candidate from loaded evidence. This never
// performs or records ExactRegeneration admission.
absl::StatusOr<ExactLiteRtProfile> DeriveExactLiteRtCpuProfile(
    const EngineSettings& engine_settings,
    const SessionConfig& resolved_session_config,
    const LoadedExactLiteRtEvidence& loaded_exact_evidence,
    SessionHandoffRuntimeClass loaded_runtime_class,
    absl::string_view canonical_loaded_execution_profile,
    const Hash256& model_artifact_hash,
    const Hash256& runtime_delegate_platform_hash,
    const SessionHandoffIdentity& session_identity);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_EXACT_LITERT_PROFILE_BUILDER_H_
