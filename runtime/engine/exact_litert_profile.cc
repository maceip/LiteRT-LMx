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

#include "runtime/engine/exact_litert_profile.h"

#include <array>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl

namespace litert::lm {

absl::Status ValidateExactLiteRtProfileAssertion(
    const ExactLiteRtProfile& derived,
    const ExactLiteRtProfileAssertion& assertion) {
  if (assertion.expected_profile_id.has_value() &&
      *assertion.expected_profile_id != derived.profile_id) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the Engine-derived exact LiteRT "
        "profile identifier.");
  }
  if (assertion.expected_backend.has_value() &&
      *assertion.expected_backend != derived.backend) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the loaded exact LiteRT backend.");
  }
  if (assertion.expected_session_identity.has_value() &&
      *assertion.expected_session_identity != derived.session_identity) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the Engine-derived session "
        "identity.");
  }
  return absl::OkStatus();
}

std::string DescribeMissingExactLiteRtEvidence(uint32_t missing_evidence) {
  struct NamedEvidence {
    ExactLiteRtEvidence evidence;
    const char* name;
  };
  constexpr std::array<NamedEvidence, 18> kNamedEvidence = {{
      {ExactLiteRtEvidence::kModelArtifact, "model_artifact"},
      {ExactLiteRtEvidence::kTokenizerContract, "tokenizer_contract"},
      {ExactLiteRtEvidence::kLiteRtModelBytecode, "litert_model_bytecode"},
      {ExactLiteRtEvidence::kRuntimeAndDelegateBinary,
       "runtime_and_delegate_binary"},
      {ExactLiteRtEvidence::kOperatingSystemAndDevice,
       "operating_system_and_device"},
      {ExactLiteRtEvidence::kMetalDeviceAndFamily,
       "metal_device_and_family"},
      {ExactLiteRtEvidence::kCompilationPrecisionAndQuantization,
       "compilation_precision_and_quantization"},
      {ExactLiteRtEvidence::kExecutionShapeThreadingAndChunking,
       "execution_shape_threading_and_chunking"},
      {ExactLiteRtEvidence::kStableCpuGreedySampler,
       "stable_cpu_greedy_sampler"},
      {ExactLiteRtEvidence::kSessionIdentity, "session_identity"},
      {ExactLiteRtEvidence::kSelectedMetalDelegate,
       "selected_metal_delegate"},
      {ExactLiteRtEvidence::kFixedPrefillSchedule,
       "fixed_prefill_schedule"},
      {ExactLiteRtEvidence::kFixedShapeDecode, "fixed_shape_decode"},
      {ExactLiteRtEvidence::kAdaptiveSplitKvDisabled,
       "adaptive_split_kv_disabled"},
      {ExactLiteRtEvidence::kQuiescentGpuExecution,
       "quiescent_gpu_execution"},
      {ExactLiteRtEvidence::kCompleteGpuSessionAndResetState,
       "complete_gpu_session_and_reset_state"},
      {ExactLiteRtEvidence::kSelectedMetalKernelPipeline,
       "selected_metal_kernel_pipeline"},
      {ExactLiteRtEvidence::kBackendEvidenceUnenumerated,
       "backend_evidence_unenumerated"},
  }};

  std::vector<std::string> names;
  uint32_t described = 0;
  for (const NamedEvidence& named : kNamedEvidence) {
    const uint32_t bit = ExactLiteRtEvidenceBit(named.evidence);
    if ((missing_evidence & bit) == 0) continue;
    names.emplace_back(named.name);
    described |= bit;
  }
  const uint32_t unknown = missing_evidence & ~described;
  if (unknown != 0) {
    names.push_back(absl::StrCat("unknown_bits_", unknown));
  }
  return absl::StrJoin(names, ", ");
}

}  // namespace litert::lm
