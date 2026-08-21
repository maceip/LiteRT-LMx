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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EXACT_LITERT_PROFILE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EXACT_LITERT_PROFILE_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Concrete LiteRT execution class. This is derived from the loaded executor;
// it is never accepted as a caller label.
enum class ExactLiteRtBackend : uint8_t {
  kUnsupported = 0,
  kCpu = 1,
  // Backend::GPU is not enough to identify Metal: LiteRT may resolve a
  // different concrete delegate. This value is discovery-only and can never
  // appear in a resolved ExactLiteRtProfile.
  kUnclassifiedGpu = 2,
  kMetalGpu = 3,
  kNpu = 4,
};

// Evidence classes that must be bound into an ExactLiteRtProfile before it can
// be presented to an independent cold-process qualification system.
enum class ExactLiteRtEvidence : uint32_t {
  kModelArtifact = uint32_t{1} << 0,
  kTokenizerContract = uint32_t{1} << 1,
  kLiteRtModelBytecode = uint32_t{1} << 2,
  kRuntimeAndDelegateBinary = uint32_t{1} << 3,
  kOperatingSystemAndDevice = uint32_t{1} << 4,
  kMetalDeviceAndFamily = uint32_t{1} << 5,
  kCompilationPrecisionAndQuantization = uint32_t{1} << 6,
  kExecutionShapeThreadingAndChunking = uint32_t{1} << 7,
  kStableCpuGreedySampler = uint32_t{1} << 8,
  kSessionIdentity = uint32_t{1} << 9,
  // Sentinel used only for capability discovery. No profile may bind this
  // value; it means the concrete backend evidence inventory is not complete.
  kBackendEvidenceUnenumerated = uint32_t{1} << 31,
};

constexpr uint32_t ExactLiteRtEvidenceBit(ExactLiteRtEvidence evidence) {
  return static_cast<uint32_t>(evidence);
}

// Engine-scoped discovery result. "Candidate" means only that this Engine can
// derive a complete profile for a compatible SessionConfig. It never means
// that independent regeneration has been demonstrated.
enum class ExactLiteRtProfileAvailability : uint8_t {
  kUnsupported = 0,
  kCandidateDerivationAvailable = 1,
  kRuntimeEvidenceUnavailable = 2,
  kMetalEvidenceNotImplemented = 3,
  kNpuUnimplemented = 4,
};

enum class ExactLiteRtQualificationRequirement : uint8_t {
  // An admission record must compare output token bytes and SHA-256 logits
  // digests from multiple independent cold worker processes with empty replay
  // catalogs. The profile itself can never satisfy this requirement.
  kIndependentColdProcessesTokensAndLogits = 1,
};

enum class ExactLiteRtSamplerIdentity : uint8_t {
  kCpuGreedyArgmaxMinIndex = 1,
};

struct ExactLiteRtProfileCapability {
  ExactLiteRtBackend backend = ExactLiteRtBackend::kUnsupported;
  ExactLiteRtProfileAvailability availability =
      ExactLiteRtProfileAvailability::kUnsupported;

  // Evidence required for the backend and the engine-scoped subset already
  // derived from the loaded Engine. Sampler and session evidence are added
  // only while resolving a concrete SessionConfig.
  uint32_t required_evidence = 0;
  uint32_t engine_derived_evidence = 0;
  ExactLiteRtQualificationRequirement qualification_requirement =
      ExactLiteRtQualificationRequirement::
          kIndependentColdProcessesTokensAndLogits;
};

// ExactRegeneration admission is intentionally not represented by this type.
// A profile is only a fully-derived identity candidate. A separate admission
// record must bind `profile_id` to repeated, independent cold-process token
// bytes and logits digests before ExactRegeneration may be claimed.
struct ExactLiteRtProfile {
  // SHA-256 of the canonical fields below.
  Hash256 profile_id;

  // Exact retained .litertlm/.task container, ordered tokenizer contract, and
  // exact embedded LiteRT prefill/decode model bytes used as compilation
  // input. External weights remain covered by model_artifact_hash.
  Hash256 model_artifact_hash;
  Hash256 tokenizer_contract_hash;
  Hash256 litert_model_bytecode_hash;

  // This digest is measured from loaded code images, including the actual
  // LiteRT/delegate/plugin code in the current process, together with
  // platform identity. CPU profiles include the OS build and hardware device.
  // A Metal profile cannot be constructed until the selected MTLDevice and
  // supported Metal families are measured as well.
  Hash256 runtime_delegate_platform_hash;

  // Hash of the executor-owned canonical runtime profile. It binds concrete
  // compiled backend, tensor/logits contract, allocation policy, compilation,
  // activation/precision/quantization flags, cache policy, and resolved
  // executor options.
  Hash256 loaded_execution_profile_hash;

  // Identity derived by the loaded Engine for this fully-resolved session.
  // It includes the model, measured runtime, and resolved session profile.
  SessionHandoffIdentity session_identity;

  ExactLiteRtBackend backend = ExactLiteRtBackend::kUnsupported;

  uint32_t bound_evidence = 0;
  ExactLiteRtQualificationRequirement qualification_requirement =
      ExactLiteRtQualificationRequirement::
          kIndependentColdProcessesTokensAndLogits;
  ExactLiteRtSamplerIdentity sampler_identity =
      ExactLiteRtSamplerIdentity::kCpuGreedyArgmaxMinIndex;

  // Exact profiles are batch one and use the stable CPU GREEDY sampler whose
  // strict comparison gives an explicit lowest-token-index tie break.
  uint32_t batch_size = 1;
  uint32_t cpu_thread_count = 0;
  int32_t prefill_chunk_size = 0;

  bool operator==(const ExactLiteRtProfile& other) const {
    return profile_id == other.profile_id &&
           model_artifact_hash == other.model_artifact_hash &&
           tokenizer_contract_hash == other.tokenizer_contract_hash &&
           litert_model_bytecode_hash == other.litert_model_bytecode_hash &&
           runtime_delegate_platform_hash ==
               other.runtime_delegate_platform_hash &&
           loaded_execution_profile_hash ==
               other.loaded_execution_profile_hash &&
           session_identity == other.session_identity &&
           backend == other.backend &&
           bound_evidence == other.bound_evidence &&
           qualification_requirement == other.qualification_requirement &&
           sampler_identity == other.sampler_identity &&
           batch_size == other.batch_size &&
           cpu_thread_count == other.cpu_thread_count &&
           prefill_chunk_size == other.prefill_chunk_size;
  }
  bool operator!=(const ExactLiteRtProfile& other) const {
    return !(*this == other);
  }
};

// Optional caller assertions are comparisons against a profile already
// derived by the loaded Engine. They cannot choose or override any identity
// field.
struct ExactLiteRtProfileAssertion {
  std::optional<Hash256> expected_profile_id;
  std::optional<ExactLiteRtBackend> expected_backend;
  std::optional<SessionHandoffIdentity> expected_session_identity;
};

absl::Status ValidateExactLiteRtProfileAssertion(
    const ExactLiteRtProfile& derived,
    const ExactLiteRtProfileAssertion& assertion);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EXACT_LITERT_PROFILE_H_
