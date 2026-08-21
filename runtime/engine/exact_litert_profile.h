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
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
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
  // The selected ML Drift Metal accelerator is proven by its live
  // Environment handles, Metal-backed executor allocations, and an actual
  // code address owned by the accelerator image. Backend::GPU is not this
  // evidence.
  kSelectedMetalDelegate = uint32_t{1} << 10,
  // The immutable, ordered set of compiled prefill shapes and the exact
  // deterministic decomposition rule are bound by the loaded executor.
  kFixedPrefillSchedule = uint32_t{1} << 11,
  // Decode uses one immutable batch-one shape rather than a runtime-selected
  // shape family.
  kFixedShapeDecode = uint32_t{1} << 12,
  // The selected Metal attention implementation proves that adaptive
  // Split-KV is disabled. Merely finding no LiteRT-LM setting is insufficient.
  kAdaptiveSplitKvDisabled = uint32_t{1} << 13,
  // Every exact GPU prefill/decode is run through an enforced quiescent,
  // request-owned execution boundary.
  kQuiescentGpuExecution = uint32_t{1} << 14,
  // Both backend-native reset and the complete GPU continuation capsule have
  // authoritative inventories. Reset-only support does not bind this bit.
  kCompleteGpuSessionAndResetState = uint32_t{1} << 15,
  // Identity of the selected, effective Metal compute pipelines/kernels after
  // delegate compilation. Delegate library bytes and requested options alone
  // do not prove this (for example, argument-buffer support may be downgraded).
  kSelectedMetalKernelPipeline = uint32_t{1} << 16,
  // Sentinel used only for capability discovery. No profile may bind this
  // value; it means the concrete backend evidence inventory is not complete.
  kBackendEvidenceUnenumerated = uint32_t{1} << 31,
};

constexpr uint32_t ExactLiteRtEvidenceBit(ExactLiteRtEvidence evidence) {
  return static_cast<uint32_t>(evidence);
}

// Canonical evidence inventories. A resolved profile binds exactly the mask
// for its concrete backend: missing bits are unsupported and extra/unknown
// bits cannot be used to manufacture a stronger-looking profile.
constexpr uint32_t ExactLiteRtCommonRequiredEvidenceMask() {
  return ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kModelArtifact) |
         ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kTokenizerContract) |
         ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kLiteRtModelBytecode) |
         ExactLiteRtEvidenceBit(
             ExactLiteRtEvidence::kRuntimeAndDelegateBinary) |
         ExactLiteRtEvidenceBit(
             ExactLiteRtEvidence::kOperatingSystemAndDevice) |
         ExactLiteRtEvidenceBit(
             ExactLiteRtEvidence::kCompilationPrecisionAndQuantization) |
         ExactLiteRtEvidenceBit(
             ExactLiteRtEvidence::kExecutionShapeThreadingAndChunking) |
         ExactLiteRtEvidenceBit(
             ExactLiteRtEvidence::kStableCpuGreedySampler) |
         ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kSessionIdentity);
}

constexpr uint32_t ExactLiteRtCpuRequiredEvidenceMask() {
  return ExactLiteRtCommonRequiredEvidenceMask();
}

constexpr uint32_t ExactLiteRtMetalRequiredEvidenceMask() {
  return ExactLiteRtCommonRequiredEvidenceMask() |
         ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kMetalDeviceAndFamily) |
         ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kSelectedMetalDelegate) |
         ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kFixedPrefillSchedule) |
         ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kFixedShapeDecode) |
         ExactLiteRtEvidenceBit(
             ExactLiteRtEvidence::kAdaptiveSplitKvDisabled) |
         ExactLiteRtEvidenceBit(
             ExactLiteRtEvidence::kQuiescentGpuExecution) |
         ExactLiteRtEvidenceBit(
             ExactLiteRtEvidence::kCompleteGpuSessionAndResetState) |
         ExactLiteRtEvidenceBit(
             ExactLiteRtEvidence::kSelectedMetalKernelPipeline);
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
  // A concrete Metal delegate/device was measured, but one or more mandatory
  // CoRun policy facts are unavailable from the loaded execution path.
  kMetalPolicyEvidenceUnavailable = 5,
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

// Immutable loaded-logits payload contract. Qualification hashes exactly one
// complete packed frame per generated token; it must never accept an arbitrary
// caller-selected byte slice.
enum class ExactLiteRtLogitsElementType : uint8_t {
  kUnsupported = 0,
  kFloat16 = 1,
  kFloat32 = 2,
};

struct ExactLiteRtLogitsFrameContract {
  ExactLiteRtLogitsElementType element_type =
      ExactLiteRtLogitsElementType::kUnsupported;
  uint32_t batch_size = 0;
  uint32_t sequence_size = 0;
  uint32_t vocabulary_size = 0;
  uint64_t byte_count = 0;

  bool operator==(const ExactLiteRtLogitsFrameContract& other) const {
    return element_type == other.element_type &&
           batch_size == other.batch_size &&
           sequence_size == other.sequence_size &&
           vocabulary_size == other.vocabulary_size &&
           byte_count == other.byte_count;
  }
  bool operator!=(const ExactLiteRtLogitsFrameContract& other) const {
    return !(*this == other);
  }
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
  // Always `required_evidence & ~engine_derived_evidence`. Session-scoped
  // evidence remains missing until ResolveExactLiteRtProfile validates the
  // concrete SessionConfig; capability discovery never pretends otherwise.
  uint32_t missing_evidence = 0;
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
  // compiled backend, tensor/logits contract, allocation policy, compilation
  // inputs, cache policy, and resolved executor options. A GPU backend must
  // additionally attest effective precision/quantization and selected pipeline
  // identity before profile construction; requested options are insufficient.
  Hash256 loaded_execution_profile_hash;

  // Non-zero only for a concrete GPU profile. This binds the executor-owned
  // fixed prefill schedule, fixed decode shape, Split-KV policy, quiescent
  // scheduling contract, CPU argmax boundary, and complete state inventory.
  Hash256 gpu_execution_policy_hash;

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

  ExactLiteRtLogitsFrameContract logits_frame;

  // Exact profiles are batch one and use the stable CPU GREEDY sampler whose
  // strict comparison gives an explicit lowest-token-index tie break.
  uint32_t batch_size = 1;
  uint32_t cpu_thread_count = 0;
  // Positive for the loaded dynamic CPU executor. -1 means prefill uses the
  // ordered static signature schedule bound by loaded_execution_profile_hash
  // (CPU) or gpu_execution_policy_hash (Metal), not an unrecorded chunk size.
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
           gpu_execution_policy_hash == other.gpu_execution_policy_hash &&
           session_identity == other.session_identity &&
           backend == other.backend &&
           bound_evidence == other.bound_evidence &&
           qualification_requirement == other.qualification_requirement &&
           sampler_identity == other.sampler_identity &&
           logits_frame == other.logits_frame &&
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

// Computes the sole canonical profile identifier. `profile_id` itself is not
// an input; every other ExactLiteRtProfile field is encoded in a fixed order
// under a versioned SHA-256 domain. This is intentionally separate from
// validation so a builder can compute the identifier before assigning it.
absl::StatusOr<Hash256> ComputeExactLiteRtProfileId(
    const ExactLiteRtProfile& profile);

// Validates the complete CPU or Metal profile contract and recomputes its
// identifier. This must be called whenever a profile crosses a construction,
// admission, repository, or runtime trust boundary; comparing a caller-
// supplied profile_id alone is never sufficient.
absl::Status ValidateExactLiteRtProfile(const ExactLiteRtProfile& profile);

absl::Status ValidateExactLiteRtProfileAssertion(
    const ExactLiteRtProfile& derived,
    const ExactLiteRtProfileAssertion& assertion);

// Stable, human-readable names for missing evidence bits. Unknown bits are
// retained numerically rather than being silently ignored.
std::string DescribeMissingExactLiteRtEvidence(uint32_t missing_evidence);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EXACT_LITERT_PROFILE_H_
