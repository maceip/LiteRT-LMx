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
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kExactLiteRtProfileIdDomain =
    "LITERT_LM_EXACT_PROFILE_V4";
constexpr absl::string_view kStableGreedySamplerContract =
    "LITERT_LM_CPU_GREEDY_ARGMAX_MIN_INDEX_V1";

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

void HashU8(uint8_t value, Sha256Hasher* hasher) {
  const char byte = static_cast<char>(value);
  hasher->Update(absl::string_view(&byte, 1));
}

void HashU32(uint32_t value, Sha256Hasher* hasher) {
  std::array<char, 4> bytes;
  for (size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] =
        static_cast<char>((value >> (24 - index * 8)) & uint32_t{0xff});
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void HashU64(uint64_t value, Sha256Hasher* hasher) {
  std::array<char, 8> bytes;
  for (size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] =
        static_cast<char>((value >> (56 - index * 8)) & uint64_t{0xff});
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void HashI32(int32_t value, Sha256Hasher* hasher) {
  HashU32(std::bit_cast<uint32_t>(value), hasher);
}

void HashFrame(absl::string_view value, Sha256Hasher* hasher) {
  HashU64(value.size(), hasher);
  hasher->Update(value);
}

void HashDigest(const Hash256& value, Sha256Hasher* hasher) {
  hasher->Update(absl::string_view(
      reinterpret_cast<const char*>(value.bytes.data()), value.bytes.size()));
}

absl::Status ValidateLogitsFrameContract(
    const ExactLiteRtLogitsFrameContract& frame) {
  if (frame.batch_size != 1 || frame.sequence_size != 1 ||
      frame.vocabulary_size == 0) {
    return absl::FailedPreconditionError(
        "Exact LiteRT logits require a non-empty batch-one, one-position "
        "frame.");
  }
  uint64_t element_byte_width;
  switch (frame.element_type) {
    case ExactLiteRtLogitsElementType::kFloat16:
      element_byte_width = 2;
      break;
    case ExactLiteRtLogitsElementType::kFloat32:
      element_byte_width = 4;
      break;
    case ExactLiteRtLogitsElementType::kUnsupported:
      return absl::FailedPreconditionError(
          "Exact LiteRT profile has no supported logits element type.");
    default:
      return absl::FailedPreconditionError(
          "Exact LiteRT profile has an unknown logits element type.");
  }
  if (frame.vocabulary_size >
          std::numeric_limits<uint64_t>::max() / element_byte_width ||
      frame.byte_count !=
          static_cast<uint64_t>(frame.vocabulary_size) * element_byte_width) {
    return absl::FailedPreconditionError(
        "Exact LiteRT logits byte count does not cover the complete packed "
        "frame.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Hash256> ComputeExactLiteRtProfileId(
    const ExactLiteRtProfile& profile) {
  Sha256Hasher hasher;
  HashFrame(kExactLiteRtProfileIdDomain, &hasher);
  HashDigest(profile.model_artifact_hash, &hasher);
  HashDigest(profile.tokenizer_contract_hash, &hasher);
  HashDigest(profile.litert_model_bytecode_hash, &hasher);
  HashDigest(profile.runtime_delegate_platform_hash, &hasher);
  HashDigest(profile.loaded_execution_profile_hash, &hasher);
  HashDigest(profile.gpu_execution_policy_hash, &hasher);
  HashDigest(profile.session_identity.model_artifact_hash, &hasher);
  HashDigest(profile.session_identity.runtime_artifact_hash, &hasher);
  HashDigest(profile.session_identity.inference_profile_hash, &hasher);
  HashU8(static_cast<uint8_t>(profile.backend), &hasher);
  HashU32(profile.bound_evidence, &hasher);
  HashU8(static_cast<uint8_t>(profile.qualification_requirement), &hasher);
  HashU8(static_cast<uint8_t>(profile.sampler_identity), &hasher);
  HashU8(static_cast<uint8_t>(profile.logits_frame.element_type), &hasher);
  HashU32(profile.logits_frame.batch_size, &hasher);
  HashU32(profile.logits_frame.sequence_size, &hasher);
  HashU32(profile.logits_frame.vocabulary_size, &hasher);
  HashU64(profile.logits_frame.byte_count, &hasher);
  HashU32(profile.batch_size, &hasher);
  HashU32(profile.cpu_thread_count, &hasher);
  HashI32(profile.prefill_chunk_size, &hasher);
  HashFrame(kStableGreedySamplerContract, &hasher);
  const Hash256 profile_id = hasher.Finalize();
  if (IsZeroHash(profile_id)) {
    return absl::InternalError(
        "Canonical exact LiteRT profile hashing produced a zero digest.");
  }
  return profile_id;
}

absl::Status ValidateExactLiteRtProfile(const ExactLiteRtProfile& profile) {
  if (IsZeroHash(profile.profile_id) ||
      IsZeroHash(profile.model_artifact_hash) ||
      IsZeroHash(profile.tokenizer_contract_hash) ||
      IsZeroHash(profile.litert_model_bytecode_hash) ||
      IsZeroHash(profile.runtime_delegate_platform_hash) ||
      IsZeroHash(profile.loaded_execution_profile_hash) ||
      IsZeroHash(profile.session_identity.model_artifact_hash) ||
      IsZeroHash(profile.session_identity.runtime_artifact_hash) ||
      IsZeroHash(profile.session_identity.inference_profile_hash)) {
    return absl::FailedPreconditionError(
        "Exact LiteRT profile has a zero artifact, execution, session, or "
        "profile identity.");
  }
  if (profile.session_identity.model_artifact_hash !=
          profile.model_artifact_hash ||
      profile.session_identity.runtime_artifact_hash !=
          profile.runtime_delegate_platform_hash) {
    return absl::FailedPreconditionError(
        "Exact LiteRT profile artifact identities disagree with its "
        "Engine-derived session identity.");
  }
  if (profile.qualification_requirement !=
      ExactLiteRtQualificationRequirement::
          kIndependentColdProcessesTokensAndLogits) {
    return absl::FailedPreconditionError(
        "Exact LiteRT profile has an unsupported qualification contract.");
  }
  if (profile.sampler_identity !=
      ExactLiteRtSamplerIdentity::kCpuGreedyArgmaxMinIndex) {
    return absl::FailedPreconditionError(
        "Exact LiteRT profile must use stable CPU GREEDY argmax with the "
        "minimum-index tie rule.");
  }
  if (profile.batch_size != 1 ||
      profile.logits_frame.batch_size != profile.batch_size) {
    return absl::FailedPreconditionError(
        "Exact LiteRT profile requires one request and one logits batch.");
  }
  ABSL_RETURN_IF_ERROR(ValidateLogitsFrameContract(profile.logits_frame));

  uint32_t required_evidence;
  switch (profile.backend) {
    case ExactLiteRtBackend::kCpu:
      required_evidence = ExactLiteRtCpuRequiredEvidenceMask();
      if (!IsZeroHash(profile.gpu_execution_policy_hash)) {
        return absl::FailedPreconditionError(
            "Exact LiteRT CPU profile cannot bind a GPU execution policy.");
      }
      if (profile.cpu_thread_count == 0 ||
          profile.cpu_thread_count >
              static_cast<uint32_t>(std::numeric_limits<int32_t>::max()) ||
          (profile.prefill_chunk_size != -1 &&
           profile.prefill_chunk_size <= 0)) {
        return absl::FailedPreconditionError(
            "Exact LiteRT CPU profile has an invalid executor-derived thread "
            "or prefill contract.");
      }
      break;
    case ExactLiteRtBackend::kMetalGpu:
      required_evidence = ExactLiteRtMetalRequiredEvidenceMask();
      if (IsZeroHash(profile.gpu_execution_policy_hash)) {
        return absl::FailedPreconditionError(
            "Exact LiteRT Metal profile has no concrete GPU execution "
            "policy identity.");
      }
      if (profile.cpu_thread_count != 1 ||
          profile.prefill_chunk_size != -1) {
        return absl::FailedPreconditionError(
            "Exact LiteRT Metal profile requires the serial CPU argmax and "
            "ordered static prefill schedule contracts.");
      }
      break;
    case ExactLiteRtBackend::kNpu:
      return absl::UnimplementedError(
          "Exact LiteRT NPU profile validation is unavailable until its "
          "backend evidence inventory is implemented.");
    default:
      return absl::FailedPreconditionError(
          "Exact LiteRT profile does not name a concrete admitted backend.");
  }
  if (profile.bound_evidence != required_evidence) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Exact LiteRT profile evidence inventory is non-canonical. Missing: ",
        DescribeMissingExactLiteRtEvidence(required_evidence &
                                           ~profile.bound_evidence),
        "; unexpected bits: ", profile.bound_evidence & ~required_evidence,
        "."));
  }

  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_profile_id,
                        ComputeExactLiteRtProfileId(profile));
  if (profile.profile_id != expected_profile_id) {
    return absl::FailedPreconditionError(
        "Exact LiteRT profile identifier does not match its canonical "
        "fields.");
  }
  return absl::OkStatus();
}

absl::Status ValidateExactLiteRtProfileAssertion(
    const ExactLiteRtProfile& derived,
    const ExactLiteRtProfileAssertion& assertion) {
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(derived));
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
