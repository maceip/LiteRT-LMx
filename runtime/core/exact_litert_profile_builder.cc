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

#include "runtime/core/exact_litert_profile_builder.h"

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
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/components/model_resources.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/proto/sampler_params.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kStableGreedySamplerContract =
    "LITERT_LM_CPU_GREEDY_ARGMAX_MIN_INDEX_V1";

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

void HashU8(uint8_t value, Sha256Hasher* hasher) {
  const char byte = static_cast<char>(value);
  hasher->Update(absl::string_view(&byte, 1));
}

void HashU32(uint32_t value, Sha256Hasher* hasher) {
  std::array<char, 4> bytes;
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (24 - i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void HashU64(uint64_t value, Sha256Hasher* hasher) {
  std::array<char, 8> bytes;
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (56 - i * 8)) & 0xff);
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

uint32_t CommonRequiredEvidence() {
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

absl::Status ValidateCpuProfileInputs(
    const EngineSettings& engine_settings,
    const SessionConfig& resolved_session_config,
    const LoadedExactLiteRtEvidence& loaded_exact_evidence,
    SessionHandoffRuntimeClass loaded_runtime_class,
    absl::string_view canonical_loaded_execution_profile,
    const Hash256& model_artifact_hash,
    const Hash256& runtime_delegate_platform_hash,
    const SessionHandoffIdentity& session_identity) {
  if (engine_settings.GetMainExecutorSettings().GetBackend() != Backend::CPU ||
      loaded_runtime_class != SessionHandoffRuntimeClass::kLiteRtCpu) {
    return absl::UnimplementedError(
        "Exact LiteRT CPU profile construction requires an executor-derived "
        "LiteRT CPU runtime class.");
  }
  if (canonical_loaded_execution_profile.empty() ||
      IsZeroHash(model_artifact_hash) ||
      IsZeroHash(loaded_exact_evidence.tokenizer_contract_hash) ||
      IsZeroHash(loaded_exact_evidence.litert_model_bytecode_hash) ||
      IsZeroHash(runtime_delegate_platform_hash) ||
      IsZeroHash(session_identity.inference_profile_hash)) {
    return absl::FailedPreconditionError(
        "Exact LiteRT profile evidence is incomplete.");
  }
  if (session_identity.model_artifact_hash != model_artifact_hash ||
      session_identity.runtime_artifact_hash !=
          runtime_delegate_platform_hash) {
    return absl::FailedPreconditionError(
        "Exact LiteRT profile evidence disagrees with the Engine-derived "
        "session identity.");
  }
  if (engine_settings.GetVisionExecutorSettings().has_value() ||
      engine_settings.GetAudioExecutorSettings().has_value() ||
      resolved_session_config.AudioModalityEnabled() ||
      resolved_session_config.VisionModalityEnabled() ||
      resolved_session_config.GetAudioEmbeddingsCallback() != nullptr) {
    return absl::UnimplementedError(
        "Exact LiteRT profile construction currently supports text-only "
        "engines and sessions.");
  }
  if (resolved_session_config.GetScopedLoraFile() != nullptr ||
      resolved_session_config.GetAudioScopedLoraFile() != nullptr) {
    return absl::UnimplementedError(
        "Exact LiteRT profile construction does not support LoRA state.");
  }
  if (resolved_session_config.UseExternalSampler() ||
      resolved_session_config.GetSamplerBackend() != Backend::CPU ||
      resolved_session_config.GetNumOutputCandidates() != 1 ||
      resolved_session_config.GetSamplerParams().type() !=
          proto::SamplerParameters::GREEDY ||
      resolved_session_config.GetSamplerParams().backend() !=
          proto::SamplerParameters::CPU) {
    return absl::UnimplementedError(
        "Exact LiteRT profiles require one output and the explicit stable CPU "
        "GREEDY sampler.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<LoadedExactLiteRtEvidence>
DeriveLoadedExactLiteRtEvidence(const Hash256& model_artifact_hash,
                                support::Tokenizer& tokenizer,
                                ModelResources& model_resources) {
  if (IsZeroHash(model_artifact_hash)) {
    return absl::FailedPreconditionError(
        "The retained model artifact digest is unavailable.");
  }

  const int vocabulary_size = tokenizer.GetVocabSize();
  if (vocabulary_size <= 0) {
    return absl::FailedPreconditionError(
        "The loaded tokenizer has no positive vocabulary size.");
  }
  if (tokenizer.GetTokenizerType() !=
          support::TokenizerType::kSentencePiece &&
      tokenizer.GetTokenizerType() !=
          support::TokenizerType::kHuggingFace) {
    return absl::UnimplementedError(
        "The loaded tokenizer type has no exact contract implementation.");
  }
  std::vector<std::string> tokens = tokenizer.GetTokens();
  if (tokens.size() != static_cast<size_t>(vocabulary_size)) {
    return absl::FailedPreconditionError(
        "The loaded tokenizer token table does not match its vocabulary "
        "size.");
  }

  Sha256Hasher tokenizer_hasher;
  HashFrame("LITERT_LM_TOKENIZER_CONTRACT_V1", &tokenizer_hasher);
  HashDigest(model_artifact_hash, &tokenizer_hasher);
  HashI32(static_cast<int32_t>(tokenizer.GetTokenizerType()),
          &tokenizer_hasher);
  HashU64(tokens.size(), &tokenizer_hasher);
  for (size_t token_id = 0; token_id < tokens.size(); ++token_id) {
    ABSL_ASSIGN_OR_RETURN(const int resolved_token_id,
                          tokenizer.TokenToId(tokens[token_id]));
    if (resolved_token_id < 0 ||
        static_cast<size_t>(resolved_token_id) != token_id) {
      return absl::FailedPreconditionError(
          "The loaded tokenizer token table is not an authoritative ordered "
          "token-id contract.");
    }
    HashU64(token_id, &tokenizer_hasher);
    HashFrame(tokens[token_id], &tokenizer_hasher);
  }
  const Hash256 tokenizer_contract_hash = tokenizer_hasher.Finalize();

  ABSL_ASSIGN_OR_RETURN(
      absl::string_view litert_model_bytecode,
      model_resources.GetTFLiteModelBuffer(ModelType::kTfLitePrefillDecode));
  if (litert_model_bytecode.empty()) {
    return absl::FailedPreconditionError(
        "The loaded prefill/decode LiteRT model bytecode is empty.");
  }
  Sha256Hasher bytecode_hasher;
  HashFrame("LITERT_LM_PREFILL_DECODE_BYTECODE_V1", &bytecode_hasher);
  HashFrame(litert_model_bytecode, &bytecode_hasher);
  const Hash256 litert_model_bytecode_hash = bytecode_hasher.Finalize();

  if (IsZeroHash(tokenizer_contract_hash) ||
      IsZeroHash(litert_model_bytecode_hash)) {
    return absl::FailedPreconditionError(
        "Loaded exact LiteRT evidence produced a zero digest.");
  }
  return LoadedExactLiteRtEvidence{
      .tokenizer_contract_hash = tokenizer_contract_hash,
      .litert_model_bytecode_hash = litert_model_bytecode_hash,
  };
}

ExactLiteRtProfileCapability DescribeExactLiteRtProfileCapability(
    Backend configured_backend, uint32_t engine_derived_evidence) {
  ExactLiteRtProfileCapability capability;
  capability.engine_derived_evidence = engine_derived_evidence;
  switch (configured_backend) {
    case Backend::CPU: {
      capability.backend = ExactLiteRtBackend::kCpu;
      capability.required_evidence = CommonRequiredEvidence();
      const uint32_t session_scoped =
          ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kStableCpuGreedySampler) |
          ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kSessionIdentity);
      const uint32_t required_engine_evidence =
          capability.required_evidence & ~session_scoped;
      capability.availability =
          (engine_derived_evidence & required_engine_evidence) ==
                  required_engine_evidence
              ? ExactLiteRtProfileAvailability::
                    kCandidateDerivationAvailable
              : ExactLiteRtProfileAvailability::kRuntimeEvidenceUnavailable;
      return capability;
    }
    case Backend::GPU:
      capability.backend = ExactLiteRtBackend::kUnclassifiedGpu;
      capability.required_evidence =
          CommonRequiredEvidence() |
          ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kMetalDeviceAndFamily);
      capability.availability =
          ExactLiteRtProfileAvailability::kMetalEvidenceNotImplemented;
      return capability;
    case Backend::NPU:
      capability.backend = ExactLiteRtBackend::kNpu;
      // NPU runtime/plugin/topology evidence is deliberately not guessed. Its
      // backend-specific evidence set remains unenumerated until a concrete
      // backend is owned; the sentinel can never be bound by a profile.
      capability.required_evidence =
          CommonRequiredEvidence() |
          ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kBackendEvidenceUnenumerated);
      capability.availability =
          ExactLiteRtProfileAvailability::kNpuUnimplemented;
      return capability;
    default:
      capability.backend = ExactLiteRtBackend::kUnsupported;
      capability.required_evidence = 0;
      capability.availability = ExactLiteRtProfileAvailability::kUnsupported;
      return capability;
  }
}

absl::StatusOr<ExactLiteRtProfile> DeriveExactLiteRtCpuProfile(
    const EngineSettings& engine_settings,
    const SessionConfig& resolved_session_config,
    const LoadedExactLiteRtEvidence& loaded_exact_evidence,
    SessionHandoffRuntimeClass loaded_runtime_class,
    absl::string_view canonical_loaded_execution_profile,
    const Hash256& model_artifact_hash,
    const Hash256& runtime_delegate_platform_hash,
    const SessionHandoffIdentity& session_identity) {
  ABSL_RETURN_IF_ERROR(ValidateCpuProfileInputs(
      engine_settings, resolved_session_config, loaded_exact_evidence,
      loaded_runtime_class, canonical_loaded_execution_profile,
      model_artifact_hash, runtime_delegate_platform_hash, session_identity));

  ABSL_ASSIGN_OR_RETURN(
      const CpuConfig cpu_config,
      engine_settings.GetMainExecutorSettings().GetBackendConfig<CpuConfig>());
  if (cpu_config.number_of_threads == 0 ||
      cpu_config.number_of_threads >
          static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
    return absl::FailedPreconditionError(
        "Exact LiteRT CPU profile has an invalid thread count.");
  }
  if (cpu_config.prefill_chunk_size == 0 ||
      cpu_config.prefill_chunk_size < -1) {
    return absl::FailedPreconditionError(
        "Exact LiteRT CPU profile has an invalid prefill chunk size.");
  }

  Sha256Hasher loaded_profile_hasher;
  HashFrame("LITERT_LM_LOADED_EXECUTION_PROFILE_V1", &loaded_profile_hasher);
  HashFrame(canonical_loaded_execution_profile, &loaded_profile_hasher);
  const Hash256 loaded_execution_profile_hash =
      loaded_profile_hasher.Finalize();
  if (IsZeroHash(loaded_execution_profile_hash)) {
    return absl::FailedPreconditionError(
        "Loaded exact LiteRT execution profile produced a zero digest.");
  }

  ExactLiteRtProfile profile{
      .model_artifact_hash = model_artifact_hash,
      .tokenizer_contract_hash =
          loaded_exact_evidence.tokenizer_contract_hash,
      .litert_model_bytecode_hash =
          loaded_exact_evidence.litert_model_bytecode_hash,
      .runtime_delegate_platform_hash = runtime_delegate_platform_hash,
      .loaded_execution_profile_hash = loaded_execution_profile_hash,
      .session_identity = session_identity,
      .backend = ExactLiteRtBackend::kCpu,
      .bound_evidence = CommonRequiredEvidence(),
      .qualification_requirement =
          ExactLiteRtQualificationRequirement::
              kIndependentColdProcessesTokensAndLogits,
      .sampler_identity =
          ExactLiteRtSamplerIdentity::kCpuGreedyArgmaxMinIndex,
      .batch_size = 1,
      .cpu_thread_count = cpu_config.number_of_threads,
      .prefill_chunk_size = cpu_config.prefill_chunk_size,
  };

  Sha256Hasher profile_hasher;
  HashFrame("LITERT_LM_EXACT_PROFILE_V1", &profile_hasher);
  HashU8(static_cast<uint8_t>(profile.backend), &profile_hasher);
  HashU32(profile.bound_evidence, &profile_hasher);
  HashU8(static_cast<uint8_t>(profile.qualification_requirement),
         &profile_hasher);
  HashU8(static_cast<uint8_t>(profile.sampler_identity), &profile_hasher);
  HashDigest(profile.model_artifact_hash, &profile_hasher);
  HashDigest(profile.tokenizer_contract_hash, &profile_hasher);
  HashDigest(profile.litert_model_bytecode_hash, &profile_hasher);
  HashDigest(profile.runtime_delegate_platform_hash, &profile_hasher);
  HashDigest(profile.loaded_execution_profile_hash, &profile_hasher);
  HashDigest(profile.session_identity.model_artifact_hash, &profile_hasher);
  HashDigest(profile.session_identity.runtime_artifact_hash, &profile_hasher);
  HashDigest(profile.session_identity.inference_profile_hash, &profile_hasher);
  HashU32(profile.batch_size, &profile_hasher);
  HashU32(profile.cpu_thread_count, &profile_hasher);
  HashI32(profile.prefill_chunk_size, &profile_hasher);
  HashFrame(kStableGreedySamplerContract, &profile_hasher);
  profile.profile_id = profile_hasher.Finalize();
  if (IsZeroHash(profile.profile_id)) {
    return absl::FailedPreconditionError(
        "Exact LiteRT profile produced a zero identifier.");
  }
  return profile;
}

}  // namespace litert::lm
