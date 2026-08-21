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

#include "runtime/core/session_handoff_identity.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/proto/sampler_params.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

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

void HashBool(bool value, Sha256Hasher* hasher) {
  HashU8(value ? 1 : 0, hasher);
}

void HashFrame(absl::string_view value, Sha256Hasher* hasher) {
  HashU64(value.size(), hasher);
  hasher->Update(value);
}

absl::Status ValidateSupportedProfile(
    const EngineSettings& engine_settings, const SessionConfig& config,
    absl::string_view canonical_runtime_profile) {
  if (canonical_runtime_profile.empty()) {
    return absl::FailedPreconditionError(
        "Loaded runtime produced no canonical identity profile.");
  }
  if (engine_settings.GetVisionExecutorSettings().has_value() ||
      engine_settings.GetAudioExecutorSettings().has_value()) {
    return absl::UnimplementedError(
        "Session capsule identity currently supports text-only engines.");
  }
  if (config.AudioModalityEnabled() || config.VisionModalityEnabled() ||
      config.GetAudioEmbeddingsCallback() != nullptr) {
    return absl::UnimplementedError(
        "Session capsule identity currently supports text-only sessions.");
  }
  if (config.GetScopedLoraFile() != nullptr ||
      config.GetAudioScopedLoraFile() != nullptr) {
    return absl::UnimplementedError(
        "Session capsule identity does not support LoRA state.");
  }
  if (config.UseExternalSampler()) {
    return absl::UnimplementedError(
        "Session capsule identity does not support an external sampler.");
  }
  if (config.GetSamplerBackend() != Backend::CPU) {
    return absl::UnimplementedError(
        "Session capsule identity currently requires the CPU sampler.");
  }
  if (config.GetNumOutputCandidates() != 1) {
    return absl::UnimplementedError(
        "Session capsule identity requires exactly one output candidate.");
  }
  if (config.GetSamplerParams().type() != proto::SamplerParameters::GREEDY) {
    return absl::UnimplementedError(
        "Session capsule identity currently requires GREEDY sampling.");
  }
  if (config.GetSamplerParams().backend() !=
      proto::SamplerParameters::CPU) {
    return absl::UnimplementedError(
        "Session capsule identity requires an explicit CPU sampler profile.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Hash256> DeriveSessionHandoffInferenceProfileHash(
    const EngineSettings& engine_settings,
    const SessionConfig& resolved_session_config,
    absl::string_view canonical_runtime_profile,
    support::TokenizerType tokenizer_type) {
  absl::Status supported = ValidateSupportedProfile(
      engine_settings, resolved_session_config, canonical_runtime_profile);
  if (!supported.ok()) return supported;

  Sha256Hasher hasher;
  HashFrame("LITERT_LM_SESSION_INFERENCE_PROFILE_V2", &hasher);
  HashFrame(canonical_runtime_profile, &hasher);
  HashI32(static_cast<int32_t>(tokenizer_type), &hasher);
  HashBool(engine_settings.GetParallelFileSectionLoading(), &hasher);
  HashBool(engine_settings.GetSingleThreadedExecution(), &hasher);
  HashBool(engine_settings.IsBenchmarkEnabled(), &hasher);
  HashBool(engine_settings.GetBenchmarkParams().has_value(), &hasher);
  if (engine_settings.GetBenchmarkParams().has_value()) {
    HashFrame(engine_settings.GetBenchmarkParams()->SerializeAsString(),
              &hasher);
  }
  HashBool(engine_settings.GetLlmMetadata().has_value(), &hasher);
  if (engine_settings.GetLlmMetadata().has_value()) {
    HashFrame(engine_settings.GetLlmMetadata()->SerializeAsString(), &hasher);
  }

  HashBool(resolved_session_config.AudioModalityEnabled(), &hasher);
  HashBool(resolved_session_config.VisionModalityEnabled(), &hasher);
  HashI32(static_cast<int32_t>(resolved_session_config.GetMemoryStrategy()),
          &hasher);
  HashFrame(resolved_session_config.GetSamplerParams().SerializeAsString(),
            &hasher);
  HashU32(
      static_cast<uint32_t>(resolved_session_config.GetStopTokenIds().size()),
      &hasher);
  for (const std::vector<int>& stop_sequence :
       resolved_session_config.GetStopTokenIds()) {
    HashU32(static_cast<uint32_t>(stop_sequence.size()), &hasher);
    for (int token_id : stop_sequence) HashI32(token_id, &hasher);
  }
  HashI32(resolved_session_config.GetStartTokenId(), &hasher);
  HashI32(resolved_session_config.GetNumOutputCandidates(), &hasher);
  HashI32(static_cast<int32_t>(resolved_session_config.GetSamplerBackend()),
          &hasher);
  HashFrame(resolved_session_config.GetPromptTemplates().SerializeAsString(),
            &hasher);
  HashFrame(resolved_session_config.GetLlmModelType().SerializeAsString(),
            &hasher);
  std::vector<int> suppressed_tokens(
      resolved_session_config.GetSuppressTokensConfig().suppress_tokens()
          .begin(),
      resolved_session_config.GetSuppressTokensConfig().suppress_tokens()
          .end());
  std::sort(suppressed_tokens.begin(), suppressed_tokens.end());
  HashU32(static_cast<uint32_t>(suppressed_tokens.size()), &hasher);
  for (int token_id : suppressed_tokens) HashI32(token_id, &hasher);
  HashBool(resolved_session_config.GetApplyPromptTemplateInSession(), &hasher);
  HashBool(resolved_session_config.UseExternalSampler(), &hasher);
  HashBool(resolved_session_config.GetScopedLoraFile() != nullptr, &hasher);
  HashBool(resolved_session_config.GetAudioScopedLoraFile() != nullptr,
           &hasher);
  HashI32(resolved_session_config.GetMaxOutputTokens(), &hasher);
  HashBool(resolved_session_config.GetAudioEmbeddingsCallback() != nullptr,
           &hasher);
  return hasher.Finalize();
}

}  // namespace litert::lm
