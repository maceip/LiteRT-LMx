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

#include "runtime/dpm/dpm_projection_runtime.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {
namespace {

bool IsValidUtf8(absl::string_view text) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t lead = bytes[index++];
    if (lead <= 0x7f) continue;

    size_t continuation_count = 0;
    uint8_t minimum_second = 0x80;
    uint8_t maximum_second = 0xbf;
    if (lead >= 0xc2 && lead <= 0xdf) {
      continuation_count = 1;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      continuation_count = 2;
      if (lead == 0xe0) minimum_second = 0xa0;
      if (lead == 0xed) maximum_second = 0x9f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      continuation_count = 3;
      if (lead == 0xf0) minimum_second = 0x90;
      if (lead == 0xf4) maximum_second = 0x8f;
    } else {
      return false;
    }
    if (continuation_count > text.size() - index ||
        bytes[index] < minimum_second || bytes[index] > maximum_second) {
      return false;
    }
    ++index;
    for (size_t offset = 1; offset < continuation_count; ++offset, ++index) {
      if (bytes[index] < 0x80 || bytes[index] > 0xbf) return false;
    }
  }
  return true;
}

absl::Status ValidateRequestedFeatures(const Engine& engine,
                                       const SessionConfig& config,
                                       uint32_t max_output_tokens) {
  switch (config.GetMemoryStrategy()) {
    case SessionConfig::MemoryStrategy::kStateful:
    case SessionConfig::MemoryStrategy::kStatelessDeterministicProjection:
      break;
    default:
      return absl::InvalidArgumentError(
          "DPM projection received an unknown session memory strategy.");
  }
  if (max_output_tokens == 0 ||
      max_output_tokens > kMaximumDPMGenerationTokens) {
    return absl::InvalidArgumentError(
        "DPM projection max output tokens must be between 1 and 65,536.");
  }
  const EngineSettings& engine_settings = engine.GetEngineSettings();
  if (engine_settings.IsBenchmarkEnabled()) {
    return absl::UnimplementedError(
        "DPM projection does not admit benchmark execution paths.");
  }
  if (engine_settings.GetVisionExecutorSettings().has_value() ||
      engine_settings.GetAudioExecutorSettings().has_value() ||
      config.AudioModalityEnabled() || config.VisionModalityEnabled() ||
      config.GetAudioEmbeddingsCallback() != nullptr) {
    return absl::UnimplementedError(
        "DPM projection requires a text-only Engine and session.");
  }
  const LlmExecutorSettings& executor =
      engine_settings.GetMainExecutorSettings();
  if (executor.GetLoraRank() != 0 || config.GetScopedLoraFile() != nullptr ||
      config.GetAudioScopedLoraFile() != nullptr) {
    return absl::UnimplementedError(
        "DPM projection does not support LoRA-enabled execution.");
  }
  if (executor.GetAdvancedSettings().has_value() &&
      executor.GetAdvancedSettings()->enable_speculative_decoding) {
    return absl::UnimplementedError(
        "DPM projection does not support MTP or speculative decode.");
  }
  if (executor.GetAdvancedSettings().has_value() &&
      (executor.GetAdvancedSettings()->is_benchmark ||
       executor.GetAdvancedSettings()->enable_profiling ||
       executor.GetAdvancedSettings()->num_logits_to_print_after_decode !=
           0)) {
    return absl::UnimplementedError(
        "DPM projection does not support benchmark, profiling, or logits "
        "debug execution paths.");
  }
  if (config.UseExternalSampler()) {
    return absl::UnimplementedError(
        "DPM projection does not support an external sampler.");
  }
  if (config.GetNumOutputCandidates() != 1) {
    return absl::InvalidArgumentError(
        "DPM projection requires exactly one output candidate.");
  }
  if (config.GetSuppressTokensConfig().enabled()) {
    return absl::UnimplementedError(
        "DPM projection does not support inherited or requested token "
        "suppression.");
  }
  if (config.GetSamplerBackend() != Backend::UNSPECIFIED &&
      config.GetSamplerBackend() != Backend::CPU) {
    return absl::InvalidArgumentError(
        "DPM projection cannot discard a requested non-CPU sampler backend.");
  }
  const proto::SamplerParameters& sampler = config.GetSamplerParams();
  if (sampler.type() != proto::SamplerParameters::TYPE_UNSPECIFIED &&
      sampler.type() != proto::SamplerParameters::GREEDY) {
    return absl::InvalidArgumentError(
        "DPM projection cannot discard a requested non-GREEDY sampler.");
  }
  if (sampler.backend() != proto::SamplerParameters::UNSPECIFIED &&
      sampler.backend() != proto::SamplerParameters::CPU) {
    return absl::InvalidArgumentError(
        "DPM projection cannot discard a requested non-CPU sampler.");
  }
  const int requested_max_output_tokens = config.GetMaxOutputTokens();
  if (requested_max_output_tokens != std::numeric_limits<int>::max() &&
      requested_max_output_tokens != static_cast<int>(max_output_tokens)) {
    return absl::InvalidArgumentError(
        "DPM projection cannot discard a different explicit session output "
        "token limit.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<EngineDPMProjectionRuntime>>
EngineDPMProjectionRuntime::Create(
    Engine* engine, SessionConfig session_config, uint32_t max_output_tokens,
    std::optional<SessionHandoffIdentity> expected_identity) {
  if (engine == nullptr) {
    return absl::InvalidArgumentError(
        "DPM projection runtime requires a loaded Engine.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateRequestedFeatures(*engine, session_config, max_output_tokens));

  // These are projection protocol invariants. TYPE_UNSPECIFIED may be resolved
  // here, but an explicit incompatible request was rejected above.
  session_config.SetNumOutputCandidates(1);
  session_config.SetSamplerBackend(Backend::CPU);
  session_config.SetApplyPromptTemplateInSession(false);
  session_config.SetMaxOutputTokens(static_cast<int>(max_output_tokens));
  session_config.SetMemoryStrategy(
      SessionConfig::MemoryStrategy::kStatelessDeterministicProjection);
  proto::SamplerParameters& sampler =
      session_config.GetMutableSamplerParams();
  sampler.set_type(proto::SamplerParameters::GREEDY);
  sampler.set_backend(proto::SamplerParameters::CPU);
  sampler.set_k(0);
  sampler.set_p(0.0f);
  sampler.set_temperature(0.0f);
  sampler.clear_seed();
  ABSL_RETURN_IF_ERROR(
      session_config.MaybeUpdateAndValidate(engine->GetEngineSettings()));
  if (session_config.GetMemoryStrategy() !=
      SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
    return absl::FailedPreconditionError(
        "DPM projection failed to pin its stateless memory strategy before "
        "identity derivation.");
  }

  ABSL_ASSIGN_OR_RETURN(
      SessionHandoffIdentity runtime_identity,
      engine->ResolveSessionHandoffIdentity(session_config));
  if (expected_identity.has_value() &&
      *expected_identity != runtime_identity) {
    return absl::FailedPreconditionError(
        "DPM projection identity assertion does not match the loaded Engine.");
  }

  auto runtime = std::unique_ptr<EngineDPMProjectionRuntime>(
      new EngineDPMProjectionRuntime(engine, std::move(session_config),
                                     max_output_tokens, runtime_identity));
  ABSL_RETURN_IF_ERROR(runtime->ValidateSupport());
  return runtime;
}

absl::Status EngineDPMProjectionRuntime::ValidateResolvedConfig(
    const SessionConfig& config) const {
  if (config.AudioModalityEnabled() || config.VisionModalityEnabled() ||
      config.GetAudioEmbeddingsCallback() != nullptr ||
      config.GetScopedLoraFile() != nullptr ||
      config.GetAudioScopedLoraFile() != nullptr ||
      config.UseExternalSampler()) {
    return absl::FailedPreconditionError(
        "Resolved DPM projection session contains an unsupported feature.");
  }
  if (config.GetNumOutputCandidates() != 1 ||
      config.GetSamplerBackend() != Backend::CPU ||
      config.GetMaxOutputTokens() != static_cast<int>(max_output_tokens_) ||
      config.GetSuppressTokensConfig().enabled() ||
      config.GetApplyPromptTemplateInSession() ||
      config.GetMemoryStrategy() !=
          SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
    return absl::FailedPreconditionError(
        "Resolved DPM projection session differs from its canonical profile.");
  }
  const proto::SamplerParameters& sampler = config.GetSamplerParams();
  if (sampler.type() != proto::SamplerParameters::GREEDY ||
      sampler.backend() != proto::SamplerParameters::CPU || sampler.k() != 0 ||
      sampler.p() != 0.0f || sampler.temperature() != 0.0f ||
      sampler.has_seed()) {
    return absl::FailedPreconditionError(
        "Resolved DPM projection session is not canonical CPU GREEDY.");
  }
  return absl::OkStatus();
}

absl::Status EngineDPMProjectionRuntime::ValidateSupport() const {
  if (engine_ == nullptr) {
    return absl::FailedPreconditionError(
        "DPM projection runtime has lost its loaded Engine.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateRequestedFeatures(*engine_, resolved_session_config_,
                                max_output_tokens_));
  ABSL_RETURN_IF_ERROR(ValidateResolvedConfig(resolved_session_config_));
  ABSL_ASSIGN_OR_RETURN(
      SessionHandoffIdentity current_identity,
      engine_->ResolveSessionHandoffIdentity(resolved_session_config_));
  if (current_identity != runtime_identity_) {
    return absl::FailedPreconditionError(
        "Loaded Engine identity changed after DPM projection admission.");
  }
  return absl::OkStatus();
}

absl::Status EngineDPMProjectionRuntime::ValidateSession(
    const Engine::Session& session) const {
  ABSL_RETURN_IF_ERROR(ValidateResolvedConfig(session.GetSessionConfig()));
  ABSL_ASSIGN_OR_RETURN(SessionHandoffIdentity identity,
                        session.GetSessionHandoffIdentity());
  if (identity != runtime_identity_) {
    return absl::FailedPreconditionError(
        "Fresh DPM projection session identity differs from its runtime.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EngineDPMProjectionRuntime::GenerateFresh(
    absl::string_view canonical_prompt) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  if (canonical_prompt.empty() || !IsValidUtf8(canonical_prompt)) {
    return absl::InvalidArgumentError(
        "DPM projection runtime requires a nonempty canonical UTF-8 prompt.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Engine::Session> session,
      engine_->CreateSession(resolved_session_config_));
  if (session == nullptr) {
    return absl::InternalError(
        "Loaded Engine returned a null DPM projection session.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSession(*session));
  // This is only a freshness assertion. Exporting a complete session capsule
  // here would incorrectly make projection reset depend on capsule support
  // (and would reject GPU before its backend-native reset can run). The
  // mandatory kStartProjection prefill below performs the transactional reset.
  ABSL_ASSIGN_OR_RETURN(const int fresh_step, session->GetCurrentStep());
  if (fresh_step != 0) {
    return absl::FailedPreconditionError(
        "DPM projection Engine did not create a step-zero fresh context.");
  }

  ABSL_RETURN_IF_ERROR(session->RunPrefill(
      {InputText(std::string(canonical_prompt))}));
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  decode_config.SetMaxOutputTokens(static_cast<int>(max_output_tokens_));
  ABSL_ASSIGN_OR_RETURN(Responses response,
                        session->RunDecode(decode_config));
  if (response.GetTaskState() != TaskState::kDone &&
      response.GetTaskState() != TaskState::kMaxNumTokensReached) {
    return absl::InternalError(
        "DPM projection decode returned a non-success terminal state.");
  }
  if (response.GetTexts().size() != 1 ||
      !IsValidUtf8(response.GetTexts()[0])) {
    return absl::DataLossError(
        "DPM projection decode did not return one UTF-8 candidate.");
  }
  return response.GetTexts()[0];
}

}  // namespace litert::lm
