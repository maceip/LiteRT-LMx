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

#include "runtime/dpm/engine_fresh_worker_contract.h"

#include <cstdint>
#include <limits>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

absl::StatusOr<SessionConfig> MakeEngineFreshWorkerSessionConfig(
    DPMReplayStage stage, uint32_t max_output_tokens) {
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayStage(stage));
  if (max_output_tokens == 0 ||
      max_output_tokens > kMaximumDPMGenerationTokens ||
      max_output_tokens >
          static_cast<uint32_t>((std::numeric_limits<int>::max)())) {
    return absl::InvalidArgumentError(
        "Exact-worker max output tokens are outside the product limit.");
  }

  SessionConfig config = SessionConfig::CreateDefault();
  config.SetAudioModalityEnabled(false);
  config.SetVisionModalityEnabled(false);
  config.SetUseExternalSampler(false);
  config.SetNumOutputCandidates(1);
  config.SetSamplerBackend(Backend::CPU);
  config.SetApplyPromptTemplateInSession(false);
  config.SetMaxOutputTokens(static_cast<int>(max_output_tokens));
  config.SetMemoryStrategy(
      stage == DPMReplayStage::kProjection
          ? SessionConfig::MemoryStrategy::kStatelessDeterministicProjection
          : SessionConfig::MemoryStrategy::kStateful);
  proto::SamplerParameters& sampler = config.GetMutableSamplerParams();
  sampler.set_type(proto::SamplerParameters::GREEDY);
  sampler.set_backend(proto::SamplerParameters::CPU);
  sampler.set_k(0);
  sampler.set_p(0.0f);
  sampler.set_temperature(0.0f);
  sampler.clear_seed();
  return config;
}

}  // namespace litert::lm
