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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_RUNTIME_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_RUNTIME_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/session_handoff.h"

namespace litert::lm {

// Narrow inference boundary for projection. One invocation must create a new
// context and perform exactly one decode. A caller cannot request sampling or
// a caller-labelled model identity through this surface.
class DPMProjectionRuntime {
 public:
  virtual ~DPMProjectionRuntime() = default;

  virtual const SessionHandoffIdentity& GetRuntimeIdentity() const = 0;
  virtual uint32_t GetMaxOutputTokens() const = 0;
  virtual absl::Status ValidateSupport() const = 0;
  virtual absl::StatusOr<std::string> GenerateFresh(
      absl::string_view canonical_prompt) = 0;
};

// Fresh-session production adapter over a loaded LiteRT-LM Engine. It pins the
// actual GREEDY sampler implementation on CPU, which uses stable argmax and
// the lowest vocabulary index for an exact tie. The model/delegate profile is
// derived by the loaded Engine and may only be asserted by the caller.
class EngineDPMProjectionRuntime final : public DPMProjectionRuntime {
 public:
  static absl::StatusOr<std::unique_ptr<EngineDPMProjectionRuntime>> Create(
      Engine* engine, SessionConfig session_config, uint32_t max_output_tokens,
      std::optional<SessionHandoffIdentity> expected_identity = std::nullopt);

  const SessionHandoffIdentity& GetRuntimeIdentity() const override {
    return runtime_identity_;
  }
  uint32_t GetMaxOutputTokens() const override { return max_output_tokens_; }
  absl::Status ValidateSupport() const override;
  absl::StatusOr<std::string> GenerateFresh(
      absl::string_view canonical_prompt) override;

 private:
  EngineDPMProjectionRuntime(Engine* engine,
                             SessionConfig resolved_session_config,
                             uint32_t max_output_tokens,
                             SessionHandoffIdentity runtime_identity)
      : engine_(engine),
        resolved_session_config_(std::move(resolved_session_config)),
        max_output_tokens_(max_output_tokens),
        runtime_identity_(runtime_identity) {}

  absl::Status ValidateResolvedConfig(const SessionConfig& config) const;
  absl::Status ValidateSession(const Engine::Session& session) const;

  Engine* const engine_;
  const SessionConfig resolved_session_config_;
  const uint32_t max_output_tokens_;
  const SessionHandoffIdentity runtime_identity_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_RUNTIME_H_
