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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ONE_SHOT_DPM_PROJECTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ONE_SHOT_DPM_PROJECTOR_H_

#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_projection_runtime.h"

namespace litert::lm {

struct OneShotDPMProjectionResult {
  DPMProjectionOutcome outcome;
  DPMProjectionManifest manifest;
  bool used_baseline = false;
};

// Production projection core. Baseline discovery and persistence deliberately
// remain outside this class: prior artifacts are carried by authoritative
// receipts, while this class verifies a selected artifact against the raw log.
// A missing or incompatible disposable baseline falls back to event zero.
class OneShotDPMProjector {
 public:
  OneShotDPMProjector(DPMEventLog* authoritative_log,
                      DPMProjectionRuntime* runtime,
                      DPMProjectionConfig config);

  absl::Status ValidateConfiguration() const;

  // Performs exactly one DPMProjectionRuntime::GenerateFresh call and no
  // repairs. The optional baseline is used only after every manifest, output,
  // prefix, correction, config, and runtime identity check succeeds.
  absl::StatusOr<OneShotDPMProjectionResult> Project(
      const DPMProjectionRequest& request,
      std::optional<DPMProjectionBaselineArtifact> baseline = std::nullopt);

 private:
  absl::Status ValidateCurrentSnapshot(
      const DPMProjectionRequest& request) const;
  absl::Status ValidateBaseline(
      const DPMProjectionRequest& request,
      const Hash256& correction_digest, const Hash256& config_hash,
      const SessionHandoffIdentity& runtime_identity,
      const DPMProjectionBaselineArtifact& baseline) const;

  DPMEventLog* const authoritative_log_;
  DPMProjectionRuntime* const runtime_;
  const DPMProjectionConfig config_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ONE_SHOT_DPM_PROJECTOR_H_
