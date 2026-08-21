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
#include "runtime/dpm/dpm_projection_replay_runtime.h"

namespace litert::lm {

// Production projection core. Baseline discovery and persistence deliberately
// remain in the raw log: this class scans authoritative prior receipts in
// descending response-event order, verifies the first compatible artifact,
// and otherwise falls back to event zero. No caller can select a baseline.
class OneShotDPMProjector final : public DPMProjectionProvider {
 public:
  OneShotDPMProjector(DPMEventLog* authoritative_log,
                      DPMProjectionReplayRuntime* runtime,
                      DPMProjectionConfig config);

  absl::Status ValidateSupport() const override;

  // Performs exactly one product replay execution and no repair stage.
  // WinnerReplay may use a prior authenticated winner without inference;
  // ExactRegeneration performs its configured independent cold processes.
  // request.log is only an identity assertion; any baseline is derived solely
  // from a freshly fetched authoritative snapshot.
  absl::StatusOr<DPMProjectionOutcome> Project(
      const DPMProjectionRequest& request) override;

 private:
  absl::StatusOr<DPMLogSnapshot> ResolveAuthoritativeSnapshot(
      const DPMProjectionRequest& request) const;
  absl::StatusOr<std::optional<DPMProjectionBaselineArtifact>>
  SelectNewestCompatibleBaseline(
      const DPMProjectionRequest& authoritative_request,
      const Hash256& correction_digest, const Hash256& config_hash,
      const SessionHandoffIdentity& runtime_identity,
      DPMReplayMode replay_mode,
      const std::optional<Hash256>& exact_profile_id) const;
  absl::Status ValidateBaseline(
      const DPMProjectionRequest& request,
      const Hash256& correction_digest, const Hash256& config_hash,
      const SessionHandoffIdentity& runtime_identity,
      DPMReplayMode replay_mode,
      const std::optional<Hash256>& exact_profile_id,
      const DPMProjectionBaselineArtifact& baseline) const;

  DPMEventLog* const authoritative_log_;
  DPMProjectionReplayRuntime* const runtime_;
  const DPMProjectionConfig config_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ONE_SHOT_DPM_PROJECTOR_H_
