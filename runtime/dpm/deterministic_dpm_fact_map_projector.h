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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DETERMINISTIC_DPM_FACT_MAP_PROJECTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DETERMINISTIC_DPM_FACT_MAP_PROJECTOR_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"    // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/canonical_replay_catalog.h"
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_fact_map.h"
#include "runtime/dpm/dpm_replay_executor.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Complete identity of the deterministic structured-memory reducer. The
// implementation artifact hash is supplied by the composition root after it
// measures the executable or library image containing this reducer; it is not
// a caller-selected model identity. The reducer contract and every limit are
// also included in the immutable configuration hash.
struct DeterministicDPMFactMapProjectorConfig {
  DPMFactMapLimits limits;
  Hash256 implementation_artifact_hash;

  // The shared WinnerReplay envelope requires a bounded token field even
  // though this projector performs no inference. Bind it to the product turn
  // limit so projection and decision receipts remain in one request domain.
  uint32_t replay_token_bound = 512;
};

// Stable semantic contract for the pure reducer, including its schema and the
// policy that untyped user/internal/tool events do not mutate structured
// memory. This identifies the algorithm contract, not a particular binary.
Hash256 GetDeterministicDPMFactMapReducerContractHash();

absl::StatusOr<Hash256> ComputeDeterministicDPMFactMapProjectorConfigHash(
    const DeterministicDPMFactMapProjectorConfig& config);

// DPMProjectionProvider backed by the pure dpm.factmap fold. Typed
// dpm.factmap.delta tool events and canonical correction tombstones are the
// only events that mutate memory. Untyped events remain visible to the agent
// through DPMEngine's canonical input but intentionally contribute no inferred
// facts. This keeps structured memory independently reproducible without
// claiming deterministic natural-language extraction.
//
// The catalog authenticates and durably replays the result for the existing
// CanonicalWinnerReplay product contract. It is not needed to make the fold
// deterministic: cold catalogs given the same authoritative prefix, runtime
// identity, implementation artifact, and configuration produce identical
// projected-memory and manifest identities.
class DeterministicDPMFactMapProjector final : public DPMProjectionProvider {
 public:
  static absl::StatusOr<std::unique_ptr<DeterministicDPMFactMapProjector>>
  Create(DPMEventLog* authoritative_log,
         CanonicalWinnerReplayCatalog* replay_catalog,
         SessionHandoffIdentity runtime_identity,
         DeterministicDPMFactMapProjectorConfig config);

  absl::Status ValidateSupport() const override;
  absl::StatusOr<DPMReplayMode> GetReplayMode() const override;
  absl::StatusOr<DPMStageCapabilities> GetCapabilities() const override;
  absl::StatusOr<DPMProjectionOutcome> Project(
      const DPMProjectionRequest& request) override;

  const Hash256& GetConfigHash() const { return config_hash_; }
  const SessionHandoffIdentity& GetRuntimeIdentity() const {
    return runtime_identity_;
  }

 private:
  DeterministicDPMFactMapProjector(
      DPMEventLog* authoritative_log,
      CanonicalWinnerReplayCatalog* replay_catalog,
      SessionHandoffIdentity runtime_identity,
      DeterministicDPMFactMapProjectorConfig config, Hash256 config_hash)
      : authoritative_log_(authoritative_log),
        runtime_identity_(runtime_identity),
        config_(config),
        config_hash_(config_hash),
        replay_executor_(replay_catalog) {}

  absl::StatusOr<DPMLogSnapshot> ResolveAuthoritativeSnapshot(
      const DPMProjectionRequest& request) const;

  DPMEventLog* const authoritative_log_;
  const SessionHandoffIdentity runtime_identity_;
  const DeterministicDPMFactMapProjectorConfig config_;
  const Hash256 config_hash_;
  CanonicalWinnerReplayExecutor replay_executor_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DETERMINISTIC_DPM_FACT_MAP_PROJECTOR_H_
