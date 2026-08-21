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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_ENGINE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_capabilities.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/session_checkpoint.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

class DPMAgentReplayRuntime;
struct ExactRegenerationDPMAgentPhysicalExecution;

class DPMClock {
 public:
  virtual ~DPMClock() = default;
  virtual int64_t NowMicros() const = 0;
};

class SystemDPMClock final : public DPMClock {
 public:
  int64_t NowMicros() const override;
};

struct DPMProjectionRequest {
  DPMLogSnapshot log;
  uint64_t input_event_index = 0;
  std::string case_id;
};

struct DPMProjectionOutcome {
  std::string projected_memory;
  // Complete identity of projected_memory. A changed embedded correction
  // digest invalidates a session checkpoint even when model/runtime identity
  // remains compatible.
  DPMProjectionManifest manifest;
};

// Projection is an explicit dependency. Production providers must treat the
// request snapshot as an assertion, refetch the current authoritative prefix,
// and derive any baseline from receipts in that prefix. There is no
// caller-selected projection cache input at this boundary.
class DPMProjectionProvider {
 public:
  virtual ~DPMProjectionProvider() = default;
  // Must validate static configuration and the resolved runtime profile
  // without inference or durable log mutation. Providers that do not expose
  // a preflight fail closed so RunTurn cannot append an input that can never
  // be projected.
  virtual absl::Status ValidateSupport() const {
    return absl::UnimplementedError(
        "DPM projection provider does not support preflight admission.");
  }
  // Returns the provider's immutable named replay guarantee without running
  // projection. DPMEngine requires this to match the agent stage before it
  // appends a recoverable input event; a provider that cannot disclose its
  // mode fails closed through the default implementation.
  virtual absl::StatusOr<DPMReplayMode> GetReplayMode() const {
    absl::StatusOr<DPMStageCapabilities> capabilities = GetCapabilities();
    if (!capabilities.ok()) return capabilities.status();
    return capabilities->replay_mode;
  }
  virtual absl::StatusOr<DPMStageCapabilities> GetCapabilities() const {
    return absl::UnimplementedError(
        "DPM projection provider does not expose capabilities.");
  }
  virtual absl::StatusOr<DPMProjectionOutcome> Project(
      const DPMProjectionRequest& request) = 0;
};

struct DPMAgentGenerationRequest {
  struct PrefillChunk {
    enum class Encoding : uint8_t {
      kUtf8Text = 1,
      kTokenIds = 2,
    };

    Encoding encoding = Encoding::kUtf8Text;
    std::string text;
    std::vector<int> token_ids;
  };

  // Canonical chunks are prefilled in order. Inputs remain exact UTF-8 bytes;
  // historical decisions use the exact executor token-history delta, including
  // stop-sequence tokens that the visible decoder response intentionally
  // filters.
  // When a checkpoint is restored this contains only the post-checkpoint
  // delta; on fallback it contains the full log-derived transcript followed
  // by the current input.
  std::vector<PrefillChunk> canonical_prefill_chunks;
  int max_output_tokens = 512;
};

struct DPMAgentGenerationOutcome {
  std::string decision_output;
  std::vector<int> decision_token_ids;
};

class DPMAgentRuntime {
 public:
  virtual ~DPMAgentRuntime() = default;

  // This identity is owned by the loaded runtime, not supplied by DPM callers.
  virtual const SessionHandoffIdentity& GetSessionHandoffIdentity() const = 0;
  // Read-only generation preflight. This is deliberately separate from
  // capsule support so WinnerReplay capability discovery can revalidate the
  // loaded Engine without requiring handoff support.
  virtual absl::Status ValidateSupport() const {
    return absl::UnimplementedError(
        "DPM agent runtime does not expose generation support preflight.");
  }
  // Must validate the fully resolved loaded session profile without creating
  // durable DPM input. A profile that cannot export and restore a complete
  // capsule fails here rather than trapping a milestone retry after inference.
  virtual absl::Status ValidateSessionHandoffSupport() const {
    return absl::UnimplementedError(
        "DPM agent runtime does not support exact session handoff.");
  }
  virtual absl::StatusOr<std::unique_ptr<Engine::Session>> CreateSession() = 0;
  virtual absl::StatusOr<DPMAgentGenerationOutcome> Generate(
      Engine::Session* session, const DPMAgentGenerationRequest& request) = 0;
};

struct DPMEngineConfig {
  // Zero disables new automatic captures. Restore remains independently
  // controlled so an existing disposable checkpoint cache can still be used.
  // A captured milestone is an own-position prefill boundary: subsequent
  // turns restore it and prefill only the exact response-event suffix. The
  // interval therefore changes inference work, not merely checkpoint names.
  uint32_t checkpoint_interval_turns = 1;
  bool restore_session_checkpoints = true;
  // Optional by default because CanonicalWinnerReplay may recover an
  // authenticated catalog winner that its current backend cannot independently
  // rematerialize byte-for-byte. Setting this true makes that condition and any
  // capture/publication failure abort the turn instead of committing without a
  // disposable checkpoint. It is invalid when automatic capture milestones
  // are disabled.
  bool require_checkpoint_at_milestone = false;
  // Product admission is capped by kMaximumDPMGenerationTokens; this is not
  // an unbounded backend integer knob.
  int max_decision_tokens = 512;
  std::string checkpoint_key_id;
  std::string checkpoint_authentication_key;
};

struct DPMTurnRequest {
  std::string operation_id;
  std::string case_id;
  std::string payload;
  DPMEvent::Kind kind = DPMEvent::Kind::kUser;
  std::optional<int64_t> timestamp_us;
  std::optional<int64_t> response_timestamp_us;
};

struct DPMTurnResult {
  std::string projected_memory;
  std::string decision_output;
  std::vector<int> decision_token_ids;
  uint64_t input_event_index = 0;
  uint64_t response_event_index = 0;
  Hash256 projection_manifest_hash;
  DPMReplayMode projection_replay_mode =
      DPMReplayMode::kCanonicalWinnerReplay;
  Hash256 projection_execution_evidence_hash;
  std::optional<Hash256> projection_exact_profile_id;
  std::optional<Hash256> projection_exact_profile_admission_record_id;
  DPMReplayMode agent_replay_mode =
      DPMReplayMode::kCanonicalWinnerReplay;
  Hash256 agent_execution_evidence_hash;
  std::optional<Hash256> agent_exact_profile_id;
  std::optional<Hash256> agent_exact_profile_admission_record_id;
  std::optional<Hash256> agent_exact_output_evidence_hash;
  uint32_t agent_exact_logit_frame_count = 0;
  bool agent_reused_canonical_winner = false;
  // This remains exclusively a live-parent WinnerReplay statement. Exact
  // worker capsules use the separate authenticated provenance below.
  bool agent_producing_session_matched_output = false;
  bool agent_worker_capsule_matched_output = false;
  std::optional<Hash256> session_checkpoint_id;
  std::optional<Hash256> restored_from_session_checkpoint_id;
  DPMCheckpointCaptureOrigin checkpoint_capture_origin =
      DPMCheckpointCaptureOrigin::kNone;
  DPMCheckpointWorkerPrefillMode agent_worker_prefill_mode =
      DPMCheckpointWorkerPrefillMode::kNone;
  Hash256 agent_physical_execution_plan_hash;
  std::optional<Hash256> agent_capsule_restore_admission_record_id;
  std::optional<DPMExactWorkerCheckpointProvenance>
      agent_exact_worker_checkpoint_provenance;
  bool restored_session_checkpoint = false;
  bool recovered_committed_turn = false;
};

// Corrections are immutable raw-log mutations, not decision-bearing turns.
// Their operation ids are idempotency keys in the same namespace as turn
// operation ids.
struct DPMCorrectionRequest {
  std::string operation_id;
  std::string case_id;
  std::string canonical_payload;
  std::optional<int64_t> timestamp_us;
};

struct DPMCorrectionResult {
  uint64_t correction_event_index = 0;
  Hash256 correction_digest;
  bool recovered_existing_correction = false;
};

// Owns the complete product turn. It is the only layer allowed to select a
// prior descriptor, restore an agent session, capture the producing session,
// and place that descriptor in the authoritative model-turn receipt.
class DPMEngine {
 public:
  DPMEngine(DPMEventLog* log, DPMProjectionProvider* projection_provider,
            DPMAgentReplayRuntime* agent_runtime,
            DPMSessionCheckpointRepository* checkpoint_repository,
            DPMEngineConfig config, DPMClock* clock = nullptr);

  absl::Status ValidateConfiguration() const;
  // Read-only, runtime-derived product contract. This reauthenticates current
  // exact/CapsuleRestore admission and never exposes checkpoint keys or lets a
  // caller select a derivative cache object.
  absl::StatusOr<DPMCapabilities> GetCapabilities() const;
  absl::StatusOr<DPMTurnResult> RunTurn(const DPMTurnRequest& request);
  absl::StatusOr<DPMCorrectionResult> AppendCorrection(
      const DPMCorrectionRequest& request);

 private:
  struct RestoreCandidate {
    DPMTurnReceipt receipt;
    DPMSessionCheckpointArtifact artifact;
  };

  absl::StatusOr<std::optional<DPMTurnResult>> RecoverCommittedTurn(
      const DPMLogSnapshot& snapshot, const DPMTurnRequest& request) const;
  absl::StatusOr<std::optional<RestoreCandidate>> FindRestoreCandidate(
      const DPMLogSnapshot& current,
      const DPMProjectionOutcome& projection) const;
  absl::Status ValidateRestoreCandidate(
      const DPMLogSnapshot& current,
      const DPMProjectionOutcome& projection,
      const RestoreCandidate& candidate) const;
  absl::StatusOr<std::vector<DPMAgentGenerationRequest::PrefillChunk>>
  BuildFullTranscriptChunks(
      const DPMLogSnapshot& snapshot,
      absl::string_view current_agent_input,
      const Hash256& current_correction_digest) const;
  absl::StatusOr<std::vector<DPMAgentGenerationRequest::PrefillChunk>>
  BuildDeltaTranscriptChunks(
      const DPMLogSnapshot& snapshot, uint64_t restored_response_event_index,
      absl::string_view current_agent_input,
      const Hash256& current_correction_digest) const;
  absl::StatusOr<std::string> BuildCanonicalAgentInput(
      const DPMTurnRequest& request, const DPMLogSnapshot& source_snapshot,
      uint64_t input_event_index,
      const DPMProjectionOutcome& projection) const;
  absl::StatusOr<Hash256> ComputeTranscriptHash(
      const DPMLogSnapshot& snapshot, absl::string_view current_agent_input,
      absl::string_view current_decision,
      const std::vector<int>& current_decision_token_ids,
      const Hash256& current_correction_digest) const;
  absl::StatusOr<DPMSessionCheckpointArtifact> CaptureProducingSession(
      Engine::Session* session, const DPMLogSnapshot& source_snapshot,
      uint64_t response_event_index,
      const DPMProjectionOutcome& projection,
      const Hash256& agent_request_hash,
      const Hash256& transcript_hash,
      const std::optional<Hash256>& restored_from_checkpoint_id,
      int64_t created_unix_micros) const;
  absl::StatusOr<DPMSessionCheckpointArtifact>
  BuildExactWorkerCheckpointArtifact(
      absl::string_view authenticated_envelope,
      const ExactRegenerationDPMAgentPhysicalExecution& physical_execution,
      const DPMLogSnapshot& source_snapshot, uint64_t response_event_index,
      const DPMProjectionOutcome& projection,
      const Hash256& agent_request_hash, const Hash256& transcript_hash,
      const Hash256& capsule_restore_admission_record_id,
      int64_t created_unix_micros) const;

  DPMEventLog* log_;
  DPMProjectionProvider* projection_provider_;
  DPMAgentReplayRuntime* agent_runtime_;
  DPMSessionCheckpointRepository* checkpoint_repository_;
  DPMEngineConfig config_;
  DPMClock* clock_;
  SystemDPMClock system_clock_;
  absl::Mutex turn_mutex_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_ENGINE_H_
