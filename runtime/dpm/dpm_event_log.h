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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_EVENT_LOG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_EVENT_LOG_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Shared product admission bounds for every event that can enter a DPM
// projection. Keeping these at the event boundary prevents durable log bytes
// that the production projector can never consume.
inline constexpr size_t kMaximumDPMEventOperationIdBytes = 16 * 1024;
inline constexpr size_t kMaximumDPMEventPayloadBytes = 16 * 1024 * 1024;
inline constexpr uint32_t kMaximumDPMGenerationTokens = 65'536;
// The canonical agent input frames one maximum event payload, one maximum
// projection, and the bounded log/case identities. The fixed allowance covers
// field labels, decimal lengths/indices, and all hexadecimal digests. Keeping
// this shared with durable decoding prevents receipt bytes from admitting an
// input the engine itself could never produce.
inline constexpr size_t kMaximumDPMCanonicalAgentInputBytes =
    2 * kMaximumDPMEventPayloadBytes +
    2 * kMaximumDPMProjectionIdentityBytes + 4 * 1024;

// The raw event log is the authority for a DPM session. Session checkpoints,
// projections, and manifests are disposable derivatives of these events.
struct DPMTurnReceipt {
  static constexpr uint32_t kFormatVersion = 3;

  uint32_t format_version = kFormatVersion;
  std::string operation_id;
  uint64_t input_event_index = 0;
  uint64_t response_event_index = 0;

  // The complete projection derivative consumed by this turn. Its embedded
  // request, output, correction, runtime, and manifest hashes are the sole
  // authoritative projection identity; parallel receipt hash fields would
  // create an avoidable split-brain validation surface.
  DPMProjectionManifest projection_manifest;
  SessionHandoffIdentity agent_session_identity;
  uint32_t max_decision_tokens = 0;
  Hash256 agent_request_hash;

  // Agent-decision execution provenance. WinnerReplay and ExactRegeneration
  // remain distinct in the immutable receipt just as they do for projection.
  DPMReplayMode agent_replay_mode =
      DPMReplayMode::kCanonicalWinnerReplay;
  Hash256 agent_replay_request_hash;
  Hash256 agent_execution_evidence_hash;
  std::optional<Hash256> agent_exact_profile_id;
  std::optional<Hash256> agent_exact_profile_admission_record_id;
  std::optional<Hash256> agent_exact_output_evidence_hash;
  uint32_t agent_exact_logit_frame_count = 0;
  bool agent_reused_canonical_winner = false;
  bool agent_producing_session_matched_output = false;

  // Stored in the immutable log so a missing KV artifact can be reconstructed
  // without treating the checkpoint repository as memory truth.
  std::string projected_memory;
  std::string canonical_agent_input;
  std::string decision_output;
  // Exact emitted token IDs are authoritative for checkpoint-free
  // reconstruction. Re-tokenizing decision_output is not guaranteed to
  // reproduce special tokens, byte fallbacks, or the decoder's final state.
  std::vector<int> decision_token_ids;

  // Hash of every canonical agent input and exact model output token sequence
  // in the current correction epoch through this turn. A changed correction
  // digest starts a fresh agent session instead of replaying stale decisions.
  Hash256 agent_transcript_hash;

  // Content address of the automatically captured producing session. Empty
  // means the checkpoint policy did not select this turn.
  std::optional<Hash256> session_checkpoint_id;
};

struct DPMEvent {
  enum class Kind : uint8_t {
    kUser = 1,
    kTool = 2,
    kInternal = 3,
    kModelTurn = 4,
    kCorrection = 5,
  };

  // Assigned by DPMEventLog::AppendIfGeneration. Event indices are contiguous
  // and zero based. Callers must leave this at zero for a new event.
  uint64_t index = 0;
  Kind kind = Kind::kUser;
  int64_t timestamp_us = 0;
  std::string operation_id;
  std::string case_id;
  std::string payload;
  std::optional<DPMTurnReceipt> turn_receipt;
};

struct DPMLogSnapshot {
  std::string log_id;
  // A log is the authoritative lineage of exactly one case. Implementations
  // bind this identity in durable log metadata and reject events for any other
  // case rather than filtering a contaminated log at read time.
  std::string case_id;
  uint64_t generation = 0;
  Hash256 prefix_hash;
  std::vector<DPMEvent> events;
};

// Append result includes the complete authoritative view after the append. A
// storage implementation may internally use an indexed/range representation;
// DPMEngine deliberately consumes this immutable value rather than consulting
// a mutable "latest" sidecar.
struct DPMAppendResult {
  uint64_t event_index = 0;
  DPMLogSnapshot snapshot;
};

// RAII ownership of a cooperative, per-log operation lease. The lease is
// distinct from the short-lived lock used by Snapshot and
// AppendIfGeneration: callers may therefore hold it across projection and
// inference while individual raw-log operations retain their strict scan and
// compare-and-append behavior.
class DPMEventLogOperationLease {
 public:
  virtual ~DPMEventLogOperationLease() = default;
};

class DPMEventLog {
 public:
  virtual ~DPMEventLog() = default;

  virtual absl::StatusOr<DPMLogSnapshot> Snapshot() const = 0;

  // Serializes complete product operations across cooperating threads and
  // processes using this log. Releasing or destroying the returned object
  // releases the lease. A process crash must also release the implementation's
  // underlying operating-system lock.
  virtual absl::StatusOr<std::unique_ptr<DPMEventLogOperationLease>>
  AcquireOperationLease() = 0;

  // Returns the cryptographic digest of exactly events [0, event_count). This
  // is used to prove that a checkpoint's cached KV prefix still corresponds to
  // the authoritative log even after later events have been appended.
  virtual absl::StatusOr<Hash256> PrefixHash(uint64_t event_count) const = 0;

  // Atomically appends only if the log is still at expected_generation. This
  // prevents a turn from committing a receipt for a projection or decision
  // that was constructed from a stale raw-log view.
  virtual absl::StatusOr<DPMAppendResult> AppendIfGeneration(
      DPMEvent event, uint64_t expected_generation) = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_EVENT_LOG_H_
