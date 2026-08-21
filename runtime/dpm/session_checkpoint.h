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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_SESSION_CHECKPOINT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_SESSION_CHECKPOINT_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

enum class DPMSessionCheckpointStage : uint8_t {
  kAgentDecision = 1,
};

// Identifies the runtime boundary that produced the authenticated capsule.
// kNone is useful while assembling a descriptor but can never describe a
// publishable artifact.
enum class DPMCheckpointCaptureOrigin : uint8_t {
  kNone = 0,
  kLiveParentSession = 1,
  kAuthenticatedFreshWorker = 2,
};

// The physical work performed by an exact worker. This deliberately mirrors
// FreshWorkerPrefillMode without making durable checkpoint descriptors depend
// on the worker transport protocol. WinnerReplay and legacy descriptors use
// kNone.
enum class DPMCheckpointWorkerPrefillMode : uint8_t {
  kNone = 0,
  kFullCanonicalPrefill = 1,
  kOwnPositionCapsuleDelta = 2,
};

// Request-scoped evidence for the fresh worker whose post-decode session was
// captured. The transient envelope is authenticated under a per-request key;
// the descriptor's existing envelope fields bind the separately rewrapped
// durable bytes. Keeping both hashes preserves the non-cyclic direction:
// worker output and transient capture, then durable rewrap, then descriptor.
struct DPMExactWorkerCheckpointProvenance {
  uint32_t run_index = 0;
  Hash256 execution_plan_hash;
  Hash256 request_envelope_hash;
  Hash256 result_envelope_hash;
  uint64_t transient_envelope_size = 0;
  Hash256 transient_envelope_hash;
  Hash256 output_evidence_hash;
};

// The descriptor is content addressed independently of the envelope. It binds
// a disposable KV cache to the exact raw-log prefix and DPM artifacts that
// produced it. A matching model/profile alone is intentionally insufficient.
struct DPMSessionCheckpointDescriptor {
  static constexpr uint32_t kLegacyFormatVersion = 1;
  static constexpr uint32_t kFormatVersion = 2;

  uint32_t format_version = kFormatVersion;
  Hash256 descriptor_id;
  std::string log_id;
  DPMSessionCheckpointStage stage =
      DPMSessionCheckpointStage::kAgentDecision;

  // Raw input prefix consumed by the turn. The response event is committed
  // after descriptor publication and names this descriptor, avoiding a hash
  // cycle between the descriptor and authoritative receipt.
  uint64_t source_event_count = 0;
  Hash256 source_prefix_hash;
  uint64_t response_event_index = 0;

  Hash256 projection_request_hash;
  Hash256 projection_manifest_hash;
  Hash256 correction_digest;
  Hash256 agent_request_hash;
  Hash256 agent_transcript_hash;

  SessionHandoffIdentity session_identity;
  std::string key_id;
  Hash256 envelope_hash;
  uint64_t envelope_size = 0;
  int64_t created_unix_micros = 0;

  // Version 2 provenance. Version 1 descriptors infer
  // CanonicalWinnerReplay plus a live-parent capture and serialize none of
  // these fields. A restored-from ID points backward to an already-published
  // descriptor and is therefore safe to include in this descriptor's content
  // address.
  DPMReplayMode replay_mode = DPMReplayMode::kCanonicalWinnerReplay;
  DPMCheckpointCaptureOrigin capture_origin =
      DPMCheckpointCaptureOrigin::kLiveParentSession;
  std::optional<Hash256> restored_from_checkpoint_id;

  // ExactRegeneration-only identities. exact_profile_id is optional at the
  // type level so WinnerReplay can prove its absence; every exact descriptor
  // must populate it and every following digest.
  std::optional<Hash256> exact_profile_id;
  Hash256 exact_profile_admission_record_id;
  Hash256 capsule_restore_admission_record_id;
  Hash256 exact_request_execution_evidence_id;
  DPMCheckpointWorkerPrefillMode worker_prefill_mode =
      DPMCheckpointWorkerPrefillMode::kNone;
  // Physical full-prefill or own-position-delta plan used by the worker.
  Hash256 execution_plan_hash;
  Hash256 exact_output_evidence_hash;
  std::optional<DPMExactWorkerCheckpointProvenance> worker_provenance;
};

struct DPMSessionCheckpointArtifact {
  DPMSessionCheckpointDescriptor descriptor;
  std::string authenticated_envelope;
};

// Computes the descriptor ID from every field except descriptor_id. The
// encoding is versioned, length-delimited, and independent of host endianness.
absl::StatusOr<Hash256> ComputeDPMSessionCheckpointId(
    const DPMSessionCheckpointDescriptor& descriptor);

absl::Status ValidateDPMSessionCheckpointArtifact(
    const DPMSessionCheckpointArtifact& artifact);

class DPMSessionCheckpointRepository {
 public:
  virtual ~DPMSessionCheckpointRepository() = default;

  // Implementations publish the complete envelope before making the
  // descriptor visible. Re-publishing an identical content address is
  // idempotent; conflicting bytes for an existing address must fail closed.
  virtual absl::Status Put(
      const DPMSessionCheckpointArtifact& artifact) = 0;
  virtual absl::StatusOr<DPMSessionCheckpointArtifact> Get(
      const Hash256& descriptor_id) const = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_SESSION_CHECKPOINT_H_
