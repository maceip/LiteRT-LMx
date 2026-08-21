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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_CAPABILITIES_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_CAPABILITIES_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/engine/session_handoff_capability.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

enum class DPMCapabilityStage : uint8_t {
  kProjection = 1,
  kAgentDecision = 2,
};

// Public guarantees remain distinct. In particular, a candidate profile is
// not ExactRegeneration admission, and successful capsule transport is not a
// CapsuleRestore qualification.
enum class DPMGuarantee : uint8_t {
  kLogAuthority = 1,
  kCanonicalWinnerReplay = 2,
  kCapsuleRestore = 3,
  kExactRegeneration = 4,
  kFailClosedCapabilities = 5,
};

enum class DPMCapabilityScope : uint32_t {
  kProjection = uint32_t{1} << 0,
  kAgentDecision = uint32_t{1} << 1,
  kExactRegeneration = uint32_t{1} << 2,
  kCapsuleRestore = uint32_t{1} << 3,
};

constexpr uint32_t DPMCapabilityScopeBit(DPMCapabilityScope scope) {
  return static_cast<uint32_t>(scope);
}

enum class DPMRestrictedFeature : uint8_t {
  kLoraState = 1,
  kAudioOrVisionState = 2,
  kPendingEmbeddings = 3,
  kMtpOrSpeculativeDecode = 4,
  kExternalOrGpuSampling = 5,
  kGraphCallbacks = 6,
  kBenchmarkProfilingOrLogitsDebugCallbacks = 7,
  kLogitsProcessorsConstraintsOrTokenSuppression = 8,
  kMultipleOutputCandidates = 9,
  kOffPositionCapsuleGraft = 10,
  kProjectionRepairLoop = 11,
};

struct DPMFeatureRestriction {
  DPMRestrictedFeature feature;
  uint32_t blocked_scopes = 0;
  const char* stable_name = nullptr;
};

// Stable process-lifetime view of the normative fail-closed feature table.
absl::Span<const DPMFeatureRestriction> GetDPMFeatureRestrictions();

struct DPMStageCapabilities {
  DPMCapabilityStage stage = DPMCapabilityStage::kProjection;
  DPMReplayMode replay_mode = DPMReplayMode::kCanonicalWinnerReplay;
  SessionHandoffIdentity runtime_identity;

  // Present together only for ExactRegeneration. The complete profile is
  // runtime-derived; the record ID is reauthenticated current admission.
  std::optional<ExactLiteRtProfile> exact_profile;
  std::optional<Hash256> exact_profile_admission_record_id;

  uint32_t max_output_tokens = 0;
};

// Validates the complete disjoint stage contract. WinnerReplay must not carry
// exact identities. ExactRegeneration must carry a canonical Engine-derived
// profile and a current nonzero admission record, and the profile must own the
// reported runtime identity.
absl::Status ValidateDPMStageCapabilities(
    const DPMStageCapabilities& capabilities);

struct DPMAgentCheckpointCapabilities {
  bool checkpoint_transport_configured = false;
  bool restore_enabled = false;
  uint32_t capture_interval_turns = 0;
  bool capture_required_at_milestone = false;

  // Exact-only formal capability/admission. WinnerReplay may still use a
  // validated disposable live-session cache without claiming CapsuleRestore.
  std::optional<SessionHandoffCapability> session_handoff_capability;
  std::optional<Hash256> capsule_restore_admission_record_id;
};

struct DPMGuaranteeAvailability {
  DPMGuarantee guarantee = DPMGuarantee::kLogAuthority;
  bool available = false;
  // Stable machine-facing reason code; empty when available.
  std::string unavailable_reason;
};

struct DPMCapabilities {
  static constexpr uint32_t kFormatVersion = 1;
  uint32_t format_version = kFormatVersion;
  DPMStageCapabilities projection;
  DPMStageCapabilities agent;
  DPMAgentCheckpointCapabilities checkpoints;
  std::vector<DPMGuaranteeAvailability> guarantees;
  std::vector<DPMFeatureRestriction> fail_closed_features;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_CAPABILITIES_H_
