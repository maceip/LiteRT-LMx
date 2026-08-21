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

#include "runtime/dpm/dpm_capabilities.h"

#include <array>
#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr uint32_t kProjection =
    DPMCapabilityScopeBit(DPMCapabilityScope::kProjection);
constexpr uint32_t kAgent =
    DPMCapabilityScopeBit(DPMCapabilityScope::kAgentDecision);
constexpr uint32_t kExact =
    DPMCapabilityScopeBit(DPMCapabilityScope::kExactRegeneration);
constexpr uint32_t kCapsule =
    DPMCapabilityScopeBit(DPMCapabilityScope::kCapsuleRestore);

constexpr std::array<DPMFeatureRestriction, 11> kRestrictions = {{
    {DPMRestrictedFeature::kLoraState,
     kProjection | kAgent | kExact | kCapsule, "lora_state"},
    {DPMRestrictedFeature::kAudioOrVisionState,
     kProjection | kAgent | kExact | kCapsule, "audio_or_vision_state"},
    {DPMRestrictedFeature::kPendingEmbeddings,
     kAgent | kExact | kCapsule, "pending_embeddings"},
    {DPMRestrictedFeature::kMtpOrSpeculativeDecode,
     kProjection | kAgent | kExact | kCapsule,
     "mtp_or_speculative_decode"},
    {DPMRestrictedFeature::kExternalOrGpuSampling,
     kProjection | kAgent | kExact | kCapsule,
     "external_or_gpu_sampling"},
    {DPMRestrictedFeature::kGraphCallbacks,
     kProjection | kAgent | kExact | kCapsule, "graph_callbacks"},
    {DPMRestrictedFeature::kBenchmarkProfilingOrLogitsDebugCallbacks,
     kProjection | kAgent | kExact | kCapsule,
     "benchmark_profiling_or_logits_debug_callbacks"},
    {DPMRestrictedFeature::kLogitsProcessorsConstraintsOrTokenSuppression,
     kProjection | kAgent | kExact | kCapsule,
     "logits_processors_constraints_or_token_suppression"},
    {DPMRestrictedFeature::kMultipleOutputCandidates,
     kProjection | kAgent | kExact | kCapsule,
     "multiple_output_candidates"},
    {DPMRestrictedFeature::kOffPositionCapsuleGraft, kCapsule,
     "off_position_capsule_graft"},
    {DPMRestrictedFeature::kProjectionRepairLoop, kProjection | kExact,
     "projection_repair_loop"},
}};

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

absl::Status ValidateRuntimeIdentity(
    const SessionHandoffIdentity& identity) {
  if (IsZeroHash(identity.model_artifact_hash) ||
      IsZeroHash(identity.runtime_artifact_hash) ||
      IsZeroHash(identity.inference_profile_hash)) {
    return absl::FailedPreconditionError(
        "DPM stage capability has an incomplete runtime identity.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Span<const DPMFeatureRestriction> GetDPMFeatureRestrictions() {
  return absl::MakeConstSpan(kRestrictions);
}

Hash256 GetDPMRestrictedFeatureContractHash() {
  std::string canonical;
  const auto append_u32 = [&canonical](uint32_t value) {
    for (int shift = 24; shift >= 0; shift -= 8) {
      canonical.push_back(static_cast<char>((value >> shift) & 0xff));
    }
  };
  const auto append_frame = [&canonical, &append_u32](
                                absl::string_view value) {
    append_u32(static_cast<uint32_t>(value.size()));
    canonical.append(value.data(), value.size());
  };
  append_u32(1);
  append_u32(static_cast<uint32_t>(kRestrictions.size()));
  for (const DPMFeatureRestriction& restriction : kRestrictions) {
    append_u32(static_cast<uint32_t>(restriction.feature));
    append_u32(restriction.blocked_scopes);
    append_frame(restriction.stable_name);
  }
  Sha256Hasher hasher;
  hasher.Update("LITERT_LMX_DPM_RESTRICTED_FEATURE_CONTRACT_SHA256_V1");
  hasher.Update(canonical);
  return hasher.Finalize();
}

absl::Status ValidateDPMStageCapabilities(
    const DPMStageCapabilities& capabilities) {
  switch (capabilities.stage) {
    case DPMCapabilityStage::kProjection:
    case DPMCapabilityStage::kAgentDecision:
      break;
    default:
      return absl::FailedPreconditionError(
          "DPM stage capability names an unknown stage.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(capabilities.replay_mode));
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(capabilities.runtime_identity));
  if (capabilities.max_output_tokens == 0 ||
      capabilities.max_output_tokens > kMaximumDPMGenerationTokens) {
    return absl::FailedPreconditionError(
        "DPM stage capability has an invalid product token bound.");
  }

  const bool has_profile = capabilities.exact_profile.has_value();
  const bool has_admission =
      capabilities.exact_profile_admission_record_id.has_value();
  switch (capabilities.replay_mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      if (has_profile || has_admission) {
        return absl::FailedPreconditionError(
            "CanonicalWinnerReplay stage capability cannot carry an exact "
            "profile or admission claim.");
      }
      break;
    case DPMReplayMode::kExactRegeneration:
      if (!has_profile || !has_admission ||
          IsZeroHash(*capabilities.exact_profile_admission_record_id)) {
        return absl::FailedPreconditionError(
            "ExactRegeneration stage capability requires a complete profile "
            "and current admission record.");
      }
      ABSL_RETURN_IF_ERROR(
          ValidateExactLiteRtProfile(*capabilities.exact_profile));
      if (capabilities.exact_profile->session_identity !=
          capabilities.runtime_identity) {
        return absl::FailedPreconditionError(
            "ExactRegeneration stage profile does not own the reported "
            "runtime identity.");
      }
      break;
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
