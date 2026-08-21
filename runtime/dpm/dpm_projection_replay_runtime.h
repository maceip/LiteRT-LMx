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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_REPLAY_RUNTIME_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_REPLAY_RUNTIME_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/canonical_replay_catalog.h"
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_projection_runtime.h"
#include "runtime/dpm/dpm_replay_executor.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

inline constexpr absl::string_view kDPMProjectionReplayContractVersion =
    "litert-lmx-dpm-projection-v1";
inline constexpr uint32_t kDPMProjectionExecutionRequestFormatVersion = 1;

// Complete model-visible projection request carried inside the generic replay
// envelope. The projection configuration is included, not merely named by a
// digest, so a fresh worker can apply the same generation and canonical-output
// limits without caller-local state. The inner request hash is recomputed by
// OneShotDPMProjector before this object is constructed and is checked again by
// every parent-side product adapter.
struct DPMProjectionExecutionRequest {
  uint32_t format_version =
      kDPMProjectionExecutionRequestFormatVersion;
  uint64_t source_event_count = 0;
  Hash256 projection_request_hash;
  Hash256 projection_config_hash;
  DPMProjectionConfig projection_config;
  std::string canonical_prompt;
};

absl::Status ValidateDPMProjectionExecutionRequest(
    const DPMProjectionExecutionRequest& request);
absl::StatusOr<std::string> EncodeDPMProjectionExecutionRequest(
    const DPMProjectionExecutionRequest& request);
absl::StatusOr<DPMProjectionExecutionRequest>
DecodeDPMProjectionExecutionRequest(absl::string_view bytes);

struct DPMProjectionReplayExecution {
  DPMReplayMode mode = DPMReplayMode::kCanonicalWinnerReplay;
  Hash256 replay_request_hash;
  // WinnerReplay uses the authenticated winner's deterministic candidate
  // binding. ExactRegeneration uses the fresh cold-run record ID. Callers must
  // interpret this field together with `mode`; the former is never
  // independent-regeneration evidence.
  Hash256 execution_evidence_hash;
  std::optional<Hash256> exact_profile_id;
  std::optional<Hash256> exact_profile_admission_record_id;
  std::string canonical_output;
  // Populated only by ExactRegeneration. These are the independently equal
  // canonical token bytes and ordered, complete logits-frame digests returned
  // by the fresh cold processes. WinnerReplay leaves both empty and makes no
  // independent-generation claim.
  std::string exact_token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> exact_logit_frames;
  // Meaningful only for CanonicalWinnerReplay. A lost publication race has
  // already discarded its local candidate and therefore reports reuse.
  bool reused_canonical_winner = false;
};

// Enforces the disjoint result contracts. In particular, WinnerReplay cannot
// carry exact-profile/token/logits claims, while ExactRegeneration must carry
// all of them and can never claim catalog reuse.
absl::Status ValidateDPMProjectionReplayExecution(
    const DPMProjectionReplayExecution& execution);

// Product-level projection execution. Implementations are one named replay
// mode; callers cannot toggle a shared object between weaker and stronger
// authority paths.
class DPMProjectionReplayRuntime {
 public:
  virtual ~DPMProjectionReplayRuntime() = default;
  virtual DPMReplayMode GetReplayMode() const = 0;
  virtual const SessionHandoffIdentity& GetRuntimeIdentity() const = 0;
  // Returns nullopt for WinnerReplay. ExactRegeneration re-derives and
  // validates the complete profile before returning its construction-time
  // profile ID, so baseline selection cannot silently cross exact profiles.
  virtual absl::StatusOr<std::optional<Hash256>> GetExactProfileId() const = 0;
  virtual uint32_t GetMaxOutputTokens() const = 0;
  virtual absl::Status ValidateSupport() const = 0;
  virtual absl::StatusOr<DPMProjectionReplayExecution> Generate(
      const CanonicalDPMProjectionRequest& request) = 0;
};

// Portable publish-once mode. The wrapped inference runtime invokes the model
// only on a catalog miss. This class owns a catalog-backed executor and cannot
// expose ExactRegeneration.
class CanonicalWinnerDPMProjectionRuntime final
    : public DPMProjectionReplayRuntime,
      private CanonicalWinnerReplayGenerator {
 public:
  static absl::StatusOr<
      std::unique_ptr<CanonicalWinnerDPMProjectionRuntime>>
  Create(DPMProjectionRuntime* inference_runtime,
         CanonicalWinnerReplayCatalog* catalog, DPMProjectionConfig config);

  DPMReplayMode GetReplayMode() const override {
    return DPMReplayMode::kCanonicalWinnerReplay;
  }
  const SessionHandoffIdentity& GetRuntimeIdentity() const override;
  absl::StatusOr<std::optional<Hash256>> GetExactProfileId() const override;
  uint32_t GetMaxOutputTokens() const override;
  absl::Status ValidateSupport() const override;
  absl::StatusOr<DPMProjectionReplayExecution> Generate(
      const CanonicalDPMProjectionRequest& request) override;

 private:
  CanonicalWinnerDPMProjectionRuntime(
      DPMProjectionRuntime* inference_runtime,
      CanonicalWinnerReplayCatalog* catalog, DPMProjectionConfig config,
      SessionHandoffIdentity runtime_identity)
      : inference_runtime_(inference_runtime),
        config_(std::move(config)),
        runtime_identity_(runtime_identity),
        replay_executor_(catalog) {}

  absl::Status ValidateSupport(DPMReplayStage expected_stage) const override;
  absl::StatusOr<CanonicalWinnerGeneratedCandidate> Generate(
      const DPMCanonicalReplayRequest& request) override;

  DPMProjectionRuntime* const inference_runtime_;
  const DPMProjectionConfig config_;
  const SessionHandoffIdentity runtime_identity_;
  CanonicalWinnerReplayExecutor replay_executor_;
};

// Catalog-free cold-process mode. It owns no catalog and delegates only to the
// ExactRegenerationExecutor. The product may expose this as independent exact
// regeneration only with the concrete Engine-owning projection adapter in the
// worker executable; the generic callback seam alone is not that proof.
class ExactRegenerationDPMProjectionRuntime final
    : public DPMProjectionReplayRuntime {
 public:
  static absl::StatusOr<
      std::unique_ptr<ExactRegenerationDPMProjectionRuntime>>
  Create(ExactRegenerationExecutor* exact_executor,
         DPMProjectionConfig config);

  DPMReplayMode GetReplayMode() const override {
    return DPMReplayMode::kExactRegeneration;
  }
  const SessionHandoffIdentity& GetRuntimeIdentity() const override {
    return derived_profile_.session_identity;
  }
  absl::StatusOr<std::optional<Hash256>> GetExactProfileId() const override;
  uint32_t GetMaxOutputTokens() const override {
    return config_.max_output_tokens;
  }
  absl::Status ValidateSupport() const override;
  absl::StatusOr<DPMProjectionReplayExecution> Generate(
      const CanonicalDPMProjectionRequest& request) override;

 private:
  ExactRegenerationDPMProjectionRuntime(
      ExactRegenerationExecutor* exact_executor,
      DPMProjectionConfig config, ExactLiteRtProfile derived_profile)
      : exact_executor_(exact_executor),
        config_(std::move(config)),
        derived_profile_(std::move(derived_profile)) {}

  ExactRegenerationExecutor* const exact_executor_;
  const DPMProjectionConfig config_;
  const ExactLiteRtProfile derived_profile_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_REPLAY_RUNTIME_H_
