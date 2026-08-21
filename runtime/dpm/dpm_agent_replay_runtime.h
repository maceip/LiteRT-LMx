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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_AGENT_REPLAY_RUNTIME_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_AGENT_REPLAY_RUNTIME_H_

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
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_replay_executor.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

inline constexpr uint32_t kDPMAgentExecutionRequestFormatVersion = 1;
inline constexpr uint32_t kDPMAgentDecisionEnvelopeFormatVersion = 1;
inline constexpr absl::string_view kDPMAgentReplayContractVersion =
    "litert-lmx-dpm-agent-decision-v1";

// Complete logical request for one agent decision. Unlike
// DPMAgentGenerationRequest::canonical_prefill_chunks, these chunks always
// start at the active correction epoch's first event. Session checkpoint
// selection can therefore change in-process prefill work without changing the
// replay key or the request sent to a cold exact worker.
struct DPMAgentExecutionRequest {
  uint32_t format_version = kDPMAgentExecutionRequestFormatVersion;
  // Derived by DPMEngine from the active transcript prefix, canonical current
  // input, generation limit, correction digest, and loaded runtime identity.
  // It is an asserted binding inside this self-contained worker request, not a
  // caller-selectable cache label. The complete chunks below remain the actual
  // model-visible input and are never replaced by this digest.
  Hash256 logical_agent_request_hash;
  Hash256 correction_digest;
  uint32_t max_output_tokens = 0;
  std::vector<DPMAgentGenerationRequest::PrefillChunk>
      full_canonical_prefill_chunks;
};

absl::Status ValidateDPMAgentExecutionRequest(
    const DPMAgentExecutionRequest& request);
absl::StatusOr<std::string> EncodeDPMAgentExecutionRequest(
    const DPMAgentExecutionRequest& request);
absl::StatusOr<DPMAgentExecutionRequest> DecodeDPMAgentExecutionRequest(
    absl::string_view bytes);

// Canonical catalog/worker output. Visible text and exact executor token IDs
// remain distinct so stop/EOS and byte-fallback tokens cannot be reconstructed
// by re-tokenizing text.
struct DPMAgentDecisionEnvelope {
  uint32_t format_version = kDPMAgentDecisionEnvelopeFormatVersion;
  std::string decision_output;
  std::string canonical_token_bytes;
};

absl::Status ValidateDPMAgentDecisionEnvelope(
    const DPMAgentDecisionEnvelope& envelope);
absl::StatusOr<std::string> EncodeDPMAgentDecisionEnvelope(
    const DPMAgentDecisionEnvelope& envelope);
absl::StatusOr<DPMAgentDecisionEnvelope> DecodeDPMAgentDecisionEnvelope(
    absl::string_view bytes);

struct DPMAgentReplayExecution {
  DPMReplayMode mode = DPMReplayMode::kCanonicalWinnerReplay;
  Hash256 replay_request_hash;
  // WinnerReplay carries the deterministic binding authenticated in its
  // create-once catalog. ExactRegeneration carries the fresh per-request
  // cold-run record ID. The former is not independent-run evidence.
  Hash256 execution_evidence_hash;
  std::optional<Hash256> exact_profile_id;
  std::optional<Hash256> exact_profile_admission_record_id;
  std::string decision_output;
  std::vector<int> decision_token_ids;
  std::string exact_token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> exact_logit_frames;
  // Present only for ExactRegeneration. This binds the canonical decision
  // envelope, DPMTOK01 bytes, and the complete ordered logits-frame evidence;
  // it is distinct from both profile admission and the cold-run record ID.
  std::optional<Hash256> exact_output_evidence_hash;
  bool reused_canonical_winner = false;

  // True only when `producing_session` is the live session that generated the
  // selected bytes. Catalog hits, lost publication races, and current exact
  // cold-worker results are false; DPMEngine must never capture those parent
  // sessions as if they produced the returned decision.
  bool producing_session_matches_output = false;
};

absl::Status ValidateDPMAgentReplayExecution(
    const DPMAgentReplayExecution& execution, uint32_t max_output_tokens);

// Product-level agent execution. Each instance has one immutable replay mode;
// callers cannot switch a shared object between replay and exact authority.
class DPMAgentReplayRuntime {
 public:
  virtual ~DPMAgentReplayRuntime() = default;
  virtual DPMReplayMode GetReplayMode() const = 0;
  virtual const SessionHandoffIdentity& GetSessionHandoffIdentity() const = 0;
  virtual absl::StatusOr<std::optional<Hash256>> GetExactProfileId() const = 0;
  // Replay/generation capability is independent from CapsuleRestore. This
  // method must not reject a usable WinnerReplay runtime merely because the
  // loaded session cannot export a complete handoff.
  virtual absl::Status ValidateSupport() const = 0;

  // Optional, separately admitted CapsuleRestore capability. DPMEngine calls
  // this only when checkpoint restore or capture is enabled. Exact agent
  // execution currently fails here because its producing session is isolated
  // in a worker and no authenticated capsule is returned to the parent.
  virtual absl::Status ValidateSessionHandoffSupport() const = 0;

  // WinnerReplay needs a parent session for optional checkpoint restore and a
  // newly published producing capsule. ExactRegeneration currently returns
  // Unimplemented here because its producing session lives in the worker; it
  // must not manufacture a matching parent session.
  virtual absl::StatusOr<std::unique_ptr<Engine::Session>> CreateSession() = 0;

  // For WinnerReplay, execution_request may contain only the post-checkpoint
  // delta while logical_request always contains the complete canonical input.
  // ExactRegeneration accepts no parent session and requires those chunk lists
  // to be identical because every cold worker starts from an empty session.
  virtual absl::StatusOr<DPMAgentReplayExecution> Generate(
      Engine::Session* producing_session,
      const DPMAgentGenerationRequest& execution_request,
      const DPMAgentExecutionRequest& logical_request) = 0;
};

class CanonicalWinnerDPMAgentRuntime final : public DPMAgentReplayRuntime {
 public:
  static absl::StatusOr<std::unique_ptr<CanonicalWinnerDPMAgentRuntime>>
  Create(DPMAgentRuntime* inference_runtime,
         CanonicalWinnerReplayCatalog* catalog);

  DPMReplayMode GetReplayMode() const override {
    return DPMReplayMode::kCanonicalWinnerReplay;
  }
  const SessionHandoffIdentity& GetSessionHandoffIdentity() const override {
    return runtime_identity_;
  }
  absl::StatusOr<std::optional<Hash256>> GetExactProfileId() const override;
  absl::Status ValidateSupport() const override;
  absl::Status ValidateSessionHandoffSupport() const override;
  absl::StatusOr<std::unique_ptr<Engine::Session>> CreateSession() override;
  absl::StatusOr<DPMAgentReplayExecution> Generate(
      Engine::Session* producing_session,
      const DPMAgentGenerationRequest& execution_request,
      const DPMAgentExecutionRequest& logical_request) override;

 private:
  CanonicalWinnerDPMAgentRuntime(
      DPMAgentRuntime* inference_runtime,
      CanonicalWinnerReplayCatalog* catalog,
      SessionHandoffIdentity runtime_identity)
      : inference_runtime_(inference_runtime),
        runtime_identity_(runtime_identity),
        replay_executor_(catalog) {}

  DPMAgentRuntime* const inference_runtime_;
  const SessionHandoffIdentity runtime_identity_;
  CanonicalWinnerReplayExecutor replay_executor_;
};

class ExactRegenerationDPMAgentRuntime final : public DPMAgentReplayRuntime {
 public:
  static absl::StatusOr<std::unique_ptr<ExactRegenerationDPMAgentRuntime>>
  Create(ExactRegenerationExecutor* exact_executor);

  DPMReplayMode GetReplayMode() const override {
    return DPMReplayMode::kExactRegeneration;
  }
  const SessionHandoffIdentity& GetSessionHandoffIdentity() const override {
    return derived_profile_.session_identity;
  }
  absl::StatusOr<std::optional<Hash256>> GetExactProfileId() const override;
  absl::Status ValidateSupport() const override;
  absl::Status ValidateSessionHandoffSupport() const override;
  absl::StatusOr<std::unique_ptr<Engine::Session>> CreateSession() override;
  absl::StatusOr<DPMAgentReplayExecution> Generate(
      Engine::Session* producing_session,
      const DPMAgentGenerationRequest& execution_request,
      const DPMAgentExecutionRequest& logical_request) override;

 private:
  ExactRegenerationDPMAgentRuntime(
      ExactRegenerationExecutor* exact_executor,
      ExactLiteRtProfile derived_profile)
      : exact_executor_(exact_executor),
        derived_profile_(std::move(derived_profile)) {}

  ExactRegenerationExecutor* const exact_executor_;
  const ExactLiteRtProfile derived_profile_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_AGENT_REPLAY_RUNTIME_H_
