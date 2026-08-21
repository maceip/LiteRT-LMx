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
#include "runtime/dpm/dpm_capabilities.h"
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
inline constexpr uint32_t kDPMAgentDeltaExecutionRequestFormatVersion = 1;
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

// Physical own-position continuation plan for a fresh exact worker. The
// complete DPMAgentExecutionRequest remains in the logical replay envelope;
// this separately authenticated value binds only the selected checkpoint and
// exact post-checkpoint chunks that replace full prefill work.
struct DPMAgentDeltaExecutionRequest {
  uint32_t format_version =
      kDPMAgentDeltaExecutionRequestFormatVersion;
  Hash256 logical_agent_request_hash;
  Hash256 correction_digest;
  Hash256 restore_checkpoint_id;
  uint64_t restored_response_event_index = 0;
  Hash256 restored_agent_transcript_hash;
  uint32_t max_output_tokens = 0;
  std::vector<DPMAgentGenerationRequest::PrefillChunk>
      canonical_delta_prefill_chunks;
};

absl::Status ValidateDPMAgentDeltaExecutionRequest(
    const DPMAgentDeltaExecutionRequest& request);
absl::StatusOr<std::string> EncodeDPMAgentDeltaExecutionRequest(
    const DPMAgentDeltaExecutionRequest& request);
absl::StatusOr<DPMAgentDeltaExecutionRequest>
DecodeDPMAgentDeltaExecutionRequest(absl::string_view bytes);
// Cross-object worker admission. This proves that the physical delta is an
// unchanged suffix of the complete logical agent input and is bound to the
// authenticated restore plan. Repository-descriptor/artifact validation is a
// separate parent-side requirement because those authority objects are never
// sent to the child.
absl::Status ValidateDPMAgentDeltaExecutionBinding(
    const DPMAgentExecutionRequest& logical_request,
    const FreshWorkerExecutionPlan& execution_plan,
    const DPMAgentDeltaExecutionRequest& delta_request);

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
  // physical-run evidence ID. The former is not independent-run evidence.
  Hash256 execution_evidence_hash;
  std::optional<Hash256> exact_profile_id;
  std::optional<Hash256> exact_profile_admission_record_id;
  std::string decision_output;
  std::vector<int> decision_token_ids;
  std::string exact_token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> exact_logit_frames;
  // Present only for ExactRegeneration. This binds the canonical decision
  // envelope, DPMTOK01 bytes, and the complete ordered logits-frame evidence;
  // it is distinct from both profile admission and the request-evidence ID.
  std::optional<Hash256> exact_output_evidence_hash;
  bool reused_canonical_winner = false;

  // True only when `producing_session` is the live session that generated the
  // selected bytes. Catalog hits, lost publication races, and current exact
  // cold-worker results are false; DPMEngine must never capture those parent
  // sessions as if they produced the returned decision.
  bool producing_session_matches_output = false;
  // Present only when this result is backed by the live session that executed
  // this exact runtime-derived physical prefill plan. Catalog-only results have
  // no physical-session plan. Exact workers populate this after all cold runs
  // independently agree on the same derived plan.
  std::optional<DPMPreparedPrefillPlan> prepared_prefill_plan;
};

absl::Status ValidateDPMAgentReplayExecution(
    const DPMAgentReplayExecution& execution, uint32_t max_output_tokens);

// Exact worker execution plus the complete request-scoped physical evidence
// needed by DPMEngine to construct a checkpoint descriptor and receipt. No
// parent Engine::Session is implied: capsule bytes remain in the caller-owned
// staging sink represented by the durable evidence below.
struct ExactRegenerationDPMAgentPhysicalExecution {
  DPMAgentReplayExecution replay_execution;
  ExactRegenerationRequestEvidence request_evidence;
  Hash256 physical_execution_plan_hash;
  FreshWorkerPrefillMode prefill_mode =
      FreshWorkerPrefillMode::kFullCanonicalPrefill;
  std::optional<Hash256> restored_checkpoint_id;
  ExactRegenerationCaptureRunPolicy capture_run_policy =
      ExactRegenerationCaptureRunPolicy::kNoCapture;
  std::optional<FreshWorkerProducingCapsuleEvidence>
      run_zero_transient_producing_capsule_evidence;
  std::optional<FreshWorkerDurableProducingCapsuleEvidence>
      durable_producing_capsule_evidence;
};

absl::Status ValidateExactRegenerationDPMAgentPhysicalExecution(
    const ExactRegenerationDPMAgentPhysicalExecution& execution,
    uint32_t max_output_tokens);

// Product-level agent execution. Each instance has one immutable replay mode;
// callers cannot switch a shared object between replay and exact authority.
class DPMAgentReplayRuntime {
 public:
  virtual ~DPMAgentReplayRuntime() = default;
  virtual DPMReplayMode GetReplayMode() const = 0;
  virtual const SessionHandoffIdentity& GetSessionHandoffIdentity() const = 0;
  virtual absl::StatusOr<std::optional<Hash256>> GetExactProfileId() const = 0;
  // Re-authenticated durable admission ID for validating exact checkpoint
  // provenance before worker launch. WinnerReplay returns nullopt.
  virtual absl::StatusOr<std::optional<Hash256>>
  GetExactProfileAdmissionRecordId() const = 0;
  // Authenticated CapsuleRestore admission and Engine-derived capability
  // identities available before generation. Either replay mode may expose
  // them; runtimes constructed without checkpoint support return nullopt.
  virtual absl::StatusOr<std::optional<Hash256>>
  GetCapsuleRestoreAdmissionRecordId() const = 0;
  virtual absl::StatusOr<std::optional<Hash256>>
  GetSessionHandoffCapabilityId() const = 0;
  virtual absl::StatusOr<DPMStageCapabilities> GetCapabilities() const = 0;
  virtual absl::StatusOr<std::optional<SessionHandoffCapability>>
  GetSessionHandoffCapability() const = 0;
  virtual absl::StatusOr<std::optional<CapsuleRestoreOperationalCoverage>>
  GetCapsuleRestoreOperationalCoverage() const = 0;
  // Replay/generation capability is independent from CapsuleRestore. This
  // method must not reject a usable WinnerReplay runtime merely because the
  // loaded session cannot export a complete handoff.
  virtual absl::Status ValidateSupport() const = 0;

  // Preflights the DPMEngine decision limit before the authoritative input is
  // appended. ExactRegeneration requires equality with its immutable worker
  // profile; WinnerReplay admits any ordinary bounded product request.
  virtual absl::Status ValidateGenerationLimit(
      uint32_t max_output_tokens) const = 0;

  // Optional, separately admitted CapsuleRestore capability. DPMEngine calls
  // this only when checkpoint restore or capture is enabled. Both replay modes
  // fail this gate until their loaded runtime reauthenticates the separate
  // CapsuleRestore admission for the selected profile and checkpoint codec.
  virtual absl::Status ValidateSessionHandoffSupport() const = 0;

  // WinnerReplay needs a parent session for optional checkpoint restore and a
  // newly published producing capsule. ExactRegeneration returns
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

  // WinnerReplay-only recovery primitive. The caller supplies the already
  // selected authenticated winner and a fresh or own-position-restored live
  // session. A successful implementation must bypass the catalog, invoke the
  // loaded inference runtime exactly once, and return success only when that
  // live session reproduces the selected winner's complete canonical bytes and
  // request-scoped evidence. ExactRegeneration has no parent session and keeps
  // this fail-closed default.
  virtual absl::StatusOr<DPMAgentReplayExecution>
  RematerializeCanonicalWinner(
      Engine::Session*, const DPMAgentGenerationRequest&,
      const DPMAgentExecutionRequest&, const DPMAgentReplayExecution&) {
    return absl::UnimplementedError(
        "This DPM agent runtime cannot rematerialize a canonical winner in a "
        "live parent session.");
  }

  // Exact-only fresh-process full/delta/capture entry point. WinnerReplay
  // leaves this unavailable; it continues to use its live parent Session.
  virtual absl::StatusOr<ExactRegenerationDPMAgentPhysicalExecution>
  GeneratePhysicalExact(
      const DPMAgentExecutionRequest&,
      const ExactRegenerationExecutionInput&) {
    return absl::UnimplementedError(
        "This DPM agent runtime has no physical exact-worker path.");
  }
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
  absl::StatusOr<std::optional<Hash256>>
  GetExactProfileAdmissionRecordId() const override;
  absl::StatusOr<std::optional<Hash256>>
  GetCapsuleRestoreAdmissionRecordId() const override;
  absl::StatusOr<std::optional<Hash256>>
  GetSessionHandoffCapabilityId() const override;
  absl::StatusOr<DPMStageCapabilities> GetCapabilities() const override;
  absl::StatusOr<std::optional<SessionHandoffCapability>>
  GetSessionHandoffCapability() const override;
  absl::StatusOr<std::optional<CapsuleRestoreOperationalCoverage>>
  GetCapsuleRestoreOperationalCoverage() const override;
  absl::Status ValidateSupport() const override;
  absl::Status ValidateGenerationLimit(
      uint32_t max_output_tokens) const override;
  absl::Status ValidateSessionHandoffSupport() const override;
  absl::StatusOr<std::unique_ptr<Engine::Session>> CreateSession() override;
  absl::StatusOr<DPMAgentReplayExecution> Generate(
      Engine::Session* producing_session,
      const DPMAgentGenerationRequest& execution_request,
      const DPMAgentExecutionRequest& logical_request) override;
  absl::StatusOr<DPMAgentReplayExecution>
  RematerializeCanonicalWinner(
      Engine::Session* producing_session,
      const DPMAgentGenerationRequest& execution_request,
      const DPMAgentExecutionRequest& logical_request,
      const DPMAgentReplayExecution& selected_winner) override;

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
  static absl::StatusOr<std::unique_ptr<ExactRegenerationDPMAgentRuntime>>
  Create(
      ExactRegenerationExecutor* exact_executor,
      CapsuleRestoreAdmissionBinding capsule_restore_admission);

  DPMReplayMode GetReplayMode() const override {
    return DPMReplayMode::kExactRegeneration;
  }
  const SessionHandoffIdentity& GetSessionHandoffIdentity() const override {
    return derived_profile_.session_identity;
  }
  absl::StatusOr<std::optional<Hash256>> GetExactProfileId() const override;
  absl::StatusOr<std::optional<Hash256>>
  GetExactProfileAdmissionRecordId() const override;
  absl::StatusOr<std::optional<Hash256>>
  GetCapsuleRestoreAdmissionRecordId() const override;
  absl::StatusOr<std::optional<Hash256>>
  GetSessionHandoffCapabilityId() const override;
  absl::StatusOr<DPMStageCapabilities> GetCapabilities() const override;
  absl::StatusOr<std::optional<SessionHandoffCapability>>
  GetSessionHandoffCapability() const override;
  absl::StatusOr<std::optional<CapsuleRestoreOperationalCoverage>>
  GetCapsuleRestoreOperationalCoverage() const override;
  absl::Status ValidateSupport() const override;
  absl::Status ValidateGenerationLimit(
      uint32_t max_output_tokens) const override;
  absl::Status ValidateSessionHandoffSupport() const override;
  absl::StatusOr<std::unique_ptr<Engine::Session>> CreateSession() override;
  absl::StatusOr<DPMAgentReplayExecution> Generate(
      Engine::Session* producing_session,
      const DPMAgentGenerationRequest& execution_request,
      const DPMAgentExecutionRequest& logical_request) override;
  absl::StatusOr<ExactRegenerationDPMAgentPhysicalExecution>
  GeneratePhysicalExact(
      const DPMAgentExecutionRequest& logical_request,
      const ExactRegenerationExecutionInput& input) override;

 private:
  ExactRegenerationDPMAgentRuntime(
      ExactRegenerationExecutor* exact_executor,
      ExactLiteRtProfile derived_profile,
      std::optional<CapsuleRestoreAdmissionBinding>
          capsule_restore_admission,
      std::optional<Hash256> capsule_restore_admission_record_id,
      std::optional<SessionHandoffCapability> session_handoff_capability,
      std::optional<CapsuleRestoreOperationalCoverage>
          capsule_restore_operational_coverage)
      : exact_executor_(exact_executor),
        derived_profile_(std::move(derived_profile)),
        capsule_restore_admission_(std::move(capsule_restore_admission)),
        capsule_restore_admission_record_id_(
            capsule_restore_admission_record_id),
        session_handoff_capability_(std::move(session_handoff_capability)),
        capsule_restore_operational_coverage_(
            std::move(capsule_restore_operational_coverage)) {}

  absl::StatusOr<AuthenticatedCapsuleRestoreAdmission>
  ResolveCurrentCapsuleRestoreAdmission() const;

  ExactRegenerationExecutor* const exact_executor_;
  const ExactLiteRtProfile derived_profile_;
  const std::optional<CapsuleRestoreAdmissionBinding>
      capsule_restore_admission_;
  const std::optional<Hash256> capsule_restore_admission_record_id_;
  const std::optional<SessionHandoffCapability>
      session_handoff_capability_;
  const std::optional<CapsuleRestoreOperationalCoverage>
      capsule_restore_operational_coverage_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_AGENT_REPLAY_RUNTIME_H_
