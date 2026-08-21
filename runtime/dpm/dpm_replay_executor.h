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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_REPLAY_EXECUTOR_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_REPLAY_EXECUTOR_H_

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
#include "runtime/dpm/capsule_restore_admission.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/exact_profile_admission.h"
#include "runtime/dpm/fresh_worker_process.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"
#include "runtime/util/byte_stream.h"

namespace litert::lm {

inline constexpr uint32_t kDPMCanonicalReplayRequestFormatVersion = 2;
// Magic, format version, stage, maximum output tokens, contract byte length,
// and payload byte length.
inline constexpr uint64_t kDPMCanonicalReplayRequestFramingBytes =
    8 + 4 + 4 + 4 + 4 + 8;

// Stage-aware request bytes supplied to exactly one of the two disjoint replay
// executors below. `canonical_payload` is the complete stage request, not an
// untrusted digest. The executor computes its hash from the versioned encoding.
struct DPMCanonicalReplayRequest {
  uint32_t format_version = kDPMCanonicalReplayRequestFormatVersion;
  DPMReplayStage stage = DPMReplayStage::kProjection;
  // Authenticated outer binding for parent-side admission before spawning a
  // worker. Stage codecs must independently require this to equal the limit
  // carried by their complete canonical payload.
  uint32_t max_output_tokens = 0;
  std::string request_contract_version;
  std::string canonical_payload;
};

absl::Status ValidateDPMCanonicalReplayRequest(
    const DPMCanonicalReplayRequest& request);
absl::StatusOr<std::string> EncodeDPMCanonicalReplayRequest(
    const DPMCanonicalReplayRequest& request);
absl::StatusOr<DPMCanonicalReplayRequest> DecodeDPMCanonicalReplayRequest(
    absl::string_view bytes);
absl::StatusOr<Hash256> ComputeDPMCanonicalReplayRequestHash(
    const DPMCanonicalReplayRequest& request);

// Output of a real model invocation that has already been canonicalized by a
// stage-specific production adapter. WinnerReplay does not reinterpret these
// bytes. The evidence hash is a deterministic binding of the request,
// runtime identity, and selected bytes; it does not cryptographically prove
// that inference occurred, is not independent-run evidence, and must never be
// presented as ExactRegeneration admission.
struct CanonicalWinnerGeneratedCandidate {
  std::string canonical_output;
  Hash256 execution_evidence_hash;
};

// Production generators own a loaded runtime and expose only its derived
// identity. There is no setter or caller-provided identity on this boundary.
class CanonicalWinnerReplayGenerator {
 public:
  virtual ~CanonicalWinnerReplayGenerator() = default;
  virtual const SessionHandoffIdentity& GetRuntimeIdentity() const = 0;
  virtual absl::Status ValidateSupport(
      DPMReplayStage expected_stage) const = 0;
  virtual absl::StatusOr<CanonicalWinnerGeneratedCandidate> Generate(
      const DPMCanonicalReplayRequest& request) = 0;
};

enum class CanonicalWinnerResolution : uint8_t {
  kReplayed = 1,
  kPublished = 2,
  kLostPublishRace = 3,
};

struct CanonicalWinnerReplayExecution {
  DPMReplayMode mode = DPMReplayMode::kCanonicalWinnerReplay;
  CanonicalWinnerResolution resolution =
      CanonicalWinnerResolution::kReplayed;
  SessionHandoffIdentity runtime_identity;
  Hash256 canonical_request_hash;
  std::string canonical_output;
  Hash256 canonical_output_hash;
  Hash256 execution_evidence_hash;

  // True only when this call generated and atomically published the exact
  // returned candidate. A catalog hit or losing publisher has no producing
  // session corresponding to the selected winner and cannot be checkpointed
  // as if it did.
  bool producing_session_matches_output = false;
};

// Publish-once replay. This type necessarily owns a catalog and has no exact
// profile, admission repository, or fresh-worker dependency.
class CanonicalWinnerReplayExecutor final {
 public:
  explicit CanonicalWinnerReplayExecutor(
      CanonicalWinnerReplayCatalog* catalog)
      : catalog_(catalog) {}

  // The generator is supplied per operation. This lets agent-decision replay
  // bind a live, turn-owned producing session without storing ephemeral
  // request/session state in a shared catalog executor.
  absl::Status ValidateSupport(CanonicalWinnerReplayGenerator* generator,
                               DPMReplayStage expected_stage) const;
  absl::StatusOr<CanonicalWinnerReplayExecution> Run(
      const DPMCanonicalReplayRequest& request,
      CanonicalWinnerReplayGenerator* generator);

 private:
  CanonicalWinnerReplayCatalog* const catalog_;
};

struct ExactRegenerationExecutorConfig {
  DPMReplayStage stage = DPMReplayStage::kProjection;
  uint32_t max_output_tokens = 0;
  FreshWorkerProcessOptions worker_process;
  ExactLiteRtProfileAssertion profile_assertion;
  FreshWorkerAuthentication authentication;
  uint32_t independent_run_count = 2;
};

// Capture is deliberately a request-scoped physical-execution policy rather
// than part of the model-affecting execution-plan hash. When capture is
// requested, only run zero may export a producing capsule; the remaining
// independent runs must reproduce the same output without exporting state.
enum class ExactRegenerationCaptureRunPolicy : uint32_t {
  kNoCapture = 1,
  kRunZeroOnly = 2,
};

// Canonical evidence for one authenticated fresh-process observation. The
// evidence ID is SHA-256 over every field below except evidence_id itself.
// Envelope hashes bind the complete HMAC-authenticated request and result;
// worker_certification_hash binds the parent-measured executable image and
// launch contract. This compact record does not duplicate model output bytes.
struct ExactRegenerationRunEvidence {
  static constexpr uint32_t kFormatVersion = 4;

  uint32_t format_version = kFormatVersion;
  Hash256 evidence_id;
  uint32_t run_index = 0;
  int64_t process_id = -1;
  Hash256 challenge_nonce;
  Hash256 worker_instance_nonce;
  Hash256 request_envelope_hash;
  Hash256 result_envelope_hash;
  Hash256 worker_certification_hash;
  Hash256 launch_spec_hash;
  Hash256 output_evidence_hash;
  std::optional<Hash256> restored_checkpoint_id;
  // Exact agent workers retain the runtime-derived physical prefill plan from
  // their authenticated result. Projection workers carry neither this plan,
  // a restored-state witness, nor restore provenance. Restore workers retain
  // their independently recomputed live-target witness; its envelope and
  // witness IDs may differ across otherwise agreeing cold runs.
  std::optional<DPMPreparedPrefillPlan> prepared_prefill_plan;
  std::optional<SessionContinuationStateWitness> restored_state_witness;
  // Complete parent-computed provenance for the selected durable checkpoint
  // rewrap. Present exactly for an own-position restore. Independent workers
  // may have different transient destination envelopes, while the durable
  // source endpoint and canonical continuation-state commitment must agree.
  std::optional<SessionHandoffReauthenticationEvidence>
      restore_reauthentication_evidence;
  std::optional<FreshWorkerProducingCapsuleEvidence>
      transient_producing_capsule_evidence;
  std::optional<FreshWorkerDurableProducingCapsuleEvidence>
      durable_producing_capsule_evidence;
};

absl::StatusOr<Hash256> ComputeExactRegenerationRunEvidenceId(
    const ExactRegenerationRunEvidence& evidence);
absl::Status ValidateExactRegenerationRunEvidence(
    const ExactRegenerationRunEvidence& evidence);

// Request-scoped equality evidence. Unlike ExactProfileAdmissionRecord, this
// object is never inserted into the durable profile-admission repository and
// cannot qualify another request. Its canonical ID binds the admitted exact
// profile, certified worker, complete logical request, physical work
// selection, and every independently authenticated process observation.
struct ExactRegenerationRequestEvidence {
  static constexpr uint32_t kFormatVersion = 4;

  uint32_t format_version = kFormatVersion;
  Hash256 evidence_id;
  Hash256 exact_profile_id;
  Hash256 worker_certification_hash;
  Hash256 profile_admission_record_id;
  SessionHandoffIdentity session_identity;
  DPMReplayStage stage = DPMReplayStage::kProjection;
  Hash256 request_execution_id;
  Hash256 canonical_request_hash;
  Hash256 physical_execution_plan_hash;
  FreshWorkerPrefillMode prefill_mode =
      FreshWorkerPrefillMode::kFullCanonicalPrefill;
  std::optional<Hash256> restored_checkpoint_id;
  ExactRegenerationCaptureRunPolicy capture_run_policy =
      ExactRegenerationCaptureRunPolicy::kNoCapture;
  FreshWorkerReplayIsolation replay_isolation =
      FreshWorkerReplayIsolation::kEmptyCatalogs;
  uint32_t run_count = 0;
  std::string authentication_key_id;
  Hash256 consensus_output_evidence_hash;
  // Present only for exact agent requests. These are the model-affecting
  // prepared-prefill commitments on which every independent worker agreed.
  // Per-run plan IDs and restored witness IDs remain intentionally distinct.
  std::optional<Hash256> agent_logical_request_hash;
  std::optional<Hash256> consensus_source_chunks_hash;
  std::optional<Hash256> consensus_resolved_token_plan_hash;
  std::optional<Hash256> consensus_shape_schedule_hash;
  std::vector<ExactRegenerationRunEvidence> runs;
};

absl::StatusOr<Hash256> ComputeExactRegenerationRequestEvidenceId(
    const ExactRegenerationRequestEvidence& evidence);
absl::Status ValidateExactRegenerationRequestEvidence(
    const ExactRegenerationRequestEvidence& evidence);

// Complete physical execution input. Delta plans require both restore
// pointers; capture plans require both staging pointers. A staging sink must
// be initially empty, transactional, and unpublished: the caller must discard
// all appended bytes if RunPhysical fails and may promote them create-once
// only after success. Restore and staging storage must not alias.
struct ExactRegenerationExecutionInput {
  FreshWorkerExecutionPlan execution_plan;
  const ByteSource* durable_restore_source = nullptr;
  const SessionHandoffOptions* durable_restore_options = nullptr;
  ByteSink* staging_capture_destination = nullptr;
  const SessionHandoffOptions* staging_capture_options = nullptr;
};

struct ExactRegenerationExecution {
  DPMReplayMode mode = DPMReplayMode::kExactRegeneration;
  ExactLiteRtProfile derived_profile;
  Hash256 canonical_request_hash;

  // Durable, authenticated evidence that this exact profile passed at least
  // one explicitly scoped cold-process qualification request.
  Hash256 profile_admission_record_id;

  // Fresh, non-catalog equality evidence for this exact request. It is not
  // inserted into the profile-admission repository and cannot become a
  // publish-once winner by accident.
  ExactRegenerationRequestEvidence request_evidence;
  std::string canonical_output;
  std::string token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> logit_frames;

  // Run zero's representative runtime-derived plan. It is exposed only after
  // all cold runs agree on the source, resolved-token, shape, profile, and
  // logical-request commitments. Projection execution leaves it absent.
  std::optional<DPMPreparedPrefillPlan> prepared_prefill_plan;

  // Exposed only after every independent run has passed equality. Capsule
  // bytes remain in the caller-owned staging sink; this is authenticated
  // metadata for promoting that sink transactionally.
  std::optional<FreshWorkerDurableProducingCapsuleEvidence>
      durable_producing_capsule_evidence;
};

// Catalog-free cold-process comparison. This type deliberately has no catalog
// pointer, catalog constructor parameter, catalog lookup, or catalog
// publication path. It derives identity from `engine`, authenticates a prior
// profile admission, then compares N new processes for the actual canonical
// request. A product may call the result ExactRegeneration only when the
// worker executable installs the concrete Engine-owning, catalog-free stage
// adapter; the generic worker callback cannot establish that fact by itself.
class ExactRegenerationExecutor final {
 public:
  // Admission is an explicit operation because Create accepts only an
  // already-admitted profile. This path still owns the concrete process
  // runner locally and cannot be redirected to the lower-level runner seam.
  static absl::StatusOr<ExactProfileAdmissionRecord> QualifyAndAdmit(
      const Engine* engine,
      ExactProfileAdmissionRepository* admission_repository,
      const ExactRegenerationExecutorConfig& config,
      const DPMCanonicalReplayRequest& qualification_request);

  // Constructs the product executor only after resolving the stage-bound
  // Engine profile and authenticating its durable admission. The owned
  // FreshWorkerProcessRunner spawns one new process per observed run.
  static absl::StatusOr<std::unique_ptr<ExactRegenerationExecutor>> Create(
      const Engine* engine,
      ExactProfileAdmissionRepository* admission_repository,
      ExactRegenerationExecutorConfig config);

  absl::Status ValidateSupport() const;

  DPMReplayStage GetReplayStage() const { return config_.stage; }
  uint32_t GetMaxOutputTokens() const { return config_.max_output_tokens; }

  // Returns the current Engine-derived identity pinned at Create. Create has
  // already authenticated admission, but the profile alone still does not
  // imply that a particular product request regenerated exactly.
  absl::StatusOr<ExactLiteRtProfile> GetDerivedProfile() const;

  // Re-resolves the Engine profile and re-authenticates the durable admission
  // repository entry. This lets a parent validate an old exact checkpoint
  // descriptor before spawning any restored worker.
  absl::StatusOr<Hash256> GetProfileAdmissionRecordId() const;

  // Resolves the binding's SessionConfig against this same loaded Engine,
  // requires its complete exact profile/session semantics to equal the
  // executor's immutable profile, derives the handoff capability, and asks
  // the repository to reauthenticate and revalidate the bound record.
  absl::StatusOr<AuthenticatedCapsuleRestoreAdmission>
  GetAuthenticatedCapsuleRestoreAdmission(
      const CapsuleRestoreAdmissionBinding& binding) const;

  absl::StatusOr<ExactRegenerationExecution> Run(
      const DPMCanonicalReplayRequest& request) const;

  // Executes the caller-selected validated full-prefill or own-position
  // delta plan through the concrete session-capable process boundary. There
  // is no implicit full-prefill fallback. Every restored run consumes the
  // same caller-owned durable source; only run zero may capture.
  absl::StatusOr<ExactRegenerationExecution> RunPhysical(
      const DPMCanonicalReplayRequest& request,
      const ExactRegenerationExecutionInput& input) const;

  // The admitted overload is mandatory for restore or capture. The overload
  // above remains available only for physical full-prefill without capsule
  // transfer, preserving exact execution without checkpoint support.
  absl::StatusOr<ExactRegenerationExecution> RunPhysical(
      const DPMCanonicalReplayRequest& request,
      const ExactRegenerationExecutionInput& input,
      const CapsuleRestoreAdmissionBinding& capsule_restore_admission) const;

 private:
  ExactRegenerationExecutor(
      const Engine* engine,
      ExactProfileAdmissionRepository* admission_repository,
      ExactRegenerationExecutorConfig config,
      FreshWorkerProcessRunner worker_runner,
      SessionConfig resolved_session_config,
      ExactLiteRtProfile derived_profile)
      : engine_(engine),
        worker_runner_(std::move(worker_runner)),
        admission_repository_(admission_repository),
        config_(std::move(config)),
        resolved_session_config_(std::move(resolved_session_config)),
        derived_profile_(std::move(derived_profile)) {}

  absl::Status ValidateBoundRequest(
      const DPMCanonicalReplayRequest& request) const;
  absl::StatusOr<ExactLiteRtProfile> ResolveCurrentProfile() const;
  absl::StatusOr<ExactRegenerationExecution> RunWithExecutionInput(
      const DPMCanonicalReplayRequest& request,
      const ExactRegenerationExecutionInput& input,
      bool capsule_free_convenience,
      const CapsuleRestoreAdmissionBinding* capsule_restore_admission) const;

  const Engine* const engine_;
  const FreshWorkerProcessRunner worker_runner_;
  ExactProfileAdmissionRepository* const admission_repository_;
  const ExactRegenerationExecutorConfig config_;
  const SessionConfig resolved_session_config_;
  const ExactLiteRtProfile derived_profile_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_REPLAY_EXECUTOR_H_
