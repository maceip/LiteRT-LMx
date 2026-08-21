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
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/exact_profile_admission.h"
#include "runtime/dpm/fresh_worker_process.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

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
  ExactProfileAdmissionRecord cold_run_evidence;
  std::string canonical_output;
  std::string token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> logit_frames;
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

  absl::StatusOr<ExactRegenerationExecution> Run(
      const DPMCanonicalReplayRequest& request) const;

 private:
  ExactRegenerationExecutor(
      const Engine* engine,
      ExactProfileAdmissionRepository* admission_repository,
      ExactRegenerationExecutorConfig config,
      SessionConfig resolved_session_config,
      ExactLiteRtProfile derived_profile)
      : engine_(engine),
        worker_runner_(config.worker_process),
        admission_repository_(admission_repository),
        config_(std::move(config)),
        resolved_session_config_(std::move(resolved_session_config)),
        derived_profile_(std::move(derived_profile)) {}

  absl::Status ValidateBoundRequest(
      const DPMCanonicalReplayRequest& request) const;
  absl::StatusOr<ExactLiteRtProfile> ResolveCurrentProfile() const;
  ExactProfileQualificationSpec MakeQualificationSpec(
      absl::string_view encoded_request) const;

  const Engine* const engine_;
  const FreshWorkerProcessRunner worker_runner_;
  ExactProfileAdmissionRepository* const admission_repository_;
  const ExactRegenerationExecutorConfig config_;
  const SessionConfig resolved_session_config_;
  const ExactLiteRtProfile derived_profile_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_REPLAY_EXECUTOR_H_
