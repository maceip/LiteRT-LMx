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

#include "runtime/dpm/deterministic_dpm_fact_map_projector.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"       // from @com_google_absl
#include "absl/strings/string_view.h"   // from @com_google_absl
#include "runtime/dpm/correction_digest.h"
#include "runtime/dpm/dpm_capabilities.h"
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_fact_map.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_replay_executor.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kReducerContractDomain =
    "LITERT_LMX_DPM_FACT_MAP_REDUCER_CONTRACT_SHA256";
constexpr absl::string_view kReducerConfigDomain =
    "LITERT_LMX_DPM_FACT_MAP_PROJECTOR_CONFIG_SHA256";
constexpr absl::string_view kReducerRequestDomain =
    "LITERT_LMX_DPM_FACT_MAP_PROJECTOR_REQUEST_SHA256";
constexpr absl::string_view kReducerEvidenceDomain =
    "LITERT_LMX_DPM_FACT_MAP_PROJECTOR_EVIDENCE_SHA256";
constexpr absl::string_view kReducerReplayContract =
    "litert-lmx-dpm-deterministic-fact-map";
constexpr absl::string_view kUntypedEventPolicy =
    "untyped-events-do-not-mutate-structured-memory";
constexpr absl::string_view kModelTurnPolicy =
    "model-turns-do-not-project-their-own-output";

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

bool IsDecisionInputKind(DPMEvent::Kind kind) {
  return kind == DPMEvent::Kind::kUser || kind == DPMEvent::Kind::kTool ||
         kind == DPMEvent::Kind::kInternal;
}

void AppendU32(uint32_t value, std::string* output) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendU64(uint64_t value, std::string* output) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendFrame(absl::string_view value, std::string* output) {
  AppendU64(value.size(), output);
  output->append(value.data(), value.size());
}

void AppendHash(const Hash256& hash, std::string* output) {
  output->append(reinterpret_cast<const char*>(hash.bytes.data()),
                 hash.bytes.size());
}

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

absl::Status ValidateRuntimeIdentity(const SessionHandoffIdentity& identity) {
  if (IsZeroHash(identity.model_artifact_hash) ||
      IsZeroHash(identity.runtime_artifact_hash) ||
      IsZeroHash(identity.inference_profile_hash)) {
    return absl::FailedPreconditionError(
        "Deterministic DPM fact-map projection requires a complete loaded "
        "Engine identity.");
  }
  return absl::OkStatus();
}

absl::Status ValidateConfig(
    const DeterministicDPMFactMapProjectorConfig& config) {
  if (IsZeroHash(config.implementation_artifact_hash)) {
    return absl::InvalidArgumentError(
        "Deterministic DPM fact-map projection requires a measured "
        "implementation artifact hash.");
  }
  if (config.replay_token_bound == 0 ||
      config.replay_token_bound > kMaximumDPMGenerationTokens) {
    return absl::InvalidArgumentError(
        "Deterministic DPM fact-map replay token binding is outside the "
        "product bound.");
  }
  // The canonical empty map exercises all limit relationships without
  // inventing a second copy of the reducer's validation rules.
  ABSL_RETURN_IF_ERROR(CanonicalizeDPMFactMap(EmptyDPMFactMap(),
                                              /*source_event_count=*/0,
                                              config.limits)
                           .status());
  return absl::OkStatus();
}

absl::Status ValidateSnapshotShape(const DPMLogSnapshot& snapshot) {
  if (snapshot.log_id.empty() || snapshot.case_id.empty() ||
      snapshot.generation == 0 ||
      snapshot.generation != snapshot.events.size() ||
      snapshot.prefix_hashes.size() != snapshot.events.size() + 1 ||
      snapshot.prefix_hashes.empty() ||
      snapshot.prefix_hashes.back() != snapshot.prefix_hash ||
      IsZeroHash(snapshot.prefix_hash)) {
    return absl::DataLossError(
        "Deterministic DPM fact-map projection received an incomplete "
        "authoritative snapshot.");
  }
  for (const Hash256& prefix : snapshot.prefix_hashes) {
    if (IsZeroHash(prefix)) {
      return absl::DataLossError(
          "Deterministic DPM fact-map projection received an empty prefix "
          "proof.");
    }
  }
  for (uint64_t index = 0; index < snapshot.events.size(); ++index) {
    const DPMEvent& event = snapshot.events[static_cast<size_t>(index)];
    if (event.index != index || event.case_id != snapshot.case_id ||
        event.timestamp_us <= 0 || event.operation_id.empty() ||
        (event.kind != DPMEvent::Kind::kModelTurn && event.payload.empty())) {
      return absl::DataLossError(
          "Deterministic DPM fact-map projection received a contaminated "
          "event prefix.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> BuildCanonicalReducerPayload(
    const DPMLogSnapshot& snapshot, uint64_t input_event_index,
    const Hash256& correction_digest, const Hash256& config_hash,
    const SessionHandoffIdentity& runtime_identity,
    const Hash256& implementation_artifact_hash) {
  ABSL_RETURN_IF_ERROR(ValidateSnapshotShape(snapshot));
  if (input_event_index + 1 != snapshot.events.size() ||
      !IsDecisionInputKind(snapshot.events.back().kind)) {
    return absl::InvalidArgumentError(
        "Deterministic DPM fact-map projection requires the pending input as "
        "the final source event.");
  }
  constexpr size_t kPayloadHeadroom = 4096;
  static_assert(kMaximumFreshWorkerRequestPayloadBytes > kPayloadHeadroom);
  ABSL_ASSIGN_OR_RETURN(
      std::string rendered_events,
      RenderCanonicalDPMEventRange(
          snapshot, /*event_range_start=*/0, snapshot.events.size(),
          kMaximumFreshWorkerRequestPayloadBytes - kPayloadHeadroom));

  std::string payload;
  payload.reserve(rendered_events.size() + snapshot.log_id.size() +
                  snapshot.case_id.size() + 512);
  payload.append("DPMFMREQ");
  AppendU32(1, &payload);
  AppendFrame(snapshot.log_id, &payload);
  AppendFrame(snapshot.case_id, &payload);
  AppendU64(snapshot.events.size(), &payload);
  AppendHash(snapshot.prefix_hash, &payload);
  AppendU64(input_event_index, &payload);
  AppendHash(correction_digest, &payload);
  AppendHash(config_hash, &payload);
  AppendHash(implementation_artifact_hash, &payload);
  AppendHash(runtime_identity.model_artifact_hash, &payload);
  AppendHash(runtime_identity.runtime_artifact_hash, &payload);
  AppendHash(runtime_identity.inference_profile_hash, &payload);
  AppendFrame(rendered_events, &payload);
  if (payload.size() > kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Canonical deterministic DPM fact-map request exceeds the shared "
        "replay envelope.");
  }
  return payload;
}

Hash256 ComputeReducerRequestHash(absl::string_view canonical_payload) {
  Sha256Hasher hasher;
  hasher.Update(kReducerRequestDomain);
  hasher.Update(canonical_payload);
  return hasher.Finalize();
}

Hash256 ComputeReducerEvidenceHash(
    const SessionHandoffIdentity& runtime_identity,
    const Hash256& replay_request_hash, const Hash256& config_hash,
    absl::string_view canonical_output) {
  Sha256Hasher hasher;
  hasher.Update(kReducerEvidenceDomain);
  hasher.Update(
      absl::string_view(reinterpret_cast<const char*>(
                            runtime_identity.model_artifact_hash.bytes.data()),
                        runtime_identity.model_artifact_hash.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(
          runtime_identity.runtime_artifact_hash.bytes.data()),
      runtime_identity.runtime_artifact_hash.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(
          runtime_identity.inference_profile_hash.bytes.data()),
      runtime_identity.inference_profile_hash.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(replay_request_hash.bytes.data()),
      replay_request_hash.bytes.size()));
  hasher.Update(
      absl::string_view(reinterpret_cast<const char*>(config_hash.bytes.data()),
                        config_hash.bytes.size()));
  std::array<char, 8> output_size{};
  const uint64_t size = canonical_output.size();
  for (size_t index = 0; index < output_size.size(); ++index) {
    output_size[index] = static_cast<char>((size >> (56 - index * 8)) & 0xff);
  }
  hasher.Update(absl::string_view(output_size.data(), output_size.size()));
  hasher.Update(canonical_output);
  return hasher.Finalize();
}

class DeterministicReductionInvocation final
    : public CanonicalWinnerReplayGenerator {
 public:
  DeterministicReductionInvocation(
      const DPMLogSnapshot* snapshot,
      const DeterministicDPMFactMapProjectorConfig* config,
      const SessionHandoffIdentity* runtime_identity,
      const Hash256* config_hash, const DPMCanonicalReplayRequest* expected,
      const std::string* canonical_output)
      : snapshot_(snapshot),
        config_(config),
        runtime_identity_(runtime_identity),
        config_hash_(config_hash),
        expected_(expected),
        canonical_output_(canonical_output) {}

  const SessionHandoffIdentity& GetRuntimeIdentity() const override {
    return *runtime_identity_;
  }

  absl::Status ValidateSupport(DPMReplayStage expected_stage) const override {
    if (expected_stage != DPMReplayStage::kProjection || snapshot_ == nullptr ||
        config_ == nullptr || runtime_identity_ == nullptr ||
        config_hash_ == nullptr || expected_ == nullptr ||
        canonical_output_ == nullptr || canonical_output_->empty()) {
      return absl::InvalidArgumentError(
          "Deterministic DPM fact-map invocation is not bound to one "
          "projection request.");
    }
    ABSL_RETURN_IF_ERROR(ValidateSnapshotShape(*snapshot_));
    ABSL_RETURN_IF_ERROR(ValidateConfig(*config_));
    return ValidateRuntimeIdentity(*runtime_identity_);
  }

  absl::StatusOr<CanonicalWinnerGeneratedCandidate> Generate(
      const DPMCanonicalReplayRequest& request) override {
    ABSL_RETURN_IF_ERROR(ValidateSupport(DPMReplayStage::kProjection));
    if (request.format_version != expected_->format_version ||
        request.stage != expected_->stage ||
        request.max_output_tokens != expected_->max_output_tokens ||
        request.request_contract != expected_->request_contract ||
        request.canonical_payload != expected_->canonical_payload) {
      return absl::FailedPreconditionError(
          "Deterministic DPM fact-map generator received another canonical "
          "request.");
    }

    ABSL_ASSIGN_OR_RETURN(const Hash256 replay_request_hash,
                          ComputeDPMCanonicalReplayRequestHash(request));
    return CanonicalWinnerGeneratedCandidate{
        .canonical_output = *canonical_output_,
        .execution_evidence_hash =
            ComputeReducerEvidenceHash(*runtime_identity_, replay_request_hash,
                                       *config_hash_, *canonical_output_),
    };
  }

 private:
  const DPMLogSnapshot* const snapshot_;
  const DeterministicDPMFactMapProjectorConfig* const config_;
  const SessionHandoffIdentity* const runtime_identity_;
  const Hash256* const config_hash_;
  const DPMCanonicalReplayRequest* const expected_;
  const std::string* const canonical_output_;
};

}  // namespace

Hash256 GetDeterministicDPMFactMapReducerContractHash() {
  Sha256Hasher hasher;
  hasher.Update(kReducerContractDomain);
  hasher.Update(kDPMFactMapSchemaId);
  hasher.Update(kDPMToolDeltaSchemaId);
  hasher.Update(DPMFactMapJsonSchema());
  hasher.Update(kUntypedEventPolicy);
  hasher.Update(kModelTurnPolicy);
  return hasher.Finalize();
}

absl::StatusOr<Hash256> ComputeDeterministicDPMFactMapProjectorConfigHash(
    const DeterministicDPMFactMapProjectorConfig& config) {
  ABSL_RETURN_IF_ERROR(ValidateConfig(config));
  std::string canonical;
  canonical.reserve(4 + 32 + 32 + 8 + 4 + 8 + 4);
  AppendU32(1, &canonical);
  AppendHash(GetDeterministicDPMFactMapReducerContractHash(), &canonical);
  AppendHash(config.implementation_artifact_hash, &canonical);
  AppendU64(config.limits.memory_budget_bytes, &canonical);
  AppendU32(config.limits.max_facts, &canonical);
  AppendU64(config.limits.max_value_bytes, &canonical);
  AppendU32(config.replay_token_bound, &canonical);
  Sha256Hasher hasher;
  hasher.Update(kReducerConfigDomain);
  hasher.Update(canonical);
  return hasher.Finalize();
}

absl::StatusOr<std::unique_ptr<DeterministicDPMFactMapProjector>>
DeterministicDPMFactMapProjector::Create(
    DPMEventLog* authoritative_log,
    CanonicalWinnerReplayCatalog* replay_catalog,
    SessionHandoffIdentity runtime_identity,
    DeterministicDPMFactMapProjectorConfig config) {
  if (authoritative_log == nullptr || replay_catalog == nullptr) {
    return absl::InvalidArgumentError(
        "Deterministic DPM fact-map projection requires an authoritative log "
        "and replay catalog.");
  }
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(runtime_identity));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 config_hash,
      ComputeDeterministicDPMFactMapProjectorConfigHash(config));
  auto projector = std::unique_ptr<DeterministicDPMFactMapProjector>(
      new DeterministicDPMFactMapProjector(authoritative_log, replay_catalog,
                                           runtime_identity, config,
                                           config_hash));
  ABSL_RETURN_IF_ERROR(projector->ValidateSupport());
  return projector;
}

absl::Status DeterministicDPMFactMapProjector::ValidateSupport() const {
  if (authoritative_log_ == nullptr) {
    return absl::FailedPreconditionError(
        "Deterministic DPM fact-map projector lost its authoritative log.");
  }
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(runtime_identity_));
  ABSL_RETURN_IF_ERROR(ValidateConfig(config_));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 current_hash,
      ComputeDeterministicDPMFactMapProjectorConfigHash(config_));
  if (current_hash != config_hash_ || IsZeroHash(config_hash_)) {
    return absl::FailedPreconditionError(
        "Deterministic DPM fact-map projector configuration changed after "
        "construction.");
  }
  return absl::OkStatus();
}

absl::StatusOr<DPMReplayMode> DeterministicDPMFactMapProjector::GetReplayMode()
    const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  return DPMReplayMode::kCanonicalWinnerReplay;
}

absl::StatusOr<DPMStageCapabilities>
DeterministicDPMFactMapProjector::GetCapabilities() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  DPMStageCapabilities capabilities{
      .stage = DPMCapabilityStage::kProjection,
      .replay_mode = DPMReplayMode::kCanonicalWinnerReplay,
      .runtime_identity = runtime_identity_,
      .max_output_tokens = config_.replay_token_bound,
  };
  ABSL_RETURN_IF_ERROR(ValidateDPMStageCapabilities(capabilities));
  return capabilities;
}

absl::StatusOr<DPMLogSnapshot>
DeterministicDPMFactMapProjector::ResolveAuthoritativeSnapshot(
    const DPMProjectionRequest& request) const {
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot current, authoritative_log_->Snapshot());
  ABSL_RETURN_IF_ERROR(ValidateSnapshotShape(current));
  if (request.case_id.empty() || current.log_id != request.log.log_id ||
      current.case_id != request.log.case_id ||
      current.case_id != request.case_id ||
      current.generation != request.log.generation ||
      current.prefix_hash != request.log.prefix_hash ||
      request.input_event_index + 1 != current.events.size() ||
      current.events.back().index != request.input_event_index ||
      !IsDecisionInputKind(current.events.back().kind) ||
      current.events.back().turn_receipt.has_value()) {
    return absl::AbortedError(
        "Deterministic DPM fact-map projection request does not name the "
        "current authoritative pending input prefix.");
  }
  return current;
}

absl::StatusOr<DPMProjectionOutcome> DeterministicDPMFactMapProjector::Project(
    const DPMProjectionRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot snapshot,
                        ResolveAuthoritativeSnapshot(request));
  ABSL_ASSIGN_OR_RETURN(const Hash256 correction_digest,
                        ComputeDPMCorrectionDigest(snapshot));
  ABSL_ASSIGN_OR_RETURN(
      DPMFactMapProjectionFold fold,
      FoldDeterministicDPMEvents(EmptyDPMFactMap(), snapshot, /*range_start=*/0,
                                 snapshot.events.size(), config_.limits));
  std::optional<std::string> no_op_delta;
  std::optional<absl::string_view> raw_delta;
  if (!fold.llm_event_indices.empty()) {
    // Untyped events are deliberately not interpreted by an LLM in this
    // provider. An empty, canonical delta makes that policy explicit while
    // retaining exact source-event accounting in the fold.
    no_op_delta = EmptyDPMFactMap();
    raw_delta = *no_op_delta;
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string deterministic_memory,
      FinalizeDPMFactMapProjection(fold, raw_delta, snapshot.events.size(),
                                   config_.limits));
  ABSL_ASSIGN_OR_RETURN(
      std::string model_visible_memory,
      RenderDPMFactValues(deterministic_memory, snapshot.events.size(),
                          config_.limits));
  bool current_input_is_model_visible = false;
  for (uint64_t event_index : fold.llm_event_indices) {
    if (event_index == request.input_event_index) {
      current_input_is_model_visible = true;
      break;
    }
  }
  std::optional<std::string> model_visible_input;
  if (!current_input_is_model_visible) {
    // A typed fact delta has already been applied to model_visible_memory.
    // Repeating the machine envelope wastes context and can distract small
    // models from the semantic state they are supposed to consume.
    model_visible_input = "Memory updated.";
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string canonical_payload,
      BuildCanonicalReducerPayload(
          snapshot, request.input_event_index, correction_digest, config_hash_,
          runtime_identity_, config_.implementation_artifact_hash));
  const Hash256 reducer_request_hash =
      ComputeReducerRequestHash(canonical_payload);
  DPMCanonicalReplayRequest replay_request{
      .stage = DPMReplayStage::kProjection,
      .max_output_tokens = config_.replay_token_bound,
      .request_contract = std::string(kReducerReplayContract),
      .canonical_payload = std::move(canonical_payload),
  };
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_replay_request_hash,
                        ComputeDPMCanonicalReplayRequestHash(replay_request));
  DeterministicReductionInvocation invocation(
      &snapshot, &config_, &runtime_identity_, &config_hash_, &replay_request,
      &deterministic_memory);
  ABSL_ASSIGN_OR_RETURN(CanonicalWinnerReplayExecution execution,
                        replay_executor_.Run(replay_request, &invocation));
  if (execution.mode != DPMReplayMode::kCanonicalWinnerReplay ||
      execution.runtime_identity != runtime_identity_ ||
      execution.canonical_request_hash != expected_replay_request_hash ||
      execution.canonical_output_hash != Sha256(execution.canonical_output) ||
      IsZeroHash(execution.execution_evidence_hash)) {
    return absl::DataLossError(
        "Deterministic DPM fact-map replay returned another request, runtime, "
        "or output identity.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const std::string verified_memory,
      CanonicalizeDPMFactMap(execution.canonical_output, snapshot.events.size(),
                             config_.limits));
  if (verified_memory != execution.canonical_output ||
      verified_memory != deterministic_memory) {
    return absl::DataLossError(
        "Deterministic DPM fact-map replay returned memory other than the "
        "independently reduced canonical output.");
  }

  DPMProjectionManifest manifest{
      .log_id = snapshot.log_id,
      .case_id = snapshot.case_id,
      .source_event_count = snapshot.events.size(),
      .source_prefix_hash = snapshot.prefix_hash,
      .input_event_index = request.input_event_index,
      .event_range_start = 0,
      .correction_digest = correction_digest,
      .config_hash = config_hash_,
      .runtime_identity = runtime_identity_,
      .request_hash = reducer_request_hash,
      .output_hash =
          ComputeCanonicalDPMProjectionOutputHash(execution.canonical_output),
      .replay_mode = DPMReplayMode::kCanonicalWinnerReplay,
      .replay_request_hash = execution.canonical_request_hash,
      .execution_evidence_hash = execution.execution_evidence_hash,
  };
  ABSL_ASSIGN_OR_RETURN(manifest.manifest_hash,
                        ComputeDPMProjectionManifestHash(manifest));
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(manifest));
  return DPMProjectionOutcome{
      .projected_memory = std::move(execution.canonical_output),
      .manifest = std::move(manifest),
      .model_visible_memory = std::move(model_visible_memory),
      .model_visible_input = std::move(model_visible_input),
  };
}

}  // namespace litert::lm
