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

#include "runtime/dpm/one_shot_dpm_projector.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/correction_digest.h"
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/exact_profile_admission.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_projection_replay_runtime.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

OneShotDPMProjector::OneShotDPMProjector(
    DPMEventLog* authoritative_log, DPMProjectionReplayRuntime* runtime,
    DPMProjectionConfig config)
    : authoritative_log_(authoritative_log),
      runtime_(runtime),
      config_(std::move(config)) {}

absl::Status OneShotDPMProjector::ValidateSupport() const {
  if (authoritative_log_ == nullptr || runtime_ == nullptr) {
    return absl::InvalidArgumentError(
        "One-shot DPM projection requires a raw event log and runtime.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionConfig(config_));
  if (runtime_->GetMaxOutputTokens() != config_.max_output_tokens) {
    return absl::FailedPreconditionError(
        "DPM projection runtime token limit differs from canonical config.");
  }
  ABSL_RETURN_IF_ERROR(runtime_->ValidateSupport());
  const DPMReplayMode replay_mode = runtime_->GetReplayMode();
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(replay_mode));
  ABSL_ASSIGN_OR_RETURN(const std::optional<Hash256> exact_profile_id,
                        runtime_->GetExactProfileId());
  if ((replay_mode == DPMReplayMode::kCanonicalWinnerReplay &&
       exact_profile_id.has_value()) ||
      (replay_mode == DPMReplayMode::kExactRegeneration &&
       (!exact_profile_id.has_value() || *exact_profile_id == Hash256{}))) {
    return absl::FailedPreconditionError(
        "DPM projection replay mode and derived exact-profile identity "
        "disagree.");
  }
  return absl::OkStatus();
}

absl::StatusOr<DPMReplayMode> OneShotDPMProjector::GetReplayMode() const {
  if (runtime_ == nullptr) {
    return absl::FailedPreconditionError(
        "One-shot DPM projection has no resolved replay runtime.");
  }
  const DPMReplayMode replay_mode = runtime_->GetReplayMode();
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(replay_mode));
  return replay_mode;
}

absl::StatusOr<DPMLogSnapshot>
OneShotDPMProjector::ResolveAuthoritativeSnapshot(
    const DPMProjectionRequest& request) const {
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot current,
                        authoritative_log_->Snapshot());
  if (request.case_id.empty() || current.log_id != request.log.log_id ||
      current.case_id != request.log.case_id ||
      current.case_id != request.case_id ||
      current.generation != request.log.generation ||
      current.prefix_hash != request.log.prefix_hash ||
      current.generation != current.events.size() || current.events.empty() ||
      current.prefix_hashes.size() != current.events.size() + 1 ||
      current.prefix_hashes.back() != current.prefix_hash ||
      request.input_event_index != current.events.size() - 1 ||
      current.events.back().index != request.input_event_index ||
      current.events.back().case_id != current.case_id ||
      (current.events.back().kind != DPMEvent::Kind::kUser &&
       current.events.back().kind != DPMEvent::Kind::kTool &&
       current.events.back().kind != DPMEvent::Kind::kInternal) ||
      current.events.back().turn_receipt.has_value()) {
    return absl::AbortedError(
        "DPM projection request does not name the current authoritative "
        "pending input prefix.");
  }
  return current;
}

absl::Status OneShotDPMProjector::ValidateBaseline(
    const DPMProjectionRequest& request,
    const Hash256& correction_digest, const Hash256& config_hash,
    const SessionHandoffIdentity& runtime_identity,
    DPMReplayMode replay_mode,
    const std::optional<Hash256>& exact_profile_id,
    const DPMProjectionBaselineArtifact& baseline) const {
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(baseline.manifest));
  if (baseline.manifest.log_id != request.log.log_id ||
      baseline.manifest.case_id != request.log.case_id ||
      baseline.manifest.source_event_count == 0 ||
      baseline.manifest.source_event_count >= request.log.events.size() ||
      baseline.manifest.input_event_index + 1 !=
          baseline.manifest.source_event_count ||
      baseline.manifest.config_hash != config_hash ||
      baseline.manifest.runtime_identity != runtime_identity ||
      baseline.manifest.replay_mode != replay_mode ||
      baseline.manifest.exact_profile_id != exact_profile_id) {
    return absl::FailedPreconditionError(
        "DPM projection baseline does not match the current log lineage and "
        "resolved projection profile.");
  }
  // The baseline's source count is exactly the new prompt's event_range_start;
  // never renumber the remaining events from zero.
  const uint64_t event_range_start = baseline.manifest.source_event_count;
  if (request.log.prefix_hashes.size() != request.log.events.size() + 1) {
    return absl::DataLossError(
        "DPM authoritative snapshot has no complete prefix-proof index.");
  }
  const Hash256 authoritative_baseline_prefix =
      request.log.prefix_hashes[static_cast<size_t>(event_range_start)];
  if (authoritative_baseline_prefix != baseline.manifest.source_prefix_hash) {
    return absl::DataLossError(
        "DPM projection baseline does not derive from the current raw-log "
        "prefix.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 baseline_correction_digest,
      ComputeDPMCorrectionDigestForPrefix(request.log, event_range_start));
  if (baseline.manifest.correction_digest != baseline_correction_digest) {
    return absl::FailedPreconditionError(
        "DPM projection baseline correction lineage does not match its own "
        "authoritative prefix.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 current_correction_digest,
      ComputeDPMCorrectionDigest(request.log));
  if (current_correction_digest != correction_digest) {
    return absl::DataLossError(
        "DPM projection correction lineage changed during baseline "
        "selection.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string canonical_baseline,
      CanonicalizeDPMProjectionOutput(baseline.projected_memory,
                                      event_range_start, config_));
  if (canonical_baseline != baseline.projected_memory ||
      ComputeCanonicalDPMProjectionOutputHash(canonical_baseline) !=
          baseline.manifest.output_hash) {
    return absl::DataLossError(
        "DPM projection baseline bytes do not match their output hash.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<DPMProjectionBaselineArtifact>>
OneShotDPMProjector::SelectNewestCompatibleBaseline(
    const DPMProjectionRequest& authoritative_request,
    const Hash256& correction_digest, const Hash256& config_hash,
    const SessionHandoffIdentity& runtime_identity,
    DPMReplayMode replay_mode,
    const std::optional<Hash256>& exact_profile_id) const {
  // Reverse durable response-event order is the unique selection rule. The
  // first compatible receipt is therefore the newest compatible ancestor for
  // every process observing this exact raw-log prefix.
  for (auto event = authoritative_request.log.events.rbegin();
       event != authoritative_request.log.events.rend(); ++event) {
    if (event->kind != DPMEvent::Kind::kModelTurn ||
        !event->turn_receipt.has_value()) {
      continue;
    }
    const DPMTurnReceipt& receipt = *event->turn_receipt;
    if ((receipt.format_version != DPMTurnReceipt::kLegacyFormatVersion &&
         receipt.format_version != DPMTurnReceipt::kFormatVersion) ||
        receipt.operation_id.empty() ||
        event->index >= authoritative_request.log.events.size() ||
        receipt.response_event_index != event->index ||
        receipt.input_event_index >= event->index ||
        receipt.input_event_index >= authoritative_request.log.events.size() ||
        receipt.input_event_index + 1 != event->index ||
        receipt.operation_id != event->operation_id ||
        receipt.decision_output != event->payload ||
        receipt.projection_manifest.source_event_count != event->index ||
        receipt.projection_manifest.input_event_index !=
            receipt.input_event_index) {
      continue;
    }
    const DPMEvent& input =
        authoritative_request.log.events[receipt.input_event_index];
    if ((input.kind != DPMEvent::Kind::kUser &&
         input.kind != DPMEvent::Kind::kTool &&
         input.kind != DPMEvent::Kind::kInternal) ||
        input.turn_receipt.has_value() ||
        input.index != receipt.input_event_index ||
        input.operation_id != receipt.operation_id ||
        input.case_id != event->case_id ||
        event->case_id != authoritative_request.log.case_id ||
        event->timestamp_us < input.timestamp_us) {
      continue;
    }
    DPMProjectionBaselineArtifact candidate{
        .manifest = receipt.projection_manifest,
        .projected_memory = receipt.projected_memory,
    };
    const absl::Status compatibility = ValidateBaseline(
        authoritative_request, correction_digest, config_hash,
        runtime_identity, replay_mode, exact_profile_id, candidate);
    if (compatibility.ok()) {
      return std::optional<DPMProjectionBaselineArtifact>(
          std::move(candidate));
    }
    // Projection receipts are disposable derivatives. A malformed manifest,
    // stale prefix, invalid correction lineage, invalid output, or any other
    // candidate-specific failure is a cache miss. Scanning continues across
    // corrections because a receipt before a correction remains the last
    // valid ancestor: the absolute delta begins at that response and contains
    // every correction needed to advance its lineage to the current digest.
  }
  return std::optional<DPMProjectionBaselineArtifact>();
}

absl::StatusOr<DPMProjectionOutcome> OneShotDPMProjector::Project(
    const DPMProjectionRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot authoritative_snapshot,
                        ResolveAuthoritativeSnapshot(request));
  DPMProjectionRequest authoritative_request{
      .log = std::move(authoritative_snapshot),
      .input_event_index = request.input_event_index,
      .case_id = request.case_id,
  };
  ABSL_ASSIGN_OR_RETURN(Hash256 correction_digest,
                        ComputeDPMCorrectionDigest(authoritative_request.log));
  ABSL_ASSIGN_OR_RETURN(Hash256 config_hash,
                        ComputeDPMProjectionConfigHash(config_));
  const SessionHandoffIdentity runtime_identity =
      runtime_->GetRuntimeIdentity();
  const DPMReplayMode replay_mode = runtime_->GetReplayMode();
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(replay_mode));
  ABSL_ASSIGN_OR_RETURN(const std::optional<Hash256> exact_profile_id,
                        runtime_->GetExactProfileId());
  ABSL_ASSIGN_OR_RETURN(
      std::optional<DPMProjectionBaselineArtifact> baseline,
      SelectNewestCompatibleBaseline(authoritative_request, correction_digest,
                                     config_hash, runtime_identity,
                                     replay_mode, exact_profile_id));

  ABSL_ASSIGN_OR_RETURN(
      CanonicalDPMProjectionRequest canonical_request,
      BuildCanonicalDPMProjectionRequest(
          authoritative_request.log, authoritative_request.input_event_index,
          correction_digest, config_, runtime_identity, baseline));
  ABSL_ASSIGN_OR_RETURN(
      Hash256 verified_request_hash,
      ComputeCanonicalDPMProjectionRequestHash(canonical_request));
  if (verified_request_hash != canonical_request.request_hash) {
    return absl::DataLossError(
        "Canonical DPM projection request changed before inference.");
  }

  // The one-shot contract has exactly this single product execution call.
  // Invalid output is returned as an error below; there is no hidden repair
  // inference. ExactRegeneration owns its N cold processes inside this call;
  // WinnerReplay may resolve an authenticated existing winner with zero model
  // calls.
  ABSL_ASSIGN_OR_RETURN(
      DPMProjectionReplayExecution execution,
      runtime_->Generate(canonical_request));
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionReplayExecution(execution));
  if (execution.mode != replay_mode ||
      execution.exact_profile_id != exact_profile_id ||
      execution.replay_request_hash == Hash256{} ||
      execution.execution_evidence_hash == Hash256{}) {
    return absl::DataLossError(
        "DPM projection execution returned incomplete or cross-mode "
        "evidence.");
  }
  std::optional<Hash256> exact_output_evidence_hash;
  uint32_t exact_logit_frame_count = 0;
  switch (execution.mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      if (execution.exact_profile_admission_record_id.has_value() ||
          !execution.exact_token_bytes.empty() ||
          !execution.exact_logit_frames.empty()) {
        return absl::DataLossError(
            "WinnerReplay projection returned exact-only evidence.");
      }
      break;
    case DPMReplayMode::kExactRegeneration:
      if (!execution.exact_profile_admission_record_id.has_value() ||
          *execution.exact_profile_admission_record_id == Hash256{} ||
          execution.exact_token_bytes.empty() ||
          execution.exact_logit_frames.empty() ||
          execution.exact_logit_frames.size() >
              std::numeric_limits<uint32_t>::max()) {
        return absl::DataLossError(
            "Exact projection returned incomplete token, logits, or profile "
            "admission evidence.");
      }
      exact_output_evidence_hash = ComputeFreshWorkerOutputEvidenceHash(
          execution.canonical_output, execution.exact_token_bytes,
          execution.exact_logit_frames);
      exact_logit_frame_count =
          static_cast<uint32_t>(execution.exact_logit_frames.size());
      break;
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string projected_memory,
      CanonicalizeDPMProjectionOutput(execution.canonical_output,
                                      canonical_request.source_event_count,
                                      config_));
  if (projected_memory != execution.canonical_output) {
    return absl::DataLossError(
        "DPM projection replay runtime returned non-canonical output.");
  }
  const Hash256 output_hash =
      ComputeCanonicalDPMProjectionOutputHash(projected_memory);

  // Revalidate both mutable surroundings before publishing a derivative.
  ABSL_RETURN_IF_ERROR(runtime_->ValidateSupport());
  if (runtime_->GetRuntimeIdentity() != runtime_identity) {
    return absl::FailedPreconditionError(
        "DPM projection runtime identity changed during inference.");
  }
  if (runtime_->GetReplayMode() != replay_mode) {
    return absl::FailedPreconditionError(
        "DPM projection replay mode changed during execution.");
  }
  ABSL_ASSIGN_OR_RETURN(const std::optional<Hash256> final_exact_profile_id,
                        runtime_->GetExactProfileId());
  if (final_exact_profile_id != exact_profile_id) {
    return absl::FailedPreconditionError(
        "DPM projection exact profile changed during execution.");
  }
  absl::StatusOr<DPMLogSnapshot> unchanged_snapshot =
      ResolveAuthoritativeSnapshot(authoritative_request);
  if (!unchanged_snapshot.ok()) return unchanged_snapshot.status();

  DPMProjectionManifest manifest{
      .log_id = canonical_request.log_id,
      .case_id = canonical_request.case_id,
      .source_event_count = canonical_request.source_event_count,
      .source_prefix_hash = canonical_request.source_prefix_hash,
      .input_event_index = canonical_request.input_event_index,
      .event_range_start = canonical_request.event_range_start,
      .baseline_manifest_hash = canonical_request.baseline_manifest_hash,
      .baseline_output_hash = canonical_request.baseline_output_hash,
      .correction_digest = canonical_request.correction_digest,
      .config_hash = canonical_request.config_hash,
      .runtime_identity = canonical_request.runtime_identity,
      .request_hash = canonical_request.request_hash,
      .output_hash = output_hash,
      .replay_mode = execution.mode,
      .replay_request_hash = execution.replay_request_hash,
      .execution_evidence_hash = execution.execution_evidence_hash,
      .exact_profile_id = execution.exact_profile_id,
      .exact_profile_admission_record_id =
          execution.exact_profile_admission_record_id,
      .exact_output_evidence_hash = exact_output_evidence_hash,
      .exact_logit_frame_count = exact_logit_frame_count,
      .reused_canonical_winner = execution.reused_canonical_winner,
  };
  ABSL_ASSIGN_OR_RETURN(manifest.manifest_hash,
                        ComputeDPMProjectionManifestHash(manifest));
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(manifest));

  return DPMProjectionOutcome{
      .projected_memory = std::move(projected_memory),
      .manifest = std::move(manifest),
  };
}

}  // namespace litert::lm
