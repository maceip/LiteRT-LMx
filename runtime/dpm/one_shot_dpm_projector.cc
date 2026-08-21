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
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/correction_digest.h"
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_projection_runtime.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

OneShotDPMProjector::OneShotDPMProjector(
    DPMEventLog* authoritative_log, DPMProjectionRuntime* runtime,
    DPMProjectionConfig config)
    : authoritative_log_(authoritative_log),
      runtime_(runtime),
      config_(std::move(config)) {}

absl::Status OneShotDPMProjector::ValidateConfiguration() const {
  if (authoritative_log_ == nullptr || runtime_ == nullptr) {
    return absl::InvalidArgumentError(
        "One-shot DPM projection requires a raw event log and runtime.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionConfig(config_));
  if (runtime_->GetMaxOutputTokens() != config_.max_output_tokens) {
    return absl::FailedPreconditionError(
        "DPM projection runtime token limit differs from canonical config.");
  }
  return runtime_->ValidateSupport();
}

absl::Status OneShotDPMProjector::ValidateCurrentSnapshot(
    const DPMProjectionRequest& request) const {
  if (request.case_id.empty() || request.case_id != request.log.case_id ||
      request.input_event_index >= request.log.events.size() ||
      request.input_event_index + 1 != request.log.events.size() ||
      request.log.generation != request.log.events.size() ||
      request.log.events[request.input_event_index].index !=
          request.input_event_index ||
      request.log.events[request.input_event_index].case_id != request.case_id) {
    return absl::InvalidArgumentError(
        "DPM projection request does not name the final input in one "
        "authoritative case prefix.");
  }
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot current,
                        authoritative_log_->Snapshot());
  if (current.log_id != request.log.log_id ||
      current.case_id != request.log.case_id ||
      current.generation != request.log.generation ||
      current.prefix_hash != request.log.prefix_hash) {
    return absl::AbortedError(
        "DPM projection request no longer matches the authoritative log.");
  }
  if (current.events.size() != request.log.events.size()) {
    return absl::AbortedError(
        "DPM projection request event count changed during admission.");
  }
  for (size_t index = 0; index < current.events.size(); ++index) {
    const DPMEvent& authoritative = current.events[index];
    const DPMEvent& supplied = request.log.events[index];
    if (authoritative.index != supplied.index ||
        authoritative.kind != supplied.kind ||
        authoritative.timestamp_us != supplied.timestamp_us ||
        authoritative.operation_id != supplied.operation_id ||
        authoritative.case_id != supplied.case_id ||
        authoritative.payload != supplied.payload) {
      return absl::DataLossError(
          "DPM projection request event bytes differ from the raw log.");
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      Hash256 authoritative_prefix,
      authoritative_log_->PrefixHash(request.log.events.size()));
  if (authoritative_prefix != request.log.prefix_hash) {
    return absl::DataLossError(
        "DPM projection source prefix hash differs from the raw event log.");
  }
  return absl::OkStatus();
}

absl::Status OneShotDPMProjector::ValidateBaseline(
    const DPMProjectionRequest& request,
    const Hash256& correction_digest, const Hash256& config_hash,
    const SessionHandoffIdentity& runtime_identity,
    const DPMProjectionBaselineArtifact& baseline) const {
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(baseline.manifest));
  if (baseline.manifest.log_id != request.log.log_id ||
      baseline.manifest.case_id != request.log.case_id ||
      baseline.manifest.source_event_count == 0 ||
      baseline.manifest.source_event_count >= request.log.events.size() ||
      baseline.manifest.input_event_index + 1 !=
          baseline.manifest.source_event_count ||
      baseline.manifest.correction_digest != correction_digest ||
      baseline.manifest.config_hash != config_hash ||
      baseline.manifest.runtime_identity != runtime_identity) {
    return absl::FailedPreconditionError(
        "DPM projection baseline does not match the current log lineage and "
        "resolved projection profile.");
  }
  // The baseline's source count is exactly the new prompt's event_range_start;
  // never renumber the remaining events from zero.
  const uint64_t event_range_start = baseline.manifest.source_event_count;
  ABSL_ASSIGN_OR_RETURN(
      Hash256 authoritative_baseline_prefix,
      authoritative_log_->PrefixHash(event_range_start));
  if (authoritative_baseline_prefix != baseline.manifest.source_prefix_hash) {
    return absl::FailedPreconditionError(
        "DPM projection baseline does not derive from the current raw-log "
        "prefix.");
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

absl::StatusOr<OneShotDPMProjectionResult> OneShotDPMProjector::Project(
    const DPMProjectionRequest& request,
    std::optional<DPMProjectionBaselineArtifact> baseline) {
  ABSL_RETURN_IF_ERROR(ValidateConfiguration());
  ABSL_RETURN_IF_ERROR(ValidateCurrentSnapshot(request));
  ABSL_ASSIGN_OR_RETURN(Hash256 correction_digest,
                        ComputeDPMCorrectionDigest(request.log));
  ABSL_ASSIGN_OR_RETURN(Hash256 config_hash,
                        ComputeDPMProjectionConfigHash(config_));
  const SessionHandoffIdentity runtime_identity =
      runtime_->GetRuntimeIdentity();

  bool used_baseline = false;
  if (baseline.has_value()) {
    const absl::Status baseline_status =
        ValidateBaseline(request, correction_digest, config_hash,
                         runtime_identity, *baseline);
    if (baseline_status.ok()) {
      used_baseline = true;
    } else {
      // Projection artifacts are disposable. Any stale, corrupt, or otherwise
      // incompatible candidate is ignored and the raw log is projected from
      // event zero. No weaker cached answer is returned.
      baseline.reset();
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      CanonicalDPMProjectionRequest canonical_request,
      BuildCanonicalDPMProjectionRequest(
          request.log, request.input_event_index, correction_digest, config_,
          runtime_identity, baseline));
  ABSL_ASSIGN_OR_RETURN(
      Hash256 verified_request_hash,
      ComputeCanonicalDPMProjectionRequestHash(canonical_request));
  if (verified_request_hash != canonical_request.request_hash) {
    return absl::DataLossError(
        "Canonical DPM projection request changed before inference.");
  }

  // The one-shot contract has exactly this single runtime call. Invalid model
  // output is returned as an error below; there is no hidden repair inference.
  ABSL_ASSIGN_OR_RETURN(
      std::string raw_output,
      runtime_->GenerateFresh(canonical_request.prompt_bytes));
  ABSL_ASSIGN_OR_RETURN(
      std::string projected_memory,
      CanonicalizeDPMProjectionOutput(raw_output,
                                      canonical_request.source_event_count,
                                      config_));
  const Hash256 output_hash =
      ComputeCanonicalDPMProjectionOutputHash(projected_memory);

  // Revalidate both mutable surroundings before publishing a derivative.
  ABSL_RETURN_IF_ERROR(runtime_->ValidateSupport());
  if (runtime_->GetRuntimeIdentity() != runtime_identity) {
    return absl::FailedPreconditionError(
        "DPM projection runtime identity changed during inference.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCurrentSnapshot(request));

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
  };
  ABSL_ASSIGN_OR_RETURN(manifest.manifest_hash,
                        ComputeDPMProjectionManifestHash(manifest));
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(manifest));

  DPMProjectionOutcome outcome{
      .projected_memory = std::move(projected_memory),
      .canonical_request_hash = manifest.request_hash,
      .manifest_hash = manifest.manifest_hash,
      .correction_digest = manifest.correction_digest,
  };
  return OneShotDPMProjectionResult{
      .outcome = std::move(outcome),
      .manifest = std::move(manifest),
      .used_baseline = used_baseline,
  };
}

}  // namespace litert::lm
