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

#include "runtime/dpm/dpm_receipt_validation.h"

#include <cstdint>
#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/session_checkpoint.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {
namespace {

constexpr uint64_t kMaximumCheckpointCapsuleBytes =
    uint64_t{8} * 1024 * 1024 * 1024;

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
}

absl::Status ValidateOptionalHash(const std::optional<Hash256>& hash,
                                  const char* field_name) {
  if (hash.has_value() && IsZeroHash(*hash)) {
    return absl::InvalidArgumentError(field_name);
  }
  return absl::OkStatus();
}

bool HasPostVersionThreeFields(const DPMTurnReceipt& receipt) {
  return receipt.restored_from_session_checkpoint_id.has_value() ||
         receipt.checkpoint_capture_origin !=
             DPMCheckpointCaptureOrigin::kNone ||
         receipt.agent_worker_prefill_mode !=
             DPMCheckpointWorkerPrefillMode::kNone ||
         !IsZeroHash(receipt.agent_physical_execution_plan_hash) ||
         receipt.agent_capsule_restore_admission_record_id.has_value() ||
         receipt.agent_capsule_restore_capability_id.has_value() ||
         receipt.agent_capsule_restore_coverage_id.has_value() ||
         receipt.agent_exact_worker_checkpoint_provenance.has_value() ||
         receipt.agent_prepared_prefill_work.has_value() ||
         receipt.published_checkpoint_capture.has_value() ||
         receipt.restored_checkpoint_capture_evidence_id.has_value() ||
         receipt.agent_capsule_restore_evidence_id.has_value() ||
         receipt.agent_rematerialized_canonical_winner;
}

bool HasPostVersionFiveFields(const DPMTurnReceipt& receipt) {
  return receipt.agent_prepared_prefill_work.has_value() ||
         receipt.published_checkpoint_capture.has_value() ||
         receipt.restored_checkpoint_capture_evidence_id.has_value() ||
         receipt.agent_capsule_restore_evidence_id.has_value();
}

bool HasCoverageV2ReceiptFields(uint32_t format_version) {
  return format_version == DPMTurnReceipt::kCoverageV2FormatVersion ||
         format_version == DPMTurnReceipt::kFormatVersion;
}

bool MatchesPreparedPrefillWorkHash(
    const Hash256& hash, const DPMPreparedPrefillWorkBinding& binding) {
  return hash == binding.plan_id ||
         hash == binding.canonical_source_chunks_hash ||
         hash == binding.resolved_token_plan_hash ||
         hash == binding.shape_schedule_hash;
}

absl::Status ValidateVersionedCapsuleRestoreProvenance(
    const DPMTurnReceipt& receipt) {
  const bool has_capability =
      receipt.agent_capsule_restore_capability_id.has_value();
  const bool has_admission =
      receipt.agent_capsule_restore_admission_record_id.has_value();
  const bool has_coverage =
      receipt.agent_capsule_restore_coverage_id.has_value();
  switch (receipt.format_version) {
    case DPMTurnReceipt::kLegacyFormatVersion:
      if (has_capability || has_admission || has_coverage) {
        return absl::InvalidArgumentError(
            "Version 3 DPM receipts cannot carry CapsuleRestore "
            "provenance.");
      }
      return absl::OkStatus();
    case DPMTurnReceipt::kPreviousFormatVersion:
      if (has_capability || has_coverage) {
        return absl::InvalidArgumentError(
            "Version 4 DPM receipts cannot carry a version 5 "
            "CapsuleRestore capability or coverage ID.");
      }
      return absl::OkStatus();
    case DPMTurnReceipt::kCoverageV1FormatVersion:
    case DPMTurnReceipt::kCoverageV2FormatVersion:
    case DPMTurnReceipt::kFormatVersion: {
      const bool uses_capsule =
          receipt.restored_from_session_checkpoint_id.has_value() ||
          receipt.session_checkpoint_id.has_value();
      if (uses_capsule != has_capability || uses_capsule != has_admission ||
          uses_capsule != has_coverage) {
        return absl::InvalidArgumentError(
            "Version 5+ capsule use requires matching CapsuleRestore "
            "capability, admission, and operational-coverage provenance.");
      }
      return absl::OkStatus();
    }
  }
  return absl::FailedPreconditionError(
      "Unsupported DPM turn receipt format version.");
}

absl::Status ValidateCoverageV2Evidence(const DPMTurnReceipt& receipt) {
  switch (receipt.format_version) {
    case DPMTurnReceipt::kLegacyFormatVersion:
    case DPMTurnReceipt::kPreviousFormatVersion:
    case DPMTurnReceipt::kCoverageV1FormatVersion:
      if (HasPostVersionFiveFields(receipt)) {
        return absl::InvalidArgumentError(
            "DPM receipt versions before 6 cannot carry Coverage V2 "
            "evidence.");
      }
      return absl::OkStatus();
    case DPMTurnReceipt::kCoverageV2FormatVersion:
    case DPMTurnReceipt::kFormatVersion:
      break;
    default:
      return absl::FailedPreconditionError(
          "Unsupported DPM turn receipt format version.");
  }

  const bool publishes_checkpoint = receipt.session_checkpoint_id.has_value();
  const bool restores_checkpoint =
      receipt.restored_from_session_checkpoint_id.has_value();
  if (publishes_checkpoint !=
      receipt.published_checkpoint_capture.has_value()) {
    return absl::InvalidArgumentError(
        "Version 6 checkpoint publication and capture-evidence binding must "
        "be present together.");
  }
  if (restores_checkpoint !=
          receipt.restored_checkpoint_capture_evidence_id.has_value() ||
      restores_checkpoint !=
          receipt.agent_capsule_restore_evidence_id.has_value()) {
    return absl::InvalidArgumentError(
        "Version 6 checkpoint restore requires both its source-capture and "
        "operation restore-evidence IDs.");
  }

  if (receipt.published_checkpoint_capture.has_value()) {
    const DPMCheckpointCaptureEvidenceBinding& binding =
        *receipt.published_checkpoint_capture;
    if (IsZeroHash(binding.capture_plan_hash) ||
        IsZeroHash(binding.capture_evidence_id) ||
        binding.capture_plan_hash == binding.capture_evidence_id ||
        binding.capture_plan_hash == *receipt.session_checkpoint_id ||
        binding.capture_evidence_id == *receipt.session_checkpoint_id) {
      return absl::InvalidArgumentError(
          "Version 6 checkpoint capture-evidence binding is incomplete or "
          "substitutes one evidence domain for another.");
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateOptionalHash(
      receipt.restored_checkpoint_capture_evidence_id,
      "Version 6 restored checkpoint has an empty source-capture evidence "
      "ID."));
  ABSL_RETURN_IF_ERROR(ValidateOptionalHash(
      receipt.agent_capsule_restore_evidence_id,
      "Version 6 restored checkpoint has an empty operation restore-evidence "
      "ID."));
  if (restores_checkpoint &&
      *receipt.restored_checkpoint_capture_evidence_id ==
          *receipt.agent_capsule_restore_evidence_id) {
    return absl::InvalidArgumentError(
        "Version 6 restore substituted source-capture evidence for operation "
        "restore evidence.");
  }
  if (restores_checkpoint &&
      (*receipt.restored_checkpoint_capture_evidence_id ==
           *receipt.restored_from_session_checkpoint_id ||
       *receipt.agent_capsule_restore_evidence_id ==
           *receipt.restored_from_session_checkpoint_id)) {
    return absl::InvalidArgumentError(
        "Version 6 restore substituted a checkpoint ID for an evidence ID.");
  }
  if (publishes_checkpoint && restores_checkpoint) {
    const DPMCheckpointCaptureEvidenceBinding& published =
        *receipt.published_checkpoint_capture;
    const Hash256& source_capture =
        *receipt.restored_checkpoint_capture_evidence_id;
    const Hash256& restore = *receipt.agent_capsule_restore_evidence_id;
    if (published.capture_plan_hash == source_capture ||
        published.capture_plan_hash == restore ||
        published.capture_evidence_id == source_capture ||
        published.capture_evidence_id == restore) {
      return absl::InvalidArgumentError(
          "Version 6 descendant capture substituted parent restore evidence "
          "for child capture provenance.");
    }
  }

  if (publishes_checkpoint || restores_checkpoint) {
    if (!receipt.agent_prepared_prefill_work.has_value()) {
      return absl::InvalidArgumentError(
          "Version 6 capsule use requires runtime-derived prepared-prefill "
          "work.");
    }
  }
  if (receipt.agent_prepared_prefill_work.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillWorkBinding(
        *receipt.agent_prepared_prefill_work));
    const DPMPreparedPrefillWorkBinding& prepared =
        *receipt.agent_prepared_prefill_work;
    if ((receipt.published_checkpoint_capture.has_value() &&
         (MatchesPreparedPrefillWorkHash(
              receipt.published_checkpoint_capture->capture_plan_hash,
              prepared) ||
          MatchesPreparedPrefillWorkHash(
              receipt.published_checkpoint_capture->capture_evidence_id,
              prepared))) ||
        (receipt.restored_checkpoint_capture_evidence_id.has_value() &&
         MatchesPreparedPrefillWorkHash(
             *receipt.restored_checkpoint_capture_evidence_id, prepared)) ||
        (receipt.agent_capsule_restore_evidence_id.has_value() &&
         MatchesPreparedPrefillWorkHash(
             *receipt.agent_capsule_restore_evidence_id, prepared))) {
      return absl::InvalidArgumentError(
          "Version 6 receipt substituted prepared-work hashes for capsule "
          "capture or restore evidence.");
    }
    const DPMPreparedPrefillStartKind start_kind =
        prepared.start_kind;
    if ((restores_checkpoint &&
         start_kind != DPMPreparedPrefillStartKind::kOwnPositionRestore) ||
        (!restores_checkpoint &&
         start_kind != DPMPreparedPrefillStartKind::kFreshSession)) {
      return absl::InvalidArgumentError(
          "Version 6 prepared-prefill work disagrees with restored-checkpoint "
          "provenance.");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateCaptureOrigin(DPMCheckpointCaptureOrigin origin) {
  switch (origin) {
    case DPMCheckpointCaptureOrigin::kNone:
    case DPMCheckpointCaptureOrigin::kLiveParentSession:
    case DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      "DPM turn receipt has an unknown checkpoint capture origin.");
}

absl::Status ValidateWorkerPrefillMode(
    DPMCheckpointWorkerPrefillMode mode) {
  switch (mode) {
    case DPMCheckpointWorkerPrefillMode::kNone:
    case DPMCheckpointWorkerPrefillMode::kFullCanonicalPrefill:
    case DPMCheckpointWorkerPrefillMode::kOwnPositionCapsuleDelta:
      return absl::OkStatus();
  }
  return absl::InvalidArgumentError(
      "DPM turn receipt has an unknown exact worker prefill mode.");
}

absl::Status ValidateExactWorkerCheckpointProvenance(
    const DPMExactWorkerCheckpointProvenance& provenance,
    const Hash256& physical_execution_plan_hash,
    const Hash256& exact_output_evidence_hash) {
  if (provenance.run_index != 0 ||
      IsZeroHash(provenance.execution_plan_hash) ||
      provenance.execution_plan_hash != physical_execution_plan_hash ||
      IsZeroHash(provenance.request_envelope_hash) ||
      IsZeroHash(provenance.result_envelope_hash) ||
      provenance.transient_envelope_size == 0 ||
      provenance.transient_envelope_size > kMaximumCheckpointCapsuleBytes ||
      IsZeroHash(provenance.transient_envelope_hash) ||
      IsZeroHash(provenance.output_evidence_hash) ||
      provenance.output_evidence_hash != exact_output_evidence_hash) {
    return absl::InvalidArgumentError(
        "DPM exact checkpoint provenance is incomplete or disagrees with "
        "the request-scoped physical execution.");
  }
  return absl::OkStatus();
}

absl::Status ValidateWinnerReplayReceipt(const DPMTurnReceipt& receipt) {
  const bool is_published_live_winner =
      !receipt.agent_reused_canonical_winner &&
      !receipt.agent_rematerialized_canonical_winner &&
      receipt.agent_producing_session_matched_output;
  const bool is_catalog_only_winner =
      receipt.agent_reused_canonical_winner &&
      !receipt.agent_rematerialized_canonical_winner &&
      !receipt.agent_producing_session_matched_output;
  const bool is_rematerialized_catalog_winner =
      receipt.agent_reused_canonical_winner &&
      receipt.agent_rematerialized_canonical_winner &&
      receipt.agent_producing_session_matched_output;
  if (receipt.agent_exact_profile_id.has_value() ||
      receipt.agent_exact_profile_admission_record_id.has_value() ||
      receipt.agent_exact_output_evidence_hash.has_value() ||
      receipt.agent_exact_logit_frame_count != 0 ||
      (!is_published_live_winner && !is_catalog_only_winner &&
       !is_rematerialized_catalog_winner)) {
    return absl::InvalidArgumentError(
        "WinnerReplay agent receipt contains exact evidence or invalid "
        "live-parent producing-session provenance.");
  }

  if (receipt.format_version != DPMTurnReceipt::kFormatVersion &&
      receipt.agent_rematerialized_canonical_winner) {
    return absl::InvalidArgumentError(
        "DPM receipt versions before 7 cannot claim explicit catalog "
        "rematerialization provenance.");
  }

  if (receipt.format_version == DPMTurnReceipt::kLegacyFormatVersion) {
    if (receipt.session_checkpoint_id.has_value() &&
        !receipt.agent_producing_session_matched_output) {
      return absl::InvalidArgumentError(
          "Legacy WinnerReplay checkpoint is not attached to its producing "
          "parent session.");
    }
    return absl::OkStatus();
  }

  if (receipt.agent_worker_prefill_mode !=
          DPMCheckpointWorkerPrefillMode::kNone ||
      !IsZeroHash(receipt.agent_physical_execution_plan_hash) ||
      receipt.agent_exact_worker_checkpoint_provenance.has_value()) {
    return absl::InvalidArgumentError(
        "WinnerReplay agent receipt contains exact physical-execution "
        "evidence.");
  }
  if (receipt.format_version == DPMTurnReceipt::kPreviousFormatVersion &&
      receipt.agent_capsule_restore_admission_record_id.has_value()) {
    return absl::InvalidArgumentError(
        "Version 4 WinnerReplay receipts cannot claim CapsuleRestore "
        "admission.");
  }
  if (HasCoverageV2ReceiptFields(receipt.format_version) &&
      receipt.agent_prepared_prefill_work.has_value() !=
          receipt.agent_producing_session_matched_output) {
    return absl::InvalidArgumentError(
        "Version 6 WinnerReplay prepared work must describe exactly the live "
        "session that produced the selected output.");
  }
  if (receipt.restored_from_session_checkpoint_id.has_value() &&
      !receipt.agent_producing_session_matched_output) {
    return absl::InvalidArgumentError(
        "WinnerReplay cannot claim a restored live session for a replayed "
        "catalog winner.");
  }
  if (receipt.session_checkpoint_id.has_value()) {
    if (!receipt.agent_producing_session_matched_output ||
        receipt.checkpoint_capture_origin !=
            DPMCheckpointCaptureOrigin::kLiveParentSession) {
      return absl::InvalidArgumentError(
          "WinnerReplay checkpoint must come from the live parent session "
          "that produced the selected output.");
    }
  } else if (receipt.checkpoint_capture_origin !=
             DPMCheckpointCaptureOrigin::kNone) {
    return absl::InvalidArgumentError(
        "WinnerReplay receipt names a checkpoint capture origin without a "
        "published checkpoint.");
  }
  return absl::OkStatus();
}

absl::Status ValidateExactRegenerationReceipt(
    const DPMTurnReceipt& receipt) {
  if (!receipt.agent_exact_profile_id.has_value() ||
      IsZeroHash(*receipt.agent_exact_profile_id) ||
      !receipt.agent_exact_profile_admission_record_id.has_value() ||
      IsZeroHash(*receipt.agent_exact_profile_admission_record_id) ||
      !receipt.agent_exact_output_evidence_hash.has_value() ||
      IsZeroHash(*receipt.agent_exact_output_evidence_hash) ||
      receipt.agent_exact_logit_frame_count == 0 ||
      receipt.agent_exact_logit_frame_count !=
          receipt.decision_token_ids.size() ||
      receipt.agent_reused_canonical_winner ||
      receipt.agent_rematerialized_canonical_winner ||
      receipt.agent_producing_session_matched_output) {
    return absl::InvalidArgumentError(
        "ExactRegeneration agent receipt is missing ordered output, profile, "
        "admission, or request-scoped execution evidence.");
  }

  // Version 3 carried no physical plan or authenticated fresh-worker capsule
  // provenance. Preserve its historical acceptance, but never let an old
  // exact receipt claim a checkpoint.
  if (receipt.format_version == DPMTurnReceipt::kLegacyFormatVersion) {
    if (receipt.session_checkpoint_id.has_value()) {
      return absl::InvalidArgumentError(
          "Version 3 ExactRegeneration receipts cannot carry checkpoints.");
    }
    return absl::OkStatus();
  }

  if (IsZeroHash(receipt.agent_physical_execution_plan_hash)) {
    return absl::InvalidArgumentError(
        "ExactRegeneration receipt is missing its physical execution plan.");
  }
  if (HasCoverageV2ReceiptFields(receipt.format_version) &&
      !receipt.agent_prepared_prefill_work.has_value()) {
    return absl::InvalidArgumentError(
        "Version 6 ExactRegeneration receipt is missing its runtime-derived "
        "prepared-prefill work.");
  }
  if (receipt.agent_execution_evidence_hash ==
          *receipt.agent_exact_profile_admission_record_id ||
      receipt.agent_execution_evidence_hash ==
          *receipt.agent_exact_profile_id ||
      receipt.agent_execution_evidence_hash ==
          *receipt.agent_exact_output_evidence_hash ||
      receipt.agent_execution_evidence_hash ==
          receipt.agent_physical_execution_plan_hash) {
    return absl::InvalidArgumentError(
        "ExactRegeneration receipt substituted reusable profile, "
        "output, or plan evidence for its request-scoped execution ID.");
  }
  switch (receipt.agent_worker_prefill_mode) {
    case DPMCheckpointWorkerPrefillMode::kFullCanonicalPrefill:
      if (receipt.restored_from_session_checkpoint_id.has_value()) {
        return absl::InvalidArgumentError(
            "Full-prefill ExactRegeneration receipt cannot claim a restored "
            "checkpoint.");
      }
      if (HasCoverageV2ReceiptFields(receipt.format_version) &&
          receipt.agent_prepared_prefill_work->start_kind !=
              DPMPreparedPrefillStartKind::kFreshSession) {
        return absl::InvalidArgumentError(
            "Version 6 exact full-prefill mode disagrees with prepared work.");
      }
      break;
    case DPMCheckpointWorkerPrefillMode::kOwnPositionCapsuleDelta:
      if (!receipt.restored_from_session_checkpoint_id.has_value()) {
        return absl::InvalidArgumentError(
            "Delta ExactRegeneration receipt must name its own-position "
            "restored checkpoint.");
      }
      if (HasCoverageV2ReceiptFields(receipt.format_version) &&
          receipt.agent_prepared_prefill_work->start_kind !=
              DPMPreparedPrefillStartKind::kOwnPositionRestore) {
        return absl::InvalidArgumentError(
            "Version 6 exact delta mode disagrees with prepared work.");
      }
      break;
    case DPMCheckpointWorkerPrefillMode::kNone:
      return absl::InvalidArgumentError(
          "ExactRegeneration receipt is missing its worker prefill "
          "mode.");
    default:
      return absl::InvalidArgumentError(
          "ExactRegeneration receipt has an unknown worker "
          "prefill mode.");
  }

  if (receipt.format_version == DPMTurnReceipt::kPreviousFormatVersion) {
    const bool uses_capsule =
        receipt.restored_from_session_checkpoint_id.has_value() ||
        receipt.session_checkpoint_id.has_value();
    if (uses_capsule !=
        receipt.agent_capsule_restore_admission_record_id.has_value()) {
      return absl::InvalidArgumentError(
          "Version 4 ExactRegeneration capsule use and CapsuleRestore "
          "admission do not match.");
    }
  }
  if (receipt.session_checkpoint_id.has_value()) {
    if (receipt.checkpoint_capture_origin !=
            DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker ||
        !receipt.agent_exact_worker_checkpoint_provenance.has_value()) {
      return absl::InvalidArgumentError(
          "ExactRegeneration checkpoint lacks authenticated run-zero worker "
          "provenance.");
    }
    ABSL_RETURN_IF_ERROR(ValidateExactWorkerCheckpointProvenance(
        *receipt.agent_exact_worker_checkpoint_provenance,
        receipt.agent_physical_execution_plan_hash,
        *receipt.agent_exact_output_evidence_hash));
  } else if (receipt.checkpoint_capture_origin !=
                 DPMCheckpointCaptureOrigin::kNone ||
             receipt.agent_exact_worker_checkpoint_provenance.has_value()) {
    return absl::InvalidArgumentError(
        "ExactRegeneration receipt carries capture provenance without a "
        "published checkpoint.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateDPMTurnReceiptReplayEvidence(
    const DPMTurnReceipt& receipt) {
  switch (receipt.format_version) {
    case DPMTurnReceipt::kLegacyFormatVersion:
    case DPMTurnReceipt::kPreviousFormatVersion:
    case DPMTurnReceipt::kCoverageV1FormatVersion:
    case DPMTurnReceipt::kCoverageV2FormatVersion:
    case DPMTurnReceipt::kFormatVersion:
      break;
    default:
      return absl::FailedPreconditionError(
          "Unsupported DPM turn receipt format version.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(receipt.agent_replay_mode));
  ABSL_RETURN_IF_ERROR(
      ValidateOptionalHash(receipt.session_checkpoint_id,
                           "DPM turn receipt contains an empty checkpoint "
                           "identity."));
  ABSL_RETURN_IF_ERROR(ValidateOptionalHash(
      receipt.restored_from_session_checkpoint_id,
      "DPM turn receipt contains an empty restored-checkpoint identity."));
  ABSL_RETURN_IF_ERROR(ValidateOptionalHash(
      receipt.agent_capsule_restore_admission_record_id,
      "DPM turn receipt contains an empty CapsuleRestore admission "
      "identity."));
  ABSL_RETURN_IF_ERROR(ValidateOptionalHash(
      receipt.agent_capsule_restore_capability_id,
      "DPM turn receipt contains an empty CapsuleRestore capability "
      "identity."));
  ABSL_RETURN_IF_ERROR(ValidateOptionalHash(
      receipt.agent_capsule_restore_coverage_id,
      "DPM turn receipt contains an empty CapsuleRestore operational "
      "coverage identity."));
  ABSL_RETURN_IF_ERROR(ValidateVersionedCapsuleRestoreProvenance(receipt));
  ABSL_RETURN_IF_ERROR(ValidateCoverageV2Evidence(receipt));
  if (receipt.session_checkpoint_id.has_value() &&
      receipt.restored_from_session_checkpoint_id.has_value() &&
      *receipt.session_checkpoint_id ==
          *receipt.restored_from_session_checkpoint_id) {
    return absl::InvalidArgumentError(
        "DPM turn receipt cannot publish and restore the same checkpoint.");
  }
  if (receipt.agent_replay_mode != receipt.projection_manifest.replay_mode ||
      IsZeroHash(receipt.agent_replay_request_hash) ||
      IsZeroHash(receipt.agent_execution_evidence_hash)) {
    return absl::InvalidArgumentError(
        "DPM agent receipt has incomplete or mixed-mode request-scoped "
        "execution evidence.");
  }
  if (receipt.format_version == DPMTurnReceipt::kLegacyFormatVersion &&
      HasPostVersionThreeFields(receipt)) {
    return absl::InvalidArgumentError(
        "Version 3 DPM receipt contains version 4 physical provenance.");
  }
  if (receipt.format_version == DPMTurnReceipt::kPreviousFormatVersion ||
      receipt.format_version == DPMTurnReceipt::kCoverageV1FormatVersion ||
      receipt.format_version == DPMTurnReceipt::kCoverageV2FormatVersion ||
      receipt.format_version == DPMTurnReceipt::kFormatVersion) {
    ABSL_RETURN_IF_ERROR(
        ValidateCaptureOrigin(receipt.checkpoint_capture_origin));
    ABSL_RETURN_IF_ERROR(
        ValidateWorkerPrefillMode(receipt.agent_worker_prefill_mode));
  }

  switch (receipt.agent_replay_mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      return ValidateWinnerReplayReceipt(receipt);
    case DPMReplayMode::kExactRegeneration:
      return ValidateExactRegenerationReceipt(receipt);
  }
  return absl::InvalidArgumentError("Unknown DPM agent replay mode.");
}

}  // namespace litert::lm
