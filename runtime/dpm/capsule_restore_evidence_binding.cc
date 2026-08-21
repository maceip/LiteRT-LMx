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

#include "runtime/dpm/capsule_restore_evidence_binding.h"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "runtime/dpm/dpm_capabilities.h"
#include "runtime/dpm/dpm_prepared_prefill_plan.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_receipt_validation.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/session_handoff_capability.h"

namespace litert::lm {
namespace {

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

struct NamedHash {
  const char* name;
  Hash256 value;
};

absl::Status ValidateDistinctHashDomains(
    std::initializer_list<NamedHash> hashes) {
  for (auto first = hashes.begin(); first != hashes.end(); ++first) {
    if (IsZeroHash(first->value)) {
      return absl::InvalidArgumentError(
          absl::StrCat("CapsuleRestore binding has an empty ", first->name,
                       " hash."));
    }
    for (auto second = first + 1; second != hashes.end(); ++second) {
      if (first->value == second->value) {
        return absl::InvalidArgumentError(absl::StrCat(
            "CapsuleRestore binding substituted ", first->name, " for ",
            second->name, "."));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateCurrentV3Admission(
    const AuthenticatedCapsuleRestoreStateWitnessAdmission& admission) {
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(admission.profile));
  ABSL_RETURN_IF_ERROR(
      ValidateSessionHandoffCapability(admission.capability));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreStateWitnessAdmissionRecord(admission.record));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreStateWitnessOperationalCoverage(
          admission.operational_coverage));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreStateWitnessOperationalContracts(
          admission.operational_coverage));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreStateWitnessAdmissionRecordForRuntime(
          admission.record, admission.profile, admission.capability,
          admission.operational_coverage.coverage_id));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 canonical_record_id,
      ComputeCapsuleRestoreStateWitnessAdmissionRecordId(admission.record));

  const CapsuleRestoreStateWitnessOperationalCoverage& coverage =
      admission.operational_coverage;
  const CapsuleRestoreStateWitnessOperationalDomain& domain =
      coverage.operational_domain;
  if (admission.record.record_id != canonical_record_id ||
      admission.record.operational_coverage != coverage ||
      coverage.runtime_derived_profile != admission.profile ||
      coverage.runtime_derived_capability != admission.capability ||
      coverage.runtime_derived_session_identity !=
          admission.profile.session_identity ||
      coverage.runtime_derived_session_identity !=
          admission.capability.session_identity ||
      admission.capability.exact_profile_id != admission.profile.profile_id ||
      admission.capability.backend != admission.profile.backend ||
      admission.capability.capsule_codec_contract_hash !=
          GetSessionHandoffCapsuleCodecContractHash()) {
    return absl::FailedPreconditionError(
        "CapsuleRestore authority admission is detached from its canonical "
        "record, complete Engine-derived profile, capability, or session "
        "identity.");
  }

  // Repeat the operation-boundary identities explicitly after the central
  // operational-contract validator. This keeps this cross-object binder
  // fail-closed if a future aggregate validator is accidentally weakened or
  // routed to a legacy evidence contract.
  if (domain.resolved_session_config_hash !=
          admission.profile.session_identity.inference_profile_hash ||
      domain.admitted_backend != admission.profile.backend ||
      domain.session_continuation_state_witness_contract_hash !=
          GetSessionContinuationStateWitnessContractHash() ||
      domain.capture_evidence_contract_hash !=
          GetCapsuleRestoreCaptureEvidenceV3ContractHash() ||
      domain.restore_evidence_contract_hash !=
          GetCapsuleRestoreRestoreEvidenceV3ContractHash() ||
      domain.deterministic_prefill_plan_contract_hash !=
          GetDPMPreparedPrefillPlanContractHash() ||
      domain.execution_shape_class_contract_hash !=
          GetDPMPreparedPrefillShapeClassContractHash() ||
      domain.restricted_feature_contract_hash !=
          GetDPMRestrictedFeatureContractHash()) {
    return absl::FailedPreconditionError(
        "CapsuleRestore authority was not qualified against the current V3 "
        "capture, restore, witness, prepared-prefill, shape, and restricted-"
        "feature contracts.");
  }

  return ValidateDistinctHashDomains(
      {{"exact profile", admission.profile.profile_id},
       {"CapsuleRestore capability", admission.capability.capability_id},
       {"CapsuleRestore admission", admission.record.record_id},
       {"CapsuleRestore coverage", coverage.coverage_id},
       {"qualification evidence", coverage.qualification_evidence_hash}});
}

absl::Status ValidateAuthoritativeV6Receipt(
    const DPMTurnReceipt& receipt) {
  if (receipt.format_version != DPMTurnReceipt::kFormatVersion) {
    return absl::FailedPreconditionError(
        "CapsuleRestore evidence binding requires an authoritative version-6 "
        "turn receipt.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateDPMProjectionManifest(receipt.projection_manifest));
  ABSL_RETURN_IF_ERROR(ValidateDPMTurnReceiptReplayEvidence(receipt));
  if (receipt.input_event_index !=
          receipt.projection_manifest.input_event_index ||
      receipt.response_event_index !=
          receipt.projection_manifest.source_event_count ||
      receipt.max_decision_tokens == 0 ||
      IsZeroHash(receipt.agent_request_hash) ||
      IsZeroHash(receipt.agent_transcript_hash)) {
    return absl::InvalidArgumentError(
        "Version-6 turn receipt is internally inconsistent at its projection, "
        "response, agent-request, or transcript boundary.");
  }
  return absl::OkStatus();
}

absl::Status ValidateReceiptAuthority(
    const DPMTurnReceipt& receipt,
    const CapsuleRestoreAuthorityV2& authority) {
  if (!receipt.agent_capsule_restore_capability_id.has_value() ||
      !receipt.agent_capsule_restore_admission_record_id.has_value() ||
      !receipt.agent_capsule_restore_coverage_id.has_value() ||
      *receipt.agent_capsule_restore_capability_id !=
          authority.capability.capability_id ||
      *receipt.agent_capsule_restore_admission_record_id !=
          authority.admission_record_id ||
      *receipt.agent_capsule_restore_coverage_id != authority.coverage_id ||
      receipt.agent_session_identity !=
          authority.capability.session_identity) {
    return absl::FailedPreconditionError(
        "Turn receipt does not carry the current complete CapsuleRestore "
        "authority and runtime-owned session identity.");
  }
  return absl::OkStatus();
}

bool ExactWorkerProvenanceEqual(
    const DPMExactWorkerCheckpointProvenance& left,
    const DPMExactWorkerCheckpointProvenance& right) {
  return left.run_index == right.run_index &&
         left.execution_plan_hash == right.execution_plan_hash &&
         left.request_envelope_hash == right.request_envelope_hash &&
         left.result_envelope_hash == right.result_envelope_hash &&
         left.transient_envelope_size == right.transient_envelope_size &&
         left.transient_envelope_hash == right.transient_envelope_hash &&
         left.output_evidence_hash == right.output_evidence_hash;
}

DPMCheckpointWorkerPrefillMode WorkerModeForPrefill(
    CapsulePrefillModeV2 mode) {
  switch (mode) {
    case CapsulePrefillModeV2::kFullCanonicalPrefill:
      return DPMCheckpointWorkerPrefillMode::kFullCanonicalPrefill;
    case CapsulePrefillModeV2::kOwnPositionCapsuleDelta:
      return DPMCheckpointWorkerPrefillMode::kOwnPositionCapsuleDelta;
  }
  return DPMCheckpointWorkerPrefillMode::kNone;
}

absl::Status ValidateCanonicalPrefillWithinCoverage(
    const CapsulePrefillPlanV2& prefill,
    const CapsuleRestoreStateWitnessOperationalDomain& domain) {
  uint64_t text_bytes = 0;
  uint64_t token_ids = 0;
  uint32_t encoding_mask = 0;
  for (const CapsuleCanonicalPrefillChunkV2& chunk :
       prefill.canonical_chunks) {
    switch (chunk.encoding) {
      case CapsuleCanonicalPrefillChunkV2::Encoding::kUtf8Text:
        if (text_bytes > domain.maximum_prefill_text_bytes ||
            chunk.utf8_text.size() >
                domain.maximum_prefill_text_bytes - text_bytes) {
          return absl::FailedPreconditionError(
              "CapsuleRestore prepared text exceeds the current "
              "authenticated operational domain.");
        }
        text_bytes += chunk.utf8_text.size();
        encoding_mask |= CapsuleRestoreStateWitnessEncodingBit(
            CapsuleRestoreStateWitnessEncoding::kUtf8Text);
        break;
      case CapsuleCanonicalPrefillChunkV2::Encoding::kExactTokenIds:
        if (token_ids > domain.maximum_prefill_token_ids ||
            chunk.token_ids.size() >
                domain.maximum_prefill_token_ids - token_ids) {
          return absl::FailedPreconditionError(
              "CapsuleRestore prepared token input exceeds the current "
              "authenticated operational domain.");
        }
        token_ids += chunk.token_ids.size();
        encoding_mask |= CapsuleRestoreStateWitnessEncodingBit(
            CapsuleRestoreStateWitnessEncoding::kExactTokenIds);
        break;
      default:
        return absl::InvalidArgumentError(
            "CapsuleRestore prepared work has an unsupported source "
            "encoding.");
    }
  }
  if (prefill.canonical_chunks.size() < domain.minimum_prefill_chunks ||
      prefill.canonical_chunks.size() > domain.maximum_prefill_chunks ||
      encoding_mask == 0 ||
      (encoding_mask & ~domain.admitted_encoding_mask) != 0) {
    return absl::FailedPreconditionError(
        "CapsuleRestore prepared work is outside the current authenticated "
        "chunk-count or source-encoding domain.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCaptureWithinCoverage(
    const CapsuleCaptureEvidenceV3& capture,
    const CapsuleRestoreStateWitnessOperationalDomain& domain) {
  const CapsuleCapturePlanV2& plan = capture.plan;
  ABSL_RETURN_IF_ERROR(
      ValidateCanonicalPrefillWithinCoverage(plan.prefill, domain));
  if (plan.generated_token_count > domain.maximum_output_tokens ||
      plan.capture_end_step < domain.minimum_checkpoint_step ||
      plan.capture_end_step > domain.maximum_checkpoint_step ||
      plan.capture_end_step > domain.maximum_context_positions) {
    return absl::FailedPreconditionError(
        "CapsuleRestore capture is outside the current authenticated output, "
        "checkpoint-position, or context domain.");
  }
  if (plan.capture_basis ==
      CapsuleCaptureBasisV2::kVerifiedParentRestore) {
    if (plan.prefill.start_step < domain.minimum_checkpoint_step ||
        plan.prefill.start_step > domain.maximum_checkpoint_step ||
        plan.prefill.end_step <= plan.prefill.start_step) {
      return absl::FailedPreconditionError(
          "CapsuleRestore descendant capture starts outside the current "
          "authenticated own-position domain.");
    }
    const uint64_t delta_positions =
        static_cast<uint64_t>(plan.prefill.end_step) -
        plan.prefill.start_step;
    if (delta_positions < domain.minimum_delta_positions ||
        delta_positions > domain.maximum_delta_positions) {
      return absl::FailedPreconditionError(
          "CapsuleRestore descendant capture delta is outside the current "
          "authenticated operational domain.");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateRestoreWithinCoverage(
    const CapsuleRestoreEvidenceV3& restore,
    const CapsuleRestoreStateWitnessOperationalDomain& domain) {
  const CapsuleRestorePlanV2& plan = restore.plan;
  ABSL_RETURN_IF_ERROR(
      ValidateCanonicalPrefillWithinCoverage(plan.prefill, domain));
  if (plan.maximum_output_tokens > domain.maximum_output_tokens ||
      plan.checkpoint_step < domain.minimum_checkpoint_step ||
      plan.checkpoint_step > domain.maximum_checkpoint_step ||
      plan.prefill.end_step <= plan.prefill.start_step) {
    return absl::FailedPreconditionError(
        "CapsuleRestore operation is outside the current authenticated "
        "output or checkpoint-position domain.");
  }
  const uint64_t delta_positions =
      static_cast<uint64_t>(plan.prefill.end_step) - plan.prefill.start_step;
  if (delta_positions < domain.minimum_delta_positions ||
      delta_positions > domain.maximum_delta_positions ||
      plan.maximum_output_tokens > domain.maximum_context_positions ||
      plan.prefill.end_step >
          domain.maximum_context_positions - plan.maximum_output_tokens) {
    return absl::FailedPreconditionError(
        "CapsuleRestore operation delta or decode exceeds the current "
        "authenticated shape and context domain.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCaptureStateJoin(
    const DPMSessionCheckpointDescriptor& descriptor,
    const DPMTurnReceipt& receipt,
    const CapsuleCaptureEvidenceV3& capture) {
  const CapsuleCapturePlanV2& plan = capture.plan;
  const CapsuleDPMCheckpointStateV2& state = plan.checkpoint_state;
  const DPMProjectionManifest& manifest = receipt.projection_manifest;
  if (descriptor.log_id != state.log_id ||
      descriptor.log_id != manifest.log_id ||
      descriptor.source_event_count != state.source_event_count ||
      descriptor.source_event_count != manifest.source_event_count ||
      descriptor.source_prefix_hash != state.source_prefix_hash ||
      descriptor.source_prefix_hash != manifest.source_prefix_hash ||
      descriptor.response_event_index != state.response_event_index ||
      descriptor.response_event_index != receipt.response_event_index ||
      descriptor.projection_request_hash !=
          state.projection_request_hash ||
      descriptor.projection_request_hash != manifest.request_hash ||
      descriptor.projection_manifest_hash !=
          state.projection_manifest_hash ||
      descriptor.projection_manifest_hash != manifest.manifest_hash ||
      descriptor.correction_digest != state.correction_digest ||
      descriptor.correction_digest != manifest.correction_digest ||
      descriptor.agent_request_hash != state.logical_agent_request_hash ||
      descriptor.agent_request_hash != receipt.agent_request_hash ||
      descriptor.agent_transcript_hash != state.agent_transcript_hash ||
      descriptor.agent_transcript_hash != receipt.agent_transcript_hash) {
    return absl::FailedPreconditionError(
        "Checkpoint descriptor, capture plan, and authoritative receipt "
        "disagree on log prefix, projection, correction, logical request, "
        "or completed transcript provenance.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCaptureParentJoin(
    const DPMSessionCheckpointDescriptor& descriptor,
    const DPMTurnReceipt& receipt,
    const CapsuleCaptureEvidenceV3& capture) {
  const CapsuleCapturePlanV2& plan = capture.plan;
  switch (plan.capture_basis) {
    case CapsuleCaptureBasisV2::kRootFreshSession:
      if (descriptor.restored_from_checkpoint_id.has_value() ||
          receipt.restored_from_session_checkpoint_id.has_value() ||
          receipt.restored_checkpoint_capture_evidence_id.has_value() ||
          receipt.agent_capsule_restore_evidence_id.has_value()) {
        return absl::FailedPreconditionError(
            "Root checkpoint capture carries restored-parent provenance.");
      }
      return absl::OkStatus();
    case CapsuleCaptureBasisV2::kVerifiedParentRestore: {
      if (!plan.parent_checkpoint_id.has_value() ||
          !plan.parent_restore_evidence_id.has_value() ||
          !capture.parent_restore_evidence.has_value() ||
          descriptor.restored_from_checkpoint_id !=
              plan.parent_checkpoint_id ||
          receipt.restored_from_session_checkpoint_id !=
              plan.parent_checkpoint_id ||
          receipt.agent_capsule_restore_evidence_id !=
              plan.parent_restore_evidence_id ||
          receipt.agent_capsule_restore_evidence_id !=
              std::optional<Hash256>(
                  capture.parent_restore_evidence->evidence_id) ||
          receipt.restored_checkpoint_capture_evidence_id !=
              std::optional<Hash256>(capture.parent_restore_evidence->plan
                                         .source_capture_evidence_id)) {
        return absl::FailedPreconditionError(
            "Verified-parent capture is detached from the receipt's source "
            "checkpoint, source-capture evidence, or restore evidence.");
      }
      return absl::OkStatus();
    }
  }
  return absl::InvalidArgumentError(
      "Checkpoint capture has an unsupported parent basis.");
}

absl::Status ValidateCaptureModeJoin(
    const DPMSessionCheckpointDescriptor& descriptor,
    const DPMTurnReceipt& receipt,
    const CapsuleCaptureEvidenceV3& capture,
    const CapsuleRestoreAuthorityV2& authority) {
  const CapsuleCapturePlanV2& plan = capture.plan;
  if (descriptor.replay_mode != receipt.agent_replay_mode ||
      descriptor.capture_origin != receipt.checkpoint_capture_origin) {
    return absl::FailedPreconditionError(
        "Checkpoint descriptor and receipt disagree on replay mode or "
        "capture origin.");
  }
  if (plan.generated_token_count != receipt.decision_token_ids.size() ||
      plan.generated_token_count > receipt.max_decision_tokens) {
    return absl::FailedPreconditionError(
        "Capture plan token count differs from the authoritative decision "
        "tokens or exceeds the receipt limit.");
  }

  switch (receipt.agent_replay_mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      if (descriptor.capture_origin !=
              DPMCheckpointCaptureOrigin::kLiveParentSession ||
          descriptor.worker_prefill_mode !=
              DPMCheckpointWorkerPrefillMode::kNone ||
          plan.producing_output_evidence_hash !=
              receipt.agent_execution_evidence_hash) {
        return absl::FailedPreconditionError(
            "WinnerReplay capture lacks its live-parent origin or canonical "
            "output-bound execution evidence.");
      }
      return absl::OkStatus();

    case DPMReplayMode::kExactRegeneration: {
      if (!descriptor.exact_profile_id.has_value() ||
          !receipt.agent_exact_profile_id.has_value() ||
          !receipt.agent_exact_profile_admission_record_id.has_value() ||
          !receipt.agent_exact_output_evidence_hash.has_value() ||
          !descriptor.worker_provenance.has_value() ||
          !receipt.agent_exact_worker_checkpoint_provenance.has_value() ||
          descriptor.capture_origin !=
              DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker ||
          *descriptor.exact_profile_id !=
              authority.capability.exact_profile_id ||
          *receipt.agent_exact_profile_id !=
              authority.capability.exact_profile_id ||
          descriptor.exact_profile_admission_record_id !=
              *receipt.agent_exact_profile_admission_record_id ||
          descriptor.exact_request_execution_evidence_id !=
              receipt.agent_execution_evidence_hash ||
          descriptor.exact_output_evidence_hash !=
              *receipt.agent_exact_output_evidence_hash ||
          descriptor.exact_output_evidence_hash !=
              plan.producing_output_evidence_hash ||
          descriptor.execution_plan_hash !=
              receipt.agent_physical_execution_plan_hash ||
          descriptor.worker_prefill_mode !=
              WorkerModeForPrefill(plan.prefill.mode) ||
          receipt.agent_worker_prefill_mode !=
              descriptor.worker_prefill_mode ||
          !ExactWorkerProvenanceEqual(
              *descriptor.worker_provenance,
              *receipt.agent_exact_worker_checkpoint_provenance)) {
        return absl::FailedPreconditionError(
            "ExactRegeneration capture is detached from its current profile, "
            "cold-profile admission, request evidence, physical plan, output, "
            "or authenticated run-zero worker provenance.");
      }
      const DPMExactWorkerCheckpointProvenance& worker =
          *descriptor.worker_provenance;
      const SessionHandoffReauthenticationEvidence& reauthentication =
          capture.transient_to_durable_reauthentication;
      if (worker.transient_envelope_hash !=
              reauthentication.source_envelope_hash ||
          worker.transient_envelope_size !=
              reauthentication.source_envelope_size ||
          worker.output_evidence_hash !=
              plan.producing_output_evidence_hash ||
          worker.execution_plan_hash != descriptor.execution_plan_hash) {
        return absl::FailedPreconditionError(
            "Exact run-zero worker provenance does not describe the capture "
            "evidence's transient producer and selected physical plan.");
      }
      return absl::OkStatus();
    }
  }
  return absl::InvalidArgumentError(
      "Checkpoint capture has an unsupported replay mode.");
}

absl::Status ValidateCaptureHashDomains(
    const DPMSessionCheckpointDescriptor& descriptor,
    const DPMTurnReceipt& receipt,
    const CapsuleCaptureEvidenceV3& capture,
    const DPMPreparedPrefillWorkBinding& prepared,
    const CapsuleRestoreAuthorityV2& authority) {
  const CapsuleDPMCheckpointStateV2& state = capture.plan.checkpoint_state;
  const SessionHandoffReauthenticationEvidence& reauthentication =
      capture.transient_to_durable_reauthentication;
  std::vector<NamedHash> hashes = {
      {"checkpoint ID", descriptor.descriptor_id},
      {"capture-plan", capture.plan.plan_hash},
      {"capture-evidence", capture.evidence_id},
      {"durable envelope", descriptor.envelope_hash},
      {"processed-history", capture.checkpoint_history_token_bytes_hash},
      {"prepared plan", prepared.plan_id},
      {"prepared source", prepared.canonical_source_chunks_hash},
      {"prepared token plan", prepared.resolved_token_plan_hash},
      {"prepared shape schedule", prepared.shape_schedule_hash},
      {"exact profile", authority.capability.exact_profile_id},
      {"CapsuleRestore capability", authority.capability.capability_id},
      {"CapsuleRestore admission", authority.admission_record_id},
      {"CapsuleRestore coverage", authority.coverage_id},
      {"qualification evidence", authority.qualification_spec_hash},
      {"source-prefix", state.source_prefix_hash},
      {"projection request", state.projection_request_hash},
      {"projection manifest", state.projection_manifest_hash},
      {"correction", state.correction_digest},
      {"agent transcript", state.agent_transcript_hash},
      {"logical agent request", state.logical_agent_request_hash},
      {"agent replay request", receipt.agent_replay_request_hash},
      {"producing output", capture.plan.producing_output_evidence_hash},
      {"capture reauthentication", reauthentication.evidence_id},
      {"canonical continuation state",
       reauthentication.canonical_continuation_state_hash},
      {"transient producer envelope", reauthentication.source_envelope_hash},
  };
  if (receipt.agent_replay_mode == DPMReplayMode::kExactRegeneration) {
    hashes.push_back({"exact profile admission",
                      *receipt.agent_exact_profile_admission_record_id});
    hashes.push_back(
        {"exact request evidence", receipt.agent_execution_evidence_hash});
    hashes.push_back(
        {"exact physical plan", receipt.agent_physical_execution_plan_hash});
    const DPMExactWorkerCheckpointProvenance& worker =
        *receipt.agent_exact_worker_checkpoint_provenance;
    hashes.push_back(
        {"exact worker request envelope", worker.request_envelope_hash});
    hashes.push_back(
        {"exact worker result envelope", worker.result_envelope_hash});
  }
  for (std::size_t first = 0; first < hashes.size(); ++first) {
    if (IsZeroHash(hashes[first].value)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CapsuleRestore capture binding has an empty ", hashes[first].name,
          " hash."));
    }
    for (std::size_t second = first + 1; second < hashes.size(); ++second) {
      if (hashes[first].value == hashes[second].value) {
        return absl::InvalidArgumentError(absl::StrCat(
            "CapsuleRestore capture binding substituted ",
            hashes[first].name, " for ", hashes[second].name, "."));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateRestoreTargetReceiptJoin(
    const CapsuleRestoreEvidenceV3& restore,
    const DPMTurnReceipt& receipt) {
  const CapsuleDPMRestoreTargetV2& target = restore.plan.target_state;
  const DPMProjectionManifest& manifest = receipt.projection_manifest;
  if (target.log_id != manifest.log_id ||
      target.source_event_count != manifest.source_event_count ||
      target.source_prefix_hash != manifest.source_prefix_hash ||
      target.prospective_response_event_index !=
          receipt.response_event_index ||
      target.projection_request_hash != manifest.request_hash ||
      target.projection_manifest_hash != manifest.manifest_hash ||
      target.correction_digest != manifest.correction_digest ||
      target.logical_agent_request_hash != receipt.agent_request_hash ||
      restore.plan.maximum_output_tokens != receipt.max_decision_tokens ||
      receipt.decision_token_ids.empty() ||
      receipt.decision_token_ids.size() >
          restore.plan.maximum_output_tokens ||
      target.agent_transcript_prefix_hash == receipt.agent_transcript_hash) {
    return absl::FailedPreconditionError(
        "Restore target and authoritative receipt disagree on the pending "
        "log prefix, projection, correction, logical request, output limit, "
        "or transcript phase.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<CapsuleRestoreAuthorityV2> BuildCapsuleRestoreAuthorityV2(
    const AuthenticatedCapsuleRestoreStateWitnessAdmission& admission) {
  ABSL_RETURN_IF_ERROR(ValidateCurrentV3Admission(admission));
  CapsuleRestoreAuthorityV2 authority{
      .capability = admission.capability,
      .admission_record_id = admission.record.record_id,
      .coverage_id = admission.operational_coverage.coverage_id,
      .qualification_spec_hash =
          admission.operational_coverage.qualification_evidence_hash,
  };
  return authority;
}

absl::Status ValidateCapsuleRestoreAuthorityV2ForAdmission(
    const CapsuleRestoreAuthorityV2& authority,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission& admission) {
  ABSL_ASSIGN_OR_RETURN(const CapsuleRestoreAuthorityV2 expected,
                        BuildCapsuleRestoreAuthorityV2(admission));
  if (authority != expected) {
    return absl::FailedPreconditionError(
        "CapsuleRestore authority differs from the freshly reauthenticated "
        "record, runtime capability, coverage, or qualification evidence.");
  }
  return absl::OkStatus();
}

absl::StatusOr<DPMPreparedPrefillWorkBinding>
BuildDPMPreparedPrefillWorkBinding(
    const DPMPreparedPrefillPlan& prepared_plan) {
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(prepared_plan));
  DPMPreparedPrefillWorkBinding binding{
      .start_kind = prepared_plan.start_kind,
      .plan_id = prepared_plan.plan_id,
      .canonical_source_chunks_hash =
          prepared_plan.canonical_source_chunks_hash,
      .resolved_token_plan_hash = prepared_plan.resolved_token_plan_hash,
      .shape_schedule_hash = prepared_plan.shape_schedule_hash,
  };
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillWorkBinding(binding));
  return binding;
}

absl::Status ValidateDPMPreparedPrefillWorkBindingForPlan(
    const DPMPreparedPrefillWorkBinding& binding,
    const DPMPreparedPrefillPlan& prepared_plan) {
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillWorkBinding(binding));
  ABSL_ASSIGN_OR_RETURN(const DPMPreparedPrefillWorkBinding expected,
                        BuildDPMPreparedPrefillWorkBinding(prepared_plan));
  if (binding != expected) {
    return absl::FailedPreconditionError(
        "Prepared-prefill work binding differs from the complete canonical "
        "runtime-derived plan.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCapsuleCaptureEvidenceV3CheckpointBinding(
    const DPMSessionCheckpointArtifact& artifact,
    const DPMTurnReceipt& source_receipt,
    const CapsuleCaptureEvidenceV3& capture_evidence,
    const CapsuleRestoreAuthorityV2& current_authority,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission&
        current_admission) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAuthorityV2ForAdmission(
      current_authority, current_admission));
  ABSL_RETURN_IF_ERROR(ValidateDPMSessionCheckpointArtifact(artifact));
  ABSL_RETURN_IF_ERROR(ValidateAuthoritativeV6Receipt(source_receipt));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleCaptureEvidenceV3(capture_evidence));
  ABSL_RETURN_IF_ERROR(ValidateCaptureWithinCoverage(
      capture_evidence,
      current_admission.operational_coverage.operational_domain));

  const DPMSessionCheckpointDescriptor& descriptor = artifact.descriptor;
  const CapsuleCapturePlanV2& plan = capture_evidence.plan;
  if (descriptor.format_version !=
          DPMSessionCheckpointDescriptor::kFormatVersion ||
      plan.authority != current_authority ||
      descriptor.descriptor_id != capture_evidence.checkpoint_id ||
      source_receipt.session_checkpoint_id !=
          std::optional<Hash256>(descriptor.descriptor_id) ||
      !source_receipt.published_checkpoint_capture.has_value() ||
      descriptor.capsule_capture_plan_hash != plan.plan_hash ||
      source_receipt.published_checkpoint_capture->capture_plan_hash !=
          plan.plan_hash ||
      source_receipt.published_checkpoint_capture->capture_evidence_id !=
          capture_evidence.evidence_id ||
      descriptor.key_id != capture_evidence.checkpoint_authentication_key_id ||
      descriptor.envelope_hash != capture_evidence.checkpoint_envelope_hash ||
      descriptor.envelope_size != capture_evidence.checkpoint_envelope_size ||
      descriptor.capsule_restore_capability_id !=
          current_authority.capability.capability_id ||
      descriptor.capsule_restore_admission_record_id !=
          current_authority.admission_record_id ||
      descriptor.capsule_restore_coverage_id !=
          current_authority.coverage_id ||
      descriptor.session_identity !=
          current_authority.capability.session_identity ||
      descriptor.key_id != current_admission.operational_coverage
                               .operational_domain
                               .checkpoint_authentication_key_id) {
    return absl::FailedPreconditionError(
        "Checkpoint artifact, receipt, capture evidence, and current "
        "authority disagree on the checkpoint, plan/evidence pair, durable "
        "envelope, authentication key, or runtime identity.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateReceiptAuthority(source_receipt, current_authority));

  ABSL_ASSIGN_OR_RETURN(
      const DPMPreparedPrefillWorkBinding prepared,
      BuildDPMPreparedPrefillWorkBinding(plan.prefill.prepared_plan));
  if (descriptor.prepared_prefill_work != prepared ||
      source_receipt.agent_prepared_prefill_work !=
          std::optional<DPMPreparedPrefillWorkBinding>(prepared)) {
    return absl::FailedPreconditionError(
        "Checkpoint descriptor or receipt does not carry the capture plan's "
        "complete runtime-derived prepared work.");
  }

  ABSL_RETURN_IF_ERROR(
      ValidateCaptureStateJoin(descriptor, source_receipt, capture_evidence));
  ABSL_RETURN_IF_ERROR(ValidateCaptureParentJoin(
      descriptor, source_receipt, capture_evidence));
  ABSL_RETURN_IF_ERROR(ValidateCaptureModeJoin(
      descriptor, source_receipt, capture_evidence, current_authority));
  return ValidateCaptureHashDomains(descriptor, source_receipt,
                                    capture_evidence, prepared,
                                    current_authority);
}

absl::Status ValidateCapsuleRestoreEvidenceV3SourceCheckpointBinding(
    const CapsuleRestoreEvidenceV3& restore_evidence,
    const CapsuleCaptureEvidenceV3& source_capture_evidence,
    const DPMSessionCheckpointArtifact& source_artifact,
    const DPMTurnReceipt& source_receipt,
    const CapsuleRestoreAuthorityV2& current_authority,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission&
        current_admission) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceV3CheckpointBinding(
      source_artifact, source_receipt, source_capture_evidence,
      current_authority, current_admission));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidenceV3ForSourceCapture(
      restore_evidence, source_capture_evidence));
  ABSL_RETURN_IF_ERROR(ValidateRestoreWithinCoverage(
      restore_evidence,
      current_admission.operational_coverage.operational_domain));
  if (restore_evidence.plan.authority != current_authority) {
    return absl::FailedPreconditionError(
        "Restore evidence does not carry the freshly reauthenticated current "
        "authority.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const DPMPreparedPrefillWorkBinding prepared,
      BuildDPMPreparedPrefillWorkBinding(
          restore_evidence.plan.prefill.prepared_plan));
  ABSL_RETURN_IF_ERROR(ValidateDistinctHashDomains(
      {{"source checkpoint", source_artifact.descriptor.descriptor_id},
       {"source capture plan", source_capture_evidence.plan.plan_hash},
       {"source capture evidence", source_capture_evidence.evidence_id},
       {"restore plan", restore_evidence.plan.plan_hash},
       {"restore evidence", restore_evidence.evidence_id},
       {"restore prepared plan", prepared.plan_id},
       {"restore prepared source", prepared.canonical_source_chunks_hash},
       {"restore prepared token plan", prepared.resolved_token_plan_hash},
       {"restore prepared shape schedule", prepared.shape_schedule_hash},
       {"exact profile", current_admission.profile.profile_id},
       {"CapsuleRestore capability",
        current_authority.capability.capability_id},
       {"CapsuleRestore admission", current_authority.admission_record_id},
       {"CapsuleRestore coverage", current_authority.coverage_id},
       {"qualification evidence",
        current_authority.qualification_spec_hash},
       {"target source-prefix",
        restore_evidence.plan.target_state.source_prefix_hash},
       {"target projection request",
        restore_evidence.plan.target_state.projection_request_hash},
       {"target projection manifest",
        restore_evidence.plan.target_state.projection_manifest_hash},
       {"target correction",
        restore_evidence.plan.target_state.correction_digest},
       {"target transcript prefix",
        restore_evidence.plan.target_state.agent_transcript_prefix_hash},
       {"target logical agent request",
        restore_evidence.plan.target_state.logical_agent_request_hash},
       {"restore reauthentication",
        restore_evidence.durable_to_transient_reauthentication.evidence_id},
       {"canonical continuation state",
        restore_evidence.durable_to_transient_reauthentication
            .canonical_continuation_state_hash},
       {"restore transient envelope",
        restore_evidence.durable_to_transient_reauthentication
            .destination_envelope_hash},
       {"restore target witness",
        restore_evidence.target_post_import.witness_id}}));
  return absl::OkStatus();
}

absl::Status ValidateCapsuleRestoreEvidenceV3TurnBinding(
    const CapsuleRestoreEvidenceV3& restore_evidence,
    const CapsuleCaptureEvidenceV3& source_capture_evidence,
    const DPMSessionCheckpointArtifact& source_artifact,
    const DPMTurnReceipt& source_receipt,
    const DPMTurnReceipt& restoring_receipt,
    const CapsuleRestoreAuthorityV2& current_authority,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission&
        current_admission) {
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreEvidenceV3SourceCheckpointBinding(
          restore_evidence, source_capture_evidence, source_artifact,
          source_receipt, current_authority, current_admission));
  ABSL_RETURN_IF_ERROR(ValidateAuthoritativeV6Receipt(restoring_receipt));
  ABSL_RETURN_IF_ERROR(
      ValidateReceiptAuthority(restoring_receipt, current_authority));
  ABSL_RETURN_IF_ERROR(
      ValidateRestoreTargetReceiptJoin(restore_evidence, restoring_receipt));

  if (restoring_receipt.restored_from_session_checkpoint_id !=
          std::optional<Hash256>(source_artifact.descriptor.descriptor_id) ||
      restoring_receipt.restored_checkpoint_capture_evidence_id !=
          std::optional<Hash256>(source_capture_evidence.evidence_id) ||
      restoring_receipt.agent_capsule_restore_evidence_id !=
          std::optional<Hash256>(restore_evidence.evidence_id)) {
    return absl::FailedPreconditionError(
        "Restoring receipt does not carry the exact source checkpoint, "
        "source-capture evidence, and operation restore evidence IDs.");
  }

  ABSL_ASSIGN_OR_RETURN(
      const DPMPreparedPrefillWorkBinding prepared,
      BuildDPMPreparedPrefillWorkBinding(
          restore_evidence.plan.prefill.prepared_plan));
  if (restoring_receipt.agent_prepared_prefill_work !=
      std::optional<DPMPreparedPrefillWorkBinding>(prepared)) {
    return absl::FailedPreconditionError(
        "Restoring receipt does not carry the restore plan's complete "
        "runtime-derived prepared work.");
  }

  switch (restoring_receipt.agent_replay_mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      if (restoring_receipt.agent_worker_prefill_mode !=
              DPMCheckpointWorkerPrefillMode::kNone) {
        return absl::FailedPreconditionError(
            "WinnerReplay restore receipt claims exact-worker prefill mode.");
      }
      break;
    case DPMReplayMode::kExactRegeneration:
      if (!restoring_receipt.agent_exact_profile_id.has_value() ||
          *restoring_receipt.agent_exact_profile_id !=
              current_authority.capability.exact_profile_id ||
          restoring_receipt.agent_worker_prefill_mode !=
              DPMCheckpointWorkerPrefillMode::kOwnPositionCapsuleDelta) {
        return absl::FailedPreconditionError(
            "Exact restoring receipt differs from the current profile or "
            "own-position worker prefill mode.");
      }
      break;
  }

  std::vector<NamedHash> hashes = {
      {"source checkpoint", source_artifact.descriptor.descriptor_id},
      {"source capture evidence", source_capture_evidence.evidence_id},
      {"restore plan", restore_evidence.plan.plan_hash},
      {"restore evidence", restore_evidence.evidence_id},
      {"prepared plan", prepared.plan_id},
      {"prepared source", prepared.canonical_source_chunks_hash},
      {"prepared token plan", prepared.resolved_token_plan_hash},
      {"prepared shape schedule", prepared.shape_schedule_hash},
      {"CapsuleRestore capability",
       current_authority.capability.capability_id},
      {"CapsuleRestore admission", current_authority.admission_record_id},
      {"CapsuleRestore coverage", current_authority.coverage_id},
      {"qualification evidence",
       current_authority.qualification_spec_hash},
      {"target source-prefix",
       restore_evidence.plan.target_state.source_prefix_hash},
      {"target projection request",
       restore_evidence.plan.target_state.projection_request_hash},
      {"target projection manifest",
       restore_evidence.plan.target_state.projection_manifest_hash},
      {"target correction",
       restore_evidence.plan.target_state.correction_digest},
      {"target transcript prefix",
       restore_evidence.plan.target_state.agent_transcript_prefix_hash},
      {"completed transcript", restoring_receipt.agent_transcript_hash},
      {"logical agent request",
       restore_evidence.plan.target_state.logical_agent_request_hash},
      {"agent replay request", restoring_receipt.agent_replay_request_hash},
      {"agent execution evidence",
       restoring_receipt.agent_execution_evidence_hash},
  };
  if (restoring_receipt.agent_replay_mode ==
      DPMReplayMode::kExactRegeneration) {
    hashes.push_back(
        {"exact profile", *restoring_receipt.agent_exact_profile_id});
    hashes.push_back(
        {"exact profile admission",
         *restoring_receipt.agent_exact_profile_admission_record_id});
    hashes.push_back(
        {"exact output", *restoring_receipt.agent_exact_output_evidence_hash});
    hashes.push_back(
        {"exact physical plan",
         restoring_receipt.agent_physical_execution_plan_hash});
  }
  for (std::size_t first = 0; first < hashes.size(); ++first) {
    if (IsZeroHash(hashes[first].value)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CapsuleRestore turn binding has an empty ", hashes[first].name,
          " hash."));
    }
    for (std::size_t second = first + 1; second < hashes.size(); ++second) {
      if (hashes[first].value == hashes[second].value) {
        return absl::InvalidArgumentError(absl::StrCat(
            "CapsuleRestore turn binding substituted ", hashes[first].name,
            " for ", hashes[second].name, "."));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
