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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_BINDING_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_BINDING_H_

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/capsule_restore_admission.h"
#include "runtime/dpm/capsule_restore_evidence.h"
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/session_checkpoint.h"

namespace litert::lm {

// Constructs the only canonical CapsuleRestoreAuthorityV2 representation for
// a freshly reauthenticated Coverage V2 admission. The legacy-named
// qualification_spec_hash field carries Coverage V2's canonical aggregate
// qualification_evidence_hash; no caller-selected assertion is admitted.
absl::StatusOr<CapsuleRestoreAuthorityV2> BuildCapsuleRestoreAuthorityV2(
    const AuthenticatedCapsuleRestoreStateWitnessAdmission& admission);

// Revalidates the complete record/profile/capability/coverage graph and the
// currently linked V3 capture/restore contracts, then requires `authority` to
// be byte-for-field equal to the canonical value above. `admission` must be a
// fresh result of ResolveAuthenticatedCapsuleRestoreStateWitnessAdmission at
// the current operation boundary; this value-level validator cannot create a
// freshness lease from a previously cached object.
absl::Status ValidateCapsuleRestoreAuthorityV2ForAdmission(
    const CapsuleRestoreAuthorityV2& authority,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission& admission);

// Losslessly projects one complete runtime-derived prefill plan into the
// compact descriptor/receipt work binding. Both functions first validate and
// canonically recompute the plan's own hashes.
absl::StatusOr<DPMPreparedPrefillWorkBinding>
BuildDPMPreparedPrefillWorkBinding(
    const DPMPreparedPrefillPlan& prepared_plan);
absl::Status ValidateDPMPreparedPrefillWorkBindingForPlan(
    const DPMPreparedPrefillWorkBinding& binding,
    const DPMPreparedPrefillPlan& prepared_plan);

// Joins a publishable version-4 checkpoint artifact to the authoritative
// version-7 receipt that published it, the complete V3 capture evidence, and
// a freshly reauthenticated current authority. This validates replicated
// fields only; raw-log prefix correctness and checkpoint candidate recency
// remain DPMEngine responsibilities.
absl::Status ValidateCapsuleCaptureEvidenceV3CheckpointBinding(
    const DPMSessionCheckpointArtifact& artifact,
    const DPMTurnReceipt& source_receipt,
    const CapsuleCaptureEvidenceV3& capture_evidence,
    const CapsuleRestoreAuthorityV2& current_authority,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission&
        current_admission);

// Pre-use source-only restore gate. It first performs the complete capture /
// artifact / source-receipt join above, then proves that `restore_evidence`
// names exactly that durable checkpoint and canonical continuation state.
// This form is usable before a new response receipt exists.
absl::Status ValidateCapsuleRestoreEvidenceV3SourceCheckpointBinding(
    const CapsuleRestoreEvidenceV3& restore_evidence,
    const CapsuleCaptureEvidenceV3& source_capture_evidence,
    const DPMSessionCheckpointArtifact& source_artifact,
    const DPMTurnReceipt& source_receipt,
    const CapsuleRestoreAuthorityV2& current_authority,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission&
        current_admission);

// Post-generation/commit gate. In addition to the source-only join, requires
// a fully assembled version-7 restoring receipt to carry the exact source
// checkpoint, source-capture evidence, operation restore evidence, target DPM
// state, prepared work, current authority, and mode-specific exact evidence.
absl::Status ValidateCapsuleRestoreEvidenceV3TurnBinding(
    const CapsuleRestoreEvidenceV3& restore_evidence,
    const CapsuleCaptureEvidenceV3& source_capture_evidence,
    const DPMSessionCheckpointArtifact& source_artifact,
    const DPMTurnReceipt& source_receipt,
    const DPMTurnReceipt& restoring_receipt,
    const CapsuleRestoreAuthorityV2& current_authority,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission&
        current_admission);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CAPSULE_RESTORE_EVIDENCE_BINDING_H_
