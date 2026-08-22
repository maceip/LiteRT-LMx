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

#include "runtime/dpm/capsule_restore_admission.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"       // from @com_google_absl
#include "absl/strings/str_cat.h"       // from @com_google_absl
#include "absl/strings/string_view.h"   // from @com_google_absl
#include "absl/time/clock.h"            // from @com_google_absl
#include "absl/time/time.h"             // from @com_google_absl
#include "runtime/dpm/capsule_restore_evidence.h"
#include "runtime/dpm/dpm_capabilities.h"
#include "runtime/dpm/dpm_prepared_prefill_plan.h"
#include "runtime/dpm/exact_decode_evidence.h"
#include "runtime/dpm/fresh_worker_process.h"
#include "runtime/engine/exact_litert_decode.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/session_handoff_codec_contract.h"
#include "runtime/platform/hash/hmac_sha256.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/proto/sampler_params.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kAdmissionMagic = {'D', 'P', 'M', 'C',
                                                 'R', 'A', '0', '1'};
constexpr uint32_t kAdmissionEnvelopeFormatVersion = 1;
constexpr uint64_t kAdmissionEnvelopeFixedBytes = 8 + 4 + 4 + 8 + 32;
constexpr uint32_t kMaximumAdmissionKeyIdBytes = 256;
constexpr uint32_t kMaximumCheckpointKeyIdBytes = 1024;
constexpr absl::string_view kAdmissionRecordDomain =
    "LITERT_LMX_CAPSULE_RESTORE_ADMISSION_RECORD_SHA256";
constexpr absl::string_view kAdmissionMacDomain =
    "LITERT_LMX_CAPSULE_RESTORE_ADMISSION_HMAC_SHA256";
constexpr absl::string_view kOperationalCoverageDomain =
    "LITERT_LMX_CAPSULE_RESTORE_OPERATIONAL_COVERAGE_SHA256";
constexpr absl::string_view kStateWitnessQualificationCaseDomain =
    "LITERT_LMX_CAPSULE_RESTORE_STATE_WITNESS_CASE_SHA256";
constexpr absl::string_view kStateWitnessQualificationEvidenceDomain =
    "LITERT_LMX_CAPSULE_RESTORE_STATE_WITNESS_EVIDENCE_SHA256";
constexpr absl::string_view kAdmissionLookupDomain =
    "LITERT_LMX_CAPSULE_RESTORE_ADMISSION_LOOKUP_SHA256";

bool IsZeroHash(const Hash256& hash) {
  uint8_t combined = 0;
  for (uint8_t byte : hash.bytes) combined |= byte;
  return combined == 0;
}

bool HasControlByte(absl::string_view text) {
  for (unsigned char byte : text) {
    if (byte < 0x20 || byte == 0x7f) return true;
  }
  return false;
}

bool IsValidUtf8(absl::string_view text) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t lead = bytes[index++];
    if (lead <= 0x7f) continue;

    size_t continuation_count = 0;
    uint8_t minimum_second = 0x80;
    uint8_t maximum_second = 0xbf;
    if (lead >= 0xc2 && lead <= 0xdf) {
      continuation_count = 1;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      continuation_count = 2;
      if (lead == 0xe0) minimum_second = 0xa0;
      if (lead == 0xed) maximum_second = 0x9f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      continuation_count = 3;
      if (lead == 0xf0) minimum_second = 0x90;
      if (lead == 0xf4) maximum_second = 0x8f;
    } else {
      return false;
    }
    if (continuation_count > text.size() - index ||
        bytes[index] < minimum_second || bytes[index] > maximum_second) {
      return false;
    }
    ++index;
    for (size_t offset = 1; offset < continuation_count; ++offset, ++index) {
      if (bytes[index] < 0x80 || bytes[index] > 0xbf) return false;
    }
  }
  return true;
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

void AppendI32(int32_t value, std::string* output) {
  AppendU32(static_cast<uint32_t>(value), output);
}

void AppendHash(const Hash256& hash, std::string* output) {
  output->append(reinterpret_cast<const char*>(hash.bytes.data()),
                 hash.bytes.size());
}

void AppendIdentity(const SessionHandoffIdentity& identity,
                    std::string* output) {
  AppendHash(identity.model_artifact_hash, output);
  AppendHash(identity.runtime_artifact_hash, output);
  AppendHash(identity.inference_profile_hash, output);
}

void AppendCapability(const SessionHandoffCapability& capability,
                      std::string* output) {
  AppendU32(capability.version, output);
  AppendHash(capability.capability_id, output);
  AppendIdentity(capability.session_identity, output);
  AppendHash(capability.exact_profile_id, output);
  AppendU32(static_cast<uint32_t>(capability.backend), output);
  AppendHash(capability.complete_state_inventory_hash, output);
  AppendHash(capability.capsule_codec_contract_hash, output);
}

void AppendProfile(const ExactLiteRtProfile& profile, std::string* output) {
  AppendHash(profile.profile_id, output);
  AppendHash(profile.model_artifact_hash, output);
  AppendHash(profile.tokenizer_contract_hash, output);
  AppendHash(profile.litert_model_bytecode_hash, output);
  AppendHash(profile.runtime_delegate_platform_hash, output);
  AppendHash(profile.loaded_execution_profile_hash, output);
  AppendHash(profile.gpu_execution_policy_hash, output);
  AppendIdentity(profile.session_identity, output);
  AppendU32(static_cast<uint32_t>(profile.backend), output);
  AppendU32(profile.bound_evidence, output);
  AppendU32(static_cast<uint32_t>(profile.qualification_requirement), output);
  AppendU32(static_cast<uint32_t>(profile.sampler_identity), output);
  AppendU32(static_cast<uint32_t>(profile.logits_frame.element_type), output);
  AppendU32(profile.logits_frame.batch_size, output);
  AppendU32(profile.logits_frame.sequence_size, output);
  AppendU32(profile.logits_frame.vocabulary_size, output);
  AppendU64(profile.logits_frame.byte_count, output);
  AppendU32(profile.batch_size, output);
  AppendU32(profile.cpu_thread_count, output);
  AppendI32(profile.prefill_chunk_size, output);
}

void AppendStateWitnessQualificationPolicy(
    const CapsuleRestoreStateWitnessQualificationPolicy& policy,
    std::string* output) {
  AppendU32(policy.format_version, output);
  AppendU32(static_cast<uint32_t>(policy.content_authority), output);
  AppendU32(policy.required_operation_evidence_mask, output);
  AppendU32(policy.required_qualification_case_kind_mask, output);
  AppendU32(policy.minimum_independent_trials_per_shape_class_and_kind, output);
  AppendU32(policy.maximum_qualified_shape_classes, output);
  AppendHash(policy.qualification_verifier_contract_hash, output);
}

void AppendStateWitnessOperationalDomain(
    const CapsuleRestoreStateWitnessOperationalDomain& domain,
    std::string* output) {
  AppendU32(domain.format_version, output);
  AppendU32(static_cast<uint32_t>(domain.capture_phase), output);
  AppendU32(static_cast<uint32_t>(domain.admitted_backend), output);
  AppendHash(domain.resolved_session_config_hash, output);
  AppendHash(domain.session_continuation_state_witness_contract_hash, output);
  AppendHash(domain.capture_evidence_contract_hash, output);
  AppendHash(domain.restore_evidence_contract_hash, output);
  AppendHash(domain.deterministic_prefill_plan_contract_hash, output);
  AppendHash(domain.execution_shape_class_contract_hash, output);
  AppendHash(domain.restricted_feature_contract_hash, output);
  AppendU64(domain.maximum_context_positions, output);
  AppendU64(domain.minimum_checkpoint_step, output);
  AppendU64(domain.maximum_checkpoint_step, output);
  AppendU64(domain.minimum_delta_positions, output);
  AppendU64(domain.maximum_delta_positions, output);
  AppendU32(domain.minimum_prefill_chunks, output);
  AppendU32(domain.maximum_prefill_chunks, output);
  AppendU64(domain.maximum_prefill_text_bytes, output);
  AppendU32(domain.maximum_prefill_token_ids, output);
  AppendU32(domain.maximum_output_tokens, output);
  AppendU32(domain.admitted_encoding_mask, output);
  AppendU32(
      static_cast<uint32_t>(domain.checkpoint_authentication_key_id.size()),
      output);
  output->append(domain.checkpoint_authentication_key_id);
  AppendU32(static_cast<uint32_t>(
                domain.operation_evidence_authentication_key_id.size()),
            output);
  output->append(domain.operation_evidence_authentication_key_id);
  AppendStateWitnessQualificationPolicy(domain.qualification_policy, output);
}

void AppendStateWitnessQualificationCaseFields(
    const CapsuleRestoreStateWitnessQualificationCaseEvidence& evidence,
    std::string* output) {
  AppendU32(evidence.format_version, output);
  AppendU32(static_cast<uint32_t>(evidence.kind), output);
  AppendHash(evidence.shape_class_hash, output);
  AppendHash(evidence.trial_identity_hash, output);
  AppendHash(evidence.source_session_instance_hash, output);
  AppendHash(evidence.target_session_instance_hash, output);
  AppendU64(evidence.checkpoint_step, output);
  AppendU64(evidence.delta_positions, output);
  AppendU32(evidence.prefill_chunk_count, output);
  AppendU64(evidence.prefill_text_bytes, output);
  AppendU32(evidence.prefill_token_ids, output);
  AppendU32(evidence.output_tokens, output);
  AppendU32(evidence.observed_encoding_mask, output);
  AppendHash(evidence.producer_state_witness_hash, output);
  AppendHash(evidence.restored_state_witness_hash, output);
  AppendHash(evidence.capture_evidence_hash, output);
  AppendHash(evidence.restore_evidence_hash, output);
  AppendHash(evidence.live_continuation_output_evidence_hash, output);
  AppendHash(evidence.restored_continuation_output_evidence_hash, output);
  AppendHash(evidence.verifier_certification_hash, output);
}

void AppendStateWitnessOperationalCoverageFields(
    const CapsuleRestoreOperationalCoverage& coverage, std::string* output) {
  AppendU32(coverage.format_version, output);
  AppendProfile(coverage.runtime_derived_profile, output);
  AppendCapability(coverage.runtime_derived_capability, output);
  AppendIdentity(coverage.runtime_derived_session_identity, output);
  AppendStateWitnessOperationalDomain(coverage.operational_domain, output);
  AppendHash(coverage.qualification_evidence_hash, output);
}

void AppendStateWitnessAdmissionRecordFields(
    const CapsuleRestoreAdmissionRecord& record, std::string* output) {
  AppendU32(record.format_version, output);
  AppendHash(record.operational_coverage.coverage_id, output);
  AppendStateWitnessOperationalCoverageFields(record.operational_coverage,
                                              output);
  AppendU32(static_cast<uint32_t>(record.qualification_cases.size()), output);
  for (const auto& evidence : record.qualification_cases) {
    AppendHash(evidence.qualification_case_id, output);
    AppendStateWitnessQualificationCaseFields(evidence, output);
  }
  AppendU64(static_cast<uint64_t>(record.qualified_unix_micros), output);
  AppendU32(static_cast<uint32_t>(record.record_authentication_key_id.size()),
            output);
  output->append(record.record_authentication_key_id);
}

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

absl::Status ValidatePublicKeyId(absl::string_view key_id,
                                 uint32_t maximum_size,
                                 absl::string_view description) {
  if (key_id.empty() || key_id.size() > maximum_size ||
      HasControlByte(key_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat(description, " key ID is invalid."));
  }
  return absl::OkStatus();
}

absl::Status ValidateDerivedProfile(const ExactLiteRtProfile& derived_profile) {
  return ValidateExactLiteRtProfile(derived_profile);
}

absl::Status ValidateProfileCapabilityAgreement(
    const ExactLiteRtProfile& profile,
    const SessionHandoffCapability& capability) {
  ABSL_RETURN_IF_ERROR(ValidateDerivedProfile(profile));
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffCapability(capability));
  if (capability.exact_profile_id != profile.profile_id ||
      capability.session_identity != profile.session_identity ||
      capability.backend != profile.backend) {
    return absl::FailedPreconditionError(
        "Engine-derived CapsuleRestore capability, exact profile, backend, "
        "and session identity do not agree.");
  }
  return absl::OkStatus();
}

absl::Status ValidateStateWitnessQualificationPolicyFields(
    const CapsuleRestoreStateWitnessQualificationPolicy& policy) {
  if (policy.format_version !=
          CapsuleRestoreStateWitnessQualificationPolicy::kFormatVersion ||
      policy.content_authority !=
          CapsuleRestoreStateWitnessContentAuthority::
              kPerOperationStateWitnessAndEvidenceOnly ||
      policy.required_operation_evidence_mask !=
          CapsuleRestoreStateWitnessRequiredOperationEvidenceMask() ||
      policy.required_qualification_case_kind_mask !=
          CapsuleRestoreStateWitnessRequiredQualificationCaseKindMask()) {
    return absl::FailedPreconditionError(
        "CapsuleRestore qualification policy would weaken the "
        "per-operation evidence or required pathway contract.");
  }
  if (policy.minimum_independent_trials_per_shape_class_and_kind < 2 ||
      policy.minimum_independent_trials_per_shape_class_and_kind >
          kMaximumCapsuleRestoreStateWitnessQualificationTrials ||
      policy.maximum_qualified_shape_classes == 0 ||
      policy.maximum_qualified_shape_classes >
          kMaximumCapsuleRestoreStateWitnessQualificationCases ||
      IsZeroHash(policy.qualification_verifier_contract_hash)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore qualification policy is incomplete or "
        "outside its product bounds.");
  }
  return absl::OkStatus();
}

absl::Status ValidateStateWitnessOperationalDomainFields(
    const CapsuleRestoreStateWitnessOperationalDomain& domain) {
  if (domain.format_version !=
          CapsuleRestoreStateWitnessOperationalDomain::kFormatVersion ||
      domain.capture_phase !=
          CapsuleRestoreStateWitnessCapturePhase::kDecodedOwnPosition ||
      domain.admitted_backend != ExactLiteRtBackend::kCpu) {
    return absl::FailedPreconditionError(
        "CapsuleRestore requires its tagged decoded "
        "own-position CPU domain.");
  }
  if (IsZeroHash(domain.resolved_session_config_hash) ||
      IsZeroHash(domain.session_continuation_state_witness_contract_hash) ||
      IsZeroHash(domain.capture_evidence_contract_hash) ||
      IsZeroHash(domain.restore_evidence_contract_hash) ||
      IsZeroHash(domain.deterministic_prefill_plan_contract_hash) ||
      IsZeroHash(domain.execution_shape_class_contract_hash) ||
      IsZeroHash(domain.restricted_feature_contract_hash)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore is missing a runtime-owned domain "
        "contract hash.");
  }
  const uint64_t maximum_int =
      static_cast<uint64_t>((std::numeric_limits<int>::max)());
  if (domain.maximum_context_positions == 0 ||
      domain.maximum_context_positions > maximum_int ||
      domain.minimum_checkpoint_step == 0 ||
      domain.minimum_checkpoint_step > domain.maximum_checkpoint_step ||
      domain.maximum_checkpoint_step > domain.maximum_context_positions ||
      domain.minimum_delta_positions == 0 ||
      domain.minimum_delta_positions > domain.maximum_delta_positions ||
      domain.maximum_delta_positions > domain.maximum_context_positions ||
      domain.minimum_prefill_chunks == 0 ||
      domain.minimum_prefill_chunks > domain.maximum_prefill_chunks ||
      domain.maximum_prefill_chunks > kMaximumCapsuleRestorePrefillChunks ||
      domain.maximum_prefill_text_bytes >
          kMaximumCapsuleRestorePrefillTextBytes ||
      domain.maximum_prefill_token_ids >
          kMaximumCapsuleRestorePrefillTokenIds ||
      domain.maximum_output_tokens == 0 ||
      domain.maximum_output_tokens > kMaximumFreshWorkerLogitFrames ||
      domain.maximum_output_tokens > domain.maximum_context_positions ||
      domain.maximum_output_tokens > maximum_int ||
      domain.minimum_checkpoint_step >
          domain.maximum_context_positions - domain.minimum_delta_positions ||
      domain.minimum_checkpoint_step + domain.minimum_delta_positions >
          domain.maximum_context_positions - 1) {
    return absl::InvalidArgumentError(
        "CapsuleRestore operational bounds are empty, "
        "inconsistent, or outside product limits.");
  }
  const uint32_t allowed_encoding_mask =
      CapsuleRestoreStateWitnessAllowedEncodingMask();
  if (domain.admitted_encoding_mask == 0 ||
      (domain.admitted_encoding_mask & ~allowed_encoding_mask) != 0) {
    return absl::InvalidArgumentError(
        "CapsuleRestore has an invalid encoding mask.");
  }
  const bool admits_text =
      (domain.admitted_encoding_mask &
       CapsuleRestoreStateWitnessEncodingBit(
           CapsuleRestoreStateWitnessEncoding::kUtf8Text)) != 0;
  const bool admits_tokens =
      (domain.admitted_encoding_mask &
       CapsuleRestoreStateWitnessEncodingBit(
           CapsuleRestoreStateWitnessEncoding::kExactTokenIds)) != 0;
  if (admits_text != (domain.maximum_prefill_text_bytes != 0) ||
      admits_tokens != (domain.maximum_prefill_token_ids != 0)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore encoding and byte/token bounds do not "
        "agree.");
  }
  ABSL_RETURN_IF_ERROR(ValidatePublicKeyId(
      domain.checkpoint_authentication_key_id, kMaximumCheckpointKeyIdBytes,
      "CapsuleRestore checkpoint authentication"));
  ABSL_RETURN_IF_ERROR(
      ValidatePublicKeyId(domain.operation_evidence_authentication_key_id,
                          kMaximumCheckpointKeyIdBytes,
                          "CapsuleRestore operation-evidence authentication"));
  if (domain.checkpoint_authentication_key_id ==
          domain.operation_evidence_authentication_key_id ||
      domain.checkpoint_authentication_key_id ==
          kFreshWorkerTransientRestoreKeyId ||
      domain.checkpoint_authentication_key_id ==
          kFreshWorkerTransientProducingKeyId ||
      domain.operation_evidence_authentication_key_id ==
          kFreshWorkerTransientRestoreKeyId ||
      domain.operation_evidence_authentication_key_id ==
          kFreshWorkerTransientProducingKeyId) {
    return absl::FailedPreconditionError(
        "CapsuleRestore checkpoint, operation evidence, and "
        "fixed request-transient capsules must use distinct key IDs.");
  }
  return ValidateStateWitnessQualificationPolicyFields(
      domain.qualification_policy);
}

bool IsStateWitnessQualificationCaseKind(
    CapsuleRestoreStateWitnessQualificationCaseKind kind) {
  return kind == CapsuleRestoreStateWitnessQualificationCaseKind::
                     kFullPrefillCaptureAndOwnPositionRestore ||
         kind == CapsuleRestoreStateWitnessQualificationCaseKind::
                     kRestoredAncestorDeltaCaptureAndOwnPositionRestore;
}

absl::Status ValidateStateWitnessQualificationCaseFields(
    const CapsuleRestoreStateWitnessQualificationCaseEvidence& evidence,
    const CapsuleRestoreStateWitnessOperationalDomain& domain,
    bool require_canonical_id) {
  ABSL_RETURN_IF_ERROR(ValidateStateWitnessOperationalDomainFields(domain));
  if (evidence.format_version !=
          CapsuleRestoreStateWitnessQualificationCaseEvidence::kFormatVersion ||
      !IsStateWitnessQualificationCaseKind(evidence.kind)) {
    return absl::FailedPreconditionError(
        "CapsuleRestore qualification case version or pathway "
        "is unsupported.");
  }
  if ((require_canonical_id && IsZeroHash(evidence.qualification_case_id)) ||
      IsZeroHash(evidence.shape_class_hash) ||
      IsZeroHash(evidence.trial_identity_hash) ||
      IsZeroHash(evidence.source_session_instance_hash) ||
      IsZeroHash(evidence.target_session_instance_hash) ||
      IsZeroHash(evidence.producer_state_witness_hash) ||
      IsZeroHash(evidence.restored_state_witness_hash) ||
      IsZeroHash(evidence.capture_evidence_hash) ||
      IsZeroHash(evidence.restore_evidence_hash) ||
      IsZeroHash(evidence.live_continuation_output_evidence_hash) ||
      IsZeroHash(evidence.restored_continuation_output_evidence_hash) ||
      IsZeroHash(evidence.verifier_certification_hash)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore qualification evidence is incomplete.");
  }
  if (evidence.source_session_instance_hash ==
          evidence.target_session_instance_hash ||
      evidence.producer_state_witness_hash !=
          evidence.restored_state_witness_hash ||
      evidence.live_continuation_output_evidence_hash !=
          evidence.restored_continuation_output_evidence_hash) {
    return absl::FailedPreconditionError(
        "CapsuleRestore qualification did not demonstrate a "
        "distinct-target own-position state and continuation match.");
  }
  if (evidence.checkpoint_step < domain.minimum_checkpoint_step ||
      evidence.checkpoint_step > domain.maximum_checkpoint_step ||
      evidence.delta_positions < domain.minimum_delta_positions ||
      evidence.delta_positions > domain.maximum_delta_positions ||
      evidence.prefill_chunk_count < domain.minimum_prefill_chunks ||
      evidence.prefill_chunk_count > domain.maximum_prefill_chunks ||
      evidence.prefill_text_bytes > domain.maximum_prefill_text_bytes ||
      evidence.prefill_token_ids > domain.maximum_prefill_token_ids ||
      evidence.output_tokens == 0 ||
      evidence.output_tokens > domain.maximum_output_tokens) {
    return absl::InvalidArgumentError(
        "CapsuleRestore qualification observation falls outside "
        "its authenticated operational domain.");
  }
  if (evidence.checkpoint_step >
          domain.maximum_context_positions - evidence.delta_positions ||
      evidence.checkpoint_step + evidence.delta_positions >
          domain.maximum_context_positions - evidence.output_tokens) {
    return absl::InvalidArgumentError(
        "CapsuleRestore qualification exceeds the context "
        "position bound.");
  }
  if (evidence.observed_encoding_mask == 0 ||
      (evidence.observed_encoding_mask & ~domain.admitted_encoding_mask) != 0) {
    return absl::InvalidArgumentError(
        "CapsuleRestore qualification observed an unadmitted "
        "encoding.");
  }
  const bool observed_text =
      (evidence.observed_encoding_mask &
       CapsuleRestoreStateWitnessEncodingBit(
           CapsuleRestoreStateWitnessEncoding::kUtf8Text)) != 0;
  const bool observed_tokens =
      (evidence.observed_encoding_mask &
       CapsuleRestoreStateWitnessEncodingBit(
           CapsuleRestoreStateWitnessEncoding::kExactTokenIds)) != 0;
  if (observed_text != (evidence.prefill_text_bytes != 0) ||
      observed_tokens != (evidence.prefill_token_ids != 0)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore qualification encoding observations do "
        "not match their byte/token counts.");
  }
  if (require_canonical_id) {
    ABSL_ASSIGN_OR_RETURN(
        const Hash256 expected_id,
        ComputeCapsuleRestoreStateWitnessQualificationCaseId(evidence));
    if (evidence.qualification_case_id != expected_id) {
      return absl::DataLossError(
          "CapsuleRestore qualification case ID is not "
          "canonical.");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateStateWitnessQualificationEvidenceFields(
    const CapsuleRestoreStateWitnessOperationalDomain& domain,
    const std::vector<CapsuleRestoreStateWitnessQualificationCaseEvidence>&
        qualification_cases) {
  ABSL_RETURN_IF_ERROR(ValidateStateWitnessOperationalDomainFields(domain));
  if (qualification_cases.empty() ||
      qualification_cases.size() >
          kMaximumCapsuleRestoreStateWitnessQualificationCases) {
    return absl::InvalidArgumentError(
        "CapsuleRestore requires a bounded nonempty "
        "qualification evidence set.");
  }

  struct CaseCounts {
    uint32_t full_prefill = 0;
    uint32_t restored_descendant = 0;
  };
  std::map<Hash256, CaseCounts> counts_by_shape_class;
  std::set<Hash256> trial_identities;
  std::set<Hash256> session_instances;
  uint32_t observed_encoding_mask = 0;
  Hash256 previous_case_id;
  bool have_previous = false;
  for (const auto& evidence : qualification_cases) {
    ABSL_RETURN_IF_ERROR(
        ValidateStateWitnessQualificationCaseFields(evidence, domain, true));
    if (have_previous && !(previous_case_id < evidence.qualification_case_id)) {
      return absl::InvalidArgumentError(
          "CapsuleRestore qualification cases must be strictly "
          "sorted by unique canonical case ID.");
    }
    previous_case_id = evidence.qualification_case_id;
    have_previous = true;
    if (!trial_identities.insert(evidence.trial_identity_hash).second) {
      return absl::InvalidArgumentError(
          "CapsuleRestore qualification reuses a trial "
          "identity.");
    }
    if (!session_instances.insert(evidence.source_session_instance_hash)
             .second ||
        !session_instances.insert(evidence.target_session_instance_hash)
             .second) {
      return absl::InvalidArgumentError(
          "CapsuleRestore qualification reuses a source or "
          "target session instance across independent trials.");
    }
    CaseCounts& counts = counts_by_shape_class[evidence.shape_class_hash];
    switch (evidence.kind) {
      case CapsuleRestoreStateWitnessQualificationCaseKind::
          kFullPrefillCaptureAndOwnPositionRestore:
        ++counts.full_prefill;
        break;
      case CapsuleRestoreStateWitnessQualificationCaseKind::
          kRestoredAncestorDeltaCaptureAndOwnPositionRestore:
        ++counts.restored_descendant;
        break;
    }
    observed_encoding_mask |= evidence.observed_encoding_mask;
  }
  if (counts_by_shape_class.size() >
      domain.qualification_policy.maximum_qualified_shape_classes) {
    return absl::InvalidArgumentError(
        "CapsuleRestore qualification evidence exceeds its "
        "shape-class policy.");
  }
  const uint32_t minimum_trials =
      domain.qualification_policy
          .minimum_independent_trials_per_shape_class_and_kind;
  for (const auto& [shape_class, counts] : counts_by_shape_class) {
    (void)shape_class;
    if (counts.full_prefill < minimum_trials ||
        counts.restored_descendant < minimum_trials) {
      return absl::FailedPreconditionError(
          "CapsuleRestore did not independently qualify both "
          "required pathways for every shape class.");
    }
  }
  if ((observed_encoding_mask & domain.admitted_encoding_mask) !=
      domain.admitted_encoding_mask) {
    return absl::FailedPreconditionError(
        "CapsuleRestore qualification did not exercise every "
        "admitted encoding.");
  }
  return absl::OkStatus();
}

absl::Status ValidateStateWitnessOperationalCoverageFields(
    const CapsuleRestoreOperationalCoverage& coverage,
    bool require_canonical_id) {
  if (coverage.format_version !=
          CapsuleRestoreOperationalCoverage::kFormatVersion ||
      (require_canonical_id && IsZeroHash(coverage.coverage_id)) ||
      IsZeroHash(coverage.qualification_evidence_hash)) {
    return absl::FailedPreconditionError(
        "CapsuleRestore authority is untagged or incomplete.");
  }
  ABSL_RETURN_IF_ERROR(ValidateProfileCapabilityAgreement(
      coverage.runtime_derived_profile, coverage.runtime_derived_capability));
  if (coverage.runtime_derived_profile.backend != ExactLiteRtBackend::kCpu ||
      coverage.runtime_derived_capability.backend != ExactLiteRtBackend::kCpu) {
    return absl::UnimplementedError(
        "CapsuleRestore currently admits only runtime-derived "
        "CPU profiles.");
  }
  if (coverage.runtime_derived_session_identity !=
          coverage.runtime_derived_profile.session_identity ||
      coverage.runtime_derived_session_identity !=
          coverage.runtime_derived_capability.session_identity) {
    return absl::FailedPreconditionError(
        "CapsuleRestore session identity disagrees with its "
        "runtime-derived profile or capability.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateStateWitnessOperationalDomainFields(coverage.operational_domain));
  if (coverage.operational_domain.admitted_backend !=
      coverage.runtime_derived_profile.backend) {
    return absl::FailedPreconditionError(
        "CapsuleRestore backend domain differs from its "
        "runtime-derived profile.");
  }
  return absl::OkStatus();
}

absl::Status ValidateStateWitnessAdmissionRecordFields(
    const CapsuleRestoreAdmissionRecord& record, bool require_canonical_id) {
  if (record.format_version != CapsuleRestoreAdmissionRecord::kFormatVersion ||
      (require_canonical_id && IsZeroHash(record.record_id)) ||
      record.qualified_unix_micros <= 0) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission record is untagged or "
        "incomplete.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreOperationalCoverage(record.operational_coverage));
  ABSL_RETURN_IF_ERROR(ValidateStateWitnessQualificationEvidenceFields(
      record.operational_coverage.operational_domain,
      record.qualification_cases));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_evidence_hash,
      ComputeCapsuleRestoreStateWitnessQualificationEvidenceHash(
          record.operational_coverage.operational_domain,
          record.qualification_cases));
  if (record.operational_coverage.qualification_evidence_hash !=
      expected_evidence_hash) {
    return absl::DataLossError(
        "CapsuleRestore qualification evidence hash is not "
        "canonical.");
  }
  ABSL_RETURN_IF_ERROR(ValidatePublicKeyId(
      record.record_authentication_key_id, kMaximumAdmissionKeyIdBytes,
      "CapsuleRestore admission-record authentication"));
  const auto& domain = record.operational_coverage.operational_domain;
  if (record.record_authentication_key_id ==
          domain.checkpoint_authentication_key_id ||
      record.record_authentication_key_id ==
          domain.operation_evidence_authentication_key_id) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission, checkpoint, and operation "
        "evidence require distinct key IDs.");
  }
  std::string encoded_fields;
  AppendStateWitnessAdmissionRecordFields(record, &encoded_fields);
  if (encoded_fields.size() > kMaximumCapsuleRestoreAdmissionEnvelopeBytes -
                                  kAdmissionEnvelopeFixedBytes - 32 -
                                  record.record_authentication_key_id.size()) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore admission record exceeds its storage "
        "limit.");
  }
  if (require_canonical_id) {
    ABSL_ASSIGN_OR_RETURN(const Hash256 expected_record_id,
                          ComputeCapsuleRestoreAdmissionRecordId(record));
    if (record.record_id != expected_record_id) {
      return absl::DataLossError(
          "CapsuleRestore admission record ID is not "
          "canonical.");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateResolvedSessionConfig(const SessionConfig& config,
                                           const Engine& engine) {
  if (config.GetMemoryStrategy() != SessionConfig::MemoryStrategy::kStateful) {
    return absl::FailedPreconditionError(
        "CapsuleRestore qualification requires an accumulating stateful "
        "session.");
  }
  if (config.UseExternalSampler() ||
      config.GetSamplerBackend() != Backend::CPU ||
      config.GetSamplerParams().type() != proto::SamplerParameters::GREEDY ||
      config.GetSamplerParams().backend() != proto::SamplerParameters::CPU) {
    return absl::UnimplementedError(
        "CapsuleRestore qualification requires the explicit CPU GREEDY "
        "min-index sampler.");
  }
  if (config.GetNumOutputCandidates() != 1 ||
      config.GetSuppressTokensConfig().enabled() ||
      config.GetApplyPromptTemplateInSession()) {
    return absl::UnimplementedError(
        "CapsuleRestore qualification requires one candidate, no token "
        "suppression, and no hidden prompt templates.");
  }
  if (config.AudioModalityEnabled() || config.VisionModalityEnabled() ||
      config.GetAudioEmbeddingsCallback() != nullptr ||
      config.GetScopedLoraFile() != nullptr ||
      config.GetAudioScopedLoraFile() != nullptr) {
    return absl::UnimplementedError(
        "CapsuleRestore qualification supports only text sessions without "
        "LoRA, audio, vision, or embedding callbacks.");
  }
  if (engine.GetEngineSettings().IsBenchmarkEnabled()) {
    return absl::UnimplementedError(
        "CapsuleRestore qualification does not admit benchmark-controlled "
        "execution.");
  }
  return absl::OkStatus();
}

class Reader {
 public:
  explicit Reader(absl::string_view bytes) : bytes_(bytes) {}

  absl::StatusOr<uint32_t> ReadU32() {
    if (remaining() < 4) return Truncated();
    uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      value = (value << 8) | static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 4;
    return value;
  }

  absl::StatusOr<uint64_t> ReadU64() {
    if (remaining() < 8) return Truncated();
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
      value = (value << 8) | static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 8;
    return value;
  }

  absl::StatusOr<int32_t> ReadI32() {
    ABSL_ASSIGN_OR_RETURN(const uint32_t encoded, ReadU32());
    const int64_t signed_value =
        encoded <= static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())
            ? static_cast<int64_t>(encoded)
            : static_cast<int64_t>(encoded) - (int64_t{1} << 32);
    return static_cast<int32_t>(signed_value);
  }

  absl::StatusOr<Hash256> ReadHash() {
    if (remaining() < 32) return Truncated();
    Hash256 hash;
    std::memcpy(hash.bytes.data(), bytes_.data() + offset_, hash.bytes.size());
    offset_ += hash.bytes.size();
    return hash;
  }

  absl::StatusOr<absl::string_view> ReadBytes(uint64_t size) {
    if (size > remaining()) return Truncated();
    const size_t safe_size = static_cast<size_t>(size);
    const absl::string_view value = bytes_.substr(offset_, safe_size);
    offset_ += safe_size;
    return value;
  }

  size_t remaining() const { return bytes_.size() - offset_; }

 private:
  absl::Status Truncated() const {
    return absl::DataLossError("Truncated CapsuleRestore admission record.");
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

absl::StatusOr<SessionHandoffIdentity> ReadIdentity(Reader* reader) {
  SessionHandoffIdentity identity;
  ABSL_ASSIGN_OR_RETURN(identity.model_artifact_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(identity.runtime_artifact_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(identity.inference_profile_hash, reader->ReadHash());
  return identity;
}

absl::StatusOr<SessionHandoffCapability> ReadCapability(Reader* reader) {
  SessionHandoffCapability capability;
  ABSL_ASSIGN_OR_RETURN(capability.version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(capability.capability_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(capability.session_identity, ReadIdentity(reader));
  ABSL_ASSIGN_OR_RETURN(capability.exact_profile_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(const uint32_t backend, reader->ReadU32());
  if (backend != static_cast<uint32_t>(ExactLiteRtBackend::kCpu) &&
      backend != static_cast<uint32_t>(ExactLiteRtBackend::kMetalGpu)) {
    return absl::UnimplementedError(
        "CapsuleRestore admission capability encodes an unsupported "
        "backend.");
  }
  capability.backend = static_cast<ExactLiteRtBackend>(backend);
  ABSL_ASSIGN_OR_RETURN(capability.complete_state_inventory_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(capability.capsule_codec_contract_hash,
                        reader->ReadHash());
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffCapability(capability));
  return capability;
}

absl::StatusOr<ExactLiteRtProfile> ReadProfile(Reader* reader) {
  ExactLiteRtProfile profile;
  ABSL_ASSIGN_OR_RETURN(profile.profile_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(profile.model_artifact_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(profile.tokenizer_contract_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(profile.litert_model_bytecode_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(profile.runtime_delegate_platform_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(profile.loaded_execution_profile_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(profile.gpu_execution_policy_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(profile.session_identity, ReadIdentity(reader));
  ABSL_ASSIGN_OR_RETURN(const uint32_t backend, reader->ReadU32());
  if (backend != static_cast<uint32_t>(ExactLiteRtBackend::kCpu)) {
    return absl::UnimplementedError(
        "CapsuleRestore profile encoding is not CPU.");
  }
  profile.backend = static_cast<ExactLiteRtBackend>(backend);
  ABSL_ASSIGN_OR_RETURN(profile.bound_evidence, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t qualification_requirement,
                        reader->ReadU32());
  if (qualification_requirement !=
      static_cast<uint32_t>(ExactLiteRtQualificationRequirement::
                                kIndependentColdProcessesTokensAndLogits)) {
    return absl::FailedPreconditionError(
        "CapsuleRestore profile has an unsupported "
        "qualification requirement.");
  }
  profile.qualification_requirement =
      static_cast<ExactLiteRtQualificationRequirement>(
          qualification_requirement);
  ABSL_ASSIGN_OR_RETURN(const uint32_t sampler_identity, reader->ReadU32());
  if (sampler_identity !=
      static_cast<uint32_t>(
          ExactLiteRtSamplerIdentity::kCpuGreedyArgmaxMinIndex)) {
    return absl::FailedPreconditionError(
        "CapsuleRestore profile has an unsupported sampler "
        "identity.");
  }
  profile.sampler_identity =
      static_cast<ExactLiteRtSamplerIdentity>(sampler_identity);
  ABSL_ASSIGN_OR_RETURN(const uint32_t logits_element_type, reader->ReadU32());
  if (logits_element_type !=
          static_cast<uint32_t>(ExactLiteRtLogitsElementType::kFloat16) &&
      logits_element_type !=
          static_cast<uint32_t>(ExactLiteRtLogitsElementType::kFloat32)) {
    return absl::FailedPreconditionError(
        "CapsuleRestore profile has an unsupported logits "
        "element type.");
  }
  profile.logits_frame.element_type =
      static_cast<ExactLiteRtLogitsElementType>(logits_element_type);
  ABSL_ASSIGN_OR_RETURN(profile.logits_frame.batch_size, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(profile.logits_frame.sequence_size, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(profile.logits_frame.vocabulary_size,
                        reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(profile.logits_frame.byte_count, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(profile.batch_size, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(profile.cpu_thread_count, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(profile.prefill_chunk_size, reader->ReadI32());
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(profile));
  return profile;
}

absl::StatusOr<std::string> ReadBoundedString(Reader* reader,
                                              uint32_t maximum_size,
                                              absl::string_view description) {
  ABSL_ASSIGN_OR_RETURN(const uint32_t size, reader->ReadU32());
  if (size > maximum_size) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds its limit."));
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view bytes, reader->ReadBytes(size));
  return std::string(bytes.data(), bytes.size());
}

absl::StatusOr<CapsuleRestoreStateWitnessQualificationPolicy>
ReadStateWitnessQualificationPolicy(Reader* reader) {
  CapsuleRestoreStateWitnessQualificationPolicy policy;
  ABSL_ASSIGN_OR_RETURN(policy.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t content_authority, reader->ReadU32());
  policy.content_authority =
      static_cast<CapsuleRestoreStateWitnessContentAuthority>(
          content_authority);
  ABSL_ASSIGN_OR_RETURN(policy.required_operation_evidence_mask,
                        reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(policy.required_qualification_case_kind_mask,
                        reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(
      policy.minimum_independent_trials_per_shape_class_and_kind,
      reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(policy.maximum_qualified_shape_classes,
                        reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(policy.qualification_verifier_contract_hash,
                        reader->ReadHash());
  ABSL_RETURN_IF_ERROR(ValidateStateWitnessQualificationPolicyFields(policy));
  return policy;
}

absl::StatusOr<CapsuleRestoreStateWitnessOperationalDomain>
ReadStateWitnessOperationalDomain(Reader* reader) {
  CapsuleRestoreStateWitnessOperationalDomain domain;
  ABSL_ASSIGN_OR_RETURN(domain.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t capture_phase, reader->ReadU32());
  domain.capture_phase =
      static_cast<CapsuleRestoreStateWitnessCapturePhase>(capture_phase);
  ABSL_ASSIGN_OR_RETURN(const uint32_t admitted_backend, reader->ReadU32());
  if (admitted_backend != static_cast<uint32_t>(ExactLiteRtBackend::kCpu)) {
    return absl::UnimplementedError(
        "CapsuleRestore domain encoding is not CPU.");
  }
  domain.admitted_backend = static_cast<ExactLiteRtBackend>(admitted_backend);
  ABSL_ASSIGN_OR_RETURN(domain.resolved_session_config_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(domain.session_continuation_state_witness_contract_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(domain.capture_evidence_contract_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(domain.restore_evidence_contract_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(domain.deterministic_prefill_plan_contract_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(domain.execution_shape_class_contract_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(domain.restricted_feature_contract_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(domain.maximum_context_positions, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(domain.minimum_checkpoint_step, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(domain.maximum_checkpoint_step, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(domain.minimum_delta_positions, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(domain.maximum_delta_positions, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(domain.minimum_prefill_chunks, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(domain.maximum_prefill_chunks, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(domain.maximum_prefill_text_bytes, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(domain.maximum_prefill_token_ids, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(domain.maximum_output_tokens, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(domain.admitted_encoding_mask, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(domain.checkpoint_authentication_key_id,
                        ReadBoundedString(reader, kMaximumCheckpointKeyIdBytes,
                                          "CapsuleRestore checkpoint key ID"));
  ABSL_ASSIGN_OR_RETURN(
      domain.operation_evidence_authentication_key_id,
      ReadBoundedString(reader, kMaximumCheckpointKeyIdBytes,
                        "CapsuleRestore operation-evidence key ID"));
  ABSL_ASSIGN_OR_RETURN(domain.qualification_policy,
                        ReadStateWitnessQualificationPolicy(reader));
  ABSL_RETURN_IF_ERROR(ValidateStateWitnessOperationalDomainFields(domain));
  return domain;
}

absl::StatusOr<CapsuleRestoreStateWitnessQualificationCaseEvidence>
ReadStateWitnessQualificationCase(
    Reader* reader, const CapsuleRestoreStateWitnessOperationalDomain& domain) {
  CapsuleRestoreStateWitnessQualificationCaseEvidence evidence;
  ABSL_ASSIGN_OR_RETURN(evidence.qualification_case_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t kind, reader->ReadU32());
  evidence.kind =
      static_cast<CapsuleRestoreStateWitnessQualificationCaseKind>(kind);
  ABSL_ASSIGN_OR_RETURN(evidence.shape_class_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.trial_identity_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.source_session_instance_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.target_session_instance_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.checkpoint_step, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(evidence.delta_positions, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(evidence.prefill_chunk_count, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(evidence.prefill_text_bytes, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(evidence.prefill_token_ids, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(evidence.output_tokens, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(evidence.observed_encoding_mask, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(evidence.producer_state_witness_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.restored_state_witness_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.capture_evidence_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.restore_evidence_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.live_continuation_output_evidence_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.restored_continuation_output_evidence_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.verifier_certification_hash,
                        reader->ReadHash());
  ABSL_RETURN_IF_ERROR(
      ValidateStateWitnessQualificationCaseFields(evidence, domain, true));
  return evidence;
}

absl::StatusOr<CapsuleRestoreOperationalCoverage>
ReadStateWitnessOperationalCoverage(Reader* reader) {
  CapsuleRestoreOperationalCoverage coverage;
  ABSL_ASSIGN_OR_RETURN(coverage.coverage_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(coverage.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(coverage.runtime_derived_profile, ReadProfile(reader));
  ABSL_ASSIGN_OR_RETURN(coverage.runtime_derived_capability,
                        ReadCapability(reader));
  ABSL_ASSIGN_OR_RETURN(coverage.runtime_derived_session_identity,
                        ReadIdentity(reader));
  ABSL_ASSIGN_OR_RETURN(coverage.operational_domain,
                        ReadStateWitnessOperationalDomain(reader));
  ABSL_ASSIGN_OR_RETURN(coverage.qualification_evidence_hash,
                        reader->ReadHash());
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreOperationalCoverage(coverage));
  return coverage;
}

absl::StatusOr<CapsuleRestoreAdmissionRecord> DecodeRecordFields(
    absl::string_view body, const Hash256& record_id,
    absl::string_view envelope_key_id) {
  Reader reader(body);
  CapsuleRestoreAdmissionRecord record;
  record.record_id = record_id;
  ABSL_ASSIGN_OR_RETURN(record.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(record.operational_coverage,
                        ReadStateWitnessOperationalCoverage(&reader));
  ABSL_ASSIGN_OR_RETURN(const uint32_t case_count, reader.ReadU32());
  if (case_count > kMaximumCapsuleRestoreStateWitnessQualificationCases) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore qualification case count exceeds its "
        "limit.");
  }
  record.qualification_cases.reserve(case_count);
  for (uint32_t index = 0; index < case_count; ++index) {
    ABSL_ASSIGN_OR_RETURN(
        CapsuleRestoreStateWitnessQualificationCaseEvidence evidence,
        ReadStateWitnessQualificationCase(
            &reader, record.operational_coverage.operational_domain));
    record.qualification_cases.push_back(std::move(evidence));
  }
  ABSL_ASSIGN_OR_RETURN(const uint64_t qualified_time, reader.ReadU64());
  if (qualified_time >
      static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
    return absl::DataLossError(
        "CapsuleRestore admission timestamp is invalid.");
  }
  record.qualified_unix_micros = static_cast<int64_t>(qualified_time);
  ABSL_ASSIGN_OR_RETURN(
      record.record_authentication_key_id,
      ReadBoundedString(&reader, kMaximumAdmissionKeyIdBytes,
                        "CapsuleRestore admission-record key ID"));
  if (record.record_authentication_key_id != envelope_key_id) {
    return absl::UnauthenticatedError(
        "CapsuleRestore admission key ID binding does not "
        "match.");
  }
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "CapsuleRestore admission record has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAdmissionRecord(record));
  return record;
}

}  // namespace

absl::Status ValidateCapsuleRestoreStateWitnessQualificationPolicy(
    const CapsuleRestoreStateWitnessQualificationPolicy& policy) {
  return ValidateStateWitnessQualificationPolicyFields(policy);
}

absl::Status ValidateCapsuleRestoreStateWitnessOperationalDomain(
    const CapsuleRestoreStateWitnessOperationalDomain& domain) {
  return ValidateStateWitnessOperationalDomainFields(domain);
}

absl::StatusOr<Hash256> ComputeCapsuleRestoreStateWitnessQualificationCaseId(
    const CapsuleRestoreStateWitnessQualificationCaseEvidence& evidence) {
  // Case-ID computation validates all evidence fields except the ID being
  // computed. A caller cannot use an empty or malformed observation as a
  // canonical qualification case.
  // This function cannot infer the operational bounds. Validate the intrinsic
  // evidence here; domain membership is enforced by the public case validator
  // and by every aggregate/record operation below.
  if (evidence.format_version !=
          CapsuleRestoreStateWitnessQualificationCaseEvidence::kFormatVersion ||
      !IsStateWitnessQualificationCaseKind(evidence.kind) ||
      IsZeroHash(evidence.shape_class_hash) ||
      IsZeroHash(evidence.trial_identity_hash) ||
      IsZeroHash(evidence.source_session_instance_hash) ||
      IsZeroHash(evidence.target_session_instance_hash) ||
      evidence.source_session_instance_hash ==
          evidence.target_session_instance_hash ||
      evidence.checkpoint_step == 0 || evidence.delta_positions == 0 ||
      evidence.prefill_chunk_count == 0 || evidence.output_tokens == 0 ||
      evidence.observed_encoding_mask == 0 ||
      (evidence.observed_encoding_mask &
       ~CapsuleRestoreStateWitnessAllowedEncodingMask()) != 0 ||
      IsZeroHash(evidence.producer_state_witness_hash) ||
      evidence.producer_state_witness_hash !=
          evidence.restored_state_witness_hash ||
      IsZeroHash(evidence.capture_evidence_hash) ||
      IsZeroHash(evidence.restore_evidence_hash) ||
      IsZeroHash(evidence.live_continuation_output_evidence_hash) ||
      evidence.live_continuation_output_evidence_hash !=
          evidence.restored_continuation_output_evidence_hash ||
      IsZeroHash(evidence.verifier_certification_hash)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore cannot canonicalize an incomplete or "
        "non-matching qualification case.");
  }
  std::string canonical;
  AppendStateWitnessQualificationCaseFields(evidence, &canonical);
  Sha256Hasher hasher;
  hasher.Update(kStateWitnessQualificationCaseDomain);
  hasher.Update(canonical);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "CapsuleRestore produced a zero qualification case "
        "ID.");
  }
  return result;
}

absl::Status ValidateCapsuleRestoreStateWitnessQualificationCaseEvidence(
    const CapsuleRestoreStateWitnessQualificationCaseEvidence& evidence,
    const CapsuleRestoreStateWitnessOperationalDomain& domain) {
  return ValidateStateWitnessQualificationCaseFields(evidence, domain, true);
}

absl::StatusOr<Hash256>
ComputeCapsuleRestoreStateWitnessQualificationEvidenceHash(
    const CapsuleRestoreStateWitnessOperationalDomain& domain,
    const std::vector<CapsuleRestoreStateWitnessQualificationCaseEvidence>&
        qualification_cases) {
  ABSL_RETURN_IF_ERROR(ValidateStateWitnessQualificationEvidenceFields(
      domain, qualification_cases));
  std::string canonical;
  AppendStateWitnessOperationalDomain(domain, &canonical);
  AppendU32(static_cast<uint32_t>(qualification_cases.size()), &canonical);
  for (const auto& evidence : qualification_cases) {
    AppendHash(evidence.qualification_case_id, &canonical);
    AppendStateWitnessQualificationCaseFields(evidence, &canonical);
  }
  Sha256Hasher hasher;
  hasher.Update(kStateWitnessQualificationEvidenceDomain);
  hasher.Update(canonical);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "CapsuleRestore produced a zero qualification evidence "
        "hash.");
  }
  return result;
}

absl::StatusOr<Hash256> ComputeCapsuleRestoreOperationalCoverageId(
    const CapsuleRestoreOperationalCoverage& coverage) {
  ABSL_RETURN_IF_ERROR(
      ValidateStateWitnessOperationalCoverageFields(coverage, false));
  std::string canonical;
  AppendStateWitnessOperationalCoverageFields(coverage, &canonical);
  Sha256Hasher hasher;
  hasher.Update(kOperationalCoverageDomain);
  hasher.Update(canonical);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError("CapsuleRestore produced a zero coverage ID.");
  }
  return result;
}

absl::Status ValidateCapsuleRestoreOperationalCoverage(
    const CapsuleRestoreOperationalCoverage& coverage) {
  ABSL_RETURN_IF_ERROR(
      ValidateStateWitnessOperationalCoverageFields(coverage, true));
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_id,
                        ComputeCapsuleRestoreOperationalCoverageId(coverage));
  if (coverage.coverage_id != expected_id) {
    return absl::DataLossError("CapsuleRestore ID is not canonical.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCapsuleRestoreOperationalContracts(
    const CapsuleRestoreOperationalCoverage& coverage) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreOperationalCoverage(coverage));
  const CapsuleRestoreStateWitnessOperationalDomain& domain =
      coverage.operational_domain;
  if (domain.resolved_session_config_hash !=
          coverage.runtime_derived_session_identity.inference_profile_hash ||
      domain.session_continuation_state_witness_contract_hash !=
          GetSessionContinuationStateWitnessContractHash() ||
      domain.capture_evidence_contract_hash !=
          GetCapsuleRestoreCaptureEvidenceContractHash() ||
      domain.restore_evidence_contract_hash !=
          GetCapsuleRestoreRestoreEvidenceContractHash() ||
      domain.deterministic_prefill_plan_contract_hash !=
          GetDPMPreparedPrefillPlanContractHash() ||
      domain.execution_shape_class_contract_hash !=
          GetDPMPreparedPrefillShapeClassContractHash() ||
      domain.restricted_feature_contract_hash !=
          GetDPMRestrictedFeatureContractHash()) {
    return absl::FailedPreconditionError(
        "CapsuleRestore was qualified against another resolved "
        "session, witness, evidence, prefill, shape, or restricted-feature "
        "contract.");
  }
  return absl::OkStatus();
}

absl::StatusOr<CapsuleRestoreOperationalCoverage>
ComputeCapsuleRestoreOperationalCoverage(
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const CapsuleRestoreStateWitnessOperationalDomain& operational_domain,
    const std::vector<CapsuleRestoreStateWitnessQualificationCaseEvidence>&
        qualification_cases) {
  ABSL_RETURN_IF_ERROR(ValidateProfileCapabilityAgreement(
      runtime_derived_profile, runtime_derived_capability));
  if (runtime_derived_profile.backend != ExactLiteRtBackend::kCpu ||
      runtime_derived_capability.backend != ExactLiteRtBackend::kCpu) {
    return absl::UnimplementedError(
        "CapsuleRestore construction currently supports only "
        "runtime-derived CPU profiles.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateStateWitnessOperationalDomainFields(operational_domain));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 qualification_evidence_hash,
      ComputeCapsuleRestoreStateWitnessQualificationEvidenceHash(
          operational_domain, qualification_cases));
  CapsuleRestoreOperationalCoverage coverage;
  coverage.runtime_derived_profile = runtime_derived_profile;
  coverage.runtime_derived_capability = runtime_derived_capability;
  coverage.runtime_derived_session_identity =
      runtime_derived_profile.session_identity;
  coverage.operational_domain = operational_domain;
  coverage.qualification_evidence_hash = qualification_evidence_hash;
  ABSL_ASSIGN_OR_RETURN(coverage.coverage_id,
                        ComputeCapsuleRestoreOperationalCoverageId(coverage));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreOperationalContracts(coverage));
  return coverage;
}

absl::StatusOr<Hash256> ComputeCapsuleRestoreAdmissionRecordId(
    const CapsuleRestoreAdmissionRecord& record) {
  ABSL_RETURN_IF_ERROR(
      ValidateStateWitnessAdmissionRecordFields(record, false));
  std::string canonical;
  AppendStateWitnessAdmissionRecordFields(record, &canonical);
  Sha256Hasher hasher;
  hasher.Update(kAdmissionRecordDomain);
  hasher.Update(canonical);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "CapsuleRestore produced a zero admission record ID.");
  }
  return result;
}

absl::Status ValidateCapsuleRestoreAdmissionRecord(
    const CapsuleRestoreAdmissionRecord& record) {
  return ValidateStateWitnessAdmissionRecordFields(record, true);
}

absl::Status ValidateCapsuleRestoreAdmissionRecordForRuntime(
    const CapsuleRestoreAdmissionRecord& record,
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const Hash256& expected_coverage_id) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAdmissionRecord(record));
  ABSL_RETURN_IF_ERROR(ValidateProfileCapabilityAgreement(
      runtime_derived_profile, runtime_derived_capability));
  if (IsZeroHash(expected_coverage_id) ||
      record.operational_coverage.coverage_id != expected_coverage_id) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission differs from the requested "
        "durable coverage ID.");
  }
  if (record.operational_coverage.runtime_derived_profile !=
          runtime_derived_profile ||
      record.operational_coverage.runtime_derived_capability !=
          runtime_derived_capability ||
      record.operational_coverage.runtime_derived_session_identity !=
          runtime_derived_profile.session_identity) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission differs from the current "
        "complete Engine-derived profile, capability, or session identity.");
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> ComputeCapsuleRestoreAdmissionLookupKey(
    const Hash256& exact_profile_id, const Hash256& capability_id,
    const Hash256& coverage_id) {
  if (IsZeroHash(exact_profile_id) || IsZeroHash(capability_id) ||
      IsZeroHash(coverage_id)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore lookup requires nonzero profile, "
        "capability, and coverage IDs.");
  }
  std::string canonical;
  AppendHash(exact_profile_id, &canonical);
  AppendHash(capability_id, &canonical);
  AppendHash(coverage_id, &canonical);
  Sha256Hasher hasher;
  hasher.Update(kAdmissionLookupDomain);
  hasher.Update(canonical);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError("CapsuleRestore produced a zero lookup key.");
  }
  return result;
}

absl::StatusOr<std::string> EncodeCapsuleRestoreAdmissionRecord(
    const CapsuleRestoreAdmissionRecord& record,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAdmissionRecord(record));
  if (record.record_authentication_key_id != authentication.key_id) {
    return absl::InvalidArgumentError(
        "CapsuleRestore admission record key ID does not match "
        "its authentication input.");
  }
  std::string body;
  AppendHash(record.record_id, &body);
  AppendStateWitnessAdmissionRecordFields(record, &body);
  if (authentication.key_id.size() >
          kMaximumCapsuleRestoreAdmissionEnvelopeBytes -
              kAdmissionEnvelopeFixedBytes ||
      body.size() > kMaximumCapsuleRestoreAdmissionEnvelopeBytes -
                        kAdmissionEnvelopeFixedBytes -
                        authentication.key_id.size()) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore admission exceeds its storage limit.");
  }
  std::string envelope;
  envelope.reserve(kAdmissionEnvelopeFixedBytes + authentication.key_id.size() +
                   body.size());
  envelope.append(kAdmissionMagic.data(), kAdmissionMagic.size());
  AppendU32(kAdmissionEnvelopeFormatVersion, &envelope);
  AppendU32(static_cast<uint32_t>(authentication.key_id.size()), &envelope);
  AppendU64(body.size(), &envelope);
  envelope.append(authentication.key_id);
  envelope.append(body);
  const Hash256 mac = HmacSha256(authentication.authentication_key,
                                 {kAdmissionMacDomain, envelope});
  AppendHash(mac, &envelope);
  return envelope;
}

absl::StatusOr<CapsuleRestoreAdmissionRecord>
DecodeCapsuleRestoreAdmissionRecord(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (envelope.size() < kAdmissionEnvelopeFixedBytes + 32 ||
      envelope.size() > kMaximumCapsuleRestoreAdmissionEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore admission envelope is outside its size "
        "bounds.");
  }
  if (std::memcmp(envelope.data(), kAdmissionMagic.data(),
                  kAdmissionMagic.size()) != 0) {
    return absl::DataLossError(
        "CapsuleRestore admission envelope magic is invalid.");
  }
  Reader header(envelope.substr(kAdmissionMagic.size()));
  ABSL_ASSIGN_OR_RETURN(const uint32_t version, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t key_id_size, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint64_t body_size, header.ReadU64());
  if (version != kAdmissionEnvelopeFormatVersion) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission envelope version is "
        "unsupported.");
  }
  if (key_id_size > kMaximumAdmissionKeyIdBytes || body_size < 32 ||
      body_size > kMaximumCapsuleRestoreAdmissionEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore admission envelope declares an "
        "oversized field.");
  }
  const uint64_t bytes_before_mac =
      uint64_t{8} + 4 + 4 + 8 + key_id_size + body_size;
  if (bytes_before_mac > kMaximumCapsuleRestoreAdmissionEnvelopeBytes - 32 ||
      envelope.size() != bytes_before_mac + 32) {
    return absl::DataLossError(
        "CapsuleRestore admission envelope length is "
        "non-canonical.");
  }
  const size_t key_offset = 8 + 4 + 4 + 8;
  const absl::string_view encoded_key =
      envelope.substr(key_offset, key_id_size);
  if (encoded_key != authentication.key_id) {
    return absl::UnauthenticatedError(
        "CapsuleRestore admission envelope key ID does not "
        "match.");
  }
  Hash256 encoded_mac;
  std::memcpy(encoded_mac.bytes.data(),
              envelope.data() + static_cast<size_t>(bytes_before_mac),
              encoded_mac.bytes.size());
  const absl::string_view authenticated_bytes =
      envelope.substr(0, static_cast<size_t>(bytes_before_mac));
  const Hash256 expected_mac =
      HmacSha256(authentication.authentication_key,
                 {kAdmissionMacDomain, authenticated_bytes});
  if (!ConstantTimeHashEquals(encoded_mac, expected_mac)) {
    return absl::UnauthenticatedError(
        "CapsuleRestore admission envelope authentication "
        "failed.");
  }
  const size_t body_offset = key_offset + key_id_size;
  Reader body_reader(
      envelope.substr(body_offset, static_cast<size_t>(body_size)));
  ABSL_ASSIGN_OR_RETURN(const Hash256 encoded_record_id,
                        body_reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(
      CapsuleRestoreAdmissionRecord record,
      DecodeRecordFields(envelope.substr(body_offset + 32,
                                         static_cast<size_t>(body_size) - 32),
                         encoded_record_id, authentication.key_id));
  if (record.record_id != encoded_record_id) {
    return absl::DataLossError(
        "CapsuleRestore admission envelope record ID is "
        "inconsistent.");
  }
  return record;
}

absl::StatusOr<AuthenticatedCapsuleRestoreAdmission>
ResolveAuthenticatedCapsuleRestoreAdmission(
    const Engine* engine, const SessionConfig& runtime_session_config,
    const CapsuleRestoreAdmissionBinding& binding) {
  if (engine == nullptr) {
    return absl::InvalidArgumentError(
        "CapsuleRestore requires a loaded authoritative Engine.");
  }
  if (binding.repository == nullptr ||
      IsZeroHash(binding.expected_coverage_id)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore binding lacks its repository or expected "
        "coverage ID.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerAuthentication(binding.record_authentication));

  SessionConfig resolved_runtime_config = runtime_session_config;
  ABSL_RETURN_IF_ERROR(resolved_runtime_config.MaybeUpdateAndValidate(
      engine->GetEngineSettings()));
  ABSL_RETURN_IF_ERROR(
      ValidateResolvedSessionConfig(resolved_runtime_config, *engine));

  ABSL_ASSIGN_OR_RETURN(
      ExactLiteRtProfile runtime_profile,
      engine->ResolveExactLiteRtProfile(resolved_runtime_config,
                                        binding.profile_assertion));
  ABSL_ASSIGN_OR_RETURN(
      SessionHandoffCapability runtime_capability,
      engine->ResolveSessionHandoffCapability(resolved_runtime_config,
                                              binding.capability_assertion));
  ABSL_ASSIGN_OR_RETURN(
      const SessionHandoffIdentity runtime_identity,
      engine->ResolveSessionHandoffIdentity(resolved_runtime_config));
  ABSL_RETURN_IF_ERROR(
      ValidateProfileCapabilityAgreement(runtime_profile, runtime_capability));
  if (runtime_identity != runtime_profile.session_identity ||
      runtime_identity != runtime_capability.session_identity) {
    return absl::FailedPreconditionError(
        "CapsuleRestore runtime SessionConfig identity disagrees with its "
        "Engine-derived profile or capability.");
  }

  ABSL_ASSIGN_OR_RETURN(
      CapsuleRestoreAdmissionRecord record,
      binding.repository->Get(runtime_profile, runtime_capability,
                              binding.expected_coverage_id,
                              binding.record_authentication));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAdmissionRecordForRuntime(
      record, runtime_profile, runtime_capability,
      binding.expected_coverage_id));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreOperationalContracts(record.operational_coverage));
  ABSL_ASSIGN_OR_RETURN(const Hash256 canonical_record_id,
                        ComputeCapsuleRestoreAdmissionRecordId(record));
  if (record.record_id != canonical_record_id ||
      record.record_authentication_key_id !=
          binding.record_authentication.key_id ||
      record.operational_coverage.runtime_derived_profile != runtime_profile ||
      record.operational_coverage.runtime_derived_capability !=
          runtime_capability ||
      record.operational_coverage.runtime_derived_session_identity !=
          runtime_identity) {
    return absl::DataLossError(
        "Authenticated CapsuleRestore admission is not "
        "canonically bound to the "
        "current Engine evidence and record key.");
  }
  CapsuleRestoreOperationalCoverage operational_coverage =
      record.operational_coverage;
  return AuthenticatedCapsuleRestoreAdmission{
      .record = std::move(record),
      .profile = std::move(runtime_profile),
      .capability = std::move(runtime_capability),
      .operational_coverage = std::move(operational_coverage),
  };
}

}  // namespace litert::lm
