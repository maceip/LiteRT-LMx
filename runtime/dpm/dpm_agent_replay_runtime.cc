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

#include "runtime/dpm/dpm_agent_replay_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/canonical_replay_catalog.h"
#include "runtime/dpm/capsule_restore_evidence_binding.h"
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_replay_executor.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/exact_profile_admission.h"
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kAgentExecutionMagic = {'D', 'P', 'M', 'A',
                                                       'G', 'N', '0', '1'};
constexpr std::array<char, 8> kAgentDeltaExecutionMagic = {
    'D', 'P', 'M', 'D', 'L', 'T', '0', '1'};
constexpr std::array<char, 8> kAgentDecisionMagic = {'D', 'P', 'M', 'D',
                                                      'E', 'C', '0', '1'};
constexpr uint64_t kAgentExecutionFixedBytes = 8 + 4 + 32 + 32 + 4 + 4;
constexpr uint64_t kAgentDeltaExecutionFixedBytes =
    8 + 4 + 32 + 32 + 32 + 8 + 32 + 4 + 4;
constexpr uint64_t kAgentChunkFramingBytes = 1 + 8;
constexpr uint64_t kCanonicalTokenFramingBytes = 8 + 4 + 4;
constexpr absl::string_view kWinnerAgentEvidenceDomain =
    "LITERT_LMX_DPM_WINNER_AGENT_EVIDENCE_SHA256_V1";

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

absl::Status ValidateRuntimeIdentity(
    const SessionHandoffIdentity& identity) {
  if (IsZeroHash(identity.model_artifact_hash) ||
      IsZeroHash(identity.runtime_artifact_hash) ||
      IsZeroHash(identity.inference_profile_hash)) {
    return absl::FailedPreconditionError(
        "DPM agent replay requires a complete runtime-derived identity.");
  }
  return absl::OkStatus();
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

void AppendHash(const Hash256& hash, std::string* output) {
  output->append(reinterpret_cast<const char*>(hash.bytes.data()),
                 hash.bytes.size());
}

class Reader {
 public:
  explicit Reader(absl::string_view bytes) : bytes_(bytes) {}

  absl::StatusOr<uint8_t> ReadU8() {
    if (remaining() < 1) return Truncated();
    return static_cast<uint8_t>(bytes_[offset_++]);
  }

  absl::StatusOr<uint32_t> ReadU32() {
    if (remaining() < 4) return Truncated();
    uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      value = (value << 8) |
              static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 4;
    return value;
  }

  absl::StatusOr<uint64_t> ReadU64() {
    if (remaining() < 8) return Truncated();
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
      value = (value << 8) |
              static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 8;
    return value;
  }

  absl::StatusOr<Hash256> ReadHash() {
    if (remaining() < Hash256{}.bytes.size()) return Truncated();
    Hash256 hash;
    std::memcpy(hash.bytes.data(), bytes_.data() + offset_, hash.bytes.size());
    offset_ += hash.bytes.size();
    return hash;
  }

  absl::StatusOr<absl::string_view> ReadBytes(uint64_t size) {
    if (size > remaining() ||
        size > (std::numeric_limits<size_t>::max)()) {
      return Truncated();
    }
    absl::string_view result =
        bytes_.substr(offset_, static_cast<size_t>(size));
    offset_ += static_cast<size_t>(size);
    return result;
  }

  size_t remaining() const { return bytes_.size() - offset_; }

 private:
  absl::Status Truncated() const {
    return absl::DataLossError("Truncated canonical DPM agent envelope.");
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

absl::StatusOr<Hash256> ComputeWinnerEvidenceHash(
    const SessionHandoffIdentity& identity,
    const DPMCanonicalReplayRequest& request,
    absl::string_view canonical_output) {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(identity));
  Sha256Hasher hasher;
  hasher.Update(kWinnerAgentEvidenceDomain);
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(identity.model_artifact_hash.bytes.data()),
      identity.model_artifact_hash.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(identity.runtime_artifact_hash.bytes.data()),
      identity.runtime_artifact_hash.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(identity.inference_profile_hash.bytes.data()),
      identity.inference_profile_hash.bytes.size()));
  ABSL_ASSIGN_OR_RETURN(const Hash256 request_hash,
                        ComputeDPMCanonicalReplayRequestHash(request));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(request_hash.bytes.data()),
      request_hash.bytes.size()));
  std::string length;
  AppendU64(canonical_output.size(), &length);
  hasher.Update(length);
  hasher.Update(canonical_output);
  return hasher.Finalize();
}

absl::Status ValidateChunk(
    const DPMAgentGenerationRequest::PrefillChunk& chunk) {
  switch (chunk.encoding) {
    case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text:
      if (chunk.text.empty() || !chunk.token_ids.empty() ||
          !IsValidUtf8(chunk.text)) {
        return absl::InvalidArgumentError(
            "Canonical DPM agent text chunk is empty, non-UTF-8, or mixed "
            "with token IDs.");
      }
      break;
    case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds:
      if (!chunk.text.empty() || chunk.token_ids.empty() ||
          chunk.token_ids.size() > kMaximumFreshWorkerTokenIds) {
        return absl::InvalidArgumentError(
            "Canonical DPM agent token chunk is empty, oversized, or mixed "
            "with text.");
      }
      for (int token_id : chunk.token_ids) {
        if (token_id < 0 ||
            static_cast<int64_t>(token_id) >
                std::numeric_limits<int32_t>::max()) {
          return absl::InvalidArgumentError(
              "Canonical DPM agent chunk contains a non-int32 token ID.");
        }
      }
      break;
    default:
      return absl::InvalidArgumentError(
          "Canonical DPM agent chunk has an unknown encoding.");
  }
  return absl::OkStatus();
}

bool ChunksEqual(
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& left,
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& right) {
  if (left.size() != right.size()) return false;
  for (size_t index = 0; index < left.size(); ++index) {
    if (left[index].encoding != right[index].encoding ||
        left[index].text != right[index].text ||
        left[index].token_ids != right[index].token_ids) {
      return false;
    }
  }
  return true;
}

absl::Status ValidatePreparedPlanSourceBindings(
    const DPMPreparedPrefillPlan& plan,
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& chunks) {
  if (plan.calls.size() != chunks.size()) {
    return absl::DataLossError(
        "Prepared DPM prefill plan changed the canonical source-call count.");
  }
  for (size_t index = 0; index < chunks.size(); ++index) {
    const DPMAgentGenerationRequest::PrefillChunk& chunk = chunks[index];
    const DPMPreparedPrefillCall& call = plan.calls[index];
    ABSL_RETURN_IF_ERROR(ValidateChunk(chunk));
    Hash256 expected_hash;
    DPMPreparedPrefillSourceEncoding expected_encoding;
    switch (chunk.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text: {
        expected_encoding = DPMPreparedPrefillSourceEncoding::kUtf8Text;
        ABSL_ASSIGN_OR_RETURN(
            expected_hash,
            ComputeDPMPreparedPrefillUtf8SourceChunkHash(chunk.text));
        break;
      }
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds: {
        expected_encoding =
            DPMPreparedPrefillSourceEncoding::kExactTokenIds;
        std::vector<int32_t> exact_ids;
        exact_ids.reserve(chunk.token_ids.size());
        for (int token_id : chunk.token_ids) {
          exact_ids.push_back(static_cast<int32_t>(token_id));
        }
        ABSL_ASSIGN_OR_RETURN(
            expected_hash,
            ComputeDPMPreparedPrefillExactTokenSourceChunkHash(exact_ids));
        break;
      }
      default:
        return absl::InvalidArgumentError(
            "Prepared DPM prefill source has an unknown encoding.");
    }
    if (call.source_encoding != expected_encoding ||
        call.source_chunk_hash != expected_hash) {
      return absl::DataLossError(
          "Prepared DPM prefill plan changed a canonical source chunk.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<CapsuleRestorePrefillChunk>>
ToCapsuleRestoreChunks(
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& chunks) {
  std::vector<CapsuleRestorePrefillChunk> converted;
  converted.reserve(chunks.size());
  for (const DPMAgentGenerationRequest::PrefillChunk& chunk : chunks) {
    ABSL_RETURN_IF_ERROR(ValidateChunk(chunk));
    CapsuleRestorePrefillChunk converted_chunk;
    switch (chunk.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text:
        converted_chunk.encoding =
            CapsuleRestorePrefillChunk::Encoding::kUtf8Text;
        converted_chunk.utf8_text = chunk.text;
        break;
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds:
        converted_chunk.encoding =
            CapsuleRestorePrefillChunk::Encoding::kExactTokenIds;
        converted_chunk.token_ids.reserve(chunk.token_ids.size());
        for (int token_id : chunk.token_ids) {
          converted_chunk.token_ids.push_back(
              static_cast<int32_t>(token_id));
        }
        break;
      default:
        return absl::InvalidArgumentError(
            "CapsuleRestore operation has an unknown chunk encoding.");
    }
    converted.push_back(std::move(converted_chunk));
  }
  return converted;
}

bool ExecutionRequestsEqual(const DPMAgentExecutionRequest& left,
                            const DPMAgentExecutionRequest& right) {
  return left.format_version == right.format_version &&
         left.logical_agent_request_hash == right.logical_agent_request_hash &&
         left.correction_digest == right.correction_digest &&
         left.max_output_tokens == right.max_output_tokens &&
         ChunksEqual(left.full_canonical_prefill_chunks,
                     right.full_canonical_prefill_chunks);
}

bool HasSameStateWitnessAdmissionAuthority(
    const AuthenticatedCapsuleRestoreStateWitnessAdmission& lhs,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission& rhs) {
  return lhs.record.record_id == rhs.record.record_id &&
         lhs.profile == rhs.profile && lhs.capability == rhs.capability &&
         lhs.operational_coverage == rhs.operational_coverage &&
         lhs.record.operational_coverage == lhs.operational_coverage &&
         rhs.record.operational_coverage == rhs.operational_coverage;
}

absl::Status ValidateStateWitnessAdmissionForAgentRuntime(
    const AuthenticatedCapsuleRestoreStateWitnessAdmission& admission,
    const SessionHandoffIdentity& expected_identity,
    const std::optional<ExactLiteRtProfile>& expected_exact_profile) {
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(expected_identity));
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
      const CapsuleRestoreAuthorityV2 authority,
      BuildCapsuleRestoreAuthorityV2(admission));
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleRestoreAuthorityV2ForAdmission(authority, admission));
  const CapsuleRestoreStateWitnessOperationalCoverage& coverage =
      admission.operational_coverage;
  const CapsuleRestoreStateWitnessOperationalDomain& domain =
      coverage.operational_domain;
  if (admission.record.operational_coverage != coverage ||
      coverage.runtime_derived_profile != admission.profile ||
      coverage.runtime_derived_capability != admission.capability ||
      coverage.runtime_derived_session_identity != expected_identity ||
      admission.profile.session_identity != expected_identity ||
      admission.capability.session_identity != expected_identity ||
      admission.capability.exact_profile_id != admission.profile.profile_id ||
      admission.capability.backend != admission.profile.backend ||
      domain.admitted_backend != admission.profile.backend ||
      domain.resolved_session_config_hash !=
          expected_identity.inference_profile_hash ||
      (expected_exact_profile.has_value() &&
       admission.profile != *expected_exact_profile)) {
    return absl::FailedPreconditionError(
        "CapsuleRestore Coverage V2 admission is detached from the loaded "
        "agent runtime's profile, capability, or operational domain.");
  }
  return absl::OkStatus();
}

absl::Status ValidateStateWitnessSourceDomain(
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& chunks,
    uint32_t maximum_output_tokens,
    const CapsuleRestoreStateWitnessOperationalDomain& domain) {
  if (chunks.size() < domain.minimum_prefill_chunks ||
      chunks.size() > domain.maximum_prefill_chunks ||
      maximum_output_tokens == 0 ||
      maximum_output_tokens > domain.maximum_output_tokens) {
    return absl::FailedPreconditionError(
        "Exact agent request is outside the CapsuleRestore Coverage V2 "
        "prefill-count or output-token domain.");
  }
  uint64_t text_bytes = 0;
  uint64_t token_ids = 0;
  uint32_t encoding_mask = 0;
  for (const DPMAgentGenerationRequest::PrefillChunk& chunk : chunks) {
    ABSL_RETURN_IF_ERROR(ValidateChunk(chunk));
    switch (chunk.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text:
        if (text_bytes > domain.maximum_prefill_text_bytes ||
            chunk.text.size() >
                domain.maximum_prefill_text_bytes - text_bytes) {
          return absl::FailedPreconditionError(
              "Exact agent text prefill exceeds the CapsuleRestore Coverage "
              "V2 domain.");
        }
        text_bytes += chunk.text.size();
        encoding_mask |= CapsuleRestoreStateWitnessEncodingBit(
            CapsuleRestoreStateWitnessEncoding::kUtf8Text);
        break;
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds:
        if (token_ids > domain.maximum_prefill_token_ids ||
            chunk.token_ids.size() >
                domain.maximum_prefill_token_ids - token_ids) {
          return absl::FailedPreconditionError(
              "Exact agent token prefill exceeds the CapsuleRestore Coverage "
              "V2 domain.");
        }
        token_ids += chunk.token_ids.size();
        encoding_mask |= CapsuleRestoreStateWitnessEncodingBit(
            CapsuleRestoreStateWitnessEncoding::kExactTokenIds);
        break;
      default:
        return absl::InvalidArgumentError(
            "Exact agent prefill contains an unknown encoding.");
    }
  }
  if (encoding_mask == 0 ||
      (encoding_mask & ~domain.admitted_encoding_mask) != 0) {
    return absl::FailedPreconditionError(
        "Exact agent prefill encoding is outside the CapsuleRestore Coverage "
        "V2 domain.");
  }
  return absl::OkStatus();
}

absl::Status ValidateStateWitnessPreparedPlanDomain(
    const DPMPreparedPrefillPlan& plan, uint32_t maximum_output_tokens,
    bool restored,
    const CapsuleRestoreStateWitnessOperationalDomain& domain) {
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(plan));
  if (plan.end_step > domain.maximum_context_positions ||
      maximum_output_tokens > domain.maximum_context_positions ||
      plan.end_step >
          domain.maximum_context_positions - maximum_output_tokens) {
    return absl::FailedPreconditionError(
        "Exact agent prefill or decode exceeds the CapsuleRestore Coverage V2 "
        "context domain.");
  }
  if (!restored) {
    if (plan.start_kind != DPMPreparedPrefillStartKind::kFreshSession ||
        plan.start_step != 0) {
      return absl::FailedPreconditionError(
          "Coverage V2 root capture did not begin from a fresh session.");
    }
    return absl::OkStatus();
  }
  if (plan.start_kind !=
          DPMPreparedPrefillStartKind::kOwnPositionRestore ||
      plan.start_step < domain.minimum_checkpoint_step ||
      plan.start_step > domain.maximum_checkpoint_step ||
      plan.end_step <= plan.start_step) {
    return absl::FailedPreconditionError(
        "Exact agent restored prefill is outside the CapsuleRestore Coverage "
        "V2 checkpoint domain.");
  }
  const uint64_t delta_positions = plan.end_step - plan.start_step;
  if (delta_positions < domain.minimum_delta_positions ||
      delta_positions > domain.maximum_delta_positions) {
    return absl::FailedPreconditionError(
        "Exact agent restored prefill is outside the CapsuleRestore Coverage "
        "V2 delta-position domain.");
  }
  return absl::OkStatus();
}

absl::Status ValidateWinnerRestoreOperationAuthority(
    const DPMAgentGenerationRequest& execution_request,
    const DPMAgentExecutionRequest& logical_request,
    const AuthenticatedCapsuleRestoreStateWitnessAdmission& admission) {
  if (!execution_request.capsule_restore_operation_v3.has_value() ||
      !execution_request.restore_checkpoint_id.has_value() ||
      !execution_request.restored_state_witness.has_value()) {
    return absl::InvalidArgumentError(
        "WinnerReplay Coverage V2 restore requires its complete operation, "
        "checkpoint, and live witness.");
  }
  const DPMAgentCapsuleRestoreOperationV3& operation =
      *execution_request.capsule_restore_operation_v3;
  if (operation.format_version !=
      DPMAgentCapsuleRestoreOperationV3::kFormatVersion) {
    return absl::FailedPreconditionError(
        "WinnerReplay CapsuleRestore operation version is unsupported.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateCapsuleCaptureEvidenceV3(operation.source_capture_evidence));
  ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(
      *execution_request.restored_state_witness));
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffReauthenticationEvidence(
      operation.durable_to_transient_reauthentication));
  ABSL_ASSIGN_OR_RETURN(
      const CapsuleRestoreAuthorityV2 expected_authority,
      BuildCapsuleRestoreAuthorityV2(admission));
  const CapsuleCaptureEvidenceV3& source =
      operation.source_capture_evidence;
  const SessionContinuationStateWitness& witness =
      *execution_request.restored_state_witness;
  const SessionHandoffReauthenticationEvidence& reauthentication =
      operation.durable_to_transient_reauthentication;
  const CapsuleRestoreStateWitnessOperationalDomain& domain =
      admission.operational_coverage.operational_domain;
  if (operation.current_authority != expected_authority ||
      source.plan.authority != expected_authority ||
      source.checkpoint_id != *execution_request.restore_checkpoint_id ||
      source.checkpoint_authentication_key_id !=
          domain.checkpoint_authentication_key_id ||
      source.plan.checkpoint_authentication_key_id !=
          domain.checkpoint_authentication_key_id ||
      operation.target_state.logical_agent_request_hash !=
          logical_request.logical_agent_request_hash ||
      operation.target_state.correction_digest !=
          logical_request.correction_digest ||
      logical_request.max_output_tokens > domain.maximum_output_tokens ||
      witness.session_identity != admission.profile.session_identity ||
      witness.phase != SessionHandoffPhase::kDecoded || !witness.ran_decode ||
      witness.current_step <= 0 ||
      static_cast<uint64_t>(witness.current_step) !=
          source.plan.capture_end_step ||
      witness.processed_history_token_bytes_hash !=
          source.checkpoint_history_token_bytes_hash ||
      reauthentication.session_identity != admission.profile.session_identity ||
      reauthentication.source_envelope_hash !=
          source.checkpoint_envelope_hash ||
      reauthentication.source_envelope_size !=
          source.checkpoint_envelope_size ||
      reauthentication.source_key_id !=
          source.checkpoint_authentication_key_id ||
      reauthentication.destination_envelope_hash != witness.envelope_hash ||
      reauthentication.destination_envelope_size != witness.envelope_size ||
      reauthentication.destination_key_id != witness.key_id ||
      reauthentication.destination_key_id !=
          kFreshWorkerTransientRestoreKeyId ||
      reauthentication.purpose !=
          kFreshWorkerDurableRestoreToTransientReauthenticationPurpose ||
      reauthentication.capsule_codec_contract_hash !=
          admission.capability.capsule_codec_contract_hash ||
      reauthentication.canonical_continuation_state_hash !=
          source.transient_to_durable_reauthentication
              .canonical_continuation_state_hash) {
    return absl::FailedPreconditionError(
        "WinnerReplay CapsuleRestore operation is outside the freshly "
        "reauthenticated authority or does not bind the durable source to the "
        "live target witness.");
  }
  return ValidateStateWitnessSourceDomain(
      execution_request.canonical_prefill_chunks,
      static_cast<uint32_t>(execution_request.max_output_tokens), domain);
}

absl::Status ValidateGenerationRequest(
    const DPMAgentGenerationRequest& request) {
  if (request.max_output_tokens <= 0 ||
      request.max_output_tokens >
          static_cast<int>(kMaximumDPMGenerationTokens) ||
      IsZeroHash(request.logical_agent_request_hash) ||
      request.canonical_prefill_chunks.empty() ||
      request.canonical_prefill_chunks.size() >
          kMaximumFreshWorkerTokenIds) {
    return absl::InvalidArgumentError(
        "DPM agent generation request is incomplete or oversized.");
  }
  if (request.restore_checkpoint_id.has_value() !=
      request.restored_state_witness.has_value()) {
    return absl::InvalidArgumentError(
        "DPM agent generation restore lacks its checkpoint or live witness.");
  }
  if (request.restore_checkpoint_id.has_value()) {
    if (IsZeroHash(*request.restore_checkpoint_id)) {
      return absl::InvalidArgumentError(
          "DPM agent generation restore has an empty checkpoint ID.");
    }
    ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(
        *request.restored_state_witness));
    if (request.restored_state_witness->phase !=
            SessionHandoffPhase::kDecoded ||
        !request.restored_state_witness->ran_decode) {
      return absl::InvalidArgumentError(
          "DPM agent generation restore witness is not a decoded producing "
          "boundary.");
    }
  }
  uint64_t total_token_ids = 0;
  for (const auto& chunk : request.canonical_prefill_chunks) {
    ABSL_RETURN_IF_ERROR(ValidateChunk(chunk));
    if (chunk.encoding ==
        DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds) {
      if (chunk.token_ids.size() >
          kMaximumFreshWorkerTokenIds - total_token_ids) {
        return absl::ResourceExhaustedError(
            "DPM agent generation request exceeds the token-input limit.");
      }
      total_token_ids += chunk.token_ids.size();
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateDecisionFields(absl::string_view decision_output,
                                    const std::vector<int32_t>& token_ids,
                                    uint32_t max_output_tokens) {
  if (decision_output.size() > kMaximumDPMEventPayloadBytes ||
      !IsValidUtf8(decision_output) || token_ids.empty() ||
      token_ids.size() > max_output_tokens) {
    return absl::DataLossError(
        "DPM agent decision text or exact token sequence is invalid.");
  }
  for (int32_t token_id : token_ids) {
    if (token_id < 0) {
      return absl::DataLossError(
          "DPM agent decision contains a negative token ID.");
    }
  }
  return absl::OkStatus();
}

class WinnerAgentInvocation final : public CanonicalWinnerReplayGenerator {
 public:
  WinnerAgentInvocation(DPMAgentRuntime* inference_runtime,
                        Engine::Session* session,
                        const DPMAgentGenerationRequest* execution_request,
                        const DPMAgentExecutionRequest* logical_request,
                        SessionHandoffIdentity runtime_identity)
      : inference_runtime_(inference_runtime),
        session_(session),
        execution_request_(execution_request),
        logical_request_(logical_request),
        runtime_identity_(runtime_identity) {}

  const SessionHandoffIdentity& GetRuntimeIdentity() const override {
    return runtime_identity_;
  }

  absl::Status ValidateSupport(DPMReplayStage expected_stage) const override {
    if (expected_stage != DPMReplayStage::kAgentDecision ||
        inference_runtime_ == nullptr || session_ == nullptr ||
        execution_request_ == nullptr || logical_request_ == nullptr) {
      return absl::InvalidArgumentError(
          "WinnerReplay agent invocation is incomplete or has another "
          "stage.");
    }
    ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(runtime_identity_));
    ABSL_RETURN_IF_ERROR(ValidateGenerationRequest(*execution_request_));
    ABSL_RETURN_IF_ERROR(
        ValidateDPMAgentExecutionRequest(*logical_request_));
    if (static_cast<uint32_t>(execution_request_->max_output_tokens) !=
            logical_request_->max_output_tokens ||
        execution_request_->logical_agent_request_hash !=
            logical_request_->logical_agent_request_hash) {
      return absl::InvalidArgumentError(
          "WinnerReplay live generation and logical request bindings differ.");
    }
    if (execution_request_->restored_state_witness.has_value() &&
        execution_request_->restored_state_witness->session_identity !=
            runtime_identity_) {
      return absl::FailedPreconditionError(
          "WinnerReplay restore witness belongs to another runtime.");
    }
    if (inference_runtime_->GetSessionHandoffIdentity() != runtime_identity_) {
      return absl::FailedPreconditionError(
          "WinnerReplay agent runtime identity changed during execution.");
    }
    ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity session_identity,
                          session_->GetSessionHandoffIdentity());
    if (session_identity != runtime_identity_) {
      return absl::FailedPreconditionError(
          "WinnerReplay agent session has another runtime identity.");
    }
    ABSL_ASSIGN_OR_RETURN(
        const std::optional<
            AuthenticatedCapsuleRestoreStateWitnessAdmission>
            state_witness_admission,
        inference_runtime_
            ->GetAuthenticatedCapsuleRestoreStateWitnessAdmission());
    const bool is_restore =
        execution_request_->restore_checkpoint_id.has_value();
    if (execution_request_->capsule_restore_operation_v3.has_value() !=
        (state_witness_admission.has_value() && is_restore)) {
      return absl::FailedPreconditionError(
          "WinnerReplay invocation requires Coverage V2 operation evidence "
          "exactly for a restore on its current loaded-Engine authority.");
    }
    if (state_witness_admission.has_value()) {
      ABSL_RETURN_IF_ERROR(ValidateStateWitnessAdmissionForAgentRuntime(
          *state_witness_admission, runtime_identity_, std::nullopt));
    }
    if (state_witness_admission.has_value() && is_restore) {
      ABSL_RETURN_IF_ERROR(ValidateWinnerRestoreOperationAuthority(
          *execution_request_, *logical_request_,
          *state_witness_admission));
    } else if (is_restore) {
      ABSL_RETURN_IF_ERROR(
          inference_runtime_->ValidateSessionHandoffSupport());
    }
    return absl::OkStatus();
  }

  absl::StatusOr<CanonicalWinnerGeneratedCandidate> Generate(
      const DPMCanonicalReplayRequest& request) override {
    ABSL_RETURN_IF_ERROR(ValidateSupport(DPMReplayStage::kAgentDecision));
    if (generate_called_) {
      return absl::FailedPreconditionError(
          "WinnerReplay attempted more than one model call for one agent "
          "request.");
    }
    generate_called_ = true;
    if (request.stage != DPMReplayStage::kAgentDecision ||
        request.max_output_tokens != logical_request_->max_output_tokens ||
        request.request_contract_version != kDPMAgentReplayContractVersion) {
      return absl::InvalidArgumentError(
          "WinnerReplay agent invocation received another request contract.");
    }
    ABSL_ASSIGN_OR_RETURN(
        const DPMAgentExecutionRequest decoded_request,
        DecodeDPMAgentExecutionRequest(request.canonical_payload));
    if (!ExecutionRequestsEqual(decoded_request, *logical_request_)) {
      return absl::FailedPreconditionError(
          "WinnerReplay agent invocation received another logical request.");
    }
    ABSL_ASSIGN_OR_RETURN(
        DPMAgentGenerationOutcome generated,
        inference_runtime_->Generate(session_, *execution_request_));
    if (!generated.prepared_prefill_plan.has_value()) {
      return absl::DataLossError(
          "WinnerReplay live generation omitted its prepared prefill plan.");
    }
    ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(
        *generated.prepared_prefill_plan));
    ABSL_RETURN_IF_ERROR(ValidatePreparedPlanSourceBindings(
        *generated.prepared_prefill_plan,
        execution_request_->canonical_prefill_chunks));
    if (generated.prepared_prefill_plan->session_identity !=
            runtime_identity_ ||
        generated.prepared_prefill_plan->logical_agent_request_hash !=
            logical_request_->logical_agent_request_hash ||
        generated.prepared_prefill_plan->restore_checkpoint_id !=
            execution_request_->restore_checkpoint_id ||
        (execution_request_->restored_state_witness.has_value() &&
         generated.prepared_prefill_plan->start_state_witness_id !=
             std::optional<Hash256>(
                 execution_request_->restored_state_witness->witness_id))) {
      return absl::DataLossError(
          "WinnerReplay prepared prefill plan differs from its runtime, "
          "logical request, or restore start.");
    }
    const bool expects_restore_evidence =
        execution_request_->capsule_restore_operation_v3.has_value();
    if (generated.capsule_restore_evidence_v3.has_value() !=
        expects_restore_evidence) {
      return absl::DataLossError(
          "WinnerReplay live generation omitted or invented Coverage V2 "
          "restore evidence.");
    }
    if (generated.capsule_restore_evidence_v3.has_value()) {
      if (!execution_request_->restore_checkpoint_id.has_value() ||
          !execution_request_->restored_state_witness.has_value()) {
        return absl::DataLossError(
            "WinnerReplay returned restore evidence for a fresh generation.");
      }
      const DPMAgentCapsuleRestoreOperationV3& operation =
          *execution_request_->capsule_restore_operation_v3;
      const CapsuleRestoreEvidenceV3& restore_evidence =
          *generated.capsule_restore_evidence_v3;
      ABSL_RETURN_IF_ERROR(
          ValidateCapsuleRestoreEvidenceV3(restore_evidence));
      ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidenceV3ForSourceCapture(
          restore_evidence, operation.source_capture_evidence));
      if (restore_evidence.plan.authority != operation.current_authority ||
          restore_evidence.plan.target_state != operation.target_state ||
          restore_evidence.plan.checkpoint_id !=
              *execution_request_->restore_checkpoint_id ||
          restore_evidence.plan.maximum_output_tokens !=
              static_cast<uint32_t>(execution_request_->max_output_tokens) ||
          !(restore_evidence.plan.prefill.prepared_plan ==
            *generated.prepared_prefill_plan) ||
          restore_evidence.durable_to_transient_reauthentication !=
              operation.durable_to_transient_reauthentication ||
          restore_evidence.target_post_import !=
              *execution_request_->restored_state_witness) {
        return absl::DataLossError(
            "WinnerReplay restore evidence differs from its current authority, "
            "checkpoint, live target, logical request, or prepared work.");
      }
    }
    std::vector<int32_t> token_ids;
    token_ids.reserve(generated.decision_token_ids.size());
    for (int token_id : generated.decision_token_ids) {
      if (token_id < 0 ||
          static_cast<int64_t>(token_id) >
              std::numeric_limits<int32_t>::max()) {
        return absl::DataLossError(
            "WinnerReplay agent generated a non-int32 token ID.");
      }
      token_ids.push_back(static_cast<int32_t>(token_id));
    }
    ABSL_RETURN_IF_ERROR(ValidateDecisionFields(
        generated.decision_output, token_ids,
        static_cast<uint32_t>(execution_request_->max_output_tokens)));
    ABSL_ASSIGN_OR_RETURN(std::string token_bytes,
                          EncodeFreshWorkerTokenIds(token_ids));
    DPMAgentDecisionEnvelope envelope{
        .decision_output = std::move(generated.decision_output),
        .canonical_token_bytes = std::move(token_bytes),
    };
    ABSL_ASSIGN_OR_RETURN(std::string canonical_output,
                          EncodeDPMAgentDecisionEnvelope(envelope));
    ABSL_ASSIGN_OR_RETURN(
        const Hash256 evidence,
        ComputeWinnerEvidenceHash(runtime_identity_, request,
                                  canonical_output));
    if (IsZeroHash(evidence)) {
      return absl::InternalError(
          "WinnerReplay agent could not bind its execution evidence.");
    }
    generated_output_hash_ = Sha256(canonical_output);
    generated_evidence_hash_ = evidence;
    prepared_prefill_plan_ =
        std::move(*generated.prepared_prefill_plan);
    capsule_restore_evidence_v3_ =
        std::move(generated.capsule_restore_evidence_v3);
    generation_succeeded_ = true;
    return CanonicalWinnerGeneratedCandidate{
        .canonical_output = std::move(canonical_output),
        .execution_evidence_hash = evidence,
    };
  }

  bool generate_called() const { return generate_called_; }
  bool generation_succeeded() const { return generation_succeeded_; }
  const std::optional<DPMPreparedPrefillPlan>& prepared_prefill_plan() const {
    return prepared_prefill_plan_;
  }
  const std::optional<CapsuleRestoreEvidenceV3>&
  capsule_restore_evidence_v3() const {
    return capsule_restore_evidence_v3_;
  }
  bool GeneratedSelectedCandidate(
      const CanonicalWinnerReplayExecution& replay) const {
    return generation_succeeded_ &&
           replay.canonical_output_hash == generated_output_hash_ &&
           replay.execution_evidence_hash == generated_evidence_hash_;
  }

 private:
  DPMAgentRuntime* const inference_runtime_;
  Engine::Session* const session_;
  const DPMAgentGenerationRequest* const execution_request_;
  const DPMAgentExecutionRequest* const logical_request_;
  const SessionHandoffIdentity runtime_identity_;
  bool generate_called_ = false;
  bool generation_succeeded_ = false;
  Hash256 generated_output_hash_;
  Hash256 generated_evidence_hash_;
  std::optional<DPMPreparedPrefillPlan> prepared_prefill_plan_;
  std::optional<CapsuleRestoreEvidenceV3> capsule_restore_evidence_v3_;
};

DPMCanonicalReplayRequest MakeReplayRequest(std::string encoded,
                                            uint32_t max_output_tokens) {
  return DPMCanonicalReplayRequest{
      .stage = DPMReplayStage::kAgentDecision,
      .max_output_tokens = max_output_tokens,
      .request_contract_version = std::string(kDPMAgentReplayContractVersion),
      .canonical_payload = std::move(encoded),
  };
}

bool DurableCapsuleEvidenceEqual(
    const FreshWorkerDurableProducingCapsuleEvidence& left,
    const FreshWorkerDurableProducingCapsuleEvidence& right) {
  return left.session_identity == right.session_identity &&
         left.key_id == right.key_id &&
         left.envelope_size == right.envelope_size &&
         left.envelope_hash == right.envelope_hash &&
         left.output_evidence_hash == right.output_evidence_hash &&
         left.reauthentication_evidence ==
             right.reauthentication_evidence;
}

absl::StatusOr<DPMAgentReplayExecution> BuildExactAgentReplayExecution(
    const ExactRegenerationExecution& exact,
    const ExactLiteRtProfile& expected_profile,
    const Hash256& expected_request_hash,
    const Hash256& expected_logical_agent_request_hash,
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>&
        expected_executed_source_chunks,
    uint32_t max_output_tokens) {
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(expected_profile));
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(exact.derived_profile));
  if (exact.mode != DPMReplayMode::kExactRegeneration ||
      exact.derived_profile != expected_profile ||
      exact.canonical_request_hash != expected_request_hash ||
      IsZeroHash(exact.profile_admission_record_id)) {
    return absl::FailedPreconditionError(
        "Exact agent result came from another request or Engine profile.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateExactRegenerationRequestEvidence(exact.request_evidence));
  if (exact.request_evidence.stage != DPMReplayStage::kAgentDecision ||
      !exact.prepared_prefill_plan.has_value() ||
      !exact.request_evidence.agent_logical_request_hash.has_value() ||
      !exact.request_evidence.consensus_source_chunks_hash.has_value() ||
      !exact.request_evidence.consensus_resolved_token_plan_hash.has_value() ||
      !exact.request_evidence.consensus_shape_schedule_hash.has_value()) {
    return absl::DataLossError(
        "Exact agent result lacks its cold-run prepared-prefill consensus.");
  }
  const DPMPreparedPrefillPlan& prepared_plan =
      *exact.prepared_prefill_plan;
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(prepared_plan));
  ABSL_RETURN_IF_ERROR(ValidatePreparedPlanSourceBindings(
      prepared_plan, expected_executed_source_chunks));
  if (prepared_plan.session_identity != expected_profile.session_identity ||
      prepared_plan.logical_agent_request_hash !=
          expected_logical_agent_request_hash ||
      *exact.request_evidence.agent_logical_request_hash !=
          expected_logical_agent_request_hash ||
      prepared_plan.canonical_source_chunks_hash !=
          *exact.request_evidence.consensus_source_chunks_hash ||
      prepared_plan.resolved_token_plan_hash !=
          *exact.request_evidence.consensus_resolved_token_plan_hash ||
      prepared_plan.shape_schedule_hash !=
          *exact.request_evidence.consensus_shape_schedule_hash) {
    return absl::FailedPreconditionError(
        "Exact agent representative plan differs from its runtime, logical "
        "request, raw source chunks, or cold-run consensus.");
  }
  const Hash256 exact_output_evidence_hash =
      ComputeFreshWorkerOutputEvidenceHash(
          exact.canonical_output, exact.token_bytes, exact.logit_frames);
  if (exact.request_evidence.exact_profile_id !=
          exact.derived_profile.profile_id ||
      exact.request_evidence.profile_admission_record_id !=
          exact.profile_admission_record_id ||
      exact.request_evidence.session_identity !=
          exact.derived_profile.session_identity ||
      exact.request_evidence.canonical_request_hash !=
          exact.canonical_request_hash ||
      exact.request_evidence.consensus_output_evidence_hash !=
          exact_output_evidence_hash) {
    return absl::DataLossError(
        "Exact agent request evidence and returned output disagree.");
  }
  const ExactRegenerationRunEvidence& run_zero =
      exact.request_evidence.runs.front();
  if (!run_zero.prepared_prefill_plan.has_value() ||
      !(*run_zero.prepared_prefill_plan == prepared_plan) ||
      run_zero.durable_producing_capsule_evidence.has_value() !=
      exact.durable_producing_capsule_evidence.has_value()) {
    return absl::DataLossError(
        "Exact agent result changed its run-zero prepared plan or durable "
        "capsule evidence.");
  }
  if (exact.durable_producing_capsule_evidence.has_value() &&
      !DurableCapsuleEvidenceEqual(
          *run_zero.durable_producing_capsule_evidence,
          *exact.durable_producing_capsule_evidence)) {
    return absl::DataLossError(
        "Exact agent run-zero and returned durable capsule evidence "
        "disagree.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const DPMAgentDecisionEnvelope envelope,
      DecodeDPMAgentDecisionEnvelope(exact.canonical_output));
  if (envelope.canonical_token_bytes != exact.token_bytes) {
    return absl::DataLossError(
        "Exact agent decision envelope and worker token evidence disagree.");
  }
  ABSL_ASSIGN_OR_RETURN(const std::vector<int32_t> exact_ids,
                        DecodeFreshWorkerTokenIds(exact.token_bytes));
  if (IsZeroHash(exact_output_evidence_hash)) {
    return absl::InternalError(
        "Exact agent could not bind its output evidence.");
  }
  std::vector<int> decision_ids(exact_ids.begin(), exact_ids.end());
  DPMAgentReplayExecution execution{
      .mode = DPMReplayMode::kExactRegeneration,
      .replay_request_hash = exact.canonical_request_hash,
      .execution_evidence_hash = exact.request_evidence.evidence_id,
      .exact_profile_id = exact.derived_profile.profile_id,
      .exact_profile_admission_record_id =
          exact.profile_admission_record_id,
      .decision_output = envelope.decision_output,
      .decision_token_ids = std::move(decision_ids),
      .exact_token_bytes = exact.token_bytes,
      .exact_logit_frames = exact.logit_frames,
      .exact_output_evidence_hash = exact_output_evidence_hash,
      .reused_canonical_winner = false,
      .producing_session_matches_output = false,
      .prepared_prefill_plan = prepared_plan,
  };
  ABSL_RETURN_IF_ERROR(
      ValidateDPMAgentReplayExecution(execution, max_output_tokens));
  return execution;
}

}  // namespace

absl::Status ValidateDPMAgentExecutionRequest(
    const DPMAgentExecutionRequest& request) {
  if (request.format_version != kDPMAgentExecutionRequestFormatVersion ||
      IsZeroHash(request.logical_agent_request_hash) ||
      IsZeroHash(request.correction_digest) || request.max_output_tokens == 0 ||
      request.max_output_tokens > kMaximumDPMGenerationTokens ||
      request.full_canonical_prefill_chunks.empty() ||
      request.full_canonical_prefill_chunks.size() >
          kMaximumFreshWorkerTokenIds) {
    return absl::InvalidArgumentError(
        "Canonical DPM agent execution request is incomplete or oversized.");
  }
  const uint64_t outer_overhead =
      kDPMCanonicalReplayRequestFramingBytes +
      kDPMAgentReplayContractVersion.size();
  if (outer_overhead > kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::InternalError(
        "DPM agent replay framing exceeds its worker protocol.");
  }
  const uint64_t maximum_inner_size =
      kMaximumFreshWorkerRequestPayloadBytes - outer_overhead;
  uint64_t encoded_size = kAgentExecutionFixedBytes;
  uint64_t total_token_ids = 0;
  for (const auto& chunk : request.full_canonical_prefill_chunks) {
    ABSL_RETURN_IF_ERROR(ValidateChunk(chunk));
    uint64_t payload_size = 0;
    if (chunk.encoding ==
        DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text) {
      payload_size = chunk.text.size();
    } else {
      if (chunk.token_ids.size() >
          kMaximumFreshWorkerTokenIds - total_token_ids) {
        return absl::ResourceExhaustedError(
            "Canonical DPM agent execution exceeds the token-input limit.");
      }
      total_token_ids += chunk.token_ids.size();
      payload_size = kCanonicalTokenFramingBytes +
                     uint64_t{4} * chunk.token_ids.size();
    }
    if (encoded_size > maximum_inner_size ||
        kAgentChunkFramingBytes > maximum_inner_size - encoded_size ||
        payload_size > maximum_inner_size - encoded_size -
                           kAgentChunkFramingBytes) {
      return absl::ResourceExhaustedError(
          "Canonical DPM agent request cannot fit one worker envelope.");
    }
    encoded_size += kAgentChunkFramingBytes + payload_size;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeDPMAgentExecutionRequest(
    const DPMAgentExecutionRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentExecutionRequest(request));
  std::string encoded;
  encoded.reserve(kAgentExecutionMagic.size() + 4 + 32 + 32 + 4 + 4);
  encoded.append(kAgentExecutionMagic.data(), kAgentExecutionMagic.size());
  AppendU32(request.format_version, &encoded);
  AppendHash(request.logical_agent_request_hash, &encoded);
  AppendHash(request.correction_digest, &encoded);
  AppendU32(request.max_output_tokens, &encoded);
  AppendU32(static_cast<uint32_t>(
      request.full_canonical_prefill_chunks.size()), &encoded);
  const uint64_t outer_overhead =
      kDPMCanonicalReplayRequestFramingBytes +
      kDPMAgentReplayContractVersion.size();
  const uint64_t maximum_inner_size =
      kMaximumFreshWorkerRequestPayloadBytes - outer_overhead;
  for (const auto& chunk : request.full_canonical_prefill_chunks) {
    std::string token_payload;
    absl::string_view payload;
    if (chunk.encoding ==
        DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text) {
      payload = chunk.text;
    } else {
      std::vector<int32_t> token_ids;
      token_ids.reserve(chunk.token_ids.size());
      for (int token_id : chunk.token_ids) {
        token_ids.push_back(static_cast<int32_t>(token_id));
      }
      ABSL_ASSIGN_OR_RETURN(token_payload,
                            EncodeFreshWorkerTokenIds(token_ids));
      payload = token_payload;
    }
    if (encoded.size() > maximum_inner_size ||
        kAgentChunkFramingBytes > maximum_inner_size - encoded.size() ||
        payload.size() > maximum_inner_size - encoded.size() -
                             kAgentChunkFramingBytes) {
      return absl::ResourceExhaustedError(
          "Canonical DPM agent execution request exceeds the worker limit.");
    }
    encoded.push_back(static_cast<char>(chunk.encoding));
    AppendU64(payload.size(), &encoded);
    encoded.append(payload.data(), payload.size());
  }
  if (encoded.size() > maximum_inner_size) {
    return absl::ResourceExhaustedError(
        "Canonical DPM agent request cannot fit its replay envelope.");
  }
  return encoded;
}

absl::StatusOr<DPMAgentExecutionRequest> DecodeDPMAgentExecutionRequest(
    absl::string_view bytes) {
  const uint64_t outer_overhead =
      kDPMCanonicalReplayRequestFramingBytes +
      kDPMAgentReplayContractVersion.size();
  if (outer_overhead > kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::InternalError(
        "DPM agent replay framing exceeds its worker protocol.");
  }
  const uint64_t maximum_inner_size =
      kMaximumFreshWorkerRequestPayloadBytes - outer_overhead;
  if (bytes.size() < kAgentExecutionFixedBytes ||
      bytes.size() > maximum_inner_size ||
      std::memcmp(bytes.data(), kAgentExecutionMagic.data(),
                  kAgentExecutionMagic.size()) != 0) {
    return absl::DataLossError(
        "Canonical DPM agent execution framing is invalid.");
  }
  Reader reader(bytes.substr(kAgentExecutionMagic.size()));
  DPMAgentExecutionRequest request;
  ABSL_ASSIGN_OR_RETURN(request.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(request.logical_agent_request_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.correction_digest, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.max_output_tokens, reader.ReadU32());
  uint32_t chunk_count;
  ABSL_ASSIGN_OR_RETURN(chunk_count, reader.ReadU32());
  if (chunk_count == 0 || chunk_count > kMaximumFreshWorkerTokenIds ||
      chunk_count >
          reader.remaining() / (kAgentChunkFramingBytes + uint64_t{1})) {
    return absl::DataLossError(
        "Canonical DPM agent execution has an invalid chunk count.");
  }
  request.full_canonical_prefill_chunks.reserve(chunk_count);
  uint64_t total_token_ids = 0;
  for (uint32_t index = 0; index < chunk_count; ++index) {
    uint8_t encoding;
    ABSL_ASSIGN_OR_RETURN(encoding, reader.ReadU8());
    uint64_t payload_size;
    ABSL_ASSIGN_OR_RETURN(payload_size, reader.ReadU64());
    absl::string_view payload;
    ABSL_ASSIGN_OR_RETURN(payload, reader.ReadBytes(payload_size));
    DPMAgentGenerationRequest::PrefillChunk chunk;
    chunk.encoding = static_cast<
        DPMAgentGenerationRequest::PrefillChunk::Encoding>(encoding);
    switch (chunk.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text:
        chunk.text.assign(payload.data(), payload.size());
        break;
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds: {
        ABSL_ASSIGN_OR_RETURN(const std::vector<int32_t> token_ids,
                              DecodeFreshWorkerTokenIds(payload));
        if (token_ids.size() >
            kMaximumFreshWorkerTokenIds - total_token_ids) {
          return absl::ResourceExhaustedError(
              "Canonical DPM agent execution exceeds the token-input limit.");
        }
        total_token_ids += token_ids.size();
        chunk.token_ids.reserve(token_ids.size());
        for (int32_t token_id : token_ids) chunk.token_ids.push_back(token_id);
        break;
      }
      default:
        return absl::DataLossError(
            "Canonical DPM agent execution has an unknown chunk encoding.");
    }
    request.full_canonical_prefill_chunks.push_back(std::move(chunk));
  }
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "Canonical DPM agent execution has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentExecutionRequest(request));
  ABSL_ASSIGN_OR_RETURN(const std::string canonical,
                        EncodeDPMAgentExecutionRequest(request));
  if (canonical != bytes) {
    return absl::DataLossError(
        "Canonical DPM agent execution is not canonically encoded.");
  }
  return request;
}

absl::Status ValidateDPMAgentDeltaExecutionRequest(
    const DPMAgentDeltaExecutionRequest& request) {
  if (request.format_version !=
          kDPMAgentDeltaExecutionRequestFormatVersion ||
      IsZeroHash(request.logical_agent_request_hash) ||
      IsZeroHash(request.correction_digest) ||
      IsZeroHash(request.restore_checkpoint_id) ||
      request.restored_response_event_index == 0 ||
      IsZeroHash(request.restored_agent_transcript_hash) ||
      request.max_output_tokens == 0 ||
      request.max_output_tokens > kMaximumDPMGenerationTokens ||
      request.canonical_delta_prefill_chunks.empty() ||
      request.canonical_delta_prefill_chunks.size() >
          kMaximumFreshWorkerTokenIds) {
    return absl::InvalidArgumentError(
        "Canonical DPM agent delta request is incomplete or oversized.");
  }

  uint64_t encoded_size = kAgentDeltaExecutionFixedBytes;
  uint64_t total_token_ids = 0;
  for (const auto& chunk : request.canonical_delta_prefill_chunks) {
    ABSL_RETURN_IF_ERROR(ValidateChunk(chunk));
    uint64_t payload_size = 0;
    if (chunk.encoding ==
        DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text) {
      payload_size = chunk.text.size();
    } else {
      if (chunk.token_ids.size() >
          kMaximumFreshWorkerTokenIds - total_token_ids) {
        return absl::ResourceExhaustedError(
            "Canonical DPM agent delta exceeds the token-input limit.");
      }
      total_token_ids += chunk.token_ids.size();
      payload_size = kCanonicalTokenFramingBytes +
                     uint64_t{4} * chunk.token_ids.size();
    }
    if (encoded_size > kMaximumFreshWorkerRequestPayloadBytes ||
        kAgentChunkFramingBytes >
            kMaximumFreshWorkerRequestPayloadBytes - encoded_size ||
        payload_size > kMaximumFreshWorkerRequestPayloadBytes -
                           encoded_size - kAgentChunkFramingBytes) {
      return absl::ResourceExhaustedError(
          "Canonical DPM agent delta cannot fit its execution-plan "
          "payload.");
    }
    encoded_size += kAgentChunkFramingBytes + payload_size;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeDPMAgentDeltaExecutionRequest(
    const DPMAgentDeltaExecutionRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentDeltaExecutionRequest(request));
  std::string encoded;
  encoded.reserve(static_cast<size_t>(kAgentDeltaExecutionFixedBytes));
  encoded.append(kAgentDeltaExecutionMagic.data(),
                 kAgentDeltaExecutionMagic.size());
  AppendU32(request.format_version, &encoded);
  AppendHash(request.logical_agent_request_hash, &encoded);
  AppendHash(request.correction_digest, &encoded);
  AppendHash(request.restore_checkpoint_id, &encoded);
  AppendU64(request.restored_response_event_index, &encoded);
  AppendHash(request.restored_agent_transcript_hash, &encoded);
  AppendU32(request.max_output_tokens, &encoded);
  AppendU32(
      static_cast<uint32_t>(request.canonical_delta_prefill_chunks.size()),
      &encoded);

  for (const auto& chunk : request.canonical_delta_prefill_chunks) {
    std::string token_payload;
    absl::string_view payload;
    if (chunk.encoding ==
        DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text) {
      payload = chunk.text;
    } else {
      std::vector<int32_t> token_ids;
      token_ids.reserve(chunk.token_ids.size());
      for (int token_id : chunk.token_ids) {
        token_ids.push_back(static_cast<int32_t>(token_id));
      }
      ABSL_ASSIGN_OR_RETURN(token_payload,
                            EncodeFreshWorkerTokenIds(token_ids));
      payload = token_payload;
    }
    if (encoded.size() > kMaximumFreshWorkerRequestPayloadBytes ||
        kAgentChunkFramingBytes >
            kMaximumFreshWorkerRequestPayloadBytes - encoded.size() ||
        payload.size() > kMaximumFreshWorkerRequestPayloadBytes -
                             encoded.size() - kAgentChunkFramingBytes) {
      return absl::ResourceExhaustedError(
          "Canonical DPM agent delta exceeds its execution-plan limit.");
    }
    encoded.push_back(static_cast<char>(chunk.encoding));
    AppendU64(payload.size(), &encoded);
    encoded.append(payload.data(), payload.size());
  }
  return encoded;
}

absl::StatusOr<DPMAgentDeltaExecutionRequest>
DecodeDPMAgentDeltaExecutionRequest(absl::string_view bytes) {
  if (bytes.size() < kAgentDeltaExecutionFixedBytes ||
      bytes.size() > kMaximumFreshWorkerRequestPayloadBytes ||
      std::memcmp(bytes.data(), kAgentDeltaExecutionMagic.data(),
                  kAgentDeltaExecutionMagic.size()) != 0) {
    return absl::DataLossError(
        "Canonical DPM agent delta framing is invalid.");
  }
  Reader reader(bytes.substr(kAgentDeltaExecutionMagic.size()));
  DPMAgentDeltaExecutionRequest request;
  ABSL_ASSIGN_OR_RETURN(request.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(request.logical_agent_request_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.correction_digest, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.restore_checkpoint_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.restored_response_event_index,
                        reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(request.restored_agent_transcript_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.max_output_tokens, reader.ReadU32());
  uint32_t chunk_count;
  ABSL_ASSIGN_OR_RETURN(chunk_count, reader.ReadU32());
  if (chunk_count == 0 || chunk_count > kMaximumFreshWorkerTokenIds ||
      chunk_count >
          reader.remaining() / (kAgentChunkFramingBytes + uint64_t{1})) {
    return absl::DataLossError(
        "Canonical DPM agent delta has an invalid chunk count.");
  }
  request.canonical_delta_prefill_chunks.reserve(chunk_count);
  uint64_t total_token_ids = 0;
  for (uint32_t index = 0; index < chunk_count; ++index) {
    uint8_t encoding;
    ABSL_ASSIGN_OR_RETURN(encoding, reader.ReadU8());
    uint64_t payload_size;
    ABSL_ASSIGN_OR_RETURN(payload_size, reader.ReadU64());
    absl::string_view payload;
    ABSL_ASSIGN_OR_RETURN(payload, reader.ReadBytes(payload_size));
    DPMAgentGenerationRequest::PrefillChunk chunk;
    chunk.encoding = static_cast<
        DPMAgentGenerationRequest::PrefillChunk::Encoding>(encoding);
    switch (chunk.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text:
        chunk.text.assign(payload.data(), payload.size());
        break;
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds: {
        ABSL_ASSIGN_OR_RETURN(const std::vector<int32_t> token_ids,
                              DecodeFreshWorkerTokenIds(payload));
        if (token_ids.size() >
            kMaximumFreshWorkerTokenIds - total_token_ids) {
          return absl::ResourceExhaustedError(
              "Canonical DPM agent delta exceeds the token-input limit.");
        }
        total_token_ids += token_ids.size();
        chunk.token_ids.reserve(token_ids.size());
        for (int32_t token_id : token_ids) chunk.token_ids.push_back(token_id);
        break;
      }
      default:
        return absl::DataLossError(
            "Canonical DPM agent delta has an unknown chunk encoding.");
    }
    request.canonical_delta_prefill_chunks.push_back(std::move(chunk));
  }
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "Canonical DPM agent delta has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentDeltaExecutionRequest(request));
  ABSL_ASSIGN_OR_RETURN(const std::string canonical,
                        EncodeDPMAgentDeltaExecutionRequest(request));
  if (canonical != bytes) {
    return absl::DataLossError(
        "Canonical DPM agent delta is not canonically encoded.");
  }
  return request;
}

absl::Status ValidateDPMAgentDeltaExecutionBinding(
    const DPMAgentExecutionRequest& logical_request,
    const FreshWorkerExecutionPlan& execution_plan,
    const DPMAgentDeltaExecutionRequest& delta_request) {
  ABSL_RETURN_IF_ERROR(
      ValidateDPMAgentExecutionRequest(logical_request));
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerExecutionPlan(execution_plan));
  ABSL_RETURN_IF_ERROR(
      ValidateDPMAgentDeltaExecutionRequest(delta_request));
  if (execution_plan.prefill_mode !=
          FreshWorkerPrefillMode::kOwnPositionCapsuleDelta ||
      !execution_plan.restore_checkpoint_id.has_value() ||
      delta_request.logical_agent_request_hash !=
          logical_request.logical_agent_request_hash ||
      delta_request.correction_digest != logical_request.correction_digest ||
      delta_request.max_output_tokens != logical_request.max_output_tokens ||
      delta_request.restore_checkpoint_id !=
          *execution_plan.restore_checkpoint_id) {
    return absl::FailedPreconditionError(
        "DPM agent delta does not match its complete logical request and "
        "authenticated restore plan.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const std::string canonical_delta,
      EncodeDPMAgentDeltaExecutionRequest(delta_request));
  if (canonical_delta != execution_plan.canonical_execution_payload) {
    return absl::DataLossError(
        "DPM agent delta bytes differ from the authenticated execution "
        "plan.");
  }
  if (delta_request.canonical_delta_prefill_chunks.size() >
      logical_request.full_canonical_prefill_chunks.size()) {
    return absl::FailedPreconditionError(
        "DPM agent delta contains more chunks than the logical request.");
  }
  const size_t suffix_start =
      logical_request.full_canonical_prefill_chunks.size() -
      delta_request.canonical_delta_prefill_chunks.size();
  for (size_t index = 0;
       index < delta_request.canonical_delta_prefill_chunks.size();
       ++index) {
    const auto& full =
        logical_request.full_canonical_prefill_chunks[suffix_start + index];
    const auto& delta =
        delta_request.canonical_delta_prefill_chunks[index];
    if (full.encoding != delta.encoding || full.text != delta.text ||
        full.token_ids != delta.token_ids) {
      return absl::FailedPreconditionError(
          "DPM agent delta is not an exact chunk-preserving suffix of the "
          "complete logical request.");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateDPMAgentDecisionEnvelope(
    const DPMAgentDecisionEnvelope& envelope) {
  if (envelope.format_version != kDPMAgentDecisionEnvelopeFormatVersion ||
      envelope.decision_output.size() > kMaximumDPMEventPayloadBytes ||
      !IsValidUtf8(envelope.decision_output) ||
      envelope.canonical_token_bytes.empty() ||
      envelope.canonical_token_bytes.size() > kMaximumFreshWorkerTokenBytes) {
    return absl::InvalidArgumentError(
        "Canonical DPM agent decision envelope is invalid or oversized.");
  }
  ABSL_ASSIGN_OR_RETURN(const std::vector<int32_t> token_ids,
                        DecodeFreshWorkerTokenIds(
                            envelope.canonical_token_bytes));
  if (token_ids.empty()) {
    return absl::InvalidArgumentError(
        "Canonical DPM agent decision has no exact token IDs.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeDPMAgentDecisionEnvelope(
    const DPMAgentDecisionEnvelope& envelope) {
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentDecisionEnvelope(envelope));
  const uint64_t encoded_size =
      kAgentDecisionMagic.size() + 4 + 8 + envelope.decision_output.size() +
      8 + envelope.canonical_token_bytes.size();
  if (encoded_size > kMaximumFreshWorkerCanonicalOutputBytes ||
      encoded_size > (std::numeric_limits<size_t>::max)()) {
    return absl::ResourceExhaustedError(
        "Canonical DPM agent decision exceeds the shared replay output "
        "limit.");
  }
  std::string encoded;
  encoded.reserve(static_cast<size_t>(encoded_size));
  encoded.append(kAgentDecisionMagic.data(), kAgentDecisionMagic.size());
  AppendU32(envelope.format_version, &encoded);
  AppendU64(envelope.decision_output.size(), &encoded);
  encoded.append(envelope.decision_output);
  AppendU64(envelope.canonical_token_bytes.size(), &encoded);
  encoded.append(envelope.canonical_token_bytes);
  return encoded;
}

absl::StatusOr<DPMAgentDecisionEnvelope> DecodeDPMAgentDecisionEnvelope(
    absl::string_view bytes) {
  constexpr size_t kFixedSize = 8 + 4 + 8 + 8;
  if (bytes.size() < kFixedSize ||
      bytes.size() > kMaximumFreshWorkerCanonicalOutputBytes ||
      std::memcmp(bytes.data(), kAgentDecisionMagic.data(),
                  kAgentDecisionMagic.size()) != 0) {
    return absl::DataLossError(
        "Canonical DPM agent decision framing is invalid.");
  }
  Reader reader(bytes.substr(kAgentDecisionMagic.size()));
  DPMAgentDecisionEnvelope envelope;
  ABSL_ASSIGN_OR_RETURN(envelope.format_version, reader.ReadU32());
  uint64_t text_size;
  ABSL_ASSIGN_OR_RETURN(text_size, reader.ReadU64());
  if (text_size > kMaximumDPMEventPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Canonical DPM agent decision text is oversized.");
  }
  absl::string_view text;
  ABSL_ASSIGN_OR_RETURN(text, reader.ReadBytes(text_size));
  envelope.decision_output.assign(text.data(), text.size());
  uint64_t token_size;
  ABSL_ASSIGN_OR_RETURN(token_size, reader.ReadU64());
  if (token_size > kMaximumFreshWorkerTokenBytes) {
    return absl::ResourceExhaustedError(
        "Canonical DPM agent decision token evidence is oversized.");
  }
  absl::string_view token_bytes;
  ABSL_ASSIGN_OR_RETURN(token_bytes, reader.ReadBytes(token_size));
  envelope.canonical_token_bytes.assign(token_bytes.data(),
                                        token_bytes.size());
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "Canonical DPM agent decision has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentDecisionEnvelope(envelope));
  ABSL_ASSIGN_OR_RETURN(const std::string canonical,
                        EncodeDPMAgentDecisionEnvelope(envelope));
  if (canonical != bytes) {
    return absl::DataLossError(
        "Canonical DPM agent decision is not canonically encoded.");
  }
  return envelope;
}

absl::Status ValidateDPMAgentReplayExecution(
    const DPMAgentReplayExecution& execution, uint32_t max_output_tokens) {
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(execution.mode));
  if (max_output_tokens == 0 ||
      max_output_tokens > kMaximumDPMGenerationTokens ||
      IsZeroHash(execution.replay_request_hash) ||
      IsZeroHash(execution.execution_evidence_hash)) {
    return absl::InvalidArgumentError(
        "DPM agent replay execution has incomplete common evidence.");
  }
  std::vector<int32_t> token_ids;
  token_ids.reserve(execution.decision_token_ids.size());
  for (int token_id : execution.decision_token_ids) {
    if (token_id < 0 ||
        static_cast<int64_t>(token_id) >
            std::numeric_limits<int32_t>::max()) {
      return absl::DataLossError(
          "DPM agent replay returned a non-int32 token ID.");
    }
    token_ids.push_back(static_cast<int32_t>(token_id));
  }
  ABSL_RETURN_IF_ERROR(ValidateDecisionFields(
      execution.decision_output, token_ids, max_output_tokens));
  if (execution.prepared_prefill_plan.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(
        *execution.prepared_prefill_plan));
  }
  if (execution.capsule_restore_evidence_v3.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidenceV3(
        *execution.capsule_restore_evidence_v3));
  }
  switch (execution.mode) {
    case DPMReplayMode::kCanonicalWinnerReplay: {
      const bool is_published_live_winner =
          !execution.reused_canonical_winner &&
          !execution.rematerialized_canonical_winner &&
          execution.producing_session_matches_output;
      const bool is_catalog_only_winner =
          execution.reused_canonical_winner &&
          !execution.rematerialized_canonical_winner &&
          !execution.producing_session_matches_output;
      const bool is_rematerialized_catalog_winner =
          execution.reused_canonical_winner &&
          execution.rematerialized_canonical_winner &&
          execution.producing_session_matches_output;
      if (execution.exact_profile_id.has_value() ||
          execution.exact_profile_admission_record_id.has_value() ||
          execution.exact_output_evidence_hash.has_value() ||
          !execution.exact_token_bytes.empty() ||
          !execution.exact_logit_frames.empty() ||
          (!is_published_live_winner && !is_catalog_only_winner &&
           !is_rematerialized_catalog_winner) ||
          execution.prepared_prefill_plan.has_value() !=
              execution.producing_session_matches_output ||
          (execution.capsule_restore_evidence_v3.has_value() &&
           (!execution.producing_session_matches_output ||
            !execution.prepared_prefill_plan.has_value() ||
            !(execution.capsule_restore_evidence_v3->plan.prefill
                  .prepared_plan == *execution.prepared_prefill_plan)))) {
        return absl::DataLossError(
            "WinnerReplay agent execution carries exact evidence or invalid "
            "producing-session provenance.");
      }
      break;
    }
    case DPMReplayMode::kExactRegeneration: {
      if (!execution.exact_profile_id.has_value() ||
          IsZeroHash(*execution.exact_profile_id) ||
          !execution.exact_profile_admission_record_id.has_value() ||
          IsZeroHash(*execution.exact_profile_admission_record_id) ||
          !execution.exact_output_evidence_hash.has_value() ||
          IsZeroHash(*execution.exact_output_evidence_hash) ||
          execution.exact_token_bytes.empty() ||
          execution.exact_logit_frames.empty() ||
          !execution.prepared_prefill_plan.has_value() ||
          execution.capsule_restore_evidence_v3.has_value() ||
          execution.reused_canonical_winner ||
          execution.rematerialized_canonical_winner ||
          execution.producing_session_matches_output) {
        return absl::DataLossError(
            "Exact agent execution is missing non-catalog evidence or "
            "claims a parent producing session.");
      }
      ABSL_ASSIGN_OR_RETURN(const std::vector<int32_t> exact_ids,
                            DecodeFreshWorkerTokenIds(
                                execution.exact_token_bytes));
      if (exact_ids != token_ids ||
          exact_ids.size() != execution.exact_logit_frames.size()) {
        return absl::DataLossError(
            "Exact agent token bytes, returned IDs, and logits frames "
            "disagree.");
      }
      DPMAgentDecisionEnvelope canonical_envelope{
          .decision_output = execution.decision_output,
          .canonical_token_bytes = execution.exact_token_bytes,
      };
      ABSL_ASSIGN_OR_RETURN(
          std::string canonical_output,
          EncodeDPMAgentDecisionEnvelope(canonical_envelope));
      FreshWorkerExecutionOutput exact_output{
          .canonical_output = canonical_output,
          .token_bytes = execution.exact_token_bytes,
          .logit_frames = execution.exact_logit_frames,
      };
      ABSL_RETURN_IF_ERROR(ValidateFreshWorkerExecutionOutput(exact_output));
      if (ComputeFreshWorkerOutputEvidenceHash(
              canonical_output, execution.exact_token_bytes,
              execution.exact_logit_frames) !=
          *execution.exact_output_evidence_hash) {
        return absl::DataLossError(
            "Exact agent output evidence digest does not bind the returned "
            "decision, tokens, and logits.");
      }
      break;
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateExactRegenerationDPMAgentPhysicalExecution(
    const ExactRegenerationDPMAgentPhysicalExecution& execution,
    uint32_t max_output_tokens) {
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentReplayExecution(
      execution.replay_execution, max_output_tokens));
  ABSL_RETURN_IF_ERROR(
      ValidateExactRegenerationRequestEvidence(execution.request_evidence));
  if (execution.replay_execution.mode !=
          DPMReplayMode::kExactRegeneration ||
      execution.request_evidence.stage !=
          DPMReplayStage::kAgentDecision ||
      !execution.replay_execution.exact_profile_id.has_value() ||
      !execution.replay_execution.exact_profile_admission_record_id
           .has_value() ||
      !execution.replay_execution.exact_output_evidence_hash.has_value() ||
      execution.replay_execution.execution_evidence_hash !=
          execution.request_evidence.evidence_id ||
      execution.replay_execution.replay_request_hash !=
          execution.request_evidence.canonical_request_hash ||
      *execution.replay_execution.exact_profile_id !=
          execution.request_evidence.exact_profile_id ||
      *execution.replay_execution.exact_profile_admission_record_id !=
          execution.request_evidence.profile_admission_record_id ||
      *execution.replay_execution.exact_output_evidence_hash !=
          execution.request_evidence.consensus_output_evidence_hash ||
      execution.physical_execution_plan_hash !=
          execution.request_evidence.physical_execution_plan_hash ||
      execution.prefill_mode != execution.request_evidence.prefill_mode ||
      execution.restored_checkpoint_id !=
          execution.request_evidence.restored_checkpoint_id ||
      execution.capture_run_policy !=
          execution.request_evidence.capture_run_policy) {
    return absl::DataLossError(
        "Exact agent physical execution disagrees with its request "
        "evidence.");
  }
  const ExactRegenerationRunEvidence& run_zero =
      execution.request_evidence.runs.front();
  if (!execution.replay_execution.prepared_prefill_plan.has_value() ||
      !run_zero.prepared_prefill_plan.has_value() ||
      !(*execution.replay_execution.prepared_prefill_plan ==
        *run_zero.prepared_prefill_plan) ||
      !execution.request_evidence.consensus_source_chunks_hash.has_value() ||
      !execution.request_evidence.consensus_resolved_token_plan_hash
           .has_value() ||
      !execution.request_evidence.consensus_shape_schedule_hash.has_value() ||
      execution.replay_execution.prepared_prefill_plan
              ->canonical_source_chunks_hash !=
          *execution.request_evidence.consensus_source_chunks_hash ||
      execution.replay_execution.prepared_prefill_plan
              ->resolved_token_plan_hash !=
          *execution.request_evidence.consensus_resolved_token_plan_hash ||
      execution.replay_execution.prepared_prefill_plan->shape_schedule_hash !=
          *execution.request_evidence.consensus_shape_schedule_hash) {
    return absl::DataLossError(
        "Exact agent physical execution changed its agreed prepared-prefill "
        "plan.");
  }
  if (execution.run_zero_restore_reauthentication_evidence.has_value() !=
          run_zero.restore_reauthentication_evidence.has_value() ||
      (execution.run_zero_restore_reauthentication_evidence.has_value() &&
       *execution.run_zero_restore_reauthentication_evidence !=
           *run_zero.restore_reauthentication_evidence)) {
    return absl::DataLossError(
        "Exact agent physical execution changed run-zero restore "
          "reauthentication provenance.");
  }
  if (execution.run_zero_restored_state_witness.has_value() !=
          run_zero.restored_state_witness.has_value() ||
      (execution.run_zero_restored_state_witness.has_value() &&
       *execution.run_zero_restored_state_witness !=
           *run_zero.restored_state_witness)) {
    return absl::DataLossError(
        "Exact agent physical execution changed run-zero's independently "
        "recomputed restored-state witness.");
  }
  const bool restored =
      execution.prefill_mode ==
      FreshWorkerPrefillMode::kOwnPositionCapsuleDelta;
  if (execution.run_zero_restore_reauthentication_evidence.has_value() !=
          restored ||
      execution.run_zero_restored_state_witness.has_value() != restored) {
    return absl::DataLossError(
        "Exact agent physical execution restore provenance disagrees with "
        "its prefill mode.");
  }
  if (execution.run_zero_transient_producing_capsule_evidence.has_value() !=
          run_zero.transient_producing_capsule_evidence.has_value() ||
      (execution.run_zero_transient_producing_capsule_evidence.has_value() &&
       !(*execution.run_zero_transient_producing_capsule_evidence ==
         *run_zero.transient_producing_capsule_evidence))) {
    return absl::DataLossError(
        "Exact agent physical execution changed run-zero transient capsule "
        "evidence.");
  }
  if (execution.durable_producing_capsule_evidence.has_value() !=
      run_zero.durable_producing_capsule_evidence.has_value()) {
    return absl::DataLossError(
        "Exact agent physical execution omitted or invented durable capsule "
        "evidence.");
  }
  if (execution.durable_producing_capsule_evidence.has_value() &&
      !DurableCapsuleEvidenceEqual(
          *execution.durable_producing_capsule_evidence,
          *run_zero.durable_producing_capsule_evidence)) {
    return absl::DataLossError(
        "Exact agent physical execution changed durable capsule evidence.");
  }
  const bool captured =
      execution.capture_run_policy ==
      ExactRegenerationCaptureRunPolicy::kRunZeroOnly;
  if (execution.run_zero_transient_producing_capsule_evidence.has_value() !=
          captured ||
      execution.durable_producing_capsule_evidence.has_value() != captured) {
    return absl::DataLossError(
        "Exact agent physical execution violates run-zero-only capture.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<CanonicalWinnerDPMAgentRuntime>>
CanonicalWinnerDPMAgentRuntime::Create(
    DPMAgentRuntime* inference_runtime,
    CanonicalWinnerReplayCatalog* catalog) {
  if (inference_runtime == nullptr || catalog == nullptr) {
    return absl::InvalidArgumentError(
        "WinnerReplay agent requires an inference runtime and catalog.");
  }
  const SessionHandoffIdentity runtime_identity =
      inference_runtime->GetSessionHandoffIdentity();
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(runtime_identity));
  auto runtime = std::unique_ptr<CanonicalWinnerDPMAgentRuntime>(
      new CanonicalWinnerDPMAgentRuntime(inference_runtime, catalog,
                                         runtime_identity));
  ABSL_RETURN_IF_ERROR(runtime->ValidateSupport());
  return runtime;
}

absl::StatusOr<std::optional<Hash256>>
CanonicalWinnerDPMAgentRuntime::GetExactProfileId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  return std::optional<Hash256>();
}

absl::StatusOr<std::optional<Hash256>>
CanonicalWinnerDPMAgentRuntime::GetExactProfileAdmissionRecordId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  return std::optional<Hash256>();
}

absl::StatusOr<std::optional<Hash256>>
CanonicalWinnerDPMAgentRuntime::GetCapsuleRestoreAdmissionRecordId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(
      std::optional<Hash256> record_id,
      inference_runtime_->GetCapsuleRestoreAdmissionRecordId());
  if (record_id.has_value() && IsZeroHash(*record_id)) {
    return absl::DataLossError(
        "WinnerReplay inference runtime returned a zero CapsuleRestore "
        "admission record ID.");
  }
  return record_id;
}

absl::StatusOr<std::optional<Hash256>>
CanonicalWinnerDPMAgentRuntime::GetSessionHandoffCapabilityId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(
      std::optional<Hash256> capability_id,
      inference_runtime_->GetSessionHandoffCapabilityId());
  if (capability_id.has_value() && IsZeroHash(*capability_id)) {
    return absl::DataLossError(
        "WinnerReplay inference runtime returned a zero session-handoff "
        "capability ID.");
  }
  return capability_id;
}

absl::StatusOr<DPMStageCapabilities>
CanonicalWinnerDPMAgentRuntime::GetCapabilities() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  DPMStageCapabilities capabilities{
      .stage = DPMCapabilityStage::kAgentDecision,
      .replay_mode = DPMReplayMode::kCanonicalWinnerReplay,
      .runtime_identity = runtime_identity_,
      .max_output_tokens = kMaximumDPMGenerationTokens,
  };
  ABSL_RETURN_IF_ERROR(ValidateDPMStageCapabilities(capabilities));
  return capabilities;
}

absl::StatusOr<std::optional<SessionHandoffCapability>>
CanonicalWinnerDPMAgentRuntime::GetSessionHandoffCapability() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(
      std::optional<SessionHandoffCapability> capability,
      inference_runtime_->GetSessionHandoffCapability());
  if (capability.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffCapability(*capability));
    if (capability->session_identity != runtime_identity_) {
      return absl::FailedPreconditionError(
          "WinnerReplay CapsuleRestore capability belongs to another "
          "runtime identity.");
    }
  }
  return capability;
}

absl::StatusOr<std::optional<CapsuleRestoreOperationalCoverage>>
CanonicalWinnerDPMAgentRuntime::GetCapsuleRestoreOperationalCoverage()
    const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(
      std::optional<CapsuleRestoreOperationalCoverage> coverage,
      inference_runtime_->GetCapsuleRestoreOperationalCoverage());
  if (coverage.has_value()) {
    ABSL_RETURN_IF_ERROR(
        ValidateCapsuleRestoreOperationalCoverage(*coverage));
  }
  return coverage;
}

absl::StatusOr<std::optional<
    AuthenticatedCapsuleRestoreStateWitnessAdmission>>
CanonicalWinnerDPMAgentRuntime::
    GetAuthenticatedCapsuleRestoreStateWitnessAdmission() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(
      std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>
          admission,
      inference_runtime_
          ->GetAuthenticatedCapsuleRestoreStateWitnessAdmission());
  if (admission.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateStateWitnessAdmissionForAgentRuntime(
        *admission, runtime_identity_, std::nullopt));
  }
  if (inference_runtime_->GetSessionHandoffIdentity() != runtime_identity_) {
    return absl::AbortedError(
        "WinnerReplay agent runtime identity changed while resolving "
        "CapsuleRestore Coverage V2 authority.");
  }
  return admission;
}

absl::Status CanonicalWinnerDPMAgentRuntime::ValidateSupport() const {
  if (inference_runtime_ == nullptr) {
    return absl::InvalidArgumentError(
        "WinnerReplay agent has no inference runtime.");
  }
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(runtime_identity_));
  if (inference_runtime_->GetSessionHandoffIdentity() != runtime_identity_) {
    return absl::FailedPreconditionError(
        "WinnerReplay agent runtime identity changed after construction.");
  }
  ABSL_RETURN_IF_ERROR(inference_runtime_->ValidateSupport());
  if (inference_runtime_->GetSessionHandoffIdentity() != runtime_identity_) {
    return absl::FailedPreconditionError(
        "WinnerReplay agent runtime changed while validating support.");
  }
  return absl::OkStatus();
}

absl::Status CanonicalWinnerDPMAgentRuntime::ValidateGenerationLimit(
    uint32_t max_output_tokens) const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  if (max_output_tokens == 0 ||
      max_output_tokens > kMaximumDPMGenerationTokens) {
    return absl::InvalidArgumentError(
        "WinnerReplay agent token limit is outside the product bound.");
  }
  return absl::OkStatus();
}

absl::Status
CanonicalWinnerDPMAgentRuntime::ValidateSessionHandoffSupport() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_RETURN_IF_ERROR(inference_runtime_->ValidateSessionHandoffSupport());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>
          state_witness_admission,
      GetAuthenticatedCapsuleRestoreStateWitnessAdmission());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<Hash256> admission_id,
      inference_runtime_->GetCapsuleRestoreAdmissionRecordId());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<Hash256> capability_id,
      inference_runtime_->GetSessionHandoffCapabilityId());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<SessionHandoffCapability> capability,
      inference_runtime_->GetSessionHandoffCapability());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<CapsuleRestoreOperationalCoverage> coverage,
      inference_runtime_->GetCapsuleRestoreOperationalCoverage());
  const bool has_any_v1 = admission_id.has_value() ||
                          capability_id.has_value() || capability.has_value() ||
                          coverage.has_value();
  const bool has_complete_v1 = admission_id.has_value() &&
                               capability_id.has_value() &&
                               capability.has_value() && coverage.has_value();
  if (state_witness_admission.has_value() && has_any_v1) {
    return absl::FailedPreconditionError(
        "WinnerReplay runtime cannot combine CapsuleRestore Coverage V1 and "
        "V2 authority.");
  }
  if (!state_witness_admission.has_value() && !has_complete_v1) {
    return absl::FailedPreconditionError(
        "WinnerReplay CapsuleRestore support requires a current admission "
        "record and complete Engine-derived capability.");
  }
  if (has_complete_v1) {
    if (IsZeroHash(*admission_id) || IsZeroHash(*capability_id)) {
      return absl::FailedPreconditionError(
          "WinnerReplay CapsuleRestore V1 authority has an empty identity.");
    }
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffCapability(*capability));
    ABSL_RETURN_IF_ERROR(
        ValidateCapsuleRestoreOperationalCoverage(*coverage));
    if (capability->capability_id != *capability_id ||
        capability->session_identity != runtime_identity_ ||
        coverage->capsule_restore_capability_id != *capability_id ||
        coverage->capsule_restore_admission_record_id != *admission_id) {
      return absl::FailedPreconditionError(
          "WinnerReplay CapsuleRestore capability IDs or runtime identity do "
          "not agree.");
    }
  }
  ABSL_RETURN_IF_ERROR(inference_runtime_->ValidateSessionHandoffSupport());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>
          state_witness_admission_after,
      GetAuthenticatedCapsuleRestoreStateWitnessAdmission());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<Hash256> admission_after,
      inference_runtime_->GetCapsuleRestoreAdmissionRecordId());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<SessionHandoffCapability> capability_after,
      inference_runtime_->GetSessionHandoffCapability());
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<CapsuleRestoreOperationalCoverage> coverage_after,
      inference_runtime_->GetCapsuleRestoreOperationalCoverage());
  if (admission_after != admission_id || capability_after != capability ||
      coverage_after != coverage ||
      state_witness_admission_after.has_value() !=
          state_witness_admission.has_value() ||
      (state_witness_admission.has_value() &&
       !HasSameStateWitnessAdmissionAuthority(
           *state_witness_admission_after, *state_witness_admission))) {
    return absl::AbortedError(
        "WinnerReplay CapsuleRestore admission changed during support "
        "validation.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Engine::Session>>
CanonicalWinnerDPMAgentRuntime::CreateSession() {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Engine::Session> session,
                        inference_runtime_->CreateSession());
  if (session == nullptr) {
    return absl::InternalError(
        "WinnerReplay agent runtime returned a null session.");
  }
  ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity identity,
                        session->GetSessionHandoffIdentity());
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(identity));
  if (identity != runtime_identity_) {
    return absl::FailedPreconditionError(
        "WinnerReplay agent created a session with another identity.");
  }
  return session;
}

absl::StatusOr<DPMAgentReplayExecution>
CanonicalWinnerDPMAgentRuntime::Generate(
    Engine::Session* producing_session,
    const DPMAgentGenerationRequest& execution_request,
    const DPMAgentExecutionRequest& logical_request) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentExecutionRequest(logical_request));
  ABSL_RETURN_IF_ERROR(ValidateGenerationRequest(execution_request));
  if (producing_session == nullptr ||
      static_cast<uint32_t>(execution_request.max_output_tokens) !=
          logical_request.max_output_tokens) {
    return absl::InvalidArgumentError(
        "WinnerReplay agent requires a live session and a matching execution "
        "request.");
  }
  std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>
      state_witness_admission_before;
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>
          current_state_witness_admission,
      GetAuthenticatedCapsuleRestoreStateWitnessAdmission());
  const bool is_restore = execution_request.restore_checkpoint_id.has_value();
  if (execution_request.capsule_restore_operation_v3.has_value() !=
      (current_state_witness_admission.has_value() && is_restore)) {
    return absl::FailedPreconditionError(
        "WinnerReplay Coverage V2 restore operation is required exactly for "
        "a restore on a Coverage V2-bound runtime.");
  }
  if (is_restore && !current_state_witness_admission.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffSupport());
  }
  if (current_state_witness_admission.has_value() && is_restore) {
    ABSL_RETURN_IF_ERROR(ValidateWinnerRestoreOperationAuthority(
        execution_request, logical_request,
        *current_state_witness_admission));
    state_witness_admission_before = *current_state_witness_admission;
  }
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMAgentExecutionRequest(logical_request));
  const DPMCanonicalReplayRequest replay_request =
      MakeReplayRequest(std::move(encoded),
                        logical_request.max_output_tokens);
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_request_hash,
      ComputeDPMCanonicalReplayRequestHash(replay_request));
  WinnerAgentInvocation invocation(inference_runtime_, producing_session,
                                   &execution_request, &logical_request,
                                   runtime_identity_);
  ABSL_ASSIGN_OR_RETURN(
      CanonicalWinnerReplayExecution replay,
      replay_executor_.Run(replay_request, &invocation));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_evidence_hash,
      ComputeWinnerEvidenceHash(runtime_identity_, replay_request,
                                replay.canonical_output));
  bool expected_generation = false;
  bool expected_producing_session_match = false;
  bool expected_reuse = false;
  switch (replay.resolution) {
    case CanonicalWinnerResolution::kReplayed:
      expected_reuse = true;
      break;
    case CanonicalWinnerResolution::kPublished:
      expected_generation = true;
      expected_producing_session_match = true;
      break;
    case CanonicalWinnerResolution::kLostPublishRace:
      expected_generation = true;
      expected_reuse = true;
      break;
    default:
      return absl::DataLossError(
          "WinnerReplay agent returned an unknown catalog resolution.");
  }
  if (replay.mode != DPMReplayMode::kCanonicalWinnerReplay ||
      replay.runtime_identity != runtime_identity_ ||
      replay.canonical_request_hash != expected_request_hash ||
      replay.canonical_output_hash != Sha256(replay.canonical_output) ||
      replay.execution_evidence_hash != expected_evidence_hash ||
      invocation.generate_called() != expected_generation ||
      invocation.generation_succeeded() != expected_generation ||
      replay.producing_session_matches_output !=
          expected_producing_session_match ||
      (expected_producing_session_match &&
       !invocation.GeneratedSelectedCandidate(replay))) {
    return absl::DataLossError(
        "WinnerReplay agent result is not bound to this request, runtime, "
        "catalog resolution, and live producing session.");
  }
  if (state_witness_admission_before.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        const std::optional<
            AuthenticatedCapsuleRestoreStateWitnessAdmission>
            current_after,
        GetAuthenticatedCapsuleRestoreStateWitnessAdmission());
    if (!current_after.has_value() ||
        !HasSameStateWitnessAdmissionAuthority(
            *current_after, *state_witness_admission_before)) {
      return absl::AbortedError(
          "WinnerReplay CapsuleRestore Coverage V2 authority changed during "
          "catalog resolution or physical generation.");
    }
  } else if (is_restore) {
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffSupport());
  }
  ABSL_ASSIGN_OR_RETURN(
      const DPMAgentDecisionEnvelope envelope,
      DecodeDPMAgentDecisionEnvelope(replay.canonical_output));
  ABSL_ASSIGN_OR_RETURN(const std::vector<int32_t> exact_ids,
                        DecodeFreshWorkerTokenIds(
                            envelope.canonical_token_bytes));
  std::vector<int> decision_ids(exact_ids.begin(), exact_ids.end());
  DPMAgentReplayExecution execution{
      .mode = DPMReplayMode::kCanonicalWinnerReplay,
      .replay_request_hash = replay.canonical_request_hash,
      .execution_evidence_hash = replay.execution_evidence_hash,
      .decision_output = envelope.decision_output,
      .decision_token_ids = std::move(decision_ids),
      .reused_canonical_winner = expected_reuse,
      .producing_session_matches_output =
          expected_producing_session_match,
      .prepared_prefill_plan =
          expected_producing_session_match
              ? invocation.prepared_prefill_plan()
              : std::optional<DPMPreparedPrefillPlan>(),
      .capsule_restore_evidence_v3 =
          expected_producing_session_match
              ? invocation.capsule_restore_evidence_v3()
              : std::optional<CapsuleRestoreEvidenceV3>(),
  };
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentReplayExecution(
      execution, logical_request.max_output_tokens));
  return execution;
}

absl::StatusOr<DPMAgentReplayExecution>
CanonicalWinnerDPMAgentRuntime::RematerializeCanonicalWinner(
    Engine::Session* producing_session,
    const DPMAgentGenerationRequest& execution_request,
    const DPMAgentExecutionRequest& logical_request,
    const DPMAgentReplayExecution& selected_winner) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentExecutionRequest(logical_request));
  ABSL_RETURN_IF_ERROR(ValidateGenerationRequest(execution_request));
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentReplayExecution(
      selected_winner, logical_request.max_output_tokens));
  if (producing_session == nullptr ||
      static_cast<uint32_t>(execution_request.max_output_tokens) !=
          logical_request.max_output_tokens) {
    return absl::InvalidArgumentError(
        "WinnerReplay rematerialization requires a live session and matching "
        "execution and logical request limits.");
  }
  if (selected_winner.mode != DPMReplayMode::kCanonicalWinnerReplay ||
      !selected_winner.reused_canonical_winner ||
      selected_winner.rematerialized_canonical_winner ||
      selected_winner.producing_session_matches_output) {
    return absl::FailedPreconditionError(
        "WinnerReplay rematerialization requires an already selected catalog "
        "winner without a live producing session.");
  }
  std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>
      state_witness_admission_before;
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>
          current_state_witness_admission,
      GetAuthenticatedCapsuleRestoreStateWitnessAdmission());
  const bool is_restore = execution_request.restore_checkpoint_id.has_value();
  if (execution_request.capsule_restore_operation_v3.has_value() !=
      (current_state_witness_admission.has_value() && is_restore)) {
    return absl::FailedPreconditionError(
        "WinnerReplay rematerialization requires Coverage V2 operation "
        "evidence exactly for a restore on a Coverage V2-bound runtime.");
  }
  if (is_restore && !current_state_witness_admission.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffSupport());
  }
  if (current_state_witness_admission.has_value() && is_restore) {
    ABSL_RETURN_IF_ERROR(ValidateWinnerRestoreOperationAuthority(
        execution_request, logical_request,
        *current_state_witness_admission));
    state_witness_admission_before = *current_state_witness_admission;
  }

  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMAgentExecutionRequest(logical_request));
  const DPMCanonicalReplayRequest replay_request =
      MakeReplayRequest(std::move(encoded), logical_request.max_output_tokens);
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_request_hash,
      ComputeDPMCanonicalReplayRequestHash(replay_request));
  if (selected_winner.replay_request_hash != expected_request_hash) {
    return absl::FailedPreconditionError(
        "Selected WinnerReplay result belongs to another logical request.");
  }

  std::vector<int32_t> selected_token_ids;
  selected_token_ids.reserve(selected_winner.decision_token_ids.size());
  for (int token_id : selected_winner.decision_token_ids) {
    if (token_id < 0 ||
        static_cast<int64_t>(token_id) >
            std::numeric_limits<int32_t>::max()) {
      return absl::DataLossError(
          "Selected WinnerReplay result contains a non-int32 token ID.");
    }
    selected_token_ids.push_back(static_cast<int32_t>(token_id));
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string selected_token_bytes,
      EncodeFreshWorkerTokenIds(selected_token_ids));
  DPMAgentDecisionEnvelope selected_envelope{
      .decision_output = selected_winner.decision_output,
      .canonical_token_bytes = std::move(selected_token_bytes),
  };
  ABSL_ASSIGN_OR_RETURN(
      const std::string selected_canonical_output,
      EncodeDPMAgentDecisionEnvelope(selected_envelope));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 selected_expected_evidence,
      ComputeWinnerEvidenceHash(runtime_identity_, replay_request,
                                selected_canonical_output));
  if (selected_winner.execution_evidence_hash != selected_expected_evidence) {
    return absl::DataLossError(
        "Selected WinnerReplay evidence does not authenticate its canonical "
        "request and output bytes.");
  }

  // Invoke the loaded runtime directly. Do not call replay_executor_: a catalog
  // lookup would return the same winner without making this session its live
  // producer. WinnerAgentInvocation itself rejects a second model call.
  WinnerAgentInvocation invocation(inference_runtime_, producing_session,
                                   &execution_request, &logical_request,
                                   runtime_identity_);
  ABSL_ASSIGN_OR_RETURN(
      CanonicalWinnerGeneratedCandidate rematerialized,
      invocation.Generate(replay_request));
  ABSL_RETURN_IF_ERROR(
      invocation.ValidateSupport(DPMReplayStage::kAgentDecision));
  if (!invocation.generate_called() || !invocation.generation_succeeded()) {
    return absl::InternalError(
        "WinnerReplay rematerialization returned without exactly one "
        "successful model call.");
  }
  if (rematerialized.canonical_output != selected_canonical_output ||
      rematerialized.execution_evidence_hash !=
          selected_winner.execution_evidence_hash) {
    return absl::FailedPreconditionError(
        "Live WinnerReplay rematerialization did not byte-match the selected "
        "canonical winner and its execution evidence.");
  }
  if (state_witness_admission_before.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        const std::optional<
            AuthenticatedCapsuleRestoreStateWitnessAdmission>
            current_after,
        GetAuthenticatedCapsuleRestoreStateWitnessAdmission());
    if (!current_after.has_value() ||
        !HasSameStateWitnessAdmissionAuthority(
            *current_after, *state_witness_admission_before)) {
      return absl::AbortedError(
          "WinnerReplay CapsuleRestore Coverage V2 authority changed during "
          "live rematerialization.");
    }
  } else if (is_restore) {
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffSupport());
  }

  DPMAgentReplayExecution live_execution = selected_winner;
  live_execution.reused_canonical_winner = true;
  live_execution.rematerialized_canonical_winner = true;
  live_execution.producing_session_matches_output = true;
  live_execution.prepared_prefill_plan = invocation.prepared_prefill_plan();
  live_execution.capsule_restore_evidence_v3 =
      invocation.capsule_restore_evidence_v3();
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentReplayExecution(
      live_execution, logical_request.max_output_tokens));
  return live_execution;
}

absl::StatusOr<std::unique_ptr<ExactRegenerationDPMAgentRuntime>>
ExactRegenerationDPMAgentRuntime::Create(
    ExactRegenerationExecutor* exact_executor) {
  if (exact_executor == nullptr) {
    return absl::InvalidArgumentError(
        "Exact agent requires an ExactRegeneration executor.");
  }
  if (exact_executor->GetReplayStage() !=
      DPMReplayStage::kAgentDecision) {
    return absl::FailedPreconditionError(
        "Exact agent executor is bound to another replay stage.");
  }
  ABSL_ASSIGN_OR_RETURN(ExactLiteRtProfile profile,
                        exact_executor->GetDerivedProfile());
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(profile));
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(profile.session_identity));
  if (IsZeroHash(profile.profile_id)) {
    return absl::FailedPreconditionError(
        "Exact agent Engine returned an empty derived profile ID.");
  }
  auto runtime = std::unique_ptr<ExactRegenerationDPMAgentRuntime>(
      new ExactRegenerationDPMAgentRuntime(exact_executor,
                                           std::move(profile), std::nullopt,
                                           std::nullopt, std::nullopt,
                                           std::nullopt, std::nullopt,
                                           std::nullopt));
  ABSL_RETURN_IF_ERROR(runtime->ValidateSupport());
  return runtime;
}

absl::StatusOr<std::unique_ptr<ExactRegenerationDPMAgentRuntime>>
ExactRegenerationDPMAgentRuntime::Create(
    ExactRegenerationExecutor* exact_executor,
    CapsuleRestoreAdmissionBinding capsule_restore_admission) {
  if (exact_executor == nullptr) {
    return absl::InvalidArgumentError(
        "Exact agent requires an ExactRegeneration executor.");
  }
  if (exact_executor->GetReplayStage() !=
      DPMReplayStage::kAgentDecision) {
    return absl::FailedPreconditionError(
        "Exact agent executor is bound to another replay stage.");
  }
  ABSL_ASSIGN_OR_RETURN(ExactLiteRtProfile profile,
                        exact_executor->GetDerivedProfile());
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(profile));
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(profile.session_identity));
  if (IsZeroHash(profile.profile_id)) {
    return absl::FailedPreconditionError(
        "Exact agent Engine returned an empty derived profile ID.");
  }
  ABSL_ASSIGN_OR_RETURN(
      AuthenticatedCapsuleRestoreAdmission admission,
      exact_executor->GetAuthenticatedCapsuleRestoreAdmission(
          capsule_restore_admission));
  if (admission.profile != profile ||
      admission.capability.exact_profile_id != profile.profile_id ||
      admission.capability.session_identity != profile.session_identity ||
      admission.capability.backend != profile.backend ||
      admission.record.capability != admission.capability ||
      admission.operational_coverage.capsule_restore_capability_id !=
          admission.capability.capability_id ||
      admission.operational_coverage
              .capsule_restore_admission_record_id !=
          admission.record.record_id ||
      IsZeroHash(admission.record.record_id)) {
    return absl::FailedPreconditionError(
        "Exact agent CapsuleRestore admission is not bound to its immutable "
        "Engine profile.");
  }
  const Hash256 admission_record_id = admission.record.record_id;
  SessionHandoffCapability capability = admission.capability;
  CapsuleRestoreOperationalCoverage operational_coverage =
      admission.operational_coverage;
  auto runtime = std::unique_ptr<ExactRegenerationDPMAgentRuntime>(
      new ExactRegenerationDPMAgentRuntime(
          exact_executor, std::move(profile),
          std::optional<CapsuleRestoreAdmissionBinding>(
              std::move(capsule_restore_admission)),
          std::optional<Hash256>(admission_record_id),
          std::optional<SessionHandoffCapability>(std::move(capability)),
          std::optional<CapsuleRestoreOperationalCoverage>(
              std::move(operational_coverage)),
          std::nullopt, std::nullopt));
  ABSL_RETURN_IF_ERROR(runtime->ValidateSupport());
  ABSL_RETURN_IF_ERROR(runtime->ValidateSessionHandoffSupport());
  return runtime;
}

absl::StatusOr<std::unique_ptr<ExactRegenerationDPMAgentRuntime>>
ExactRegenerationDPMAgentRuntime::Create(
    ExactRegenerationExecutor* exact_executor,
    CapsuleRestoreStateWitnessAdmissionBinding
        capsule_restore_state_witness_admission) {
  if (exact_executor == nullptr) {
    return absl::InvalidArgumentError(
        "Exact agent requires an ExactRegeneration executor.");
  }
  if (exact_executor->GetReplayStage() !=
      DPMReplayStage::kAgentDecision) {
    return absl::FailedPreconditionError(
        "Exact agent executor is bound to another replay stage.");
  }
  ABSL_ASSIGN_OR_RETURN(ExactLiteRtProfile profile,
                        exact_executor->GetDerivedProfile());
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(profile));
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(profile.session_identity));
  if (IsZeroHash(profile.profile_id)) {
    return absl::FailedPreconditionError(
        "Exact agent Engine returned an empty derived profile ID.");
  }
  ABSL_ASSIGN_OR_RETURN(
      AuthenticatedCapsuleRestoreStateWitnessAdmission admission,
      exact_executor->GetAuthenticatedCapsuleRestoreStateWitnessAdmission(
          capsule_restore_state_witness_admission));
  ABSL_RETURN_IF_ERROR(ValidateStateWitnessAdmissionForAgentRuntime(
      admission, profile.session_identity,
      std::optional<ExactLiteRtProfile>(profile)));
  auto runtime = std::unique_ptr<ExactRegenerationDPMAgentRuntime>(
      new ExactRegenerationDPMAgentRuntime(
          exact_executor, std::move(profile), std::nullopt, std::nullopt,
          std::nullopt, std::nullopt,
          std::optional<CapsuleRestoreStateWitnessAdmissionBinding>(
              std::move(capsule_restore_state_witness_admission)),
          std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>(
              std::move(admission))));
  ABSL_RETURN_IF_ERROR(runtime->ValidateSupport());
  ABSL_RETURN_IF_ERROR(runtime->ValidateSessionHandoffSupport());
  return runtime;
}

absl::StatusOr<std::optional<Hash256>>
ExactRegenerationDPMAgentRuntime::GetExactProfileId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  return std::optional<Hash256>(derived_profile_.profile_id);
}

absl::StatusOr<std::optional<Hash256>>
ExactRegenerationDPMAgentRuntime::GetExactProfileAdmissionRecordId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 admission_id,
      exact_executor_->GetProfileAdmissionRecordId());
  if (IsZeroHash(admission_id)) {
    return absl::DataLossError(
        "Exact agent admission repository returned a zero record ID.");
  }
  return std::optional<Hash256>(admission_id);
}

absl::StatusOr<AuthenticatedCapsuleRestoreAdmission>
ExactRegenerationDPMAgentRuntime::ResolveCurrentCapsuleRestoreAdmission()
    const {
  const bool has_binding = capsule_restore_admission_.has_value();
  if (has_binding != capsule_restore_admission_record_id_.has_value() ||
      has_binding != session_handoff_capability_.has_value() ||
      has_binding != capsule_restore_operational_coverage_.has_value()) {
    return absl::InternalError(
        "Exact agent CapsuleRestore binding is internally inconsistent.");
  }
  if (!has_binding) {
    return absl::FailedPreconditionError(
        "Exact agent was constructed without CapsuleRestore admission.");
  }
  ABSL_ASSIGN_OR_RETURN(
      AuthenticatedCapsuleRestoreAdmission current,
      exact_executor_->GetAuthenticatedCapsuleRestoreAdmission(
          *capsule_restore_admission_));
  if (current.record.record_id !=
          *capsule_restore_admission_record_id_ ||
      current.profile != derived_profile_ ||
      current.capability != *session_handoff_capability_ ||
      current.operational_coverage !=
          *capsule_restore_operational_coverage_ ||
      current.record.capability != current.capability) {
    return absl::AbortedError(
        "Authenticated CapsuleRestore admission or Engine-derived "
        "capability changed after exact-runtime construction.");
  }
  return current;
}

absl::StatusOr<AuthenticatedCapsuleRestoreStateWitnessAdmission>
ExactRegenerationDPMAgentRuntime::
    ResolveCurrentCapsuleRestoreStateWitnessAdmission() const {
  const bool has_binding =
      capsule_restore_state_witness_admission_.has_value();
  const bool has_initial =
      initial_capsule_restore_state_witness_admission_.has_value();
  const bool has_any_v1 = capsule_restore_admission_.has_value() ||
                          capsule_restore_admission_record_id_.has_value() ||
                          session_handoff_capability_.has_value() ||
                          capsule_restore_operational_coverage_.has_value();
  if (has_binding != has_initial || has_any_v1) {
    return absl::InternalError(
        "Exact agent CapsuleRestore Coverage V2 binding is internally "
        "inconsistent or combined with Coverage V1.");
  }
  if (!has_binding) {
    return absl::FailedPreconditionError(
        "Exact agent was constructed without CapsuleRestore Coverage V2 "
        "admission.");
  }
  ABSL_ASSIGN_OR_RETURN(
      AuthenticatedCapsuleRestoreStateWitnessAdmission current,
      exact_executor_->GetAuthenticatedCapsuleRestoreStateWitnessAdmission(
          *capsule_restore_state_witness_admission_));
  ABSL_RETURN_IF_ERROR(ValidateStateWitnessAdmissionForAgentRuntime(
      current, derived_profile_.session_identity,
      std::optional<ExactLiteRtProfile>(derived_profile_)));
  if (!HasSameStateWitnessAdmissionAuthority(
          current, *initial_capsule_restore_state_witness_admission_)) {
    return absl::AbortedError(
        "Authenticated CapsuleRestore Coverage V2 authority changed after "
        "exact-runtime construction.");
  }
  return current;
}

absl::StatusOr<std::optional<Hash256>>
ExactRegenerationDPMAgentRuntime::GetCapsuleRestoreAdmissionRecordId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  if (!capsule_restore_admission_.has_value()) {
    return std::optional<Hash256>();
  }
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  return std::optional<Hash256>(current.record.record_id);
}

absl::StatusOr<std::optional<Hash256>>
ExactRegenerationDPMAgentRuntime::GetSessionHandoffCapabilityId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  if (!capsule_restore_admission_.has_value()) {
    return std::optional<Hash256>();
  }
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  return std::optional<Hash256>(current.capability.capability_id);
}

absl::StatusOr<DPMStageCapabilities>
ExactRegenerationDPMAgentRuntime::GetCapabilities() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 current_admission_record_id,
      exact_executor_->GetProfileAdmissionRecordId());
  if (IsZeroHash(current_admission_record_id)) {
    return absl::DataLossError(
        "Exact agent capability has an empty current admission record.");
  }
  DPMStageCapabilities capabilities{
      .stage = DPMCapabilityStage::kAgentDecision,
      .replay_mode = DPMReplayMode::kExactRegeneration,
      .runtime_identity = derived_profile_.session_identity,
      .exact_profile = derived_profile_,
      .exact_profile_admission_record_id = current_admission_record_id,
      .max_output_tokens = exact_executor_->GetMaxOutputTokens(),
  };
  ABSL_RETURN_IF_ERROR(ValidateDPMStageCapabilities(capabilities));
  return capabilities;
}

absl::StatusOr<std::optional<SessionHandoffCapability>>
ExactRegenerationDPMAgentRuntime::GetSessionHandoffCapability() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  if (!capsule_restore_admission_.has_value()) {
    return std::optional<SessionHandoffCapability>();
  }
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  return std::optional<SessionHandoffCapability>(current.capability);
}

absl::StatusOr<std::optional<CapsuleRestoreOperationalCoverage>>
ExactRegenerationDPMAgentRuntime::GetCapsuleRestoreOperationalCoverage()
    const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  if (!capsule_restore_admission_.has_value()) {
    return std::optional<CapsuleRestoreOperationalCoverage>();
  }
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreOperationalCoverage(
      current.operational_coverage));
  return std::optional<CapsuleRestoreOperationalCoverage>(
      current.operational_coverage);
}

absl::StatusOr<std::optional<
    AuthenticatedCapsuleRestoreStateWitnessAdmission>>
ExactRegenerationDPMAgentRuntime::
    GetAuthenticatedCapsuleRestoreStateWitnessAdmission() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  if (!capsule_restore_state_witness_admission_.has_value()) {
    return std::optional<
        AuthenticatedCapsuleRestoreStateWitnessAdmission>();
  }
  ABSL_ASSIGN_OR_RETURN(
      AuthenticatedCapsuleRestoreStateWitnessAdmission current,
      ResolveCurrentCapsuleRestoreStateWitnessAdmission());
  return std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>(
      std::move(current));
}

absl::Status ExactRegenerationDPMAgentRuntime::ValidateSupport() const {
  if (exact_executor_ == nullptr) {
    return absl::InvalidArgumentError(
        "Exact agent has no regeneration executor.");
  }
  if (exact_executor_->GetReplayStage() !=
          DPMReplayStage::kAgentDecision ||
      exact_executor_->GetMaxOutputTokens() == 0) {
    return absl::FailedPreconditionError(
        "Exact agent executor is not bound to a valid agent-decision "
        "profile.");
  }
  const bool has_v1_binding =
      capsule_restore_admission_.has_value();
  if (has_v1_binding !=
          capsule_restore_admission_record_id_.has_value() ||
      has_v1_binding != session_handoff_capability_.has_value() ||
      has_v1_binding !=
          capsule_restore_operational_coverage_.has_value()) {
    return absl::InternalError(
        "Exact agent CapsuleRestore Coverage V1 binding is internally "
        "inconsistent.");
  }
  const bool has_v2_binding =
      capsule_restore_state_witness_admission_.has_value();
  if (has_v2_binding !=
          initial_capsule_restore_state_witness_admission_.has_value() ||
      (has_v1_binding && has_v2_binding)) {
    return absl::InternalError(
        "Exact agent CapsuleRestore Coverage V2 binding is incomplete or "
        "combined with Coverage V1.");
  }
  ABSL_RETURN_IF_ERROR(exact_executor_->ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(const ExactLiteRtProfile current,
                        exact_executor_->GetDerivedProfile());
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(current));
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(derived_profile_));
  if (current != derived_profile_) {
    return absl::FailedPreconditionError(
        "Exact agent Engine profile changed after construction.");
  }
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(current.session_identity));
  if (IsZeroHash(current.profile_id)) {
    return absl::FailedPreconditionError(
        "Exact agent Engine returned an empty derived profile ID.");
  }
  if (has_v1_binding) {
    ABSL_ASSIGN_OR_RETURN(
        const AuthenticatedCapsuleRestoreAdmission capsule_admission,
        ResolveCurrentCapsuleRestoreAdmission());
    (void)capsule_admission;
  }
  if (has_v2_binding) {
    ABSL_ASSIGN_OR_RETURN(
        const AuthenticatedCapsuleRestoreStateWitnessAdmission
            state_witness_admission,
        ResolveCurrentCapsuleRestoreStateWitnessAdmission());
    (void)state_witness_admission;
  }
  return absl::OkStatus();
}

absl::Status ExactRegenerationDPMAgentRuntime::ValidateGenerationLimit(
    uint32_t max_output_tokens) const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  if (max_output_tokens != exact_executor_->GetMaxOutputTokens()) {
    return absl::FailedPreconditionError(
        "DPM agent token limit differs from the immutable exact-worker "
        "profile.");
  }
  return absl::OkStatus();
}

absl::Status
ExactRegenerationDPMAgentRuntime::ValidateSessionHandoffSupport() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  if (capsule_restore_state_witness_admission_.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        const AuthenticatedCapsuleRestoreStateWitnessAdmission before,
        ResolveCurrentCapsuleRestoreStateWitnessAdmission());
    ABSL_ASSIGN_OR_RETURN(
        const AuthenticatedCapsuleRestoreStateWitnessAdmission after,
        ResolveCurrentCapsuleRestoreStateWitnessAdmission());
    if (!HasSameStateWitnessAdmissionAuthority(before, after)) {
      return absl::AbortedError(
          "Exact agent CapsuleRestore Coverage V2 authority changed during "
          "support validation.");
    }
    return absl::OkStatus();
  }
  ABSL_ASSIGN_OR_RETURN(
      const AuthenticatedCapsuleRestoreAdmission current,
      ResolveCurrentCapsuleRestoreAdmission());
  if (current.record.record_id !=
          *capsule_restore_admission_record_id_ ||
      current.profile != derived_profile_ ||
      current.capability.capability_id !=
          session_handoff_capability_->capability_id ||
      current.operational_coverage !=
          *capsule_restore_operational_coverage_) {
    return absl::AbortedError(
        "Exact agent CapsuleRestore admission binding changed during "
        "support validation.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Engine::Session>>
ExactRegenerationDPMAgentRuntime::CreateSession() {
  return absl::UnimplementedError(
      "Exact agent sessions live in fresh workers; use GeneratePhysicalExact "
      "instead of manufacturing a parent producing session.");
}

absl::StatusOr<DPMAgentReplayExecution>
ExactRegenerationDPMAgentRuntime::Generate(
    Engine::Session* producing_session,
    const DPMAgentGenerationRequest& execution_request,
    const DPMAgentExecutionRequest& logical_request) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentExecutionRequest(logical_request));
  ABSL_RETURN_IF_ERROR(ValidateGenerationRequest(execution_request));
  ABSL_RETURN_IF_ERROR(
      ValidateGenerationLimit(logical_request.max_output_tokens));
  if (producing_session != nullptr ||
      static_cast<uint32_t>(execution_request.max_output_tokens) !=
          logical_request.max_output_tokens ||
      execution_request.restore_checkpoint_id.has_value() ||
      execution_request.restored_state_witness.has_value() ||
      execution_request.capsule_restore_operation_v3.has_value() ||
      !ChunksEqual(execution_request.canonical_prefill_chunks,
                   logical_request.full_canonical_prefill_chunks)) {
    return absl::InvalidArgumentError(
        "Exact agent requires no parent session and the complete logical "
        "prefill request.");
  }
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMAgentExecutionRequest(logical_request));
  const DPMCanonicalReplayRequest replay_request =
      MakeReplayRequest(std::move(encoded),
                        logical_request.max_output_tokens);
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_request_hash,
      ComputeDPMCanonicalReplayRequestHash(replay_request));
  ABSL_ASSIGN_OR_RETURN(
      ExactRegenerationExecution exact,
      exact_executor_->Run(replay_request));
  if (exact.request_evidence.prefill_mode !=
          FreshWorkerPrefillMode::kFullCanonicalPrefill ||
      exact.request_evidence.restored_checkpoint_id.has_value() ||
      exact.request_evidence.capture_run_policy !=
          ExactRegenerationCaptureRunPolicy::kNoCapture ||
      exact.durable_producing_capsule_evidence.has_value()) {
    return absl::DataLossError(
        "Exact agent convenience execution was not full prefill without "
        "capture.");
  }
  return BuildExactAgentReplayExecution(
      exact, derived_profile_, expected_request_hash,
      logical_request.logical_agent_request_hash,
      logical_request.full_canonical_prefill_chunks,
      logical_request.max_output_tokens);
}

absl::StatusOr<ExactRegenerationDPMAgentPhysicalExecution>
ExactRegenerationDPMAgentRuntime::GeneratePhysicalExact(
    const DPMAgentExecutionRequest& logical_request,
    const ExactRegenerationExecutionInput& input) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentExecutionRequest(logical_request));
  ABSL_RETURN_IF_ERROR(
      ValidateGenerationLimit(logical_request.max_output_tokens));
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMAgentExecutionRequest(logical_request));
  const DPMCanonicalReplayRequest replay_request =
      MakeReplayRequest(std::move(encoded),
                        logical_request.max_output_tokens);
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_request_hash,
      ComputeDPMCanonicalReplayRequestHash(replay_request));
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerExecutionPlan(input.execution_plan));
  if (input.execution_plan.logical_replay_request_hash !=
      expected_request_hash) {
    return absl::FailedPreconditionError(
        "Exact agent physical plan is not bound to its complete logical "
        "request.");
  }
  std::optional<DPMAgentDeltaExecutionRequest> decoded_delta_request;
  switch (input.execution_plan.prefill_mode) {
    case FreshWorkerPrefillMode::kFullCanonicalPrefill:
      break;
    case FreshWorkerPrefillMode::kOwnPositionCapsuleDelta: {
      ABSL_ASSIGN_OR_RETURN(
          const DPMAgentDeltaExecutionRequest delta_request,
          DecodeDPMAgentDeltaExecutionRequest(
              input.execution_plan.canonical_execution_payload));
      ABSL_RETURN_IF_ERROR(ValidateDPMAgentDeltaExecutionBinding(
          logical_request, input.execution_plan, delta_request));
      decoded_delta_request = delta_request;
      break;
    }
    default:
      return absl::InvalidArgumentError(
          "Exact agent physical plan has an unknown prefill mode.");
  }
  const bool transfers_capsule =
      input.execution_plan.prefill_mode ==
          FreshWorkerPrefillMode::kOwnPositionCapsuleDelta ||
      input.execution_plan.capture_producing_capsule;
  std::optional<AuthenticatedCapsuleRestoreAdmission>
      v1_capsule_admission_before;
  std::optional<AuthenticatedCapsuleRestoreStateWitnessAdmission>
      v2_capsule_admission_before;
  if (transfers_capsule) {
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffSupport());
    if (capsule_restore_state_witness_admission_.has_value()) {
      ABSL_ASSIGN_OR_RETURN(v2_capsule_admission_before,
                            ResolveCurrentCapsuleRestoreStateWitnessAdmission());
      const CapsuleRestoreStateWitnessOperationalDomain& domain =
          v2_capsule_admission_before->operational_coverage.operational_domain;
      const std::vector<DPMAgentGenerationRequest::PrefillChunk>&
          physical_chunks =
              decoded_delta_request.has_value()
                  ? decoded_delta_request->canonical_delta_prefill_chunks
                  : logical_request.full_canonical_prefill_chunks;
      ABSL_RETURN_IF_ERROR(ValidateStateWitnessSourceDomain(
          physical_chunks, logical_request.max_output_tokens, domain));
      if (decoded_delta_request.has_value() &&
          (input.durable_restore_options == nullptr ||
           input.durable_restore_source == nullptr ||
           input.durable_restore_options->key_id !=
               domain.checkpoint_authentication_key_id ||
           input.execution_plan.restore_durable_envelope_size == 0 ||
           input.execution_plan.restore_durable_envelope_size >
               kMaximumCapsuleRestoreCheckpointBytes)) {
        return absl::FailedPreconditionError(
            "Exact worker restore transport is outside CapsuleRestore "
            "Coverage V2.");
      }
      if (input.execution_plan.capture_producing_capsule &&
          (input.staging_capture_options == nullptr ||
           input.staging_capture_destination == nullptr ||
           input.staging_capture_options->key_id !=
               domain.checkpoint_authentication_key_id)) {
        return absl::FailedPreconditionError(
            "Exact worker capture transport is outside CapsuleRestore "
            "Coverage V2.");
      }
    } else {
      ABSL_ASSIGN_OR_RETURN(v1_capsule_admission_before,
                            ResolveCurrentCapsuleRestoreAdmission());
      const CapsuleRestoreOperationalCoverage& coverage =
          v1_capsule_admission_before->operational_coverage;
      if (input.execution_plan.capture_producing_capsule &&
          input.execution_plan.prefill_mode ==
              FreshWorkerPrefillMode::kOwnPositionCapsuleDelta) {
        return absl::FailedPreconditionError(
            "CapsuleRestore Coverage V1 cannot recursively capture a restored "
            "continuation.");
      }
      if (decoded_delta_request.has_value()) {
        ABSL_ASSIGN_OR_RETURN(
            const std::vector<CapsuleRestorePrefillChunk> delta_chunks,
            ToCapsuleRestoreChunks(
                decoded_delta_request->canonical_delta_prefill_chunks));
        ABSL_ASSIGN_OR_RETURN(
            const Hash256 continuation_workload_hash,
            ComputeCapsuleRestoreContinuationWorkloadHash(
                delta_chunks, decoded_delta_request->max_output_tokens));
        if (continuation_workload_hash !=
                coverage.restore_continuation_workload_hash ||
            decoded_delta_request->max_output_tokens !=
                coverage.continuation_output_tokens ||
            input.execution_plan.restore_durable_envelope_size !=
                coverage.checkpoint_envelope_size ||
            input.durable_restore_options == nullptr ||
            input.durable_restore_options->key_id !=
                coverage.checkpoint_authentication_key_id) {
          return absl::FailedPreconditionError(
              "Exact worker restore request is outside CapsuleRestore "
              "Coverage V1.");
        }
      }
      if (input.execution_plan.capture_producing_capsule) {
        ABSL_ASSIGN_OR_RETURN(
            const std::vector<CapsuleRestorePrefillChunk> full_chunks,
            ToCapsuleRestoreChunks(
                logical_request.full_canonical_prefill_chunks));
        ABSL_ASSIGN_OR_RETURN(
            const Hash256 capture_workload_hash,
            ComputeCapsuleRestoreCheckpointWorkloadHash(
                full_chunks, logical_request.max_output_tokens));
        if (capture_workload_hash !=
                coverage.checkpoint_capture_workload_hash ||
            logical_request.max_output_tokens !=
                coverage.checkpoint_output_tokens ||
            input.staging_capture_options == nullptr ||
            input.staging_capture_options->key_id !=
                coverage.checkpoint_authentication_key_id) {
          return absl::FailedPreconditionError(
              "Exact worker capture request is outside CapsuleRestore "
              "Coverage V1.");
        }
      }
    }
  }
  absl::StatusOr<ExactRegenerationExecution> exact_result = [&]()
      -> absl::StatusOr<ExactRegenerationExecution> {
    if (transfers_capsule && v1_capsule_admission_before.has_value()) {
      return exact_executor_->RunPhysical(
          replay_request, input, *capsule_restore_admission_);
    }
    if (transfers_capsule && v2_capsule_admission_before.has_value()) {
      return exact_executor_->RunPhysical(
          replay_request, input, *capsule_restore_state_witness_admission_);
    }
    return exact_executor_->RunPhysical(replay_request, input);
  }();
  ABSL_ASSIGN_OR_RETURN(ExactRegenerationExecution exact,
                        std::move(exact_result));
  if (input.execution_plan.capture_producing_capsule &&
      v1_capsule_admission_before.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        const AuthenticatedCapsuleRestoreAdmission current_capsule,
        ResolveCurrentCapsuleRestoreAdmission());
    if (!exact.durable_producing_capsule_evidence.has_value() ||
        exact.durable_producing_capsule_evidence->envelope_size !=
            current_capsule.operational_coverage
                .checkpoint_envelope_size ||
        exact.durable_producing_capsule_evidence->key_id !=
            current_capsule.operational_coverage
                .checkpoint_authentication_key_id) {
      return absl::FailedPreconditionError(
          "Exact worker produced a capsule outside CapsuleRestore Coverage "
          "V1.");
    }
  }
  if (v2_capsule_admission_before.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        const AuthenticatedCapsuleRestoreStateWitnessAdmission
            current_capsule_after,
        ResolveCurrentCapsuleRestoreStateWitnessAdmission());
    if (!HasSameStateWitnessAdmissionAuthority(
            current_capsule_after, *v2_capsule_admission_before)) {
      return absl::AbortedError(
          "Exact worker CapsuleRestore Coverage V2 authority changed during "
          "physical work.");
    }
    if (!exact.prepared_prefill_plan.has_value()) {
      return absl::DataLossError(
          "Exact worker omitted its runtime-derived prepared-prefill plan.");
    }
    const bool restored = decoded_delta_request.has_value();
    ABSL_RETURN_IF_ERROR(ValidateStateWitnessPreparedPlanDomain(
        *exact.prepared_prefill_plan, logical_request.max_output_tokens,
        restored,
        current_capsule_after.operational_coverage.operational_domain));
    if (input.execution_plan.capture_producing_capsule) {
      const CapsuleRestoreStateWitnessOperationalDomain& domain =
          current_capsule_after.operational_coverage.operational_domain;
      if (exact.request_evidence.runs.empty() ||
          !exact.request_evidence.runs.front()
               .transient_producing_capsule_evidence.has_value()) {
        return absl::DataLossError(
            "Exact worker capture omitted its run-zero transient producer "
            "witnesses.");
      }
      const SessionContinuationStateWitness& producing_witness =
          exact.request_evidence.runs.front()
              .transient_producing_capsule_evidence->producer_first_export;
      if (!exact.durable_producing_capsule_evidence.has_value() ||
          exact.durable_producing_capsule_evidence->key_id !=
              domain.checkpoint_authentication_key_id ||
          exact.durable_producing_capsule_evidence->envelope_size == 0 ||
          exact.durable_producing_capsule_evidence->envelope_size >
              kMaximumCapsuleRestoreCheckpointBytes ||
          producing_witness.current_step <= 0 ||
          static_cast<uint64_t>(producing_witness.current_step) <
              domain.minimum_checkpoint_step ||
          static_cast<uint64_t>(producing_witness.current_step) >
              domain.maximum_checkpoint_step ||
          static_cast<uint64_t>(producing_witness.current_step) >
              domain.maximum_context_positions) {
        return absl::FailedPreconditionError(
            "Exact worker produced a capsule outside CapsuleRestore Coverage "
            "V2.");
      }
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      DPMAgentReplayExecution replay_execution,
      BuildExactAgentReplayExecution(
          exact, derived_profile_, expected_request_hash,
          logical_request.logical_agent_request_hash,
          decoded_delta_request.has_value()
              ? decoded_delta_request->canonical_delta_prefill_chunks
              : logical_request.full_canonical_prefill_chunks,
          logical_request.max_output_tokens));
  const ExactRegenerationCaptureRunPolicy expected_capture_policy =
      input.execution_plan.capture_producing_capsule
          ? ExactRegenerationCaptureRunPolicy::kRunZeroOnly
          : ExactRegenerationCaptureRunPolicy::kNoCapture;
  if (exact.request_evidence.physical_execution_plan_hash !=
          input.execution_plan.plan_hash ||
      exact.request_evidence.prefill_mode !=
          input.execution_plan.prefill_mode ||
      exact.request_evidence.restored_checkpoint_id !=
          input.execution_plan.restore_checkpoint_id ||
      exact.request_evidence.capture_run_policy !=
          expected_capture_policy) {
    return absl::DataLossError(
        "Exact agent worker evidence changed the selected physical plan.");
  }
  const ExactRegenerationRunEvidence& run_zero =
      exact.request_evidence.runs.front();
  ExactRegenerationDPMAgentPhysicalExecution physical{
      .replay_execution = std::move(replay_execution),
      .request_evidence = exact.request_evidence,
      .physical_execution_plan_hash =
          exact.request_evidence.physical_execution_plan_hash,
      .prefill_mode = exact.request_evidence.prefill_mode,
      .restored_checkpoint_id =
          exact.request_evidence.restored_checkpoint_id,
      .capture_run_policy = exact.request_evidence.capture_run_policy,
      .run_zero_restore_reauthentication_evidence =
          run_zero.restore_reauthentication_evidence,
      .run_zero_restored_state_witness =
          run_zero.restored_state_witness,
      .run_zero_transient_producing_capsule_evidence =
          run_zero.transient_producing_capsule_evidence,
      .durable_producing_capsule_evidence =
          exact.durable_producing_capsule_evidence,
  };
  ABSL_RETURN_IF_ERROR(
      ValidateExactRegenerationDPMAgentPhysicalExecution(
          physical, logical_request.max_output_tokens));
  return physical;
}

}  // namespace litert::lm
