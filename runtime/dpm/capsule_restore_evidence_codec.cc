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

#include "runtime/dpm/capsule_restore_evidence_codec.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_prepared_prefill_plan.h"
#include "runtime/platform/hash/hmac_sha256.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kEnvelopeMagic = {
    'D', 'P', 'M', 'C', 'E', 'V', '0', '2'};
constexpr absl::string_view kAuthenticationDomain =
    "LITERT_LMX_CAPSULE_CAPTURE_EVIDENCE_HMAC_SHA256_V2";
constexpr std::array<char, 8> kCaptureEnvelopeV3Magic = {
    'D', 'P', 'M', 'C', 'E', 'V', '0', '3'};
constexpr std::array<char, 8> kRestoreEnvelopeV3Magic = {
    'D', 'P', 'M', 'R', 'E', 'V', '0', '3'};
constexpr absl::string_view kCaptureAuthenticationV3Domain =
    "LITERT_LMX_CAPSULE_CAPTURE_EVIDENCE_HMAC_SHA256_V3";
constexpr absl::string_view kRestoreAuthenticationV3Domain =
    "LITERT_LMX_CAPSULE_RESTORE_EVIDENCE_HMAC_SHA256_V3";
constexpr uint64_t kEnvelopeHeaderBytes = 8 + 4 + 4 + 8;
constexpr uint64_t kAuthenticationTagBytes = 32;
constexpr uint32_t kMaximumOperationEvidenceKeyIdBytes = 256;

void AppendU8(uint8_t value, std::string* output) {
  output->push_back(static_cast<char>(value));
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

void AppendString(absl::string_view value, std::string* output) {
  AppendU32(static_cast<uint32_t>(value.size()), output);
  output->append(value.data(), value.size());
}

void AppendBlob(absl::string_view value, std::string* output) {
  AppendU64(value.size(), output);
  output->append(value.data(), value.size());
}

void AppendOptionalHash(const std::optional<Hash256>& value,
                        std::string* output) {
  AppendU8(value.has_value() ? 1 : 0, output);
  if (value.has_value()) AppendHash(*value, output);
}

void AppendOptionalU64(const std::optional<uint64_t>& value,
                       std::string* output) {
  AppendU8(value.has_value() ? 1 : 0, output);
  if (value.has_value()) AppendU64(*value, output);
}

class CanonicalReader {
 public:
  explicit CanonicalReader(absl::string_view bytes) : bytes_(bytes) {}

  bool empty() const { return offset_ == bytes_.size(); }
  uint64_t remaining() const { return bytes_.size() - offset_; }

  absl::StatusOr<uint8_t> ReadU8() {
    ABSL_RETURN_IF_ERROR(Require(1, "uint8"));
    return static_cast<uint8_t>(bytes_[offset_++]);
  }

  absl::StatusOr<uint32_t> ReadU32() {
    ABSL_RETURN_IF_ERROR(Require(4, "uint32"));
    uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      value = (value << 8) |
              static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 4;
    return value;
  }

  absl::StatusOr<uint64_t> ReadU64() {
    ABSL_RETURN_IF_ERROR(Require(8, "uint64"));
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
      value = (value << 8) |
              static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 8;
    return value;
  }

  absl::StatusOr<int32_t> ReadI32() {
    ABSL_ASSIGN_OR_RETURN(const uint32_t value, ReadU32());
    return static_cast<int32_t>(value);
  }

  absl::StatusOr<Hash256> ReadHash() {
    Hash256 hash;
    ABSL_RETURN_IF_ERROR(Require(hash.bytes.size(), "SHA-256 digest"));
    std::memcpy(hash.bytes.data(), bytes_.data() + offset_, hash.bytes.size());
    offset_ += hash.bytes.size();
    return hash;
  }

  absl::StatusOr<std::string> ReadString(uint64_t maximum,
                                         absl::string_view description) {
    ABSL_ASSIGN_OR_RETURN(const uint32_t size, ReadU32());
    if (size > maximum) {
      return absl::ResourceExhaustedError(
          absl::StrCat(description, " exceeds its durable bound."));
    }
    ABSL_RETURN_IF_ERROR(Require(size, description));
    std::string value(bytes_.data() + offset_, size);
    offset_ += size;
    return value;
  }

  absl::StatusOr<std::string> ReadBlob(uint64_t maximum,
                                       absl::string_view description) {
    ABSL_ASSIGN_OR_RETURN(const uint64_t size, ReadU64());
    if (size > maximum) {
      return absl::ResourceExhaustedError(
          absl::StrCat(description, " exceeds its durable bound."));
    }
    ABSL_RETURN_IF_ERROR(Require(size, description));
    std::string value(bytes_.data() + offset_, static_cast<size_t>(size));
    offset_ += static_cast<size_t>(size);
    return value;
  }

  absl::StatusOr<std::optional<Hash256>> ReadOptionalHash() {
    ABSL_ASSIGN_OR_RETURN(const uint8_t present, ReadU8());
    if (present == 0) return std::optional<Hash256>();
    if (present != 1) {
      return absl::DataLossError(
          "Coverage V2 evidence has a noncanonical optional-hash tag.");
    }
    ABSL_ASSIGN_OR_RETURN(const Hash256 value, ReadHash());
    return std::optional<Hash256>(value);
  }

  absl::StatusOr<std::optional<uint64_t>> ReadOptionalU64() {
    ABSL_ASSIGN_OR_RETURN(const uint8_t present, ReadU8());
    if (present == 0) return std::optional<uint64_t>();
    if (present != 1) {
      return absl::DataLossError(
          "Coverage V2 evidence has a noncanonical optional-uint64 tag.");
    }
    ABSL_ASSIGN_OR_RETURN(const uint64_t value, ReadU64());
    return std::optional<uint64_t>(value);
  }

 private:
  absl::Status Require(uint64_t count,
                       absl::string_view description) const {
    if (count > remaining()) {
      return absl::DataLossError(absl::StrCat(
          "Truncated Coverage V2 evidence ", description, "."));
    }
    return absl::OkStatus();
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

void AppendIdentity(const SessionHandoffIdentity& identity,
                    std::string* output) {
  AppendHash(identity.model_artifact_hash, output);
  AppendHash(identity.runtime_artifact_hash, output);
  AppendHash(identity.inference_profile_hash, output);
}

absl::StatusOr<SessionHandoffIdentity> ReadIdentity(CanonicalReader* reader) {
  SessionHandoffIdentity identity;
  ABSL_ASSIGN_OR_RETURN(identity.model_artifact_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(identity.runtime_artifact_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(identity.inference_profile_hash, reader->ReadHash());
  return identity;
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

absl::StatusOr<SessionHandoffCapability> ReadCapability(
    CanonicalReader* reader) {
  SessionHandoffCapability capability;
  ABSL_ASSIGN_OR_RETURN(capability.version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(capability.capability_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(capability.session_identity, ReadIdentity(reader));
  ABSL_ASSIGN_OR_RETURN(capability.exact_profile_id, reader->ReadHash());
  uint32_t backend;
  ABSL_ASSIGN_OR_RETURN(backend, reader->ReadU32());
  capability.backend = static_cast<ExactLiteRtBackend>(backend);
  ABSL_ASSIGN_OR_RETURN(capability.complete_state_inventory_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(capability.capsule_codec_contract_hash,
                        reader->ReadHash());
  return capability;
}

void AppendAuthority(const CapsuleRestoreAuthorityV2& authority,
                     std::string* output) {
  AppendCapability(authority.capability, output);
  AppendHash(authority.admission_record_id, output);
  AppendHash(authority.coverage_id, output);
  AppendHash(authority.qualification_spec_hash, output);
}

absl::StatusOr<CapsuleRestoreAuthorityV2> ReadAuthority(
    CanonicalReader* reader) {
  CapsuleRestoreAuthorityV2 authority;
  ABSL_ASSIGN_OR_RETURN(authority.capability, ReadCapability(reader));
  ABSL_ASSIGN_OR_RETURN(authority.admission_record_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(authority.coverage_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(authority.qualification_spec_hash, reader->ReadHash());
  return authority;
}

void AppendCheckpointState(const CapsuleDPMCheckpointStateV2& state,
                           std::string* output) {
  AppendString(state.log_id, output);
  AppendU64(state.source_event_count, output);
  AppendHash(state.source_prefix_hash, output);
  AppendU64(state.response_event_index, output);
  AppendHash(state.projection_request_hash, output);
  AppendHash(state.projection_manifest_hash, output);
  AppendHash(state.correction_digest, output);
  AppendHash(state.agent_transcript_hash, output);
  AppendHash(state.logical_agent_request_hash, output);
}

absl::StatusOr<CapsuleDPMCheckpointStateV2> ReadCheckpointState(
    CanonicalReader* reader) {
  CapsuleDPMCheckpointStateV2 state;
  ABSL_ASSIGN_OR_RETURN(
      state.log_id,
      reader->ReadString(kMaximumCapsuleEvidenceLogIdBytes, "log ID"));
  ABSL_ASSIGN_OR_RETURN(state.source_event_count, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(state.source_prefix_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.response_event_index, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(state.projection_request_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.projection_manifest_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.correction_digest, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.agent_transcript_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.logical_agent_request_hash, reader->ReadHash());
  return state;
}

void AppendRestoreTarget(const CapsuleDPMRestoreTargetV2& state,
                         std::string* output) {
  AppendString(state.log_id, output);
  AppendU64(state.source_event_count, output);
  AppendHash(state.source_prefix_hash, output);
  AppendU64(state.prospective_response_event_index, output);
  AppendHash(state.projection_request_hash, output);
  AppendHash(state.projection_manifest_hash, output);
  AppendHash(state.correction_digest, output);
  AppendHash(state.agent_transcript_prefix_hash, output);
  AppendHash(state.logical_agent_request_hash, output);
}

absl::StatusOr<CapsuleDPMRestoreTargetV2> ReadRestoreTarget(
    CanonicalReader* reader) {
  CapsuleDPMRestoreTargetV2 state;
  ABSL_ASSIGN_OR_RETURN(
      state.log_id,
      reader->ReadString(kMaximumCapsuleEvidenceLogIdBytes, "log ID"));
  ABSL_ASSIGN_OR_RETURN(state.source_event_count, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(state.source_prefix_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.prospective_response_event_index,
                        reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(state.projection_request_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.projection_manifest_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.correction_digest, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.agent_transcript_prefix_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(state.logical_agent_request_hash, reader->ReadHash());
  return state;
}

void AppendCanonicalChunks(
    const std::vector<CapsuleCanonicalPrefillChunkV2>& chunks,
    std::string* output) {
  AppendU32(static_cast<uint32_t>(chunks.size()), output);
  for (const CapsuleCanonicalPrefillChunkV2& chunk : chunks) {
    AppendU32(static_cast<uint32_t>(chunk.encoding), output);
    if (chunk.encoding ==
        CapsuleCanonicalPrefillChunkV2::Encoding::kUtf8Text) {
      AppendString(chunk.utf8_text, output);
    } else {
      AppendU32(static_cast<uint32_t>(chunk.token_ids.size()), output);
      for (int32_t token_id : chunk.token_ids) AppendI32(token_id, output);
    }
  }
}

absl::StatusOr<std::vector<CapsuleCanonicalPrefillChunkV2>>
ReadCanonicalChunks(CanonicalReader* reader) {
  ABSL_ASSIGN_OR_RETURN(const uint32_t count, reader->ReadU32());
  if (count == 0 || count > kMaximumCapsuleEvidencePrefillChunks) {
    return absl::ResourceExhaustedError(
        "Coverage V2 evidence declares an invalid canonical chunk count.");
  }
  std::vector<CapsuleCanonicalPrefillChunkV2> chunks;
  chunks.reserve(count);
  uint64_t total_text_bytes = 0;
  uint64_t total_token_ids = 0;
  for (uint32_t index = 0; index < count; ++index) {
    CapsuleCanonicalPrefillChunkV2 chunk;
    uint32_t encoding;
    ABSL_ASSIGN_OR_RETURN(encoding, reader->ReadU32());
    chunk.encoding =
        static_cast<CapsuleCanonicalPrefillChunkV2::Encoding>(encoding);
    switch (chunk.encoding) {
      case CapsuleCanonicalPrefillChunkV2::Encoding::kUtf8Text: {
        ABSL_ASSIGN_OR_RETURN(
            chunk.utf8_text,
            reader->ReadString(kMaximumCapsuleEvidencePrefillTextBytes,
                               "canonical prefill text"));
        if (chunk.utf8_text.size() >
            kMaximumCapsuleEvidencePrefillTextBytes - total_text_bytes) {
          return absl::ResourceExhaustedError(
              "Coverage V2 evidence exceeds its aggregate text bound.");
        }
        total_text_bytes += chunk.utf8_text.size();
        break;
      }
      case CapsuleCanonicalPrefillChunkV2::Encoding::kExactTokenIds: {
        uint32_t token_count;
        ABSL_ASSIGN_OR_RETURN(token_count, reader->ReadU32());
        if (token_count == 0 ||
            token_count > kMaximumCapsuleEvidenceTokenIds - total_token_ids ||
            uint64_t{token_count} * 4 > reader->remaining()) {
          return absl::ResourceExhaustedError(
              "Coverage V2 evidence declares an invalid token count.");
        }
        chunk.token_ids.reserve(token_count);
        for (uint32_t token_index = 0; token_index < token_count;
             ++token_index) {
          ABSL_ASSIGN_OR_RETURN(const int32_t token_id, reader->ReadI32());
          chunk.token_ids.push_back(token_id);
        }
        total_token_ids += token_count;
        break;
      }
      default:
        return absl::UnimplementedError(
            "Coverage V2 evidence contains an unsupported chunk encoding.");
    }
    chunks.push_back(std::move(chunk));
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCanonicalPrefillChunksV2(chunks));
  return chunks;
}

absl::Status AppendPrefillPlan(const CapsulePrefillPlanV2& plan,
                               std::string* output) {
  AppendU32(static_cast<uint32_t>(plan.mode), output);
  AppendU64(plan.event_range_start, output);
  AppendU64(plan.event_range_end, output);
  AppendU32(plan.start_step, output);
  AppendU32(plan.end_step, output);
  AppendCanonicalChunks(plan.canonical_chunks, output);
  AppendHash(plan.canonical_full_prefill_chunks_hash, output);
  AppendHash(plan.canonical_delta_chunks_hash, output);
  ABSL_ASSIGN_OR_RETURN(const std::string prepared,
                        EncodeDPMPreparedPrefillPlan(plan.prepared_plan));
  AppendBlob(prepared, output);
  return absl::OkStatus();
}

absl::StatusOr<CapsulePrefillPlanV2> ReadPrefillPlan(
    CanonicalReader* reader) {
  CapsulePrefillPlanV2 plan;
  uint32_t mode;
  ABSL_ASSIGN_OR_RETURN(mode, reader->ReadU32());
  plan.mode = static_cast<CapsulePrefillModeV2>(mode);
  ABSL_ASSIGN_OR_RETURN(plan.event_range_start, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(plan.event_range_end, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(plan.start_step, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.end_step, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.canonical_chunks, ReadCanonicalChunks(reader));
  ABSL_ASSIGN_OR_RETURN(plan.canonical_full_prefill_chunks_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.canonical_delta_chunks_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(
      const std::string prepared,
      reader->ReadBlob(kMaximumDPMPreparedPrefillPlanBytes,
                       "prepared prefill plan"));
  ABSL_ASSIGN_OR_RETURN(plan.prepared_plan,
                        DecodeDPMPreparedPrefillPlan(prepared));
  ABSL_RETURN_IF_ERROR(ValidateCapsulePrefillPlanV2(plan));
  return plan;
}

absl::Status AppendCapturePlan(const CapsuleCapturePlanV2& plan,
                               std::string* output) {
  AppendU32(plan.format_version, output);
  AppendHash(plan.plan_hash, output);
  AppendAuthority(plan.authority, output);
  AppendU32(static_cast<uint32_t>(plan.capture_basis), output);
  AppendCheckpointState(plan.checkpoint_state, output);
  AppendHash(plan.agent_transcript_prefix_hash, output);
  ABSL_RETURN_IF_ERROR(AppendPrefillPlan(plan.prefill, output));
  AppendHash(plan.producing_output_evidence_hash, output);
  AppendU32(plan.generated_token_count, output);
  AppendU32(plan.capture_end_step, output);
  AppendString(plan.checkpoint_authentication_key_id, output);
  AppendOptionalHash(plan.parent_checkpoint_id, output);
  AppendOptionalU64(plan.parent_response_event_index, output);
  AppendOptionalHash(plan.parent_restore_evidence_id, output);
  return absl::OkStatus();
}

absl::StatusOr<CapsuleCapturePlanV2> ReadCapturePlan(
    CanonicalReader* reader) {
  CapsuleCapturePlanV2 plan;
  ABSL_ASSIGN_OR_RETURN(plan.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.plan_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.authority, ReadAuthority(reader));
  uint32_t capture_basis;
  ABSL_ASSIGN_OR_RETURN(capture_basis, reader->ReadU32());
  plan.capture_basis = static_cast<CapsuleCaptureBasisV2>(capture_basis);
  ABSL_ASSIGN_OR_RETURN(plan.checkpoint_state, ReadCheckpointState(reader));
  ABSL_ASSIGN_OR_RETURN(plan.agent_transcript_prefix_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.prefill, ReadPrefillPlan(reader));
  ABSL_ASSIGN_OR_RETURN(plan.producing_output_evidence_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.generated_token_count, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.capture_end_step, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(
      plan.checkpoint_authentication_key_id,
      reader->ReadString(kMaximumCapsuleEvidenceKeyIdBytes,
                         "checkpoint authentication key ID"));
  ABSL_ASSIGN_OR_RETURN(plan.parent_checkpoint_id, reader->ReadOptionalHash());
  ABSL_ASSIGN_OR_RETURN(plan.parent_response_event_index,
                        reader->ReadOptionalU64());
  ABSL_ASSIGN_OR_RETURN(plan.parent_restore_evidence_id,
                        reader->ReadOptionalHash());
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCapturePlanV2(plan));
  return plan;
}

absl::Status AppendRestorePlan(const CapsuleRestorePlanV2& plan,
                               std::string* output) {
  AppendU32(plan.format_version, output);
  AppendHash(plan.plan_hash, output);
  AppendAuthority(plan.authority, output);
  AppendHash(plan.source_capture_plan_hash, output);
  AppendHash(plan.source_capture_evidence_id, output);
  AppendHash(plan.checkpoint_id, output);
  AppendCheckpointState(plan.checkpoint_state, output);
  AppendHash(plan.checkpoint_envelope_hash, output);
  AppendU64(plan.checkpoint_envelope_size, output);
  AppendString(plan.checkpoint_authentication_key_id, output);
  AppendU32(plan.checkpoint_step, output);
  AppendHash(plan.checkpoint_history_token_bytes_hash, output);
  AppendRestoreTarget(plan.target_state, output);
  ABSL_RETURN_IF_ERROR(AppendPrefillPlan(plan.prefill, output));
  AppendU32(plan.maximum_output_tokens, output);
  return absl::OkStatus();
}

absl::StatusOr<CapsuleRestorePlanV2> ReadRestorePlan(
    CanonicalReader* reader) {
  CapsuleRestorePlanV2 plan;
  ABSL_ASSIGN_OR_RETURN(plan.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.plan_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.authority, ReadAuthority(reader));
  ABSL_ASSIGN_OR_RETURN(plan.source_capture_plan_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.source_capture_evidence_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.checkpoint_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.checkpoint_state, ReadCheckpointState(reader));
  ABSL_ASSIGN_OR_RETURN(plan.checkpoint_envelope_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.checkpoint_envelope_size, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(
      plan.checkpoint_authentication_key_id,
      reader->ReadString(kMaximumCapsuleEvidenceKeyIdBytes,
                         "checkpoint authentication key ID"));
  ABSL_ASSIGN_OR_RETURN(plan.checkpoint_step, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.checkpoint_history_token_bytes_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.target_state, ReadRestoreTarget(reader));
  ABSL_ASSIGN_OR_RETURN(plan.prefill, ReadPrefillPlan(reader));
  ABSL_ASSIGN_OR_RETURN(plan.maximum_output_tokens, reader->ReadU32());
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestorePlanV2(plan));
  return plan;
}

void AppendWitness(const SessionContinuationStateWitness& witness,
                   std::string* output) {
  AppendU32(witness.format_version, output);
  AppendHash(witness.witness_id, output);
  AppendIdentity(witness.session_identity, output);
  AppendU8(static_cast<uint8_t>(witness.phase), output);
  AppendI32(witness.current_step, output);
  AppendU8(witness.ran_decode ? 1 : 0, output);
  AppendHash(witness.processed_history_token_bytes_hash, output);
  AppendHash(witness.envelope_hash, output);
  AppendU64(witness.envelope_size, output);
  AppendString(witness.key_id, output);
}

absl::StatusOr<SessionContinuationStateWitness> ReadWitness(
    CanonicalReader* reader) {
  SessionContinuationStateWitness witness;
  ABSL_ASSIGN_OR_RETURN(witness.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(witness.witness_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(witness.session_identity, ReadIdentity(reader));
  uint8_t phase;
  ABSL_ASSIGN_OR_RETURN(phase, reader->ReadU8());
  witness.phase = static_cast<SessionHandoffPhase>(phase);
  ABSL_ASSIGN_OR_RETURN(witness.current_step, reader->ReadI32());
  uint8_t ran_decode;
  ABSL_ASSIGN_OR_RETURN(ran_decode, reader->ReadU8());
  if (ran_decode > 1) {
    return absl::DataLossError(
        "Coverage V2 evidence has a noncanonical ran-decode flag.");
  }
  witness.ran_decode = ran_decode != 0;
  ABSL_ASSIGN_OR_RETURN(witness.processed_history_token_bytes_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(witness.envelope_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(witness.envelope_size, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(
      witness.key_id,
      reader->ReadString(kMaximumCapsuleEvidenceKeyIdBytes,
                         "witness authentication key ID"));
  ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(witness));
  return witness;
}

void AppendReauthenticationEvidence(
    const SessionHandoffReauthenticationEvidence& evidence,
    std::string* output) {
  AppendU32(evidence.format_version, output);
  AppendHash(evidence.evidence_id, output);
  AppendIdentity(evidence.session_identity, output);
  AppendHash(evidence.canonical_continuation_state_hash, output);
  AppendHash(evidence.source_envelope_hash, output);
  AppendU64(evidence.source_envelope_size, output);
  AppendString(evidence.source_key_id, output);
  AppendHash(evidence.destination_envelope_hash, output);
  AppendU64(evidence.destination_envelope_size, output);
  AppendString(evidence.destination_key_id, output);
  AppendHash(evidence.capsule_codec_contract_hash, output);
  AppendHash(evidence.reauthentication_contract_hash, output);
  AppendString(evidence.purpose, output);
}

absl::StatusOr<SessionHandoffReauthenticationEvidence>
ReadReauthenticationEvidence(CanonicalReader* reader) {
  SessionHandoffReauthenticationEvidence evidence;
  ABSL_ASSIGN_OR_RETURN(evidence.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(evidence.evidence_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.session_identity, ReadIdentity(reader));
  ABSL_ASSIGN_OR_RETURN(evidence.canonical_continuation_state_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.source_envelope_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.source_envelope_size, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(
      evidence.source_key_id,
      reader->ReadString(kMaximumCapsuleEvidenceKeyIdBytes,
                         "reauthentication source key ID"));
  ABSL_ASSIGN_OR_RETURN(evidence.destination_envelope_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.destination_envelope_size,
                        reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(
      evidence.destination_key_id,
      reader->ReadString(kMaximumCapsuleEvidenceKeyIdBytes,
                         "reauthentication destination key ID"));
  ABSL_ASSIGN_OR_RETURN(evidence.capsule_codec_contract_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.reauthentication_contract_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(
      evidence.purpose,
      reader->ReadString(
          kMaximumSessionHandoffReauthenticationPurposeBytes,
          "reauthentication purpose"));
  ABSL_RETURN_IF_ERROR(
      ValidateSessionHandoffReauthenticationEvidence(evidence));
  return evidence;
}

absl::Status AppendRestoreEvidence(const CapsuleRestoreEvidenceV2& evidence,
                                   std::string* output) {
  AppendU32(evidence.format_version, output);
  AppendHash(evidence.evidence_id, output);
  ABSL_RETURN_IF_ERROR(AppendRestorePlan(evidence.plan, output));
  AppendWitness(evidence.target_post_import, output);
  return absl::OkStatus();
}

absl::StatusOr<CapsuleRestoreEvidenceV2> ReadRestoreEvidence(
    CanonicalReader* reader) {
  CapsuleRestoreEvidenceV2 evidence;
  ABSL_ASSIGN_OR_RETURN(evidence.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(evidence.evidence_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.plan, ReadRestorePlan(reader));
  ABSL_ASSIGN_OR_RETURN(evidence.target_post_import, ReadWitness(reader));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidenceV2(evidence));
  return evidence;
}

absl::StatusOr<std::string> EncodeCaptureEvidenceBody(
    const CapsuleCaptureEvidenceV2& evidence) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceV2(evidence));
  std::string body;
  AppendU32(evidence.format_version, &body);
  AppendHash(evidence.evidence_id, &body);
  ABSL_RETURN_IF_ERROR(AppendCapturePlan(evidence.plan, &body));
  AppendHash(evidence.checkpoint_id, &body);
  AppendHash(evidence.checkpoint_envelope_hash, &body);
  AppendU64(evidence.checkpoint_envelope_size, &body);
  AppendString(evidence.checkpoint_authentication_key_id, &body);
  AppendHash(evidence.checkpoint_history_token_bytes_hash, &body);
  AppendWitness(evidence.producer_before_export, &body);
  AppendWitness(evidence.producer_after_export, &body);
  AppendWitness(evidence.fresh_import_target, &body);
  AppendU8(evidence.parent_restore_evidence.has_value() ? 1 : 0, &body);
  if (evidence.parent_restore_evidence.has_value()) {
    ABSL_RETURN_IF_ERROR(
        AppendRestoreEvidence(*evidence.parent_restore_evidence, &body));
  }
  return body;
}

absl::StatusOr<CapsuleCaptureEvidenceV2> DecodeCaptureEvidenceBody(
    absl::string_view body) {
  CanonicalReader reader(body);
  CapsuleCaptureEvidenceV2 evidence;
  ABSL_ASSIGN_OR_RETURN(evidence.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(evidence.evidence_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.plan, ReadCapturePlan(&reader));
  ABSL_ASSIGN_OR_RETURN(evidence.checkpoint_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.checkpoint_envelope_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.checkpoint_envelope_size, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(
      evidence.checkpoint_authentication_key_id,
      reader.ReadString(kMaximumCapsuleEvidenceKeyIdBytes,
                        "checkpoint authentication key ID"));
  ABSL_ASSIGN_OR_RETURN(evidence.checkpoint_history_token_bytes_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.producer_before_export,
                        ReadWitness(&reader));
  ABSL_ASSIGN_OR_RETURN(evidence.producer_after_export,
                        ReadWitness(&reader));
  ABSL_ASSIGN_OR_RETURN(evidence.fresh_import_target, ReadWitness(&reader));
  uint8_t has_parent;
  ABSL_ASSIGN_OR_RETURN(has_parent, reader.ReadU8());
  if (has_parent > 1) {
    return absl::DataLossError(
        "Coverage V2 evidence has a noncanonical parent-restore flag.");
  }
  if (has_parent != 0) {
    ABSL_ASSIGN_OR_RETURN(evidence.parent_restore_evidence,
                          ReadRestoreEvidence(&reader));
  }
  if (!reader.empty()) {
    return absl::DataLossError(
        "Authenticated Coverage V2 capture evidence has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceV2(evidence));
  return evidence;
}

absl::Status AppendRestoreEvidenceV3(
    const CapsuleRestoreEvidenceV3& evidence, std::string* output) {
  AppendU32(evidence.format_version, output);
  AppendHash(evidence.evidence_id, output);
  ABSL_RETURN_IF_ERROR(AppendRestorePlan(evidence.plan, output));
  AppendReauthenticationEvidence(
      evidence.durable_to_transient_reauthentication, output);
  AppendWitness(evidence.target_post_import, output);
  return absl::OkStatus();
}

absl::StatusOr<CapsuleRestoreEvidenceV3> ReadRestoreEvidenceV3(
    CanonicalReader* reader) {
  CapsuleRestoreEvidenceV3 evidence;
  ABSL_ASSIGN_OR_RETURN(evidence.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(evidence.evidence_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.plan, ReadRestorePlan(reader));
  ABSL_ASSIGN_OR_RETURN(
      evidence.durable_to_transient_reauthentication,
      ReadReauthenticationEvidence(reader));
  ABSL_ASSIGN_OR_RETURN(evidence.target_post_import, ReadWitness(reader));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidenceV3(evidence));
  return evidence;
}

absl::StatusOr<std::string> EncodeRestoreEvidenceV3Body(
    const CapsuleRestoreEvidenceV3& evidence) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidenceV3(evidence));
  std::string body;
  ABSL_RETURN_IF_ERROR(AppendRestoreEvidenceV3(evidence, &body));
  return body;
}

absl::StatusOr<CapsuleRestoreEvidenceV3> DecodeRestoreEvidenceV3Body(
    absl::string_view body) {
  CanonicalReader reader(body);
  ABSL_ASSIGN_OR_RETURN(CapsuleRestoreEvidenceV3 evidence,
                        ReadRestoreEvidenceV3(&reader));
  if (!reader.empty()) {
    return absl::DataLossError(
        "Authenticated Coverage V3 restore evidence has trailing bytes.");
  }
  return evidence;
}

absl::StatusOr<std::string> EncodeCaptureEvidenceV3Body(
    const CapsuleCaptureEvidenceV3& evidence) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceV3(evidence));
  std::string body;
  AppendU32(evidence.format_version, &body);
  AppendHash(evidence.evidence_id, &body);
  ABSL_RETURN_IF_ERROR(AppendCapturePlan(evidence.plan, &body));
  AppendHash(evidence.checkpoint_id, &body);
  AppendHash(evidence.checkpoint_envelope_hash, &body);
  AppendU64(evidence.checkpoint_envelope_size, &body);
  AppendString(evidence.checkpoint_authentication_key_id, &body);
  AppendHash(evidence.checkpoint_history_token_bytes_hash, &body);
  AppendWitness(evidence.producer_before_export, &body);
  AppendWitness(evidence.producer_after_export, &body);
  AppendWitness(evidence.fresh_import_target, &body);
  AppendReauthenticationEvidence(
      evidence.transient_to_durable_reauthentication, &body);
  AppendU8(evidence.parent_restore_evidence.has_value() ? 1 : 0, &body);
  if (evidence.parent_restore_evidence.has_value()) {
    ABSL_RETURN_IF_ERROR(
        AppendRestoreEvidenceV3(*evidence.parent_restore_evidence, &body));
  }
  return body;
}

absl::StatusOr<CapsuleCaptureEvidenceV3> DecodeCaptureEvidenceV3Body(
    absl::string_view body) {
  CanonicalReader reader(body);
  CapsuleCaptureEvidenceV3 evidence;
  ABSL_ASSIGN_OR_RETURN(evidence.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(evidence.evidence_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.plan, ReadCapturePlan(&reader));
  ABSL_ASSIGN_OR_RETURN(evidence.checkpoint_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.checkpoint_envelope_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.checkpoint_envelope_size,
                        reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(
      evidence.checkpoint_authentication_key_id,
      reader.ReadString(kMaximumCapsuleEvidenceKeyIdBytes,
                        "V3 checkpoint authentication key ID"));
  ABSL_ASSIGN_OR_RETURN(evidence.checkpoint_history_token_bytes_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(evidence.producer_before_export,
                        ReadWitness(&reader));
  ABSL_ASSIGN_OR_RETURN(evidence.producer_after_export,
                        ReadWitness(&reader));
  ABSL_ASSIGN_OR_RETURN(evidence.fresh_import_target, ReadWitness(&reader));
  ABSL_ASSIGN_OR_RETURN(
      evidence.transient_to_durable_reauthentication,
      ReadReauthenticationEvidence(&reader));
  uint8_t has_parent;
  ABSL_ASSIGN_OR_RETURN(has_parent, reader.ReadU8());
  if (has_parent > 1) {
    return absl::DataLossError(
        "Coverage V3 evidence has a noncanonical parent-restore flag.");
  }
  if (has_parent != 0) {
    ABSL_ASSIGN_OR_RETURN(evidence.parent_restore_evidence,
                          ReadRestoreEvidenceV3(&reader));
  }
  if (!reader.empty()) {
    return absl::DataLossError(
        "Authenticated Coverage V3 capture evidence has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceV3(evidence));
  return evidence;
}

bool OperationKeyCollidesWithRestoreEvidenceV3(
    absl::string_view operation_key_id,
    const CapsuleRestoreEvidenceV3& evidence) {
  const SessionHandoffReauthenticationEvidence& reauthentication =
      evidence.durable_to_transient_reauthentication;
  return operation_key_id ==
             evidence.plan.checkpoint_authentication_key_id ||
         operation_key_id == reauthentication.source_key_id ||
         operation_key_id == reauthentication.destination_key_id ||
         operation_key_id == evidence.target_post_import.key_id;
}

absl::Status ValidateOperationAuthenticationBindingV3(
    const CapsuleRestoreEvidenceV3& evidence,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (OperationKeyCollidesWithRestoreEvidenceV3(authentication.key_id,
                                                 evidence)) {
    return absl::InvalidArgumentError(
        "Coverage V3 operation evidence, durable checkpoint, and transient "
        "restore envelopes require distinct authentication key IDs.");
  }
  return absl::OkStatus();
}

absl::Status ValidateOperationAuthenticationBindingV3(
    const CapsuleCaptureEvidenceV3& evidence,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  const SessionHandoffReauthenticationEvidence& reauthentication =
      evidence.transient_to_durable_reauthentication;
  if (authentication.key_id ==
          evidence.plan.checkpoint_authentication_key_id ||
      authentication.key_id == evidence.checkpoint_authentication_key_id ||
      authentication.key_id == reauthentication.source_key_id ||
      authentication.key_id == reauthentication.destination_key_id ||
      authentication.key_id == evidence.producer_before_export.key_id ||
      (evidence.parent_restore_evidence.has_value() &&
       OperationKeyCollidesWithRestoreEvidenceV3(
           authentication.key_id, *evidence.parent_restore_evidence))) {
    return absl::InvalidArgumentError(
        "Coverage V3 operation evidence, durable checkpoints, and transient "
        "capsules require distinct authentication key IDs.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeAuthenticatedV3Envelope(
    absl::string_view body, const std::array<char, 8>& magic,
    uint32_t version, absl::string_view authentication_domain,
    uint64_t maximum_envelope_bytes,
    const FreshWorkerAuthentication& authentication,
    absl::string_view description) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (authentication.key_id.size() >
          (std::numeric_limits<uint32_t>::max)() ||
      kEnvelopeHeaderBytes + kAuthenticationTagBytes >
          maximum_envelope_bytes ||
      authentication.key_id.size() >
          maximum_envelope_bytes - kEnvelopeHeaderBytes -
              kAuthenticationTagBytes ||
      body.size() > maximum_envelope_bytes - kEnvelopeHeaderBytes -
                        kAuthenticationTagBytes -
                        authentication.key_id.size()) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Authenticated ", description,
                     " exceeds its envelope bound."));
  }

  std::string envelope;
  envelope.reserve(static_cast<size_t>(
      kEnvelopeHeaderBytes + authentication.key_id.size() + body.size() +
      kAuthenticationTagBytes));
  envelope.append(magic.data(), magic.size());
  AppendU32(version, &envelope);
  AppendU32(static_cast<uint32_t>(authentication.key_id.size()), &envelope);
  AppendU64(body.size(), &envelope);
  envelope.append(authentication.key_id);
  envelope.append(body.data(), body.size());
  const Hash256 tag = HmacSha256(
      authentication.authentication_key,
      {authentication_domain, envelope});
  AppendHash(tag, &envelope);
  return envelope;
}

absl::StatusOr<absl::string_view> DecodeAuthenticatedV3EnvelopeBody(
    absl::string_view envelope, const std::array<char, 8>& magic,
    uint32_t expected_version, absl::string_view authentication_domain,
    uint64_t maximum_envelope_bytes,
    const FreshWorkerAuthentication& authentication,
    absl::string_view description) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (envelope.size() < kEnvelopeHeaderBytes + kAuthenticationTagBytes ||
      envelope.size() > maximum_envelope_bytes) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Authenticated ", description,
        " envelope is outside its size bounds."));
  }
  if (std::memcmp(envelope.data(), magic.data(), magic.size()) != 0) {
    return absl::DataLossError(
        absl::StrCat("Authenticated ", description, " magic is invalid."));
  }

  CanonicalReader header(envelope.substr(magic.size()));
  ABSL_ASSIGN_OR_RETURN(const uint32_t version, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t key_id_size, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint64_t body_size, header.ReadU64());
  if (version != expected_version) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Authenticated ", description, " version is unsupported."));
  }
  if (key_id_size > kMaximumOperationEvidenceKeyIdBytes ||
      body_size > maximum_envelope_bytes ||
      key_id_size > maximum_envelope_bytes - kEnvelopeHeaderBytes -
                        kAuthenticationTagBytes ||
      body_size > maximum_envelope_bytes - kEnvelopeHeaderBytes -
                      kAuthenticationTagBytes - key_id_size) {
    return absl::DataLossError(
        absl::StrCat("Authenticated ", description,
                     " framing is invalid."));
  }
  const uint64_t authenticated_size =
      kEnvelopeHeaderBytes + key_id_size + body_size;
  if (authenticated_size >
          maximum_envelope_bytes - kAuthenticationTagBytes ||
      envelope.size() != authenticated_size + kAuthenticationTagBytes) {
    return absl::DataLossError(
        absl::StrCat("Authenticated ", description,
                     " length is noncanonical."));
  }

  const absl::string_view encoded_key =
      envelope.substr(kEnvelopeHeaderBytes, key_id_size);
  if (encoded_key != authentication.key_id) {
    return absl::UnauthenticatedError(absl::StrCat(
        description, " authentication key ID does not match."));
  }
  Hash256 encoded_tag;
  std::memcpy(encoded_tag.bytes.data(),
              envelope.data() + static_cast<size_t>(authenticated_size),
              encoded_tag.bytes.size());
  const absl::string_view authenticated_bytes =
      envelope.substr(0, static_cast<size_t>(authenticated_size));
  const Hash256 expected_tag = HmacSha256(
      authentication.authentication_key,
      {authentication_domain, authenticated_bytes});
  if (!ConstantTimeHashEquals(encoded_tag, expected_tag)) {
    return absl::UnauthenticatedError(
        absl::StrCat(description, " envelope authentication failed."));
  }
  const size_t body_offset =
      static_cast<size_t>(kEnvelopeHeaderBytes + key_id_size);
  return envelope.substr(body_offset, static_cast<size_t>(body_size));
}

absl::Status ValidateOperationAuthenticationBinding(
    const CapsuleCaptureEvidenceV2& evidence,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (authentication.key_id ==
          evidence.plan.checkpoint_authentication_key_id ||
      authentication.key_id == evidence.checkpoint_authentication_key_id ||
      (evidence.parent_restore_evidence.has_value() &&
       authentication.key_id == evidence.parent_restore_evidence->plan
                                    .checkpoint_authentication_key_id)) {
    return absl::InvalidArgumentError(
        "Coverage V2 operation evidence and checkpoint envelopes require "
        "distinct authentication key IDs.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::string> EncodeAuthenticatedCapsuleCaptureEvidenceV2(
    const CapsuleCaptureEvidenceV2& evidence,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(
      ValidateOperationAuthenticationBinding(evidence, authentication));
  ABSL_ASSIGN_OR_RETURN(const std::string body,
                        EncodeCaptureEvidenceBody(evidence));
  if (authentication.key_id.size() >
          (std::numeric_limits<uint32_t>::max)() ||
      kEnvelopeHeaderBytes + kAuthenticationTagBytes >
          kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes ||
      authentication.key_id.size() >
          kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes -
              kEnvelopeHeaderBytes - kAuthenticationTagBytes ||
      body.size() >
          kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes -
              kEnvelopeHeaderBytes - kAuthenticationTagBytes -
              authentication.key_id.size()) {
    return absl::ResourceExhaustedError(
        "Authenticated Coverage V2 capture evidence exceeds its envelope "
        "bound.");
  }

  std::string envelope;
  envelope.reserve(static_cast<size_t>(
      kEnvelopeHeaderBytes + authentication.key_id.size() + body.size() +
      kAuthenticationTagBytes));
  envelope.append(kEnvelopeMagic.data(), kEnvelopeMagic.size());
  AppendU32(kAuthenticatedCapsuleCaptureEvidenceV2EnvelopeVersion, &envelope);
  AppendU32(static_cast<uint32_t>(authentication.key_id.size()), &envelope);
  AppendU64(body.size(), &envelope);
  envelope.append(authentication.key_id);
  envelope.append(body);
  const Hash256 tag = HmacSha256(
      authentication.authentication_key, {kAuthenticationDomain, envelope});
  AppendHash(tag, &envelope);
  return envelope;
}

absl::StatusOr<CapsuleCaptureEvidenceV2>
DecodeAuthenticatedCapsuleCaptureEvidenceV2(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (envelope.size() < kEnvelopeHeaderBytes + kAuthenticationTagBytes ||
      envelope.size() >
          kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "Authenticated Coverage V2 capture-evidence envelope is outside its "
        "size bounds.");
  }
  if (std::memcmp(envelope.data(), kEnvelopeMagic.data(),
                  kEnvelopeMagic.size()) != 0) {
    return absl::DataLossError(
        "Authenticated Coverage V2 capture-evidence magic is invalid.");
  }

  CanonicalReader header(envelope.substr(kEnvelopeMagic.size()));
  ABSL_ASSIGN_OR_RETURN(const uint32_t version, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t key_id_size, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint64_t body_size, header.ReadU64());
  if (version != kAuthenticatedCapsuleCaptureEvidenceV2EnvelopeVersion) {
    return absl::FailedPreconditionError(
        "Authenticated Coverage V2 capture-evidence version is unsupported.");
  }
  if (key_id_size > kMaximumOperationEvidenceKeyIdBytes ||
      body_size > kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes ||
      key_id_size >
          kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes -
              kEnvelopeHeaderBytes - kAuthenticationTagBytes ||
      body_size >
          kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes -
              kEnvelopeHeaderBytes - kAuthenticationTagBytes - key_id_size) {
    return absl::DataLossError(
        "Authenticated Coverage V2 capture-evidence framing is invalid.");
  }
  const uint64_t authenticated_size =
      kEnvelopeHeaderBytes + key_id_size + body_size;
  if (authenticated_size >
          kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes -
              kAuthenticationTagBytes ||
      envelope.size() != authenticated_size + kAuthenticationTagBytes) {
    return absl::DataLossError(
        "Authenticated Coverage V2 capture-evidence length is noncanonical.");
  }

  const absl::string_view encoded_key =
      envelope.substr(kEnvelopeHeaderBytes, key_id_size);
  if (encoded_key != authentication.key_id) {
    return absl::UnauthenticatedError(
        "Coverage V2 capture-evidence authentication key ID does not match.");
  }
  Hash256 encoded_tag;
  std::memcpy(encoded_tag.bytes.data(),
              envelope.data() + static_cast<size_t>(authenticated_size),
              encoded_tag.bytes.size());
  const absl::string_view authenticated_bytes =
      envelope.substr(0, static_cast<size_t>(authenticated_size));
  const Hash256 expected_tag = HmacSha256(authentication.authentication_key,
                                           {kAuthenticationDomain,
                                            authenticated_bytes});
  if (!ConstantTimeHashEquals(encoded_tag, expected_tag)) {
    return absl::UnauthenticatedError(
        "Coverage V2 capture-evidence envelope authentication failed.");
  }

  // No evidence body byte is parsed until the complete envelope has passed its
  // HMAC check above.
  const size_t body_offset =
      static_cast<size_t>(kEnvelopeHeaderBytes + key_id_size);
  ABSL_ASSIGN_OR_RETURN(
      CapsuleCaptureEvidenceV2 evidence,
      DecodeCaptureEvidenceBody(
          envelope.substr(body_offset, static_cast<size_t>(body_size))));
  ABSL_RETURN_IF_ERROR(
      ValidateOperationAuthenticationBinding(evidence, authentication));
  ABSL_ASSIGN_OR_RETURN(
      const std::string canonical,
      EncodeAuthenticatedCapsuleCaptureEvidenceV2(evidence, authentication));
  if (canonical != envelope) {
    return absl::DataLossError(
        "Authenticated Coverage V2 capture evidence is not canonically "
        "encoded.");
  }
  return evidence;
}

absl::StatusOr<std::string> EncodeAuthenticatedCapsuleCaptureEvidenceV3(
    const CapsuleCaptureEvidenceV3& evidence,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(
      ValidateOperationAuthenticationBindingV3(evidence, authentication));
  ABSL_ASSIGN_OR_RETURN(const std::string body,
                        EncodeCaptureEvidenceV3Body(evidence));
  return EncodeAuthenticatedV3Envelope(
      body, kCaptureEnvelopeV3Magic,
      kAuthenticatedCapsuleCaptureEvidenceV3EnvelopeVersion,
      kCaptureAuthenticationV3Domain,
      kMaximumAuthenticatedCapsuleCaptureEvidenceV3EnvelopeBytes,
      authentication, "Coverage V3 capture evidence");
}

absl::StatusOr<CapsuleCaptureEvidenceV3>
DecodeAuthenticatedCapsuleCaptureEvidenceV3(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication) {
  ABSL_ASSIGN_OR_RETURN(
      const absl::string_view body,
      DecodeAuthenticatedV3EnvelopeBody(
          envelope, kCaptureEnvelopeV3Magic,
          kAuthenticatedCapsuleCaptureEvidenceV3EnvelopeVersion,
          kCaptureAuthenticationV3Domain,
          kMaximumAuthenticatedCapsuleCaptureEvidenceV3EnvelopeBytes,
          authentication, "Coverage V3 capture evidence"));
  ABSL_ASSIGN_OR_RETURN(CapsuleCaptureEvidenceV3 evidence,
                        DecodeCaptureEvidenceV3Body(body));
  ABSL_RETURN_IF_ERROR(
      ValidateOperationAuthenticationBindingV3(evidence, authentication));
  ABSL_ASSIGN_OR_RETURN(
      const std::string canonical,
      EncodeAuthenticatedCapsuleCaptureEvidenceV3(evidence, authentication));
  if (canonical != envelope) {
    return absl::DataLossError(
        "Authenticated Coverage V3 capture evidence is not canonically "
        "encoded.");
  }
  return evidence;
}

absl::StatusOr<std::string> EncodeAuthenticatedCapsuleRestoreEvidenceV3(
    const CapsuleRestoreEvidenceV3& evidence,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(
      ValidateOperationAuthenticationBindingV3(evidence, authentication));
  ABSL_ASSIGN_OR_RETURN(const std::string body,
                        EncodeRestoreEvidenceV3Body(evidence));
  return EncodeAuthenticatedV3Envelope(
      body, kRestoreEnvelopeV3Magic,
      kAuthenticatedCapsuleRestoreEvidenceV3EnvelopeVersion,
      kRestoreAuthenticationV3Domain,
      kMaximumAuthenticatedCapsuleRestoreEvidenceV3EnvelopeBytes,
      authentication, "Coverage V3 restore evidence");
}

absl::StatusOr<CapsuleRestoreEvidenceV3>
DecodeAuthenticatedCapsuleRestoreEvidenceV3(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication) {
  ABSL_ASSIGN_OR_RETURN(
      const absl::string_view body,
      DecodeAuthenticatedV3EnvelopeBody(
          envelope, kRestoreEnvelopeV3Magic,
          kAuthenticatedCapsuleRestoreEvidenceV3EnvelopeVersion,
          kRestoreAuthenticationV3Domain,
          kMaximumAuthenticatedCapsuleRestoreEvidenceV3EnvelopeBytes,
          authentication, "Coverage V3 restore evidence"));
  ABSL_ASSIGN_OR_RETURN(CapsuleRestoreEvidenceV3 evidence,
                        DecodeRestoreEvidenceV3Body(body));
  ABSL_RETURN_IF_ERROR(
      ValidateOperationAuthenticationBindingV3(evidence, authentication));
  ABSL_ASSIGN_OR_RETURN(
      const std::string canonical,
      EncodeAuthenticatedCapsuleRestoreEvidenceV3(evidence, authentication));
  if (canonical != envelope) {
    return absl::DataLossError(
        "Authenticated Coverage V3 restore evidence is not canonically "
        "encoded.");
  }
  return evidence;
}

}  // namespace litert::lm
