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
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/dpm/exact_decode_evidence.h"
#include "runtime/engine/exact_litert_decode.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/session_handoff_codec_contract.h"
#include "runtime/platform/hash/hmac_sha256.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/proto/sampler_params.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kAdmissionMagic = {'D', 'P', 'M', 'C', 'R', 'A',
                                                  '0', '1'};
constexpr uint32_t kAdmissionEnvelopeVersion = 1;
constexpr uint64_t kAdmissionEnvelopeFixedBytes = 8 + 4 + 4 + 8 + 32;
constexpr uint32_t kMaximumAdmissionKeyIdBytes = 256;
constexpr uint32_t kMaximumCheckpointKeyIdBytes = 1024;
constexpr absl::string_view kQualificationSpecDomain =
    "LITERT_LMX_CAPSULE_RESTORE_QUALIFICATION_SPEC_SHA256_V1";
constexpr absl::string_view kLookupKeyDomain =
    "LITERT_LMX_CAPSULE_RESTORE_ADMISSION_LOOKUP_SHA256_V1";
constexpr absl::string_view kAdmissionRecordDomain =
    "LITERT_LMX_CAPSULE_RESTORE_ADMISSION_RECORD_SHA256_V1";
constexpr absl::string_view kAdmissionMacDomain =
    "LITERT_LMX_CAPSULE_RESTORE_ADMISSION_HMAC_SHA256_V1";

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

void AppendLogitFrame(const FreshWorkerLogitFrameEvidence& frame,
                      std::string* output) {
  AppendU32(static_cast<uint32_t>(frame.element_type), output);
  AppendU32(frame.element_byte_width, output);
  AppendU32(frame.batch_size, output);
  AppendU32(frame.sequence_size, output);
  AppendU32(frame.vocabulary_size, output);
  AppendU64(frame.byte_count, output);
  AppendHash(frame.sha256, output);
}

void AppendContinuationEvidence(
    const CapsuleRestoreContinuationEvidence& evidence,
    std::string* output) {
  AppendU64(evidence.visible_output.size(), output);
  output->append(evidence.visible_output);
  AppendU64(evidence.token_bytes.size(), output);
  output->append(evidence.token_bytes);
  AppendU32(static_cast<uint32_t>(evidence.logit_frames.size()), output);
  for (const FreshWorkerLogitFrameEvidence& frame : evidence.logit_frames) {
    AppendLogitFrame(frame, output);
  }
  AppendHash(evidence.output_evidence_hash, output);
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

absl::Status ValidateDerivedProfile(
    const ExactLiteRtProfile& derived_profile) {
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

absl::Status ValidateResolvedSessionConfig(const SessionConfig& config,
                                           const Engine& engine) {
  if (config.GetMemoryStrategy() != SessionConfig::MemoryStrategy::kStateful) {
    return absl::FailedPreconditionError(
        "CapsuleRestore qualification requires an accumulating stateful "
        "session.");
  }
  if (config.UseExternalSampler() ||
      config.GetSamplerBackend() != Backend::CPU ||
      config.GetSamplerParams().type() !=
          proto::SamplerParameters::GREEDY ||
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

absl::Status ValidateChunk(const CapsuleRestorePrefillChunk& chunk) {
  switch (chunk.encoding) {
    case CapsuleRestorePrefillChunk::Encoding::kUtf8Text:
      if (chunk.utf8_text.empty() || !chunk.token_ids.empty() ||
          !IsValidUtf8(chunk.utf8_text)) {
        return absl::InvalidArgumentError(
            "A CapsuleRestore text chunk is empty, mixed, or not UTF-8.");
      }
      return absl::OkStatus();
    case CapsuleRestorePrefillChunk::Encoding::kExactTokenIds:
      if (!chunk.utf8_text.empty() || chunk.token_ids.empty()) {
        return absl::InvalidArgumentError(
            "A CapsuleRestore token chunk is empty or mixed with text.");
      }
      for (int32_t token_id : chunk.token_ids) {
        if (token_id < 0) {
          return absl::InvalidArgumentError(
              "A CapsuleRestore exact-token chunk contains a negative ID.");
        }
      }
      return absl::OkStatus();
    default:
      return absl::InvalidArgumentError(
          "A CapsuleRestore prefill chunk has an unknown encoding.");
  }
}

void AppendChunks(const std::vector<CapsuleRestorePrefillChunk>& chunks,
                  std::string* output) {
  AppendU32(static_cast<uint32_t>(chunks.size()), output);
  for (const CapsuleRestorePrefillChunk& chunk : chunks) {
    AppendU32(static_cast<uint32_t>(chunk.encoding), output);
    switch (chunk.encoding) {
      case CapsuleRestorePrefillChunk::Encoding::kUtf8Text:
        AppendU64(chunk.utf8_text.size(), output);
        output->append(chunk.utf8_text);
        break;
      case CapsuleRestorePrefillChunk::Encoding::kExactTokenIds:
        AppendU32(static_cast<uint32_t>(chunk.token_ids.size()), output);
        for (int32_t token_id : chunk.token_ids) {
          AppendI32(token_id, output);
        }
        break;
    }
  }
}

absl::Status ValidateContinuationEvidence(
    const CapsuleRestoreContinuationEvidence& evidence) {
  if (evidence.visible_output.size() >
          kMaximumFreshWorkerCanonicalOutputBytes ||
      !IsValidUtf8(evidence.visible_output)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore continuation visible bytes are oversized or not "
        "UTF-8.");
  }
  if (evidence.token_bytes.size() > kMaximumFreshWorkerTokenBytes ||
      evidence.logit_frames.empty() ||
      evidence.logit_frames.size() > kMaximumFreshWorkerLogitFrames) {
    return absl::InvalidArgumentError(
        "CapsuleRestore continuation token or logits evidence is outside "
        "its bounds.");
  }
  ABSL_ASSIGN_OR_RETURN(const std::vector<int32_t> token_ids,
                        DecodeFreshWorkerTokenIds(evidence.token_bytes));
  if (token_ids.empty() || token_ids.size() != evidence.logit_frames.size()) {
    return absl::InvalidArgumentError(
        "CapsuleRestore continuation requires one ordered logits frame per "
        "DPMTOK01 token.");
  }
  const FreshWorkerLogitFrameEvidence& first = evidence.logit_frames.front();
  for (size_t index = 0; index < evidence.logit_frames.size(); ++index) {
    const FreshWorkerLogitFrameEvidence& frame = evidence.logit_frames[index];
    ABSL_RETURN_IF_ERROR(ValidateFreshWorkerLogitFrameEvidence(frame));
    if (frame.batch_size != 1 || frame.sequence_size != 1 ||
        frame.element_type != first.element_type ||
        frame.element_byte_width != first.element_byte_width ||
        frame.batch_size != first.batch_size ||
        frame.sequence_size != first.sequence_size ||
        frame.vocabulary_size != first.vocabulary_size ||
        frame.byte_count != first.byte_count ||
        static_cast<uint32_t>(token_ids[index]) >= frame.vocabulary_size) {
      return absl::FailedPreconditionError(
          "CapsuleRestore continuation logits frames or sampled tokens do "
          "not share one exact packed contract.");
    }
  }
  const Hash256 expected_hash = ComputeFreshWorkerOutputEvidenceHash(
      evidence.visible_output, evidence.token_bytes, evidence.logit_frames);
  if (IsZeroHash(evidence.output_evidence_hash) ||
      evidence.output_evidence_hash != expected_hash) {
    return absl::DataLossError(
        "CapsuleRestore continuation output-evidence hash is inconsistent.");
  }
  return absl::OkStatus();
}

absl::Status ValidateContinuationForProfile(
    const CapsuleRestoreContinuationEvidence& evidence,
    const ExactLiteRtProfile& profile, uint32_t maximum_output_tokens) {
  ABSL_RETURN_IF_ERROR(ValidateContinuationEvidence(evidence));
  ABSL_ASSIGN_OR_RETURN(const std::vector<int32_t> token_ids,
                        DecodeFreshWorkerTokenIds(evidence.token_bytes));
  if (maximum_output_tokens == 0 ||
      token_ids.size() > maximum_output_tokens) {
    return absl::FailedPreconditionError(
        "CapsuleRestore continuation exceeds its qualified generation "
        "bound.");
  }
  FreshWorkerLogitElementType expected_type;
  uint32_t expected_width = 0;
  switch (profile.logits_frame.element_type) {
    case ExactLiteRtLogitsElementType::kFloat16:
      expected_type = FreshWorkerLogitElementType::kFloat16;
      expected_width = 2;
      break;
    case ExactLiteRtLogitsElementType::kFloat32:
      expected_type = FreshWorkerLogitElementType::kFloat32;
      expected_width = 4;
      break;
    default:
      return absl::FailedPreconditionError(
          "The runtime-derived profile has an unsupported logits type.");
  }
  for (const FreshWorkerLogitFrameEvidence& frame : evidence.logit_frames) {
    if (frame.element_type != expected_type ||
        frame.element_byte_width != expected_width ||
        frame.batch_size != profile.logits_frame.batch_size ||
        frame.sequence_size != profile.logits_frame.sequence_size ||
        frame.vocabulary_size != profile.logits_frame.vocabulary_size ||
        frame.byte_count != profile.logits_frame.byte_count) {
      return absl::FailedPreconditionError(
          "CapsuleRestore continuation logits do not match the current "
          "Engine-derived profile.");
    }
  }
  return absl::OkStatus();
}

std::string EncodeRecordFields(const CapsuleRestoreAdmissionRecord& record) {
  std::string body;
  AppendU32(record.format_version, &body);
  AppendCapability(record.capability, &body);
  AppendHash(record.qualification_spec_hash, &body);
  AppendU64(record.checkpoint_step, &body);
  AppendU64(record.checkpoint_history_token_bytes.size(), &body);
  body.append(record.checkpoint_history_token_bytes);
  AppendHash(record.checkpoint_envelope_hash, &body);
  AppendU64(record.checkpoint_envelope_size, &body);
  AppendU32(
      static_cast<uint32_t>(record.checkpoint_authentication_key_id.size()),
      &body);
  body.append(record.checkpoint_authentication_key_id);
  AppendContinuationEvidence(record.live_continuation, &body);
  AppendContinuationEvidence(record.restored_continuation, &body);
  AppendU64(static_cast<uint64_t>(record.qualified_unix_micros), &body);
  AppendU32(static_cast<uint32_t>(record.record_authentication_key_id.size()),
            &body);
  body.append(record.record_authentication_key_id);
  return body;
}

absl::Status ValidateRecordFields(const CapsuleRestoreAdmissionRecord& record,
                                  bool validate_record_id) {
  if (record.format_version != CapsuleRestoreAdmissionRecord::kFormatVersion) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission record version is unsupported.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffCapability(record.capability));
  if (IsZeroHash(record.qualification_spec_hash) ||
      record.checkpoint_step == 0 ||
      record.checkpoint_step >
          static_cast<uint64_t>((std::numeric_limits<int>::max)()) ||
      record.checkpoint_history_token_bytes.size() >
          kMaximumFreshWorkerTokenBytes ||
      IsZeroHash(record.checkpoint_envelope_hash) ||
      record.checkpoint_envelope_size == 0 ||
      record.qualified_unix_micros <= 0) {
    return absl::InvalidArgumentError(
        "CapsuleRestore admission checkpoint or timestamp metadata is "
        "incomplete.");
  }
  ABSL_RETURN_IF_ERROR(ValidatePublicKeyId(
      record.checkpoint_authentication_key_id, kMaximumCheckpointKeyIdBytes,
      "CapsuleRestore checkpoint authentication"));
  ABSL_RETURN_IF_ERROR(ValidatePublicKeyId(
      record.record_authentication_key_id, kMaximumAdmissionKeyIdBytes,
      "CapsuleRestore admission-record authentication"));
  if (record.checkpoint_authentication_key_id ==
      record.record_authentication_key_id) {
    return absl::FailedPreconditionError(
        "CapsuleRestore checkpoint and admission record must use distinct "
        "authentication key IDs.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<int32_t> checkpoint_history,
      DecodeFreshWorkerTokenIds(record.checkpoint_history_token_bytes));
  if (checkpoint_history.empty() ||
      checkpoint_history.size() != record.checkpoint_step) {
    return absl::DataLossError(
        "CapsuleRestore checkpoint step does not match its full DPMTOK01 "
        "history.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateContinuationEvidence(record.live_continuation));
  ABSL_RETURN_IF_ERROR(
      ValidateContinuationEvidence(record.restored_continuation));
  if (record.live_continuation != record.restored_continuation) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission continuations are not byte-identical.");
  }
  // Bound the complete encoded object before record-ID computation allocates
  // its canonical byte string. The fixed allowance deliberately exceeds all
  // scalar/capability framing; each frame uses 60 bytes on wire and is rounded
  // up here so this check remains conservative if framing grows within v1.
  uint64_t encoded_upper_bound = 4096;
  const auto add_bytes = [&](uint64_t bytes) -> absl::Status {
    if (bytes > kMaximumCapsuleRestoreAdmissionEnvelopeBytes -
                    encoded_upper_bound) {
      return absl::ResourceExhaustedError(
          "CapsuleRestore admission record exceeds its storage limit.");
    }
    encoded_upper_bound += bytes;
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(
      add_bytes(record.checkpoint_history_token_bytes.size()));
  ABSL_RETURN_IF_ERROR(
      add_bytes(record.checkpoint_authentication_key_id.size()));
  ABSL_RETURN_IF_ERROR(
      add_bytes(record.live_continuation.visible_output.size()));
  ABSL_RETURN_IF_ERROR(add_bytes(record.live_continuation.token_bytes.size()));
  ABSL_RETURN_IF_ERROR(add_bytes(
      static_cast<uint64_t>(record.live_continuation.logit_frames.size()) *
      64));
  ABSL_RETURN_IF_ERROR(
      add_bytes(record.restored_continuation.visible_output.size()));
  ABSL_RETURN_IF_ERROR(
      add_bytes(record.restored_continuation.token_bytes.size()));
  ABSL_RETURN_IF_ERROR(add_bytes(
      static_cast<uint64_t>(record.restored_continuation.logit_frames.size()) *
      64));
  // The public record key ID occurs once in the body and once in the envelope.
  ABSL_RETURN_IF_ERROR(
      add_bytes(record.record_authentication_key_id.size() * uint64_t{2}));
  if (validate_record_id) {
    if (IsZeroHash(record.record_id)) {
      return absl::DataLossError(
          "CapsuleRestore admission record ID is absent.");
    }
    ABSL_ASSIGN_OR_RETURN(const Hash256 expected_id,
                          ComputeCapsuleRestoreAdmissionRecordId(record));
    if (record.record_id != expected_id) {
      return absl::DataLossError(
          "CapsuleRestore admission record ID is not canonical.");
    }
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
    return absl::DataLossError(
        "Truncated CapsuleRestore admission record.");
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

absl::StatusOr<FreshWorkerLogitFrameEvidence> ReadLogitFrame(
    Reader* reader) {
  FreshWorkerLogitFrameEvidence frame;
  ABSL_ASSIGN_OR_RETURN(const uint32_t element_type, reader->ReadU32());
  frame.element_type =
      static_cast<FreshWorkerLogitElementType>(element_type);
  ABSL_ASSIGN_OR_RETURN(frame.element_byte_width, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(frame.batch_size, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(frame.sequence_size, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(frame.vocabulary_size, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(frame.byte_count, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(frame.sha256, reader->ReadHash());
  return frame;
}

absl::StatusOr<CapsuleRestoreContinuationEvidence>
ReadContinuationEvidence(Reader* reader) {
  CapsuleRestoreContinuationEvidence evidence;
  ABSL_ASSIGN_OR_RETURN(const uint64_t visible_size, reader->ReadU64());
  if (visible_size > kMaximumFreshWorkerCanonicalOutputBytes) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore continuation visible bytes exceed their limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view visible,
                        reader->ReadBytes(visible_size));
  evidence.visible_output.assign(visible.data(), visible.size());

  ABSL_ASSIGN_OR_RETURN(const uint64_t token_size, reader->ReadU64());
  if (token_size > kMaximumFreshWorkerTokenBytes) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore continuation token bytes exceed their limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view token_bytes,
                        reader->ReadBytes(token_size));
  evidence.token_bytes.assign(token_bytes.data(), token_bytes.size());

  ABSL_ASSIGN_OR_RETURN(const uint32_t frame_count, reader->ReadU32());
  if (frame_count > kMaximumFreshWorkerLogitFrames) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore continuation logits exceed their limit.");
  }
  evidence.logit_frames.reserve(frame_count);
  for (uint32_t index = 0; index < frame_count; ++index) {
    ABSL_ASSIGN_OR_RETURN(FreshWorkerLogitFrameEvidence frame,
                          ReadLogitFrame(reader));
    evidence.logit_frames.push_back(frame);
  }
  ABSL_ASSIGN_OR_RETURN(evidence.output_evidence_hash, reader->ReadHash());
  ABSL_RETURN_IF_ERROR(ValidateContinuationEvidence(evidence));
  return evidence;
}

absl::StatusOr<CapsuleRestoreAdmissionRecord> DecodeRecordFields(
    absl::string_view body, const Hash256& record_id,
    absl::string_view envelope_key_id) {
  Reader reader(body);
  CapsuleRestoreAdmissionRecord record;
  record.record_id = record_id;
  ABSL_ASSIGN_OR_RETURN(record.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(record.capability, ReadCapability(&reader));
  ABSL_ASSIGN_OR_RETURN(record.qualification_spec_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(record.checkpoint_step, reader.ReadU64());

  ABSL_ASSIGN_OR_RETURN(const uint64_t history_size, reader.ReadU64());
  if (history_size > kMaximumFreshWorkerTokenBytes) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore checkpoint history exceeds its limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view history,
                        reader.ReadBytes(history_size));
  record.checkpoint_history_token_bytes.assign(history.data(), history.size());
  ABSL_ASSIGN_OR_RETURN(record.checkpoint_envelope_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(record.checkpoint_envelope_size, reader.ReadU64());

  ABSL_ASSIGN_OR_RETURN(const uint32_t checkpoint_key_size, reader.ReadU32());
  if (checkpoint_key_size > kMaximumCheckpointKeyIdBytes) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore checkpoint key ID exceeds its limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view checkpoint_key,
                        reader.ReadBytes(checkpoint_key_size));
  record.checkpoint_authentication_key_id.assign(checkpoint_key.data(),
                                                 checkpoint_key.size());
  ABSL_ASSIGN_OR_RETURN(record.live_continuation,
                        ReadContinuationEvidence(&reader));
  ABSL_ASSIGN_OR_RETURN(record.restored_continuation,
                        ReadContinuationEvidence(&reader));

  ABSL_ASSIGN_OR_RETURN(const uint64_t encoded_time, reader.ReadU64());
  if (encoded_time >
      static_cast<uint64_t>((std::numeric_limits<int64_t>::max)())) {
    return absl::DataLossError(
        "CapsuleRestore admission timestamp is invalid.");
  }
  record.qualified_unix_micros = static_cast<int64_t>(encoded_time);
  ABSL_ASSIGN_OR_RETURN(const uint32_t record_key_size, reader.ReadU32());
  if (record_key_size > kMaximumAdmissionKeyIdBytes) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore admission key ID exceeds its limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view record_key,
                        reader.ReadBytes(record_key_size));
  if (record_key != envelope_key_id) {
    return absl::UnauthenticatedError(
        "CapsuleRestore admission key ID binding does not match.");
  }
  record.record_authentication_key_id.assign(record_key.data(),
                                             record_key.size());
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "CapsuleRestore admission record has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAdmissionRecord(record));
  return record;
}

struct ResolvedQualification {
  SessionConfig config = SessionConfig::CreateDefault();
  ExactLiteRtProfile profile;
  SessionHandoffCapability capability;
  SessionHandoffIdentity identity;
  Hash256 specification_hash;
};

absl::StatusOr<ResolvedQualification> ResolveQualification(
    Engine* engine, const CapsuleRestoreQualificationSpec& spec) {
  if (engine == nullptr) {
    return absl::FailedPreconditionError(
        "CapsuleRestore qualification has no authoritative loaded Engine.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreQualificationSpec(spec));
  ResolvedQualification resolved;
  resolved.config = spec.session_config;
  ABSL_RETURN_IF_ERROR(
      resolved.config.MaybeUpdateAndValidate(engine->GetEngineSettings()));
  ABSL_RETURN_IF_ERROR(ValidateResolvedSessionConfig(resolved.config, *engine));
  ABSL_ASSIGN_OR_RETURN(
      resolved.profile,
      engine->ResolveExactLiteRtProfile(resolved.config,
                                        spec.exact_profile_assertion));
  ABSL_ASSIGN_OR_RETURN(
      resolved.capability,
      engine->ResolveSessionHandoffCapability(resolved.config,
                                               spec.capability_assertion));
  ABSL_ASSIGN_OR_RETURN(
      resolved.identity,
      engine->ResolveSessionHandoffIdentity(resolved.config));
  ABSL_RETURN_IF_ERROR(ValidateProfileCapabilityAgreement(
      resolved.profile, resolved.capability));
  if (resolved.identity != resolved.profile.session_identity ||
      resolved.identity != resolved.capability.session_identity) {
    return absl::FailedPreconditionError(
        "Engine-derived handoff identity disagrees with the exact profile "
        "or CapsuleRestore capability.");
  }
  ABSL_ASSIGN_OR_RETURN(
      resolved.specification_hash,
      ComputeCapsuleRestoreQualificationSpecHash(
          resolved.profile, resolved.capability, spec));
  return resolved;
}

absl::Status ValidateSessionIdentity(const Engine::Session& session,
                                     const SessionHandoffIdentity& expected) {
  ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity actual,
                        session.GetSessionHandoffIdentity());
  if (actual != expected) {
    return absl::FailedPreconditionError(
        "CapsuleRestore session identity differs from the loaded Engine.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::pair<int, std::vector<int>>> InspectSingleHistory(
    const Engine::Session& session) {
  ABSL_ASSIGN_OR_RETURN(const int step, session.GetCurrentStep());
  ABSL_ASSIGN_OR_RETURN(const std::vector<std::vector<int>> histories,
                        session.GetExactProcessedTokenHistory());
  if (step < 0 || histories.size() != 1 ||
      histories.front().size() != static_cast<size_t>(step)) {
    return absl::DataLossError(
        "CapsuleRestore session step and single exact token history differ.");
  }
  return std::make_pair(step, histories.front());
}

absl::Status RequireFreshSession(const Engine::Session& session,
                                 const SessionHandoffIdentity& identity) {
  ABSL_RETURN_IF_ERROR(ValidateSessionIdentity(session, identity));
  ABSL_ASSIGN_OR_RETURN(auto observation, InspectSingleHistory(session));
  if (observation.first != 0 || !observation.second.empty()) {
    return absl::FailedPreconditionError(
        "CapsuleRestore qualification requires a step-zero session with one "
        "empty exact token history.");
  }
  return absl::OkStatus();
}

absl::Status PrefillChunks(
    const std::vector<CapsuleRestorePrefillChunk>& chunks,
    int vocabulary_size, Engine::Session* session) {
  if (session == nullptr || chunks.empty() || vocabulary_size <= 0) {
    return absl::InvalidArgumentError(
        "CapsuleRestore prefill requires a session, chunks, and vocabulary.");
  }
  for (const CapsuleRestorePrefillChunk& chunk : chunks) {
    ABSL_RETURN_IF_ERROR(ValidateChunk(chunk));
    std::vector<InputData> contents;
    switch (chunk.encoding) {
      case CapsuleRestorePrefillChunk::Encoding::kUtf8Text:
        contents.emplace_back(InputText(std::string(chunk.utf8_text)));
        break;
      case CapsuleRestorePrefillChunk::Encoding::kExactTokenIds: {
        std::vector<int> exact_ids;
        exact_ids.reserve(chunk.token_ids.size());
        for (int32_t token_id : chunk.token_ids) {
          if (token_id < 0 || token_id >= vocabulary_size) {
            return absl::InvalidArgumentError(
                "CapsuleRestore prefill contains a token outside the loaded "
                "vocabulary.");
          }
          exact_ids.push_back(static_cast<int>(token_id));
        }
        ABSL_ASSIGN_OR_RETURN(
            auto token_buffer,
            support::Tokenizer::TokenIdsToTensorBuffer(exact_ids));
        contents.emplace_back(InputText(std::move(token_buffer)));
        break;
      }
      default:
        return absl::InvalidArgumentError(
            "CapsuleRestore prefill contains an unknown chunk encoding.");
    }
    // One call per exact chunk is part of the qualification contract.
    ABSL_RETURN_IF_ERROR(session->RunPrefill(contents));
  }
  return absl::OkStatus();
}

absl::StatusOr<CapsuleRestoreContinuationEvidence> RunCanonicalExactDecode(
    Engine::Session* session, const ExactLiteRtProfile& profile,
    uint32_t maximum_output_tokens) {
  if (session == nullptr) {
    return absl::InvalidArgumentError(
        "CapsuleRestore exact decode has no session.");
  }
  ABSL_ASSIGN_OR_RETURN(
      ExactLiteRtDecodeResult decoded,
      session->RunExactDecode(static_cast<int>(maximum_output_tokens)));
  ABSL_ASSIGN_OR_RETURN(
      CanonicalExactDecodeEvidence canonical,
      CanonicalizeExactLiteRtDecodeEvidence(decoded, profile,
                                             maximum_output_tokens));
  CapsuleRestoreContinuationEvidence evidence;
  evidence.visible_output = std::move(canonical.visible_output);
  evidence.token_bytes = std::move(canonical.token_bytes);
  evidence.logit_frames = std::move(canonical.logit_frames);
  evidence.output_evidence_hash = ComputeFreshWorkerOutputEvidenceHash(
      evidence.visible_output, evidence.token_bytes, evidence.logit_frames);
  ABSL_RETURN_IF_ERROR(
      ValidateContinuationForProfile(evidence, profile,
                                     maximum_output_tokens));
  return evidence;
}

absl::Status ValidateLrtSess1Envelope(absl::string_view envelope) {
  if (envelope.size() < kSessionHandoffEnvelopeMagic.size() + 4 ||
      std::memcmp(envelope.data(), kSessionHandoffEnvelopeMagic.data(),
                  kSessionHandoffEnvelopeMagic.size()) != 0) {
    return absl::DataLossError(
        "Session export did not produce an LRTSESS1 envelope.");
  }
  uint32_t version = 0;
  for (size_t index = 0; index < 4; ++index) {
    version = (version << 8) |
              static_cast<uint8_t>(
                  envelope[kSessionHandoffEnvelopeMagic.size() + index]);
  }
  if (version != kSessionHandoffEnvelopeVersion) {
    return absl::FailedPreconditionError(
        "Session export used an unsupported LRTSESS1 version.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCheckpointAuthentication(
    const SessionHandoffOptions& checkpoint_authentication,
    const FreshWorkerAuthentication& record_authentication) {
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerAuthentication(record_authentication));
  ABSL_RETURN_IF_ERROR(ValidatePublicKeyId(
      checkpoint_authentication.key_id, kMaximumCheckpointKeyIdBytes,
      "CapsuleRestore checkpoint authentication"));
  if (checkpoint_authentication.authentication_key.size() < 32) {
    return absl::InvalidArgumentError(
        "CapsuleRestore checkpoint authentication key must contain at least "
        "32 bytes.");
  }
  if (checkpoint_authentication.key_id == record_authentication.key_id) {
    return absl::FailedPreconditionError(
        "CapsuleRestore checkpoint and admission record require distinct "
        "authentication key IDs.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateCapsuleRestoreQualificationSpec(
    const CapsuleRestoreQualificationSpec& spec) {
  if (spec.checkpoint_prefix_chunks.empty() || spec.delta_chunks.empty()) {
    return absl::InvalidArgumentError(
        "CapsuleRestore qualification requires nonempty checkpoint-prefix "
        "and delta chunk sequences.");
  }
  const uint64_t chunk_count =
      static_cast<uint64_t>(spec.checkpoint_prefix_chunks.size()) +
      static_cast<uint64_t>(spec.delta_chunks.size());
  if (chunk_count > kMaximumCapsuleRestorePrefillChunks) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore qualification has too many prefill chunks.");
  }
  if (spec.checkpoint_output_tokens == 0 ||
      spec.checkpoint_output_tokens > kMaximumFreshWorkerLogitFrames ||
      spec.continuation_output_tokens == 0 ||
      spec.continuation_output_tokens > kMaximumFreshWorkerLogitFrames ||
      spec.checkpoint_output_tokens >
          static_cast<uint32_t>((std::numeric_limits<int>::max)()) ||
      spec.continuation_output_tokens >
          static_cast<uint32_t>((std::numeric_limits<int>::max)())) {
    return absl::InvalidArgumentError(
        "CapsuleRestore decode limits are outside the exact-decode range.");
  }

  uint64_t text_bytes = 0;
  uint64_t token_ids = 0;
  const auto inspect_chunks = [&](const auto& chunks) -> absl::Status {
    for (const CapsuleRestorePrefillChunk& chunk : chunks) {
      ABSL_RETURN_IF_ERROR(ValidateChunk(chunk));
      if (chunk.utf8_text.size() >
          kMaximumCapsuleRestorePrefillTextBytes - text_bytes) {
        return absl::ResourceExhaustedError(
            "CapsuleRestore qualification text exceeds its limit.");
      }
      text_bytes += chunk.utf8_text.size();
      if (chunk.token_ids.size() >
          kMaximumCapsuleRestorePrefillTokenIds - token_ids) {
        return absl::ResourceExhaustedError(
            "CapsuleRestore qualification token IDs exceed their limit.");
      }
      token_ids += chunk.token_ids.size();
    }
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(inspect_chunks(spec.checkpoint_prefix_chunks));
  return inspect_chunks(spec.delta_chunks);
}

absl::StatusOr<Hash256> ComputeCapsuleRestoreQualificationSpecHash(
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const CapsuleRestoreQualificationSpec& spec) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreQualificationSpec(spec));
  ABSL_RETURN_IF_ERROR(ValidateProfileCapabilityAgreement(
      runtime_derived_profile, runtime_derived_capability));
  std::string canonical;
  AppendU32(CapsuleRestoreAdmissionRecord::kFormatVersion, &canonical);
  AppendProfile(runtime_derived_profile, &canonical);
  AppendCapability(runtime_derived_capability, &canonical);
  AppendChunks(spec.checkpoint_prefix_chunks, &canonical);
  AppendU32(spec.checkpoint_output_tokens, &canonical);
  AppendChunks(spec.delta_chunks, &canonical);
  AppendU32(spec.continuation_output_tokens, &canonical);
  Sha256Hasher hasher;
  hasher.Update(kQualificationSpecDomain);
  hasher.Update(canonical);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "CapsuleRestore qualification produced a zero specification hash.");
  }
  return result;
}

absl::StatusOr<Hash256> ComputeCapsuleRestoreAdmissionLookupKey(
    const Hash256& capability_id, const Hash256& qualification_spec_hash) {
  if (IsZeroHash(capability_id) || IsZeroHash(qualification_spec_hash)) {
    return absl::InvalidArgumentError(
        "CapsuleRestore admission lookup requires nonzero capability and "
        "specification hashes.");
  }
  Sha256Hasher hasher;
  hasher.Update(kLookupKeyDomain);
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(capability_id.bytes.data()),
      capability_id.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(qualification_spec_hash.bytes.data()),
      qualification_spec_hash.bytes.size()));
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "CapsuleRestore admission produced a zero lookup key.");
  }
  return result;
}

absl::StatusOr<Hash256> ComputeCapsuleRestoreAdmissionRecordId(
    const CapsuleRestoreAdmissionRecord& record) {
  ABSL_RETURN_IF_ERROR(ValidateRecordFields(record, false));
  const std::string fields = EncodeRecordFields(record);
  Sha256Hasher hasher;
  hasher.Update(kAdmissionRecordDomain);
  hasher.Update(fields);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "CapsuleRestore admission produced a zero record ID.");
  }
  return result;
}

absl::Status ValidateCapsuleRestoreAdmissionRecord(
    const CapsuleRestoreAdmissionRecord& record) {
  return ValidateRecordFields(record, true);
}

absl::Status ValidateCapsuleRestoreAdmissionRecordForRuntime(
    const CapsuleRestoreAdmissionRecord& record,
    const ExactLiteRtProfile& runtime_derived_profile,
    const SessionHandoffCapability& runtime_derived_capability,
    const CapsuleRestoreQualificationSpec& spec) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAdmissionRecord(record));
  ABSL_RETURN_IF_ERROR(ValidateProfileCapabilityAgreement(
      runtime_derived_profile, runtime_derived_capability));
  if (record.capability != runtime_derived_capability) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission capability differs from the current "
        "Engine-derived capability.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_specification_hash,
      ComputeCapsuleRestoreQualificationSpecHash(
          runtime_derived_profile, runtime_derived_capability, spec));
  if (record.qualification_spec_hash != expected_specification_hash) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission was produced for a different canonical "
        "qualification specification.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<int32_t> checkpoint_history,
      DecodeFreshWorkerTokenIds(record.checkpoint_history_token_bytes));
  for (int32_t token_id : checkpoint_history) {
    if (static_cast<uint32_t>(token_id) >=
        runtime_derived_profile.logits_frame.vocabulary_size) {
      return absl::FailedPreconditionError(
          "CapsuleRestore checkpoint history contains a token outside the "
          "current vocabulary.");
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateContinuationForProfile(
      record.live_continuation, runtime_derived_profile,
      spec.continuation_output_tokens));
  return ValidateContinuationForProfile(
      record.restored_continuation, runtime_derived_profile,
      spec.continuation_output_tokens);
}

absl::StatusOr<std::string> EncodeCapsuleRestoreAdmissionRecord(
    const CapsuleRestoreAdmissionRecord& record,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAdmissionRecord(record));
  if (record.record_authentication_key_id != authentication.key_id) {
    return absl::InvalidArgumentError(
        "CapsuleRestore admission record key ID does not match its "
        "authentication input.");
  }
  std::string body;
  AppendHash(record.record_id, &body);
  body.append(EncodeRecordFields(record));
  if (authentication.key_id.size() >
          kMaximumCapsuleRestoreAdmissionEnvelopeBytes -
              kAdmissionEnvelopeFixedBytes ||
      body.size() > kMaximumCapsuleRestoreAdmissionEnvelopeBytes -
                        kAdmissionEnvelopeFixedBytes -
                        authentication.key_id.size()) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore admission record exceeds its storage limit.");
  }
  std::string envelope;
  envelope.reserve(kAdmissionEnvelopeFixedBytes + authentication.key_id.size() +
                   body.size());
  envelope.append(kAdmissionMagic.data(), kAdmissionMagic.size());
  AppendU32(kAdmissionEnvelopeVersion, &envelope);
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
        "CapsuleRestore admission envelope is outside its size bounds.");
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
  if (version != kAdmissionEnvelopeVersion) {
    return absl::FailedPreconditionError(
        "CapsuleRestore admission envelope version is unsupported.");
  }
  if (key_id_size > kMaximumAdmissionKeyIdBytes ||
      body_size < 32 ||
      body_size > kMaximumCapsuleRestoreAdmissionEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "CapsuleRestore admission envelope declares an oversized field.");
  }
  const uint64_t bytes_before_mac =
      uint64_t{8} + 4 + 4 + 8 + key_id_size + body_size;
  if (bytes_before_mac >
          kMaximumCapsuleRestoreAdmissionEnvelopeBytes - 32 ||
      envelope.size() != bytes_before_mac + 32) {
    return absl::DataLossError(
        "CapsuleRestore admission envelope length is non-canonical.");
  }
  const size_t key_offset = 8 + 4 + 4 + 8;
  const absl::string_view encoded_key =
      envelope.substr(key_offset, key_id_size);
  if (encoded_key != authentication.key_id) {
    return absl::UnauthenticatedError(
        "CapsuleRestore admission envelope key ID does not match.");
  }
  Hash256 encoded_mac;
  std::memcpy(encoded_mac.bytes.data(),
              envelope.data() + static_cast<size_t>(bytes_before_mac),
              encoded_mac.bytes.size());
  const absl::string_view authenticated_bytes =
      envelope.substr(0, static_cast<size_t>(bytes_before_mac));
  const Hash256 expected_mac = HmacSha256(
      authentication.authentication_key,
      {kAdmissionMacDomain, authenticated_bytes});
  if (!ConstantTimeHashEquals(encoded_mac, expected_mac)) {
    return absl::UnauthenticatedError(
        "CapsuleRestore admission envelope authentication failed.");
  }
  const size_t body_offset = key_offset + key_id_size;
  Reader body_reader(
      envelope.substr(body_offset, static_cast<size_t>(body_size)));
  ABSL_ASSIGN_OR_RETURN(const Hash256 encoded_record_id,
                        body_reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(
      CapsuleRestoreAdmissionRecord record,
      DecodeRecordFields(
          envelope.substr(body_offset + 32,
                          static_cast<size_t>(body_size) - 32),
          encoded_record_id, authentication.key_id));
  if (record.record_id != encoded_record_id) {
    return absl::DataLossError(
        "CapsuleRestore admission envelope record ID is inconsistent.");
  }
  return record;
}

absl::StatusOr<CapsuleRestoreAdmissionRecord>
CapsuleRestoreQualifier::Qualify(
    const CapsuleRestoreQualificationSpec& spec,
    const SessionHandoffOptions& checkpoint_authentication,
    const FreshWorkerAuthentication& record_authentication) const {
  ABSL_RETURN_IF_ERROR(ValidateCheckpointAuthentication(
      checkpoint_authentication, record_authentication));
  ABSL_ASSIGN_OR_RETURN(
      const ResolvedQualification resolved,
      ResolveQualification(authoritative_engine_, spec));
  if (checkpoint_authentication.expected_identity.has_value() &&
      *checkpoint_authentication.expected_identity != resolved.identity) {
    return absl::FailedPreconditionError(
        "Caller checkpoint identity assertion differs from the loaded "
        "Engine.");
  }
  SessionHandoffOptions bound_checkpoint_authentication =
      checkpoint_authentication;
  bound_checkpoint_authentication.expected_identity = resolved.identity;

  const int vocabulary_size =
      authoritative_engine_->GetTokenizer().GetVocabSize();
  if (vocabulary_size <= 0 ||
      static_cast<uint32_t>(vocabulary_size) !=
          resolved.profile.logits_frame.vocabulary_size) {
    return absl::FailedPreconditionError(
        "Loaded tokenizer vocabulary differs from the exact logits "
        "contract.");
  }

  // Source creation count is intentionally exactly one.
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Engine::Session> source,
      authoritative_engine_->CreateSession(resolved.config));
  if (source == nullptr) {
    return absl::InternalError(
        "Engine returned a null CapsuleRestore source session.");
  }
  ABSL_RETURN_IF_ERROR(RequireFreshSession(*source, resolved.identity));
  ABSL_RETURN_IF_ERROR(PrefillChunks(spec.checkpoint_prefix_chunks,
                                    vocabulary_size, source.get()));
  // Exactly one checkpoint decode creates the post-decode producing boundary.
  ABSL_ASSIGN_OR_RETURN(
      const CapsuleRestoreContinuationEvidence checkpoint_decode,
      RunCanonicalExactDecode(source.get(), resolved.profile,
                              spec.checkpoint_output_tokens));
  // The canonicalization above is required even though its output is retained
  // indirectly through the full exact session history rather than duplicated
  // in the admission record.
  (void)checkpoint_decode;

  ABSL_ASSIGN_OR_RETURN(const auto checkpoint_observation,
                        InspectSingleHistory(*source));
  if (checkpoint_observation.first <= 0 ||
      checkpoint_observation.second.empty()) {
    return absl::FailedPreconditionError(
        "CapsuleRestore source did not reach a producing checkpoint "
        "boundary.");
  }
  std::vector<int32_t> checkpoint_token_ids;
  checkpoint_token_ids.reserve(checkpoint_observation.second.size());
  for (int token_id : checkpoint_observation.second) {
    if (token_id < 0 || token_id >= vocabulary_size) {
      return absl::DataLossError(
          "CapsuleRestore source history contains an invalid token ID.");
    }
    checkpoint_token_ids.push_back(static_cast<int32_t>(token_id));
  }
  ABSL_ASSIGN_OR_RETURN(
      const std::string checkpoint_history_token_bytes,
      EncodeFreshWorkerTokenIds(checkpoint_token_ids));
  ABSL_ASSIGN_OR_RETURN(
      const std::string checkpoint_envelope,
      source->ExportHandoff(bound_checkpoint_authentication));
  ABSL_RETURN_IF_ERROR(ValidateLrtSess1Envelope(checkpoint_envelope));
  ABSL_RETURN_IF_ERROR(ValidateSessionIdentity(*source, resolved.identity));
  ABSL_ASSIGN_OR_RETURN(const auto source_after_export,
                        InspectSingleHistory(*source));
  if (source_after_export != checkpoint_observation) {
    return absl::FailedPreconditionError(
        "CapsuleRestore export changed the live source step or exact token "
        "history.");
  }
  // LRTSESS1 is canonical. A second quiescent observation must therefore be
  // byte-identical; this catches export implementations that mutate hidden
  // state while taking a snapshot. State is observed once more after the
  // second export, and the later live-vs-restored continuation comparison
  // remains the final load-bearing-state check.
  ABSL_ASSIGN_OR_RETURN(
      const std::string repeated_checkpoint_envelope,
      source->ExportHandoff(bound_checkpoint_authentication));
  ABSL_RETURN_IF_ERROR(
      ValidateLrtSess1Envelope(repeated_checkpoint_envelope));
  if (repeated_checkpoint_envelope != checkpoint_envelope) {
    return absl::FailedPreconditionError(
        "CapsuleRestore export is not a stable canonical observation of the "
        "same quiescent source state.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSessionIdentity(*source, resolved.identity));
  ABSL_ASSIGN_OR_RETURN(const auto source_after_repeated_export,
                        InspectSingleHistory(*source));
  if (source_after_repeated_export != checkpoint_observation) {
    return absl::FailedPreconditionError(
        "Repeated CapsuleRestore export changed the source step or history.");
  }

  // Target creation count is intentionally exactly one. Freshness is proven
  // immediately before the transactional import API is invoked.
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Engine::Session> target,
      authoritative_engine_->CreateSession(resolved.config));
  if (target == nullptr) {
    return absl::InternalError(
        "Engine returned a null CapsuleRestore target session.");
  }
  ABSL_RETURN_IF_ERROR(RequireFreshSession(*target, resolved.identity));
  ABSL_RETURN_IF_ERROR(target->ImportHandoff(
      checkpoint_envelope, bound_checkpoint_authentication));
  ABSL_RETURN_IF_ERROR(ValidateSessionIdentity(*target, resolved.identity));
  ABSL_ASSIGN_OR_RETURN(const auto restored_checkpoint_observation,
                        InspectSingleHistory(*target));
  if (restored_checkpoint_observation != checkpoint_observation) {
    return absl::FailedPreconditionError(
        "Restored CapsuleRestore target step/history differs from its live "
        "source before delta prefill.");
  }

  // Apply each exact delta boundary once to each session. There is no merged
  // prefill, retry, repair, alternate checkpoint, or full-prefill fallback.
  for (const CapsuleRestorePrefillChunk& chunk : spec.delta_chunks) {
    ABSL_RETURN_IF_ERROR(
        PrefillChunks({chunk}, vocabulary_size, source.get()));
    ABSL_RETURN_IF_ERROR(
        PrefillChunks({chunk}, vocabulary_size, target.get()));
  }
  ABSL_ASSIGN_OR_RETURN(const auto live_before_continuation,
                        InspectSingleHistory(*source));
  ABSL_ASSIGN_OR_RETURN(const auto restored_before_continuation,
                        InspectSingleHistory(*target));
  if (live_before_continuation != restored_before_continuation) {
    return absl::FailedPreconditionError(
        "Live and restored CapsuleRestore sessions differ after identical "
        "delta-prefill work.");
  }

  // Exactly one continuation decode is run on each session with the same
  // explicit bound; both are independently canonicalized before comparison.
  ABSL_ASSIGN_OR_RETURN(
      const CapsuleRestoreContinuationEvidence live_continuation,
      RunCanonicalExactDecode(source.get(), resolved.profile,
                              spec.continuation_output_tokens));
  ABSL_ASSIGN_OR_RETURN(
      const CapsuleRestoreContinuationEvidence restored_continuation,
      RunCanonicalExactDecode(target.get(), resolved.profile,
                              spec.continuation_output_tokens));
  if (live_continuation != restored_continuation) {
    return absl::FailedPreconditionError(
        "CapsuleRestore live/restored visible bytes, DPMTOK01 tokens, "
        "ordered full-logits evidence, or output-evidence hashes differ.");
  }
  ABSL_ASSIGN_OR_RETURN(const auto live_after_continuation,
                        InspectSingleHistory(*source));
  ABSL_ASSIGN_OR_RETURN(const auto restored_after_continuation,
                        InspectSingleHistory(*target));
  if (live_after_continuation != restored_after_continuation) {
    return absl::FailedPreconditionError(
        "CapsuleRestore live/restored final steps or exact histories differ.");
  }

  ABSL_RETURN_IF_ERROR(ValidateSessionIdentity(*source, resolved.identity));
  ABSL_RETURN_IF_ERROR(ValidateSessionIdentity(*target, resolved.identity));
  ABSL_ASSIGN_OR_RETURN(
      const ResolvedQualification after,
      ResolveQualification(authoritative_engine_, spec));
  if (after.profile != resolved.profile ||
      after.capability != resolved.capability ||
      after.identity != resolved.identity ||
      after.specification_hash != resolved.specification_hash) {
    return absl::AbortedError(
        "Engine-derived exact profile, CapsuleRestore capability, session "
        "identity, or qualification hash changed during qualification.");
  }

  CapsuleRestoreAdmissionRecord record;
  record.capability = resolved.capability;
  record.qualification_spec_hash = resolved.specification_hash;
  record.checkpoint_step =
      static_cast<uint64_t>(checkpoint_observation.first);
  record.checkpoint_history_token_bytes = checkpoint_history_token_bytes;
  record.checkpoint_envelope_hash = Sha256(checkpoint_envelope);
  record.checkpoint_envelope_size = checkpoint_envelope.size();
  record.checkpoint_authentication_key_id =
      checkpoint_authentication.key_id;
  record.live_continuation = live_continuation;
  record.restored_continuation = restored_continuation;
  record.qualified_unix_micros = absl::ToUnixMicros(absl::Now());
  if (record.qualified_unix_micros <= 0) {
    return absl::FailedPreconditionError(
        "System clock cannot timestamp CapsuleRestore admission.");
  }
  record.record_authentication_key_id = record_authentication.key_id;
  ABSL_ASSIGN_OR_RETURN(record.record_id,
                        ComputeCapsuleRestoreAdmissionRecordId(record));
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreAdmissionRecordForRuntime(
      record, resolved.profile, resolved.capability, spec));
  return record;
}

absl::StatusOr<CapsuleRestoreAdmissionRecord>
CapsuleRestoreQualifier::QualifyAndAdmit(
    const CapsuleRestoreQualificationSpec& spec,
    const SessionHandoffOptions& checkpoint_authentication,
    const FreshWorkerAuthentication& record_authentication,
    CapsuleRestoreAdmissionRepository* repository) const {
  if (repository == nullptr) {
    return absl::InvalidArgumentError(
        "CapsuleRestore admission repository is missing.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCheckpointAuthentication(
      checkpoint_authentication, record_authentication));
  ABSL_ASSIGN_OR_RETURN(
      const ResolvedQualification resolved,
      ResolveQualification(authoritative_engine_, spec));
  absl::StatusOr<CapsuleRestoreAdmissionRecord> existing = repository->Get(
      resolved.profile, resolved.capability, spec, record_authentication);
  if (existing.ok()) return std::move(*existing);
  if (existing.status().code() != absl::StatusCode::kNotFound) {
    return existing.status();
  }

  ABSL_ASSIGN_OR_RETURN(
      CapsuleRestoreAdmissionRecord record,
      Qualify(spec, checkpoint_authentication, record_authentication));
  if (record.capability != resolved.capability ||
      record.qualification_spec_hash != resolved.specification_hash) {
    return absl::AbortedError(
        "CapsuleRestore runtime identity changed before publication.");
  }
  const absl::Status put =
      repository->PutIfAbsent(record, record_authentication);
  if (put.ok()) return record;
  if (put.code() != absl::StatusCode::kAlreadyExists) return put;
  return repository->Get(resolved.profile, resolved.capability, spec,
                         record_authentication);
}

}  // namespace litert::lm
