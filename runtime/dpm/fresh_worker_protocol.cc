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

#include "runtime/dpm/fresh_worker_protocol.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/hmac_sha256.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kRequestMagic = {'D', 'P', 'M', 'W', 'R', 'Q',
                                                '0', '3'};
constexpr std::array<char, 8> kResultMagic = {'D', 'P', 'M', 'W', 'R', 'S',
                                               '0', '3'};
constexpr std::array<char, 8> kTokenBytesMagic = {'D', 'P', 'M', 'T', 'O', 'K',
                                                   '0', '1'};
constexpr std::array<char, 8> kExecutionPlanMagic = {
    'D', 'P', 'M', 'P', 'L', 'N', '0', '1'};
constexpr uint32_t kRequestKind = 1;
constexpr uint32_t kResultKind = 2;
constexpr uint64_t kMaximumResultStatusMessageBytes = 4096;
constexpr uint64_t kMaximumKeyIdBytes = 256;
constexpr uint64_t kMaximumAuthenticationKeyBytes = 4096;
constexpr uint32_t kMaximumContinuationWitnessKeyIdBytes = 1024;
constexpr uint64_t kEnvelopeFixedBytes = 8 + 4 + 4 + 4 + 8 + 32;
constexpr absl::string_view kRequestMacDomain =
    "LITERT_LMX_FRESH_WORKER_REQUEST_HMAC_SHA256_V3";
constexpr absl::string_view kResultMacDomain =
    "LITERT_LMX_FRESH_WORKER_RESULT_HMAC_SHA256_V3";
constexpr absl::string_view kExecutionPlanHashDomain =
    "LITERT_LMX_FRESH_WORKER_EXECUTION_PLAN_SHA256_V1";
constexpr absl::string_view kOutputEvidenceHashDomain =
    "LITERT_LMX_FRESH_WORKER_OUTPUT_EVIDENCE_SHA256_V1";

bool IsZeroHash(const Hash256& hash) {
  uint8_t combined = 0;
  for (uint8_t byte : hash.bytes) combined |= byte;
  return combined == 0;
}

bool IsZeroIdentity(const SessionHandoffIdentity& identity) {
  return IsZeroHash(identity.model_artifact_hash) &&
         IsZeroHash(identity.runtime_artifact_hash) &&
         IsZeroHash(identity.inference_profile_hash);
}

bool IsCompleteIdentity(const SessionHandoffIdentity& identity) {
  return !IsZeroHash(identity.model_artifact_hash) &&
         !IsZeroHash(identity.runtime_artifact_hash) &&
         !IsZeroHash(identity.inference_profile_hash);
}

bool HasControlByte(absl::string_view text) {
  for (unsigned char byte : text) {
    if (byte < 0x20 || byte == 0x7f) return true;
  }
  return false;
}

bool IsKnownStatusCode(absl::StatusCode status_code) {
  const int value = static_cast<int>(status_code);
  return value >= static_cast<int>(absl::StatusCode::kOk) &&
         value <= static_cast<int>(absl::StatusCode::kUnauthenticated);
}

absl::StatusOr<uint32_t> ExpectedLogitElementByteWidth(
    FreshWorkerLogitElementType element_type) {
  switch (element_type) {
    case FreshWorkerLogitElementType::kFloat16:
      return 2;
    case FreshWorkerLogitElementType::kFloat32:
      return 4;
    case FreshWorkerLogitElementType::kUnsupported:
      break;
  }
  return absl::FailedPreconditionError(
      "Fresh-worker logits element type is unsupported.");
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

void AppendIdentity(const SessionHandoffIdentity& identity,
                    std::string* output) {
  AppendHash(identity.model_artifact_hash, output);
  AppendHash(identity.runtime_artifact_hash, output);
  AppendHash(identity.inference_profile_hash, output);
}

void AppendContinuationWitness(
    const SessionContinuationStateWitness& witness, std::string* output) {
  AppendU32(witness.format_version, output);
  AppendHash(witness.witness_id, output);
  AppendIdentity(witness.session_identity, output);
  AppendU32(static_cast<uint32_t>(witness.phase), output);
  AppendU32(static_cast<uint32_t>(witness.current_step), output);
  AppendU32(witness.ran_decode ? 1 : 0, output);
  AppendHash(witness.processed_history_token_bytes_hash, output);
  AppendHash(witness.envelope_hash, output);
  AppendU64(witness.envelope_size, output);
  AppendU32(static_cast<uint32_t>(witness.key_id.size()), output);
  output->append(witness.key_id);
}

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

void HashU32(uint32_t value, Sha256Hasher* hasher) {
  std::string bytes;
  AppendU32(value, &bytes);
  hasher->Update(bytes);
}

void HashU64(uint64_t value, Sha256Hasher* hasher) {
  std::string bytes;
  AppendU64(value, &bytes);
  hasher->Update(bytes);
}

void HashValue(const Hash256& value, Sha256Hasher* hasher) {
  hasher->Update(absl::string_view(
      reinterpret_cast<const char*>(value.bytes.data()), value.bytes.size()));
}

void HashFrame(absl::string_view bytes, Sha256Hasher* hasher) {
  HashU64(bytes.size(), hasher);
  hasher->Update(bytes);
}

absl::Status ValidateExecutionPlanFields(
    const FreshWorkerExecutionPlan& plan) {
  if (plan.format_version != kFreshWorkerExecutionPlanFormatVersion) {
    return absl::FailedPreconditionError(
        "Fresh-worker execution-plan version is unsupported.");
  }
  if (IsZeroHash(plan.logical_replay_request_hash)) {
    return absl::InvalidArgumentError(
        "Fresh-worker execution plan requires a logical replay hash.");
  }
  if (plan.canonical_execution_payload.size() >
      kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker physical execution payload exceeds the protocol "
        "limit.");
  }

  switch (plan.prefill_mode) {
    case FreshWorkerPrefillMode::kFullCanonicalPrefill:
      if (plan.restore_checkpoint_id.has_value() ||
          !IsZeroIdentity(plan.session_identity) ||
          !IsZeroHash(plan.restore_durable_envelope_hash) ||
          plan.restore_durable_envelope_size != 0 ||
          !plan.canonical_execution_payload.empty()) {
        return absl::InvalidArgumentError(
            "Full-prefill execution plans cannot carry restore or delta "
            "fields.");
      }
      break;
    case FreshWorkerPrefillMode::kOwnPositionCapsuleDelta:
      if (!plan.restore_checkpoint_id.has_value() ||
          IsZeroHash(*plan.restore_checkpoint_id) ||
          !IsCompleteIdentity(plan.session_identity) ||
          IsZeroHash(plan.restore_durable_envelope_hash) ||
          plan.restore_durable_envelope_size == 0 ||
          plan.canonical_execution_payload.empty()) {
        return absl::InvalidArgumentError(
            "Own-position execution plans require complete checkpoint, "
            "identity, envelope, and delta bindings.");
      }
      break;
    default:
      return absl::InvalidArgumentError(
          "Fresh-worker execution plan has an unknown prefill mode.");
  }
  return absl::OkStatus();
}

std::string EncodeExecutionPlanHashFields(
    const FreshWorkerExecutionPlan& plan) {
  std::string fields;
  fields.reserve(kExecutionPlanMagic.size() + 4 + 4 + 32 + 4 + 32 + 96 +
                 32 + 8 + 8 + plan.canonical_execution_payload.size());
  fields.append(kExecutionPlanMagic.data(), kExecutionPlanMagic.size());
  AppendU32(plan.format_version, &fields);
  AppendU32(static_cast<uint32_t>(plan.prefill_mode), &fields);
  AppendHash(plan.logical_replay_request_hash, &fields);
  AppendU32(plan.restore_checkpoint_id.has_value() ? 1 : 0, &fields);
  if (plan.restore_checkpoint_id.has_value()) {
    AppendHash(*plan.restore_checkpoint_id, &fields);
  }
  AppendIdentity(plan.session_identity, &fields);
  AppendHash(plan.restore_durable_envelope_hash, &fields);
  AppendU64(plan.restore_durable_envelope_size, &fields);
  AppendU64(plan.canonical_execution_payload.size(), &fields);
  fields.append(plan.canonical_execution_payload);
  return fields;
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
    const absl::string_view result = bytes_.substr(offset_, safe_size);
    offset_ += safe_size;
    return result;
  }

  size_t remaining() const { return bytes_.size() - offset_; }

 private:
  absl::Status Truncated() const {
    return absl::DataLossError("Truncated fresh-worker protocol value.");
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

absl::StatusOr<SessionHandoffIdentity> ReadIdentity(Reader* reader) {
  if (reader == nullptr) {
    return absl::InternalError(
        "Fresh-worker protocol identity reader is null.");
  }
  SessionHandoffIdentity identity;
  ABSL_ASSIGN_OR_RETURN(identity.model_artifact_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(identity.runtime_artifact_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(identity.inference_profile_hash, reader->ReadHash());
  return identity;
}

absl::StatusOr<SessionContinuationStateWitness> ReadContinuationWitness(
    Reader* reader) {
  if (reader == nullptr) {
    return absl::InternalError(
        "Fresh-worker continuation witness reader is null.");
  }
  SessionContinuationStateWitness witness;
  ABSL_ASSIGN_OR_RETURN(witness.format_version, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(witness.witness_id, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(witness.session_identity, ReadIdentity(reader));
  ABSL_ASSIGN_OR_RETURN(const uint32_t phase, reader->ReadU32());
  if (phase > static_cast<uint32_t>(SessionHandoffPhase::kDecoded)) {
    return absl::DataLossError(
        "Fresh-worker continuation witness phase is invalid.");
  }
  witness.phase = static_cast<SessionHandoffPhase>(phase);
  ABSL_ASSIGN_OR_RETURN(const uint32_t current_step, reader->ReadU32());
  if (current_step >
      static_cast<uint32_t>((std::numeric_limits<int32_t>::max)())) {
    return absl::DataLossError(
        "Fresh-worker continuation witness step is not representable.");
  }
  witness.current_step = static_cast<int32_t>(current_step);
  ABSL_ASSIGN_OR_RETURN(const uint32_t ran_decode, reader->ReadU32());
  if (ran_decode > 1) {
    return absl::DataLossError(
        "Fresh-worker continuation witness decode flag is invalid.");
  }
  witness.ran_decode = ran_decode != 0;
  ABSL_ASSIGN_OR_RETURN(witness.processed_history_token_bytes_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(witness.envelope_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(witness.envelope_size, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(const uint32_t key_id_size, reader->ReadU32());
  if (key_id_size == 0 ||
      key_id_size > kMaximumContinuationWitnessKeyIdBytes) {
    return absl::DataLossError(
        "Fresh-worker continuation witness key ID is invalid.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view key_id,
                        reader->ReadBytes(key_id_size));
  witness.key_id.assign(key_id.data(), key_id.size());
  ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(witness));
  return witness;
}

absl::StatusOr<std::string> EncodeExecutionPlan(
    const FreshWorkerExecutionPlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerExecutionPlan(plan));
  std::string encoded;
  encoded.reserve(kExecutionPlanMagic.size() + 4 + 32 + 4 + 32 + 4 + 32 +
                  96 + 32 + 8 + 8 +
                  plan.canonical_execution_payload.size() + 4);
  encoded.append(kExecutionPlanMagic.data(), kExecutionPlanMagic.size());
  AppendU32(plan.format_version, &encoded);
  AppendHash(plan.plan_hash, &encoded);
  AppendU32(static_cast<uint32_t>(plan.prefill_mode), &encoded);
  AppendHash(plan.logical_replay_request_hash, &encoded);
  AppendU32(plan.restore_checkpoint_id.has_value() ? 1 : 0, &encoded);
  if (plan.restore_checkpoint_id.has_value()) {
    AppendHash(*plan.restore_checkpoint_id, &encoded);
  }
  AppendIdentity(plan.session_identity, &encoded);
  AppendHash(plan.restore_durable_envelope_hash, &encoded);
  AppendU64(plan.restore_durable_envelope_size, &encoded);
  AppendU64(plan.canonical_execution_payload.size(), &encoded);
  encoded.append(plan.canonical_execution_payload);
  AppendU32(plan.capture_producing_capsule ? 1 : 0, &encoded);
  return encoded;
}

absl::StatusOr<FreshWorkerExecutionPlan> DecodeExecutionPlan(
    absl::string_view encoded) {
  constexpr uint64_t kMinimumPlanBytes =
      8 + 4 + 32 + 4 + 32 + 4 + 96 + 32 + 8 + 8 + 4;
  if (encoded.size() < kMinimumPlanBytes ||
      encoded.size() > kMaximumFreshWorkerEnvelopeBytes ||
      std::memcmp(encoded.data(), kExecutionPlanMagic.data(),
                  kExecutionPlanMagic.size()) != 0) {
    return absl::DataLossError(
        "Fresh-worker execution-plan framing is invalid.");
  }
  Reader reader(encoded.substr(kExecutionPlanMagic.size()));
  FreshWorkerExecutionPlan plan;
  ABSL_ASSIGN_OR_RETURN(plan.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.plan_hash, reader.ReadHash());
  uint32_t prefill_mode;
  ABSL_ASSIGN_OR_RETURN(prefill_mode, reader.ReadU32());
  plan.prefill_mode = static_cast<FreshWorkerPrefillMode>(prefill_mode);
  ABSL_ASSIGN_OR_RETURN(plan.logical_replay_request_hash, reader.ReadHash());
  uint32_t has_restore_checkpoint;
  ABSL_ASSIGN_OR_RETURN(has_restore_checkpoint, reader.ReadU32());
  if (has_restore_checkpoint > 1) {
    return absl::DataLossError(
        "Fresh-worker execution-plan checkpoint flag is invalid.");
  }
  if (has_restore_checkpoint != 0) {
    Hash256 checkpoint_id;
    ABSL_ASSIGN_OR_RETURN(checkpoint_id, reader.ReadHash());
    plan.restore_checkpoint_id = checkpoint_id;
  }
  ABSL_ASSIGN_OR_RETURN(plan.session_identity, ReadIdentity(&reader));
  ABSL_ASSIGN_OR_RETURN(plan.restore_durable_envelope_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.restore_durable_envelope_size,
                        reader.ReadU64());
  uint64_t execution_payload_size;
  ABSL_ASSIGN_OR_RETURN(execution_payload_size, reader.ReadU64());
  if (execution_payload_size > kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker physical execution payload exceeds the protocol "
        "limit.");
  }
  absl::string_view execution_payload;
  ABSL_ASSIGN_OR_RETURN(execution_payload,
                        reader.ReadBytes(execution_payload_size));
  plan.canonical_execution_payload.assign(execution_payload.data(),
                                          execution_payload.size());
  uint32_t capture_producing_capsule;
  ABSL_ASSIGN_OR_RETURN(capture_producing_capsule, reader.ReadU32());
  if (capture_producing_capsule > 1 || reader.remaining() != 0) {
    return absl::DataLossError(
        "Fresh-worker execution-plan capture flag or length is invalid.");
  }
  plan.capture_producing_capsule = capture_producing_capsule != 0;
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerExecutionPlan(plan));
  ABSL_ASSIGN_OR_RETURN(const std::string canonical,
                        EncodeExecutionPlan(plan));
  if (canonical != encoded) {
    return absl::DataLossError(
        "Fresh-worker execution plan is not canonically encoded.");
  }
  return plan;
}

absl::Status ValidateProducingCapsuleEvidence(
    const FreshWorkerProducingCapsuleEvidence& evidence) {
  if (!IsCompleteIdentity(evidence.session_identity) ||
      evidence.transient_envelope_size == 0 ||
      IsZeroHash(evidence.transient_envelope_hash) ||
      IsZeroHash(evidence.output_evidence_hash)) {
    return absl::InvalidArgumentError(
        "Fresh-worker producing-capsule evidence is incomplete.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(
      evidence.producer_first_export));
  ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(
      evidence.producer_second_export));
  ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(
      evidence.fresh_import_target));
  if (evidence.producer_first_export != evidence.producer_second_export ||
      evidence.producer_first_export != evidence.fresh_import_target) {
    return absl::DataLossError(
        "Fresh-worker producing-capsule continuation witnesses disagree.");
  }
  const SessionContinuationStateWitness& witness =
      evidence.producer_first_export;
  if (witness.session_identity != evidence.session_identity ||
      witness.phase != SessionHandoffPhase::kDecoded ||
      !witness.ran_decode || witness.current_step <= 0 ||
      witness.envelope_size != evidence.transient_envelope_size ||
      witness.envelope_hash != evidence.transient_envelope_hash) {
    return absl::DataLossError(
        "Fresh-worker producing-capsule witness does not describe the "
        "sealed decoded producer state.");
  }
  return absl::OkStatus();
}

struct DecodedEnvelopeBody {
  absl::string_view body;
};

absl::StatusOr<std::string> EncodeEnvelope(
    const std::array<char, 8>& magic, uint32_t kind,
    absl::string_view mac_domain, absl::string_view body,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (body.size() > kMaximumFreshWorkerEnvelopeBytes - kEnvelopeFixedBytes -
                        authentication.key_id.size()) {
    return absl::ResourceExhaustedError(
        "Fresh-worker envelope body exceeds the protocol limit.");
  }
  std::string envelope;
  envelope.reserve(kEnvelopeFixedBytes + authentication.key_id.size() +
                   body.size());
  envelope.append(magic.data(), magic.size());
  AppendU32(kFreshWorkerProtocolVersion, &envelope);
  AppendU32(kind, &envelope);
  AppendU32(static_cast<uint32_t>(authentication.key_id.size()), &envelope);
  AppendU64(body.size(), &envelope);
  envelope.append(authentication.key_id);
  envelope.append(body);
  const Hash256 mac =
      HmacSha256(authentication.authentication_key, {mac_domain, envelope});
  AppendHash(mac, &envelope);
  return envelope;
}

absl::StatusOr<DecodedEnvelopeBody> DecodeEnvelope(
    absl::string_view envelope, const std::array<char, 8>& expected_magic,
    uint32_t expected_kind, absl::string_view mac_domain,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (envelope.size() < kEnvelopeFixedBytes ||
      envelope.size() > kMaximumFreshWorkerEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker envelope size is outside protocol bounds.");
  }
  if (std::memcmp(envelope.data(), expected_magic.data(),
                  expected_magic.size()) != 0) {
    return absl::DataLossError("Fresh-worker envelope magic is invalid.");
  }
  Reader header(envelope.substr(expected_magic.size()));
  ABSL_ASSIGN_OR_RETURN(const uint32_t version, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t kind, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t key_id_size, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint64_t body_size, header.ReadU64());
  if (version != kFreshWorkerProtocolVersion || kind != expected_kind) {
    return absl::FailedPreconditionError(
        "Fresh-worker envelope version or kind is unsupported.");
  }
  if (key_id_size > kMaximumKeyIdBytes ||
      body_size > kMaximumFreshWorkerEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker envelope declares an oversized field.");
  }
  const uint64_t bytes_before_mac =
      uint64_t{8} + 4 + 4 + 4 + 8 + key_id_size + body_size;
  if (bytes_before_mac > kMaximumFreshWorkerEnvelopeBytes - 32 ||
      envelope.size() != bytes_before_mac + 32) {
    return absl::DataLossError(
        "Fresh-worker envelope length is non-canonical.");
  }

  const size_t key_id_offset = 8 + 4 + 4 + 4 + 8;
  const absl::string_view encoded_key_id =
      envelope.substr(key_id_offset, key_id_size);
  if (encoded_key_id != authentication.key_id) {
    return absl::UnauthenticatedError(
        "Fresh-worker envelope authentication key ID does not match.");
  }
  Hash256 encoded_mac;
  std::memcpy(encoded_mac.bytes.data(),
              envelope.data() + static_cast<size_t>(bytes_before_mac),
              encoded_mac.bytes.size());
  const absl::string_view authenticated_bytes =
      envelope.substr(0, static_cast<size_t>(bytes_before_mac));
  const Hash256 expected_mac = HmacSha256(
      authentication.authentication_key, {mac_domain, authenticated_bytes});
  if (!ConstantTimeHashEquals(encoded_mac, expected_mac)) {
    return absl::UnauthenticatedError(
        "Fresh-worker envelope authentication failed.");
  }

  const size_t body_offset = key_id_offset + key_id_size;
  return DecodedEnvelopeBody{
      .body = envelope.substr(body_offset, static_cast<size_t>(body_size))};
}

absl::StatusOr<std::string> EncodeRequestBody(
    const FreshWorkerRequest& request) {
  ABSL_ASSIGN_OR_RETURN(const std::string execution_plan,
                        EncodeExecutionPlan(request.execution_plan));
  std::string body;
  body.reserve(4 + 32 + 32 + 4 + 4 + 32 + 8 +
               request.request_payload.size() + 8 + execution_plan.size());
  AppendU32(request.format_version, &body);
  AppendHash(request.exact_profile_hash, &body);
  AppendHash(request.qualification_id, &body);
  AppendU32(request.run_index, &body);
  AppendU32(request.run_count, &body);
  AppendHash(request.challenge_nonce, &body);
  AppendU64(request.request_payload.size(), &body);
  body.append(request.request_payload);
  AppendU64(execution_plan.size(), &body);
  body.append(execution_plan);
  return body;
}

absl::StatusOr<FreshWorkerRequest> DecodeRequestBody(absl::string_view body) {
  Reader reader(body);
  FreshWorkerRequest request;
  ABSL_ASSIGN_OR_RETURN(request.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(request.exact_profile_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.qualification_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.run_index, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(request.run_count, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(request.challenge_nonce, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(const uint64_t payload_size, reader.ReadU64());
  if (payload_size > kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker request payload exceeds the protocol limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view payload,
                        reader.ReadBytes(payload_size));
  request.request_payload.assign(payload.data(), payload.size());
  uint64_t execution_plan_size;
  ABSL_ASSIGN_OR_RETURN(execution_plan_size, reader.ReadU64());
  if (execution_plan_size > kMaximumFreshWorkerEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker execution plan exceeds the envelope limit.");
  }
  absl::string_view execution_plan;
  ABSL_ASSIGN_OR_RETURN(execution_plan,
                        reader.ReadBytes(execution_plan_size));
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "Fresh-worker request body has trailing bytes.");
  }
  ABSL_ASSIGN_OR_RETURN(request.execution_plan,
                        DecodeExecutionPlan(execution_plan));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(request));
  return request;
}

absl::StatusOr<std::string> EncodeResultBody(
    const FreshWorkerResult& result) {
  std::string prepared_prefill_plan;
  if (result.prepared_prefill_plan.has_value()) {
    ABSL_ASSIGN_OR_RETURN(
        prepared_prefill_plan,
        EncodeDPMPreparedPrefillPlan(*result.prepared_prefill_plan));
  }
  std::string body;
  body.reserve(4 + 32 * 10 + 96 + 8 + 4 * 6 + 8 +
               result.status_message.size() +
               result.token_bytes.size() +
               result.logit_frames.size() * (5 * 4 + 8 + 32));
  AppendU32(result.format_version, &body);
  AppendHash(result.exact_profile_hash, &body);
  AppendHash(result.qualification_id, &body);
  AppendU32(result.run_index, &body);
  AppendU32(result.run_count, &body);
  AppendHash(result.challenge_nonce, &body);
  AppendHash(result.request_envelope_hash, &body);
  AppendHash(result.worker_instance_nonce, &body);
  AppendHash(result.execution_plan_hash, &body);
  AppendU32(result.prepared_prefill_plan.has_value() ? 1 : 0, &body);
  if (result.prepared_prefill_plan.has_value()) {
    AppendU64(prepared_prefill_plan.size(), &body);
    body.append(prepared_prefill_plan);
  }
  AppendU32(result.restored_checkpoint_id.has_value() ? 1 : 0, &body);
  if (result.restored_checkpoint_id.has_value()) {
    AppendHash(*result.restored_checkpoint_id, &body);
  }
  AppendU32(result.restored_state_witness.has_value() ? 1 : 0, &body);
  if (result.restored_state_witness.has_value()) {
    AppendContinuationWitness(*result.restored_state_witness, &body);
  }
  AppendU32(result.producing_capsule_evidence.has_value() ? 1 : 0, &body);
  if (result.producing_capsule_evidence.has_value()) {
    const FreshWorkerProducingCapsuleEvidence& evidence =
        *result.producing_capsule_evidence;
    AppendIdentity(evidence.session_identity, &body);
    AppendU64(evidence.transient_envelope_size, &body);
    AppendHash(evidence.transient_envelope_hash, &body);
    AppendHash(evidence.output_evidence_hash, &body);
    AppendContinuationWitness(evidence.producer_first_export, &body);
    AppendContinuationWitness(evidence.producer_second_export, &body);
    AppendContinuationWitness(evidence.fresh_import_target, &body);
  }
  AppendU32(static_cast<uint32_t>(result.replay_isolation), &body);
  AppendU32(static_cast<uint32_t>(result.status_code), &body);
  AppendU32(static_cast<uint32_t>(result.status_message.size()), &body);
  body.append(result.status_message);
  AppendU64(result.canonical_output.size(), &body);
  body.append(result.canonical_output);
  AppendU64(result.token_bytes.size(), &body);
  body.append(result.token_bytes);
  AppendU32(static_cast<uint32_t>(result.logit_frames.size()), &body);
  for (const FreshWorkerLogitFrameEvidence& frame : result.logit_frames) {
    AppendU32(static_cast<uint32_t>(frame.element_type), &body);
    AppendU32(frame.element_byte_width, &body);
    AppendU32(frame.batch_size, &body);
    AppendU32(frame.sequence_size, &body);
    AppendU32(frame.vocabulary_size, &body);
    AppendU64(frame.byte_count, &body);
    AppendHash(frame.sha256, &body);
  }
  return body;
}

absl::StatusOr<FreshWorkerResult> DecodeResultBody(absl::string_view body) {
  Reader reader(body);
  FreshWorkerResult result;
  ABSL_ASSIGN_OR_RETURN(result.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(result.exact_profile_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(result.qualification_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(result.run_index, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(result.run_count, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(result.challenge_nonce, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(result.request_envelope_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(result.worker_instance_nonce, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(result.execution_plan_hash, reader.ReadHash());
  uint32_t has_prepared_prefill_plan;
  ABSL_ASSIGN_OR_RETURN(has_prepared_prefill_plan, reader.ReadU32());
  if (has_prepared_prefill_plan > 1) {
    return absl::DataLossError(
        "Fresh-worker prepared-prefill-plan flag is invalid.");
  }
  if (has_prepared_prefill_plan != 0) {
    uint64_t prepared_prefill_plan_size;
    ABSL_ASSIGN_OR_RETURN(prepared_prefill_plan_size, reader.ReadU64());
    if (prepared_prefill_plan_size > kMaximumDPMPreparedPrefillPlanBytes) {
      return absl::ResourceExhaustedError(
          "Fresh-worker prepared prefill plan exceeds its protocol limit.");
    }
    absl::string_view prepared_prefill_plan;
    ABSL_ASSIGN_OR_RETURN(
        prepared_prefill_plan,
        reader.ReadBytes(prepared_prefill_plan_size));
    ABSL_ASSIGN_OR_RETURN(
        result.prepared_prefill_plan,
        DecodeDPMPreparedPrefillPlan(prepared_prefill_plan));
  }
  uint32_t has_restored_checkpoint;
  ABSL_ASSIGN_OR_RETURN(has_restored_checkpoint, reader.ReadU32());
  if (has_restored_checkpoint > 1) {
    return absl::DataLossError(
        "Fresh-worker restored-checkpoint flag is invalid.");
  }
  if (has_restored_checkpoint != 0) {
    Hash256 checkpoint_id;
    ABSL_ASSIGN_OR_RETURN(checkpoint_id, reader.ReadHash());
    result.restored_checkpoint_id = checkpoint_id;
  }
  uint32_t has_restored_state_witness;
  ABSL_ASSIGN_OR_RETURN(has_restored_state_witness, reader.ReadU32());
  if (has_restored_state_witness > 1) {
    return absl::DataLossError(
        "Fresh-worker restored-state-witness flag is invalid.");
  }
  if (has_restored_state_witness != 0) {
    ABSL_ASSIGN_OR_RETURN(result.restored_state_witness,
                          ReadContinuationWitness(&reader));
  }
  uint32_t has_producing_capsule;
  ABSL_ASSIGN_OR_RETURN(has_producing_capsule, reader.ReadU32());
  if (has_producing_capsule > 1) {
    return absl::DataLossError(
        "Fresh-worker producing-capsule flag is invalid.");
  }
  if (has_producing_capsule != 0) {
    FreshWorkerProducingCapsuleEvidence evidence;
    ABSL_ASSIGN_OR_RETURN(evidence.session_identity,
                          ReadIdentity(&reader));
    ABSL_ASSIGN_OR_RETURN(evidence.transient_envelope_size,
                          reader.ReadU64());
    ABSL_ASSIGN_OR_RETURN(evidence.transient_envelope_hash,
                          reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(evidence.output_evidence_hash,
                          reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(evidence.producer_first_export,
                          ReadContinuationWitness(&reader));
    ABSL_ASSIGN_OR_RETURN(evidence.producer_second_export,
                          ReadContinuationWitness(&reader));
    ABSL_ASSIGN_OR_RETURN(evidence.fresh_import_target,
                          ReadContinuationWitness(&reader));
    result.producing_capsule_evidence = evidence;
  }
  ABSL_ASSIGN_OR_RETURN(const uint32_t replay_isolation, reader.ReadU32());
  result.replay_isolation =
      static_cast<FreshWorkerReplayIsolation>(replay_isolation);
  ABSL_ASSIGN_OR_RETURN(const uint32_t encoded_status_code,
                        reader.ReadU32());
  result.status_code = static_cast<absl::StatusCode>(encoded_status_code);
  ABSL_ASSIGN_OR_RETURN(const uint32_t status_message_size, reader.ReadU32());
  if (status_message_size > kMaximumResultStatusMessageBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker status message exceeds the protocol limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view status_message,
                        reader.ReadBytes(status_message_size));
  result.status_message.assign(status_message.data(), status_message.size());
  ABSL_ASSIGN_OR_RETURN(const uint64_t canonical_output_size,
                        reader.ReadU64());
  if (canonical_output_size > kMaximumFreshWorkerCanonicalOutputBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker canonical output exceeds the protocol limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view canonical_output,
                        reader.ReadBytes(canonical_output_size));
  result.canonical_output.assign(canonical_output.data(),
                                 canonical_output.size());
  ABSL_ASSIGN_OR_RETURN(const uint64_t token_bytes_size, reader.ReadU64());
  if (token_bytes_size > kMaximumFreshWorkerTokenBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker token output exceeds the protocol limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view token_bytes,
                        reader.ReadBytes(token_bytes_size));
  result.token_bytes.assign(token_bytes.data(), token_bytes.size());
  ABSL_ASSIGN_OR_RETURN(const uint32_t frame_count, reader.ReadU32());
  if (frame_count > kMaximumFreshWorkerLogitFrames) {
    return absl::ResourceExhaustedError(
        "Fresh-worker logits evidence exceeds the protocol limit.");
  }
  result.logit_frames.reserve(frame_count);
  for (uint32_t index = 0; index < frame_count; ++index) {
    FreshWorkerLogitFrameEvidence frame;
    ABSL_ASSIGN_OR_RETURN(const uint32_t element_type, reader.ReadU32());
    frame.element_type =
        static_cast<FreshWorkerLogitElementType>(element_type);
    ABSL_ASSIGN_OR_RETURN(frame.element_byte_width, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(frame.batch_size, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(frame.sequence_size, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(frame.vocabulary_size, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(frame.byte_count, reader.ReadU64());
    ABSL_ASSIGN_OR_RETURN(frame.sha256, reader.ReadHash());
    result.logit_frames.push_back(frame);
  }
  if (reader.remaining() != 0) {
    return absl::DataLossError("Fresh-worker result body has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerResult(result));
  return result;
}

}  // namespace

absl::StatusOr<Hash256> ComputeFreshWorkerExecutionPlanHash(
    const FreshWorkerExecutionPlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateExecutionPlanFields(plan));
  Sha256Hasher hasher;
  hasher.Update(kExecutionPlanHashDomain);
  hasher.Update(EncodeExecutionPlanHashFields(plan));
  return hasher.Finalize();
}

absl::Status ValidateFreshWorkerExecutionPlan(
    const FreshWorkerExecutionPlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateExecutionPlanFields(plan));
  if (IsZeroHash(plan.plan_hash)) {
    return absl::InvalidArgumentError(
        "Fresh-worker execution plan requires a nonzero canonical hash.");
  }
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_hash,
                        ComputeFreshWorkerExecutionPlanHash(plan));
  if (plan.plan_hash != expected_hash) {
    return absl::DataLossError(
        "Fresh-worker execution-plan hash is not canonical.");
  }
  return absl::OkStatus();
}

absl::Status ValidateFreshWorkerAuthentication(
    const FreshWorkerAuthentication& authentication) {
  if (authentication.key_id.empty() ||
      authentication.key_id.size() > kMaximumKeyIdBytes ||
      HasControlByte(authentication.key_id)) {
    return absl::InvalidArgumentError(
        "Fresh-worker authentication key ID is invalid.");
  }
  if (authentication.authentication_key.size() < 32 ||
      authentication.authentication_key.size() >
          kMaximumAuthenticationKeyBytes) {
    return absl::InvalidArgumentError(
        "Fresh-worker authentication key must contain 32 to 4096 bytes.");
  }
  return absl::OkStatus();
}

absl::Status ValidateFreshWorkerRequest(const FreshWorkerRequest& request) {
  if (request.format_version != kFreshWorkerProtocolVersion) {
    return absl::FailedPreconditionError(
        "Fresh-worker request version is unsupported.");
  }
  if (IsZeroHash(request.exact_profile_hash) ||
      IsZeroHash(request.qualification_id) ||
      IsZeroHash(request.challenge_nonce)) {
    return absl::InvalidArgumentError(
        "Fresh-worker request identity fields must be nonzero.");
  }
  if (request.run_count < 2 ||
      request.run_count > kMaximumFreshWorkerRuns ||
      request.run_index >= request.run_count) {
    return absl::InvalidArgumentError(
        "Fresh-worker run index or run count is invalid.");
  }
  if (request.request_payload.empty()) {
    return absl::InvalidArgumentError(
        "Fresh-worker request payload must not be empty.");
  }
  if (request.request_payload.size() >
      kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker request payload exceeds the protocol limit.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerExecutionPlan(request.execution_plan));
  if (request.execution_plan.logical_replay_request_hash !=
      Sha256(request.request_payload)) {
    return absl::FailedPreconditionError(
        "Fresh-worker execution plan is not bound to the complete logical "
        "request payload.");
  }
  return absl::OkStatus();
}

absl::Status ValidateFreshWorkerExecutionOutput(
    const FreshWorkerExecutionOutput& output) {
  if (output.canonical_output.empty() ||
      output.canonical_output.size() >
          kMaximumFreshWorkerCanonicalOutputBytes) {
    return absl::InvalidArgumentError(
        "Fresh-worker canonical stage output is empty or oversized.");
  }
  if (output.token_bytes.size() > kMaximumFreshWorkerTokenBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker token output exceeds the protocol limit.");
  }
  absl::StatusOr<std::vector<int32_t>> decoded_token_ids =
      DecodeFreshWorkerTokenIds(output.token_bytes);
  if (!decoded_token_ids.ok()) return decoded_token_ids.status();
  if (decoded_token_ids->empty() ||
      output.logit_frames.size() != decoded_token_ids->size()) {
    return absl::InvalidArgumentError(
        "Fresh-worker output requires one full logits frame per sampled token.");
  }
  if (output.logit_frames.empty() ||
      output.logit_frames.size() > kMaximumFreshWorkerLogitFrames) {
    return absl::InvalidArgumentError(
        "Fresh-worker output must contain bounded logits evidence.");
  }
  const FreshWorkerLogitFrameEvidence& first_frame = output.logit_frames.front();
  for (size_t index = 0; index < output.logit_frames.size(); ++index) {
    const FreshWorkerLogitFrameEvidence& frame = output.logit_frames[index];
    ABSL_RETURN_IF_ERROR(ValidateFreshWorkerLogitFrameEvidence(frame));
    if (frame.batch_size != 1 || frame.sequence_size != 1) {
      return absl::FailedPreconditionError(
          "Exact fresh-worker logits must be one-candidate, one-position frames.");
    }
    if (frame.element_type != first_frame.element_type ||
        frame.element_byte_width != first_frame.element_byte_width ||
        frame.batch_size != first_frame.batch_size ||
        frame.sequence_size != first_frame.sequence_size ||
        frame.vocabulary_size != first_frame.vocabulary_size) {
      return absl::FailedPreconditionError(
          "Fresh-worker logits tensor contract changed during generation.");
    }
    if (static_cast<uint32_t>((*decoded_token_ids)[index]) >=
        frame.vocabulary_size) {
      return absl::InvalidArgumentError(
          "Fresh-worker output contains a token outside the logits vocabulary.");
    }
  }
  return absl::OkStatus();
}

Hash256 ComputeFreshWorkerOutputEvidenceHash(
    absl::string_view canonical_output,
    absl::string_view token_bytes,
    const std::vector<FreshWorkerLogitFrameEvidence>& logit_frames) {
  Sha256Hasher hasher;
  hasher.Update(kOutputEvidenceHashDomain);
  HashFrame(canonical_output, &hasher);
  HashFrame(token_bytes, &hasher);
  HashU32(static_cast<uint32_t>(logit_frames.size()), &hasher);
  for (const FreshWorkerLogitFrameEvidence& frame : logit_frames) {
    HashU32(static_cast<uint32_t>(frame.element_type), &hasher);
    HashU32(frame.element_byte_width, &hasher);
    HashU32(frame.batch_size, &hasher);
    HashU32(frame.sequence_size, &hasher);
    HashU32(frame.vocabulary_size, &hasher);
    HashU64(frame.byte_count, &hasher);
    HashValue(frame.sha256, &hasher);
  }
  return hasher.Finalize();
}

absl::Status ValidateFreshWorkerLogitFrameEvidence(
    const FreshWorkerLogitFrameEvidence& frame) {
  ABSL_ASSIGN_OR_RETURN(
      const uint32_t expected_element_width,
      ExpectedLogitElementByteWidth(frame.element_type));
  if (frame.element_byte_width != expected_element_width ||
      frame.batch_size == 0 || frame.sequence_size == 0 ||
      frame.vocabulary_size == 0 || IsZeroHash(frame.sha256)) {
    return absl::InvalidArgumentError(
        "Fresh-worker logits tensor evidence is incomplete.");
  }
  uint64_t expected_byte_count = frame.element_byte_width;
  const std::array<uint32_t, 3> dimensions = {
      frame.batch_size, frame.sequence_size, frame.vocabulary_size};
  for (uint32_t dimension : dimensions) {
    if (expected_byte_count >
        std::numeric_limits<uint64_t>::max() / dimension) {
      return absl::ResourceExhaustedError(
          "Fresh-worker logits tensor extent overflows uint64.");
    }
    expected_byte_count *= dimension;
  }
  if (frame.byte_count != expected_byte_count) {
    return absl::DataLossError(
        "Fresh-worker logits byte count does not match its tensor contract.");
  }
  return absl::OkStatus();
}

absl::Status ValidateFreshWorkerResult(const FreshWorkerResult& result) {
  if (result.format_version != kFreshWorkerProtocolVersion) {
    return absl::FailedPreconditionError(
        "Fresh-worker result version is unsupported.");
  }
  if (IsZeroHash(result.exact_profile_hash) ||
      IsZeroHash(result.qualification_id) ||
      IsZeroHash(result.challenge_nonce)) {
    return absl::InvalidArgumentError(
        "Fresh-worker result identity fields must be nonzero.");
  }
  if (result.run_count < 2 ||
      result.run_count > kMaximumFreshWorkerRuns ||
      result.run_index >= result.run_count) {
    return absl::InvalidArgumentError(
        "Fresh-worker result run index or run count is invalid.");
  }
  if (IsZeroHash(result.request_envelope_hash) ||
      IsZeroHash(result.worker_instance_nonce) ||
      IsZeroHash(result.execution_plan_hash)) {
    return absl::InvalidArgumentError(
        "Fresh-worker result evidence bindings must be nonzero.");
  }
  if (result.restored_checkpoint_id.has_value() &&
      IsZeroHash(*result.restored_checkpoint_id)) {
    return absl::InvalidArgumentError(
        "Fresh-worker restored checkpoint ID must be nonzero.");
  }
  if (result.prepared_prefill_plan.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(
        *result.prepared_prefill_plan));
  }
  if (result.restored_state_witness.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(
        *result.restored_state_witness));
  }
  if (result.producing_capsule_evidence.has_value()) {
    ABSL_RETURN_IF_ERROR(ValidateProducingCapsuleEvidence(
        *result.producing_capsule_evidence));
  }
  if (!IsKnownStatusCode(result.status_code) ||
      result.status_message.size() > kMaximumResultStatusMessageBytes) {
    return absl::InvalidArgumentError(
        "Fresh-worker result status is invalid.");
  }
  if (result.status_code == absl::StatusCode::kOk) {
    if (result.replay_isolation !=
        FreshWorkerReplayIsolation::kEmptyCatalogs) {
      return absl::FailedPreconditionError(
          "Successful fresh-worker result lacks the required empty-catalog attestation.");
    }
    if (!result.status_message.empty()) {
      return absl::InvalidArgumentError(
          "Successful fresh-worker result must not carry an error message.");
    }
    ABSL_RETURN_IF_ERROR(ValidateFreshWorkerExecutionOutput(
        FreshWorkerExecutionOutput{.canonical_output = result.canonical_output,
                                   .token_bytes = result.token_bytes,
                                   .logit_frames = result.logit_frames}));
    if (!result.prepared_prefill_plan.has_value()) {
      if (result.restored_checkpoint_id.has_value() ||
          result.restored_state_witness.has_value()) {
        return absl::DataLossError(
            "A successful worker result without a prepared prefill plan "
            "cannot claim restored state.");
      }
    } else {
      const DPMPreparedPrefillPlan& plan =
          *result.prepared_prefill_plan;
      switch (plan.start_kind) {
        case DPMPreparedPrefillStartKind::kFreshSession:
          if (result.restored_checkpoint_id.has_value() ||
              result.restored_state_witness.has_value()) {
            return absl::DataLossError(
                "A fresh prepared prefill result claims restored state.");
          }
          break;
        case DPMPreparedPrefillStartKind::kOwnPositionRestore:
          if (!result.restored_checkpoint_id.has_value() ||
              !result.restored_state_witness.has_value() ||
              plan.restore_checkpoint_id != result.restored_checkpoint_id ||
              plan.start_state_witness_id !=
                  result.restored_state_witness->witness_id ||
              plan.session_identity !=
                  result.restored_state_witness->session_identity ||
              plan.start_step != static_cast<uint64_t>(
                                     result.restored_state_witness->current_step) ||
              plan.start_history_token_count != plan.start_step ||
              plan.start_history_token_bytes_hash !=
                  result.restored_state_witness
                      ->processed_history_token_bytes_hash ||
              result.restored_state_witness->phase !=
                  SessionHandoffPhase::kDecoded ||
              !result.restored_state_witness->ran_decode ||
              result.restored_state_witness->current_step <= 0) {
            return absl::DataLossError(
                "An own-position prepared prefill result lacks its exact "
                "checkpoint and live restored-state witness bindings.");
          }
          break;
        default:
          return absl::InternalError(
              "Validated prepared prefill plan lost its start kind.");
      }
    }
    if (result.producing_capsule_evidence.has_value() &&
        (!result.prepared_prefill_plan.has_value() ||
         result.producing_capsule_evidence->session_identity !=
             result.prepared_prefill_plan->session_identity)) {
      return absl::DataLossError(
          "Fresh-worker producing capsule lacks its prepared prefill plan "
          "or has a different session identity.");
    }
    if (result.producing_capsule_evidence.has_value() &&
        result.producing_capsule_evidence->output_evidence_hash !=
            ComputeFreshWorkerOutputEvidenceHash(
                result.canonical_output, result.token_bytes,
                result.logit_frames)) {
      return absl::DataLossError(
          "Fresh-worker producing capsule is not bound to the returned "
          "output evidence.");
    }
    return absl::OkStatus();
  }
  if (result.replay_isolation != FreshWorkerReplayIsolation::kUnverified ||
      result.status_message.empty() || !result.canonical_output.empty() ||
      !result.token_bytes.empty() ||
      !result.logit_frames.empty() ||
      result.prepared_prefill_plan.has_value() ||
      result.restored_checkpoint_id.has_value() ||
      result.restored_state_witness.has_value() ||
      result.producing_capsule_evidence.has_value()) {
    return absl::InvalidArgumentError(
        "Failed fresh-worker result has non-canonical output, prepared "
        "prefill, restore, or capsule fields.");
  }
  return absl::OkStatus();
}

absl::Status ValidateFreshWorkerResultForRequest(
    const FreshWorkerResult& result, const FreshWorkerRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(request));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerResult(result));
  if (result.format_version != request.format_version ||
      result.exact_profile_hash != request.exact_profile_hash ||
      result.qualification_id != request.qualification_id ||
      result.run_index != request.run_index ||
      result.run_count != request.run_count ||
      result.challenge_nonce != request.challenge_nonce ||
      result.execution_plan_hash != request.execution_plan.plan_hash) {
    return absl::UnauthenticatedError(
        "Fresh-worker result is not bound to its request and execution "
        "plan.");
  }
  if (result.status_code != absl::StatusCode::kOk) {
    return absl::OkStatus();
  }

  switch (request.execution_plan.prefill_mode) {
    case FreshWorkerPrefillMode::kFullCanonicalPrefill:
      if (result.restored_checkpoint_id.has_value() ||
          result.restored_state_witness.has_value() ||
          (result.prepared_prefill_plan.has_value() &&
           result.prepared_prefill_plan->start_kind !=
               DPMPreparedPrefillStartKind::kFreshSession)) {
        return absl::DataLossError(
            "Full-prefill worker result claims a restored start state.");
      }
      break;
    case FreshWorkerPrefillMode::kOwnPositionCapsuleDelta:
      if (!result.restored_checkpoint_id.has_value() ||
          result.restored_checkpoint_id !=
              request.execution_plan.restore_checkpoint_id ||
          !result.restored_state_witness.has_value() ||
          !result.prepared_prefill_plan.has_value() ||
          result.prepared_prefill_plan->start_kind !=
              DPMPreparedPrefillStartKind::kOwnPositionRestore ||
          result.prepared_prefill_plan->restore_checkpoint_id !=
              request.execution_plan.restore_checkpoint_id ||
          result.prepared_prefill_plan->session_identity !=
              request.execution_plan.session_identity ||
          result.restored_state_witness->session_identity !=
              request.execution_plan.session_identity) {
        return absl::DataLossError(
            "Own-position worker result did not confirm its checkpoint, "
            "runtime plan, and live restored-state witness.");
      }
      break;
    default:
      return absl::InternalError(
          "Validated fresh-worker request lost its prefill mode.");
  }

  if (result.producing_capsule_evidence.has_value() !=
      request.execution_plan.capture_producing_capsule) {
    return absl::DataLossError(
        "Fresh-worker result capsule evidence disagrees with capture "
        "policy.");
  }
  if (result.producing_capsule_evidence.has_value() &&
      (!result.prepared_prefill_plan.has_value() ||
       result.producing_capsule_evidence->session_identity !=
           result.prepared_prefill_plan->session_identity ||
       (request.execution_plan.prefill_mode ==
            FreshWorkerPrefillMode::kOwnPositionCapsuleDelta &&
        result.producing_capsule_evidence->session_identity !=
            request.execution_plan.session_identity))) {
    return absl::DataLossError(
        "Fresh-worker producing capsule identity differs from its prepared "
        "or restored session identity.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeFreshWorkerTokenIds(
    const std::vector<int32_t>& token_ids) {
  if (token_ids.size() > kMaximumFreshWorkerTokenIds) {
    return absl::ResourceExhaustedError(
        "Fresh-worker token count exceeds the protocol limit.");
  }
  std::string encoded;
  encoded.reserve(kTokenBytesMagic.size() + 4 + 4 + token_ids.size() * 4);
  encoded.append(kTokenBytesMagic.data(), kTokenBytesMagic.size());
  AppendU32(kFreshWorkerTokenEncodingVersion, &encoded);
  AppendU32(static_cast<uint32_t>(token_ids.size()), &encoded);
  for (int32_t token_id : token_ids) {
    if (token_id < 0) {
      return absl::InvalidArgumentError(
          "Fresh-worker token IDs must be nonnegative int32 values.");
    }
    AppendU32(static_cast<uint32_t>(token_id), &encoded);
  }
  return encoded;
}

absl::StatusOr<std::vector<int32_t>> DecodeFreshWorkerTokenIds(
    absl::string_view canonical_token_bytes) {
  constexpr uint64_t kHeaderBytes = 8 + 4 + 4;
  if (canonical_token_bytes.size() < kHeaderBytes ||
      canonical_token_bytes.size() > kMaximumFreshWorkerTokenBytes ||
      std::memcmp(canonical_token_bytes.data(), kTokenBytesMagic.data(),
                  kTokenBytesMagic.size()) != 0) {
    return absl::DataLossError(
        "Fresh-worker token bytes are not canonical DPMTOK01.");
  }
  Reader reader(canonical_token_bytes.substr(kTokenBytesMagic.size()));
  ABSL_ASSIGN_OR_RETURN(const uint32_t version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t token_count, reader.ReadU32());
  if (version != kFreshWorkerTokenEncodingVersion) {
    return absl::FailedPreconditionError(
        "Fresh-worker token encoding version is unsupported.");
  }
  if (token_count > kMaximumFreshWorkerTokenIds ||
      reader.remaining() != static_cast<uint64_t>(token_count) * 4) {
    return absl::DataLossError(
        "Fresh-worker token encoding length is non-canonical.");
  }
  std::vector<int32_t> token_ids;
  token_ids.reserve(token_count);
  for (uint32_t index = 0; index < token_count; ++index) {
    ABSL_ASSIGN_OR_RETURN(const uint32_t encoded_token_id, reader.ReadU32());
    if (encoded_token_id >
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      return absl::DataLossError(
          "Fresh-worker token encoding contains a negative int32 ID.");
    }
    token_ids.push_back(static_cast<int32_t>(encoded_token_id));
  }
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "Fresh-worker token encoding has trailing bytes.");
  }
  return token_ids;
}

absl::StatusOr<FreshWorkerLogitFrameEvidence>
MakeFreshWorkerLogitFrameEvidence(
    absl::string_view canonical_logit_bytes,
    FreshWorkerLogitElementType element_type, uint32_t batch_size,
    uint32_t sequence_size, uint32_t vocabulary_size) {
  if (canonical_logit_bytes.empty()) {
    return absl::InvalidArgumentError(
        "A logits evidence frame cannot be empty.");
  }
  Sha256Hasher hasher;
  hasher.Update(canonical_logit_bytes);
  ABSL_ASSIGN_OR_RETURN(const uint32_t element_byte_width,
                        ExpectedLogitElementByteWidth(element_type));
  FreshWorkerLogitFrameEvidence frame{
      .element_type = element_type,
      .element_byte_width = element_byte_width,
      .batch_size = batch_size,
      .sequence_size = sequence_size,
      .vocabulary_size = vocabulary_size,
      .byte_count = canonical_logit_bytes.size(),
      .sha256 = hasher.Finalize()};
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerLogitFrameEvidence(frame));
  return frame;
}

absl::StatusOr<std::string> EncodeFreshWorkerRequest(
    const FreshWorkerRequest& request,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(request));
  ABSL_ASSIGN_OR_RETURN(const std::string body, EncodeRequestBody(request));
  return EncodeEnvelope(kRequestMagic, kRequestKind, kRequestMacDomain,
                        body, authentication);
}

absl::StatusOr<FreshWorkerRequest> DecodeFreshWorkerRequest(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication) {
  ABSL_ASSIGN_OR_RETURN(
      const DecodedEnvelopeBody decoded,
      DecodeEnvelope(envelope, kRequestMagic, kRequestKind, kRequestMacDomain,
                     authentication));
  return DecodeRequestBody(decoded.body);
}

absl::StatusOr<std::string> EncodeFreshWorkerResult(
    const FreshWorkerResult& result,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerResult(result));
  ABSL_ASSIGN_OR_RETURN(const std::string body, EncodeResultBody(result));
  return EncodeEnvelope(kResultMagic, kResultKind, kResultMacDomain,
                        body, authentication);
}

absl::StatusOr<FreshWorkerResult> DecodeFreshWorkerResult(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication) {
  ABSL_ASSIGN_OR_RETURN(
      const DecodedEnvelopeBody decoded,
      DecodeEnvelope(envelope, kResultMagic, kResultKind, kResultMacDomain,
                     authentication));
  return DecodeResultBody(decoded.body);
}

Hash256 HashFreshWorkerEnvelope(absl::string_view envelope) {
  Sha256Hasher hasher;
  hasher.Update(envelope);
  return hasher.Finalize();
}

}  // namespace litert::lm
