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
                                                '0', '1'};
constexpr std::array<char, 8> kResultMagic = {'D', 'P', 'M', 'W', 'R', 'S',
                                               '0', '1'};
constexpr std::array<char, 8> kTokenBytesMagic = {'D', 'P', 'M', 'T', 'O', 'K',
                                                   '0', '1'};
constexpr uint32_t kRequestKind = 1;
constexpr uint32_t kResultKind = 2;
constexpr uint64_t kMaximumResultStatusMessageBytes = 4096;
constexpr uint64_t kMaximumKeyIdBytes = 256;
constexpr uint64_t kMaximumAuthenticationKeyBytes = 4096;
constexpr uint64_t kEnvelopeFixedBytes = 8 + 4 + 4 + 4 + 8 + 32;
constexpr absl::string_view kRequestMacDomain =
    "LITERT_LMX_FRESH_WORKER_REQUEST_HMAC_SHA256_V1";
constexpr absl::string_view kResultMacDomain =
    "LITERT_LMX_FRESH_WORKER_RESULT_HMAC_SHA256_V1";

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

std::string EncodeRequestBody(const FreshWorkerRequest& request) {
  std::string body;
  body.reserve(4 + 32 + 32 + 4 + 4 + 32 + 8 +
               request.request_payload.size());
  AppendU32(request.format_version, &body);
  AppendHash(request.exact_profile_hash, &body);
  AppendHash(request.qualification_id, &body);
  AppendU32(request.run_index, &body);
  AppendU32(request.run_count, &body);
  AppendHash(request.challenge_nonce, &body);
  AppendU64(request.request_payload.size(), &body);
  body.append(request.request_payload);
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
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "Fresh-worker request body has trailing bytes.");
  }
  request.request_payload.assign(payload.data(), payload.size());
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(request));
  return request;
}

std::string EncodeResultBody(const FreshWorkerResult& result) {
  std::string body;
  body.reserve(4 + 32 * 5 + 4 * 4 + 8 + result.status_message.size() +
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
  if (request.request_payload.size() >
      kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker request payload exceeds the protocol limit.");
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
  FreshWorkerRequest binding;
  binding.format_version = result.format_version;
  binding.exact_profile_hash = result.exact_profile_hash;
  binding.qualification_id = result.qualification_id;
  binding.run_index = result.run_index;
  binding.run_count = result.run_count;
  binding.challenge_nonce = result.challenge_nonce;
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(binding));
  if (IsZeroHash(result.request_envelope_hash) ||
      IsZeroHash(result.worker_instance_nonce)) {
    return absl::InvalidArgumentError(
        "Fresh-worker result evidence bindings must be nonzero.");
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
    return ValidateFreshWorkerExecutionOutput(
        FreshWorkerExecutionOutput{.canonical_output = result.canonical_output,
                                   .token_bytes = result.token_bytes,
                                   .logit_frames = result.logit_frames});
  }
  if (result.replay_isolation != FreshWorkerReplayIsolation::kUnverified ||
      result.status_message.empty() || !result.canonical_output.empty() ||
      !result.token_bytes.empty() ||
      !result.logit_frames.empty()) {
    return absl::InvalidArgumentError(
        "Failed fresh-worker result has non-canonical output fields.");
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
  AppendU32(kFreshWorkerProtocolVersion, &encoded);
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
  if (version != kFreshWorkerProtocolVersion) {
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
  return EncodeEnvelope(kRequestMagic, kRequestKind, kRequestMacDomain,
                        EncodeRequestBody(request), authentication);
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
  return EncodeEnvelope(kResultMagic, kResultKind, kResultMacDomain,
                        EncodeResultBody(result), authentication);
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
