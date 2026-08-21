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

#include "runtime/core/session_handoff_codec.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ios>
#include <limits>
#include <locale>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {
namespace {

constexpr size_t kMacSize = 32;
constexpr size_t kHmacBlockSize = 64;
constexpr size_t kMinimumKeySize = 32;
constexpr uint32_t kMaximumKeyIdSize = 1024;
constexpr uint32_t kMaximumRandomEngineStateSize = 64 * 1024;
constexpr uint32_t kSupportedCandidateCount = 1;
constexpr uint32_t kMaximumTokensPerCandidate = 16 * 1024 * 1024;
constexpr size_t kMaximumTotalTokenIds = 16 * 1024 * 1024;
constexpr uint32_t kMaximumWitnessDpmTokenIds = 1'000'000;
constexpr size_t kIoChunkSize = 2 * 1024 * 1024;
constexpr size_t kTokenWriteChunkSize = 64 * 1024;
constexpr absl::string_view kContinuationStateWitnessDomain =
    "LITERT_LMX_SESSION_CONTINUATION_STATE_WITNESS_SHA256_V1";
constexpr absl::string_view kContinuationStateWitnessContractDomain =
    "LITERT_LMX_SESSION_CONTINUATION_STATE_WITNESS_CONTRACT_SHA256_V1";

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

void AppendWitnessU32(uint32_t value, std::string* output) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendWitnessI32(int32_t value, std::string* output) {
  AppendWitnessU32(std::bit_cast<uint32_t>(value), output);
}

void AppendWitnessU64(uint64_t value, std::string* output) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendWitnessHash(const Hash256& hash, std::string* output) {
  output->append(reinterpret_cast<const char*>(hash.bytes.data()),
                 hash.bytes.size());
}

absl::Status ValidateContinuationStateWitnessFields(
    const SessionContinuationStateWitness& witness,
    bool require_witness_id) {
  if (witness.format_version !=
      SessionContinuationStateWitness::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Unsupported session continuation-state witness version.");
  }
  if ((require_witness_id && IsZeroHash(witness.witness_id)) ||
      IsZeroHash(witness.session_identity.model_artifact_hash) ||
      IsZeroHash(witness.session_identity.runtime_artifact_hash) ||
      IsZeroHash(witness.session_identity.inference_profile_hash) ||
      witness.current_step < 0 ||
      static_cast<uint32_t>(witness.current_step) >
          kMaximumWitnessDpmTokenIds ||
      IsZeroHash(witness.processed_history_token_bytes_hash) ||
      IsZeroHash(witness.envelope_hash) || witness.envelope_size == 0 ||
      witness.envelope_size > kMaximumSessionHandoffEnvelopeBytes ||
      witness.key_id.empty() || witness.key_id.size() > kMaximumKeyIdSize) {
    return absl::InvalidArgumentError(
        "Session continuation-state witness is incomplete or outside its "
        "canonical bounds.");
  }
  switch (witness.phase) {
    case SessionHandoffPhase::kFresh:
      if (witness.current_step != 0 || witness.ran_decode) {
        return absl::InvalidArgumentError(
            "Fresh session continuation-state witness contains non-fresh "
            "state.");
      }
      break;
    case SessionHandoffPhase::kPrefilled:
      if (witness.current_step == 0 || witness.ran_decode) {
        return absl::InvalidArgumentError(
            "Prefilled session continuation-state witness has inconsistent "
            "state.");
      }
      break;
    case SessionHandoffPhase::kDecoded:
      if (witness.current_step == 0 || !witness.ran_decode) {
        return absl::InvalidArgumentError(
            "Decoded session continuation-state witness has inconsistent "
            "state.");
      }
      break;
    default:
      return absl::InvalidArgumentError(
          "Session continuation-state witness has an unknown phase.");
  }
  return absl::OkStatus();
}

std::string EncodeContinuationStateWitnessFields(
    const SessionContinuationStateWitness& witness) {
  std::string canonical;
  canonical.reserve(4 + 96 + 1 + 4 + 1 + 32 + 32 + 8 + 4 +
                    witness.key_id.size());
  AppendWitnessU32(witness.format_version, &canonical);
  AppendWitnessHash(witness.session_identity.model_artifact_hash,
                    &canonical);
  AppendWitnessHash(witness.session_identity.runtime_artifact_hash,
                    &canonical);
  AppendWitnessHash(witness.session_identity.inference_profile_hash,
                    &canonical);
  canonical.push_back(static_cast<char>(witness.phase));
  AppendWitnessI32(witness.current_step, &canonical);
  canonical.push_back(witness.ran_decode ? '\1' : '\0');
  AppendWitnessHash(witness.processed_history_token_bytes_hash, &canonical);
  AppendWitnessHash(witness.envelope_hash, &canonical);
  AppendWitnessU64(witness.envelope_size, &canonical);
  AppendWitnessU32(static_cast<uint32_t>(witness.key_id.size()), &canonical);
  canonical.append(witness.key_id);
  return canonical;
}

absl::Status ValidateOptions(
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options) {
  if (IsZeroHash(authoritative_identity.model_artifact_hash) ||
      IsZeroHash(authoritative_identity.runtime_artifact_hash) ||
      IsZeroHash(authoritative_identity.inference_profile_hash)) {
    return absl::InvalidArgumentError(
        "Engine-derived session handoff identity hashes must all be nonzero.");
  }
  if (options.expected_identity.has_value() &&
      *options.expected_identity != authoritative_identity) {
    return absl::FailedPreconditionError(
        "Session handoff identity assertion does not match the loaded "
        "session.");
  }
  if (options.key_id.empty() || options.key_id.size() > kMaximumKeyIdSize) {
    return absl::InvalidArgumentError(
        "Session handoff key_id must contain 1 to 1024 bytes.");
  }
  if (options.authentication_key.size() < kMinimumKeySize) {
    return absl::InvalidArgumentError(
        "Session handoff authentication key must contain at least 32 bytes.");
  }
  return absl::OkStatus();
}

absl::Status ValidateSnapshot(const SessionHandoffSnapshot& snapshot,
                              uint64_t serialized_state_size) {
  const auto& executor = snapshot.executor;
  if (executor.current_step < 0) {
    return absl::InvalidArgumentError(
        "Session handoff current_step must be nonnegative.");
  }
  if (executor.last_prefill_token_id < 0) {
    return absl::UnimplementedError(
        "Session handoff does not support multimodal prefill token IDs.");
  }
  if (serialized_state_size == 0 &&
      snapshot.phase != SessionHandoffPhase::kFresh) {
    return absl::InvalidArgumentError(
        "Non-fresh session handoff must include serialized LiteRT state.");
  }
  if (!executor.runtime_config.sampler_params.has_value()) {
    return absl::InvalidArgumentError(
        "Session handoff must include resolved sampler parameters.");
  }
  const proto::SamplerParameters& sampler =
      *executor.runtime_config.sampler_params;
  if (!proto::SamplerParameters::Type_IsValid(sampler.type()) ||
      sampler.type() != proto::SamplerParameters::GREEDY) {
    return absl::UnimplementedError(
        "Session handoff currently supports only GREEDY sampling.");
  }
  if (!proto::SamplerParameters::Backend_IsValid(sampler.backend()) ||
      sampler.backend() != proto::SamplerParameters::CPU) {
    return absl::UnimplementedError(
        "Session handoff currently supports only CPU sampling.");
  }
  if (!executor.runtime_config.output_heads.has_value() ||
      *executor.runtime_config.output_heads != kSupportedCandidateCount) {
    return absl::UnimplementedError(
        "Session handoff currently supports exactly one output candidate.");
  }
  if (executor.runtime_config.tokens_per_decode.has_value() &&
      *executor.runtime_config.tokens_per_decode <= 0) {
    return absl::InvalidArgumentError(
        "Session handoff tokens_per_decode must be positive when present.");
  }
  if (executor.runtime_config.tokens_per_decode.has_value() &&
      *executor.runtime_config.tokens_per_decode != 1) {
    return absl::UnimplementedError(
        "Session handoff currently supports one token per decode step.");
  }
  const auto& tokens = executor.processed_tokens;
  if (tokens.processed_token_ids.size() != kSupportedCandidateCount) {
    return absl::UnimplementedError(
        "Session handoff currently supports exactly one token candidate.");
  }
  const size_t processed_count = tokens.processed_token_ids[0].size();
  if (processed_count > kMaximumTokensPerCandidate) {
    return absl::ResourceExhaustedError(
        "Session handoff token history exceeds the supported limit.");
  }
  size_t total_token_ids = 0;
  for (const auto& candidate : tokens.processed_token_ids) {
    if (candidate.size() != processed_count) {
      return absl::InvalidArgumentError(
          "Session handoff candidates have different processed lengths.");
    }
    if (candidate.size() > kMaximumTotalTokenIds - total_token_ids) {
      return absl::ResourceExhaustedError(
          "Session handoff total token history exceeds the supported limit.");
    }
    total_token_ids += candidate.size();
  }
  if (tokens.pending_token_ids.size() >
      kMaximumTotalTokenIds - total_token_ids) {
    return absl::ResourceExhaustedError(
        "Session handoff total token history exceeds the supported limit.");
  }
  for (const auto& candidate : tokens.processed_token_ids) {
    for (int token_id : candidate) {
      if (token_id < 0) {
        return absl::UnimplementedError(
            "Session handoff does not support multimodal token IDs.");
      }
    }
  }
  if (!tokens.pending_token_ids.empty() &&
      tokens.pending_token_ids.size() != tokens.processed_token_ids.size()) {
    return absl::InvalidArgumentError(
        "Session handoff pending-token count does not match candidates.");
  }
  for (int token_id : tokens.pending_token_ids) {
    if (token_id < 0) {
      return absl::UnimplementedError(
          "Session handoff does not support multimodal pending tokens.");
    }
  }
  const size_t token_count =
      processed_count + (tokens.pending_token_ids.empty() ? 0 : 1);
  if (token_count > static_cast<size_t>(std::numeric_limits<int>::max()) ||
      executor.current_step != static_cast<int>(token_count)) {
    return absl::InvalidArgumentError(
        "Session handoff current_step does not match token history.");
  }
  switch (snapshot.phase) {
    case SessionHandoffPhase::kFresh:
      if (executor.current_step != 0 || executor.ran_decode ||
          tokens.processed_token_ids.size() != 1 ||
          !tokens.pending_token_ids.empty() ||
          executor.last_prefill_token_id != 0) {
        return absl::InvalidArgumentError(
            "Fresh session handoff contains non-fresh state.");
      }
      break;
    case SessionHandoffPhase::kPrefilled:
      if (executor.current_step == 0 || executor.ran_decode ||
          tokens.processed_token_ids.size() != 1 ||
          tokens.pending_token_ids.size() != 1 ||
          tokens.pending_token_ids[0] != executor.last_prefill_token_id) {
        return absl::InvalidArgumentError(
            "Prefilled session handoff has inconsistent executor state.");
      }
      break;
    case SessionHandoffPhase::kDecoded:
      if (executor.current_step == 0 || !executor.ran_decode ||
          tokens.pending_token_ids.empty()) {
        return absl::InvalidArgumentError(
            "Decoded session handoff has inconsistent executor state.");
      }
      break;
    default:
      return absl::InvalidArgumentError(
          "Session handoff contains an unknown session phase.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> SerializeRandomEngine(
    const std::default_random_engine& engine) {
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << engine;
  if (!stream.good()) {
    return absl::InternalError(
        "Failed to serialize session sampler random engine.");
  }
  std::string result = stream.str();
  if (result.empty() || result.size() > kMaximumRandomEngineStateSize) {
    return absl::InternalError(
        "Serialized session sampler random engine has an invalid size.");
  }
  return result;
}

absl::Status ValidateEncodedEnvelopeSize(
    const SessionHandoffSnapshot& snapshot, uint64_t serialized_state_size,
    size_t key_id_size, size_t random_engine_state_size) {
  uint64_t total_size = 0;
  const auto add = [&total_size](uint64_t byte_count) -> bool {
    if (byte_count >
        kMaximumSessionHandoffEnvelopeBytes - total_size) {
      return false;
    }
    total_size += byte_count;
    return true;
  };

  // Fixed body fields through runtime_config_flags, including the three
  // identity hashes and key-id length, but excluding the key-id bytes.
  if (!add(kSessionHandoffEnvelopeMagic.size() + sizeof(uint32_t) +
           3 * Hash256{}.bytes.size() + sizeof(uint32_t) +
           sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint16_t) +
           sizeof(int32_t) + sizeof(int32_t) + sizeof(uint8_t)) ||
      !add(key_id_size)) {
    return absl::ResourceExhaustedError(
        "Session handoff metadata exceeds the supported transport limit.");
  }

  const auto& runtime_config = snapshot.executor.runtime_config;
  if (runtime_config.sampler_params.has_value()) {
    // type, k, p, temperature, has_seed, optional seed, backend.
    if (!add(4 + 4 + 4 + 4 + 1 +
             (runtime_config.sampler_params->has_seed() ? 4 : 0) + 4)) {
      return absl::ResourceExhaustedError(
          "Session handoff sampler metadata exceeds the transport limit.");
    }
  }
  if (runtime_config.output_heads.has_value() && !add(sizeof(int32_t))) {
    return absl::ResourceExhaustedError(
        "Session handoff runtime metadata exceeds the transport limit.");
  }
  if (runtime_config.tokens_per_decode.has_value() && !add(sizeof(int32_t))) {
    return absl::ResourceExhaustedError(
        "Session handoff runtime metadata exceeds the transport limit.");
  }
  if (!add(sizeof(uint32_t)) || !add(random_engine_state_size) ||
      !add(sizeof(uint32_t) + sizeof(uint32_t))) {
    return absl::ResourceExhaustedError(
        "Session handoff runtime metadata exceeds the transport limit.");
  }

  const auto& tokens = snapshot.executor.processed_tokens;
  const uint64_t candidate_count = tokens.processed_token_ids.size();
  const uint64_t processed_count = tokens.processed_token_ids.front().size();
  if (candidate_count != 0 &&
      processed_count >
          kMaximumSessionHandoffEnvelopeBytes / candidate_count / 4) {
    return absl::ResourceExhaustedError(
        "Session handoff token metadata exceeds the transport limit.");
  }
  if (!add(candidate_count * processed_count * 4) ||
      !add(sizeof(uint8_t)) ||
      (!tokens.pending_token_ids.empty() && !add(candidate_count * 4)) ||
      !add(sizeof(uint64_t)) || !add(serialized_state_size) ||
      !add(kMacSize)) {
    return absl::ResourceExhaustedError(
        "Session handoff envelope exceeds the supported transport limit.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::default_random_engine> ParseRandomEngine(
    absl::string_view serialized) {
  if (serialized.empty() || serialized.size() > kMaximumRandomEngineStateSize) {
    return absl::DataLossError(
        "Session handoff random engine state has an invalid size.");
  }
  std::istringstream stream{std::string(serialized)};
  stream.imbue(std::locale::classic());
  std::default_random_engine engine;
  stream >> engine;
  if (stream.fail()) {
    return absl::DataLossError(
        "Session handoff random engine state is malformed.");
  }
  stream >> std::ws;
  if (!stream.eof()) {
    return absl::DataLossError(
        "Session handoff random engine state has trailing data.");
  }
  ABSL_ASSIGN_OR_RETURN(std::string canonical, SerializeRandomEngine(engine));
  if (canonical != serialized) {
    return absl::DataLossError(
        "Session handoff random engine state is not canonical.");
  }
  return engine;
}

class HmacSha256State {
 public:
  explicit HmacSha256State(absl::string_view key) {
    std::array<uint8_t, kHmacBlockSize> normalized_key{};
    if (key.size() > normalized_key.size()) {
      Sha256Hasher key_hasher;
      const Hash256 key_hash = key_hasher.OneShot(key);
      std::copy(key_hash.bytes.begin(), key_hash.bytes.end(),
                normalized_key.begin());
    } else if (!key.empty()) {
      std::memcpy(normalized_key.data(), key.data(), key.size());
    }
    std::array<char, kHmacBlockSize> inner_pad{};
    for (size_t i = 0; i < normalized_key.size(); ++i) {
      inner_pad[i] = static_cast<char>(normalized_key[i] ^ 0x36);
      outer_pad_[i] = static_cast<char>(normalized_key[i] ^ 0x5c);
    }
    inner_.Update(absl::string_view(inner_pad.data(), inner_pad.size()));
  }

  void Update(absl::string_view bytes) { inner_.Update(bytes); }

  Hash256 Finalize() {
    const Hash256 inner_hash = inner_.Finalize();
    Sha256Hasher outer;
    outer.Update(absl::string_view(outer_pad_.data(), outer_pad_.size()));
    outer.Update(absl::string_view(
        reinterpret_cast<const char*>(inner_hash.bytes.data()),
        inner_hash.bytes.size()));
    return outer.Finalize();
  }

 private:
  Sha256Hasher inner_;
  std::array<char, kHmacBlockSize> outer_pad_{};
};

class AuthenticatedByteSink final : public ByteSink {
 public:
  AuthenticatedByteSink(ByteSink* downstream, HmacSha256State* hmac)
      : downstream_(downstream), hmac_(hmac) {}

  absl::Status Append(absl::string_view bytes) override {
    if (downstream_ == nullptr || hmac_ == nullptr) {
      return absl::FailedPreconditionError(
          "Session handoff authenticated sink is incomplete.");
    }
    ABSL_RETURN_IF_ERROR(downstream_->Append(bytes));
    hmac_->Update(bytes);
    return absl::OkStatus();
  }

 private:
  ByteSink* downstream_;
  HmacSha256State* hmac_;
};

class ExactSizeByteSink final : public ByteSink {
 public:
  ExactSizeByteSink(ByteSink* downstream, uint64_t expected_size)
      : downstream_(downstream), expected_size_(expected_size) {}

  absl::Status Append(absl::string_view bytes) override {
    if (downstream_ == nullptr) {
      return absl::FailedPreconditionError(
          "Session handoff state sink is incomplete.");
    }
    if (bytes.size() > expected_size_ - emitted_size_) {
      return absl::DataLossError(
          "LiteRT state emitted more bytes than its reported size.");
    }
    ABSL_RETURN_IF_ERROR(downstream_->Append(bytes));
    emitted_size_ += bytes.size();
    return absl::OkStatus();
  }

  absl::Status Finish() const {
    if (emitted_size_ != expected_size_) {
      return absl::DataLossError(
          "LiteRT state emitted fewer bytes than its reported size.");
    }
    return absl::OkStatus();
  }

 private:
  ByteSink* downstream_;
  uint64_t expected_size_;
  uint64_t emitted_size_ = 0;
};

absl::Status WriteU8(ByteSink* sink, uint8_t value) {
  const char byte = static_cast<char>(value);
  return sink->Append(absl::string_view(&byte, 1));
}

absl::Status WriteU16(ByteSink* sink, uint16_t value) {
  const std::array<char, 2> bytes = {
      static_cast<char>((value >> 8) & 0xff),
      static_cast<char>(value & 0xff),
  };
  return sink->Append(absl::string_view(bytes.data(), bytes.size()));
}

absl::Status WriteU32(ByteSink* sink, uint32_t value) {
  std::array<char, 4> bytes;
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (24 - 8 * i)) & 0xff);
  }
  return sink->Append(absl::string_view(bytes.data(), bytes.size()));
}

absl::Status WriteI32(ByteSink* sink, int32_t value) {
  return WriteU32(sink, std::bit_cast<uint32_t>(value));
}

absl::Status WriteFloat(ByteSink* sink, float value) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  static_assert(std::numeric_limits<float>::is_iec559);
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return WriteU32(sink, bits);
}

absl::Status WriteU64(ByteSink* sink, uint64_t value) {
  std::array<char, 8> bytes;
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (56 - 8 * i)) & 0xff);
  }
  return sink->Append(absl::string_view(bytes.data(), bytes.size()));
}

absl::Status WriteHash(ByteSink* sink, const Hash256& hash) {
  return sink->Append(absl::string_view(
      reinterpret_cast<const char*>(hash.bytes.data()), hash.bytes.size()));
}

absl::Status WriteTokenIds(ByteSink* sink, absl::Span<const int> token_ids) {
  std::array<char, kTokenWriteChunkSize> bytes;
  size_t used = 0;
  for (int token_id : token_ids) {
    const uint32_t value = std::bit_cast<uint32_t>(token_id);
    for (size_t i = 0; i < sizeof(value); ++i) {
      bytes[used++] = static_cast<char>((value >> (24 - 8 * i)) & 0xff);
    }
    if (used == bytes.size()) {
      ABSL_RETURN_IF_ERROR(
          sink->Append(absl::string_view(bytes.data(), bytes.size())));
      used = 0;
    }
  }
  if (used != 0) {
    ABSL_RETURN_IF_ERROR(sink->Append(absl::string_view(bytes.data(), used)));
  }
  return absl::OkStatus();
}

absl::Status EncodeSessionHandoffParts(
    const SessionHandoffSnapshot& snapshot, uint64_t serialized_state_size,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options, ByteSink* sink,
    absl::FunctionRef<absl::Status(ByteSink*)> write_serialized_state) {
  if (sink == nullptr) {
    return absl::InvalidArgumentError(
        "Session handoff output sink must not be null.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateOptions(authoritative_identity, options));
  ABSL_RETURN_IF_ERROR(ValidateSnapshot(snapshot, serialized_state_size));
  ABSL_ASSIGN_OR_RETURN(std::string random_engine_state,
                        SerializeRandomEngine(snapshot.executor.random_engine));
  ABSL_RETURN_IF_ERROR(ValidateEncodedEnvelopeSize(
      snapshot, serialized_state_size, options.key_id.size(),
      random_engine_state.size()));

  HmacSha256State hmac(options.authentication_key);
  AuthenticatedByteSink body(sink, &hmac);
  ABSL_RETURN_IF_ERROR(
      body.Append(absl::string_view(kSessionHandoffEnvelopeMagic.data(),
                                   kSessionHandoffEnvelopeMagic.size())));
  ABSL_RETURN_IF_ERROR(WriteU32(&body, kSessionHandoffEnvelopeVersion));
  ABSL_RETURN_IF_ERROR(
      WriteHash(&body, authoritative_identity.model_artifact_hash));
  ABSL_RETURN_IF_ERROR(
      WriteHash(&body, authoritative_identity.runtime_artifact_hash));
  ABSL_RETURN_IF_ERROR(
      WriteHash(&body, authoritative_identity.inference_profile_hash));
  ABSL_RETURN_IF_ERROR(
      WriteU32(&body, static_cast<uint32_t>(options.key_id.size())));
  ABSL_RETURN_IF_ERROR(body.Append(options.key_id));
  ABSL_RETURN_IF_ERROR(WriteU8(&body, static_cast<uint8_t>(snapshot.phase)));
  ABSL_RETURN_IF_ERROR(WriteU8(&body, snapshot.executor.ran_decode ? 1 : 0));
  ABSL_RETURN_IF_ERROR(WriteU16(&body, 0));
  ABSL_RETURN_IF_ERROR(WriteI32(&body, snapshot.executor.current_step));
  ABSL_RETURN_IF_ERROR(
      WriteI32(&body, snapshot.executor.last_prefill_token_id));

  uint8_t runtime_config_flags = 0;
  if (snapshot.executor.runtime_config.sampler_params.has_value()) {
    runtime_config_flags |= 1;
  }
  if (snapshot.executor.runtime_config.output_heads.has_value()) {
    runtime_config_flags |= 2;
  }
  if (snapshot.executor.runtime_config.tokens_per_decode.has_value()) {
    runtime_config_flags |= 4;
  }
  ABSL_RETURN_IF_ERROR(WriteU8(&body, runtime_config_flags));
  if (snapshot.executor.runtime_config.sampler_params.has_value()) {
    const proto::SamplerParameters& sampler =
        *snapshot.executor.runtime_config.sampler_params;
    ABSL_RETURN_IF_ERROR(
        WriteI32(&body, static_cast<int32_t>(sampler.type())));
    ABSL_RETURN_IF_ERROR(WriteI32(&body, sampler.k()));
    ABSL_RETURN_IF_ERROR(WriteFloat(&body, sampler.p()));
    ABSL_RETURN_IF_ERROR(WriteFloat(&body, sampler.temperature()));
    ABSL_RETURN_IF_ERROR(WriteU8(&body, sampler.has_seed() ? 1 : 0));
    if (sampler.has_seed()) {
      ABSL_RETURN_IF_ERROR(WriteI32(&body, sampler.seed()));
    }
    ABSL_RETURN_IF_ERROR(
        WriteI32(&body, static_cast<int32_t>(sampler.backend())));
  }
  if (snapshot.executor.runtime_config.output_heads.has_value()) {
    ABSL_RETURN_IF_ERROR(
        WriteI32(&body, *snapshot.executor.runtime_config.output_heads));
  }
  if (snapshot.executor.runtime_config.tokens_per_decode.has_value()) {
    ABSL_RETURN_IF_ERROR(
        WriteI32(&body, *snapshot.executor.runtime_config.tokens_per_decode));
  }
  ABSL_RETURN_IF_ERROR(
      WriteU32(&body, static_cast<uint32_t>(random_engine_state.size())));
  ABSL_RETURN_IF_ERROR(body.Append(random_engine_state));

  const auto& tokens = snapshot.executor.processed_tokens;
  ABSL_RETURN_IF_ERROR(WriteU32(
      &body, static_cast<uint32_t>(tokens.processed_token_ids.size())));
  ABSL_RETURN_IF_ERROR(WriteU32(
      &body, static_cast<uint32_t>(tokens.processed_token_ids[0].size())));
  for (const auto& candidate : tokens.processed_token_ids) {
    ABSL_RETURN_IF_ERROR(WriteTokenIds(&body, absl::MakeConstSpan(candidate)));
  }
  ABSL_RETURN_IF_ERROR(
      WriteU8(&body, tokens.pending_token_ids.empty() ? 0 : 1));
  if (!tokens.pending_token_ids.empty()) {
    ABSL_RETURN_IF_ERROR(
        WriteTokenIds(&body, absl::MakeConstSpan(tokens.pending_token_ids)));
  }

  ABSL_RETURN_IF_ERROR(WriteU64(&body, serialized_state_size));
  ExactSizeByteSink exact_state(&body, serialized_state_size);
  ABSL_RETURN_IF_ERROR(write_serialized_state(&exact_state));
  ABSL_RETURN_IF_ERROR(exact_state.Finish());

  const Hash256 mac = hmac.Finalize();
  return sink->Append(absl::string_view(
      reinterpret_cast<const char*>(mac.bytes.data()), mac.bytes.size()));
}

absl::StatusOr<Hash256> HmacSha256Source(absl::string_view key,
                                         const ByteSource& source,
                                         uint64_t size) {
  if (size > source.Size()) {
    return absl::OutOfRangeError(
        "Session handoff authentication range exceeds its byte source.");
  }
  HmacSha256State hmac(key);
  std::string chunk(
      static_cast<size_t>(std::min<uint64_t>(size, kIoChunkSize)), '\0');
  uint64_t offset = 0;
  while (offset < size) {
    const size_t count = static_cast<size_t>(
        std::min<uint64_t>(chunk.size(), size - offset));
    ABSL_RETURN_IF_ERROR(
        source.ReadAt(offset, absl::MakeSpan(chunk).subspan(0, count)));
    hmac.Update(absl::string_view(chunk.data(), count));
    offset += count;
  }
  return hmac.Finalize();
}

bool ConstantTimeEquals(absl::Span<const char> bytes, const Hash256& hash) {
  if (bytes.size() != hash.bytes.size()) return false;
  uint8_t difference = 0;
  for (size_t i = 0; i < hash.bytes.size(); ++i) {
    difference |= static_cast<uint8_t>(bytes[i]) ^ hash.bytes[i];
  }
  return difference == 0;
}

class SourceReader {
 public:
  SourceReader(const ByteSource& source, uint64_t limit)
      : source_(source), limit_(limit) {}

  absl::StatusOr<uint8_t> U8() {
    std::array<char, 1> bytes;
    ABSL_RETURN_IF_ERROR(Read(absl::MakeSpan(bytes)));
    return static_cast<uint8_t>(bytes[0]);
  }

  absl::StatusOr<uint16_t> U16() {
    std::array<char, 2> bytes;
    ABSL_RETURN_IF_ERROR(Read(absl::MakeSpan(bytes)));
    return (static_cast<uint16_t>(static_cast<unsigned char>(bytes[0])) << 8) |
           static_cast<uint16_t>(static_cast<unsigned char>(bytes[1]));
  }

  absl::StatusOr<uint32_t> U32() {
    std::array<char, 4> bytes;
    ABSL_RETURN_IF_ERROR(Read(absl::MakeSpan(bytes)));
    uint32_t value = 0;
    for (char byte : bytes) {
      value = (value << 8) | static_cast<unsigned char>(byte);
    }
    return value;
  }

  absl::StatusOr<int32_t> I32() {
    ABSL_ASSIGN_OR_RETURN(uint32_t value, U32());
    return std::bit_cast<int32_t>(value);
  }

  absl::StatusOr<float> Float() {
    ABSL_ASSIGN_OR_RETURN(uint32_t bits, U32());
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }

  absl::StatusOr<uint64_t> U64() {
    std::array<char, 8> bytes;
    ABSL_RETURN_IF_ERROR(Read(absl::MakeSpan(bytes)));
    uint64_t value = 0;
    for (char byte : bytes) {
      value = (value << 8) | static_cast<unsigned char>(byte);
    }
    return value;
  }

  absl::StatusOr<Hash256> Hash() {
    std::array<char, 32> bytes;
    ABSL_RETURN_IF_ERROR(Read(absl::MakeSpan(bytes)));
    Hash256 hash;
    std::memcpy(hash.bytes.data(), bytes.data(), bytes.size());
    return hash;
  }

  absl::StatusOr<std::string> String(uint64_t size) {
    if (size > std::numeric_limits<size_t>::max()) {
      return absl::ResourceExhaustedError(
          "Session handoff field exceeds addressable memory.");
    }
    std::string bytes(static_cast<size_t>(size), '\0');
    ABSL_RETURN_IF_ERROR(Read(absl::MakeSpan(bytes)));
    return bytes;
  }

  absl::Status Read(absl::Span<char> destination) {
    if (destination.size() > remaining()) {
      return absl::DataLossError("Session handoff envelope is truncated.");
    }
    ABSL_RETURN_IF_ERROR(source_.ReadAt(offset_, destination));
    offset_ += destination.size();
    return absl::OkStatus();
  }

  absl::Status Skip(uint64_t size) {
    if (size > remaining()) {
      return absl::DataLossError("Session handoff envelope is truncated.");
    }
    offset_ += size;
    return absl::OkStatus();
  }

  uint64_t offset() const { return offset_; }
  uint64_t remaining() const { return limit_ - offset_; }
  bool empty() const { return offset_ == limit_; }

 private:
  const ByteSource& source_;
  uint64_t limit_;
  uint64_t offset_ = 0;
};

}  // namespace

Hash256 GetSessionContinuationStateWitnessContractHash() {
  std::string canonical;
  const auto append_frame = [&canonical](absl::string_view value) {
    AppendWitnessU32(static_cast<uint32_t>(value.size()), &canonical);
    canonical.append(value.data(), value.size());
  };
  AppendWitnessU32(SessionContinuationStateWitness::kFormatVersion,
                   &canonical);
  AppendWitnessU64(kMaximumSessionHandoffEnvelopeBytes, &canonical);
  AppendWitnessU32(kMaximumWitnessDpmTokenIds, &canonical);
  append_frame("ENGINE_DERIVED_SESSION_IDENTITY");
  append_frame("DPMTOK01_COMPLETE_HISTORY_WITH_PENDING_TOKEN");
  append_frame("ENVELOPE_SHA256_SIZE_AND_KEY_ID");
  append_frame("IMPORT_TARGET_CANONICAL_REEXPORT_EQUALITY");
  Sha256Hasher hasher;
  hasher.Update(kContinuationStateWitnessContractDomain);
  hasher.Update(canonical);
  return hasher.Finalize();
}

absl::StatusOr<Hash256> ComputeSessionContinuationStateWitnessId(
    const SessionContinuationStateWitness& witness) {
  ABSL_RETURN_IF_ERROR(
      ValidateContinuationStateWitnessFields(witness, false));
  Sha256Hasher hasher;
  hasher.Update(kContinuationStateWitnessDomain);
  hasher.Update(EncodeContinuationStateWitnessFields(witness));
  const Hash256 witness_id = hasher.Finalize();
  if (IsZeroHash(witness_id)) {
    return absl::InternalError(
        "Session continuation-state witness produced a zero ID.");
  }
  return witness_id;
}

absl::Status ValidateSessionContinuationStateWitness(
    const SessionContinuationStateWitness& witness) {
  ABSL_RETURN_IF_ERROR(
      ValidateContinuationStateWitnessFields(witness, true));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 canonical_id,
      ComputeSessionContinuationStateWitnessId(witness));
  if (witness.witness_id != canonical_id) {
    return absl::DataLossError(
        "Session continuation-state witness ID is not canonical.");
  }
  return absl::OkStatus();
}

absl::Status EncodeSessionHandoffTo(
    const SessionHandoffSnapshot& snapshot, const StateInterface& state,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options, ByteSink* sink) {
  if (!snapshot.executor.serialized_state.empty()) {
    return absl::InvalidArgumentError(
        "Streamed session handoff metadata must not duplicate state bytes.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateOptions(authoritative_identity, options));
  ABSL_RETURN_IF_ERROR(ValidateSnapshot(
      snapshot, snapshot.phase == SessionHandoffPhase::kFresh ? 0 : 1));
  uint64_t serialized_state_size = 0;
  if (snapshot.phase != SessionHandoffPhase::kFresh) {
    ABSL_ASSIGN_OR_RETURN(serialized_state_size, state.SerializedSize());
  }
  return EncodeSessionHandoffParts(
      snapshot, serialized_state_size, authoritative_identity, options, sink,
      [&state, serialized_state_size](ByteSink* state_sink) {
        return serialized_state_size == 0 ? absl::OkStatus()
                                          : state.SerializeTo(state_sink);
      });
}

absl::StatusOr<std::string> EncodeSessionHandoff(
    const SessionHandoffSnapshot& snapshot,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options) {
  const uint64_t serialized_state_size =
      snapshot.phase == SessionHandoffPhase::kFresh
          ? 0
          : static_cast<uint64_t>(
                snapshot.executor.serialized_state.size());
  std::string envelope;
  StringByteSink sink(&envelope);
  ABSL_RETURN_IF_ERROR(EncodeSessionHandoffParts(
      snapshot, serialized_state_size, authoritative_identity, options, &sink,
      [&snapshot, serialized_state_size](ByteSink* state_sink) {
        return serialized_state_size == 0
                   ? absl::OkStatus()
                   : state_sink->Append(snapshot.executor.serialized_state);
      }));
  return envelope;
}

absl::StatusOr<DecodedSessionHandoff> DecodeSessionHandoffFrom(
    const ByteSource& envelope,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options) {
  ABSL_RETURN_IF_ERROR(
      ValidateOptions(authoritative_identity, options));
  constexpr uint64_t kMinimumEnvelopeSize =
      kSessionHandoffEnvelopeMagic.size() + sizeof(uint32_t) + kMacSize;
  if (envelope.Size() < kMinimumEnvelopeSize ||
      envelope.Size() > kMaximumSessionHandoffEnvelopeBytes) {
    return absl::DataLossError(
        "Session handoff envelope size is outside the supported range.");
  }
  const uint64_t body_size = envelope.Size() - kMacSize;
  ABSL_ASSIGN_OR_RETURN(
      Hash256 computed_mac,
      HmacSha256Source(options.authentication_key, envelope, body_size));
  std::array<char, kMacSize> supplied_mac;
  ABSL_RETURN_IF_ERROR(
      envelope.ReadAt(body_size, absl::MakeSpan(supplied_mac)));
  if (!ConstantTimeEquals(absl::MakeConstSpan(supplied_mac), computed_mac)) {
    return absl::UnauthenticatedError("Session handoff authentication failed.");
  }

  SourceReader reader(envelope, body_size);
  std::array<char, kSessionHandoffEnvelopeMagic.size()> magic;
  ABSL_RETURN_IF_ERROR(reader.Read(absl::MakeSpan(magic)));
  if (magic != kSessionHandoffEnvelopeMagic) {
    return absl::DataLossError("Session handoff envelope has invalid magic.");
  }
  ABSL_ASSIGN_OR_RETURN(uint32_t version, reader.U32());
  if (version != kSessionHandoffEnvelopeVersion) {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported session handoff version: ", version));
  }

  SessionHandoffIdentity identity;
  ABSL_ASSIGN_OR_RETURN(identity.model_artifact_hash, reader.Hash());
  ABSL_ASSIGN_OR_RETURN(identity.runtime_artifact_hash, reader.Hash());
  ABSL_ASSIGN_OR_RETURN(identity.inference_profile_hash, reader.Hash());
  ABSL_ASSIGN_OR_RETURN(uint32_t key_id_size, reader.U32());
  if (key_id_size == 0 || key_id_size > kMaximumKeyIdSize) {
    return absl::DataLossError("Session handoff key_id size is invalid.");
  }
  ABSL_ASSIGN_OR_RETURN(std::string key_id, reader.String(key_id_size));
  if (key_id != options.key_id) {
    return absl::FailedPreconditionError(
        "Session handoff key_id does not match the expected key.");
  }
  if (identity != authoritative_identity) {
    return absl::FailedPreconditionError(
        "Session handoff model/runtime/profile identity does not match.");
  }

  DecodedSessionHandoff decoded;
  SessionHandoffSnapshot& snapshot = decoded.snapshot;
  ABSL_ASSIGN_OR_RETURN(uint8_t phase, reader.U8());
  switch (phase) {
    case static_cast<uint8_t>(SessionHandoffPhase::kFresh):
      snapshot.phase = SessionHandoffPhase::kFresh;
      break;
    case static_cast<uint8_t>(SessionHandoffPhase::kPrefilled):
      snapshot.phase = SessionHandoffPhase::kPrefilled;
      break;
    case static_cast<uint8_t>(SessionHandoffPhase::kDecoded):
      snapshot.phase = SessionHandoffPhase::kDecoded;
      break;
    default:
      return absl::DataLossError("Session handoff session phase is invalid.");
  }
  ABSL_ASSIGN_OR_RETURN(uint8_t ran_decode, reader.U8());
  if (ran_decode > 1) {
    return absl::DataLossError("Session handoff ran_decode flag is invalid.");
  }
  snapshot.executor.ran_decode = ran_decode != 0;
  ABSL_ASSIGN_OR_RETURN(uint16_t reserved, reader.U16());
  if (reserved != 0) {
    return absl::DataLossError("Session handoff reserved flags must be zero.");
  }
  ABSL_ASSIGN_OR_RETURN(snapshot.executor.current_step, reader.I32());
  ABSL_ASSIGN_OR_RETURN(snapshot.executor.last_prefill_token_id, reader.I32());
  ABSL_ASSIGN_OR_RETURN(uint8_t runtime_config_flags, reader.U8());
  if ((runtime_config_flags & ~uint8_t{7}) != 0) {
    return absl::DataLossError(
        "Session handoff runtime-config flags are invalid.");
  }
  if ((runtime_config_flags & 1) != 0) {
    proto::SamplerParameters sampler;
    ABSL_ASSIGN_OR_RETURN(int32_t type, reader.I32());
    if (!proto::SamplerParameters::Type_IsValid(type)) {
      return absl::DataLossError("Session handoff sampler type is invalid.");
    }
    sampler.set_type(static_cast<proto::SamplerParameters::Type>(type));
    ABSL_ASSIGN_OR_RETURN(int32_t k, reader.I32());
    sampler.set_k(k);
    ABSL_ASSIGN_OR_RETURN(float p, reader.Float());
    sampler.set_p(p);
    ABSL_ASSIGN_OR_RETURN(float temperature, reader.Float());
    sampler.set_temperature(temperature);
    ABSL_ASSIGN_OR_RETURN(uint8_t has_seed, reader.U8());
    if (has_seed > 1) {
      return absl::DataLossError(
          "Session handoff sampler seed flag is invalid.");
    }
    if (has_seed != 0) {
      ABSL_ASSIGN_OR_RETURN(int32_t seed, reader.I32());
      sampler.set_seed(seed);
    }
    ABSL_ASSIGN_OR_RETURN(int32_t backend, reader.I32());
    if (!proto::SamplerParameters::Backend_IsValid(backend)) {
      return absl::DataLossError("Session handoff sampler backend is invalid.");
    }
    sampler.set_backend(
        static_cast<proto::SamplerParameters::Backend>(backend));
    snapshot.executor.runtime_config.sampler_params = std::move(sampler);
  }
  if ((runtime_config_flags & 2) != 0) {
    ABSL_ASSIGN_OR_RETURN(int32_t output_heads, reader.I32());
    snapshot.executor.runtime_config.output_heads = output_heads;
  }
  if ((runtime_config_flags & 4) != 0) {
    ABSL_ASSIGN_OR_RETURN(int32_t tokens_per_decode, reader.I32());
    snapshot.executor.runtime_config.tokens_per_decode = tokens_per_decode;
  }
  ABSL_ASSIGN_OR_RETURN(uint32_t random_size, reader.U32());
  if (random_size == 0 || random_size > kMaximumRandomEngineStateSize) {
    return absl::DataLossError(
        "Session handoff random engine state size is invalid.");
  }
  ABSL_ASSIGN_OR_RETURN(std::string random_bytes, reader.String(random_size));
  ABSL_ASSIGN_OR_RETURN(snapshot.executor.random_engine,
                        ParseRandomEngine(random_bytes));

  ABSL_ASSIGN_OR_RETURN(uint32_t candidate_count, reader.U32());
  ABSL_ASSIGN_OR_RETURN(uint32_t processed_count, reader.U32());
  if (candidate_count != kSupportedCandidateCount ||
      processed_count > kMaximumTokensPerCandidate) {
    return absl::DataLossError(
        "Session handoff token-history dimensions are invalid.");
  }
  if (processed_count != 0 &&
      candidate_count > std::numeric_limits<size_t>::max() / processed_count) {
    return absl::ResourceExhaustedError(
        "Session handoff token-history dimensions overflow.");
  }
  const size_t processed_token_ids =
      static_cast<size_t>(candidate_count) * processed_count;
  if (processed_token_ids > kMaximumTotalTokenIds) {
    return absl::ResourceExhaustedError(
        "Session handoff total token history exceeds the supported limit.");
  }
  if (processed_token_ids > reader.remaining() / sizeof(int32_t)) {
    return absl::DataLossError("Session handoff token history is truncated.");
  }
  auto& candidates = snapshot.executor.processed_tokens.processed_token_ids;
  candidates.resize(candidate_count);
  for (auto& candidate : candidates) {
    candidate.reserve(processed_count);
    for (uint32_t i = 0; i < processed_count; ++i) {
      ABSL_ASSIGN_OR_RETURN(int32_t token_id, reader.I32());
      candidate.push_back(token_id);
    }
  }
  ABSL_ASSIGN_OR_RETURN(uint8_t has_pending, reader.U8());
  if (has_pending > 1) {
    return absl::DataLossError(
        "Session handoff pending-token flag is invalid.");
  }
  if (has_pending != 0) {
    if (candidate_count > kMaximumTotalTokenIds - processed_token_ids) {
      return absl::ResourceExhaustedError(
          "Session handoff total token history exceeds the supported limit.");
    }
    if (candidate_count > reader.remaining() / sizeof(int32_t)) {
      return absl::DataLossError(
          "Session handoff pending-token history is truncated.");
    }
    auto& pending = snapshot.executor.processed_tokens.pending_token_ids;
    pending.reserve(candidate_count);
    for (uint32_t i = 0; i < candidate_count; ++i) {
      ABSL_ASSIGN_OR_RETURN(int32_t token_id, reader.I32());
      pending.push_back(token_id);
    }
  }

  ABSL_ASSIGN_OR_RETURN(decoded.serialized_state_size, reader.U64());
  decoded.serialized_state_offset = reader.offset();
  ABSL_RETURN_IF_ERROR(reader.Skip(decoded.serialized_state_size));
  if (!reader.empty()) {
    return absl::DataLossError(
        "Session handoff envelope has trailing authenticated bytes.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateSnapshot(snapshot, decoded.serialized_state_size));
  return decoded;
}

absl::Status ReauthenticateSessionHandoffTo(
    const ByteSource& source,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& source_options,
    const SessionHandoffOptions& destination_options,
    ByteSink* destination) {
  if (destination == nullptr) {
    return absl::InvalidArgumentError(
        "Session handoff reauthentication destination must not be null.");
  }
  // ByteSource and ByteSink do not expose their backing storage, so callers
  // must also honor the documented non-aliasing contract. This catches the
  // directly-identical interface-object case before any source or destination
  // operation.
  if (static_cast<const void*>(&source) ==
      static_cast<const void*>(destination)) {
    return absl::InvalidArgumentError(
        "Session handoff reauthentication source and destination must not "
        "be the same object.");
  }

  // DecodeSessionHandoffFrom authenticates the complete source before parsing
  // metadata. Keep this operation ahead of every destination write so an
  // unauthenticated or malformed source can never produce a partial rewrap.
  ABSL_ASSIGN_OR_RETURN(
      DecodedSessionHandoff decoded,
      DecodeSessionHandoffFrom(source, authoritative_identity,
                               source_options));
  if (!decoded.snapshot.executor.serialized_state.empty()) {
    return absl::DataLossError(
        "Streamed session handoff decode unexpectedly materialized state.");
  }
  if (decoded.serialized_state_offset > source.Size() ||
      decoded.serialized_state_size >
          source.Size() - decoded.serialized_state_offset) {
    return absl::DataLossError(
        "Decoded session handoff state range exceeds its source.");
  }

  const ByteSourceView serialized_state(
      &source, decoded.serialized_state_offset,
      decoded.serialized_state_size);
  if (serialized_state.Size() != decoded.serialized_state_size) {
    return absl::DataLossError(
        "Decoded session handoff state range is not readable.");
  }

  return EncodeSessionHandoffParts(
      decoded.snapshot, decoded.serialized_state_size,
      authoritative_identity, destination_options, destination,
      [&serialized_state](ByteSink* state_sink) -> absl::Status {
        if (state_sink == nullptr) {
          return absl::InternalError(
              "Session handoff reauthentication state sink is null.");
        }
        if (serialized_state.Size() == 0) return absl::OkStatus();

        std::string chunk(static_cast<size_t>(std::min<uint64_t>(
                              serialized_state.Size(), kIoChunkSize)),
                          '\0');
        uint64_t offset = 0;
        while (offset < serialized_state.Size()) {
          const size_t count = static_cast<size_t>(std::min<uint64_t>(
              chunk.size(), serialized_state.Size() - offset));
          ABSL_RETURN_IF_ERROR(serialized_state.ReadAt(
              offset, absl::MakeSpan(chunk).subspan(0, count)));
          ABSL_RETURN_IF_ERROR(
              state_sink->Append(absl::string_view(chunk.data(), count)));
          offset += count;
        }
        return absl::OkStatus();
      });
}

absl::StatusOr<SessionHandoffSnapshot> DecodeSessionHandoff(
    absl::string_view envelope,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options) {
  StringByteSource source(envelope);
  ABSL_ASSIGN_OR_RETURN(DecodedSessionHandoff decoded,
                        DecodeSessionHandoffFrom(
                            source, authoritative_identity, options));
  if (decoded.serialized_state_size > std::numeric_limits<size_t>::max()) {
    return absl::ResourceExhaustedError(
        "Session handoff serialized state is too large for this platform.");
  }
  decoded.snapshot.executor.serialized_state.assign(
      static_cast<size_t>(decoded.serialized_state_size), '\0');
  ABSL_RETURN_IF_ERROR(source.ReadAt(
      decoded.serialized_state_offset,
      absl::MakeSpan(decoded.snapshot.executor.serialized_state)));
  return std::move(decoded.snapshot);
}

}  // namespace litert::lm
