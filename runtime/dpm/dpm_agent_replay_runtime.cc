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
constexpr std::array<char, 8> kAgentDecisionMagic = {'D', 'P', 'M', 'D',
                                                      'E', 'C', '0', '1'};
constexpr uint64_t kAgentExecutionFixedBytes = 8 + 4 + 32 + 32 + 4 + 4;
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

bool ExecutionRequestsEqual(const DPMAgentExecutionRequest& left,
                            const DPMAgentExecutionRequest& right) {
  return left.format_version == right.format_version &&
         left.logical_agent_request_hash == right.logical_agent_request_hash &&
         left.correction_digest == right.correction_digest &&
         left.max_output_tokens == right.max_output_tokens &&
         ChunksEqual(left.full_canonical_prefill_chunks,
                     right.full_canonical_prefill_chunks);
}

absl::Status ValidateGenerationRequest(
    const DPMAgentGenerationRequest& request) {
  if (request.max_output_tokens <= 0 ||
      request.max_output_tokens >
          static_cast<int>(kMaximumDPMGenerationTokens) ||
      request.canonical_prefill_chunks.empty() ||
      request.canonical_prefill_chunks.size() >
          kMaximumFreshWorkerTokenIds) {
    return absl::InvalidArgumentError(
        "DPM agent generation request is incomplete or oversized.");
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
        logical_request_->max_output_tokens) {
      return absl::InvalidArgumentError(
          "WinnerReplay live generation and logical request limits differ.");
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
    generation_succeeded_ = true;
    return CanonicalWinnerGeneratedCandidate{
        .canonical_output = std::move(canonical_output),
        .execution_evidence_hash = evidence,
    };
  }

  bool generate_called() const { return generate_called_; }
  bool generation_succeeded() const { return generation_succeeded_; }
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
};

DPMCanonicalReplayRequest MakeReplayRequest(std::string encoded) {
  return DPMCanonicalReplayRequest{
      .stage = DPMReplayStage::kAgentDecision,
      .request_contract_version = std::string(kDPMAgentReplayContractVersion),
      .canonical_payload = std::move(encoded),
  };
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
  switch (execution.mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      if (execution.exact_profile_id.has_value() ||
          execution.exact_profile_admission_record_id.has_value() ||
          execution.exact_output_evidence_hash.has_value() ||
          !execution.exact_token_bytes.empty() ||
          !execution.exact_logit_frames.empty() ||
          execution.producing_session_matches_output ==
              execution.reused_canonical_winner) {
        return absl::DataLossError(
            "WinnerReplay agent execution carries exact evidence or invalid "
            "producing-session provenance.");
      }
      break;
    case DPMReplayMode::kExactRegeneration: {
      if (!execution.exact_profile_id.has_value() ||
          IsZeroHash(*execution.exact_profile_id) ||
          !execution.exact_profile_admission_record_id.has_value() ||
          IsZeroHash(*execution.exact_profile_admission_record_id) ||
          !execution.exact_output_evidence_hash.has_value() ||
          IsZeroHash(*execution.exact_output_evidence_hash) ||
          execution.exact_token_bytes.empty() ||
          execution.exact_logit_frames.empty() ||
          execution.reused_canonical_winner ||
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
  return absl::OkStatus();
}

absl::Status
CanonicalWinnerDPMAgentRuntime::ValidateSessionHandoffSupport() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  return inference_runtime_->ValidateSessionHandoffSupport();
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
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMAgentExecutionRequest(logical_request));
  const DPMCanonicalReplayRequest replay_request =
      MakeReplayRequest(std::move(encoded));
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
  };
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentReplayExecution(
      execution, logical_request.max_output_tokens));
  return execution;
}

absl::StatusOr<std::unique_ptr<ExactRegenerationDPMAgentRuntime>>
ExactRegenerationDPMAgentRuntime::Create(
    ExactRegenerationExecutor* exact_executor) {
  if (exact_executor == nullptr) {
    return absl::InvalidArgumentError(
        "Exact agent requires an ExactRegeneration executor.");
  }
  ABSL_ASSIGN_OR_RETURN(ExactLiteRtProfile profile,
                        exact_executor->GetDerivedProfile());
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(profile.session_identity));
  if (IsZeroHash(profile.profile_id)) {
    return absl::FailedPreconditionError(
        "Exact agent Engine returned an empty derived profile ID.");
  }
  auto runtime = std::unique_ptr<ExactRegenerationDPMAgentRuntime>(
      new ExactRegenerationDPMAgentRuntime(exact_executor,
                                           std::move(profile)));
  ABSL_RETURN_IF_ERROR(runtime->ValidateSupport());
  return runtime;
}

absl::StatusOr<std::optional<Hash256>>
ExactRegenerationDPMAgentRuntime::GetExactProfileId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  return std::optional<Hash256>(derived_profile_.profile_id);
}

absl::Status ExactRegenerationDPMAgentRuntime::ValidateSupport() const {
  if (exact_executor_ == nullptr) {
    return absl::InvalidArgumentError(
        "Exact agent has no regeneration executor.");
  }
  ABSL_RETURN_IF_ERROR(exact_executor_->ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(const ExactLiteRtProfile current,
                        exact_executor_->GetDerivedProfile());
  if (current != derived_profile_) {
    return absl::FailedPreconditionError(
        "Exact agent Engine profile changed after construction.");
  }
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(current.session_identity));
  if (IsZeroHash(current.profile_id)) {
    return absl::FailedPreconditionError(
        "Exact agent Engine returned an empty derived profile ID.");
  }
  return absl::OkStatus();
}

absl::Status
ExactRegenerationDPMAgentRuntime::ValidateSessionHandoffSupport() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  return absl::UnimplementedError(
      "Exact agent CapsuleRestore requires an authenticated producing-session "
      "envelope returned by the fresh worker; parent-session substitution is "
      "forbidden.");
}

absl::StatusOr<std::unique_ptr<Engine::Session>>
ExactRegenerationDPMAgentRuntime::CreateSession() {
  return absl::UnimplementedError(
      "Exact agent producing sessions live in fresh workers; parent capsule "
      "import is unavailable until the worker returns an authenticated "
      "producing-session envelope.");
}

absl::StatusOr<DPMAgentReplayExecution>
ExactRegenerationDPMAgentRuntime::Generate(
    Engine::Session* producing_session,
    const DPMAgentGenerationRequest& execution_request,
    const DPMAgentExecutionRequest& logical_request) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentExecutionRequest(logical_request));
  ABSL_RETURN_IF_ERROR(ValidateGenerationRequest(execution_request));
  if (producing_session != nullptr ||
      static_cast<uint32_t>(execution_request.max_output_tokens) !=
          logical_request.max_output_tokens ||
      !ChunksEqual(execution_request.canonical_prefill_chunks,
                   logical_request.full_canonical_prefill_chunks)) {
    return absl::InvalidArgumentError(
        "Exact agent requires no parent session and the complete logical "
        "prefill request.");
  }
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMAgentExecutionRequest(logical_request));
  const DPMCanonicalReplayRequest replay_request =
      MakeReplayRequest(std::move(encoded));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_request_hash,
      ComputeDPMCanonicalReplayRequestHash(replay_request));
  ABSL_ASSIGN_OR_RETURN(
      ExactRegenerationExecution exact,
      exact_executor_->Run(replay_request));
  if (exact.mode != DPMReplayMode::kExactRegeneration ||
      exact.derived_profile != derived_profile_ ||
      exact.canonical_request_hash != expected_request_hash ||
      IsZeroHash(exact.profile_admission_record_id) ||
      IsZeroHash(exact.cold_run_evidence.record_id)) {
    return absl::FailedPreconditionError(
        "Exact agent result came from another request or Engine profile.");
  }
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecordForProfile(
      exact.cold_run_evidence, exact.derived_profile));
  if (exact.cold_run_evidence.canonical_output != exact.canonical_output ||
      exact.cold_run_evidence.token_bytes != exact.token_bytes ||
      exact.cold_run_evidence.logit_frames != exact.logit_frames) {
    return absl::DataLossError(
        "Exact agent cold-run record and returned output evidence disagree.");
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
  const Hash256 exact_output_evidence_hash =
      ComputeFreshWorkerOutputEvidenceHash(
          exact.canonical_output, exact.token_bytes, exact.logit_frames);
  if (IsZeroHash(exact_output_evidence_hash)) {
    return absl::InternalError(
        "Exact agent could not bind its output evidence.");
  }
  std::vector<int> decision_ids(exact_ids.begin(), exact_ids.end());
  DPMAgentReplayExecution execution{
      .mode = DPMReplayMode::kExactRegeneration,
      .replay_request_hash = exact.canonical_request_hash,
      .execution_evidence_hash = exact.cold_run_evidence.record_id,
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
  };
  ABSL_RETURN_IF_ERROR(ValidateDPMAgentReplayExecution(
      execution, logical_request.max_output_tokens));
  return execution;
}

}  // namespace litert::lm
