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

#include "runtime/dpm/dpm_projection_replay_runtime.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_replay_executor.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kProjectionExecutionMagic = {
    'D', 'P', 'M', 'P', 'R', 'J', '0', '1'};
// Fixed bytes in DPMPRJ01 apart from schema ID, schema JSON, and prompt bytes.
constexpr uint64_t kProjectionExecutionFramingBytes =
    8 + 4 + 8 + 32 + 32 + 4 + 8 + 8 + 4 + 8 + 8 + 4 + 8 + 8;
constexpr absl::string_view kWinnerEvidenceDomain =
    "LITERT_LMX_CANONICAL_WINNER_PROJECTION_EVIDENCE_V1";

bool IsZeroHash(const Hash256& hash) {
  uint8_t combined = 0;
  for (uint8_t byte : hash.bytes) combined |= byte;
  return combined == 0;
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
    const absl::string_view result =
        bytes_.substr(offset_, static_cast<size_t>(size));
    offset_ += static_cast<size_t>(size);
    return result;
  }

  size_t remaining() const { return bytes_.size() - offset_; }

 private:
  absl::Status Truncated() const {
    return absl::DataLossError(
        "Truncated DPM projection execution request.");
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

bool ProjectionConfigsEqual(const DPMProjectionConfig& left,
                            const DPMProjectionConfig& right) {
  return left.format_version == right.format_version &&
         left.schema_id == right.schema_id &&
         left.schema_json == right.schema_json &&
         left.max_output_tokens == right.max_output_tokens &&
         left.memory_budget_bytes == right.memory_budget_bytes &&
         left.max_event_range_bytes == right.max_event_range_bytes &&
         left.max_items_per_section == right.max_items_per_section &&
         left.max_item_bytes == right.max_item_bytes;
}

absl::Status ValidateRuntimeIdentity(
    const SessionHandoffIdentity& identity) {
  if (IsZeroHash(identity.model_artifact_hash) ||
      IsZeroHash(identity.runtime_artifact_hash) ||
      IsZeroHash(identity.inference_profile_hash)) {
    return absl::FailedPreconditionError(
        "Projection replay requires a complete runtime-derived identity.");
  }
  return absl::OkStatus();
}

Hash256 ComputeWinnerEvidenceHash(
    const SessionHandoffIdentity& identity,
    const Hash256& canonical_replay_request_hash,
    absl::string_view canonical_output) {
  Sha256Hasher hasher;
  hasher.Update(kWinnerEvidenceDomain);
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(identity.model_artifact_hash.bytes.data()),
      identity.model_artifact_hash.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(identity.runtime_artifact_hash.bytes.data()),
      identity.runtime_artifact_hash.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(identity.inference_profile_hash.bytes.data()),
      identity.inference_profile_hash.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(canonical_replay_request_hash.bytes.data()),
      canonical_replay_request_hash.bytes.size()));
  std::string length;
  AppendU64(canonical_output.size(), &length);
  hasher.Update(length);
  hasher.Update(canonical_output);
  return hasher.Finalize();
}

absl::Status ValidateCanonicalProjectionRequest(
    const CanonicalDPMProjectionRequest& request,
    const SessionHandoffIdentity& expected_identity,
    const DPMProjectionConfig& expected_config) {
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_config_hash,
                        ComputeDPMProjectionConfigHash(expected_config));
  ABSL_ASSIGN_OR_RETURN(const Hash256 computed,
                        ComputeCanonicalDPMProjectionRequestHash(request));
  if (computed != request.request_hash ||
      request.runtime_identity != expected_identity ||
      request.config_hash != expected_config_hash) {
    return absl::FailedPreconditionError(
        "Projection execution request changed or names another runtime or "
        "projection configuration.");
  }
  return absl::OkStatus();
}

DPMCanonicalReplayRequest MakeReplayRequest(
    std::string encoded_projection_request, uint32_t max_output_tokens) {
  return DPMCanonicalReplayRequest{
      .stage = DPMReplayStage::kProjection,
      .max_output_tokens = max_output_tokens,
      .request_contract_version =
          std::string(kDPMProjectionReplayContractVersion),
      .canonical_payload = std::move(encoded_projection_request),
  };
}

}  // namespace

absl::Status ValidateDPMProjectionExecutionRequest(
    const DPMProjectionExecutionRequest& request) {
  if (request.format_version !=
          kDPMProjectionExecutionRequestFormatVersion ||
      request.source_event_count == 0 ||
      IsZeroHash(request.projection_request_hash) ||
      IsZeroHash(request.projection_config_hash) ||
      request.canonical_prompt.empty() ||
      request.canonical_prompt.size() >
          kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::InvalidArgumentError(
        "DPM projection execution request is incomplete or oversized.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 computed_config_hash,
      ComputeDPMProjectionConfigHash(request.projection_config));
  if (computed_config_hash != request.projection_config_hash) {
    return absl::FailedPreconditionError(
        "DPM projection execution request configuration hash is stale.");
  }

  // Keep the complete inner request plus its fixed generic replay envelope
  // within the one-request fresh-worker protocol limit. WinnerReplay uses the
  // same canonical request bytes so the two named modes cannot disagree about
  // what request a hash denotes.
  uint64_t encoded_envelope_size =
      kProjectionExecutionFramingBytes +
      kDPMCanonicalReplayRequestFramingBytes +
      kDPMProjectionReplayContractVersion.size();
  const auto add_bounded = [&encoded_envelope_size](size_t size) {
    if (size > kMaximumFreshWorkerRequestPayloadBytes -
                   encoded_envelope_size) {
      return false;
    }
    encoded_envelope_size += size;
    return true;
  };
  if (encoded_envelope_size > kMaximumFreshWorkerRequestPayloadBytes ||
      !add_bounded(request.projection_config.schema_id.size()) ||
      !add_bounded(request.projection_config.schema_json.size()) ||
      !add_bounded(request.canonical_prompt.size())) {
    return absl::ResourceExhaustedError(
        "DPM projection execution request cannot fit one canonical worker "
        "envelope.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeDPMProjectionExecutionRequest(
    const DPMProjectionExecutionRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionExecutionRequest(request));
  const uint64_t encoded_size =
      kProjectionExecutionFramingBytes +
      request.projection_config.schema_id.size() +
      request.projection_config.schema_json.size() +
      request.canonical_prompt.size();
  std::string encoded;
  encoded.reserve(static_cast<size_t>(encoded_size));
  encoded.append(kProjectionExecutionMagic.data(),
                 kProjectionExecutionMagic.size());
  AppendU32(request.format_version, &encoded);
  AppendU64(request.source_event_count, &encoded);
  AppendHash(request.projection_request_hash, &encoded);
  AppendHash(request.projection_config_hash, &encoded);
  AppendU32(request.projection_config.format_version, &encoded);
  AppendU64(request.projection_config.schema_id.size(), &encoded);
  encoded.append(request.projection_config.schema_id);
  AppendU64(request.projection_config.schema_json.size(), &encoded);
  encoded.append(request.projection_config.schema_json);
  AppendU32(request.projection_config.max_output_tokens, &encoded);
  AppendU64(request.projection_config.memory_budget_bytes, &encoded);
  AppendU64(request.projection_config.max_event_range_bytes, &encoded);
  AppendU32(request.projection_config.max_items_per_section, &encoded);
  AppendU64(request.projection_config.max_item_bytes, &encoded);
  AppendU64(request.canonical_prompt.size(), &encoded);
  encoded.append(request.canonical_prompt);
  return encoded;
}

absl::StatusOr<DPMProjectionExecutionRequest>
DecodeDPMProjectionExecutionRequest(absl::string_view bytes) {
  if (bytes.size() < kProjectionExecutionFramingBytes ||
      bytes.size() > kMaximumFreshWorkerRequestPayloadBytes ||
      std::memcmp(bytes.data(), kProjectionExecutionMagic.data(),
                  kProjectionExecutionMagic.size()) != 0) {
    return absl::DataLossError(
        "DPM projection execution request framing is invalid.");
  }
  Reader reader(bytes.substr(kProjectionExecutionMagic.size()));
  DPMProjectionExecutionRequest request;
  ABSL_ASSIGN_OR_RETURN(request.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(request.source_event_count, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(request.projection_request_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.projection_config_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(request.projection_config.format_version,
                        reader.ReadU32());
  uint64_t schema_id_size;
  ABSL_ASSIGN_OR_RETURN(schema_id_size, reader.ReadU64());
  absl::string_view schema_id;
  ABSL_ASSIGN_OR_RETURN(schema_id, reader.ReadBytes(schema_id_size));
  request.projection_config.schema_id.assign(schema_id.data(),
                                             schema_id.size());
  uint64_t schema_json_size;
  ABSL_ASSIGN_OR_RETURN(schema_json_size, reader.ReadU64());
  absl::string_view schema_json;
  ABSL_ASSIGN_OR_RETURN(schema_json, reader.ReadBytes(schema_json_size));
  request.projection_config.schema_json.assign(schema_json.data(),
                                               schema_json.size());
  ABSL_ASSIGN_OR_RETURN(request.projection_config.max_output_tokens,
                        reader.ReadU32());
  uint64_t memory_budget_bytes;
  ABSL_ASSIGN_OR_RETURN(memory_budget_bytes, reader.ReadU64());
  uint64_t max_event_range_bytes;
  ABSL_ASSIGN_OR_RETURN(max_event_range_bytes, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(request.projection_config.max_items_per_section,
                        reader.ReadU32());
  uint64_t max_item_bytes;
  ABSL_ASSIGN_OR_RETURN(max_item_bytes, reader.ReadU64());
  if (memory_budget_bytes > (std::numeric_limits<size_t>::max)() ||
      max_event_range_bytes > (std::numeric_limits<size_t>::max)() ||
      max_item_bytes > (std::numeric_limits<size_t>::max)()) {
    return absl::ResourceExhaustedError(
        "DPM projection execution request contains unrepresentable limits.");
  }
  request.projection_config.memory_budget_bytes =
      static_cast<size_t>(memory_budget_bytes);
  request.projection_config.max_event_range_bytes =
      static_cast<size_t>(max_event_range_bytes);
  request.projection_config.max_item_bytes =
      static_cast<size_t>(max_item_bytes);
  uint64_t prompt_size;
  ABSL_ASSIGN_OR_RETURN(prompt_size, reader.ReadU64());
  absl::string_view prompt;
  ABSL_ASSIGN_OR_RETURN(prompt, reader.ReadBytes(prompt_size));
  request.canonical_prompt.assign(prompt.data(), prompt.size());
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "DPM projection execution request has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionExecutionRequest(request));
  ABSL_ASSIGN_OR_RETURN(const std::string canonical,
                        EncodeDPMProjectionExecutionRequest(request));
  if (canonical != bytes) {
    return absl::DataLossError(
        "DPM projection execution request is not canonical.");
  }
  return request;
}

absl::Status ValidateDPMProjectionReplayExecution(
    const DPMProjectionReplayExecution& execution) {
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(execution.mode));
  if (IsZeroHash(execution.replay_request_hash) ||
      IsZeroHash(execution.execution_evidence_hash) ||
      execution.canonical_output.empty() ||
      execution.canonical_output.size() >
          kMaximumFreshWorkerCanonicalOutputBytes) {
    return absl::InvalidArgumentError(
        "DPM projection replay result is incomplete or oversized.");
  }
  switch (execution.mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      if (execution.exact_profile_id.has_value() ||
          execution.exact_profile_admission_record_id.has_value() ||
          !execution.exact_token_bytes.empty() ||
          !execution.exact_logit_frames.empty()) {
        return absl::FailedPreconditionError(
            "CanonicalWinnerReplay cannot carry exact-regeneration "
            "evidence.");
      }
      return absl::OkStatus();
    case DPMReplayMode::kExactRegeneration:
      if (!execution.exact_profile_id.has_value() ||
          IsZeroHash(*execution.exact_profile_id) ||
          !execution.exact_profile_admission_record_id.has_value() ||
          IsZeroHash(*execution.exact_profile_admission_record_id) ||
          execution.reused_canonical_winner) {
        return absl::FailedPreconditionError(
            "ExactRegeneration lacks profile admission or claims catalog "
            "reuse.");
      }
      return ValidateFreshWorkerExecutionOutput(FreshWorkerExecutionOutput{
          .canonical_output = execution.canonical_output,
          .token_bytes = execution.exact_token_bytes,
          .logit_frames = execution.exact_logit_frames,
      });
  }
  return absl::InvalidArgumentError(
      "DPM projection replay result uses an unknown mode.");
}

absl::StatusOr<std::unique_ptr<CanonicalWinnerDPMProjectionRuntime>>
CanonicalWinnerDPMProjectionRuntime::Create(
    DPMProjectionRuntime* inference_runtime,
    CanonicalWinnerReplayCatalog* catalog, DPMProjectionConfig config) {
  if (inference_runtime == nullptr || catalog == nullptr) {
    return absl::InvalidArgumentError(
        "WinnerReplay projection requires inference and catalog.");
  }
  const SessionHandoffIdentity runtime_identity =
      inference_runtime->GetRuntimeIdentity();
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(runtime_identity));
  auto runtime = std::unique_ptr<CanonicalWinnerDPMProjectionRuntime>(
      new CanonicalWinnerDPMProjectionRuntime(inference_runtime, catalog,
                                              std::move(config),
                                              runtime_identity));
  ABSL_RETURN_IF_ERROR(runtime->ValidateSupport());
  return runtime;
}

const SessionHandoffIdentity&
CanonicalWinnerDPMProjectionRuntime::GetRuntimeIdentity() const {
  return runtime_identity_;
}

absl::StatusOr<std::optional<Hash256>>
CanonicalWinnerDPMProjectionRuntime::GetExactProfileId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  return std::optional<Hash256>();
}

uint32_t CanonicalWinnerDPMProjectionRuntime::GetMaxOutputTokens() const {
  return config_.max_output_tokens;
}

absl::Status CanonicalWinnerDPMProjectionRuntime::ValidateSupport() const {
  return ValidateSupport(DPMReplayStage::kProjection);
}

absl::Status CanonicalWinnerDPMProjectionRuntime::ValidateSupport(
    DPMReplayStage expected_stage) const {
  if (expected_stage != DPMReplayStage::kProjection ||
      inference_runtime_ == nullptr) {
    return absl::InvalidArgumentError(
        "WinnerReplay projection adapter received another stage.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionConfig(config_));
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(runtime_identity_));
  if (inference_runtime_->GetRuntimeIdentity() != runtime_identity_) {
    return absl::FailedPreconditionError(
        "WinnerReplay projection runtime identity changed after "
        "construction.");
  }
  if (inference_runtime_->GetMaxOutputTokens() != config_.max_output_tokens) {
    return absl::FailedPreconditionError(
        "WinnerReplay projection token limit differs from its config.");
  }
  ABSL_RETURN_IF_ERROR(inference_runtime_->ValidateSupport());
  if (inference_runtime_->GetRuntimeIdentity() != runtime_identity_ ||
      inference_runtime_->GetMaxOutputTokens() != config_.max_output_tokens) {
    return absl::FailedPreconditionError(
        "WinnerReplay projection runtime changed while validating support.");
  }
  return absl::OkStatus();
}

absl::StatusOr<CanonicalWinnerGeneratedCandidate>
CanonicalWinnerDPMProjectionRuntime::Generate(
    const DPMCanonicalReplayRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateDPMCanonicalReplayRequest(request));
  if (request.stage != DPMReplayStage::kProjection ||
      request.max_output_tokens != config_.max_output_tokens ||
      request.request_contract_version !=
          kDPMProjectionReplayContractVersion) {
    return absl::InvalidArgumentError(
        "WinnerReplay projection received another request contract.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const DPMProjectionExecutionRequest projection,
      DecodeDPMProjectionExecutionRequest(request.canonical_payload));
  if (!ProjectionConfigsEqual(projection.projection_config, config_)) {
    return absl::FailedPreconditionError(
        "WinnerReplay projection request carries another configuration.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 canonical_replay_request_hash,
      ComputeDPMCanonicalReplayRequestHash(request));
  ABSL_ASSIGN_OR_RETURN(
      const std::string raw_output,
      inference_runtime_->GenerateFresh(projection.canonical_prompt));
  ABSL_ASSIGN_OR_RETURN(
      std::string canonical_output,
      CanonicalizeDPMProjectionOutput(raw_output,
                                      projection.source_event_count, config_));
  return CanonicalWinnerGeneratedCandidate{
      .canonical_output = canonical_output,
      .execution_evidence_hash = ComputeWinnerEvidenceHash(
          GetRuntimeIdentity(), canonical_replay_request_hash,
          canonical_output),
  };
}

absl::StatusOr<DPMProjectionReplayExecution>
CanonicalWinnerDPMProjectionRuntime::Generate(
    const CanonicalDPMProjectionRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_RETURN_IF_ERROR(ValidateCanonicalProjectionRequest(
      request, GetRuntimeIdentity(), config_));
  DPMProjectionExecutionRequest projection{
      .source_event_count = request.source_event_count,
      .projection_request_hash = request.request_hash,
      .projection_config_hash = request.config_hash,
      .projection_config = config_,
      .canonical_prompt = request.prompt_bytes,
  };
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMProjectionExecutionRequest(projection));
  const DPMCanonicalReplayRequest replay_request =
      MakeReplayRequest(std::move(encoded), config_.max_output_tokens);
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_replay_request_hash,
      ComputeDPMCanonicalReplayRequestHash(replay_request));
  ABSL_ASSIGN_OR_RETURN(
      CanonicalWinnerReplayExecution replay,
      replay_executor_.Run(replay_request, this));
  ABSL_ASSIGN_OR_RETURN(
      const std::string verified,
      CanonicalizeDPMProjectionOutput(replay.canonical_output,
                                      request.source_event_count, config_));
  if (replay.mode != DPMReplayMode::kCanonicalWinnerReplay ||
      replay.runtime_identity != runtime_identity_ ||
      replay.canonical_request_hash != expected_replay_request_hash ||
      replay.canonical_output_hash != Sha256(replay.canonical_output) ||
      IsZeroHash(replay.execution_evidence_hash) ||
      verified != replay.canonical_output) {
    return absl::DataLossError(
        "Canonical winner projection result is not bound to this canonical "
        "request.");
  }
  DPMProjectionReplayExecution execution{
      .mode = DPMReplayMode::kCanonicalWinnerReplay,
      .replay_request_hash = replay.canonical_request_hash,
      .execution_evidence_hash = replay.execution_evidence_hash,
      .canonical_output = replay.canonical_output,
      .reused_canonical_winner =
          replay.resolution != CanonicalWinnerResolution::kPublished,
  };
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionReplayExecution(execution));
  return execution;
}

absl::StatusOr<std::unique_ptr<ExactRegenerationDPMProjectionRuntime>>
ExactRegenerationDPMProjectionRuntime::Create(
    ExactRegenerationExecutor* exact_executor,
    DPMProjectionConfig config) {
  if (exact_executor == nullptr) {
    return absl::InvalidArgumentError(
        "Exact projection requires an ExactRegeneration executor.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionConfig(config));
  if (exact_executor->GetReplayStage() != DPMReplayStage::kProjection ||
      exact_executor->GetMaxOutputTokens() != config.max_output_tokens) {
    return absl::FailedPreconditionError(
        "Exact projection executor is bound to another stage or token "
        "limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const ExactLiteRtProfile profile,
                        exact_executor->GetDerivedProfile());
  ABSL_RETURN_IF_ERROR(ValidateRuntimeIdentity(profile.session_identity));
  auto runtime = std::unique_ptr<ExactRegenerationDPMProjectionRuntime>(
      new ExactRegenerationDPMProjectionRuntime(
          exact_executor, std::move(config), profile));
  ABSL_RETURN_IF_ERROR(runtime->ValidateSupport());
  return runtime;
}

absl::Status ExactRegenerationDPMProjectionRuntime::ValidateSupport() const {
  if (exact_executor_ == nullptr) {
    return absl::InvalidArgumentError(
        "Exact projection has no regeneration executor.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionConfig(config_));
  if (exact_executor_->GetReplayStage() != DPMReplayStage::kProjection ||
      exact_executor_->GetMaxOutputTokens() != config_.max_output_tokens) {
    return absl::FailedPreconditionError(
        "Exact projection executor binding changed or does not match its "
        "projection configuration.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateRuntimeIdentity(derived_profile_.session_identity));
  ABSL_RETURN_IF_ERROR(exact_executor_->ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(const ExactLiteRtProfile profile,
                        exact_executor_->GetDerivedProfile());
  if (profile != derived_profile_) {
    return absl::FailedPreconditionError(
        "Exact projection Engine profile changed after construction.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<Hash256>>
ExactRegenerationDPMProjectionRuntime::GetExactProfileId() const {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  return std::optional<Hash256>(derived_profile_.profile_id);
}

absl::StatusOr<DPMProjectionReplayExecution>
ExactRegenerationDPMProjectionRuntime::Generate(
    const CanonicalDPMProjectionRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_RETURN_IF_ERROR(ValidateCanonicalProjectionRequest(
      request, derived_profile_.session_identity, config_));
  DPMProjectionExecutionRequest projection{
      .source_event_count = request.source_event_count,
      .projection_request_hash = request.request_hash,
      .projection_config_hash = request.config_hash,
      .projection_config = config_,
      .canonical_prompt = request.prompt_bytes,
  };
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMProjectionExecutionRequest(projection));
  const DPMCanonicalReplayRequest replay_request =
      MakeReplayRequest(std::move(encoded), config_.max_output_tokens);
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_replay_request_hash,
      ComputeDPMCanonicalReplayRequestHash(replay_request));
  ABSL_ASSIGN_OR_RETURN(
      ExactRegenerationExecution exact,
      exact_executor_->Run(replay_request));
  if (exact.mode != DPMReplayMode::kExactRegeneration ||
      exact.derived_profile != derived_profile_ ||
      exact.canonical_request_hash != expected_replay_request_hash ||
      IsZeroHash(exact.profile_admission_record_id)) {
    return absl::FailedPreconditionError(
        "Exact projection result came from another request or exact "
        "profile.");
  }
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecordForProfile(
      exact.cold_run_evidence, exact.derived_profile));
  ABSL_ASSIGN_OR_RETURN(
      const std::string verified,
      CanonicalizeDPMProjectionOutput(exact.canonical_output,
                                      request.source_event_count, config_));
  if (verified != exact.canonical_output ||
      IsZeroHash(exact.cold_run_evidence.record_id) ||
      exact.cold_run_evidence.canonical_output != exact.canonical_output ||
      exact.cold_run_evidence.token_bytes != exact.token_bytes ||
      exact.cold_run_evidence.logit_frames != exact.logit_frames) {
    return absl::DataLossError(
        "Exact projection worker output is not canonical evidence.");
  }
  DPMProjectionReplayExecution execution{
      .mode = DPMReplayMode::kExactRegeneration,
      .replay_request_hash = exact.canonical_request_hash,
      .execution_evidence_hash = exact.cold_run_evidence.record_id,
      .exact_profile_id = derived_profile_.profile_id,
      .exact_profile_admission_record_id =
          exact.profile_admission_record_id,
      .canonical_output = exact.canonical_output,
      .exact_token_bytes = std::move(exact.token_bytes),
      .exact_logit_frames = std::move(exact.logit_frames),
      .reused_canonical_winner = false,
  };
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionReplayExecution(execution));
  return execution;
}

}  // namespace litert::lm
