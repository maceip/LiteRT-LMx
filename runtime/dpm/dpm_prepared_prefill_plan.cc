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

#include "runtime/dpm/dpm_prepared_prefill_plan.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kSourceChunksDomain =
    "litert-lmx-dpm-prepared-prefill-source-chunks-v1";
constexpr absl::string_view kUtf8SourceChunkDomain =
    "litert-lmx-dpm-prefill-utf8-source-chunk-v1";
constexpr absl::string_view kTokenSourceChunkDomain =
    "litert-lmx-dpm-prefill-token-source-chunk-v1";
constexpr absl::string_view kResolvedTokenPlanDomain =
    "litert-lmx-dpm-prepared-prefill-token-plan-v1";
constexpr absl::string_view kShapeScheduleDomain =
    "litert-lmx-dpm-prepared-prefill-shape-schedule-v1";
constexpr absl::string_view kPlanIdDomain =
    "litert-lmx-dpm-prepared-prefill-plan-id-v1";
constexpr absl::string_view kPreparedPrefillPlanMagic = "DPMPRP01";

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
}

bool HasValidIdentity(const SessionHandoffIdentity& identity) {
  return !IsZeroHash(identity.model_artifact_hash) &&
         !IsZeroHash(identity.runtime_artifact_hash) &&
         !IsZeroHash(identity.inference_profile_hash);
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

void AppendOptionalHash(const std::optional<Hash256>& hash,
                        std::string* output) {
  AppendU32(hash.has_value() ? 1 : 0, output);
  if (hash.has_value()) AppendHash(*hash, output);
}

Hash256 HashCanonical(absl::string_view domain,
                      absl::string_view canonical) {
  Sha256Hasher hasher;
  hasher.Update(domain);
  hasher.Update(canonical);
  return hasher.Finalize();
}

class CanonicalReader {
 public:
  explicit CanonicalReader(absl::string_view bytes) : bytes_(bytes) {}

  absl::StatusOr<uint32_t> ReadU32() {
    if (remaining() < 4) {
      return absl::DataLossError(
          "Prepared DPM prefill plan ended inside a uint32 field.");
    }
    uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      value = (value << 8) |
              static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 4;
    return value;
  }

  absl::StatusOr<uint64_t> ReadU64() {
    if (remaining() < 8) {
      return absl::DataLossError(
          "Prepared DPM prefill plan ended inside a uint64 field.");
    }
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
      value = (value << 8) |
              static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 8;
    return value;
  }

  absl::StatusOr<Hash256> ReadHash() {
    if (remaining() < Hash256{}.bytes.size()) {
      return absl::DataLossError(
          "Prepared DPM prefill plan ended inside a digest field.");
    }
    Hash256 hash;
    std::memcpy(hash.bytes.data(), bytes_.data() + offset_, hash.bytes.size());
    offset_ += hash.bytes.size();
    return hash;
  }

  absl::StatusOr<std::optional<Hash256>> ReadOptionalHash() {
    ABSL_ASSIGN_OR_RETURN(const uint32_t present, ReadU32());
    if (present == 0) return std::optional<Hash256>();
    if (present != 1) {
      return absl::DataLossError(
          "Prepared DPM prefill plan has a noncanonical optional digest tag.");
    }
    ABSL_ASSIGN_OR_RETURN(const Hash256 hash, ReadHash());
    return std::optional<Hash256>(hash);
  }

  absl::StatusOr<absl::string_view> ReadBytes(uint64_t size) {
    if (size > remaining()) {
      return absl::DataLossError(
          "Prepared DPM prefill plan ended inside a byte field.");
    }
    const absl::string_view result = bytes_.substr(offset_, size);
    offset_ += size;
    return result;
  }

  uint64_t remaining() const { return bytes_.size() - offset_; }

 private:
  absl::string_view bytes_;
  uint64_t offset_ = 0;
};

absl::Status ValidateBaseFields(const DPMPreparedPrefillPlan& plan,
                                bool require_canonical_hashes) {
  if (plan.format_version != DPMPreparedPrefillPlan::kFormatVersion) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill plan has an unsupported format version.");
  }
  if (!HasValidIdentity(plan.session_identity) ||
      IsZeroHash(plan.logical_agent_request_hash) ||
      IsZeroHash(plan.canonical_source_chunks_hash) ||
      IsZeroHash(plan.start_history_token_bytes_hash) ||
      (require_canonical_hashes &&
       (IsZeroHash(plan.plan_id) ||
        IsZeroHash(plan.resolved_token_plan_hash) ||
        IsZeroHash(plan.shape_schedule_hash)))) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill plan has an empty identity or digest.");
  }
  if (plan.batch_size != 1 || plan.vocabulary_size == 0) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill requires batch one and a nonzero vocabulary.");
  }
  if (plan.start_step != plan.start_history_token_count) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill start step differs from its exact history "
        "count.");
  }
  switch (plan.start_kind) {
    case DPMPreparedPrefillStartKind::kFreshSession:
      if (plan.start_step != 0 || plan.restore_checkpoint_id.has_value() ||
          plan.start_state_witness_id.has_value()) {
        return absl::InvalidArgumentError(
            "Fresh prepared DPM prefill has restore-only state bindings.");
      }
      break;
    case DPMPreparedPrefillStartKind::kOwnPositionRestore:
      if (plan.start_step == 0 || !plan.restore_checkpoint_id.has_value() ||
          IsZeroHash(*plan.restore_checkpoint_id) ||
          !plan.start_state_witness_id.has_value() ||
          IsZeroHash(*plan.start_state_witness_id)) {
        return absl::InvalidArgumentError(
            "Own-position prepared DPM prefill lacks its checkpoint or live "
            "state witness.");
      }
      break;
    default:
      return absl::InvalidArgumentError(
          "Prepared DPM prefill has an unknown start kind.");
  }
  if (plan.calls.empty() ||
      plan.calls.size() > kMaximumDPMPreparedPrefillCalls) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill has an empty or excessive call sequence.");
  }

  uint64_t position = plan.start_step;
  uint64_t total_tokens = 0;
  for (const DPMPreparedPrefillCall& call : plan.calls) {
    if (IsZeroHash(call.source_chunk_hash) ||
        call.start_token_position != position ||
        call.end_token_position <= call.start_token_position ||
        call.token_segments.empty() ||
        call.token_segments.size() >
            kMaximumDPMPreparedPrefillSegmentsPerCall) {
      return absl::InvalidArgumentError(
          "Prepared DPM prefill call has invalid provenance, position, or "
          "segments.");
    }
    switch (call.source_encoding) {
      case DPMPreparedPrefillSourceEncoding::kUtf8Text:
      case DPMPreparedPrefillSourceEncoding::kExactTokenIds:
        break;
      default:
        return absl::InvalidArgumentError(
            "Prepared DPM prefill call has an unknown source encoding.");
    }

    uint64_t call_tokens = 0;
    for (const DPMPreparedPrefillSegment& segment : call.token_segments) {
      if (segment.token_ids.empty()) {
        return absl::InvalidArgumentError(
            "Prepared DPM prefill contains an empty token segment.");
      }
      if (segment.token_ids.size() >
          kMaximumDPMPreparedPrefillTokenIds - total_tokens) {
        return absl::ResourceExhaustedError(
            "Prepared DPM prefill exceeds its token bound.");
      }
      for (int32_t token_id : segment.token_ids) {
        if (token_id < 0 ||
            static_cast<uint32_t>(token_id) >= plan.vocabulary_size) {
          return absl::InvalidArgumentError(
              "Prepared DPM prefill contains an out-of-vocabulary token.");
        }
      }
      call_tokens += segment.token_ids.size();
      total_tokens += segment.token_ids.size();
    }
    if (call.end_token_position - call.start_token_position != call_tokens) {
      return absl::InvalidArgumentError(
          "Prepared DPM prefill call positions differ from its exact token "
          "count.");
    }
    position = call.end_token_position;
  }
  if (position != plan.end_step || plan.end_step <= plan.start_step ||
      plan.end_step > kMaximumDPMPreparedPrefillTokenIds) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill end step differs from its physical calls or "
        "exceeds the admitted exact-history bound.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Hash256> ComputeDPMPreparedPrefillUtf8SourceChunkHash(
    absl::string_view utf8_text) {
  if (utf8_text.empty() ||
      utf8_text.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError(
        "Prepared DPM UTF-8 source chunk is empty or exceeds its canonical "
        "length field.");
  }
  std::string canonical;
  AppendU32(static_cast<uint32_t>(utf8_text.size()), &canonical);
  canonical.append(utf8_text.data(), utf8_text.size());
  const Hash256 result = HashCanonical(kUtf8SourceChunkDomain, canonical);
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "Prepared DPM UTF-8 source hashing produced an empty digest.");
  }
  return result;
}

absl::StatusOr<Hash256> ComputeDPMPreparedPrefillExactTokenSourceChunkHash(
    absl::Span<const int32_t> token_ids) {
  if (token_ids.empty() ||
      token_ids.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::InvalidArgumentError(
        "Prepared DPM exact-token source is empty or exceeds its canonical "
        "count field.");
  }
  std::string canonical;
  AppendU32(static_cast<uint32_t>(token_ids.size()), &canonical);
  for (int32_t token_id : token_ids) {
    if (token_id < 0) {
      return absl::InvalidArgumentError(
          "Prepared DPM exact-token source contains a negative token ID.");
    }
    AppendI32(token_id, &canonical);
  }
  const Hash256 result = HashCanonical(kTokenSourceChunkDomain, canonical);
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "Prepared DPM exact-token source hashing produced an empty digest.");
  }
  return result;
}

absl::StatusOr<Hash256> ComputeDPMPreparedPrefillSourceChunksHash(
    const std::vector<DPMPreparedPrefillCall>& calls) {
  if (calls.empty() || calls.size() > kMaximumDPMPreparedPrefillCalls) {
    return absl::InvalidArgumentError(
        "Cannot hash an empty or excessive prepared prefill source list.");
  }
  std::string canonical;
  AppendU32(DPMPreparedPrefillPlan::kFormatVersion, &canonical);
  AppendU32(static_cast<uint32_t>(calls.size()), &canonical);
  for (const DPMPreparedPrefillCall& call : calls) {
    if (IsZeroHash(call.source_chunk_hash)) {
      return absl::InvalidArgumentError(
          "Cannot hash an empty prepared prefill source digest.");
    }
    AppendU32(static_cast<uint32_t>(call.source_encoding), &canonical);
    AppendHash(call.source_chunk_hash, &canonical);
  }
  const Hash256 result = HashCanonical(kSourceChunksDomain, canonical);
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "Prepared prefill source hashing produced an empty digest.");
  }
  return result;
}

absl::StatusOr<Hash256> ComputeDPMPreparedPrefillResolvedTokenPlanHash(
    const DPMPreparedPrefillPlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateBaseFields(plan, false));
  std::string canonical;
  AppendU32(plan.format_version, &canonical);
  AppendIdentity(plan.session_identity, &canonical);
  AppendHash(plan.logical_agent_request_hash, &canonical);
  AppendHash(plan.canonical_source_chunks_hash, &canonical);
  AppendU32(static_cast<uint32_t>(plan.start_kind), &canonical);
  AppendOptionalHash(plan.restore_checkpoint_id, &canonical);
  AppendU64(plan.start_step, &canonical);
  AppendU64(plan.start_history_token_count, &canonical);
  AppendHash(plan.start_history_token_bytes_hash, &canonical);
  AppendU32(plan.batch_size, &canonical);
  AppendU32(plan.vocabulary_size, &canonical);
  AppendU64(plan.end_step, &canonical);
  AppendU32(static_cast<uint32_t>(plan.calls.size()), &canonical);
  for (const DPMPreparedPrefillCall& call : plan.calls) {
    AppendU32(static_cast<uint32_t>(call.source_encoding), &canonical);
    AppendHash(call.source_chunk_hash, &canonical);
    AppendU64(call.start_token_position, &canonical);
    AppendU64(call.end_token_position, &canonical);
    AppendU32(static_cast<uint32_t>(call.token_segments.size()), &canonical);
    for (const DPMPreparedPrefillSegment& segment : call.token_segments) {
      AppendU32(static_cast<uint32_t>(segment.token_ids.size()), &canonical);
      for (int32_t token_id : segment.token_ids) {
        AppendI32(token_id, &canonical);
      }
    }
  }
  const Hash256 result = HashCanonical(kResolvedTokenPlanDomain, canonical);
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "Prepared prefill token-plan hashing produced an empty digest.");
  }
  return result;
}

absl::StatusOr<Hash256> ComputeDPMPreparedPrefillShapeScheduleHash(
    const DPMPreparedPrefillPlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateBaseFields(plan, false));
  std::string canonical;
  AppendU32(plan.format_version, &canonical);
  AppendIdentity(plan.session_identity, &canonical);
  AppendU32(plan.batch_size, &canonical);
  AppendU64(plan.start_step, &canonical);
  AppendU64(plan.end_step, &canonical);
  AppendU32(static_cast<uint32_t>(plan.calls.size()), &canonical);
  for (const DPMPreparedPrefillCall& call : plan.calls) {
    AppendU64(call.start_token_position, &canonical);
    AppendU64(call.end_token_position, &canonical);
    AppendU32(static_cast<uint32_t>(call.token_segments.size()), &canonical);
    for (const DPMPreparedPrefillSegment& segment : call.token_segments) {
      AppendU64(segment.token_ids.size(), &canonical);
    }
  }
  const Hash256 result = HashCanonical(kShapeScheduleDomain, canonical);
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "Prepared prefill shape hashing produced an empty digest.");
  }
  return result;
}

absl::StatusOr<Hash256> ComputeDPMPreparedPrefillPlanId(
    const DPMPreparedPrefillPlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateBaseFields(plan, false));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 source_chunks_hash,
      ComputeDPMPreparedPrefillSourceChunksHash(plan.calls));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 resolved_token_plan_hash,
      ComputeDPMPreparedPrefillResolvedTokenPlanHash(plan));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 shape_schedule_hash,
      ComputeDPMPreparedPrefillShapeScheduleHash(plan));
  if (source_chunks_hash != plan.canonical_source_chunks_hash) {
    return absl::InvalidArgumentError(
        "Prepared prefill source-list hash is not canonical.");
  }

  std::string canonical;
  AppendU32(plan.format_version, &canonical);
  AppendU32(static_cast<uint32_t>(plan.start_kind), &canonical);
  AppendOptionalHash(plan.restore_checkpoint_id, &canonical);
  AppendOptionalHash(plan.start_state_witness_id, &canonical);
  AppendHash(source_chunks_hash, &canonical);
  AppendHash(resolved_token_plan_hash, &canonical);
  AppendHash(shape_schedule_hash, &canonical);
  const Hash256 result = HashCanonical(kPlanIdDomain, canonical);
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "Prepared prefill plan-id hashing produced an empty digest.");
  }
  return result;
}

absl::Status ValidateDPMPreparedPrefillPlan(
    const DPMPreparedPrefillPlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateBaseFields(plan, true));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_sources,
      ComputeDPMPreparedPrefillSourceChunksHash(plan.calls));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_tokens,
      ComputeDPMPreparedPrefillResolvedTokenPlanHash(plan));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_shapes,
      ComputeDPMPreparedPrefillShapeScheduleHash(plan));
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_id,
                        ComputeDPMPreparedPrefillPlanId(plan));
  if (plan.canonical_source_chunks_hash != expected_sources ||
      plan.resolved_token_plan_hash != expected_tokens ||
      plan.shape_schedule_hash != expected_shapes ||
      plan.plan_id != expected_id) {
    return absl::FailedPreconditionError(
        "Prepared DPM prefill plan has noncanonical provenance, token, shape, "
        "or plan digests.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeDPMPreparedPrefillPlan(
    const DPMPreparedPrefillPlan& plan) {
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(plan));
  std::string encoded;
  encoded.append(kPreparedPrefillPlanMagic.data(),
                 kPreparedPrefillPlanMagic.size());
  AppendU32(plan.format_version, &encoded);
  AppendHash(plan.plan_id, &encoded);
  AppendIdentity(plan.session_identity, &encoded);
  AppendHash(plan.logical_agent_request_hash, &encoded);
  AppendHash(plan.canonical_source_chunks_hash, &encoded);
  AppendU32(static_cast<uint32_t>(plan.start_kind), &encoded);
  AppendOptionalHash(plan.restore_checkpoint_id, &encoded);
  AppendOptionalHash(plan.start_state_witness_id, &encoded);
  AppendU64(plan.start_step, &encoded);
  AppendU64(plan.start_history_token_count, &encoded);
  AppendHash(plan.start_history_token_bytes_hash, &encoded);
  AppendU32(plan.batch_size, &encoded);
  AppendU32(plan.vocabulary_size, &encoded);
  AppendU64(plan.end_step, &encoded);
  AppendU32(static_cast<uint32_t>(plan.calls.size()), &encoded);
  for (const DPMPreparedPrefillCall& call : plan.calls) {
    AppendU32(static_cast<uint32_t>(call.source_encoding), &encoded);
    AppendHash(call.source_chunk_hash, &encoded);
    AppendU64(call.start_token_position, &encoded);
    AppendU64(call.end_token_position, &encoded);
    AppendU32(static_cast<uint32_t>(call.token_segments.size()), &encoded);
    for (const DPMPreparedPrefillSegment& segment : call.token_segments) {
      AppendU32(static_cast<uint32_t>(segment.token_ids.size()), &encoded);
      for (int32_t token_id : segment.token_ids) {
        AppendI32(token_id, &encoded);
      }
    }
  }
  AppendHash(plan.resolved_token_plan_hash, &encoded);
  AppendHash(plan.shape_schedule_hash, &encoded);
  if (encoded.size() > kMaximumDPMPreparedPrefillPlanBytes) {
    return absl::ResourceExhaustedError(
        "Prepared DPM prefill plan encoding exceeds its transport bound.");
  }
  return encoded;
}

absl::StatusOr<DPMPreparedPrefillPlan> DecodeDPMPreparedPrefillPlan(
    absl::string_view bytes) {
  if (bytes.size() > kMaximumDPMPreparedPrefillPlanBytes ||
      bytes.size() < kPreparedPrefillPlanMagic.size() + 4) {
    return absl::ResourceExhaustedError(
        "Prepared DPM prefill plan bytes are empty, truncated, or oversized.");
  }
  CanonicalReader reader(bytes);
  ABSL_ASSIGN_OR_RETURN(
      const absl::string_view magic,
      reader.ReadBytes(kPreparedPrefillPlanMagic.size()));
  if (magic != kPreparedPrefillPlanMagic) {
    return absl::DataLossError(
        "Prepared DPM prefill plan has an invalid magic value.");
  }

  DPMPreparedPrefillPlan plan;
  ABSL_ASSIGN_OR_RETURN(plan.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.plan_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.session_identity.model_artifact_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.session_identity.runtime_artifact_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.session_identity.inference_profile_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.logical_agent_request_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.canonical_source_chunks_hash, reader.ReadHash());
  uint32_t start_kind = 0;
  ABSL_ASSIGN_OR_RETURN(start_kind, reader.ReadU32());
  plan.start_kind = static_cast<DPMPreparedPrefillStartKind>(start_kind);
  ABSL_ASSIGN_OR_RETURN(plan.restore_checkpoint_id,
                        reader.ReadOptionalHash());
  ABSL_ASSIGN_OR_RETURN(plan.start_state_witness_id,
                        reader.ReadOptionalHash());
  ABSL_ASSIGN_OR_RETURN(plan.start_step, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(plan.start_history_token_count, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(plan.start_history_token_bytes_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.batch_size, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.vocabulary_size, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(plan.end_step, reader.ReadU64());
  uint32_t call_count = 0;
  ABSL_ASSIGN_OR_RETURN(call_count, reader.ReadU32());
  if (call_count == 0 || call_count > kMaximumDPMPreparedPrefillCalls) {
    return absl::DataLossError(
        "Prepared DPM prefill plan has an invalid call count.");
  }
  plan.calls.reserve(call_count);
  uint64_t total_token_ids = 0;
  for (uint32_t call_index = 0; call_index < call_count; ++call_index) {
    DPMPreparedPrefillCall call;
    uint32_t source_encoding = 0;
    ABSL_ASSIGN_OR_RETURN(source_encoding, reader.ReadU32());
    call.source_encoding =
        static_cast<DPMPreparedPrefillSourceEncoding>(source_encoding);
    ABSL_ASSIGN_OR_RETURN(call.source_chunk_hash, reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(call.start_token_position, reader.ReadU64());
    ABSL_ASSIGN_OR_RETURN(call.end_token_position, reader.ReadU64());
    uint32_t segment_count = 0;
    ABSL_ASSIGN_OR_RETURN(segment_count, reader.ReadU32());
    if (segment_count == 0 ||
        segment_count > kMaximumDPMPreparedPrefillSegmentsPerCall) {
      return absl::DataLossError(
          "Prepared DPM prefill plan has an invalid segment count.");
    }
    call.token_segments.reserve(segment_count);
    for (uint32_t segment_index = 0; segment_index < segment_count;
         ++segment_index) {
      uint32_t token_count = 0;
      ABSL_ASSIGN_OR_RETURN(token_count, reader.ReadU32());
      if (token_count == 0 ||
          token_count > kMaximumDPMPreparedPrefillTokenIds -
                            total_token_ids) {
        return absl::DataLossError(
            "Prepared DPM prefill plan has an invalid token count.");
      }
      DPMPreparedPrefillSegment segment;
      segment.token_ids.reserve(token_count);
      for (uint32_t token_index = 0; token_index < token_count;
           ++token_index) {
        uint32_t encoded_token = 0;
        ABSL_ASSIGN_OR_RETURN(encoded_token, reader.ReadU32());
        if (encoded_token >
            static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
          return absl::DataLossError(
              "Prepared DPM prefill plan contains a negative token ID.");
        }
        segment.token_ids.push_back(static_cast<int32_t>(encoded_token));
      }
      total_token_ids += token_count;
      call.token_segments.push_back(std::move(segment));
    }
    plan.calls.push_back(std::move(call));
  }
  ABSL_ASSIGN_OR_RETURN(plan.resolved_token_plan_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(plan.shape_schedule_hash, reader.ReadHash());
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "Prepared DPM prefill plan has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(plan));
  ABSL_ASSIGN_OR_RETURN(const std::string canonical,
                        EncodeDPMPreparedPrefillPlan(plan));
  if (canonical != bytes) {
    return absl::DataLossError(
        "Prepared DPM prefill plan bytes are not canonical.");
  }
  return plan;
}

}  // namespace litert::lm
