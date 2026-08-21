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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PREPARED_PREFILL_PLAN_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PREPARED_PREFILL_PLAN_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

inline constexpr uint32_t kMaximumDPMPreparedPrefillCalls = 65'536;
inline constexpr uint32_t kMaximumDPMPreparedPrefillSegmentsPerCall = 4;
inline constexpr uint64_t kMaximumDPMPreparedPrefillTokenIds = 1'000'000;
inline constexpr uint64_t kMaximumDPMPreparedPrefillPlanBytes =
    uint64_t{64} * 1024 * 1024;

enum class DPMPreparedPrefillStartKind : uint32_t {
  // A newly created step-zero session. The plan explicitly resolves any BOS
  // insertion that ordinary Session preprocessing would otherwise hide.
  kFreshSession = 1,
  // A post-decode own-position session authenticated by a concrete restore
  // operation. Off-position grafting is intentionally not representable.
  kOwnPositionRestore = 2,
};

enum class DPMPreparedPrefillSourceEncoding : uint32_t {
  kUtf8Text = 1,
  kExactTokenIds = 2,
};

// One exact tensor-buffer segment supplied to a preprocessed prefill call.
// A fresh first call can have a separate one-token BOS segment followed by the
// source chunk's resolved tokens; later calls normally contain one segment.
struct DPMPreparedPrefillSegment {
  std::vector<int32_t> token_ids;

  bool operator==(const DPMPreparedPrefillSegment& other) const {
    return token_ids == other.token_ids;
  }
};

// One physical Session prefill call. The source chunk hash commits to the raw
// canonical text or exact token IDs and its encoding without retaining a
// second copy of the potentially large logical request in this execution plan.
struct DPMPreparedPrefillCall {
  DPMPreparedPrefillSourceEncoding source_encoding =
      DPMPreparedPrefillSourceEncoding::kUtf8Text;
  Hash256 source_chunk_hash;
  uint64_t start_token_position = 0;
  uint64_t end_token_position = 0;
  std::vector<DPMPreparedPrefillSegment> token_segments;

  bool operator==(const DPMPreparedPrefillCall& other) const {
    return source_encoding == other.source_encoding &&
           source_chunk_hash == other.source_chunk_hash &&
           start_token_position == other.start_token_position &&
           end_token_position == other.end_token_position &&
           token_segments == other.token_segments;
  }
};

// Runtime-derived physical prefill plan. This is deliberately distinct from
// the logical DPM replay request: checkpoints may change the selected suffix
// and starting state without changing the logical request or winner key.
//
// `resolved_token_plan_hash` commits to every exact token ID, call/segment
// boundary, and semantic start history, but intentionally excludes the
// transport-specific witness ID. Independent exact workers reauthenticate the
// same state under per-run keys, so their envelope witnesses differ while the
// model-affecting token plan must agree. `plan_id` binds that run's witness;
// `shape_schedule_hash` separately commits to the batch-one API shape schedule.
// SessionHandoffIdentity pins the loaded tokenizer, backend, compiled runtime,
// and internal prefill-chunking contract.
struct DPMPreparedPrefillPlan {
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;
  Hash256 plan_id;
  SessionHandoffIdentity session_identity;
  Hash256 logical_agent_request_hash;
  Hash256 canonical_source_chunks_hash;

  DPMPreparedPrefillStartKind start_kind =
      DPMPreparedPrefillStartKind::kFreshSession;
  std::optional<Hash256> restore_checkpoint_id;
  std::optional<Hash256> start_state_witness_id;
  uint64_t start_step = 0;
  uint64_t start_history_token_count = 0;
  Hash256 start_history_token_bytes_hash;

  uint32_t batch_size = 1;
  uint32_t vocabulary_size = 0;
  uint64_t end_step = 0;
  std::vector<DPMPreparedPrefillCall> calls;

  Hash256 resolved_token_plan_hash;
  Hash256 shape_schedule_hash;

  bool operator==(const DPMPreparedPrefillPlan& other) const {
    return format_version == other.format_version &&
           plan_id == other.plan_id &&
           session_identity == other.session_identity &&
           logical_agent_request_hash == other.logical_agent_request_hash &&
           canonical_source_chunks_hash ==
               other.canonical_source_chunks_hash &&
           start_kind == other.start_kind &&
           restore_checkpoint_id == other.restore_checkpoint_id &&
           start_state_witness_id == other.start_state_witness_id &&
           start_step == other.start_step &&
           start_history_token_count == other.start_history_token_count &&
           start_history_token_bytes_hash ==
               other.start_history_token_bytes_hash &&
           batch_size == other.batch_size &&
           vocabulary_size == other.vocabulary_size &&
           end_step == other.end_step && calls == other.calls &&
           resolved_token_plan_hash == other.resolved_token_plan_hash &&
           shape_schedule_hash == other.shape_schedule_hash;
  }
};

// Canonical commitments to the source bytes that produced one prepared call.
// These are public so replay/evidence boundaries can independently join a
// runtime-derived plan back to the logical chunks without re-tokenizing text
// or duplicating the hashing contract.
absl::StatusOr<Hash256> ComputeDPMPreparedPrefillUtf8SourceChunkHash(
    absl::string_view utf8_text);
absl::StatusOr<Hash256> ComputeDPMPreparedPrefillExactTokenSourceChunkHash(
    absl::Span<const int32_t> token_ids);

// Hash of the ordered source_chunk_hash values and their encodings. This lets
// the plan validator prove that no source chunk was reordered or dropped.
absl::StatusOr<Hash256> ComputeDPMPreparedPrefillSourceChunksHash(
    const std::vector<DPMPreparedPrefillCall>& calls);

absl::StatusOr<Hash256> ComputeDPMPreparedPrefillResolvedTokenPlanHash(
    const DPMPreparedPrefillPlan& plan);
absl::StatusOr<Hash256> ComputeDPMPreparedPrefillShapeScheduleHash(
    const DPMPreparedPrefillPlan& plan);
absl::StatusOr<Hash256> ComputeDPMPreparedPrefillPlanId(
    const DPMPreparedPrefillPlan& plan);
absl::Status ValidateDPMPreparedPrefillPlan(
    const DPMPreparedPrefillPlan& plan);

// Canonical transport form used inside authenticated fresh-worker results.
// This encoding is content-addressed but not authenticated by itself; callers
// must retain the enclosing worker/result authentication boundary.
absl::StatusOr<std::string> EncodeDPMPreparedPrefillPlan(
    const DPMPreparedPrefillPlan& plan);
absl::StatusOr<DPMPreparedPrefillPlan> DecodeDPMPreparedPrefillPlan(
    absl::string_view bytes);

// Runtime-owned identities used by Coverage V2 admission. The plan contract
// binds the canonical field/hash/transport semantics; the shape-class contract
// separately binds batch-one call and tensor-segment scheduling.
Hash256 GetDPMPreparedPrefillPlanContractHash();
Hash256 GetDPMPreparedPrefillShapeClassContractHash();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PREPARED_PREFILL_PLAN_H_
