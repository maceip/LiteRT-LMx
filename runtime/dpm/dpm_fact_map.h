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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_FACT_MAP_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_FACT_MAP_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"     // from @com_google_absl
#include "absl/strings/string_view.h" // from @com_google_absl
#include "absl/types/span.h"          // from @com_google_absl
#include "runtime/dpm/dpm_event_log.h"

namespace litert::lm {

inline constexpr absl::string_view kDPMFactMapSchemaId = "dpm.factmap.v1";
inline constexpr absl::string_view kDPMToolDeltaSchemaId =
    "dpm.factmap.delta.v1";

struct DPMFactMapLimits {
  size_t memory_budget_bytes = 4096;
  uint32_t max_facts = 4096;
  size_t max_value_bytes = 4096;
};

struct DPMFact {
  std::string id;
  std::string value;
  uint64_t source_event = 0;
  uint64_t valid_from = 0;
  std::optional<uint64_t> retracted_at;
  std::optional<std::string> prev;

  bool operator==(const DPMFact &other) const = default;
};

struct DPMFactMap {
  std::vector<DPMFact> facts;
};

struct DPMFactRetraction {
  std::string id;
  uint64_t source_event = 0;

  bool operator==(const DPMFactRetraction &other) const = default;
};

// Result of folding all deterministic events in a projection range. Events
// left in llm_event_indices are the only raw events exposed to the model.
// canonical_memory contains the deterministic state before the model delta.
// Retractions are retained separately as well as applied to already-known
// facts so a full-log rebuild can apply a later tombstone to a fact first
// materialized by the model delta.
struct DPMFactMapProjectionFold {
  uint64_t event_range_start = 0;
  uint64_t source_event_count = 0;
  std::string canonical_memory;
  std::vector<uint64_t> llm_event_indices;
  std::vector<DPMFactRetraction> retractions;
};

// Frozen JSON schema for the model-emitted fact delta. The model emits the
// same shape as dpm.factmap.v1 but only includes changed facts.
const std::string &DPMFactMapJsonSchema();

// Canonical empty full map.
std::string EmptyDPMFactMap();

// Validates a complete fact map and emits deterministic compact JSON with
// facts sorted by id and every fact field in frozen order.
absl::StatusOr<std::string>
CanonicalizeDPMFactMap(absl::string_view raw_output,
                       uint64_t source_event_count,
                       const DPMFactMapLimits &limits);

// Validates a model-produced delta. Every fact must cite one of the exact raw
// event indices exposed in the prompt, never a deterministic tool/correction
// event or an event outside the current source prefix.
absl::StatusOr<std::string>
CanonicalizeDPMFactDelta(absl::string_view raw_output,
                         absl::Span<const uint64_t> allowed_source_events,
                         uint64_t source_event_count,
                         const DPMFactMapLimits &limits);

// Pure C++ last-writer merge. Higher source_event wins. Equal-source
// disagreement fails closed; unchanged facts are copied byte-for-byte through
// the canonical encoder without another model call.
absl::StatusOr<std::string> MergeDPMFactMap(absl::string_view canonical_base,
                                            absl::string_view canonical_delta,
                                            uint64_t source_event_count,
                                            const DPMFactMapLimits &limits);

// Applies typed tool deltas and correction tombstones from [range_start,
// range_end). Non-typed user/internal/model/tool events are returned as the
// exact model-visible delta set.
absl::StatusOr<DPMFactMapProjectionFold>
FoldDeterministicDPMEvents(absl::string_view canonical_base,
                           const DPMLogSnapshot &snapshot, uint64_t range_start,
                           uint64_t range_end, const DPMFactMapLimits &limits);

// Completes a deterministic fold. A model delta is required exactly when the
// fold retained model-visible events and may cite only those absolute event
// indices. Deterministic retractions are then replayed with last-writer
// semantics so correction behavior is identical for baseline and full-log
// reconstruction.
absl::StatusOr<std::string>
FinalizeDPMFactMapProjection(const DPMFactMapProjectionFold &fold,
                             std::optional<absl::string_view> raw_model_delta,
                             uint64_t source_event_count,
                             const DPMFactMapLimits &limits);

// Canonical correction payload accepted by DPMEngine::AppendCorrection:
// {"retract":["fact-id",...]}. IDs must be unique and sorted.
absl::StatusOr<std::string>
CanonicalizeDPMRetractionPayload(absl::string_view raw_payload);

// Compact model-visible summary. Values are UTF-8-safe prefixes only; the full
// baseline values never re-enter an incremental projection prompt.
absl::StatusOr<std::string> RenderDPMFactHeads(absl::string_view canonical_map,
                                               uint64_t source_event_count,
                                               const DPMFactMapLimits &limits,
                                               size_t maximum_head_bytes,
                                               size_t maximum_total_bytes);

} // namespace litert::lm

#endif // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_FACT_MAP_H_
