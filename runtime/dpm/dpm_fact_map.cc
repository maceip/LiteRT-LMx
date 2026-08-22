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

#include "runtime/dpm/dpm_fact_map.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"       // from @com_google_absl
#include "absl/strings/string_view.h"   // from @com_google_absl
#include "absl/types/span.h"            // from @com_google_absl
#include "nlohmann/json.hpp"            // from @nlohmann_json

namespace litert::lm {
namespace {

constexpr size_t kMaximumFactIdBytes = 128;
constexpr size_t kMaximumRetractions = 4096;

bool IsValidUtf8(absl::string_view text) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t lead = bytes[index++];
    if (lead <= 0x7f) continue;
    size_t continuations = 0;
    uint8_t minimum_second = 0x80;
    uint8_t maximum_second = 0xbf;
    if (lead >= 0xc2 && lead <= 0xdf) {
      continuations = 1;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      continuations = 2;
      if (lead == 0xe0) minimum_second = 0xa0;
      if (lead == 0xed) maximum_second = 0x9f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      continuations = 3;
      if (lead == 0xf0) minimum_second = 0x90;
      if (lead == 0xf4) maximum_second = 0x8f;
    } else {
      return false;
    }
    if (continuations > text.size() - index || bytes[index] < minimum_second ||
        bytes[index] > maximum_second) {
      return false;
    }
    ++index;
    for (size_t i = 1; i < continuations; ++i, ++index) {
      if (bytes[index] < 0x80 || bytes[index] > 0xbf) return false;
    }
  }
  return true;
}

bool IsValidFactId(absl::string_view id) {
  if (id.empty() || id.size() > kMaximumFactIdBytes) return false;
  for (unsigned char byte : id) {
    const bool alpha_numeric = (byte >= 'a' && byte <= 'z') ||
                               (byte >= 'A' && byte <= 'Z') ||
                               (byte >= '0' && byte <= '9');
    if (!alpha_numeric && byte != '.' && byte != '_' && byte != ':' &&
        byte != '-') {
      return false;
    }
  }
  return true;
}

absl::Status ValidateLimits(const DPMFactMapLimits& limits) {
  if (limits.memory_budget_bytes == 0 || limits.max_facts == 0 ||
      limits.max_value_bytes == 0 ||
      limits.max_value_bytes > limits.memory_budget_bytes) {
    return absl::InvalidArgumentError(
        "DPM fact-map limits must be positive and internally bounded.");
  }
  return absl::OkStatus();
}

absl::StatusOr<nlohmann::ordered_json> ParseStrictJson(absl::string_view raw) {
  bool duplicate_key = false;
  std::vector<std::set<std::string>> object_keys;
  try {
    nlohmann::ordered_json parsed = nlohmann::ordered_json::parse(
        raw.begin(), raw.end(),
        [&duplicate_key, &object_keys](
            int, nlohmann::ordered_json::parse_event_t event,
            nlohmann::ordered_json& value) {
          if (event == nlohmann::ordered_json::parse_event_t::object_start) {
            object_keys.emplace_back();
          } else if (event == nlohmann::ordered_json::parse_event_t::key) {
            if (object_keys.empty() ||
                !object_keys.back().insert(value.get<std::string>()).second) {
              duplicate_key = true;
            }
          } else if (event ==
                     nlohmann::ordered_json::parse_event_t::object_end) {
            if (!object_keys.empty()) object_keys.pop_back();
          }
          return true;
        },
        /*allow_exceptions=*/true, /*ignore_comments=*/false);
    if (duplicate_key) {
      return absl::InvalidArgumentError(
          "DPM fact-map JSON contains a duplicate object key.");
    }
    return parsed;
  } catch (const std::exception&) {
    return absl::InvalidArgumentError(
        "DPM fact-map bytes are not valid strict JSON.");
  }
}

absl::Status ValidateFact(
    const DPMFact& fact, uint64_t source_event_count,
    const DPMFactMapLimits& limits,
    const std::optional<std::set<uint64_t>>& allowed_source_events) {
  if (!IsValidFactId(fact.id) || fact.value.empty() ||
      fact.value.size() > limits.max_value_bytes || !IsValidUtf8(fact.value)) {
    return absl::InvalidArgumentError(
        "DPM fact requires a bounded canonical id and nonempty UTF-8 value.");
  }
  if (source_event_count == 0 || fact.source_event >= source_event_count ||
      fact.valid_from > fact.source_event) {
    return absl::InvalidArgumentError(
        "DPM fact provenance falls outside its authoritative event prefix.");
  }
  if (allowed_source_events.has_value() &&
      !allowed_source_events->contains(fact.source_event)) {
    return absl::InvalidArgumentError(
        "DPM fact cites an event that was not exposed to the model.");
  }
  if (fact.retracted_at.has_value() &&
      (*fact.retracted_at < fact.valid_from ||
       *fact.retracted_at >= source_event_count)) {
    return absl::InvalidArgumentError(
        "DPM fact tombstone falls outside its authoritative lifetime.");
  }
  if (fact.prev.has_value() &&
      (!IsValidFactId(*fact.prev) || *fact.prev == fact.id)) {
    return absl::InvalidArgumentError(
        "DPM fact prev must name a different canonical fact id.");
  }
  return absl::OkStatus();
}

absl::StatusOr<DPMFact> ParseFact(
    const nlohmann::ordered_json& value, uint64_t source_event_count,
    const DPMFactMapLimits& limits,
    const std::optional<std::set<uint64_t>>& allowed_source_events) {
  if (!value.is_object() || value.size() != 6 || !value.contains("id") ||
      !value.contains("value") || !value.contains("source_event") ||
      !value.contains("valid_from") || !value.contains("retracted_at") ||
      !value.contains("prev") || !value["id"].is_string() ||
      !value["value"].is_string() ||
      !value["source_event"].is_number_unsigned() ||
      !value["valid_from"].is_number_unsigned() ||
      (!value["retracted_at"].is_null() &&
       !value["retracted_at"].is_number_unsigned()) ||
      (!value["prev"].is_null() && !value["prev"].is_string())) {
    return absl::InvalidArgumentError(
        "Every DPM fact must contain exactly the six frozen typed fields.");
  }
  DPMFact fact{
      .id = value["id"].get<std::string>(),
      .value = value["value"].get<std::string>(),
      .source_event = value["source_event"].get<uint64_t>(),
      .valid_from = value["valid_from"].get<uint64_t>(),
  };
  if (!value["retracted_at"].is_null()) {
    fact.retracted_at = value["retracted_at"].get<uint64_t>();
  }
  if (!value["prev"].is_null()) {
    fact.prev = value["prev"].get<std::string>();
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFact(fact, source_event_count, limits, allowed_source_events));
  return fact;
}

absl::StatusOr<DPMFactMap> ParseFactMap(
    absl::string_view raw, uint64_t source_event_count,
    const DPMFactMapLimits& limits,
    const std::optional<std::set<uint64_t>>& allowed_source_events) {
  ABSL_RETURN_IF_ERROR(ValidateLimits(limits));
  if (raw.empty() || raw.size() > limits.memory_budget_bytes ||
      !IsValidUtf8(raw)) {
    return absl::InvalidArgumentError(
        "DPM fact map is not bounded nonempty UTF-8 JSON.");
  }
  ABSL_ASSIGN_OR_RETURN(nlohmann::ordered_json parsed, ParseStrictJson(raw));
  if (!parsed.is_object() || parsed.size() != 2 ||
      !parsed.contains("schema_id") || !parsed["schema_id"].is_string() ||
      parsed["schema_id"].get<std::string>() != kDPMFactMapSchemaId ||
      !parsed.contains("facts") || !parsed["facts"].is_array() ||
      parsed["facts"].size() > limits.max_facts) {
    return absl::InvalidArgumentError(
        "DPM fact map must contain exactly schema_id and bounded facts.");
  }
  DPMFactMap map;
  map.facts.reserve(parsed["facts"].size());
  std::set<std::string> ids;
  for (const nlohmann::ordered_json& item : parsed["facts"]) {
    ABSL_ASSIGN_OR_RETURN(
        DPMFact fact,
        ParseFact(item, source_event_count, limits, allowed_source_events));
    if (!ids.insert(fact.id).second) {
      return absl::InvalidArgumentError(
          "DPM fact map contains a duplicate fact id.");
    }
    map.facts.push_back(std::move(fact));
  }
  return map;
}

nlohmann::ordered_json EncodeFact(const DPMFact& fact) {
  nlohmann::ordered_json encoded = nlohmann::ordered_json::object();
  encoded["id"] = fact.id;
  encoded["value"] = fact.value;
  encoded["source_event"] = fact.source_event;
  encoded["valid_from"] = fact.valid_from;
  encoded["retracted_at"] = fact.retracted_at.has_value()
                                ? nlohmann::ordered_json(*fact.retracted_at)
                                : nlohmann::ordered_json(nullptr);
  encoded["prev"] = fact.prev.has_value() ? nlohmann::ordered_json(*fact.prev)
                                          : nlohmann::ordered_json(nullptr);
  return encoded;
}

absl::StatusOr<std::string> EncodeFactMap(DPMFactMap map,
                                          const DPMFactMapLimits& limits) {
  std::sort(map.facts.begin(), map.facts.end(),
            [](const DPMFact& left, const DPMFact& right) {
              return left.id < right.id;
            });
  nlohmann::ordered_json encoded = nlohmann::ordered_json::object();
  encoded["schema_id"] = kDPMFactMapSchemaId;
  encoded["facts"] = nlohmann::ordered_json::array();
  for (const DPMFact& fact : map.facts) {
    encoded["facts"].push_back(EncodeFact(fact));
  }
  const std::string bytes = encoded.dump();
  if (bytes.size() > limits.memory_budget_bytes) {
    return absl::ResourceExhaustedError(
        "Canonical DPM fact map exceeds its memory budget.");
  }
  return bytes;
}

absl::Status MergeFacts(const DPMFactMap& delta,
                        std::map<std::string, DPMFact>* facts) {
  for (const DPMFact& incoming : delta.facts) {
    auto existing = facts->find(incoming.id);
    if (existing == facts->end()) {
      facts->emplace(incoming.id, incoming);
      continue;
    }
    if (incoming.source_event > existing->second.source_event) {
      existing->second = incoming;
    } else if (incoming.source_event == existing->second.source_event &&
               !(incoming == existing->second)) {
      return absl::DataLossError(
          "DPM fact merge has conflicting values at one source event.");
    }
  }
  return absl::OkStatus();
}

absl::Status ApplyRetractions(absl::Span<const std::string> ids,
                              uint64_t event_index, bool require_existing,
                              std::map<std::string, DPMFact>* facts) {
  for (const std::string& id : ids) {
    auto existing = facts->find(id);
    if (existing == facts->end()) {
      if (require_existing) {
        return absl::FailedPreconditionError(
            "DPM correction retracts an unknown fact id.");
      }
      continue;
    }
    if (event_index < existing->second.source_event) {
      continue;
    }
    existing->second.source_event = event_index;
    existing->second.retracted_at = event_index;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::string>> ParseRetractions(
    const nlohmann::ordered_json& value) {
  if (!value.is_array() || value.size() > kMaximumRetractions) {
    return absl::InvalidArgumentError(
        "DPM retractions must be a bounded JSON array.");
  }
  std::vector<std::string> ids;
  ids.reserve(value.size());
  for (const nlohmann::ordered_json& item : value) {
    if (!item.is_string()) {
      return absl::InvalidArgumentError(
          "Every DPM retraction must be a fact id string.");
    }
    std::string id = item.get<std::string>();
    if (!IsValidFactId(id)) {
      return absl::InvalidArgumentError(
          "DPM retraction contains a non-canonical fact id.");
    }
    ids.push_back(std::move(id));
  }
  if (!std::is_sorted(ids.begin(), ids.end()) ||
      std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
    return absl::InvalidArgumentError(
        "DPM retraction ids must be unique and sorted.");
  }
  return ids;
}

absl::StatusOr<std::optional<std::pair<DPMFactMap, std::vector<std::string>>>>
ParseTypedToolDelta(absl::string_view payload, uint64_t event_index,
                    uint64_t source_event_count,
                    const DPMFactMapLimits& limits) {
  absl::StatusOr<nlohmann::ordered_json> parsed = ParseStrictJson(payload);
  if (!parsed.ok() || !parsed->is_object() || !parsed->contains("schema_id") ||
      !(*parsed)["schema_id"].is_string() ||
      (*parsed)["schema_id"].get<std::string>() != kDPMToolDeltaSchemaId) {
    return std::optional<std::pair<DPMFactMap, std::vector<std::string>>>();
  }
  if (parsed->size() != 3 || !parsed->contains("facts") ||
      !(*parsed)["facts"].is_array() || !parsed->contains("retract")) {
    return absl::InvalidArgumentError(
        "Typed DPM tool delta has fields outside its frozen schema.");
  }
  const std::set<uint64_t> allowed_sources = {event_index};
  DPMFactMap delta;
  if ((*parsed)["facts"].size() > limits.max_facts) {
    return absl::ResourceExhaustedError(
        "Typed DPM tool delta contains too many facts.");
  }
  std::set<std::string> ids;
  for (const nlohmann::ordered_json& item : (*parsed)["facts"]) {
    ABSL_ASSIGN_OR_RETURN(DPMFact fact, ParseFact(item, source_event_count,
                                                  limits, allowed_sources));
    if (!ids.insert(fact.id).second) {
      return absl::InvalidArgumentError(
          "Typed DPM tool delta contains a duplicate fact id.");
    }
    delta.facts.push_back(std::move(fact));
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<std::string> retract,
                        ParseRetractions((*parsed)["retract"]));
  return std::optional<std::pair<DPMFactMap, std::vector<std::string>>>(
      std::in_place, std::move(delta), std::move(retract));
}

absl::StatusOr<std::vector<std::string>> ParseCorrectionRetractions(
    absl::string_view payload) {
  ABSL_ASSIGN_OR_RETURN(nlohmann::ordered_json parsed,
                        ParseStrictJson(payload));
  if (!parsed.is_object() || parsed.size() != 1 ||
      !parsed.contains("retract")) {
    return absl::InvalidArgumentError(
        "DPM correction must contain exactly a retract array.");
  }
  return ParseRetractions(parsed["retract"]);
}

DPMFactMap MapValues(const std::map<std::string, DPMFact>& values) {
  DPMFactMap map;
  map.facts.reserve(values.size());
  for (const auto& [id, fact] : values) map.facts.push_back(fact);
  return map;
}

std::string Utf8Head(absl::string_view value, size_t maximum_bytes) {
  size_t length = std::min(value.size(), maximum_bytes);
  while (length > 0 && !IsValidUtf8(value.substr(0, length))) --length;
  return std::string(value.substr(0, length));
}

}  // namespace

const std::string& DPMFactMapJsonSchema() {
  static const std::string* schema = new std::string(
      R"({"type":"object","additionalProperties":false,"required":["schema_id","facts"],"properties":{"schema_id":{"const":"dpm.factmap"},"facts":{"type":"array","items":{"type":"object","additionalProperties":false,"required":["id","value","source_event","valid_from","retracted_at","prev"],"properties":{"id":{"type":"string","minLength":1,"maxLength":128,"pattern":"^[A-Za-z0-9._:-]+$"},"value":{"type":"string","minLength":1},"source_event":{"type":"integer","minimum":0},"valid_from":{"type":"integer","minimum":0},"retracted_at":{"type":["integer","null"],"minimum":0},"prev":{"type":["string","null"]}}}}}})");
  return *schema;
}

std::string EmptyDPMFactMap() {
  return std::string("{\"schema_id\":\"dpm.factmap\",\"facts\":[]}");
}

absl::StatusOr<std::string> CanonicalizeDPMFactMap(
    absl::string_view raw_output, uint64_t source_event_count,
    const DPMFactMapLimits& limits) {
  ABSL_ASSIGN_OR_RETURN(
      DPMFactMap map,
      ParseFactMap(raw_output, source_event_count, limits, std::nullopt));
  return EncodeFactMap(std::move(map), limits);
}

absl::StatusOr<std::string> CanonicalizeDPMFactDelta(
    absl::string_view raw_output,
    absl::Span<const uint64_t> allowed_source_events,
    uint64_t source_event_count, const DPMFactMapLimits& limits) {
  const std::set<uint64_t> allowed(allowed_source_events.begin(),
                                   allowed_source_events.end());
  if (allowed.empty()) {
    return absl::InvalidArgumentError(
        "A model fact delta requires at least one exposed source event.");
  }
  ABSL_ASSIGN_OR_RETURN(
      DPMFactMap map,
      ParseFactMap(raw_output, source_event_count, limits, allowed));
  return EncodeFactMap(std::move(map), limits);
}

absl::StatusOr<std::string> MergeDPMFactMap(absl::string_view canonical_base,
                                            absl::string_view canonical_delta,
                                            uint64_t source_event_count,
                                            const DPMFactMapLimits& limits) {
  ABSL_ASSIGN_OR_RETURN(
      DPMFactMap base,
      ParseFactMap(canonical_base, source_event_count, limits, std::nullopt));
  ABSL_ASSIGN_OR_RETURN(
      DPMFactMap delta,
      ParseFactMap(canonical_delta, source_event_count, limits, std::nullopt));
  std::map<std::string, DPMFact> facts;
  for (DPMFact& fact : base.facts) facts.emplace(fact.id, std::move(fact));
  ABSL_RETURN_IF_ERROR(MergeFacts(delta, &facts));
  return EncodeFactMap(MapValues(facts), limits);
}

absl::StatusOr<DPMFactMapProjectionFold> FoldDeterministicDPMEvents(
    absl::string_view canonical_base, const DPMLogSnapshot& snapshot,
    uint64_t range_start, uint64_t range_end, const DPMFactMapLimits& limits) {
  if (snapshot.generation != snapshot.events.size() ||
      range_start > range_end || range_end > snapshot.events.size() ||
      range_end == 0) {
    return absl::InvalidArgumentError(
        "DPM deterministic fold received an invalid event range.");
  }
  // A baseline is authoritative only for [0, range_start). Parsing it against
  // range_end would accidentally admit facts that cite the very delta this
  // fold is supposed to derive.
  ABSL_ASSIGN_OR_RETURN(
      DPMFactMap base,
      ParseFactMap(canonical_base, range_start, limits, std::nullopt));
  std::map<std::string, DPMFact> facts;
  for (DPMFact& fact : base.facts) facts.emplace(fact.id, std::move(fact));

  DPMFactMapProjectionFold result{
      .event_range_start = range_start,
      .source_event_count = range_end,
  };
  for (uint64_t index = range_start; index < range_end; ++index) {
    const DPMEvent& event = snapshot.events[index];
    if (event.index != index || event.case_id != snapshot.case_id) {
      return absl::DataLossError(
          "DPM deterministic fold received a contaminated event prefix.");
    }
    if (event.kind == DPMEvent::Kind::kCorrection) {
      ABSL_ASSIGN_OR_RETURN(std::vector<std::string> retract,
                            ParseCorrectionRetractions(event.payload));
      for (const std::string& id : retract) {
        result.retractions.push_back(
            DPMFactRetraction{.id = id, .source_event = event.index});
      }
      ABSL_RETURN_IF_ERROR(ApplyRetractions(retract, event.index,
                                            /*require_existing=*/false,
                                            &facts));
      continue;
    }
    if (event.kind == DPMEvent::Kind::kTool) {
      ABSL_ASSIGN_OR_RETURN(
          auto typed,
          ParseTypedToolDelta(event.payload, event.index, range_end, limits));
      if (typed.has_value()) {
        ABSL_RETURN_IF_ERROR(MergeFacts(typed->first, &facts));
        for (const std::string& id : typed->second) {
          result.retractions.push_back(
              DPMFactRetraction{.id = id, .source_event = event.index});
        }
        ABSL_RETURN_IF_ERROR(ApplyRetractions(
            typed->second, event.index, /*require_existing=*/false, &facts));
        continue;
      }
    }
    // A model turn is already represented by the prior input's projection
    // receipt and exact decision-token transcript. Re-projecting its visible
    // response would both double-count that turn and make a typed-tool-only
    // delta spuriously invoke the LLM. Only source/user, internal, and
    // untyped-tool events are model-visible projection inputs.
    if (event.kind == DPMEvent::Kind::kModelTurn) continue;
    result.llm_event_indices.push_back(event.index);
  }
  ABSL_ASSIGN_OR_RETURN(result.canonical_memory,
                        EncodeFactMap(MapValues(facts), limits));
  return result;
}

absl::StatusOr<std::string> FinalizeDPMFactMapProjection(
    const DPMFactMapProjectionFold& fold,
    std::optional<absl::string_view> raw_model_delta,
    uint64_t source_event_count, const DPMFactMapLimits& limits) {
  const bool requires_model_delta = !fold.llm_event_indices.empty();
  if (fold.source_event_count != source_event_count ||
      fold.event_range_start > source_event_count) {
    return absl::InvalidArgumentError(
        "DPM fact-map fold does not name this source prefix.");
  }
  if (requires_model_delta != raw_model_delta.has_value()) {
    return absl::InvalidArgumentError(
        "DPM fact-map finalization requires a model delta exactly when the "
        "deterministic fold retained model-visible events.");
  }
  if (!std::is_sorted(fold.llm_event_indices.begin(),
                      fold.llm_event_indices.end()) ||
      std::adjacent_find(fold.llm_event_indices.begin(),
                         fold.llm_event_indices.end()) !=
          fold.llm_event_indices.end()) {
    return absl::InvalidArgumentError(
        "DPM fact-map model-visible event indices must be unique and sorted.");
  }
  for (uint64_t event_index : fold.llm_event_indices) {
    if (event_index < fold.event_range_start ||
        event_index >= source_event_count) {
      return absl::InvalidArgumentError(
          "DPM fact-map model-visible event falls outside the source prefix.");
    }
  }

  ABSL_ASSIGN_OR_RETURN(DPMFactMap base,
                        ParseFactMap(fold.canonical_memory, source_event_count,
                                     limits, std::nullopt));
  std::map<std::string, DPMFact> facts;
  for (DPMFact& fact : base.facts) facts.emplace(fact.id, std::move(fact));

  if (raw_model_delta.has_value()) {
    const std::set<uint64_t> allowed_sources(fold.llm_event_indices.begin(),
                                             fold.llm_event_indices.end());
    ABSL_ASSIGN_OR_RETURN(DPMFactMap delta,
                          ParseFactMap(*raw_model_delta, source_event_count,
                                       limits, allowed_sources));
    ABSL_RETURN_IF_ERROR(MergeFacts(delta, &facts));
  }

  uint64_t previous_retraction_event = 0;
  bool have_previous_retraction = false;
  for (const DPMFactRetraction& retraction : fold.retractions) {
    if (!IsValidFactId(retraction.id) ||
        retraction.source_event < fold.event_range_start ||
        retraction.source_event >= source_event_count ||
        (have_previous_retraction &&
         retraction.source_event < previous_retraction_event)) {
      return absl::InvalidArgumentError(
          "DPM fact-map retractions must be canonical and event-ordered.");
    }
    const std::array<std::string, 1> id = {retraction.id};
    ABSL_RETURN_IF_ERROR(ApplyRetractions(id, retraction.source_event,
                                          /*require_existing=*/true, &facts));
    previous_retraction_event = retraction.source_event;
    have_previous_retraction = true;
  }
  return EncodeFactMap(MapValues(facts), limits);
}

absl::StatusOr<std::string> CanonicalizeDPMRetractionPayload(
    absl::string_view raw_payload) {
  ABSL_ASSIGN_OR_RETURN(std::vector<std::string> ids,
                        ParseCorrectionRetractions(raw_payload));
  nlohmann::ordered_json canonical = nlohmann::ordered_json::object();
  canonical["retract"] = ids;
  return canonical.dump();
}

absl::StatusOr<std::string> RenderDPMFactValues(
    absl::string_view canonical_map, uint64_t source_event_count,
    const DPMFactMapLimits& limits) {
  ABSL_ASSIGN_OR_RETURN(
      DPMFactMap map,
      ParseFactMap(canonical_map, source_event_count, limits, std::nullopt));
  std::sort(map.facts.begin(), map.facts.end(),
            [](const DPMFact& left, const DPMFact& right) {
              return left.id < right.id;
            });
  nlohmann::ordered_json values = nlohmann::ordered_json::object();
  for (const DPMFact& fact : map.facts) {
    if (!fact.retracted_at.has_value()) values[fact.id] = fact.value;
  }
  std::string rendered = values.dump();
  if (rendered.size() > limits.memory_budget_bytes) {
    return absl::ResourceExhaustedError(
        "DPM fact values exceed their model-visible memory budget.");
  }
  return rendered;
}

absl::StatusOr<std::string> RenderDPMFactHeads(absl::string_view canonical_map,
                                               uint64_t source_event_count,
                                               const DPMFactMapLimits& limits,
                                               size_t maximum_head_bytes,
                                               size_t maximum_total_bytes) {
  if (maximum_head_bytes == 0 || maximum_total_bytes == 0) {
    return absl::InvalidArgumentError(
        "DPM fact heads require positive byte limits.");
  }
  ABSL_ASSIGN_OR_RETURN(
      DPMFactMap map,
      ParseFactMap(canonical_map, source_event_count, limits, std::nullopt));
  std::sort(map.facts.begin(), map.facts.end(),
            [](const DPMFact& left, const DPMFact& right) {
              return left.id < right.id;
            });
  std::string rendered;
  for (const DPMFact& fact : map.facts) {
    if (fact.retracted_at.has_value()) continue;
    nlohmann::ordered_json head = nlohmann::ordered_json::object();
    head["id"] = fact.id;
    head["head"] = Utf8Head(fact.value, maximum_head_bytes);
    head["source_event"] = fact.source_event;
    head["retracted_at"] = fact.retracted_at.has_value()
                               ? nlohmann::ordered_json(*fact.retracted_at)
                               : nlohmann::ordered_json(nullptr);
    head["prev"] = fact.prev.has_value() ? nlohmann::ordered_json(*fact.prev)
                                         : nlohmann::ordered_json(nullptr);
    std::string line = head.dump();
    if (rendered.size() > maximum_total_bytes ||
        line.size() + 1 > maximum_total_bytes - rendered.size()) {
      return absl::ResourceExhaustedError(
          "DPM fact heads exceed their prompt byte budget.");
    }
    rendered.append(line);
    rendered.push_back('\n');
  }
  return rendered;
}

}  // namespace litert::lm
