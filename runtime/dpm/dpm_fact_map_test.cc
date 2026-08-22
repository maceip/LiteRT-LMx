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

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "absl/status/status.h"    // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl

namespace litert::lm {
namespace {

constexpr char kFactMapTestdata[] = "litert_lm/runtime/dpm/testdata/factmap/";

std::string ReadTestdata(const std::string& name) {
  const std::filesystem::path path =
      std::filesystem::path(::testing::SrcDir()) / kFactMapTestdata / name;
  std::ifstream stream(path, std::ios::binary);
  EXPECT_TRUE(stream.is_open()) << path;
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

std::string WithoutTrailingNewline(std::string value) {
  if (!value.empty() && value.back() == '\n') value.pop_back();
  return value;
}

DPMEvent Event(uint64_t index, DPMEvent::Kind kind, std::string payload) {
  return DPMEvent{
      .index = index,
      .kind = kind,
      .timestamp_us = static_cast<int64_t>(index + 1),
      .operation_id = "op-" + std::to_string(index),
      .case_id = "case",
      .payload = std::move(payload),
  };
}

TEST(DPMFactMapTest, CanonicalizesFrozenGolden) {
  DPMFactMapLimits limits;
  limits.memory_budget_bytes = 4096;
  auto canonical = CanonicalizeDPMFactMap(ReadTestdata("canonical_input.json"),
                                          /*source_event_count=*/3, limits);
  ASSERT_TRUE(canonical.ok()) << canonical.status();
  EXPECT_EQ(*canonical,
            WithoutTrailingNewline(ReadTestdata("canonical.golden.json")));
}

TEST(DPMFactMapTest, TypedToolAndCorrectionFoldWithoutModel) {
  const std::string base =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"customer.name","value":"Zoë","source_event":0,"valid_from":0,"retracted_at":null,"prev":null}]})";
  DPMLogSnapshot snapshot{
      .log_id = "log",
      .case_id = "case",
      .generation = 3,
      .events =
          {
              Event(0, DPMEvent::Kind::kUser, "Customer is Zoë"),
              Event(
                  1, DPMEvent::Kind::kTool,
                  R"({"schema_id":"dpm.factmap.delta","facts":[{"id":"risk.level","value":"high","source_event":1,"valid_from":1,"retracted_at":null,"prev":null}],"retract":[]})"),
              Event(2, DPMEvent::Kind::kCorrection,
                    R"({"retract":["customer.name"]})"),
          },
  };
  auto fold = FoldDeterministicDPMEvents(base, snapshot, /*range_start=*/1,
                                         /*range_end=*/3, DPMFactMapLimits());
  ASSERT_TRUE(fold.ok()) << fold.status();
  EXPECT_TRUE(fold->llm_event_indices.empty());
  ASSERT_EQ(fold->retractions.size(), 1);
  EXPECT_EQ(fold->retractions[0],
            (DPMFactRetraction{.id = "customer.name", .source_event = 2}));
  auto finalized = FinalizeDPMFactMapProjection(
      *fold, std::nullopt, /*source_event_count=*/3, DPMFactMapLimits());
  ASSERT_TRUE(finalized.ok()) << finalized.status();
  EXPECT_EQ(*finalized,
            WithoutTrailingNewline(ReadTestdata("fold.golden.json")));
}

TEST(DPMFactMapTest, UntypedEventsAreTheOnlyModelVisibleEvents) {
  DPMLogSnapshot snapshot{
      .log_id = "log",
      .case_id = "case",
      .generation = 2,
      .events =
          {
              Event(0, DPMEvent::Kind::kUser, "Customer is Zoë"),
              Event(
                  1, DPMEvent::Kind::kTool,
                  R"({"schema_id":"dpm.factmap.delta","facts":[{"id":"risk.level","value":"high","source_event":1,"valid_from":1,"retracted_at":null,"prev":null}],"retract":[]})"),
          },
  };
  auto fold =
      FoldDeterministicDPMEvents(EmptyDPMFactMap(), snapshot, /*range_start=*/0,
                                 /*range_end=*/2, DPMFactMapLimits());
  ASSERT_TRUE(fold.ok()) << fold.status();
  EXPECT_EQ(fold->llm_event_indices, std::vector<uint64_t>({0}));
  EXPECT_NE(fold->canonical_memory.find("risk.level"), std::string::npos);

  const std::string model_delta =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"customer.name","value":"Zoë","source_event":0,"valid_from":0,"retracted_at":null,"prev":null}]})";
  auto finalized = FinalizeDPMFactMapProjection(
      *fold, absl::string_view(model_delta), /*source_event_count=*/2,
      DPMFactMapLimits());
  ASSERT_TRUE(finalized.ok()) << finalized.status();
  EXPECT_NE(finalized->find("customer.name"), std::string::npos);
  EXPECT_NE(finalized->find("risk.level"), std::string::npos);
}

TEST(DPMFactMapTest, FullRebuildAppliesCorrectionAfterModelDelta) {
  DPMLogSnapshot snapshot{
      .log_id = "log",
      .case_id = "case",
      .generation = 2,
      .events =
          {
              Event(0, DPMEvent::Kind::kUser, "Customer is Zoë"),
              Event(1, DPMEvent::Kind::kCorrection,
                    R"({"retract":["customer.name"]})"),
          },
  };
  auto fold = FoldDeterministicDPMEvents(EmptyDPMFactMap(), snapshot,
                                         /*range_start=*/0, /*range_end=*/2,
                                         DPMFactMapLimits());
  ASSERT_TRUE(fold.ok()) << fold.status();
  EXPECT_EQ(fold->llm_event_indices, std::vector<uint64_t>({0}));
  ASSERT_EQ(fold->retractions.size(), 1);

  const std::string model_delta =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"customer.name","value":"Zoë","source_event":0,"valid_from":0,"retracted_at":null,"prev":null}]})";
  auto finalized = FinalizeDPMFactMapProjection(
      *fold, absl::string_view(model_delta), /*source_event_count=*/2,
      DPMFactMapLimits());
  ASSERT_TRUE(finalized.ok()) << finalized.status();
  EXPECT_NE(finalized->find(R"("source_event":1)"), std::string::npos);
  EXPECT_NE(finalized->find(R"("retracted_at":1)"), std::string::npos);
}

TEST(DPMFactMapTest, NewerFactUpdateWinsOverEarlierRetraction) {
  DPMLogSnapshot snapshot{
      .log_id = "log",
      .case_id = "case",
      .generation = 3,
      .events =
          {
              Event(0, DPMEvent::Kind::kUser, "Initial risk"),
              Event(1, DPMEvent::Kind::kCorrection,
                    R"({"retract":["risk.level"]})"),
              Event(
                  2, DPMEvent::Kind::kTool,
                  R"({"schema_id":"dpm.factmap.delta","facts":[{"id":"risk.level","value":"high","source_event":2,"valid_from":2,"retracted_at":null,"prev":null}],"retract":[]})"),
          },
  };
  auto fold = FoldDeterministicDPMEvents(EmptyDPMFactMap(), snapshot,
                                         /*range_start=*/0, /*range_end=*/3,
                                         DPMFactMapLimits());
  ASSERT_TRUE(fold.ok()) << fold.status();
  const std::string model_delta =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"risk.level","value":"low","source_event":0,"valid_from":0,"retracted_at":null,"prev":null}]})";
  auto finalized = FinalizeDPMFactMapProjection(
      *fold, absl::string_view(model_delta), /*source_event_count=*/3,
      DPMFactMapLimits());
  ASSERT_TRUE(finalized.ok()) << finalized.status();
  EXPECT_NE(finalized->find(R"("value":"high")"), std::string::npos);
  EXPECT_NE(finalized->find(R"("source_event":2)"), std::string::npos);
  EXPECT_NE(finalized->find(R"("retracted_at":null)"), std::string::npos);
}

TEST(DPMFactMapTest, UnknownRetractionFailsDuringFinalization) {
  DPMLogSnapshot snapshot{
      .log_id = "log",
      .case_id = "case",
      .generation = 1,
      .events =
          {
              Event(0, DPMEvent::Kind::kCorrection,
                    R"({"retract":["missing"]})"),
          },
  };
  auto fold = FoldDeterministicDPMEvents(EmptyDPMFactMap(), snapshot,
                                         /*range_start=*/0, /*range_end=*/1,
                                         DPMFactMapLimits());
  ASSERT_TRUE(fold.ok()) << fold.status();
  auto finalized = FinalizeDPMFactMapProjection(
      *fold, std::nullopt, /*source_event_count=*/1, DPMFactMapLimits());
  EXPECT_EQ(finalized.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(DPMFactMapTest, FinalizationRejectsAnotherSourcePrefix) {
  DPMLogSnapshot snapshot{
      .log_id = "log",
      .case_id = "case",
      .generation = 1,
      .events = {Event(0, DPMEvent::Kind::kUser, "Customer is Zoë")},
  };
  auto fold = FoldDeterministicDPMEvents(EmptyDPMFactMap(), snapshot,
                                         /*range_start=*/0, /*range_end=*/1,
                                         DPMFactMapLimits());
  ASSERT_TRUE(fold.ok()) << fold.status();
  const std::string model_delta =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"customer.name","value":"Zoë","source_event":0,"valid_from":0,"retracted_at":null,"prev":null}]})";
  auto finalized = FinalizeDPMFactMapProjection(
      *fold, absl::string_view(model_delta), /*source_event_count=*/2,
      DPMFactMapLimits());
  EXPECT_EQ(finalized.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(DPMFactMapTest, FoldRejectsBaselineThatCitesItsOwnDelta) {
  const std::string contaminated_base =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"future","value":"invalid","source_event":1,"valid_from":1,"retracted_at":null,"prev":null}]})";
  DPMLogSnapshot snapshot{
      .log_id = "log",
      .case_id = "case",
      .generation = 2,
      .events =
          {
              Event(0, DPMEvent::Kind::kUser, "first"),
              Event(1, DPMEvent::Kind::kUser, "second"),
          },
  };
  auto fold =
      FoldDeterministicDPMEvents(contaminated_base, snapshot, /*range_start=*/1,
                                 /*range_end=*/2, DPMFactMapLimits());
  EXPECT_EQ(fold.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(DPMFactMapTest, ModelReceiptEventsAreNotProjectedAgain) {
  DPMLogSnapshot snapshot{
      .log_id = "log",
      .case_id = "case",
      .generation = 3,
      .events =
          {
              Event(0, DPMEvent::Kind::kUser, "Customer is Zoë"),
              Event(1, DPMEvent::Kind::kModelTurn, "prior decision"),
              Event(
                  2, DPMEvent::Kind::kTool,
                  R"({"schema_id":"dpm.factmap.delta","facts":[{"id":"risk.level","value":"high","source_event":2,"valid_from":2,"retracted_at":null,"prev":null}],"retract":[]})"),
          },
  };
  auto fold =
      FoldDeterministicDPMEvents(EmptyDPMFactMap(), snapshot, /*range_start=*/1,
                                 /*range_end=*/3, DPMFactMapLimits());
  ASSERT_TRUE(fold.ok()) << fold.status();
  EXPECT_TRUE(fold->llm_event_indices.empty());
  EXPECT_NE(fold->canonical_memory.find("risk.level"), std::string::npos);
}

TEST(DPMFactMapTest, DeltaProvenanceMustNameAnExposedRawEvent) {
  const std::string delta =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"customer.name","value":"Zoë","source_event":1,"valid_from":1,"retracted_at":null,"prev":null}]})";
  const std::vector<uint64_t> exposed = {0};
  auto canonical = CanonicalizeDPMFactDelta(
      delta, exposed, /*source_event_count=*/2, DPMFactMapLimits());
  EXPECT_EQ(canonical.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(DPMFactMapTest, MergeRejectsEqualEventConflict) {
  const std::string base =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"risk.level","value":"low","source_event":1,"valid_from":1,"retracted_at":null,"prev":null}]})";
  const std::string delta =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"risk.level","value":"high","source_event":1,"valid_from":1,"retracted_at":null,"prev":null}]})";
  auto merged = MergeDPMFactMap(base, delta, /*source_event_count=*/2,
                                DPMFactMapLimits());
  EXPECT_EQ(merged.status().code(), absl::StatusCode::kDataLoss);
}

TEST(DPMFactMapTest, HeadsHideFullAndRetractedValues) {
  const std::string map =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"active","value":"abcdefghij","source_event":0,"valid_from":0,"retracted_at":null,"prev":null},{"id":"gone","value":"secret-value","source_event":1,"valid_from":0,"retracted_at":1,"prev":null}]})";
  auto heads = RenderDPMFactHeads(map, /*source_event_count=*/2,
                                  DPMFactMapLimits(), /*maximum_head_bytes=*/4,
                                  /*maximum_total_bytes=*/1024);
  ASSERT_TRUE(heads.ok()) << heads.status();
  EXPECT_NE(heads->find(R"("head":"abcd")"), std::string::npos);
  EXPECT_EQ(heads->find("efghij"), std::string::npos);
  EXPECT_EQ(heads->find("secret-value"), std::string::npos);
  EXPECT_EQ(heads->find(R"("id":"gone")"), std::string::npos);
}

TEST(DPMFactMapTest, ValuesRenderSortedLiveMemoryWithoutProvenance) {
  const std::string map =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"zeta","value":"last","source_event":0,"valid_from":0,"retracted_at":null,"prev":null},{"id":"alpha","value":"first","source_event":1,"valid_from":1,"retracted_at":null,"prev":null},{"id":"gone","value":"secret","source_event":2,"valid_from":0,"retracted_at":2,"prev":null}]})";
  auto values =
      RenderDPMFactValues(map, /*source_event_count=*/3, DPMFactMapLimits());
  ASSERT_TRUE(values.ok()) << values.status();
  EXPECT_EQ(*values, R"({"alpha":"first","zeta":"last"})");
}

TEST(DPMFactMapTest, CorrectionPayloadMustBeSortedAndUnique) {
  auto canonical =
      CanonicalizeDPMRetractionPayload(R"({"retract":["beta","alpha"]})");
  EXPECT_EQ(canonical.status().code(), absl::StatusCode::kInvalidArgument);
  canonical =
      CanonicalizeDPMRetractionPayload(R"({"retract":["alpha","beta"]})");
  ASSERT_TRUE(canonical.ok()) << canonical.status();
  EXPECT_EQ(*canonical, R"({"retract":["alpha","beta"]})");
}

}  // namespace
}  // namespace litert::lm
