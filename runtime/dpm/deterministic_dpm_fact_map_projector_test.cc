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

#include "runtime/dpm/deterministic_dpm_fact_map_projector.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"        // from @com_google_absl
#include "absl/status/statusor.h"      // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "gtest/gtest.h"               // from @com_google_googletest
#include "runtime/dpm/canonical_replay_catalog.h"
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

Hash256 Digest(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

class NoopLease final : public DPMEventLogOperationLease {};

class FixedEventLog final : public DPMEventLog {
 public:
  explicit FixedEventLog(DPMLogSnapshot snapshot)
      : snapshot_(std::move(snapshot)) {}

  absl::StatusOr<DPMLogSnapshot> Snapshot() const override { return snapshot_; }

  absl::StatusOr<std::unique_ptr<DPMEventLogOperationLease>>
  AcquireOperationLease() override {
    return std::make_unique<NoopLease>();
  }

  absl::StatusOr<Hash256> PrefixHash(uint64_t event_count) const override {
    if (event_count >= snapshot_.prefix_hashes.size()) {
      return absl::OutOfRangeError("test prefix is outside the snapshot");
    }
    return snapshot_.prefix_hashes[static_cast<size_t>(event_count)];
  }

  absl::StatusOr<DPMAppendResult> AppendIfGeneration(DPMEvent,
                                                     uint64_t) override {
    return absl::UnimplementedError("fixed test log is read only");
  }

 private:
  DPMLogSnapshot snapshot_;
};

class MemoryWinnerCatalog final : public CanonicalWinnerReplayCatalog {
 public:
  absl::StatusOr<CanonicalReplayWinner> Get(
      const CanonicalReplayKey& key) const override {
    if (!winner_.has_value() || !(winner_->key == key)) {
      return absl::NotFoundError("test winner is absent");
    }
    ++get_hit_count_;
    return *winner_;
  }

  absl::StatusOr<PublishCanonicalReplayResult> Publish(
      const CanonicalReplayCandidate& candidate) override {
    if (winner_.has_value()) {
      if (!(winner_->key == candidate.key)) {
        return absl::AlreadyExistsError(
            "test catalog already contains another request");
      }
      return PublishCanonicalReplayResult{
          .published = false,
          .winner = *winner_,
      };
    }
    CanonicalReplayWinner winner{
        .key = candidate.key,
        .canonical_output = candidate.canonical_output,
        .canonical_output_hash = Digest(candidate.canonical_output),
        .execution_evidence_hash = candidate.execution_evidence_hash,
        .authentication_key_id = "test-key",
        .authentication_tag = Digest("test-authentication-tag"),
    };
    winner_ = winner;
    ++publish_count_;
    return PublishCanonicalReplayResult{
        .published = true,
        .winner = std::move(winner),
    };
  }

  int publish_count() const { return publish_count_; }
  int get_hit_count() const { return get_hit_count_; }

 private:
  std::optional<CanonicalReplayWinner> winner_;
  int publish_count_ = 0;
  mutable int get_hit_count_ = 0;
};

SessionHandoffIdentity TestRuntimeIdentity() {
  return SessionHandoffIdentity{
      .model_artifact_hash = Digest("model"),
      .runtime_artifact_hash = Digest("runtime"),
      .inference_profile_hash = Digest("profile"),
  };
}

DPMLogSnapshot TestSnapshot() {
  DPMLogSnapshot snapshot{
      .log_id = "deterministic-memory-log",
      .case_id = "case-1",
      .generation = 2,
      .prefix_hash = Digest("prefix-2"),
      .prefix_hashes = {Digest("genesis"), Digest("prefix-1"),
                        Digest("prefix-2")},
  };
  snapshot.events.push_back(DPMEvent{
      .index = 0,
      .kind = DPMEvent::Kind::kTool,
      .timestamp_us = 1,
      .operation_id = "typed-fact",
      .case_id = snapshot.case_id,
      .payload =
          R"({"schema_id":"dpm.factmap.delta","facts":[{"id":"customer.name","value":"Zoe","source_event":0,"valid_from":0,"retracted_at":null,"prev":null}],"retract":[]})",
  });
  snapshot.events.push_back(DPMEvent{
      .index = 1,
      .kind = DPMEvent::Kind::kUser,
      .timestamp_us = 2,
      .operation_id = "ask-agent",
      .case_id = snapshot.case_id,
      .payload = "What is the customer's name?",
  });
  return snapshot;
}

DPMLogSnapshot TypedFactSnapshot() {
  DPMLogSnapshot snapshot{
      .log_id = "deterministic-memory-log",
      .case_id = "case-1",
      .generation = 1,
      .prefix_hash = Digest("typed-prefix-1"),
      .prefix_hashes = {Digest("typed-genesis"), Digest("typed-prefix-1")},
  };
  snapshot.events.push_back(DPMEvent{
      .index = 0,
      .kind = DPMEvent::Kind::kTool,
      .timestamp_us = 1,
      .operation_id = "typed-fact",
      .case_id = snapshot.case_id,
      .payload =
          R"({"schema_id":"dpm.factmap.delta","facts":[{"id":"customer.name","value":"Zoe","source_event":0,"valid_from":0,"retracted_at":null,"prev":null}],"retract":[]})",
  });
  return snapshot;
}

TEST(DeterministicDPMFactMapProjectorTest,
     ColdCatalogsAndWarmReplayHaveOneManifestIdentity) {
  FixedEventLog log(TestSnapshot());
  MemoryWinnerCatalog first_catalog;
  MemoryWinnerCatalog second_catalog;
  DeterministicDPMFactMapProjectorConfig config{
      .implementation_artifact_hash = Digest("same-dpm-binary"),
      .replay_token_bound = 32,
  };

  auto first_or = DeterministicDPMFactMapProjector::Create(
      &log, &first_catalog, TestRuntimeIdentity(), config);
  ASSERT_TRUE(first_or.ok()) << first_or.status();
  auto second_or = DeterministicDPMFactMapProjector::Create(
      &log, &second_catalog, TestRuntimeIdentity(), config);
  ASSERT_TRUE(second_or.ok()) << second_or.status();

  DPMProjectionRequest request{
      .log = TestSnapshot(),
      .input_event_index = 1,
      .case_id = "case-1",
  };
  auto first_cold_or = (*first_or)->Project(request);
  ASSERT_TRUE(first_cold_or.ok()) << first_cold_or.status();
  auto second_cold_or = (*second_or)->Project(request);
  ASSERT_TRUE(second_cold_or.ok()) << second_cold_or.status();
  auto first_warm_or = (*first_or)->Project(request);
  ASSERT_TRUE(first_warm_or.ok()) << first_warm_or.status();

  const std::string expected =
      R"({"schema_id":"dpm.factmap","facts":[{"id":"customer.name","value":"Zoe","source_event":0,"valid_from":0,"retracted_at":null,"prev":null}]})";
  EXPECT_EQ(first_cold_or->projected_memory, expected);
  EXPECT_EQ(second_cold_or->projected_memory, expected);
  EXPECT_EQ(first_warm_or->projected_memory, expected);
  EXPECT_EQ(first_cold_or->manifest.manifest_hash,
            second_cold_or->manifest.manifest_hash);
  EXPECT_EQ(first_cold_or->manifest.manifest_hash,
            first_warm_or->manifest.manifest_hash);
  EXPECT_EQ(first_cold_or->manifest.execution_evidence_hash,
            first_warm_or->manifest.execution_evidence_hash);
  EXPECT_EQ(first_cold_or->manifest.replay_mode,
            DPMReplayMode::kCanonicalWinnerReplay);
  EXPECT_EQ(first_catalog.publish_count(), 1);
  EXPECT_EQ(first_catalog.get_hit_count(), 1);
  EXPECT_EQ(second_catalog.publish_count(), 1);
  ASSERT_TRUE(first_cold_or->model_visible_memory.has_value());
  EXPECT_EQ(*first_cold_or->model_visible_memory, R"({"customer.name":"Zoe"})");
  EXPECT_FALSE(first_cold_or->model_visible_input.has_value());
}

TEST(DeterministicDPMFactMapProjectorTest,
     TypedPendingInputUsesCompactAcknowledgment) {
  FixedEventLog log(TypedFactSnapshot());
  MemoryWinnerCatalog catalog;
  DeterministicDPMFactMapProjectorConfig config{
      .implementation_artifact_hash = Digest("same-dpm-binary"),
      .replay_token_bound = 32,
  };
  auto projector_or = DeterministicDPMFactMapProjector::Create(
      &log, &catalog, TestRuntimeIdentity(), config);
  ASSERT_TRUE(projector_or.ok()) << projector_or.status();
  DPMProjectionRequest request{
      .log = TypedFactSnapshot(),
      .input_event_index = 0,
      .case_id = "case-1",
  };
  auto outcome_or = (*projector_or)->Project(request);
  ASSERT_TRUE(outcome_or.ok()) << outcome_or.status();
  ASSERT_TRUE(outcome_or->model_visible_memory.has_value());
  EXPECT_EQ(*outcome_or->model_visible_memory, R"({"customer.name":"Zoe"})");
  ASSERT_TRUE(outcome_or->model_visible_input.has_value());
  EXPECT_EQ(*outcome_or->model_visible_input, "Memory updated.");
}

TEST(DeterministicDPMFactMapProjectorTest,
     ImplementationArtifactParticipatesInReducerConfigurationIdentity) {
  DeterministicDPMFactMapProjectorConfig first{
      .implementation_artifact_hash = Digest("binary-a"),
  };
  DeterministicDPMFactMapProjectorConfig second = first;
  second.implementation_artifact_hash = Digest("binary-b");
  auto first_hash = ComputeDeterministicDPMFactMapProjectorConfigHash(first);
  auto second_hash = ComputeDeterministicDPMFactMapProjectorConfigHash(second);
  ASSERT_TRUE(first_hash.ok()) << first_hash.status();
  ASSERT_TRUE(second_hash.ok()) << second_hash.status();
  EXPECT_NE(*first_hash, *second_hash);
}

}  // namespace
}  // namespace litert::lm
