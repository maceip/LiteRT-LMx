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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_PROMPT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_PROMPT_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Every field influences canonical request bytes or strict output admission.
// There is deliberately no temperature, seed, or repair count: projection is
// exactly one fresh GREEDY CPU inference and malformed output fails closed.
struct DPMProjectionConfig {
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;
  std::string schema_id;
  // A compact JSON object in canonical nlohmann ordered-json form. It gives
  // task-specific context but cannot change the fixed top-level output shape.
  std::string schema_json;
  uint32_t max_output_tokens = 4096;
  size_t memory_budget_bytes = 4096;
  size_t max_event_range_bytes = 16 * 1024 * 1024;
  uint32_t max_items_per_section = 4096;
  size_t max_item_bytes = 4096;
};

// A previously produced projection may accelerate a later one only when the
// caller supplies its complete manifest and exact canonical output. The
// projector revalidates both and compares the baseline's prefix hash with the
// authoritative DPMEventLog before using it.
struct DPMProjectionBaselineArtifact {
  DPMProjectionManifest manifest;
  std::string projected_memory;
};

// Immutable request bytes passed to one runtime invocation. The request hash
// binds the prompt bytes plus every non-prompt identity that affects their
// interpretation.
struct CanonicalDPMProjectionRequest {
  std::string prompt_bytes;
  std::string log_id;
  std::string case_id;
  uint64_t source_event_count = 0;
  Hash256 source_prefix_hash;
  uint64_t input_event_index = 0;
  uint64_t event_range_start = 0;
  std::optional<Hash256> baseline_manifest_hash;
  std::optional<Hash256> baseline_output_hash;
  Hash256 correction_digest;
  Hash256 config_hash;
  SessionHandoffIdentity runtime_identity;
  Hash256 request_hash;
};

absl::Status ValidateDPMProjectionConfig(const DPMProjectionConfig& config);

// SHA-256 of the versioned canonical configuration encoding.
absl::StatusOr<Hash256> ComputeDPMProjectionConfigHash(
    const DPMProjectionConfig& config);

// Renders exactly events [event_range_start, event_range_end), retaining each
// event's absolute durable zero-based index. Prompt citations are one-based,
// so event index N is always cited as [N+1], even in a delta range.
absl::StatusOr<std::string> RenderCanonicalDPMEventRange(
    const DPMLogSnapshot& snapshot, uint64_t event_range_start,
    uint64_t event_range_end, size_t maximum_bytes);

// Builds deterministic full-log or baseline-plus-delta prompt bytes. Baseline
// compatibility with an authoritative earlier prefix is enforced by the
// projector, which has access to DPMEventLog::PrefixHash.
absl::StatusOr<CanonicalDPMProjectionRequest>
BuildCanonicalDPMProjectionRequest(
    const DPMLogSnapshot& snapshot, uint64_t input_event_index,
    const Hash256& correction_digest, const DPMProjectionConfig& config,
    const SessionHandoffIdentity& runtime_identity,
    const std::optional<DPMProjectionBaselineArtifact>& baseline);

// Recomputes the canonical request hash and rejects structurally incomplete
// requests. This helper hashes exact prompt bytes; it never rebuilds a prompt.
absl::StatusOr<Hash256> ComputeCanonicalDPMProjectionRequestHash(
    const CanonicalDPMProjectionRequest& request);

// Validates exactly {Facts,Reasoning,Compliance}, canonicalizes field order and
// JSON encoding, and rejects missing, malformed, duplicate, zero, or
// out-of-prefix citations. No markdown envelope or repair extraction is
// accepted.
absl::StatusOr<std::string> CanonicalizeDPMProjectionOutput(
    absl::string_view raw_output, uint64_t source_event_count,
    const DPMProjectionConfig& config);

// SHA-256 of the canonical projected-memory bytes.
Hash256 ComputeCanonicalDPMProjectionOutputHash(
    absl::string_view canonical_output);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_PROMPT_H_
