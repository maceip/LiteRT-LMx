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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_MANIFEST_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_MANIFEST_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Log and case identities are embedded in manifests and projection requests.
// Keeping one product bound here prevents the two admission paths from
// accepting different identity domains.
inline constexpr std::size_t kMaximumDPMProjectionIdentityBytes = 16 * 1024;

// Content-addressed description of one independently executed DPM projection.
// The raw event log remains authoritative. This object names a disposable
// projection derived from an exact log prefix and an Engine-owned runtime
// identity; it is never a replacement source of memory truth.
struct DPMProjectionManifest {
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;
  std::string log_id;
  std::string case_id;

  // The source is events [0, source_event_count). input_event_index is the
  // pending decision input and must be the final event in that prefix.
  uint64_t source_event_count = 0;
  Hash256 source_prefix_hash;
  uint64_t input_event_index = 0;

  // The prompt renders events [event_range_start, source_event_count) using
  // their absolute durable indices. Zero means a full-log projection. A
  // nonzero start requires both baseline hashes below and means the prompt
  // asked the model to update that canonical baseline with this exact delta.
  uint64_t event_range_start = 0;
  std::optional<Hash256> baseline_manifest_hash;
  std::optional<Hash256> baseline_output_hash;

  Hash256 correction_digest;
  Hash256 config_hash;
  SessionHandoffIdentity runtime_identity;
  Hash256 request_hash;
  Hash256 output_hash;

  // SHA-256 over every preceding field using the versioned canonical binary
  // encoding implemented by ComputeDPMProjectionManifestHash.
  Hash256 manifest_hash;
};

// Computes the manifest content address. `manifest_hash` is deliberately
// excluded from its own digest.
absl::StatusOr<Hash256> ComputeDPMProjectionManifestHash(
    const DPMProjectionManifest& manifest);

// Validates structural invariants and the embedded manifest_hash.
absl::Status ValidateDPMProjectionManifest(
    const DPMProjectionManifest& manifest);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PROJECTION_MANIFEST_H_
