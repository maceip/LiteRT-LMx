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

#include "runtime/dpm/correction_digest.h"

#include <array>
#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kCorrectionRootDomain =
    "DPM_CORRECTION_LINEAGE_ROOT_SHA256_V1";
constexpr absl::string_view kCorrectionStepDomain =
    "DPM_CORRECTION_LINEAGE_STEP_SHA256_V1";

void UpdateU64(uint64_t value, Sha256Hasher* hasher) {
  std::array<char, 8> bytes{};
  for (int shift = 56, i = 0; shift >= 0; shift -= 8, ++i) {
    bytes[i] = static_cast<char>((value >> shift) & 0xff);
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void UpdateI64(int64_t value, Sha256Hasher* hasher) {
  UpdateU64(static_cast<uint64_t>(value), hasher);
}

void UpdateFrame(char tag, absl::string_view bytes, Sha256Hasher* hasher) {
  hasher->Update(absl::string_view(&tag, 1));
  UpdateU64(bytes.size(), hasher);
  hasher->Update(bytes);
}

}  // namespace

absl::StatusOr<Hash256> InitialDPMCorrectionDigest(
    absl::string_view log_id, absl::string_view case_id) {
  if (log_id.empty() || case_id.empty()) {
    return absl::InvalidArgumentError(
        "A DPM correction digest requires immutable log and case ids.");
  }
  Sha256Hasher hasher;
  hasher.Update(kCorrectionRootDomain);
  UpdateFrame('L', log_id, &hasher);
  UpdateFrame('C', case_id, &hasher);
  return hasher.Finalize();
}

absl::StatusOr<Hash256> AdvanceDPMCorrectionDigest(
    const Hash256& previous, const DPMEvent& correction) {
  if (previous == Hash256{}) {
    return absl::InvalidArgumentError(
        "DPM correction lineage cannot advance from an empty digest.");
  }
  if (correction.kind != DPMEvent::Kind::kCorrection ||
      correction.operation_id.empty() || correction.case_id.empty() ||
      correction.payload.empty() || correction.turn_receipt.has_value()) {
    return absl::InvalidArgumentError(
        "Only a complete authoritative correction event can advance the "
        "correction lineage.");
  }
  Sha256Hasher hasher;
  hasher.Update(kCorrectionStepDomain);
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(previous.bytes.data()),
      previous.bytes.size()));
  UpdateU64(correction.index, &hasher);
  UpdateI64(correction.timestamp_us, &hasher);
  UpdateFrame('O', correction.operation_id, &hasher);
  UpdateFrame('C', correction.case_id, &hasher);
  UpdateFrame('P', correction.payload, &hasher);
  return hasher.Finalize();
}

absl::StatusOr<Hash256> ComputeDPMCorrectionDigest(
    const DPMLogSnapshot& snapshot) {
  if (snapshot.log_id.empty() || snapshot.case_id.empty()) {
    return absl::InvalidArgumentError(
        "A DPM correction digest requires immutable log and case ids.");
  }
  if (snapshot.generation != snapshot.events.size()) {
    return absl::DataLossError(
        "DPM correction digest received an inconsistent log generation.");
  }

  for (uint64_t i = 0; i < snapshot.events.size(); ++i) {
    const DPMEvent& event = snapshot.events[i];
    if (event.index != i || event.case_id != snapshot.case_id) {
      return absl::DataLossError(
          "DPM correction digest received a contaminated log snapshot.");
    }
    if (event.kind != DPMEvent::Kind::kCorrection) continue;
    if (event.operation_id.empty() || event.payload.empty() ||
        event.turn_receipt.has_value()) {
      return absl::DataLossError(
          "Authoritative DPM correction event is malformed.");
    }
  }

  ABSL_ASSIGN_OR_RETURN(
      Hash256 digest,
      InitialDPMCorrectionDigest(snapshot.log_id, snapshot.case_id));
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kCorrection) continue;
    ABSL_ASSIGN_OR_RETURN(digest,
                          AdvanceDPMCorrectionDigest(digest, event));
  }
  return digest;
}

}  // namespace litert::lm
