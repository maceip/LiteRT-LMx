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

#include "runtime/dpm/dpm_projection_manifest.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kManifestDomain =
    "LITERT_LMX_DPM_PROJECTION_MANIFEST_SHA256";

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

bool ContainsControlByte(absl::string_view value) {
  for (unsigned char byte : value) {
    if (byte < 0x20 || byte == 0x7f) return true;
  }
  return false;
}

bool IsValidUtf8(absl::string_view text) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t lead = bytes[index++];
    if (lead <= 0x7f) continue;

    size_t continuation_count = 0;
    uint8_t minimum_second = 0x80;
    uint8_t maximum_second = 0xbf;
    if (lead >= 0xc2 && lead <= 0xdf) {
      continuation_count = 1;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      continuation_count = 2;
      if (lead == 0xe0) minimum_second = 0xa0;
      if (lead == 0xed) maximum_second = 0x9f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      continuation_count = 3;
      if (lead == 0xf0) minimum_second = 0x90;
      if (lead == 0xf4) maximum_second = 0x8f;
    } else {
      return false;
    }
    if (continuation_count > text.size() - index ||
        bytes[index] < minimum_second || bytes[index] > maximum_second) {
      return false;
    }
    ++index;
    for (size_t offset = 1; offset < continuation_count; ++offset, ++index) {
      if (bytes[index] < 0x80 || bytes[index] > 0xbf) return false;
    }
  }
  return true;
}

void UpdateU8(uint8_t value, Sha256Hasher* hasher) {
  const char byte = static_cast<char>(value);
  hasher->Update(absl::string_view(&byte, 1));
}

void UpdateU32(uint32_t value, Sha256Hasher* hasher) {
  std::array<char, 4> bytes{};
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (24 - i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void UpdateU64(uint64_t value, Sha256Hasher* hasher) {
  std::array<char, 8> bytes{};
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (56 - i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void UpdateFrame(char tag, absl::string_view bytes, Sha256Hasher* hasher) {
  hasher->Update(absl::string_view(&tag, 1));
  UpdateU64(bytes.size(), hasher);
  hasher->Update(bytes);
}

void UpdateHash(char tag, const Hash256& hash, Sha256Hasher* hasher) {
  UpdateFrame(tag,
              absl::string_view(
                  reinterpret_cast<const char*>(hash.bytes.data()),
                  hash.bytes.size()),
              hasher);
}

absl::Status ValidateManifestFields(const DPMProjectionManifest& manifest) {
  if (manifest.format_version != DPMProjectionManifest::kFormatVersion) {
    return absl::InvalidArgumentError(
        "Unsupported DPM projection manifest format version.");
  }
  if (manifest.log_id.empty() || manifest.case_id.empty() ||
      manifest.log_id.size() > kMaximumDPMProjectionIdentityBytes ||
      manifest.case_id.size() > kMaximumDPMProjectionIdentityBytes ||
      !IsValidUtf8(manifest.log_id) || !IsValidUtf8(manifest.case_id) ||
      ContainsControlByte(manifest.log_id) ||
      ContainsControlByte(manifest.case_id)) {
    return absl::InvalidArgumentError(
        "DPM projection manifest requires bounded UTF-8 log and case "
        "identities without control bytes.");
  }
  if (manifest.source_event_count == 0 ||
      manifest.input_event_index >= manifest.source_event_count ||
      manifest.input_event_index != manifest.source_event_count - 1 ||
      manifest.event_range_start > manifest.input_event_index) {
    return absl::InvalidArgumentError(
        "DPM projection manifest has an invalid source event range.");
  }
  const bool has_baseline_manifest =
      manifest.baseline_manifest_hash.has_value();
  const bool has_baseline_output = manifest.baseline_output_hash.has_value();
  if (has_baseline_manifest != has_baseline_output ||
      (manifest.event_range_start == 0 && has_baseline_manifest) ||
      (manifest.event_range_start != 0 && !has_baseline_manifest)) {
    return absl::InvalidArgumentError(
        "DPM projection manifest baseline hashes do not match its event "
        "range mode.");
  }
  if (IsZeroHash(manifest.source_prefix_hash) ||
      IsZeroHash(manifest.correction_digest) ||
      IsZeroHash(manifest.config_hash) ||
      IsZeroHash(manifest.runtime_identity.model_artifact_hash) ||
      IsZeroHash(manifest.runtime_identity.runtime_artifact_hash) ||
      IsZeroHash(manifest.runtime_identity.inference_profile_hash) ||
      IsZeroHash(manifest.request_hash) || IsZeroHash(manifest.output_hash) ||
      IsZeroHash(manifest.replay_request_hash) ||
      IsZeroHash(manifest.execution_evidence_hash) ||
      (manifest.baseline_manifest_hash.has_value() &&
       IsZeroHash(*manifest.baseline_manifest_hash)) ||
      (manifest.baseline_output_hash.has_value() &&
       IsZeroHash(*manifest.baseline_output_hash))) {
    return absl::InvalidArgumentError(
        "DPM projection manifest contains an empty cryptographic identity.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(manifest.replay_mode));
  switch (manifest.replay_mode) {
    case DPMReplayMode::kCanonicalWinnerReplay:
      if (manifest.exact_profile_id.has_value() ||
          manifest.exact_profile_admission_record_id.has_value() ||
          manifest.exact_output_evidence_hash.has_value() ||
          manifest.exact_logit_frame_count != 0) {
        return absl::InvalidArgumentError(
            "WinnerReplay projection manifest must not claim an exact "
            "profile, admission record, or exact execution evidence.");
      }
      break;
    case DPMReplayMode::kExactRegeneration:
      if (!manifest.exact_profile_id.has_value() ||
          IsZeroHash(*manifest.exact_profile_id) ||
          !manifest.exact_profile_admission_record_id.has_value() ||
          IsZeroHash(*manifest.exact_profile_admission_record_id) ||
          !manifest.exact_output_evidence_hash.has_value() ||
          IsZeroHash(*manifest.exact_output_evidence_hash) ||
          manifest.exact_logit_frame_count == 0 ||
          manifest.exact_logit_frame_count >
              kMaximumDPMProjectionExactLogitFrames) {
        return absl::InvalidArgumentError(
            "ExactRegeneration projection manifest requires a non-empty "
            "derived profile, admission record, and ordered execution "
            "evidence.");
      }
      break;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Hash256> ComputeDPMProjectionManifestHash(
    const DPMProjectionManifest& manifest) {
  ABSL_RETURN_IF_ERROR(ValidateManifestFields(manifest));

  Sha256Hasher hasher;
  hasher.Update(kManifestDomain);
  UpdateU32(manifest.format_version, &hasher);
  UpdateFrame('L', manifest.log_id, &hasher);
  UpdateFrame('C', manifest.case_id, &hasher);
  UpdateU64(manifest.source_event_count, &hasher);
  UpdateHash('P', manifest.source_prefix_hash, &hasher);
  UpdateU64(manifest.input_event_index, &hasher);
  UpdateU64(manifest.event_range_start, &hasher);
  UpdateU8(manifest.baseline_manifest_hash.has_value() ? 1 : 0, &hasher);
  if (manifest.baseline_manifest_hash.has_value()) {
    UpdateHash('B', *manifest.baseline_manifest_hash, &hasher);
    UpdateHash('b', *manifest.baseline_output_hash, &hasher);
  }
  UpdateHash('D', manifest.correction_digest, &hasher);
  UpdateHash('G', manifest.config_hash, &hasher);
  UpdateHash('M', manifest.runtime_identity.model_artifact_hash, &hasher);
  UpdateHash('R', manifest.runtime_identity.runtime_artifact_hash, &hasher);
  UpdateHash('I', manifest.runtime_identity.inference_profile_hash, &hasher);
  UpdateHash('Q', manifest.request_hash, &hasher);
  UpdateHash('O', manifest.output_hash, &hasher);
  UpdateU8(static_cast<uint8_t>(manifest.replay_mode), &hasher);
  UpdateHash('Y', manifest.replay_request_hash, &hasher);
  UpdateHash('E', manifest.execution_evidence_hash, &hasher);
  UpdateU8(manifest.exact_profile_id.has_value() ? 1 : 0, &hasher);
  if (manifest.exact_profile_id.has_value()) {
    UpdateHash('X', *manifest.exact_profile_id, &hasher);
  }
  UpdateU8(manifest.exact_profile_admission_record_id.has_value() ? 1 : 0,
           &hasher);
  if (manifest.exact_profile_admission_record_id.has_value()) {
    UpdateHash('A', *manifest.exact_profile_admission_record_id, &hasher);
  }
  UpdateU8(manifest.exact_output_evidence_hash.has_value() ? 1 : 0, &hasher);
  if (manifest.exact_output_evidence_hash.has_value()) {
    UpdateHash('V', *manifest.exact_output_evidence_hash, &hasher);
  }
  UpdateU32(manifest.exact_logit_frame_count, &hasher);
  return hasher.Finalize();
}

absl::Status ValidateDPMProjectionManifest(
    const DPMProjectionManifest& manifest) {
  ABSL_RETURN_IF_ERROR(ValidateManifestFields(manifest));
  if (IsZeroHash(manifest.manifest_hash)) {
    return absl::InvalidArgumentError(
        "DPM projection manifest has no content address.");
  }
  ABSL_ASSIGN_OR_RETURN(Hash256 expected,
                        ComputeDPMProjectionManifestHash(manifest));
  if (expected != manifest.manifest_hash) {
    return absl::DataLossError(
        "DPM projection manifest content address does not match its fields.");
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
