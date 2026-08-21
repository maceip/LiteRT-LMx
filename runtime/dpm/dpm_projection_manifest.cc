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
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kManifestDomain =
    "LITERT_LMX_DPM_PROJECTION_MANIFEST_SHA256_V1";
constexpr size_t kMaximumIdentityBytes = 16 * 1024;

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

bool ContainsControlByte(absl::string_view value) {
  for (unsigned char byte : value) {
    if (byte < 0x20 || byte == 0x7f) return true;
  }
  return false;
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
      manifest.log_id.size() > kMaximumIdentityBytes ||
      manifest.case_id.size() > kMaximumIdentityBytes ||
      ContainsControlByte(manifest.log_id) ||
      ContainsControlByte(manifest.case_id)) {
    return absl::InvalidArgumentError(
        "DPM projection manifest requires bounded log and case identities "
        "without control bytes.");
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
      (manifest.baseline_manifest_hash.has_value() &&
       IsZeroHash(*manifest.baseline_manifest_hash)) ||
      (manifest.baseline_output_hash.has_value() &&
       IsZeroHash(*manifest.baseline_output_hash))) {
    return absl::InvalidArgumentError(
        "DPM projection manifest contains an empty cryptographic identity.");
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
