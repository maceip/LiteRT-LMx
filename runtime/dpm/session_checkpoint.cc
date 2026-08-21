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

#include "runtime/dpm/session_checkpoint.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kDescriptorMagic = {'D', 'P', 'M', 'K',
                                                   'V', '0', '0', '1'};

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
}

void AppendU32(uint32_t value, std::string* output) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendU64(uint64_t value, std::string* output) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendI64(int64_t value, std::string* output) {
  AppendU64(static_cast<uint64_t>(value), output);
}

absl::Status AppendString(absl::string_view value, std::string* output) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError(
        "DPM checkpoint descriptor string is too large.");
  }
  AppendU32(static_cast<uint32_t>(value.size()), output);
  output->append(value.data(), value.size());
  return absl::OkStatus();
}

void AppendHash(const Hash256& value, std::string* output) {
  output->append(reinterpret_cast<const char*>(value.bytes.data()),
                 value.bytes.size());
}

absl::Status ValidateDescriptorFields(
    const DPMSessionCheckpointDescriptor& descriptor) {
  if (descriptor.format_version !=
      DPMSessionCheckpointDescriptor::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Unsupported DPM session checkpoint descriptor version.");
  }
  if (descriptor.log_id.empty()) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint requires a log id.");
  }
  if (descriptor.stage != DPMSessionCheckpointStage::kAgentDecision) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint has an unsupported capture stage.");
  }
  if (descriptor.response_event_index != descriptor.source_event_count) {
    return absl::InvalidArgumentError(
        "DPM checkpoint response index must immediately follow its source "
        "prefix.");
  }
  if (descriptor.key_id.empty() || descriptor.key_id.size() > 1024) {
    return absl::InvalidArgumentError(
        "DPM checkpoint authentication key id must contain 1 to 1024 "
        "bytes.");
  }
  if (descriptor.created_unix_micros <= 0) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint requires a positive creation timestamp.");
  }
  if (descriptor.envelope_size == 0 || IsZeroHash(descriptor.envelope_hash)) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint requires a non-empty envelope.");
  }
  if (IsZeroHash(descriptor.source_prefix_hash) ||
      IsZeroHash(descriptor.projection_request_hash) ||
      IsZeroHash(descriptor.projection_manifest_hash) ||
      IsZeroHash(descriptor.correction_digest) ||
      IsZeroHash(descriptor.agent_request_hash) ||
      IsZeroHash(descriptor.agent_transcript_hash) ||
      IsZeroHash(descriptor.session_identity.model_artifact_hash) ||
      IsZeroHash(descriptor.session_identity.runtime_artifact_hash) ||
      IsZeroHash(descriptor.session_identity.inference_profile_hash)) {
    return absl::InvalidArgumentError(
        "DPM session checkpoint is missing a required identity hash.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeDescriptorForHash(
    const DPMSessionCheckpointDescriptor& descriptor) {
  ABSL_RETURN_IF_ERROR(ValidateDescriptorFields(descriptor));
  std::string bytes;
  bytes.reserve(512 + descriptor.log_id.size() + descriptor.key_id.size());
  bytes.append(kDescriptorMagic.data(), kDescriptorMagic.size());
  AppendU32(descriptor.format_version, &bytes);
  ABSL_RETURN_IF_ERROR(AppendString(descriptor.log_id, &bytes));
  bytes.push_back(static_cast<char>(descriptor.stage));
  AppendU64(descriptor.source_event_count, &bytes);
  AppendHash(descriptor.source_prefix_hash, &bytes);
  AppendU64(descriptor.response_event_index, &bytes);
  AppendHash(descriptor.projection_request_hash, &bytes);
  AppendHash(descriptor.projection_manifest_hash, &bytes);
  AppendHash(descriptor.correction_digest, &bytes);
  AppendHash(descriptor.agent_request_hash, &bytes);
  AppendHash(descriptor.agent_transcript_hash, &bytes);
  AppendHash(descriptor.session_identity.model_artifact_hash, &bytes);
  AppendHash(descriptor.session_identity.runtime_artifact_hash, &bytes);
  AppendHash(descriptor.session_identity.inference_profile_hash, &bytes);
  ABSL_RETURN_IF_ERROR(AppendString(descriptor.key_id, &bytes));
  AppendHash(descriptor.envelope_hash, &bytes);
  AppendU64(descriptor.envelope_size, &bytes);
  AppendI64(descriptor.created_unix_micros, &bytes);
  return bytes;
}

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

}  // namespace

absl::StatusOr<Hash256> ComputeDPMSessionCheckpointId(
    const DPMSessionCheckpointDescriptor& descriptor) {
  ABSL_ASSIGN_OR_RETURN(std::string canonical,
                        EncodeDescriptorForHash(descriptor));
  return Sha256(canonical);
}

absl::Status ValidateDPMSessionCheckpointArtifact(
    const DPMSessionCheckpointArtifact& artifact) {
  ABSL_RETURN_IF_ERROR(ValidateDescriptorFields(artifact.descriptor));
  if (artifact.authenticated_envelope.size() !=
      artifact.descriptor.envelope_size) {
    return absl::DataLossError(
        "DPM session checkpoint envelope size does not match descriptor.");
  }
  if (Sha256(artifact.authenticated_envelope) !=
      artifact.descriptor.envelope_hash) {
    return absl::DataLossError(
        "DPM session checkpoint envelope hash does not match descriptor.");
  }
  ABSL_ASSIGN_OR_RETURN(Hash256 expected_id,
                        ComputeDPMSessionCheckpointId(artifact.descriptor));
  if (expected_id != artifact.descriptor.descriptor_id) {
    return absl::DataLossError(
        "DPM session checkpoint descriptor id is not canonical.");
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
