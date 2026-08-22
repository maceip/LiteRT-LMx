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

#include "runtime/engine/session_handoff_capability.h"

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

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

void HashU8(uint8_t value, Sha256Hasher* hasher) {
  const char byte = static_cast<char>(value);
  hasher->Update(absl::string_view(&byte, 1));
}

void HashU32(uint32_t value, Sha256Hasher* hasher) {
  std::array<char, 4> bytes;
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (24 - i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void HashU64(uint64_t value, Sha256Hasher* hasher) {
  std::array<char, 8> bytes;
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (56 - i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void HashFrame(absl::string_view value, Sha256Hasher* hasher) {
  HashU64(value.size(), hasher);
  hasher->Update(value);
}

template <size_t Size>
void HashMagic(const std::array<char, Size>& magic, Sha256Hasher* hasher) {
  HashFrame(absl::string_view(magic.data(), magic.size()), hasher);
}

void HashDigest(const Hash256& value, Sha256Hasher* hasher) {
  hasher->Update(absl::string_view(
      reinterpret_cast<const char*>(value.bytes.data()), value.bytes.size()));
}

absl::Status ValidateCapabilityFields(
    const SessionHandoffCapability& capability) {
  if (capability.version != kSessionHandoffCapabilityVersion) {
    return absl::UnimplementedError(
        "Unsupported session handoff capability version.");
  }
  if (IsZeroHash(capability.session_identity.model_artifact_hash) ||
      IsZeroHash(capability.session_identity.runtime_artifact_hash) ||
      IsZeroHash(capability.session_identity.inference_profile_hash) ||
      IsZeroHash(capability.exact_profile_id) ||
      IsZeroHash(capability.complete_state_inventory_hash) ||
      IsZeroHash(capability.capsule_codec_contract_hash)) {
    return absl::InvalidArgumentError(
        "Session handoff capability evidence hashes must all be nonzero.");
  }
  switch (capability.backend) {
    case ExactLiteRtBackend::kCpu:
    case ExactLiteRtBackend::kMetalGpu:
      break;
    case ExactLiteRtBackend::kNpu:
      return absl::UnimplementedError(
          "Session handoff capability has no complete NPU capsule "
          "inventory.");
    case ExactLiteRtBackend::kUnclassifiedGpu:
      return absl::UnimplementedError(
          "An unclassified GPU cannot own a session handoff capability.");
    default:
      return absl::UnimplementedError(
          "Unsupported backend cannot own a session handoff capability.");
  }
  if (capability.capsule_codec_contract_hash !=
      GetSessionHandoffCapsuleCodecContractHash()) {
    return absl::FailedPreconditionError(
        "Session handoff capability does not bind the current LRTSESS1 and "
        "LRTST001 codec contract.");
  }
  return absl::OkStatus();
}

}  // namespace

Hash256 GetSessionHandoffCapsuleCodecContractHash() {
  Sha256Hasher hasher;
  HashFrame("LITERT_LM_SESSION_HANDOFF_CAPSULE_CODEC_CONTRACT_V1", &hasher);
  HashMagic(kSessionHandoffEnvelopeMagic, &hasher);
  HashU32(kSessionHandoffEnvelopeVersion, &hasher);
  HashFrame("HMAC_SHA256_COMPLETE_ENVELOPE_V1", &hasher);
  HashFrame("AUTHENTICATE_BEFORE_PARSE_V1", &hasher);
  HashMagic(kLiteRtStateSnapshotMagic, &hasher);
  HashU32(kLiteRtStateSnapshotVersion, &hasher);
  HashFrame("SHA256_STATE_TRAILER_V1", &hasher);
  HashFrame("CANONICAL_BIG_ENDIAN_FIELDS_V1", &hasher);
  return hasher.Finalize();
}

absl::StatusOr<Hash256> ComputeSessionHandoffCapabilityId(
    const SessionHandoffCapability& capability) {
  ABSL_RETURN_IF_ERROR(ValidateCapabilityFields(capability));
  Sha256Hasher hasher;
  HashFrame("LITERT_LM_SESSION_HANDOFF_CAPABILITY_V2", &hasher);
  HashU32(capability.version, &hasher);
  HashDigest(capability.session_identity.model_artifact_hash, &hasher);
  HashDigest(capability.session_identity.runtime_artifact_hash, &hasher);
  HashDigest(capability.session_identity.inference_profile_hash, &hasher);
  HashDigest(capability.exact_profile_id, &hasher);
  HashU8(static_cast<uint8_t>(capability.backend), &hasher);
  HashDigest(capability.complete_state_inventory_hash, &hasher);
  HashDigest(capability.capsule_codec_contract_hash, &hasher);
  const Hash256 capability_id = hasher.Finalize();
  if (IsZeroHash(capability_id)) {
    return absl::InternalError(
        "Session handoff capability produced a zero identifier.");
  }
  return capability_id;
}

absl::Status ValidateSessionHandoffCapability(
    const SessionHandoffCapability& capability) {
  ABSL_ASSIGN_OR_RETURN(const Hash256 expected_id,
                        ComputeSessionHandoffCapabilityId(capability));
  if (IsZeroHash(capability.capability_id) ||
      capability.capability_id != expected_id) {
    return absl::InvalidArgumentError(
        "Session handoff capability identifier is not canonical.");
  }
  return absl::OkStatus();
}

absl::Status ValidateSessionHandoffCapabilityAssertion(
    const SessionHandoffCapability& derived,
    const SessionHandoffCapabilityAssertion& assertion) {
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffCapability(derived));
  if (assertion.expected_capability_id.has_value() &&
      *assertion.expected_capability_id != derived.capability_id) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the Engine-derived session handoff "
        "capability identifier.");
  }
  if (assertion.expected_session_identity.has_value() &&
      *assertion.expected_session_identity != derived.session_identity) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the Engine-derived session "
        "identity.");
  }
  if (assertion.expected_exact_profile_id.has_value() &&
      *assertion.expected_exact_profile_id != derived.exact_profile_id) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the Engine-derived exact LiteRT "
        "profile identifier.");
  }
  if (assertion.expected_backend.has_value() &&
      *assertion.expected_backend != derived.backend) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the Engine-derived handoff "
        "backend.");
  }
  if (assertion.expected_complete_state_inventory_hash.has_value() &&
      *assertion.expected_complete_state_inventory_hash !=
          derived.complete_state_inventory_hash) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the executor-derived complete "
        "session-state inventory.");
  }
  if (assertion.expected_capsule_codec_contract_hash.has_value() &&
      *assertion.expected_capsule_codec_contract_hash !=
          derived.capsule_codec_contract_hash) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the runtime-owned capsule codec "
        "contract.");
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
