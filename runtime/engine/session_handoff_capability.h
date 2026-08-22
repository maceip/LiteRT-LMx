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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_CAPABILITY_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_CAPABILITY_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/engine/session_handoff_codec_contract.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Version of the canonical capability identity below. This is deliberately
// independent of the LRTSESS1 and LRTST001 wire-format versions: changing
// either wire contract changes `capsule_codec_contract_hash`, while changing
// the capability field set or an authoritative field's semantics changes this
// version.
inline constexpr uint32_t kSessionHandoffCapabilityVersion = 2;

// Engine-derived proof that a concrete exact LiteRT profile can export and
// restore the complete execution capsule represented by the named codec
// contract. This is a capability identity, not CapsuleRestore admission:
// producer-versus-restore equality must be demonstrated separately.
struct SessionHandoffCapability {
  static constexpr uint32_t kFormatVersion =
      kSessionHandoffCapabilityVersion;

  uint32_t version = kFormatVersion;

  // SHA-256 of the canonical, versioned fields below.
  Hash256 capability_id;

  SessionHandoffIdentity session_identity;
  Hash256 exact_profile_id;
  ExactLiteRtBackend backend = ExactLiteRtBackend::kUnsupported;

  // Executor-owned digest of every admitted mutable or reconstructible state
  // element needed for continuation. Dynamic buffers bind their structural
  // axis and per-entry layout, while each capsule's codec binds its concrete
  // live capacity. A backend cannot derive this digest merely from a
  // requested backend or model label.
  Hash256 complete_state_inventory_hash;

  // Canonical identity of the authenticated LRTSESS1 envelope and its nested
  // LRTST001 LiteRT-state snapshot contract.
  Hash256 capsule_codec_contract_hash;

  bool operator==(const SessionHandoffCapability& other) const {
    return version == other.version &&
           capability_id == other.capability_id &&
           session_identity == other.session_identity &&
           exact_profile_id == other.exact_profile_id &&
           backend == other.backend &&
           complete_state_inventory_hash ==
               other.complete_state_inventory_hash &&
           capsule_codec_contract_hash ==
               other.capsule_codec_contract_hash;
  }
  bool operator!=(const SessionHandoffCapability& other) const {
    return !(*this == other);
  }
};

// Every optional field is an assertion against Engine-derived evidence. None
// of these values may select or replace a loaded runtime fact.
struct SessionHandoffCapabilityAssertion {
  std::optional<Hash256> expected_capability_id;
  std::optional<SessionHandoffIdentity> expected_session_identity;
  std::optional<Hash256> expected_exact_profile_id;
  std::optional<ExactLiteRtBackend> expected_backend;
  std::optional<Hash256> expected_complete_state_inventory_hash;
  std::optional<Hash256> expected_capsule_codec_contract_hash;
};

// Returns the stable SHA-256 identity of the currently implemented LRTSESS1
// plus LRTST001 codec contract. The hash is derived from the same exported
// magic/version constants consumed by the codecs themselves.
Hash256 GetSessionHandoffCapsuleCodecContractHash();

// Computes the canonical capability identifier after validating every
// authoritative field other than `capability_id`.
absl::StatusOr<Hash256> ComputeSessionHandoffCapabilityId(
    const SessionHandoffCapability& capability);

// Validates all fields and verifies that `capability_id` is the canonical
// digest. Unknown versions and unsupported backends fail closed.
absl::Status ValidateSessionHandoffCapability(
    const SessionHandoffCapability& capability);

absl::Status ValidateSessionHandoffCapabilityAssertion(
    const SessionHandoffCapability& derived,
    const SessionHandoffCapabilityAssertion& assertion);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_CAPABILITY_H_
