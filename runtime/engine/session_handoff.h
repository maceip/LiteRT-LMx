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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_H_

#include <cstdint>
#include <optional>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Shared upper bound for one complete authenticated LRTSESS1 capsule. This is
// the transport/product bound used by continuation witnesses; admission-record
// envelopes have a separate, much smaller bound.
inline constexpr uint64_t kMaximumSessionHandoffEnvelopeBytes =
    uint64_t{8} * 1024 * 1024 * 1024;

// Immutable identity of the loaded inference profile that is allowed to
// consume a session handoff. Every field is the SHA-256 digest of the exact
// retained model artifact, measured loaded runtime/delegate artifacts, or the
// fully-resolved inference profile. Engines derive this identity; callers may
// compare against it but never supply the identity sealed into an envelope.
struct SessionHandoffIdentity {
  Hash256 model_artifact_hash;
  Hash256 runtime_artifact_hash;
  Hash256 inference_profile_hash;

  bool operator==(const SessionHandoffIdentity& other) const {
    return model_artifact_hash == other.model_artifact_hash &&
           runtime_artifact_hash == other.runtime_artifact_hash &&
           inference_profile_hash == other.inference_profile_hash;
  }
  bool operator!=(const SessionHandoffIdentity& other) const {
    return !(*this == other);
  }
};

// Authentication inputs and an optional assertion for a session handoff. The
// key is never placed in the serialized envelope. Applications should use a
// distinct, secret key of at least 32 bytes and rotate it by changing key_id.
//
// `expected_identity` is only an assertion against the identity already owned
// by the loaded session. It can reject a mismatched session but cannot choose
// the identity that is encoded or accepted. There is deliberately no field by
// which a caller can supply the authoritative identity.
struct SessionHandoffOptions {
  std::string key_id;
  std::string authentication_key;
  std::optional<SessionHandoffIdentity> expected_identity;
};

// Public continuation phase carried by the authenticated LRTSESS1 envelope
// and independently committed by a live-session witness.
enum class SessionHandoffPhase : uint8_t {
  kFresh = 0,
  kPrefilled = 1,
  kDecoded = 2,
};

// Runtime-derived evidence for one exact live continuation state. The
// processed-history digest is SHA-256 over the complete canonical DPMTOK01
// bytes (including a backend pending token when present). The envelope digest
// covers the exact canonical LRTSESS1 bytes emitted by the live session,
// including its authentication tag. State and identity evidence are
// runtime-derived. Options select the authenticated key ID/key and may assert,
// but cannot supply or relabel, the Engine-owned session identity.
struct SessionContinuationStateWitness {
  static constexpr uint32_t kFormatVersion = 1;

  uint32_t format_version = kFormatVersion;
  Hash256 witness_id;
  SessionHandoffIdentity session_identity;
  SessionHandoffPhase phase = SessionHandoffPhase::kFresh;
  int32_t current_step = 0;
  bool ran_decode = false;
  Hash256 processed_history_token_bytes_hash;
  Hash256 envelope_hash;
  uint64_t envelope_size = 0;
  std::string key_id;

  bool operator==(const SessionContinuationStateWitness& other) const {
    return format_version == other.format_version &&
           witness_id == other.witness_id &&
           session_identity == other.session_identity &&
           phase == other.phase && current_step == other.current_step &&
           ran_decode == other.ran_decode &&
           processed_history_token_bytes_hash ==
               other.processed_history_token_bytes_hash &&
           envelope_hash == other.envelope_hash &&
           envelope_size == other.envelope_size && key_id == other.key_id;
  }
  bool operator!=(const SessionContinuationStateWitness& other) const {
    return !(*this == other);
  }
};

// Canonical versioned content address over every witness field except
// witness_id. Validation recomputes this ID and rejects partial, unsupported,
// or noncanonical evidence.
absl::StatusOr<Hash256> ComputeSessionContinuationStateWitnessId(
    const SessionContinuationStateWitness& witness);
absl::Status ValidateSessionContinuationStateWitness(
    const SessionContinuationStateWitness& witness);

// Stable identity of the V1 witness field set and the live-target re-export
// semantics. Coverage admission binds this value so a later witness evolution
// cannot be accepted under an older operational contract.
Hash256 GetSessionContinuationStateWitnessContractHash();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_H_
