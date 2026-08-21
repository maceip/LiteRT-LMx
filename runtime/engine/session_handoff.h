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

#include <optional>
#include <string>

#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

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

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_H_
