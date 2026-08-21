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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_HANDOFF_CODEC_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_HANDOFF_CODEC_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/engine/session_handoff.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/state_interface.h"
#include "runtime/util/byte_stream.h"

namespace litert::lm {

enum class SessionHandoffPhase : uint8_t {
  kFresh = 0,
  kPrefilled = 1,
  kDecoded = 2,
};

struct SessionHandoffSnapshot {
  SessionHandoffPhase phase = SessionHandoffPhase::kFresh;
  ExecutorSessionSnapshot executor;
};

// Metadata decoded from an authenticated envelope plus the exact byte range
// containing its canonical LiteRT state. The state remains in the caller's
// ByteSource so large KV caches do not need a second in-memory copy.
struct DecodedSessionHandoff {
  SessionHandoffSnapshot snapshot;
  uint64_t serialized_state_offset = 0;
  uint64_t serialized_state_size = 0;
};

// Streams a canonical authenticated envelope directly from live LiteRT state.
// The state size is measured and its emitted byte count is checked before the
// MAC is committed.
absl::Status EncodeSessionHandoffTo(
    const SessionHandoffSnapshot& snapshot, const StateInterface& state,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options, ByteSink* sink);

// Authenticates the complete source before parsing any metadata and returns a
// checked subrange for state import. It performs no executor/session mutation.
absl::StatusOr<DecodedSessionHandoff> DecodeSessionHandoffFrom(
    const ByteSource& envelope,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options);

// Authenticates and canonically decodes the complete source before emitting
// any destination bytes, then streams the exact serialized-state subrange
// into an otherwise identical envelope authenticated by destination_options.
// This performs no session creation, import, or executor mutation. `source`
// must remain immutable and must not alias storage mutated by `destination`
// for the duration of the call. As with other streaming encoders, callers
// must discard destination bytes if a later source or destination I/O error
// is returned; durable publication must be transactional around this call.
absl::Status ReauthenticateSessionHandoffTo(
    const ByteSource& source,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& source_options,
    const SessionHandoffOptions& destination_options,
    ByteSink* destination);

// Produces a canonical versioned envelope authenticated with HMAC-SHA256.
// The key itself is caller-owned and is never serialized.
absl::StatusOr<std::string> EncodeSessionHandoff(
    const SessionHandoffSnapshot& snapshot,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options);

// Authenticates before parsing, validates the expected identity and key id,
// and returns a fully validated in-memory snapshot. This compatibility helper
// materializes state; production handoff paths use DecodeSessionHandoffFrom.
absl::StatusOr<SessionHandoffSnapshot> DecodeSessionHandoff(
    absl::string_view envelope,
    const SessionHandoffIdentity& authoritative_identity,
    const SessionHandoffOptions& options);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_CORE_SESSION_HANDOFF_CODEC_H_
