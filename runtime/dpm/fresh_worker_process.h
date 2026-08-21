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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FRESH_WORKER_PROCESS_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FRESH_WORKER_PROCESS_H_

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"
#include "runtime/util/byte_stream.h"

namespace litert::lm {

// Deliberately high to avoid colliding with stdin/stdout/stderr and ordinary
// application descriptors. The concrete process runner derives a unique
// request-bound transport key from the caller's durable authentication key and
// passes only that derived key through this inherited pipe, never argv or the
// environment.
inline constexpr int kFreshWorkerAuthenticationFd = 198;

// Session capsules use a separate authenticated frame and do not consume the
// bounded request/result-envelope budget. The independent limit prevents a
// compromised peer from consuming unbounded local storage before capsule
// authentication completes.
inline constexpr uint64_t kMaximumFreshWorkerCapsuleBytes =
    uint64_t{8} * 1024 * 1024 * 1024;

struct FreshWorkerProcessOptions {
  // Must be an absolute path to a regular executable. `arguments` excludes
  // argv[0]. No string is ever interpreted by a shell.
  std::filesystem::path executable_path;
  std::vector<std::string> arguments;
  absl::Duration timeout = absl::Minutes(10);
  absl::Duration termination_grace = absl::Milliseconds(250);

  // Checked throughout parent-side I/O and process waiting. Cancellation is
  // fail-closed and terminates the child before returning.
  std::function<bool()> cancellation_requested;
};

// Parent-only durable inputs and outputs for the session-capable process
// boundary. Durable keys are used solely in the parent and never enter the
// worker authentication prelude, request envelope, command line, or
// environment. Pointer pairs must either both be present or both be absent.
// Capture options must contain the complete Engine-derived expected identity;
// this assertion prevents a full-prefill worker from selecting the identity
// of the durable capsule it asks the parent to publish.
// A capture destination must publish transactionally: if
// RunWithSessionHandoff returns an error, any bytes appended to it must be
// discarded. Restore and capture storage must not alias, and every pointer
// remains caller-owned for the complete synchronous call. Any file descriptor
// hidden behind a caller implementation must be close-on-exec; the generic
// ByteSource/ByteSink interfaces cannot inspect that property.
struct FreshWorkerSessionHandoffTransfer {
  const ByteSource* durable_restore_source = nullptr;
  const SessionHandoffOptions* durable_restore_options = nullptr;
  ByteSink* durable_capture_destination = nullptr;
  const SessionHandoffOptions* durable_capture_options = nullptr;
};

// Parent-observed durable form of an authenticated producing capsule. These
// are the exact fields needed to construct a checkpoint descriptor without
// trusting worker-supplied durable metadata. The result retains the worker's
// authenticated transient evidence and output binding.
struct FreshWorkerDurableProducingCapsuleEvidence {
  SessionHandoffIdentity session_identity;
  std::string key_id;
  uint64_t envelope_size = 0;
  Hash256 envelope_hash;
  Hash256 output_evidence_hash;
};

struct FreshWorkerProcessObservation {
  FreshWorkerResult result;
  int64_t process_id = -1;

  // Digests of the exact authenticated protocol envelopes consumed and
  // emitted by this process. The request digest is repeated inside `result`;
  // both copies must agree before Run returns.
  Hash256 request_envelope_hash;
  Hash256 result_envelope_hash;

  // Hash of the canonical executable path plus exact argv bytes. It is launch
  // routing evidence only, not a substitute for executable/delegate digests in
  // the derived ExactLiteRtProfile.
  Hash256 launch_spec_hash;

  // Present only after a requested producing capsule has been authenticated,
  // rewrapped under the caller's durable parent-only key, and completely
  // appended to the caller's destination.
  std::optional<FreshWorkerDurableProducingCapsuleEvidence>
      durable_producing_capsule_evidence;
};

class FreshWorkerRunner {
 public:
  virtual ~FreshWorkerRunner() = default;
  // Contract seam for orchestration and fault injection. Implementations other
  // than FreshWorkerProcessRunner do not, by this interface alone, prove a new
  // OS process, authenticated IPC, or empty replay catalogs. A product exact
  // path must own the concrete process runner rather than accept an arbitrary
  // caller-supplied implementation.
  virtual absl::StatusOr<FreshWorkerProcessObservation> Run(
      const FreshWorkerRequest& request,
      const FreshWorkerAuthentication& authentication) const = 0;
};

// Spawns one new OS process for every Run call, derives a request-specific
// transport HMAC key from the supplied durable key, writes one bounded request
// plus an explicit capsule/zero frame, accepts the matching response pair, and
// requires clean process exit. Only the derived transport key enters the
// worker. It never reuses a process and never invokes a command shell.
class FreshWorkerProcessRunner final : public FreshWorkerRunner {
 public:
  explicit FreshWorkerProcessRunner(FreshWorkerProcessOptions options);

  absl::StatusOr<FreshWorkerProcessObservation> Run(
      const FreshWorkerRequest& request,
      const FreshWorkerAuthentication& authentication) const override;

  // Concrete session-capable process boundary. The ordinary virtual Run path
  // intentionally remains full-prefill and capsule-free for exact-profile
  // admission. This path sends one bounded request frame followed by one
  // separately bounded transient capsule frame (which may be empty), and
  // accepts the corresponding result/capsule pair. Durable handoff keys never
  // cross the process boundary.
  absl::StatusOr<FreshWorkerProcessObservation> RunWithSessionHandoff(
      const FreshWorkerRequest& request,
      const FreshWorkerAuthentication& authentication,
      const FreshWorkerSessionHandoffTransfer& transfer) const;

 private:
  FreshWorkerProcessOptions options_;
};

// Output plus the profile digest derived by the engine loaded inside this
// worker process. The generic boundary compares the digest with the request
// before it will authenticate a successful result. These fields are a contract
// seam, not proof by themselves: the concrete product callback must own the
// loaded Engine and catalog-free runtime construction.
struct FreshWorkerDerivedExecution {
  Hash256 derived_exact_profile_hash;
  // Exact qualification is independent regeneration. The product worker
  // integration must construct its runtime with no replay catalog and attest
  // that fact; an omitted/default value fails closed. A custom callback merely
  // setting this enum is not independent evidence of isolation.
  FreshWorkerReplayIsolation replay_isolation =
      FreshWorkerReplayIsolation::kUnverified;
  FreshWorkerExecutionOutput output;

  // Session-capable callbacks attest which own-position checkpoint was
  // restored and whether they exported the producing session through the
  // controlled sink. The producing identity is runtime-derived and is used by
  // the generic boundary to authenticate that export; it is never supplied by
  // the request caller. `exported_producing_capsule` also attests that export
  // happened only after `output` was finalized; the product-owned adapter must
  // establish that sequencing because this generic callback cannot observe
  // operations inside the callback.
  std::optional<Hash256> restored_checkpoint_id;
  bool exported_producing_capsule = false;
  SessionHandoffIdentity producing_session_identity;
};

// The session-capable callback receives only per-request transient handoff
// capabilities. A restore pair is present only for an own-position delta
// plan; a producing pair is present only for post-output capture. No durable
// parent key or parent Session is exposed. The callback must not retain any
// context pointer after returning.
struct FreshWorkerExecutionContext {
  const FreshWorkerRequest& request;
  const ByteSource* const restore_source;
  const SessionHandoffOptions* const restore_options;
  ByteSink* const producing_sink;
  const SessionHandoffOptions* const producing_options;
};

using FreshWorkerExecutionCallback = std::function<
    absl::StatusOr<FreshWorkerDerivedExecution>(
        const FreshWorkerExecutionContext&)>;
using FreshWorkerCapsuleFreeExecutionCallback = std::function<
    absl::StatusOr<FreshWorkerDerivedExecution>(const FreshWorkerRequest&)>;

// Worker-side contract entry point. Both overloads use the same authenticated
// request-plus-capsule and result-plus-capsule framing, so one fixed worker
// executable needs no unauthenticated argv mode selector. The legacy-shaped
// callback overload is an explicit full-prefill/no-capture adapter; a restore
// or capture request fails closed before reaching that callback. Model load,
// profile derivation, and catalog-free construction remain the concrete
// product callback's responsibility.
absl::Status RunFreshWorkerOnce(
    const FreshWorkerCapsuleFreeExecutionCallback& execute);
absl::Status RunFreshWorkerOnce(
    const FreshWorkerExecutionCallback& execute);

// Descriptive alias for the session-capable overload above.
absl::Status RunFreshWorkerWithSessionHandoffOnce(
    const FreshWorkerExecutionCallback& execute);

// Cryptographically strong OS randomness used for qualification challenges
// and worker-instance nonces. Fails instead of falling back to a PRNG.
absl::StatusOr<Hash256> GenerateFreshWorkerNonce();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FRESH_WORKER_PROCESS_H_
