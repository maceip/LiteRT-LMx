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
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Deliberately high to avoid colliding with stdin/stdout/stderr and ordinary
// application descriptors. The concrete process runner derives a unique
// request-bound transport key from the caller's durable authentication key and
// passes only that derived key through this inherited pipe, never argv or the
// environment.
inline constexpr int kFreshWorkerAuthenticationFd = 198;

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
// transport HMAC key from the supplied durable key, writes one bounded request,
// accepts one bounded response, and requires clean process exit. Only the
// derived transport key enters the worker. It never reuses a process and never
// invokes a command shell.
class FreshWorkerProcessRunner final : public FreshWorkerRunner {
 public:
  explicit FreshWorkerProcessRunner(FreshWorkerProcessOptions options);

  absl::StatusOr<FreshWorkerProcessObservation> Run(
      const FreshWorkerRequest& request,
      const FreshWorkerAuthentication& authentication) const override;

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
};

using FreshWorkerExecutionCallback = std::function<
    absl::StatusOr<FreshWorkerDerivedExecution>(const FreshWorkerRequest&)>;

// Worker-side contract entry point. Reads the derived request transport key from
// kFreshWorkerAuthenticationFd, consumes exactly one framed request on stdin,
// invokes `execute` once, emits exactly one authenticated result on stdout,
// and returns. Model load/profile derivation and catalog-free construction
// belong inside the concrete product `execute` adapter; this generic callback
// boundary cannot establish either fact on its own.
absl::Status RunFreshWorkerOnce(const FreshWorkerExecutionCallback& execute);

// Cryptographically strong OS randomness used for qualification challenges
// and worker-instance nonces. Fails instead of falling back to a PRNG.
absl::StatusOr<Hash256> GenerateFreshWorkerNonce();

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FRESH_WORKER_PROCESS_H_
