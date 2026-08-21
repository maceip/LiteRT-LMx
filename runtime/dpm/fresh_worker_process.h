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
#include "absl/strings/string_view.h"  // from @com_google_absl
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

// Public, versioned non-secret key IDs for the parent-derived transient
// envelopes. They let durable evidence validators distinguish a live worker
// witness from the caller's parent-only checkpoint key without exposing either
// authentication key.
inline constexpr absl::string_view kFreshWorkerTransientRestoreKeyId =
    "litert-lmx-fresh-worker-restore-v3";
inline constexpr absl::string_view kFreshWorkerTransientProducingKeyId =
    "litert-lmx-fresh-worker-producing-v3";

// Stable, evidence-bound purposes for the two parent-only capsule rewraps.
// These labels are public protocol identifiers, not authentication material.
inline constexpr absl::string_view
    kFreshWorkerDurableRestoreToTransientReauthenticationPurpose =
        "litert-lmx/fresh-worker/durable-restore-to-transient/v1";
inline constexpr absl::string_view
    kFreshWorkerTransientProducingToDurableReauthenticationPurpose =
        "litert-lmx/fresh-worker/transient-producing-to-durable/v1";

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

enum class FreshWorkerEnvironmentContract : uint32_t {
  // Exact workers receive an environment containing only the terminating
  // null pointer. Model paths, delegate choices, settings, keys, and mode
  // selectors therefore cannot arrive through ambient process state.
  kEmpty = 1,
};

// Immutable parent-derived measurement of the configured worker executable.
// The executable image hash covers every file byte (including an embedded
// Mach-O code-signature blob when present). File metadata is additionally
// committed so replacing an executable with byte-identical but independently
// installed storage still creates a different certification object. This does
// not validate a platform code signature or prove that the measured bytes
// implement the locally asserted Engine-adapter contract.
class FreshWorkerCertification final {
 public:
  static constexpr uint32_t kFormatVersion = 1;

  FreshWorkerCertification(const FreshWorkerCertification&) = default;
  FreshWorkerCertification(FreshWorkerCertification&&) = default;
  FreshWorkerCertification& operator=(const FreshWorkerCertification&) =
      delete;
  FreshWorkerCertification& operator=(FreshWorkerCertification&&) = delete;

  static absl::StatusOr<FreshWorkerCertification> Create(
      const FreshWorkerProcessOptions& options);

  uint32_t format_version() const { return format_version_; }
  const std::string& canonical_executable_path() const {
    return canonical_executable_path_;
  }
  const std::vector<std::string>& canonical_arguments() const {
    return canonical_arguments_;
  }
  const Hash256& executable_image_hash() const {
    return executable_image_hash_;
  }
  FreshWorkerEnvironmentContract environment_contract() const {
    return environment_contract_;
  }
  const std::string& engine_adapter_contract_version() const {
    return engine_adapter_contract_version_;
  }
  const Hash256& certification_hash() const { return certification_hash_; }

  // Reopens the canonical path without following a final symlink, rehashes
  // the complete image, and requires every committed file-identity field to
  // remain unchanged. This detects path/image drift around path-based spawn;
  // it is not an fd-based attestation of the exact image executed by the OS.
  absl::Status ValidateExecutableImage() const;

 private:
  friend class FreshWorkerProcessRunner;

  FreshWorkerCertification() = default;

  // Canonicalized POSIX identity and timestamps captured from one open file
  // descriptor before and after hashing. Kept private because consumers must
  // compare the complete certification digest rather than cherry-pick weaker
  // metadata fields.
  uint64_t device_ = 0;
  uint64_t inode_ = 0;
  uint64_t file_size_ = 0;
  uint32_t file_mode_ = 0;
  int64_t modification_time_seconds_ = 0;
  int64_t modification_time_nanoseconds_ = 0;
  int64_t status_change_time_seconds_ = 0;
  int64_t status_change_time_nanoseconds_ = 0;

  uint32_t format_version_ = kFormatVersion;
  std::string canonical_executable_path_;
  std::vector<std::string> canonical_arguments_;
  Hash256 executable_image_hash_;
  FreshWorkerEnvironmentContract environment_contract_ =
      FreshWorkerEnvironmentContract::kEmpty;
  std::string engine_adapter_contract_version_;
  Hash256 certification_hash_;
};

// Domain-separated launch identity used by durable evidence. Keeping this
// derivation public lets admission and per-request validators prove that a
// serialized launch hash actually commits its worker certification digest.
Hash256 ComputeFreshWorkerLaunchSpecHash(
    const Hash256& worker_certification_hash);

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
  // Parent-computed provenance for the authenticated transient-to-durable
  // rewrap. Its source endpoint is the worker's sealed producing capsule and
  // its destination endpoint is exactly the durable envelope above.
  SessionHandoffReauthenticationEvidence reauthentication_evidence;
};

struct FreshWorkerProcessObservation {
  FreshWorkerResult result;
  int64_t process_id = -1;

  // Digests of the exact authenticated protocol envelopes consumed and
  // emitted by this process. The request digest is repeated inside `result`;
  // both copies must agree before Run returns.
  Hash256 request_envelope_hash;
  Hash256 result_envelope_hash;

  // Parent-derived immutable certification of the exact executable image,
  // stable file identity, argv, empty environment contract, and Engine adapter
  // version label. The launch-spec hash explicitly commits this digest. The
  // adapter label is a local configuration assertion, and neither digest is a
  // substitute for the loaded model/delegate/profile identity derived inside
  // the worker.
  Hash256 worker_certification_hash;
  Hash256 launch_spec_hash;

  // Present only for a successful own-position restore. The parent
  // authenticates the selected durable capsule and retains the complete
  // durable-to-transient rewrap provenance before the transient envelope can
  // enter the worker. Failed observations deliberately publish none.
  std::optional<SessionHandoffReauthenticationEvidence>
      restore_reauthentication_evidence;

  // Present only after a requested producing capsule has been authenticated,
  // rewrapped under the caller's durable parent-only key, and completely
  // appended to the caller's destination.
  std::optional<FreshWorkerDurableProducingCapsuleEvidence>
      durable_producing_capsule_evidence;
};

// Spawns one new OS process for every Run call, derives a request-specific
// transport HMAC key from the supplied durable key, writes one bounded request
// plus an explicit capsule/zero frame, accepts the matching response pair, and
// requires clean process exit. Only the derived transport key enters the
// worker. It never reuses a process and never invokes a command shell.
class FreshWorkerProcessRunner final {
 public:
  // Resolves and hashes the worker executable once. Construction fails unless
  // the path is canonical, non-symlink, regular, executable, and stable across
  // the complete image read. Every Run revalidates the same image immediately
  // before spawn, immediately after spawn, and after process exit. These
  // checks detect a changed stable path/image but do not make path-based
  // posix_spawn equivalent to fd-based exec attestation.
  static absl::StatusOr<FreshWorkerProcessRunner> Create(
      FreshWorkerProcessOptions options);

  const FreshWorkerCertification& certification() const {
    return certification_;
  }

  absl::StatusOr<FreshWorkerProcessObservation> Run(
      const FreshWorkerRequest& request,
      const FreshWorkerAuthentication& authentication) const;

  // Concrete session-capable process boundary. The ordinary Run path
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
  FreshWorkerProcessRunner(FreshWorkerProcessOptions options,
                           FreshWorkerCertification certification);

  FreshWorkerProcessOptions options_;
  const FreshWorkerCertification certification_;
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

  // Runtime-produced physical agent prefill evidence. A successful exact
  // agent execution must provide this plan. A delta execution must also carry
  // the independently recomputed live import witness used to prepare and
  // execute it. Projection and failed executions carry neither.
  std::optional<DPMPreparedPrefillPlan> prepared_prefill_plan;
  std::optional<SessionContinuationStateWitness> restored_state_witness;

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

  struct ProducingCapsuleContinuationWitnesses {
    SessionContinuationStateWitness producer_first_export;
    SessionContinuationStateWitness producer_second_export;
    SessionContinuationStateWitness fresh_import_target;
  };
  std::optional<ProducingCapsuleContinuationWitnesses>
      producing_capsule_continuation_witnesses;
};

// Narrow worker-owned access to the transient producing-capsule file. It
// exposes only the ByteSink used by live export and a one-way seal-for-read
// operation. No durable source, key, destination, or publication capability
// crosses this boundary. The callback must not retain this object.
class FreshWorkerTransientCapsuleAccess {
 public:
  virtual ~FreshWorkerTransientCapsuleAccess() = default;
  virtual ByteSink* sink() = 0;
  virtual absl::StatusOr<const ByteSource*> SealForRead() = 0;
};

// The session-capable callback receives only per-request transient handoff
// capabilities. A restore pair is present only for an own-position delta
// plan; a producing options/access set is present only for post-output
// capture. No durable parent key or parent Session is exposed. The callback
// must not retain any context pointer after returning.
struct FreshWorkerExecutionContext {
  const FreshWorkerRequest& request;
  const ByteSource* const restore_source;
  const SessionHandoffOptions* const restore_options;
  ByteSink* const producing_sink;
  const SessionHandoffOptions* const producing_options;
  FreshWorkerTransientCapsuleAccess* const producing_capsule_access;
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
