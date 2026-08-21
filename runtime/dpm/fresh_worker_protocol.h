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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FRESH_WORKER_PROTOCOL_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FRESH_WORKER_PROTOCOL_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_prepared_prefill_plan.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

inline constexpr uint32_t kFreshWorkerProtocolVersion = 3;
// DPMTOK01 is an independent canonical product format. Worker-envelope
// evolution must not silently change already-recorded exact token bytes.
inline constexpr uint32_t kFreshWorkerTokenEncodingVersion = 1;
inline constexpr uint32_t kFreshWorkerExecutionPlanFormatVersion = 1;
inline constexpr uint32_t kMaximumFreshWorkerRuns = 64;
inline constexpr uint64_t kMaximumFreshWorkerRequestPayloadBytes =
    uint64_t{16} * 1024 * 1024;
inline constexpr uint64_t kMaximumFreshWorkerTokenBytes =
    uint64_t{64} * 1024 * 1024;
inline constexpr uint64_t kMaximumFreshWorkerCanonicalOutputBytes =
    uint64_t{16} * 1024 * 1024;
inline constexpr uint32_t kMaximumFreshWorkerLogitFrames = 1'000'000;
inline constexpr uint32_t kMaximumFreshWorkerTokenIds = 1'000'000;
inline constexpr uint64_t kMaximumFreshWorkerEnvelopeBytes =
    uint64_t{128} * 1024 * 1024;

// Caller-owned authentication material. The public key ID is serialized for
// rotation and lookup. The secret authentication key is never serialized in a
// request, result, admission record, command line, or environment variable.
struct FreshWorkerAuthentication {
  std::string key_id;
  std::string authentication_key;
};

// Physical prefill work is deliberately separate from the complete logical
// replay request. A checkpoint can therefore change work without changing the
// logical replay key. Off-position grafting is not represented.
enum class FreshWorkerPrefillMode : uint32_t {
  kFullCanonicalPrefill = 1,
  kOwnPositionCapsuleDelta = 2,
};

struct FreshWorkerExecutionPlan {
  uint32_t format_version = kFreshWorkerExecutionPlanFormatVersion;

  // SHA-256 of every canonical model-affecting field below except this digest
  // itself. `capture_producing_capsule` is also excluded because capture runs
  // only after output has been finalized; the enclosing request HMAC still
  // authenticates that policy bit.
  Hash256 plan_hash;
  FreshWorkerPrefillMode prefill_mode =
      FreshWorkerPrefillMode::kFullCanonicalPrefill;

  // SHA-256 of FreshWorkerRequest::request_payload. The payload remains the
  // complete canonical logical replay request in both prefill modes.
  Hash256 logical_replay_request_hash;

  // Restore-only bindings. Full-prefill plans must leave all of them empty or
  // zero. The capsule bytes are transported separately and never enter the
  // authenticated request envelope.
  std::optional<Hash256> restore_checkpoint_id;
  SessionHandoffIdentity session_identity;
  Hash256 restore_durable_envelope_hash;
  uint64_t restore_durable_envelope_size = 0;
  std::string canonical_execution_payload;

  bool capture_producing_capsule = false;
};

absl::Status ValidateFreshWorkerExecutionPlan(
    const FreshWorkerExecutionPlan& plan);
absl::StatusOr<Hash256> ComputeFreshWorkerExecutionPlanHash(
    const FreshWorkerExecutionPlan& plan);

// One independently launched worker consumes exactly one request. The exact
// profile digest must be derived by the loaded runtime; this protocol treats it
// only as an opaque binding and never lets it stand in for runtime derivation.
struct FreshWorkerRequest {
  uint32_t format_version = kFreshWorkerProtocolVersion;
  Hash256 exact_profile_hash;
  Hash256 qualification_id;
  uint32_t run_index = 0;
  uint32_t run_count = 0;
  Hash256 challenge_nonce;
  std::string request_payload;
  FreshWorkerExecutionPlan execution_plan;
};

// Stable wire identity for the element representation hashed in a logits
// frame. Exact modes currently admit only the two floating-point
// representations consumed by LiteRT-LM's CPU greedy sampler.
enum class FreshWorkerLogitElementType : uint32_t {
  kUnsupported = 0,
  kFloat16 = 1,
  kFloat32 = 2,
};

// SHA-256 evidence for one complete, full-vocabulary logits buffer, in
// generation order and before sampling. The compact tensor is described as
// [batch_size, sequence_size, vocabulary_size]. A batch entry is one output
// candidate. Successful output
// requires exactly one frame for every canonical sampled token ID.
//
// `byte_count` is part of equality and must equal the overflow-checked product
// of all three dimensions and `element_byte_width`. This proves that the hash
// covers the declared compact tensor extent. Exact-profile qualification must
// additionally compare the declared contract with the contract derived by the
// loaded Engine; self-declared dimensions alone are not runtime proof.
struct FreshWorkerLogitFrameEvidence {
  FreshWorkerLogitElementType element_type =
      FreshWorkerLogitElementType::kUnsupported;
  uint32_t element_byte_width = 0;
  uint32_t batch_size = 0;
  uint32_t sequence_size = 0;
  uint32_t vocabulary_size = 0;
  uint64_t byte_count = 0;
  Hash256 sha256;

  bool operator==(const FreshWorkerLogitFrameEvidence& other) const {
    return element_type == other.element_type &&
           element_byte_width == other.element_byte_width &&
           batch_size == other.batch_size &&
           sequence_size == other.sequence_size &&
           vocabulary_size == other.vocabulary_size &&
           byte_count == other.byte_count && sha256 == other.sha256;
  }
};

// The model-specific worker callback returns only execution output. Protocol
// bindings, the process nonce, request digest, and authentication tag are
// supplied by the generic one-request worker boundary.
struct FreshWorkerExecutionOutput {
  // Exact stage product (for example projection JSON or agent-decision text)
  // in the stage's canonical byte representation. This is compared in
  // addition to, never instead of, token and logits evidence.
  std::string canonical_output;
  // Canonical DPMTOK01 encoding produced by EncodeFreshWorkerTokenIds. Raw
  // text, native-endian integers, visible-text filtering, or omitted EOS/stop
  // token IDs are forbidden. The concrete product executor adapter must pass
  // every sampled ID, including the terminating ID; this generic codec can
  // validate the supplied sequence but cannot infer that an upstream adapter
  // omitted a token.
  std::string token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> logit_frames;
};

enum class FreshWorkerReplayIsolation : uint32_t {
  kUnverified = 0,
  kEmptyCatalogs = 1,
};

// Authenticated evidence for a producing-session capsule streamed separately
// from the bounded result envelope. The transient envelope uses a per-request
// key; the parent must authenticate and rewrap it before durable publication.
struct FreshWorkerProducingCapsuleEvidence {
  SessionHandoffIdentity session_identity;
  uint64_t transient_envelope_size = 0;
  Hash256 transient_envelope_hash;
  Hash256 output_evidence_hash;

  // Runtime-produced continuation witnesses. These must be canonical and
  // byte-for-byte equal: the producing session's first export wrote the
  // transient capsule, its second export used a digest-only sink, and a newly
  // created target imported that sealed capsule and independently re-exported
  // its committed live state. No field may be synthesized from request or
  // checkpoint metadata.
  SessionContinuationStateWitness producer_first_export;
  SessionContinuationStateWitness producer_second_export;
  SessionContinuationStateWitness fresh_import_target;

  bool operator==(const FreshWorkerProducingCapsuleEvidence& other) const {
    return session_identity == other.session_identity &&
           transient_envelope_size == other.transient_envelope_size &&
           transient_envelope_hash == other.transient_envelope_hash &&
           output_evidence_hash == other.output_evidence_hash &&
           producer_first_export == other.producer_first_export &&
           producer_second_export == other.producer_second_export &&
           fresh_import_target == other.fresh_import_target;
  }
};

// An authenticated observation, not a claim that a profile is universally
// deterministic. A non-OK result carries no model output and cannot be used by
// admission.
struct FreshWorkerResult {
  uint32_t format_version = kFreshWorkerProtocolVersion;
  Hash256 exact_profile_hash;
  Hash256 qualification_id;
  uint32_t run_index = 0;
  uint32_t run_count = 0;
  Hash256 challenge_nonce;
  Hash256 request_envelope_hash;
  Hash256 worker_instance_nonce;
  Hash256 execution_plan_hash;
  // Present on every successful exact agent execution. Projection workers
  // carry neither a prepared plan nor a restore witness.
  std::optional<DPMPreparedPrefillPlan> prepared_prefill_plan;
  std::optional<Hash256> restored_checkpoint_id;
  // Present only for a successful own-position delta run and copied from the
  // live target returned by ImportHandoffFromWithWitness.
  std::optional<SessionContinuationStateWitness> restored_state_witness;
  std::optional<FreshWorkerProducingCapsuleEvidence>
      producing_capsule_evidence;
  FreshWorkerReplayIsolation replay_isolation =
      FreshWorkerReplayIsolation::kUnverified;
  absl::StatusCode status_code = absl::StatusCode::kOk;
  std::string status_message;
  std::string canonical_output;
  std::string token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> logit_frames;
};

absl::Status ValidateFreshWorkerAuthentication(
    const FreshWorkerAuthentication& authentication);
absl::Status ValidateFreshWorkerRequest(const FreshWorkerRequest& request);
absl::Status ValidateFreshWorkerExecutionOutput(
    const FreshWorkerExecutionOutput& output);
// Canonical digest over the complete stage output, canonical token bytes, and
// ordered full-logits evidence. This is the output binding used by both
// profile admission and producing-session capsule evidence.
Hash256 ComputeFreshWorkerOutputEvidenceHash(
    absl::string_view canonical_output,
    absl::string_view token_bytes,
    const std::vector<FreshWorkerLogitFrameEvidence>& logit_frames);
absl::Status ValidateFreshWorkerResult(const FreshWorkerResult& result);
// Cross-checks a self-valid result against the authenticated request plan.
// Failure results carry the execution-plan hash but no prepared-prefill,
// restore, continuation-witness, or capsule claim.
absl::Status ValidateFreshWorkerResultForRequest(
    const FreshWorkerResult& result, const FreshWorkerRequest& request);
absl::Status ValidateFreshWorkerLogitFrameEvidence(
    const FreshWorkerLogitFrameEvidence& frame);

// The sole admitted generated-token representation. DPMTOK01 is:
// 8-byte magic, explicit version 1, big-endian uint32 count, followed by
// exactly `count` nonnegative big-endian int32 token IDs. Decode rejects
// out-of-range counts, negative IDs, truncation, and trailing bytes.
absl::StatusOr<std::string> EncodeFreshWorkerTokenIds(
    const std::vector<int32_t>& token_ids);
absl::StatusOr<std::vector<int32_t>> DecodeFreshWorkerTokenIds(
    absl::string_view canonical_token_bytes);

// Hashes the complete canonical bytes of one logits buffer. The caller must
// pass the exact produced buffer bytes in tensor order, with no text or numeric
// reformatting between inference and this function.
absl::StatusOr<FreshWorkerLogitFrameEvidence>
MakeFreshWorkerLogitFrameEvidence(
    absl::string_view canonical_logit_bytes,
    FreshWorkerLogitElementType element_type, uint32_t batch_size,
    uint32_t sequence_size, uint32_t vocabulary_size);

// Canonical, bounded, HMAC-SHA256 authenticated envelopes. Decode authenticates
// the entire envelope before parsing body fields.
absl::StatusOr<std::string> EncodeFreshWorkerRequest(
    const FreshWorkerRequest& request,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<FreshWorkerRequest> DecodeFreshWorkerRequest(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<std::string> EncodeFreshWorkerResult(
    const FreshWorkerResult& result,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<FreshWorkerResult> DecodeFreshWorkerResult(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication);

Hash256 HashFreshWorkerEnvelope(absl::string_view envelope);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_FRESH_WORKER_PROTOCOL_H_
