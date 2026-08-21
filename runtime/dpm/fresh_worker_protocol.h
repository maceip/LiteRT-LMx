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
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

inline constexpr uint32_t kFreshWorkerProtocolVersion = 1;
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
absl::Status ValidateFreshWorkerResult(const FreshWorkerResult& result);
absl::Status ValidateFreshWorkerLogitFrameEvidence(
    const FreshWorkerLogitFrameEvidence& frame);

// The sole admitted generated-token representation. DPMTOK01 is:
// 8-byte magic, big-endian uint32 version, big-endian uint32 count, followed
// by exactly `count` nonnegative big-endian int32 token IDs. Decode rejects
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
