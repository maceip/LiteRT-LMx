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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CANONICAL_REPLAY_CATALOG_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CANONICAL_REPLAY_CATALOG_H_

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

inline constexpr size_t kMaximumCanonicalReplayContractBytes = 16 * 1024;
inline constexpr size_t kMaximumCanonicalReplayOutputBytes =
    size_t{64} * 1024 * 1024;

// One model-visible request under one runtime identity. Production adapters
// must copy this identity from the loaded Engine and treat any caller-provided
// identity as an assertion before constructing the key. This storage primitive
// validates and authenticates the identity but cannot prove its provenance.
// The canonical request bytes remain outside the catalog; only their SHA-256
// is used as the request key.
struct CanonicalReplayKey {
  DPMReplayStage stage = DPMReplayStage::kProjection;
  SessionHandoffIdentity runtime_identity;
  Hash256 canonical_request_hash;
  std::string request_contract_version;

  bool operator==(const CanonicalReplayKey& other) const {
    return stage == other.stage &&
           runtime_identity == other.runtime_identity &&
           canonical_request_hash == other.canonical_request_hash &&
           request_contract_version == other.request_contract_version;
  }
};

// A candidate is still caller-owned and must never be returned after a
// competing publisher wins. `canonical_output` is an opaque, stage-specific,
// canonical envelope; projection and decision adapters own its typed codec.
// The catalog authenticates the supplied evidence hash but does not establish
// that it came from inference; that is the publishing adapter's obligation.
struct CanonicalReplayCandidate {
  CanonicalReplayKey key;
  std::string canonical_output;
  Hash256 execution_evidence_hash;
};

// Durable authenticated winner. The HMAC covers every field except the tag
// itself, including the exact output bytes and asserted runtime identity. The
// production adapter must already have checked that assertion against its
// loaded Engine before publication.
struct CanonicalReplayWinner {
  CanonicalReplayKey key;
  std::string canonical_output;
  Hash256 canonical_output_hash;
  Hash256 execution_evidence_hash;
  std::string authentication_key_id;
  Hash256 authentication_tag;

  bool operator==(const CanonicalReplayWinner& other) const {
    return key == other.key && canonical_output == other.canonical_output &&
           canonical_output_hash == other.canonical_output_hash &&
           execution_evidence_hash == other.execution_evidence_hash &&
           authentication_key_id == other.authentication_key_id &&
           authentication_tag == other.authentication_tag;
  }
};

struct PublishCanonicalReplayResult {
  // True only for the process that atomically created the winner. Both the
  // winner and every loser receive the subsequently reloaded durable bytes.
  bool published = false;
  CanonicalReplayWinner winner;
};

struct CanonicalReplayAuthentication {
  // Rotating this identifier creates a new catalog verification and path
  // domain. It is persisted; the key never is. The caller must retain the
  // matching key to resolve existing winners.
  std::string key_id;
  std::string authentication_key;
};

absl::Status ValidateCanonicalReplayKey(const CanonicalReplayKey& key);
absl::Status ValidateCanonicalReplayCandidate(
    const CanonicalReplayCandidate& candidate);
absl::Status ValidateCanonicalReplayWinner(
    const CanonicalReplayWinner& winner,
    const CanonicalReplayAuthentication& authentication);
absl::StatusOr<std::string> EncodeCanonicalReplayKey(
    const CanonicalReplayKey& key);
absl::StatusOr<Hash256> ComputeCanonicalReplayKeyHash(
    const CanonicalReplayKey& key);

class CanonicalWinnerReplayCatalog {
 public:
  virtual ~CanonicalWinnerReplayCatalog() = default;

  virtual absl::StatusOr<CanonicalReplayWinner> Get(
      const CanonicalReplayKey& key) const = 0;

  // Atomically selects one immutable, authenticated winner. The catalog must
  // never replace a prior winner. A losing publisher receives the committed
  // winner and must discard its candidate.
  virtual absl::StatusOr<PublishCanonicalReplayResult> Publish(
      const CanonicalReplayCandidate& candidate) = 0;
};

// Local, correctness-first implementation. Each request maps to a single
// create-once file; there is no mutable "latest" pointer. This class exists
// only for DPMReplayMode::kCanonicalWinnerReplay, and Create rejects an exact
// mode explicitly. Higher-level dispatch must likewise keep ExactRegeneration
// free of catalog reads and writes. The caller must provide an existing
// absolute storage root that is not writable by group or others; the
// implementation creates, permission-checks, and durably syncs every
// catalog-owned directory below it.
class LocalFilesystemCanonicalWinnerReplayCatalog final
    : public CanonicalWinnerReplayCatalog {
 public:
  static absl::StatusOr<
      std::unique_ptr<LocalFilesystemCanonicalWinnerReplayCatalog>>
  Create(std::filesystem::path root,
         CanonicalReplayAuthentication authentication,
         DPMReplayMode mode);

  absl::StatusOr<CanonicalReplayWinner> Get(
      const CanonicalReplayKey& key) const override;
  absl::StatusOr<PublishCanonicalReplayResult> Publish(
      const CanonicalReplayCandidate& candidate) override;

  absl::StatusOr<std::filesystem::path> WinnerPathFor(
      const CanonicalReplayKey& key) const;

 private:
  LocalFilesystemCanonicalWinnerReplayCatalog(
      std::filesystem::path root,
      CanonicalReplayAuthentication authentication)
      : root_(std::move(root)),
        authentication_(std::move(authentication)) {}

  std::filesystem::path root_;
  CanonicalReplayAuthentication authentication_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CANONICAL_REPLAY_CATALOG_H_
