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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_EXACT_PROFILE_ADMISSION_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_EXACT_PROFILE_ADMISSION_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/fresh_worker_process.h"
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

enum class ExactProfileAdmissionEvidenceKind : uint32_t {
  // Narrow statement valid only when this record was produced by the
  // product-owned concrete process runner/Engine adapter: the recorded
  // independently spawned processes produced equal bytes for this one
  // canonical qualification request. The serialized enum cannot prove which
  // FreshWorkerRunner implementation produced it.
  kIndependentColdRunEquality = 1,
};

struct ExactProfileQualificationSpec {
  // Settings are resolved by the already-loaded authoritative Engine.
  // Assertions can only reject that result; this surface accepts no caller-
  // supplied profile/model/runtime identity.
  std::optional<SessionConfig> session_config;
  ExactLiteRtProfileAssertion profile_assertion;
  std::string canonical_request_payload;
  uint32_t independent_run_count = 0;
};

struct ExactProfileAdmissionRunEvidence {
  uint32_t run_index = 0;
  int64_t process_id = -1;
  Hash256 challenge_nonce;
  Hash256 worker_instance_nonce;
  Hash256 request_envelope_hash;
  Hash256 result_envelope_hash;
  Hash256 launch_spec_hash;
  Hash256 output_evidence_hash;
};

// Durable record of observed equality. It does not prove that its producer used
// the concrete process runner, and it does not claim that unobserved prompts,
// another profile, or a future software/hardware environment will be
// deterministic. The product admission surface must control record production;
// consumers must key lookups by the complete Engine-derived profile.
struct ExactProfileAdmissionRecord {
  static constexpr uint32_t kFormatVersion = 2;

  uint32_t format_version = kFormatVersion;
  Hash256 record_id;
  ExactProfileAdmissionEvidenceKind evidence_kind =
      ExactProfileAdmissionEvidenceKind::kIndependentColdRunEquality;
  FreshWorkerReplayIsolation replay_isolation =
      FreshWorkerReplayIsolation::kEmptyCatalogs;
  Hash256 exact_profile_hash;
  Hash256 qualification_id;
  Hash256 qualification_request_hash;
  Hash256 request_payload_hash;
  uint64_t request_payload_size = 0;
  // Canonical full-prefill plan used by every admission run. Profile
  // admission is deliberately capsule-free; a restore or capture plan can
  // never be serialized as profile qualification evidence.
  Hash256 execution_plan_hash;
  bool capture_producing_capsule = false;
  uint32_t independent_run_count = 0;
  int64_t qualified_unix_micros = 0;
  std::string authentication_key_id;
  std::vector<ExactProfileAdmissionRunEvidence> runs;

  // Consensus output after byte-for-byte comparison of every run.
  std::string canonical_output;
  Hash256 canonical_output_hash;
  std::string token_bytes;
  Hash256 token_bytes_hash;
  std::vector<FreshWorkerLogitFrameEvidence> logit_frames;
};

Hash256 ComputeExactProfileQualificationRequestHash(
    const ExactLiteRtProfile& derived_profile,
    const ExactProfileQualificationSpec& spec);
absl::StatusOr<Hash256> ComputeExactProfileAdmissionRecordId(
    const ExactProfileAdmissionRecord& record);
absl::Status ValidateExactProfileAdmissionRecord(
    const ExactProfileAdmissionRecord& record);
// Revalidates the stored evidence against the complete identity and packed
// logits-frame contract derived by the currently loaded Engine.
absl::Status ValidateExactProfileAdmissionRecordForProfile(
    const ExactProfileAdmissionRecord& record,
    const ExactLiteRtProfile& runtime_derived_profile);

// Canonical authenticated storage envelope used by admission repositories.
// The authentication key is never included in the envelope.
absl::StatusOr<std::string> EncodeExactProfileAdmissionRecord(
    const ExactProfileAdmissionRecord& record,
    const FreshWorkerAuthentication& authentication);
absl::StatusOr<ExactProfileAdmissionRecord>
DecodeExactProfileAdmissionRecord(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication);

class ExactProfileAdmissionRepository {
 public:
  virtual ~ExactProfileAdmissionRepository() = default;

  // Create-once by exact profile. Identical re-publication is idempotent;
  // conflicting evidence for the same profile fails closed.
  virtual absl::Status PutIfAbsent(
      const ExactProfileAdmissionRecord& record,
      const FreshWorkerAuthentication& authentication) = 0;
  virtual absl::StatusOr<ExactProfileAdmissionRecord> Get(
      const ExactLiteRtProfile& runtime_derived_profile,
      const FreshWorkerAuthentication& authentication) const = 0;
};

class ExactProfileQualifier {
 public:
  ExactProfileQualifier(const Engine* authoritative_engine,
                        const FreshWorkerRunner* runner)
      : authoritative_engine_(authoritative_engine), runner_(runner) {}

  // Runs N serial observations. With the concrete product-owned
  // FreshWorkerProcessRunner and worker Engine adapter, these are cold OS
  // processes; an injected FreshWorkerRunner is only a contract seam and must
  // never be surfaced as cold-process proof. Serial execution avoids concurrent
  // serving shape/batch effects and makes every concrete observation an
  // isolated batch-1 qualification attempt controlled by the worker
  // integration.
  // The returned evidence is not durable admission until PutIfAbsent succeeds;
  // consumers of ExactRegeneration must resolve admission through a repository.
  absl::StatusOr<ExactProfileAdmissionRecord> Qualify(
      const ExactProfileQualificationSpec& spec,
      const FreshWorkerAuthentication& authentication) const;

  absl::StatusOr<ExactProfileAdmissionRecord> QualifyAndAdmit(
      const ExactProfileQualificationSpec& spec,
      const FreshWorkerAuthentication& authentication,
      ExactProfileAdmissionRepository* repository) const;

 private:
  const Engine* authoritative_engine_;
  const FreshWorkerRunner* runner_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_EXACT_PROFILE_ADMISSION_H_
