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

#include "runtime/dpm/dpm_replay_executor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/canonical_replay_catalog.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/exact_profile_admission.h"
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kCanonicalRequestMagic = {'D', 'P', 'M', 'R',
                                                        'E', 'Q', '0', '1'};

bool IsZeroHash(const Hash256& hash) {
  uint8_t combined = 0;
  for (uint8_t byte : hash.bytes) combined |= byte;
  return combined == 0;
}

bool HasControlByte(absl::string_view bytes) {
  for (unsigned char byte : bytes) {
    if (byte < 0x20 || byte == 0x7f) return true;
  }
  return false;
}

void AppendU32(uint32_t value, std::string* output) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendU64(uint64_t value, std::string* output) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

class Reader {
 public:
  explicit Reader(absl::string_view bytes) : bytes_(bytes) {}

  absl::StatusOr<uint32_t> ReadU32() {
    if (remaining() < 4) return Truncated();
    uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      value = (value << 8) |
              static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 4;
    return value;
  }

  absl::StatusOr<uint64_t> ReadU64() {
    if (remaining() < 8) return Truncated();
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
      value = (value << 8) |
              static_cast<uint8_t>(bytes_[offset_ + index]);
    }
    offset_ += 8;
    return value;
  }

  absl::StatusOr<absl::string_view> ReadBytes(uint64_t size) {
    if (size > remaining() || size > std::string().max_size()) {
      return Truncated();
    }
    const absl::string_view result =
        bytes_.substr(offset_, static_cast<size_t>(size));
    offset_ += static_cast<size_t>(size);
    return result;
  }

  size_t remaining() const { return bytes_.size() - offset_; }

 private:
  absl::Status Truncated() const {
    return absl::DataLossError("Truncated canonical DPM replay request.");
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

absl::Status ValidateWinnerForKey(const CanonicalReplayWinner& winner,
                                  const CanonicalReplayKey& expected_key) {
  ABSL_RETURN_IF_ERROR(ValidateCanonicalReplayKey(winner.key));
  if (!(winner.key == expected_key)) {
    return absl::DataLossError(
        "Canonical replay catalog returned a winner for another request.");
  }
  if (winner.canonical_output.empty() ||
      winner.canonical_output.size() > kMaximumCanonicalReplayOutputBytes ||
      winner.canonical_output_hash != Sha256(winner.canonical_output) ||
      IsZeroHash(winner.execution_evidence_hash) ||
      winner.authentication_key_id.empty() ||
      IsZeroHash(winner.authentication_tag)) {
    return absl::DataLossError(
        "Canonical replay catalog returned an incomplete winner.");
  }
  return absl::OkStatus();
}

absl::Status ValidateExactExecutionRecord(
    const ExactProfileAdmissionRecord& record,
    const ExactLiteRtProfile& profile,
    const ExactProfileQualificationSpec& qualification_spec,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(
      ValidateExactProfileAdmissionRecordForProfile(record, profile));
  if (record.independent_run_count !=
          qualification_spec.independent_run_count ||
      record.qualification_request_hash !=
          ComputeExactProfileQualificationRequestHash(profile,
                                                      qualification_spec) ||
      record.request_payload_size !=
          qualification_spec.canonical_request_payload.size() ||
      record.request_payload_hash !=
          Sha256(qualification_spec.canonical_request_payload) ||
      record.authentication_key_id != authentication.key_id ||
      record.replay_isolation != FreshWorkerReplayIsolation::kEmptyCatalogs) {
    return absl::DataLossError(
        "Fresh exact execution evidence does not bind the resolved request "
        "and profile.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::Status ValidateDPMCanonicalReplayRequest(
    const DPMCanonicalReplayRequest& request) {
  if (request.format_version != kDPMCanonicalReplayRequestFormatVersion) {
    return absl::FailedPreconditionError(
        "Canonical DPM replay request version is unsupported.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayStage(request.stage));
  if (request.request_contract_version.empty() ||
      request.request_contract_version.size() >
          kMaximumCanonicalReplayContractBytes ||
      HasControlByte(request.request_contract_version)) {
    return absl::InvalidArgumentError(
        "Canonical DPM replay request contract is invalid.");
  }
  if (request.canonical_payload.empty() ||
      request.canonical_payload.size() >
          kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::InvalidArgumentError(
        "Canonical DPM replay request payload is empty or oversized.");
  }
  const uint64_t available_after_framing =
      kMaximumFreshWorkerRequestPayloadBytes -
      kDPMCanonicalReplayRequestFramingBytes;
  if (request.request_contract_version.size() > available_after_framing ||
      request.canonical_payload.size() >
          available_after_framing -
              request.request_contract_version.size()) {
    return absl::ResourceExhaustedError(
        "Canonical DPM replay request does not fit its worker payload "
        "envelope.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeDPMCanonicalReplayRequest(
    const DPMCanonicalReplayRequest& request) {
  ABSL_RETURN_IF_ERROR(ValidateDPMCanonicalReplayRequest(request));
  const uint64_t fixed_size = kDPMCanonicalReplayRequestFramingBytes;
  const uint64_t encoded_size =
      fixed_size + request.request_contract_version.size() +
      request.canonical_payload.size();
  if (encoded_size > kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Encoded canonical DPM replay request exceeds the worker limit.");
  }
  std::string encoded;
  encoded.reserve(static_cast<size_t>(encoded_size));
  encoded.append(kCanonicalRequestMagic.data(), kCanonicalRequestMagic.size());
  AppendU32(request.format_version, &encoded);
  AppendU32(static_cast<uint32_t>(request.stage), &encoded);
  AppendU32(static_cast<uint32_t>(request.request_contract_version.size()),
            &encoded);
  AppendU64(request.canonical_payload.size(), &encoded);
  encoded.append(request.request_contract_version);
  encoded.append(request.canonical_payload);
  return encoded;
}

absl::StatusOr<DPMCanonicalReplayRequest> DecodeDPMCanonicalReplayRequest(
    absl::string_view bytes) {
  if (bytes.size() < kDPMCanonicalReplayRequestFramingBytes ||
      bytes.size() > kMaximumFreshWorkerRequestPayloadBytes ||
      std::memcmp(bytes.data(), kCanonicalRequestMagic.data(),
                  kCanonicalRequestMagic.size()) != 0) {
    return absl::DataLossError(
        "Canonical DPM replay request framing is invalid.");
  }
  Reader reader(bytes.substr(kCanonicalRequestMagic.size()));
  DPMCanonicalReplayRequest request;
  ABSL_ASSIGN_OR_RETURN(request.format_version, reader.ReadU32());
  uint32_t stage;
  ABSL_ASSIGN_OR_RETURN(stage, reader.ReadU32());
  request.stage = static_cast<DPMReplayStage>(stage);
  uint32_t contract_size;
  ABSL_ASSIGN_OR_RETURN(contract_size, reader.ReadU32());
  uint64_t payload_size;
  ABSL_ASSIGN_OR_RETURN(payload_size, reader.ReadU64());
  if (contract_size > kMaximumCanonicalReplayContractBytes ||
      payload_size > kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Canonical DPM replay request declares an oversized field.");
  }
  absl::string_view contract;
  ABSL_ASSIGN_OR_RETURN(contract, reader.ReadBytes(contract_size));
  request.request_contract_version.assign(contract.data(), contract.size());
  absl::string_view payload;
  ABSL_ASSIGN_OR_RETURN(payload, reader.ReadBytes(payload_size));
  request.canonical_payload.assign(payload.data(), payload.size());
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "Canonical DPM replay request has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMCanonicalReplayRequest(request));
  ABSL_ASSIGN_OR_RETURN(const std::string canonical,
                        EncodeDPMCanonicalReplayRequest(request));
  if (canonical != bytes) {
    return absl::DataLossError(
        "Canonical DPM replay request is not canonically encoded.");
  }
  return request;
}

absl::StatusOr<Hash256> ComputeDPMCanonicalReplayRequestHash(
    const DPMCanonicalReplayRequest& request) {
  ABSL_ASSIGN_OR_RETURN(const std::string encoded,
                        EncodeDPMCanonicalReplayRequest(request));
  return Sha256(encoded);
}

absl::Status CanonicalWinnerReplayExecutor::ValidateSupport(
    CanonicalWinnerReplayGenerator* generator,
    DPMReplayStage expected_stage) const {
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayStage(expected_stage));
  if (generator == nullptr || catalog_ == nullptr) {
    return absl::InvalidArgumentError(
        "Canonical WinnerReplay requires a generator and catalog.");
  }
  return generator->ValidateSupport(expected_stage);
}

absl::StatusOr<CanonicalWinnerReplayExecution>
CanonicalWinnerReplayExecutor::Run(
    const DPMCanonicalReplayRequest& request,
    CanonicalWinnerReplayGenerator* generator) {
  ABSL_RETURN_IF_ERROR(ValidateDPMCanonicalReplayRequest(request));
  ABSL_RETURN_IF_ERROR(ValidateSupport(generator, request.stage));
  const SessionHandoffIdentity runtime_identity =
      generator->GetRuntimeIdentity();
  ABSL_ASSIGN_OR_RETURN(const Hash256 request_hash,
                        ComputeDPMCanonicalReplayRequestHash(request));
  CanonicalReplayKey key{
      .stage = request.stage,
      .runtime_identity = runtime_identity,
      .canonical_request_hash = request_hash,
      .request_contract_version = request.request_contract_version,
  };
  ABSL_RETURN_IF_ERROR(ValidateCanonicalReplayKey(key));

  absl::StatusOr<CanonicalReplayWinner> existing = catalog_->Get(key);
  if (existing.ok()) {
    ABSL_RETURN_IF_ERROR(ValidateWinnerForKey(*existing, key));
    ABSL_RETURN_IF_ERROR(generator->ValidateSupport(request.stage));
    if (generator->GetRuntimeIdentity() != runtime_identity) {
      return absl::FailedPreconditionError(
          "WinnerReplay runtime identity changed during catalog lookup.");
    }
    return CanonicalWinnerReplayExecution{
        .resolution = CanonicalWinnerResolution::kReplayed,
        .runtime_identity = runtime_identity,
        .canonical_request_hash = request_hash,
        .canonical_output = existing->canonical_output,
        .canonical_output_hash = existing->canonical_output_hash,
        .execution_evidence_hash = existing->execution_evidence_hash,
        .producing_session_matches_output = false,
    };
  }
  if (existing.status().code() != absl::StatusCode::kNotFound) {
    return existing.status();
  }

  ABSL_ASSIGN_OR_RETURN(CanonicalWinnerGeneratedCandidate candidate,
                        generator->Generate(request));
  if (candidate.canonical_output.empty() ||
      candidate.canonical_output.size() >
          kMaximumCanonicalReplayOutputBytes ||
      IsZeroHash(candidate.execution_evidence_hash)) {
    return absl::DataLossError(
        "Canonical WinnerReplay generator returned an invalid candidate.");
  }
  ABSL_RETURN_IF_ERROR(generator->ValidateSupport(request.stage));
  if (generator->GetRuntimeIdentity() != runtime_identity) {
    return absl::FailedPreconditionError(
        "WinnerReplay runtime identity changed during generation.");
  }
  CanonicalReplayCandidate catalog_candidate{
      .key = key,
      .canonical_output = candidate.canonical_output,
      .execution_evidence_hash = candidate.execution_evidence_hash,
  };
  ABSL_ASSIGN_OR_RETURN(PublishCanonicalReplayResult published,
                        catalog_->Publish(catalog_candidate));
  ABSL_RETURN_IF_ERROR(ValidateWinnerForKey(published.winner, key));
  ABSL_RETURN_IF_ERROR(generator->ValidateSupport(request.stage));
  if (generator->GetRuntimeIdentity() != runtime_identity) {
    return absl::FailedPreconditionError(
        "WinnerReplay runtime identity changed during publication.");
  }
  if (published.published &&
      (published.winner.canonical_output != candidate.canonical_output ||
       published.winner.execution_evidence_hash !=
           candidate.execution_evidence_hash)) {
    return absl::DataLossError(
        "Canonical replay catalog changed the publishing candidate.");
  }
  return CanonicalWinnerReplayExecution{
      .resolution = published.published
                        ? CanonicalWinnerResolution::kPublished
                        : CanonicalWinnerResolution::kLostPublishRace,
      .runtime_identity = runtime_identity,
      .canonical_request_hash = request_hash,
      .canonical_output = published.winner.canonical_output,
      .canonical_output_hash = published.winner.canonical_output_hash,
      .execution_evidence_hash = published.winner.execution_evidence_hash,
      .producing_session_matches_output = published.published,
  };
}

absl::StatusOr<ExactLiteRtProfile>
ExactRegenerationExecutor::ResolveCurrentProfile() const {
  if (engine_ == nullptr) {
    return absl::InvalidArgumentError(
        "ExactRegeneration requires a loaded authoritative Engine.");
  }
  return engine_->ResolveExactLiteRtProfile(config_.session_config,
                                            config_.profile_assertion);
}

absl::StatusOr<ExactLiteRtProfile>
ExactRegenerationExecutor::GetDerivedProfile() const {
  return ResolveCurrentProfile();
}

ExactProfileQualificationSpec
ExactRegenerationExecutor::MakeQualificationSpec(
    absl::string_view encoded_request) const {
  ExactProfileQualificationSpec spec;
  spec.session_config = config_.session_config;
  spec.profile_assertion = config_.profile_assertion;
  spec.canonical_request_payload.assign(encoded_request.data(),
                                        encoded_request.size());
  spec.independent_run_count = config_.independent_run_count;
  return spec;
}

absl::Status ExactRegenerationExecutor::ValidateSupport() const {
  if (worker_runner_ == nullptr || admission_repository_ == nullptr) {
    return absl::InvalidArgumentError(
        "ExactRegeneration requires a process runner and admission "
        "repository.");
  }
  if (config_.independent_run_count < 2 ||
      config_.independent_run_count > kMaximumFreshWorkerRuns) {
    return absl::InvalidArgumentError(
        "ExactRegeneration requires 2 to 64 cold runs per request.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerAuthentication(config_.authentication));
  ABSL_ASSIGN_OR_RETURN(const ExactLiteRtProfile profile,
                        ResolveCurrentProfile());
  ABSL_ASSIGN_OR_RETURN(
      const ExactProfileAdmissionRecord admission,
      admission_repository_->Get(profile, config_.authentication));
  ABSL_RETURN_IF_ERROR(
      ValidateExactProfileAdmissionRecordForProfile(admission, profile));
  if (admission.authentication_key_id != config_.authentication.key_id ||
      admission.replay_isolation != FreshWorkerReplayIsolation::kEmptyCatalogs) {
    return absl::FailedPreconditionError(
        "Authenticated admission does not match the Engine-derived exact "
        "profile.");
  }
  return absl::OkStatus();
}

absl::StatusOr<ExactProfileAdmissionRecord>
ExactRegenerationExecutor::AdmitProfile(
    const DPMCanonicalReplayRequest& qualification_request) const {
  ABSL_RETURN_IF_ERROR(
      ValidateDPMCanonicalReplayRequest(qualification_request));
  if (engine_ == nullptr || worker_runner_ == nullptr ||
      admission_repository_ == nullptr) {
    return absl::InvalidArgumentError(
        "Exact profile admission requires Engine, worker, and repository.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const std::string encoded_request,
      EncodeDPMCanonicalReplayRequest(qualification_request));
  ExactProfileQualifier qualifier(engine_, worker_runner_);
  return qualifier.QualifyAndAdmit(MakeQualificationSpec(encoded_request),
                                   config_.authentication,
                                   admission_repository_);
}

absl::StatusOr<ExactRegenerationExecution> ExactRegenerationExecutor::Run(
    const DPMCanonicalReplayRequest& request) const {
  ABSL_RETURN_IF_ERROR(ValidateDPMCanonicalReplayRequest(request));
  ABSL_RETURN_IF_ERROR(ValidateSupport());
  ABSL_ASSIGN_OR_RETURN(const ExactLiteRtProfile profile_before,
                        ResolveCurrentProfile());
  ABSL_ASSIGN_OR_RETURN(
      const ExactProfileAdmissionRecord admission,
      admission_repository_->Get(profile_before, config_.authentication));
  ABSL_ASSIGN_OR_RETURN(const std::string encoded_request,
                        EncodeDPMCanonicalReplayRequest(request));
  ABSL_ASSIGN_OR_RETURN(const Hash256 request_hash,
                        ComputeDPMCanonicalReplayRequestHash(request));
  const ExactProfileQualificationSpec qualification_spec =
      MakeQualificationSpec(encoded_request);
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecordForProfile(
      admission, profile_before));
  if (admission.authentication_key_id != config_.authentication.key_id ||
      admission.replay_isolation != FreshWorkerReplayIsolation::kEmptyCatalogs) {
    return absl::FailedPreconditionError(
        "Authenticated admission changed or no longer matches the "
        "Engine-derived exact profile.");
  }

  ExactProfileQualifier qualifier(engine_, worker_runner_);
  ABSL_ASSIGN_OR_RETURN(
      ExactProfileAdmissionRecord cold_run,
      qualifier.Qualify(qualification_spec, config_.authentication));
  ABSL_RETURN_IF_ERROR(ValidateExactExecutionRecord(
      cold_run, profile_before, qualification_spec, config_.authentication));
  ABSL_ASSIGN_OR_RETURN(const ExactLiteRtProfile profile_after,
                        ResolveCurrentProfile());
  if (profile_after != profile_before) {
    return absl::AbortedError(
        "Engine-derived exact profile changed during cold regeneration.");
  }
  return ExactRegenerationExecution{
      .derived_profile = profile_before,
      .canonical_request_hash = request_hash,
      .profile_admission_record_id = admission.record_id,
      .cold_run_evidence = cold_run,
      .canonical_output = cold_run.canonical_output,
      .token_bytes = cold_run.token_bytes,
      .logit_frames = cold_run.logit_frames,
  };
}

}  // namespace litert::lm
