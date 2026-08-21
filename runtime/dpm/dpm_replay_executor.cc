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
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/engine_fresh_worker_contract.h"
#include "runtime/dpm/exact_profile_admission.h"
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kCanonicalRequestMagic = {'D', 'P', 'M', 'R',
                                                        'E', 'Q', '0', '2'};

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
  if (request.max_output_tokens == 0 ||
      request.max_output_tokens > kMaximumDPMGenerationTokens) {
    return absl::InvalidArgumentError(
        "Canonical DPM replay request token limit is outside the product "
        "bound.");
  }
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
  AppendU32(request.max_output_tokens, &encoded);
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
  ABSL_ASSIGN_OR_RETURN(request.max_output_tokens, reader.ReadU32());
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

namespace {

struct ResolvedExactExecutorProfile {
  SessionConfig session_config;
  ExactLiteRtProfile profile;
};

absl::Status ValidateExactExecutorConfig(
    const Engine* engine,
    const ExactProfileAdmissionRepository* admission_repository,
    const ExactRegenerationExecutorConfig& config) {
  if (engine == nullptr || admission_repository == nullptr) {
    return absl::InvalidArgumentError(
        "ExactRegeneration requires a loaded Engine and admission "
        "repository.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayStage(config.stage));
  if (config.independent_run_count < 2 ||
      config.independent_run_count > kMaximumFreshWorkerRuns) {
    return absl::InvalidArgumentError(
        "ExactRegeneration requires 2 to 64 cold runs per request.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerAuthentication(config.authentication));
  if (config.worker_process.executable_path.empty() ||
      !config.worker_process.executable_path.is_absolute()) {
    return absl::InvalidArgumentError(
        "ExactRegeneration requires an absolute concrete worker executable "
        "path.");
  }
  ABSL_ASSIGN_OR_RETURN(
      SessionConfig ignored,
      MakeEngineFreshWorkerSessionConfig(config.stage,
                                         config.max_output_tokens));
  (void)ignored;
  return absl::OkStatus();
}

absl::StatusOr<ResolvedExactExecutorProfile> ResolveExactExecutorProfile(
    const Engine* engine, const ExactRegenerationExecutorConfig& config) {
  ABSL_ASSIGN_OR_RETURN(
      SessionConfig session_config,
      MakeEngineFreshWorkerSessionConfig(config.stage,
                                         config.max_output_tokens));
  ABSL_RETURN_IF_ERROR(
      session_config.MaybeUpdateAndValidate(engine->GetEngineSettings()));
  ABSL_ASSIGN_OR_RETURN(
      ExactLiteRtProfile profile,
      engine->ResolveExactLiteRtProfile(session_config,
                                        config.profile_assertion));
  return ResolvedExactExecutorProfile{
      .session_config = std::move(session_config),
      .profile = std::move(profile),
  };
}

absl::Status ValidateAdmissionForExactProfile(
    const ExactProfileAdmissionRecord& admission,
    const ExactLiteRtProfile& profile,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(
      ValidateExactProfileAdmissionRecordForProfile(admission, profile));
  if (admission.authentication_key_id != authentication.key_id ||
      admission.replay_isolation !=
          FreshWorkerReplayIsolation::kEmptyCatalogs) {
    return absl::FailedPreconditionError(
        "Authenticated admission does not match the Engine-derived exact "
        "profile and empty-catalog contract.");
  }
  return absl::OkStatus();
}

absl::Status ValidateRequestForExactExecutorConfig(
    const DPMCanonicalReplayRequest& request,
    const ExactRegenerationExecutorConfig& config) {
  ABSL_RETURN_IF_ERROR(ValidateDPMCanonicalReplayRequest(request));
  if (request.stage != config.stage ||
      request.max_output_tokens != config.max_output_tokens) {
    return absl::FailedPreconditionError(
        "Canonical replay request stage or token limit differs from the "
        "immutable exact-worker profile.");
  }
  return absl::OkStatus();
}

ExactProfileQualificationSpec MakeExactQualificationSpec(
    const SessionConfig& session_config,
    const ExactRegenerationExecutorConfig& config,
    absl::string_view encoded_request) {
  ExactProfileQualificationSpec spec;
  spec.session_config = session_config;
  spec.profile_assertion = config.profile_assertion;
  spec.canonical_request_payload.assign(encoded_request.data(),
                                        encoded_request.size());
  spec.independent_run_count = config.independent_run_count;
  return spec;
}

}  // namespace

absl::StatusOr<ExactProfileAdmissionRecord>
ExactRegenerationExecutor::QualifyAndAdmit(
    const Engine* engine,
    ExactProfileAdmissionRepository* admission_repository,
    const ExactRegenerationExecutorConfig& config,
    const DPMCanonicalReplayRequest& qualification_request) {
  ABSL_RETURN_IF_ERROR(
      ValidateExactExecutorConfig(engine, admission_repository, config));
  ABSL_RETURN_IF_ERROR(
      ValidateRequestForExactExecutorConfig(qualification_request, config));
  ABSL_ASSIGN_OR_RETURN(ResolvedExactExecutorProfile resolved,
                        ResolveExactExecutorProfile(engine, config));
  ABSL_ASSIGN_OR_RETURN(
      const std::string encoded_request,
      EncodeDPMCanonicalReplayRequest(qualification_request));
  FreshWorkerProcessRunner worker_runner(config.worker_process);
  ExactProfileQualifier qualifier(engine, &worker_runner);
  ABSL_ASSIGN_OR_RETURN(
      ExactProfileAdmissionRecord admission,
      qualifier.QualifyAndAdmit(
          MakeExactQualificationSpec(resolved.session_config, config,
                                     encoded_request),
          config.authentication, admission_repository));
  ABSL_RETURN_IF_ERROR(ValidateAdmissionForExactProfile(
      admission, resolved.profile, config.authentication));
  return admission;
}

absl::StatusOr<std::unique_ptr<ExactRegenerationExecutor>>
ExactRegenerationExecutor::Create(
    const Engine* engine,
    ExactProfileAdmissionRepository* admission_repository,
    ExactRegenerationExecutorConfig config) {
  ABSL_RETURN_IF_ERROR(
      ValidateExactExecutorConfig(engine, admission_repository, config));
  ABSL_ASSIGN_OR_RETURN(ResolvedExactExecutorProfile resolved,
                        ResolveExactExecutorProfile(engine, config));
  ABSL_ASSIGN_OR_RETURN(
      const ExactProfileAdmissionRecord admission,
      admission_repository->Get(resolved.profile, config.authentication));
  ABSL_RETURN_IF_ERROR(ValidateAdmissionForExactProfile(
      admission, resolved.profile, config.authentication));
  return std::unique_ptr<ExactRegenerationExecutor>(
      new ExactRegenerationExecutor(
          engine, admission_repository, std::move(config),
          std::move(resolved.session_config), std::move(resolved.profile)));
}

absl::Status ExactRegenerationExecutor::ValidateBoundRequest(
    const DPMCanonicalReplayRequest& request) const {
  return ValidateRequestForExactExecutorConfig(request, config_);
}

absl::StatusOr<ExactLiteRtProfile>
ExactRegenerationExecutor::ResolveCurrentProfile() const {
  if (engine_ == nullptr) {
    return absl::InvalidArgumentError(
        "ExactRegeneration requires a loaded authoritative Engine.");
  }
  ABSL_ASSIGN_OR_RETURN(
      ExactLiteRtProfile current,
      engine_->ResolveExactLiteRtProfile(resolved_session_config_,
                                         config_.profile_assertion));
  if (current != derived_profile_) {
    return absl::FailedPreconditionError(
        "Engine-derived exact profile changed after executor construction.");
  }
  return current;
}

absl::StatusOr<ExactLiteRtProfile>
ExactRegenerationExecutor::GetDerivedProfile() const {
  return ResolveCurrentProfile();
}

ExactProfileQualificationSpec
ExactRegenerationExecutor::MakeQualificationSpec(
    absl::string_view encoded_request) const {
  return MakeExactQualificationSpec(resolved_session_config_, config_,
                                    encoded_request);
}

absl::Status ExactRegenerationExecutor::ValidateSupport() const {
  if (engine_ == nullptr || admission_repository_ == nullptr) {
    return absl::InvalidArgumentError(
        "ExactRegeneration requires a loaded Engine and admission "
        "repository.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateExactExecutorConfig(engine_, admission_repository_, config_));
  ABSL_ASSIGN_OR_RETURN(const ExactLiteRtProfile profile,
                        ResolveCurrentProfile());
  ABSL_ASSIGN_OR_RETURN(
      const ExactProfileAdmissionRecord admission,
      admission_repository_->Get(profile, config_.authentication));
  return ValidateAdmissionForExactProfile(admission, profile,
                                          config_.authentication);
}

absl::StatusOr<ExactRegenerationExecution> ExactRegenerationExecutor::Run(
    const DPMCanonicalReplayRequest& request) const {
  // Reject cross-stage and cross-limit requests before ValidateSupport can
  // touch the repository and, critically, before any worker process is born.
  ABSL_RETURN_IF_ERROR(ValidateBoundRequest(request));
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
  ABSL_RETURN_IF_ERROR(ValidateAdmissionForExactProfile(
      admission, profile_before, config_.authentication));

  ExactProfileQualifier qualifier(engine_, &worker_runner_);
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
