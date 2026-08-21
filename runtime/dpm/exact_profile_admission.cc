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

#include "runtime/dpm/exact_profile_admission.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "runtime/platform/hash/hmac_sha256.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kAdmissionMagic = {'D', 'P', 'M', 'A', 'D', 'M',
                                                  '0', '1'};
constexpr uint32_t kAdmissionEnvelopeVersion = 1;
constexpr uint64_t kMaximumAdmissionEnvelopeBytes =
    kMaximumFreshWorkerEnvelopeBytes;
constexpr uint64_t kAdmissionEnvelopeFixedBytes = 8 + 4 + 4 + 8 + 32;
constexpr uint64_t kMaximumAdmissionKeyIdBytes = 256;
constexpr absl::string_view kQualificationRequestDomain =
    "LITERT_LMX_EXACT_PROFILE_QUALIFICATION_REQUEST_SHA256_V1";
constexpr absl::string_view kOutputEvidenceDomain =
    "LITERT_LMX_FRESH_WORKER_OUTPUT_EVIDENCE_SHA256_V1";
constexpr absl::string_view kAdmissionRecordDomain =
    "LITERT_LMX_EXACT_PROFILE_ADMISSION_RECORD_SHA256_V1";
constexpr absl::string_view kAdmissionMacDomain =
    "LITERT_LMX_EXACT_PROFILE_ADMISSION_HMAC_SHA256_V1";

bool IsZeroHash(const Hash256& hash) {
  uint8_t combined = 0;
  for (uint8_t byte : hash.bytes) combined |= byte;
  return combined == 0;
}

bool HasControlByte(absl::string_view text) {
  for (unsigned char byte : text) {
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

void AppendHash(const Hash256& hash, std::string* output) {
  output->append(reinterpret_cast<const char*>(hash.bytes.data()),
                 hash.bytes.size());
}

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

void HashU32(uint32_t value, Sha256Hasher* hasher) {
  std::string bytes;
  AppendU32(value, &bytes);
  hasher->Update(bytes);
}

void HashU64(uint64_t value, Sha256Hasher* hasher) {
  std::string bytes;
  AppendU64(value, &bytes);
  hasher->Update(bytes);
}

void HashValue(const Hash256& hash, Sha256Hasher* hasher) {
  hasher->Update(absl::string_view(
      reinterpret_cast<const char*>(hash.bytes.data()), hash.bytes.size()));
}

void HashFrame(absl::string_view bytes, Sha256Hasher* hasher) {
  HashU64(bytes.size(), hasher);
  hasher->Update(bytes);
}

Hash256 ComputeQualificationRequestHashFromDigest(
    const Hash256& exact_profile_hash, uint32_t run_count,
    const Hash256& request_payload_hash, uint64_t request_payload_size) {
  Sha256Hasher hasher;
  hasher.Update(kQualificationRequestDomain);
  HashU32(ExactProfileAdmissionRecord::kFormatVersion, &hasher);
  HashValue(exact_profile_hash, &hasher);
  HashU32(run_count, &hasher);
  HashU64(request_payload_size, &hasher);
  HashValue(request_payload_hash, &hasher);
  return hasher.Finalize();
}

std::string EncodeRecordFields(const ExactProfileAdmissionRecord& record) {
  std::string body;
  body.reserve(4 + 32 * 8 + 4 * 4 + 8 * 3 +
               record.authentication_key_id.size() +
               record.runs.size() * (4 + 8 + 32 * 6) +
               record.token_bytes.size() +
               record.logit_frames.size() * (5 * 4 + 8 + 32));
  AppendU32(record.format_version, &body);
  AppendU32(static_cast<uint32_t>(record.evidence_kind), &body);
  AppendU32(static_cast<uint32_t>(record.replay_isolation), &body);
  AppendHash(record.exact_profile_hash, &body);
  AppendHash(record.qualification_id, &body);
  AppendHash(record.qualification_request_hash, &body);
  AppendHash(record.request_payload_hash, &body);
  AppendU64(record.request_payload_size, &body);
  AppendU32(record.independent_run_count, &body);
  AppendU64(static_cast<uint64_t>(record.qualified_unix_micros), &body);
  AppendU32(static_cast<uint32_t>(record.authentication_key_id.size()), &body);
  body.append(record.authentication_key_id);
  AppendU32(static_cast<uint32_t>(record.runs.size()), &body);
  for (const ExactProfileAdmissionRunEvidence& run : record.runs) {
    AppendU32(run.run_index, &body);
    AppendU64(static_cast<uint64_t>(run.process_id), &body);
    AppendHash(run.challenge_nonce, &body);
    AppendHash(run.worker_instance_nonce, &body);
    AppendHash(run.request_envelope_hash, &body);
    AppendHash(run.result_envelope_hash, &body);
    AppendHash(run.launch_spec_hash, &body);
    AppendHash(run.output_evidence_hash, &body);
  }
  AppendU64(record.canonical_output.size(), &body);
  body.append(record.canonical_output);
  AppendHash(record.canonical_output_hash, &body);
  AppendU64(record.token_bytes.size(), &body);
  body.append(record.token_bytes);
  AppendHash(record.token_bytes_hash, &body);
  AppendU32(static_cast<uint32_t>(record.logit_frames.size()), &body);
  for (const FreshWorkerLogitFrameEvidence& frame : record.logit_frames) {
    AppendU32(static_cast<uint32_t>(frame.element_type), &body);
    AppendU32(frame.element_byte_width, &body);
    AppendU32(frame.batch_size, &body);
    AppendU32(frame.sequence_size, &body);
    AppendU32(frame.vocabulary_size, &body);
    AppendU64(frame.byte_count, &body);
    AppendHash(frame.sha256, &body);
  }
  return body;
}

absl::Status ValidateAdmissionRecordFields(
    const ExactProfileAdmissionRecord& record, bool validate_record_id) {
  if (record.format_version != ExactProfileAdmissionRecord::kFormatVersion ||
      record.evidence_kind !=
          ExactProfileAdmissionEvidenceKind::kIndependentColdRunEquality ||
      record.replay_isolation != FreshWorkerReplayIsolation::kEmptyCatalogs) {
    return absl::FailedPreconditionError(
        "Exact-profile admission record version or evidence kind is unsupported.");
  }
  if (IsZeroHash(record.exact_profile_hash) ||
      IsZeroHash(record.qualification_id) ||
      IsZeroHash(record.qualification_request_hash) ||
      IsZeroHash(record.request_payload_hash)) {
    return absl::InvalidArgumentError(
        "Exact-profile admission record identity is incomplete.");
  }
  if (record.request_payload_size >
          kMaximumFreshWorkerRequestPayloadBytes ||
      record.independent_run_count < 2 ||
      record.independent_run_count > kMaximumFreshWorkerRuns ||
      record.runs.size() != record.independent_run_count ||
      record.qualified_unix_micros <= 0) {
    return absl::InvalidArgumentError(
        "Exact-profile admission record run metadata is invalid.");
  }
  if (record.authentication_key_id.empty() ||
      record.authentication_key_id.size() > kMaximumAdmissionKeyIdBytes ||
      HasControlByte(record.authentication_key_id)) {
    return absl::InvalidArgumentError(
        "Exact-profile admission authentication key ID is invalid.");
  }
  const Hash256 expected_request_hash =
      ComputeQualificationRequestHashFromDigest(
          record.exact_profile_hash, record.independent_run_count,
          record.request_payload_hash, record.request_payload_size);
  if (record.qualification_request_hash != expected_request_hash) {
    return absl::DataLossError(
        "Exact-profile qualification request hash is inconsistent.");
  }

  FreshWorkerExecutionOutput output{.canonical_output = record.canonical_output,
                                    .token_bytes = record.token_bytes,
                                    .logit_frames = record.logit_frames};
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerExecutionOutput(output));
  if (record.canonical_output_hash != Sha256(record.canonical_output)) {
    return absl::DataLossError(
        "Exact-profile admission canonical output hash is inconsistent.");
  }
  if (record.token_bytes_hash != Sha256(record.token_bytes)) {
    return absl::DataLossError(
        "Exact-profile admission token bytes hash is inconsistent.");
  }
  const Hash256 expected_output_hash = ComputeFreshWorkerOutputEvidenceHash(
      record.canonical_output, record.token_bytes, record.logit_frames);

  std::set<int64_t> process_ids;
  std::set<Hash256> challenges;
  std::set<Hash256> worker_nonces;
  std::set<Hash256> request_envelopes;
  std::set<Hash256> result_envelopes;
  Hash256 launch_spec_hash;
  for (uint32_t index = 0; index < record.runs.size(); ++index) {
    const ExactProfileAdmissionRunEvidence& run = record.runs[index];
    if (run.run_index != index || run.process_id <= 0 ||
        IsZeroHash(run.challenge_nonce) ||
        IsZeroHash(run.worker_instance_nonce) ||
        IsZeroHash(run.request_envelope_hash) ||
        IsZeroHash(run.result_envelope_hash) ||
        IsZeroHash(run.launch_spec_hash) ||
        run.output_evidence_hash != expected_output_hash) {
      return absl::DataLossError(
          "Exact-profile admission run evidence is incomplete.");
    }
    if (!process_ids.insert(run.process_id).second ||
        !challenges.insert(run.challenge_nonce).second ||
        !worker_nonces.insert(run.worker_instance_nonce).second ||
        !request_envelopes.insert(run.request_envelope_hash).second ||
        !result_envelopes.insert(run.result_envelope_hash).second) {
      return absl::FailedPreconditionError(
          "Exact-profile admission reused cold-process evidence.");
    }
    if (index == 0) {
      launch_spec_hash = run.launch_spec_hash;
    } else if (run.launch_spec_hash != launch_spec_hash) {
      return absl::FailedPreconditionError(
          "Exact-profile admission mixed worker launch specifications.");
    }
  }

  if (validate_record_id) {
    if (IsZeroHash(record.record_id)) {
      return absl::DataLossError(
          "Exact-profile admission record ID is absent.");
    }
    ABSL_ASSIGN_OR_RETURN(const Hash256 expected_id,
                          ComputeExactProfileAdmissionRecordId(record));
    if (record.record_id != expected_id) {
      return absl::DataLossError(
          "Exact-profile admission record ID is inconsistent.");
    }
  }
  return absl::OkStatus();
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

  absl::StatusOr<Hash256> ReadHash() {
    if (remaining() < 32) return Truncated();
    Hash256 hash;
    std::memcpy(hash.bytes.data(), bytes_.data() + offset_, hash.bytes.size());
    offset_ += hash.bytes.size();
    return hash;
  }

  absl::StatusOr<absl::string_view> ReadBytes(uint64_t size) {
    if (size > remaining()) return Truncated();
    const size_t safe_size = static_cast<size_t>(size);
    const absl::string_view result = bytes_.substr(offset_, safe_size);
    offset_ += safe_size;
    return result;
  }

  size_t remaining() const { return bytes_.size() - offset_; }

 private:
  absl::Status Truncated() const {
    return absl::DataLossError(
        "Truncated exact-profile admission record.");
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

absl::StatusOr<ExactProfileAdmissionRecord> DecodeRecordFields(
    absl::string_view body, const Hash256& record_id,
    const std::string& key_id) {
  Reader reader(body);
  ExactProfileAdmissionRecord record;
  record.record_id = record_id;
  ABSL_ASSIGN_OR_RETURN(record.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t evidence_kind, reader.ReadU32());
  record.evidence_kind =
      static_cast<ExactProfileAdmissionEvidenceKind>(evidence_kind);
  ABSL_ASSIGN_OR_RETURN(const uint32_t replay_isolation, reader.ReadU32());
  record.replay_isolation =
      static_cast<FreshWorkerReplayIsolation>(replay_isolation);
  ABSL_ASSIGN_OR_RETURN(record.exact_profile_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(record.qualification_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(record.qualification_request_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(record.request_payload_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(record.request_payload_size, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(record.independent_run_count, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint64_t encoded_time, reader.ReadU64());
  if (encoded_time >
      static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    return absl::DataLossError(
        "Exact-profile admission timestamp is invalid.");
  }
  record.qualified_unix_micros = static_cast<int64_t>(encoded_time);
  ABSL_ASSIGN_OR_RETURN(const uint32_t encoded_key_id_size, reader.ReadU32());
  if (encoded_key_id_size > kMaximumAdmissionKeyIdBytes) {
    return absl::ResourceExhaustedError(
        "Exact-profile admission key ID exceeds its limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view encoded_key_id,
                        reader.ReadBytes(encoded_key_id_size));
  if (encoded_key_id != key_id) {
    return absl::UnauthenticatedError(
        "Exact-profile admission key ID binding does not match.");
  }
  record.authentication_key_id = key_id;
  ABSL_ASSIGN_OR_RETURN(const uint32_t run_count, reader.ReadU32());
  if (run_count > kMaximumFreshWorkerRuns) {
    return absl::ResourceExhaustedError(
        "Exact-profile admission run evidence exceeds its limit.");
  }
  record.runs.reserve(run_count);
  for (uint32_t index = 0; index < run_count; ++index) {
    ExactProfileAdmissionRunEvidence run;
    ABSL_ASSIGN_OR_RETURN(run.run_index, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(const uint64_t encoded_pid, reader.ReadU64());
    if (encoded_pid >
        static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
      return absl::DataLossError(
          "Exact-profile admission process ID is invalid.");
    }
    run.process_id = static_cast<int64_t>(encoded_pid);
    ABSL_ASSIGN_OR_RETURN(run.challenge_nonce, reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(run.worker_instance_nonce, reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(run.request_envelope_hash, reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(run.result_envelope_hash, reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(run.launch_spec_hash, reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(run.output_evidence_hash, reader.ReadHash());
    record.runs.push_back(run);
  }
  ABSL_ASSIGN_OR_RETURN(const uint64_t canonical_output_size,
                        reader.ReadU64());
  if (canonical_output_size > kMaximumFreshWorkerCanonicalOutputBytes) {
    return absl::ResourceExhaustedError(
        "Exact-profile admission canonical output exceeds its limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view canonical_output,
                        reader.ReadBytes(canonical_output_size));
  record.canonical_output.assign(canonical_output.data(),
                                 canonical_output.size());
  ABSL_ASSIGN_OR_RETURN(record.canonical_output_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(const uint64_t token_size, reader.ReadU64());
  if (token_size > kMaximumFreshWorkerTokenBytes) {
    return absl::ResourceExhaustedError(
        "Exact-profile admission token bytes exceed their limit.");
  }
  ABSL_ASSIGN_OR_RETURN(const absl::string_view token_bytes,
                        reader.ReadBytes(token_size));
  record.token_bytes.assign(token_bytes.data(), token_bytes.size());
  ABSL_ASSIGN_OR_RETURN(record.token_bytes_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(const uint32_t frame_count, reader.ReadU32());
  if (frame_count > kMaximumFreshWorkerLogitFrames) {
    return absl::ResourceExhaustedError(
        "Exact-profile admission logits evidence exceeds its limit.");
  }
  record.logit_frames.reserve(frame_count);
  for (uint32_t index = 0; index < frame_count; ++index) {
    FreshWorkerLogitFrameEvidence frame;
    ABSL_ASSIGN_OR_RETURN(const uint32_t element_type, reader.ReadU32());
    frame.element_type =
        static_cast<FreshWorkerLogitElementType>(element_type);
    ABSL_ASSIGN_OR_RETURN(frame.element_byte_width, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(frame.batch_size, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(frame.sequence_size, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(frame.vocabulary_size, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(frame.byte_count, reader.ReadU64());
    ABSL_ASSIGN_OR_RETURN(frame.sha256, reader.ReadHash());
    record.logit_frames.push_back(frame);
  }
  if (reader.remaining() != 0) {
    return absl::DataLossError(
        "Exact-profile admission record has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecord(record));
  return record;
}

absl::Status ValidateQualificationSpec(
    const ExactProfileQualificationSpec& spec) {
  if (!spec.session_config.has_value()) {
    return absl::InvalidArgumentError(
        "Exact-profile qualification requires session settings for Engine resolution.");
  }
  if (spec.independent_run_count < 2 ||
      spec.independent_run_count > kMaximumFreshWorkerRuns) {
    return absl::InvalidArgumentError(
        "Exact-profile qualification requires 2 to 64 independent runs.");
  }
  if (spec.canonical_request_payload.empty()) {
    return absl::InvalidArgumentError(
        "Exact-profile qualification request payload is empty.");
  }
  if (spec.canonical_request_payload.size() >
      kMaximumFreshWorkerRequestPayloadBytes) {
    return absl::ResourceExhaustedError(
        "Exact-profile qualification request exceeds its limit.");
  }
  return absl::OkStatus();
}

absl::Status ValidateDerivedProfileForQualification(
    const ExactLiteRtProfile& derived_profile) {
  if (IsZeroHash(derived_profile.profile_id) ||
      derived_profile.qualification_requirement !=
          ExactLiteRtQualificationRequirement::
              kIndependentColdProcessesTokensAndLogits ||
      derived_profile.sampler_identity !=
          ExactLiteRtSamplerIdentity::kCpuGreedyArgmaxMinIndex ||
      derived_profile.batch_size != 1 ||
      derived_profile.logits_frame.batch_size != 1 ||
      derived_profile.logits_frame.sequence_size != 1 ||
      derived_profile.logits_frame.vocabulary_size == 0 ||
      derived_profile.logits_frame.byte_count == 0) {
    return absl::FailedPreconditionError(
        "The loaded Engine did not derive an admissible batch-one exact profile.");
  }
  uint64_t element_byte_width = 0;
  switch (derived_profile.logits_frame.element_type) {
    case ExactLiteRtLogitsElementType::kFloat16:
      element_byte_width = 2;
      break;
    case ExactLiteRtLogitsElementType::kFloat32:
      element_byte_width = 4;
      break;
    case ExactLiteRtLogitsElementType::kUnsupported:
      return absl::FailedPreconditionError(
          "The loaded Engine did not derive a supported logits element type.");
    default:
      return absl::FailedPreconditionError(
          "The loaded Engine derived an unknown logits element type.");
  }
  if (derived_profile.logits_frame.vocabulary_size >
          std::numeric_limits<uint64_t>::max() / element_byte_width ||
      derived_profile.logits_frame.byte_count !=
          static_cast<uint64_t>(derived_profile.logits_frame.vocabulary_size) *
              element_byte_width) {
    return absl::FailedPreconditionError(
        "The loaded Engine derived an inconsistent packed logits extent.");
  }
  return absl::OkStatus();
}

absl::Status ValidateLogitFramesMatchDerivedProfile(
    const ExactLiteRtProfile& derived_profile,
    const std::vector<FreshWorkerLogitFrameEvidence>& frames) {
  FreshWorkerLogitElementType expected_element_type;
  uint32_t expected_element_byte_width = 0;
  switch (derived_profile.logits_frame.element_type) {
    case ExactLiteRtLogitsElementType::kFloat16:
      expected_element_type = FreshWorkerLogitElementType::kFloat16;
      expected_element_byte_width = 2;
      break;
    case ExactLiteRtLogitsElementType::kFloat32:
      expected_element_type = FreshWorkerLogitElementType::kFloat32;
      expected_element_byte_width = 4;
      break;
    case ExactLiteRtLogitsElementType::kUnsupported:
      return absl::FailedPreconditionError(
          "The Engine-derived logits contract is unsupported.");
    default:
      return absl::FailedPreconditionError(
          "The Engine-derived logits contract has an unknown element type.");
  }
  if (frames.empty()) {
    return absl::FailedPreconditionError(
        "Exact-profile evidence contains no logits frames.");
  }
  for (const FreshWorkerLogitFrameEvidence& frame : frames) {
    ABSL_RETURN_IF_ERROR(ValidateFreshWorkerLogitFrameEvidence(frame));
    if (frame.element_type != expected_element_type ||
        frame.element_byte_width != expected_element_byte_width ||
        frame.batch_size != derived_profile.logits_frame.batch_size ||
        frame.sequence_size != derived_profile.logits_frame.sequence_size ||
        frame.vocabulary_size !=
            derived_profile.logits_frame.vocabulary_size ||
        frame.byte_count != derived_profile.logits_frame.byte_count) {
      return absl::FailedPreconditionError(
          "Fresh-worker logits evidence does not match the Engine-derived exact profile.");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateAdmissionMatchesSpec(
    const ExactProfileAdmissionRecord& record,
    const ExactLiteRtProfile& derived_profile,
    const ExactProfileQualificationSpec& spec,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecordForProfile(
      record, derived_profile));
  if (record.exact_profile_hash != derived_profile.profile_id ||
      record.independent_run_count != spec.independent_run_count ||
      record.qualification_request_hash !=
          ComputeExactProfileQualificationRequestHash(derived_profile, spec) ||
      record.request_payload_hash != Sha256(spec.canonical_request_payload) ||
      record.request_payload_size != spec.canonical_request_payload.size() ||
      record.authentication_key_id != authentication.key_id) {
    return absl::AlreadyExistsError(
        "The exact profile already has admission for a different qualification request.");
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> GenerateUniqueNonce(const std::set<Hash256>& used) {
  for (int attempt = 0; attempt < 16; ++attempt) {
    ABSL_ASSIGN_OR_RETURN(const Hash256 nonce, GenerateFreshWorkerNonce());
    if (used.find(nonce) == used.end()) return nonce;
  }
  return absl::InternalError(
      "The OS random source repeatedly returned a reused nonce.");
}

}  // namespace

Hash256 ComputeExactProfileQualificationRequestHash(
    const ExactLiteRtProfile& derived_profile,
    const ExactProfileQualificationSpec& spec) {
  return ComputeQualificationRequestHashFromDigest(
      derived_profile.profile_id, spec.independent_run_count,
      Sha256(spec.canonical_request_payload),
      spec.canonical_request_payload.size());
}

Hash256 ComputeFreshWorkerOutputEvidenceHash(
    absl::string_view canonical_output,
    absl::string_view token_bytes,
    const std::vector<FreshWorkerLogitFrameEvidence>& logit_frames) {
  Sha256Hasher hasher;
  hasher.Update(kOutputEvidenceDomain);
  HashFrame(canonical_output, &hasher);
  HashFrame(token_bytes, &hasher);
  HashU32(static_cast<uint32_t>(logit_frames.size()), &hasher);
  for (const FreshWorkerLogitFrameEvidence& frame : logit_frames) {
    HashU32(static_cast<uint32_t>(frame.element_type), &hasher);
    HashU32(frame.element_byte_width, &hasher);
    HashU32(frame.batch_size, &hasher);
    HashU32(frame.sequence_size, &hasher);
    HashU32(frame.vocabulary_size, &hasher);
    HashU64(frame.byte_count, &hasher);
    HashValue(frame.sha256, &hasher);
  }
  return hasher.Finalize();
}

absl::StatusOr<Hash256> ComputeExactProfileAdmissionRecordId(
    const ExactProfileAdmissionRecord& record) {
  ABSL_RETURN_IF_ERROR(ValidateAdmissionRecordFields(record, false));
  Sha256Hasher hasher;
  hasher.Update(kAdmissionRecordDomain);
  const std::string fields = EncodeRecordFields(record);
  hasher.Update(fields);
  return hasher.Finalize();
}

absl::Status ValidateExactProfileAdmissionRecord(
    const ExactProfileAdmissionRecord& record) {
  return ValidateAdmissionRecordFields(record, true);
}

absl::Status ValidateExactProfileAdmissionRecordForProfile(
    const ExactProfileAdmissionRecord& record,
    const ExactLiteRtProfile& runtime_derived_profile) {
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecord(record));
  ABSL_RETURN_IF_ERROR(
      ValidateDerivedProfileForQualification(runtime_derived_profile));
  if (record.exact_profile_hash != runtime_derived_profile.profile_id) {
    return absl::FailedPreconditionError(
        "Exact-profile admission does not match the Engine-derived profile.");
  }
  return ValidateLogitFramesMatchDerivedProfile(runtime_derived_profile,
                                                 record.logit_frames);
}

absl::StatusOr<std::string> EncodeExactProfileAdmissionRecord(
    const ExactProfileAdmissionRecord& record,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecord(record));
  if (record.authentication_key_id != authentication.key_id) {
    return absl::InvalidArgumentError(
        "Exact-profile admission record key ID does not match authentication.");
  }
  std::string body;
  AppendHash(record.record_id, &body);
  body.append(EncodeRecordFields(record));
  if (body.size() > kMaximumAdmissionEnvelopeBytes -
                        kAdmissionEnvelopeFixedBytes -
                        authentication.key_id.size()) {
    return absl::ResourceExhaustedError(
        "Exact-profile admission record exceeds its storage limit.");
  }
  std::string envelope;
  envelope.reserve(kAdmissionEnvelopeFixedBytes + authentication.key_id.size() +
                   body.size());
  envelope.append(kAdmissionMagic.data(), kAdmissionMagic.size());
  AppendU32(kAdmissionEnvelopeVersion, &envelope);
  AppendU32(static_cast<uint32_t>(authentication.key_id.size()), &envelope);
  AppendU64(body.size(), &envelope);
  envelope.append(authentication.key_id);
  envelope.append(body);
  const Hash256 mac = HmacSha256(authentication.authentication_key,
                                 {kAdmissionMacDomain, envelope});
  AppendHash(mac, &envelope);
  return envelope;
}

absl::StatusOr<ExactProfileAdmissionRecord>
DecodeExactProfileAdmissionRecord(
    absl::string_view envelope,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (envelope.size() < kAdmissionEnvelopeFixedBytes ||
      envelope.size() > kMaximumAdmissionEnvelopeBytes ||
      std::memcmp(envelope.data(), kAdmissionMagic.data(),
                  kAdmissionMagic.size()) != 0) {
    return absl::DataLossError(
        "Exact-profile admission envelope framing is invalid.");
  }
  Reader header(envelope.substr(kAdmissionMagic.size()));
  ABSL_ASSIGN_OR_RETURN(const uint32_t version, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t key_id_size, header.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint64_t body_size, header.ReadU64());
  if (version != kAdmissionEnvelopeVersion) {
    return absl::FailedPreconditionError(
        "Exact-profile admission envelope version is unsupported.");
  }
  if (key_id_size > kMaximumAdmissionKeyIdBytes ||
      body_size > kMaximumAdmissionEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "Exact-profile admission envelope declares an oversized field.");
  }
  const uint64_t bytes_before_mac =
      uint64_t{8} + 4 + 4 + 8 + key_id_size + body_size;
  if (bytes_before_mac > kMaximumAdmissionEnvelopeBytes - 32 ||
      envelope.size() != bytes_before_mac + 32) {
    return absl::DataLossError(
        "Exact-profile admission envelope length is non-canonical.");
  }
  const size_t key_id_offset = 8 + 4 + 4 + 8;
  const absl::string_view encoded_key_id =
      envelope.substr(key_id_offset, key_id_size);
  if (encoded_key_id != authentication.key_id) {
    return absl::UnauthenticatedError(
        "Exact-profile admission envelope key ID does not match.");
  }
  Hash256 encoded_mac;
  std::memcpy(encoded_mac.bytes.data(),
              envelope.data() + static_cast<size_t>(bytes_before_mac),
              encoded_mac.bytes.size());
  const absl::string_view authenticated_bytes =
      envelope.substr(0, static_cast<size_t>(bytes_before_mac));
  const Hash256 expected_mac = HmacSha256(
      authentication.authentication_key,
      {kAdmissionMacDomain, authenticated_bytes});
  if (!ConstantTimeHashEquals(encoded_mac, expected_mac)) {
    return absl::UnauthenticatedError(
        "Exact-profile admission envelope authentication failed.");
  }
  const size_t body_offset = key_id_offset + key_id_size;
  Reader body_reader(
      envelope.substr(body_offset, static_cast<size_t>(body_size)));
  ABSL_ASSIGN_OR_RETURN(const Hash256 encoded_record_id,
                        body_reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(
      ExactProfileAdmissionRecord record,
      DecodeRecordFields(
          envelope.substr(body_offset + 32,
                          static_cast<size_t>(body_size) - 32),
          encoded_record_id,
          authentication.key_id));
  if (record.record_id != encoded_record_id) {
    return absl::DataLossError(
        "Exact-profile admission envelope record ID is inconsistent.");
  }
  return record;
}

absl::StatusOr<ExactProfileAdmissionRecord> ExactProfileQualifier::Qualify(
    const ExactProfileQualificationSpec& spec,
    const FreshWorkerAuthentication& authentication) const {
  if (runner_ == nullptr) {
    return absl::FailedPreconditionError(
        "Exact-profile qualification has no fresh-worker runner.");
  }
  if (authoritative_engine_ == nullptr) {
    return absl::FailedPreconditionError(
        "Exact-profile qualification has no authoritative loaded Engine.");
  }
  ABSL_RETURN_IF_ERROR(ValidateQualificationSpec(spec));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_ASSIGN_OR_RETURN(
      const ExactLiteRtProfile derived_profile,
      authoritative_engine_->ResolveExactLiteRtProfile(
          *spec.session_config, spec.profile_assertion));
  ABSL_RETURN_IF_ERROR(
      ValidateDerivedProfileForQualification(derived_profile));

  std::set<Hash256> allocated_nonces;
  ABSL_ASSIGN_OR_RETURN(const Hash256 qualification_id,
                        GenerateUniqueNonce(allocated_nonces));
  allocated_nonces.insert(qualification_id);
  const Hash256 qualification_request_hash =
      ComputeExactProfileQualificationRequestHash(derived_profile, spec);
  const Hash256 payload_hash = Sha256(spec.canonical_request_payload);

  ExactProfileAdmissionRecord record;
  record.exact_profile_hash = derived_profile.profile_id;
  record.qualification_id = qualification_id;
  record.qualification_request_hash = qualification_request_hash;
  record.request_payload_hash = payload_hash;
  record.request_payload_size = spec.canonical_request_payload.size();
  record.independent_run_count = spec.independent_run_count;
  record.authentication_key_id = authentication.key_id;
  record.runs.reserve(spec.independent_run_count);

  std::set<int64_t> process_ids;
  std::set<Hash256> worker_nonces;
  Hash256 expected_launch_spec_hash;
  Hash256 expected_output_hash;
  for (uint32_t run_index = 0; run_index < spec.independent_run_count;
       ++run_index) {
    ABSL_ASSIGN_OR_RETURN(const Hash256 challenge,
                          GenerateUniqueNonce(allocated_nonces));
    allocated_nonces.insert(challenge);
    FreshWorkerRequest request;
    request.exact_profile_hash = derived_profile.profile_id;
    request.qualification_id = qualification_id;
    request.run_index = run_index;
    request.run_count = spec.independent_run_count;
    request.challenge_nonce = challenge;
    request.request_payload = spec.canonical_request_payload;

    ABSL_ASSIGN_OR_RETURN(
        FreshWorkerProcessObservation observation,
        runner_->Run(request, authentication));
    ABSL_RETURN_IF_ERROR(ValidateFreshWorkerResult(observation.result));
    if (observation.result.exact_profile_hash != derived_profile.profile_id ||
        observation.result.qualification_id != qualification_id ||
        observation.result.run_index != run_index ||
        observation.result.run_count != spec.independent_run_count ||
        observation.result.challenge_nonce != challenge ||
        observation.result.request_envelope_hash !=
            observation.request_envelope_hash) {
      return absl::UnauthenticatedError(
          "Fresh-worker observation is not bound to its qualification run.");
    }
    if (observation.result.status_code != absl::StatusCode::kOk) {
      return absl::Status(observation.result.status_code,
                          observation.result.status_message);
    }
    if (observation.result.replay_isolation !=
        FreshWorkerReplayIsolation::kEmptyCatalogs) {
      return absl::FailedPreconditionError(
          "Exact-profile qualification lacks the worker's empty-catalog attestation.");
    }
    ABSL_RETURN_IF_ERROR(ValidateLogitFramesMatchDerivedProfile(
        derived_profile, observation.result.logit_frames));
    if (observation.process_id <= 0 ||
        !process_ids.insert(observation.process_id).second) {
      return absl::FailedPreconditionError(
          "Exact-profile qualification did not use unique fresh processes.");
    }
    if (!worker_nonces.insert(observation.result.worker_instance_nonce).second) {
      return absl::FailedPreconditionError(
          "Exact-profile qualification reused a worker-instance nonce.");
    }
    if (IsZeroHash(observation.request_envelope_hash) ||
        IsZeroHash(observation.result_envelope_hash) ||
        IsZeroHash(observation.launch_spec_hash)) {
      return absl::DataLossError(
          "Fresh-worker observation omitted authenticated process evidence.");
    }

    const Hash256 output_hash = ComputeFreshWorkerOutputEvidenceHash(
        observation.result.canonical_output, observation.result.token_bytes,
        observation.result.logit_frames);
    if (run_index == 0) {
      record.replay_isolation = observation.result.replay_isolation;
      expected_launch_spec_hash = observation.launch_spec_hash;
      expected_output_hash = output_hash;
      record.canonical_output = observation.result.canonical_output;
      record.canonical_output_hash = Sha256(record.canonical_output);
      record.token_bytes = observation.result.token_bytes;
      record.token_bytes_hash = Sha256(record.token_bytes);
      record.logit_frames = observation.result.logit_frames;
    } else {
      if (observation.launch_spec_hash != expected_launch_spec_hash) {
        return absl::FailedPreconditionError(
            "Exact-profile qualification changed worker launch specification.");
      }
      if (observation.result.canonical_output != record.canonical_output ||
          observation.result.token_bytes != record.token_bytes ||
          observation.result.logit_frames != record.logit_frames ||
          output_hash != expected_output_hash) {
        return absl::FailedPreconditionError(
            "Independent cold runs produced different canonical output, token, or logits bytes.");
      }
    }
    record.runs.push_back(ExactProfileAdmissionRunEvidence{
        .run_index = run_index,
        .process_id = observation.process_id,
        .challenge_nonce = challenge,
        .worker_instance_nonce = observation.result.worker_instance_nonce,
        .request_envelope_hash = observation.request_envelope_hash,
        .result_envelope_hash = observation.result_envelope_hash,
        .launch_spec_hash = observation.launch_spec_hash,
        .output_evidence_hash = output_hash});
  }

  record.qualified_unix_micros = absl::ToUnixMicros(absl::Now());
  if (record.qualified_unix_micros <= 0) {
    return absl::FailedPreconditionError(
        "System clock cannot timestamp exact-profile admission.");
  }
  ABSL_ASSIGN_OR_RETURN(record.record_id,
                        ComputeExactProfileAdmissionRecordId(record));
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecord(record));
  return record;
}

absl::StatusOr<ExactProfileAdmissionRecord>
ExactProfileQualifier::QualifyAndAdmit(
    const ExactProfileQualificationSpec& spec,
    const FreshWorkerAuthentication& authentication,
    ExactProfileAdmissionRepository* repository) const {
  if (repository == nullptr) {
    return absl::InvalidArgumentError(
        "Exact-profile admission repository is missing.");
  }
  ABSL_RETURN_IF_ERROR(ValidateQualificationSpec(spec));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  if (authoritative_engine_ == nullptr) {
    return absl::FailedPreconditionError(
        "Exact-profile admission has no authoritative loaded Engine.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const ExactLiteRtProfile derived_profile,
      authoritative_engine_->ResolveExactLiteRtProfile(
          *spec.session_config, spec.profile_assertion));
  ABSL_RETURN_IF_ERROR(
      ValidateDerivedProfileForQualification(derived_profile));
  absl::StatusOr<ExactProfileAdmissionRecord> existing =
      repository->Get(derived_profile, authentication);
  if (existing.ok()) {
    ABSL_RETURN_IF_ERROR(
        ValidateAdmissionMatchesSpec(*existing, derived_profile, spec,
                                     authentication));
    return std::move(*existing);
  }
  if (existing.status().code() != absl::StatusCode::kNotFound) {
    return existing.status();
  }
  ABSL_ASSIGN_OR_RETURN(ExactProfileAdmissionRecord record,
                        Qualify(spec, authentication));
  if (record.exact_profile_hash != derived_profile.profile_id) {
    return absl::AbortedError(
        "The authoritative exact profile changed during qualification.");
  }
  const absl::Status put_status =
      repository->PutIfAbsent(record, authentication);
  if (put_status.ok()) return record;
  if (put_status.code() != absl::StatusCode::kAlreadyExists) {
    return put_status;
  }
  ABSL_ASSIGN_OR_RETURN(
      ExactProfileAdmissionRecord winner,
      repository->Get(derived_profile, authentication));
  ABSL_RETURN_IF_ERROR(
      ValidateAdmissionMatchesSpec(winner, derived_profile, spec,
                                   authentication));
  // Another qualifier won create-once publication. Return only that durable,
  // authenticated record; never leak this call's unpublished candidate.
  return winner;
}

}  // namespace litert::lm
