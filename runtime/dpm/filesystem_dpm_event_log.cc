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

#include "runtime/dpm/filesystem_dpm_event_log.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/correction_digest.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_receipt_validation.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kLogMagic = {'D', 'P', 'M', 'L', 'O', 'G',
                                            '0', '4'};
constexpr std::array<char, 8> kRecordMagic = {'D', 'P', 'M', 'E', 'V', 'T',
                                               '0', '4'};
constexpr uint32_t kLogFormatVersion = 4;
constexpr uint32_t kEventFormatVersion = 4;
// A maximum model-turn record contains three payload-sized strings (the event
// payload, projected memory, and the receipt's decision output), the canonical
// agent input, all repeated operation/case/manifest identities, the complete
// token vector, and a conservative allowance for fixed framing and hashes.
constexpr uint64_t kMaxRecordBytes =
    uint64_t{3} * kMaximumDPMEventPayloadBytes +
    kMaximumDPMCanonicalAgentInputBytes +
    uint64_t{3} * kMaximumDPMProjectionIdentityBytes +
    uint64_t{2} * kMaximumDPMEventOperationIdBytes +
    uint64_t{kMaximumDPMGenerationTokens} * sizeof(int32_t) + 4 * 1024;
constexpr uint64_t kRecordHeaderSize =
    kRecordMagic.size() + sizeof(uint64_t) + 32 + 32;
constexpr absl::string_view kGenesisDomain = "DPM_EVENT_LOG_GENESIS_SHA256_V4";
constexpr absl::string_view kRecordDomain = "DPM_EVENT_LOG_PREFIX_SHA256_V4";
static_assert(sizeof(size_t) <= sizeof(uint64_t));
static_assert(sizeof(int) >= sizeof(int32_t));

int ExtraOpenFlags() {
  int flags = 0;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  return flags;
}

class ScopedFd {
 public:
  ScopedFd() = default;
  explicit ScopedFd(int fd) : fd_(fd) {}
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ScopedFd(ScopedFd&& other) noexcept : fd_(other.Release()) {}
  ScopedFd& operator=(ScopedFd&& other) noexcept {
    if (this != &other) Reset(other.Release());
    return *this;
  }
  ~ScopedFd() { Reset(); }

  int get() const { return fd_; }
  int Release() {
    const int fd = fd_;
    fd_ = -1;
    return fd;
  }
  void Reset(int fd = -1) {
    if (fd_ >= 0) close(fd_);
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

class ScopedFileLock {
 public:
  ScopedFileLock() = default;
  explicit ScopedFileLock(ScopedFd fd) : fd_(std::move(fd)) {}
  ScopedFileLock(const ScopedFileLock&) = delete;
  ScopedFileLock& operator=(const ScopedFileLock&) = delete;
  ScopedFileLock(ScopedFileLock&&) = default;
  ScopedFileLock& operator=(ScopedFileLock&&) = default;
  ~ScopedFileLock() {
    if (fd_.get() >= 0) {
      while (flock(fd_.get(), LOCK_UN) != 0 && errno == EINTR) {
      }
    }
  }

 private:
  ScopedFd fd_;
};

class FilesystemOperationLease final : public DPMEventLogOperationLease {
 public:
  explicit FilesystemOperationLease(ScopedFileLock lock)
      : lock_(std::move(lock)) {}

 private:
  ScopedFileLock lock_;
};

absl::StatusOr<ScopedFd> OpenRegularFile(const std::filesystem::path& path,
                                         int flags, mode_t mode = 0) {
  int fd;
  do {
    fd = open(path.c_str(), flags | ExtraOpenFlags(), mode);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to open DPM storage file ",
                            path.string()));
  }
  struct stat stat_buffer;
  if (fstat(fd, &stat_buffer) != 0) {
    const int saved_errno = errno;
    close(fd);
    return absl::ErrnoToStatus(
        saved_errno,
        absl::StrCat("Unable to inspect DPM storage file ", path.string()));
  }
  if (!S_ISREG(stat_buffer.st_mode)) {
    close(fd);
    return absl::FailedPreconditionError(
        absl::StrCat("DPM storage path is not a regular file: ",
                     path.string()));
  }
  return ScopedFd(fd);
}

absl::StatusOr<ScopedFileLock> AcquireFileLock(
    const std::filesystem::path& path, bool exclusive, bool create) {
  int flags = O_RDWR;
  if (create) flags |= O_CREAT;
  ABSL_ASSIGN_OR_RETURN(ScopedFd fd,
                        OpenRegularFile(path, flags, S_IRUSR | S_IWUSR));
  const int operation = exclusive ? LOCK_EX : LOCK_SH;
  int result;
  do {
    result = flock(fd.get(), operation);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to lock DPM storage file ",
                            path.string()));
  }
  return ScopedFileLock(std::move(fd));
}

absl::Status SyncFd(int fd, absl::string_view description) {
  int result;
  do {
    result = fsync(fd);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to fsync ", description));
  }
  return absl::OkStatus();
}

absl::Status SyncDirectory(const std::filesystem::path& directory) {
  int flags = O_RDONLY;
#ifdef O_DIRECTORY
  flags |= O_DIRECTORY;
#endif
  int fd;
  do {
    fd = open(directory.c_str(), flags | ExtraOpenFlags());
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno,
        absl::StrCat("Unable to open DPM directory ", directory.string()));
  }
  ScopedFd scoped_fd(fd);
  return SyncFd(fd, directory.string());
}

absl::Status WriteAll(int fd, absl::string_view bytes,
                      absl::string_view description) {
  size_t offset = 0;
  while (offset < bytes.size()) {
    ssize_t written;
    do {
      written = write(fd, bytes.data() + offset, bytes.size() - offset);
    } while (written < 0 && errno == EINTR);
    if (written < 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("Unable to write ", description));
    }
    if (written == 0) {
      return absl::DataLossError(
          absl::StrCat("Zero-byte write while storing ", description));
    }
    offset += static_cast<size_t>(written);
  }
  return absl::OkStatus();
}

absl::Status ReadExactAt(int fd, uint64_t offset, char* output, size_t size,
                         absl::string_view description) {
  if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) ||
      size > static_cast<uint64_t>(std::numeric_limits<off_t>::max()) -
                 offset) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds addressable file offsets."));
  }
  size_t consumed = 0;
  while (consumed < size) {
    ssize_t read_count;
    do {
      read_count = pread(fd, output + consumed, size - consumed,
                         static_cast<off_t>(offset + consumed));
    } while (read_count < 0 && errno == EINTR);
    if (read_count < 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("Unable to read ", description));
    }
    if (read_count == 0) {
      return absl::DataLossError(
          absl::StrCat("Truncated ", description));
    }
    consumed += static_cast<size_t>(read_count);
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> FileSize(int fd, absl::string_view description) {
  struct stat stat_buffer;
  if (fstat(fd, &stat_buffer) != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to inspect ", description));
  }
  if (!S_ISREG(stat_buffer.st_mode) || stat_buffer.st_size < 0) {
    return absl::FailedPreconditionError(
        absl::StrCat(description, " is not a regular file."));
  }
  return static_cast<uint64_t>(stat_buffer.st_size);
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

void AppendI64(int64_t value, std::string* output) {
  AppendU64(static_cast<uint64_t>(value), output);
}

void AppendI32(int32_t value, std::string* output) {
  AppendU32(static_cast<uint32_t>(value), output);
}

absl::Status AppendString(absl::string_view value, std::string* output) {
  AppendU64(static_cast<uint64_t>(value.size()), output);
  output->append(value.data(), value.size());
  return absl::OkStatus();
}

void AppendHash(const Hash256& hash, std::string* output) {
  output->append(reinterpret_cast<const char*>(hash.bytes.data()),
                 hash.bytes.size());
}

void AppendPreparedPrefillWorkBinding(
    const DPMPreparedPrefillWorkBinding& binding, std::string* output) {
  AppendU32(static_cast<uint32_t>(binding.start_kind), output);
  AppendHash(binding.plan_id, output);
  AppendHash(binding.canonical_source_chunks_hash, output);
  AppendHash(binding.resolved_token_plan_hash, output);
  AppendHash(binding.shape_schedule_hash, output);
}

uint64_t DecodeU64(const char* bytes) {
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value = (value << 8) | static_cast<unsigned char>(bytes[i]);
  }
  return value;
}

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
}

bool IsSupportedTurnReceiptVersion(uint32_t format_version) {
  switch (format_version) {
    case DPMTurnReceipt::kLegacyFormatVersion:
    case DPMTurnReceipt::kPreviousFormatVersion:
    case DPMTurnReceipt::kCoverageV1FormatVersion:
    case DPMTurnReceipt::kFormatVersion:
      return true;
    default:
      return false;
  }
}

bool HasVersionFourReceiptFields(uint32_t format_version) {
  switch (format_version) {
    case DPMTurnReceipt::kPreviousFormatVersion:
    case DPMTurnReceipt::kCoverageV1FormatVersion:
    case DPMTurnReceipt::kFormatVersion:
      return true;
    case DPMTurnReceipt::kLegacyFormatVersion:
    default:
      return false;
  }
}

bool HasVersionFiveReceiptFields(uint32_t format_version) {
  switch (format_version) {
    case DPMTurnReceipt::kCoverageV1FormatVersion:
    case DPMTurnReceipt::kFormatVersion:
      return true;
    case DPMTurnReceipt::kLegacyFormatVersion:
    case DPMTurnReceipt::kPreviousFormatVersion:
    default:
      return false;
  }
}

bool ContainsControlByte(absl::string_view text) {
  for (unsigned char byte : text) {
    if (byte < 0x20 || byte == 0x7f) return true;
  }
  return false;
}

bool IsValidUtf8(absl::string_view text) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t lead = bytes[index++];
    if (lead <= 0x7f) continue;

    size_t continuation_count = 0;
    uint8_t minimum_second = 0x80;
    uint8_t maximum_second = 0xbf;
    if (lead >= 0xc2 && lead <= 0xdf) {
      continuation_count = 1;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      continuation_count = 2;
      if (lead == 0xe0) minimum_second = 0xa0;
      if (lead == 0xed) maximum_second = 0x9f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      continuation_count = 3;
      if (lead == 0xf0) minimum_second = 0x90;
      if (lead == 0xf4) maximum_second = 0x8f;
    } else {
      return false;
    }
    if (continuation_count > text.size() - index ||
        bytes[index] < minimum_second || bytes[index] > maximum_second) {
      return false;
    }
    ++index;
    for (size_t offset = 1; offset < continuation_count; ++offset, ++index) {
      if (bytes[index] < 0x80 || bytes[index] > 0xbf) return false;
    }
  }
  return true;
}

absl::StatusOr<std::string> BuildLogHeader(absl::string_view log_id,
                                           absl::string_view case_id) {
  if (log_id.empty()) {
    return absl::InvalidArgumentError("DPM filesystem log requires a log id.");
  }
  if (case_id.empty()) {
    return absl::InvalidArgumentError(
        "DPM filesystem log requires a case id.");
  }
  if (log_id.size() > kMaximumDPMProjectionIdentityBytes ||
      case_id.size() > kMaximumDPMProjectionIdentityBytes ||
      !IsValidUtf8(log_id) || !IsValidUtf8(case_id) ||
      ContainsControlByte(log_id) || ContainsControlByte(case_id)) {
    return absl::InvalidArgumentError(
        "DPM log and case identities must be bounded UTF-8 without control "
        "bytes.");
  }
  std::string header;
  header.append(kLogMagic.data(), kLogMagic.size());
  AppendU32(kLogFormatVersion, &header);
  AppendU32(static_cast<uint32_t>(log_id.size()), &header);
  header.append(log_id.data(), log_id.size());
  AppendU32(static_cast<uint32_t>(case_id.size()), &header);
  header.append(case_id.data(), case_id.size());
  return header;
}

Hash256 ComputeGenesisHash(absl::string_view header) {
  Sha256Hasher hasher;
  hasher.Update(kGenesisDomain);
  std::string length;
  AppendU64(header.size(), &length);
  hasher.Update(length);
  hasher.Update(header);
  return hasher.Finalize();
}

Hash256 ComputeNextPrefixHash(const Hash256& previous,
                              absl::string_view canonical_event) {
  Sha256Hasher hasher;
  hasher.Update(kRecordDomain);
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(previous.bytes.data()),
      previous.bytes.size()));
  std::string length;
  AppendU64(canonical_event.size(), &length);
  hasher.Update(length);
  hasher.Update(canonical_event);
  return hasher.Finalize();
}

class CanonicalReader {
 public:
  explicit CanonicalReader(absl::string_view bytes) : bytes_(bytes) {}

  bool empty() const { return offset_ == bytes_.size(); }

  absl::StatusOr<uint8_t> ReadU8() {
    ABSL_RETURN_IF_ERROR(Require(1));
    return static_cast<uint8_t>(bytes_[offset_++]);
  }

  absl::StatusOr<uint32_t> ReadU32() {
    ABSL_RETURN_IF_ERROR(Require(4));
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i) {
      value = (value << 8) |
              static_cast<unsigned char>(bytes_[offset_ + i]);
    }
    offset_ += 4;
    return value;
  }

  absl::StatusOr<uint64_t> ReadU64() {
    ABSL_RETURN_IF_ERROR(Require(8));
    const uint64_t value = DecodeU64(bytes_.data() + offset_);
    offset_ += 8;
    return value;
  }

  absl::StatusOr<int64_t> ReadI64() {
    ABSL_ASSIGN_OR_RETURN(uint64_t value, ReadU64());
    return static_cast<int64_t>(value);
  }

  absl::StatusOr<int32_t> ReadI32() {
    ABSL_ASSIGN_OR_RETURN(uint32_t value, ReadU32());
    return static_cast<int32_t>(value);
  }

  absl::StatusOr<std::string> ReadString(
      size_t maximum_bytes, absl::string_view field_name) {
    ABSL_ASSIGN_OR_RETURN(uint64_t length, ReadU64());
    if (length > static_cast<uint64_t>(maximum_bytes)) {
      return absl::DataLossError(absl::StrCat(
          "Canonical DPM ", field_name, " exceeds its product limit."));
    }
    if (length > bytes_.size() - offset_) {
      return absl::DataLossError("Truncated canonical DPM event string.");
    }
    std::string value(bytes_.data() + offset_, static_cast<size_t>(length));
    offset_ += static_cast<size_t>(length);
    return value;
  }

  absl::StatusOr<Hash256> ReadHash() {
    Hash256 hash;
    ABSL_RETURN_IF_ERROR(Require(hash.bytes.size()));
    std::memcpy(hash.bytes.data(), bytes_.data() + offset_,
                hash.bytes.size());
    offset_ += hash.bytes.size();
    return hash;
  }

 private:
  absl::Status Require(size_t count) const {
    if (count > bytes_.size() - offset_) {
      return absl::DataLossError("Truncated canonical DPM event.");
    }
    return absl::OkStatus();
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

absl::Status AppendProjectionManifest(
    const DPMProjectionManifest& manifest, std::string* output) {
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(manifest));
  AppendU32(manifest.format_version, output);
  ABSL_RETURN_IF_ERROR(AppendString(manifest.log_id, output));
  ABSL_RETURN_IF_ERROR(AppendString(manifest.case_id, output));
  AppendU64(manifest.source_event_count, output);
  AppendHash(manifest.source_prefix_hash, output);
  AppendU64(manifest.input_event_index, output);
  AppendU64(manifest.event_range_start, output);
  output->push_back(manifest.baseline_manifest_hash.has_value() ? 1 : 0);
  if (manifest.baseline_manifest_hash.has_value()) {
    AppendHash(*manifest.baseline_manifest_hash, output);
    AppendHash(*manifest.baseline_output_hash, output);
  }
  AppendHash(manifest.correction_digest, output);
  AppendHash(manifest.config_hash, output);
  AppendHash(manifest.runtime_identity.model_artifact_hash, output);
  AppendHash(manifest.runtime_identity.runtime_artifact_hash, output);
  AppendHash(manifest.runtime_identity.inference_profile_hash, output);
  AppendHash(manifest.request_hash, output);
  AppendHash(manifest.output_hash, output);
  output->push_back(static_cast<char>(manifest.replay_mode));
  AppendHash(manifest.replay_request_hash, output);
  AppendHash(manifest.execution_evidence_hash, output);
  output->push_back(manifest.exact_profile_id.has_value() ? 1 : 0);
  if (manifest.exact_profile_id.has_value()) {
    AppendHash(*manifest.exact_profile_id, output);
  }
  output->push_back(
      manifest.exact_profile_admission_record_id.has_value() ? 1 : 0);
  if (manifest.exact_profile_admission_record_id.has_value()) {
    AppendHash(*manifest.exact_profile_admission_record_id, output);
  }
  output->push_back(manifest.exact_output_evidence_hash.has_value() ? 1 : 0);
  if (manifest.exact_output_evidence_hash.has_value()) {
    AppendHash(*manifest.exact_output_evidence_hash, output);
  }
  AppendU32(manifest.exact_logit_frame_count, output);
  output->push_back(manifest.reused_canonical_winner ? 1 : 0);
  AppendHash(manifest.manifest_hash, output);
  return absl::OkStatus();
}

absl::StatusOr<DPMProjectionManifest> ReadProjectionManifest(
    CanonicalReader* reader) {
  if (reader == nullptr) {
    return absl::InvalidArgumentError(
        "Canonical DPM projection manifest reader is null.");
  }
  DPMProjectionManifest manifest;
  ABSL_ASSIGN_OR_RETURN(manifest.format_version, reader->ReadU32());
  if (manifest.format_version != DPMProjectionManifest::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Unsupported canonical DPM projection manifest version.");
  }
  ABSL_ASSIGN_OR_RETURN(
      manifest.log_id,
      reader->ReadString(kMaximumDPMProjectionIdentityBytes,
                         "projection manifest log id"));
  ABSL_ASSIGN_OR_RETURN(
      manifest.case_id,
      reader->ReadString(kMaximumDPMProjectionIdentityBytes,
                         "projection manifest case id"));
  ABSL_ASSIGN_OR_RETURN(manifest.source_event_count, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(manifest.source_prefix_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(manifest.input_event_index, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(manifest.event_range_start, reader->ReadU64());
  ABSL_ASSIGN_OR_RETURN(uint8_t has_baseline, reader->ReadU8());
  if (has_baseline > 1) {
    return absl::DataLossError(
        "Non-canonical DPM projection baseline-presence flag.");
  }
  if (has_baseline == 1) {
    ABSL_ASSIGN_OR_RETURN(Hash256 baseline_manifest, reader->ReadHash());
    ABSL_ASSIGN_OR_RETURN(Hash256 baseline_output, reader->ReadHash());
    manifest.baseline_manifest_hash = baseline_manifest;
    manifest.baseline_output_hash = baseline_output;
  }
  ABSL_ASSIGN_OR_RETURN(manifest.correction_digest, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(manifest.config_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(manifest.runtime_identity.model_artifact_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(manifest.runtime_identity.runtime_artifact_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(manifest.runtime_identity.inference_profile_hash,
                        reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(manifest.request_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(manifest.output_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(uint8_t replay_mode, reader->ReadU8());
  manifest.replay_mode = static_cast<DPMReplayMode>(replay_mode);
  ABSL_ASSIGN_OR_RETURN(manifest.replay_request_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(manifest.execution_evidence_hash, reader->ReadHash());
  ABSL_ASSIGN_OR_RETURN(uint8_t has_exact_profile, reader->ReadU8());
  if (has_exact_profile > 1) {
    return absl::DataLossError(
        "Non-canonical DPM projection exact-profile-presence flag.");
  }
  if (has_exact_profile == 1) {
    ABSL_ASSIGN_OR_RETURN(Hash256 exact_profile, reader->ReadHash());
    manifest.exact_profile_id = exact_profile;
  }
  ABSL_ASSIGN_OR_RETURN(uint8_t has_exact_admission, reader->ReadU8());
  if (has_exact_admission > 1) {
    return absl::DataLossError(
        "Non-canonical DPM projection exact-admission-presence flag.");
  }
  if (has_exact_admission == 1) {
    ABSL_ASSIGN_OR_RETURN(Hash256 exact_admission, reader->ReadHash());
    manifest.exact_profile_admission_record_id = exact_admission;
  }
  ABSL_ASSIGN_OR_RETURN(uint8_t has_exact_output_evidence, reader->ReadU8());
  if (has_exact_output_evidence > 1) {
    return absl::DataLossError(
        "Non-canonical DPM projection exact-output-evidence-presence flag.");
  }
  if (has_exact_output_evidence == 1) {
    ABSL_ASSIGN_OR_RETURN(Hash256 exact_output_evidence, reader->ReadHash());
    manifest.exact_output_evidence_hash = exact_output_evidence;
  }
  ABSL_ASSIGN_OR_RETURN(manifest.exact_logit_frame_count, reader->ReadU32());
  ABSL_ASSIGN_OR_RETURN(uint8_t reused_winner, reader->ReadU8());
  if (reused_winner > 1) {
    return absl::DataLossError(
        "Non-canonical DPM projection winner-reuse flag.");
  }
  manifest.reused_canonical_winner = reused_winner == 1;
  ABSL_ASSIGN_OR_RETURN(manifest.manifest_hash, reader->ReadHash());
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(manifest));
  return manifest;
}

absl::Status ValidateEvent(const DPMEvent& event,
                           const std::vector<DPMEvent>& prior_events,
                           absl::string_view log_id,
                           absl::string_view case_id,
                           const Hash256& prior_prefix_hash) {
  switch (event.kind) {
    case DPMEvent::Kind::kUser:
    case DPMEvent::Kind::kTool:
    case DPMEvent::Kind::kInternal:
    case DPMEvent::Kind::kModelTurn:
    case DPMEvent::Kind::kCorrection:
      break;
    default:
      return absl::InvalidArgumentError("Unknown DPM event kind.");
  }
  if (event.operation_id.empty()) {
    return absl::InvalidArgumentError(
        "DPM event requires a non-empty operation id.");
  }
  if (event.timestamp_us <= 0) {
    return absl::InvalidArgumentError(
        "DPM event requires a positive timestamp.");
  }
  if (event.case_id != case_id) {
    return absl::FailedPreconditionError(
        "DPM event case id does not match the immutable log case.");
  }
  if (event.operation_id.size() > kMaximumDPMEventOperationIdBytes ||
      event.payload.size() > kMaximumDPMEventPayloadBytes ||
      !IsValidUtf8(event.operation_id) || !IsValidUtf8(event.case_id) ||
      !IsValidUtf8(event.payload) ||
      ContainsControlByte(event.operation_id) ||
      ContainsControlByte(event.case_id)) {
    return absl::InvalidArgumentError(
        "DPM event identities or payload are invalid or exceed product "
        "bounds.");
  }
  if (event.kind != DPMEvent::Kind::kModelTurn) {
    if (event.turn_receipt.has_value()) {
      return absl::InvalidArgumentError(
          "Only a DPM model event may contain a turn receipt.");
    }
    if (event.payload.empty()) {
      return absl::InvalidArgumentError(
          "DPM input/correction event requires non-empty payload bytes.");
    }
    for (const DPMEvent& prior : prior_events) {
      if (prior.operation_id == event.operation_id) {
        return absl::AlreadyExistsError(
            "DPM operation id is already bound in this log.");
      }
    }
    if (!prior_events.empty()) {
      const DPMEvent& tail = prior_events.back();
      if (tail.kind == DPMEvent::Kind::kUser ||
          tail.kind == DPMEvent::Kind::kTool ||
          tail.kind == DPMEvent::Kind::kInternal) {
        return absl::FailedPreconditionError(
            "DPM log has an incomplete tail turn that must be resumed.");
      }
    }
    return absl::OkStatus();
  }
  if (!event.turn_receipt.has_value()) {
    return absl::InvalidArgumentError(
        "DPM model event requires a turn receipt.");
  }
  const DPMTurnReceipt& receipt = *event.turn_receipt;
  if (!IsSupportedTurnReceiptVersion(receipt.format_version)) {
    return absl::FailedPreconditionError(
        "Unsupported DPM turn receipt format version.");
  }
  if (receipt.operation_id != event.operation_id ||
      receipt.response_event_index != event.index ||
      receipt.decision_output != event.payload) {
    return absl::InvalidArgumentError(
        "DPM model event and turn receipt are not identically bound.");
  }
  if (receipt.input_event_index >= event.index ||
      receipt.input_event_index >= prior_events.size()) {
    return absl::InvalidArgumentError(
        "DPM turn receipt names an invalid input event.");
  }
  const DPMEvent& input = prior_events[receipt.input_event_index];
  if ((input.kind != DPMEvent::Kind::kUser &&
       input.kind != DPMEvent::Kind::kTool &&
       input.kind != DPMEvent::Kind::kInternal) ||
      input.turn_receipt ||
      input.operation_id != receipt.operation_id ||
      input.case_id != event.case_id ||
      receipt.response_event_index != receipt.input_event_index + 1 ||
      prior_events.size() != receipt.response_event_index ||
      event.timestamp_us < input.timestamp_us) {
    return absl::InvalidArgumentError(
        "DPM turn receipt is not bound to its authoritative input event.");
  }
  for (const DPMEvent& prior : prior_events) {
    if (prior.kind == DPMEvent::Kind::kModelTurn &&
        prior.operation_id == receipt.operation_id) {
      return absl::AlreadyExistsError(
          "DPM operation already has a committed model turn.");
    }
  }
  const DPMProjectionManifest& projection = receipt.projection_manifest;
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(projection));
  if (projection.log_id != log_id || projection.case_id != case_id ||
      projection.source_event_count != event.index ||
      projection.source_event_count != prior_events.size() ||
      projection.source_prefix_hash != prior_prefix_hash ||
      projection.input_event_index != receipt.input_event_index ||
      ComputeCanonicalDPMProjectionOutputHash(receipt.projected_memory) !=
          projection.output_hash) {
    return absl::InvalidArgumentError(
        "DPM turn receipt projection manifest is not bound to its raw-log "
        "source and projected-memory bytes.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_projection_correction_digest,
      ComputeDPMCorrectionDigestForPrefix(log_id, case_id, prior_events,
                                          prior_events.size()));
  if (projection.correction_digest !=
      expected_projection_correction_digest) {
    return absl::InvalidArgumentError(
        "DPM projection correction lineage is not derived from its raw-log "
        "source prefix.");
  }
  if (projection.baseline_manifest_hash.has_value()) {
    if (projection.event_range_start >= prior_events.size()) {
      return absl::InvalidArgumentError(
          "DPM projection baseline range does not name a prior response.");
    }
    const DPMEvent& baseline_response =
        prior_events[static_cast<size_t>(projection.event_range_start)];
    if (baseline_response.kind != DPMEvent::Kind::kModelTurn ||
        !baseline_response.turn_receipt.has_value()) {
      return absl::InvalidArgumentError(
          "DPM projection baseline does not name an authoritative receipt.");
    }
    const DPMProjectionManifest& baseline =
        baseline_response.turn_receipt->projection_manifest;
    ABSL_ASSIGN_OR_RETURN(
        const Hash256 expected_baseline_correction_digest,
        ComputeDPMCorrectionDigestForPrefix(
            log_id, case_id, prior_events, baseline.source_event_count));
    if (baseline.source_event_count != projection.event_range_start ||
        baseline.manifest_hash != *projection.baseline_manifest_hash ||
        baseline.output_hash != *projection.baseline_output_hash ||
        baseline.correction_digest != expected_baseline_correction_digest ||
        baseline.config_hash != projection.config_hash ||
        baseline.runtime_identity != projection.runtime_identity ||
        baseline.replay_mode != projection.replay_mode ||
        baseline.exact_profile_id != projection.exact_profile_id) {
      return absl::InvalidArgumentError(
          "DPM projection baseline lineage is not the compatible prior "
          "authoritative receipt named by its event range.");
    }
  }
  if (receipt.projected_memory.empty() ||
      receipt.projected_memory.size() > kMaximumDPMEventPayloadBytes ||
      receipt.canonical_agent_input.empty() ||
      receipt.canonical_agent_input.size() >
          kMaximumDPMCanonicalAgentInputBytes ||
      receipt.decision_token_ids.empty() ||
      !IsValidUtf8(receipt.projected_memory) ||
      !IsValidUtf8(receipt.canonical_agent_input) ||
      !IsValidUtf8(receipt.decision_output) ||
      IsZeroHash(receipt.agent_session_identity.model_artifact_hash) ||
      IsZeroHash(receipt.agent_session_identity.runtime_artifact_hash) ||
      IsZeroHash(receipt.agent_session_identity.inference_profile_hash) ||
      receipt.max_decision_tokens == 0 ||
      receipt.max_decision_tokens > kMaximumDPMGenerationTokens ||
      IsZeroHash(receipt.agent_request_hash) ||
      IsZeroHash(receipt.agent_transcript_hash)) {
    return absl::InvalidArgumentError(
        "DPM turn receipt is missing reconstruction data.");
  }
  if (receipt.decision_token_ids.size() > receipt.max_decision_tokens) {
    return absl::InvalidArgumentError(
        "DPM turn receipt exceeds its committed decision-token limit.");
  }
  for (int token_id : receipt.decision_token_ids) {
    if (token_id < 0) {
      return absl::InvalidArgumentError(
          "DPM decision token id is outside canonical int32 range.");
    }
    if constexpr (sizeof(int) > sizeof(int32_t)) {
      if (token_id > std::numeric_limits<int32_t>::max()) {
        return absl::InvalidArgumentError(
            "DPM decision token id is outside canonical int32 range.");
      }
    }
  }
  if (receipt.session_checkpoint_id.has_value() &&
      IsZeroHash(*receipt.session_checkpoint_id)) {
    return absl::InvalidArgumentError(
        "DPM turn receipt contains an empty checkpoint identity.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMTurnReceiptReplayEvidence(receipt));
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeEventCanonical(const DPMEvent& event) {
  std::string bytes;
  AppendU32(kEventFormatVersion, &bytes);
  AppendU64(event.index, &bytes);
  bytes.push_back(static_cast<char>(event.kind));
  AppendI64(event.timestamp_us, &bytes);
  ABSL_RETURN_IF_ERROR(AppendString(event.operation_id, &bytes));
  ABSL_RETURN_IF_ERROR(AppendString(event.case_id, &bytes));
  ABSL_RETURN_IF_ERROR(AppendString(event.payload, &bytes));
  bytes.push_back(event.turn_receipt.has_value() ? 1 : 0);
  if (!event.turn_receipt.has_value()) return bytes;

  const DPMTurnReceipt& receipt = *event.turn_receipt;
  ABSL_RETURN_IF_ERROR(ValidateDPMTurnReceiptReplayEvidence(receipt));
  AppendU32(receipt.format_version, &bytes);
  ABSL_RETURN_IF_ERROR(AppendString(receipt.operation_id, &bytes));
  AppendU64(receipt.input_event_index, &bytes);
  AppendU64(receipt.response_event_index, &bytes);
  ABSL_RETURN_IF_ERROR(
      AppendProjectionManifest(receipt.projection_manifest, &bytes));
  AppendHash(receipt.agent_session_identity.model_artifact_hash, &bytes);
  AppendHash(receipt.agent_session_identity.runtime_artifact_hash, &bytes);
  AppendHash(receipt.agent_session_identity.inference_profile_hash, &bytes);
  AppendU32(receipt.max_decision_tokens, &bytes);
  AppendHash(receipt.agent_request_hash, &bytes);
  bytes.push_back(static_cast<char>(receipt.agent_replay_mode));
  AppendHash(receipt.agent_replay_request_hash, &bytes);
  AppendHash(receipt.agent_execution_evidence_hash, &bytes);
  bytes.push_back(receipt.agent_exact_profile_id.has_value() ? 1 : 0);
  if (receipt.agent_exact_profile_id.has_value()) {
    AppendHash(*receipt.agent_exact_profile_id, &bytes);
  }
  bytes.push_back(
      receipt.agent_exact_profile_admission_record_id.has_value() ? 1 : 0);
  if (receipt.agent_exact_profile_admission_record_id.has_value()) {
    AppendHash(*receipt.agent_exact_profile_admission_record_id, &bytes);
  }
  bytes.push_back(
      receipt.agent_exact_output_evidence_hash.has_value() ? 1 : 0);
  if (receipt.agent_exact_output_evidence_hash.has_value()) {
    AppendHash(*receipt.agent_exact_output_evidence_hash, &bytes);
  }
  AppendU32(receipt.agent_exact_logit_frame_count, &bytes);
  bytes.push_back(receipt.agent_reused_canonical_winner ? 1 : 0);
  bytes.push_back(receipt.agent_producing_session_matched_output ? 1 : 0);
  ABSL_RETURN_IF_ERROR(AppendString(receipt.projected_memory, &bytes));
  ABSL_RETURN_IF_ERROR(AppendString(receipt.canonical_agent_input, &bytes));
  ABSL_RETURN_IF_ERROR(AppendString(receipt.decision_output, &bytes));
  AppendU64(receipt.decision_token_ids.size(), &bytes);
  for (int token_id : receipt.decision_token_ids) {
    AppendI32(static_cast<int32_t>(token_id), &bytes);
  }
  AppendHash(receipt.agent_transcript_hash, &bytes);
  bytes.push_back(receipt.session_checkpoint_id.has_value() ? 1 : 0);
  if (receipt.session_checkpoint_id.has_value()) {
    AppendHash(*receipt.session_checkpoint_id, &bytes);
  }
  if (HasVersionFourReceiptFields(receipt.format_version)) {
    bytes.push_back(
        receipt.restored_from_session_checkpoint_id.has_value() ? 1 : 0);
    if (receipt.restored_from_session_checkpoint_id.has_value()) {
      AppendHash(*receipt.restored_from_session_checkpoint_id, &bytes);
    }
    bytes.push_back(static_cast<char>(receipt.checkpoint_capture_origin));
    bytes.push_back(static_cast<char>(receipt.agent_worker_prefill_mode));
    AppendHash(receipt.agent_physical_execution_plan_hash, &bytes);
    bytes.push_back(
        receipt.agent_capsule_restore_admission_record_id.has_value() ? 1
                                                                      : 0);
    if (receipt.agent_capsule_restore_admission_record_id.has_value()) {
      AppendHash(*receipt.agent_capsule_restore_admission_record_id, &bytes);
    }
    bytes.push_back(
        receipt.agent_exact_worker_checkpoint_provenance.has_value() ? 1 : 0);
    if (receipt.agent_exact_worker_checkpoint_provenance.has_value()) {
      const DPMExactWorkerCheckpointProvenance& provenance =
          *receipt.agent_exact_worker_checkpoint_provenance;
      AppendU32(provenance.run_index, &bytes);
      AppendHash(provenance.execution_plan_hash, &bytes);
      AppendHash(provenance.request_envelope_hash, &bytes);
      AppendHash(provenance.result_envelope_hash, &bytes);
      AppendU64(provenance.transient_envelope_size, &bytes);
      AppendHash(provenance.transient_envelope_hash, &bytes);
      AppendHash(provenance.output_evidence_hash, &bytes);
    }
    if (HasVersionFiveReceiptFields(receipt.format_version)) {
      bytes.push_back(
          receipt.agent_capsule_restore_capability_id.has_value() ? 1 : 0);
      if (receipt.agent_capsule_restore_capability_id.has_value()) {
        AppendHash(*receipt.agent_capsule_restore_capability_id, &bytes);
      }
      bytes.push_back(
          receipt.agent_capsule_restore_coverage_id.has_value() ? 1 : 0);
      if (receipt.agent_capsule_restore_coverage_id.has_value()) {
        AppendHash(*receipt.agent_capsule_restore_coverage_id, &bytes);
      }
    }
    if (receipt.format_version == DPMTurnReceipt::kFormatVersion) {
      bytes.push_back(receipt.agent_prepared_prefill_work.has_value() ? 1 : 0);
      if (receipt.agent_prepared_prefill_work.has_value()) {
        AppendPreparedPrefillWorkBinding(
            *receipt.agent_prepared_prefill_work, &bytes);
      }
      bytes.push_back(receipt.published_checkpoint_capture.has_value() ? 1
                                                                      : 0);
      if (receipt.published_checkpoint_capture.has_value()) {
        AppendHash(receipt.published_checkpoint_capture->capture_plan_hash,
                   &bytes);
        AppendHash(receipt.published_checkpoint_capture->capture_evidence_id,
                   &bytes);
      }
      bytes.push_back(
          receipt.restored_checkpoint_capture_evidence_id.has_value() ? 1 : 0);
      if (receipt.restored_checkpoint_capture_evidence_id.has_value()) {
        AppendHash(*receipt.restored_checkpoint_capture_evidence_id, &bytes);
      }
      bytes.push_back(
          receipt.agent_capsule_restore_evidence_id.has_value() ? 1 : 0);
      if (receipt.agent_capsule_restore_evidence_id.has_value()) {
        AppendHash(*receipt.agent_capsule_restore_evidence_id, &bytes);
      }
    }
  }
  return bytes;
}

absl::StatusOr<DPMEvent> DecodeEventCanonical(absl::string_view bytes) {
  CanonicalReader reader(bytes);
  ABSL_ASSIGN_OR_RETURN(uint32_t format_version, reader.ReadU32());
  if (format_version != kEventFormatVersion) {
    return absl::FailedPreconditionError(
        "Unsupported canonical DPM event format version.");
  }
  DPMEvent event;
  ABSL_ASSIGN_OR_RETURN(event.index, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(uint8_t kind, reader.ReadU8());
  event.kind = static_cast<DPMEvent::Kind>(kind);
  ABSL_ASSIGN_OR_RETURN(event.timestamp_us, reader.ReadI64());
  ABSL_ASSIGN_OR_RETURN(
      event.operation_id,
      reader.ReadString(kMaximumDPMEventOperationIdBytes,
                        "event operation id"));
  ABSL_ASSIGN_OR_RETURN(
      event.case_id,
      reader.ReadString(kMaximumDPMProjectionIdentityBytes, "event case id"));
  ABSL_ASSIGN_OR_RETURN(
      event.payload,
      reader.ReadString(kMaximumDPMEventPayloadBytes, "event payload"));
  ABSL_ASSIGN_OR_RETURN(uint8_t has_receipt, reader.ReadU8());
  if (has_receipt > 1) {
    return absl::DataLossError(
        "Non-canonical DPM event receipt-presence flag.");
  }
  if (has_receipt == 1) {
    DPMTurnReceipt receipt;
    ABSL_ASSIGN_OR_RETURN(receipt.format_version, reader.ReadU32());
    if (!IsSupportedTurnReceiptVersion(receipt.format_version)) {
      return absl::FailedPreconditionError(
          "Unsupported canonical DPM turn receipt version.");
    }
    ABSL_ASSIGN_OR_RETURN(
        receipt.operation_id,
        reader.ReadString(kMaximumDPMEventOperationIdBytes,
                          "turn receipt operation id"));
    ABSL_ASSIGN_OR_RETURN(receipt.input_event_index, reader.ReadU64());
    ABSL_ASSIGN_OR_RETURN(receipt.response_event_index, reader.ReadU64());
    ABSL_ASSIGN_OR_RETURN(receipt.projection_manifest,
                          ReadProjectionManifest(&reader));
    ABSL_ASSIGN_OR_RETURN(receipt.agent_session_identity.model_artifact_hash,
                          reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(receipt.agent_session_identity.runtime_artifact_hash,
                          reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(
        receipt.agent_session_identity.inference_profile_hash,
        reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(receipt.max_decision_tokens, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(receipt.agent_request_hash, reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(uint8_t agent_replay_mode, reader.ReadU8());
    receipt.agent_replay_mode = static_cast<DPMReplayMode>(agent_replay_mode);
    ABSL_ASSIGN_OR_RETURN(receipt.agent_replay_request_hash,
                          reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(receipt.agent_execution_evidence_hash,
                          reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(uint8_t has_agent_exact_profile, reader.ReadU8());
    if (has_agent_exact_profile > 1) {
      return absl::DataLossError(
          "Non-canonical DPM agent exact-profile-presence flag.");
    }
    if (has_agent_exact_profile == 1) {
      ABSL_ASSIGN_OR_RETURN(Hash256 exact_profile, reader.ReadHash());
      receipt.agent_exact_profile_id = exact_profile;
    }
    ABSL_ASSIGN_OR_RETURN(uint8_t has_agent_exact_admission, reader.ReadU8());
    if (has_agent_exact_admission > 1) {
      return absl::DataLossError(
          "Non-canonical DPM agent exact-admission-presence flag.");
    }
    if (has_agent_exact_admission == 1) {
      ABSL_ASSIGN_OR_RETURN(Hash256 exact_admission, reader.ReadHash());
      receipt.agent_exact_profile_admission_record_id = exact_admission;
    }
    ABSL_ASSIGN_OR_RETURN(uint8_t has_agent_exact_output, reader.ReadU8());
    if (has_agent_exact_output > 1) {
      return absl::DataLossError(
          "Non-canonical DPM agent exact-output-evidence-presence flag.");
    }
    if (has_agent_exact_output == 1) {
      ABSL_ASSIGN_OR_RETURN(Hash256 exact_output, reader.ReadHash());
      receipt.agent_exact_output_evidence_hash = exact_output;
    }
    ABSL_ASSIGN_OR_RETURN(receipt.agent_exact_logit_frame_count,
                          reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(uint8_t reused_agent_winner, reader.ReadU8());
    ABSL_ASSIGN_OR_RETURN(uint8_t producing_agent_session, reader.ReadU8());
    if (reused_agent_winner > 1 || producing_agent_session > 1) {
      return absl::DataLossError(
          "Non-canonical DPM agent replay provenance flag.");
    }
    receipt.agent_reused_canonical_winner = reused_agent_winner == 1;
    receipt.agent_producing_session_matched_output =
        producing_agent_session == 1;
    ABSL_ASSIGN_OR_RETURN(
        receipt.projected_memory,
        reader.ReadString(kMaximumDPMEventPayloadBytes,
                          "turn receipt projected memory"));
    ABSL_ASSIGN_OR_RETURN(
        receipt.canonical_agent_input,
        reader.ReadString(kMaximumDPMCanonicalAgentInputBytes,
                          "turn receipt canonical agent input"));
    ABSL_ASSIGN_OR_RETURN(
        receipt.decision_output,
        reader.ReadString(kMaximumDPMEventPayloadBytes,
                          "turn receipt decision output"));
    ABSL_ASSIGN_OR_RETURN(uint64_t token_count, reader.ReadU64());
    if (token_count > kMaximumDPMGenerationTokens ||
        token_count > bytes.size() / sizeof(int32_t)) {
      return absl::DataLossError(
          "Canonical DPM token vector exceeds the product limit.");
    }
    receipt.decision_token_ids.reserve(static_cast<size_t>(token_count));
    for (uint64_t i = 0; i < token_count; ++i) {
      ABSL_ASSIGN_OR_RETURN(int32_t token_id, reader.ReadI32());
      receipt.decision_token_ids.push_back(static_cast<int>(token_id));
    }
    ABSL_ASSIGN_OR_RETURN(receipt.agent_transcript_hash, reader.ReadHash());
    ABSL_ASSIGN_OR_RETURN(uint8_t has_checkpoint, reader.ReadU8());
    if (has_checkpoint > 1) {
      return absl::DataLossError(
          "Non-canonical DPM checkpoint-presence flag.");
    }
    if (has_checkpoint == 1) {
      ABSL_ASSIGN_OR_RETURN(Hash256 checkpoint_id, reader.ReadHash());
      receipt.session_checkpoint_id = checkpoint_id;
    }
    if (HasVersionFourReceiptFields(receipt.format_version)) {
      ABSL_ASSIGN_OR_RETURN(uint8_t has_restored_checkpoint, reader.ReadU8());
      if (has_restored_checkpoint > 1) {
        return absl::DataLossError(
            "Non-canonical DPM restored-checkpoint-presence flag.");
      }
      if (has_restored_checkpoint == 1) {
        ABSL_ASSIGN_OR_RETURN(Hash256 restored_checkpoint, reader.ReadHash());
        receipt.restored_from_session_checkpoint_id = restored_checkpoint;
      }
      ABSL_ASSIGN_OR_RETURN(uint8_t checkpoint_capture_origin,
                            reader.ReadU8());
      receipt.checkpoint_capture_origin =
          static_cast<DPMCheckpointCaptureOrigin>(checkpoint_capture_origin);
      ABSL_ASSIGN_OR_RETURN(uint8_t agent_worker_prefill_mode,
                            reader.ReadU8());
      receipt.agent_worker_prefill_mode =
          static_cast<DPMCheckpointWorkerPrefillMode>(
              agent_worker_prefill_mode);
      ABSL_ASSIGN_OR_RETURN(receipt.agent_physical_execution_plan_hash,
                            reader.ReadHash());
      ABSL_ASSIGN_OR_RETURN(uint8_t has_capsule_restore_admission,
                            reader.ReadU8());
      if (has_capsule_restore_admission > 1) {
        return absl::DataLossError(
            "Non-canonical DPM CapsuleRestore-admission-presence flag.");
      }
      if (has_capsule_restore_admission == 1) {
        ABSL_ASSIGN_OR_RETURN(Hash256 capsule_restore_admission,
                              reader.ReadHash());
        receipt.agent_capsule_restore_admission_record_id =
            capsule_restore_admission;
      }
      ABSL_ASSIGN_OR_RETURN(uint8_t has_worker_checkpoint_provenance,
                            reader.ReadU8());
      if (has_worker_checkpoint_provenance > 1) {
        return absl::DataLossError(
            "Non-canonical DPM worker-checkpoint-provenance-presence flag.");
      }
      if (has_worker_checkpoint_provenance == 1) {
        DPMExactWorkerCheckpointProvenance provenance;
        ABSL_ASSIGN_OR_RETURN(provenance.run_index, reader.ReadU32());
        ABSL_ASSIGN_OR_RETURN(provenance.execution_plan_hash,
                              reader.ReadHash());
        ABSL_ASSIGN_OR_RETURN(provenance.request_envelope_hash,
                              reader.ReadHash());
        ABSL_ASSIGN_OR_RETURN(provenance.result_envelope_hash,
                              reader.ReadHash());
        ABSL_ASSIGN_OR_RETURN(provenance.transient_envelope_size,
                              reader.ReadU64());
        ABSL_ASSIGN_OR_RETURN(provenance.transient_envelope_hash,
                              reader.ReadHash());
        ABSL_ASSIGN_OR_RETURN(provenance.output_evidence_hash,
                              reader.ReadHash());
        receipt.agent_exact_worker_checkpoint_provenance = provenance;
      }
      if (HasVersionFiveReceiptFields(receipt.format_version)) {
        ABSL_ASSIGN_OR_RETURN(uint8_t has_capsule_restore_capability,
                              reader.ReadU8());
        if (has_capsule_restore_capability > 1) {
          return absl::DataLossError(
              "Non-canonical DPM CapsuleRestore-capability-presence flag.");
        }
        if (has_capsule_restore_capability == 1) {
          ABSL_ASSIGN_OR_RETURN(Hash256 capsule_restore_capability,
                                reader.ReadHash());
          receipt.agent_capsule_restore_capability_id =
              capsule_restore_capability;
        }
        ABSL_ASSIGN_OR_RETURN(uint8_t has_capsule_restore_coverage,
                              reader.ReadU8());
        if (has_capsule_restore_coverage > 1) {
          return absl::DataLossError(
              "Non-canonical DPM CapsuleRestore-coverage-presence flag.");
        }
        if (has_capsule_restore_coverage == 1) {
          ABSL_ASSIGN_OR_RETURN(Hash256 capsule_restore_coverage,
                                reader.ReadHash());
          receipt.agent_capsule_restore_coverage_id =
              capsule_restore_coverage;
        }
      }
      if (receipt.format_version == DPMTurnReceipt::kFormatVersion) {
        ABSL_ASSIGN_OR_RETURN(uint8_t has_prepared_prefill_work,
                              reader.ReadU8());
        if (has_prepared_prefill_work > 1) {
          return absl::DataLossError(
              "Non-canonical DPM prepared-prefill-work-presence flag.");
        }
        if (has_prepared_prefill_work == 1) {
          DPMPreparedPrefillWorkBinding binding;
          ABSL_ASSIGN_OR_RETURN(uint32_t start_kind, reader.ReadU32());
          binding.start_kind =
              static_cast<DPMPreparedPrefillStartKind>(start_kind);
          ABSL_ASSIGN_OR_RETURN(binding.plan_id, reader.ReadHash());
          ABSL_ASSIGN_OR_RETURN(binding.canonical_source_chunks_hash,
                                reader.ReadHash());
          ABSL_ASSIGN_OR_RETURN(binding.resolved_token_plan_hash,
                                reader.ReadHash());
          ABSL_ASSIGN_OR_RETURN(binding.shape_schedule_hash,
                                reader.ReadHash());
          receipt.agent_prepared_prefill_work = binding;
        }

        ABSL_ASSIGN_OR_RETURN(uint8_t has_published_checkpoint_capture,
                              reader.ReadU8());
        if (has_published_checkpoint_capture > 1) {
          return absl::DataLossError(
              "Non-canonical DPM checkpoint-capture-evidence-presence flag.");
        }
        if (has_published_checkpoint_capture == 1) {
          DPMCheckpointCaptureEvidenceBinding binding;
          ABSL_ASSIGN_OR_RETURN(binding.capture_plan_hash, reader.ReadHash());
          ABSL_ASSIGN_OR_RETURN(binding.capture_evidence_id,
                                reader.ReadHash());
          receipt.published_checkpoint_capture = binding;
        }

        ABSL_ASSIGN_OR_RETURN(
            uint8_t has_restored_checkpoint_capture_evidence,
            reader.ReadU8());
        if (has_restored_checkpoint_capture_evidence > 1) {
          return absl::DataLossError(
              "Non-canonical DPM restored-checkpoint-capture-evidence-presence "
              "flag.");
        }
        if (has_restored_checkpoint_capture_evidence == 1) {
          ABSL_ASSIGN_OR_RETURN(Hash256 capture_evidence_id,
                                reader.ReadHash());
          receipt.restored_checkpoint_capture_evidence_id =
              capture_evidence_id;
        }

        ABSL_ASSIGN_OR_RETURN(uint8_t has_capsule_restore_evidence,
                              reader.ReadU8());
        if (has_capsule_restore_evidence > 1) {
          return absl::DataLossError(
              "Non-canonical DPM capsule-restore-evidence-presence flag.");
        }
        if (has_capsule_restore_evidence == 1) {
          ABSL_ASSIGN_OR_RETURN(Hash256 restore_evidence_id,
                                reader.ReadHash());
          receipt.agent_capsule_restore_evidence_id = restore_evidence_id;
        }
      }
    }
    ABSL_RETURN_IF_ERROR(ValidateDPMTurnReceiptReplayEvidence(receipt));
    event.turn_receipt = std::move(receipt);
  }
  if (!reader.empty()) {
    return absl::DataLossError("Trailing bytes in canonical DPM event.");
  }
  return event;
}

absl::StatusOr<std::string> BuildRecord(absl::string_view canonical_event,
                                        const Hash256& previous,
                                        const Hash256& current) {
  if (canonical_event.size() > kMaxRecordBytes) {
    return absl::ResourceExhaustedError(
        "Canonical DPM event exceeds the durable record limit.");
  }
  std::string record;
  record.reserve(kRecordHeaderSize + canonical_event.size());
  record.append(kRecordMagic.data(), kRecordMagic.size());
  AppendU64(canonical_event.size(), &record);
  AppendHash(previous, &record);
  AppendHash(current, &record);
  record.append(canonical_event.data(), canonical_event.size());
  return record;
}

struct ScanResult {
  DPMLogSnapshot snapshot;
  uint64_t file_size = 0;
};

absl::StatusOr<ScanResult> ScanLog(int fd, absl::string_view log_id,
                                   absl::string_view case_id) {
  ABSL_ASSIGN_OR_RETURN(std::string expected_header,
                        BuildLogHeader(log_id, case_id));
  ABSL_ASSIGN_OR_RETURN(uint64_t file_size, FileSize(fd, "DPM event log"));
  if (file_size < expected_header.size()) {
    return absl::DataLossError("Truncated DPM event log header.");
  }
  std::string actual_header(expected_header.size(), '\0');
  ABSL_RETURN_IF_ERROR(ReadExactAt(fd, 0, actual_header.data(),
                                   actual_header.size(),
                                   "DPM event log header"));
  if (actual_header != expected_header) {
    return absl::DataLossError(
        "DPM event log header, version, log id, or case id does not match.");
  }

  ScanResult result;
  result.snapshot.log_id.assign(log_id.data(), log_id.size());
  result.snapshot.case_id.assign(case_id.data(), case_id.size());
  result.file_size = file_size;
  Hash256 prefix = ComputeGenesisHash(expected_header);
  result.snapshot.prefix_hashes.push_back(prefix);
  uint64_t offset = expected_header.size();
  while (offset < file_size) {
    if (file_size - offset < kRecordHeaderSize) {
      return absl::DataLossError("Truncated DPM event record header.");
    }
    std::array<char, kRecordHeaderSize> record_header{};
    ABSL_RETURN_IF_ERROR(ReadExactAt(fd, offset, record_header.data(),
                                     record_header.size(),
                                     "DPM event record header"));
    if (std::memcmp(record_header.data(), kRecordMagic.data(),
                    kRecordMagic.size()) != 0) {
      return absl::DataLossError("Invalid DPM event record magic.");
    }
    const uint64_t payload_size =
        DecodeU64(record_header.data() + kRecordMagic.size());
    if (payload_size > kMaxRecordBytes) {
      return absl::ResourceExhaustedError(
          "DPM event record exceeds the supported size limit.");
    }
    if (payload_size > file_size - offset - kRecordHeaderSize) {
      return absl::DataLossError("Truncated DPM event record payload.");
    }
    Hash256 stored_previous;
    Hash256 stored_current;
    const size_t previous_offset = kRecordMagic.size() + sizeof(uint64_t);
    std::memcpy(stored_previous.bytes.data(),
                record_header.data() + previous_offset,
                stored_previous.bytes.size());
    std::memcpy(stored_current.bytes.data(),
                record_header.data() + previous_offset +
                    stored_previous.bytes.size(),
                stored_current.bytes.size());
    if (stored_previous != prefix) {
      return absl::DataLossError("Broken DPM event prefix-hash link.");
    }
    std::string canonical_event(static_cast<size_t>(payload_size), '\0');
    ABSL_RETURN_IF_ERROR(ReadExactAt(
        fd, offset + kRecordHeaderSize, canonical_event.data(),
        canonical_event.size(), "DPM canonical event payload"));
    const Hash256 expected_current =
        ComputeNextPrefixHash(prefix, canonical_event);
    if (stored_current != expected_current) {
      return absl::DataLossError("Corrupt DPM event prefix hash.");
    }
    ABSL_ASSIGN_OR_RETURN(DPMEvent event,
                          DecodeEventCanonical(canonical_event));
    if (event.index != result.snapshot.events.size()) {
      return absl::DataLossError(
          "DPM event log contains non-contiguous indices.");
    }
    absl::Status validation = ValidateEvent(
        event, result.snapshot.events, log_id, case_id, prefix);
    if (!validation.ok()) {
      return absl::DataLossError(absl::StrCat(
          "Invalid authoritative DPM event: ", validation.message()));
    }
    ABSL_ASSIGN_OR_RETURN(std::string reencoded,
                          EncodeEventCanonical(event));
    if (reencoded != canonical_event) {
      return absl::DataLossError("Non-canonical DPM event encoding.");
    }
    result.snapshot.events.push_back(std::move(event));
    prefix = expected_current;
    result.snapshot.prefix_hashes.push_back(prefix);
    offset += kRecordHeaderSize + payload_size;
  }
  ABSL_ASSIGN_OR_RETURN(uint64_t final_file_size,
                        FileSize(fd, "DPM event log"));
  if (final_file_size != file_size) {
    return absl::DataLossError(
        "DPM event log changed during a locked strict scan.");
  }
  result.snapshot.generation = result.snapshot.events.size();
  result.snapshot.prefix_hash = prefix;
  return result;
}

// A complete append can remain readable in the page cache even when the
// appending process observed an fsync failure. Never accept such bytes as an
// authoritative prefix merely because they parse: first establish a successful
// durability barrier while the event-file lock excludes writers. This also
// makes recovery after an uncertain append explicit--a later operation may
// adopt it only after this barrier succeeds.
absl::StatusOr<ScanResult> DurablyScanLog(int fd, absl::string_view log_id,
                                         absl::string_view case_id) {
  ABSL_RETURN_IF_ERROR(SyncFd(fd, "DPM event log before authoritative scan"));
  return ScanLog(fd, log_id, case_id);
}

absl::StatusOr<std::pair<ScopedFd, std::filesystem::path>> CreateTempFile(
    const std::filesystem::path& directory, absl::string_view stem) {
  std::string pattern =
      (directory / absl::StrCat(".", stem, ".tmp.XXXXXX")).string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  int fd = mkstemp(writable.data());
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to create DPM temporary file in ",
                            directory.string()));
  }
#ifdef FD_CLOEXEC
  if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) {
    const int saved_errno = errno;
    close(fd);
    unlink(writable.data());
    return absl::ErrnoToStatus(
        saved_errno, "Unable to mark DPM temporary file close-on-exec");
  }
#endif
  return std::make_pair(ScopedFd(fd),
                        std::filesystem::path(writable.data()));
}

absl::Status PublishInitialLog(const std::filesystem::path& directory,
                               const std::filesystem::path& log_path,
                               absl::string_view header) {
  ABSL_ASSIGN_OR_RETURN(auto temporary,
                        CreateTempFile(directory, "events"));
  ScopedFd fd = std::move(temporary.first);
  const std::filesystem::path temporary_path = std::move(temporary.second);
  absl::Status status = WriteAll(fd.get(), header, "DPM event log header");
  if (status.ok()) status = SyncFd(fd.get(), "DPM event log header");
  if (status.ok()) {
    int result;
    do {
      result = link(temporary_path.c_str(), log_path.c_str());
    } while (result != 0 && errno == EINTR);
    if (result != 0 && errno != EEXIST) {
      status = absl::ErrnoToStatus(
          errno, "Unable to publish initial DPM event log");
    }
  }
  if (status.ok()) status = SyncDirectory(directory);
  const int unlink_result = unlink(temporary_path.c_str());
  if (status.ok() && unlink_result != 0 && errno != ENOENT) {
    status = absl::ErrnoToStatus(
        errno, "Unable to remove published DPM temporary log");
  }
  if (status.ok()) status = SyncDirectory(directory);
  return status;
}

}  // namespace

FilesystemDPMEventLog::FilesystemDPMEventLog(
    std::filesystem::path directory, std::string log_id,
    std::string case_id)
    : directory_(std::move(directory)),
      log_path_(directory_ / "events.dpm"),
      lock_path_(directory_ / ".events.lock"),
      operation_lock_path_(directory_ / ".operation.lock"),
      log_id_(std::move(log_id)),
      case_id_(std::move(case_id)) {}

absl::StatusOr<std::unique_ptr<FilesystemDPMEventLog>>
FilesystemDPMEventLog::Create(std::filesystem::path directory,
                              std::string log_id, std::string case_id) {
  if (directory.empty()) {
    return absl::InvalidArgumentError(
        "DPM filesystem log requires a storage directory.");
  }
  ABSL_RETURN_IF_ERROR(BuildLogHeader(log_id, case_id).status());
  std::error_code path_error;
  directory = std::filesystem::absolute(directory, path_error);
  if (path_error) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to resolve DPM event-log directory: ",
        path_error.message()));
  }
  directory = std::filesystem::weakly_canonical(directory, path_error);
  if (path_error) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to normalize DPM event-log directory: ",
        path_error.message()));
  }
  auto log = std::unique_ptr<FilesystemDPMEventLog>(
      new FilesystemDPMEventLog(std::move(directory), std::move(log_id),
                                std::move(case_id)));
  ABSL_RETURN_IF_ERROR(log->Initialize());
  return log;
}

absl::Status FilesystemDPMEventLog::Initialize() {
  std::error_code error;
  std::filesystem::create_directories(directory_, error);
  if (error || !std::filesystem::is_directory(directory_, error)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Unable to create DPM event-log directory ",
                     directory_.string(), ": ", error.message()));
  }
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireFileLock(lock_path_, true, true));
  // The operation lock has no payload. Creating it while the append lock is
  // held establishes its durable name without taking locks in the inverse of
  // the runtime order (operation lease, then individual append lock).
  ABSL_ASSIGN_OR_RETURN(
      ScopedFd operation_lock_fd,
      OpenRegularFile(operation_lock_path_, O_RDWR | O_CREAT,
                      S_IRUSR | S_IWUSR));
  absl::StatusOr<ScopedFd> existing =
      OpenRegularFile(log_path_, O_RDWR);
  if (!existing.ok()) {
    if (!absl::IsNotFound(existing.status())) return existing.status();
    ABSL_ASSIGN_OR_RETURN(std::string header,
                          BuildLogHeader(log_id_, case_id_));
    ABSL_RETURN_IF_ERROR(PublishInitialLog(directory_, log_path_, header));
    existing = OpenRegularFile(log_path_, O_RDWR);
    if (!existing.ok()) return existing.status();
  }
  ABSL_RETURN_IF_ERROR(
      DurablyScanLog(existing->get(), log_id_, case_id_).status());
  ABSL_RETURN_IF_ERROR(SyncDirectory(directory_));
  return SyncDirectory(directory_.parent_path());
}

absl::StatusOr<DPMLogSnapshot> FilesystemDPMEventLog::Snapshot() const {
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireFileLock(lock_path_, false, false));
  ABSL_ASSIGN_OR_RETURN(ScopedFd fd, OpenRegularFile(log_path_, O_RDWR));
  ABSL_ASSIGN_OR_RETURN(ScanResult scanned,
                        DurablyScanLog(fd.get(), log_id_, case_id_));
  return std::move(scanned.snapshot);
}

absl::StatusOr<std::unique_ptr<DPMEventLogOperationLease>>
FilesystemDPMEventLog::AcquireOperationLease() {
  ABSL_ASSIGN_OR_RETURN(
      ScopedFileLock lock,
      AcquireFileLock(operation_lock_path_, true, false));
  return std::unique_ptr<DPMEventLogOperationLease>(
      new FilesystemOperationLease(std::move(lock)));
}

absl::StatusOr<Hash256> FilesystemDPMEventLog::PrefixHash(
    uint64_t event_count) const {
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireFileLock(lock_path_, false, false));
  ABSL_ASSIGN_OR_RETURN(ScopedFd fd, OpenRegularFile(log_path_, O_RDWR));
  ABSL_ASSIGN_OR_RETURN(ScanResult scanned,
                        DurablyScanLog(fd.get(), log_id_, case_id_));
  if (event_count > scanned.snapshot.generation) {
    return absl::OutOfRangeError(absl::StrCat(
        "DPM prefix event count ", event_count,
        " exceeds log generation ", scanned.snapshot.generation, "."));
  }
  return scanned.snapshot.prefix_hashes[static_cast<size_t>(event_count)];
}

absl::StatusOr<DPMAppendResult>
FilesystemDPMEventLog::AppendIfGeneration(DPMEvent event,
                                          uint64_t expected_generation) {
  if (event.index != 0) {
    return absl::InvalidArgumentError(
        "Caller must leave a new DPM event index at zero.");
  }
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireFileLock(lock_path_, true, false));
  ABSL_ASSIGN_OR_RETURN(
      ScopedFd fd,
      OpenRegularFile(log_path_, O_RDWR | O_APPEND));
  ABSL_ASSIGN_OR_RETURN(ScanResult scanned,
                        DurablyScanLog(fd.get(), log_id_, case_id_));
  if (scanned.snapshot.generation != expected_generation) {
    return absl::AbortedError(absl::StrCat(
        "DPM append generation changed: expected ", expected_generation,
        ", authoritative generation is ", scanned.snapshot.generation, "."));
  }
  if (scanned.snapshot.generation ==
      std::numeric_limits<uint64_t>::max()) {
    return absl::ResourceExhaustedError(
        "DPM event log cannot address another event.");
  }
  event.index = scanned.snapshot.generation;
  ABSL_RETURN_IF_ERROR(ValidateEvent(
      event, scanned.snapshot.events, log_id_, case_id_,
      scanned.snapshot.prefix_hashes.back()));
  ABSL_ASSIGN_OR_RETURN(std::string canonical_event,
                        EncodeEventCanonical(event));
  const Hash256 previous = scanned.snapshot.prefix_hashes.back();
  const Hash256 current =
      ComputeNextPrefixHash(previous, canonical_event);
  ABSL_ASSIGN_OR_RETURN(std::string record,
                        BuildRecord(canonical_event, previous, current));
  ABSL_RETURN_IF_ERROR(WriteAll(fd.get(), record, "DPM event record"));
  ABSL_RETURN_IF_ERROR(SyncFd(fd.get(), "DPM event log"));

  DPMAppendResult result;
  result.event_index = event.index;
  scanned.snapshot.events.push_back(std::move(event));
  scanned.snapshot.generation = scanned.snapshot.events.size();
  scanned.snapshot.prefix_hash = current;
  scanned.snapshot.prefix_hashes.push_back(current);
  result.snapshot = std::move(scanned.snapshot);
  return result;
}

}  // namespace litert::lm
