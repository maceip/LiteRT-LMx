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

#include "runtime/dpm/filesystem_session_checkpoint_repository.h"

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
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kStoredDescriptorMagic = {
    'D', 'P', 'M', 'D', 'S', 'C', '0', '1'};
constexpr uint32_t kStoredDescriptorFormatVersion = 1;
constexpr uint64_t kMaxStoredDescriptorBytes = uint64_t{1024} * 1024;
constexpr uint64_t kMaxStoredEnvelopeBytes = uint64_t{8} * 1024 * 1024 * 1024;
static_assert(sizeof(size_t) <= sizeof(uint64_t));

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

absl::StatusOr<ScopedFd> OpenRegularFile(const std::filesystem::path& path,
                                         int flags, mode_t mode = 0) {
  int fd;
  do {
    fd = open(path.c_str(), flags | ExtraOpenFlags(), mode);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to open DPM checkpoint file ",
                            path.string()));
  }
  struct stat stat_buffer;
  if (fstat(fd, &stat_buffer) != 0) {
    const int saved_errno = errno;
    close(fd);
    return absl::ErrnoToStatus(
        saved_errno,
        absl::StrCat("Unable to inspect DPM checkpoint file ",
                     path.string()));
  }
  if (!S_ISREG(stat_buffer.st_mode)) {
    close(fd);
    return absl::FailedPreconditionError(absl::StrCat(
        "DPM checkpoint path is not a regular file: ", path.string()));
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
        errno, absl::StrCat("Unable to lock DPM checkpoint repository ",
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
        errno, absl::StrCat("Unable to open DPM checkpoint directory ",
                            directory.string()));
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
      return absl::DataLossError(absl::StrCat("Truncated ", description));
    }
    consumed += static_cast<size_t>(read_count);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> AllocateFileBytes(
    uint64_t size, absl::string_view description) {
  if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
    if (size > std::numeric_limits<size_t>::max()) {
      return absl::ResourceExhaustedError(absl::StrCat(
          description, " exceeds addressable process memory."));
    }
  }
  const size_t allocation_size = static_cast<size_t>(size);
  if (allocation_size > std::string().max_size()) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds the string allocation limit."));
  }
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
  try {
    return std::string(allocation_size, '\0');
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Unable to allocate memory for ", description, "."));
  } catch (const std::length_error&) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds the string allocation limit."));
  }
#else
  return std::string(allocation_size, '\0');
#endif
}

absl::StatusOr<std::string> ReadWholeFile(
    const std::filesystem::path& path, uint64_t size_limit,
    absl::string_view description,
    std::optional<uint64_t> expected_size = std::nullopt) {
  if (expected_size.has_value() && *expected_size > size_limit) {
    return absl::ResourceExhaustedError(absl::StrCat(
        description, " declares a size beyond the repository limit."));
  }
  ABSL_ASSIGN_OR_RETURN(ScopedFd fd, OpenRegularFile(path, O_RDONLY));
  struct stat stat_buffer;
  if (fstat(fd.get(), &stat_buffer) != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to inspect ", description));
  }
  if (stat_buffer.st_size < 0 ||
      static_cast<uint64_t>(stat_buffer.st_size) > size_limit) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds the repository size limit."));
  }
  const uint64_t observed_size = static_cast<uint64_t>(stat_buffer.st_size);
  if (expected_size.has_value() && observed_size != *expected_size) {
    return absl::DataLossError(
        absl::StrCat(description, " size does not match its descriptor."));
  }
  ABSL_ASSIGN_OR_RETURN(std::string bytes,
                        AllocateFileBytes(observed_size, description));
  ABSL_RETURN_IF_ERROR(
      ReadExactAt(fd.get(), 0, bytes.data(), bytes.size(), description));
  struct stat final_stat_buffer;
  if (fstat(fd.get(), &final_stat_buffer) != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to re-inspect ", description));
  }
  if (final_stat_buffer.st_size != stat_buffer.st_size) {
    return absl::DataLossError(
        absl::StrCat(description, " changed during an immutable read."));
  }
  return bytes;
}

absl::Status EnsureNonSymlinkDirectory(const std::filesystem::path& directory,
                                       absl::string_view description) {
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unable to create ", description, " ", directory.string(), ": ",
        error.message()));
  }
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(directory, error);
  if (error) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unable to inspect ", description, " ", directory.string(), ": ",
        error.message()));
  }
  if (std::filesystem::is_symlink(status)) {
    return absl::FailedPreconditionError(absl::StrCat(
        description, " must not be a symbolic link: ", directory.string()));
  }
  if (!std::filesystem::is_directory(status)) {
    return absl::FailedPreconditionError(absl::StrCat(
        description, " is not a directory: ", directory.string()));
  }
  return absl::OkStatus();
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
        errno, absl::StrCat("Unable to create checkpoint temporary file in ",
                            directory.string()));
  }
#ifdef FD_CLOEXEC
  if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) {
    const int saved_errno = errno;
    close(fd);
    unlink(writable.data());
    return absl::ErrnoToStatus(
        saved_errno, "Unable to mark checkpoint temporary file close-on-exec");
  }
#endif
  return std::make_pair(ScopedFd(fd),
                        std::filesystem::path(writable.data()));
}

absl::Status PublishCreateOnce(const std::filesystem::path& directory,
                               const std::filesystem::path& final_path,
                               absl::string_view bytes,
                               uint64_t existing_size_limit,
                               absl::string_view description) {
  absl::StatusOr<std::string> existing =
      ReadWholeFile(final_path, existing_size_limit, description);
  if (existing.ok()) {
    if (*existing == bytes) return SyncDirectory(directory);
    return absl::DataLossError(absl::StrCat(
        "Conflicting bytes already exist for content-addressed ",
        description, "."));
  }
  if (!absl::IsNotFound(existing.status())) return existing.status();

  ABSL_ASSIGN_OR_RETURN(auto temporary,
                        CreateTempFile(directory, "checkpoint"));
  ScopedFd fd = std::move(temporary.first);
  const std::filesystem::path temporary_path = std::move(temporary.second);
  absl::Status status = WriteAll(fd.get(), bytes, description);
  if (status.ok()) status = SyncFd(fd.get(), description);
  bool published = false;
  if (status.ok()) {
    int result;
    do {
      result = link(temporary_path.c_str(), final_path.c_str());
    } while (result != 0 && errno == EINTR);
    if (result == 0) {
      published = true;
    } else if (errno == EEXIST) {
      absl::StatusOr<std::string> raced =
          ReadWholeFile(final_path, existing_size_limit, description);
      if (!raced.ok()) {
        status = raced.status();
      } else if (*raced != bytes) {
        status = absl::DataLossError(absl::StrCat(
            "Conflicting bytes won create-once publication for ",
            description, "."));
      }
    } else {
      status = absl::ErrnoToStatus(
          errno, absl::StrCat("Unable to publish ", description));
    }
  }
  if (status.ok() && published) status = SyncDirectory(directory);
  const int unlink_result = unlink(temporary_path.c_str());
  if (status.ok() && unlink_result != 0 && errno != ENOENT) {
    status = absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to remove temporary ", description));
  }
  if (status.ok()) status = SyncDirectory(directory);
  return status;
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

absl::Status AppendString(absl::string_view value, std::string* output) {
  if (value.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError(
        "DPM checkpoint descriptor string is too large.");
  }
  AppendU32(static_cast<uint32_t>(value.size()), output);
  output->append(value.data(), value.size());
  return absl::OkStatus();
}

void AppendHash(const Hash256& hash, std::string* output) {
  output->append(reinterpret_cast<const char*>(hash.bytes.data()),
                 hash.bytes.size());
}

void AppendOptionalHash(const std::optional<Hash256>& hash,
                        std::string* output) {
  output->push_back(hash.has_value() ? '\1' : '\0');
  if (hash.has_value()) AppendHash(*hash, output);
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
    uint64_t value = 0;
    for (int i = 0; i < 8; ++i) {
      value = (value << 8) |
              static_cast<unsigned char>(bytes_[offset_ + i]);
    }
    offset_ += 8;
    return value;
  }

  absl::StatusOr<int64_t> ReadI64() {
    ABSL_ASSIGN_OR_RETURN(uint64_t value, ReadU64());
    return static_cast<int64_t>(value);
  }

  absl::StatusOr<std::string> ReadString() {
    ABSL_ASSIGN_OR_RETURN(uint32_t length, ReadU32());
    if (length > bytes_.size() - offset_) {
      return absl::DataLossError(
          "Truncated stored DPM descriptor string.");
    }
    std::string value(bytes_.data() + offset_, length);
    offset_ += length;
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

  absl::StatusOr<std::optional<Hash256>> ReadOptionalHash() {
    ABSL_ASSIGN_OR_RETURN(uint8_t present, ReadU8());
    if (present > 1) {
      return absl::DataLossError(
          "Stored DPM descriptor has a non-canonical optional-hash marker.");
    }
    if (present == 0) return std::nullopt;
    ABSL_ASSIGN_OR_RETURN(Hash256 hash, ReadHash());
    return std::optional<Hash256>(hash);
  }

 private:
  absl::Status Require(size_t count) const {
    if (count > bytes_.size() - offset_) {
      return absl::DataLossError("Truncated stored DPM descriptor.");
    }
    return absl::OkStatus();
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

absl::StatusOr<std::string> EncodeStoredDescriptor(
    const DPMSessionCheckpointDescriptor& descriptor) {
  ABSL_ASSIGN_OR_RETURN(Hash256 expected_id,
                        ComputeDPMSessionCheckpointId(descriptor));
  if (expected_id != descriptor.descriptor_id) {
    return absl::InvalidArgumentError(
        "DPM checkpoint descriptor id is not canonical.");
  }
  std::string bytes;
  bytes.append(kStoredDescriptorMagic.data(), kStoredDescriptorMagic.size());
  AppendU32(kStoredDescriptorFormatVersion, &bytes);
  AppendU32(descriptor.format_version, &bytes);
  AppendHash(descriptor.descriptor_id, &bytes);
  ABSL_RETURN_IF_ERROR(AppendString(descriptor.log_id, &bytes));
  bytes.push_back(static_cast<char>(descriptor.stage));
  AppendU64(descriptor.source_event_count, &bytes);
  AppendHash(descriptor.source_prefix_hash, &bytes);
  AppendU64(descriptor.response_event_index, &bytes);
  AppendHash(descriptor.projection_request_hash, &bytes);
  AppendHash(descriptor.projection_manifest_hash, &bytes);
  AppendHash(descriptor.correction_digest, &bytes);
  AppendHash(descriptor.agent_request_hash, &bytes);
  AppendHash(descriptor.agent_transcript_hash, &bytes);
  AppendHash(descriptor.session_identity.model_artifact_hash, &bytes);
  AppendHash(descriptor.session_identity.runtime_artifact_hash, &bytes);
  AppendHash(descriptor.session_identity.inference_profile_hash, &bytes);
  ABSL_RETURN_IF_ERROR(AppendString(descriptor.key_id, &bytes));
  AppendHash(descriptor.envelope_hash, &bytes);
  AppendU64(descriptor.envelope_size, &bytes);
  AppendI64(descriptor.created_unix_micros, &bytes);
  if (descriptor.format_version ==
          DPMSessionCheckpointDescriptor::kPreviousFormatVersion ||
      descriptor.format_version ==
          DPMSessionCheckpointDescriptor::kFormatVersion) {
    bytes.push_back(static_cast<char>(descriptor.replay_mode));
    bytes.push_back(static_cast<char>(descriptor.capture_origin));
    AppendOptionalHash(descriptor.restored_from_checkpoint_id, &bytes);
    AppendOptionalHash(descriptor.exact_profile_id, &bytes);
    AppendHash(descriptor.exact_profile_admission_record_id, &bytes);
    if (descriptor.format_version ==
        DPMSessionCheckpointDescriptor::kFormatVersion) {
      AppendHash(descriptor.capsule_restore_capability_id, &bytes);
    }
    AppendHash(descriptor.capsule_restore_admission_record_id, &bytes);
    if (descriptor.format_version ==
        DPMSessionCheckpointDescriptor::kFormatVersion) {
      AppendHash(descriptor.capsule_restore_coverage_id, &bytes);
    }
    AppendHash(descriptor.exact_request_execution_evidence_id, &bytes);
    bytes.push_back(static_cast<char>(descriptor.worker_prefill_mode));
    AppendHash(descriptor.execution_plan_hash, &bytes);
    AppendHash(descriptor.exact_output_evidence_hash, &bytes);
    bytes.push_back(descriptor.worker_provenance.has_value() ? '\1' : '\0');
    if (descriptor.worker_provenance.has_value()) {
      const DPMExactWorkerCheckpointProvenance& provenance =
          *descriptor.worker_provenance;
      AppendU32(provenance.run_index, &bytes);
      AppendHash(provenance.execution_plan_hash, &bytes);
      AppendHash(provenance.request_envelope_hash, &bytes);
      AppendHash(provenance.result_envelope_hash, &bytes);
      AppendU64(provenance.transient_envelope_size, &bytes);
      AppendHash(provenance.transient_envelope_hash, &bytes);
      AppendHash(provenance.output_evidence_hash, &bytes);
    }
  }
  if (bytes.size() > kMaxStoredDescriptorBytes) {
    return absl::ResourceExhaustedError(
        "DPM checkpoint descriptor exceeds its durable size limit.");
  }
  return bytes;
}

absl::StatusOr<DPMSessionCheckpointDescriptor> DecodeStoredDescriptor(
    absl::string_view bytes) {
  if (bytes.size() > kMaxStoredDescriptorBytes) {
    return absl::ResourceExhaustedError(
        "Stored DPM descriptor exceeds its durable size limit.");
  }
  if (bytes.size() < kStoredDescriptorMagic.size() ||
      std::memcmp(bytes.data(), kStoredDescriptorMagic.data(),
                  kStoredDescriptorMagic.size()) != 0) {
    return absl::DataLossError("Invalid stored DPM descriptor magic.");
  }
  CanonicalReader reader(bytes.substr(kStoredDescriptorMagic.size()));
  ABSL_ASSIGN_OR_RETURN(uint32_t stored_version, reader.ReadU32());
  if (stored_version != kStoredDescriptorFormatVersion) {
    return absl::FailedPreconditionError(
        "Unsupported stored DPM descriptor format version.");
  }
  DPMSessionCheckpointDescriptor descriptor;
  ABSL_ASSIGN_OR_RETURN(descriptor.format_version, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(descriptor.descriptor_id, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.log_id, reader.ReadString());
  ABSL_ASSIGN_OR_RETURN(uint8_t stage, reader.ReadU8());
  descriptor.stage = static_cast<DPMSessionCheckpointStage>(stage);
  ABSL_ASSIGN_OR_RETURN(descriptor.source_event_count, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(descriptor.source_prefix_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.response_event_index, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(descriptor.projection_request_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.projection_manifest_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.correction_digest, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.agent_request_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.agent_transcript_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.session_identity.model_artifact_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.session_identity.runtime_artifact_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.session_identity.inference_profile_hash,
                        reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.key_id, reader.ReadString());
  ABSL_ASSIGN_OR_RETURN(descriptor.envelope_hash, reader.ReadHash());
  ABSL_ASSIGN_OR_RETURN(descriptor.envelope_size, reader.ReadU64());
  ABSL_ASSIGN_OR_RETURN(descriptor.created_unix_micros, reader.ReadI64());

  switch (descriptor.format_version) {
    case DPMSessionCheckpointDescriptor::kLegacyFormatVersion:
      // These values are inferred rather than serialized in DPMDSC01 v1.
      descriptor.replay_mode = DPMReplayMode::kCanonicalWinnerReplay;
      descriptor.capture_origin =
          DPMCheckpointCaptureOrigin::kLiveParentSession;
      break;

    case DPMSessionCheckpointDescriptor::kPreviousFormatVersion:
    case DPMSessionCheckpointDescriptor::kFormatVersion: {
      uint8_t replay_mode;
      ABSL_ASSIGN_OR_RETURN(replay_mode, reader.ReadU8());
      descriptor.replay_mode = static_cast<DPMReplayMode>(replay_mode);
      uint8_t capture_origin;
      ABSL_ASSIGN_OR_RETURN(capture_origin, reader.ReadU8());
      descriptor.capture_origin =
          static_cast<DPMCheckpointCaptureOrigin>(capture_origin);
      ABSL_ASSIGN_OR_RETURN(descriptor.restored_from_checkpoint_id,
                            reader.ReadOptionalHash());
      ABSL_ASSIGN_OR_RETURN(descriptor.exact_profile_id,
                            reader.ReadOptionalHash());
      ABSL_ASSIGN_OR_RETURN(descriptor.exact_profile_admission_record_id,
                            reader.ReadHash());
      if (descriptor.format_version ==
          DPMSessionCheckpointDescriptor::kFormatVersion) {
        ABSL_ASSIGN_OR_RETURN(descriptor.capsule_restore_capability_id,
                              reader.ReadHash());
      }
      ABSL_ASSIGN_OR_RETURN(descriptor.capsule_restore_admission_record_id,
                            reader.ReadHash());
      if (descriptor.format_version ==
          DPMSessionCheckpointDescriptor::kFormatVersion) {
        ABSL_ASSIGN_OR_RETURN(descriptor.capsule_restore_coverage_id,
                              reader.ReadHash());
      }
      ABSL_ASSIGN_OR_RETURN(descriptor.exact_request_execution_evidence_id,
                            reader.ReadHash());
      uint8_t worker_prefill_mode;
      ABSL_ASSIGN_OR_RETURN(worker_prefill_mode, reader.ReadU8());
      descriptor.worker_prefill_mode =
          static_cast<DPMCheckpointWorkerPrefillMode>(worker_prefill_mode);
      ABSL_ASSIGN_OR_RETURN(descriptor.execution_plan_hash,
                            reader.ReadHash());
      ABSL_ASSIGN_OR_RETURN(descriptor.exact_output_evidence_hash,
                            reader.ReadHash());
      uint8_t has_worker_provenance;
      ABSL_ASSIGN_OR_RETURN(has_worker_provenance, reader.ReadU8());
      if (has_worker_provenance > 1) {
        return absl::DataLossError(
            "Stored DPM descriptor has a non-canonical worker-provenance "
            "marker.");
      }
      if (has_worker_provenance != 0) {
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
        descriptor.worker_provenance = std::move(provenance);
      }
      break;
    }

    default:
      return absl::FailedPreconditionError(
          "Unsupported DPM session checkpoint descriptor version.");
  }
  if (!reader.empty()) {
    return absl::DataLossError("Trailing bytes in stored DPM descriptor.");
  }
  ABSL_ASSIGN_OR_RETURN(Hash256 expected_id,
                        ComputeDPMSessionCheckpointId(descriptor));
  if (expected_id != descriptor.descriptor_id) {
    return absl::DataLossError(
        "Stored DPM descriptor id is not its canonical content address.");
  }
  ABSL_ASSIGN_OR_RETURN(std::string reencoded,
                        EncodeStoredDescriptor(descriptor));
  if (reencoded != bytes) {
    return absl::DataLossError("Non-canonical stored DPM descriptor.");
  }
  return descriptor;
}

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
}

}  // namespace

FilesystemDPMSessionCheckpointRepository::
    FilesystemDPMSessionCheckpointRepository(std::filesystem::path directory)
    : directory_(std::move(directory)),
      envelope_directory_(directory_ / "envelopes"),
      descriptor_directory_(directory_ / "descriptors"),
      lock_path_(directory_ / ".repository.lock") {}

absl::StatusOr<
    std::unique_ptr<FilesystemDPMSessionCheckpointRepository>>
FilesystemDPMSessionCheckpointRepository::Create(
    std::filesystem::path directory) {
  if (directory.empty()) {
    return absl::InvalidArgumentError(
        "DPM checkpoint repository requires a storage directory.");
  }
  std::error_code path_error;
  directory = std::filesystem::absolute(directory, path_error);
  if (path_error) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to resolve DPM checkpoint directory: ",
        path_error.message()));
  }
  directory = std::filesystem::weakly_canonical(directory, path_error);
  if (path_error) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to normalize DPM checkpoint directory: ",
        path_error.message()));
  }
  auto repository =
      std::unique_ptr<FilesystemDPMSessionCheckpointRepository>(
          new FilesystemDPMSessionCheckpointRepository(std::move(directory)));
  ABSL_RETURN_IF_ERROR(repository->Initialize());
  return repository;
}

absl::Status FilesystemDPMSessionCheckpointRepository::Initialize() {
  ABSL_RETURN_IF_ERROR(EnsureNonSymlinkDirectory(
      envelope_directory_, "DPM envelope directory"));
  ABSL_RETURN_IF_ERROR(EnsureNonSymlinkDirectory(
      descriptor_directory_, "DPM descriptor directory"));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireFileLock(lock_path_, true, true));
  ABSL_RETURN_IF_ERROR(SyncDirectory(envelope_directory_));
  ABSL_RETURN_IF_ERROR(SyncDirectory(descriptor_directory_));
  ABSL_RETURN_IF_ERROR(SyncDirectory(directory_));
  return SyncDirectory(directory_.parent_path());
}

absl::Status FilesystemDPMSessionCheckpointRepository::Put(
    const DPMSessionCheckpointArtifact& artifact) {
  ABSL_RETURN_IF_ERROR(ValidateDPMSessionCheckpointArtifact(artifact));
  if (artifact.authenticated_envelope.size() > kMaxStoredEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "DPM session checkpoint envelope exceeds the repository limit.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string descriptor_bytes,
      EncodeStoredDescriptor(artifact.descriptor));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireFileLock(lock_path_, true, false));

  const std::filesystem::path envelope_path =
      envelope_directory_ /
      absl::StrCat(artifact.descriptor.envelope_hash.ToHex(), ".envelope");
  ABSL_RETURN_IF_ERROR(PublishCreateOnce(
      envelope_directory_, envelope_path, artifact.authenticated_envelope,
      kMaxStoredEnvelopeBytes, "DPM checkpoint envelope"));

  // Descriptor publication is the visibility boundary. The preceding call
  // has fsynced both envelope contents and its directory entry.
  const std::filesystem::path descriptor_path =
      descriptor_directory_ /
      absl::StrCat(artifact.descriptor.descriptor_id.ToHex(), ".descriptor");
  return PublishCreateOnce(descriptor_directory_, descriptor_path,
                           descriptor_bytes, kMaxStoredDescriptorBytes,
                           "DPM checkpoint descriptor");
}

absl::StatusOr<DPMSessionCheckpointArtifact>
FilesystemDPMSessionCheckpointRepository::Get(
    const Hash256& descriptor_id) const {
  if (IsZeroHash(descriptor_id)) {
    return absl::InvalidArgumentError(
        "Cannot load an empty DPM checkpoint descriptor id.");
  }
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireFileLock(lock_path_, false, false));
  const std::filesystem::path descriptor_path =
      descriptor_directory_ /
      absl::StrCat(descriptor_id.ToHex(), ".descriptor");
  ABSL_ASSIGN_OR_RETURN(
      std::string descriptor_bytes,
      ReadWholeFile(descriptor_path, kMaxStoredDescriptorBytes,
                    "DPM checkpoint descriptor"));
  ABSL_ASSIGN_OR_RETURN(DPMSessionCheckpointDescriptor descriptor,
                        DecodeStoredDescriptor(descriptor_bytes));
  if (descriptor.descriptor_id != descriptor_id) {
    return absl::DataLossError(
        "DPM descriptor filename and canonical identity disagree.");
  }
  const std::filesystem::path envelope_path =
      envelope_directory_ /
      absl::StrCat(descriptor.envelope_hash.ToHex(), ".envelope");
  ABSL_ASSIGN_OR_RETURN(
      std::string envelope,
      ReadWholeFile(envelope_path, kMaxStoredEnvelopeBytes,
                    "DPM checkpoint envelope", descriptor.envelope_size));
  DPMSessionCheckpointArtifact artifact{std::move(descriptor),
                                        std::move(envelope)};
  ABSL_RETURN_IF_ERROR(ValidateDPMSessionCheckpointArtifact(artifact));
  return artifact;
}

}  // namespace litert::lm
