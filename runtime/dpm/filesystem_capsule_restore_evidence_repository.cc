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

#include "runtime/dpm/filesystem_capsule_restore_evidence_repository.h"

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
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

bool IsZeroHash(const Hash256& hash) {
  uint8_t combined = 0;
  for (uint8_t byte : hash.bytes) combined |= byte;
  return combined == 0;
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
    const int value = fd_;
    fd_ = -1;
    return value;
  }
  void Reset(int fd = -1) {
    if (fd_ >= 0) {
      while (close(fd_) != 0 && errno == EINTR) {
      }
    }
    fd_ = fd;
  }

 private:
  int fd_ = -1;
};

class ScopedFileLock {
 public:
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

absl::StatusOr<ScopedFd> OpenRegularOwnerOnlyFile(
    const std::filesystem::path& path, int flags, mode_t mode = 0) {
  int fd;
  do {
    fd = open(path.c_str(), flags | ExtraOpenFlags(), mode);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno,
        absl::StrCat("Unable to open Coverage V2 evidence file ",
                     path.string()));
  }
  struct stat file_stat;
  if (fstat(fd, &file_stat) != 0) {
    const int saved_errno = errno;
    close(fd);
    return absl::ErrnoToStatus(
        saved_errno,
        absl::StrCat("Unable to inspect Coverage V2 evidence file ",
                     path.string()));
  }
  if (!S_ISREG(file_stat.st_mode)) {
    close(fd);
    return absl::FailedPreconditionError(absl::StrCat(
        "Coverage V2 evidence path is not a regular file: ", path.string()));
  }
  if (file_stat.st_uid != geteuid() ||
      (file_stat.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
    close(fd);
    return absl::PermissionDeniedError(
        "Coverage V2 evidence files must be current-user-owned and "
        "owner-only.");
  }
  return ScopedFd(fd);
}

absl::StatusOr<ScopedFileLock> AcquireLock(
    const std::filesystem::path& path, bool exclusive) {
  ABSL_ASSIGN_OR_RETURN(
      ScopedFd fd,
      OpenRegularOwnerOnlyFile(path, O_RDWR | O_CREAT,
                               S_IRUSR | S_IWUSR));
  int result;
  do {
    result = flock(fd.get(), exclusive ? LOCK_EX : LOCK_SH);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to lock Coverage V2 evidence repository.");
  }
  return ScopedFileLock(std::move(fd));
}

absl::Status ValidateSecureDirectory(const std::filesystem::path& directory,
                                     absl::string_view description) {
  struct stat directory_stat;
  if (lstat(directory.c_str(), &directory_stat) != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to inspect ", description, "."));
  }
  if (S_ISLNK(directory_stat.st_mode) ||
      !S_ISDIR(directory_stat.st_mode)) {
    return absl::FailedPreconditionError(
        absl::StrCat(description, " must be a non-symlink directory."));
  }
  if (directory_stat.st_uid != geteuid()) {
    return absl::PermissionDeniedError(
        absl::StrCat(description, " must be owned by the current user."));
  }
  if ((directory_stat.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
    return absl::PermissionDeniedError(
        absl::StrCat(description, " must be owner-only."));
  }
  if ((directory_stat.st_mode & S_IRWXU) != S_IRWXU) {
    return absl::PermissionDeniedError(
        absl::StrCat(description,
                     " must be owner-readable, writable, and searchable."));
  }
  return absl::OkStatus();
}

absl::Status ValidateNoSymlinkPathComponents(
    const std::filesystem::path& absolute_path) {
  std::filesystem::path cursor = absolute_path.root_path();
  for (const std::filesystem::path& component : absolute_path.relative_path()) {
    cursor /= component;
    struct stat component_stat;
    if (lstat(cursor.c_str(), &component_stat) != 0) {
      return absl::ErrnoToStatus(
          errno,
          "Unable to inspect a Coverage V2 evidence-root path component.");
    }
    if (S_ISLNK(component_stat.st_mode)) {
      return absl::FailedPreconditionError(
          "Coverage V2 evidence root must not traverse symbolic links.");
    }
  }
  return absl::OkStatus();
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
        errno, "Unable to open Coverage V2 evidence directory.");
  }
  ScopedFd scoped(fd);
  return SyncFd(fd, "Coverage V2 evidence directory");
}

absl::Status EnsureOwnedDirectory(const std::filesystem::path& parent,
                                  const std::filesystem::path& directory,
                                  absl::string_view description) {
  ABSL_RETURN_IF_ERROR(
      ValidateSecureDirectory(parent, "Coverage V2 evidence parent"));
  bool created = false;
  int result;
  do {
    result = mkdir(directory.c_str(), S_IRWXU);
  } while (result != 0 && errno == EINTR);
  if (result == 0) {
    created = true;
  } else if (errno != EEXIST) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to create ", description, "."));
  }
  ABSL_RETURN_IF_ERROR(ValidateSecureDirectory(directory, description));
  if (created) {
    ABSL_RETURN_IF_ERROR(SyncDirectory(directory));
    ABSL_RETURN_IF_ERROR(SyncDirectory(parent));
  }
  return absl::OkStatus();
}

absl::Status ValidateRepositoryTree(
    const std::filesystem::path& root,
    const std::filesystem::path& product_directory,
    const std::filesystem::path& version_directory,
    const std::filesystem::path& records_directory) {
  ABSL_RETURN_IF_ERROR(ValidateNoSymlinkPathComponents(root));
  ABSL_RETURN_IF_ERROR(
      ValidateSecureDirectory(root, "Coverage V2 evidence root"));
  ABSL_RETURN_IF_ERROR(ValidateSecureDirectory(
      product_directory, "Coverage V2 evidence product directory"));
  ABSL_RETURN_IF_ERROR(ValidateSecureDirectory(
      version_directory, "Coverage V2 evidence version directory"));
  return ValidateSecureDirectory(records_directory,
                                 "Coverage V2 evidence records directory");
}

absl::Status ValidateV3RepositoryTree(
    const std::filesystem::path& root,
    const std::filesystem::path& product_directory,
    const std::filesystem::path& version_directory,
    const std::filesystem::path& capture_records_directory,
    const std::filesystem::path& restore_records_directory) {
  ABSL_RETURN_IF_ERROR(ValidateNoSymlinkPathComponents(root));
  ABSL_RETURN_IF_ERROR(
      ValidateSecureDirectory(root, "Coverage V3 evidence root"));
  ABSL_RETURN_IF_ERROR(ValidateSecureDirectory(
      product_directory, "Coverage V3 evidence product directory"));
  ABSL_RETURN_IF_ERROR(ValidateSecureDirectory(
      version_directory, "Coverage V3 evidence version directory"));
  ABSL_RETURN_IF_ERROR(ValidateSecureDirectory(
      capture_records_directory,
      "Coverage V3 capture-evidence records directory"));
  return ValidateSecureDirectory(
      restore_records_directory,
      "Coverage V3 restore-evidence records directory");
}

absl::Status WriteAll(int fd, absl::string_view bytes) {
  size_t offset = 0;
  while (offset < bytes.size()) {
    ssize_t count;
    do {
      count = write(fd, bytes.data() + offset, bytes.size() - offset);
    } while (count < 0 && errno == EINTR);
    if (count < 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to write Coverage V2 capture evidence.");
    }
    if (count == 0) {
      return absl::DataLossError(
          "Zero-byte write while publishing Coverage V2 capture evidence.");
    }
    offset += static_cast<size_t>(count);
  }
  return absl::OkStatus();
}

absl::Status ReadExactAt(int fd, char* output, size_t size) {
  size_t offset = 0;
  while (offset < size) {
    ssize_t count;
    do {
      count = pread(fd, output + offset, size - offset,
                    static_cast<off_t>(offset));
    } while (count < 0 && errno == EINTR);
    if (count < 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to read Coverage V2 capture evidence.");
    }
    if (count == 0) {
      return absl::DataLossError(
          "Coverage V2 capture-evidence file is truncated.");
    }
    offset += static_cast<size_t>(count);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> AllocateBytes(uint64_t size) {
  if constexpr (sizeof(size_t) < sizeof(uint64_t)) {
    if (size > (std::numeric_limits<size_t>::max)()) {
      return absl::ResourceExhaustedError(
          "Coverage V2 capture evidence exceeds addressable memory.");
    }
  }
  const size_t allocation_size = static_cast<size_t>(size);
  if (allocation_size > std::string().max_size()) {
    return absl::ResourceExhaustedError(
        "Coverage V2 capture evidence exceeds the string allocation limit.");
  }
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
  try {
    return std::string(allocation_size, '\0');
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError(
        "Unable to allocate Coverage V2 capture-evidence storage.");
  } catch (const std::length_error&) {
    return absl::ResourceExhaustedError(
        "Coverage V2 capture evidence exceeds the allocation limit.");
  }
#else
  return std::string(allocation_size, '\0');
#endif
}

absl::StatusOr<std::string> ReadWholeFile(
    const std::filesystem::path& path) {
  ABSL_ASSIGN_OR_RETURN(ScopedFd fd,
                        OpenRegularOwnerOnlyFile(path, O_RDONLY));
  struct stat before;
  if (fstat(fd.get(), &before) != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to inspect Coverage V2 capture evidence.");
  }
  if (before.st_size < 0 ||
      static_cast<uint64_t>(before.st_size) >
          kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "Coverage V2 capture-evidence file exceeds its storage bound.");
  }
  if ((before.st_mode & S_IRUSR) == 0 ||
      (before.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0) {
    return absl::PermissionDeniedError(
        "Coverage V2 capture-evidence records must be owner-readable and "
        "read-only.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string bytes,
      AllocateBytes(static_cast<uint64_t>(before.st_size)));
  ABSL_RETURN_IF_ERROR(ReadExactAt(fd.get(), bytes.data(), bytes.size()));
  struct stat after;
  if (fstat(fd.get(), &after) != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to re-inspect Coverage V2 capture evidence.");
  }
  if (after.st_dev != before.st_dev || after.st_ino != before.st_ino ||
      after.st_size != before.st_size || after.st_mtime != before.st_mtime ||
      after.st_ctime != before.st_ctime) {
    return absl::DataLossError(
        "Coverage V2 capture evidence changed during its immutable read.");
  }
  return bytes;
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
        errno, "Unable to create Coverage V2 evidence temporary file.");
  }
  ScopedFd scoped(fd);
  if (fchmod(fd, S_IRUSR | S_IWUSR) != 0) {
    const int saved_errno = errno;
    unlink(writable.data());
    return absl::ErrnoToStatus(
        saved_errno, "Unable to protect Coverage V2 evidence temporary file.");
  }
  const int descriptor_flags = fcntl(fd, F_GETFD);
  if (descriptor_flags < 0 ||
      fcntl(fd, F_SETFD, descriptor_flags | FD_CLOEXEC) != 0) {
    const int saved_errno = errno;
    unlink(writable.data());
    return absl::ErrnoToStatus(
        saved_errno,
        "Unable to mark Coverage V2 evidence temporary file close-on-exec.");
  }
  return std::make_pair(std::move(scoped),
                        std::filesystem::path(writable.data()));
}

std::filesystem::path RecordPath(const std::filesystem::path& directory,
                                 const Hash256& checkpoint_id,
                                 const Hash256& evidence_id) {
  return directory /
         absl::StrCat(checkpoint_id.ToHex(), "-", evidence_id.ToHex(),
                      ".capture-v2");
}

absl::Status ValidateStoredIdentity(
    const CapsuleCaptureEvidenceV2& evidence,
    const Hash256& checkpoint_id, const Hash256& evidence_id) {
  if (evidence.checkpoint_id != checkpoint_id ||
      evidence.evidence_id != evidence_id) {
    return absl::DataLossError(
        "Coverage V2 capture evidence does not match its composite "
        "checkpoint/evidence storage key.");
  }
  return absl::OkStatus();
}

absl::Status PublishCreateOnce(
    const std::filesystem::path& directory,
    const std::filesystem::path& final_path, absl::string_view bytes,
    const Hash256& checkpoint_id, const Hash256& evidence_id,
    const FreshWorkerAuthentication& authentication) {
  absl::StatusOr<std::string> existing = ReadWholeFile(final_path);
  if (existing.ok()) {
    if (*existing == bytes) return absl::OkStatus();
    ABSL_ASSIGN_OR_RETURN(
        const CapsuleCaptureEvidenceV2 decoded,
        DecodeAuthenticatedCapsuleCaptureEvidenceV2(*existing,
                                                     authentication));
    ABSL_RETURN_IF_ERROR(
        ValidateStoredIdentity(decoded, checkpoint_id, evidence_id));
    return absl::AlreadyExistsError(
        "Conflicting authenticated bytes already exist for this Coverage V2 "
        "checkpoint/evidence pair.");
  }
  if (!absl::IsNotFound(existing.status())) return existing.status();

  ABSL_ASSIGN_OR_RETURN(
      auto temporary,
      CreateTempFile(directory,
                     absl::StrCat(checkpoint_id.ToHex(), "-",
                                  evidence_id.ToHex())));
  ScopedFd fd = std::move(temporary.first);
  const std::filesystem::path temporary_path = std::move(temporary.second);
  const auto remove_temporary = [&temporary_path]() {
    int result;
    do {
      result = unlink(temporary_path.c_str());
    } while (result != 0 && errno == EINTR);
  };

  absl::Status status = WriteAll(fd.get(), bytes);
  if (status.ok()) {
    int result;
    do {
      result = fchmod(fd.get(), S_IRUSR);
    } while (result != 0 && errno == EINTR);
    if (result != 0) {
      status = absl::ErrnoToStatus(
          errno, "Unable to make Coverage V2 capture evidence read-only.");
    }
  }
  if (status.ok()) {
    status = SyncFd(fd.get(), "Coverage V2 capture evidence");
  }
  fd.Reset();
  if (!status.ok()) {
    remove_temporary();
    return status;
  }

  int link_result;
  do {
    link_result = link(temporary_path.c_str(), final_path.c_str());
  } while (link_result != 0 && errno == EINTR);
  if (link_result != 0) {
    const int saved_errno = errno;
    remove_temporary();
    if (saved_errno != EEXIST) {
      return absl::ErrnoToStatus(
          saved_errno, "Unable to publish Coverage V2 capture evidence.");
    }
    ABSL_ASSIGN_OR_RETURN(const std::string raced, ReadWholeFile(final_path));
    if (raced == bytes) return absl::OkStatus();
    ABSL_ASSIGN_OR_RETURN(
        const CapsuleCaptureEvidenceV2 decoded,
        DecodeAuthenticatedCapsuleCaptureEvidenceV2(raced, authentication));
    ABSL_RETURN_IF_ERROR(
        ValidateStoredIdentity(decoded, checkpoint_id, evidence_id));
    return absl::AlreadyExistsError(
        "Conflicting authenticated bytes won create-once publication for this "
        "Coverage V2 checkpoint/evidence pair.");
  }
  ABSL_RETURN_IF_ERROR(SyncDirectory(directory));
  remove_temporary();
  return SyncDirectory(directory);
}

static_assert(
    kMaximumAuthenticatedCapsuleCaptureEvidenceV3EnvelopeBytes ==
        kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes);
static_assert(
    kMaximumAuthenticatedCapsuleRestoreEvidenceV3EnvelopeBytes ==
        kMaximumAuthenticatedCapsuleCaptureEvidenceV2EnvelopeBytes);

std::filesystem::path CaptureV3RecordPath(
    const std::filesystem::path& directory, const Hash256& checkpoint_id,
    const Hash256& evidence_id) {
  return directory /
         absl::StrCat(checkpoint_id.ToHex(), "-", evidence_id.ToHex(),
                      ".capture-v3");
}

std::filesystem::path RestoreV3RecordPath(
    const std::filesystem::path& directory, const Hash256& checkpoint_id,
    const Hash256& evidence_id) {
  return directory /
         absl::StrCat(checkpoint_id.ToHex(), "-", evidence_id.ToHex(),
                      ".restore-v3");
}

absl::Status ValidateStoredCaptureV3Identity(
    const CapsuleCaptureEvidenceV3& evidence,
    const Hash256& checkpoint_id, const Hash256& evidence_id) {
  if (evidence.checkpoint_id != checkpoint_id ||
      evidence.evidence_id != evidence_id) {
    return absl::DataLossError(
        "Coverage V3 capture evidence does not match its composite "
        "checkpoint/evidence storage key.");
  }
  return absl::OkStatus();
}

absl::Status ValidateStoredRestoreV3Identity(
    const CapsuleRestoreEvidenceV3& evidence,
    const Hash256& checkpoint_id, const Hash256& evidence_id) {
  if (evidence.plan.checkpoint_id != checkpoint_id ||
      evidence.evidence_id != evidence_id) {
    return absl::DataLossError(
        "Coverage V3 restore evidence does not match its composite "
        "checkpoint/evidence storage key.");
  }
  return absl::OkStatus();
}

template <typename Evidence, typename Decode, typename ValidateIdentity>
absl::Status PublishCreateOnceV3(
    const std::filesystem::path& directory,
    const std::filesystem::path& final_path, absl::string_view bytes,
    const Hash256& checkpoint_id, const Hash256& evidence_id,
    const FreshWorkerAuthentication& authentication, Decode decode,
    ValidateIdentity validate_identity, absl::string_view description) {
  // All three evidence envelope bounds are deliberately identical. Reusing
  // the V2 immutable reader preserves its owner-only, no-follow, bounded-read,
  // and before/after inode checks without allowing V2/V3 wire reinterpretation.
  absl::StatusOr<std::string> existing = ReadWholeFile(final_path);
  if (existing.ok()) {
    if (*existing == bytes) return absl::OkStatus();
    ABSL_ASSIGN_OR_RETURN(const Evidence decoded,
                          decode(*existing, authentication));
    ABSL_RETURN_IF_ERROR(
        validate_identity(decoded, checkpoint_id, evidence_id));
    return absl::AlreadyExistsError(absl::StrCat(
        "Conflicting authenticated bytes already exist for this ",
        description, " checkpoint/evidence pair."));
  }
  if (!absl::IsNotFound(existing.status())) return existing.status();

  ABSL_ASSIGN_OR_RETURN(
      auto temporary,
      CreateTempFile(directory,
                     absl::StrCat(checkpoint_id.ToHex(), "-",
                                  evidence_id.ToHex())));
  ScopedFd fd = std::move(temporary.first);
  const std::filesystem::path temporary_path = std::move(temporary.second);
  const auto remove_temporary = [&temporary_path]() {
    int result;
    do {
      result = unlink(temporary_path.c_str());
    } while (result != 0 && errno == EINTR);
  };

  absl::Status status = WriteAll(fd.get(), bytes);
  if (status.ok()) {
    int result;
    do {
      result = fchmod(fd.get(), S_IRUSR);
    } while (result != 0 && errno == EINTR);
    if (result != 0) {
      status = absl::ErrnoToStatus(
          errno, absl::StrCat("Unable to make ", description,
                              " evidence read-only."));
    }
  }
  if (status.ok()) {
    status = SyncFd(fd.get(), absl::StrCat(description, " evidence"));
  }
  fd.Reset();
  if (!status.ok()) {
    remove_temporary();
    return status;
  }

  int link_result;
  do {
    link_result = link(temporary_path.c_str(), final_path.c_str());
  } while (link_result != 0 && errno == EINTR);
  if (link_result != 0) {
    const int saved_errno = errno;
    remove_temporary();
    if (saved_errno != EEXIST) {
      return absl::ErrnoToStatus(
          saved_errno,
          absl::StrCat("Unable to publish ", description, " evidence."));
    }
    ABSL_ASSIGN_OR_RETURN(const std::string raced, ReadWholeFile(final_path));
    if (raced == bytes) return absl::OkStatus();
    ABSL_ASSIGN_OR_RETURN(const Evidence decoded,
                          decode(raced, authentication));
    ABSL_RETURN_IF_ERROR(
        validate_identity(decoded, checkpoint_id, evidence_id));
    return absl::AlreadyExistsError(absl::StrCat(
        "Conflicting authenticated bytes won create-once publication for this ",
        description, " checkpoint/evidence pair."));
  }
  ABSL_RETURN_IF_ERROR(SyncDirectory(directory));
  remove_temporary();
  return SyncDirectory(directory);
}

}  // namespace

FilesystemCapsuleRestoreEvidenceRepository::
    FilesystemCapsuleRestoreEvidenceRepository(std::filesystem::path root)
    : root_(std::move(root)),
      product_directory_(root_ / "capsule-restore-evidence"),
      version_directory_(product_directory_ / "v2"),
      records_directory_(version_directory_ / "records"),
      lock_path_(version_directory_ / "repository.lock"),
      v3_version_directory_(product_directory_ / "v3"),
      v3_capture_records_directory_(v3_version_directory_ /
                                    "capture-records"),
      v3_restore_records_directory_(v3_version_directory_ /
                                    "restore-records"),
      v3_lock_path_(v3_version_directory_ / "repository.lock") {}

absl::StatusOr<
    std::unique_ptr<FilesystemCapsuleRestoreEvidenceRepository>>
FilesystemCapsuleRestoreEvidenceRepository::Create(
    std::filesystem::path root) {
  if (root.empty() || !root.is_absolute()) {
    return absl::InvalidArgumentError(
        "Coverage V2 evidence repository requires an absolute existing root.");
  }
  for (const std::filesystem::path& component : root.relative_path()) {
    if (component == "." || component == "..") {
      return absl::InvalidArgumentError(
          "Coverage V2 evidence root must not contain dot components.");
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateNoSymlinkPathComponents(root));
  ABSL_RETURN_IF_ERROR(
      ValidateSecureDirectory(root, "Coverage V2 evidence root"));
  auto repository =
      std::unique_ptr<FilesystemCapsuleRestoreEvidenceRepository>(
          new FilesystemCapsuleRestoreEvidenceRepository(std::move(root)));
  ABSL_RETURN_IF_ERROR(repository->Initialize());
  return repository;
}

absl::Status FilesystemCapsuleRestoreEvidenceRepository::Initialize() {
  ABSL_RETURN_IF_ERROR(EnsureOwnedDirectory(
      root_, product_directory_, "Coverage V2 evidence product directory"));
  ABSL_RETURN_IF_ERROR(EnsureOwnedDirectory(
      product_directory_, version_directory_,
      "Coverage V2 evidence version directory"));
  ABSL_RETURN_IF_ERROR(EnsureOwnedDirectory(
      version_directory_, records_directory_,
      "Coverage V2 evidence records directory"));
  {
    ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                          AcquireLock(lock_path_, true));
    ABSL_RETURN_IF_ERROR(ValidateRepositoryTree(
        root_, product_directory_, version_directory_, records_directory_));
    ABSL_RETURN_IF_ERROR(SyncDirectory(version_directory_));
  }

  ABSL_RETURN_IF_ERROR(EnsureOwnedDirectory(
      product_directory_, v3_version_directory_,
      "Coverage V3 evidence version directory"));
  ABSL_RETURN_IF_ERROR(EnsureOwnedDirectory(
      v3_version_directory_, v3_capture_records_directory_,
      "Coverage V3 capture-evidence records directory"));
  ABSL_RETURN_IF_ERROR(EnsureOwnedDirectory(
      v3_version_directory_, v3_restore_records_directory_,
      "Coverage V3 restore-evidence records directory"));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireLock(v3_lock_path_, true));
  ABSL_RETURN_IF_ERROR(ValidateV3RepositoryTree(
      root_, product_directory_, v3_version_directory_,
      v3_capture_records_directory_, v3_restore_records_directory_));
  return SyncDirectory(v3_version_directory_);
}

absl::Status FilesystemCapsuleRestoreEvidenceRepository::PutIfAbsent(
    const CapsuleCaptureEvidenceV2& evidence,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceV2(evidence));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_ASSIGN_OR_RETURN(
      const std::string bytes,
      EncodeAuthenticatedCapsuleCaptureEvidenceV2(evidence, authentication));
  ABSL_RETURN_IF_ERROR(ValidateRepositoryTree(
      root_, product_directory_, version_directory_, records_directory_));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock, AcquireLock(lock_path_, true));
  ABSL_RETURN_IF_ERROR(ValidateRepositoryTree(
      root_, product_directory_, version_directory_, records_directory_));
  const std::filesystem::path path =
      RecordPath(records_directory_, evidence.checkpoint_id,
                 evidence.evidence_id);
  return PublishCreateOnce(records_directory_, path, bytes,
                           evidence.checkpoint_id, evidence.evidence_id,
                           authentication);
}

absl::StatusOr<CapsuleCaptureEvidenceV2>
FilesystemCapsuleRestoreEvidenceRepository::Get(
    const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
    const FreshWorkerAuthentication& authentication) const {
  if (IsZeroHash(checkpoint_id) || IsZeroHash(expected_evidence_id)) {
    return absl::InvalidArgumentError(
        "Coverage V2 evidence lookup requires exact nonzero checkpoint and "
        "evidence IDs.");
  }
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_RETURN_IF_ERROR(ValidateRepositoryTree(
      root_, product_directory_, version_directory_, records_directory_));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock, AcquireLock(lock_path_, false));
  ABSL_RETURN_IF_ERROR(ValidateRepositoryTree(
      root_, product_directory_, version_directory_, records_directory_));
  const std::filesystem::path path =
      RecordPath(records_directory_, checkpoint_id, expected_evidence_id);
  ABSL_ASSIGN_OR_RETURN(const std::string bytes, ReadWholeFile(path));
  ABSL_ASSIGN_OR_RETURN(
      CapsuleCaptureEvidenceV2 evidence,
      DecodeAuthenticatedCapsuleCaptureEvidenceV2(bytes, authentication));
  ABSL_RETURN_IF_ERROR(
      ValidateStoredIdentity(evidence, checkpoint_id, expected_evidence_id));
  return evidence;
}

absl::Status
FilesystemCapsuleRestoreEvidenceRepository::PutCaptureV3IfAbsent(
    const CapsuleCaptureEvidenceV3& evidence,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleCaptureEvidenceV3(evidence));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_ASSIGN_OR_RETURN(
      const std::string bytes,
      EncodeAuthenticatedCapsuleCaptureEvidenceV3(evidence, authentication));
  ABSL_RETURN_IF_ERROR(ValidateV3RepositoryTree(
      root_, product_directory_, v3_version_directory_,
      v3_capture_records_directory_, v3_restore_records_directory_));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireLock(v3_lock_path_, true));
  ABSL_RETURN_IF_ERROR(ValidateV3RepositoryTree(
      root_, product_directory_, v3_version_directory_,
      v3_capture_records_directory_, v3_restore_records_directory_));
  const std::filesystem::path path = CaptureV3RecordPath(
      v3_capture_records_directory_, evidence.checkpoint_id,
      evidence.evidence_id);
  return PublishCreateOnceV3<CapsuleCaptureEvidenceV3>(
      v3_capture_records_directory_, path, bytes, evidence.checkpoint_id,
      evidence.evidence_id, authentication,
      &DecodeAuthenticatedCapsuleCaptureEvidenceV3,
      &ValidateStoredCaptureV3Identity, "Coverage V3 capture evidence");
}

absl::StatusOr<CapsuleCaptureEvidenceV3>
FilesystemCapsuleRestoreEvidenceRepository::GetCaptureV3(
    const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
    const FreshWorkerAuthentication& authentication) const {
  if (IsZeroHash(checkpoint_id) || IsZeroHash(expected_evidence_id)) {
    return absl::InvalidArgumentError(
        "Coverage V3 capture-evidence lookup requires exact nonzero "
        "checkpoint and evidence IDs.");
  }
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_RETURN_IF_ERROR(ValidateV3RepositoryTree(
      root_, product_directory_, v3_version_directory_,
      v3_capture_records_directory_, v3_restore_records_directory_));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireLock(v3_lock_path_, false));
  ABSL_RETURN_IF_ERROR(ValidateV3RepositoryTree(
      root_, product_directory_, v3_version_directory_,
      v3_capture_records_directory_, v3_restore_records_directory_));
  const std::filesystem::path path = CaptureV3RecordPath(
      v3_capture_records_directory_, checkpoint_id, expected_evidence_id);
  ABSL_ASSIGN_OR_RETURN(const std::string bytes, ReadWholeFile(path));
  ABSL_ASSIGN_OR_RETURN(
      CapsuleCaptureEvidenceV3 evidence,
      DecodeAuthenticatedCapsuleCaptureEvidenceV3(bytes, authentication));
  ABSL_RETURN_IF_ERROR(ValidateStoredCaptureV3Identity(
      evidence, checkpoint_id, expected_evidence_id));
  return evidence;
}

absl::Status
FilesystemCapsuleRestoreEvidenceRepository::PutRestoreV3IfAbsent(
    const CapsuleRestoreEvidenceV3& evidence,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateCapsuleRestoreEvidenceV3(evidence));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_ASSIGN_OR_RETURN(
      const std::string bytes,
      EncodeAuthenticatedCapsuleRestoreEvidenceV3(evidence, authentication));
  ABSL_RETURN_IF_ERROR(ValidateV3RepositoryTree(
      root_, product_directory_, v3_version_directory_,
      v3_capture_records_directory_, v3_restore_records_directory_));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireLock(v3_lock_path_, true));
  ABSL_RETURN_IF_ERROR(ValidateV3RepositoryTree(
      root_, product_directory_, v3_version_directory_,
      v3_capture_records_directory_, v3_restore_records_directory_));
  const std::filesystem::path path = RestoreV3RecordPath(
      v3_restore_records_directory_, evidence.plan.checkpoint_id,
      evidence.evidence_id);
  return PublishCreateOnceV3<CapsuleRestoreEvidenceV3>(
      v3_restore_records_directory_, path, bytes,
      evidence.plan.checkpoint_id, evidence.evidence_id, authentication,
      &DecodeAuthenticatedCapsuleRestoreEvidenceV3,
      &ValidateStoredRestoreV3Identity, "Coverage V3 restore evidence");
}

absl::StatusOr<CapsuleRestoreEvidenceV3>
FilesystemCapsuleRestoreEvidenceRepository::GetRestoreV3(
    const Hash256& checkpoint_id, const Hash256& expected_evidence_id,
    const FreshWorkerAuthentication& authentication) const {
  if (IsZeroHash(checkpoint_id) || IsZeroHash(expected_evidence_id)) {
    return absl::InvalidArgumentError(
        "Coverage V3 restore-evidence lookup requires exact nonzero "
        "checkpoint and evidence IDs.");
  }
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_RETURN_IF_ERROR(ValidateV3RepositoryTree(
      root_, product_directory_, v3_version_directory_,
      v3_capture_records_directory_, v3_restore_records_directory_));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock,
                        AcquireLock(v3_lock_path_, false));
  ABSL_RETURN_IF_ERROR(ValidateV3RepositoryTree(
      root_, product_directory_, v3_version_directory_,
      v3_capture_records_directory_, v3_restore_records_directory_));
  const std::filesystem::path path = RestoreV3RecordPath(
      v3_restore_records_directory_, checkpoint_id, expected_evidence_id);
  ABSL_ASSIGN_OR_RETURN(const std::string bytes, ReadWholeFile(path));
  ABSL_ASSIGN_OR_RETURN(
      CapsuleRestoreEvidenceV3 evidence,
      DecodeAuthenticatedCapsuleRestoreEvidenceV3(bytes, authentication));
  ABSL_RETURN_IF_ERROR(ValidateStoredRestoreV3Identity(
      evidence, checkpoint_id, expected_evidence_id));
  return evidence;
}

}  // namespace litert::lm
