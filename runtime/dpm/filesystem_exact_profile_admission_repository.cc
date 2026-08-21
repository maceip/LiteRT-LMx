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

#include "runtime/dpm/filesystem_exact_profile_admission_repository.h"

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

constexpr uint64_t kMaximumStoredAdmissionBytes =
    kMaximumFreshWorkerEnvelopeBytes;

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
    const int result = fd_;
    fd_ = -1;
    return result;
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

absl::StatusOr<ScopedFd> OpenRegularFile(const std::filesystem::path& path,
                                         int flags, mode_t mode = 0) {
  int fd;
  do {
    fd = open(path.c_str(), flags | ExtraOpenFlags(), mode);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno,
        absl::StrCat("Unable to open exact-profile admission file ",
                     path.string()));
  }
  struct stat file_stat;
  if (fstat(fd, &file_stat) != 0) {
    const int saved_errno = errno;
    close(fd);
    return absl::ErrnoToStatus(
        saved_errno,
        absl::StrCat("Unable to inspect exact-profile admission file ",
                     path.string()));
  }
  if (!S_ISREG(file_stat.st_mode)) {
    close(fd);
    return absl::FailedPreconditionError(absl::StrCat(
        "Exact-profile admission path is not a regular file: ",
        path.string()));
  }
  if (file_stat.st_uid != geteuid() ||
      (file_stat.st_mode & (S_IRWXG | S_IRWXO)) != 0) {
    close(fd);
    return absl::PermissionDeniedError(
        "Exact-profile admission files must be current-user-owned and owner-only.");
  }
  return ScopedFd(fd);
}

absl::StatusOr<ScopedFileLock> AcquireLock(
    const std::filesystem::path& path, bool exclusive) {
  ABSL_ASSIGN_OR_RETURN(
      ScopedFd fd,
      OpenRegularFile(path, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR));
  int result;
  do {
    result = flock(fd.get(), exclusive ? LOCK_EX : LOCK_SH);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to lock exact-profile admission repository.");
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
    return absl::PermissionDeniedError(absl::StrCat(
        description, " must be owner-only."));
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
          errno, "Unable to inspect an admission-root path component.");
    }
    if (S_ISLNK(component_stat.st_mode)) {
      return absl::FailedPreconditionError(
          "Exact-profile admission root must not traverse symbolic links.");
    }
  }
  return absl::OkStatus();
}

absl::Status SyncDirectory(const std::filesystem::path& directory);

absl::Status EnsureOwnedDirectory(const std::filesystem::path& parent,
                                  const std::filesystem::path& directory,
                                  absl::string_view description) {
  ABSL_RETURN_IF_ERROR(
      ValidateSecureDirectory(parent, "admission parent directory"));
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
    const std::filesystem::path& repository_directory,
    const std::filesystem::path& records_directory) {
  ABSL_RETURN_IF_ERROR(ValidateNoSymlinkPathComponents(root));
  ABSL_RETURN_IF_ERROR(
      ValidateSecureDirectory(root, "exact-profile admission root"));
  ABSL_RETURN_IF_ERROR(ValidateSecureDirectory(
      root / "exact-profile-admission", "admission product directory"));
  ABSL_RETURN_IF_ERROR(ValidateSecureDirectory(
      repository_directory, "admission version directory"));
  return ValidateSecureDirectory(records_directory,
                                 "admission records directory");
}

absl::Status SyncFd(int fd, absl::string_view description) {
  int result;
  do {
    result = fsync(fd);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("Unable to fsync ", description));
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
        errno, "Unable to open exact-profile admission directory.");
  }
  ScopedFd scoped(fd);
  return SyncFd(fd, "exact-profile admission directory");
}

absl::Status WriteAll(int fd, absl::string_view bytes) {
  size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        write(fd, bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to write exact-profile admission record.");
    }
    if (count == 0) {
      return absl::DataLossError(
          "Zero-byte write while publishing exact-profile admission.");
    }
    offset += static_cast<size_t>(count);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> ReadWholeFile(
    const std::filesystem::path& path) {
  ABSL_ASSIGN_OR_RETURN(ScopedFd fd, OpenRegularFile(path, O_RDONLY));
  struct stat before;
  if (fstat(fd.get(), &before) != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to inspect exact-profile admission record.");
  }
  if (before.st_size < 0 ||
      static_cast<uint64_t>(before.st_size) > kMaximumStoredAdmissionBytes ||
      static_cast<uint64_t>(before.st_size) > std::string().max_size()) {
    return absl::ResourceExhaustedError(
        "Exact-profile admission record exceeds its storage limit.");
  }
  if ((before.st_mode & S_IRUSR) == 0 ||
      (before.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0) {
    return absl::PermissionDeniedError(
        "Exact-profile admission records must be owner-readable and read-only.");
  }
  std::string bytes(static_cast<size_t>(before.st_size), '\0');
  size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count = pread(fd.get(), bytes.data() + offset,
                                bytes.size() - offset,
                                static_cast<off_t>(offset));
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to read exact-profile admission record.");
    }
    if (count == 0) {
      return absl::DataLossError(
          "Exact-profile admission record is truncated.");
    }
    offset += static_cast<size_t>(count);
  }
  struct stat after;
  if (fstat(fd.get(), &after) != 0 || after.st_size != before.st_size ||
      after.st_mtime != before.st_mtime) {
    return absl::DataLossError(
        "Exact-profile admission record changed during its read.");
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
        errno, "Unable to create exact-profile admission temporary file.");
  }
  ScopedFd scoped(fd);
  if (fchmod(fd, S_IRUSR | S_IWUSR) != 0) {
    const int saved_errno = errno;
    unlink(writable.data());
    return absl::ErrnoToStatus(
        saved_errno,
        "Unable to protect exact-profile admission temporary file.");
  }
  ABSL_RETURN_IF_ERROR(
      [&]() -> absl::Status {
        const int flags = fcntl(fd, F_GETFD);
        if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) != 0) {
          const int saved_errno = errno;
          unlink(writable.data());
          return absl::ErrnoToStatus(
              saved_errno,
              "Unable to mark admission temporary file close-on-exec.");
        }
        return absl::OkStatus();
      }());
  return std::make_pair(std::move(scoped),
                        std::filesystem::path(writable.data()));
}

std::filesystem::path RecordPath(const std::filesystem::path& directory,
                                 const Hash256& exact_profile_hash) {
  return directory /
         absl::StrCat(exact_profile_hash.ToHex(), ".admission");
}

}  // namespace

FilesystemExactProfileAdmissionRepository::
    FilesystemExactProfileAdmissionRepository(std::filesystem::path root)
    : root_(std::move(root)),
      // Admission v2 binds the authenticated full-prefill execution plan and
      // worker protocol v2. Keep it in a disjoint create-once namespace so a
      // disposable v1 admission can never block or masquerade as v2 proof.
      directory_(root_ / "exact-profile-admission" / "v2"),
      records_directory_(directory_ / "records"),
      lock_path_(directory_ / "repository.lock") {}

absl::StatusOr<std::unique_ptr<FilesystemExactProfileAdmissionRepository>>
FilesystemExactProfileAdmissionRepository::Create(
    std::filesystem::path directory) {
  if (directory.empty() || !directory.is_absolute()) {
    return absl::InvalidArgumentError(
        "Exact-profile admission repository requires an absolute existing root.");
  }
  for (const std::filesystem::path& component : directory.relative_path()) {
    if (component == "." || component == "..") {
      return absl::InvalidArgumentError(
          "Exact-profile admission root must not contain dot components.");
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateNoSymlinkPathComponents(directory));
  ABSL_RETURN_IF_ERROR(
      ValidateSecureDirectory(directory, "exact-profile admission root"));
  auto repository = std::unique_ptr<FilesystemExactProfileAdmissionRepository>(
      new FilesystemExactProfileAdmissionRepository(std::move(directory)));
  ABSL_RETURN_IF_ERROR(repository->Initialize());
  return repository;
}

absl::Status FilesystemExactProfileAdmissionRepository::Initialize() {
  const std::filesystem::path product = root_ / "exact-profile-admission";
  ABSL_RETURN_IF_ERROR(EnsureOwnedDirectory(
      root_, product, "admission product directory"));
  ABSL_RETURN_IF_ERROR(EnsureOwnedDirectory(
      product, directory_, "admission version directory"));
  ABSL_RETURN_IF_ERROR(EnsureOwnedDirectory(
      directory_, records_directory_, "admission records directory"));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock, AcquireLock(lock_path_, true));
  return ValidateRepositoryTree(root_, directory_, records_directory_);
}

absl::Status FilesystemExactProfileAdmissionRepository::PutIfAbsent(
    const ExactProfileAdmissionRecord& record,
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecord(record));
  ABSL_RETURN_IF_ERROR(
      ValidateRepositoryTree(root_, directory_, records_directory_));
  ABSL_ASSIGN_OR_RETURN(const std::string bytes,
                        EncodeExactProfileAdmissionRecord(record,
                                                          authentication));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock, AcquireLock(lock_path_, true));
  const std::filesystem::path final_path =
      RecordPath(records_directory_, record.exact_profile_hash);

  std::error_code exists_error;
  if (std::filesystem::exists(final_path, exists_error)) {
    if (exists_error) {
      return absl::FailedPreconditionError(
          "Unable to inspect existing exact-profile admission record.");
    }
    ABSL_ASSIGN_OR_RETURN(const std::string existing, ReadWholeFile(final_path));
    if (existing == bytes) return absl::OkStatus();
    ABSL_ASSIGN_OR_RETURN(
        const ExactProfileAdmissionRecord existing_record,
        DecodeExactProfileAdmissionRecord(existing, authentication));
    if (existing_record.exact_profile_hash != record.exact_profile_hash) {
      return absl::DataLossError(
          "Existing admission is stored under the wrong exact profile.");
    }
    return absl::AlreadyExistsError(
        "Conflicting admission already exists for this exact profile.");
  }
  if (exists_error) {
    return absl::FailedPreconditionError(
        "Unable to inspect exact-profile admission record path.");
  }

  ABSL_ASSIGN_OR_RETURN(auto temporary,
                        CreateTempFile(records_directory_,
                                       record.exact_profile_hash.ToHex()));
  ScopedFd temp_fd = std::move(temporary.first);
  const std::filesystem::path temp_path = std::move(temporary.second);
  const auto remove_temp = [&temp_path]() {
    while (unlink(temp_path.c_str()) != 0 && errno == EINTR) {
    }
  };
  absl::Status write_status = WriteAll(temp_fd.get(), bytes);
  if (write_status.ok()) {
    int chmod_result;
    do {
      chmod_result = fchmod(temp_fd.get(), S_IRUSR);
    } while (chmod_result != 0 && errno == EINTR);
    if (chmod_result != 0) {
      write_status = absl::ErrnoToStatus(
          errno, "Unable to make exact-profile admission record read-only.");
    }
  }
  if (write_status.ok()) {
    write_status = SyncFd(temp_fd.get(), "exact-profile admission record");
  }
  temp_fd.Reset();
  if (!write_status.ok()) {
    remove_temp();
    return write_status;
  }

  int link_result;
  do {
    link_result = link(temp_path.c_str(), final_path.c_str());
  } while (link_result != 0 && errno == EINTR);
  if (link_result != 0) {
    const int saved_errno = errno;
    remove_temp();
    if (saved_errno == EEXIST) {
      ABSL_ASSIGN_OR_RETURN(const std::string existing,
                            ReadWholeFile(final_path));
      if (existing == bytes) return absl::OkStatus();
      ABSL_ASSIGN_OR_RETURN(
          const ExactProfileAdmissionRecord existing_record,
          DecodeExactProfileAdmissionRecord(existing, authentication));
      if (existing_record.exact_profile_hash != record.exact_profile_hash) {
        return absl::DataLossError(
            "Concurrent admission is stored under the wrong exact profile.");
      }
      return absl::AlreadyExistsError(
          "Conflicting admission was concurrently published for this profile.");
    }
    return absl::ErrnoToStatus(
        saved_errno, "Unable to publish exact-profile admission record.");
  }
  ABSL_RETURN_IF_ERROR(SyncDirectory(records_directory_));
  remove_temp();
  return absl::OkStatus();
}

absl::StatusOr<ExactProfileAdmissionRecord>
FilesystemExactProfileAdmissionRepository::Get(
    const ExactLiteRtProfile& runtime_derived_profile,
    const FreshWorkerAuthentication& authentication) const {
  const Hash256& exact_profile_hash = runtime_derived_profile.profile_id;
  bool all_zero = true;
  for (uint8_t byte : exact_profile_hash.bytes) all_zero &= byte == 0;
  if (all_zero) {
    return absl::InvalidArgumentError(
        "Exact-profile admission lookup requires a nonzero profile digest.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateRepositoryTree(root_, directory_, records_directory_));
  ABSL_ASSIGN_OR_RETURN(ScopedFileLock lock, AcquireLock(lock_path_, false));
  const std::filesystem::path path =
      RecordPath(records_directory_, exact_profile_hash);
  ABSL_ASSIGN_OR_RETURN(const std::string bytes, ReadWholeFile(path));
  ABSL_ASSIGN_OR_RETURN(
      ExactProfileAdmissionRecord record,
      DecodeExactProfileAdmissionRecord(bytes, authentication));
  ABSL_RETURN_IF_ERROR(ValidateExactProfileAdmissionRecordForProfile(
      record, runtime_derived_profile));
  return record;
}

}  // namespace litert::lm
