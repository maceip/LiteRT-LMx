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

#include "runtime/dpm/canonical_replay_catalog.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
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
#include "runtime/platform/hash/hmac_sha256.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kKeyMagic = {'D', 'P', 'M', 'R', 'K', 'E', 'Y',
                                            '1'};
constexpr std::array<char, 8> kWinnerMagic = {'D', 'P', 'M', 'W', 'I', 'N',
                                               '0', '1'};
constexpr uint32_t kKeyFormatVersion = 1;
constexpr uint32_t kWinnerFormatVersion = 1;
constexpr size_t kAuthenticationTagBytes = 32;
constexpr size_t kMinimumAuthenticationKeyBytes = 32;
constexpr size_t kMaximumAuthenticationKeyBytes = 4096;
constexpr uint64_t kMaximumWinnerFileBytes =
    static_cast<uint64_t>(kMaximumCanonicalReplayOutputBytes) +
    2 * kMaximumCanonicalReplayContractBytes + 1024;
constexpr absl::string_view kAuthenticationDomain =
    "LITERT_LM_CANONICAL_WINNER_REPLAY_HMAC_SHA256_V1";
constexpr absl::string_view kAuthenticationPathDomain =
    "LITERT_LM_CANONICAL_WINNER_REPLAY_KEY_ID_PATH_SHA256_V1";

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
}

absl::Status ValidateBoundedIdentityString(absl::string_view value,
                                            absl::string_view description) {
  if (value.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat(description, " must not be empty."));
  }
  if (value.size() > kMaximumCanonicalReplayContractBytes) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds the replay format limit."));
  }
  for (unsigned char byte : value) {
    if (byte < 0x20 || byte > 0x7e) {
      return absl::InvalidArgumentError(absl::StrCat(
          description, " must contain printable ASCII bytes only."));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateAuthentication(
    const CanonicalReplayAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateBoundedIdentityString(
      authentication.key_id, "Canonical replay authentication key id"));
  if (authentication.authentication_key.size() <
          kMinimumAuthenticationKeyBytes ||
      authentication.authentication_key.size() >
          kMaximumAuthenticationKeyBytes) {
    return absl::InvalidArgumentError(
        "Canonical replay authentication requires 32 to 4096 key bytes.");
  }
  return absl::OkStatus();
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

absl::Status AppendString(absl::string_view value, size_t maximum_size,
                          std::string* output) {
  if (value.size() > maximum_size ||
      value.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError(
        "Canonical replay string exceeds its format limit.");
  }
  AppendU32(static_cast<uint32_t>(value.size()), output);
  output->append(value.data(), value.size());
  return absl::OkStatus();
}

absl::Status AppendBytes(absl::string_view value, size_t maximum_size,
                         std::string* output) {
  if (value.size() > maximum_size) {
    return absl::ResourceExhaustedError(
        "Canonical replay output exceeds its format limit.");
  }
  AppendU64(static_cast<uint64_t>(value.size()), output);
  output->append(value.data(), value.size());
  return absl::OkStatus();
}

class CanonicalReader {
 public:
  explicit CanonicalReader(absl::string_view bytes) : bytes_(bytes) {}

  bool empty() const { return offset_ == bytes_.size(); }

  absl::Status ReadMagic(absl::string_view magic,
                         absl::string_view description) {
    ABSL_RETURN_IF_ERROR(Require(magic.size(), description));
    if (std::memcmp(bytes_.data() + offset_, magic.data(), magic.size()) != 0) {
      return absl::DataLossError(
          absl::StrCat(description, " has invalid magic."));
    }
    offset_ += magic.size();
    return absl::OkStatus();
  }

  absl::StatusOr<uint32_t> ReadU32(absl::string_view description) {
    ABSL_RETURN_IF_ERROR(Require(sizeof(uint32_t), description));
    uint32_t value = 0;
    for (int index = 0; index < 4; ++index) {
      value = (value << 8) |
              static_cast<unsigned char>(bytes_[offset_ + index]);
    }
    offset_ += sizeof(uint32_t);
    return value;
  }

  absl::StatusOr<uint64_t> ReadU64(absl::string_view description) {
    ABSL_RETURN_IF_ERROR(Require(sizeof(uint64_t), description));
    uint64_t value = 0;
    for (int index = 0; index < 8; ++index) {
      value = (value << 8) |
              static_cast<unsigned char>(bytes_[offset_ + index]);
    }
    offset_ += sizeof(uint64_t);
    return value;
  }

  absl::StatusOr<Hash256> ReadHash(absl::string_view description) {
    Hash256 hash;
    ABSL_RETURN_IF_ERROR(Require(hash.bytes.size(), description));
    std::memcpy(hash.bytes.data(), bytes_.data() + offset_, hash.bytes.size());
    offset_ += hash.bytes.size();
    return hash;
  }

  absl::StatusOr<std::string> ReadString(size_t maximum_size,
                                         absl::string_view description) {
    ABSL_ASSIGN_OR_RETURN(const uint32_t size, ReadU32(description));
    if (size > maximum_size) {
      return absl::ResourceExhaustedError(
          absl::StrCat(description, " exceeds its format limit."));
    }
    return ReadSizedString(size, description);
  }

  absl::StatusOr<std::string> ReadBytes(size_t maximum_size,
                                        absl::string_view description) {
    ABSL_ASSIGN_OR_RETURN(const uint64_t size, ReadU64(description));
    if (size > maximum_size || size > std::numeric_limits<size_t>::max()) {
      return absl::ResourceExhaustedError(
          absl::StrCat(description, " exceeds its format limit."));
    }
    return ReadSizedString(static_cast<size_t>(size), description);
  }

 private:
  absl::Status Require(size_t size, absl::string_view description) const {
    if (size > bytes_.size() - offset_) {
      return absl::DataLossError(
          absl::StrCat("Truncated ", description, "."));
    }
    return absl::OkStatus();
  }

  absl::StatusOr<std::string> ReadSizedString(
      size_t size, absl::string_view description) {
    ABSL_RETURN_IF_ERROR(Require(size, description));
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
    try {
      std::string value(bytes_.data() + offset_, size);
      offset_ += size;
      return value;
    } catch (const std::bad_alloc&) {
      return absl::ResourceExhaustedError(
          absl::StrCat("Unable to allocate ", description, "."));
    } catch (const std::length_error&) {
      return absl::ResourceExhaustedError(
          absl::StrCat(description, " exceeds the string allocation limit."));
    }
#else
    std::string value(bytes_.data() + offset_, size);
    offset_ += size;
    return value;
#endif
  }

  absl::string_view bytes_;
  size_t offset_ = 0;
};

absl::StatusOr<CanonicalReplayKey> DecodeCanonicalReplayKey(
    absl::string_view bytes) {
  if (bytes.size() > 2 * kMaximumCanonicalReplayContractBytes + 256) {
    return absl::ResourceExhaustedError(
        "Canonical replay key exceeds its format limit.");
  }
  CanonicalReader reader(bytes);
  ABSL_RETURN_IF_ERROR(reader.ReadMagic(
      absl::string_view(kKeyMagic.data(), kKeyMagic.size()),
      "canonical replay key"));
  ABSL_ASSIGN_OR_RETURN(const uint32_t version,
                        reader.ReadU32("canonical replay key version"));
  if (version != kKeyFormatVersion) {
    return absl::DataLossError(
        "Unsupported canonical replay key format version.");
  }

  CanonicalReplayKey key;
  ABSL_ASSIGN_OR_RETURN(const uint32_t stage,
                        reader.ReadU32("canonical replay stage"));
  key.stage = static_cast<DPMReplayStage>(stage);
  ABSL_ASSIGN_OR_RETURN(
      key.runtime_identity.model_artifact_hash,
      reader.ReadHash("canonical replay model artifact hash"));
  ABSL_ASSIGN_OR_RETURN(
      key.runtime_identity.runtime_artifact_hash,
      reader.ReadHash("canonical replay runtime artifact hash"));
  ABSL_ASSIGN_OR_RETURN(
      key.runtime_identity.inference_profile_hash,
      reader.ReadHash("canonical replay inference profile hash"));
  ABSL_ASSIGN_OR_RETURN(
      key.canonical_request_hash,
      reader.ReadHash("canonical replay request hash"));
  ABSL_ASSIGN_OR_RETURN(
      key.request_contract_version,
      reader.ReadString(kMaximumCanonicalReplayContractBytes,
                        "canonical replay request contract"));
  if (!reader.empty()) {
    return absl::DataLossError(
        "Canonical replay key has trailing bytes.");
  }
  ABSL_RETURN_IF_ERROR(ValidateCanonicalReplayKey(key));
  ABSL_ASSIGN_OR_RETURN(std::string canonical, EncodeCanonicalReplayKey(key));
  if (canonical != bytes) {
    return absl::DataLossError(
        "Canonical replay key is not canonically encoded.");
  }
  return key;
}

absl::StatusOr<std::string> EncodeWinnerBody(
    const CanonicalReplayWinner& winner) {
  ABSL_RETURN_IF_ERROR(ValidateCanonicalReplayKey(winner.key));
  if (winner.canonical_output.empty()) {
    return absl::InvalidArgumentError(
        "Canonical replay output must not be empty.");
  }
  if (winner.canonical_output.size() >
      kMaximumCanonicalReplayOutputBytes) {
    return absl::ResourceExhaustedError(
        "Canonical replay output exceeds its format limit.");
  }
  if (IsZeroHash(winner.canonical_output_hash) ||
      IsZeroHash(winner.execution_evidence_hash)) {
    return absl::InvalidArgumentError(
        "Canonical replay output and evidence hashes must be nonzero.");
  }
  ABSL_RETURN_IF_ERROR(ValidateBoundedIdentityString(
      winner.authentication_key_id,
      "Canonical replay authentication key id"));

  ABSL_ASSIGN_OR_RETURN(std::string encoded_key,
                        EncodeCanonicalReplayKey(winner.key));
  std::string body(kWinnerMagic.data(), kWinnerMagic.size());
  AppendU32(kWinnerFormatVersion, &body);
  ABSL_RETURN_IF_ERROR(AppendString(encoded_key, 2 *
                                                    kMaximumCanonicalReplayContractBytes +
                                                256,
                                    &body));
  ABSL_RETURN_IF_ERROR(AppendBytes(winner.canonical_output,
                                   kMaximumCanonicalReplayOutputBytes, &body));
  AppendHash(winner.canonical_output_hash, &body);
  AppendHash(winner.execution_evidence_hash, &body);
  ABSL_RETURN_IF_ERROR(AppendString(winner.authentication_key_id,
                                    kMaximumCanonicalReplayContractBytes,
                                    &body));
  return body;
}

absl::StatusOr<std::string> EncodeWinner(
    const CanonicalReplayWinner& winner,
    const CanonicalReplayAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(
      ValidateCanonicalReplayWinner(winner, authentication));
  ABSL_ASSIGN_OR_RETURN(std::string bytes, EncodeWinnerBody(winner));
  AppendHash(winner.authentication_tag, &bytes);
  if (bytes.size() > kMaximumWinnerFileBytes) {
    return absl::ResourceExhaustedError(
        "Canonical replay winner exceeds its file format limit.");
  }
  return bytes;
}

absl::StatusOr<CanonicalReplayWinner> DecodeWinner(
    absl::string_view bytes,
    const CanonicalReplayAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateAuthentication(authentication));
  if (bytes.size() > kMaximumWinnerFileBytes) {
    return absl::ResourceExhaustedError(
        "Canonical replay winner exceeds its file format limit.");
  }
  if (bytes.size() < kWinnerMagic.size() + sizeof(uint32_t) +
                         kAuthenticationTagBytes) {
    return absl::DataLossError("Canonical replay winner is truncated.");
  }

  const absl::string_view body =
      bytes.substr(0, bytes.size() - kAuthenticationTagBytes);
  Hash256 stored_tag;
  std::memcpy(stored_tag.bytes.data(),
              bytes.data() + bytes.size() - kAuthenticationTagBytes,
              stored_tag.bytes.size());
  const Hash256 computed_tag = HmacSha256(
      authentication.authentication_key, {kAuthenticationDomain, body});
  if (!ConstantTimeHashEquals(stored_tag, computed_tag)) {
    return absl::UnauthenticatedError(
        "Canonical replay winner authentication failed.");
  }

  CanonicalReader reader(body);
  ABSL_RETURN_IF_ERROR(reader.ReadMagic(
      absl::string_view(kWinnerMagic.data(), kWinnerMagic.size()),
      "canonical replay winner"));
  ABSL_ASSIGN_OR_RETURN(const uint32_t version,
                        reader.ReadU32("canonical replay winner version"));
  if (version != kWinnerFormatVersion) {
    return absl::DataLossError(
        "Unsupported canonical replay winner format version.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string encoded_key,
      reader.ReadString(2 * kMaximumCanonicalReplayContractBytes + 256,
                        "canonical replay encoded key"));

  CanonicalReplayWinner winner;
  ABSL_ASSIGN_OR_RETURN(winner.key, DecodeCanonicalReplayKey(encoded_key));
  ABSL_ASSIGN_OR_RETURN(
      winner.canonical_output,
      reader.ReadBytes(kMaximumCanonicalReplayOutputBytes,
                       "canonical replay output"));
  ABSL_ASSIGN_OR_RETURN(
      winner.canonical_output_hash,
      reader.ReadHash("canonical replay output hash"));
  ABSL_ASSIGN_OR_RETURN(
      winner.execution_evidence_hash,
      reader.ReadHash("canonical replay execution evidence hash"));
  ABSL_ASSIGN_OR_RETURN(
      winner.authentication_key_id,
      reader.ReadString(kMaximumCanonicalReplayContractBytes,
                        "canonical replay authentication key id"));
  if (!reader.empty()) {
    return absl::DataLossError(
        "Canonical replay winner has trailing authenticated bytes.");
  }
  winner.authentication_tag = stored_tag;
  ABSL_RETURN_IF_ERROR(
      ValidateCanonicalReplayWinner(winner, authentication));
  ABSL_ASSIGN_OR_RETURN(std::string canonical,
                        EncodeWinner(winner, authentication));
  if (canonical != bytes) {
    return absl::DataLossError(
        "Canonical replay winner is not canonically encoded.");
  }
  return winner;
}

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

absl::StatusOr<ScopedFd> OpenRegularFile(const std::filesystem::path& path,
                                         int flags, mode_t mode = 0) {
  int fd;
  do {
    fd = open(path.c_str(), flags | ExtraOpenFlags(), mode);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Unable to open canonical replay file ",
                            path.string()));
  }
  struct stat file_stat;
  if (fstat(fd, &file_stat) != 0) {
    const int saved_errno = errno;
    close(fd);
    return absl::ErrnoToStatus(
        saved_errno, absl::StrCat("Unable to inspect canonical replay file ",
                                  path.string()));
  }
  if (!S_ISREG(file_stat.st_mode)) {
    close(fd);
    return absl::FailedPreconditionError(absl::StrCat(
        "Canonical replay path is not a regular file: ", path.string()));
  }
  return ScopedFd(fd);
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
        errno, absl::StrCat("Unable to open canonical replay directory ",
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

absl::Status ReadExact(int fd, char* output, size_t size,
                       absl::string_view description) {
  size_t offset = 0;
  while (offset < size) {
    ssize_t count;
    do {
      count = pread(fd, output + offset, size - offset,
                    static_cast<off_t>(offset));
    } while (count < 0 && errno == EINTR);
    if (count < 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("Unable to read ", description));
    }
    if (count == 0) {
      return absl::DataLossError(absl::StrCat("Truncated ", description));
    }
    offset += static_cast<size_t>(count);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> ReadWholeFile(
    const std::filesystem::path& path, uint64_t maximum_size) {
  ABSL_ASSIGN_OR_RETURN(ScopedFd fd, OpenRegularFile(path, O_RDONLY));
  struct stat file_stat;
  if (fstat(fd.get(), &file_stat) != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to inspect canonical replay winner.");
  }
  if (file_stat.st_size < 0 ||
      static_cast<uint64_t>(file_stat.st_size) > maximum_size ||
      static_cast<uint64_t>(file_stat.st_size) >
          std::numeric_limits<size_t>::max()) {
    return absl::ResourceExhaustedError(
        "Canonical replay winner exceeds its file limit.");
  }
  const size_t size = static_cast<size_t>(file_stat.st_size);
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS) || defined(_CPPUNWIND)
  std::string bytes;
  try {
    bytes.resize(size);
  } catch (const std::bad_alloc&) {
    return absl::ResourceExhaustedError(
        "Unable to allocate canonical replay winner bytes.");
  } catch (const std::length_error&) {
    return absl::ResourceExhaustedError(
        "Canonical replay winner exceeds the string allocation limit.");
  }
#else
  std::string bytes(size, '\0');
#endif
  ABSL_RETURN_IF_ERROR(
      ReadExact(fd.get(), bytes.data(), bytes.size(),
                "canonical replay winner"));
  struct stat final_stat;
  if (fstat(fd.get(), &final_stat) != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to re-inspect canonical replay winner.");
  }
  if (final_stat.st_size != file_stat.st_size) {
    return absl::DataLossError(
        "Canonical replay winner changed during immutable read.");
  }
  return bytes;
}

absl::Status EnsureNonSymlinkDirectory(const std::filesystem::path& directory,
                                       absl::string_view description) {
  std::error_code error;
  const bool created = std::filesystem::create_directories(directory, error);
  if (error) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unable to create ", description, " ", directory.string(), ": ",
        error.message()));
  }
  if (created) {
    std::filesystem::permissions(
        directory, std::filesystem::perms::owner_all,
        std::filesystem::perm_options::replace |
            std::filesystem::perm_options::nofollow,
        error);
    if (error) {
      return absl::PermissionDeniedError(absl::StrCat(
          "Unable to restrict ", description, " ", directory.string(), ": ",
          error.message()));
    }
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
  const std::filesystem::perms writable_by_others =
      std::filesystem::perms::group_write |
      std::filesystem::perms::others_write;
  if ((status.permissions() & writable_by_others) !=
      std::filesystem::perms::none) {
    return absl::PermissionDeniedError(absl::StrCat(
        description, " must not be group- or other-writable: ",
        directory.string()));
  }
  if (created) {
    ABSL_RETURN_IF_ERROR(SyncDirectory(directory));
    const std::filesystem::path parent = directory.parent_path();
    if (!parent.empty() && parent != directory) {
      ABSL_RETURN_IF_ERROR(SyncDirectory(parent));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateNonSymlinkDirectory(
    const std::filesystem::path& directory,
    absl::string_view description) {
  std::error_code error;
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
  const std::filesystem::perms writable_by_others =
      std::filesystem::perms::group_write |
      std::filesystem::perms::others_write;
  if ((status.permissions() & writable_by_others) !=
      std::filesystem::perms::none) {
    return absl::PermissionDeniedError(absl::StrCat(
        description, " must not be group- or other-writable: ",
        directory.string()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::pair<ScopedFd, std::filesystem::path>> CreateTempFile(
    const std::filesystem::path& directory) {
  std::string pattern = (directory / ".winner.tmp.XXXXXX").string();
  std::vector<char> writable(pattern.begin(), pattern.end());
  writable.push_back('\0');
  int fd = mkstemp(writable.data());
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to create canonical replay temporary file.");
  }
  if (fchmod(fd, S_IRUSR | S_IWUSR) != 0) {
    const int saved_errno = errno;
    close(fd);
    unlink(writable.data());
    return absl::ErrnoToStatus(
        saved_errno, "Unable to restrict canonical replay temporary file.");
  }
#ifdef FD_CLOEXEC
  if (fcntl(fd, F_SETFD, FD_CLOEXEC) != 0) {
    const int saved_errno = errno;
    close(fd);
    unlink(writable.data());
    return absl::ErrnoToStatus(
        saved_errno,
        "Unable to mark canonical replay temporary file close-on-exec.");
  }
#endif
  return std::make_pair(ScopedFd(fd),
                        std::filesystem::path(writable.data()));
}

absl::StatusOr<bool> PublishCreateOnce(
    const std::filesystem::path& directory,
    const std::filesystem::path& final_path, absl::string_view bytes) {
  ABSL_ASSIGN_OR_RETURN(auto temporary, CreateTempFile(directory));
  ScopedFd fd = std::move(temporary.first);
  const std::filesystem::path temporary_path = std::move(temporary.second);

  absl::Status status =
      WriteAll(fd.get(), bytes, "canonical replay winner");
  if (status.ok()) {
    int chmod_result;
    do {
      chmod_result = fchmod(fd.get(), S_IRUSR);
    } while (chmod_result != 0 && errno == EINTR);
    if (chmod_result != 0) {
      status = absl::ErrnoToStatus(
          errno, "Unable to make canonical replay winner read-only.");
    }
  }
  if (status.ok()) {
    status = SyncFd(fd.get(), "canonical replay winner");
  }

  bool published = false;
  if (status.ok()) {
    int result;
    do {
      result = link(temporary_path.c_str(), final_path.c_str());
    } while (result != 0 && errno == EINTR);
    if (result == 0) {
      published = true;
      status = SyncDirectory(directory);
    } else if (errno != EEXIST) {
      status = absl::ErrnoToStatus(
          errno, "Unable to publish canonical replay winner.");
    }
  }

  int unlink_result;
  do {
    unlink_result = unlink(temporary_path.c_str());
  } while (unlink_result != 0 && errno == EINTR);
  if (status.ok() && unlink_result != 0 && errno != ENOENT) {
    status = absl::ErrnoToStatus(
        errno, "Unable to remove canonical replay temporary file.");
  }
  if (status.ok()) status = SyncDirectory(directory);
  ABSL_RETURN_IF_ERROR(status);
  return published;
}

}  // namespace

absl::Status ValidateCanonicalReplayKey(const CanonicalReplayKey& key) {
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayStage(key.stage));
  if (IsZeroHash(key.runtime_identity.model_artifact_hash) ||
      IsZeroHash(key.runtime_identity.runtime_artifact_hash) ||
      IsZeroHash(key.runtime_identity.inference_profile_hash)) {
    return absl::InvalidArgumentError(
        "Canonical replay runtime identity hashes must be nonzero.");
  }
  if (IsZeroHash(key.canonical_request_hash)) {
    return absl::InvalidArgumentError(
        "Canonical replay request hash must be nonzero.");
  }
  return ValidateBoundedIdentityString(
      key.request_contract_version,
      "Canonical replay request contract version");
}

absl::Status ValidateCanonicalReplayCandidate(
    const CanonicalReplayCandidate& candidate) {
  ABSL_RETURN_IF_ERROR(ValidateCanonicalReplayKey(candidate.key));
  if (candidate.canonical_output.empty()) {
    return absl::InvalidArgumentError(
        "Canonical replay candidate output must not be empty.");
  }
  if (candidate.canonical_output.size() >
      kMaximumCanonicalReplayOutputBytes) {
    return absl::ResourceExhaustedError(
        "Canonical replay candidate output exceeds its format limit.");
  }
  if (IsZeroHash(candidate.execution_evidence_hash)) {
    return absl::InvalidArgumentError(
        "Canonical replay candidate requires nonzero execution evidence.");
  }
  return absl::OkStatus();
}

absl::Status ValidateCanonicalReplayWinner(
    const CanonicalReplayWinner& winner,
    const CanonicalReplayAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateAuthentication(authentication));
  ABSL_RETURN_IF_ERROR(ValidateCanonicalReplayKey(winner.key));
  if (winner.canonical_output.empty()) {
    return absl::InvalidArgumentError(
        "Canonical replay winner output must not be empty.");
  }
  if (winner.canonical_output.size() >
      kMaximumCanonicalReplayOutputBytes) {
    return absl::ResourceExhaustedError(
        "Canonical replay winner output exceeds its format limit.");
  }
  Sha256Hasher output_hasher;
  output_hasher.Update(winner.canonical_output);
  if (output_hasher.Finalize() != winner.canonical_output_hash) {
    return absl::DataLossError(
        "Canonical replay winner output hash does not match its bytes.");
  }
  if (IsZeroHash(winner.execution_evidence_hash)) {
    return absl::InvalidArgumentError(
        "Canonical replay winner requires nonzero execution evidence.");
  }
  if (winner.authentication_key_id != authentication.key_id) {
    return absl::UnauthenticatedError(
        "Canonical replay winner authentication key id is not configured.");
  }
  ABSL_ASSIGN_OR_RETURN(std::string body, EncodeWinnerBody(winner));
  const Hash256 expected_tag = HmacSha256(
      authentication.authentication_key, {kAuthenticationDomain, body});
  if (!ConstantTimeHashEquals(expected_tag, winner.authentication_tag)) {
    return absl::UnauthenticatedError(
        "Canonical replay winner authentication tag is invalid.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> EncodeCanonicalReplayKey(
    const CanonicalReplayKey& key) {
  ABSL_RETURN_IF_ERROR(ValidateCanonicalReplayKey(key));
  std::string bytes(kKeyMagic.data(), kKeyMagic.size());
  AppendU32(kKeyFormatVersion, &bytes);
  AppendU32(static_cast<uint32_t>(key.stage), &bytes);
  AppendHash(key.runtime_identity.model_artifact_hash, &bytes);
  AppendHash(key.runtime_identity.runtime_artifact_hash, &bytes);
  AppendHash(key.runtime_identity.inference_profile_hash, &bytes);
  AppendHash(key.canonical_request_hash, &bytes);
  ABSL_RETURN_IF_ERROR(AppendString(key.request_contract_version,
                                    kMaximumCanonicalReplayContractBytes,
                                    &bytes));
  return bytes;
}

absl::StatusOr<Hash256> ComputeCanonicalReplayKeyHash(
    const CanonicalReplayKey& key) {
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeCanonicalReplayKey(key));
  Sha256Hasher hasher;
  hasher.Update(encoded);
  return hasher.Finalize();
}

static Hash256 ComputeAuthenticationPathHash(absl::string_view key_id) {
  Sha256Hasher hasher;
  hasher.Update(kAuthenticationPathDomain);
  std::array<char, 8> encoded_size{};
  const uint64_t size = key_id.size();
  for (size_t index = 0; index < encoded_size.size(); ++index) {
    encoded_size[index] =
        static_cast<char>((size >> (56 - index * 8)) & 0xff);
  }
  hasher.Update(
      absl::string_view(encoded_size.data(), encoded_size.size()));
  hasher.Update(key_id);
  return hasher.Finalize();
}

static absl::Status EnsureCatalogDirectoryTree(
    const std::filesystem::path& root,
    const Hash256& authentication_path_hash) {
  const std::filesystem::path product = root / "canonical-winner-replay";
  const std::filesystem::path version = product / "v1";
  const std::filesystem::path authentication_domain =
      version / authentication_path_hash.ToHex();
  ABSL_RETURN_IF_ERROR(
      ValidateNonSymlinkDirectory(root, "canonical replay root"));
  ABSL_RETURN_IF_ERROR(
      EnsureNonSymlinkDirectory(product, "canonical replay product directory"));
  ABSL_RETURN_IF_ERROR(
      EnsureNonSymlinkDirectory(version, "canonical replay version directory"));
  ABSL_RETURN_IF_ERROR(EnsureNonSymlinkDirectory(
      authentication_domain, "canonical replay authentication directory"));
  ABSL_RETURN_IF_ERROR(EnsureNonSymlinkDirectory(
      authentication_domain /
          std::string(DPMReplayStageToString(DPMReplayStage::kProjection)),
      "canonical replay projection directory"));
  return EnsureNonSymlinkDirectory(
      authentication_domain /
          std::string(
              DPMReplayStageToString(DPMReplayStage::kAgentDecision)),
      "canonical replay agent-decision directory");
}

static absl::Status ValidateCatalogDirectoryTree(
    const std::filesystem::path& root,
    const Hash256& authentication_path_hash, DPMReplayStage stage) {
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayStage(stage));
  const std::filesystem::path product = root / "canonical-winner-replay";
  const std::filesystem::path version = product / "v1";
  const std::filesystem::path authentication_domain =
      version / authentication_path_hash.ToHex();
  ABSL_RETURN_IF_ERROR(
      ValidateNonSymlinkDirectory(root, "canonical replay root"));
  ABSL_RETURN_IF_ERROR(ValidateNonSymlinkDirectory(
      product, "canonical replay product directory"));
  ABSL_RETURN_IF_ERROR(ValidateNonSymlinkDirectory(
      version, "canonical replay version directory"));
  ABSL_RETURN_IF_ERROR(ValidateNonSymlinkDirectory(
      authentication_domain, "canonical replay authentication directory"));
  return ValidateNonSymlinkDirectory(
      authentication_domain / std::string(DPMReplayStageToString(stage)),
      "canonical replay stage directory");
}

absl::StatusOr<
    std::unique_ptr<LocalFilesystemCanonicalWinnerReplayCatalog>>
LocalFilesystemCanonicalWinnerReplayCatalog::Create(
    std::filesystem::path root,
    CanonicalReplayAuthentication authentication, DPMReplayMode mode) {
  if (root.empty() || !root.is_absolute()) {
    return absl::InvalidArgumentError(
        "Canonical replay catalog requires an absolute storage root.");
  }
  for (const std::filesystem::path& component : root.relative_path()) {
    if (component == "." || component == "..") {
      return absl::InvalidArgumentError(
          "Canonical replay catalog root must not contain dot path "
          "components.");
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(mode));
  if (mode != DPMReplayMode::kCanonicalWinnerReplay) {
    return absl::InvalidArgumentError(
        "Exact regeneration cannot be configured with a canonical-winner "
        "catalog.");
  }
  ABSL_RETURN_IF_ERROR(ValidateAuthentication(authentication));
  const Hash256 authentication_path_hash =
      ComputeAuthenticationPathHash(authentication.key_id);
  ABSL_RETURN_IF_ERROR(
      EnsureCatalogDirectoryTree(root, authentication_path_hash));
  return std::unique_ptr<LocalFilesystemCanonicalWinnerReplayCatalog>(
      new LocalFilesystemCanonicalWinnerReplayCatalog(
          std::move(root), std::move(authentication)));
}

absl::StatusOr<std::filesystem::path>
LocalFilesystemCanonicalWinnerReplayCatalog::WinnerPathFor(
    const CanonicalReplayKey& key) const {
  if (root_.empty() || !root_.is_absolute()) {
    return absl::FailedPreconditionError(
        "Canonical replay catalog has no absolute storage root.");
  }
  ABSL_RETURN_IF_ERROR(ValidateAuthentication(authentication_));
  ABSL_ASSIGN_OR_RETURN(const Hash256 key_hash,
                        ComputeCanonicalReplayKeyHash(key));
  const Hash256 authentication_path_hash =
      ComputeAuthenticationPathHash(authentication_.key_id);
  ABSL_RETURN_IF_ERROR(ValidateCatalogDirectoryTree(
      root_, authentication_path_hash, key.stage));
  return root_ / "canonical-winner-replay" / "v1" /
         authentication_path_hash.ToHex() /
         std::string(DPMReplayStageToString(key.stage)) /
         absl::StrCat(key_hash.ToHex(), ".winner");
}

absl::StatusOr<CanonicalReplayWinner>
LocalFilesystemCanonicalWinnerReplayCatalog::Get(
    const CanonicalReplayKey& key) const {
  ABSL_ASSIGN_OR_RETURN(const std::filesystem::path path, WinnerPathFor(key));
  ABSL_ASSIGN_OR_RETURN(std::string bytes,
                        ReadWholeFile(path, kMaximumWinnerFileBytes));
  ABSL_ASSIGN_OR_RETURN(CanonicalReplayWinner winner,
                        DecodeWinner(bytes, authentication_));
  if (!(winner.key == key)) {
    return absl::DataLossError(
        "Canonical replay winner key does not match its content address.");
  }
  return winner;
}

absl::StatusOr<PublishCanonicalReplayResult>
LocalFilesystemCanonicalWinnerReplayCatalog::Publish(
    const CanonicalReplayCandidate& candidate) {
  ABSL_RETURN_IF_ERROR(ValidateCanonicalReplayCandidate(candidate));
  ABSL_ASSIGN_OR_RETURN(const std::filesystem::path winner_path,
                        WinnerPathFor(candidate.key));
  const std::filesystem::path directory = winner_path.parent_path();
  ABSL_RETURN_IF_ERROR(ValidateNonSymlinkDirectory(
      directory, "canonical replay winner directory"));

  CanonicalReplayWinner proposed;
  proposed.key = candidate.key;
  proposed.canonical_output = candidate.canonical_output;
  Sha256Hasher output_hasher;
  output_hasher.Update(proposed.canonical_output);
  proposed.canonical_output_hash = output_hasher.Finalize();
  proposed.execution_evidence_hash = candidate.execution_evidence_hash;
  proposed.authentication_key_id = authentication_.key_id;
  ABSL_ASSIGN_OR_RETURN(std::string body, EncodeWinnerBody(proposed));
  proposed.authentication_tag = HmacSha256(
      authentication_.authentication_key, {kAuthenticationDomain, body});
  ABSL_ASSIGN_OR_RETURN(std::string bytes,
                        EncodeWinner(proposed, authentication_));

  ABSL_ASSIGN_OR_RETURN(
      const bool published,
      PublishCreateOnce(directory, winner_path, bytes));
  // Neither publisher nor loser exposes caller-owned bytes. Reload and
  // authenticate the immutable file so the result is exactly the durable
  // winner selected by the filesystem create-once operation.
  ABSL_ASSIGN_OR_RETURN(CanonicalReplayWinner winner, Get(candidate.key));
  return PublishCanonicalReplayResult{
      .published = published,
      .winner = std::move(winner),
  };
}

}  // namespace litert::lm
