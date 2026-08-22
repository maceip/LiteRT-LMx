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

#include "runtime/dpm/fresh_worker_process.h"

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/core/session_handoff_codec.h"
#include "runtime/dpm/engine_fresh_worker_contract.h"
#include "runtime/platform/hash/hmac_sha256.h"
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kIpcFrameMagic = {'D', 'P', 'M', 'I', 'P', 'C',
                                                 '0', '1'};
constexpr std::array<char, 8> kCapsuleFrameMagic = {
    'D', 'P', 'M', 'C', 'A', 'P', '0', '1'};
constexpr std::array<char, 8> kAuthenticationMagic = {
    'D', 'P', 'M', 'K', 'E', 'Y', '0', '1'};
constexpr uint32_t kAuthenticationPreludeFormatVersion = 1;
constexpr uint32_t kMaximumAuthenticationKeyIdBytes = 256;
constexpr uint32_t kMinimumAuthenticationKeyBytes = 32;
constexpr uint32_t kMaximumAuthenticationKeyBytes = 4096;
constexpr uint32_t kMaximumArguments = 128;
constexpr uint64_t kMaximumArgumentBytes = 64 * 1024;
constexpr uint64_t kMaximumFailureMessageBytes = 4096;
constexpr absl::Duration kMaximumWorkerTimeout = absl::Hours(1);
constexpr absl::Duration kMaximumTerminationGrace = absl::Seconds(5);
constexpr absl::string_view kLaunchSpecDomain =
    "LITERT_LMX_FRESH_WORKER_LAUNCH_SPEC_SHA256";
constexpr absl::string_view kWorkerCertificationDomain =
    "LITERT_LMX_FRESH_WORKER_CERTIFICATION_SHA256";
constexpr absl::string_view kTransportKeyDomain =
    "LITERT_LMX_FRESH_WORKER_PER_REQUEST_TRANSPORT_KEY_HMAC_SHA256";
constexpr absl::string_view kCapsuleKeyDomain =
    "LITERT_LMX_FRESH_WORKER_CAPSULE_KEY_HMAC_SHA256";
constexpr absl::string_view kRestoreCapsuleDirection = "RESTORE_TO_WORKER";
constexpr absl::string_view kProducingCapsuleDirection =
    "PRODUCING_FROM_WORKER";
constexpr uint64_t kCapsuleIoChunkBytes = uint64_t{1024} * 1024;

bool IsZeroHash(const Hash256& hash) {
  uint8_t combined = 0;
  for (uint8_t byte : hash.bytes) combined |= byte;
  return combined == 0;
}

bool IsCompleteSessionHandoffIdentity(
    const SessionHandoffIdentity& identity) {
  return !IsZeroHash(identity.model_artifact_hash) &&
         !IsZeroHash(identity.runtime_artifact_hash) &&
         !IsZeroHash(identity.inference_profile_hash);
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

uint32_t ReadU32(const char* bytes) {
  uint32_t value = 0;
  for (int index = 0; index < 4; ++index) {
    value = (value << 8) | static_cast<uint8_t>(bytes[index]);
  }
  return value;
}

uint64_t ReadU64(const char* bytes) {
  uint64_t value = 0;
  for (int index = 0; index < 8; ++index) {
    value = (value << 8) | static_cast<uint8_t>(bytes[index]);
  }
  return value;
}

void SecureErase(std::string* bytes) {
  if (bytes == nullptr || bytes->empty()) return;
  volatile char* cursor = bytes->data();
  for (size_t index = 0; index < bytes->size(); ++index) cursor[index] = 0;
  bytes->clear();
}

class SecretEraser {
 public:
  explicit SecretEraser(std::string* secret) : secret_(secret) {}
  SecretEraser(const SecretEraser&) = delete;
  SecretEraser& operator=(const SecretEraser&) = delete;
  ~SecretEraser() { SecureErase(secret_); }

 private:
  std::string* secret_;
};

absl::Status ValidateProcessOptions(const FreshWorkerProcessOptions& options) {
  if (options.executable_path.empty() ||
      !options.executable_path.is_absolute()) {
    return absl::InvalidArgumentError(
        "Fresh-worker executable path must be absolute.");
  }
  if (options.arguments.size() > kMaximumArguments) {
    return absl::ResourceExhaustedError(
        "Fresh-worker argument count exceeds the process limit.");
  }
  uint64_t argument_bytes = 0;
  for (const std::string& argument : options.arguments) {
    if (argument.find('\0') != std::string::npos ||
        argument.size() > kMaximumArgumentBytes - argument_bytes) {
      return absl::InvalidArgumentError(
          "Fresh-worker arguments are invalid or oversized.");
    }
    argument_bytes += argument.size();
  }
  if (options.timeout <= absl::ZeroDuration() ||
      options.timeout > kMaximumWorkerTimeout ||
      options.termination_grace < absl::ZeroDuration() ||
      options.termination_grace > kMaximumTerminationGrace) {
    return absl::InvalidArgumentError(
        "Fresh-worker timeout or termination grace is invalid.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> CanonicalExecutablePath(
    const std::filesystem::path& executable_path) {
  std::error_code error;
  const std::filesystem::file_status input_status =
      std::filesystem::symlink_status(executable_path, error);
  if (error || std::filesystem::is_symlink(input_status) ||
      !std::filesystem::is_regular_file(input_status)) {
    return absl::FailedPreconditionError(
        "Fresh-worker executable must be a non-symlink regular file.");
  }
  const std::filesystem::path canonical =
      std::filesystem::canonical(executable_path, error);
  if (error || canonical.empty()) {
    return absl::FailedPreconditionError(
        "Fresh-worker executable path cannot be canonicalized.");
  }
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  if (access(canonical.c_str(), X_OK) != 0) {
    return absl::ErrnoToStatus(
        errno, "Fresh-worker executable is not executable.");
  }
#endif
  return canonical.string();
}

std::string EncodeFrameHeader(const std::array<char, 8>& magic,
                              uint64_t payload_size) {
  std::string frame;
  frame.reserve(magic.size() + 8);
  frame.append(magic.data(), magic.size());
  AppendU64(payload_size, &frame);
  return frame;
}

std::string EncodeIpcFrame(absl::string_view envelope) {
  std::string frame = EncodeFrameHeader(kIpcFrameMagic, envelope.size());
  frame.reserve(frame.size() + envelope.size());
  frame.append(envelope);
  return frame;
}

absl::StatusOr<std::string> BuildAuthenticationPrelude(
    const FreshWorkerAuthentication& authentication) {
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  std::string prelude;
  prelude.reserve(kAuthenticationMagic.size() + 12 +
                  authentication.key_id.size() +
                  authentication.authentication_key.size());
  prelude.append(kAuthenticationMagic.data(), kAuthenticationMagic.size());
  AppendU32(kAuthenticationPreludeFormatVersion, &prelude);
  AppendU32(static_cast<uint32_t>(authentication.key_id.size()), &prelude);
  AppendU32(static_cast<uint32_t>(authentication.authentication_key.size()),
            &prelude);
  prelude.append(authentication.key_id);
  prelude.append(authentication.authentication_key);
  return prelude;
}

absl::StatusOr<FreshWorkerAuthentication> DeriveTransportAuthentication(
    const FreshWorkerAuthentication& master_authentication,
    const FreshWorkerRequest& request) {
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerAuthentication(master_authentication));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(request));

  Sha256Hasher payload_hasher;
  payload_hasher.Update(request.request_payload);
  const Hash256 payload_hash = payload_hasher.Finalize();

  std::string binding;
  binding.reserve(4 + 32 + 32 + 4 + 4 + 32 + 32 + 4 + 4 +
                  master_authentication.key_id.size() + 8 + 32);
  AppendU32(request.format_version, &binding);
  AppendHash(request.exact_profile_hash, &binding);
  AppendHash(request.qualification_id, &binding);
  AppendU32(request.run_index, &binding);
  AppendU32(request.run_count, &binding);
  AppendHash(request.challenge_nonce, &binding);
  AppendHash(request.execution_plan.plan_hash, &binding);
  AppendU32(request.execution_plan.capture_producing_capsule ? 1 : 0,
            &binding);
  AppendU32(static_cast<uint32_t>(master_authentication.key_id.size()),
            &binding);
  binding.append(master_authentication.key_id);
  AppendU64(request.request_payload.size(), &binding);
  AppendHash(payload_hash, &binding);

  const Hash256 derived_key = HmacSha256(
      master_authentication.authentication_key, {kTransportKeyDomain, binding});
  return FreshWorkerAuthentication{
      .key_id = master_authentication.key_id,
      .authentication_key = std::string(
      reinterpret_cast<const char*>(derived_key.bytes.data()),
          derived_key.bytes.size())};
}

SessionHandoffOptions DeriveCapsuleOptions(
    const FreshWorkerAuthentication& transport_authentication,
    const Hash256& request_envelope_hash, absl::string_view direction,
    absl::string_view key_id,
    std::optional<SessionHandoffIdentity> expected_identity) {
  const absl::string_view request_hash_bytes(
      reinterpret_cast<const char*>(request_envelope_hash.bytes.data()),
      request_envelope_hash.bytes.size());
  const Hash256 derived_key = HmacSha256(
      transport_authentication.authentication_key,
      {kCapsuleKeyDomain, direction, request_hash_bytes});
  return SessionHandoffOptions{
      .key_id = std::string(key_id),
      .authentication_key = std::string(
          reinterpret_cast<const char*>(derived_key.bytes.data()),
          derived_key.bytes.size()),
      .expected_identity = std::move(expected_identity)};
}

absl::Status ValidateSessionHandoffTransfer(
    const FreshWorkerRequest& request,
    const FreshWorkerSessionHandoffTransfer& transfer) {
  const bool has_restore_source = transfer.durable_restore_source != nullptr;
  const bool has_restore_options =
      transfer.durable_restore_options != nullptr;
  if (has_restore_source != has_restore_options) {
    return absl::InvalidArgumentError(
        "Durable restore source and options must be supplied together.");
  }
  const bool has_capture_destination =
      transfer.durable_capture_destination != nullptr;
  const bool has_capture_options =
      transfer.durable_capture_options != nullptr;
  if (has_capture_destination != has_capture_options) {
    return absl::InvalidArgumentError(
        "Durable capture destination and options must be supplied together.");
  }

  const bool restore_required =
      request.execution_plan.prefill_mode ==
      FreshWorkerPrefillMode::kOwnPositionCapsuleDelta;
  if (restore_required != has_restore_source) {
    return absl::InvalidArgumentError(
        "Fresh-worker restore transfer disagrees with the execution plan.");
  }
  if (request.execution_plan.prefill_mode ==
          FreshWorkerPrefillMode::kFullCanonicalPrefill &&
      has_restore_source) {
    return absl::InvalidArgumentError(
        "Full-prefill execution cannot receive a restore capsule.");
  }
  if (request.execution_plan.capture_producing_capsule !=
      has_capture_destination) {
    return absl::InvalidArgumentError(
        "Fresh-worker durable capture transfer disagrees with capture "
        "policy.");
  }
  if (has_capture_options &&
      (!transfer.durable_capture_options->expected_identity.has_value() ||
       !IsCompleteSessionHandoffIdentity(
           *transfer.durable_capture_options->expected_identity))) {
    return absl::InvalidArgumentError(
        "Durable capture requires a complete parent-asserted session "
        "identity.");
  }
  if (restore_required &&
      request.execution_plan.restore_durable_envelope_size >
          kMaximumFreshWorkerCapsuleBytes) {
    return absl::ResourceExhaustedError(
        "Durable restore capsule exceeds the fresh-worker capsule limit.");
  }
  if (restore_required &&
      transfer.durable_restore_source->Size() !=
          request.execution_plan.restore_durable_envelope_size) {
    return absl::DataLossError(
        "Durable restore capsule size differs from its execution-plan "
        "binding.");
  }
  if (restore_required && has_capture_options &&
      transfer.durable_capture_options->expected_identity.has_value() &&
      *transfer.durable_capture_options->expected_identity !=
          request.execution_plan.session_identity) {
    return absl::FailedPreconditionError(
        "Durable capture identity assertion differs from the restored "
        "session identity.");
  }
  return absl::OkStatus();
}

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)

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

struct Pipe {
  ScopedFd read;
  ScopedFd write;
};

absl::Status SetCloseOnExec(int fd) {
  const int flags = fcntl(fd, F_GETFD);
  if (flags < 0 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) != 0) {
    return absl::ErrnoToStatus(errno,
                               "Unable to mark worker pipe close-on-exec.");
  }
  return absl::OkStatus();
}

struct ExecutableInspection {
  uint64_t device = 0;
  uint64_t inode = 0;
  uint64_t file_size = 0;
  uint32_t file_mode = 0;
  int64_t modification_time_seconds = 0;
  int64_t modification_time_nanoseconds = 0;
  int64_t status_change_time_seconds = 0;
  int64_t status_change_time_nanoseconds = 0;
  Hash256 image_hash;
};

ExecutableInspection InspectionFromStat(const struct stat& value) {
  ExecutableInspection inspection;
  inspection.device = static_cast<uint64_t>(value.st_dev);
  inspection.inode = static_cast<uint64_t>(value.st_ino);
  inspection.file_size = static_cast<uint64_t>(value.st_size);
  inspection.file_mode = static_cast<uint32_t>(value.st_mode);
#if defined(__APPLE__)
  inspection.modification_time_seconds = value.st_mtimespec.tv_sec;
  inspection.modification_time_nanoseconds = value.st_mtimespec.tv_nsec;
  inspection.status_change_time_seconds = value.st_ctimespec.tv_sec;
  inspection.status_change_time_nanoseconds = value.st_ctimespec.tv_nsec;
#else
  inspection.modification_time_seconds = value.st_mtim.tv_sec;
  inspection.modification_time_nanoseconds = value.st_mtim.tv_nsec;
  inspection.status_change_time_seconds = value.st_ctim.tv_sec;
  inspection.status_change_time_nanoseconds = value.st_ctim.tv_nsec;
#endif
  return inspection;
}

bool SameExecutableMetadata(const ExecutableInspection& left,
                            const ExecutableInspection& right) {
  return left.device == right.device && left.inode == right.inode &&
         left.file_size == right.file_size &&
         left.file_mode == right.file_mode &&
         left.modification_time_seconds == right.modification_time_seconds &&
         left.modification_time_nanoseconds ==
             right.modification_time_nanoseconds &&
         left.status_change_time_seconds == right.status_change_time_seconds &&
         left.status_change_time_nanoseconds ==
             right.status_change_time_nanoseconds;
}

absl::StatusOr<ExecutableInspection> InspectExecutableImage(
    absl::string_view canonical_path) {
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  int descriptor;
  do {
    descriptor = open(std::string(canonical_path).c_str(), flags);
  } while (descriptor < 0 && errno == EINTR);
  if (descriptor < 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to open the certified fresh-worker executable.");
  }
  ScopedFd file(descriptor);
#ifndef O_CLOEXEC
  ABSL_RETURN_IF_ERROR(SetCloseOnExec(file.get()));
#endif

  struct stat before;
  if (fstat(file.get(), &before) != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to inspect the fresh-worker executable.");
  }
  if (!S_ISREG(before.st_mode) || before.st_size < 0) {
    return absl::FailedPreconditionError(
        "Certified fresh-worker executable is not a regular file.");
  }
  if (access(std::string(canonical_path).c_str(), X_OK) != 0) {
    return absl::ErrnoToStatus(
        errno, "Certified fresh-worker image is not executable.");
  }
  const ExecutableInspection before_inspection = InspectionFromStat(before);

  Sha256Hasher image_hasher;
  std::array<char, 64 * 1024> buffer{};
  uint64_t offset = 0;
  while (offset < before_inspection.file_size) {
    const uint64_t remaining = before_inspection.file_size - offset;
    const size_t requested = static_cast<size_t>(
        std::min<uint64_t>(remaining, buffer.size()));
    if (offset >
        static_cast<uint64_t>((std::numeric_limits<off_t>::max)())) {
      return absl::ResourceExhaustedError(
          "Fresh-worker executable offset is not representable.");
    }
    const ssize_t read_count =
        pread(file.get(), buffer.data(), requested, static_cast<off_t>(offset));
    if (read_count < 0 && errno == EINTR) continue;
    if (read_count < 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to hash the fresh-worker executable.");
    }
    if (read_count == 0) {
      return absl::DataLossError(
          "Fresh-worker executable changed while it was being hashed.");
    }
    image_hasher.Update(
        absl::string_view(buffer.data(), static_cast<size_t>(read_count)));
    offset += static_cast<uint64_t>(read_count);
  }

  struct stat after;
  if (fstat(file.get(), &after) != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to re-inspect the fresh-worker executable.");
  }
  const ExecutableInspection after_inspection = InspectionFromStat(after);
  if (!SameExecutableMetadata(before_inspection, after_inspection)) {
    return absl::AbortedError(
        "Fresh-worker executable changed while its certification was read.");
  }

  struct stat path_after;
  if (lstat(std::string(canonical_path).c_str(), &path_after) != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to re-resolve the fresh-worker executable path.");
  }
  const ExecutableInspection path_inspection = InspectionFromStat(path_after);
  if (S_ISLNK(path_after.st_mode) || !S_ISREG(path_after.st_mode) ||
      !SameExecutableMetadata(after_inspection, path_inspection)) {
    return absl::AbortedError(
        "Fresh-worker executable path changed during certification.");
  }

  ExecutableInspection result = after_inspection;
  result.image_hash = image_hasher.Finalize();
  return result;
}

Hash256 ComputeWorkerCertificationHash(
    absl::string_view canonical_path,
    const std::vector<std::string>& canonical_arguments,
    const ExecutableInspection& inspection,
    FreshWorkerEnvironmentContract environment_contract,
    absl::string_view engine_adapter_contract) {
  Sha256Hasher hasher;
  hasher.Update(kWorkerCertificationDomain);
  std::string encoded;
  AppendU32(FreshWorkerCertification::kFormatVersion, &encoded);
  AppendU64(canonical_path.size(), &encoded);
  encoded.append(canonical_path.data(), canonical_path.size());
  AppendU32(static_cast<uint32_t>(canonical_arguments.size()), &encoded);
  for (const std::string& argument : canonical_arguments) {
    AppendU64(argument.size(), &encoded);
    encoded.append(argument);
  }
  AppendHash(inspection.image_hash, &encoded);
  AppendU64(inspection.device, &encoded);
  AppendU64(inspection.inode, &encoded);
  AppendU64(inspection.file_size, &encoded);
  AppendU32(inspection.file_mode, &encoded);
  AppendU64(static_cast<uint64_t>(inspection.modification_time_seconds),
            &encoded);
  AppendU64(static_cast<uint64_t>(inspection.modification_time_nanoseconds),
            &encoded);
  AppendU64(static_cast<uint64_t>(inspection.status_change_time_seconds),
            &encoded);
  AppendU64(static_cast<uint64_t>(inspection.status_change_time_nanoseconds),
            &encoded);
  AppendU32(static_cast<uint32_t>(environment_contract), &encoded);
  AppendU64(engine_adapter_contract.size(), &encoded);
  encoded.append(engine_adapter_contract.data(),
                 engine_adapter_contract.size());
  hasher.Update(encoded);
  return hasher.Finalize();
}

// Private, unlinked storage used while a capsule is unauthenticated or carries
// a transient per-request key. No pathname survives construction, the
// descriptor is never inherited across exec, and reads are unavailable until
// the writer seals the exact extent.
class UnlinkedTempByteFile final : public ByteSource,
                                   public ByteSink,
                                   public FreshWorkerTransientCapsuleAccess {
 public:
  UnlinkedTempByteFile(const UnlinkedTempByteFile&) = delete;
  UnlinkedTempByteFile& operator=(const UnlinkedTempByteFile&) = delete;
  UnlinkedTempByteFile(UnlinkedTempByteFile&&) noexcept = default;
  UnlinkedTempByteFile& operator=(UnlinkedTempByteFile&&) noexcept = default;

  static absl::StatusOr<UnlinkedTempByteFile> Create() {
    char path[] = "/tmp/litert-lmx-fresh-capsule.XXXXXX";
    const int descriptor = mkstemp(path);
    if (descriptor < 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to create private fresh-worker capsule storage.");
    }
    ScopedFd file(descriptor);
    if (fchmod(file.get(), S_IRUSR | S_IWUSR) != 0) {
      const int saved_errno = errno;
      unlink(path);
      return absl::ErrnoToStatus(
          saved_errno, "Unable to restrict fresh-worker capsule storage.");
    }
    const absl::Status close_on_exec = SetCloseOnExec(file.get());
    if (!close_on_exec.ok()) {
      unlink(path);
      return close_on_exec;
    }
    struct stat file_stat;
    if (fstat(file.get(), &file_stat) != 0) {
      const int saved_errno = errno;
      unlink(path);
      return absl::ErrnoToStatus(
          saved_errno, "Unable to inspect fresh-worker capsule storage.");
    }
    if (!S_ISREG(file_stat.st_mode)) {
      unlink(path);
      return absl::FailedPreconditionError(
          "Fresh-worker capsule storage is not a regular file.");
    }
    if (unlink(path) != 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to unlink private fresh-worker capsule storage.");
    }
    return UnlinkedTempByteFile(std::move(file));
  }

  uint64_t Size() const override { return size_; }

  ByteSink* sink() override { return static_cast<ByteSink*>(this); }

  absl::StatusOr<const ByteSource*> SealForRead() override {
    ABSL_RETURN_IF_ERROR(Seal());
    return static_cast<const ByteSource*>(this);
  }

  absl::Status Append(absl::string_view bytes) override {
    if (sealed_) {
      return absl::FailedPreconditionError(
          "Fresh-worker capsule storage is already sealed.");
    }
    if (bytes.size() > kMaximumFreshWorkerCapsuleBytes - size_) {
      return absl::ResourceExhaustedError(
          "Fresh-worker capsule exceeds its independent storage limit.");
    }
    size_t source_offset = 0;
    while (source_offset < bytes.size()) {
      if (size_ > static_cast<uint64_t>(
                      std::numeric_limits<off_t>::max())) {
        return absl::ResourceExhaustedError(
            "Fresh-worker capsule offset is not representable.");
      }
      const size_t count = std::min<size_t>(
          bytes.size() - source_offset,
          static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
      const ssize_t written =
          pwrite(file_.get(), bytes.data() + source_offset, count,
                 static_cast<off_t>(size_));
      if (written < 0 && errno == EINTR) continue;
      if (written < 0) {
        return absl::ErrnoToStatus(
            errno, "Unable to write fresh-worker capsule storage.");
      }
      if (written == 0) {
        return absl::DataLossError(
            "Fresh-worker capsule storage accepted zero bytes.");
      }
      source_offset += static_cast<size_t>(written);
      size_ += static_cast<uint64_t>(written);
    }
    return absl::OkStatus();
  }

  absl::Status Seal() {
    if (sealed_) return absl::OkStatus();
    struct stat file_stat;
    if (fstat(file_.get(), &file_stat) != 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to seal fresh-worker capsule storage.");
    }
    if (!S_ISREG(file_stat.st_mode) || file_stat.st_nlink != 0 ||
        file_stat.st_size < 0 ||
        static_cast<uint64_t>(file_stat.st_size) != size_) {
      return absl::DataLossError(
          "Fresh-worker capsule storage extent or ownership changed.");
    }
    sealed_ = true;
    return absl::OkStatus();
  }

  absl::Status ReadAt(uint64_t offset,
                      absl::Span<char> destination) const override {
    if (!sealed_) {
      return absl::FailedPreconditionError(
          "Fresh-worker capsule storage must be sealed before reading.");
    }
    if (offset > size_ || destination.size() > size_ - offset) {
      return absl::OutOfRangeError(
          "Fresh-worker capsule storage read exceeds bounds.");
    }
    size_t destination_offset = 0;
    while (destination_offset < destination.size()) {
      const uint64_t absolute_offset = offset + destination_offset;
      if (absolute_offset > static_cast<uint64_t>(
                                std::numeric_limits<off_t>::max())) {
        return absl::OutOfRangeError(
            "Fresh-worker capsule read offset is not representable.");
      }
      const size_t count = std::min<size_t>(
          destination.size() - destination_offset,
          static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
      const ssize_t read_count =
          pread(file_.get(), destination.data() + destination_offset, count,
                static_cast<off_t>(absolute_offset));
      if (read_count < 0 && errno == EINTR) continue;
      if (read_count < 0) {
        return absl::ErrnoToStatus(
            errno, "Unable to read fresh-worker capsule storage.");
      }
      if (read_count == 0) {
        return absl::DataLossError(
            "Fresh-worker capsule storage was truncated.");
      }
      destination_offset += static_cast<size_t>(read_count);
    }
    return absl::OkStatus();
  }

 private:
  explicit UnlinkedTempByteFile(ScopedFd file) : file_(std::move(file)) {}

  ScopedFd file_;
  uint64_t size_ = 0;
  bool sealed_ = false;
};

absl::StatusOr<Hash256> HashCapsuleSource(const ByteSource& source) {
  if (source.Size() > kMaximumFreshWorkerCapsuleBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker capsule exceeds its independent storage limit.");
  }
  Sha256Hasher hasher;
  const size_t chunk_size = static_cast<size_t>(std::min<uint64_t>(
      std::max<uint64_t>(source.Size(), 1), kCapsuleIoChunkBytes));
  std::vector<char> chunk(chunk_size);
  uint64_t offset = 0;
  while (offset < source.Size()) {
    const size_t count = static_cast<size_t>(std::min<uint64_t>(
        chunk.size(), source.Size() - offset));
    ABSL_RETURN_IF_ERROR(source.ReadAt(
        offset, absl::MakeSpan(chunk).subspan(0, count)));
    hasher.Update(absl::string_view(chunk.data(), count));
    offset += count;
  }
  return hasher.Finalize();
}

class BoundedHashingByteSink final : public ByteSink {
 public:
  explicit BoundedHashingByteSink(ByteSink* destination)
      : destination_(destination) {}

  absl::Status Append(absl::string_view bytes) override {
    if (destination_ == nullptr) {
      return absl::InvalidArgumentError(
          "Durable capsule destination must not be null.");
    }
    if (bytes.size() > kMaximumFreshWorkerCapsuleBytes - size_) {
      return absl::ResourceExhaustedError(
          "Durable fresh-worker capsule exceeds its storage limit.");
    }
    ABSL_RETURN_IF_ERROR(destination_->Append(bytes));
    hasher_.Update(bytes);
    size_ += bytes.size();
    return absl::OkStatus();
  }

  uint64_t Size() const { return size_; }
  Hash256 Finalize() { return hasher_.Finalize(); }

 private:
  ByteSink* destination_;
  Sha256Hasher hasher_;
  uint64_t size_ = 0;
};

absl::Status SetNonBlocking(int fd) {
  const int flags = fcntl(fd, F_GETFL);
  if (flags < 0 || fcntl(fd, F_SETFL, flags | O_NONBLOCK) != 0) {
    return absl::ErrnoToStatus(errno,
                               "Unable to make worker pipe nonblocking.");
  }
  return absl::OkStatus();
}

absl::Status SuppressSigpipeOnWriteFd(int fd) {
#ifdef F_SETNOSIGPIPE
  if (fcntl(fd, F_SETNOSIGPIPE, 1) != 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to suppress SIGPIPE on a worker write pipe.");
  }
#endif
  return absl::OkStatus();
}

absl::StatusOr<Pipe> MakePipe() {
  int descriptors[2];
  int result;
  do {
    result = pipe(descriptors);
  } while (result != 0 && errno == EINTR);
  if (result != 0) {
    return absl::ErrnoToStatus(errno, "Unable to create worker pipe.");
  }
  Pipe pipe{.read = ScopedFd(descriptors[0]),
            .write = ScopedFd(descriptors[1])};
  const auto move_to_safe_descriptor = [](ScopedFd* descriptor) -> absl::Status {
    constexpr int kFirstSafeSourceFd = kFreshWorkerAuthenticationFd + 1;
    if (descriptor->get() >= kFirstSafeSourceFd) {
      return absl::OkStatus();
    }
    int duplicate;
    do {
#ifdef F_DUPFD_CLOEXEC
      duplicate =
          fcntl(descriptor->get(), F_DUPFD_CLOEXEC, kFirstSafeSourceFd);
#else
      duplicate = fcntl(descriptor->get(), F_DUPFD, kFirstSafeSourceFd);
#endif
    } while (duplicate < 0 && errno == EINTR);
    if (duplicate < 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to relocate a fresh-worker pipe descriptor.");
    }
    ScopedFd safe_duplicate(duplicate);
#ifndef F_DUPFD_CLOEXEC
    ABSL_RETURN_IF_ERROR(SetCloseOnExec(safe_duplicate.get()));
#endif
    descriptor->Reset(safe_duplicate.Release());
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(move_to_safe_descriptor(&pipe.read));
  ABSL_RETURN_IF_ERROR(move_to_safe_descriptor(&pipe.write));
  ABSL_RETURN_IF_ERROR(SetCloseOnExec(pipe.read.get()));
  ABSL_RETURN_IF_ERROR(SetCloseOnExec(pipe.write.get()));
  return pipe;
}

class ScopedSigpipeBlock {
 public:
  ScopedSigpipeBlock() {
#if defined(__APPLE__) && defined(F_SETNOSIGPIPE)
    // macOS suppresses SIGPIPE per write FD instead. There is no signal-mask
    // state to restore and no destructor wait.
    active_ = true;
#else
    sigemptyset(&set_);
    sigaddset(&set_, SIGPIPE);
    sigset_t pending;
    had_pending_sigpipe_ =
        sigpending(&pending) == 0 && sigismember(&pending, SIGPIPE) == 1;
    active_ = pthread_sigmask(SIG_BLOCK, &set_, &old_set_) == 0;
#endif
  }
  ScopedSigpipeBlock(const ScopedSigpipeBlock&) = delete;
  ScopedSigpipeBlock& operator=(const ScopedSigpipeBlock&) = delete;
  ~ScopedSigpipeBlock() {
    if (!active_) return;
#if defined(__APPLE__) && defined(F_SETNOSIGPIPE)
    return;
#else
    if (!had_pending_sigpipe_) {
#if defined(__linux__)
      timespec zero = {.tv_sec = 0, .tv_nsec = 0};
      while (sigtimedwait(&set_, nullptr, &zero) < 0 && errno == EINTR) {
      }
#else
      // Platforms without sigtimedwait: first enqueue a blocked,
      // thread-directed SIGPIPE. Standard-signal coalescing means one sigwait
      // then consumes either it or the already-pending write signal without a
      // pending-state check/wait race.
      if (pthread_kill(pthread_self(), SIGPIPE) == 0) {
        int received_signal = 0;
        sigwait(&set_, &received_signal);
      }
#endif
    }
    pthread_sigmask(SIG_SETMASK, &old_set_, nullptr);
#endif
  }

  bool active() const { return active_; }

 private:
  sigset_t set_{};
  sigset_t old_set_{};
  bool had_pending_sigpipe_ = false;
  bool active_ = false;
};

using SteadyClock = std::chrono::steady_clock;

class RunControl {
 public:
  explicit RunControl(const FreshWorkerProcessOptions& options)
      : options_(options),
        deadline_(SteadyClock::now() + std::chrono::nanoseconds(
                                           absl::ToInt64Nanoseconds(
                                               options.timeout))) {}

  absl::Status Check() const {
    if (options_.cancellation_requested &&
        options_.cancellation_requested()) {
      return absl::CancelledError("Fresh-worker run was cancelled.");
    }
    if (SteadyClock::now() >= deadline_) {
      return absl::DeadlineExceededError(
          "Fresh-worker run exceeded its deadline.");
    }
    return absl::OkStatus();
  }

  int PollMilliseconds() const {
    const auto remaining = deadline_ - SteadyClock::now();
    if (remaining <= SteadyClock::duration::zero()) return 0;
    const auto milliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(remaining);
    const int64_t rounded = std::max<int64_t>(1, milliseconds.count());
    return static_cast<int>(std::min<int64_t>(rounded, 50));
  }

  SteadyClock::time_point deadline() const { return deadline_; }

 private:
  const FreshWorkerProcessOptions& options_;
  SteadyClock::time_point deadline_;
};

absl::Status PollFor(int fd, short events, const RunControl& control) {
  while (true) {
    ABSL_RETURN_IF_ERROR(control.Check());
    pollfd descriptor{.fd = fd, .events = events, .revents = 0};
    const int result = poll(&descriptor, 1, control.PollMilliseconds());
    if (result < 0 && errno == EINTR) continue;
    if (result < 0) {
      return absl::ErrnoToStatus(errno, "Fresh-worker pipe polling failed.");
    }
    if (result == 0) continue;
    if ((descriptor.revents & POLLNVAL) != 0) {
      return absl::InternalError("Fresh-worker pipe became invalid.");
    }
    if ((descriptor.revents & (events | POLLERR | POLLHUP)) != 0) {
      return absl::OkStatus();
    }
  }
}

absl::Status WriteWithDeadline(int fd, absl::string_view bytes,
                               const RunControl& control) {
  size_t offset = 0;
  while (offset < bytes.size()) {
    ABSL_RETURN_IF_ERROR(PollFor(fd, POLLOUT, control));
    const ssize_t count = write(fd, bytes.data() + offset,
                                bytes.size() - offset);
    if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (count < 0) {
      return absl::ErrnoToStatus(errno,
                                 "Unable to write fresh-worker pipe.");
    }
    if (count == 0) {
      return absl::DataLossError("Fresh-worker pipe accepted zero bytes.");
    }
    offset += static_cast<size_t>(count);
  }
  return absl::OkStatus();
}

absl::Status WriteCapsuleFrameWithDeadline(
    int fd, const ByteSource* source, const RunControl& control) {
  const uint64_t size = source == nullptr ? 0 : source->Size();
  if (size > kMaximumFreshWorkerCapsuleBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker capsule frame exceeds its independent limit.");
  }
  const std::string header = EncodeFrameHeader(kCapsuleFrameMagic, size);
  ABSL_RETURN_IF_ERROR(WriteWithDeadline(fd, header, control));
  if (size == 0) return absl::OkStatus();

  std::vector<char> chunk(static_cast<size_t>(std::min<uint64_t>(
      size, kCapsuleIoChunkBytes)));
  uint64_t offset = 0;
  while (offset < size) {
    const size_t count = static_cast<size_t>(
        std::min<uint64_t>(chunk.size(), size - offset));
    ABSL_RETURN_IF_ERROR(
        source->ReadAt(offset, absl::MakeSpan(chunk).subspan(0, count)));
    ABSL_RETURN_IF_ERROR(WriteWithDeadline(
        fd, absl::string_view(chunk.data(), count), control));
    offset += count;
  }
  return absl::OkStatus();
}

absl::Status ReadExactWithDeadline(int fd, char* output, size_t size,
                                   const RunControl& control) {
  size_t offset = 0;
  while (offset < size) {
    ABSL_RETURN_IF_ERROR(PollFor(fd, POLLIN, control));
    const ssize_t count = read(fd, output + offset, size - offset);
    if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (count < 0) {
      return absl::ErrnoToStatus(errno,
                                 "Unable to read fresh-worker pipe.");
    }
    if (count == 0) {
      return absl::DataLossError("Fresh-worker response was truncated.");
    }
    offset += static_cast<size_t>(count);
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> ReadCapsuleFrameWithDeadline(
    int fd, UnlinkedTempByteFile* destination,
    const RunControl& control) {
  if (destination == nullptr) {
    return absl::InvalidArgumentError(
        "Fresh-worker capsule destination is null.");
  }
  std::array<char, 16> header{};
  ABSL_RETURN_IF_ERROR(
      ReadExactWithDeadline(fd, header.data(), header.size(), control));
  if (std::memcmp(header.data(), kCapsuleFrameMagic.data(),
                  kCapsuleFrameMagic.size()) != 0) {
    return absl::DataLossError(
        "Fresh-worker capsule has invalid framing magic.");
  }
  const uint64_t size = ReadU64(header.data() + 8);
  if (size > kMaximumFreshWorkerCapsuleBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker capsule frame exceeds its independent limit.");
  }
  std::vector<char> chunk(static_cast<size_t>(std::min<uint64_t>(
      std::max<uint64_t>(size, 1), kCapsuleIoChunkBytes)));
  uint64_t offset = 0;
  while (offset < size) {
    const size_t count = static_cast<size_t>(
        std::min<uint64_t>(chunk.size(), size - offset));
    ABSL_RETURN_IF_ERROR(
        ReadExactWithDeadline(fd, chunk.data(), count, control));
    ABSL_RETURN_IF_ERROR(destination->Append(
        absl::string_view(chunk.data(), count)));
    offset += count;
  }
  ABSL_RETURN_IF_ERROR(destination->Seal());
  return size;
}

absl::Status ReadZeroCapsuleFrameWithDeadline(
    int fd, const RunControl& control) {
  std::array<char, 16> header{};
  ABSL_RETURN_IF_ERROR(
      ReadExactWithDeadline(fd, header.data(), header.size(), control));
  if (std::memcmp(header.data(), kCapsuleFrameMagic.data(),
                  kCapsuleFrameMagic.size()) != 0 ||
      ReadU64(header.data() + 8) != 0) {
    return absl::DataLossError(
        "Capsule-free fresh worker emitted a nonzero or invalid capsule "
        "frame.");
  }
  return absl::OkStatus();
}

absl::Status ExpectEofWithDeadline(int fd, const RunControl& control) {
  char trailing;
  while (true) {
    ABSL_RETURN_IF_ERROR(PollFor(fd, POLLIN, control));
    const ssize_t count = read(fd, &trailing, 1);
    if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) {
      continue;
    }
    if (count < 0) {
      return absl::ErrnoToStatus(errno,
                                 "Unable to finish fresh-worker response.");
    }
    if (count == 0) return absl::OkStatus();
    return absl::DataLossError(
        "Fresh-worker emitted more than one response frame.");
  }
}

absl::Status ReadExactBlocking(int fd, char* output, size_t size,
                               absl::string_view description) {
  size_t offset = 0;
  while (offset < size) {
    const ssize_t count = read(fd, output + offset, size - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      return absl::ErrnoToStatus(errno,
                                 absl::StrCat("Unable to read ", description));
    }
    if (count == 0) {
      return absl::DataLossError(absl::StrCat("Truncated ", description));
    }
    offset += static_cast<size_t>(count);
  }
  return absl::OkStatus();
}

absl::Status WriteAllBlocking(int fd, absl::string_view bytes,
                              absl::string_view description) {
  size_t offset = 0;
  while (offset < bytes.size()) {
    const ssize_t count =
        write(fd, bytes.data() + offset, bytes.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("Unable to write ", description));
    }
    if (count == 0) {
      return absl::DataLossError(
          absl::StrCat("Zero-byte write to ", description));
    }
    offset += static_cast<size_t>(count);
  }
  return absl::OkStatus();
}

absl::Status WriteCapsuleFrameBlocking(int fd, const ByteSource* source,
                                       absl::string_view description) {
  const uint64_t size = source == nullptr ? 0 : source->Size();
  if (size > kMaximumFreshWorkerCapsuleBytes) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds the capsule framing limit."));
  }
  ABSL_RETURN_IF_ERROR(WriteAllBlocking(
      fd, EncodeFrameHeader(kCapsuleFrameMagic, size), description));
  if (size == 0) return absl::OkStatus();

  std::vector<char> chunk(static_cast<size_t>(std::min<uint64_t>(
      size, kCapsuleIoChunkBytes)));
  uint64_t offset = 0;
  while (offset < size) {
    const size_t count = static_cast<size_t>(
        std::min<uint64_t>(chunk.size(), size - offset));
    ABSL_RETURN_IF_ERROR(
        source->ReadAt(offset, absl::MakeSpan(chunk).subspan(0, count)));
    ABSL_RETURN_IF_ERROR(WriteAllBlocking(
        fd, absl::string_view(chunk.data(), count), description));
    offset += count;
  }
  return absl::OkStatus();
}

absl::Status ExpectEofBlocking(int fd, absl::string_view description) {
  char trailing;
  while (true) {
    const ssize_t count = read(fd, &trailing, 1);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      return absl::ErrnoToStatus(
          errno, absl::StrCat("Unable to finish reading ", description));
    }
    if (count == 0) return absl::OkStatus();
    return absl::DataLossError(
        absl::StrCat(description, " has trailing bytes."));
  }
}

absl::StatusOr<std::string> ReadIpcFrameBlockingWithoutEof(
    int fd, absl::string_view description) {
  std::array<char, 16> header{};
  ABSL_RETURN_IF_ERROR(
      ReadExactBlocking(fd, header.data(), header.size(), description));
  if (std::memcmp(header.data(), kIpcFrameMagic.data(),
                  kIpcFrameMagic.size()) != 0) {
    return absl::DataLossError(
        absl::StrCat(description, " has invalid framing magic."));
  }
  const uint64_t envelope_size = ReadU64(header.data() + 8);
  if (envelope_size == 0 ||
      envelope_size > kMaximumFreshWorkerEnvelopeBytes ||
      envelope_size > std::string().max_size()) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds the framing limit."));
  }
  std::string envelope(static_cast<size_t>(envelope_size), '\0');
  ABSL_RETURN_IF_ERROR(ReadExactBlocking(fd, envelope.data(), envelope.size(),
                                         description));
  return envelope;
}

absl::StatusOr<uint64_t> ReadCapsuleFrameBlocking(
    int fd, UnlinkedTempByteFile* destination,
    absl::string_view description) {
  if (destination == nullptr) {
    return absl::InvalidArgumentError(
        "Fresh-worker capsule destination is null.");
  }
  std::array<char, 16> header{};
  ABSL_RETURN_IF_ERROR(
      ReadExactBlocking(fd, header.data(), header.size(), description));
  if (std::memcmp(header.data(), kCapsuleFrameMagic.data(),
                  kCapsuleFrameMagic.size()) != 0) {
    return absl::DataLossError(
        absl::StrCat(description, " has invalid framing magic."));
  }
  const uint64_t size = ReadU64(header.data() + 8);
  if (size > kMaximumFreshWorkerCapsuleBytes) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds the capsule framing limit."));
  }
  std::vector<char> chunk(static_cast<size_t>(std::min<uint64_t>(
      std::max<uint64_t>(size, 1), kCapsuleIoChunkBytes)));
  uint64_t offset = 0;
  while (offset < size) {
    const size_t count = static_cast<size_t>(
        std::min<uint64_t>(chunk.size(), size - offset));
    ABSL_RETURN_IF_ERROR(
        ReadExactBlocking(fd, chunk.data(), count, description));
    ABSL_RETURN_IF_ERROR(destination->Append(
        absl::string_view(chunk.data(), count)));
    offset += count;
  }
  ABSL_RETURN_IF_ERROR(destination->Seal());
  return size;
}

absl::Status ReadZeroCapsuleFrameBlocking(
    int fd, absl::string_view description) {
  std::array<char, 16> header{};
  ABSL_RETURN_IF_ERROR(
      ReadExactBlocking(fd, header.data(), header.size(), description));
  if (std::memcmp(header.data(), kCapsuleFrameMagic.data(),
                  kCapsuleFrameMagic.size()) != 0 ||
      ReadU64(header.data() + 8) != 0) {
    return absl::DataLossError(
        absl::StrCat(description,
                     " must be an explicit zero-size capsule frame."));
  }
  return absl::OkStatus();
}

absl::Status WriteIpcFrameBlocking(int fd, absl::string_view envelope,
                                   absl::string_view description) {
  if (envelope.empty() || envelope.size() > kMaximumFreshWorkerEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        absl::StrCat(description, " exceeds the framing limit."));
  }
  return WriteAllBlocking(fd, EncodeIpcFrame(envelope), description);
}

absl::StatusOr<FreshWorkerAuthentication> ReadAuthenticationPrelude(int fd) {
  std::array<char, 20> header{};
  ABSL_RETURN_IF_ERROR(ReadExactBlocking(fd, header.data(), header.size(),
                                         "worker authentication prelude"));
  if (std::memcmp(header.data(), kAuthenticationMagic.data(),
                  kAuthenticationMagic.size()) != 0 ||
      ReadU32(header.data() + 8) != kAuthenticationPreludeFormatVersion) {
    return absl::DataLossError(
        "Fresh-worker authentication prelude is invalid.");
  }
  const uint32_t key_id_size = ReadU32(header.data() + 12);
  const uint32_t key_size = ReadU32(header.data() + 16);
  if (key_id_size == 0 || key_id_size > kMaximumAuthenticationKeyIdBytes ||
      key_size < kMinimumAuthenticationKeyBytes ||
      key_size > kMaximumAuthenticationKeyBytes) {
    return absl::ResourceExhaustedError(
        "Fresh-worker authentication prelude exceeds its limits.");
  }
  FreshWorkerAuthentication authentication;
  authentication.key_id.resize(key_id_size);
  authentication.authentication_key.resize(key_size);
  ABSL_RETURN_IF_ERROR(ReadExactBlocking(
      fd, authentication.key_id.data(), authentication.key_id.size(),
      "worker authentication key ID"));
  ABSL_RETURN_IF_ERROR(ReadExactBlocking(
      fd, authentication.authentication_key.data(),
      authentication.authentication_key.size(), "worker authentication key"));
  ABSL_RETURN_IF_ERROR(
      ExpectEofBlocking(fd, "worker authentication prelude"));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  return authentication;
}

class SpawnActions {
 public:
  SpawnActions() : initialized_(posix_spawn_file_actions_init(&actions_) == 0) {}
  SpawnActions(const SpawnActions&) = delete;
  SpawnActions& operator=(const SpawnActions&) = delete;
  ~SpawnActions() {
    if (initialized_) posix_spawn_file_actions_destroy(&actions_);
  }
  bool initialized() const { return initialized_; }
  posix_spawn_file_actions_t* get() { return &actions_; }

 private:
  posix_spawn_file_actions_t actions_{};
  bool initialized_ = false;
};

class SpawnAttributes {
 public:
  SpawnAttributes() : initialized_(posix_spawnattr_init(&attributes_) == 0) {}
  SpawnAttributes(const SpawnAttributes&) = delete;
  SpawnAttributes& operator=(const SpawnAttributes&) = delete;
  ~SpawnAttributes() {
    if (initialized_) posix_spawnattr_destroy(&attributes_);
  }
  bool initialized() const { return initialized_; }
  posix_spawnattr_t* get() { return &attributes_; }

 private:
  posix_spawnattr_t attributes_{};
  bool initialized_ = false;
};

class ChildGuard {
 public:
  ChildGuard(pid_t pid, absl::Duration grace) : pid_(pid), grace_(grace) {}
  ChildGuard(const ChildGuard&) = delete;
  ChildGuard& operator=(const ChildGuard&) = delete;
  ~ChildGuard() { TerminateAndReap(); }

  absl::StatusOr<int> Wait(const RunControl& control) {
    while (true) {
      ABSL_RETURN_IF_ERROR(control.Check());
      int status = 0;
      const pid_t result = waitpid(pid_, &status, WNOHANG);
      if (result == pid_) {
        const pid_t process_group = pid_;
        pid_ = -1;
        if (kill(-process_group, 0) == 0 || errno == EPERM) {
          kill(-process_group, SIGKILL);
          return absl::FailedPreconditionError(
              "Fresh worker left another process in its isolated process group.");
        }
        return status;
      }
      if (result < 0 && errno == EINTR) continue;
      if (result < 0) {
        return absl::ErrnoToStatus(errno,
                                   "Unable to wait for fresh worker.");
      }
      poll(nullptr, 0, std::max(1, control.PollMilliseconds()));
    }
  }

 private:
  void TerminateAndReap() {
    if (pid_ <= 0) return;
    // Signal the isolated group for descendants and the direct child as a
    // fallback if hostile worker code moved itself to another process group.
    kill(-pid_, SIGTERM);
    kill(pid_, SIGTERM);
    const auto grace_deadline =
        SteadyClock::now() + std::chrono::nanoseconds(
                                 absl::ToInt64Nanoseconds(grace_));
    while (SteadyClock::now() < grace_deadline) {
      int status = 0;
      const pid_t result = waitpid(pid_, &status, WNOHANG);
      if (result == pid_ || (result < 0 && errno == ECHILD)) {
        const pid_t process_group = pid_;
        pid_ = -1;
        kill(-process_group, SIGKILL);
        return;
      }
      if (result < 0 && errno != EINTR) break;
      poll(nullptr, 0, 10);
    }
    kill(-pid_, SIGKILL);
    kill(pid_, SIGKILL);
    int status = 0;
    while (waitpid(pid_, &status, 0) < 0 && errno == EINTR) {
    }
    pid_ = -1;
  }

  pid_t pid_ = -1;
  absl::Duration grace_;
};

absl::Status AddDupAction(posix_spawn_file_actions_t* actions, int source,
                          int target) {
  const int error = posix_spawn_file_actions_adddup2(actions, source, target);
  if (error != 0) {
    return absl::ErrnoToStatus(error,
                               "Unable to configure fresh-worker pipe.");
  }
  return absl::OkStatus();
}

absl::StatusOr<pid_t> SpawnWorker(
    const FreshWorkerCertification& certification, int request_read_fd,
    int result_write_fd, int authentication_read_fd) {
  SpawnActions actions;
  SpawnAttributes attributes;
  if (!actions.initialized() || !attributes.initialized()) {
    return absl::InternalError(
        "Unable to initialize fresh-worker spawn configuration.");
  }
  ABSL_RETURN_IF_ERROR(AddDupAction(actions.get(), request_read_fd, STDIN_FILENO));
  ABSL_RETURN_IF_ERROR(AddDupAction(actions.get(), result_write_fd, STDOUT_FILENO));
  ABSL_RETURN_IF_ERROR(AddDupAction(actions.get(), authentication_read_fd,
                                    kFreshWorkerAuthenticationFd));
  const int stderr_error = posix_spawn_file_actions_addopen(
      actions.get(), STDERR_FILENO, "/dev/null", O_WRONLY, 0);
  if (stderr_error != 0) {
    return absl::ErrnoToStatus(
        stderr_error, "Unable to configure fresh-worker stderr.");
  }

  sigset_t empty_mask;
  sigemptyset(&empty_mask);
  int error = posix_spawnattr_setsigmask(attributes.get(), &empty_mask);
  if (error == 0) {
    error = posix_spawnattr_setpgroup(attributes.get(), 0);
  }
  if (error == 0) {
    short spawn_flags = static_cast<short>(POSIX_SPAWN_SETSIGMASK |
                                           POSIX_SPAWN_SETPGROUP);
#ifdef POSIX_SPAWN_CLOEXEC_DEFAULT
    // On platforms that provide it, admit only the three descriptors named by
    // file actions instead of inheriting unrelated application descriptors.
    spawn_flags =
        static_cast<short>(spawn_flags | POSIX_SPAWN_CLOEXEC_DEFAULT);
#endif
    error = posix_spawnattr_setflags(attributes.get(), spawn_flags);
  }
  if (error != 0) {
    return absl::ErrnoToStatus(
        error, "Unable to configure fresh-worker signal mask.");
  }

  std::string executable(certification.canonical_executable_path());
  std::vector<std::string> owned_arguments;
  owned_arguments.reserve(certification.canonical_arguments().size() + 1);
  owned_arguments.push_back(executable);
  owned_arguments.insert(owned_arguments.end(),
                         certification.canonical_arguments().begin(),
                         certification.canonical_arguments().end());
  std::vector<char*> argv;
  argv.reserve(owned_arguments.size() + 1);
  for (std::string& argument : owned_arguments) argv.push_back(argument.data());
  argv.push_back(nullptr);

  // Exact workers never inherit ambient process state. In particular, model
  // selection, plugin search paths, precision flags, and backend toggles
  // cannot be smuggled past the certified EngineSettings through env vars.
  char* empty_environment[] = {nullptr};

  pid_t pid = -1;
  // The runner brackets this path-based launch with complete image and stable
  // identity checks. posix_spawn still resolves a pathname inside the OS; this
  // is change detection, not an atomic fd-based assertion of the executed
  // image.
  error = posix_spawn(&pid, executable.c_str(), actions.get(), attributes.get(),
                      argv.data(), empty_environment);
  if (error != 0) {
    return absl::ErrnoToStatus(error, "Unable to spawn fresh worker.");
  }
  return pid;
}

absl::StatusOr<std::string> ReadParentResultFrameWithoutEof(
    int fd, const RunControl& control) {
  std::array<char, 16> header{};
  ABSL_RETURN_IF_ERROR(
      ReadExactWithDeadline(fd, header.data(), header.size(), control));
  if (std::memcmp(header.data(), kIpcFrameMagic.data(),
                  kIpcFrameMagic.size()) != 0) {
    return absl::DataLossError(
        "Fresh-worker response has invalid framing magic.");
  }
  const uint64_t envelope_size = ReadU64(header.data() + 8);
  if (envelope_size == 0 ||
      envelope_size > kMaximumFreshWorkerEnvelopeBytes ||
      envelope_size > std::string().max_size()) {
    return absl::ResourceExhaustedError(
        "Fresh-worker response exceeds the framing limit.");
  }
  std::string envelope(static_cast<size_t>(envelope_size), '\0');
  ABSL_RETURN_IF_ERROR(ReadExactWithDeadline(
      fd, envelope.data(), envelope.size(), control));
  return envelope;
}

#endif  // defined(__APPLE__) || defined(__linux__) || defined(__unix__)

FreshWorkerResult MakeFailureResult(const FreshWorkerRequest& request,
                                    const Hash256& request_envelope_hash,
                                    const Hash256& worker_nonce,
                                    const absl::Status& status) {
  FreshWorkerResult result;
  result.exact_profile_hash = request.exact_profile_hash;
  result.qualification_id = request.qualification_id;
  result.run_index = request.run_index;
  result.run_count = request.run_count;
  result.challenge_nonce = request.challenge_nonce;
  result.request_envelope_hash = request_envelope_hash;
  result.worker_instance_nonce = worker_nonce;
  result.execution_plan_hash = request.execution_plan.plan_hash;
  result.status_code = status.ok() ? absl::StatusCode::kInternal : status.code();
  result.status_message = status.ok() ? "Worker execution failed."
                                      : std::string(status.message());
  if (result.status_message.empty()) result.status_message = "Worker failed.";
  if (result.status_message.size() > kMaximumFailureMessageBytes) {
    result.status_message.resize(kMaximumFailureMessageBytes);
  }
  return result;
}

}  // namespace

Hash256 ComputeFreshWorkerLaunchSpecHash(
    const Hash256& worker_certification_hash) {
  Sha256Hasher hasher;
  hasher.Update(kLaunchSpecDomain);
  std::string framing;
  AppendU32(FreshWorkerCertification::kFormatVersion, &framing);
  AppendHash(worker_certification_hash, &framing);
  hasher.Update(framing);
  return hasher.Finalize();
}

absl::StatusOr<FreshWorkerCertification> FreshWorkerCertification::Create(
    const FreshWorkerProcessOptions& options) {
  ABSL_RETURN_IF_ERROR(ValidateProcessOptions(options));
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  ABSL_ASSIGN_OR_RETURN(
      const std::string canonical_path,
      CanonicalExecutablePath(options.executable_path));
  ABSL_ASSIGN_OR_RETURN(const ExecutableInspection inspection,
                        InspectExecutableImage(canonical_path));

  FreshWorkerCertification certification;
  certification.canonical_executable_path_ = canonical_path;
  certification.canonical_arguments_ = options.arguments;
  certification.executable_image_hash_ = inspection.image_hash;
  certification.device_ = inspection.device;
  certification.inode_ = inspection.inode;
  certification.file_size_ = inspection.file_size;
  certification.file_mode_ = inspection.file_mode;
  certification.modification_time_seconds_ =
      inspection.modification_time_seconds;
  certification.modification_time_nanoseconds_ =
      inspection.modification_time_nanoseconds;
  certification.status_change_time_seconds_ =
      inspection.status_change_time_seconds;
  certification.status_change_time_nanoseconds_ =
      inspection.status_change_time_nanoseconds;
  certification.environment_contract_ =
      FreshWorkerEnvironmentContract::kEmpty;
  // This is the product's expected adapter contract under local launcher and
  // packaging trust. It contributes to authenticated evidence but does not infer
  // the executable's semantics from its bytes.
  certification.engine_adapter_contract_.assign(
      kEngineFreshWorkerAdapterContract.data(),
      kEngineFreshWorkerAdapterContract.size());
  certification.certification_hash_ = ComputeWorkerCertificationHash(
      certification.canonical_executable_path_,
      certification.canonical_arguments_, inspection,
      certification.environment_contract_,
      certification.engine_adapter_contract_);
  if (IsZeroHash(certification.executable_image_hash_) ||
      IsZeroHash(certification.certification_hash_)) {
    return absl::InternalError(
        "Fresh-worker certification produced an empty digest.");
  }
  return certification;
#else
  return absl::UnimplementedError(
      "Fresh-worker executable certification is not implemented on this "
      "platform.");
#endif
}

absl::Status FreshWorkerCertification::ValidateExecutableImage() const {
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  if (format_version_ != kFormatVersion ||
      canonical_executable_path_.empty() ||
      environment_contract_ != FreshWorkerEnvironmentContract::kEmpty ||
      absl::string_view(engine_adapter_contract_) !=
          kEngineFreshWorkerAdapterContract ||
      IsZeroHash(executable_image_hash_) || IsZeroHash(certification_hash_)) {
    return absl::FailedPreconditionError(
        "Fresh-worker certification contract is incomplete or unsupported.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const std::string current_canonical_path,
      CanonicalExecutablePath(canonical_executable_path_));
  if (current_canonical_path != canonical_executable_path_) {
    return absl::AbortedError(
        "Fresh-worker executable canonical path changed after certification.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const ExecutableInspection current,
      InspectExecutableImage(canonical_executable_path_));
  if (current.device != device_ || current.inode != inode_ ||
      current.file_size != file_size_ || current.file_mode != file_mode_ ||
      current.modification_time_seconds != modification_time_seconds_ ||
      current.modification_time_nanoseconds !=
          modification_time_nanoseconds_ ||
      current.status_change_time_seconds != status_change_time_seconds_ ||
      current.status_change_time_nanoseconds !=
          status_change_time_nanoseconds_ ||
      current.image_hash != executable_image_hash_) {
    return absl::AbortedError(
        "Fresh-worker executable image or stable file identity changed after "
        "certification.");
  }
  const Hash256 expected_certification_hash = ComputeWorkerCertificationHash(
      canonical_executable_path_, canonical_arguments_, current,
      environment_contract_, engine_adapter_contract_);
  if (expected_certification_hash != certification_hash_) {
    return absl::DataLossError(
        "Fresh-worker certification digest is internally inconsistent.");
  }
  return absl::OkStatus();
#else
  return absl::UnimplementedError(
      "Fresh-worker executable certification is not implemented on this "
      "platform.");
#endif
}

absl::StatusOr<FreshWorkerProcessRunner> FreshWorkerProcessRunner::Create(
    FreshWorkerProcessOptions options) {
  ABSL_ASSIGN_OR_RETURN(FreshWorkerCertification certification,
                        FreshWorkerCertification::Create(options));
  options.executable_path = certification.canonical_executable_path();
  options.arguments = certification.canonical_arguments();
  return FreshWorkerProcessRunner(std::move(options),
                                  std::move(certification));
}

FreshWorkerProcessRunner::FreshWorkerProcessRunner(
    FreshWorkerProcessOptions options,
    FreshWorkerCertification certification)
    : options_(std::move(options)),
      certification_(std::move(certification)) {}

absl::StatusOr<FreshWorkerProcessObservation> FreshWorkerProcessRunner::Run(
    const FreshWorkerRequest& request,
    const FreshWorkerAuthentication& authentication) const {
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  ABSL_RETURN_IF_ERROR(ValidateProcessOptions(options_));
  ABSL_RETURN_IF_ERROR(certification_.ValidateExecutableImage());
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(request));
  if (request.execution_plan.prefill_mode !=
          FreshWorkerPrefillMode::kFullCanonicalPrefill ||
      request.execution_plan.capture_producing_capsule) {
    return absl::UnimplementedError(
        "The ordinary fresh-worker runner does not transport session "
        "capsules; use the authenticated capsule-stream path.");
  }
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_ASSIGN_OR_RETURN(
      FreshWorkerAuthentication transport_authentication,
      DeriveTransportAuthentication(authentication, request));
  SecretEraser transport_key_eraser(
      &transport_authentication.authentication_key);
  ABSL_ASSIGN_OR_RETURN(const std::string request_envelope,
                        EncodeFreshWorkerRequest(request,
                                                 transport_authentication));
  const Hash256 request_envelope_hash =
      HashFreshWorkerEnvelope(request_envelope);
  std::string request_frame = EncodeIpcFrame(request_envelope);
  ABSL_ASSIGN_OR_RETURN(std::string authentication_prelude,
                        BuildAuthenticationPrelude(transport_authentication));
  SecretEraser prelude_eraser(&authentication_prelude);

  ABSL_ASSIGN_OR_RETURN(Pipe request_pipe, MakePipe());
  ABSL_ASSIGN_OR_RETURN(Pipe result_pipe, MakePipe());
  ABSL_ASSIGN_OR_RETURN(Pipe authentication_pipe, MakePipe());

  ScopedSigpipeBlock sigpipe_block;
  if (!sigpipe_block.active()) {
    return absl::InternalError(
        "Unable to block SIGPIPE for fresh-worker communication.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const pid_t pid,
      SpawnWorker(certification_, request_pipe.read.get(),
                  result_pipe.write.get(), authentication_pipe.read.get()));
  ChildGuard child(pid, options_.termination_grace);
  ABSL_RETURN_IF_ERROR(certification_.ValidateExecutableImage());
  request_pipe.read.Reset();
  result_pipe.write.Reset();
  authentication_pipe.read.Reset();

  ABSL_RETURN_IF_ERROR(SetNonBlocking(request_pipe.write.get()));
  ABSL_RETURN_IF_ERROR(SetNonBlocking(result_pipe.read.get()));
  ABSL_RETURN_IF_ERROR(SetNonBlocking(authentication_pipe.write.get()));
  ABSL_RETURN_IF_ERROR(SuppressSigpipeOnWriteFd(request_pipe.write.get()));
  ABSL_RETURN_IF_ERROR(
      SuppressSigpipeOnWriteFd(authentication_pipe.write.get()));
  const RunControl control(options_);

  ABSL_RETURN_IF_ERROR(WriteWithDeadline(authentication_pipe.write.get(),
                                         authentication_prelude, control));
  authentication_pipe.write.Reset();
  SecureErase(&authentication_prelude);
  ABSL_RETURN_IF_ERROR(
      WriteWithDeadline(request_pipe.write.get(), request_frame, control));
  request_frame.clear();
  ABSL_RETURN_IF_ERROR(WriteCapsuleFrameWithDeadline(
      request_pipe.write.get(), nullptr, control));
  request_pipe.write.Reset();

  ABSL_ASSIGN_OR_RETURN(
      const std::string result_envelope,
      ReadParentResultFrameWithoutEof(result_pipe.read.get(), control));
  ABSL_RETURN_IF_ERROR(
      ReadZeroCapsuleFrameWithDeadline(result_pipe.read.get(), control));
  ABSL_RETURN_IF_ERROR(ExpectEofWithDeadline(result_pipe.read.get(), control));
  result_pipe.read.Reset();
  ABSL_ASSIGN_OR_RETURN(const int process_status, child.Wait(control));
  ABSL_RETURN_IF_ERROR(certification_.ValidateExecutableImage());
  if (!WIFEXITED(process_status) || WEXITSTATUS(process_status) != 0) {
    return absl::DataLossError(
        "Fresh worker did not exit successfully after its capsule-free "
        "response frames.");
  }
  ABSL_ASSIGN_OR_RETURN(
      FreshWorkerResult result,
      DecodeFreshWorkerResult(result_envelope, transport_authentication));
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerResultForRequest(result, request));
  if (result.exact_profile_hash != request.exact_profile_hash ||
      result.qualification_id != request.qualification_id ||
      result.run_index != request.run_index ||
      result.run_count != request.run_count ||
      result.challenge_nonce != request.challenge_nonce ||
      result.request_envelope_hash != request_envelope_hash ||
      result.execution_plan_hash != request.execution_plan.plan_hash) {
    return absl::UnauthenticatedError(
        "Capsule-free fresh-worker result is not bound to its exact request "
        "or emitted an unexpected capsule.");
  }
  return FreshWorkerProcessObservation{
      .result = std::move(result),
      .process_id = static_cast<int64_t>(pid),
      .request_envelope_hash = request_envelope_hash,
      .result_envelope_hash = HashFreshWorkerEnvelope(result_envelope),
      .worker_certification_hash = certification_.certification_hash(),
      .launch_spec_hash = ComputeFreshWorkerLaunchSpecHash(
          certification_.certification_hash())};
#else
  return absl::UnimplementedError(
      "Fresh-worker process isolation is not implemented on this platform.");
#endif
}

absl::StatusOr<FreshWorkerProcessObservation>
FreshWorkerProcessRunner::RunWithSessionHandoff(
    const FreshWorkerRequest& request,
    const FreshWorkerAuthentication& authentication,
    const FreshWorkerSessionHandoffTransfer& transfer) const {
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  ABSL_RETURN_IF_ERROR(ValidateProcessOptions(options_));
  ABSL_RETURN_IF_ERROR(certification_.ValidateExecutableImage());
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(request));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffTransfer(request, transfer));
  ABSL_ASSIGN_OR_RETURN(
      FreshWorkerAuthentication transport_authentication,
      DeriveTransportAuthentication(authentication, request));
  SecretEraser transport_key_eraser(
      &transport_authentication.authentication_key);

  // Encode and authenticate the request before deriving capsule keys. The
  // exact envelope hash is a non-circular physical-execution binding because
  // capsule bytes never enter this envelope.
  ABSL_ASSIGN_OR_RETURN(
      const std::string request_envelope,
      EncodeFreshWorkerRequest(request, transport_authentication));
  const Hash256 request_envelope_hash =
      HashFreshWorkerEnvelope(request_envelope);
  std::string request_frame = EncodeIpcFrame(request_envelope);

  const bool restore_required =
      request.execution_plan.prefill_mode ==
      FreshWorkerPrefillMode::kOwnPositionCapsuleDelta;
  const std::optional<SessionHandoffIdentity> expected_session_identity =
      restore_required
          ? std::optional<SessionHandoffIdentity>(
                request.execution_plan.session_identity)
          : std::nullopt;
  SessionHandoffOptions transient_restore_options = DeriveCapsuleOptions(
      transport_authentication, request_envelope_hash,
      kRestoreCapsuleDirection, kFreshWorkerTransientRestoreKeyId,
      expected_session_identity);
  SecretEraser transient_restore_key_eraser(
      &transient_restore_options.authentication_key);
  SessionHandoffOptions transient_producing_options = DeriveCapsuleOptions(
      transport_authentication, request_envelope_hash,
      kProducingCapsuleDirection, kFreshWorkerTransientProducingKeyId,
      expected_session_identity);
  if (request.execution_plan.capture_producing_capsule) {
    transient_producing_options.expected_identity =
        *transfer.durable_capture_options->expected_identity;
  }
  SecretEraser transient_producing_key_eraser(
      &transient_producing_options.authentication_key);

  ABSL_ASSIGN_OR_RETURN(UnlinkedTempByteFile transient_restore,
                        UnlinkedTempByteFile::Create());
  std::optional<Hash256> durable_restore_hash;
  std::optional<SessionHandoffReauthenticationEvidence>
      restore_reauthentication_evidence;
  if (restore_required) {
    ABSL_ASSIGN_OR_RETURN(
        durable_restore_hash,
        HashCapsuleSource(*transfer.durable_restore_source));
    if (*durable_restore_hash !=
        request.execution_plan.restore_durable_envelope_hash) {
      return absl::DataLossError(
          "Durable restore capsule hash differs from its execution-plan "
          "binding.");
    }
    ABSL_ASSIGN_OR_RETURN(
        restore_reauthentication_evidence,
        ReauthenticateSessionHandoffToWithEvidence(
            *transfer.durable_restore_source,
            request.execution_plan.session_identity,
            *transfer.durable_restore_options, transient_restore_options,
            kFreshWorkerDurableRestoreToTransientReauthenticationPurpose,
            &transient_restore));
  }
  ABSL_RETURN_IF_ERROR(transient_restore.Seal());
  if (restore_required && transient_restore.Size() == 0) {
    return absl::DataLossError(
        "Authenticated restore capsule rewrap produced no bytes.");
  }
  std::optional<Hash256> transient_restore_hash;
  if (restore_required) {
    ABSL_ASSIGN_OR_RETURN(transient_restore_hash,
                          HashCapsuleSource(transient_restore));
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffReauthenticationEvidence(
        *restore_reauthentication_evidence));
    const SessionHandoffReauthenticationEvidence& reauthentication =
        *restore_reauthentication_evidence;
    if (reauthentication.session_identity !=
            request.execution_plan.session_identity ||
        reauthentication.purpose !=
            kFreshWorkerDurableRestoreToTransientReauthenticationPurpose ||
        reauthentication.source_envelope_hash !=
            request.execution_plan.restore_durable_envelope_hash ||
        reauthentication.source_envelope_hash != *durable_restore_hash ||
        reauthentication.source_envelope_size !=
            request.execution_plan.restore_durable_envelope_size ||
        reauthentication.source_envelope_size !=
            transfer.durable_restore_source->Size() ||
        reauthentication.source_key_id !=
            transfer.durable_restore_options->key_id ||
        reauthentication.destination_envelope_hash !=
            *transient_restore_hash ||
        reauthentication.destination_envelope_size !=
            transient_restore.Size() ||
        reauthentication.destination_key_id !=
            transient_restore_options.key_id) {
      return absl::DataLossError(
          "Restore reauthentication evidence does not bind the selected "
          "durable capsule and exact transient worker envelope.");
    }
  }

  // Allocate response storage before spawning when capture is requested so
  // every post-spawn local failure is a transport/process failure, never an
  // accidental fallback to materializing capsule bytes in memory.
  std::optional<UnlinkedTempByteFile> transient_producing;
  if (request.execution_plan.capture_producing_capsule) {
    ABSL_ASSIGN_OR_RETURN(UnlinkedTempByteFile storage,
                          UnlinkedTempByteFile::Create());
    transient_producing.emplace(std::move(storage));
  }
  ABSL_ASSIGN_OR_RETURN(std::string authentication_prelude,
                        BuildAuthenticationPrelude(transport_authentication));
  SecretEraser prelude_eraser(&authentication_prelude);

  ABSL_ASSIGN_OR_RETURN(Pipe request_pipe, MakePipe());
  ABSL_ASSIGN_OR_RETURN(Pipe result_pipe, MakePipe());
  ABSL_ASSIGN_OR_RETURN(Pipe authentication_pipe, MakePipe());

  ScopedSigpipeBlock sigpipe_block;
  if (!sigpipe_block.active()) {
    return absl::InternalError(
        "Unable to block SIGPIPE for fresh-worker communication.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const pid_t pid,
      SpawnWorker(certification_, request_pipe.read.get(),
                  result_pipe.write.get(), authentication_pipe.read.get()));
  ChildGuard child(pid, options_.termination_grace);
  ABSL_RETURN_IF_ERROR(certification_.ValidateExecutableImage());
  request_pipe.read.Reset();
  result_pipe.write.Reset();
  authentication_pipe.read.Reset();

  ABSL_RETURN_IF_ERROR(SetNonBlocking(request_pipe.write.get()));
  ABSL_RETURN_IF_ERROR(SetNonBlocking(result_pipe.read.get()));
  ABSL_RETURN_IF_ERROR(SetNonBlocking(authentication_pipe.write.get()));
  ABSL_RETURN_IF_ERROR(SuppressSigpipeOnWriteFd(request_pipe.write.get()));
  ABSL_RETURN_IF_ERROR(
      SuppressSigpipeOnWriteFd(authentication_pipe.write.get()));
  const RunControl control(options_);

  ABSL_RETURN_IF_ERROR(WriteWithDeadline(authentication_pipe.write.get(),
                                         authentication_prelude, control));
  authentication_pipe.write.Reset();
  SecureErase(&authentication_prelude);
  ABSL_RETURN_IF_ERROR(
      WriteWithDeadline(request_pipe.write.get(), request_frame, control));
  request_frame.clear();
  ABSL_RETURN_IF_ERROR(WriteCapsuleFrameWithDeadline(
      request_pipe.write.get(), restore_required ? &transient_restore : nullptr,
      control));
  request_pipe.write.Reset();

  ABSL_ASSIGN_OR_RETURN(
      const std::string result_envelope,
      ReadParentResultFrameWithoutEof(result_pipe.read.get(), control));
  uint64_t transient_producing_size = 0;
  if (request.execution_plan.capture_producing_capsule) {
    ABSL_ASSIGN_OR_RETURN(
        transient_producing_size,
        ReadCapsuleFrameWithDeadline(result_pipe.read.get(),
                                     &*transient_producing, control));
    ABSL_RETURN_IF_ERROR(transient_producing->Seal());
  } else {
    ABSL_RETURN_IF_ERROR(
        ReadZeroCapsuleFrameWithDeadline(result_pipe.read.get(), control));
  }
  ABSL_RETURN_IF_ERROR(ExpectEofWithDeadline(result_pipe.read.get(), control));
  result_pipe.read.Reset();
  ABSL_ASSIGN_OR_RETURN(const int process_status, child.Wait(control));
  ABSL_RETURN_IF_ERROR(certification_.ValidateExecutableImage());
  if (!WIFEXITED(process_status) || WEXITSTATUS(process_status) != 0) {
    return absl::DataLossError(
        "Fresh worker did not exit successfully after its response and "
        "capsule frames.");
  }

  ABSL_ASSIGN_OR_RETURN(
      FreshWorkerResult result,
      DecodeFreshWorkerResult(result_envelope, transport_authentication));
  ABSL_RETURN_IF_ERROR(
      ValidateFreshWorkerResultForRequest(result, request));
  if (result.request_envelope_hash != request_envelope_hash) {
    return absl::UnauthenticatedError(
        "Fresh-worker result is not bound to the authenticated request "
        "envelope.");
  }
  if (result.status_code == absl::StatusCode::kOk && restore_required) {
    if (!result.restored_state_witness.has_value() ||
        !transient_restore_hash.has_value()) {
      return absl::DataLossError(
          "Successful own-position worker omitted its live restore witness.");
    }
    const SessionContinuationStateWitness& restore_witness =
        *result.restored_state_witness;
    if (restore_witness.session_identity !=
            request.execution_plan.session_identity ||
        restore_witness.phase != SessionHandoffPhase::kDecoded ||
        !restore_witness.ran_decode || restore_witness.current_step <= 0 ||
        restore_witness.envelope_size != transient_restore.Size() ||
        restore_witness.envelope_hash != *transient_restore_hash ||
        restore_witness.key_id != transient_restore_options.key_id) {
      return absl::DataLossError(
          "Worker restore witness does not describe the exact authenticated "
          "transient capsule supplied by the parent.");
    }
  }

  std::optional<FreshWorkerDurableProducingCapsuleEvidence>
      durable_capsule_evidence;
  if (result.status_code != absl::StatusCode::kOk) {
    // Failed executions do not publish success-path capsule provenance even
    // when a restore rewrap necessarily occurred before worker launch.
    restore_reauthentication_evidence.reset();
    if (transient_producing_size != 0) {
      return absl::DataLossError(
          "Failed fresh worker emitted a producing capsule.");
    }
  } else if (request.execution_plan.capture_producing_capsule) {
    if (!result.producing_capsule_evidence.has_value()) {
      return absl::DataLossError(
          "Successful capture result omitted capsule evidence.");
    }
    const FreshWorkerProducingCapsuleEvidence& transient_evidence =
        *result.producing_capsule_evidence;
    if (transient_evidence.session_identity !=
        *transfer.durable_capture_options->expected_identity) {
      return absl::FailedPreconditionError(
          "Producing capsule identity differs from the parent-derived "
          "durable capture assertion.");
    }
    if (transient_producing_size == 0 ||
        transient_producing_size != transient_evidence.transient_envelope_size) {
      return absl::DataLossError(
          "Producing capsule size differs from authenticated result "
          "evidence.");
    }
    const SessionContinuationStateWitness& producing_witness =
        transient_evidence.producer_first_export;
    if (transient_evidence.producer_second_export != producing_witness ||
        transient_evidence.fresh_import_target != producing_witness ||
        producing_witness.session_identity !=
            transient_evidence.session_identity ||
        producing_witness.envelope_size != transient_producing_size ||
        producing_witness.envelope_hash !=
            transient_evidence.transient_envelope_hash ||
        producing_witness.key_id != transient_producing_options.key_id) {
      return absl::DataLossError(
          "Producing continuation witnesses do not bind the exact "
          "transient capsule source endpoint.");
    }
    ABSL_ASSIGN_OR_RETURN(const Hash256 transient_hash,
                          HashCapsuleSource(*transient_producing));
    if (transient_hash != transient_evidence.transient_envelope_hash) {
      return absl::DataLossError(
          "Producing capsule hash differs from authenticated result "
          "evidence.");
    }
    // Authenticate the complete transient capsule before any durable output.
    absl::StatusOr<DecodedSessionHandoff> decoded_transient =
        DecodeSessionHandoffFrom(*transient_producing,
                                 transient_evidence.session_identity,
                                 transient_producing_options);
    if (!decoded_transient.ok()) return decoded_transient.status();

    BoundedHashingByteSink durable_sink(
        transfer.durable_capture_destination);
    ABSL_ASSIGN_OR_RETURN(
        SessionHandoffReauthenticationEvidence capture_reauthentication,
        ReauthenticateSessionHandoffToWithEvidence(
            *transient_producing, transient_evidence.session_identity,
            transient_producing_options,
            *transfer.durable_capture_options,
            kFreshWorkerTransientProducingToDurableReauthenticationPurpose,
            &durable_sink));
    if (durable_sink.Size() == 0) {
      return absl::DataLossError(
          "Durable producing-capsule rewrap produced no bytes.");
    }
    const Hash256 durable_hash = durable_sink.Finalize();
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffReauthenticationEvidence(
        capture_reauthentication));
    if (capture_reauthentication.session_identity !=
            transient_evidence.session_identity ||
        capture_reauthentication.purpose !=
            kFreshWorkerTransientProducingToDurableReauthenticationPurpose ||
        capture_reauthentication.source_envelope_hash != transient_hash ||
        capture_reauthentication.source_envelope_hash !=
            transient_evidence.transient_envelope_hash ||
        capture_reauthentication.source_envelope_size !=
            transient_producing_size ||
        capture_reauthentication.source_envelope_size !=
            transient_evidence.transient_envelope_size ||
        capture_reauthentication.source_key_id !=
            transient_producing_options.key_id ||
        capture_reauthentication.destination_envelope_hash != durable_hash ||
        capture_reauthentication.destination_envelope_size !=
            durable_sink.Size() ||
        capture_reauthentication.destination_key_id !=
            transfer.durable_capture_options->key_id) {
      return absl::DataLossError(
          "Producing-capsule reauthentication evidence does not bind the "
          "worker transient source and exact durable destination.");
    }
    durable_capsule_evidence =
        FreshWorkerDurableProducingCapsuleEvidence{
            .session_identity = transient_evidence.session_identity,
            .key_id = transfer.durable_capture_options->key_id,
            .envelope_size = durable_sink.Size(),
            .envelope_hash = durable_hash,
            .output_evidence_hash = transient_evidence.output_evidence_hash,
            .reauthentication_evidence =
                std::move(capture_reauthentication)};
  } else if (transient_producing_size != 0) {
    return absl::DataLossError(
        "Fresh worker emitted an unrequested producing capsule.");
  }

  return FreshWorkerProcessObservation{
      .result = std::move(result),
      .process_id = static_cast<int64_t>(pid),
      .request_envelope_hash = request_envelope_hash,
      .result_envelope_hash = HashFreshWorkerEnvelope(result_envelope),
      .worker_certification_hash = certification_.certification_hash(),
      .launch_spec_hash = ComputeFreshWorkerLaunchSpecHash(
          certification_.certification_hash()),
      .restore_reauthentication_evidence =
          std::move(restore_reauthentication_evidence),
      .durable_producing_capsule_evidence =
          std::move(durable_capsule_evidence)};
#else
  return absl::UnimplementedError(
      "Fresh-worker process isolation is not implemented on this platform.");
#endif
}

absl::Status RunFreshWorkerOnce(
    const FreshWorkerCapsuleFreeExecutionCallback& execute) {
  if (!execute) {
    return absl::InvalidArgumentError(
        "Fresh-worker execution callback is missing.");
  }
  return RunFreshWorkerOnce(FreshWorkerExecutionCallback(
      [&execute](const FreshWorkerExecutionContext& context)
          -> absl::StatusOr<FreshWorkerDerivedExecution> {
        if (context.request.execution_plan.prefill_mode !=
                FreshWorkerPrefillMode::kFullCanonicalPrefill ||
            context.request.execution_plan.capture_producing_capsule ||
            context.restore_source != nullptr ||
            context.restore_options != nullptr ||
            context.producing_sink != nullptr ||
            context.producing_options != nullptr ||
            context.producing_capsule_access != nullptr) {
          return absl::UnimplementedError(
              "The configured worker adapter is capsule-free.");
        }
        ABSL_ASSIGN_OR_RETURN(FreshWorkerDerivedExecution execution,
                              execute(context.request));
        if (execution.restored_checkpoint_id.has_value() ||
            execution.exported_producing_capsule) {
          return absl::FailedPreconditionError(
              "Capsule-free worker execution claimed restore or capture.");
        }
        return execution;
      }));
}

absl::Status RunFreshWorkerOnce(
    const FreshWorkerExecutionCallback& execute) {
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  if (!execute) {
    return absl::InvalidArgumentError(
        "Fresh-worker execution callback is missing.");
  }
  ScopedFd authentication_fd(kFreshWorkerAuthenticationFd);
  ABSL_ASSIGN_OR_RETURN(
      FreshWorkerAuthentication authentication,
      ReadAuthenticationPrelude(authentication_fd.get()));
  authentication_fd.Reset();
  SecretEraser authentication_eraser(&authentication.authentication_key);

  ScopedFd input_fd(STDIN_FILENO);
  ABSL_ASSIGN_OR_RETURN(
      const std::string request_envelope,
      ReadIpcFrameBlockingWithoutEof(input_fd.get(),
                                     "fresh-worker request"));
  ABSL_ASSIGN_OR_RETURN(
      const FreshWorkerRequest request,
      DecodeFreshWorkerRequest(request_envelope, authentication));
  const Hash256 request_envelope_hash =
      HashFreshWorkerEnvelope(request_envelope);
  ABSL_ASSIGN_OR_RETURN(const Hash256 worker_nonce,
                        GenerateFreshWorkerNonce());

  const bool restore_required =
      request.execution_plan.prefill_mode ==
      FreshWorkerPrefillMode::kOwnPositionCapsuleDelta;
  const std::optional<SessionHandoffIdentity> expected_session_identity =
      restore_required
          ? std::optional<SessionHandoffIdentity>(
                request.execution_plan.session_identity)
          : std::nullopt;
  SessionHandoffOptions transient_restore_options = DeriveCapsuleOptions(
      authentication, request_envelope_hash, kRestoreCapsuleDirection,
      kFreshWorkerTransientRestoreKeyId, expected_session_identity);
  SecretEraser transient_restore_key_eraser(
      &transient_restore_options.authentication_key);
  SessionHandoffOptions transient_producing_options = DeriveCapsuleOptions(
      authentication, request_envelope_hash, kProducingCapsuleDirection,
      kFreshWorkerTransientProducingKeyId, expected_session_identity);
  SecretEraser transient_producing_key_eraser(
      &transient_producing_options.authentication_key);

  const auto emit_result =
      [&](const FreshWorkerResult& result,
          const ByteSource* capsule) -> absl::Status {
    // Closing unread input prevents a malformed or locally failed child from
    // deadlocking with a parent that is still writing the capsule frame.
    input_fd.Reset();
    ABSL_RETURN_IF_ERROR(
        ValidateFreshWorkerResultForRequest(result, request));
    ABSL_ASSIGN_OR_RETURN(
        const std::string result_envelope,
        EncodeFreshWorkerResult(result, authentication));
    ABSL_RETURN_IF_ERROR(WriteIpcFrameBlocking(
        STDOUT_FILENO, result_envelope, "fresh-worker result"));
    return WriteCapsuleFrameBlocking(
        STDOUT_FILENO, capsule, "fresh-worker producing capsule");
  };
  const auto emit_failure = [&](const absl::Status& status) -> absl::Status {
    return emit_result(MakeFailureResult(request, request_envelope_hash,
                                         worker_nonce, status),
                       nullptr);
  };

  std::optional<UnlinkedTempByteFile> restore_storage;
  if (restore_required) {
    absl::StatusOr<UnlinkedTempByteFile> restore_storage_or =
        UnlinkedTempByteFile::Create();
    if (!restore_storage_or.ok()) {
      return emit_failure(restore_storage_or.status());
    }
    restore_storage.emplace(std::move(restore_storage_or).value());
    absl::StatusOr<uint64_t> restore_size_or = ReadCapsuleFrameBlocking(
        input_fd.get(), &*restore_storage, "fresh-worker restore capsule");
    if (!restore_size_or.ok()) {
      return emit_failure(restore_size_or.status());
    }
    if (*restore_size_or == 0) {
      return emit_failure(absl::DataLossError(
          "Own-position execution did not receive a restore capsule."));
    }
    const absl::Status seal_restore = restore_storage->Seal();
    if (!seal_restore.ok()) return emit_failure(seal_restore);
  } else {
    const absl::Status zero_restore = ReadZeroCapsuleFrameBlocking(
        input_fd.get(), "fresh-worker restore capsule");
    if (!zero_restore.ok()) return emit_failure(zero_restore);
  }
  const absl::Status input_eof =
      ExpectEofBlocking(input_fd.get(), "fresh-worker input frames");
  if (!input_eof.ok()) return emit_failure(input_eof);
  if (restore_required) {
    absl::StatusOr<DecodedSessionHandoff> decoded_restore =
        DecodeSessionHandoffFrom(*restore_storage,
                                 request.execution_plan.session_identity,
                                 transient_restore_options);
    if (!decoded_restore.ok()) {
      return emit_failure(decoded_restore.status());
    }
  }

  const bool capture_requested =
      request.execution_plan.capture_producing_capsule;
  std::optional<UnlinkedTempByteFile> producing_storage;
  if (capture_requested) {
    absl::StatusOr<UnlinkedTempByteFile> producing_storage_or =
        UnlinkedTempByteFile::Create();
    if (!producing_storage_or.ok()) {
      return emit_failure(producing_storage_or.status());
    }
    producing_storage.emplace(std::move(producing_storage_or).value());
  }
  const FreshWorkerExecutionContext context{
      .request = request,
      .restore_source = restore_required ? &*restore_storage : nullptr,
      .restore_options =
          restore_required ? &transient_restore_options : nullptr,
      .producing_sink = capture_requested ? &*producing_storage : nullptr,
      .producing_options =
          capture_requested ? &transient_producing_options : nullptr,
      .producing_capsule_access =
          capture_requested ? &*producing_storage : nullptr};

  absl::StatusOr<FreshWorkerDerivedExecution> execution = execute(context);
  if (!execution.ok()) return emit_failure(execution.status());
  if (execution->derived_exact_profile_hash != request.exact_profile_hash) {
    return emit_failure(absl::FailedPreconditionError(
        "Worker-derived ExactLiteRtProfile does not match the request."));
  }
  if (execution->replay_isolation !=
      FreshWorkerReplayIsolation::kEmptyCatalogs) {
    return emit_failure(absl::FailedPreconditionError(
        "Exact worker did not attest catalog-free construction."));
  }
  if (execution->prepared_prefill_plan.has_value()) {
    const absl::Status plan_status = ValidateDPMPreparedPrefillPlan(
        *execution->prepared_prefill_plan);
    if (!plan_status.ok()) return emit_failure(plan_status);
  }
  if (execution->restored_state_witness.has_value()) {
    const absl::Status witness_status =
        ValidateSessionContinuationStateWitness(
            *execution->restored_state_witness);
    if (!witness_status.ok()) return emit_failure(witness_status);
  }
  if (restore_required) {
    if (!execution->restored_checkpoint_id.has_value() ||
        execution->restored_checkpoint_id !=
            request.execution_plan.restore_checkpoint_id ||
        !execution->prepared_prefill_plan.has_value() ||
        execution->prepared_prefill_plan->start_kind !=
            DPMPreparedPrefillStartKind::kOwnPositionRestore ||
        execution->prepared_prefill_plan->restore_checkpoint_id !=
            request.execution_plan.restore_checkpoint_id ||
        execution->prepared_prefill_plan->session_identity !=
            request.execution_plan.session_identity ||
        !execution->restored_state_witness.has_value() ||
        execution->restored_state_witness->session_identity !=
            request.execution_plan.session_identity ||
        execution->prepared_prefill_plan->start_state_witness_id !=
            execution->restored_state_witness->witness_id ||
        execution->restored_state_witness->phase !=
            SessionHandoffPhase::kDecoded ||
        !execution->restored_state_witness->ran_decode ||
        execution->restored_state_witness->current_step <= 0 ||
        execution->restored_state_witness->envelope_size !=
            restore_storage->Size() ||
        execution->restored_state_witness->key_id !=
            transient_restore_options.key_id) {
      return emit_failure(absl::FailedPreconditionError(
          "Worker did not attest the requested checkpoint with its runtime "
          "prepared plan and exact live restore witness."));
    }
    absl::StatusOr<Hash256> restore_hash_or =
        HashCapsuleSource(*restore_storage);
    if (!restore_hash_or.ok()) return emit_failure(restore_hash_or.status());
    if (execution->restored_state_witness->envelope_hash != *restore_hash_or) {
      return emit_failure(absl::DataLossError(
          "Worker live restore witness does not hash the received transient "
          "capsule."));
    }
  } else if (execution->restored_checkpoint_id.has_value() ||
             execution->restored_state_witness.has_value() ||
             (execution->prepared_prefill_plan.has_value() &&
              execution->prepared_prefill_plan->start_kind !=
                  DPMPreparedPrefillStartKind::kFreshSession)) {
    return emit_failure(absl::FailedPreconditionError(
        "Full-prefill worker claimed a restored start state."));
  }
  if (execution->exported_producing_capsule != capture_requested) {
    return emit_failure(absl::FailedPreconditionError(
        "Worker producing-capsule export disagrees with capture policy."));
  }
  if (execution->producing_capsule_continuation_witnesses.has_value() !=
      capture_requested) {
    return emit_failure(absl::FailedPreconditionError(
        "Worker producing-capsule continuation witnesses disagree with "
        "capture policy."));
  }
  if (capture_requested && !execution->prepared_prefill_plan.has_value()) {
    return emit_failure(absl::FailedPreconditionError(
        "A producing-capsule capture lacks its runtime prepared prefill "
        "plan."));
  }
  const absl::Status output_status =
      ValidateFreshWorkerExecutionOutput(execution->output);
  if (!output_status.ok()) return emit_failure(output_status);

  FreshWorkerResult result;
  result.exact_profile_hash = request.exact_profile_hash;
  result.qualification_id = request.qualification_id;
  result.run_index = request.run_index;
  result.run_count = request.run_count;
  result.challenge_nonce = request.challenge_nonce;
  result.request_envelope_hash = request_envelope_hash;
  result.worker_instance_nonce = worker_nonce;
  result.execution_plan_hash = request.execution_plan.plan_hash;
  result.prepared_prefill_plan =
      std::move(execution->prepared_prefill_plan);
  result.restored_checkpoint_id = execution->restored_checkpoint_id;
  result.restored_state_witness =
      std::move(execution->restored_state_witness);
  result.replay_isolation = execution->replay_isolation;
  result.canonical_output = std::move(execution->output.canonical_output);
  result.token_bytes = std::move(execution->output.token_bytes);
  result.logit_frames = std::move(execution->output.logit_frames);

  const ByteSource* producing_capsule = nullptr;
  if (capture_requested) {
    const absl::Status seal_status = producing_storage->Seal();
    if (!seal_status.ok()) return emit_failure(seal_status);
    if (producing_storage->Size() == 0) {
      return emit_failure(absl::DataLossError(
          "Worker claimed producing-capsule export without bytes."));
    }
    absl::StatusOr<DecodedSessionHandoff> decoded_producing =
        DecodeSessionHandoffFrom(
            *producing_storage, execution->producing_session_identity,
            transient_producing_options);
    if (!decoded_producing.ok()) {
      return emit_failure(decoded_producing.status());
    }
    absl::StatusOr<Hash256> producing_hash_or =
        HashCapsuleSource(*producing_storage);
    if (!producing_hash_or.ok()) {
      return emit_failure(producing_hash_or.status());
    }
    const Hash256 producing_hash = *producing_hash_or;
    const FreshWorkerDerivedExecution::
        ProducingCapsuleContinuationWitnesses& witnesses =
            *execution->producing_capsule_continuation_witnesses;
    if (witnesses.producer_first_export.key_id !=
        transient_producing_options.key_id) {
      return emit_failure(absl::DataLossError(
          "Producing-capsule continuation witness uses an unexpected "
          "transient key ID."));
    }
    result.producing_capsule_evidence =
        FreshWorkerProducingCapsuleEvidence{
            .session_identity = execution->producing_session_identity,
            .transient_envelope_size = producing_storage->Size(),
            .transient_envelope_hash = producing_hash,
            .output_evidence_hash = ComputeFreshWorkerOutputEvidenceHash(
                result.canonical_output, result.token_bytes,
                result.logit_frames),
            .producer_first_export = witnesses.producer_first_export,
            .producer_second_export = witnesses.producer_second_export,
            .fresh_import_target = witnesses.fresh_import_target};
    producing_capsule = &*producing_storage;
  }
  const absl::Status result_status =
      ValidateFreshWorkerResultForRequest(result, request);
  if (!result_status.ok()) return emit_failure(result_status);
  absl::StatusOr<std::string> result_envelope =
      EncodeFreshWorkerResult(result, authentication);
  if (!result_envelope.ok()) return emit_failure(result_envelope.status());
  input_fd.Reset();
  ABSL_RETURN_IF_ERROR(WriteIpcFrameBlocking(
      STDOUT_FILENO, *result_envelope, "fresh-worker result"));
  return WriteCapsuleFrameBlocking(
      STDOUT_FILENO, producing_capsule,
      "fresh-worker producing capsule");
#else
  return absl::UnimplementedError(
      "Fresh-worker process isolation is not implemented on this platform.");
#endif
}

absl::Status RunFreshWorkerWithSessionHandoffOnce(
    const FreshWorkerExecutionCallback& execute) {
  return RunFreshWorkerOnce(execute);
}

absl::StatusOr<Hash256> GenerateFreshWorkerNonce() {
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  int flags = O_RDONLY;
#ifdef O_CLOEXEC
  flags |= O_CLOEXEC;
#endif
#ifdef O_NOFOLLOW
  flags |= O_NOFOLLOW;
#endif
  int fd;
  do {
    fd = open("/dev/urandom", flags);
  } while (fd < 0 && errno == EINTR);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, "Unable to open the OS random source for a worker nonce.");
  }
  ScopedFd random(fd);
  struct stat source_stat;
  if (fstat(random.get(), &source_stat) != 0 ||
      !S_ISCHR(source_stat.st_mode)) {
    return absl::FailedPreconditionError(
        "The worker nonce source is not a character device.");
  }
  Hash256 nonce;
  size_t offset = 0;
  while (offset < nonce.bytes.size()) {
    const ssize_t count = read(random.get(), nonce.bytes.data() + offset,
                               nonce.bytes.size() - offset);
    if (count < 0 && errno == EINTR) continue;
    if (count < 0) {
      return absl::ErrnoToStatus(
          errno, "Unable to read the OS random source for a worker nonce.");
    }
    if (count == 0) {
      return absl::DataLossError(
          "The OS random source returned an incomplete worker nonce.");
    }
    offset += static_cast<size_t>(count);
  }
  if (IsZeroHash(nonce)) {
    return absl::DataLossError("The OS random source returned a zero nonce.");
  }
  return nonce;
#else
  return absl::UnimplementedError(
      "Fresh-worker nonce generation is not implemented on this platform.");
#endif
}

}  // namespace litert::lm
