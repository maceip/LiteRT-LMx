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
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
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
#include "runtime/platform/hash/hmac_sha256.h"
#include "runtime/platform/hash/sha256_hasher.h"

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
extern char** environ;
#endif

namespace litert::lm {
namespace {

constexpr std::array<char, 8> kIpcFrameMagic = {'D', 'P', 'M', 'I', 'P', 'C',
                                                 '0', '1'};
constexpr std::array<char, 8> kAuthenticationMagic = {
    'D', 'P', 'M', 'K', 'E', 'Y', '0', '1'};
constexpr uint32_t kAuthenticationPreludeVersion = 1;
constexpr uint32_t kMaximumAuthenticationKeyIdBytes = 256;
constexpr uint32_t kMinimumAuthenticationKeyBytes = 32;
constexpr uint32_t kMaximumAuthenticationKeyBytes = 4096;
constexpr uint32_t kMaximumArguments = 128;
constexpr uint64_t kMaximumArgumentBytes = 64 * 1024;
constexpr uint64_t kMaximumFailureMessageBytes = 4096;
constexpr absl::Duration kMaximumWorkerTimeout = absl::Hours(1);
constexpr absl::Duration kMaximumTerminationGrace = absl::Seconds(5);
constexpr absl::string_view kLaunchSpecDomain =
    "LITERT_LMX_FRESH_WORKER_LAUNCH_SPEC_SHA256_V1";
constexpr absl::string_view kTransportKeyDomain =
    "LITERT_LMX_FRESH_WORKER_PER_REQUEST_TRANSPORT_KEY_HMAC_SHA256_V1";

bool IsZeroHash(const Hash256& hash) {
  uint8_t combined = 0;
  for (uint8_t byte : hash.bytes) combined |= byte;
  return combined == 0;
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

Hash256 HashLaunchSpec(absl::string_view executable_path,
                       const std::vector<std::string>& arguments) {
  Sha256Hasher hasher;
  hasher.Update(kLaunchSpecDomain);
  std::string length;
  AppendU64(executable_path.size(), &length);
  hasher.Update(length);
  hasher.Update(executable_path);
  length.clear();
  AppendU32(static_cast<uint32_t>(arguments.size()), &length);
  hasher.Update(length);
  for (const std::string& argument : arguments) {
    length.clear();
    AppendU64(argument.size(), &length);
    hasher.Update(length);
    hasher.Update(argument);
  }
  return hasher.Finalize();
}

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

std::string EncodeIpcFrame(absl::string_view envelope) {
  std::string frame;
  frame.reserve(kIpcFrameMagic.size() + 8 + envelope.size());
  frame.append(kIpcFrameMagic.data(), kIpcFrameMagic.size());
  AppendU64(envelope.size(), &frame);
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
  AppendU32(kAuthenticationPreludeVersion, &prelude);
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
  binding.reserve(4 + 32 + 32 + 4 + 4 + 32 + 4 +
                  master_authentication.key_id.size() + 8 + 32);
  AppendU32(request.format_version, &binding);
  AppendHash(request.exact_profile_hash, &binding);
  AppendHash(request.qualification_id, &binding);
  AppendU32(request.run_index, &binding);
  AppendU32(request.run_count, &binding);
  AppendHash(request.challenge_nonce, &binding);
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

absl::StatusOr<std::string> ReadIpcFrameBlocking(
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
  ABSL_RETURN_IF_ERROR(ExpectEofBlocking(fd, description));
  return envelope;
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
      ReadU32(header.data() + 8) != kAuthenticationPreludeVersion) {
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
    absl::string_view executable_path,
    const std::vector<std::string>& arguments, int request_read_fd,
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

  std::string executable(executable_path);
  std::vector<std::string> owned_arguments;
  owned_arguments.reserve(arguments.size() + 1);
  owned_arguments.push_back(executable);
  owned_arguments.insert(owned_arguments.end(), arguments.begin(),
                         arguments.end());
  std::vector<char*> argv;
  argv.reserve(owned_arguments.size() + 1);
  for (std::string& argument : owned_arguments) argv.push_back(argument.data());
  argv.push_back(nullptr);

  pid_t pid = -1;
  error = posix_spawn(&pid, executable.c_str(), actions.get(), attributes.get(),
                      argv.data(), environ);
  if (error != 0) {
    return absl::ErrnoToStatus(error, "Unable to spawn fresh worker.");
  }
  return pid;
}

absl::StatusOr<std::string> ReadParentResultFrame(
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
  ABSL_RETURN_IF_ERROR(ExpectEofWithDeadline(fd, control));
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

FreshWorkerProcessRunner::FreshWorkerProcessRunner(
    FreshWorkerProcessOptions options)
    : options_(std::move(options)) {}

absl::StatusOr<FreshWorkerProcessObservation> FreshWorkerProcessRunner::Run(
    const FreshWorkerRequest& request,
    const FreshWorkerAuthentication& authentication) const {
#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)
  ABSL_RETURN_IF_ERROR(ValidateProcessOptions(options_));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerRequest(request));
  ABSL_RETURN_IF_ERROR(ValidateFreshWorkerAuthentication(authentication));
  ABSL_ASSIGN_OR_RETURN(
      const std::string executable_path,
      CanonicalExecutablePath(options_.executable_path));
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
      SpawnWorker(executable_path, options_.arguments, request_pipe.read.get(),
                  result_pipe.write.get(), authentication_pipe.read.get()));
  ChildGuard child(pid, options_.termination_grace);
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
  request_pipe.write.Reset();
  request_frame.clear();

  ABSL_ASSIGN_OR_RETURN(const std::string result_envelope,
                        ReadParentResultFrame(result_pipe.read.get(), control));
  result_pipe.read.Reset();
  ABSL_ASSIGN_OR_RETURN(const int process_status, child.Wait(control));
  if (!WIFEXITED(process_status) || WEXITSTATUS(process_status) != 0) {
    return absl::DataLossError(
        "Fresh worker did not exit successfully after its one response.");
  }
  ABSL_ASSIGN_OR_RETURN(
      FreshWorkerResult result,
      DecodeFreshWorkerResult(result_envelope, transport_authentication));
  if (result.exact_profile_hash != request.exact_profile_hash ||
      result.qualification_id != request.qualification_id ||
      result.run_index != request.run_index ||
      result.run_count != request.run_count ||
      result.challenge_nonce != request.challenge_nonce ||
      result.request_envelope_hash != request_envelope_hash) {
    return absl::UnauthenticatedError(
        "Fresh-worker result is not bound to its exact request.");
  }
  return FreshWorkerProcessObservation{
      .result = std::move(result),
      .process_id = static_cast<int64_t>(pid),
      .request_envelope_hash = request_envelope_hash,
      .result_envelope_hash = HashFreshWorkerEnvelope(result_envelope),
      .launch_spec_hash = HashLaunchSpec(executable_path, options_.arguments)};
#else
  return absl::UnimplementedError(
      "Fresh-worker process isolation is not implemented on this platform.");
#endif
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
  ABSL_ASSIGN_OR_RETURN(
      const std::string request_envelope,
      ReadIpcFrameBlocking(STDIN_FILENO, "fresh-worker request"));
  ABSL_ASSIGN_OR_RETURN(
      const FreshWorkerRequest request,
      DecodeFreshWorkerRequest(request_envelope, authentication));
  const Hash256 request_envelope_hash =
      HashFreshWorkerEnvelope(request_envelope);
  ABSL_ASSIGN_OR_RETURN(const Hash256 worker_nonce,
                        GenerateFreshWorkerNonce());

  FreshWorkerResult result;
  absl::StatusOr<FreshWorkerDerivedExecution> execution = execute(request);
  if (!execution.ok()) {
    result = MakeFailureResult(request, request_envelope_hash, worker_nonce,
                               execution.status());
  } else if (execution->derived_exact_profile_hash !=
             request.exact_profile_hash) {
    result = MakeFailureResult(
        request, request_envelope_hash, worker_nonce,
        absl::FailedPreconditionError(
            "Worker-derived ExactLiteRtProfile does not match the request."));
  } else if (execution->replay_isolation !=
             FreshWorkerReplayIsolation::kEmptyCatalogs) {
    result = MakeFailureResult(
        request, request_envelope_hash, worker_nonce,
        absl::FailedPreconditionError(
            "Exact qualification worker did not attest catalog-free construction."));
  } else {
    const absl::Status output_status =
        ValidateFreshWorkerExecutionOutput(execution->output);
    if (!output_status.ok()) {
      result = MakeFailureResult(request, request_envelope_hash, worker_nonce,
                                 output_status);
    } else {
      result.exact_profile_hash = request.exact_profile_hash;
      result.qualification_id = request.qualification_id;
      result.run_index = request.run_index;
      result.run_count = request.run_count;
      result.challenge_nonce = request.challenge_nonce;
      result.request_envelope_hash = request_envelope_hash;
      result.worker_instance_nonce = worker_nonce;
      result.replay_isolation = execution->replay_isolation;
      result.canonical_output = std::move(execution->output.canonical_output);
      result.token_bytes = std::move(execution->output.token_bytes);
      result.logit_frames = std::move(execution->output.logit_frames);
    }
  }
  ABSL_ASSIGN_OR_RETURN(const std::string result_envelope,
                        EncodeFreshWorkerResult(result, authentication));
  ABSL_RETURN_IF_ERROR(WriteIpcFrameBlocking(
      STDOUT_FILENO, result_envelope, "fresh-worker result"));
  return absl::OkStatus();
#else
  return absl::UnimplementedError(
      "Fresh-worker process isolation is not implemented on this platform.");
#endif
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
