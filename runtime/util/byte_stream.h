// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_BYTE_STREAM_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_BYTE_STREAM_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl

namespace litert::lm {

// Minimal synchronous byte interfaces used by state snapshots and session
// handoff. ByteSource is random-access so callers can authenticate a complete
// object before parsing or mutating live inference state. A ByteSource is an
// immutable snapshot for its complete lifetime: Size and the bytes returned
// for a range must not change between calls. Implementations must either return
// every requested byte or fail; short successful reads are not permitted.
class ByteSink {
 public:
  virtual ~ByteSink() = default;
  virtual absl::Status Append(absl::string_view bytes) = 0;
};

class ByteSource {
 public:
  virtual ~ByteSource() = default;
  virtual uint64_t Size() const = 0;
  virtual absl::Status ReadAt(uint64_t offset,
                              absl::Span<char> destination) const = 0;
};

class StringByteSink final : public ByteSink {
 public:
  explicit StringByteSink(std::string* destination)
      : destination_(destination) {}

  absl::Status Append(absl::string_view bytes) override {
    if (destination_ == nullptr) {
      return absl::FailedPreconditionError(
          "StringByteSink destination is null.");
    }
    if (bytes.size() > destination_->max_size() - destination_->size()) {
      return absl::ResourceExhaustedError(
          "StringByteSink output exceeds std::string capacity.");
    }
    if (!bytes.empty()) {
      destination_->append(bytes.data(), bytes.size());
    }
    return absl::OkStatus();
  }

 private:
  std::string* destination_;
};

class StringByteSource final : public ByteSource {
 public:
  explicit StringByteSource(absl::string_view bytes) : bytes_(bytes) {}

  uint64_t Size() const override { return bytes_.size(); }

  absl::Status ReadAt(uint64_t offset,
                      absl::Span<char> destination) const override {
    if (offset > bytes_.size() ||
        destination.size() > bytes_.size() - static_cast<size_t>(offset)) {
      return absl::OutOfRangeError("StringByteSource read exceeds bounds.");
    }
    if (!destination.empty()) {
      std::memcpy(destination.data(), bytes_.data() + offset,
                  destination.size());
    }
    return absl::OkStatus();
  }

 private:
  absl::string_view bytes_;
};

// A checked view over a subrange of another source. The parent must outlive the
// view. Construction with an invalid range produces an empty invalid view;
// reads then fail closed.
class ByteSourceView final : public ByteSource {
 public:
  ByteSourceView(const ByteSource* parent, uint64_t offset, uint64_t size)
      : parent_(parent), offset_(offset), size_(size) {
    valid_ = parent_ != nullptr && offset_ <= parent_->Size() &&
             size_ <= parent_->Size() - offset_;
  }

  uint64_t Size() const override { return valid_ ? size_ : 0; }

  absl::Status ReadAt(uint64_t offset,
                      absl::Span<char> destination) const override {
    if (!valid_) {
      return absl::FailedPreconditionError(
          "ByteSourceView was constructed with an invalid range.");
    }
    if (offset > size_ || destination.size() > size_ - offset) {
      return absl::OutOfRangeError("ByteSourceView read exceeds bounds.");
    }
    if (offset_ > std::numeric_limits<uint64_t>::max() - offset) {
      return absl::OutOfRangeError("ByteSourceView offset overflows.");
    }
    return parent_->ReadAt(offset_ + offset, destination);
  }

 private:
  const ByteSource* parent_;
  uint64_t offset_;
  uint64_t size_;
  bool valid_ = false;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_UTIL_BYTE_STREAM_H_
