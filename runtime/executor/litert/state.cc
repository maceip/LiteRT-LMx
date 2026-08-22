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

#include "runtime/executor/litert/state.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/executor/common_utils.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/state_interface.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {

namespace {

constexpr int kDynamicDimValue = -1;

// The wire encoding is deliberately independent of host endianness and hash
// map iteration order. Version 1 always uses SHA-256 as the final 32 bytes.
// This state format intentionally has a distinct magic from the older
// LitertKVCache format because LitertState also carries generalized state
// kinds and allocation policies.
constexpr size_t kSnapshotDigestSize = 32;

using SnapshotDigest = Hash256;
static_assert(std::tuple_size<decltype(SnapshotDigest::bytes)>::value ==
                  kSnapshotDigestSize,
              "LRTST001 requires a 32-byte digest");
constexpr uint64_t kMaxSha256InputBytes =
    std::numeric_limits<uint64_t>::max() / 8;

absl::StatusOr<SnapshotDigest> Sha256SourcePrefix(const ByteSource& source,
                                                  uint64_t size) {
  if (size > source.Size()) {
    return absl::OutOfRangeError(
        "LiteRT state digest range exceeds its byte source.");
  }
  if (size > kMaxSha256InputBytes) {
    return absl::ResourceExhaustedError(
        "LiteRT state snapshot exceeds the SHA-256 input limit.");
  }
  constexpr size_t kDigestChunkBytes = 2 * 1024 * 1024;
  std::string chunk;
  chunk.resize(static_cast<size_t>(
      std::min<uint64_t>(size, static_cast<uint64_t>(kDigestChunkBytes))));
  Sha256Hasher hasher;
  uint64_t offset = 0;
  while (offset < size) {
    const size_t count = static_cast<size_t>(std::min<uint64_t>(
        chunk.size(), size - offset));
    RETURN_IF_ERROR(
        source.ReadAt(offset, absl::MakeSpan(chunk).subspan(0, count)));
    hasher.Update(absl::string_view(chunk.data(), count));
    offset += count;
  }
  return hasher.Finalize();
}

class DigestingByteSink {
 public:
  explicit DigestingByteSink(ByteSink* sink) : sink_(sink) {}

  absl::Status Append(absl::string_view bytes) {
    if (sink_ == nullptr) {
      return absl::InvalidArgumentError(
          "LiteRT state snapshot sink must not be null.");
    }
    const uintmax_t byte_count = static_cast<uintmax_t>(bytes.size());
    if (byte_count > kMaxSha256InputBytes - bytes_hashed_) {
      return absl::ResourceExhaustedError(
          "LiteRT state snapshot exceeds the SHA-256 input limit.");
    }
    RETURN_IF_ERROR(sink_->Append(bytes));
    hasher_.Update(bytes);
    bytes_hashed_ += static_cast<uint64_t>(byte_count);
    return absl::OkStatus();
  }

  SnapshotDigest Finalize() { return hasher_.Finalize(); }

 private:
  ByteSink* sink_;
  Sha256Hasher hasher_;
  uint64_t bytes_hashed_ = 0;
};

struct ConstSnapshotBuffer {
  absl::string_view name;
  const TensorBuffer* buffer;
  int32_t state_type;
  int32_t dynamic_dim;
};

struct MutableSnapshotBuffer {
  absl::string_view name;
  TensorBuffer* buffer;
  int32_t state_type;
  int32_t dynamic_dim;
  proto::StateBuffer::Type live_state_type;
  std::optional<int> live_dynamic_dim;
};

struct ParsedSnapshotBuffer {
  MutableSnapshotBuffer target;
  TensorBufferType buffer_type = TensorBufferType::kUnknown;
  uint64_t payload_offset = 0;
  size_t payload_size = 0;
  bool shape_changed = false;
};

struct MetalSnapshotBuffer {
  ParsedSnapshotBuffer parsed;
  std::string rollback_bytes;
  bool has_rollback = false;
};

template <typename NamedBuffer>
void SortSnapshotBuffers(std::vector<NamedBuffer>& buffers) {
  std::sort(buffers.begin(), buffers.end(),
            [](const NamedBuffer& lhs, const NamedBuffer& rhs) {
              return lhs.name < rhs.name;
            });
}

void AppendU8(uint8_t value, std::string& output) {
  output.push_back(static_cast<char>(value));
}

void AppendU32(uint32_t value, std::string& output) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output.push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendI32(int32_t value, std::string& output) {
  AppendU32(static_cast<uint32_t>(value), output);
}

void AppendU64(uint64_t value, std::string& output) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output.push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

class SourceSnapshotReader {
 public:
  SourceSnapshotReader(const ByteSource& source, uint64_t size)
      : source_(source), size_(size) {}

  absl::StatusOr<uint8_t> ReadU8() {
    std::array<char, 1> bytes;
    RETURN_IF_ERROR(ReadBytes(absl::MakeSpan(bytes)));
    return static_cast<uint8_t>(bytes[0]);
  }

  absl::StatusOr<uint32_t> ReadU32() {
    std::array<char, 4> bytes;
    RETURN_IF_ERROR(ReadBytes(absl::MakeSpan(bytes)));
    uint32_t value = 0;
    for (char byte : bytes) {
      value = (value << 8) | static_cast<uint8_t>(byte);
    }
    return value;
  }

  absl::StatusOr<int32_t> ReadI32() {
    ASSIGN_OR_RETURN(uint32_t value, ReadU32());
    if (value <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
      return static_cast<int32_t>(value);
    }
    return -1 - static_cast<int32_t>(
                    std::numeric_limits<uint32_t>::max() - value);
  }

  absl::StatusOr<uint64_t> ReadU64() {
    std::array<char, 8> bytes;
    RETURN_IF_ERROR(ReadBytes(absl::MakeSpan(bytes)));
    uint64_t value = 0;
    for (char byte : bytes) {
      value = (value << 8) | static_cast<uint8_t>(byte);
    }
    return value;
  }

  absl::StatusOr<std::string> ReadString(size_t size) {
    if (size > remaining()) {
      return absl::DataLossError("Truncated LiteRT state snapshot");
    }
    std::string value(size, '\0');
    RETURN_IF_ERROR(ReadBytes(absl::MakeSpan(value)));
    return value;
  }

  absl::Status ReadBytes(absl::Span<char> destination) {
    if (destination.size() > remaining()) {
      return absl::DataLossError("Truncated LiteRT state snapshot");
    }
    RETURN_IF_ERROR(source_.ReadAt(offset_, destination));
    offset_ += destination.size();
    return absl::OkStatus();
  }

  absl::Status Skip(uint64_t size) {
    if (size > remaining()) {
      return absl::DataLossError("Truncated LiteRT state snapshot");
    }
    offset_ += size;
    return absl::OkStatus();
  }

  uint64_t offset() const { return offset_; }
  uint64_t remaining() const { return size_ - offset_; }
  bool empty() const { return offset_ == size_; }

 private:
  const ByteSource& source_;
  uint64_t size_;
  uint64_t offset_ = 0;
};

absl::Status IncompatibleSnapshot(absl::string_view field,
                                  absl::string_view tensor_name = {}) {
  if (tensor_name.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat("LiteRT state snapshot has incompatible ", field));
  }
  return absl::FailedPreconditionError(absl::StrCat(
      "LiteRT state snapshot has incompatible ", field, " for ", tensor_name));
}

template <typename CanonicalBank, typename SecondaryBank>
absl::Status ValidateMatchingStateBankNames(
    const CanonicalBank& canonical_bank, const SecondaryBank& secondary_bank) {
  if (canonical_bank.size() != secondary_bank.size()) {
    return absl::FailedPreconditionError(
        "LiteRT state banks have different tensor counts");
  }
  for (const auto& [name, _] : secondary_bank) {
    if (!canonical_bank.contains(name)) {
      return absl::FailedPreconditionError(
          "LiteRT state banks have different tensor names");
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateBufferByteRange(size_t packed_size, size_t size,
                                     size_t offset,
                                     absl::string_view tensor_name) {
  if (offset > size || packed_size > size - offset) {
    return absl::FailedPreconditionError(absl::StrCat(
        "LiteRT state buffer has an invalid byte range for ", tensor_name));
  }
  return absl::OkStatus();
}

absl::Status NullMappedAddress(TensorBuffer& buffer,
                               absl::string_view operation) {
  auto unlock_status = buffer.Unlock();
  if (!unlock_status) {
    return absl::InternalError(absl::StrCat(
        "LiteRT tensor buffer returned a null mapped address during ",
        operation, "; unlock also failed: ", unlock_status.Error().Message()));
  }
  return absl::InternalError(absl::StrCat(
      "LiteRT tensor buffer returned a null mapped address during ",
      operation));
}

absl::Status ValidateEquivalentBuffer(const TensorBuffer& expected,
                                      const TensorBuffer& actual,
                                      absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(auto expected_buffer_type, expected.BufferType());
  LITERT_ASSIGN_OR_RETURN(auto actual_buffer_type, actual.BufferType());
  if (expected_buffer_type != actual_buffer_type) {
    return IncompatibleSnapshot("staging buffer type", tensor_name);
  }

  LITERT_ASSIGN_OR_RETURN(auto expected_tensor_type, expected.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto actual_tensor_type, actual.TensorType());
  if (expected_tensor_type != actual_tensor_type) {
    return IncompatibleSnapshot("staging tensor type", tensor_name);
  }

  LITERT_ASSIGN_OR_RETURN(size_t expected_packed_size, expected.PackedSize());
  LITERT_ASSIGN_OR_RETURN(size_t actual_packed_size, actual.PackedSize());
  LITERT_ASSIGN_OR_RETURN(size_t expected_size, expected.Size());
  LITERT_ASSIGN_OR_RETURN(size_t actual_size, actual.Size());
  LITERT_ASSIGN_OR_RETURN(size_t expected_offset, expected.Offset());
  LITERT_ASSIGN_OR_RETURN(size_t actual_offset, actual.Offset());
  RETURN_IF_ERROR(ValidateBufferByteRange(expected_packed_size, expected_size,
                                          expected_offset, tensor_name));
  RETURN_IF_ERROR(ValidateBufferByteRange(actual_packed_size, actual_size,
                                          actual_offset, tensor_name));
  if (expected_packed_size != actual_packed_size ||
      expected_size != actual_size || expected_offset != actual_offset) {
    return IncompatibleSnapshot("staging buffer layout", tensor_name);
  }
  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> CreateStagingBuffer(
    Environment& env, const TensorBuffer& target,
    absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, target.TensorType());
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, target.BufferType());
  LITERT_ASSIGN_OR_RETURN(size_t size, target.Size());
  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer staged,
      TensorBuffer::CreateManaged(env, buffer_type, tensor_type, size));
  ABSL_RETURN_IF_ERROR(ValidateEquivalentBuffer(target, staged, tensor_name));
  return staged;
}

absl::StatusOr<TensorBuffer> CreateReshapedStagingBuffer(
    Environment& env, const TensorBuffer& target, int dynamic_dim,
    int num_entries, absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(auto target_tensor_type, target.TensorType());
  const Layout& target_layout = target_tensor_type.Layout();
  if (dynamic_dim < 0 ||
      static_cast<uint32_t>(dynamic_dim) >= target_layout.Rank()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "LiteRT state has an invalid dynamic dimension for ", tensor_name));
  }
  if (target_layout.HasStrides()) {
    return absl::UnimplementedError(absl::StrCat(
        "Cannot reshape a strided LiteRT state buffer for ", tensor_name));
  }

  std::vector<int> dimensions(target_layout.Dimensions().begin(),
                              target_layout.Dimensions().end());
  dimensions[dynamic_dim] = num_entries;
  Layout reshaped_layout(Dimensions(dimensions.begin(), dimensions.end()));
  RankedTensorType reshaped_tensor_type(target_tensor_type.ElementType(),
                                        std::move(reshaped_layout));
  LITERT_ASSIGN_OR_RETURN(size_t reshaped_size,
                          reshaped_tensor_type.Bytes());
  LITERT_ASSIGN_OR_RETURN(auto buffer_type, target.BufferType());
  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer staged,
      TensorBuffer::CreateManaged(env, buffer_type, reshaped_tensor_type,
                                  reshaped_size));

  LITERT_ASSIGN_OR_RETURN(auto staged_buffer_type, staged.BufferType());
  LITERT_ASSIGN_OR_RETURN(auto staged_tensor_type, staged.TensorType());
  LITERT_ASSIGN_OR_RETURN(size_t staged_packed_size, staged.PackedSize());
  LITERT_ASSIGN_OR_RETURN(size_t staged_size, staged.Size());
  LITERT_ASSIGN_OR_RETURN(size_t staged_offset, staged.Offset());
  RETURN_IF_ERROR(ValidateBufferByteRange(
      staged_packed_size, staged_size, staged_offset, tensor_name));
  if (staged_buffer_type != buffer_type ||
      staged_tensor_type != reshaped_tensor_type ||
      staged_packed_size != reshaped_size || staged_size != reshaped_size ||
      staged_offset != 0) {
    return IncompatibleSnapshot("reshaped staging buffer layout", tensor_name);
  }
  return staged;
}

absl::StatusOr<std::string> ReadBufferBytes(TensorBuffer& buffer,
                                            size_t packed_size) {
  std::string bytes;
  if (packed_size > bytes.max_size()) {
    return absl::ResourceExhaustedError(
        "LiteRT rollback buffer exceeds std::string capacity.");
  }
  bytes.resize(packed_size);
  LITERT_ASSIGN_OR_RETURN(void* address,
                          buffer.Lock(TensorBuffer::LockMode::kRead));
  if (packed_size != 0) {
    if (address == nullptr) {
      return NullMappedAddress(buffer, "rollback read");
    }
    std::memcpy(bytes.data(), address, packed_size);
  }
  LITERT_RETURN_IF_ERROR(buffer.Unlock());
  return bytes;
}

absl::Status WriteBufferBytes(TensorBuffer& buffer, absl::string_view bytes,
                              bool* bytes_written) {
  *bytes_written = false;
  LITERT_ASSIGN_OR_RETURN(void* address,
                          buffer.Lock(TensorBuffer::LockMode::kWrite));
  if (!bytes.empty()) {
    if (address == nullptr) {
      return NullMappedAddress(buffer, "rollback write");
    }
    std::memcpy(address, bytes.data(), bytes.size());
  }
  *bytes_written = true;
  LITERT_RETURN_IF_ERROR(buffer.Unlock());
  return absl::OkStatus();
}

absl::Status WriteBufferFromSource(TensorBuffer& buffer,
                                   const ByteSource& source,
                                   uint64_t source_offset, size_t size,
                                   bool* buffer_may_be_modified) {
  *buffer_may_be_modified = false;
  LITERT_ASSIGN_OR_RETURN(void* address,
                          buffer.Lock(TensorBuffer::LockMode::kWrite));
  absl::Status read_status = absl::OkStatus();
  if (size != 0) {
    if (address == nullptr) {
      // A backend that reports a successful write lock with no mapped address
      // is already outside the lock contract. Conservatively include this
      // buffer in Metal rollback because unlock may still have synchronized
      // backend-owned staging memory.
      *buffer_may_be_modified = true;
      return NullMappedAddress(buffer, "snapshot restore");
    }
    // A failing source is allowed to have filled a prefix before reporting its
    // error, so mark the buffer dirty before entering the call.
    *buffer_may_be_modified = true;
    read_status = source.ReadAt(
        source_offset, absl::Span<char>(static_cast<char*>(address), size));
  }
  auto unlock_status = buffer.Unlock();
  if (!read_status.ok()) return read_status;
  LITERT_RETURN_IF_ERROR(unlock_status);
  *buffer_may_be_modified = size != 0;
  return absl::OkStatus();
}

absl::Status RollBackMetalBuffers(std::vector<MetalSnapshotBuffer>& buffers,
                                  size_t count) {
  absl::Status first_failure = absl::OkStatus();
  while (count > 0) {
    --count;
    if (!buffers[count].has_rollback) continue;
    bool bytes_written = false;
    const absl::Status rollback = WriteBufferBytes(
        *buffers[count].parsed.target.buffer, buffers[count].rollback_bytes,
        &bytes_written);
    if (!rollback.ok() && first_failure.ok()) first_failure = rollback;
  }
  return first_failure;
}

absl::StatusOr<std::optional<int>> GetDynamicDimIndex(
    const CompiledModel& compiled_model, absl::string_view signature,
    absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(
      const RankedTensorType ranked_tensor_type,
      compiled_model.GetInputTensorType(signature, tensor_name));
  auto dimensions = ranked_tensor_type.Layout().Dimensions();
  std::optional<int> dynamic_dim_index;
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      RET_CHECK(!dynamic_dim_index.has_value())
          << "Multiple dynamic dimensions are not supported.";
      dynamic_dim_index = i;
    }
  }
  return dynamic_dim_index;
}

absl::Status ResolveDynamicShape(CompiledModel& compiled_model,
                                 absl::string_view signature,
                                 absl::string_view tensor_name, int new_value) {
  LITERT_ASSIGN_OR_RETURN(
      const RankedTensorType ranked_tensor_type,
      compiled_model.GetInputTensorType(signature, tensor_name));
  auto dimensions = ranked_tensor_type.Layout().Dimensions();

  bool has_dynamic_dim = false;
  std::vector<int> new_shape;
  new_shape.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (dimensions[i] == kDynamicDimValue) {
      has_dynamic_dim = true;
      new_shape.push_back(new_value);
    } else {
      new_shape.push_back(dimensions[i]);
    }
  }

  if (has_dynamic_dim) {
    LITERT_RETURN_IF_ERROR(
        compiled_model.ResizeInputTensor(signature, tensor_name, new_shape));
  }

  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> ResizeTensorBuffer(Environment& env,
                                                TensorBuffer& tensor_buffer,
                                                int dynamic_dim_index,
                                                int num_entries_to_insert) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType& tensor_type,
                          tensor_buffer.TensorType());
  RET_CHECK(!tensor_type.Layout().HasStrides());
  auto dimensions = tensor_type.Layout().Dimensions();
  std::vector<int> new_dimensions;
  new_dimensions.reserve(dimensions.size());
  for (int i = 0; i < dimensions.size(); ++i) {
    if (i == dynamic_dim_index) {
      new_dimensions.push_back(dimensions[i] + num_entries_to_insert);
    } else {
      new_dimensions.push_back(dimensions[i]);
    }
  }

  LITERT_ASSIGN_OR_RETURN(TensorBufferType buffer_type,
                          tensor_buffer.BufferType());
  Layout new_layout(Dimensions(new_dimensions.begin(), new_dimensions.end()));
  auto new_out_type =
      RankedTensorType(tensor_type.ElementType(), std::move(new_layout));
  LITERT_ASSIGN_OR_RETURN(size_t new_size, new_out_type.Bytes());

  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer new_tensor_buffer,
      TensorBuffer::CreateManaged(env, buffer_type, new_out_type, new_size));
  LITERT_RETURN_IF_ERROR(new_tensor_buffer.Clear());

  LITERT_ASSIGN_OR_RETURN(auto tensor_buffer_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              tensor_buffer, TensorBuffer::LockMode::kRead));
  auto* tensor_buffer_ptr =
      static_cast<uint8_t*>(tensor_buffer_lock_and_addr.second);
  LITERT_ASSIGN_OR_RETURN(
      auto new_tensor_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(new_tensor_buffer,
                                     TensorBuffer::LockMode::kWrite));
  auto* new_tensor_buffer_ptr =
      static_cast<uint8_t*>(new_tensor_buffer_lock_and_addr.second);
  std::optional<size_t> element_size = GetByteWidth(tensor_type.ElementType());
  RET_CHECK(element_size.has_value());

  ABSL_RETURN_IF_ERROR(ExpandBuffer(tensor_buffer_ptr, dimensions,
                                    new_tensor_buffer_ptr, new_dimensions,
                                    element_size.value()));

  return new_tensor_buffer;
}

absl::Status SelectAndCopyBuffer(TensorBuffer& dst, const TensorBuffer& src,
                                 int batch_index) {
  LITERT_ASSIGN_OR_RETURN(
      auto src_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(src, TensorBuffer::LockMode::kRead));
  const char* src_buffer_ptr =
      static_cast<const char*>(src_buffer_lock_and_addr.second);

  LITERT_ASSIGN_OR_RETURN(
      auto dst_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(dst, TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(size_t dst_buffer_size, dst.PackedSize());
  char* dst_buffer_ptr =
      static_cast<char*>(const_cast<void*>(dst_buffer_lock_and_addr.second));
  // This copy is based on the assumption that the KV cache buffers are in the
  // layout of [batch * X, ...] or [1, batch * X, ...] where X could be 1 or
  // more and X doesn't make values interleaved across batches which is true
  // for the current LLM models of all backends.
  src_buffer_ptr += batch_index * dst_buffer_size;
  memcpy(dst_buffer_ptr, src_buffer_ptr, dst_buffer_size);
  return absl::OkStatus();
}

absl::Status BroadcastAndCopyBuffer(TensorBuffer& dst, int dst_batch_size,
                                    const TensorBuffer& src) {
  LITERT_ASSIGN_OR_RETURN(
      auto src_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(src, TensorBuffer::LockMode::kRead));
  LITERT_ASSIGN_OR_RETURN(size_t src_buffer_size, src.PackedSize());
  const char* src_buffer_ptr =
      static_cast<const char*>(src_buffer_lock_and_addr.second);

  LITERT_ASSIGN_OR_RETURN(
      auto dst_buffer_lock_and_addr,
      TensorBufferScopedLock::Create(dst, TensorBuffer::LockMode::kWrite));
  char* dst_buffer_ptr =
      static_cast<char*>(const_cast<void*>(dst_buffer_lock_and_addr.second));

  for (int i = 0; i < dst_batch_size; ++i) {
    memcpy(dst_buffer_ptr, src_buffer_ptr, src_buffer_size);
    dst_buffer_ptr += src_buffer_size;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::Create(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name,
    const proto::ExecutorMetadata* executor_metadata,
    AllocationPolicy allocation_policy, int batch_size,
    bool clear_kv_cache_before_prefill) {
  if (executor_metadata == nullptr) {
    return HeuristicBasedCreate(env, compiled_model, signature_name,
                                allocation_policy, batch_size,
                                clear_kv_cache_before_prefill);
  }

  return MetadataBasedCreate(env, compiled_model, signature_name,
                             *executor_metadata, allocation_policy, batch_size,
                             clear_kv_cache_before_prefill);
}

absl::Status LitertState::ValidateSessionHandoffSupport() const {
  if (allocation_policy_ == AllocationPolicy::kGpuOptimizedInplace) {
    // GPU-optimized state is created from compiled-model output buffers and is
    // deliberately omitted from most graph inputs. It is restorable only when
    // those authoritative buffers are the actual Metal allocations whose
    // synchronized lock contract SerializeTo/LoadFrom use. A generic GPU label
    // or a host buffer with this allocation-policy enum is not sufficient.
    return ValidateMetalStateStorageForExactProfile();
  }
  return ValidateStateInventory(/*allow_gpu_optimized=*/false);
}

absl::StatusOr<Hash256>
LitertState::GetSessionHandoffStateInventoryHash() const {
  RETURN_IF_ERROR(ValidateSessionHandoffSupport());

  std::string inventory;
  const auto append_frame = [&inventory](absl::string_view value) {
    AppendU64(value.size(), inventory);
    inventory.append(value.data(), value.size());
  };
  append_frame("LITERT_LM_LRTST001_COMPLETE_STATE_INVENTORY_V2");
  append_frame(absl::string_view(kLiteRtStateSnapshotMagic.data(),
                                 kLiteRtStateSnapshotMagic.size()));
  AppendU32(kLiteRtStateSnapshotVersion, inventory);
  AppendU32(static_cast<uint32_t>(allocation_policy_), inventory);
  AppendI32(batch_size_, inventory);
  const bool has_dynamic_capacity =
      std::any_of(bank_1_state_buffers_.begin(),
                  bank_1_state_buffers_.end(), [](const auto& named_buffer) {
                    return named_buffer.second.dynamic_dim.has_value();
                  });
  AppendU8(has_dynamic_capacity ? 1 : 0, inventory);
  if (has_dynamic_capacity) {
    append_frame(
        "DYNAMIC_CAPACITY_NORMALIZED_AXIS_AND_PACKED_BYTES_PER_ENTRY_V2");
  } else {
    append_frame("FIXED_CAPACITY_EXACT_EXTENT_V2");
    AppendI32(num_entries_, inventory);
  }
  AppendU8(authoritative_state_inventory_ ? 1 : 0, inventory);
  // LRTST001 serializes the logical active state into whichever bank is
  // active in the fresh import target. Physical bank identity is not a
  // continuation fact, but both possible physical schemas are.
  append_frame("LOGICAL_ACTIVE_STATE_WITH_FRESH_TARGET_BANK_V1");

  const uint32_t bank_count = bank_2_state_buffers_.has_value() ? 2 : 1;
  AppendU32(bank_count, inventory);
  const auto append_bank =
      [this, &inventory, &append_frame](
          uint32_t bank_index,
          const absl::flat_hash_map<std::string, StateBuffer>& bank)
      -> absl::Status {
    if (bank.size() != bank_1_state_buffers_.size() ||
        bank.size() > std::numeric_limits<uint32_t>::max()) {
      return absl::FailedPreconditionError(
          "LiteRT state inventory banks have different tensor counts.");
    }
    std::vector<std::string> names;
    names.reserve(bank.size());
    for (const auto& [name, _] : bank) names.push_back(name);
    std::sort(names.begin(), names.end());

    AppendU32(bank_index, inventory);
    AppendU32(static_cast<uint32_t>(names.size()), inventory);
    for (const std::string& name : names) {
      const auto physical = bank.find(name);
      const auto canonical = bank_1_state_buffers_.find(name);
      if (physical == bank.end() || canonical == bank_1_state_buffers_.end()) {
        return absl::FailedPreconditionError(
            "LiteRT state inventory banks have different tensor names.");
      }
      if (name.empty() ||
          name.size() > std::numeric_limits<uint32_t>::max()) {
        return absl::FailedPreconditionError(
            "LiteRT state inventory contains an invalid tensor name.");
      }
      const StateBuffer& buffer = physical->second;
      if (buffer.type != canonical->second.type) {
        return absl::FailedPreconditionError(
            "LiteRT state inventory banks disagree about tensor type.");
      }
      LITERT_ASSIGN_OR_RETURN(const TensorBufferType buffer_type,
                              buffer.buffer.BufferType());
      LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                              buffer.buffer.TensorType());
      LITERT_ASSIGN_OR_RETURN(const size_t packed_size,
                              buffer.buffer.PackedSize());
      LITERT_ASSIGN_OR_RETURN(const size_t size, buffer.buffer.Size());
      LITERT_ASSIGN_OR_RETURN(const size_t offset, buffer.buffer.Offset());
      RETURN_IF_ERROR(
          ValidateBufferByteRange(packed_size, size, offset, name));
      const Layout& layout = tensor_type.Layout();
      if (layout.Rank() > std::numeric_limits<uint32_t>::max()) {
        return absl::ResourceExhaustedError(
            "LiteRT state inventory tensor rank exceeds uint32.");
      }
      const int logical_dynamic_dim =
          canonical->second.dynamic_dim.value_or(kDynamicDimValue);
      if (logical_dynamic_dim >= 0 &&
          static_cast<uint32_t>(logical_dynamic_dim) >= layout.Rank()) {
        return absl::FailedPreconditionError(
            "LiteRT state inventory contains an invalid dynamic axis.");
      }
      if (logical_dynamic_dim >= 0 && layout.HasStrides()) {
        return absl::UnimplementedError(
            "Dynamic strided LiteRT state has no capacity-invariant session "
            "handoff inventory contract.");
      }

      AppendU32(static_cast<uint32_t>(name.size()), inventory);
      inventory.append(name);
      AppendI32(static_cast<int32_t>(buffer.type), inventory);
      // LRTST001 uses the primary bank's dynamic dimension as the logical
      // snapshot contract. A ping-pong output bank may intentionally carry no
      // resize marker of its own, so bind both facts instead of conflating
      // them or silently discarding the physical-bank metadata.
      AppendI32(logical_dynamic_dim, inventory);
      AppendI32(buffer.dynamic_dim.value_or(kDynamicDimValue), inventory);
      AppendU32(static_cast<uint32_t>(buffer_type), inventory);
      AppendU32(static_cast<uint32_t>(tensor_type.ElementType()), inventory);
      AppendU32(layout.Rank(), inventory);
      for (uint32_t dimension_index = 0; dimension_index < layout.Rank();
           ++dimension_index) {
        AppendI32(logical_dynamic_dim ==
                          static_cast<int>(dimension_index)
                      ? kDynamicDimValue
                      : layout.Dimensions()[dimension_index],
                  inventory);
      }
      AppendU8(layout.HasStrides() ? 1 : 0, inventory);
      if (layout.HasStrides()) {
        for (uint32_t stride : layout.Strides()) {
          AppendU32(stride, inventory);
        }
      }
      if (logical_dynamic_dim >= 0) {
        LITERT_ASSIGN_OR_RETURN(const size_t logical_packed_size,
                                tensor_type.Bytes());
        if (packed_size != logical_packed_size || num_entries_ <= 0 ||
            packed_size % static_cast<size_t>(num_entries_) != 0 ||
            size != packed_size || offset != 0) {
          return absl::FailedPreconditionError(
              "Dynamic LiteRT state does not have a canonical tight "
              "bytes-per-entry contract.");
        }
        // The capability binds only capacity-invariant structure. The nested
        // LRTST001 snapshot continues to bind the live dimensions, packed
        // size, backing size, offset, payload, and digest for each capsule.
        append_frame(
            "DYNAMIC_TIGHT_BUFFER_CONCRETE_LAYOUT_BOUND_BY_LRTST001_V2");
        AppendU64(packed_size / static_cast<size_t>(num_entries_), inventory);
      } else {
        append_frame("FIXED_BUFFER_CONCRETE_LAYOUT_V2");
        AppendU64(packed_size, inventory);
        AppendU64(size, inventory);
        AppendU64(offset, inventory);
      }
      AppendU8(buffer.buffer.IsMetalMemory() ? 1 : 0, inventory);
    }
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(append_bank(/*bank_index=*/0, bank_1_state_buffers_));
  if (bank_2_state_buffers_.has_value()) {
    RETURN_IF_ERROR(append_bank(/*bank_index=*/1, *bank_2_state_buffers_));
  }

  Sha256Hasher hasher;
  const Hash256 inventory_hash = hasher.OneShot(inventory);
  if (inventory_hash == Hash256{}) {
    return absl::InternalError(
        "LiteRT state inventory produced a zero digest.");
  }
  return inventory_hash;
}

absl::Status LitertState::ValidateDeterministicProjectionResetSupport() const {
  return ValidateStateInventory(/*allow_gpu_optimized=*/true);
}

absl::Status LitertState::ValidateMetalStateStorageForExactProfile() const {
  RETURN_IF_ERROR(ValidateStateInventory(/*allow_gpu_optimized=*/true));
  for (const auto& [name, state_buffer] : bank_1_state_buffers_) {
    if (!state_buffer.buffer.IsMetalMemory()) {
      return absl::UnimplementedError(absl::StrCat(
          "Exact Metal profile requires Metal-backed state buffer ", name,
          "."));
    }
  }
  if (bank_2_state_buffers_.has_value()) {
    for (const auto& [name, state_buffer] : *bank_2_state_buffers_) {
      if (!state_buffer.buffer.IsMetalMemory()) {
        return absl::UnimplementedError(absl::StrCat(
            "Exact Metal profile requires Metal-backed secondary state "
            "buffer ",
            name, "."));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status LitertState::ValidateStateInventory(
    bool allow_gpu_optimized) const {
  if (!authoritative_state_inventory_) {
    return absl::UnimplementedError(
        "Continuation-state operations require an authoritative "
        "executor-metadata state inventory; heuristic discovery is not "
        "sufficient.");
  }
  if (batch_size_ <= 0 || num_entries_ <= 0) {
    return absl::FailedPreconditionError(
        "LiteRT state has an invalid batch or entry capacity.");
  }
  switch (allocation_policy_) {
    case AllocationPolicy::kInplace:
      if (bank_2_state_buffers_.has_value()) {
        return absl::FailedPreconditionError(
            "In-place LiteRT state unexpectedly has a second bank.");
      }
      break;
    case AllocationPolicy::kPingPong:
      if (!bank_2_state_buffers_.has_value()) {
        return absl::FailedPreconditionError(
            "Ping-pong LiteRT state is missing its second bank.");
      }
      break;
    case AllocationPolicy::kGpuOptimizedInplace:
      if (!allow_gpu_optimized) {
        return absl::UnimplementedError(
            "This continuation-state operation has not admitted "
            "GPU-optimized LiteRT state.");
      }
      if (bank_2_state_buffers_.has_value()) {
        return absl::FailedPreconditionError(
            "GPU-optimized in-place LiteRT state unexpectedly has a second "
            "bank.");
      }
      break;
    default:
      return absl::FailedPreconditionError(
          "LiteRT state has an unknown allocation policy.");
  }
  if (bank_1_state_buffers_.empty()) {
    return absl::FailedPreconditionError(
        "LiteRT state has no inventoried continuation buffers.");
  }
  if (bank_2_state_buffers_.has_value()) {
    RETURN_IF_ERROR(ValidateMatchingStateBankNames(
        bank_1_state_buffers_, *bank_2_state_buffers_));
  }

  for (const auto& [name, state_buffer] : bank_1_state_buffers_) {
    if (name.empty()) {
      return absl::FailedPreconditionError(
          "LiteRT state inventory contains an unnamed buffer.");
    }
    switch (state_buffer.type) {
      case proto::StateBuffer::TYPE_GLOBAL_KEY_CACHE:
      case proto::StateBuffer::TYPE_GLOBAL_VALUE_CACHE:
      case proto::StateBuffer::TYPE_LOCAL_KEY_CACHE:
      case proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE:
      case proto::StateBuffer::TYPE_LINEAR_ATTENTION:
        break;
      case proto::StateBuffer::TYPE_UNSPECIFIED:
      default:
        return absl::UnimplementedError(absl::StrCat(
            "Continuation-state operations have not admitted LiteRT state "
            "buffer type ",
            state_buffer.type, " for ", name, "."));
    }

    LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                            state_buffer.buffer.TensorType());
    const Layout& layout = tensor_type.Layout();
    if (state_buffer.dynamic_dim.has_value()) {
      const int dynamic_dim = *state_buffer.dynamic_dim;
      if (dynamic_dim < 0 ||
          static_cast<uint32_t>(dynamic_dim) >= layout.Rank()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "LiteRT state has an invalid dynamic dimension for ", name, "."));
      }
      if (layout.Dimensions()[dynamic_dim] != num_entries_) {
        return absl::FailedPreconditionError(absl::StrCat(
            "LiteRT state capacity does not match the dynamic dimension for ",
            name, "."));
      }
    }
    LITERT_ASSIGN_OR_RETURN(const size_t packed_size,
                            state_buffer.buffer.PackedSize());
    LITERT_ASSIGN_OR_RETURN(const size_t size, state_buffer.buffer.Size());
    LITERT_ASSIGN_OR_RETURN(const size_t offset,
                            state_buffer.buffer.Offset());
    RETURN_IF_ERROR(ValidateBufferByteRange(packed_size, size, offset, name));

    if (bank_2_state_buffers_.has_value()) {
      const auto secondary = bank_2_state_buffers_->find(name);
      if (secondary == bank_2_state_buffers_->end() ||
          secondary->second.type != state_buffer.type) {
        return absl::FailedPreconditionError(absl::StrCat(
            "LiteRT state banks disagree about the type of ", name, "."));
      }
      LITERT_ASSIGN_OR_RETURN(const RankedTensorType secondary_tensor_type,
                              secondary->second.buffer.TensorType());
      if (secondary_tensor_type != tensor_type) {
        return absl::FailedPreconditionError(absl::StrCat(
            "LiteRT state banks disagree about the tensor type of ", name,
            "."));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<uint64_t> LitertState::SerializedSize() const {
  if (batch_size_ < 0 || num_entries_ < 0) {
    return absl::FailedPreconditionError(
        "Cannot size an invalid LiteRT state shape");
  }
  switch (allocation_policy_) {
    case AllocationPolicy::kInplace:
    case AllocationPolicy::kGpuOptimizedInplace:
      if (bank_2_state_buffers_.has_value()) {
        return absl::FailedPreconditionError(
            "In-place LiteRT state unexpectedly has a second bank");
      }
      break;
    case AllocationPolicy::kPingPong:
      if (!bank_2_state_buffers_.has_value()) {
        return absl::FailedPreconditionError(
            "Ping-pong LiteRT state is missing its second bank");
      }
      break;
    default:
      return absl::FailedPreconditionError(
          "LiteRT state has an unknown allocation policy");
  }
  if (bank_2_state_buffers_.has_value()) {
    RETURN_IF_ERROR(ValidateMatchingStateBankNames(
        bank_1_state_buffers_, *bank_2_state_buffers_));
  }

  const auto* active_state_buffers = &bank_1_state_buffers_;
  if (bank_2_state_buffers_.has_value() && !bank_1_is_input_) {
    active_state_buffers = &*bank_2_state_buffers_;
  }
  if (active_state_buffers->size() != bank_1_state_buffers_.size() ||
      active_state_buffers->size() > std::numeric_limits<uint32_t>::max()) {
    return absl::FailedPreconditionError(
        "LiteRT state has an invalid active tensor bank");
  }

  uint64_t total =
      kLiteRtStateSnapshotMagic.size() + 5 * sizeof(uint32_t);
  auto add = [&total](uint64_t bytes) -> absl::Status {
    if (bytes > std::numeric_limits<uint64_t>::max() - total) {
      return absl::ResourceExhaustedError(
          "LiteRT state snapshot byte count overflow.");
    }
    total += bytes;
    return absl::OkStatus();
  };
  for (const auto& [name, active_state_buffer] : *active_state_buffers) {
    const auto canonical = bank_1_state_buffers_.find(name);
    if (canonical == bank_1_state_buffers_.end()) {
      return absl::FailedPreconditionError(
          "LiteRT state banks have different tensor names");
    }
    if (active_state_buffer.type != canonical->second.type) {
      return absl::FailedPreconditionError(
          "LiteRT state banks have different state types");
    }
    const int dynamic_dim =
        canonical->second.dynamic_dim.value_or(kDynamicDimValue);
    if (dynamic_dim < kDynamicDimValue ||
        name.size() > std::numeric_limits<uint32_t>::max()) {
      return absl::FailedPreconditionError(
          "LiteRT state has invalid snapshot metadata");
    }
    LITERT_ASSIGN_OR_RETURN(auto buffer_type,
                            active_state_buffer.buffer.BufferType());
    (void)buffer_type;
    LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                            active_state_buffer.buffer.TensorType());
    const Layout& layout = tensor_type.Layout();
    if (dynamic_dim >= 0 &&
        static_cast<uint32_t>(dynamic_dim) >= layout.Rank()) {
      return absl::FailedPreconditionError(
          "LiteRT state has an out-of-range dynamic dimension");
    }
    LITERT_ASSIGN_OR_RETURN(size_t packed_size,
                            active_state_buffer.buffer.PackedSize());
    LITERT_ASSIGN_OR_RETURN(size_t size,
                            active_state_buffer.buffer.Size());
    LITERT_ASSIGN_OR_RETURN(size_t offset,
                            active_state_buffer.buffer.Offset());
    RETURN_IF_ERROR(
        ValidateBufferByteRange(packed_size, size, offset, name));

    RETURN_IF_ERROR(add(sizeof(uint32_t)));
    RETURN_IF_ERROR(add(name.size()));
    // state type, dynamic dim, buffer type, element type, and rank.
    RETURN_IF_ERROR(add(5 * sizeof(uint32_t)));
    RETURN_IF_ERROR(
        add(static_cast<uint64_t>(layout.Dimensions().size()) *
            sizeof(uint32_t)));
    RETURN_IF_ERROR(add(sizeof(uint8_t)));
    if (layout.HasStrides()) {
      RETURN_IF_ERROR(
          add(static_cast<uint64_t>(layout.Strides().size()) *
              sizeof(uint32_t)));
    }
    RETURN_IF_ERROR(add(3 * sizeof(uint64_t)));
    RETURN_IF_ERROR(add(packed_size));
  }
  if (total > kMaxSha256InputBytes) {
    return absl::ResourceExhaustedError(
        "LiteRT state snapshot exceeds the SHA-256 input limit.");
  }
  RETURN_IF_ERROR(add(kSnapshotDigestSize));
  return total;
}

absl::StatusOr<std::string> LitertState::Serialize() const {
  ASSIGN_OR_RETURN(const uint64_t serialized_size, SerializedSize());
  std::string output;
  if (serialized_size > std::numeric_limits<size_t>::max() ||
      serialized_size > output.max_size()) {
    return absl::ResourceExhaustedError(
        "LiteRT state snapshot exceeds std::string capacity.");
  }
  output.reserve(static_cast<size_t>(serialized_size));
  StringByteSink sink(&output);
  RETURN_IF_ERROR(SerializeTo(&sink));
  if (output.size() != serialized_size) {
    return absl::InternalError(
        "LiteRT state snapshot size changed during serialization.");
  }
  return output;
}

absl::Status LitertState::SerializeTo(ByteSink* sink) const {
  if (sink == nullptr) {
    return absl::InvalidArgumentError(
        "LiteRT state snapshot sink must not be null.");
  }
  if (batch_size_ < 0 || num_entries_ < 0) {
    return absl::FailedPreconditionError(
        "Cannot serialize an invalid LiteRT state shape");
  }

  switch (allocation_policy_) {
    case AllocationPolicy::kInplace:
    case AllocationPolicy::kGpuOptimizedInplace:
      if (bank_2_state_buffers_.has_value()) {
        return absl::FailedPreconditionError(
            "In-place LiteRT state unexpectedly has a second bank");
      }
      break;
    case AllocationPolicy::kPingPong:
      if (!bank_2_state_buffers_.has_value()) {
        return absl::FailedPreconditionError(
            "Ping-pong LiteRT state is missing its second bank");
      }
      break;
    default:
      return absl::FailedPreconditionError(
          "LiteRT state has an unknown allocation policy");
  }
  if (bank_2_state_buffers_.has_value()) {
    RETURN_IF_ERROR(ValidateMatchingStateBankNames(
        bank_1_state_buffers_, *bank_2_state_buffers_));
  }

  const auto* active_state_buffers = &bank_1_state_buffers_;
  if (bank_2_state_buffers_.has_value() && !bank_1_is_input_) {
    active_state_buffers = &*bank_2_state_buffers_;
  }
  if (active_state_buffers->size() != bank_1_state_buffers_.size()) {
    return absl::FailedPreconditionError(
        "LiteRT state banks have different tensor counts");
  }

  std::vector<ConstSnapshotBuffer> buffers;
  buffers.reserve(active_state_buffers->size());
  for (const auto& [name, active_state_buffer] : *active_state_buffers) {
    auto canonical = bank_1_state_buffers_.find(name);
    if (canonical == bank_1_state_buffers_.end()) {
      return absl::FailedPreconditionError(
          "LiteRT state banks have different tensor names");
    }
    if (active_state_buffer.type != canonical->second.type) {
      return absl::FailedPreconditionError(
          "LiteRT state banks have different state types");
    }
    const int dynamic_dim =
        canonical->second.dynamic_dim.value_or(kDynamicDimValue);
    if (dynamic_dim < kDynamicDimValue) {
      return absl::FailedPreconditionError(
          "LiteRT state has an invalid dynamic dimension");
    }
    buffers.push_back(ConstSnapshotBuffer{
        .name = name,
        .buffer = &active_state_buffer.buffer,
        .state_type = static_cast<int32_t>(canonical->second.type),
        .dynamic_dim = dynamic_dim,
    });
  }
  SortSnapshotBuffers(buffers);
  if (buffers.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError(
        "Too many tensors in LiteRT state snapshot");
  }

  DigestingByteSink writer(sink);
  std::string header(kLiteRtStateSnapshotMagic.data(),
                     kLiteRtStateSnapshotMagic.size());
  AppendU32(kLiteRtStateSnapshotVersion, header);
  AppendU32(static_cast<uint32_t>(allocation_policy_), header);
  AppendU32(static_cast<uint32_t>(batch_size_), header);
  AppendU32(static_cast<uint32_t>(num_entries_), header);
  AppendU32(static_cast<uint32_t>(buffers.size()), header);
  RETURN_IF_ERROR(writer.Append(header));

  for (const ConstSnapshotBuffer& named_buffer : buffers) {
    if (named_buffer.name.size() > std::numeric_limits<uint32_t>::max()) {
      return absl::ResourceExhaustedError(
          "Tensor name is too large for LiteRT state snapshot");
    }

    LITERT_ASSIGN_OR_RETURN(auto buffer_type,
                            named_buffer.buffer->BufferType());
    LITERT_ASSIGN_OR_RETURN(auto tensor_type,
                            named_buffer.buffer->TensorType());
    const Layout& layout = tensor_type.Layout();
    if (named_buffer.dynamic_dim >= 0 &&
        static_cast<uint32_t>(named_buffer.dynamic_dim) >= layout.Rank()) {
      return absl::FailedPreconditionError(
          "LiteRT state has an out-of-range dynamic dimension");
    }
    LITERT_ASSIGN_OR_RETURN(size_t packed_size,
                            named_buffer.buffer->PackedSize());
    LITERT_ASSIGN_OR_RETURN(size_t size, named_buffer.buffer->Size());
    LITERT_ASSIGN_OR_RETURN(size_t offset, named_buffer.buffer->Offset());
    RETURN_IF_ERROR(ValidateBufferByteRange(
        packed_size, size, offset, named_buffer.name));

    std::string metadata;
    AppendU32(static_cast<uint32_t>(named_buffer.name.size()), metadata);
    metadata.append(named_buffer.name.data(), named_buffer.name.size());
    AppendI32(named_buffer.state_type, metadata);
    AppendI32(named_buffer.dynamic_dim, metadata);
    AppendU32(static_cast<uint32_t>(buffer_type), metadata);
    AppendU32(static_cast<uint32_t>(tensor_type.ElementType()), metadata);
    AppendU32(layout.Rank(), metadata);
    for (int32_t dimension : layout.Dimensions()) {
      AppendI32(dimension, metadata);
    }
    AppendU8(layout.HasStrides() ? 1 : 0, metadata);
    if (layout.HasStrides()) {
      for (uint32_t stride : layout.Strides()) {
        AppendU32(stride, metadata);
      }
    }
    AppendU64(static_cast<uint64_t>(packed_size), metadata);
    AppendU64(static_cast<uint64_t>(size), metadata);
    AppendU64(static_cast<uint64_t>(offset), metadata);
    RETURN_IF_ERROR(writer.Append(metadata));

    // Duplicate provides a non-const handle to the same allocation so that an
    // unlock failure can be observed and returned to the caller.
    LITERT_ASSIGN_OR_RETURN(TensorBuffer readable,
                            named_buffer.buffer->Duplicate());
    LITERT_ASSIGN_OR_RETURN(void* address,
                            readable.Lock(TensorBuffer::LockMode::kRead));
    absl::Status append_status = absl::OkStatus();
    if (packed_size != 0) {
      if (address == nullptr) {
        return NullMappedAddress(readable, "snapshot serialization");
      }
      append_status = writer.Append(absl::string_view(
          static_cast<const char*>(address), packed_size));
    }
    auto unlock_status = readable.Unlock();
    if (!append_status.ok()) return append_status;
    LITERT_RETURN_IF_ERROR(unlock_status);
  }

  const SnapshotDigest digest = writer.Finalize();
  return sink->Append(absl::string_view(
      reinterpret_cast<const char*>(digest.bytes.data()),
      digest.bytes.size()));
}

absl::Status LitertState::Load(absl::string_view serialized_state) {
  StringByteSource source(serialized_state);
  return LoadFrom(source, /*target_is_disposable=*/false);
}

absl::Status LitertState::ValidateSnapshotHeaderForImport(
    const ByteSource& source, int required_consumed_entries) const {
  if (required_consumed_entries < 0) {
    return absl::InvalidArgumentError(
        "Required LiteRT state entry count must not be negative.");
  }
  constexpr uint64_t kSnapshotHeaderSize =
      kLiteRtStateSnapshotMagic.size() + 5 * sizeof(uint32_t);
  if (source.Size() < kSnapshotHeaderSize + kSnapshotDigestSize) {
    return absl::DataLossError("Truncated LiteRT state snapshot");
  }

  SourceSnapshotReader reader(source, source.Size() - kSnapshotDigestSize);
  std::array<char, kLiteRtStateSnapshotMagic.size()> magic;
  RETURN_IF_ERROR(reader.ReadBytes(absl::MakeSpan(magic)));
  if (!std::equal(magic.begin(), magic.end(),
                  kLiteRtStateSnapshotMagic.begin())) {
    return absl::DataLossError("Invalid LiteRT state snapshot magic");
  }
  ABSL_ASSIGN_OR_RETURN(const uint32_t version, reader.ReadU32());
  if (version != kLiteRtStateSnapshotVersion) {
    return IncompatibleSnapshot("format version");
  }
  ABSL_ASSIGN_OR_RETURN(const uint32_t allocation_policy, reader.ReadU32());
  if (allocation_policy >
      static_cast<uint32_t>(AllocationPolicy::kGpuOptimizedInplace)) {
    return absl::DataLossError(
        "Invalid allocation policy in LiteRT state snapshot");
  }
  if (allocation_policy != static_cast<uint32_t>(allocation_policy_)) {
    return IncompatibleSnapshot("allocation policy");
  }
  ABSL_ASSIGN_OR_RETURN(const uint32_t batch_size, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t num_entries, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(const uint32_t tensor_count, reader.ReadU32());
  if (batch_size_ < 0 || batch_size != static_cast<uint32_t>(batch_size_)) {
    return IncompatibleSnapshot("batch size");
  }
  if (num_entries >
      static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return absl::DataLossError(
        "Invalid entry count in LiteRT state snapshot");
  }
  if (tensor_count != bank_1_state_buffers_.size()) {
    return IncompatibleSnapshot("tensor count");
  }
  const int snapshot_num_entries = static_cast<int>(num_entries);
  if (snapshot_num_entries < required_consumed_entries) {
    return absl::FailedPreconditionError(absl::StrCat(
        "LiteRT state snapshot capacity ", snapshot_num_entries,
        " is smaller than the ", required_consumed_entries,
        " consumed token positions in the session handoff."));
  }
  if (snapshot_num_entries != num_entries_) {
    if (snapshot_num_entries == 0) {
      return IncompatibleSnapshot("entry count");
    }
    if (bank_2_state_buffers_.has_value()) {
      return absl::UnimplementedError(
          "Cannot reshape a ping-pong LiteRT state during snapshot restore");
    }
    const bool has_dynamic_buffer = std::any_of(
        bank_1_state_buffers_.begin(), bank_1_state_buffers_.end(),
        [](const auto& named_buffer) {
          return named_buffer.second.dynamic_dim.has_value();
        });
    if (!has_dynamic_buffer) {
      return IncompatibleSnapshot("entry count");
    }
  }
  return absl::OkStatus();
}

absl::Status LitertState::LoadFrom(const ByteSource& source,
                                   bool target_is_disposable) {
  const uint64_t source_size = source.Size();
  if (source_size <= kSnapshotDigestSize) {
    return absl::DataLossError("Truncated LiteRT state snapshot");
  }
  if (batch_size_ < 0 || num_entries_ < 0) {
    return absl::FailedPreconditionError(
        "Cannot load into an invalid LiteRT state shape");
  }

  switch (allocation_policy_) {
    case AllocationPolicy::kInplace:
    case AllocationPolicy::kGpuOptimizedInplace:
      if (bank_2_state_buffers_.has_value()) {
        return absl::FailedPreconditionError(
            "In-place LiteRT state unexpectedly has a second bank");
      }
      break;
    case AllocationPolicy::kPingPong:
      if (!bank_2_state_buffers_.has_value()) {
        return absl::FailedPreconditionError(
            "Ping-pong LiteRT state is missing its second bank");
      }
      break;
    default:
      return absl::FailedPreconditionError(
          "LiteRT state has an unknown allocation policy");
  }
  if (bank_2_state_buffers_.has_value()) {
    RETURN_IF_ERROR(ValidateMatchingStateBankNames(
        bank_1_state_buffers_, *bank_2_state_buffers_));
  }

  const uint64_t encoded_size = source_size - kSnapshotDigestSize;
  ASSIGN_OR_RETURN(const SnapshotDigest expected_digest,
                   Sha256SourcePrefix(source, encoded_size));
  std::array<char, kSnapshotDigestSize> encoded_digest;
  RETURN_IF_ERROR(source.ReadAt(encoded_size, absl::MakeSpan(encoded_digest)));
  uint8_t digest_difference = 0;
  for (size_t i = 0; i < expected_digest.bytes.size(); ++i) {
    digest_difference |=
        expected_digest.bytes[i] ^ static_cast<uint8_t>(encoded_digest[i]);
  }
  if (digest_difference != 0) {
    return absl::DataLossError("LiteRT state snapshot digest mismatch");
  }

  auto* active_state_buffers = &bank_1_state_buffers_;
  if (bank_2_state_buffers_.has_value() && !bank_1_is_input_) {
    active_state_buffers = &*bank_2_state_buffers_;
  }
  if (active_state_buffers->size() != bank_1_state_buffers_.size()) {
    return absl::FailedPreconditionError(
        "LiteRT state banks have different tensor counts");
  }

  std::vector<MutableSnapshotBuffer> expected_buffers;
  expected_buffers.reserve(active_state_buffers->size());
  for (auto& [name, active_state_buffer] : *active_state_buffers) {
    auto canonical = bank_1_state_buffers_.find(name);
    if (canonical == bank_1_state_buffers_.end()) {
      return absl::FailedPreconditionError(
          "LiteRT state banks have different tensor names");
    }
    if (active_state_buffer.type != canonical->second.type) {
      return absl::FailedPreconditionError(
          "LiteRT state banks have different state types");
    }
    expected_buffers.push_back(MutableSnapshotBuffer{
        .name = name,
        .buffer = &active_state_buffer.buffer,
        .state_type = static_cast<int32_t>(canonical->second.type),
        .dynamic_dim =
            canonical->second.dynamic_dim.value_or(kDynamicDimValue),
        .live_state_type = active_state_buffer.type,
        .live_dynamic_dim = active_state_buffer.dynamic_dim,
    });
  }
  SortSnapshotBuffers(expected_buffers);

  SourceSnapshotReader reader(source, encoded_size);
  std::array<char, kLiteRtStateSnapshotMagic.size()> magic;
  RETURN_IF_ERROR(reader.ReadBytes(absl::MakeSpan(magic)));
  if (!std::equal(magic.begin(), magic.end(),
                  kLiteRtStateSnapshotMagic.begin())) {
    return absl::DataLossError("Invalid LiteRT state snapshot magic");
  }
  ABSL_ASSIGN_OR_RETURN(uint32_t version, reader.ReadU32());
  if (version != kLiteRtStateSnapshotVersion) {
    return IncompatibleSnapshot("format version");
  }
  ABSL_ASSIGN_OR_RETURN(uint32_t allocation_policy, reader.ReadU32());
  if (allocation_policy >
      static_cast<uint32_t>(AllocationPolicy::kGpuOptimizedInplace)) {
    return absl::DataLossError(
        "Invalid allocation policy in LiteRT state snapshot");
  }
  if (allocation_policy != static_cast<uint32_t>(allocation_policy_)) {
    return IncompatibleSnapshot("allocation policy");
  }

  ABSL_ASSIGN_OR_RETURN(uint32_t batch_size, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(uint32_t num_entries, reader.ReadU32());
  ABSL_ASSIGN_OR_RETURN(uint32_t tensor_count, reader.ReadU32());
  if (batch_size != static_cast<uint32_t>(batch_size_)) {
    return IncompatibleSnapshot("batch size");
  }
  if (num_entries >
      static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return absl::DataLossError(
        "Invalid entry count in LiteRT state snapshot");
  }
  if (tensor_count != expected_buffers.size()) {
    return IncompatibleSnapshot("tensor count");
  }
  const int snapshot_num_entries = static_cast<int>(num_entries);
  const bool entry_count_changed = snapshot_num_entries != num_entries_;
  if (entry_count_changed) {
    if (snapshot_num_entries == 0) {
      return IncompatibleSnapshot("entry count");
    }
    if (bank_2_state_buffers_.has_value()) {
      return absl::UnimplementedError(
          "Cannot reshape a ping-pong LiteRT state during snapshot restore");
    }
    const bool has_dynamic_buffer =
        std::any_of(expected_buffers.begin(), expected_buffers.end(),
                    [](const MutableSnapshotBuffer& buffer) {
                      return buffer.dynamic_dim >= 0;
                    });
    if (!has_dynamic_buffer) {
      return IncompatibleSnapshot("entry count");
    }
  }

  std::vector<ParsedSnapshotBuffer> parsed_buffers;
  parsed_buffers.reserve(expected_buffers.size());
  for (const MutableSnapshotBuffer& expected : expected_buffers) {
    ABSL_ASSIGN_OR_RETURN(uint32_t name_size, reader.ReadU32());
    if (name_size > 1024 * 1024) {
      return absl::ResourceExhaustedError(
          "Tensor name exceeds LiteRT state snapshot limit");
    }
    ASSIGN_OR_RETURN(std::string name, reader.ReadString(name_size));
    if (name != expected.name) {
      return IncompatibleSnapshot("canonical tensor order or name");
    }

    ABSL_ASSIGN_OR_RETURN(int32_t state_type, reader.ReadI32());
    ABSL_ASSIGN_OR_RETURN(int32_t dynamic_dim, reader.ReadI32());
    if (state_type != expected.state_type) {
      return IncompatibleSnapshot("state type", name);
    }
    if (dynamic_dim != expected.dynamic_dim) {
      return IncompatibleSnapshot("dynamic dimension", name);
    }

    LITERT_ASSIGN_OR_RETURN(auto target_buffer_type,
                            expected.buffer->BufferType());
    LITERT_ASSIGN_OR_RETURN(auto target_tensor_type,
                            expected.buffer->TensorType());
    const Layout& target_layout = target_tensor_type.Layout();
    if (expected.dynamic_dim >= 0 &&
        static_cast<uint32_t>(expected.dynamic_dim) >= target_layout.Rank()) {
      return absl::FailedPreconditionError(
          "LiteRT state has an out-of-range dynamic dimension");
    }
    LITERT_ASSIGN_OR_RETURN(size_t target_packed_size,
                            expected.buffer->PackedSize());
    LITERT_ASSIGN_OR_RETURN(size_t target_size, expected.buffer->Size());
    LITERT_ASSIGN_OR_RETURN(size_t target_offset, expected.buffer->Offset());
    RETURN_IF_ERROR(ValidateBufferByteRange(
        target_packed_size, target_size, target_offset, name));

    ABSL_ASSIGN_OR_RETURN(uint32_t buffer_type, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(uint32_t element_type, reader.ReadU32());
    ABSL_ASSIGN_OR_RETURN(uint32_t rank, reader.ReadU32());
    if (buffer_type != static_cast<uint32_t>(target_buffer_type)) {
      return IncompatibleSnapshot("buffer type", name);
    }
    if (element_type !=
        static_cast<uint32_t>(target_tensor_type.ElementType())) {
      return IncompatibleSnapshot("element type", name);
    }
    if (rank != target_layout.Rank()) {
      return IncompatibleSnapshot("rank", name);
    }
    std::vector<int32_t> snapshot_dimensions;
    snapshot_dimensions.reserve(rank);
    bool shape_changed = false;
    for (uint32_t dimension_index = 0; dimension_index < rank;
         ++dimension_index) {
      ABSL_ASSIGN_OR_RETURN(int32_t dimension, reader.ReadI32());
      const int32_t target_dimension =
          target_layout.Dimensions()[dimension_index];
      if (dimension < 0) {
        return absl::DataLossError(absl::StrCat(
            "Invalid dimension in LiteRT state snapshot for ", name));
      }
      if (expected.dynamic_dim == static_cast<int32_t>(dimension_index)) {
        if (target_dimension != num_entries_) {
          return absl::FailedPreconditionError(absl::StrCat(
              "LiteRT state has an inconsistent live dynamic dimension for ",
              name));
        }
        if (dimension != snapshot_num_entries) {
          return IncompatibleSnapshot("dynamic dimension size", name);
        }
      } else if (dimension != target_dimension) {
        return IncompatibleSnapshot("dimensions", name);
      }
      shape_changed |= dimension != target_dimension;
      snapshot_dimensions.push_back(dimension);
    }
    ABSL_ASSIGN_OR_RETURN(uint8_t has_strides, reader.ReadU8());
    if (has_strides > 1) {
      return absl::DataLossError(
          "Invalid stride flag in LiteRT state snapshot");
    }
    if ((has_strides != 0) != target_layout.HasStrides()) {
      return IncompatibleSnapshot("stride layout", name);
    }
    if (target_layout.HasStrides()) {
      for (uint32_t target_stride : target_layout.Strides()) {
        ABSL_ASSIGN_OR_RETURN(uint32_t stride, reader.ReadU32());
        if (stride != target_stride) {
          return IncompatibleSnapshot("strides", name);
        }
      }
    }

    size_t expected_packed_size = target_packed_size;
    size_t expected_size = target_size;
    size_t expected_offset = target_offset;
    if (shape_changed) {
      if (expected.dynamic_dim < 0 || target_layout.HasStrides()) {
        return IncompatibleSnapshot("dynamic dimensions", name);
      }
      Layout snapshot_layout(
          Dimensions(snapshot_dimensions.begin(), snapshot_dimensions.end()));
      RankedTensorType snapshot_tensor_type(target_tensor_type.ElementType(),
                                            std::move(snapshot_layout));
      LITERT_ASSIGN_OR_RETURN(expected_packed_size,
                              snapshot_tensor_type.Bytes());
      // LitertState::Resize produces a managed, tightly packed allocation.
      // Requiring that canonical layout prevents a snapshot from using a
      // shape change to smuggle arbitrary backing-store offsets or padding.
      expected_size = expected_packed_size;
      expected_offset = 0;
    }

    ABSL_ASSIGN_OR_RETURN(uint64_t packed_size, reader.ReadU64());
    ABSL_ASSIGN_OR_RETURN(uint64_t size, reader.ReadU64());
    ABSL_ASSIGN_OR_RETURN(uint64_t offset, reader.ReadU64());
    if (packed_size != static_cast<uint64_t>(expected_packed_size)) {
      return IncompatibleSnapshot("packed byte size", name);
    }
    if (size != static_cast<uint64_t>(expected_size)) {
      return IncompatibleSnapshot("underlying byte size", name);
    }
    if (offset != static_cast<uint64_t>(expected_offset)) {
      return IncompatibleSnapshot("buffer offset", name);
    }
    const uint64_t payload_offset = reader.offset();
    RETURN_IF_ERROR(reader.Skip(expected_packed_size));
    parsed_buffers.push_back(ParsedSnapshotBuffer{
        .target = expected,
        .buffer_type = target_buffer_type,
        .payload_offset = payload_offset,
        .payload_size = expected_packed_size,
        .shape_changed = shape_changed,
    });
  }
  if (!reader.empty()) {
    return absl::DataLossError("Trailing data in LiteRT state snapshot");
  }

  // Stage and populate every non-Metal tensor before replacing the live map.
  // LiteRT currently avoids managed Metal allocations because they leak on the
  // supported Apple GPU path (b/505373949). Metal tensors therefore retain
  // their existing allocation: first retain an ordinary duplicate handle and
  // a host rollback copy, then update them only after every other allocation,
  // lock, copy, and structural check has succeeded. Any failed Metal write is
  // rolled back before this method returns.
  absl::flat_hash_map<std::string, StateBuffer> staged_state_buffers;
  staged_state_buffers.reserve(active_state_buffers->size());
  std::vector<MetalSnapshotBuffer> metal_buffers;
  for (const ParsedSnapshotBuffer& parsed : parsed_buffers) {
    if (::litert::IsMetalMemory(parsed.buffer_type)) {
      if (parsed.shape_changed) {
        return absl::UnimplementedError(
            "Cannot reshape a Metal LiteRT state during snapshot restore");
      }
      LITERT_ASSIGN_OR_RETURN(TensorBuffer retained,
                              parsed.target.buffer->Duplicate());
      ABSL_RETURN_IF_ERROR(ValidateEquivalentBuffer(
          *parsed.target.buffer, retained, parsed.target.name));
      std::string rollback_bytes;
      if (!target_is_disposable) {
        ABSL_ASSIGN_OR_RETURN(
            rollback_bytes,
            ReadBufferBytes(*parsed.target.buffer, parsed.payload_size));
      }
      staged_state_buffers.emplace(
          std::string(parsed.target.name),
          StateBuffer{
              .buffer = std::move(retained),
              .type = parsed.target.live_state_type,
              .dynamic_dim = parsed.target.live_dynamic_dim,
          });
      metal_buffers.push_back(MetalSnapshotBuffer{
          .parsed = parsed,
          .rollback_bytes = std::move(rollback_bytes),
          .has_rollback = !target_is_disposable,
      });
      continue;
    }
    absl::StatusOr<TensorBuffer> staged_or =
        parsed.shape_changed
            ? CreateReshapedStagingBuffer(
                  env_, *parsed.target.buffer, parsed.target.dynamic_dim,
                  snapshot_num_entries, parsed.target.name)
            : CreateStagingBuffer(env_, *parsed.target.buffer,
                                  parsed.target.name);
    if (!staged_or.ok()) return staged_or.status();
    TensorBuffer staged = std::move(*staged_or);
    LITERT_ASSIGN_OR_RETURN(void* address,
                            staged.Lock(TensorBuffer::LockMode::kWrite));
    absl::Status read_status = absl::OkStatus();
    if (parsed.payload_size != 0) {
      if (address == nullptr) {
        return NullMappedAddress(staged, "staged snapshot restore");
      }
      read_status = source.ReadAt(
          parsed.payload_offset,
          absl::Span<char>(static_cast<char*>(address), parsed.payload_size));
    }
    auto unlock_status = staged.Unlock();
    if (!read_status.ok()) return read_status;
    LITERT_RETURN_IF_ERROR(unlock_status);
    staged_state_buffers.emplace(
        std::string(parsed.target.name),
        StateBuffer{
            .buffer = std::move(staged),
            .type = parsed.target.live_state_type,
            .dynamic_dim = parsed.target.live_dynamic_dim,
        });
  }

  for (size_t i = 0; i < metal_buffers.size(); ++i) {
    bool buffer_may_be_modified = false;
    const absl::Status write = WriteBufferFromSource(
        *metal_buffers[i].parsed.target.buffer, source,
        metal_buffers[i].parsed.payload_offset,
        metal_buffers[i].parsed.payload_size, &buffer_may_be_modified);
    if (!write.ok()) {
      if (!target_is_disposable) {
        const size_t rollback_count =
            i + (buffer_may_be_modified ? 1 : 0);
        const absl::Status rollback =
            RollBackMetalBuffers(metal_buffers, rollback_count);
        if (!rollback.ok()) {
          return absl::DataLossError(absl::StrCat(
              "LiteRT Metal state restore failed and rollback could not be "
              "verified: restore=",
              write.message(), "; rollback=", rollback.message()));
        }
      }
      return write;
    }
  }

  active_state_buffers->swap(staged_state_buffers);
  num_entries_ = snapshot_num_entries;
  return absl::OkStatus();
}

absl::Status LitertState::SelectAndCopyFrom(StateInterface& other,
                                            int batch_index) {
  auto other_litert = dynamic_cast<LitertState*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertState.";
  RET_CHECK(!bank_2_state_buffers_.has_value());
  RET_CHECK(!other_litert->bank_2_state_buffers_.has_value());
  RET_CHECK_GT(other_litert->batch_size_, batch_size_);
  RET_CHECK_LT(batch_index, other_litert->batch_size_);
  RET_CHECK_EQ(num_entries_, other_litert->num_entries_);

  for (auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    auto it = other_litert->bank_1_state_buffers_.find(input_name);
    RET_CHECK(it != other_litert->bank_1_state_buffers_.end());
    ABSL_RETURN_IF_ERROR(SelectAndCopyBuffer(state_buffer.buffer,
                                             it->second.buffer, batch_index));
  }
  return absl::OkStatus();
}

absl::Status LitertState::BroadcastAndCopyFrom(StateInterface& other) {
  auto other_litert = dynamic_cast<LitertState*>(&other);
  RET_CHECK(other_litert != nullptr) << "Only support LitertState.";
  RET_CHECK(!bank_2_state_buffers_.has_value());
  RET_CHECK(!other_litert->bank_2_state_buffers_.has_value());
  RET_CHECK_EQ(other_litert->batch_size_, 1);
  RET_CHECK_GT(batch_size_, other_litert->batch_size_);
  RET_CHECK_EQ(num_entries_, other_litert->num_entries_);

  for (auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    auto it = other_litert->bank_1_state_buffers_.find(input_name);
    RET_CHECK(it != other_litert->bank_1_state_buffers_.end());
    ABSL_RETURN_IF_ERROR(BroadcastAndCopyBuffer(
        state_buffer.buffer, batch_size_, it->second.buffer));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<StateInterface>> LitertState::DeepCopy() const {
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers;
  for (const auto& [name, state_buffer] : bank_1_state_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto buf_copy,
                            CopyTensorBuffer(env_, state_buffer.buffer));
    bank_1_state_buffers[name] = StateBuffer{
        .buffer = std::move(buf_copy),
        .type = state_buffer.type,
        .dynamic_dim = state_buffer.dynamic_dim,
    };
  }

  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers;
  if (bank_2_state_buffers_.has_value()) {
    bank_2_state_buffers.emplace();
    auto& map = *bank_2_state_buffers;
    for (const auto& [name, state_buffer] : *bank_2_state_buffers_) {
      LITERT_ASSIGN_OR_RETURN(auto buf_copy,
                              CopyTensorBuffer(env_, state_buffer.buffer));
      map[name] = StateBuffer{
          .buffer = std::move(buf_copy),
          .type = state_buffer.type,
          .dynamic_dim = state_buffer.dynamic_dim,
      };
    }
  }

  auto copy = absl::WrapUnique(new LitertState(
      batch_size_, num_entries_, env_, std::move(bank_1_state_buffers),
      std::move(bank_2_state_buffers), allocation_policy_,
      authoritative_state_inventory_));
  copy->bank_1_is_input_ = bank_1_is_input_;

  return copy;
}

bool LitertState::Contains(absl::string_view tensor_name) const {
  return bank_1_state_buffers_.contains(tensor_name);
}

absl::Status LitertState::Clear() {
  for (auto& [_, state_buffer] : bank_1_state_buffers_) {
    LITERT_RETURN_IF_ERROR(state_buffer.buffer.Clear());
  }
  if (bank_2_state_buffers_.has_value()) {
    for (auto& [_, state_buffer] : *bank_2_state_buffers_) {
      LITERT_RETURN_IF_ERROR(state_buffer.buffer.Clear());
    }
  }
  // A cleared ping-pong state must start from the same physical input bank as
  // a newly created state. Otherwise an even/odd number of prior executions
  // changes which cleared bank is presented first after reset.
  bank_1_is_input_ = true;
  return absl::OkStatus();
}

absl::Status LitertState::Resize(CompiledModel& compiled_model,
                                 absl::string_view signature_name,
                                 int num_entries) {
  RET_CHECK(!bank_2_state_buffers_.has_value())
          .SetCode(absl::StatusCode::kInvalidArgument)
      << "Out of place KV cache cannot be resized.";

  bool has_dynamic = false;
  for (const auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      has_dynamic = true;
      break;
    }
  }
  if (!has_dynamic) {
    return absl::InvalidArgumentError(
        "KV cache is not dynamic and cannot be resized.");
  }

  int entries_to_add = num_entries - num_entries_;
  if (entries_to_add <= 0) {
    return absl::OkStatus();
  }

  for (const auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, num_entries));
    }
  }

  for (auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          state_buffer.buffer,
          ResizeTensorBuffer(env_, state_buffer.buffer,
                             state_buffer.dynamic_dim.value(), entries_to_add));
    }
  }
  num_entries_ = num_entries;
  return absl::OkStatus();
}

absl::StatusOr<LitertState::StateBuffers> LitertState::GetStateBuffers(
    CompiledModel& compiled_model, absl::string_view signature_name) {
  LITERT_RETURN_IF_ERROR(SyncShapes(compiled_model, signature_name));
  auto* input_bank = &bank_1_state_buffers_;
  auto* output_bank = &bank_1_state_buffers_;

  if (bank_2_state_buffers_.has_value()) {
    if (bank_1_is_input_) {
      output_bank = &bank_2_state_buffers_.value();
    } else {
      input_bank = &bank_2_state_buffers_.value();
    }
    bank_1_is_input_ = !bank_1_is_input_;
  }

  StateBuffers buffers;
  const bool should_skip_inputs =
      allocation_policy_ == AllocationPolicy::kGpuOptimizedInplace;
  const bool is_prefill = absl::StartsWith(signature_name, "prefill");

  for (const auto& [input_name, state_buffer] : *input_bank) {
    const bool is_local_kv_cache =
        state_buffer.type == proto::StateBuffer::TYPE_LOCAL_KEY_CACHE ||
        state_buffer.type == proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE;
    if (should_skip_inputs && (!is_prefill || !is_local_kv_cache)) {
      // For GPU optimized in place updates, we are required to pass the local
      // KV cache buffers as inputs too in the prefill stage.
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto duplicated, state_buffer.buffer.Duplicate());
    buffers.input_buffers[input_name] = std::move(duplicated);
  }

  for (const auto& [input_name, state_buffer] : *output_bank) {
    LITERT_ASSIGN_OR_RETURN(auto duplicated, state_buffer.buffer.Duplicate());
    buffers.output_buffers[input_name] = std::move(duplicated);
  }
  return buffers;
}

absl::Status LitertState::SyncShapes(CompiledModel& compiled_model,
                                     absl::string_view signature_name) {
  for (const auto& [input_name, state_buffer] : bank_1_state_buffers_) {
    if (state_buffer.dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, num_entries_));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::HeuristicBasedCreate(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name, AllocationPolicy allocation_policy,
    int batch_size, bool clear_kv_cache_before_prefill) {
  LITERT_ASSIGN_OR_RETURN(
      auto sig_input_names,
      compiled_model.GetSignatureInputNames(signature_name));
  LITERT_ASSIGN_OR_RETURN(
      auto sig_output_names,
      compiled_model.GetSignatureOutputNames(signature_name));

  std::string k_root_name;
  std::string v_root_name;
  ABSL_RETURN_IF_ERROR(GetKVCacheRootNames(sig_input_names, sig_output_names,
                                           k_root_name, v_root_name));

  std::vector<std::string> key_cache_input_names;
  std::vector<std::string> value_cache_input_names;
  std::string mask_input_name;
  for (auto input_name : sig_input_names) {
    if (absl::StartsWith(input_name, k_root_name) ||
        absl::StartsWith(input_name, "kv_cache_c_")) {
      key_cache_input_names.push_back(std::string(input_name));
    }
    if (absl::StartsWith(input_name, v_root_name) ||
        absl::StartsWith(input_name, "kv_cache_c_")) {
      value_cache_input_names.push_back(std::string(input_name));
    }
    if (absl::StrContains(input_name, "mask")) {
      mask_input_name = input_name;
    }
  }
  if (key_cache_input_names.empty() || value_cache_input_names.empty()) {
    return absl::FailedPreconditionError("No KV cache inputs found.");
  }

  ABSL_ASSIGN_OR_RETURN(std::optional<int> k_dynamic_dim,
                        GetDynamicDimIndex(compiled_model, signature_name,
                                           key_cache_input_names[0]));
  ABSL_ASSIGN_OR_RETURN(std::optional<int> v_dynamic_dim,
                        GetDynamicDimIndex(compiled_model, signature_name,
                                           value_cache_input_names[0]));
  RET_CHECK(k_dynamic_dim.has_value() == v_dynamic_dim.has_value());

  auto create_and_init_buffer =
      [&](absl::string_view input_name, proto::StateBuffer::Type type,
          const std::optional<int>& dynamic_dim, bool is_output,
          bool clear_buffer) -> absl::StatusOr<StateBuffer> {
    if (dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, /*new_value=*/1));
    }

    // LiteRT macros don't support trinary operators. Hence the following block.
    std::optional<TensorBuffer> buffer;
    if (is_output) {
      LITERT_ASSIGN_OR_RETURN(buffer, compiled_model.CreateOutputBuffer(
                                          signature_name, input_name));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          buffer, compiled_model.CreateInputBuffer(signature_name, input_name));
    }

    if (clear_buffer) {
      LITERT_RETURN_IF_ERROR(buffer->Clear());
    }
    return StateBuffer{
        .buffer = std::move(buffer.value()),
        .type = type,
        .dynamic_dim = dynamic_dim,
    };
  };

  const bool is_gpu_optimized =
      allocation_policy == AllocationPolicy::kGpuOptimizedInplace;
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers;

  for (const auto& input_name : key_cache_input_names) {
    LITERT_ASSIGN_OR_RETURN(
        bank_1_state_buffers[input_name],
        create_and_init_buffer(input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
                               k_dynamic_dim, /*is_output=*/is_gpu_optimized,
                               /*clear_buffer=*/clear_kv_cache_before_prefill));
  }
  for (const auto& input_name : value_cache_input_names) {
    LITERT_ASSIGN_OR_RETURN(
        bank_1_state_buffers[input_name],
        create_and_init_buffer(input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
                               v_dynamic_dim, /*is_output=*/is_gpu_optimized,
                               /*clear_buffer=*/clear_kv_cache_before_prefill));
  }

  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers;
  if (allocation_policy == AllocationPolicy::kPingPong) {
    auto& bank_2 = bank_2_state_buffers.emplace();
    for (const auto& input_name : key_cache_input_names) {
      LITERT_ASSIGN_OR_RETURN(
          bank_2[input_name],
          create_and_init_buffer(
              input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
              /*dynamic_dim=*/std::nullopt, /*is_output=*/true,
              /*clear_buffer=*/false));
    }
    for (const auto& input_name : value_cache_input_names) {
      LITERT_ASSIGN_OR_RETURN(
          bank_2[input_name],
          create_and_init_buffer(
              input_name, proto::StateBuffer::TYPE_UNSPECIFIED,
              /*dynamic_dim=*/std::nullopt, /*is_output=*/true,
              /*clear_buffer=*/false));
    }
  }

  int context_size = 1;
  const bool is_dynamic_kv_cache = k_dynamic_dim.has_value();
  if (!is_dynamic_kv_cache) {
    if (!mask_input_name.empty()) {
      // Mask is our best bet for inferring context size. Key and value tensors
      // have different layouts and as such cannot be used directly.
      LITERT_ASSIGN_OR_RETURN(
          const RankedTensorType mask_tensor_type,
          compiled_model.GetInputTensorType(signature_name, mask_input_name));
      auto dims = mask_tensor_type.Layout().Dimensions();
      // Expect [1, 1, Sequence, KV Length]
      RET_CHECK_EQ(dims.size(), 4);
      context_size = dims[3];
    } else {
      // Fallback: get capacity from key cache tensor layout.
      // Usually key cache layout is [batch, num_heads, sequence_length,
      // head_dim]
      LITERT_ASSIGN_OR_RETURN(const RankedTensorType k_tensor_type,
                              compiled_model.GetInputTensorType(
                                  signature_name, key_cache_input_names[0]));
      auto dims = k_tensor_type.Layout().Dimensions();
      RET_CHECK_GE(dims.size(), 3);
      context_size = dims[2];
    }
  }

  return absl::WrapUnique(new LitertState(
      batch_size, context_size, env, std::move(bank_1_state_buffers),
      std::move(bank_2_state_buffers), allocation_policy,
      /*authoritative_state_inventory=*/false));
}

absl::StatusOr<std::unique_ptr<LitertState>> LitertState::MetadataBasedCreate(
    Environment& env, CompiledModel& compiled_model,
    absl::string_view signature_name,
    const proto::ExecutorMetadata& executor_metadata,
    AllocationPolicy allocation_policy, int batch_size,
    bool clear_kv_cache_before_prefill) {
  struct StateBufferMeta {
    std::string name;
    proto::StateBuffer::Type type;
    std::optional<int> dynamic_dim;
    bool is_global;
  };

  std::vector<StateBufferMeta> state_buffer_metas;
  std::optional<int> max_supported_sequence_size;

  for (const auto& state_buffer :
       executor_metadata.llm_executor_metadata().state_buffers()) {
    std::vector<absl::string_view> names;
    if (!state_buffer.prefill_input_name().empty()) {
      names.push_back(state_buffer.prefill_input_name());
    }
    if (!state_buffer.prefill_output_name().empty()) {
      names.push_back(state_buffer.prefill_output_name());
    }
    if (!state_buffer.decode_input_name().empty()) {
      names.push_back(state_buffer.decode_input_name());
    }
    if (!state_buffer.decode_output_name().empty()) {
      names.push_back(state_buffer.decode_output_name());
    }
    for (size_t i = 1; i < names.size(); ++i) {
      RET_CHECK_EQ(names[0], names[i])
          << "Current implementation requires all state names in StateBuffer"
             " to be the same: "
          << names[0] << " vs " << names[i];
    }

    RET_CHECK(!names.empty()) << "At least one state name must be defined";
    std::string input_name = std::string(names[0]);

    bool is_global = false;
    switch (state_buffer.type()) {
      case proto::StateBuffer::TYPE_GLOBAL_KEY_CACHE:
      case proto::StateBuffer::TYPE_GLOBAL_VALUE_CACHE:
        is_global = true;
        break;
      case proto::StateBuffer::TYPE_LOCAL_KEY_CACHE:
      case proto::StateBuffer::TYPE_LOCAL_VALUE_CACHE:
      case proto::StateBuffer::TYPE_LINEAR_ATTENTION:
        break;
      default:
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported state buffer type: ", state_buffer.type()));
    }

    LITERT_ASSIGN_OR_RETURN(
        const RankedTensorType ranked_tensor_type,
        compiled_model.GetInputTensorType(signature_name, input_name));
    auto dimensions = ranked_tensor_type.Layout().Dimensions();

    std::optional<int> dynamic_dim;
    if (state_buffer.has_sequence_axis()) {
      int axis = state_buffer.sequence_axis();
      RET_CHECK_GE(axis, 0);
      RET_CHECK_LT(axis, dimensions.size());
      if (dimensions[axis] == kDynamicDimValue) {
        dynamic_dim = axis;
      } else if (is_global) {
        int seq_size = dimensions[axis];
        if (max_supported_sequence_size.has_value()) {
          max_supported_sequence_size =
              std::min(*max_supported_sequence_size, seq_size);
        } else {
          max_supported_sequence_size = seq_size;
        }
      }
    } else {
      RET_CHECK(state_buffer.type() ==
                proto::StateBuffer::TYPE_LINEAR_ATTENTION)
          << "Sequence axis must be defined for state buffers in the current "
             "implementation.";
    }

    state_buffer_metas.push_back(StateBufferMeta{
        .name = std::move(input_name),
        .type = state_buffer.type(),
        .dynamic_dim = dynamic_dim,
        .is_global = is_global,
    });
  }

  if (state_buffer_metas.empty()) {
    return absl::InvalidArgumentError(
        "No state buffers found for the current signature");
  }

  auto create_and_init_buffer =
      [&](absl::string_view input_name, proto::StateBuffer::Type type,
          const std::optional<int>& dynamic_dim, bool is_output,
          bool clear_buffer) -> absl::StatusOr<StateBuffer> {
    if (dynamic_dim.has_value()) {
      ABSL_RETURN_IF_ERROR(ResolveDynamicShape(compiled_model, signature_name,
                                               input_name, /*new_value=*/1));
    }

    // LiteRT macros don't support trinary operators. Hence the following block.
    std::optional<TensorBuffer> buffer;
    if (is_output) {
      LITERT_ASSIGN_OR_RETURN(buffer, compiled_model.CreateOutputBuffer(
                                          signature_name, input_name));
    } else {
      LITERT_ASSIGN_OR_RETURN(
          buffer, compiled_model.CreateInputBuffer(signature_name, input_name));
    }
    if (clear_buffer) {
      LITERT_RETURN_IF_ERROR(buffer->Clear());
    }
    return StateBuffer{
        .buffer = std::move(buffer.value()),
        .type = type,
        .dynamic_dim = dynamic_dim,
    };
  };

  const bool is_gpu_optimized =
      allocation_policy == AllocationPolicy::kGpuOptimizedInplace;
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers;
  for (const auto& meta : state_buffer_metas) {
    LITERT_ASSIGN_OR_RETURN(
        bank_1_state_buffers[meta.name],
        create_and_init_buffer(meta.name, meta.type, meta.dynamic_dim,
                               /*is_output=*/is_gpu_optimized,
                               /*clear_buffer=*/clear_kv_cache_before_prefill));
  }

  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers;
  if (allocation_policy == AllocationPolicy::kPingPong) {
    auto& bank_2 = bank_2_state_buffers.emplace();
    for (const auto& meta : state_buffer_metas) {
      LITERT_ASSIGN_OR_RETURN(
          bank_2[meta.name],
          create_and_init_buffer(meta.name, meta.type,
                                 /*dynamic_dim=*/std::nullopt,
                                 /*is_output=*/true,
                                 /*clear_buffer=*/false));
    }
  }

  int context_size = 1;
  bool is_dynamic = false;
  for (const auto& meta : state_buffer_metas) {
    if (meta.dynamic_dim.has_value()) {
      is_dynamic = true;
      break;
    }
  }
  if (!is_dynamic) {
    if (max_supported_sequence_size.has_value()) {
      context_size = *max_supported_sequence_size;
    } else {
      context_size = std::numeric_limits<int>::max();
    }
  }

  return absl::WrapUnique(new LitertState(
      batch_size, context_size, env, std::move(bank_1_state_buffers),
      std::move(bank_2_state_buffers), allocation_policy,
      /*authoritative_state_inventory=*/true));
}

}  // namespace litert::lm
