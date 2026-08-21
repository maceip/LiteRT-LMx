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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_STATE_INTERFACE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_STATE_INTERFACE_H_

#include <cstdint>
#include <memory>
#include <limits>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/util/byte_stream.h"

namespace litert::lm {

// The KV cache interface including all K and V buffers for a model.
class StateInterface {
 public:
  virtual ~StateInterface() = default;

  // Returns the total number of entries in the KV cache per block.
  virtual int GetNumEntries() const = 0;

  // Returns the batch size of the KV cache.
  virtual int GetBatchSize() const = 0;

  // Serializes the KV cache to a byte string.
  virtual absl::StatusOr<std::string> Serialize() const = 0;

  // Returns the exact byte count produced by SerializeTo. Implementations with
  // large device state should override this without materializing the bytes.
  virtual absl::StatusOr<uint64_t> SerializedSize() const {
    absl::StatusOr<std::string> bytes = Serialize();
    if (!bytes.ok()) return bytes.status();
    return static_cast<uint64_t>(bytes->size());
  }

  // Streams the canonical serialization to `sink`. The default adapter keeps
  // legacy state implementations source-compatible; production LiteRT state
  // overrides it with direct tensor-buffer streaming.
  virtual absl::Status SerializeTo(ByteSink* sink) const {
    if (sink == nullptr) {
      return absl::InvalidArgumentError("State byte sink must not be null.");
    }
    absl::StatusOr<std::string> bytes = Serialize();
    if (!bytes.ok()) return bytes.status();
    return sink->Append(*bytes);
  }

  // Loads the KV cache from a serialized byte string.
  virtual absl::Status Load(absl::string_view serialized_state) = 0;

  // Loads from random-access bytes. `target_is_disposable` permits an
  // implementation to avoid retaining rollback copies while populating a
  // staged context that the caller guarantees will be discarded on failure.
  // The default adapter materializes legacy state and ignores that hint.
  virtual absl::Status LoadFrom(const ByteSource& source,
                                bool target_is_disposable = false) {
    (void)target_is_disposable;
    const uint64_t source_size = source.Size();
    if (source_size > std::numeric_limits<size_t>::max()) {
      return absl::ResourceExhaustedError(
          "Serialized state exceeds addressable memory.");
    }
    std::string bytes;
    if (source_size > bytes.max_size()) {
      return absl::ResourceExhaustedError(
          "Serialized state exceeds std::string capacity.");
    }
    bytes.resize(static_cast<size_t>(source_size));
    absl::Status read = source.ReadAt(0, absl::MakeSpan(bytes));
    if (!read.ok()) return read;
    return Load(bytes);
  }

  // Selects a single batch from the other KV cache and copies it to this KV
  // cache.
  // Example:
  //   This has shape [1, ...] and other has shape [3, ...]. Then we can select
  //   batch x from other and copy it to this
  //   (i.e., other[x, :, ...] -> this[0, :, ...]).
  virtual absl::Status SelectAndCopyFrom(StateInterface& other,
                                         int batch_index) = 0;

  // Broadcasts the source KV with batch size 1 to this KV cache with batch size
  // > 1.
  // Example:
  //   This has shape [3, ...] and other has shape [1, ...]. Then we can copy
  //   other[0, :, ...] -> this[0, :, ...], this[1, :, ...], this[2, :, ...].
  virtual absl::Status BroadcastAndCopyFrom(StateInterface& other) = 0;

  // Deep copies the KV cache. This is an expensive operation. Use sparingly.
  virtual absl::StatusOr<std::unique_ptr<StateInterface>> DeepCopy() const = 0;

  // Returns true if this state container holds the tensor with the specified
  // `tensor_name`. The lookup maps to the signature input/output names of the
  // model.
  virtual bool Contains(absl::string_view tensor_name) const = 0;

  // Zeroes out all the buffers held in this state container. Should attempt to
  // perform the operation asynchronously if the underlying buffers support
  // asynchronous operations.
  virtual absl::Status Clear() = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_STATE_INTERFACE_H_
