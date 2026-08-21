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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_STATE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_STATE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/session_handoff_codec_contract.h"
#include "runtime/executor/state_interface.h"
#include "runtime/platform/hash/hasher.h"
#include "runtime/proto/executor_metadata.pb.h"

namespace litert::lm {

class LitertState : public StateInterface {
 public:
  enum class AllocationPolicy {
    // In-place update. The same buffer is used for both input and output KV
    // cache tensors.
    kInplace = 0,
    // Ping-pong update. Two sets of buffers are used and swapped between
    // inputs and outputs at each step. Required when the backend/model
    // does not support updating buffers in-place.
    kPingPong = 1,
    // GPU-optimized in-place update. Similar to kInplace, but optimized for
    // GPU backends that handle KV cache updates internally. In this mode,
    // KV cache buffers are allocated as GPU buffers (outputs) and we skip
    // passing them as inputs during execution to avoid overhead.
    kGpuOptimizedInplace = 2,
  };

  static absl::StatusOr<std::unique_ptr<LitertState>> Create(
      Environment& env, CompiledModel& compiled_model,
      absl::string_view signature_name,
      const proto::ExecutorMetadata* executor_metadata,
      AllocationPolicy allocation_policy, int batch_size,
      bool clear_kv_cache_before_prefill = true);

  int GetNumEntries() const override { return num_entries_; };

  int GetBatchSize() const override { return batch_size_; };

  AllocationPolicy allocation_policy() const { return allocation_policy_; }

  // True only when this state was constructed from runtime-owned executor
  // metadata that explicitly enumerated its generalized state buffers.
  bool HasAuthoritativeStateInventory() const {
    return authoritative_state_inventory_;
  }

  size_t StateBufferCount() const { return bank_1_state_buffers_.size(); }

  // Fails closed unless the allocation policy and every generalized state
  // buffer type are admitted for exact session handoff. GPU-optimized in-place
  // state is admitted only when every authoritative buffer is live Metal
  // memory and therefore participates in LRTST001's synchronized host staging
  // and transactional in-place restore. Heuristic KV discovery is deliberately
  // insufficient proof even of the caller-bound state inventory. This method
  // does not prove that a Metal delegate has no hidden continuation state;
  // complete capsule support additionally requires delegate-owned evidence.
  absl::Status ValidateSessionHandoffSupport() const;

  // Returns a canonical digest of the complete, executor-metadata-backed
  // LRTST001 state schema. Payload bytes and the mutable active-bank selector
  // are deliberately excluded; both ping-pong banks and their layouts are
  // included so the digest describes the full restorable structure.
  absl::StatusOr<Hash256> GetSessionHandoffStateInventoryHash() const;

  // Fails closed unless every history-dependent buffer is authoritatively
  // inventoried and can be replaced as one reset transaction. Unlike session
  // handoff, backend-native GPU-optimized allocations are admitted here
  // because reset recreates them instead of serializing them.
  absl::Status ValidateDeterministicProjectionResetSupport() const;

  // Stronger exact-profile check for a concrete Metal executor. Every
  // authoritatively inventoried continuation buffer in every state bank must
  // be backed by Metal memory. This proves the live backend allocation rather
  // than trusting Backend::GPU, but remains necessary rather than sufficient:
  // selected-pipeline policy and hidden delegate state are outside LitertState.
  absl::Status ValidateMetalStateStorageForExactProfile() const;

  // Serializes the logical input bank into a canonical, versioned snapshot.
  // The snapshot binds the state structure and buffer layout and includes an
  // integrity digest. The caller remains responsible for authenticating the
  // snapshot and binding it to a model/backend/profile identity.
  //
  // Serialization requires every buffer to support a host-readable lock. GPU
  // buffers, including Metal buffers, are staged through LiteRT's synchronized
  // lock contract. A backend that cannot provide that contract is rejected
  // rather than producing a partial snapshot. The caller must keep inference
  // quiescent until this operation returns.
  absl::StatusOr<std::string> Serialize() const override;
  absl::StatusOr<uint64_t> SerializedSize() const override;
  absl::Status SerializeTo(ByteSink* sink) const override;

  // Loads a snapshot only when its complete structure matches this state. A
  // dynamic sequence dimension may differ: non-Metal buffers are recreated at
  // the producer's canonical shape and committed with the entry count only
  // after every payload has been validated and staged. All other dimensions
  // and layout properties must match exactly. Metal buffers retain their live
  // GPU allocations and use a host rollback copy; a Metal shape change is
  // rejected to avoid the known managed-Metal allocation leak. The caller must
  // keep inference quiescent until this operation returns.
  absl::Status Load(absl::string_view serialized_state) override;
  absl::Status LoadFrom(const ByteSource& source,
                        bool target_is_disposable = false) override;

  // Reads only the immutable LRTST001 header and validates it against this
  // live state and the number of model-consumed token positions. This method
  // does not allocate, lock, write, or otherwise mutate live state. LoadFrom
  // still authenticates and validates the complete snapshot before commit.
  absl::Status ValidateSnapshotHeaderForImport(
      const ByteSource& source, int required_consumed_entries) const;

  absl::Status SelectAndCopyFrom(StateInterface& other,
                                 int batch_index) override;

  absl::Status BroadcastAndCopyFrom(StateInterface& other) override;

  absl::StatusOr<std::unique_ptr<StateInterface>> DeepCopy() const override;

  bool Contains(absl::string_view tensor_name) const override;

  absl::Status Clear() override;

  // Resizes the KV cache to the given number of entries (sequence length).
  // Note: Resize is a no-op if the requested size is smaller than the current
  // size.
  absl::Status Resize(CompiledModel& compiled_model,
                      absl::string_view signature_name, int num_entries);

  struct StateBuffers {
    absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers;
    absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers;
  };

  // For backends that support inplace update, this returns a single set of
  // state cache buffers that can be used for both input and output (i.e,
  // input_buffers and output_buffers point to the same data).
  // For backends that don't support inplace update, this returns two distinct
  // sets of state cache buffers, one for input and one for output. On each
  // call, the input/output buffers will be swapped.
  absl::StatusOr<StateBuffers> GetStateBuffers(
      CompiledModel& compiled_model, absl::string_view signature_name);

 private:
  struct StateBuffer {
    TensorBuffer buffer;
    proto::StateBuffer::Type type = proto::StateBuffer::TYPE_UNSPECIFIED;
    std::optional<int> dynamic_dim;
  };

  absl::Status ValidateStateInventory(bool allow_gpu_optimized) const;

  absl::Status SyncShapes(CompiledModel& compiled_model,
                          absl::string_view signature_name);

  static absl::StatusOr<std::unique_ptr<LitertState>> HeuristicBasedCreate(
      Environment& env, CompiledModel& compiled_model,
      absl::string_view signature_name, AllocationPolicy allocation_policy,
      int batch_size, bool clear_kv_cache_before_prefill);

  static absl::StatusOr<std::unique_ptr<LitertState>> MetadataBasedCreate(
      Environment& env, CompiledModel& compiled_model,
      absl::string_view signature_name,
      const proto::ExecutorMetadata& executor_metadata,
      AllocationPolicy allocation_policy, int batch_size,
      bool clear_kv_cache_before_prefill);

  LitertState(
      int batch_size, int num_entries, Environment& env,
      absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers,
      std::optional<absl::flat_hash_map<std::string, StateBuffer>>
          bank_2_state_buffers,
      AllocationPolicy allocation_policy, bool authoritative_state_inventory)
      : batch_size_(batch_size),
        num_entries_(num_entries),
        env_(env),
        bank_1_state_buffers_(std::move(bank_1_state_buffers)),
        bank_2_state_buffers_(std::move(bank_2_state_buffers)),
        allocation_policy_(allocation_policy),
        authoritative_state_inventory_(authoritative_state_inventory) {}

  // Batch size of the KV cache buffers.
  int batch_size_;
  // Number of entries in the KV cache.
  int num_entries_;
  // Environment to create new TensorBuffers (required for resizing).
  Environment& env_;
  // Primary state buffers.
  absl::flat_hash_map<std::string, StateBuffer> bank_1_state_buffers_;
  // Secondary state buffers - only used when inplace update is not
  // supported.
  std::optional<absl::flat_hash_map<std::string, StateBuffer>>
      bank_2_state_buffers_;

  bool bank_1_is_input_ = true;

  AllocationPolicy allocation_policy_;

  // Construction provenance is immutable session-handoff evidence. A caller
  // cannot upgrade heuristic state discovery after the model is loaded.
  const bool authoritative_state_inventory_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_LITERT_STATE_H_
