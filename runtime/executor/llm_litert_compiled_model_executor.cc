// Copyright 2025 The ODML Authors.
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

#include "runtime/executor/llm_litert_compiled_model_executor.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"  // from @litert
#include "litert/c/litert_model.h"  // from @litert
#include "litert/c/litert_op_code.h"  // from @litert
#include "litert/cc/internal/litert_handle.h"  // from @litert
#include "litert/cc/litert_any.h"  // from @litert
#include "litert/cc/litert_compiled_model.h"  // from @litert
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_environment_options.h"  // from @litert
#include "litert/cc/litert_expected.h"  // from @litert
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_model.h"  // from @litert
#include "litert/cc/litert_model_types.h"  // from @litert
#include "litert/cc/litert_options.h"  // from @litert
#include "litert/cc/litert_profiler.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "litert/cc/litert_tensor_buffer_types.h"  // from @litert
#include "runtime/components/constrained_decoding/logits_processor.h"
#include "runtime/components/embedding_lookup/embedding_lookup_manager.h"
#include "runtime/components/greedy_cpu_sampler.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/engine/exact_litert_decode.h"
#include "runtime/executor/common_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert/state.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/llm_executor_processed_tokens.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_executor_settings_utils.h"
#include "runtime/executor/llm_litert_compiled_model_cache_utils.h"
#include "runtime/executor/llm_litert_mtp_drafter.h"
#include "runtime/executor/state_interface.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/log_tensor_buffer.h"
#include "runtime/util/lora_util.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

using ::absl::Span;

// Names of the signature runners, used to get the signature runners from the
// interpreter.
constexpr absl::string_view kPrefillSignatureRunner = "prefill";
constexpr absl::string_view kDecodeSignatureRunner = "decode";
constexpr int kDynamicDimValue = -1;

bool OpMayCarryUncapturedSessionState(LiteRtOpCode opcode) {
  switch (opcode) {
    case kLiteRtOpCodeTflHashtableLookup:
    case kLiteRtOpCodeTflLstm:
    case kLiteRtOpCodeTflRnn:
    case kLiteRtOpCodeTflSvdf:
    case kLiteRtOpCodeTflCustom:
    case kLiteRtOpCodeTflUnidirectionalSequenceRnn:
    case kLiteRtOpCodeTflUnidirectionalSequenceLstm:
    case kLiteRtOpCodeTflBidirectionalSequenceRnn:
    case kLiteRtOpCodeTflDelegate:
    case kLiteRtOpCodeTflBidirectionalSequenceLstm:
    case kLiteRtOpCodeTflCallOnce:
    case kLiteRtOpCodeTflHashtable:
    case kLiteRtOpCodeTflHashtableFind:
    case kLiteRtOpCodeTflHashtableImport:
    case kLiteRtOpCodeTflHashtableSize:
    case kLiteRtOpCodeTflVarHandle:
    case kLiteRtOpCodeTflReadVariable:
    case kLiteRtOpCodeTflAssignVariable:
    case kLiteRtOpCodeTflRandomStandardNormal:
    case kLiteRtOpCodeTflRandomUniform:
    case kLiteRtOpCodeTflMultinomial:
    case kLiteRtOpCodeShloCustomCall:
    case kLiteRtOpCodeShloRngBitGenerator:
    case kLiteRtOpCodeShloComposite:
      return true;
    default:
      return false;
  }
}

absl::Status ValidateTensorHasNoOpaqueStateType(
    LiteRtTensor tensor, LiteRtParamIndex subgraph_index,
    LiteRtParamIndex op_index, absl::string_view tensor_role,
    LiteRtParamIndex tensor_index) {
  if (tensor == nullptr) {
    return absl::FailedPreconditionError(
        "Session handoff encountered a null LiteRT operation tensor.");
  }
  LiteRtTensorTypeId type_id{};
  if (LiteRtGetTensorTypeId(tensor, &type_id) != kLiteRtStatusOk) {
    return absl::FailedPreconditionError(
        "Session handoff could not inspect a LiteRT tensor type.");
  }
  LiteRtElementType element_type{};
  switch (type_id) {
    case kLiteRtRankedTensorType: {
      LiteRtRankedTensorType ranked_type{};
      if (LiteRtGetRankedTensorType(tensor, &ranked_type) != kLiteRtStatusOk) {
        return absl::FailedPreconditionError(
            "Session handoff could not inspect a ranked LiteRT tensor.");
      }
      element_type = ranked_type.element_type;
      break;
    }
    case kLiteRtUnrankedTensorType: {
      LiteRtUnrankedTensorType unranked_type{};
      if (LiteRtGetUnrankedTensorType(tensor, &unranked_type) !=
          kLiteRtStatusOk) {
        return absl::FailedPreconditionError(
            "Session handoff could not inspect an unranked LiteRT tensor.");
      }
      element_type = unranked_type.element_type;
      break;
    }
    default:
      return absl::UnimplementedError(
          "Session handoff has not admitted an unknown LiteRT tensor type.");
  }
  if (element_type == kLiteRtElementTypeTfResource ||
      element_type == kLiteRtElementTypeTfVariant) {
    return absl::UnimplementedError(absl::StrCat(
        "Session handoff has not admitted opaque resource/variant ",
        tensor_role, " tensor ", tensor_index, " at subgraph ",
        subgraph_index, ", operation ", op_index, "."));
  }
  return absl::OkStatus();
}

absl::Status ValidateLoadedModelStatefulness(const Model& model) {
  if (model.Get() == nullptr) {
    return absl::FailedPreconditionError(
        "Session handoff cannot inspect a null loaded LiteRT model.");
  }
  LiteRtParamIndex subgraph_count = 0;
  if (LiteRtGetNumModelSubgraphs(model.Get(), &subgraph_count) !=
      kLiteRtStatusOk) {
    return absl::FailedPreconditionError(
        "Session handoff could not enumerate loaded LiteRT subgraphs.");
  }
  if (subgraph_count == 0) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT model has no inspectable subgraphs.");
  }
  for (LiteRtParamIndex subgraph_index = 0;
       subgraph_index < subgraph_count; ++subgraph_index) {
    LiteRtSubgraph subgraph = nullptr;
    if (LiteRtGetModelSubgraph(model.Get(), subgraph_index, &subgraph) !=
            kLiteRtStatusOk ||
        subgraph == nullptr) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Session handoff could not inspect LiteRT subgraph ",
          subgraph_index, "."));
    }
    LiteRtParamIndex op_count = 0;
    if (LiteRtGetNumSubgraphOps(subgraph, &op_count) != kLiteRtStatusOk) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Session handoff could not enumerate operations in LiteRT "
          "subgraph ",
          subgraph_index, "."));
    }
    for (LiteRtParamIndex op_index = 0; op_index < op_count; ++op_index) {
      LiteRtOp op = nullptr;
      if (LiteRtGetSubgraphOp(subgraph, op_index, &op) != kLiteRtStatusOk ||
          op == nullptr) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Session handoff could not inspect LiteRT operation ", op_index,
            " in subgraph ", subgraph_index, "."));
      }
      LiteRtOpCode opcode{};
      if (LiteRtGetOpCode(op, &opcode) != kLiteRtStatusOk) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Session handoff could not identify LiteRT operation ", op_index,
            " in subgraph ", subgraph_index, "."));
      }
      if (OpMayCarryUncapturedSessionState(opcode)) {
        return absl::UnimplementedError(absl::StrCat(
            "Session handoff has not admitted LiteRT opcode ",
            static_cast<int>(opcode), " at subgraph ", subgraph_index,
            ", operation ", op_index,
            "; its complete continuation state cannot be proven."));
      }
      LiteRtParamIndex input_count = 0;
      if (LiteRtGetNumOpInputs(op, &input_count) != kLiteRtStatusOk) {
        return absl::FailedPreconditionError(
            "Session handoff could not enumerate LiteRT operation inputs.");
      }
      for (LiteRtParamIndex input_index = 0; input_index < input_count;
           ++input_index) {
        LiteRtTensor input = nullptr;
        if (LiteRtGetOpInput(op, input_index, &input) != kLiteRtStatusOk) {
          return absl::FailedPreconditionError(
              "Session handoff could not inspect a LiteRT operation input.");
        }
        ABSL_RETURN_IF_ERROR(ValidateTensorHasNoOpaqueStateType(
            input, subgraph_index, op_index, "input", input_index));
      }
      LiteRtParamIndex output_count = 0;
      if (LiteRtGetNumOpOutputs(op, &output_count) != kLiteRtStatusOk) {
        return absl::FailedPreconditionError(
            "Session handoff could not enumerate LiteRT operation outputs.");
      }
      for (LiteRtParamIndex output_index = 0; output_index < output_count;
           ++output_index) {
        LiteRtTensor output = nullptr;
        if (LiteRtGetOpOutput(op, output_index, &output) != kLiteRtStatusOk) {
          return absl::FailedPreconditionError(
              "Session handoff could not inspect a LiteRT operation output.");
        }
        ABSL_RETURN_IF_ERROR(ValidateTensorHasNoOpaqueStateType(
            output, subgraph_index, op_index, "output", output_index));
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ValidateAuthoritativeStateMetadata(
    const proto::ExecutorMetadata* executor_metadata,
    const LitertState& state, absl::string_view state_label) {
  if (executor_metadata == nullptr ||
      !executor_metadata->has_llm_executor_metadata()) {
    return absl::UnimplementedError(
        "Session handoff requires runtime-owned LLM executor metadata.");
  }
  const auto& metadata_buffers =
      executor_metadata->llm_executor_metadata().state_buffers();
  if (metadata_buffers.empty()) {
    return absl::UnimplementedError(
        "Session handoff requires a non-empty executor-metadata state "
        "inventory.");
  }

  absl::flat_hash_set<std::string> inventoried_names;
  inventoried_names.reserve(metadata_buffers.size());
  for (const proto::StateBuffer& metadata_buffer : metadata_buffers) {
    const std::string& name = metadata_buffer.prefill_input_name();
    if (name.empty() || metadata_buffer.prefill_output_name().empty() ||
        metadata_buffer.decode_input_name().empty() ||
        metadata_buffer.decode_output_name().empty()) {
      return absl::UnimplementedError(
          "Session handoff requires every state buffer to be explicit in "
          "prefill/decode inputs and outputs.");
    }
    if (metadata_buffer.prefill_output_name() != name ||
        metadata_buffer.decode_input_name() != name ||
        metadata_buffer.decode_output_name() != name) {
      return absl::UnimplementedError(
          "Session handoff has not admitted aliased state-buffer names "
          "across prefill and decode signatures.");
    }
    if (!inventoried_names.insert(name).second) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Executor metadata contains duplicate state buffer ", name, "."));
    }
    if (!state.Contains(name)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Executor metadata state buffer ", name,
          " is absent from the loaded ", state_label, "."));
    }
  }
  if (inventoried_names.size() != state.StateBufferCount()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Executor metadata does not account for every loaded ", state_label,
        " buffer."));
  }
  return absl::OkStatus();
}

void AppendRuntimeU8(uint8_t value, std::string* output) {
  output->push_back(static_cast<char>(value));
}

void AppendRuntimeU32(uint32_t value, std::string* output) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendRuntimeU64(uint64_t value, std::string* output) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendRuntimeI32(int32_t value, std::string* output) {
  AppendRuntimeU32(std::bit_cast<uint32_t>(value), output);
}

void AppendRuntimeBool(bool value, std::string* output) {
  AppendRuntimeU8(value ? 1 : 0, output);
}

void AppendRuntimeBytes(absl::string_view value, std::string* output) {
  AppendRuntimeU32(static_cast<uint32_t>(value.size()), output);
  output->append(value.data(), value.size());
}

void AppendRuntimeOptionalBool(const std::optional<bool>& value,
                               std::string* output) {
  AppendRuntimeBool(value.has_value(), output);
  if (value.has_value()) AppendRuntimeBool(*value, output);
}

void AppendRuntimeOptionalInt(const std::optional<int>& value,
                              std::string* output) {
  AppendRuntimeBool(value.has_value(), output);
  if (value.has_value()) AppendRuntimeI32(*value, output);
}

absl::StatusOr<const void*> GetUniqueEnvironmentPointer(
    const EnvironmentOptions& options, EnvironmentOptions::Tag tag,
    absl::string_view label) {
  const void* result = nullptr;
  bool found = false;
  for (const EnvironmentOptions::Option& option : options.GetOptions()) {
    if (option.tag != tag) continue;
    if (found) {
      return absl::FailedPreconditionError(
          absl::StrCat("Loaded Environment contains duplicate ", label,
                       " options."));
    }
    found = true;
    if (const auto* value = std::get_if<const void*>(&option.value)) {
      result = *value;
    } else if (const auto* value = std::get_if<void*>(&option.value)) {
      result = *value;
    } else {
      return absl::FailedPreconditionError(
          absl::StrCat("Loaded Environment ", label,
                       " option has the wrong value type."));
    }
  }
  if (!found || result == nullptr) {
    return absl::FailedPreconditionError(
        absl::StrCat("Loaded Environment has no non-null ", label,
                     " option."));
  }
  return result;
}

bool HasEnvironmentTag(const EnvironmentOptions& options,
                       EnvironmentOptions::Tag tag) {
  for (const EnvironmentOptions::Option& option : options.GetOptions()) {
    if (option.tag == tag) return true;
  }
  return false;
}

template <typename BufferMap>
absl::Status AppendFixedTensorMapContract(absl::string_view label,
                                          const BufferMap& buffers,
                                          std::string* output) {
  std::vector<absl::string_view> names;
  names.reserve(buffers.size());
  for (const auto& [name, _] : buffers) names.push_back(name);
  std::sort(names.begin(), names.end());
  AppendRuntimeBytes(label, output);
  AppendRuntimeU32(static_cast<uint32_t>(names.size()), output);
  for (absl::string_view name : names) {
    if (name.empty()) {
      return absl::FailedPreconditionError(
          "Loaded decode tensor contract contains an empty name.");
    }
    const auto found = buffers.find(name);
    if (found == buffers.end()) {
      return absl::InternalError(
          "Loaded decode tensor contract changed while being measured.");
    }
    const TensorBuffer& buffer = found->second;
    LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                            buffer.TensorType());
    LITERT_ASSIGN_OR_RETURN(const TensorBufferType buffer_type,
                            buffer.BufferType());
    LITERT_ASSIGN_OR_RETURN(const size_t packed_size, buffer.PackedSize());
    LITERT_ASSIGN_OR_RETURN(const size_t size, buffer.Size());
    LITERT_ASSIGN_OR_RETURN(const size_t offset, buffer.Offset());
    const Layout& layout = tensor_type.Layout();
    for (int dimension : layout.Dimensions()) {
      if (dimension <= 0) {
        return absl::UnimplementedError(absl::StrCat(
            "Exact Metal decode requires a fixed positive shape for ", name,
            "."));
      }
    }
    AppendRuntimeBytes(name, output);
    AppendRuntimeI32(static_cast<int32_t>(tensor_type.ElementType()), output);
    AppendRuntimeI32(static_cast<int32_t>(buffer_type), output);
    AppendRuntimeU32(layout.Rank(), output);
    for (int dimension : layout.Dimensions()) {
      AppendRuntimeI32(dimension, output);
    }
    AppendRuntimeBool(layout.HasStrides(), output);
    if (layout.HasStrides()) {
      for (uint32_t stride : layout.Strides()) {
        AppendRuntimeU32(stride, output);
      }
    }
    AppendRuntimeU64(packed_size, output);
    AppendRuntimeU64(size, output);
    AppendRuntimeU64(offset, output);
    AppendRuntimeBool(buffer.IsMetalMemory(), output);
  }
  return absl::OkStatus();
}

absl::Status AppendStaticPrefillScheduleContract(
    absl::string_view label, absl::string_view scheduling_rule,
    const CompiledModel& compiled_model, const ModelSignatures& signatures,
    const SortedPrefillSignatureMap& prefill_schedule, std::string* output) {
  if (prefill_schedule.empty()) {
    return absl::FailedPreconditionError(
        "Loaded executor has no fixed prefill signature schedule.");
  }
  AppendRuntimeBytes(label, output);
  AppendRuntimeBytes(scheduling_rule, output);
  AppendRuntimeU32(static_cast<uint32_t>(prefill_schedule.size()), output);
  for (const auto& [sequence_length, signature_name] : prefill_schedule) {
    if (sequence_length <= 0 || signature_name.empty()) {
      return absl::FailedPreconditionError(
          "Loaded prefill schedule contains an invalid shape or signature.");
    }
    LITERT_ASSIGN_OR_RETURN(
        const RankedTensorType positions_type,
        compiled_model.GetInputTensorType(signature_name,
                                          signatures.input_positions));
    const Layout& positions_layout = positions_type.Layout();
    const auto position_dimensions = positions_layout.Dimensions();
    const bool implicit_batch_one =
        position_dimensions.size() == 1 &&
        position_dimensions[0] == sequence_length;
    const bool explicit_batch_one =
        position_dimensions.size() == 2 && position_dimensions[0] == 1 &&
        position_dimensions[1] == sequence_length;
    if (!implicit_batch_one && !explicit_batch_one) {
      return absl::UnimplementedError(
          "Exact prefill requires every compiled input-position signature to "
          "have fixed batch one and the advertised sequence length.");
    }
    AppendRuntimeI32(sequence_length, output);
    AppendRuntimeBytes(signature_name, output);
    AppendRuntimeI32(static_cast<int32_t>(positions_type.ElementType()), output);
    AppendRuntimeU32(positions_layout.Rank(), output);
    for (int dimension : position_dimensions) {
      AppendRuntimeI32(dimension, output);
    }
    AppendRuntimeBool(positions_layout.HasStrides(), output);
    if (positions_layout.HasStrides()) {
      for (uint32_t stride : positions_layout.Strides()) {
        AppendRuntimeU32(stride, output);
      }
    }
  }
  return absl::OkStatus();
}

bool RuntimeConfigsEqual(const RuntimeConfig& lhs, const RuntimeConfig& rhs) {
  if (lhs.sampler_params.has_value() != rhs.sampler_params.has_value() ||
      lhs.output_heads != rhs.output_heads ||
      lhs.tokens_per_decode != rhs.tokens_per_decode) {
    return false;
  }
  return !lhs.sampler_params.has_value() ||
         lhs.sampler_params->SerializeAsString() ==
             rhs.sampler_params->SerializeAsString();
}

absl::StatusOr<bool> HasDynamicDim(const CompiledModel& compiled_model,
                                   absl::string_view signature,
                                   absl::string_view tensor_name) {
  LITERT_ASSIGN_OR_RETURN(
      const RankedTensorType ranked_tensor_type,
      compiled_model.GetInputTensorType(signature, tensor_name));
  for (int dim : ranked_tensor_type.Layout().Dimensions()) {
    if (dim == kDynamicDimValue) {
      return true;
    }
  }
  return false;
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

// Builds the output tensor type for the embedding lookup. The output tensor
// type is the same as the input tensor type, except the first dimension is the
// number of tokens.
absl::StatusOr<RankedTensorType> GetEmbeddingLookupOutputTensorType(
    int num_tokens, const RankedTensorType& output_element_type) {
  if (num_tokens == 1) {
    return output_element_type;
  } else if (num_tokens == 0) {
    return absl::InvalidArgumentError(
        "Number of tokens must be greater than 0.");
  }

  const auto& dims = output_element_type.Layout().Dimensions();
  if (dims.size() < 3) {
    return absl::InvalidArgumentError("Tensor type must have rank 3 or more.");
  }
  if (dims[0] != 1 || dims[1] != 1) {
    return absl::InvalidArgumentError(
        "Element type must have first two dimensions as 1.");
  }
  Dimensions embedding_dims(dims.begin(), dims.end());
  embedding_dims[1] = num_tokens;
  return RankedTensorType(output_element_type.ElementType(),
                          Layout(std::move(embedding_dims)));
}

// Returns a subspan of the given span for a chunk at the given index.
template <typename T>
absl::Span<const T> GetSpanForChunk(absl::Span<T> span, int num_chunks,
                                    int chunk_index) {
  size_t total_size = span.size();
  size_t chunk_size = total_size / num_chunks;
  return span.subspan(chunk_size * chunk_index, chunk_size);
}

absl::StatusOr<TensorBuffer> CreateFP16OutputBuffer(
    Environment& env, CompiledModel& compiled_model, size_t signature_index,
    absl::string_view output_name, size_t output_index) {
  LITERT_ASSIGN_OR_RETURN(
      std::vector<Layout> runtime_layouts,
      compiled_model.GetOutputTensorLayouts(signature_index,
                                            /*update_allocation=*/true));
  // Use runtime layout.
  Layout runtime_layout = runtime_layouts[output_index];
  LITERT_ASSIGN_OR_RETURN(
      auto requirements,
      compiled_model.GetOutputBufferRequirements(signature_index, output_name));
  LITERT_ASSIGN_OR_RETURN(auto strides, requirements.Strides());
  if (!strides.empty()) {
    auto dims = runtime_layout.Dimensions();
    runtime_layout = Layout(litert::Dimensions(dims.begin(), dims.end()),
                            litert::Strides(strides.begin(), strides.end()));
  }
  RankedTensorType new_tensor_type(litert::ElementType::Float16,
                                   std::move(runtime_layout));
  LITERT_ASSIGN_OR_RETURN(size_t size, requirements.BufferSize());
  LITERT_ASSIGN_OR_RETURN(auto buffer_types, requirements.SupportedTypes());
  if (buffer_types.empty()) {
    return absl::InternalError("No supported buffer types found.");
  }
  auto buffer_type = buffer_types[0];
  LITERT_ASSIGN_OR_RETURN(
      auto buffer, TensorBuffer::CreateManaged(
                       env, buffer_type, std::move(new_tensor_type), size));
  return buffer;
}

absl::StatusOr<TensorBuffer> CreateHostOutputBuffer(
    Environment& env, CompiledModel& compiled_model, size_t signature_index,
    size_t output_index, RankedTensorType tensor_type) {
  LITERT_ASSIGN_OR_RETURN(
      std::vector<Layout> runtime_layouts,
      compiled_model.GetOutputTensorLayouts(signature_index,
                                            /*update_allocation=*/true));
  Layout runtime_layout = runtime_layouts[output_index];
  LITERT_ASSIGN_OR_RETURN(auto requirements,
                          compiled_model.GetOutputBufferRequirements(
                              signature_index, output_index));
  LITERT_ASSIGN_OR_RETURN(auto strides, requirements.Strides());
  if (!strides.empty()) {
    auto dims = runtime_layout.Dimensions();
    runtime_layout = Layout(litert::Dimensions(dims.begin(), dims.end()),
                            litert::Strides(strides.begin(), strides.end()));
  }
  RankedTensorType host_tensor_type(tensor_type.ElementType(),
                                    std::move(runtime_layout));
  LITERT_ASSIGN_OR_RETURN(size_t size, requirements.BufferSize());
  LITERT_ASSIGN_OR_RETURN(auto buffer, TensorBuffer::CreateManaged(
                                           env, TensorBufferType::kHostMemory,
                                           std::move(host_tensor_type), size));
  return buffer;
}

}  // namespace

absl::Status LlmLiteRtCompiledModelExecutorBase::CreatePrefillInputBuffers(
    absl::string_view prefill_signature, int sequence_length,
    int context_length,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_input_buffers) {
  auto dyn_shape_resolver = [&](absl::string_view tensor_name) -> absl::Status {
    return ResolveDynamicShape(*compiled_model_, prefill_signature, tensor_name,
                               sequence_length);
  };
  // Create input_token, positions and attn_mask buffers after determining
  // the prefill length.
  if (!signatures_.input_tokens.empty()) {
    ABSL_RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_tokens));
    LITERT_ASSIGN_OR_RETURN(auto tokens_buffer,
                            compiled_model_->CreateInputBuffer(
                                prefill_signature, signatures_.input_tokens));
    prefill_input_buffers[signatures_.input_tokens] = std::move(tokens_buffer);
  } else {
    // If input_tokens is empty, we must have input_embeddings.
    if (!signatures_.input_embeddings.has_value()) {
      return absl::FailedPreconditionError(
          "Input tokens or embeddings must be provided.");
    }
    if (embedding_lookup_ == nullptr) {
      return absl::FailedPreconditionError(
          "Input embeddings required by signature but embedding lookup "
          "model is not initialized.");
    }
    ABSL_RETURN_IF_ERROR(
        dyn_shape_resolver(signatures_.input_embeddings.value()));
    LITERT_ASSIGN_OR_RETURN(
        auto embeddings_buffer,
        compiled_model_->CreateInputBuffer(
            prefill_signature, signatures_.input_embeddings.value()));
    prefill_input_buffers[signatures_.input_embeddings.value()] =
        std::move(embeddings_buffer);

    // We may have per layer embedding as well.
    if (signatures_.input_per_layer_embeddings.has_value()) {
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Input per layer embeddings required by signature but "
            "embedding lookup model is not initialized.");
      }
      ABSL_RETURN_IF_ERROR(
          dyn_shape_resolver(signatures_.input_per_layer_embeddings.value()));
      LITERT_ASSIGN_OR_RETURN(
          auto per_layer_embeddings_buffer,
          compiled_model_->CreateInputBuffer(
              prefill_signature,
              signatures_.input_per_layer_embeddings.value()));
      prefill_input_buffers[signatures_.input_per_layer_embeddings.value()] =
          std::move(per_layer_embeddings_buffer);
    }
  }
  ABSL_RETURN_IF_ERROR(dyn_shape_resolver(signatures_.input_positions));
  LITERT_ASSIGN_OR_RETURN(auto positions_buffer,
                          compiled_model_->CreateInputBuffer(
                              prefill_signature, signatures_.input_positions));
  prefill_input_buffers[signatures_.input_positions] =
      std::move(positions_buffer);

  if (signatures_.input_attn_mask.has_value()) {
    ABSL_ASSIGN_OR_RETURN(bool is_attn_dyn,
                          HasDynamicDim(*compiled_model_, prefill_signature,
                                        signatures_.input_attn_mask.value()));
    if (is_attn_dyn) {
      std::vector<int> new_shape = {1, 1, sequence_length, context_length};
      LITERT_RETURN_IF_ERROR(compiled_model_->ResizeInputTensor(
          prefill_signature, signatures_.input_attn_mask.value(), new_shape));
    }

    LITERT_ASSIGN_OR_RETURN(
        auto attn_mask_buffer,
        compiled_model_->CreateInputBuffer(
            prefill_signature, signatures_.input_attn_mask.value()));
    prefill_input_buffers[signatures_.input_attn_mask.value()] =
        std::move(attn_mask_buffer);
    if (signatures_.input_attn_mask_local.has_value()) {
      auto attn_mask_local_buffer = compiled_model_->CreateInputBuffer(
          prefill_signature, signatures_.input_attn_mask_local.value());
      prefill_input_buffers[signatures_.input_attn_mask_local.value()] =
          std::move(*attn_mask_local_buffer);
    }
  }
  if (signatures_.input_int32_param.has_value()) {
    LITERT_ASSIGN_OR_RETURN(
        auto param_tensor_buffer,
        compiled_model_->CreateInputBuffer(
            prefill_signature, signatures_.input_int32_param.value()));
    prefill_input_buffers[signatures_.input_int32_param.value()] =
        std::move(param_tensor_buffer);
  }
  return absl::OkStatus();
}

// Allocates and initializes non-KV-cache output buffers for a given prefill
// signature. KV-cache buffers are skipped as they are managed independently
// by LitertState.
absl::Status LlmLiteRtCompiledModelExecutorBase::CreatePrefillOutputBuffers(
    absl::string_view prefill_signature, int sequence_length,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_output_buffers) {
  LITERT_ASSIGN_OR_RETURN(auto signature,
                          compiled_model_->FindSignature(prefill_signature));

  for (auto output_name : signature.OutputNames()) {
    // Skip KV-cache state tensors; their lifecycle and memory allocation are
    // owned and maintained entirely by LitertState.
    if (IsKVCacheTensor(output_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto output_buffer,
        compiled_model_->CreateOutputBuffer(prefill_signature, output_name));
    prefill_output_buffers[output_name] = std::move(output_buffer);
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::FillInputBufferWithToken(
    const std::vector<std::shared_ptr<TokenData>>& unprocessed_token,
    TensorBuffer& input_buffer, bool is_per_layer_embedding) {
  if (unprocessed_token.empty()) {
    return absl::InvalidArgumentError("Unprocessed token is null.");
  }

  LITERT_ASSIGN_OR_RETURN(auto input_buffer_lock_and_addr,
                          TensorBufferScopedLock::Create(
                              input_buffer, TensorBuffer::LockMode::kWrite));
  LITERT_ASSIGN_OR_RETURN(size_t packed_size, input_buffer.PackedSize());
  size_t stride = packed_size / unprocessed_token.size();
  char* input_buffer_ptr =
      static_cast<char*>(input_buffer_lock_and_addr.second);
  for (const auto& token : unprocessed_token) {
    size_t size_to_fill = 0;
    if (token->embedding().empty()) {
      size_to_fill = sizeof(int32_t);
      RET_CHECK_GE(stride, size_to_fill);
      // If the token has no embedding, the input_buffer should takes token id.
      *reinterpret_cast<int32_t*>(input_buffer_ptr) = token->id();
    } else if (is_per_layer_embedding) {
      size_to_fill = token->per_layer_embedding().size() * sizeof(float);
      RET_CHECK_GE(stride, size_to_fill);
      memcpy(input_buffer_ptr, token->per_layer_embedding().data(),
             size_to_fill);
    } else {
      size_to_fill = token->embedding().size() * sizeof(float);
      RET_CHECK_GE(stride, size_to_fill);
      memcpy(input_buffer_ptr, token->embedding().data(), size_to_fill);
    }

    if (stride > size_to_fill) {
      memset(input_buffer_ptr + size_to_fill, 0, stride - size_to_fill);
    }
    input_buffer_ptr += stride;
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::RollBackProcessedTokens() {
  int current_step = llm_context_->runtime_state().current_step;
  ProcessedTokens& processed_tokens =
      llm_context_->processed_context().processed_tokens();
  if (current_step == processed_tokens.TokenCount()) {
    return absl::OkStatus();
  }
  if (current_step == 0) {
    ABSL_RETURN_IF_ERROR(processed_tokens.RollBackToStep(0));
  } else {
    auto token_at_step = processed_tokens.GetTokenAtStep(current_step - 1);
    ABSL_RETURN_IF_ERROR(processed_tokens.RollBackToStep(current_step - 1));
    if (!token_at_step.empty()) {
      RET_CHECK_EQ(token_at_step.size(), 1);
      // Multimodal input cannot become a pending input token.
      if (token_at_step.at(0) > 0) {
        ABSL_RETURN_IF_ERROR(processed_tokens.AddPendingInputToken(
            {std::make_shared<TokenData>(token_at_step.at(0))}));
      } else {
        processed_tokens.AddProcessedTokens({token_at_step.at(0)});
      }
    }
  }

  // Reset sampler input handling as the step is rolled back.
  if (sampler_ != nullptr && sampler_->HandlesInput()) {
    ABSL_RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrepareFirstPrefillAfterDecode(
    int token_index_to_reduce) {
  if (!llm_context_->runtime_state().ran_decode && !force_prepare_needed_) {
    return absl::OkStatus();
  }

  force_prepare_needed_ = false;
  llm_context_->runtime_state().ran_decode = false;

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }

  if (output_heads > 1) {
    LITERT_RETURN_IF_ERROR(llm_context_->processed_context()
                               .processed_tokens()
                               .ReduceTokenCandidates(token_index_to_reduce));
    RET_CHECK(state_ != nullptr);
    RET_CHECK(decode_state_ != nullptr);
    LITERT_RETURN_IF_ERROR(
        state_->SelectAndCopyFrom(*decode_state_, token_index_to_reduce));
  }

  // Reset sampler input handling if it handles input for next decode.
  if (sampler_ != nullptr && sampler_->HandlesInput()) {
    ABSL_RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrefillInternal(
    absl::string_view prefill_signature,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_output_buffers,
    Span<const int> ids, bool async) {
  ABSL_RETURN_IF_ERROR(RollBackProcessedTokens());

  auto [internal_start_step_initial, pending_input_token_initial] =
      llm_context_->processed_context()
          .processed_tokens()
          .GetNextUnprocessedToken();

  {
    // Fill the input buffers with scoped locks.
    auto& prefill_input_pos =
        prefill_input_buffers[signatures_.input_positions];
    LITERT_ASSIGN_OR_RETURN(auto prefill_input_pos_size,
                            prefill_input_pos.PackedSize());
    LITERT_ASSIGN_OR_RETURN(
        auto prefill_input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(prefill_input_pos,
                                       TensorBuffer::LockMode::kWrite));
    auto* prefill_input_pos_ptr =
        static_cast<int32_t*>(prefill_input_pos_lock_and_addr.second);

    memset(prefill_input_pos_ptr, 0, prefill_input_pos_size);
    if (signatures_.input_attn_mask.has_value()) {
      ABSL_RETURN_IF_ERROR(InitializeAttentionMask(
          prefill_input_buffers[signatures_.input_attn_mask.value()],
          use_fp16_precision_));
      if (signatures_.input_attn_mask_local.has_value()) {
        ABSL_RETURN_IF_ERROR(InitializeAttentionMask(
            prefill_input_buffers[signatures_.input_attn_mask_local.value()],
            use_fp16_precision_));
      }
    }
    // TODO(b/425396146): Add the unit tests for checking the prefill length.
    // We always hold one pending token in the input ids for the next
    // prefill or decode step.
    int prefill_length = ids.size() - 1;

    // Check if have a pending input token. Note that 'internal_start_step' is
    // always equal to the number of processed tokens plus 1.
    auto [internal_start_step, pending_input_token] =
        llm_context_->processed_context()
            .processed_tokens()
            .GetNextUnprocessedToken();
    RET_CHECK_LE(pending_input_token.size(), 1);
    const int start_step = internal_start_step;
    const bool has_pending_input_token = !pending_input_token.empty();
    const bool use_token_as_lookup = !signatures_.input_tokens.empty();
    const bool use_per_layer_embedding =
        signatures_.input_per_layer_embeddings.has_value();
    // If there is no pending input token and no input token to prefill, we can
    // skip the prefill by storing the token as a pending input token.
    bool skip_prefill = !has_pending_input_token && prefill_length == 0;
    if (!skip_prefill) {
      int input_idx = 0;
      if (has_pending_input_token) {
        if (use_token_as_lookup) {
          ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
              pending_input_token,
              prefill_input_buffers[signatures_.input_tokens]));
        } else {
          ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
              pending_input_token,
              prefill_input_buffers[signatures_.input_embeddings.value()]));
          if (use_per_layer_embedding) {
            ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
                pending_input_token,
                prefill_input_buffers[signatures_.input_per_layer_embeddings
                                          .value()],
                /*is_per_layer_embedding=*/true));
          }
        }
        prefill_input_pos_ptr[input_idx] = internal_start_step;
        ABSL_RETURN_IF_ERROR(llm_context_->processed_context()
                                 .processed_tokens()
                                 .MarkPendingInputTokenAsProcessed());
        llm_context_->runtime_state().current_step = internal_start_step + 1;

        ++prefill_input_pos_ptr;
        ++input_idx;
      }
      std::transform(prefill_input_pos_ptr,
                     prefill_input_pos_ptr + prefill_length,
                     prefill_input_pos_ptr, [&](int token) mutable {
                       return llm_context_->runtime_state().current_step++;
                     });
      std::vector<int> processed_input_tokens(ids.begin(),
                                              ids.begin() + prefill_length);
      llm_context_->processed_context().processed_tokens().AddProcessedTokens(
          processed_input_tokens);

      if (use_token_as_lookup) {
        auto& prefill_input_buffer =
            prefill_input_buffers[signatures_.input_tokens];
        LITERT_ASSIGN_OR_RETURN(
            auto prefill_input_lock_and_addr,
            TensorBufferScopedLock::Create(prefill_input_buffer,
                                           TensorBuffer::LockMode::kWrite));
        int32_t* prefill_input_ptr =
            static_cast<int32_t*>(prefill_input_lock_and_addr.second);
        if (!has_pending_input_token) {
          LITERT_ASSIGN_OR_RETURN(auto prefill_input_size,
                                  prefill_input_buffer.PackedSize());
          // If there is a pending input token, the zeros and the pending input
          // token id are already filled in the above
          // FillInputBufferWithToken() function, so we cannot zero out the
          // whole prefill input buffer here.
          //
          // If there is no pending input token, we need to zero out the whole
          // prefill input buffer.
          memset(prefill_input_ptr, 0, prefill_input_size);
        }
        memcpy(prefill_input_ptr + input_idx, processed_input_tokens.data(),
               processed_input_tokens.size() * sizeof(int32_t));
      } else {
        // If not using token as lookup, we must have input_embeddings. There is
        // no need to create input_embeddings_ptr because TensorBuffer locking
        // and filling is handled by the embedding lookup.
        if (embedding_lookup_ == nullptr) {
          return absl::FailedPreconditionError(
              "Prefill requires embedding_lookup_ when use_token_as_lookup is "
              "false, but embedding_lookup_ is null.");
        }
        TensorBuffer* prefill_input_embeddings_buffer =
            &(prefill_input_buffers[signatures_.input_embeddings.value()]);
        ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
            processed_input_tokens, prefill_input_embeddings_buffer,
            /*offset=*/input_idx));

        // We may have per layer embedding as well.
        if (signatures_.input_per_layer_embeddings) {
          if (per_layer_embedding_lookup_ == nullptr) {
            return absl::FailedPreconditionError(
                "Prefill requires per_layer_embedding_lookup_ when signature "
                "has input_per_layer_embeddings, but per_layer_embedding_"
                "lookup_ is null.");
          }
          TensorBuffer* prefill_input_per_layer_embeddings_buffer =
              &(prefill_input_buffers[signatures_.input_per_layer_embeddings
                                          .value()]);
          ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
              processed_input_tokens, prefill_input_per_layer_embeddings_buffer,
              /*offset=*/input_idx));
        }
      }
      if (signatures_.input_attn_mask.has_value()) {
        const AttentionMaskParams attn_params =
            GetAttentionMaskParams(executor_metadata_);
        auto tokens_copy = llm_context_->processed_context()
                               .processed_tokens()
                               .GetCopyOfTokens();
        absl::Span<const int> token_ids_span =
            tokens_copy.empty() ? absl::Span<const int>()
                                : absl::MakeConstSpan(tokens_copy[0]);

        ABSL_RETURN_IF_ERROR(FillAttentionMask(
            prefill_input_buffers[signatures_.input_attn_mask.value()],
            start_step,
            /*steps=*/prefill_length + input_idx, attn_params.global_type,
            token_ids_span,
            /*sliding_window_size=*/std::nullopt));
        if (signatures_.input_attn_mask_local.has_value()) {
          ABSL_LOG(INFO) << "filling local attention mask";
          ABSL_RETURN_IF_ERROR(FillAttentionMask(
              prefill_input_buffers[signatures_.input_attn_mask_local.value()],
              start_step,
              /*steps=*/prefill_length + input_idx, attn_params.local_type,
              token_ids_span, attn_params.sliding_window_size));
        }
      }
      if (gpu_optimized_single_buffer_cache_) {
        LITERT_RETURN_IF_ERROR(signatures_.input_int32_param.has_value());
        ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
            prefill_input_buffers[signatures_.input_int32_param.value()],
            start_step, ids.size()));
      }
    }

    // Add the last token of the current input as a pending input token, to be
    // used in the next prefill or decode.
    auto last_input_token = std::make_shared<TokenData>(ids.back());
    if (!use_token_as_lookup) {
      // Look up the embeddings for the last token so they can be used in the
      // next prefill or decode. This has to be done now in the case of
      // multi-modal prefill so the embeddings are used in the correct order.
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Prefill requires embedding_lookup_ for the last pending token "
            "when use_token_as_lookup is false, but embedding_lookup_ is "
            "null.");
      }
      ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
          last_input_token->id(), last_input_token->mutable_embedding()));
      if (use_per_layer_embedding) {
        if (per_layer_embedding_lookup_ == nullptr) {
          return absl::FailedPreconditionError(
              "Prefill requires per_layer_embedding_lookup_ for the last "
              "pending token, but per_layer_embedding_lookup_ is null.");
        }
        ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            last_input_token->id(),
            last_input_token->mutable_per_layer_embedding()));
      }
    }
    // Add the last input token to the pending input token list.
    ABSL_RETURN_IF_ERROR(
        llm_context_->processed_context()
            .processed_tokens()
            .AddPendingInputToken({std::move(last_input_token)}));
    ++llm_context_->runtime_state().current_step;
    if (skip_prefill) {
      return absl::OkStatus();
    }
  }
  return BindTensorsAndRunPrefill(prefill_signature, prefill_input_buffers,
                                  prefill_output_buffers, async);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunPrefill(
    absl::string_view prefill_signature,
    absl::flat_hash_map<absl::string_view, TensorBuffer>& prefill_input_buffers,
    absl::flat_hash_map<absl::string_view, TensorBuffer>&
        prefill_output_buffers,
    bool async) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> input_buffers;
  for (const auto& [input_name, input_buffer] : prefill_input_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    input_buffers[input_name] = std::move(input_buffer_dup);
  }

  LitertState* litert_state = nullptr;
  if (state_ != nullptr) {
    litert_state = dynamic_cast<LitertState*>(state_.get());
    RET_CHECK(litert_state != nullptr);
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> output_buffers;

  if (litert_state != nullptr) {
    LITERT_ASSIGN_OR_RETURN(
        auto state_buffers,
        litert_state->GetStateBuffers(*compiled_model_, prefill_signature));
    for (auto& [name, buffer] : state_buffers.input_buffers) {
      input_buffers[name] = std::move(buffer);
    }
    for (auto& [name, buffer] : state_buffers.output_buffers) {
      buffer.ClearEvent();
      output_buffers[name] = std::move(buffer);
    }
  }
  // Bind non-KV-cache output buffers to the final output buffers map.
  // Duplicate buffer handles and clear completion events so they are ready
  // for the upcoming graph execution.
  for (const auto& [output_name, output_buffer] : prefill_output_buffers) {
    LITERT_ASSIGN_OR_RETURN(auto output_buffer_dup, output_buffer.Duplicate());
    output_buffer_dup.ClearEvent();
    output_buffers[output_name] = std::move(output_buffer_dup);
  }

  if (pre_graph_run_callback_) {
    ABSL_ASSIGN_OR_RETURN(auto current_step, GetCurrentStep());
    pre_graph_run_callback_(prefill_signature, current_step, input_buffers);
  }

  litert::Options run_options = GetRunOptions();
  if (async) {
    LITERT_RETURN_IF_ERROR(compiled_model_->RunAsync(
        prefill_signature, input_buffers, output_buffers, async, &run_options));
  } else {
    LITERT_RETURN_IF_ERROR(compiled_model_->Run(
        prefill_signature, input_buffers, output_buffers, &run_options));
  }

  if (post_graph_run_callback_) {
    ABSL_ASSIGN_OR_RETURN(auto current_step, GetCurrentStep());
    post_graph_run_callback_(prefill_signature, current_step, output_buffers);
  }

  return absl::OkStatus();
}

absl::StatusOr<ProcessedTokens::StepAndToken>
LlmLiteRtCompiledModelExecutorBase::GetTokenToDecode(
    const ExecutorInputs& inputs) {
  ABSL_RETURN_IF_ERROR(RollBackProcessedTokens());

  if (inputs.GetTextDataPtr().ok()) {
    LITERT_ASSIGN_OR_RETURN(auto token_ids_buffer, inputs.GetTextTokenIdsPtr());
    auto input_tensor_size = token_ids_buffer->PackedSize();
    if (input_tensor_size && *input_tensor_size != 0) {
      int output_heads = 1;
      if (llm_context_->runtime_config().output_heads.has_value()) {
        output_heads = llm_context_->runtime_config().output_heads.value();
      }
      // Input token ids provided, so use it regardless of whether next input
      // token id is set.
      RET_CHECK_EQ(*input_tensor_size, output_heads * sizeof(int32_t));
      LITERT_ASSIGN_OR_RETURN(
          auto ids, ReferTensorBufferAsSpan<int32_t>(*token_ids_buffer));
      if (ids[0] >= 0) {
        // If the input token id is >= 0, it means the input token is provided
        // by the user. In this case, we should invalidate the pending input
        // token and add the input token as a pending input token.
        llm_context_->processed_context()
            .processed_tokens()
            .InvalidatePendingInputToken();
        std::vector<std::shared_ptr<TokenData>> token;
        token.reserve(output_heads);
        for (int i = 0; i < output_heads; ++i) {
          token.push_back(std::make_shared<TokenData>(ids[i]));
        }
        ABSL_RETURN_IF_ERROR(llm_context_->processed_context()
                                 .processed_tokens()
                                 .AddPendingInputToken(token));
      }
    }
  }

  // Here we must have a pending input token to decode that's either coming from
  // the previous prefill or decode, or we just added one from the inputs.
  for (const auto& token : llm_context_->processed_context()
                               .processed_tokens()
                               .GetNextUnprocessedToken()
                               .token) {
    // If the token has no embedding, we will look up the embedding for the
    // token here. This reduces the complexity for internal or external
    // sampling.
    if (signatures_.input_embeddings.has_value() &&
        token->mutable_embedding().empty()) {
      if (embedding_lookup_ == nullptr) {
        return absl::FailedPreconditionError(
            "Decode requires embedding_lookup_ when input_embeddings are used, "
            "but embedding_lookup_ is null.");
      }
      ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupDecode(
          token->id(), token->mutable_embedding()));
      if (signatures_.input_per_layer_embeddings.has_value()) {
        if (per_layer_embedding_lookup_ == nullptr) {
          return absl::FailedPreconditionError(
              "Decode requires per_layer_embedding_lookup_ when required by "
              "signature, but per_layer_embedding_lookup_ is null.");
        }
        ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupDecode(
            token->id(), token->mutable_per_layer_embedding()));
      }
    }
  }
  return llm_context_->processed_context()
      .processed_tokens()
      .GetNextUnprocessedToken();
}

absl::Status
LlmLiteRtCompiledModelExecutorBase::ConsumePendingOrAddProcessedToken(
    const std::vector<std::shared_ptr<TokenData>>& token) {
  auto status = llm_context_->processed_context()
                    .processed_tokens()
                    .MarkPendingInputTokenAsProcessed();
  if (status.ok() || status.code() != absl::StatusCode::kNotFound) {
    return status;
  }

  // If the pending input token was not used, we should add the token to the
  // processed tokens.
  std::vector<int> processed_tokens;
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  processed_tokens.reserve(output_heads);
  for (const auto& t : token) {
    processed_tokens.push_back(t->id());
  }
  llm_context_->processed_context().processed_tokens().AddProcessedTokens(
      processed_tokens);
  ++llm_context_->runtime_state().current_step;
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::DecodeInternal(
    const std::vector<std::shared_ptr<TokenData>>& token,
    TensorBuffer& output_logits) {
  int step = llm_context_->runtime_state().current_step - 1;
  if (sampler_ && sampler_->HandlesInput()) {
    // The sampler has already been running decode for this step. Check if
    // output_logits is the one used last time, i.e. by
    // BindTensorsAndRunDecodeStatic().
    LITERT_RETURN_IF_ERROR(
        output_logits.Get() ==
        decode_output_buffers_[signatures_.output_logits].Get());
    return absl::OkStatus();
  }

  const bool use_token_as_lookup = !signatures_.input_tokens.empty();
  const bool use_per_layer_embedding =
      signatures_.input_per_layer_embeddings.has_value();

  // Fill the input buffers with scoped locks.
  if (use_token_as_lookup) {
    ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
        token, decode_input_buffers_[signatures_.input_tokens]));
  } else {
    if (!signatures_.input_embeddings.has_value()) {
      return absl::InvalidArgumentError(
          "Input tokens or embeddings must be provided.");
    }
    ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
        token, decode_input_buffers_[signatures_.input_embeddings.value()]));
    if (use_per_layer_embedding) {
      ABSL_RETURN_IF_ERROR(FillInputBufferWithToken(
          token,
          decode_input_buffers_[signatures_.input_per_layer_embeddings.value()],
          /*is_per_layer_embedding=*/true));
    }
  }

  {
    LITERT_ASSIGN_OR_RETURN(
        auto input_pos_type,
        decode_input_buffers_[signatures_.input_positions].TensorType());
    LITERT_ASSIGN_OR_RETURN(
        auto input_pos_lock_and_addr,
        TensorBufferScopedLock::Create(
            decode_input_buffers_[signatures_.input_positions],
            TensorBuffer::LockMode::kWrite));
    auto* input_pos_ptr = static_cast<int32_t*>(input_pos_lock_and_addr.second);
    if (input_pos_type.Layout().Dimensions()[0] == 1) {
      *input_pos_ptr = step;
    } else {
      int output_heads = 1;
      if (llm_context_->runtime_config().output_heads.has_value()) {
        output_heads = llm_context_->runtime_config().output_heads.value();
      }
      RET_CHECK_EQ(input_pos_type.Layout().Dimensions()[0], output_heads);
      LITERT_ASSIGN_OR_RETURN(
          auto input_pos_size,
          decode_input_buffers_[signatures_.input_positions].PackedSize());
      size_t offset = input_pos_size / output_heads / sizeof(int32_t);
      for (int i = 0; i < output_heads; ++i) {
        input_pos_ptr[i * offset] = step;
      }
    }
  }

  if (signatures_.input_attn_mask.has_value()) {
    ABSL_RETURN_IF_ERROR(InitializeAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()],
        use_fp16_precision_));
    if (signatures_.input_attn_mask_local.has_value()) {
      ABSL_RETURN_IF_ERROR(InitializeAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask_local.value()],
          use_fp16_precision_));
    }
    const AttentionMaskParams attn_params =
        GetAttentionMaskParams(executor_metadata_);
    auto tokens_copy =
        llm_context_->processed_context().processed_tokens().GetCopyOfTokens();
    absl::Span<const int> token_ids_span =
        tokens_copy.empty() ? absl::Span<const int>()
                            : absl::MakeConstSpan(tokens_copy[0]);

    ABSL_RETURN_IF_ERROR(FillAttentionMask(
        decode_input_buffers_[signatures_.input_attn_mask.value()], step,
        /*steps=*/1, attn_params.global_type, token_ids_span,
        /*sliding_window_size=*/std::nullopt));
    if (signatures_.input_attn_mask_local.has_value()) {
      ABSL_RETURN_IF_ERROR(FillAttentionMask(
          decode_input_buffers_[signatures_.input_attn_mask_local.value()],
          step,
          /*steps=*/1, attn_params.local_type, token_ids_span,
          attn_params.sliding_window_size));
    }
  }
  if (gpu_optimized_single_buffer_cache_) {
    LITERT_RETURN_IF_ERROR(signatures_.input_int32_param.has_value());
    ABSL_RETURN_IF_ERROR(FillSingleBufferCacheParamTensor(
        decode_input_buffers_[signatures_.input_int32_param.value()], step, 1));
  }

  return BindTensorsAndRunDecode(&output_logits);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecode(
    TensorBuffer* output_logits) {
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  for (const auto& [input_name, input_buffer] : decode_input_buffers_) {
    LITERT_ASSIGN_OR_RETURN(auto input_buffer_dup, input_buffer.Duplicate());
    decode_input_buffers[input_name] = std::move(input_buffer_dup);
  }

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  StateInterface* active_state =
      (output_heads > 1) ? decode_state_.get() : state_.get();
  RET_CHECK(active_state != nullptr);

  auto* litert_state = dynamic_cast<LitertState*>(active_state);
  RET_CHECK(litert_state != nullptr);

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  for (const auto& [output_name, output_buffer] : decode_output_buffers_) {
    // LITERT_ASSIGN_OR_RETURN() causes a compilation error on windows.
    auto output_buffer_dup =
        output_logits && output_name == signatures_.output_logits
            ? output_logits->Duplicate()
            : output_buffer.Duplicate();
    RET_CHECK(output_buffer_dup) << "Failed to duplicate output buffer.";
    output_buffer_dup->ClearEvent();
    decode_output_buffers[output_name] = std::move(*output_buffer_dup);
  }

  LITERT_ASSIGN_OR_RETURN(
      auto state_buffers,
      litert_state->GetStateBuffers(*compiled_model_, kDecodeSignatureRunner));
  for (auto& [name, buffer] : state_buffers.input_buffers) {
    decode_input_buffers[name] = std::move(buffer);
  }
  for (auto& [name, buffer] : state_buffers.output_buffers) {
    buffer.ClearEvent();
    decode_output_buffers[name] = std::move(buffer);
  }

  if (pre_graph_run_callback_) {
    ABSL_ASSIGN_OR_RETURN(auto current_step, GetCurrentStep());
    pre_graph_run_callback_(kDecodeSignatureRunner, current_step,
                            decode_input_buffers);
  }

  litert::Options run_options = GetRunOptions();
  bool async = true;
  LITERT_RETURN_IF_ERROR(
      compiled_model_->RunAsync(kDecodeSignatureRunner, decode_input_buffers,
                                decode_output_buffers, async, &run_options));

  if (post_graph_run_callback_) {
    ABSL_ASSIGN_OR_RETURN(auto current_step, GetCurrentStep());
    post_graph_run_callback_(kDecodeSignatureRunner, current_step,
                             decode_output_buffers);
  }

  return absl::OkStatus();
}

int LlmLiteRtCompiledModelExecutorBase::BindTensorsAndRunDecodeStatic(
    void* arg) {
  auto self = static_cast<LlmLiteRtCompiledModelExecutorBase*>(arg);
  // Run decode with default output_logits.
  auto status = self->BindTensorsAndRunDecode(/*output_logits=*/nullptr);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to bind tensors and run decode: " << status;
  }
  return status.raw_code();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::PrepareFirstDecode() {
  if (llm_context_->runtime_state().ran_decode && !force_prepare_needed_) {
    return absl::OkStatus();
  }
  force_prepare_needed_ = false;
  // Mark that we have run decode at least once.
  llm_context_->runtime_state().ran_decode = true;

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }

  if (output_heads <= 1) {
    return absl::OkStatus();
  }

  LITERT_RETURN_IF_ERROR(llm_context_->processed_context()
                             .processed_tokens()
                             .BroadcastTokenCandidates(output_heads));

  RET_CHECK(state_ != nullptr);
  RET_CHECK(decode_state_ != nullptr);
  LITERT_RETURN_IF_ERROR(decode_state_->BroadcastAndCopyFrom(*state_));

  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtCompiledModelExecutorBase::Decode() {
  return Decode(ExecutorDecodeParams());
}

absl::StatusOr<std::vector<std::vector<int>>>
LlmLiteRtCompiledModelExecutorBase::Decode(
    const ExecutorDecodeParams& decode_params) {

  const std::shared_ptr<ExactLiteRtDecodeCapture>& exact_capture =
      decode_params.GetExactLiteRtDecodeCapture();
  if (exact_capture != nullptr) {
    // Every rejection in this block precedes DecodeLogits and therefore
    // precedes graph/session mutation. Exact evidence is confined to the one-
    // token compiled path whose CPU sampler implementation is the stable
    // strict-greater-than scan with the minimum-index tie rule. Model compute
    // may use a separately admitted CPU or GPU backend.
    if (mtp_drafter_ != nullptr) {
      return absl::UnimplementedError(
          "Exact decode does not support speculative or MTP decoding.");
    }
    if (!decode_params.GetLogitsProcessorList().empty()) {
      return absl::UnimplementedError(
          "Exact decode does not support logits processors or constraints.");
    }
    if (HasGraphRunCallbacks()) {
      return absl::UnimplementedError(
          "Exact decode does not support graph callbacks.");
    }
    const LlmExecutorSettings settings = [this]() {
      absl::MutexLock lock(executor_settings_mutex_);
      return executor_settings_;
    }();
    if (compiled_model_ == nullptr || llm_context_ == nullptr) {
      return absl::FailedPreconditionError(
          "Exact decode has no loaded compiled model or active context.");
    }
    if (compiled_backend_ != Backend::CPU &&
        compiled_backend_ != Backend::GPU) {
      return absl::UnimplementedError(
          "Exact decode supports only loaded LiteRT CPU or GPU compiled-model "
          "execution.");
    }
    if (settings.GetBackend() != compiled_backend_) {
      return absl::FailedPreconditionError(
          "Exact decode executor settings no longer match the compiled "
          "backend.");
    }
    ABSL_ASSIGN_OR_RETURN(const Backend configured_sampler_backend,
                          GetSamplerBackend(settings));
    if (configured_sampler_backend != Backend::CPU) {
      return absl::UnimplementedError(
          "Exact decode requires CPU-side stable GREEDY token selection.");
    }
    if (settings.GetLoraRank() != 0 ||
        llm_context_->processed_context().lora_id().has_value()) {
      return absl::UnimplementedError(
          "Exact decode does not support configured or active LoRA state.");
    }
    const RuntimeConfig& runtime_config = llm_context_->runtime_config();
    if (!runtime_config.output_heads.has_value() ||
        *runtime_config.output_heads != 1 ||
        !runtime_config.tokens_per_decode.has_value() ||
        *runtime_config.tokens_per_decode != 1) {
      return absl::UnimplementedError(
          "Exact decode requires one output head and one token per compiled "
          "decode invocation.");
    }
    if (!runtime_config.sampler_params.has_value() ||
        runtime_config.sampler_params->type() !=
            proto::SamplerParameters::GREEDY ||
        runtime_config.sampler_params->backend() !=
            proto::SamplerParameters::CPU) {
      return absl::UnimplementedError(
          "Exact decode runtime context is not the explicit CPU GREEDY "
          "sampler profile.");
    }
    const RuntimeState& runtime_state = llm_context_->runtime_state();
    if (runtime_state.current_step < 0 || runtime_state.rand_gen == nullptr) {
      return absl::FailedPreconditionError(
          "Exact decode runtime continuation state is incomplete.");
    }
    const ProcessedTokens& processed_tokens =
        llm_context_->processed_context().processed_tokens();
    // This ProcessedTokens-level validation is intentionally narrower than
    // executor session handoff: it rejects pending embeddings and malformed
    // candidate state without imposing capsule/cache requirements.
    ABSL_RETURN_IF_ERROR(processed_tokens.ValidateSessionHandoffSupport());
    if (processed_tokens.TokenCount() != runtime_state.current_step) {
      return absl::FailedPreconditionError(
          "Exact decode step does not match its processed-token history.");
    }
    ABSL_ASSIGN_OR_RETURN(const int loaded_vocabulary_size,
                          GetLoadedVocabularySizeForSessionHandoff());
    ABSL_RETURN_IF_ERROR(
        processed_tokens.ValidateTokenIds(loaded_vocabulary_size));

    const ExactLiteRtLogitsFrameContract& expected_logits =
        exact_capture->contract();
    if (expected_logits.batch_size != 1 ||
        expected_logits.sequence_size != 1 ||
        expected_logits.vocabulary_size !=
            static_cast<uint32_t>(loaded_vocabulary_size)) {
      return absl::FailedPreconditionError(
          "Exact decode capture contract differs from the loaded executor "
          "shape.");
    }
    const auto loaded_logits = decode_output_buffers_.find(
        absl::string_view(signatures_.output_logits));
    if (signatures_.output_logits.empty() ||
        loaded_logits == decode_output_buffers_.end()) {
      return absl::FailedPreconditionError(
          "Exact decode has no loaded logits output allocation.");
    }
    LITERT_ASSIGN_OR_RETURN(const RankedTensorType loaded_logits_type,
                            loaded_logits->second.TensorType());
    const ElementType expected_element_type =
        expected_logits.element_type == ExactLiteRtLogitsElementType::kFloat16
            ? ElementType::Float16
            : ElementType::Float32;
    if (loaded_logits_type.ElementType() != expected_element_type ||
        loaded_logits_type.Layout().HasStrides()) {
      return absl::FailedPreconditionError(
          "Exact decode capture contract differs from the loaded packed "
          "logits representation.");
    }
    const auto loaded_logits_dimensions =
        loaded_logits_type.Layout().Dimensions();
    if (loaded_logits_dimensions.size() != 3 ||
        loaded_logits_dimensions[0] != 1 ||
        loaded_logits_dimensions[1] != 1 ||
        loaded_logits_dimensions[2] != loaded_vocabulary_size) {
      return absl::FailedPreconditionError(
          "Exact decode loaded logits are not [1, 1, vocabulary].");
    }
    LITERT_ASSIGN_OR_RETURN(const size_t loaded_logits_bytes,
                            loaded_logits->second.PackedSize());
    if (loaded_logits_bytes != expected_logits.byte_count) {
      return absl::FailedPreconditionError(
          "Exact decode capture byte extent differs from the loaded complete "
          "logits frame.");
    }

    if (!settings.GetAdvancedSettings().has_value()) {
      return absl::FailedPreconditionError(
          "Exact decode has no resolved compiled executor settings.");
    }
    const AdvancedSettings& advanced = *settings.GetAdvancedSettings();
    if (advanced.is_benchmark || advanced.enable_profiling ||
        advanced.num_logits_to_print_after_decode != 0 ||
        advanced.enable_speculative_decoding) {
      return absl::UnimplementedError(
          "Exact decode does not support benchmark, profiling, or logits "
          "debug callbacks.");
    }

    ActivationDataType logits_data_type;
    switch (exact_capture->contract().element_type) {
      case ExactLiteRtLogitsElementType::kFloat16:
        logits_data_type = ActivationDataType::FLOAT16;
        break;
      case ExactLiteRtLogitsElementType::kFloat32:
        logits_data_type = ActivationDataType::FLOAT32;
        break;
      default:
        return absl::UnimplementedError(
            "Exact decode requires an Engine-derived FP16 or FP32 logits "
            "contract.");
    }
    ABSL_RETURN_IF_ERROR(InitializeSampler(logits_data_type));
    if (initialized_sampler_backend_ != Backend::CPU ||
        initialized_sampler_type_ !=
            static_cast<int>(proto::SamplerParameters::GREEDY) ||
        dynamic_cast<GreedyCpuSampler*>(sampler_.get()) == nullptr ||
        sampler_handles_input_) {
      return absl::FailedPreconditionError(
          "Exact decode did not resolve to the stable CPU GREEDY min-index "
          "sampler path.");
    }
  }

  std::vector<std::vector<int>> output_tokens_vector;
  if (mtp_drafter_ == nullptr) {
    ABSL_ASSIGN_OR_RETURN(auto decoded_logits,
                          DecodeLogits(ExecutorInputs(), decode_params));
    std::optional<TensorBuffer> output_tokens;
    {
      LITERT_ASSIGN_OR_RETURN(auto decoded_logits_type,
                              decoded_logits.TensorType());
      auto dimensions = decoded_logits_type.Layout().Dimensions();
      // Shape of decoded_logits is [batch_size, Token_length, vocab_size].
      RET_CHECK_EQ(dimensions.size(), 3);
      LITERT_ASSIGN_OR_RETURN(
          output_tokens,
          CreateTensorBuffer<int>({dimensions[0], dimensions[1]}));
    }
    if (exact_capture != nullptr) {
      ABSL_RETURN_IF_ERROR(
          exact_capture->CaptureLogitsBeforeSampling(decoded_logits));
    }
    ABSL_RETURN_IF_ERROR(SampleLogits(decoded_logits, *output_tokens));
    LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                            CopyFromTensorBuffer2D<int>(*output_tokens));
    if (exact_capture != nullptr) {
      ABSL_RETURN_IF_ERROR(
          exact_capture->CaptureSampledTokenIds(output_tokens_vector));
    }
  } else {
    // MTP keeps an internal state of the last time it was called and will
    // use those projected activations to kick off the next draft steps. As
    // such, we need to do a single decode step on the first decode call after
    // prefill and provide the projected activations to the MTP drafted only
    // once.
    StateInterface* active_state = state_.get();
    RET_CHECK(active_state != nullptr);

    bool last_run_is_decode = llm_context_->runtime_state().ran_decode;
    if (last_run_is_decode) {
      ABSL_ASSIGN_OR_RETURN(auto step_and_token,
                            GetTokenToDecode(ExecutorInputs()));
      ABSL_RETURN_IF_ERROR(
          ConsumePendingOrAddProcessedToken(step_and_token.token));
      // Output: [Batch, drafted and verified tokens]
      LITERT_ASSIGN_OR_RETURN(
          output_tokens_vector,
          mtp_drafter_->Draft(step_and_token.step,
                              step_and_token.token[0]->id(),
                              /*activations=*/std::nullopt, *active_state));
      RET_CHECK_EQ(output_tokens_vector.size(), 1);
      llm_context_->runtime_state().current_step +=
          output_tokens_vector[0].size();
    } else {
      int token_id = -1;
      {
        ABSL_ASSIGN_OR_RETURN(auto decoded_logits,
                              DecodeLogits(ExecutorInputs(), decode_params));
        LITERT_ASSIGN_OR_RETURN(auto decoded_logits_type,
                                decoded_logits.TensorType());
        auto dimensions = decoded_logits_type.Layout().Dimensions();
        // Shape of decoded_logits is [batch_size, Token_length, vocab_size].
        RET_CHECK_EQ(dimensions.size(), 3);
        LITERT_ASSIGN_OR_RETURN(
            auto output_tokens,
            CreateTensorBuffer<int>({dimensions[0], dimensions[1]}));
        ABSL_RETURN_IF_ERROR(SampleLogits(decoded_logits, output_tokens));
        LITERT_ASSIGN_OR_RETURN(output_tokens_vector,
                                CopyFromTensorBuffer2D<int>(output_tokens));
        RET_CHECK_EQ(output_tokens_vector.size(), 1);
        RET_CHECK_EQ(output_tokens_vector[0].size(), 1);
        token_id = output_tokens_vector[0][0];
      }

      RET_CHECK(decode_output_buffers_.contains("activations"));
      LITERT_ASSIGN_OR_RETURN(
          auto activations, decode_output_buffers_["activations"].Duplicate());
      // Note: Position remains the same as the prefill step. However,
      // current_step is incremented in DecodeLogits and as such needs to be
      // decremented.
      LITERT_ASSIGN_OR_RETURN(
          output_tokens_vector,
          mtp_drafter_->Draft(llm_context_->runtime_state().current_step - 1,
                              token_id, std::move(activations), *active_state));
      llm_context_->runtime_state().current_step +=
          output_tokens_vector[0].size();
      output_tokens_vector[0].insert(output_tokens_vector[0].begin(), token_id);
    }
  }

  // Check for any invalid token ids and set them to zero, if any.
  bool has_invalid_output_token = false;
  for (int batch = 0; batch < output_tokens_vector.size(); ++batch) {
    for (int token_idx = 0; token_idx < output_tokens_vector[batch].size();
         ++token_idx) {
      if (output_tokens_vector[batch][token_idx] < 0) {
        has_invalid_output_token = true;
        output_tokens_vector[batch][token_idx] = 0;
      }
    }
  }
  if (has_invalid_output_token) {
    absl::MutexLock lock(executor_settings_mutex_);
    const auto& advanced_settings = executor_settings_.GetAdvancedSettings();
    if (advanced_settings.has_value() &&
        advanced_settings->error_on_invalid_sampled_token_id) {
      return absl::InternalError(
          "Invalid decode and sample result. The sampled token is negative. "
          "This is caused by invalid sampling or sampling from an invalid "
          "logits tensor, usually an overflowed logits tensor.");
    }
    ABSL_LOG(WARNING) << "Invalid decode and sample result. The sampled token "
                         "is casted to 0 to avoid crash.";
  }

  // Update context with the assumption that there is one output per head.
  // We must change this when doing drafter based decoding.
  std::vector<int> processed_tokens;
  std::vector<std::shared_ptr<TokenData>> pending_tokens;
  for (auto& output_head_tokens : output_tokens_vector) {
    for (int i = 0; i < output_head_tokens.size(); ++i) {
      // Last token is reserved as pending input token.
      if (i == output_head_tokens.size() - 1) {
        pending_tokens.push_back(
            std::make_shared<TokenData>(output_head_tokens[i]));
      } else {
        processed_tokens.push_back(output_head_tokens[i]);
      }
    }
  }
  if (!processed_tokens.empty()) {
    llm_context_->processed_context().processed_tokens().AddProcessedTokens(
        processed_tokens);
  }
  ABSL_RETURN_IF_ERROR(
      llm_context_->processed_context().processed_tokens().AddPendingInputToken(
          pending_tokens));

  return output_tokens_vector;
}

absl::Status LlmLiteRtCompiledModelExecutorBase::Decode(
    const ExecutorInputs& inputs, TensorBuffer& output_logits) {
  ABSL_RETURN_IF_ERROR(PrepareFirstDecode());
  ABSL_ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  ABSL_RETURN_IF_ERROR(DecodeInternal(step_and_token.token, output_logits));
  ABSL_RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));
  ++llm_context_->runtime_state().current_step;
  return absl::OkStatus();
}

absl::StatusOr<TensorBuffer> LlmLiteRtCompiledModelExecutorBase::DecodeLogits(
    const ExecutorInputs& inputs) {
  return DecodeLogits(inputs, ExecutorDecodeParams());
}

absl::StatusOr<TensorBuffer> LlmLiteRtCompiledModelExecutorBase::DecodeLogits(
    const ExecutorInputs& inputs, const ExecutorDecodeParams& decode_params) {
  LITERT_ASSIGN_OR_RETURN(
      auto output_logits,
      decode_output_buffers_[signatures_.output_logits].Duplicate());

  bool last_run_is_decode = llm_context_->runtime_state().ran_decode;
  ABSL_RETURN_IF_ERROR(PrepareFirstDecode());
  ABSL_ASSIGN_OR_RETURN(auto step_and_token, GetTokenToDecode(inputs));
  ABSL_RETURN_IF_ERROR(DecodeInternal(step_and_token.token, output_logits));
  ABSL_RETURN_IF_ERROR(ConsumePendingOrAddProcessedToken(step_and_token.token));

  if (!decode_params.GetLogitsProcessorList().empty() &&
      !step_and_token.token.empty()) {
    int output_heads = 1;
    if (llm_context_->runtime_config().output_heads.has_value()) {
      output_heads = llm_context_->runtime_config().output_heads.value();
    }

    RET_CHECK_EQ(step_and_token.token.size(), output_heads);
    std::vector<int> current_token_ids;
    current_token_ids.reserve(output_heads);
    for (const auto& token : step_and_token.token) {
      current_token_ids.push_back(token->id());
    }
    // Update constraint state only with decode ids.
    if (last_run_is_decode) {
      for (LogitsProcessor* logits_processor :
           decode_params.GetLogitsProcessorList()) {
        ABSL_RETURN_IF_ERROR(
            logits_processor->UpdateState(absl::MakeSpan(current_token_ids)));
      }
    }

    LITERT_ASSIGN_OR_RETURN(auto output_logits_buffer_type,
                            output_logits.BufferType());
    // If the output logits are already on the host memory, use the buffer
    // directly.
    if (output_logits_buffer_type == TensorBufferType::kHostMemory) {
      // Process logits based on the current constraint state.
      for (LogitsProcessor* logits_processor :
           decode_params.GetLogitsProcessorList()) {
        ABSL_RETURN_IF_ERROR(logits_processor->ProcessLogits(output_logits));
      }
    } else {
      // For GPU, we always copy the logits to CPU and mask them, then write
      // them back to GPU.
      LITERT_ASSIGN_OR_RETURN(RankedTensorType logits_tensor_type,
                              output_logits.TensorType());
      if (logits_tensor_type.ElementType() == ElementType::Float32) {
        // Copy the logits from the tensor buffer to a vector.
        LITERT_ASSIGN_OR_RETURN(auto logits_vector,
                                CopyFromTensorBuffer<float>(output_logits));
        // Process the logits using the logits processor.
        for (LogitsProcessor* logits_processor :
             decode_params.GetLogitsProcessorList()) {
          ABSL_RETURN_IF_ERROR(logits_processor->ProcessLogits(
              absl::MakeSpan(logits_vector.data(), logits_vector.size()),
              logits_tensor_type.Layout().Dimensions()));
        }
        // Write the processed logits back to the tensor buffer.
        output_logits.Write(
            absl::MakeConstSpan(logits_vector.data(), logits_vector.size()));
      } else if (logits_tensor_type.ElementType() ==
                 litert::ElementType::Float16) {
        // Copy the logits from the tensor buffer to a vector.
        LITERT_ASSIGN_OR_RETURN(
            auto logits_vector,
            CopyFromTensorBuffer<tflite::half>(output_logits));

        // Process the logits using the logits processor.
        for (LogitsProcessor* logits_processor :
             decode_params.GetLogitsProcessorList()) {
          ABSL_RETURN_IF_ERROR(logits_processor->ProcessLogits(
              absl::MakeSpan(logits_vector.data(), logits_vector.size()),
              logits_tensor_type.Layout().Dimensions()));
        }
        // Write the processed logits back to the tensor buffer.
        output_logits.Write(
            absl::MakeConstSpan(logits_vector.data(), logits_vector.size()));
      } else {
        return absl::InvalidArgumentError(
            "Output logits are not in float32 or float16 type.");
      }
    }
  }

  ++llm_context_->runtime_state().current_step;

  std::optional<AdvancedSettings> advanced_settings;
  {
    absl::MutexLock lock(executor_settings_mutex_);
    advanced_settings = executor_settings_.GetAdvancedSettings();
  }
  if (advanced_settings &&
      advanced_settings->num_logits_to_print_after_decode > 0) {
    LogTensor(output_logits,
              advanced_settings->num_logits_to_print_after_decode, "Logits")
        .IgnoreError();
  }
  return output_logits;
}

absl::StatusOr<std::string>
LlmLiteRtCompiledModelExecutorBase::GetPrefillSignatureKey() const {
  std::string prefill_signature_key;
  for (int i = 0; i < model_.GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto sig, model_.GetSignature(i));
    absl::string_view key = sig.Key();
    if (absl::StartsWith(key, kPrefillSignatureRunner)) {
      prefill_signature_key = key;
      break;
    }
  }
  RET_CHECK(!prefill_signature_key.empty());
  return prefill_signature_key;
}

absl::StatusOr<std::unique_ptr<StateInterface>>
LlmLiteRtCompiledModelExecutorBase::CloneState() const {
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  StateInterface* active_state =
      (output_heads > 1) ? decode_state_.get() : state_.get();
  if (active_state == nullptr) {
    return nullptr;
  }
  return active_state->DeepCopy();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::RestoreState(
    std::unique_ptr<StateInterface> state) {
  if (state == nullptr) {
    return absl::OkStatus();
  }
  if (state->GetBatchSize() > 1) {
    decode_state_ = std::move(state);
  } else {
    state_ = std::move(state);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmLiteRtCompiledModelExecutorBase::CreateNewContext(
    std::optional<uint32_t> lora_id, RuntimeConfig runtime_config) const {
  std::unique_ptr<ProcessedContext> processed_context =
      std::make_unique<LlmProcessedContext>(lora_id, nullptr);

  auto runtime_state = std::make_unique<RuntimeState>();
  if (runtime_config.sampler_params.has_value()) {
    runtime_state->rand_gen = std::make_shared<std::default_random_engine>(
        runtime_config.sampler_params->seed());
  } else {
    runtime_state->rand_gen = std::make_shared<std::default_random_engine>(0);
  }

  return std::make_unique<LlmContext>(
      std::move(processed_context),
      std::make_unique<RuntimeConfig>(std::move(runtime_config)),
      std::move(runtime_state));
}

absl::StatusOr<std::unique_ptr<LlmContext>>
LlmLiteRtCompiledModelExecutorBase::CloneContext() const {
  std::optional<uint32_t> lora_id;
  ABSL_ASSIGN_OR_RETURN(auto state, CloneState());
  ProcessedTokens new_processed_tokens =
      llm_context_->processed_context().processed_tokens();
  auto new_processed_context = std::make_unique<LlmProcessedContext>(
      std::move(lora_id), std::move(state), std::move(new_processed_tokens));
  auto new_runtime_config =
      std::make_unique<RuntimeConfig>(llm_context_->runtime_config());
  auto new_runtime_state =
      std::make_unique<RuntimeState>(llm_context_->runtime_state());
  return std::make_unique<LlmContext>(std::move(new_processed_context),
                                      std::move(new_runtime_config),
                                      std::move(new_runtime_state));
}

absl::Status LlmLiteRtCompiledModelExecutorBase::RestoreContext(
    std::unique_ptr<LlmContext> context_data) {
  if (context_data == nullptr) {
    return absl::InvalidArgumentError("Cannot restore a null LLM context.");
  }
  std::optional<ContextSampler> replacement_sampler;
  if (sampler_ != nullptr) {
    absl::Status bind_status = BindSamplerToContext(
        context_data->runtime_config(), context_data->runtime_state());
    if (!bind_status.ok()) {
      // Sampling algorithm is session configuration, not immutable executor
      // state. Stage a replacement so a previous session cannot prevent a
      // fresh CPU-GREEDY projection context from becoming active.
      absl::StatusOr<ContextSampler> staged = CreateSamplerForContext(
          context_data->runtime_config(), context_data->runtime_state());
      if (!staged.ok()) {
        BindSamplerToContext(llm_context_->runtime_config(),
                             llm_context_->runtime_state())
            .IgnoreError();
        return staged.status();
      }
      replacement_sampler = std::move(*staged);
    }
  }
  llm_context_ = std::move(context_data);

  // We can keep our kv cache buffers if this is the first step. This lets us
  // restore from LlmContexts at step 0 with an empty kv cache.
  if (llm_context_->runtime_state().current_step > 0) {
    auto restored_state = std::move(
        static_cast<LlmProcessedContext&>(llm_context_->processed_context())
            .state());
    ABSL_RETURN_IF_ERROR(RestoreState(std::move(restored_state)));
  }

  if (replacement_sampler.has_value()) {
    sampler_ = std::move(replacement_sampler->sampler);
    initialized_sampler_backend_ = replacement_sampler->backend;
    initialized_sampler_type_ = replacement_sampler->sampler_type;
    gpu_sampler_max_top_k_ = replacement_sampler->max_top_k;
    // Input-handling buffers are execution-history state. Algorithm switches
    // conservatively return to executor-managed decode inputs.
    sampler_handles_input_ = false;
    decode_prev_input_pos_ = TensorBuffer();
    decode_prev_mask_ = TensorBuffer();
    decode_prev_param_ = TensorBuffer();
  }

  force_prepare_needed_ = true;

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::UpdateRuntimeConfig(
    const RuntimeConfig& runtime_config) {
  std::optional<ContextSampler> replacement_sampler;
  if (sampler_ != nullptr) {
    absl::Status bind_status =
        BindSamplerToContext(runtime_config, llm_context_->runtime_state());
    if (!bind_status.ok()) {
      absl::StatusOr<ContextSampler> staged = CreateSamplerForContext(
          runtime_config, llm_context_->runtime_state());
      if (!staged.ok()) {
        BindSamplerToContext(llm_context_->runtime_config(),
                             llm_context_->runtime_state())
            .IgnoreError();
        return staged.status();
      }
      replacement_sampler = std::move(*staged);
    }
  }
  llm_context_->runtime_config() = runtime_config;
  if (replacement_sampler.has_value()) {
    sampler_ = std::move(replacement_sampler->sampler);
    initialized_sampler_backend_ = replacement_sampler->backend;
    initialized_sampler_type_ = replacement_sampler->sampler_type;
    gpu_sampler_max_top_k_ = replacement_sampler->max_top_k;
    sampler_handles_input_ = false;
    decode_prev_input_pos_ = TensorBuffer();
    decode_prev_mask_ = TensorBuffer();
    decode_prev_param_ = TensorBuffer();
    force_prepare_needed_ = true;
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::UpdateRuntimeState(
    const RuntimeState& runtime_state) {
  if (sampler_ != nullptr) {
    absl::Status bind_status =
        BindSamplerToContext(llm_context_->runtime_config(), runtime_state);
    if (!bind_status.ok()) {
      BindSamplerToContext(llm_context_->runtime_config(),
                           llm_context_->runtime_state())
          .IgnoreError();
      return bind_status;
    }
  }
  llm_context_->runtime_state() = runtime_state;
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::BindSamplerToContext(
    const RuntimeConfig& runtime_config, const RuntimeState& runtime_state) {
  if (sampler_ == nullptr) return absl::OkStatus();
  if (runtime_state.rand_gen == nullptr) {
    return absl::FailedPreconditionError(
        "Sampler context has no random engine.");
  }
  proto::SamplerParameters sampler_params;
  if (runtime_config.sampler_params.has_value()) {
    sampler_params = *runtime_config.sampler_params;
  }
  if (sampler_params.type() == proto::SamplerParameters::TYPE_UNSPECIFIED) {
    sampler_params.set_type(proto::SamplerParameters::TOP_P);
    sampler_params.set_k(1);
    sampler_params.set_p(0.0f);
    sampler_params.set_temperature(1.0f);
    sampler_params.set_seed(0);
  }
  if (!initialized_sampler_type_.has_value() ||
      *initialized_sampler_type_ != static_cast<int>(sampler_params.type())) {
    return absl::FailedPreconditionError(
        "Cannot change the sampling algorithm of a live sampler; "
        "sampling-type changes require a newly initialized executor.");
  }
  return sampler_->UpdateConfig(sampler_params,
                                runtime_config.output_heads.value_or(1),
                                runtime_state.rand_gen);
}

absl::StatusOr<LlmLiteRtCompiledModelExecutorBase::ContextSampler>
LlmLiteRtCompiledModelExecutorBase::CreateSamplerForContext(
    const RuntimeConfig& runtime_config,
    const RuntimeState& runtime_state) const {
  if (runtime_state.rand_gen == nullptr) {
    return absl::FailedPreconditionError(
        "Sampler context has no random engine.");
  }
  if (!runtime_config.output_heads.has_value() ||
      *runtime_config.output_heads <= 0) {
    return absl::FailedPreconditionError(
        "Sampler context has no positive output-head count.");
  }
  proto::SamplerParameters sampler_params;
  if (runtime_config.sampler_params.has_value()) {
    sampler_params = *runtime_config.sampler_params;
  }
  if (sampler_params.type() == proto::SamplerParameters::TYPE_UNSPECIFIED) {
    sampler_params.set_type(proto::SamplerParameters::TOP_P);
    sampler_params.set_k(1);
    sampler_params.set_p(0.0f);
    sampler_params.set_temperature(1.0f);
    sampler_params.set_seed(0);
  }
  const LlmExecutorSettings settings = [this]() {
    absl::MutexLock lock(executor_settings_mutex_);
    return executor_settings_;
  }();
  ABSL_ASSIGN_OR_RETURN(const Backend sampler_backend,
                        GetSamplerBackend(settings));
  ABSL_ASSIGN_OR_RETURN(const int vocabulary_size,
                        GetLoadedVocabularySizeForSessionHandoff());
  const int output_heads = *runtime_config.output_heads;
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Sampler> sampler,
      CreateSampler(sampler_backend, output_heads, sampler_params, env_,
                    /*sequence_size=*/1, vocabulary_size,
                    logits_data_type_));
  ABSL_RETURN_IF_ERROR(sampler->UpdateConfig(
      sampler_params, output_heads, runtime_state.rand_gen));
  return ContextSampler{
      .sampler = std::move(sampler),
      .backend = sampler_backend,
      .sampler_type = static_cast<int>(sampler_params.type()),
      .max_top_k = sampler_params.k(),
  };
}

absl::Status LlmLiteRtCompiledModelExecutorBase::InitializeSampler(
    std::optional<ActivationDataType> logits_data_type) {
  if (sampler_ != nullptr) {
    return absl::OkStatus();
  }

  // Use the provided activation data type if available, otherwise fallback to
  // the member variable.
  auto data_type = logits_data_type.value_or(logits_data_type_);

  ABSL_ASSIGN_OR_RETURN(auto vocab_size, GetVocabSize());
  LlmExecutorSettings settings = [this]() {
    absl::MutexLock lock(executor_settings_mutex_);
    return executor_settings_;
  }();
  ABSL_ASSIGN_OR_RETURN(auto sampler_backend, GetSamplerBackend(settings));
  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }
  proto::SamplerParameters sampler_params;
  if (llm_context_->runtime_config().sampler_params.has_value()) {
    sampler_params = llm_context_->runtime_config().sampler_params.value();
  }
  if (sampler_params.type() == proto::SamplerParameters::TYPE_UNSPECIFIED) {
    sampler_params.set_type(proto::SamplerParameters::TOP_P);
    sampler_params.set_k(1);
    sampler_params.set_p(0.0f);
    sampler_params.set_temperature(1.0f);
    sampler_params.set_seed(0);
  }
  const int sampler_type = static_cast<int>(sampler_params.type());

  gpu_sampler_max_top_k_ = sampler_params.k();

  ABSL_ASSIGN_OR_RETURN(
      sampler_,
      CreateSampler(sampler_backend, output_heads, std::move(sampler_params),
                    env_, /*sequence_size=*/1, vocab_size, data_type));
  initialized_sampler_backend_ = sampler_backend;
  initialized_sampler_type_ = sampler_type;
  absl::Status bind_status = BindSamplerToContext(
      llm_context_->runtime_config(), llm_context_->runtime_state());
  if (!bind_status.ok()) {
    sampler_.reset();
    initialized_sampler_backend_.reset();
    initialized_sampler_type_.reset();
    return bind_status;
  }

  // Disable GPU token copy for models that run embedding on the GPU.
  const bool runs_embedding_on_gpu = (embedding_lookup_ == nullptr);

  // If the sampler can handle input, prepare the input tensors for it.
  bool sampler_handles_input = true;
  {
    absl::MutexLock lock(executor_settings_mutex_);
    if (executor_settings_.GetAdvancedSettings().has_value()) {
      sampler_handles_input =
          executor_settings_.GetAdvancedSettings()->sampler_handles_input;
    }
  }
  sampler_handles_input_ =
      sampler_handles_input && sampler_->CanHandleInput() &&
      runs_embedding_on_gpu && !signatures_.input_tokens.empty() &&
      !signatures_.input_attn_mask_local.has_value();
  if (sampler_handles_input_) {
    ABSL_LOG(INFO) << "Sampler will handle decode input tensors.";
    if (!decode_prev_input_pos_) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_input_pos_,
          compiled_model_->CreateInputBuffer(kDecodeSignatureRunner,
                                             signatures_.input_positions));
    }
    if (!decode_prev_mask_ && signatures_.input_attn_mask.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_mask_,
          compiled_model_->CreateInputBuffer(kDecodeSignatureRunner,
                                             *signatures_.input_attn_mask));
    }
    if (!decode_prev_param_ && signatures_.input_int32_param.has_value()) {
      LITERT_ASSIGN_OR_RETURN(
          decode_prev_param_,
          compiled_model_->CreateInputBuffer(kDecodeSignatureRunner,
                                             *signatures_.input_int32_param));
    }
    // Set, then reset the input handling to get the underlying model ready, but
    // not to bind the input tensors.
    ABSL_RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/false));
    ABSL_RETURN_IF_ERROR(SetSamplerInputHandling(/*reset=*/true));
  }

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SwapSamplerInputTensors() {
  // Move the input_pos and mask to previous ones.
  std::swap(decode_prev_input_pos_,
            decode_input_buffers_[signatures_.input_positions]);
  if (signatures_.input_attn_mask.has_value()) {
    std::swap(decode_prev_mask_,
              decode_input_buffers_[*signatures_.input_attn_mask]);
  }
  if (signatures_.input_int32_param.has_value()) {
    std::swap(decode_prev_param_,
              decode_input_buffers_[*signatures_.input_int32_param]);
  }
  return SetSamplerInputHandling(/*reset=*/false);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SetSamplerInputHandling(
    bool reset) {
  if (reset) {
    return sampler_->SetInferenceFuncAndInputTensors(nullptr, nullptr, nullptr,
                                                     nullptr, nullptr, nullptr,
                                                     nullptr, nullptr, nullptr);
  }

  bool has_input_attn_mask = signatures_.input_attn_mask.has_value();
  bool has_input_int32_param = signatures_.input_int32_param.has_value();
  return sampler_->SetInferenceFuncAndInputTensors(
      BindTensorsAndRunDecodeStatic, this,
      &decode_input_buffers_[signatures_.input_tokens], &decode_prev_input_pos_,
      &decode_input_buffers_[signatures_.input_positions],
      has_input_attn_mask ? &decode_prev_mask_ : nullptr,
      has_input_attn_mask ? &decode_input_buffers_[*signatures_.input_attn_mask]
                          : nullptr,
      has_input_int32_param ? &decode_prev_param_ : nullptr,
      has_input_int32_param
          ? &decode_input_buffers_[*signatures_.input_int32_param]
          : nullptr);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SampleLogits(
    const TensorBuffer& logits, TensorBuffer& ids_tensor) {
  if (sampler_ == nullptr) {
    LITERT_ASSIGN_OR_RETURN(auto logits_tensor_type, logits.TensorType());
    ActivationDataType logits_data_type;
    if (logits_tensor_type.ElementType() == ElementType::Float16) {
      logits_data_type = ActivationDataType::FLOAT16;
    } else if (logits_tensor_type.ElementType() == ElementType::Float32) {
      logits_data_type = ActivationDataType::FLOAT32;
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported logits data type for sampler: ",
                       static_cast<int>(logits_tensor_type.ElementType())));
    }

    ABSL_RETURN_IF_ERROR(InitializeSampler(logits_data_type));
  }

  if (sampler_handles_input_) {
    ABSL_RETURN_IF_ERROR(SwapSamplerInputTensors());
  }

  ABSL_RETURN_IF_ERROR(sampler_->SampleToIdAndScoreBuffer(
      logits, ids_tensor, /*scores_tensor=*/nullptr));
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::UpdateExecutorSettings(
    const LlmExecutorSettings& executor_settings) {
  absl::MutexLock lock(executor_settings_mutex_);
  executor_settings_ = executor_settings;
  return absl::OkStatus();
}

litert::Options LlmLiteRtCompiledModelExecutorBase::GetRunOptions() const {
  absl::MutexLock lock(executor_settings_mutex_);
  litert::Options run_options;
  if (executor_settings_.GetAdvancedSettings().has_value()) {
#if defined(__APPLE__)
    const auto& advanced_settings = *executor_settings_.GetAdvancedSettings();
    auto gpu_options = run_options.GetGpuOptions();
    if (gpu_options.HasValue()) {
      (void)gpu_options->EnableMetalResidencySet(
          advanced_settings.gpu_enable_metal_residency_set);
    }
#endif
  }
  return run_options;
}

absl::Status LlmLiteRtCompiledModelExecutorBase::SetCurrentStep(int new_step) {
  ABSL_ASSIGN_OR_RETURN(auto old_step, GetCurrentStep());
  if (old_step == new_step) {
    return absl::OkStatus();
  }

  int max_step = old_step;
  ABSL_ASSIGN_OR_RETURN(auto processed_tokens, GetProcessedTokens());
  max_step = processed_tokens->TokenCount();
  RET_CHECK_LE(new_step, max_step).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be greater than the max step: " << max_step;
  RET_CHECK_GE(new_step, 0).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be negative.";
  if (new_step == max_step) {
    llm_context_->runtime_state().current_step = new_step;
    return absl::OkStatus();
  }
  RET_CHECK_LE(new_step, max_step).SetCode(absl::StatusCode::kInvalidArgument)
      << "New step cannot be greater than the max step: " << max_step;
  if (new_step < 0) {
    // Current step is negative after rolling back. This can only happen when
    // the user wants to set the step to 0 while there is a pending input token.
    // Thus we can roll back executor state to step 0.
    return Reset();
  }
  llm_context_->runtime_state().current_step = new_step;

  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::Reset() {
  llm_context_->runtime_state().current_step = 0;
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::
    ValidateDeterministicProjectionSupport() const {
  if (compiled_model_ == nullptr) {
    return absl::FailedPreconditionError(
        "Deterministic projection reset has no loaded compiled LiteRT "
        "model.");
  }
  if (compiled_backend_ != Backend::CPU && compiled_backend_ != Backend::GPU) {
    return absl::UnimplementedError(
        "Deterministic projection reset is implemented only for LiteRT CPU "
        "and GPU compiled-model executors.");
  }
  const LlmExecutorSettings settings = [this]() {
    absl::MutexLock lock(executor_settings_mutex_);
    return executor_settings_;
  }();
  if (settings.GetBackend() != compiled_backend_) {
    return absl::FailedPreconditionError(
        "Mutable executor settings no longer match the compiled backend.");
  }
  ABSL_ASSIGN_OR_RETURN(const Backend configured_sampler_backend,
                        GetSamplerBackend(settings));
  if (configured_sampler_backend != Backend::CPU) {
    return absl::UnimplementedError(
        "Deterministic projection requires CPU-side stable GREEDY selection "
        "for LiteRT logits.");
  }
  if (mtp_drafter_ != nullptr ||
      (settings.GetAdvancedSettings().has_value() &&
       settings.GetAdvancedSettings()->enable_speculative_decoding)) {
    return absl::UnimplementedError(
        "Deterministic projection does not support speculative or MTP "
        "decoder state.");
  }
  if (HasGraphRunCallbacks()) {
    return absl::UnimplementedError(
        "Deterministic projection does not support graph callback state or "
        "side effects.");
  }
  ABSL_RETURN_IF_ERROR(ValidateLoadedModelStatefulness(model_));

  const auto* state = dynamic_cast<const LitertState*>(state_.get());
  if (state == nullptr) {
    return absl::UnimplementedError(
        "Deterministic projection requires an inventoried LitertState "
        "allocation.");
  }
  ABSL_RETURN_IF_ERROR(
      state->ValidateDeterministicProjectionResetSupport());
  ABSL_RETURN_IF_ERROR(ValidateAuthoritativeStateMetadata(
      executor_metadata_, *state, "primary LiteRT state"));
  const auto* decode_state =
      dynamic_cast<const LitertState*>(decode_state_.get());
  if (decode_state_ != nullptr && decode_state == nullptr) {
    return absl::UnimplementedError(
        "Deterministic projection cannot inventory a non-LiteRT decode "
        "state.");
  }
  if (decode_state != nullptr) {
    ABSL_RETURN_IF_ERROR(
        decode_state->ValidateDeterministicProjectionResetSupport());
    ABSL_RETURN_IF_ERROR(ValidateAuthoritativeStateMetadata(
        executor_metadata_, *decode_state, "LiteRT decode state"));
  }
  return absl::OkStatus();
}

absl::Status
LlmLiteRtCompiledModelExecutorBase::ResetForDeterministicProjection() {
  ABSL_RETURN_IF_ERROR(ValidateDeterministicProjectionSupport());
  if (llm_context_ == nullptr) {
    return absl::FailedPreconditionError(
        "Deterministic projection reset requires an active LiteRT context.");
  }
  if (llm_context_->processed_context().lora_id().has_value()) {
    return absl::UnimplementedError(
        "Deterministic projection reset does not support LoRA state.");
  }

  const RuntimeConfig runtime_config = llm_context_->runtime_config();
  if (!runtime_config.output_heads.has_value() ||
      *runtime_config.output_heads != 1) {
    return absl::UnimplementedError(
        "Deterministic projection reset requires exactly one output "
        "candidate.");
  }
  if (!runtime_config.tokens_per_decode.has_value() ||
      *runtime_config.tokens_per_decode != 1) {
    return absl::UnimplementedError(
        "Deterministic projection reset requires one token per decode "
        "operation.");
  }
  if (!runtime_config.sampler_params.has_value() ||
      runtime_config.sampler_params->type() !=
          proto::SamplerParameters::GREEDY ||
      runtime_config.sampler_params->backend() !=
          proto::SamplerParameters::CPU) {
    return absl::UnimplementedError(
        "Deterministic projection reset requires the explicit CPU GREEDY "
        "sampler contract.");
  }

  // Stage every fallible replacement before touching the live continuation.
  // CPU state is copied and cleared so dynamic capacity is preserved. Metal
  // and other GPU state must be allocated through the compiled model; a host
  // deep copy would silently change its backend storage contract.
  const bool recreate_backend_native_state = compiled_backend_ == Backend::GPU;
  std::string prefill_signature;
  if (recreate_backend_native_state) {
    ABSL_ASSIGN_OR_RETURN(prefill_signature, GetPrefillSignatureKey());
  }
  auto make_fresh_state =
      [this, recreate_backend_native_state](
          const StateInterface* current,
          absl::string_view signature)
      -> absl::StatusOr<std::unique_ptr<StateInterface>> {
    if (current == nullptr) return std::unique_ptr<StateInterface>();
    if (!recreate_backend_native_state) {
      ABSL_ASSIGN_OR_RETURN(std::unique_ptr<StateInterface> staged,
                            current->DeepCopy());
      ABSL_RETURN_IF_ERROR(staged->Clear());
      return staged;
    }
    const auto* litert_state = dynamic_cast<const LitertState*>(current);
    if (litert_state == nullptr) {
      return absl::FailedPreconditionError(
          "GPU deterministic projection reset requires native LitertState "
          "storage.");
    }
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<LitertState> staged,
        LitertState::Create(env_, *compiled_model_, signature,
                            executor_metadata_,
                            litert_state->allocation_policy(),
                            litert_state->GetBatchSize(),
                            /*clear_kv_cache_before_prefill=*/true));
    ABSL_RETURN_IF_ERROR(
        staged->ValidateDeterministicProjectionResetSupport());
    return std::unique_ptr<StateInterface>(std::move(staged));
  };

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<StateInterface> staged_state,
      make_fresh_state(state_.get(), prefill_signature));
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<StateInterface> staged_decode_state,
      make_fresh_state(decode_state_.get(), kDecodeSignatureRunner));
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<LlmContext> staged_context,
      CreateNewContext(/*lora_id=*/std::nullopt, runtime_config));
  if (staged_context->runtime_state().rand_gen == nullptr) {
    return absl::FailedPreconditionError(
        "Fresh deterministic projection context has no random engine.");
  }

  proto::SamplerParameters sampler_params = *runtime_config.sampler_params;
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Sampler> staged_sampler,
      CreateSampler(Backend::CPU, /*batch_size=*/1, sampler_params));
  ABSL_RETURN_IF_ERROR(staged_sampler->UpdateConfig(
      sampler_params, /*batch_size=*/1,
      staged_context->runtime_state().rand_gen));

  // Ownership moves and scalar assignments below are the reset commit point.
  // No live state is changed unless all state, context, and sampler staging
  // succeeded.
  state_ = std::move(staged_state);
  decode_state_ = std::move(staged_decode_state);
  llm_context_ = std::move(staged_context);
  sampler_ = std::move(staged_sampler);
  initialized_sampler_backend_ = Backend::CPU;
  initialized_sampler_type_ =
      static_cast<int>(proto::SamplerParameters::GREEDY);
  gpu_sampler_max_top_k_ = 0;
  sampler_handles_input_ = false;
  decode_prev_input_pos_ = TensorBuffer();
  decode_prev_mask_ = TensorBuffer();
  decode_prev_param_ = TensorBuffer();
  force_prepare_needed_ = true;
  return absl::OkStatus();
}

absl::Status
LlmLiteRtCompiledModelExecutorBase::ValidateSessionHandoffSupport() const {
  if (compiled_model_ == nullptr) {
    return absl::FailedPreconditionError(
        "Session handoff has no loaded compiled LiteRT model.");
  }
  if (compiled_backend_ != Backend::CPU) {
    return absl::UnimplementedError(
        "Session handoff currently admits only a compiled LiteRT CPU "
        "backend.");
  }
  if (!session_handoff_compile_caches_disabled_) {
    return absl::UnimplementedError(
        "Session handoff requires compilation caches to have been disabled "
        "before the model was compiled.");
  }
  LlmExecutorSettings settings = [this]() {
    absl::MutexLock lock(executor_settings_mutex_);
    return executor_settings_;
  }();
  if (settings.GetBackend() != compiled_backend_) {
    return absl::FailedPreconditionError(
        "Mutable executor settings no longer match the compiled backend.");
  }
  if (llm_context_ == nullptr) {
    return absl::FailedPreconditionError(
        "LiteRT session handoff has no active context.");
  }
  if (llm_context_->processed_context().lora_id().has_value()) {
    return absl::UnimplementedError(
        "Session handoff does not support LoRA state.");
  }
  if (mtp_drafter_ != nullptr) {
    return absl::UnimplementedError(
        "Session handoff does not support speculative/MTP decoder state.");
  }
  if (HasGraphRunCallbacks()) {
    return absl::UnimplementedError(
        "Session handoff does not support graph callback state.");
  }
  ABSL_RETURN_IF_ERROR(ValidateLoadedModelStatefulness(model_));

  const auto* state = dynamic_cast<const LitertState*>(state_.get());
  if (state == nullptr) {
    return absl::UnimplementedError(
        "Session handoff requires an inventoried LitertState allocation.");
  }
  ABSL_RETURN_IF_ERROR(state->ValidateSessionHandoffSupport());
  ABSL_RETURN_IF_ERROR(ValidateAuthoritativeStateMetadata(
      executor_metadata_, *state, "primary LiteRT state"));
  const auto* decode_state =
      dynamic_cast<const LitertState*>(decode_state_.get());
  if (decode_state_ != nullptr && decode_state == nullptr) {
    return absl::UnimplementedError(
        "Session handoff cannot inventory a non-LiteRT decode state.");
  }
  if (decode_state != nullptr) {
    ABSL_RETURN_IF_ERROR(decode_state->ValidateSessionHandoffSupport());
    ABSL_RETURN_IF_ERROR(ValidateAuthoritativeStateMetadata(
        executor_metadata_, *decode_state, "LiteRT decode state"));
  }

  const RuntimeState& runtime_state = llm_context_->runtime_state();
  if (runtime_state.current_step < 0 || runtime_state.rand_gen == nullptr) {
    return absl::FailedPreconditionError(
        "Session handoff runtime continuation state is incomplete.");
  }
  const ProcessedTokens& processed_tokens =
      llm_context_->processed_context().processed_tokens();
  ABSL_RETURN_IF_ERROR(processed_tokens.ValidateSessionHandoffSupport());
  if (processed_tokens.TokenCount() != runtime_state.current_step) {
    return absl::FailedPreconditionError(
        "Live session step does not match its processed-token history.");
  }
  ABSL_ASSIGN_OR_RETURN(const int vocabulary_size,
                        GetLoadedVocabularySizeForSessionHandoff());
  ABSL_RETURN_IF_ERROR(processed_tokens.ValidateTokenIds(vocabulary_size));
  if (processed_tokens.ProcessedTokenCount() > state->GetNumEntries()) {
    return absl::FailedPreconditionError(
        "Live LiteRT state capacity is smaller than its consumed token "
        "history.");
  }

  ABSL_ASSIGN_OR_RETURN(Backend configured_sampler_backend,
                        GetSamplerBackend(settings));
  const Backend sampler_backend = sampler_ == nullptr
                                      ? configured_sampler_backend
                                      : initialized_sampler_backend_.value_or(
                                            Backend::UNSPECIFIED);
  if (sampler_backend != Backend::CPU) {
    return absl::UnimplementedError(
        "Session handoff currently supports only the CPU sampler backend.");
  }
  const RuntimeConfig& runtime_config = llm_context_->runtime_config();
  if (!runtime_config.output_heads.has_value() ||
      *runtime_config.output_heads != 1) {
    return absl::UnimplementedError(
        "Session handoff currently supports exactly one output candidate.");
  }
  if (!runtime_config.tokens_per_decode.has_value() ||
      *runtime_config.tokens_per_decode != 1) {
    return absl::UnimplementedError(
        "Session handoff currently supports one token per decode step.");
  }
  if (!runtime_config.sampler_params.has_value() ||
      runtime_config.sampler_params->type() !=
          proto::SamplerParameters::GREEDY) {
    return absl::UnimplementedError(
        "Session handoff currently supports only GREEDY sampling.");
  }
  if (runtime_config.sampler_params->backend() !=
      proto::SamplerParameters::CPU) {
    return absl::UnimplementedError(
        "Session handoff requires an explicit CPU sampler profile.");
  }
  if (sampler_ != nullptr &&
      initialized_sampler_type_ !=
          static_cast<int>(proto::SamplerParameters::GREEDY)) {
    return absl::FailedPreconditionError(
        "Session handoff runtime is not backed by the GREEDY sampler.");
  }
  if (sampler_ != nullptr && sampler_handles_input_) {
    return absl::UnimplementedError(
        "Session handoff does not preserve sampler-managed decode inputs.");
  }
  const auto is_derived_decode_input = [this](absl::string_view name) {
    return name == signatures_.input_tokens ||
           name == signatures_.input_positions ||
           (signatures_.input_attn_mask.has_value() &&
            name == *signatures_.input_attn_mask) ||
           (signatures_.input_attn_mask_local.has_value() &&
            name == *signatures_.input_attn_mask_local) ||
           (signatures_.input_embeddings.has_value() &&
            name == *signatures_.input_embeddings) ||
           (signatures_.input_per_layer_embeddings.has_value() &&
            name == *signatures_.input_per_layer_embeddings) ||
           (signatures_.input_int32_param.has_value() &&
            name == *signatures_.input_int32_param);
  };
  for (const auto& [name, _] : decode_input_buffers_) {
    if (!is_derived_decode_input(name)) {
      return absl::UnimplementedError(absl::StrCat(
          "Session handoff cannot reconstruct decode input buffer: ", name));
    }
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::VisitSessionState(
    absl::FunctionRef<absl::Status(const StateInterface&)> visitor) const {
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffSupport());
  if (llm_context_ == nullptr) {
    return absl::FailedPreconditionError(
        "LiteRT session handoff has no active context.");
  }
  if (llm_context_->processed_context().lora_id().has_value()) {
    return absl::UnimplementedError(
        "Session handoff does not support LoRA state.");
  }
  const int output_heads =
      llm_context_->runtime_config().output_heads.value_or(1);
  const StateInterface* active_state =
      output_heads > 1 && llm_context_->runtime_state().ran_decode
          ? decode_state_.get()
          : state_.get();
  if (active_state == nullptr) {
    return absl::FailedPreconditionError(
        "LiteRT session handoff has no active executor state.");
  }
  return visitor(*active_state);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::ImportSessionStateFrom(
    const ExecutorSessionSnapshot& snapshot,
    const ByteSource& serialized_state) {
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffSupport());
  if (!snapshot.serialized_state.empty()) {
    return absl::InvalidArgumentError(
        "Streamed session handoff metadata must not duplicate state bytes.");
  }
  if (llm_context_ == nullptr) {
    return absl::FailedPreconditionError(
        "LiteRT session handoff has no active context.");
  }
  if (llm_context_->processed_context().lora_id().has_value()) {
    return absl::UnimplementedError(
        "Session handoff does not support LoRA state.");
  }
  RuntimeState& live_runtime = llm_context_->runtime_state();
  ProcessedTokens& live_tokens =
      llm_context_->processed_context().processed_tokens();
  if (live_runtime.current_step != 0 || live_runtime.ran_decode ||
      live_tokens.TokenCount() != 0) {
    return absl::FailedPreconditionError(
        "Session handoff import target executor context must be fresh.");
  }
  if (!RuntimeConfigsEqual(llm_context_->runtime_config(),
                           snapshot.runtime_config)) {
    return absl::FailedPreconditionError(
        "Session handoff target runtime configuration is incompatible.");
  }
  if (live_runtime.rand_gen == nullptr) {
    return absl::FailedPreconditionError(
        "Session handoff target random engine is unavailable.");
  }
  ABSL_ASSIGN_OR_RETURN(const int vocabulary_size,
                        GetLoadedVocabularySizeForSessionHandoff());
  ABSL_RETURN_IF_ERROR(ProcessedTokens::ValidateSnapshotTokenIds(
      snapshot.processed_tokens, vocabulary_size));
  if (snapshot.last_prefill_token_id < 0 ||
      snapshot.last_prefill_token_id >= vocabulary_size) {
    return absl::InvalidArgumentError(
        "Session handoff last prefill token is outside the loaded "
        "vocabulary.");
  }
  ABSL_ASSIGN_OR_RETURN(
      ProcessedTokens restored_tokens,
      ProcessedTokens::FromSnapshot(snapshot.processed_tokens));
  if (snapshot.current_step < 0 ||
      restored_tokens.TokenCount() != snapshot.current_step) {
    return absl::InvalidArgumentError(
        "Session handoff current step does not match token history.");
  }

  const int output_heads = *llm_context_->runtime_config().output_heads;
  if (snapshot.processed_tokens.processed_token_ids.size() != 1) {
    return absl::InvalidArgumentError(
        "Session handoff token candidates do not match runtime state.");
  }
  StateInterface* active_state =
      output_heads > 1 && snapshot.ran_decode ? decode_state_.get()
                                              : state_.get();
  if (active_state == nullptr) {
    return absl::FailedPreconditionError(
        "LiteRT session handoff target has no active executor state.");
  }
  if (serialized_state.Size() == 0) {
    if (snapshot.current_step != 0 || snapshot.ran_decode ||
        restored_tokens.TokenCount() != 0) {
      return absl::DataLossError(
          "Only a fresh session handoff may omit serialized state.");
    }
  } else {
    auto* litert_state = dynamic_cast<LitertState*>(active_state);
    if (litert_state == nullptr) {
      return absl::UnimplementedError(
          "Session handoff import requires a LitertState target.");
    }
    const size_t consumed_entries =
        snapshot.processed_tokens.processed_token_ids[0].size();
    if (consumed_entries >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      return absl::ResourceExhaustedError(
          "Session handoff consumed-token count exceeds the supported state "
          "range.");
    }
    ABSL_RETURN_IF_ERROR(litert_state->ValidateSnapshotHeaderForImport(
        serialized_state, static_cast<int>(consumed_entries)));
    // LoadFrom is transactional for live host and Metal buffers. All other
    // fallible work is complete before this state-import commit point.
    ABSL_RETURN_IF_ERROR(active_state->LoadFrom(
        serialized_state, /*target_is_disposable=*/false));
  }

  live_tokens = std::move(restored_tokens);
  live_runtime.current_step = snapshot.current_step;
  live_runtime.ran_decode = snapshot.ran_decode;
  *live_runtime.rand_gen = snapshot.random_engine;
  force_prepare_needed_ = true;
  return absl::OkStatus();
}

absl::StatusOr<ExactLiteRtLogitsFrameContract>
LlmLiteRtCompiledModelExecutorBase::GetExactLiteRtLogitsFrameContract() const {
  if (signatures_.output_logits.empty()) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT executor has no resolved decode logits signature.");
  }
  const auto logits = decode_output_buffers_.find(
      absl::string_view(signatures_.output_logits));
  if (logits == decode_output_buffers_.end()) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT executor has no decode logits allocation.");
  }
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          logits->second.TensorType());
  ExactLiteRtLogitsElementType element_type =
      ExactLiteRtLogitsElementType::kUnsupported;
  uint64_t element_byte_count = 0;
  switch (tensor_type.ElementType()) {
    case ElementType::Float16:
      element_type = ExactLiteRtLogitsElementType::kFloat16;
      element_byte_count = 2;
      break;
    case ElementType::Float32:
      element_type = ExactLiteRtLogitsElementType::kFloat32;
      element_byte_count = 4;
      break;
    default:
      return absl::UnimplementedError(
          "Exact logits capture requires loaded FP16 or FP32 logits.");
  }
  if (tensor_type.Layout().HasStrides()) {
    return absl::UnimplementedError(
        "Exact logits capture requires a packed logits allocation.");
  }
  const auto dimensions = tensor_type.Layout().Dimensions();
  if (dimensions.size() != 3 || dimensions[0] != 1 || dimensions[1] != 1 ||
      dimensions[2] <= 0) {
    return absl::UnimplementedError(
        "Exact logits capture requires loaded [1, 1, vocabulary] logits.");
  }
  LITERT_ASSIGN_OR_RETURN(const size_t element_count,
                          tensor_type.Layout().NumElements());
  if (element_count != static_cast<size_t>(dimensions[2])) {
    return absl::FailedPreconditionError(
        "Loaded logits element count does not match its vocabulary extent.");
  }
  const uint64_t vocabulary_size = static_cast<uint64_t>(dimensions[2]);
  if (vocabulary_size >
      std::numeric_limits<uint64_t>::max() / element_byte_count) {
    return absl::ResourceExhaustedError(
        "Loaded exact logits frame byte count overflows uint64.");
  }
  const uint64_t expected_byte_count =
      vocabulary_size * element_byte_count;
  LITERT_ASSIGN_OR_RETURN(const size_t packed_byte_count,
                          logits->second.PackedSize());
  if (packed_byte_count != expected_byte_count) {
    return absl::FailedPreconditionError(
        "Loaded logits packed bytes do not cover the complete vocabulary "
        "frame.");
  }
  return ExactLiteRtLogitsFrameContract{
      .element_type = element_type,
      .batch_size = 1,
      .sequence_size = 1,
      .vocabulary_size = static_cast<uint32_t>(dimensions[2]),
      .byte_count = expected_byte_count,
  };
}

absl::StatusOr<SessionHandoffRuntimeProfile>
LlmLiteRtCompiledModelExecutorBase::GetSessionHandoffRuntimeProfile() const {
  if (compiled_model_ == nullptr || state_ == nullptr) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT executor is missing compiled model or state evidence.");
  }
  if (mtp_drafter_ != nullptr) {
    return absl::UnimplementedError(
        "Loaded runtime identity does not support speculative/MTP state.");
  }
  if (HasGraphRunCallbacks()) {
    return absl::UnimplementedError(
        "Loaded runtime identity does not support graph callbacks.");
  }
  ABSL_RETURN_IF_ERROR(ValidateLoadedModelStatefulness(model_));

  LlmExecutorSettings settings = [this]() {
    absl::MutexLock lock(executor_settings_mutex_);
    return executor_settings_;
  }();
  if (settings.GetBackend() != compiled_backend_) {
    return absl::FailedPreconditionError(
        "Loaded executor settings do not match the compiled backend.");
  }
  ABSL_ASSIGN_OR_RETURN(const Backend sampler_backend,
                        GetSamplerBackend(settings));
  if (sampler_backend != Backend::CPU) {
    return absl::UnimplementedError(
        "Loaded runtime identity currently requires the CPU sampler.");
  }
  if (compiled_backend_ != Backend::CPU && compiled_backend_ != Backend::GPU) {
    return absl::UnimplementedError(
        "Exact loaded runtime identity supports only concrete LiteRT CPU or "
        "Metal executors.");
  }
  const LlmExecutorSettings& compiled_settings = compiled_executor_settings_;
  if (compiled_settings.GetBackend() != compiled_backend_) {
    return absl::FailedPreconditionError(
        "Immutable compilation settings disagree with the compiled backend.");
  }
  if (!compiled_settings.GetAdvancedSettings().has_value()) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT executor has no resolved advanced settings.");
  }
  const bool caches_disabled_by_sentinel =
      compiled_settings.GetCacheDir() == ":nocache";
  if (!session_handoff_compile_caches_disabled_ ||
      compiled_settings.GetScopedCacheFile() != nullptr ||
      compiled_settings.GetScopedProgramCacheFile() != nullptr ||
      (!caches_disabled_by_sentinel &&
       (!compiled_settings.IsWeightCacheDisabled() ||
        !compiled_settings.IsProgramCacheDisabled()))) {
    return absl::UnimplementedError(
        "Loaded runtime identity requires weight and program caches to be "
        "disabled before model compilation; exact cache artifact evidence is "
        "not available.");
  }

  // Validate the actual compiled decode output that the internal GREEDY
  // sampler will consume. Session metadata cannot establish this contract:
  // SampleLogits() receives a duplicate of this loaded buffer, and
  // GreedyCpuSampler accepts only one packed FP16/FP32 logits row.
  if (signatures_.output_logits.empty()) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT executor has no resolved decode logits signature.");
  }
  const auto logits_buffer_it = decode_output_buffers_.find(
      absl::string_view(signatures_.output_logits));
  if (logits_buffer_it == decode_output_buffers_.end()) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT executor has no decode logits output buffer.");
  }
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType logits_tensor_type,
                          logits_buffer_it->second.TensorType());
  const ElementType logits_element_type = logits_tensor_type.ElementType();
  SessionHandoffLogitsElementType runtime_logits_element_type =
      SessionHandoffLogitsElementType::kUnsupported;
  uint64_t logits_element_byte_count = 0;
  switch (logits_element_type) {
    case ElementType::Float16:
      runtime_logits_element_type =
          SessionHandoffLogitsElementType::kFloat16;
      logits_element_byte_count = 2;
      break;
    case ElementType::Float32:
      runtime_logits_element_type =
          SessionHandoffLogitsElementType::kFloat32;
      logits_element_byte_count = 4;
      break;
    default:
      return absl::UnimplementedError(
          "Session handoff GREEDY sampling requires FP16 or FP32 decode "
          "logits.");
  }
  const auto& logits_layout = logits_tensor_type.Layout();
  if (logits_layout.HasStrides()) {
    return absl::UnimplementedError(
        "Session handoff GREEDY sampling requires packed decode logits.");
  }
  const auto logits_dimensions = logits_layout.Dimensions();
  if (logits_dimensions.size() != 3) {
    return absl::UnimplementedError(
        "Session handoff GREEDY sampling requires decode logits shaped "
        "[batch, sequence, vocabulary].");
  }
  if (logits_dimensions[0] != 1) {
    return absl::UnimplementedError(
        "Session handoff GREEDY sampling requires exactly one loaded output "
        "head.");
  }
  if (logits_dimensions[1] != 1) {
    return absl::UnimplementedError(
        "Session handoff GREEDY sampling requires exactly one decode "
        "sequence position.");
  }
  const int logits_vocabulary_size = logits_dimensions[2];
  if (logits_vocabulary_size <= 0) {
    return absl::FailedPreconditionError(
        "Loaded decode logits have no positive vocabulary dimension.");
  }
  LITERT_ASSIGN_OR_RETURN(const size_t logits_element_count,
                          logits_layout.NumElements());
  if (logits_element_count !=
      static_cast<size_t>(logits_vocabulary_size)) {
    return absl::FailedPreconditionError(
        "Loaded decode logits element count is inconsistent with its "
        "single-row vocabulary shape.");
  }
  const uint64_t logits_vocabulary_size_u64 =
      static_cast<uint64_t>(logits_vocabulary_size);
  if (logits_vocabulary_size_u64 >
      std::numeric_limits<uint64_t>::max() / logits_element_byte_count) {
    return absl::ResourceExhaustedError(
        "Loaded decode logits frame byte count overflows uint64.");
  }
  const uint64_t logits_frame_byte_count =
      logits_vocabulary_size_u64 * logits_element_byte_count;
  LITERT_ASSIGN_OR_RETURN(const size_t packed_logits_byte_count,
                          logits_buffer_it->second.PackedSize());
  if (packed_logits_byte_count != logits_frame_byte_count) {
    return absl::FailedPreconditionError(
        "Loaded decode logits packed bytes do not exactly match the complete "
        "runtime-derived tensor frame.");
  }
  if (llm_context_ == nullptr ||
      !llm_context_->runtime_config().output_heads.has_value()) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT executor has no resolved output-head contract.");
  }
  if (*llm_context_->runtime_config().output_heads != 1) {
    return absl::UnimplementedError(
        "Session handoff GREEDY sampling requires one executor output head.");
  }
  const auto* state = dynamic_cast<const LitertState*>(state_.get());
  if (state == nullptr) {
    return absl::UnimplementedError(
        "Loaded runtime identity cannot measure a non-LiteRT state "
        "allocation.");
  }
  const auto* decode_state =
      dynamic_cast<const LitertState*>(decode_state_.get());
  if (decode_state_ != nullptr && decode_state == nullptr) {
    return absl::UnimplementedError(
        "Loaded runtime identity cannot measure the decode state "
        "allocation.");
  }
  if (compiled_backend_ == Backend::CPU) {
    ABSL_RETURN_IF_ERROR(state->ValidateSessionHandoffSupport());
  } else {
    ABSL_RETURN_IF_ERROR(
        state->ValidateDeterministicProjectionResetSupport());
    ABSL_RETURN_IF_ERROR(state->ValidateMetalStateStorageForExactProfile());
  }
  ABSL_RETURN_IF_ERROR(ValidateAuthoritativeStateMetadata(
      executor_metadata_, *state, "primary LiteRT state"));
  if (decode_state != nullptr) {
    if (compiled_backend_ == Backend::CPU) {
      ABSL_RETURN_IF_ERROR(decode_state->ValidateSessionHandoffSupport());
    } else {
      ABSL_RETURN_IF_ERROR(
          decode_state->ValidateDeterministicProjectionResetSupport());
      ABSL_RETURN_IF_ERROR(
          decode_state->ValidateMetalStateStorageForExactProfile());
    }
    ABSL_RETURN_IF_ERROR(ValidateAuthoritativeStateMetadata(
        executor_metadata_, *decode_state, "LiteRT decode state"));
  }

  uint8_t executor_shape = 0;
  if (dynamic_cast<const LlmLiteRtCompiledModelExecutorStatic*>(this) !=
      nullptr) {
    executor_shape = 1;
  } else if (dynamic_cast<const LlmLiteRtCompiledModelExecutorDynamic*>(this) !=
             nullptr) {
    executor_shape = 2;
  } else {
    return absl::UnimplementedError(
        "Loaded runtime identity cannot classify the concrete executor.");
  }

  const AdvancedSettings& advanced =
      *compiled_settings.GetAdvancedSettings();

  std::string profile;
  AppendRuntimeBytes("LITERT_LM_SESSION_RUNTIME_PROFILE_V3", &profile);
  // This subsection is derived from the compiled TensorBuffer, not from
  // caller-provided sampler or session labels. Keep it in the identity so a
  // different loaded logits contract cannot share an exact-handoff profile.
  AppendRuntimeBytes("GREEDY_DECODE_LOGITS_CONTRACT_V1", &profile);
  AppendRuntimeI32(static_cast<int32_t>(logits_element_type), &profile);
  AppendRuntimeBool(logits_layout.HasStrides(), &profile);
  AppendRuntimeU32(static_cast<uint32_t>(logits_dimensions.size()), &profile);
  for (int dimension : logits_dimensions) {
    AppendRuntimeI32(dimension, &profile);
  }
  AppendRuntimeU32(static_cast<uint32_t>(logits_element_count), &profile);
  AppendRuntimeU64(logits_frame_byte_count, &profile);
  AppendRuntimeU8(executor_shape, &profile);
  AppendRuntimeI32(static_cast<int32_t>(compiled_backend_), &profile);
  AppendRuntimeI32(static_cast<int32_t>(sampler_backend), &profile);
  AppendRuntimeBool(compiled_settings.GetActivationDataType().has_value(),
                    &profile);
  if (compiled_settings.GetActivationDataType().has_value()) {
    AppendRuntimeI32(
        static_cast<int32_t>(*compiled_settings.GetActivationDataType()),
        &profile);
  }
  AppendRuntimeBool(compiled_settings.IsMixedPrecisionEnabled(), &profile);
  AppendRuntimeU32(compiled_settings.GetMaxNumTokens(), &profile);
  AppendRuntimeU32(compiled_settings.GetMaxNumImages(), &profile);
  AppendRuntimeU32(compiled_settings.GetLoraRank(), &profile);
  AppendRuntimeI32(
      static_cast<int32_t>(
          compiled_settings.GetModelAssets().fake_weights_mode()),
      &profile);

  uint32_t runtime_cpu_thread_count = 0;
  int32_t runtime_prefill_chunk_size = 0;
  if (compiled_backend_ == Backend::CPU) {
    ABSL_ASSIGN_OR_RETURN(
        const CpuConfig cpu,
        compiled_settings.GetBackendConfig<CpuConfig>());
    if (cpu.number_of_threads == 0) {
      return absl::FailedPreconditionError(
          "Loaded CPU executor has no positive compiled thread count.");
    }
    runtime_cpu_thread_count = cpu.number_of_threads;
    AppendRuntimeBytes("CPU_COMPILATION_INPUTS_V2", &profile);
    AppendRuntimeU32(cpu.kv_increment_size, &profile);
    AppendRuntimeI32(cpu.prefill_chunk_size, &profile);
    AppendRuntimeU32(cpu.number_of_threads, &profile);
    AppendRuntimeBool(cpu.enable_ynnpack, &profile);
    // These are the exact fixed options applied by CreateCompilationOptions.
    // Their implementation and numeric XNNPACK defaults are also sealed by the
    // loaded executable/delegate image digest.
    AppendRuntimeBytes(
        "XNNPACK_DEFAULT_PLUS_DYNAMIC_FULLY_CONNECTED_PLUS_LATEST_OPERATORS_V1",
        &profile);
    AppendRuntimeBool(/*compress_quantization_zero_points=*/true, &profile);
    AppendRuntimeBool(/*hardware_accelerator_cpu_only=*/true, &profile);

    if (const auto* static_executor =
            dynamic_cast<const LlmLiteRtCompiledModelExecutorStatic*>(this)) {
      runtime_prefill_chunk_size = -1;
      ABSL_RETURN_IF_ERROR(AppendStaticPrefillScheduleContract(
          "CPU_STATIC_PREFILL_SIGNATURES_V1",
          "STATIC_PREFILL_LONGEST_FIT_GREEDY_V1", *compiled_model_,
          signatures_,
          static_executor->prefill_signature_map_for_exact_profile(),
          &profile));
    } else if (const auto* dynamic_executor =
                   dynamic_cast<const LlmLiteRtCompiledModelExecutorDynamic*>(
                       this)) {
      runtime_prefill_chunk_size =
          dynamic_executor->prefill_chunk_size_for_exact_profile();
      if (runtime_prefill_chunk_size != cpu.prefill_chunk_size ||
          dynamic_executor->kv_increment_size_for_exact_profile() !=
              cpu.kv_increment_size) {
        return absl::FailedPreconditionError(
            "Loaded dynamic CPU executor does not match its retained "
            "compilation-time chunking contract.");
      }
      AppendRuntimeBytes("CPU_DYNAMIC_PREFILL_CHUNKING_V1", &profile);
      AppendRuntimeI32(runtime_prefill_chunk_size, &profile);
      AppendRuntimeU32(
          dynamic_executor->kv_increment_size_for_exact_profile(), &profile);
    } else {
      return absl::UnimplementedError(
          "Loaded CPU prefill executor has no exact chunking contract.");
    }
    ABSL_RETURN_IF_ERROR(AppendFixedTensorMapContract(
        "CPU_DECODE_INPUTS_V1", decode_input_buffers_, &profile));
    ABSL_RETURN_IF_ERROR(AppendFixedTensorMapContract(
        "CPU_DECODE_OUTPUTS_V1", decode_output_buffers_, &profile));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        const GpuConfig gpu,
        compiled_settings.GetBackendConfig<GpuConfig>());
    AppendRuntimeBytes("GPU_COMPILATION_INPUTS_V1", &profile);
    AppendRuntimeU32(gpu.max_top_k, &profile);
    AppendRuntimeBool(gpu.external_tensor_mode, &profile);
    // These values are the immutable inputs from which
    // CreateCompilationOptions constructed the actual GpuOptions passed to
    // CompiledModel::Create. The measured delegate code image binds the
    // option-default implementation that interpreted them.
    AppendRuntimeI32(static_cast<int32_t>(
                         compiled_settings.GetActivationDataType().value_or(
                             ActivationDataType::FLOAT16)),
                     &profile);
    // The exact sampler is a serial CPU scan. Metal prefill is governed by the
    // ordered static signature schedule rather than CpuConfig chunking.
    runtime_cpu_thread_count = 1;
    runtime_prefill_chunk_size = -1;
  }

  AppendRuntimeU32(
      static_cast<uint32_t>(advanced.prefill_batch_sizes.size()), &profile);
  for (int batch_size : advanced.prefill_batch_sizes) {
    AppendRuntimeI32(batch_size, &profile);
  }
  AppendRuntimeI32(advanced.num_output_candidates, &profile);
  AppendRuntimeBool(advanced.configure_magic_numbers, &profile);
  AppendRuntimeBool(advanced.verify_magic_numbers, &profile);
  AppendRuntimeBool(advanced.clear_kv_cache_before_prefill, &profile);
  AppendRuntimeU32(advanced.num_logits_to_print_after_decode, &profile);
  AppendRuntimeBool(advanced.gpu_madvise_original_shared_tensors, &profile);
  AppendRuntimeBool(advanced.gpu_enable_metal_residency_set, &profile);
  AppendRuntimeBool(advanced.is_benchmark, &profile);
  AppendRuntimeBool(advanced.enable_profiling, &profile);
  AppendRuntimeBytes(advanced.preferred_device_substr, &profile);
  AppendRuntimeI32(advanced.num_threads_to_upload, &profile);
  AppendRuntimeI32(advanced.num_threads_to_compile, &profile);
  AppendRuntimeBool(advanced.convert_weights_on_gpu, &profile);
  AppendRuntimeBool(
      advanced.wait_for_weights_conversion_complete_in_benchmark, &profile);
  AppendRuntimeBool(advanced.optimize_shader_compilation, &profile);
  AppendRuntimeBool(advanced.cache_compiled_shaders_only, &profile);
  AppendRuntimeBool(advanced.share_constant_tensors, &profile);
  AppendRuntimeBool(advanced.sampler_handles_input, &profile);
  AppendRuntimeOptionalBool(advanced.allow_src_quantized_fc_conv_ops,
                            &profile);
  AppendRuntimeOptionalBool(advanced.hint_waiting_for_completion, &profile);
  AppendRuntimeOptionalBool(advanced.gpu_context_low_priority, &profile);
  AppendRuntimeBool(advanced.enable_speculative_decoding, &profile);
  AppendRuntimeBool(advanced.disable_delegate_clustering, &profile);
  AppendRuntimeOptionalInt(advanced.hint_kernel_batch_size, &profile);
  AppendRuntimeBool(advanced.error_on_invalid_sampled_token_id, &profile);

  AppendRuntimeU32(
      static_cast<uint32_t>(compiled_settings.GetSelectedSignatures().size()),
      &profile);
  for (const std::string& signature :
       compiled_settings.GetSelectedSignatures()) {
    AppendRuntimeBytes(signature, &profile);
  }
  AppendRuntimeBool(compiled_settings.IsWeightCacheDisabled(), &profile);
  AppendRuntimeBool(compiled_settings.IsProgramCacheDisabled(), &profile);
  AppendRuntimeBool(!compiled_settings.GetCacheDir().empty(), &profile);
  AppendRuntimeBool(compiled_settings.GetScopedCacheFile() != nullptr,
                    &profile);
  AppendRuntimeBool(compiled_settings.GetScopedProgramCacheFile() != nullptr,
                    &profile);
  AppendRuntimeBool(!compiled_settings.GetLitertDispatchLibDir().empty(),
                    &profile);
  AppendRuntimeBool(!weight_cache_path_.empty(), &profile);

  AppendRuntimeI32(static_cast<int32_t>(state->allocation_policy()), &profile);
  AppendRuntimeI32(state->GetNumEntries(), &profile);
  AppendRuntimeI32(state->GetBatchSize(), &profile);
  AppendRuntimeBool(decode_state != nullptr, &profile);
  if (decode_state != nullptr) {
    AppendRuntimeI32(
        static_cast<int32_t>(decode_state->allocation_policy()), &profile);
    AppendRuntimeI32(decode_state->GetNumEntries(), &profile);
    AppendRuntimeI32(decode_state->GetBatchSize(), &profile);
  }
  AppendRuntimeBool(use_fp16_precision_, &profile);
  AppendRuntimeI32(static_cast<int32_t>(logits_data_type_), &profile);
  AppendRuntimeBool(gpu_optimized_single_buffer_cache_, &profile);
  AppendRuntimeBool(embedding_lookup_ != nullptr, &profile);
  AppendRuntimeBool(per_layer_embedding_lookup_ != nullptr, &profile);
  AppendRuntimeBool(executor_metadata_ != nullptr, &profile);

  auto holder = env_.GetHolder();
  if (holder.runtime == nullptr ||
      holder.runtime->CompiledModelGetProfiler == nullptr) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT runtime proxy has no measurable code anchor.");
  }
  const auto runtime_function = holder.runtime->CompiledModelGetProfiler;
  static_assert(sizeof(runtime_function) == sizeof(uintptr_t));
  uintptr_t runtime_code_anchor = 0;
  std::memcpy(&runtime_code_anchor, &runtime_function,
              sizeof(runtime_code_anchor));
  if (runtime_code_anchor == 0) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT runtime code anchor is zero.");
  }

  if (compiled_backend_ == Backend::GPU) {
#if !defined(__APPLE__)
    return absl::UnimplementedError(
        "Concrete Metal runtime identity is available only on Apple "
        "platforms.");
#else
    if (executor_shape != 1) {
      return absl::UnimplementedError(
          "Exact Metal runtime identity requires the statically shaped "
          "LiteRT executor.");
    }
    if (!logits_buffer_it->second.IsMetalMemory()) {
      return absl::UnimplementedError(
          "Backend::GPU did not produce a Metal-backed live decode logits "
          "allocation.");
    }
    LITERT_ASSIGN_OR_RETURN(const bool fully_accelerated,
                            compiled_model_->IsFullyAccelerated());
    if (!fully_accelerated) {
      return absl::UnimplementedError(
          "Loaded Metal model is not fully accelerated; a CPU/delegate "
          "partition cannot satisfy the exact Metal profile.");
    }
    LITERT_ASSIGN_OR_RETURN(auto environment_options, env_.GetOptions());
    if (HasEnvironmentTag(environment_options,
                          EnvironmentOptions::Tag::kWebGpuDevice) ||
        HasEnvironmentTag(environment_options,
                          EnvironmentOptions::Tag::kWebGpuQueue) ||
        HasEnvironmentTag(environment_options,
                          EnvironmentOptions::Tag::kWebGpuInstance) ||
        HasEnvironmentTag(environment_options,
                          EnvironmentOptions::Tag::kOpenClDeviceId) ||
        HasEnvironmentTag(environment_options,
                          EnvironmentOptions::Tag::kOpenClCommandQueue) ||
        HasEnvironmentTag(environment_options,
                          EnvironmentOptions::Tag::kVulkanEnvironment)) {
      return absl::UnimplementedError(
          "Loaded GPU Environment exposes a non-Metal accelerator alongside "
          "Metal; the concrete selected delegate is ambiguous.");
    }
    ABSL_ASSIGN_OR_RETURN(
        const void* metal_device,
        GetUniqueEnvironmentPointer(environment_options,
                                    EnvironmentOptions::Tag::kMetalDevice,
                                    "MTLDevice"));
    ABSL_ASSIGN_OR_RETURN(
        const void* metal_command_queue,
        GetUniqueEnvironmentPointer(
            environment_options,
            EnvironmentOptions::Tag::kMetalCommandQueue,
            "MTLCommandQueue"));
    ABSL_ASSIGN_OR_RETURN(
        const void* accelerator_callback,
        GetUniqueEnvironmentPointer(
            environment_options,
            EnvironmentOptions::Tag::kCallbackOnGpuEnvDestroy,
            "selected accelerator destruction callback"));
    // The callback user data is owned by the selected accelerator. Requiring
    // it distinguishes the ML Drift Metal-owned Environment contract from a
    // caller that merely inserted device-shaped labels.
    ABSL_ASSIGN_OR_RETURN(
        const void* accelerator_user_data,
        GetUniqueEnvironmentPointer(
            environment_options,
            EnvironmentOptions::Tag::kCallbackUserDataOnGpuEnvDestroy,
            "selected accelerator callback state"));
    (void)accelerator_user_data;

    const auto* static_executor =
        dynamic_cast<const LlmLiteRtCompiledModelExecutorStatic*>(this);
    if (static_executor == nullptr) {
      return absl::UnimplementedError(
          "Exact Metal prefill scheduling requires the static executor.");
    }
    const SortedPrefillSignatureMap& prefill_schedule =
        static_executor->prefill_signature_map_for_exact_profile();
    std::string metal_policy;
    AppendRuntimeBytes("LITERT_LM_METAL_CORUN_POLICY_V1", &metal_policy);
    // This is queried from the compiled model after delegate application. It
    // is stronger than the fully-delegated compilation hint, although it still
    // does not identify the selected kernels or their reduction policy.
    AppendRuntimeBool(fully_accelerated, &metal_policy);
    uint32_t derived_corun_evidence = MetalCoRunEvidenceBit(
        MetalCoRunEvidence::kSelectedMetalDelegate);
    ABSL_RETURN_IF_ERROR(AppendStaticPrefillScheduleContract(
        "METAL_STATIC_PREFILL_SIGNATURES_V1",
        "STATIC_PREFILL_CAUTIOUS_GREEDY_V1", *compiled_model_,
        signatures_, prefill_schedule, &metal_policy));
    derived_corun_evidence |= MetalCoRunEvidenceBit(
        MetalCoRunEvidence::kFixedPrefillSchedule);

    ABSL_RETURN_IF_ERROR(AppendFixedTensorMapContract(
        "METAL_DECODE_INPUTS_V1", decode_input_buffers_, &metal_policy));
    ABSL_RETURN_IF_ERROR(AppendFixedTensorMapContract(
        "METAL_DECODE_OUTPUTS_V1", decode_output_buffers_, &metal_policy));
    AppendRuntimeI32(static_cast<int32_t>(state->allocation_policy()),
                     &metal_policy);
    AppendRuntimeI32(state->GetNumEntries(), &metal_policy);
    AppendRuntimeI32(state->GetBatchSize(), &metal_policy);
    AppendRuntimeBool(decode_state != nullptr, &metal_policy);
    if (decode_state != nullptr) {
      AppendRuntimeI32(
          static_cast<int32_t>(decode_state->allocation_policy()),
          &metal_policy);
      AppendRuntimeI32(decode_state->GetNumEntries(), &metal_policy);
      AppendRuntimeI32(decode_state->GetBatchSize(), &metal_policy);
    }
    derived_corun_evidence |=
        MetalCoRunEvidenceBit(MetalCoRunEvidence::kFixedShapeDecode);

    // LiteRT-LM-visible state has an authoritative reset inventory and reset
    // recreates its backend-native Metal buffers. Delegate-owned mutable state
    // is not enumerable, and GPU session handoff is still rejected by
    // ValidateSessionHandoffSupport, so the combined session-and-reset bit must
    // remain absent.
    ABSL_RETURN_IF_ERROR(ValidateDeterministicProjectionSupport());
    AppendRuntimeBool(/*litert_lm_visible_reset_inventory=*/true,
                      &metal_policy);
    AppendRuntimeBool(/*complete_delegate_reset_inventory=*/false,
                      &metal_policy);
    AppendRuntimeBool(/*complete_gpu_session_capsule=*/false, &metal_policy);

    // Pinned LiteRT/ML Drift exposes neither the selected attention kernel's
    // adaptive Split-KV policy nor an executor hook that enforces isolated,
    // quiescent fixed-shape decode. Record the missing bits, never intended
    // values. A future concrete hook must set them from the live delegate.
    AppendRuntimeU32(RequiredMetalCoRunEvidence(), &metal_policy);
    AppendRuntimeU32(derived_corun_evidence, &metal_policy);
    AppendRuntimeBytes(
        "MISSING_HOOK:ML_DRIFT_SELECTED_ATTENTION_SPLIT_KV_POLICY_V1",
        &metal_policy);
    AppendRuntimeBytes(
        "MISSING_HOOK:ML_DRIFT_EFFECTIVE_METAL_COMPILATION_FLAGS_V1",
        &metal_policy);
    AppendRuntimeBytes(
        "MISSING_HOOK:ML_DRIFT_SELECTED_METAL_PIPELINE_DIGEST_V1",
        &metal_policy);
    AppendRuntimeBytes(
        "MISSING_HOOK:LITERT_METAL_QUIESCENT_FIXED_DECODE_BOUNDARY_V1",
        &metal_policy);
    AppendRuntimeBytes(
        "MISSING_HOOK:LITERT_COMPLETE_GPU_SESSION_CAPSULE_INVENTORY_V1",
        &metal_policy);
    AppendRuntimeBytes(metal_policy, &profile);

    const uintptr_t selected_accelerator_code_anchor =
        reinterpret_cast<uintptr_t>(accelerator_callback);
    if (selected_accelerator_code_anchor == 0) {
      return absl::FailedPreconditionError(
          "Selected Metal accelerator callback code anchor is zero.");
    }
    return SessionHandoffRuntimeProfile{
        .runtime_class = SessionHandoffRuntimeClass::kLiteRtMetal,
        .canonical_profile = std::move(profile),
        .logits_element_type = runtime_logits_element_type,
        .logits_batch_size = logits_dimensions[0],
        .logits_sequence_size = logits_dimensions[1],
        .logits_vocabulary_size = logits_vocabulary_size,
        .logits_frame_byte_count = logits_frame_byte_count,
        .cpu_thread_count = runtime_cpu_thread_count,
        .prefill_chunk_size = runtime_prefill_chunk_size,
        .runtime_code_anchor = runtime_code_anchor,
        .metal_corun =
            MetalCoRunRuntimeEvidence{
                .metal_device = metal_device,
                .metal_command_queue = metal_command_queue,
                .selected_accelerator_code_anchor =
                    selected_accelerator_code_anchor,
                .derived_evidence = derived_corun_evidence,
                .canonical_policy = std::move(metal_policy),
            },
    };
#endif
  }

  return SessionHandoffRuntimeProfile{
      .runtime_class = SessionHandoffRuntimeClass::kLiteRtCpu,
      .canonical_profile = std::move(profile),
      .logits_element_type = runtime_logits_element_type,
      .logits_batch_size = logits_dimensions[0],
      .logits_sequence_size = logits_dimensions[1],
      .logits_vocabulary_size = logits_vocabulary_size,
      .logits_frame_byte_count = logits_frame_byte_count,
      .cpu_thread_count = runtime_cpu_thread_count,
      .prefill_chunk_size = runtime_prefill_chunk_size,
      .runtime_code_anchor = runtime_code_anchor,
  };
}

absl::StatusOr<int> LlmLiteRtCompiledModelExecutorBase::
    GetLoadedVocabularySizeForSessionHandoff() const {
  if (signatures_.output_logits.empty()) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT executor has no decode logits signature.");
  }
  const auto logits = decode_output_buffers_.find(signatures_.output_logits);
  if (logits == decode_output_buffers_.end()) {
    return absl::FailedPreconditionError(
        "Loaded LiteRT executor has no decode logits allocation.");
  }

  LITERT_ASSIGN_OR_RETURN(const RankedTensorType logits_tensor_type,
                          logits->second.TensorType());
  if (logits_tensor_type.ElementType() != ElementType::Float16 &&
      logits_tensor_type.ElementType() != ElementType::Float32) {
    return absl::UnimplementedError(
        "Session handoff requires FP16 or FP32 decode logits.");
  }
  if (logits_tensor_type.Layout().HasStrides()) {
    return absl::UnimplementedError(
        "Session handoff requires packed decode logits.");
  }
  const auto dimensions = logits_tensor_type.Layout().Dimensions();
  if (dimensions.size() != 3 || dimensions[0] != 1 || dimensions[1] != 1 ||
      dimensions[2] <= 0) {
    return absl::FailedPreconditionError(
        "Loaded decode logits must have shape [1, 1, vocabulary].");
  }
  return dimensions[2];
}

absl::StatusOr<int> LlmLiteRtCompiledModelExecutorBase::GetVocabSize() {
  const auto logits = decode_output_buffers_.find(signatures_.output_logits);
  if (logits == decode_output_buffers_.end()) {
    return absl::NotFoundError("Output logits info not found.");
  }
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType logits_tensor_type,
                          logits->second.TensorType());
  RET_CHECK_EQ(logits_tensor_type.Layout().Dimensions().size(), 3);
  return logits_tensor_type.Layout().Dimensions()[2];
}

absl::StatusOr<litert::Profiler>
LlmLiteRtCompiledModelExecutorBase::GetProfiler() const {
  if (compiled_model_ == nullptr) {
    return absl::FailedPreconditionError("Compiled model is null.");
  }
  auto holder = env_.GetHolder();
  if (holder.runtime == nullptr) {
    return absl::FailedPreconditionError(
        "LiteRT runtime proxy is null in environment.");
  }
  if (holder.handle == nullptr) {
    return absl::FailedPreconditionError("LiteRT environment handle is null.");
  }
  LiteRtProfiler profiler = nullptr;
  LITERT_RETURN_IF_ERROR(holder.runtime->CompiledModelGetProfiler(
      compiled_model_->Get(), &profiler));
  return litert::Profiler(profiler, litert::OwnHandle::kNo);
}

absl::Status LlmLiteRtCompiledModelExecutorBase::StartProfiling() {
  ABSL_ASSIGN_OR_RETURN(auto profiler, GetProfiler());
  LITERT_RETURN_IF_ERROR(profiler.StartProfiling());
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorBase::StopProfiling() {
  ABSL_ASSIGN_OR_RETURN(auto profiler, GetProfiler());
  LITERT_RETURN_IF_ERROR(profiler.StopProfiling());
  return absl::OkStatus();
}

absl::StatusOr<std::string>
LlmLiteRtCompiledModelExecutorBase::GetProfileSummary() {
  ABSL_ASSIGN_OR_RETURN(auto profiler, GetProfiler());
  LITERT_ASSIGN_OR_RETURN(auto summary,
                          profiler.GetProfileSummary(compiled_model_->Get()));
  return summary;
}

/* ===========================================================================*/
/* LlmLiteRtCompiledModelExecutorStatic */
/* ===========================================================================*/

absl::Status LlmLiteRtCompiledModelExecutorStatic::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {

  int output_heads = 1;
  if (llm_context_->runtime_config().output_heads.has_value()) {
    output_heads = llm_context_->runtime_config().output_heads.value();
  }

  // For now, we reduce the input and processed tokens for prefill only with
  // the first input and processed tokens. This should be updated if user select
  // the decode output candidate.
  constexpr int kTokenIndexToReduce = 0;
  LITERT_RETURN_IF_ERROR(PrepareFirstPrefillAfterDecode(kTokenIndexToReduce));

  LITERT_ASSIGN_OR_RETURN(auto token_ids_buffer, inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, token_ids_buffer->TensorType());
  // Accept batch size 1 or output_heads though prefill handles only the
  // first batch element.
  int32_t input_batch_size = tensor_type.Layout().Dimensions()[0];
  if (input_batch_size != 1) {
    RET_CHECK_EQ(input_batch_size, output_heads);
  }
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";

  if (embedding_lookup_ != nullptr) {
    ABSL_RETURN_IF_ERROR(embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  }

  LITERT_ASSIGN_OR_RETURN(auto ids,
                          ReferTensorBufferAsSpan<int32_t>(*token_ids_buffer));
  // Reduce the input ids only with one user selected.
  auto input_length = ids.size() / input_batch_size;
  ids = ids.subspan(kTokenIndexToReduce * input_length, input_length);
  int remaining_capacity =
      state_->GetNumEntries() - llm_context_->runtime_state().current_step;

  const bool is_cpu = [this]() {
    absl::MutexLock lock(executor_settings_mutex_);
    return executor_settings_.GetBackend() == Backend::CPU;
  }();
  ABSL_ASSIGN_OR_RETURN(auto work_groups, GetOptimizedPrefillWorkGroups(
                                              prefill_signature_map_,
                                              ids.size(), remaining_capacity,
                                              /*use_greedy_chunking=*/is_cpu));
  for (int i = 0; i < work_groups.size(); ++i) {
    const auto& prefill_signature = work_groups[i].first;
    int prefill_length = work_groups[i].second;
    // Keep track of the signatures that have already had their buffers
    // created only create them once.
    if (!prefill_input_buffers_.contains(prefill_signature)) {
      prefill_input_buffers_[prefill_signature] = {};
      ABSL_RETURN_IF_ERROR(CreatePrefillInputBuffers(
          prefill_signature, prefill_length, prefill_length,
          prefill_input_buffers_[prefill_signature]));
    }
    if (!prefill_output_buffers_.contains(prefill_signature)) {
      prefill_output_buffers_[prefill_signature] = {};
      ABSL_RETURN_IF_ERROR(CreatePrefillOutputBuffers(
          prefill_signature, prefill_length,
          prefill_output_buffers_[prefill_signature]));
    }

    // TODO: b/494284915 - Switch to use async prefill for Metal backend.
    if (!do_prefill_sync_.has_value()) {
      do_prefill_sync_ = std::any_of(
          prefill_input_buffers_[prefill_signature].begin(),
          prefill_input_buffers_[prefill_signature].end(),
          [](const auto& pair) { return pair.second.IsMetalMemory(); });
    }
    bool async = !*do_prefill_sync_ &&
                 (i < work_groups.size() - 1 || !params.GetWaitForCompletion());
    ABSL_RETURN_IF_ERROR(PrefillInternal(
        prefill_signature, prefill_input_buffers_[prefill_signature],
        prefill_output_buffers_[prefill_signature],
        ids.subspan(/*pos=*/0, prefill_length), async));
    ids = ids.subspan(/*pos=*/prefill_length);
  }
  RET_CHECK_EQ(ids.size(), 0).SetCode(absl::StatusCode::kInternal)
      << "Work groups not covering the entire prefill input.";

  if (embedding_lookup_ != nullptr) {
    ABSL_RETURN_IF_ERROR(embedding_lookup_->CleanupMultiModalEmbeddings());
  }

  return absl::OkStatus();
}

// static
// Creates a LlmLiteRtCompiledModelExecutorStatic from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorStatic>>
LlmLiteRtCompiledModelExecutorStatic::Create(
    LlmExecutorSettings executor_settings, Environment& lrt_env,
    ModelResources& resources) {
  ABSL_ASSIGN_OR_RETURN(
      auto litert_model,
      resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));
  std::string cache_path = executor_settings.GetCacheDir();
  auto activation_data_type = ActivationDataType::FLOAT16;
  // TODO: b/433590109 - Some GPUs do not support FP16, so we need to check the
  // capabilities of the GPU and set the activation data type accordingly.
  if (executor_settings.GetActivationDataType().has_value()) {
    activation_data_type = executor_settings.GetActivationDataType().value();
  }
  const Backend backend = executor_settings.GetBackend();
  bool use_generic_npu_compiler_plugin = false;
  if (backend == Backend::NPU) {
    auto npu_config = executor_settings.GetBackendConfig<NpuConfig>();
    use_generic_npu_compiler_plugin =
        npu_config.ok() && npu_config->use_generic_litert_compiler_plugin;
  }
  bool use_fp16_precision =
      activation_data_type == ActivationDataType::FLOAT16 &&
      backend == Backend::GPU;

  if (!litert_model || !*litert_model) {
    return absl::InternalError("Failed to build LiteRt model");
  }

  const proto::ExecutorMetadata* executor_metadata = nullptr;
  auto executor_metadata_or = resources.GetExecutorMetadata();
  if (executor_metadata_or.ok()) {
    executor_metadata = *executor_metadata_or;
  }

  absl::string_view prefill_signature_key = "";
  for (int i = 0; i < litert_model->GetNumSignatures(); ++i) {
    LITERT_ASSIGN_OR_RETURN(auto sig, litert_model->GetSignature(i));
    absl::string_view key = sig.Key();
    if (absl::StartsWith(key, kPrefillSignatureRunner)) {
      prefill_signature_key = key;
      break;
    }
  }

  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  ABSL_ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  LITERT_ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(executor_settings, activation_data_type,
                               &signatures));

  ABSL_RETURN_IF_ERROR(SetExternalWeightOptions(
      resources, ModelType::kTfLitePrefillDecode, compilation_options));

  std::unique_ptr<CompiledModel> compiled_model;
  {
    LITERT_ASSIGN_OR_RETURN(auto compiled_model_tmp,
                            CompiledModel::Create(lrt_env, litert_model->Get(),
                                                  compilation_options));
    compiled_model =
        std::make_unique<CompiledModel>(std::move(compiled_model_tmp));
  }

  ABSL_ASSIGN_OR_RETURN(
      auto prefill_runner_set,
      GetPrefillRunnerSetFromModel(
          *litert_model, kPrefillSignatureRunner,
          /*input_positions_name=*/signatures.input_positions));
  RET_CHECK(!prefill_runner_set.empty()) << "No prefill runner available.";

  LitertState::AllocationPolicy allocation_policy =
      LitertState::AllocationPolicy::kInplace;
  if (backend == Backend::GPU) {
    if (signatures.input_int32_param.has_value()) {
      allocation_policy = LitertState::AllocationPolicy::kGpuOptimizedInplace;
    } else {
      allocation_policy = LitertState::AllocationPolicy::kPingPong;
    }
  }

  bool clear_kv_cache_before_prefill =
      !executor_settings.GetAdvancedSettings() ||
      executor_settings.GetAdvancedSettings()->clear_kv_cache_before_prefill;

  LITERT_ASSIGN_OR_RETURN(
      auto state,
      LitertState::Create(lrt_env, *compiled_model, prefill_signature_key,
                          executor_metadata, allocation_policy,
                          /*batch_size=*/1, clear_kv_cache_before_prefill));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;
  for (auto input_name : decode_signature.InputNames()) {
    if (IsLoRAInputName(input_name)) {
      // We let LoraManager handle LoRA inputs.
      continue;
    }
    if (state->Contains(input_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(
        auto input_buffer,
        compiled_model->CreateInputBuffer(kDecodeSignatureRunner, input_name));
    decode_input_buffers[input_name] = std::move(input_buffer);
  }
  LITERT_ASSIGN_OR_RETURN(
      size_t decode_signature_index,
      compiled_model->GetSignatureIndex(kDecodeSignatureRunner));
  for (size_t i = 0; i < decode_signature.OutputNames().size(); ++i) {
    auto output_name = decode_signature.OutputNames()[i];
    if (state->Contains(output_name)) {
      continue;
    }
    // If we are using the GPU sampler and the model is compiled with FP16
    // precision, we force the output logits to be FP16 as the
    // GPU sampler supports FP16 inputs.
    // If we use CPU sampler or the model is executed with FP32 / mixed
    // precision, we will keep the logits in FP32
    auto sampler_backend = GetSamplerBackend(executor_settings);

    if (output_name == signatures.output_logits && use_fp16_precision &&
        sampler_backend.ok() && *sampler_backend == Backend::GPU) {
      LITERT_ASSIGN_OR_RETURN(
          size_t signature_index,
          compiled_model->GetSignatureIndex(kDecodeSignatureRunner));
      LITERT_ASSIGN_OR_RETURN(
          auto output_buffer,
          CreateFP16OutputBuffer(lrt_env, *compiled_model, signature_index,
                                 output_name, i));
      decode_output_buffers[output_name] = std::move(output_buffer);
    } else {
      auto output_buffer_or = compiled_model->CreateOutputBuffer(
          kDecodeSignatureRunner, output_name);
      if (output_buffer_or) {
        decode_output_buffers[output_name] = std::move(*output_buffer_or);
        continue;
      }
      if (!use_generic_npu_compiler_plugin) {
        LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                                std::move(output_buffer_or));
        decode_output_buffers[output_name] = std::move(output_buffer);
        continue;
      }
      ABSL_LOG(WARNING) << "Falling back to host memory for NPU decode output '"
                        << output_name
                        << "' after compiled-model output buffer allocation "
                        << "failed: " << output_buffer_or.Error().Message();
      LITERT_ASSIGN_OR_RETURN(auto output_tensor_type,
                              decode_signature.OutputTensorType(i));
      ABSL_ASSIGN_OR_RETURN(
          auto output_buffer,
          CreateHostOutputBuffer(lrt_env, *compiled_model,
                                 decode_signature_index, i,
                                 std::move(output_tensor_type)));
      decode_output_buffers[output_name] = std::move(output_buffer);
    }
  }

  LITERT_ASSIGN_OR_RETURN(
      auto output_logits_buffer,
      decode_output_buffers[signatures.output_logits].Duplicate());
  LITERT_ASSIGN_OR_RETURN(auto output_logits_buffer_tensor_type,
                          output_logits_buffer.TensorType());
  RET_CHECK(output_logits_buffer_tensor_type.Layout().Dimensions().size() == 3)
      << "Output logits must be (batch, seq, vocab)";
  int batch_size = output_logits_buffer_tensor_type.Layout().Dimensions()[0];

  std::unique_ptr<LitertState> decode_state;
  if (batch_size > 1) {
    ABSL_VLOG(1) << "Decode batch size is larger than 1. Allocate decode "
                 << "only KV cache buffers.";
    LITERT_ASSIGN_OR_RETURN(
        decode_state,
        LitertState::Create(lrt_env, *compiled_model, kDecodeSignatureRunner,
                            executor_metadata, allocation_policy, batch_size,
                            clear_kv_cache_before_prefill));
  }

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  ABSL_RETURN_IF_ERROR(InitializeEmbeddingLookups(
      lrt_env, resources, embedding_lookup, per_layer_embedding_lookup));
  std::unique_ptr<LlmLiteRtMtpDrafter> mtp_drafter;
  {
    const auto& advanced_settings = executor_settings.GetAdvancedSettings();
    if (advanced_settings.has_value() &&
        advanced_settings->enable_speculative_decoding) {
      RET_CHECK_EQ(batch_size, 1)
          << "Speculative decoding (MTP) only supports a single output head.";
      RET_CHECK_NE(embedding_lookup, nullptr);
      std::optional<std::reference_wrapper<EmbeddingLookupManager>>
          ple_manager_opt;
      if (per_layer_embedding_lookup) {
        ple_manager_opt = std::ref(*per_layer_embedding_lookup);
      }
      ABSL_ASSIGN_OR_RETURN(
          mtp_drafter,
          LlmLiteRtMtpDrafter::Create(lrt_env, resources, executor_settings,
                                      *compiled_model, *embedding_lookup,
                                      ple_manager_opt));
    }
  }

  bool enable_profiling =
      executor_settings.GetAdvancedSettings() &&
      executor_settings.GetAdvancedSettings()->enable_profiling;
  auto executor = absl::WrapUnique(new LlmLiteRtCompiledModelExecutorStatic(
      std::move(executor_settings), lrt_env, litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(state),
      std::move(decode_state), std::move(prefill_runner_set), signatures,
      batch_size, std::move(cache_path), std::move(embedding_lookup),
      std::move(per_layer_embedding_lookup), use_fp16_precision,
      activation_data_type, std::move(mtp_drafter), executor_metadata));

  if (enable_profiling) {
    auto status = executor->StartProfiling();
    if (!status.ok()) {
      ABSL_LOG(WARNING) << "Failed to start profiling: " << status;
    }
  }
  return executor;
}

/* ===========================================================================*/
/* LlmLiteRtCompiledModelExecutorDynamic */
/* ===========================================================================*/

absl::Status LlmLiteRtCompiledModelExecutorDynamic::Prefill(
    const ExecutorInputs& inputs, const ExecutorPrefillParams& params) {

  // Only accept batch size 1 for now.
  LITERT_RETURN_IF_ERROR(PrepareFirstPrefillAfterDecode(0));

  if (embedding_lookup_ != nullptr) {
    ABSL_RETURN_IF_ERROR(embedding_lookup_->UpdateMultiModalEmbeddings(inputs));
  }
  auto cleanup = absl::MakeCleanup([this]() {
    if (embedding_lookup_ != nullptr) {
      embedding_lookup_->CleanupMultiModalEmbeddings().IgnoreError();
    }
  });

  LITERT_ASSIGN_OR_RETURN(auto token_ids_buffer, inputs.GetTextTokenIdsPtr());
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, token_ids_buffer->TensorType());
  RET_CHECK_EQ(tensor_type.Layout().Dimensions()[0], 1);
  RET_CHECK_GT(tensor_type.Layout().Dimensions()[1], 0)
      << "Prefill token ids must be non-empty.";
  LITERT_ASSIGN_OR_RETURN(absl::Span<int> ids,
                          ReferTensorBufferAsSpan<int32_t>(*token_ids_buffer));

  if (prefill_chunk_size_ <= 0) {
    return PrefillInternal(ids, params);
  }

  while (!ids.empty()) {
    int chunk_size =
        std::min(static_cast<int>(ids.size()), prefill_chunk_size_);
    absl::Span<int> chunk_ids = ids.first(chunk_size);
    ids = ids.subspan(chunk_size);
    ABSL_RETURN_IF_ERROR(PrefillInternal(chunk_ids, params));
  }
  return absl::OkStatus();
}

absl::Status LlmLiteRtCompiledModelExecutorDynamic::PrefillInternal(
    absl::Span<int> ids, const ExecutorPrefillParams& params) {
  ABSL_RETURN_IF_ERROR(RollBackProcessedTokens());
  // Check if have a pending input token. Note that 'internal_start_step' is
  // always equal to the number of processed tokens plus 1.
  ProcessedTokens::StepAndToken step_and_token =
      llm_context_->processed_context()
          .processed_tokens()
          .GetNextUnprocessedToken();
  bool has_pending_input_token = !step_and_token.token.empty();
  int prefill_length = has_pending_input_token ? ids.size() : ids.size() - 1;
  // If there is no pending input token and no input token to prefill, we can
  // return early by storing the token as a pending input token.
  if (!has_pending_input_token && prefill_length == 0) {
    auto pending_token = std::make_shared<TokenData>(ids[0]);
    if (embedding_lookup_ != nullptr) {
      ABSL_RETURN_IF_ERROR(embedding_lookup_->LookupPrefill(
          pending_token->id(), pending_token->mutable_embedding()));
      if (per_layer_embedding_lookup_ != nullptr) {
        ABSL_RETURN_IF_ERROR(per_layer_embedding_lookup_->LookupPrefill(
            pending_token->id(), pending_token->mutable_per_layer_embedding()));
      }
    }
    ABSL_RETURN_IF_ERROR(llm_context_->processed_context()
                             .processed_tokens()
                             .AddPendingInputToken({std::move(pending_token)}));
    ++llm_context_->runtime_state().current_step;
    return absl::OkStatus();
  }

  auto* litert_state = dynamic_cast<LitertState*>(state_.get());
  RET_CHECK(litert_state != nullptr);

  int kv_length = litert_state->GetNumEntries();
  if (kv_length == 1 && step_and_token.step == 0) {
    LITERT_RETURN_IF_ERROR(litert_state->Resize(
        *compiled_model_, kPrefillSignatureRunner, prefill_length));
    kv_length = prefill_length;
  } else {
    int free_kv_entries = kv_length - step_and_token.step;
    if (prefill_length > free_kv_entries) {
      int new_kv_seq_len = kv_length + prefill_length;
      LITERT_RETURN_IF_ERROR(litert_state->Resize(
          *compiled_model_, kPrefillSignatureRunner, new_kv_seq_len));
      kv_length = new_kv_seq_len;
    }
  }

  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_input_buffers;
  ABSL_RETURN_IF_ERROR(CreatePrefillInputBuffers(
      "prefill", prefill_length, kv_length, prefill_input_buffers));
  absl::flat_hash_map<absl::string_view, TensorBuffer> prefill_output_buffers;
  ABSL_RETURN_IF_ERROR(CreatePrefillOutputBuffers("prefill", prefill_length,
                                                  prefill_output_buffers));

  bool async = !params.GetWaitForCompletion();
  return LlmLiteRtCompiledModelExecutorBase::PrefillInternal(
      "prefill", prefill_input_buffers, prefill_output_buffers, ids, async);
}

absl::Status LlmLiteRtCompiledModelExecutorDynamic::DecodeInternal(
    const std::vector<std::shared_ptr<TokenData>>& token,
    TensorBuffer& output_logits) {
  auto* litert_state = dynamic_cast<LitertState*>(state_.get());
  RET_CHECK(litert_state != nullptr);

  int current_kv_len = litert_state->GetNumEntries();

  if (current_kv_len <= llm_context_->runtime_state().current_step - 1) {
    int entries_to_add = kv_increament_size_;
    int new_kv_len = current_kv_len + entries_to_add;
    LITERT_RETURN_IF_ERROR(litert_state->Resize(
        *compiled_model_, kDecodeSignatureRunner, new_kv_len));
    current_kv_len = new_kv_len;
  }

  ABSL_RETURN_IF_ERROR(ResolveDynamicShape(*compiled_model_, "decode",
                                           signatures_.input_attn_mask.value(),
                                           current_kv_len));
  LITERT_ASSIGN_OR_RETURN(
      decode_input_buffers_[signatures_.input_attn_mask.value()],
      compiled_model_->CreateInputBuffer("decode",
                                         signatures_.input_attn_mask.value()));

  return LlmLiteRtCompiledModelExecutorBase::DecodeInternal(token,
                                                            output_logits);
}

// static
// Creates a LlmLiteRtCompiledModelExecutorDynamic from a LiteRt model.
absl::StatusOr<std::unique_ptr<LlmLiteRtCompiledModelExecutorDynamic>>
LlmLiteRtCompiledModelExecutorDynamic::Create(
    LlmExecutorSettings executor_settings, Environment& lrt_env,
    ModelResources& resources) {
  ABSL_ASSIGN_OR_RETURN(
      auto litert_model,
      resources.GetTFLiteModel(ModelType::kTfLitePrefillDecode));

  const proto::ExecutorMetadata* executor_metadata = nullptr;
  auto executor_metadata_or = resources.GetExecutorMetadata();
  if (executor_metadata_or.ok()) {
    executor_metadata = *executor_metadata_or;
  }
  ABSL_ASSIGN_OR_RETURN(
      auto compilation_options,
      CreateCompilationOptions(executor_settings, ActivationDataType::FLOAT32,
                               /*signatures=*/std::nullopt));
  std::string weight_cache_path = executor_settings.GetCacheDir();

  const Backend backend = executor_settings.GetBackend();
  RET_CHECK_EQ(backend, Backend::CPU)
      << "LlmLiteRtCompiledModelExecutorDynamic only supports CPU backend.";
  uint32_t kv_increament_size = 0;
  int prefill_chunk_size = -1;
  {
    ABSL_ASSIGN_OR_RETURN(const auto& cpu_config,
                          executor_settings.GetBackendConfig<CpuConfig>());
    kv_increament_size = cpu_config.kv_increment_size;
    prefill_chunk_size = cpu_config.prefill_chunk_size;
    RET_CHECK_GT(kv_increament_size, 0)
        << "KV increment size must be greater than 0.";
  }

  std::unique_ptr<CompiledModel> compiled_model;
  {
    LITERT_ASSIGN_OR_RETURN(auto compiled_model_tmp,
                            CompiledModel::Create(lrt_env, litert_model->Get(),
                                                  compilation_options));
    compiled_model =
        std::make_unique<CompiledModel>(std::move(compiled_model_tmp));
  }

  LITERT_ASSIGN_OR_RETURN(auto decode_signature,
                          litert_model->FindSignature(kDecodeSignatureRunner));
  ABSL_ASSIGN_OR_RETURN(
      ModelSignatures signatures,
      GetModelSignaturesFromInputOutputNames(decode_signature.InputNames(),
                                             decode_signature.OutputNames()));

  LITERT_ASSIGN_OR_RETURN(
      const SimpleTensor& output_logits_tensor,
      decode_signature.OutputTensor(signatures.output_logits));
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType output_logits_tensor_type,
                          output_logits_tensor.RankedTensorType());
  RET_CHECK(output_logits_tensor_type.Layout().Dimensions().size() == 3)
      << "Output logits must be (batch, seq, vocab)";
  int batch_size = output_logits_tensor_type.Layout().Dimensions()[0];
  RET_CHECK_EQ(batch_size, 1) << "Only support batch size 1 for now.";

  bool clear_kv_cache_before_prefill =
      !executor_settings.GetAdvancedSettings() ||
      executor_settings.GetAdvancedSettings()->clear_kv_cache_before_prefill;

  LITERT_ASSIGN_OR_RETURN(
      auto state, LitertState::Create(
                      lrt_env, *compiled_model, "prefill", executor_metadata,
                      LitertState::AllocationPolicy::kInplace, batch_size,
                      clear_kv_cache_before_prefill));

  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_input_buffers;
  absl::flat_hash_map<absl::string_view, TensorBuffer> decode_output_buffers;

  for (auto input_name : decode_signature.InputNames()) {
    if (state->Contains(input_name)) {
      continue;
    }
    bool is_attn_mask_input =
        signatures.input_attn_mask.has_value() &&
        absl::StartsWith(input_name, signatures.input_attn_mask.value());
    if (!is_attn_mask_input) {
      LITERT_ASSIGN_OR_RETURN(auto input_buffer,
                              compiled_model->CreateInputBuffer(
                                  kDecodeSignatureRunner, input_name));
      decode_input_buffers[input_name] = std::move(input_buffer);
    }
  }
  for (auto output_name : decode_signature.OutputNames()) {
    if (state->Contains(output_name)) {
      continue;
    }
    LITERT_ASSIGN_OR_RETURN(auto output_buffer,
                            compiled_model->CreateOutputBuffer(
                                kDecodeSignatureRunner, output_name));
    decode_output_buffers[output_name] = std::move(output_buffer);
  }

  std::unique_ptr<EmbeddingLookupManager> embedding_lookup;
  std::unique_ptr<EmbeddingLookupManager> per_layer_embedding_lookup;
  ABSL_RETURN_IF_ERROR(InitializeEmbeddingLookups(
      lrt_env, resources, embedding_lookup, per_layer_embedding_lookup));

  bool enable_profiling =
      executor_settings.GetAdvancedSettings() &&
      executor_settings.GetAdvancedSettings()->enable_profiling;
  auto executor = absl::WrapUnique(new LlmLiteRtCompiledModelExecutorDynamic(
      std::move(executor_settings), lrt_env, litert_model,
      std::move(compiled_model), std::move(decode_input_buffers),
      std::move(decode_output_buffers), std::move(state), prefill_chunk_size,
      kv_increament_size, signatures, batch_size, std::move(weight_cache_path),
      std::move(embedding_lookup), std::move(per_layer_embedding_lookup),
      /*use_fp16_precision=*/false,
      /*logits_data_type=*/LogitsDataType::FLOAT32,
      /*mtp_drafter=*/nullptr, executor_metadata));
  if (enable_profiling) {
    auto status = executor->StartProfiling();
    if (!status.ok()) {
      ABSL_LOG(WARNING) << "Failed to start profiling: " << status;
    }
  }
  return executor;
}

}  // namespace litert::lm
