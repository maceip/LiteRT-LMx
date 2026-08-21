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

#include "runtime/components/greedy_cpu_sampler.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "tflite/types/half.h"  // from @litert

namespace litert::lm {
namespace {

struct GreedyInputShape {
  size_t row_count;
  size_t vocab_size;
};

bool IsFp32Nan(float value) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  static_assert(std::numeric_limits<float>::is_iec559);
  const uint32_t bits = std::bit_cast<uint32_t>(value);
  constexpr uint32_t kExponentMask = 0x7f800000u;
  constexpr uint32_t kMantissaMask = 0x007fffffu;
  return (bits & kExponentMask) == kExponentMask &&
         (bits & kMantissaMask) != 0;
}

absl::Status ValidateConfig(
    const proto::SamplerParameters& sampler_params, int batch_size,
    int sequence_size) {
  if (!proto::SamplerParameters::Type_IsValid(sampler_params.type()) ||
      sampler_params.type() != proto::SamplerParameters::GREEDY) {
    return absl::InvalidArgumentError(
        "GreedyCpuSampler requires sampler type GREEDY.");
  }
  if (!proto::SamplerParameters::Backend_IsValid(sampler_params.backend())) {
    return absl::InvalidArgumentError("Invalid GREEDY sampler backend.");
  }
  if (sampler_params.backend() == proto::SamplerParameters::GPU) {
    return absl::UnimplementedError(
        "GREEDY sampling is supported only on CPU.");
  }
  if (batch_size <= 0) {
    return absl::InvalidArgumentError("batch_size must be positive.");
  }
  if (sequence_size <= 0) {
    return absl::InvalidArgumentError("sequence_size must be positive.");
  }
  return absl::OkStatus();
}

absl::StatusOr<GreedyInputShape> ValidateInputTensor(
    const TensorBuffer& tensor, int batch_size, int sequence_size) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
  if (tensor_type.ElementType() != ElementType::Float32 &&
      tensor_type.ElementType() != ElementType::Float16) {
    return absl::InvalidArgumentError(
        "GREEDY sampler logits must have FP32 or FP16 element type.");
  }
  if (tensor_type.Layout().HasStrides()) {
    return absl::UnimplementedError(
        "GREEDY sampler does not support explicitly strided logits.");
  }

  const auto dimensions = tensor_type.Layout().Dimensions();
  int vocab_size = 0;
  if (dimensions.size() == 2 && sequence_size == 1) {
    if (dimensions[0] != batch_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "GREEDY logits batch dimension does not match configured batch: ",
          dimensions[0], " vs ", batch_size, "."));
    }
    vocab_size = dimensions[1];
  } else if (dimensions.size() == 3) {
    if (dimensions[0] != batch_size || dimensions[1] != sequence_size) {
      return absl::InvalidArgumentError(absl::StrCat(
          "GREEDY logits shape does not match configured batch/sequence: [",
          dimensions[0], ", ", dimensions[1], "] vs [", batch_size, ", ",
          sequence_size, "]."));
    }
    vocab_size = dimensions[2];
  } else {
    return absl::InvalidArgumentError(
        "GREEDY logits must have shape [batch, vocab] for sequence size 1 "
        "or [batch, sequence, vocab].");
  }
  if (vocab_size <= 0) {
    return absl::InvalidArgumentError(
        "GREEDY logits vocabulary dimension must be positive.");
  }

  const size_t batch = static_cast<size_t>(batch_size);
  const size_t sequence = static_cast<size_t>(sequence_size);
  if (batch > std::numeric_limits<size_t>::max() / sequence) {
    return absl::ResourceExhaustedError(
        "GREEDY sampler batch/sequence size overflows addressable memory.");
  }
  const size_t row_count = batch * sequence;
  const size_t vocab = static_cast<size_t>(vocab_size);
  if (row_count > std::numeric_limits<size_t>::max() / vocab) {
    return absl::ResourceExhaustedError(
        "GREEDY logits element count overflows addressable memory.");
  }
  LITERT_ASSIGN_OR_RETURN(auto element_count,
                          tensor_type.Layout().NumElements());
  if (element_count != row_count * vocab) {
    return absl::InvalidArgumentError(
        "GREEDY logits layout has an inconsistent element count.");
  }
  return GreedyInputShape{.row_count = row_count, .vocab_size = vocab};
}

absl::Status ValidateOutputTensor(const TensorBuffer& tensor,
                                  ElementType element_type, int batch_size,
                                  int sequence_size,
                                  const std::string& tensor_name) {
  LITERT_ASSIGN_OR_RETURN(auto tensor_type, tensor.TensorType());
  if (tensor_type.ElementType() != element_type) {
    return absl::InvalidArgumentError(absl::StrCat(
        "GREEDY ", tensor_name, " tensor has an unsupported element type."));
  }
  if (tensor_type.Layout().HasStrides()) {
    return absl::UnimplementedError(absl::StrCat(
        "GREEDY sampler does not support explicitly strided ", tensor_name,
        " tensors."));
  }

  const auto dimensions = tensor_type.Layout().Dimensions();
  const bool flat_single_sequence =
      sequence_size == 1 && dimensions.size() == 1 &&
      dimensions[0] == batch_size;
  const bool explicit_sequence =
      dimensions.size() == 2 && dimensions[0] == batch_size &&
      dimensions[1] == sequence_size;
  if (!flat_single_sequence && !explicit_sequence) {
    return absl::InvalidArgumentError(absl::StrCat(
        "GREEDY ", tensor_name,
        " tensor must have shape [batch] for sequence size 1 or "
        "[batch, sequence]."));
  }

  LITERT_ASSIGN_OR_RETURN(auto element_count,
                          tensor_type.Layout().NumElements());
  const size_t expected_count = static_cast<size_t>(batch_size) *
                                static_cast<size_t>(sequence_size);
  if (element_count != expected_count) {
    return absl::InvalidArgumentError(absl::StrCat(
        "GREEDY ", tensor_name,
        " tensor layout has an inconsistent element count."));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> ReadLogits(
    const TensorBuffer& logits_tensor, ElementType element_type) {
  if (element_type == ElementType::Float32) {
    LITERT_ASSIGN_OR_RETURN(auto logits,
                            CopyFromTensorBuffer<float>(logits_tensor));
    return logits;
  }
  LITERT_ASSIGN_OR_RETURN(
      auto fp16_logits,
      CopyFromTensorBuffer<tflite::half>(logits_tensor));
  std::vector<float> logits;
  logits.reserve(fp16_logits.size());
  for (tflite::half value : fp16_logits) {
    logits.push_back(static_cast<float>(value));
  }
  return logits;
}

}  // namespace

absl::StatusOr<std::unique_ptr<GreedyCpuSampler>> GreedyCpuSampler::Create(
    int batch_size, int sequence_size,
    const proto::SamplerParameters& sampler_params) {
  absl::Status status =
      ValidateConfig(sampler_params, batch_size, sequence_size);
  if (!status.ok()) return status;
  return absl::WrapUnique(new GreedyCpuSampler(batch_size, sequence_size));
}

absl::Status GreedyCpuSampler::SampleToIdAndScoreBuffer(
    const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
    TensorBuffer* scores_tensor) {
  LITERT_ASSIGN_OR_RETURN(
      const GreedyInputShape shape,
      ValidateInputTensor(logits_tensor, batch_size_, sequence_size_));
  LITERT_RETURN_IF_ERROR(ValidateOutputTensor(
      ids_tensor, ElementType::Int32, batch_size_, sequence_size_, "IDs"));
  if (scores_tensor != nullptr) {
    LITERT_RETURN_IF_ERROR(ValidateOutputTensor(
        *scores_tensor, ElementType::Float32, batch_size_, sequence_size_,
        "scores"));
  }

  LITERT_ASSIGN_OR_RETURN(auto logits_type, logits_tensor.TensorType());
  LITERT_ASSIGN_OR_RETURN(
      std::vector<float> logits,
      ReadLogits(logits_tensor, logits_type.ElementType()));
  if (logits.size() != shape.row_count * shape.vocab_size) {
    return absl::InvalidArgumentError(
        "GREEDY logits payload does not match its declared shape.");
  }

  // Validate every value before mutating either output buffer.
  for (size_t i = 0; i < logits.size(); ++i) {
    if (IsFp32Nan(logits[i])) {
      return absl::InvalidArgumentError(
          absl::StrCat("GREEDY logits contain NaN at flat index ", i, "."));
    }
  }

  std::vector<int32_t> sampled_ids(shape.row_count);
  for (size_t row = 0; row < shape.row_count; ++row) {
    const size_t row_offset = row * shape.vocab_size;
    float maximum = logits[row_offset];
    int32_t selected_id = 0;
    for (size_t vocab_index = 1; vocab_index < shape.vocab_size;
         ++vocab_index) {
      const float value = logits[row_offset + vocab_index];
      // Strict comparison makes the lowest vocabulary index win exact ties.
      if (value > maximum) {
        maximum = value;
        selected_id = static_cast<int32_t>(vocab_index);
      }
    }
    sampled_ids[row] = selected_id;
  }

  LITERT_RETURN_IF_ERROR(
      ids_tensor.Write(absl::MakeConstSpan(sampled_ids)));
  if (scores_tensor != nullptr) {
    const std::vector<float> scores(shape.row_count, 0.0f);
    LITERT_RETURN_IF_ERROR(
        scores_tensor->Write(absl::MakeConstSpan(scores)));
  }
  return absl::OkStatus();
}

absl::Status GreedyCpuSampler::UpdateConfig(
    const proto::SamplerParameters& sampler_params, int batch_size,
    std::shared_ptr<std::default_random_engine> rand_gen) {
  // A GREEDY sampler deliberately neither consumes nor retains random state.
  (void)rand_gen;
  absl::Status status =
      ValidateConfig(sampler_params, batch_size, sequence_size_);
  if (!status.ok()) return status;
  batch_size_ = batch_size;
  return absl::OkStatus();
}

}  // namespace litert::lm
