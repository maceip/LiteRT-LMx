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

#include "runtime/engine/exact_litert_decode.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_element_type.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

absl::StatusOr<uint64_t> ElementByteWidth(
    ExactLiteRtLogitsElementType element_type) {
  switch (element_type) {
    case ExactLiteRtLogitsElementType::kFloat16:
      return uint64_t{2};
    case ExactLiteRtLogitsElementType::kFloat32:
      return uint64_t{4};
    default:
      return absl::UnimplementedError(
          "Exact decode requires an Engine-derived FP16 or FP32 logits "
          "contract.");
  }
}

absl::Status ValidateContract(
    const ExactLiteRtLogitsFrameContract& contract) {
  if (contract.batch_size != 1 || contract.sequence_size != 1) {
    return absl::UnimplementedError(
        "Exact decode requires batch-one, one-position logits.");
  }
  if (contract.vocabulary_size == 0) {
    return absl::FailedPreconditionError(
        "Exact decode logits contract has an empty vocabulary.");
  }
  if (contract.vocabulary_size >
      static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    return absl::ResourceExhaustedError(
        "Exact decode vocabulary exceeds LiteRT dimension range.");
  }
  absl::StatusOr<uint64_t> element_bytes =
      ElementByteWidth(contract.element_type);
  if (!element_bytes.ok()) return element_bytes.status();
  if (contract.vocabulary_size >
      std::numeric_limits<uint64_t>::max() / *element_bytes) {
    return absl::ResourceExhaustedError(
        "Exact decode logits frame byte count overflows uint64.");
  }
  if (contract.byte_count !=
      static_cast<uint64_t>(contract.vocabulary_size) * *element_bytes) {
    return absl::FailedPreconditionError(
        "Exact decode logits contract does not cover the complete "
        "full-vocabulary frame.");
  }
  if (contract.byte_count > std::numeric_limits<size_t>::max()) {
    return absl::ResourceExhaustedError(
        "Exact decode logits frame exceeds addressable memory.");
  }
  return absl::OkStatus();
}

absl::Status ValidateTensorAgainstContract(
    const ::litert::TensorBuffer& logits,
    const ExactLiteRtLogitsFrameContract& contract) {
  LITERT_ASSIGN_OR_RETURN(const ::litert::RankedTensorType tensor_type,
                          logits.TensorType());
  const ::litert::ElementType expected_element_type =
      contract.element_type == ExactLiteRtLogitsElementType::kFloat16
          ? ::litert::ElementType::Float16
          : ::litert::ElementType::Float32;
  if (tensor_type.ElementType() != expected_element_type) {
    return absl::DataLossError(
        "Produced logits element type differs from the Engine-derived exact "
        "contract.");
  }
  if (tensor_type.Layout().HasStrides()) {
    return absl::UnimplementedError(
        "Exact decode cannot hash explicitly strided logits as a canonical "
        "packed frame.");
  }
  const auto dimensions = tensor_type.Layout().Dimensions();
  if (dimensions.size() != 3 || dimensions[0] != 1 || dimensions[1] != 1 ||
      dimensions[2] != static_cast<int>(contract.vocabulary_size)) {
    return absl::DataLossError(
        "Produced logits shape differs from the Engine-derived [1, 1, "
        "vocabulary] contract.");
  }
  LITERT_ASSIGN_OR_RETURN(const size_t element_count,
                          tensor_type.Layout().NumElements());
  if (element_count != static_cast<size_t>(contract.vocabulary_size)) {
    return absl::DataLossError(
        "Produced logits element count differs from the complete vocabulary "
        "extent.");
  }
  LITERT_ASSIGN_OR_RETURN(const size_t packed_size, logits.PackedSize());
  if (packed_size != static_cast<size_t>(contract.byte_count)) {
    return absl::DataLossError(
        "Produced logits packed byte count differs from the Engine-derived "
        "exact frame extent.");
  }
  return absl::OkStatus();
}

}  // namespace

// static
absl::StatusOr<std::shared_ptr<ExactLiteRtDecodeCapture>>
ExactLiteRtDecodeCapture::Create(ExactLiteRtLogitsFrameContract contract,
                                 uint32_t maximum_frames) {
  absl::Status contract_status = ValidateContract(contract);
  if (!contract_status.ok()) return contract_status;
  if (maximum_frames == 0 ||
      maximum_frames > kMaximumExactLiteRtDecodeFrames) {
    return absl::InvalidArgumentError(
        "Exact decode maximum frame count is outside the admitted range.");
  }
  return std::shared_ptr<ExactLiteRtDecodeCapture>(
      new ExactLiteRtDecodeCapture(contract, maximum_frames));
}

absl::Status ExactLiteRtDecodeCapture::CaptureLogitsBeforeSampling(
    const ::litert::TensorBuffer& logits) {
  absl::Status tensor_status = ValidateTensorAgainstContract(logits, contract_);
  if (!tensor_status.ok()) return tensor_status;

  LITERT_ASSIGN_OR_RETURN(
      auto lock_and_address,
      ::litert::TensorBufferScopedLock::Create<const void>(
          logits,
          ::litert::TensorBuffer::LockMode::kRead));
  if (lock_and_address.second == nullptr) {
    return absl::DataLossError(
        "Exact decode logits read lock returned a null address.");
  }
  const absl::string_view packed_bytes(
      static_cast<const char*>(lock_and_address.second),
      static_cast<size_t>(contract_.byte_count));
  Sha256Hasher hasher;
  const Hash256 digest = hasher.OneShot(packed_bytes);

  absl::MutexLock lock(mutex_);
  if (sealed_) {
    return absl::FailedPreconditionError(
        "Exact decode capture is already sealed.");
  }
  if (pending_logits_sha256_.has_value()) {
    return absl::DataLossError(
        "Exact decode produced a second logits frame before sampling the "
        "first.");
  }
  if (frames_.size() >= maximum_frames_) {
    return absl::ResourceExhaustedError(
        "Exact decode exceeded its admitted logits frame count.");
  }
  pending_logits_sha256_ = digest;
  return absl::OkStatus();
}

absl::Status ExactLiteRtDecodeCapture::CaptureSampledTokenIds(
    const std::vector<std::vector<int>>& sampled_token_ids) {
  if (sampled_token_ids.size() != 1 ||
      sampled_token_ids.front().size() != 1) {
    return absl::DataLossError(
        "Exact decode requires exactly one sampled token from one candidate "
        "per logits frame.");
  }
  const int sampled_token_id = sampled_token_ids.front().front();
  if (sampled_token_id < 0 ||
      static_cast<uint64_t>(sampled_token_id) >= contract_.vocabulary_size) {
    return absl::DataLossError(
        "Exact decode sampled a token outside the Engine-derived "
        "vocabulary.");
  }

  absl::MutexLock lock(mutex_);
  if (sealed_) {
    return absl::FailedPreconditionError(
        "Exact decode capture is already sealed.");
  }
  if (!pending_logits_sha256_.has_value()) {
    return absl::DataLossError(
        "Exact decode sampled a token without a preceding captured logits "
        "frame.");
  }
  const uint64_t frame_index = frames_.size();
  frames_.push_back(ExactLiteRtLogitsFrameEvidence{
      .frame_index = frame_index,
      .contract = contract_,
      .sha256 = *pending_logits_sha256_,
      .sampled_token_id = static_cast<int32_t>(sampled_token_id),
  });
  pending_logits_sha256_.reset();
  return absl::OkStatus();
}

absl::StatusOr<ExactLiteRtDecodeEvidence> ExactLiteRtDecodeCapture::Finish() {
  absl::MutexLock lock(mutex_);
  if (sealed_) {
    return absl::FailedPreconditionError(
        "Exact decode capture was already consumed.");
  }
  sealed_ = true;
  if (pending_logits_sha256_.has_value()) {
    return absl::DataLossError(
        "Exact decode ended with an unpaired logits frame.");
  }
  if (frames_.empty()) {
    return absl::DataLossError(
        "Exact decode completed without capturing any logits frames.");
  }

  ExactLiteRtDecodeEvidence evidence;
  evidence.logits_frame_contract = contract_;
  evidence.sampled_token_ids.reserve(frames_.size());
  evidence.logits_frames = std::move(frames_);
  for (size_t index = 0; index < evidence.logits_frames.size(); ++index) {
    const ExactLiteRtLogitsFrameEvidence& frame =
        evidence.logits_frames[index];
    if (frame.frame_index != index || frame.contract != contract_ ||
        frame.sampled_token_id < 0) {
      return absl::DataLossError(
          "Exact decode frame ordering or token pairing is inconsistent.");
    }
    evidence.sampled_token_ids.push_back(frame.sampled_token_id);
  }
  if (evidence.sampled_token_ids.size() != evidence.logits_frames.size()) {
    return absl::DataLossError(
        "Exact decode token and logits frame counts differ.");
  }
  return evidence;
}

}  // namespace litert::lm
