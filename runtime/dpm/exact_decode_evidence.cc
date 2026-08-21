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

#include "runtime/dpm/exact_decode_evidence.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/engine/exact_litert_decode.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/io_types.h"

namespace litert::lm {
namespace {

bool IsValidUtf8(absl::string_view text) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t lead = bytes[index++];
    if (lead <= 0x7f) continue;

    size_t continuation_count = 0;
    uint8_t minimum_second = 0x80;
    uint8_t maximum_second = 0xbf;
    if (lead >= 0xc2 && lead <= 0xdf) {
      continuation_count = 1;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      continuation_count = 2;
      if (lead == 0xe0) minimum_second = 0xa0;
      if (lead == 0xed) maximum_second = 0x9f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      continuation_count = 3;
      if (lead == 0xf0) minimum_second = 0x90;
      if (lead == 0xf4) maximum_second = 0x8f;
    } else {
      return false;
    }
    if (continuation_count > text.size() - index ||
        bytes[index] < minimum_second || bytes[index] > maximum_second) {
      return false;
    }
    ++index;
    for (size_t offset = 1; offset < continuation_count; ++offset, ++index) {
      if (bytes[index] < 0x80 || bytes[index] > 0xbf) return false;
    }
  }
  return true;
}

absl::StatusOr<FreshWorkerLogitElementType> ConvertElementType(
    ExactLiteRtLogitsElementType element_type) {
  switch (element_type) {
    case ExactLiteRtLogitsElementType::kFloat16:
      return FreshWorkerLogitElementType::kFloat16;
    case ExactLiteRtLogitsElementType::kFloat32:
      return FreshWorkerLogitElementType::kFloat32;
    case ExactLiteRtLogitsElementType::kUnsupported:
      break;
  }
  return absl::UnimplementedError(
      "Exact decode produced an unsupported logits element type.");
}

}  // namespace

absl::StatusOr<CanonicalExactDecodeEvidence>
CanonicalizeExactLiteRtDecodeEvidence(
    const ExactLiteRtDecodeResult& decoded,
    const ExactLiteRtProfile& derived_profile,
    uint32_t max_output_tokens) {
  ABSL_RETURN_IF_ERROR(ValidateExactLiteRtProfile(derived_profile));
  if (decoded.responses.GetTaskState() != TaskState::kDone &&
      decoded.responses.GetTaskState() != TaskState::kMaxNumTokensReached) {
    return absl::InternalError(
        "Exact decode returned a non-success terminal state.");
  }
  if (decoded.responses.GetTexts().size() != 1 ||
      !IsValidUtf8(decoded.responses.GetTexts().front())) {
    return absl::DataLossError(
        "Exact decode did not return one UTF-8 visible candidate.");
  }

  const ExactLiteRtDecodeEvidence& evidence = decoded.evidence;
  if (max_output_tokens == 0 ||
      evidence.logits_frame_contract != derived_profile.logits_frame ||
      evidence.sampled_token_ids.empty() ||
      evidence.sampled_token_ids.size() > max_output_tokens ||
      evidence.sampled_token_ids.size() != evidence.logits_frames.size()) {
    return absl::DataLossError(
        "Exact decode evidence differs from the Engine-derived logits "
        "contract or generation bound.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const FreshWorkerLogitElementType element_type,
      ConvertElementType(evidence.logits_frame_contract.element_type));
  const uint32_t element_byte_width =
      element_type == FreshWorkerLogitElementType::kFloat16 ? 2 : 4;

  CanonicalExactDecodeEvidence canonical;
  canonical.visible_output = decoded.responses.GetTexts().front();
  canonical.logit_frames.reserve(evidence.logits_frames.size());
  for (size_t index = 0; index < evidence.logits_frames.size(); ++index) {
    const ExactLiteRtLogitsFrameEvidence& frame =
        evidence.logits_frames[index];
    if (frame.frame_index != index ||
        frame.contract != evidence.logits_frame_contract ||
        frame.sampled_token_id != evidence.sampled_token_ids[index] ||
        frame.sampled_token_id < 0 ||
        static_cast<uint32_t>(frame.sampled_token_id) >=
            evidence.logits_frame_contract.vocabulary_size) {
      return absl::DataLossError(
          "Exact decode logits frames are not ordered and paired with every "
          "sampled token.");
    }
    FreshWorkerLogitFrameEvidence worker_frame{
        .element_type = element_type,
        .element_byte_width = element_byte_width,
        .batch_size = evidence.logits_frame_contract.batch_size,
        .sequence_size = evidence.logits_frame_contract.sequence_size,
        .vocabulary_size = evidence.logits_frame_contract.vocabulary_size,
        .byte_count = evidence.logits_frame_contract.byte_count,
        .sha256 = frame.sha256,
    };
    ABSL_RETURN_IF_ERROR(
        ValidateFreshWorkerLogitFrameEvidence(worker_frame));
    canonical.logit_frames.push_back(worker_frame);
  }
  ABSL_ASSIGN_OR_RETURN(
      canonical.token_bytes,
      EncodeFreshWorkerTokenIds(evidence.sampled_token_ids));
  return canonical;
}

}  // namespace litert::lm
