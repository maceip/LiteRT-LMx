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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EXACT_LITERT_DECODE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EXACT_LITERT_DECODE_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/io_types.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Matches the bounded fresh-worker wire contract without making the core
// inference path depend on DPM protocol types.
inline constexpr uint32_t kMaximumExactLiteRtDecodeFrames = 1'000'000;

// Evidence for one complete raw logits tensor and the token sampled from it.
// `sha256` is computed while the tensor is read-locked, before any sampler is
// invoked. The vector position and frame_index are generation order. Stop and
// EOS tokens are retained here even when Responses intentionally filters them.
struct ExactLiteRtLogitsFrameEvidence {
  uint64_t frame_index = 0;
  ExactLiteRtLogitsFrameContract contract;
  Hash256 sha256;
  int32_t sampled_token_id = -1;

  bool operator==(const ExactLiteRtLogitsFrameEvidence& other) const {
    return frame_index == other.frame_index && contract == other.contract &&
           sha256 == other.sha256 &&
           sampled_token_id == other.sampled_token_id;
  }
};

struct ExactLiteRtDecodeEvidence {
  ExactLiteRtLogitsFrameContract logits_frame_contract;
  std::vector<int32_t> sampled_token_ids;
  std::vector<ExactLiteRtLogitsFrameEvidence> logits_frames;
};

// Blocking exact-decode output. Responses contains the ordinary filtered text
// product. Evidence is the authoritative generated-token stream paired with
// pre-sampling full-vocabulary logits hashes.
struct ExactLiteRtDecodeResult {
  ExactLiteRtDecodeResult(Responses response,
                          ExactLiteRtDecodeEvidence decode_evidence)
      : responses(std::move(response)), evidence(std::move(decode_evidence)) {}

  Responses responses;
  ExactLiteRtDecodeEvidence evidence;
};

// Request-owned capture shared across SessionAdvanced, an execution-manager
// task, Tasks::Decode, and ExecutorDecodeParams. Shared ownership is required:
// threaded managers may execute after the initiating stack has returned. The
// object accepts exactly one sampled token for each pending logits frame.
class ExactLiteRtDecodeCapture final {
 public:
  static absl::StatusOr<std::shared_ptr<ExactLiteRtDecodeCapture>> Create(
      ExactLiteRtLogitsFrameContract contract, uint32_t maximum_frames);

  ExactLiteRtDecodeCapture(const ExactLiteRtDecodeCapture&) = delete;
  ExactLiteRtDecodeCapture& operator=(const ExactLiteRtDecodeCapture&) =
      delete;

  // Validates and hashes the complete packed [1, 1, vocabulary] FP16/FP32
  // tensor. This must be called immediately before sampling.
  absl::Status CaptureLogitsBeforeSampling(
      const ::litert::TensorBuffer& logits);

  // Completes the one pending frame. Exact mode accepts one candidate and one
  // sampled token per compiled decode invocation.
  absl::Status CaptureSampledTokenIds(
      const std::vector<std::vector<int>>& sampled_token_ids);

  // Seals the object and returns evidence only when every frame has exactly
  // one nonnegative sampled token and at least one frame was produced.
  absl::StatusOr<ExactLiteRtDecodeEvidence> Finish();

  const ExactLiteRtLogitsFrameContract& contract() const { return contract_; }

 private:
  ExactLiteRtDecodeCapture(ExactLiteRtLogitsFrameContract contract,
                           uint32_t maximum_frames)
      : contract_(contract), maximum_frames_(maximum_frames) {}

  const ExactLiteRtLogitsFrameContract contract_;
  const uint32_t maximum_frames_;

  mutable absl::Mutex mutex_;
  std::optional<Hash256> pending_logits_sha256_ ABSL_GUARDED_BY(mutex_);
  std::vector<ExactLiteRtLogitsFrameEvidence> frames_
      ABSL_GUARDED_BY(mutex_);
  bool sealed_ ABSL_GUARDED_BY(mutex_) = false;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_EXACT_LITERT_DECODE_H_
