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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_EXACT_DECODE_EVIDENCE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_EXACT_DECODE_EVIDENCE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/engine/exact_litert_decode.h"
#include "runtime/engine/exact_litert_profile.h"

namespace litert::lm {

// Canonical, stage-independent evidence for one exact decode. Visible output
// remains separate from DPMTOK01 because EOS/stop and byte-fallback tokens are
// not recoverable by re-tokenizing text. The ordered logits frames cover every
// sampled token before ordinary response filtering.
struct CanonicalExactDecodeEvidence {
  std::string visible_output;
  std::string token_bytes;
  std::vector<FreshWorkerLogitFrameEvidence> logit_frames;
};

// Validates the terminal response, the loaded full-logits tensor contract,
// every ordered frame/token pairing, and the product generation bound. It
// preserves the exact sampled token sequence as DPMTOK01.
absl::StatusOr<CanonicalExactDecodeEvidence>
CanonicalizeExactLiteRtDecodeEvidence(
    const ExactLiteRtDecodeResult& decoded,
    const ExactLiteRtProfile& derived_profile,
    uint32_t max_output_tokens);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_EXACT_DECODE_EVIDENCE_H_
