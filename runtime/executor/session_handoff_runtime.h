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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SESSION_HANDOFF_RUNTIME_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SESSION_HANDOFF_RUNTIME_H_

#include <cstdint>
#include <optional>
#include <string>

namespace litert::lm {

// Concrete loaded-runtime class whose artifacts can be measured by the
// platform implementation. A backend must not claim a class until its actual
// delegate/device evidence is available; unsupported classes fail closed.
enum class SessionHandoffRuntimeClass : uint8_t {
  kLiteRtCpu = 1,
  // This class may be returned only after the loaded executor proves the
  // selected ML Drift Metal accelerator with live Environment handles,
  // Metal-backed allocations, and a selected-accelerator code anchor.
  kLiteRtMetal = 2,
};

// Element type read from the live compiled decode-logits tensor. This is a
// runtime measurement, not a requested activation-type label.
enum class SessionHandoffLogitsElementType : uint8_t {
  kUnsupported = 0,
  kFloat16 = 1,
  kFloat32 = 2,
};

// Executor-owned evidence for the fixed-shape, isolated Metal policy required
// by ExactRegeneration. These bits describe live implementation facts, not
// desired settings. Missing bits must fail closed.
enum class MetalCoRunEvidence : uint32_t {
  kSelectedMetalDelegate = uint32_t{1} << 0,
  kFixedPrefillSchedule = uint32_t{1} << 1,
  kFixedShapeDecode = uint32_t{1} << 2,
  kAdaptiveSplitKvDisabled = uint32_t{1} << 3,
  kQuiescentExecution = uint32_t{1} << 4,
  kCompleteSessionAndResetState = uint32_t{1} << 5,
  kSelectedKernelPipeline = uint32_t{1} << 6,
};

constexpr uint32_t MetalCoRunEvidenceBit(MetalCoRunEvidence evidence) {
  return static_cast<uint32_t>(evidence);
}

constexpr uint32_t RequiredMetalCoRunEvidence() {
  return MetalCoRunEvidenceBit(MetalCoRunEvidence::kSelectedMetalDelegate) |
         MetalCoRunEvidenceBit(MetalCoRunEvidence::kFixedPrefillSchedule) |
         MetalCoRunEvidenceBit(MetalCoRunEvidence::kFixedShapeDecode) |
         MetalCoRunEvidenceBit(MetalCoRunEvidence::kAdaptiveSplitKvDisabled) |
         MetalCoRunEvidenceBit(MetalCoRunEvidence::kQuiescentExecution) |
         MetalCoRunEvidenceBit(
             MetalCoRunEvidence::kCompleteSessionAndResetState) |
         MetalCoRunEvidenceBit(MetalCoRunEvidence::kSelectedKernelPipeline);
}

struct MetalCoRunRuntimeEvidence {
  // Addresses are transient measurement handles. They are never serialized or
  // hashed directly; the platform layer resolves them to immutable code-image
  // bytes and stable MTLDevice properties.
  const void* metal_device = nullptr;
  const void* metal_command_queue = nullptr;
  uintptr_t selected_accelerator_code_anchor = 0;

  uint32_t derived_evidence = 0;

  // Canonical executor-owned facts: actual loaded tensor contracts, fixed
  // prefill signature schedule, compilation inputs, and state/reset inventory.
  // It deliberately records missing policy facts as absent evidence rather
  // than encoding an intended value.
  std::string canonical_policy;
};

// Canonical description captured from the live concrete executor after model
// compilation and all executor setting updates. This is not a caller label.
struct SessionHandoffRuntimeProfile {
  SessionHandoffRuntimeClass runtime_class;
  std::string canonical_profile;

  // Immutable frame contract read from the live compiled decode-logits tensor.
  // The frame byte count is the overflow-checked packed tensor payload, not a
  // caller-provided buffer length. Zero/unsupported fields mean evidence is
  // unavailable.
  SessionHandoffLogitsElementType logits_element_type =
      SessionHandoffLogitsElementType::kUnsupported;
  int32_t logits_batch_size = 0;
  int32_t logits_sequence_size = 0;
  int32_t logits_vocabulary_size = 0;
  uint64_t logits_frame_byte_count = 0;

  // Executor-retained values that were actually used to construct the loaded
  // runtime. Static prefill schedules use -1 and bind their ordered signature
  // contract in canonical_profile instead of claiming a dynamic chunk size.
  uint32_t cpu_thread_count = 0;
  int32_t prefill_chunk_size = 0;

  // Address of a function dispatched by the live LiteRT runtime proxy. The
  // platform layer uses it only to locate and measure the loaded image; the
  // ASLR-dependent address itself is never hashed.
  uintptr_t runtime_code_anchor = 0;

  // Present only for kLiteRtMetal after concrete Metal selection is proven.
  std::optional<MetalCoRunRuntimeEvidence> metal_corun;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SESSION_HANDOFF_RUNTIME_H_
