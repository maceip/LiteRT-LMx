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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_METAL_DELEGATE_EXACT_EVIDENCE_ABI_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_METAL_DELEGATE_EXACT_EVIDENCE_ABI_H_

#include <stdint.h>

// Optional ABI exported by the selected ML Drift Metal accelerator image.
// LiteRT-LM resolves this symbol from the same loaded Mach-O image as the
// accelerator-owned Environment destruction callback. The ABI is deliberately
// model-instance scoped: global delegate defaults, requested GpuOptions, and a
// device name are not evidence of the pipelines or policies actually selected
// for a compiled model.
//
// This contract is additive to public LiteRT. An unmodified delegate has no
// symbol and therefore supplies no delegate-owned ExactRegeneration evidence.
// Ordinary LiteRT execution remains available, while the exact Metal profile
// fails closed.

#define LITERT_LM_METAL_EXACT_EVIDENCE_ABI_VERSION_V1 1u
#define LITERT_LM_METAL_EXACT_EVIDENCE_SHA256_SIZE_V1 32u
#define LITERT_LM_QUERY_METAL_EXACT_EVIDENCE_SYMBOL_V1 \
  "LiteRtLmQueryMetalExactEvidenceV1"

typedef enum LiteRtLmMetalExactEvidenceQueryStatusV1 {
  kLiteRtLmMetalExactEvidenceQuerySuccessV1 = 0,
  kLiteRtLmMetalExactEvidenceQueryUnsupportedV1 = 1,
  kLiteRtLmMetalExactEvidenceQueryInvalidArgumentV1 = 2,
  kLiteRtLmMetalExactEvidenceQueryInternalV1 = 3,
} LiteRtLmMetalExactEvidenceQueryStatusV1;

// Presence bits for independently verifiable response sections. A successful
// query may intentionally return a strict subset. Fields belonging to an
// absent section must be zero so that there is one canonical response.
typedef enum LiteRtLmMetalExactEvidenceFlagsV1 {
  kLiteRtLmMetalExactEvidenceEffectiveCompilationV1 = 1u << 0,
  kLiteRtLmMetalExactEvidenceSelectedPipelineSetV1 = 1u << 1,
  kLiteRtLmMetalExactEvidenceAdaptiveSplitKvPolicyV1 = 1u << 2,
  kLiteRtLmMetalExactEvidenceSynchronousRunCompletionV1 = 1u << 3,
  kLiteRtLmMetalExactEvidenceCompleteContinuationStateV1 = 1u << 4,
  kLiteRtLmMetalExactEvidenceCompleteResetStateV1 = 1u << 5,
} LiteRtLmMetalExactEvidenceFlagsV1;

#define LITERT_LM_METAL_EXACT_EVIDENCE_KNOWN_FLAGS_V1 \
  ((uint32_t)kLiteRtLmMetalExactEvidenceEffectiveCompilationV1 | \
   (uint32_t)kLiteRtLmMetalExactEvidenceSelectedPipelineSetV1 | \
   (uint32_t)kLiteRtLmMetalExactEvidenceAdaptiveSplitKvPolicyV1 | \
   (uint32_t)kLiteRtLmMetalExactEvidenceSynchronousRunCompletionV1 | \
   (uint32_t)kLiteRtLmMetalExactEvidenceCompleteContinuationStateV1 | \
   (uint32_t)kLiteRtLmMetalExactEvidenceCompleteResetStateV1)

typedef enum LiteRtLmMetalCalculationPrecisionV1 {
  kLiteRtLmMetalCalculationPrecisionUnspecifiedV1 = 0,
  kLiteRtLmMetalCalculationPrecisionFloat32V1 = 1,
  kLiteRtLmMetalCalculationPrecisionFloat16V1 = 2,
  kLiteRtLmMetalCalculationPrecisionFloat32Float16V1 = 3,
} LiteRtLmMetalCalculationPrecisionV1;

typedef enum LiteRtLmMetalBooleanFactV1 {
  kLiteRtLmMetalBooleanFactUnspecifiedV1 = 0,
  kLiteRtLmMetalBooleanFactFalseV1 = 1,
  kLiteRtLmMetalBooleanFactTrueV1 = 2,
} LiteRtLmMetalBooleanFactV1;

typedef enum LiteRtLmMetalWaitTypeV1 {
  kLiteRtLmMetalWaitTypeUnspecifiedV1 = 0,
  kLiteRtLmMetalWaitTypeActiveV1 = 1,
  kLiteRtLmMetalWaitTypePassiveV1 = 2,
  kLiteRtLmMetalWaitTypeDoNotWaitV1 = 3,
} LiteRtLmMetalWaitTypeV1;

typedef enum LiteRtLmMetalAdaptiveSplitKvPolicyV1 {
  kLiteRtLmMetalAdaptiveSplitKvPolicyUnspecifiedV1 = 0,
  // The delegate inspected every selected attention pipeline for this exact
  // compiled model and proved that no runtime shape/chunk/batch decision can
  // enable or retune Split-KV.
  kLiteRtLmMetalAdaptiveSplitKvDisabledForAllSelectedPipelinesV1 = 1,
  kLiteRtLmMetalAdaptiveSplitKvEnabledOrShapeDependentV1 = 2,
} LiteRtLmMetalAdaptiveSplitKvPolicyV1;

typedef enum LiteRtLmMetalRunCompletionPolicyV1 {
  kLiteRtLmMetalRunCompletionPolicyUnspecifiedV1 = 0,
  // A synchronous CompiledModel::Run returns only after every command buffer
  // submitted for that invocation has completed and every bound output/state
  // buffer is host-lock-visible. This does not itself prove that the caller
  // used the synchronous API or excluded another session.
  kLiteRtLmMetalRunWaitsForAllSubmittedWorkV1 = 1,
  kLiteRtLmMetalRunMayReturnWithSubmittedWorkPendingV1 = 2,
} LiteRtLmMetalRunCompletionPolicyV1;

typedef enum LiteRtLmMetalContinuationStatePolicyV1 {
  kLiteRtLmMetalContinuationStatePolicyUnspecifiedV1 = 0,
  // All history-dependent state is carried by the caller-bound generalized
  // state tensors. The selected compiled pipelines retain no hidden recurrent,
  // convolutional, sampler, or attention continuation state.
  kLiteRtLmMetalAllContinuationStateInBoundStateTensorsV1 = 1,
  kLiteRtLmMetalDelegateOwnsHiddenContinuationStateV1 = 2,
} LiteRtLmMetalContinuationStatePolicyV1;

typedef enum LiteRtLmMetalResetStatePolicyV1 {
  kLiteRtLmMetalResetStatePolicyUnspecifiedV1 = 0,
  // Replacing every bound generalized state tensor with a freshly allocated,
  // cleared backend-native tensor restores the delegate to its fresh-session
  // continuation state. The compiled model may remain loaded.
  kLiteRtLmMetalFreshBoundStateTensorsResetAllContinuationStateV1 = 1,
  kLiteRtLmMetalResetRequiresAdditionalDelegateMutationV1 = 2,
} LiteRtLmMetalResetStatePolicyV1;

typedef struct LiteRtLmMetalExactEvidenceQueryV1 {
  uint32_t struct_size;
  uint32_t abi_version;

  // The live opaque LiteRtCompiledModel handle. The implementation must query
  // the exact instance and echo it in the response. The accelerator/runtime
  // must have registered this handle with the concrete applied delegate and
  // its DelegateKernel instances during compiled-model construction; guessing
  // that association from a shared device, queue, model bytes, or global
  // delegate defaults is invalid.
  const void* compiled_model_instance;
  const void* selected_accelerator_state;
  const void* metal_device;
  const void* metal_command_queue;

  uint64_t reserved[4];
} LiteRtLmMetalExactEvidenceQueryV1;

typedef struct LiteRtLmMetalExactEvidenceV1 {
  uint32_t struct_size;
  uint32_t abi_version;

  // Exact live-instance bindings echoed from the query after the delegate has
  // resolved its per-compiled-model implementation state.
  const void* compiled_model_instance;
  const void* selected_accelerator_state;
  const void* metal_device;
  const void* metal_command_queue;

  uint32_t evidence_flags;

  // kEffectiveCompilation section. The digest must cover all effective
  // model-specific Metal compilation/code-generation flags, including any
  // precision, fast-math, quantization, accumulation, argument-buffer, and
  // wait-policy values not represented explicitly below.
  uint32_t effective_calculation_precision;
  uint32_t effective_f32_accumulation_for_fp16;
  uint32_t effective_argument_buffers;
  uint32_t effective_wait_type;
  uint8_t effective_compilation_flags_sha256
      [LITERT_LM_METAL_EXACT_EVIDENCE_SHA256_SIZE_V1];

  // kSelectedPipelineSet section. This is an ordered, model-instance-specific
  // digest over every selected Metal pipeline and its specialization/function
  // constants and dispatch policy, across every compiled signature. Hashing
  // source text, requested options, or a delegate binary alone is insufficient.
  uint32_t selected_pipeline_count;
  uint8_t selected_pipeline_set_sha256
      [LITERT_LM_METAL_EXACT_EVIDENCE_SHA256_SIZE_V1];

  // Independently present policy sections.
  uint32_t adaptive_split_kv_policy;
  uint32_t synchronous_run_completion_policy;
  uint32_t continuation_state_policy;
  uint32_t reset_state_policy;

  // Present only when both complete-state sections are present. It binds the
  // delegate-side proof that no continuation/reset state exists outside the
  // caller-bound tensor inventory.
  uint8_t delegate_state_inventory_sha256
      [LITERT_LM_METAL_EXACT_EVIDENCE_SHA256_SIZE_V1];

  uint64_t reserved[8];
} LiteRtLmMetalExactEvidenceV1;

typedef int32_t (*LiteRtLmQueryMetalExactEvidenceFnV1)(
    const LiteRtLmMetalExactEvidenceQueryV1* query,
    LiteRtLmMetalExactEvidenceV1* evidence);

#ifdef __cplusplus
extern "C" {
#endif

// Implemented only by a Metal accelerator that can supply the evidence above.
// LiteRT-LM resolves this by name and never links to it directly. On
// kUnsupported the implementation must preserve struct_size/abi_version and
// leave every remaining response byte zero. On kSuccess each absent section
// likewise has a zero payload.
int32_t LiteRtLmQueryMetalExactEvidenceV1(
    const LiteRtLmMetalExactEvidenceQueryV1* query,
    LiteRtLmMetalExactEvidenceV1* evidence);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_METAL_DELEGATE_EXACT_EVIDENCE_ABI_H_
