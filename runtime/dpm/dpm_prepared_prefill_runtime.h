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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PREPARED_PREFILL_RUNTIME_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PREPARED_PREFILL_RUNTIME_H_

#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_prepared_prefill_plan.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Runtime-owned starting state for prefill preparation. A restored plan must
// carry the witness independently recomputed by ImportHandoffFromWithWitness;
// a caller-selected checkpoint ID without that live-target evidence is
// insufficient.
struct DPMPreparedPrefillStart {
  DPMPreparedPrefillStartKind kind =
      DPMPreparedPrefillStartKind::kFreshSession;
  std::optional<Hash256> restore_checkpoint_id;
  std::optional<SessionContinuationStateWitness> restored_state_witness;
};

// Resolves canonical text through the tokenizer owned by `engine`, including
// the exact first-call BOS behavior hidden by ordinary Session preprocessing.
// The returned plan contains only exact token buffers and can therefore be
// executed without a second tokenization or prompt-template decision.
absl::StatusOr<DPMPreparedPrefillPlan> PrepareDPMEnginePrefillPlan(
    Engine* engine, Engine::Session* session,
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& source_chunks,
    const Hash256& logical_agent_request_hash,
    const DPMPreparedPrefillStart& start);

// Executes the exact preprocessed token buffers in the plan. The restored
// witness is required again for own-position plans so a plan ID alone cannot
// authorize an arbitrary live session. The implementation checks each call's
// resulting step and finally compares the complete exact token history.
absl::Status ExecuteDPMEnginePrefillPlan(
    Engine::Session* session, const DPMPreparedPrefillPlan& plan,
    const std::optional<SessionContinuationStateWitness>&
        restored_state_witness = std::nullopt);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_DPM_PREPARED_PREFILL_RUNTIME_H_
