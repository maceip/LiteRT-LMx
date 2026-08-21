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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_FRESH_WORKER_ADAPTER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_FRESH_WORKER_ADAPTER_H_

#include <cstdint>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/engine/engine_settings.h"

namespace litert::lm {

// Constructs the only SessionConfig admitted by the concrete exact worker for
// `stage`. The parent ExactRegenerationExecutor must derive its profile from
// this same configuration. The request may select only its bounded generation
// limit; model artifacts, backend settings, sampler identity, modalities, and
// all other execution policy remain fixed by the worker executable.
absl::StatusOr<SessionConfig> MakeEngineFreshWorkerSessionConfig(
    DPMReplayStage stage, uint32_t max_output_tokens);

// Concrete one-request worker entry point. `fixed_engine_settings` must be
// assembled by the worker executable before this function reads stdin. The
// authenticated request cannot replace them and cannot inject an Engine,
// replay catalog, inference callback, session capsule, or backend selector.
//
// After authentication and canonical stage decoding, this function creates
// one kAdvancedLiteRTCompiledModel Engine, derives and checks its
// ExactLiteRtProfile, creates one fresh session, runs full prefill and one
// exact decode, and exits through RunFreshWorkerOnce's authenticated result
// boundary. No replay catalog is constructed or consulted. This adapter is
// one-shot, but it does not by itself prove that its host is a new OS process;
// FreshWorkerProcessRunner and exact-profile admission own that independently
// observed process boundary.
absl::Status RunEngineFreshWorkerOnce(
    EngineSettings fixed_engine_settings);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_FRESH_WORKER_ADAPTER_H_
