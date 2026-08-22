# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_litertlm_library(runtime_dpm STATIC
  canonical_replay_catalog.cc
  capsule_restore_admission.cc
  capsule_restore_evidence.cc
  capsule_restore_evidence_binding.cc
  capsule_restore_evidence_codec.cc
  correction_digest.cc
  dpm_agent_replay_runtime.cc
  dpm_capabilities.cc
  dpm_engine.cc
  dpm_prepared_prefill_plan.cc
  dpm_prepared_prefill_runtime.cc
  dpm_projection_manifest.cc
  dpm_projection_prompt.cc
  dpm_projection_replay_runtime.cc
  dpm_projection_runtime.cc
  dpm_receipt_validation.cc
  dpm_replay_executor.cc
  dpm_replay_mode.cc
  engine_dpm_agent.cc
  engine_fresh_worker_adapter.cc
  engine_fresh_worker_contract.cc
  exact_decode_evidence.cc
  exact_profile_admission.cc
  filesystem_capsule_restore_admission_repository.cc
  filesystem_capsule_restore_evidence_repository.cc
  filesystem_dpm_event_log.cc
  filesystem_exact_profile_admission_repository.cc
  filesystem_session_checkpoint_repository.cc
  fresh_worker_process.cc
  fresh_worker_protocol.cc
  one_shot_dpm_projector.cc
  session_checkpoint.cc
)
add_library(LiteRTLM::Runtime::DPM ALIAS runtime_dpm)

target_include_directories(runtime_dpm
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_dpm
  PUBLIC
    LiteRTLM::Runtime::Core
    LiteRTLM::Runtime::Engine
    LiteRTLM::Runtime::Executor
    LiteRTLM::Runtime::Platform
    LiteRTLM::Runtime::Util
    LiteRTLM::nlohmann_json::nlohmann_json
    LITERTLM_DEPS
)

# Optional dpm.factmap schema support. The production DPM engine remains
# schema-agnostic and does not link this module in the reverse direction.
add_litertlm_library(runtime_dpm_fact_map STATIC
  dpm_fact_map.cc
  deterministic_dpm_fact_map_projector.cc
)
add_library(LiteRTLM::Runtime::DPMFactMap ALIAS runtime_dpm_fact_map)

target_include_directories(runtime_dpm_fact_map
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_dpm_fact_map
  PUBLIC
    LiteRTLM::Runtime::DPM
    LiteRTLM::nlohmann_json::nlohmann_json
    LITERTLM_DEPS
)
