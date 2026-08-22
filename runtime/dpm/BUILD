# Copyright 2026 The ODML Authors.
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

package(
    default_hdrs_check = "strict",
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

cc_library(
    name = "dpm",
    srcs = [
        "canonical_replay_catalog.cc",
        "capsule_restore_admission.cc",
        "capsule_restore_evidence.cc",
        "capsule_restore_evidence_binding.cc",
        "capsule_restore_evidence_codec.cc",
        "correction_digest.cc",
        "dpm_agent_replay_runtime.cc",
        "dpm_capabilities.cc",
        "dpm_engine.cc",
        "dpm_prepared_prefill_plan.cc",
        "dpm_prepared_prefill_runtime.cc",
        "dpm_projection_manifest.cc",
        "dpm_projection_prompt.cc",
        "dpm_projection_replay_runtime.cc",
        "dpm_projection_runtime.cc",
        "dpm_receipt_validation.cc",
        "dpm_replay_executor.cc",
        "dpm_replay_mode.cc",
        "engine_dpm_agent.cc",
        "engine_fresh_worker_adapter.cc",
        "engine_fresh_worker_contract.cc",
        "exact_decode_evidence.cc",
        "exact_profile_admission.cc",
        "filesystem_capsule_restore_admission_repository.cc",
        "filesystem_capsule_restore_evidence_repository.cc",
        "filesystem_dpm_event_log.cc",
        "filesystem_exact_profile_admission_repository.cc",
        "filesystem_session_checkpoint_repository.cc",
        "fresh_worker_process.cc",
        "fresh_worker_protocol.cc",
        "one_shot_dpm_projector.cc",
        "session_checkpoint.cc",
    ],
    hdrs = [
        "canonical_replay_catalog.h",
        "capsule_restore_admission.h",
        "capsule_restore_evidence.h",
        "capsule_restore_evidence_binding.h",
        "capsule_restore_evidence_codec.h",
        "correction_digest.h",
        "dpm_agent_replay_runtime.h",
        "dpm_capabilities.h",
        "dpm_engine.h",
        "dpm_event_log.h",
        "dpm_prepared_prefill_plan.h",
        "dpm_prepared_prefill_runtime.h",
        "dpm_projection_manifest.h",
        "dpm_projection_prompt.h",
        "dpm_projection_replay_runtime.h",
        "dpm_projection_runtime.h",
        "dpm_receipt_validation.h",
        "dpm_replay_executor.h",
        "dpm_replay_mode.h",
        "engine_dpm_agent.h",
        "engine_fresh_worker_adapter.h",
        "engine_fresh_worker_contract.h",
        "exact_decode_evidence.h",
        "exact_profile_admission.h",
        "filesystem_capsule_restore_admission_repository.h",
        "filesystem_capsule_restore_evidence_repository.h",
        "filesystem_dpm_event_log.h",
        "filesystem_exact_profile_admission_repository.h",
        "filesystem_session_checkpoint_repository.h",
        "fresh_worker_process.h",
        "fresh_worker_protocol.h",
        "one_shot_dpm_projector.h",
        "session_checkpoint.h",
    ],
    deps = [
        "//runtime/core:exact_litert_profile_builder",
        "//runtime/core:session_advanced",
        "//runtime/core:session_handoff_codec",
        "//runtime/core:session_handoff_identity",
        "//runtime/engine:cpu_affinity_utils",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/engine:exact_litert_decode",
        "//runtime/engine:exact_litert_profile",
        "//runtime/engine:io_types",
        "//runtime/engine:session_handoff",
        "//runtime/engine:session_handoff_capability",
        "//runtime/engine:session_handoff_codec_contract",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:session_handoff_runtime",
        "//runtime/executor/litert:state",
        "//runtime/platform:runtime_artifact_identity",
        "//runtime/platform/hash",
        "//runtime/proto:sampler_params_cc_proto",
        "//runtime/util:byte_stream",
        "//runtime/util:convert_tensor_buffer",
        "//support/tokenizer",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/cleanup",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/functional:any_invocable",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/time",
        "@com_google_absl//absl/types:span",
        "@litert//litert/cc:litert_tensor_buffer",
        "@nlohmann_json//:json",
    ],
)

# Optional domain schema built above the schema-agnostic DPM core. Keeping it
# separate prevents the engine from hard-coding one projection representation.
cc_library(
    name = "dpm_fact_map",
    srcs = ["dpm_fact_map.cc"],
    hdrs = ["dpm_fact_map.h"],
    deps = [
        ":dpm",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_absl//absl/types:span",
        "@nlohmann_json//:json",
    ],
)

# Product adapter for the pure structured-memory reducer. This stays above the
# schema-agnostic DPM engine while implementing its projection-provider
# boundary, so applications can opt into dpm.factmap without coupling that
# representation into the low-level LiteRT-LM executor.
cc_library(
    name = "deterministic_dpm_fact_map_projector",
    srcs = ["deterministic_dpm_fact_map_projector.cc"],
    hdrs = ["deterministic_dpm_fact_map_projector.h"],
    deps = [
        ":dpm",
        ":dpm_fact_map",
        "//runtime/engine:session_handoff",
        "//runtime/platform/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_test(
    name = "deterministic_dpm_fact_map_projector_test",
    srcs = ["deterministic_dpm_fact_map_projector_test.cc"],
    deps = [
        ":deterministic_dpm_fact_map_projector",
        ":dpm",
        "//runtime/engine:session_handoff",
        "//runtime/platform/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:string_view",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dpm_fact_map_test",
    srcs = ["dpm_fact_map_test.cc"],
    data = glob(["testdata/factmap/*"]),
    deps = [
        ":dpm_fact_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "dpm_main",
    srcs = ["dpm_main.cc"],
    additional_linker_inputs = select({
        "@platforms//os:windows": [
            "@litert//litert/c:windows_exported_symbols.def",
        ],
        "@platforms//os:linux": ["//runtime/engine:litert_lm_main.exported_symbols"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@litert//litert:litert_runtime_link_mode_dynamic": [],
        "@platforms//os:ios": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:macos": ["-Wl,-exported_symbol,_LiteRt*"],
        "@platforms//os:windows": [
            "/DEF:$(location @litert//litert/c:windows_exported_symbols.def)",
        ],
        "@platforms//os:linux": ["-Wl,--dynamic-list=$(location //runtime/engine:litert_lm_main.exported_symbols)"],
        "//conditions:default": ["-Wl,--export-dynamic-symbol=LiteRt*"],
    }) + select({
        "@platforms//os:android": ["-lEGL", "-lGLESv3"],
        "//conditions:default": [],
    }),
    deps = [
        ":deterministic_dpm_fact_map_projector",
        ":dpm",
        ":dpm_fact_map",
        "//runtime/engine:engine_factory",
        "//runtime/engine:engine_impl_selected",
        "//runtime/engine:engine_interface",
        "//runtime/engine:engine_settings",
        "//runtime/executor:executor_settings_base",
        "//runtime/executor:llm_executor_settings",
        "//runtime/executor:session_handoff_runtime",
        "//runtime/platform:runtime_artifact_identity",
        "//runtime/platform/hash",
        "@com_google_absl//absl/base:log_severity",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/log:globals",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
        "@nlohmann_json//:json",
    ],
)
