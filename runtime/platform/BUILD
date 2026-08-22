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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

objc_library(
    name = "apple_metal_identity",
    srcs = ["apple_metal_identity.mm"],
    hdrs = ["apple_metal_identity.h"],
    sdk_frameworks = [
        "Foundation",
        "Metal",
    ],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ],
)

cc_library(
    name = "runtime_artifact_identity",
    srcs = ["runtime_artifact_identity.cc"],
    hdrs = ["runtime_artifact_identity.h"],
    deps = [
        "//runtime/executor:session_handoff_runtime",
        "//runtime/platform/hash",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:status_macros",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:string_view",
    ] + select({
        "@platforms//os:macos": [":apple_metal_identity"],
        "@platforms//os:ios": [":apple_metal_identity"],
        "//conditions:default": [],
    }),
)
