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

add_subdirectory(hash)

set(PLATFORM_SRCS
  runtime_artifact_identity.cc
)

if(APPLE)
  list(APPEND PLATFORM_SRCS apple_metal_identity.mm)
endif()

add_litertlm_library(runtime_platform_runtime_artifact_identity STATIC
  ${PLATFORM_SRCS}
)
add_library(LiteRTLM::Runtime::Platform::RuntimeArtifactIdentity
  ALIAS runtime_platform_runtime_artifact_identity)

target_include_directories(runtime_platform_runtime_artifact_identity
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_platform_runtime_artifact_identity
  PUBLIC
    LiteRTLM::Runtime::Platform::Hash
    LITERTLM_DEPS
)

if(APPLE)
  target_link_libraries(runtime_platform_runtime_artifact_identity
    PUBLIC
      "-framework Foundation"
      "-framework Metal"
  )
endif()

add_library(runtime_platform_libs INTERFACE)
add_library(LiteRTLM::Runtime::Platform ALIAS runtime_platform_libs)

target_link_libraries(runtime_platform_libs INTERFACE
  LiteRTLM::Runtime::Platform::Hash
  LiteRTLM::Runtime::Platform::RuntimeArtifactIdentity
)
