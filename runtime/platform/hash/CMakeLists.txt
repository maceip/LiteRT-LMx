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

add_litertlm_library(runtime_platform_hash STATIC
  hasher.cc
  hmac_sha256.cc
  sha256_hasher.cc
)
add_library(LiteRTLM::Runtime::Platform::Hash ALIAS runtime_platform_hash)

target_include_directories(runtime_platform_hash
  PUBLIC
    ${LITERTLM_GENERATED_SRC_DIR}
    ${LITERTLM_INCLUDE_PATHS}
)

target_link_libraries(runtime_platform_hash
  PUBLIC
    LITERTLM_DEPS
)
