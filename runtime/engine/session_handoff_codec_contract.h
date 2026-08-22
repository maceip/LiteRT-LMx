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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_CODEC_CONTRACT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_CODEC_CONTRACT_H_

#include <array>
#include <cstdint>

namespace litert::lm {

// Single source of truth for the on-wire identities consumed by the
// LRTSESS1 envelope codec and the nested LRTST001 LiteRT-state codec.
inline constexpr std::array<char, 8> kSessionHandoffEnvelopeMagic = {
    'L', 'R', 'T', 'S', 'E', 'S', 'S', '1'};
inline constexpr uint32_t kSessionHandoffEnvelopeVersion = 1;

inline constexpr std::array<char, 8> kLiteRtStateSnapshotMagic = {
    'L', 'R', 'T', 'S', 'T', '0', '0', '1'};
inline constexpr uint32_t kLiteRtStateSnapshotVersion = 1;

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_SESSION_HANDOFF_CODEC_CONTRACT_H_
