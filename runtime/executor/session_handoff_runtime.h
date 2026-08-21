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
#include <string>

namespace litert::lm {

// Concrete loaded-runtime class whose artifacts can be measured by the
// platform implementation. A backend must not claim a class until its actual
// delegate/device evidence is available; unsupported classes fail closed.
enum class SessionHandoffRuntimeClass : uint8_t {
  kLiteRtCpu = 1,
};

// Canonical description captured from the live concrete executor after model
// compilation and all executor setting updates. This is not a caller label.
struct SessionHandoffRuntimeProfile {
  SessionHandoffRuntimeClass runtime_class;
  std::string canonical_profile;

  // Vocabulary dimension read from the loaded compiled decode-logits tensor.
  // Exact handoff admission compares this executor-owned evidence with the
  // tokenizer loaded by the Engine; zero means the evidence is unavailable.
  int32_t logits_vocabulary_size = 0;

  // Address of a function dispatched by the live LiteRT runtime proxy. The
  // platform layer uses it only to locate and measure the loaded image; the
  // ASLR-dependent address itself is never hashed.
  uintptr_t runtime_code_anchor = 0;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_EXECUTOR_SESSION_HANDOFF_RUNTIME_H_
