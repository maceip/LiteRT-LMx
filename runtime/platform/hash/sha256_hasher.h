// Copyright 2026 Google LLC.
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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_HASH_SHA256_HASHER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_HASH_SHA256_HASHER_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// SHA-256 (FIPS 180-4). Pure C++ streaming implementation used as the sole
// digest primitive by the session-handoff protocol.
//
// A FIPS-validated deployment should swap this implementation for
// BoringSSL EVP at build time; the on-wire bytes and Hash256 surface are
// algorithm-stable.
class Sha256Hasher : public Hasher {
 public:
  Sha256Hasher();
  ~Sha256Hasher() override = default;

  void Update(absl::string_view data) override;
  Hash256 Finalize() override;

 private:
  void ProcessBlock(const uint8_t block[64]);

  std::array<uint32_t, 8> state_;
  std::array<uint8_t, 64> buffer_;
  size_t buffer_len_ = 0;
  uint64_t total_len_bits_ = 0;
  bool finalized_ = false;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_HASH_SHA256_HASHER_H_
