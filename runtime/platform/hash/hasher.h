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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_HASH_HASHER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_HASH_HASHER_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {

// Stable 32-byte digest value shared by handoff and higher-level artifact
// identities. The producing algorithm is part of the enclosing protocol; the
// session-handoff protocol uses SHA-256 exclusively.
struct Hash256 {
  std::array<uint8_t, 32> bytes{};

  bool operator==(const Hash256& other) const { return bytes == other.bytes; }
  bool operator!=(const Hash256& other) const { return !(*this == other); }
  bool operator<(const Hash256& other) const {
    return std::memcmp(bytes.data(), other.bytes.data(), bytes.size()) < 0;
  }

  // Lowercase hex. 64 characters.
  std::string ToHex() const;

  // Inverse of ToHex. On malformed input, returns the zero hash and sets
  // `*ok` to false when `ok` is non-null.
  static Hash256 FromHex(absl::string_view hex, bool* ok);

  template <typename H>
  friend H AbslHashValue(H h, const Hash256& value) {
    return H::combine_contiguous(std::move(h), value.bytes.data(),
                                 value.bytes.size());
  }
};

// Minimal streaming digest interface. Algorithms are selected by the concrete
// type so no unrelated hash factory or platform stack is pulled into handoff.
class Hasher {
 public:
  virtual ~Hasher() = default;

  virtual void Update(absl::string_view data) = 0;
  virtual Hash256 Finalize() = 0;

  Hash256 OneShot(absl::string_view data) {
    Update(data);
    return Finalize();
  }
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_PLATFORM_HASH_HASHER_H_
