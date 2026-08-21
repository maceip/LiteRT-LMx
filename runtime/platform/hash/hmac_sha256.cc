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

#include "runtime/platform/hash/hmac_sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/sha256_hasher.h"

namespace litert::lm {
namespace {

constexpr size_t kSha256BlockSize = 64;

}  // namespace

Hash256 HmacSha256(absl::string_view key,
                   std::initializer_list<absl::string_view> parts) {
  std::array<uint8_t, kSha256BlockSize> normalized_key{};
  if (key.size() > normalized_key.size()) {
    Sha256Hasher key_hasher;
    key_hasher.Update(key);
    const Hash256 key_hash = key_hasher.Finalize();
    std::memcpy(normalized_key.data(), key_hash.bytes.data(),
                key_hash.bytes.size());
  } else if (!key.empty()) {
    std::memcpy(normalized_key.data(), key.data(), key.size());
  }

  std::array<char, kSha256BlockSize> inner_pad{};
  std::array<char, kSha256BlockSize> outer_pad{};
  for (size_t index = 0; index < normalized_key.size(); ++index) {
    inner_pad[index] = static_cast<char>(normalized_key[index] ^ 0x36u);
    outer_pad[index] = static_cast<char>(normalized_key[index] ^ 0x5cu);
  }

  Sha256Hasher inner;
  inner.Update(absl::string_view(inner_pad.data(), inner_pad.size()));
  for (absl::string_view part : parts) inner.Update(part);
  const Hash256 inner_hash = inner.Finalize();

  Sha256Hasher outer;
  outer.Update(absl::string_view(outer_pad.data(), outer_pad.size()));
  outer.Update(absl::string_view(
      reinterpret_cast<const char*>(inner_hash.bytes.data()),
      inner_hash.bytes.size()));
  return outer.Finalize();
}

bool ConstantTimeHashEquals(const Hash256& left, const Hash256& right) {
  uint8_t difference = 0;
  for (size_t index = 0; index < left.bytes.size(); ++index) {
    difference |= left.bytes[index] ^ right.bytes[index];
  }
  return difference == 0;
}

}  // namespace litert::lm
