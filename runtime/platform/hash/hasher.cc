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

#include "runtime/platform/hash/hasher.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {
namespace {

constexpr char kHexDigits[] = "0123456789abcdef";

int FromHexDigit(char value) {
  if (value >= '0' && value <= '9') return value - '0';
  if (value >= 'a' && value <= 'f') return value - 'a' + 10;
  if (value >= 'A' && value <= 'F') return value - 'A' + 10;
  return -1;
}

}  // namespace

std::string Hash256::ToHex() const {
  std::string output(64, '\0');
  for (size_t i = 0; i < bytes.size(); ++i) {
    output[i * 2] = kHexDigits[(bytes[i] >> 4) & 0x0f];
    output[i * 2 + 1] = kHexDigits[bytes[i] & 0x0f];
  }
  return output;
}

Hash256 Hash256::FromHex(absl::string_view hex, bool* ok) {
  Hash256 output;
  if (hex.size() != output.bytes.size() * 2) {
    if (ok != nullptr) *ok = false;
    return output;
  }
  for (size_t i = 0; i < output.bytes.size(); ++i) {
    const int high = FromHexDigit(hex[i * 2]);
    const int low = FromHexDigit(hex[i * 2 + 1]);
    if (high < 0 || low < 0) {
      if (ok != nullptr) *ok = false;
      return Hash256{};
    }
    output.bytes[i] = static_cast<uint8_t>((high << 4) | low);
  }
  if (ok != nullptr) *ok = true;
  return output;
}

}  // namespace litert::lm
