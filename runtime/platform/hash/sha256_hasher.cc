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

#include "runtime/platform/hash/sha256_hasher.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {
namespace {

constexpr uint32_t kInitialState[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

constexpr uint32_t kRoundConstants[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu,
    0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u,
    0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u,
    0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u,
    0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u,
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u,
    0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u, 0x1e376c08u,
    0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu,
    0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

inline uint32_t Rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}
inline uint32_t Choose(uint32_t e, uint32_t f, uint32_t g) {
  return (e & f) ^ (~e & g);
}
inline uint32_t Majority(uint32_t a, uint32_t b, uint32_t c) {
  return (a & b) ^ (a & c) ^ (b & c);
}
inline uint32_t Sigma0(uint32_t x) {
  return Rotr(x, 2) ^ Rotr(x, 13) ^ Rotr(x, 22);
}
inline uint32_t Sigma1(uint32_t x) {
  return Rotr(x, 6) ^ Rotr(x, 11) ^ Rotr(x, 25);
}
inline uint32_t Gamma0(uint32_t x) {
  return Rotr(x, 7) ^ Rotr(x, 18) ^ (x >> 3);
}
inline uint32_t Gamma1(uint32_t x) {
  return Rotr(x, 17) ^ Rotr(x, 19) ^ (x >> 10);
}

}  // namespace

Sha256Hasher::Sha256Hasher() {
  for (size_t i = 0; i < 8; ++i) state_[i] = kInitialState[i];
  buffer_.fill(0);
}

void Sha256Hasher::ProcessBlock(const uint8_t block[64]) {
  uint32_t w[64];
  for (size_t i = 0; i < 16; ++i) {
    w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
           (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
           (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
           (static_cast<uint32_t>(block[i * 4 + 3]));
  }
  for (size_t i = 16; i < 64; ++i) {
    w[i] = Gamma1(w[i - 2]) + w[i - 7] + Gamma0(w[i - 15]) + w[i - 16];
  }

  uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
  uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];

  for (size_t i = 0; i < 64; ++i) {
    const uint32_t t1 = h + Sigma1(e) + Choose(e, f, g) + kRoundConstants[i] +
                        w[i];
    const uint32_t t2 = Sigma0(a) + Majority(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

void Sha256Hasher::Update(absl::string_view data) {
  if (finalized_) return;
  total_len_bits_ += static_cast<uint64_t>(data.size()) * 8;
  size_t offset = 0;
  if (buffer_len_ > 0) {
    const size_t to_copy = std::min<size_t>(64 - buffer_len_, data.size());
    std::memcpy(buffer_.data() + buffer_len_, data.data(), to_copy);
    buffer_len_ += to_copy;
    offset += to_copy;
    if (buffer_len_ == 64) {
      ProcessBlock(buffer_.data());
      buffer_len_ = 0;
    }
  }
  while (data.size() - offset >= 64) {
    ProcessBlock(reinterpret_cast<const uint8_t*>(data.data() + offset));
    offset += 64;
  }
  if (offset < data.size()) {
    const size_t remain = data.size() - offset;
    std::memcpy(buffer_.data(), data.data() + offset, remain);
    buffer_len_ = remain;
  }
}

Hash256 Sha256Hasher::Finalize() {
  Hash256 digest;
  if (finalized_) {
    // Returning the zero hash on misuse is benign; the caller-visible
    // contract is that Finalize is one-shot.
    return digest;
  }
  finalized_ = true;

  // Append the 0x80 separator byte.
  buffer_[buffer_len_++] = 0x80;
  if (buffer_len_ > 56) {
    while (buffer_len_ < 64) buffer_[buffer_len_++] = 0;
    ProcessBlock(buffer_.data());
    buffer_len_ = 0;
  }
  while (buffer_len_ < 56) buffer_[buffer_len_++] = 0;

  // Big-endian total length in bits.
  for (int i = 7; i >= 0; --i) {
    buffer_[buffer_len_++] =
        static_cast<uint8_t>((total_len_bits_ >> (i * 8)) & 0xff);
  }
  ProcessBlock(buffer_.data());

  for (size_t i = 0; i < 8; ++i) {
    digest.bytes[i * 4] = static_cast<uint8_t>(state_[i] >> 24);
    digest.bytes[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16);
    digest.bytes[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8);
    digest.bytes[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
  }
  return digest;
}

}  // namespace litert::lm
