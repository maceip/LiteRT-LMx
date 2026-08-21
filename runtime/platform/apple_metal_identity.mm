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

#include "runtime/platform/apple_metal_identity.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <TargetConditionals.h>

#include <array>
#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

namespace litert::lm {
namespace {

void AppendU8(uint8_t value, std::string* output) {
  output->push_back(static_cast<char>(value));
}

void AppendU32(uint32_t value, std::string* output) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendU64(uint64_t value, std::string* output) {
  for (int shift = 56; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

void AppendBytes(absl::string_view value, std::string* output) {
  AppendU64(value.size(), output);
  output->append(value.data(), value.size());
}

absl::StatusOr<std::string> Utf8String(NSString* value,
                                       absl::string_view label) {
  if (value == nil || value.length == 0) {
    return absl::FailedPreconditionError(
        std::string(label.data(), label.size()) +
        " is unavailable on the selected MTLDevice.");
  }
  const char* bytes = value.UTF8String;
  if (bytes == nullptr) {
    return absl::FailedPreconditionError(
        std::string(label.data(), label.size()) +
        " has no UTF-8 representation.");
  }
  return std::string(bytes);
}

}  // namespace

absl::StatusOr<std::string> DeriveMacOsMetalDeviceIdentity(
    const void* metal_device, const void* metal_command_queue) {
#if !TARGET_OS_OSX
  (void)metal_device;
  (void)metal_command_queue;
  return absl::UnimplementedError(
      "Exact Metal device identity is currently defined only for macOS.");
#else
  if (metal_device == nullptr || metal_command_queue == nullptr) {
    return absl::FailedPreconditionError(
        "Selected Metal device or command queue evidence is null.");
  }

  @autoreleasepool {
    id<MTLDevice> device = (__bridge id<MTLDevice>)metal_device;
    id<MTLCommandQueue> queue =
        (__bridge id<MTLCommandQueue>)metal_command_queue;
    if (device == nil || queue == nil || queue.device != device) {
      return absl::FailedPreconditionError(
          "The selected Metal command queue is not owned by the measured "
          "MTLDevice.");
    }
    if (@available(macOS 10.15, *)) {
      if (![device respondsToSelector:@selector(supportsFamily:)]) {
        return absl::UnimplementedError(
            "The selected MTLDevice does not expose Metal-family evidence.");
      }
    } else {
      return absl::UnimplementedError(
          "The macOS Metal-family API is unavailable on this OS.");
    }

    const uint64_t registry_id = device.registryID;
    if (registry_id == 0) {
      return absl::FailedPreconditionError(
          "The selected MTLDevice has no stable registry identifier.");
    }
    absl::StatusOr<std::string> name = Utf8String(device.name, "MTLDevice name");
    if (!name.ok()) return name.status();

    NSString* architecture_name = nil;
    if (@available(macOS 10.15, *)) {
      architecture_name = device.architecture.name;
    }
    absl::StatusOr<std::string> architecture =
        Utf8String(architecture_name, "MTLDevice architecture name");
    if (!architecture.ok()) return architecture.status();

    std::string identity;
    AppendBytes("LITERT_LM_MACOS_METAL_DEVICE_V1", &identity);
    AppendU64(registry_id, &identity);
    AppendBytes(*name, &identity);
    AppendBytes(*architecture, &identity);
    AppendU8(device.isLowPower ? 1 : 0, &identity);
    AppendU8(device.isHeadless ? 1 : 0, &identity);
    AppendU8(device.isRemovable ? 1 : 0, &identity);
    if (@available(macOS 10.15, *)) {
      AppendU8(device.hasUnifiedMemory ? 1 : 0, &identity);
    }
    AppendU64(device.maxBufferLength, &identity);
    AppendU64(device.recommendedMaxWorkingSetSize, &identity);
    const MTLSize max_threads = device.maxThreadsPerThreadgroup;
    AppendU64(max_threads.width, &identity);
    AppendU64(max_threads.height, &identity);
    AppendU64(max_threads.depth, &identity);
    AppendU32(static_cast<uint32_t>(device.readWriteTextureSupport),
              &identity);
    AppendU32(static_cast<uint32_t>(device.argumentBuffersSupport), &identity);

    // MTLGPUFamily values are grouped by Apple (1000), Mac (2000), Common
    // (3000), Mac Catalyst (4000), and Metal feature family (5000). Query a
    // versioned bounded range in every namespace so the identity records the
    // actual capabilities instead of inferring a family from the device name.
    // A future family outside this schema receives a new identity schema; the
    // exact macOS build is independently bound by runtime artifact identity.
    constexpr std::array<NSInteger, 5> kFamilyBases = {
        1000, 2000, 3000, 4000, 5000};
    constexpr NSInteger kFamilyValuesPerNamespace = 32;
    AppendU32(static_cast<uint32_t>(kFamilyBases.size()), &identity);
    AppendU32(static_cast<uint32_t>(kFamilyValuesPerNamespace), &identity);
    for (NSInteger base : kFamilyBases) {
      AppendU32(static_cast<uint32_t>(base), &identity);
      for (NSInteger offset = 1; offset <= kFamilyValuesPerNamespace;
           ++offset) {
        const auto family = static_cast<MTLGPUFamily>(base + offset);
        AppendU8([device supportsFamily:family] ? 1 : 0, &identity);
      }
    }
    return identity;
  }
#endif
}

}  // namespace litert::lm
