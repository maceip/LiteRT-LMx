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

#include "runtime/platform/runtime_artifact_identity.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/sha256_hasher.h"

#if defined(__linux__) || defined(__ANDROID__)
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sched.h>
#include <sys/auxv.h>
#include <sys/utsname.h>
#include <unistd.h>
#if defined(__ANDROID__)
#include <sys/system_properties.h>
#endif
#endif

#if defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach/vm_statistics.h>
#include <mach-o/dyld.h>
#include <mach-o/loader.h>
#include <mach/vm_prot.h>
#include <sys/sysctl.h>
#include "runtime/platform/apple_metal_identity.h"
#endif

namespace litert::lm {
namespace {

constexpr absl::string_view kRuntimeArtifactDomain =
    "LITERT_LM_LOADED_RUNTIME_ARTIFACT_V4";

void HashU32(uint32_t value, Sha256Hasher* hasher) {
  std::array<char, 4> bytes;
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (24 - i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void HashU64(uint64_t value, Sha256Hasher* hasher) {
  std::array<char, 8> bytes;
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = static_cast<char>((value >> (56 - i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(bytes.data(), bytes.size()));
}

void HashFrame(absl::string_view value, Sha256Hasher* hasher) {
  HashU64(value.size(), hasher);
  hasher->Update(value);
}

#if defined(__APPLE__)

constexpr uint32_t kMaximumLoadCommandsBytes = 64 * 1024 * 1024;
constexpr uint64_t kMaximumMeasuredSegmentBytes = uint64_t{8} * 1024 * 1024 *
                                                  1024;
constexpr uint32_t kMaximumCodeSignatureBytes = 512 * 1024 * 1024;
constexpr size_t kMaximumSysctlBytes = 1024 * 1024;

// Code-signing blobs use network byte order. These wire constants and field
// offsets are stable parts of Apple's embedded-signature format; duplicating
// the small subset used here avoids a dependency on private Kernel headers.
constexpr uint32_t kCsMagicCodeDirectory = 0xfade0c02;
constexpr uint32_t kCsMagicEmbeddedSignature = 0xfade0cc0;
constexpr uint32_t kCsSlotCodeDirectory = 0;
constexpr uint32_t kCsSlotAlternateCodeDirectoryFirst = 0x1000;
constexpr uint32_t kCsSlotAlternateCodeDirectoryLimit = 0x1005;
constexpr uint32_t kCsEarliestVersion = 0x20001;
constexpr uint32_t kCsSupportsScatter = 0x20100;
constexpr uint32_t kCsSupportsTeamId = 0x20200;
constexpr uint32_t kCsSupportsCodeLimit64 = 0x20300;
constexpr uint32_t kCsSupportsExecSegment = 0x20400;
constexpr uint32_t kCsSupportsRuntime = 0x20500;
constexpr uint32_t kCsSupportsLinkage = 0x20600;
constexpr size_t kCsGenericBlobHeaderSize = 8;
constexpr size_t kCsSuperBlobHeaderSize = 12;
constexpr size_t kCsBlobIndexSize = 8;
constexpr size_t kCsCodeDirectoryEarliestSize = 44;
constexpr uint32_t kMaximumCodeSignatureBlobs = 4096;

uint32_t ReadBigEndianU32(const uint8_t* bytes) {
  return (static_cast<uint32_t>(bytes[0]) << 24) |
         (static_cast<uint32_t>(bytes[1]) << 16) |
         (static_cast<uint32_t>(bytes[2]) << 8) |
         static_cast<uint32_t>(bytes[3]);
}

uint64_t ReadBigEndianU64(const uint8_t* bytes) {
  return (static_cast<uint64_t>(ReadBigEndianU32(bytes)) << 32) |
         ReadBigEndianU32(bytes + 4);
}

absl::Status ValidateMappedRange(uintptr_t address, uint64_t size,
                                 bool require_immutable,
                                 absl::string_view description) {
  if (size == 0) return absl::OkStatus();
  if (size > std::numeric_limits<uintptr_t>::max() - address) {
    return absl::FailedPreconditionError(
        absl::StrCat(description, " address range overflows."));
  }
  const mach_vm_address_t end =
      static_cast<mach_vm_address_t>(address + size);
  mach_vm_address_t cursor = static_cast<mach_vm_address_t>(address);
  while (cursor < end) {
    mach_vm_address_t region_address = cursor;
    mach_vm_size_t region_size = 0;
    vm_region_basic_info_data_64_t region_info = {};
    mach_msg_type_number_t region_info_count =
        VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object_name = MACH_PORT_NULL;
    const kern_return_t result = mach_vm_region(
        mach_task_self(), &region_address, &region_size,
        VM_REGION_BASIC_INFO_64,
        reinterpret_cast<vm_region_info_t>(&region_info), &region_info_count,
        &object_name);
    if (object_name != MACH_PORT_NULL) {
      (void)mach_port_deallocate(mach_task_self(), object_name);
    }
    if (result != KERN_SUCCESS) {
      return absl::UnimplementedError(absl::StrCat(
          "Cannot inspect ", description,
          " memory protections; mach_vm_region returned ", result, "."));
    }
    if (region_size == 0 ||
        region_size > std::numeric_limits<mach_vm_address_t>::max() -
                          region_address) {
      return absl::FailedPreconditionError(
          absl::StrCat(description, " has an invalid VM region."));
    }
    const mach_vm_address_t region_end = region_address + region_size;
    if (region_address > cursor || region_end <= cursor) {
      return absl::FailedPreconditionError(
          absl::StrCat(description, " is not completely mapped."));
    }
    if ((region_info.protection & VM_PROT_READ) == 0) {
      return absl::UnimplementedError(
          absl::StrCat(description, " contains an unreadable range."));
    }
    if (require_immutable &&
        ((region_info.protection | region_info.max_protection) &
         VM_PROT_WRITE) != 0) {
      return absl::UnimplementedError(
          absl::StrCat(description,
                       " contains a writable or write-capable range."));
    }
    cursor = std::min(end, region_end);
  }
  return absl::OkStatus();
}

absl::Status FaultAndValidateCodeSignedPages(
    uintptr_t address, uint64_t size, absl::string_view description) {
  if (size == 0) return absl::OkStatus();
  if (vm_page_size == 0 ||
      (vm_page_size & (static_cast<vm_size_t>(vm_page_size) - 1)) != 0) {
    return absl::UnimplementedError(
        "Loaded runtime identity cannot determine the VM page size.");
  }
  if (size > std::numeric_limits<uintptr_t>::max() - address) {
    return absl::FailedPreconditionError(
        absl::StrCat(description, " page range overflows."));
  }
  ABSL_RETURN_IF_ERROR(ValidateMappedRange(
      address, size, /*require_immutable=*/false, description));
  const uintptr_t end = address + size;
  const uintptr_t page_size = static_cast<uintptr_t>(vm_page_size);

  // A query does not fault an absent page. Read one byte from every mapped VM
  // page first so a successful query describes the loaded image page rather
  // than merely reporting that it has not been instantiated.
  uint8_t observed = 0;
  uintptr_t cursor = address;
  while (cursor < end) {
    observed = static_cast<uint8_t>(
        observed ^ *reinterpret_cast<volatile const uint8_t*>(cursor));
    const uintptr_t page = cursor - (cursor % page_size);
    if (page_size > std::numeric_limits<uintptr_t>::max() - page) {
      return absl::FailedPreconditionError(
          absl::StrCat(description, " page iteration overflows."));
    }
    const uintptr_t next_page = page + page_size;
    if (next_page >= end) break;
    cursor = next_page;
  }
  (void)observed;

  uintptr_t page = address - (address % page_size);
  while (page < end) {
    integer_t disposition = 0;
    integer_t reference_count = 0;
    const kern_return_t result = mach_vm_page_query(
        mach_task_self(), static_cast<mach_vm_offset_t>(page), &disposition,
        &reference_count);
    (void)reference_count;
    if (result != KERN_SUCCESS) {
      return absl::UnimplementedError(absl::StrCat(
          "Cannot inspect ", description, " code-signing state; "
          "mach_vm_page_query returned ",
          result, "."));
    }
    if ((disposition & VM_PAGE_QUERY_PAGE_PRESENT) == 0 ||
        (disposition & VM_PAGE_QUERY_PAGE_CS_VALIDATED) == 0 ||
        (disposition & VM_PAGE_QUERY_PAGE_CS_TAINTED) != 0) {
      return absl::UnimplementedError(
          absl::StrCat(description,
                       " lacks untainted kernel-validated code-signature "
                       "page evidence."));
    }
    if (page_size > std::numeric_limits<uintptr_t>::max() - page) {
      return absl::FailedPreconditionError(
          absl::StrCat(description, " page iteration overflows."));
    }
    page += page_size;
  }
  return absl::OkStatus();
}

absl::Status ValidateStrongCodeDirectory(const uint8_t* bytes, size_t size,
                                         uint64_t expected_code_limit) {
  if (size < kCsCodeDirectoryEarliestSize ||
      ReadBigEndianU32(bytes) != kCsMagicCodeDirectory ||
      ReadBigEndianU32(bytes + 4) != size) {
    return absl::FailedPreconditionError(
        "Loaded image has a malformed CodeDirectory.");
  }
  const uint32_t version = ReadBigEndianU32(bytes + 8);
  if (version < kCsEarliestVersion || version > kCsSupportsLinkage) {
    return absl::UnimplementedError(
        "Loaded image uses an unsupported CodeDirectory version.");
  }
  const uint32_t hash_offset = ReadBigEndianU32(bytes + 16);
  const uint32_t identifier_offset = ReadBigEndianU32(bytes + 20);
  const uint32_t special_slots = ReadBigEndianU32(bytes + 24);
  const uint32_t code_slots = ReadBigEndianU32(bytes + 28);
  uint64_t code_limit = ReadBigEndianU32(bytes + 32);
  const uint8_t hash_size = bytes[36];
  const uint8_t hash_type = bytes[37];
  const uint8_t page_shift = bytes[39];
  if (ReadBigEndianU32(bytes + 40) != 0) {
    return absl::FailedPreconditionError(
        "Loaded image CodeDirectory has a nonzero reserved field.");
  }
  size_t fixed_size = kCsCodeDirectoryEarliestSize;
  uint32_t team_id_offset = 0;
  if (version >= kCsSupportsScatter) {
    if (size < 48) {
      return absl::FailedPreconditionError(
          "Loaded image has a truncated scatter-capable CodeDirectory.");
    }
    if (ReadBigEndianU32(bytes + 44) != 0) {
      return absl::UnimplementedError(
          "Scattered CodeDirectory coverage is not supported for exact "
          "loaded runtime identity.");
    }
    fixed_size = 48;
  }
  if (version >= kCsSupportsTeamId) {
    if (size < 52) {
      return absl::FailedPreconditionError(
          "Loaded image has a truncated team-capable CodeDirectory.");
    }
    team_id_offset = ReadBigEndianU32(bytes + 48);
    fixed_size = 52;
  }
  if (version >= kCsSupportsCodeLimit64) {
    if (size < 64) {
      return absl::FailedPreconditionError(
          "Loaded image has a truncated 64-bit CodeDirectory.");
    }
    if (ReadBigEndianU32(bytes + 52) != 0) {
      return absl::FailedPreconditionError(
          "Loaded image CodeDirectory has a nonzero reserved field.");
    }
    const uint64_t code_limit_64 = ReadBigEndianU64(bytes + 56);
    if (code_limit_64 != 0) code_limit = code_limit_64;
    fixed_size = 64;
  }
  if (version >= kCsSupportsExecSegment) {
    if (size < 88) {
      return absl::FailedPreconditionError(
          "Loaded image has a truncated executable-segment CodeDirectory.");
    }
    fixed_size = 88;
  }
  if (version >= kCsSupportsRuntime) {
    if (size < 96) {
      return absl::FailedPreconditionError(
          "Loaded image has a truncated runtime-capable CodeDirectory.");
    }
    if (ReadBigEndianU32(bytes + 92) != 0) {
      return absl::UnimplementedError(
          "Pre-encryption CodeDirectory coverage is not supported for exact "
          "loaded runtime identity.");
    }
    fixed_size = 96;
  }
  if (version >= kCsSupportsLinkage) {
    if (size < 108) {
      return absl::FailedPreconditionError(
          "Loaded image has a truncated linkage-capable CodeDirectory.");
    }
    if (bytes[96] != 0 || bytes[97] != 0 || bytes[98] != 0 ||
        bytes[99] != 0 || ReadBigEndianU32(bytes + 100) != 0 ||
        ReadBigEndianU32(bytes + 104) != 0) {
      return absl::UnimplementedError(
          "Linkage CodeDirectory coverage is not supported for exact loaded "
          "runtime identity.");
    }
    fixed_size = 108;
  }
  if (code_limit != expected_code_limit) {
    return absl::UnimplementedError(
        "Loaded image CodeDirectory does not cover every byte before its "
        "embedded signature.");
  }

  uint8_t expected_hash_size = 0;
  switch (hash_type) {
    case 2:  // SHA-256.
      expected_hash_size = 32;
      break;
    case 4:  // SHA-384.
      expected_hash_size = 48;
      break;
    case 1:  // SHA-1.
    case 3:  // Truncated SHA-256.
      return absl::UnimplementedError(
          "Loaded image has a weak CodeDirectory hash algorithm.");
    default:
      return absl::UnimplementedError(
          "Loaded image uses an unsupported CodeDirectory hash algorithm.");
  }
  if (hash_size != expected_hash_size) {
    return absl::FailedPreconditionError(
        "Loaded image CodeDirectory has an invalid hash size.");
  }
  if (page_shift != 12 && page_shift != 14) {
    return absl::UnimplementedError(
        "Loaded image uses an unsupported CodeDirectory page size.");
  }
  const uint64_t page_size = uint64_t{1} << page_shift;
  const uint64_t expected_code_slots =
      code_limit / page_size + (code_limit % page_size != 0 ? 1 : 0);
  if (expected_code_slots != code_slots) {
    return absl::FailedPreconditionError(
        "Loaded image CodeDirectory does not contain a complete page-hash "
        "array.");
  }
  const uint64_t special_hash_bytes =
      static_cast<uint64_t>(special_slots) * hash_size;
  const uint64_t code_hash_bytes =
      static_cast<uint64_t>(code_slots) * hash_size;
  if (hash_offset > size || special_hash_bytes > hash_offset ||
      hash_offset - special_hash_bytes < fixed_size ||
      code_hash_bytes != size - hash_offset) {
    return absl::FailedPreconditionError(
        "Loaded image CodeDirectory hash slots are out of bounds.");
  }
  const size_t variable_data_end =
      static_cast<size_t>(hash_offset - special_hash_bytes);
  auto validate_string_offset =
      [bytes, fixed_size, variable_data_end](uint32_t offset,
                                             absl::string_view description)
      -> absl::Status {
    if (offset < fixed_size || offset >= variable_data_end ||
        std::memchr(bytes + offset, '\0', variable_data_end - offset) ==
            nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat("Loaded image CodeDirectory has an invalid ",
                       description, "."));
    }
    return absl::OkStatus();
  };
  ABSL_RETURN_IF_ERROR(
      validate_string_offset(identifier_offset, "identifier"));
  if (team_id_offset != 0) {
    ABSL_RETURN_IF_ERROR(
        validate_string_offset(team_id_offset, "team identifier"));
  }
  return absl::OkStatus();
}

absl::Status ValidateEmbeddedCodeSignature(const uint8_t* bytes, size_t size,
                                           uint64_t expected_code_limit) {
  if (bytes == nullptr || size < kCsGenericBlobHeaderSize) {
    return absl::FailedPreconditionError(
        "Loaded image has a truncated embedded code signature.");
  }
  const uint32_t magic = ReadBigEndianU32(bytes);
  const uint32_t blob_size = ReadBigEndianU32(bytes + 4);
  if (blob_size < kCsGenericBlobHeaderSize || blob_size > size) {
    return absl::FailedPreconditionError(
        "Loaded image has invalid embedded code-signature bounds.");
  }

  bool found_code_directory = false;
  if (magic == kCsMagicCodeDirectory) {
    ABSL_RETURN_IF_ERROR(ValidateStrongCodeDirectory(
        bytes, blob_size, expected_code_limit));
    found_code_directory = true;
  } else if (magic == kCsMagicEmbeddedSignature) {
    if (blob_size < kCsSuperBlobHeaderSize) {
      return absl::FailedPreconditionError(
          "Loaded image has a truncated code-signature SuperBlob.");
    }
    const uint32_t count = ReadBigEndianU32(bytes + 8);
    if (count > kMaximumCodeSignatureBlobs ||
        count > (blob_size - kCsSuperBlobHeaderSize) / kCsBlobIndexSize) {
      return absl::FailedPreconditionError(
          "Loaded image code-signature index is out of bounds.");
    }
    const size_t index_end =
        kCsSuperBlobHeaderSize + static_cast<size_t>(count) * kCsBlobIndexSize;
    struct SubBlobRange {
      uint32_t type;
      uint32_t begin;
      uint32_t end;
    };
    std::vector<SubBlobRange> sub_blob_ranges;
    sub_blob_ranges.reserve(count);
    for (uint32_t index = 0; index < count; ++index) {
      const uint8_t* entry =
          bytes + kCsSuperBlobHeaderSize + index * kCsBlobIndexSize;
      const uint32_t type = ReadBigEndianU32(entry);
      const uint32_t offset = ReadBigEndianU32(entry + 4);
      if (offset < index_end || offset > blob_size - kCsGenericBlobHeaderSize) {
        return absl::FailedPreconditionError(
            "Loaded image code-signature sub-blob is out of bounds.");
      }
      const uint32_t sub_blob_size = ReadBigEndianU32(bytes + offset + 4);
      if (sub_blob_size < kCsGenericBlobHeaderSize ||
          sub_blob_size > blob_size - offset) {
        return absl::FailedPreconditionError(
            "Loaded image code-signature sub-blob has invalid bounds.");
      }
      const uint32_t sub_blob_end = offset + sub_blob_size;
      for (const SubBlobRange& prior : sub_blob_ranges) {
        if (prior.type == type ||
            (offset < prior.end && prior.begin < sub_blob_end)) {
          return absl::FailedPreconditionError(
              "Loaded image code-signature sub-blobs are ambiguous.");
        }
      }
      sub_blob_ranges.push_back(
          SubBlobRange{.type = type, .begin = offset, .end = sub_blob_end});
      const bool is_code_directory =
          type == kCsSlotCodeDirectory ||
          (type >= kCsSlotAlternateCodeDirectoryFirst &&
           type < kCsSlotAlternateCodeDirectoryLimit);
      if (is_code_directory) {
        ABSL_RETURN_IF_ERROR(ValidateStrongCodeDirectory(
            bytes + offset, sub_blob_size, expected_code_limit));
        found_code_directory = true;
      }
    }
  } else {
    return absl::UnimplementedError(
        "Loaded image has an unsupported embedded code-signature format.");
  }
  if (!found_code_directory) {
    return absl::UnimplementedError(
        "Loaded image lacks a full-coverage SHA-256 or SHA-384 "
        "CodeDirectory.");
  }
  for (size_t offset = blob_size; offset < size; ++offset) {
    if (bytes[offset] != 0) {
      return absl::UnimplementedError(
          "Loaded image code-signature padding contains unclassified data.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> ReadSysctl(absl::string_view name) {
  const std::string nul_terminated_name(name);
  size_t size = 0;
  if (::sysctlbyname(nul_terminated_name.c_str(), nullptr, &size, nullptr, 0) !=
      0) {
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("sysctl size failed for ", name));
  }
  if (size == 0 || size > kMaximumSysctlBytes) {
    return absl::FailedPreconditionError(
        absl::StrCat("sysctl returned invalid evidence size for ", name));
  }
  std::string value(size, '\0');
  if (::sysctlbyname(nul_terminated_name.c_str(), value.data(), &size, nullptr,
                     0) != 0) {
    return absl::ErrnoToStatus(errno,
                               absl::StrCat("sysctl read failed for ", name));
  }
  if (size == 0 || size > value.size()) {
    return absl::FailedPreconditionError(
        absl::StrCat("sysctl changed evidence size for ", name));
  }
  value.resize(size);
  return value;
}

struct MeasuredImage {
  Hash256 digest;
  bool contains_runtime_anchor = false;
  bool has_digest = false;
};

absl::StatusOr<MeasuredImage> MeasureImage(const mach_header* image_header,
                                           uintptr_t runtime_anchor,
                                           bool include_without_anchor) {
  if (image_header == nullptr) {
    return absl::FailedPreconditionError(
        "dyld returned a null loaded-image header.");
  }
  const uintptr_t header_address =
      reinterpret_cast<uintptr_t>(image_header);
  ABSL_RETURN_IF_ERROR(ValidateMappedRange(
      header_address, sizeof(mach_header_64), /*require_immutable=*/true,
      "Loaded Mach-O header"));
  if (image_header->magic != MH_MAGIC_64) {
    return absl::UnimplementedError(
        "Loaded runtime identity requires a 64-bit Mach-O image.");
  }
  const auto* header = reinterpret_cast<const mach_header_64*>(image_header);
  if (header->sizeofcmds > kMaximumLoadCommandsBytes ||
      header->ncmds > header->sizeofcmds / sizeof(load_command)) {
    return absl::FailedPreconditionError(
        "Loaded Mach-O image has invalid load-command bounds.");
  }
  const uint64_t header_and_commands_size =
      sizeof(*header) + static_cast<uint64_t>(header->sizeofcmds);
  ABSL_RETURN_IF_ERROR(ValidateMappedRange(
      header_address, header_and_commands_size, /*require_immutable=*/true,
      "Loaded Mach-O header and load commands"));

  const uint8_t* commands = reinterpret_cast<const uint8_t*>(header + 1);
  struct SegmentEvidence {
    segment_command_64 segment;
    bool has_relocations = false;
    bool has_self_modifying_code = false;
  };
  size_t command_offset = 0;
  uint64_t base_vmaddr = 0;
  uint64_t maximum_file_end = 0;
  bool found_base = false;
  std::vector<SegmentEvidence> segments;
  std::vector<size_t> executable_segments;
  executable_segments.reserve(2);
  std::optional<linkedit_data_command> code_signature;
  for (uint32_t index = 0; index < header->ncmds; ++index) {
    if (header->sizeofcmds - command_offset < sizeof(load_command)) {
      return absl::FailedPreconditionError(
          "Loaded Mach-O image has a truncated load command.");
    }
    load_command command;
    std::memcpy(&command, commands + command_offset, sizeof(command));
    if (command.cmdsize < sizeof(load_command) ||
        command.cmdsize > header->sizeofcmds - command_offset ||
        command.cmdsize % sizeof(uint64_t) != 0) {
      return absl::FailedPreconditionError(
          "Loaded Mach-O image has an invalid load command.");
    }
    if (command.cmd == LC_SEGMENT_64) {
      if (command.cmdsize < sizeof(segment_command_64)) {
        return absl::FailedPreconditionError(
            "Loaded Mach-O image has a truncated segment command.");
      }
      segment_command_64 segment;
      std::memcpy(&segment, commands + command_offset, sizeof(segment));
      if (segment.filesize > segment.vmsize ||
          segment.filesize > kMaximumMeasuredSegmentBytes ||
          segment.fileoff >
              std::numeric_limits<uint64_t>::max() - segment.filesize ||
          segment.vmaddr >
              std::numeric_limits<uint64_t>::max() - segment.vmsize) {
        return absl::FailedPreconditionError(
            "Loaded Mach-O image has invalid segment bounds.");
      }
      const uint64_t file_end = segment.fileoff + segment.filesize;
      maximum_file_end = std::max(maximum_file_end, file_end);
      if (segment.nsects >
              (command.cmdsize - sizeof(segment_command_64)) /
                  sizeof(section_64) ||
          command.cmdsize !=
              sizeof(segment_command_64) +
                  static_cast<uint64_t>(segment.nsects) *
                      sizeof(section_64)) {
        return absl::FailedPreconditionError(
            "Loaded Mach-O image has invalid section-command bounds.");
      }
      SegmentEvidence evidence{.segment = segment};
      const uint8_t* sections =
          commands + command_offset + sizeof(segment_command_64);
      for (uint32_t section_index = 0; section_index < segment.nsects;
           ++section_index) {
        section_64 section;
        std::memcpy(&section,
                    sections + static_cast<size_t>(section_index) *
                                   sizeof(section_64),
                    sizeof(section));
        if (section.addr < segment.vmaddr ||
            section.size > segment.vmsize - (section.addr - segment.vmaddr)) {
          return absl::FailedPreconditionError(
              "Loaded Mach-O image has a section outside its VM segment.");
        }
        const uint32_t section_type = section.flags & SECTION_TYPE;
        const bool is_zero_fill =
            section_type == S_ZEROFILL || section_type == S_GB_ZEROFILL ||
            section_type == S_THREAD_LOCAL_ZEROFILL;
        if (!is_zero_fill && section.size != 0) {
          const uint64_t section_file_offset = section.offset;
          if (section_file_offset < segment.fileoff ||
              section_file_offset - segment.fileoff > segment.filesize ||
              section.size >
                  segment.filesize -
                      (section_file_offset - segment.fileoff)) {
            return absl::FailedPreconditionError(
                "Loaded Mach-O image has a section outside its file segment.");
          }
        }
        evidence.has_relocations =
            evidence.has_relocations ||
            (section.flags & (S_ATTR_EXT_RELOC | S_ATTR_LOC_RELOC)) != 0;
        evidence.has_self_modifying_code =
            evidence.has_self_modifying_code ||
            (section.flags & S_ATTR_SELF_MODIFYING_CODE) != 0;
      }
      if (segment.fileoff == 0 && segment.filesize >= sizeof(*header)) {
        if (found_base && base_vmaddr != segment.vmaddr) {
          return absl::FailedPreconditionError(
              "Loaded Mach-O image has ambiguous base segments.");
        }
        base_vmaddr = segment.vmaddr;
        found_base = true;
      }
      if ((segment.initprot & VM_PROT_EXECUTE) != 0 &&
          segment.filesize != 0) {
        executable_segments.push_back(segments.size());
      }
      segments.push_back(evidence);
    } else if (command.cmd == LC_CODE_SIGNATURE) {
      if (command.cmdsize != sizeof(linkedit_data_command) ||
          code_signature.has_value()) {
        return absl::FailedPreconditionError(
            "Loaded Mach-O image has an invalid or duplicate code-signature "
            "command.");
      }
      linkedit_data_command signature;
      std::memcpy(&signature, commands + command_offset, sizeof(signature));
      if (signature.datasize == 0 ||
          signature.datasize > kMaximumCodeSignatureBytes ||
          signature.dataoff >
              std::numeric_limits<uint32_t>::max() - signature.datasize) {
        return absl::FailedPreconditionError(
            "Loaded Mach-O image has invalid code-signature bounds.");
      }
      code_signature = signature;
    }
    command_offset += command.cmdsize;
  }
  if (!found_base || command_offset != header->sizeofcmds ||
      executable_segments.empty()) {
    return absl::FailedPreconditionError(
        "Loaded Mach-O image lacks a complete executable layout.");
  }

  for (size_t first = 0; first < segments.size(); ++first) {
    const segment_command_64& left = segments[first].segment;
    for (size_t second = first + 1; second < segments.size(); ++second) {
      const segment_command_64& right = segments[second].segment;
      if (left.filesize != 0 && right.filesize != 0 &&
          left.fileoff < right.fileoff + right.filesize &&
          right.fileoff < left.fileoff + left.filesize) {
        return absl::FailedPreconditionError(
            "Loaded Mach-O image has overlapping file segments.");
      }
      if (left.vmsize != 0 && right.vmsize != 0 &&
          left.vmaddr < right.vmaddr + right.vmsize &&
          right.vmaddr < left.vmaddr + left.vmsize) {
        return absl::FailedPreconditionError(
            "Loaded Mach-O image has overlapping VM segments.");
      }
    }
  }

  auto segment_runtime_address =
      [header_address, base_vmaddr](const segment_command_64& segment)
      -> absl::StatusOr<uintptr_t> {
    if (segment.vmaddr < base_vmaddr ||
        segment.vmaddr - base_vmaddr >
            std::numeric_limits<uintptr_t>::max() - header_address) {
      return absl::FailedPreconditionError(
          "Loaded Mach-O segment address overflows.");
    }
    return header_address +
           static_cast<uintptr_t>(segment.vmaddr - base_vmaddr);
  };

  bool contains_anchor = false;
  for (size_t segment_index : executable_segments) {
    const segment_command_64& segment = segments[segment_index].segment;
    ABSL_ASSIGN_OR_RETURN(const uintptr_t address,
                          segment_runtime_address(segment));
    if (segment.filesize >
        std::numeric_limits<uintptr_t>::max() - address) {
      return absl::FailedPreconditionError(
          "Loaded Mach-O executable segment range overflows.");
    }
    const uintptr_t end = address + static_cast<uintptr_t>(segment.filesize);
    contains_anchor = contains_anchor ||
                      (runtime_anchor >= address && runtime_anchor < end);
  }
  if (!include_without_anchor && !contains_anchor) {
    return MeasuredImage{.contains_runtime_anchor = false,
                         .has_digest = false};
  }

  if (!code_signature.has_value()) {
    return absl::UnimplementedError(
        "Loaded image lacks an embedded code-signature commitment.");
  }
  const uint64_t signature_end =
      static_cast<uint64_t>(code_signature->dataoff) +
      code_signature->datasize;
  if (signature_end != maximum_file_end) {
    return absl::UnimplementedError(
        "Loaded image has file-backed content outside its embedded "
        "code-signature boundary.");
  }

  const SegmentEvidence* signature_segment = nullptr;
  for (const SegmentEvidence& evidence : segments) {
    const segment_command_64& segment = evidence.segment;
    if (code_signature->dataoff < segment.fileoff) continue;
    const uint64_t signature_segment_offset =
        code_signature->dataoff - segment.fileoff;
    if (signature_segment_offset <= segment.filesize &&
        code_signature->datasize <=
            segment.filesize - signature_segment_offset) {
      if (signature_segment != nullptr) {
        return absl::FailedPreconditionError(
            "Loaded image code-signature mapping is ambiguous.");
      }
      signature_segment = &evidence;
    }
  }
  if (signature_segment == nullptr ||
      (signature_segment->segment.initprot & VM_PROT_READ) == 0 ||
      (signature_segment->segment.initprot & VM_PROT_WRITE) != 0 ||
      (signature_segment->segment.maxprot & VM_PROT_WRITE) != 0 ||
      (signature_segment->segment.flags & SG_PROTECTED_VERSION_1) != 0) {
    return absl::UnimplementedError(
        "Loaded image code-signature bytes are not in an immutable readable "
        "mapping.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const uintptr_t signature_segment_address,
      segment_runtime_address(signature_segment->segment));
  const uint64_t signature_offset_in_segment =
      code_signature->dataoff - signature_segment->segment.fileoff;
  if (signature_offset_in_segment >
      std::numeric_limits<uintptr_t>::max() - signature_segment_address) {
    return absl::FailedPreconditionError(
        "Loaded image code-signature address overflows.");
  }
  const uintptr_t signature_address =
      signature_segment_address + signature_offset_in_segment;
  ABSL_RETURN_IF_ERROR(ValidateMappedRange(
      signature_address, code_signature->datasize,
      /*require_immutable=*/true, "Loaded Mach-O code signature"));
  const auto* signature_bytes =
      reinterpret_cast<const uint8_t*>(signature_address);
  ABSL_RETURN_IF_ERROR(ValidateEmbeddedCodeSignature(
      signature_bytes, code_signature->datasize, code_signature->dataoff));

  // The CodeDirectory commits the original bytes in [0, dataoff), including
  // bytes that dyld may later relocate in writable segments. Bind that
  // commitment to the image actually loaded by requiring every mapped,
  // file-backed page in the covered range to have untainted kernel code-sign
  // validation evidence. Immutable loaded bytes are also hashed directly
  // below; mutable relocated process data is deliberately not hashed.
  for (const SegmentEvidence& evidence : segments) {
    const segment_command_64& segment = evidence.segment;
    if (segment.filesize == 0 ||
        segment.fileoff >= code_signature->dataoff) {
      continue;
    }
    if ((segment.flags & SG_PROTECTED_VERSION_1) != 0) {
      return absl::UnimplementedError(
          "Loaded image has protected file-backed pages whose signing state "
          "cannot be measured safely.");
    }
    const uint64_t signed_size = std::min(
        segment.filesize,
        static_cast<uint64_t>(code_signature->dataoff) - segment.fileoff);
    ABSL_ASSIGN_OR_RETURN(const uintptr_t address,
                          segment_runtime_address(segment));
    ABSL_RETURN_IF_ERROR(FaultAndValidateCodeSignedPages(
        address, signed_size, "Loaded Mach-O file-backed segment"));
  }

  std::vector<size_t> immutable_segments;
  immutable_segments.reserve(segments.size());
  for (size_t index = 0; index < segments.size(); ++index) {
    const SegmentEvidence& evidence = segments[index];
    const segment_command_64& segment = evidence.segment;
    if (segment.filesize == 0 ||
        (segment.initprot & VM_PROT_READ) == 0 ||
        (segment.initprot & VM_PROT_WRITE) != 0 ||
        (segment.maxprot & VM_PROT_WRITE) != 0) {
      continue;
    }
    if ((segment.flags & SG_PROTECTED_VERSION_1) != 0 ||
        evidence.has_relocations) {
      return absl::UnimplementedError(
          "Loaded image has a protected or relocated read-only segment whose "
          "in-memory bytes are not stable identity evidence.");
    }
    ABSL_ASSIGN_OR_RETURN(const uintptr_t address,
                          segment_runtime_address(segment));
    ABSL_RETURN_IF_ERROR(ValidateMappedRange(
        address, segment.filesize, /*require_immutable=*/true,
        "Loaded Mach-O read-only segment"));
    immutable_segments.push_back(index);
  }

  for (size_t segment_index : executable_segments) {
    const SegmentEvidence& evidence = segments[segment_index];
    const segment_command_64& segment = evidence.segment;
    if ((segment.initprot & VM_PROT_READ) == 0 ||
        (segment.initprot & VM_PROT_WRITE) != 0 ||
        (segment.maxprot & VM_PROT_WRITE) != 0 ||
        (segment.flags & SG_PROTECTED_VERSION_1) != 0 ||
        evidence.has_relocations || evidence.has_self_modifying_code) {
      return absl::UnimplementedError(
          "Loaded image has unreadable, writable, relocated, protected, or "
          "self-modifying executable bytes.");
    }
    if (std::find(immutable_segments.begin(), immutable_segments.end(),
                  segment_index) == immutable_segments.end()) {
      return absl::FailedPreconditionError(
          "Loaded executable segment was not admitted as immutable evidence.");
    }
  }

  Sha256Hasher image_hasher;
  HashFrame("LITERT_LM_LOADED_MACHO_IMAGE_V2", &image_hasher);
  HashFrame(absl::string_view(reinterpret_cast<const char*>(header),
                              sizeof(*header) + header->sizeofcmds),
            &image_hasher);
  HashU32(static_cast<uint32_t>(immutable_segments.size()), &image_hasher);
  for (size_t segment_index : immutable_segments) {
    const segment_command_64& segment = segments[segment_index].segment;
    ABSL_ASSIGN_OR_RETURN(const uintptr_t address,
                          segment_runtime_address(segment));
    HashU64(segment.vmaddr - base_vmaddr, &image_hasher);
    HashU64(segment.fileoff, &image_hasher);
    HashU64(segment.filesize, &image_hasher);
    image_hasher.Update(absl::string_view(
        reinterpret_cast<const char*>(address),
        static_cast<size_t>(segment.filesize)));
  }
  HashU64(code_signature->dataoff, &image_hasher);
  HashFrame(absl::string_view(
                reinterpret_cast<const char*>(signature_bytes),
                static_cast<size_t>(code_signature->datasize)),
            &image_hasher);
  return MeasuredImage{.digest = image_hasher.Finalize(),
                       .contains_runtime_anchor = contains_anchor,
                       .has_digest = true};
}

absl::StatusOr<Hash256> MeasureImageContainingAnchor(uintptr_t code_anchor) {
  if (code_anchor == 0) {
    return absl::FailedPreconditionError(
        "Cannot measure a zero loaded-image code anchor.");
  }
  for (int attempt = 0; attempt < 3; ++attempt) {
    const uint32_t image_count = ::_dyld_image_count();
    std::vector<const mach_header*> image_headers;
    std::vector<std::string> image_names;
    image_headers.reserve(image_count);
    image_names.reserve(image_count);
    std::optional<Hash256> anchored_digest;
    for (uint32_t index = 0; index < image_count; ++index) {
      const mach_header* image_header = ::_dyld_get_image_header(index);
      const char* image_name = ::_dyld_get_image_name(index);
      if (image_header == nullptr || image_name == nullptr ||
          image_name[0] == '\0') {
        return absl::FailedPreconditionError(
            "dyld reported incomplete anchored-image evidence.");
      }
      image_headers.push_back(image_header);
      image_names.emplace_back(image_name);
      ABSL_ASSIGN_OR_RETURN(
          MeasuredImage measured,
          MeasureImage(image_header, code_anchor,
                       /*include_without_anchor=*/false));
      if (!measured.contains_runtime_anchor) continue;
      if (!measured.has_digest || anchored_digest.has_value()) {
        return absl::FailedPreconditionError(
            "Loaded code anchor has missing or ambiguous image identity.");
      }
      anchored_digest = measured.digest;
    }
    if (::_dyld_image_count() != image_count) continue;
    bool stable = true;
    for (uint32_t index = 0; index < image_count; ++index) {
      const char* image_name = ::_dyld_get_image_name(index);
      if (::_dyld_get_image_header(index) != image_headers[index] ||
          image_name == nullptr || image_names[index] != image_name) {
        stable = false;
        break;
      }
    }
    if (!stable) continue;
    if (!anchored_digest.has_value()) {
      return absl::FailedPreconditionError(
          "No loaded Mach-O image contains the requested code anchor.");
    }
    return *anchored_digest;
  }
  return absl::UnavailableError(
      "Loaded image set changed during anchored identity measurement.");
}

absl::StatusOr<Hash256> MeasureAppleCpuRuntimeArtifact(
    const SessionHandoffRuntimeProfile& profile) {
  if (profile.runtime_code_anchor == 0 || profile.canonical_profile.empty()) {
    return absl::FailedPreconditionError(
        "Loaded CPU runtime profile lacks measurable evidence.");
  }
  // Bind only the image that owns the live LiteRT dispatch function. The host
  // executable and unrelated application/plugin images are not inference
  // artifacts and must not make an otherwise identical fresh worker derive a
  // different profile.
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 runtime_image_digest,
      MeasureImageContainingAnchor(profile.runtime_code_anchor));

  constexpr std::array<absl::string_view, 7> kPlatformEvidence = {
      "kern.osversion", "kern.osrelease", "hw.model",     "hw.machine",
      "hw.cputype",     "hw.cpusubtype",  "hw.cpufamily",
  };
  Sha256Hasher hasher;
  HashFrame(kRuntimeArtifactDomain, &hasher);
  // V4 makes the executable-container class explicit so an ELF CPU image can
  // never enter an admission created under the earlier Apple-only namespace.
  HashFrame("APPLE_MACHO_CPU_V1", &hasher);
  HashU32(static_cast<uint32_t>(profile.runtime_class), &hasher);
  HashFrame(profile.canonical_profile, &hasher);
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(runtime_image_digest.bytes.data()),
      runtime_image_digest.bytes.size()));
  for (absl::string_view name : kPlatformEvidence) {
    ABSL_ASSIGN_OR_RETURN(std::string value, ReadSysctl(name));
    HashFrame(name, &hasher);
    HashFrame(value, &hasher);
  }
  return hasher.Finalize();
}

absl::StatusOr<Hash256> MeasureAppleMetalRuntimeArtifact(
    const SessionHandoffRuntimeProfile& profile) {
  if (profile.runtime_code_anchor == 0 || profile.canonical_profile.empty() ||
      !profile.metal_corun.has_value()) {
    return absl::FailedPreconditionError(
        "Loaded Metal runtime profile lacks measurable evidence.");
  }
  const MetalCoRunRuntimeEvidence& metal = *profile.metal_corun;
  if (metal.metal_device == nullptr || metal.metal_command_queue == nullptr ||
      metal.selected_accelerator_code_anchor == 0 ||
      metal.canonical_policy.empty() ||
      (metal.derived_evidence &
       MetalCoRunEvidenceBit(MetalCoRunEvidence::kSelectedMetalDelegate)) ==
          0) {
    return absl::FailedPreconditionError(
        "Loaded Metal runtime lacks selected accelerator, device, queue, or "
        "policy evidence.");
  }

  // The runtime API table and the selected accelerator callback are separate
  // live code anchors. Measuring both prevents a generic LiteRT runtime image
  // plus a GPU label from impersonating the actually selected Metal plugin.
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 runtime_image_digest,
      MeasureImageContainingAnchor(profile.runtime_code_anchor));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 accelerator_image_digest,
      MeasureImageContainingAnchor(metal.selected_accelerator_code_anchor));
  ABSL_ASSIGN_OR_RETURN(
      std::string metal_device_identity,
      DeriveMacOsMetalDeviceIdentity(metal.metal_device,
                                     metal.metal_command_queue));

  constexpr std::array<absl::string_view, 7> kPlatformEvidence = {
      "kern.osversion", "kern.osrelease", "hw.model",     "hw.machine",
      "hw.cputype",     "hw.cpusubtype",  "hw.cpufamily",
  };
  Sha256Hasher hasher;
  HashFrame("LITERT_LM_LOADED_METAL_RUNTIME_ARTIFACT_V2", &hasher);
  HashU32(static_cast<uint32_t>(profile.runtime_class), &hasher);
  HashFrame(profile.canonical_profile, &hasher);
  HashFrame(metal.canonical_policy, &hasher);
  HashFrame(metal_device_identity, &hasher);
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(runtime_image_digest.bytes.data()),
      runtime_image_digest.bytes.size()));
  hasher.Update(absl::string_view(
      reinterpret_cast<const char*>(accelerator_image_digest.bytes.data()),
      accelerator_image_digest.bytes.size()));
  for (absl::string_view name : kPlatformEvidence) {
    ABSL_ASSIGN_OR_RETURN(std::string value, ReadSysctl(name));
    HashFrame(name, &hasher);
    HashFrame(value, &hasher);
  }
  return hasher.Finalize();
}

#endif  // defined(__APPLE__)

#if defined(__linux__) || defined(__ANDROID__)

constexpr size_t kMaximumProcEvidenceBytes = 32 * 1024 * 1024;
constexpr size_t kMaximumPlatformEvidenceBytes = 1024 * 1024;
constexpr uint64_t kMaximumElfExecutableBytes = uint64_t{8} * 1024 * 1024 *
                                                1024;
constexpr size_t kMaximumAuxiliaryStringBytes = 4096;

absl::StatusOr<std::string> ReadBoundedFile(absl::string_view path,
                                            size_t maximum_bytes) {
  const std::string nul_terminated_path(path);
  const int fd =
      ::open(nul_terminated_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Cannot open platform evidence ", path));
  }

  std::string contents;
  std::array<char, 16 * 1024> buffer;
  while (contents.size() <= maximum_bytes) {
    const size_t remaining = maximum_bytes + 1 - contents.size();
    const size_t request = std::min(remaining, buffer.size());
    const ssize_t bytes_read = ::read(fd, buffer.data(), request);
    if (bytes_read < 0) {
      if (errno == EINTR) continue;
      const int read_errno = errno;
      (void)::close(fd);
      return absl::ErrnoToStatus(
          read_errno, absl::StrCat("Cannot read platform evidence ", path));
    }
    if (bytes_read == 0) break;
    contents.append(buffer.data(), static_cast<size_t>(bytes_read));
  }
  if (::close(fd) != 0) {
    return absl::ErrnoToStatus(
        errno, absl::StrCat("Cannot close platform evidence ", path));
  }
  if (contents.size() > maximum_bytes) {
    return absl::ResourceExhaustedError(
        absl::StrCat("Platform evidence exceeds the supported bound: ",
                     path));
  }
  return contents;
}

std::string NormalizeEvidenceText(absl::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  bool pending_space = false;
  for (unsigned char character : value) {
    if (character == '\0' || std::isspace(character)) {
      pending_space = !normalized.empty();
      continue;
    }
    if (pending_space) normalized.push_back(' ');
    normalized.push_back(static_cast<char>(character));
    pending_space = false;
  }
  return normalized;
}

std::string NormalizeCpuInfoKey(absl::string_view key) {
  std::string normalized = NormalizeEvidenceText(key);
  for (char& character : normalized) {
    character = static_cast<char>(
        std::tolower(static_cast<unsigned char>(character)));
  }
  return normalized;
}

bool IsStableCpuInfoKey(absl::string_view key) {
  constexpr std::array<absl::string_view, 22> kStableKeys = {
      "vendor_id",        "cpu family",       "model",
      "model name",       "stepping",         "microcode",
      "flags",            "bugs",             "cpu implementer",
      "cpu architecture", "cpu variant",      "cpu part",
      "cpu revision",     "features",         "isa",
      "uarch",            "mmu",              "fpu",
      "hardware",         "revision",         "platform",
      "isa extensions",
  };
  return std::find(kStableKeys.begin(), kStableKeys.end(), key) !=
         kStableKeys.end();
}

absl::StatusOr<std::string> CanonicalizeCpuInfo(
    absl::string_view cpu_info) {
  std::vector<std::string> records;
  std::vector<std::string> fields;
  size_t cursor = 0;
  while (cursor <= cpu_info.size()) {
    const size_t line_end = cpu_info.find('\n', cursor);
    const size_t bounded_end =
        line_end == absl::string_view::npos ? cpu_info.size() : line_end;
    absl::string_view line = cpu_info.substr(cursor, bounded_end - cursor);
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (NormalizeEvidenceText(line).empty()) {
      if (!fields.empty()) {
        std::sort(fields.begin(), fields.end());
        std::string record;
        for (const std::string& field : fields) {
          record.append(field);
          record.push_back('\n');
        }
        records.push_back(std::move(record));
        fields.clear();
      }
    } else {
      const size_t separator = line.find(':');
      if (separator != absl::string_view::npos) {
        const std::string key = NormalizeCpuInfoKey(line.substr(0, separator));
        const std::string value =
            NormalizeEvidenceText(line.substr(separator + 1));
        if (IsStableCpuInfoKey(key) && !value.empty()) {
          fields.push_back(absl::StrCat(key, "=", value));
        }
      }
    }
    if (line_end == absl::string_view::npos) break;
    cursor = line_end + 1;
  }
  if (!fields.empty()) {
    std::sort(fields.begin(), fields.end());
    std::string record;
    for (const std::string& field : fields) {
      record.append(field);
      record.push_back('\n');
    }
    records.push_back(std::move(record));
  }
  if (records.empty()) {
    return absl::UnimplementedError(
        "Linux CPU identity has no stable /proc/cpuinfo fields.");
  }
  std::sort(records.begin(), records.end());
  std::string canonical;
  Sha256Hasher canonical_hasher;
  HashFrame("LITERT_LM_CANONICAL_CPUINFO_V1", &canonical_hasher);
  HashU64(records.size(), &canonical_hasher);
  for (const std::string& record : records) {
    HashFrame(record, &canonical_hasher);
  }
  const Hash256 digest = canonical_hasher.Finalize();
  canonical.assign(reinterpret_cast<const char*>(digest.bytes.data()),
                   digest.bytes.size());
  return canonical;
}

struct LinuxPlatformEvidence {
  std::array<std::string, 4> uname_fields;
  std::vector<uint32_t> effective_cpu_affinity;
  uint64_t page_size = 0;
  uint64_t hardware_capabilities = 0;
  bool has_hardware_capabilities_2 = false;
  uint64_t hardware_capabilities_2 = 0;
  std::string auxiliary_platform;
  std::string auxiliary_base_platform;
  std::string canonical_cpu_info;
  std::vector<std::pair<std::string, std::string>> named_evidence;

  bool operator==(const LinuxPlatformEvidence& other) const {
    return uname_fields == other.uname_fields &&
           effective_cpu_affinity == other.effective_cpu_affinity &&
           page_size == other.page_size &&
           hardware_capabilities == other.hardware_capabilities &&
           has_hardware_capabilities_2 ==
               other.has_hardware_capabilities_2 &&
           hardware_capabilities_2 == other.hardware_capabilities_2 &&
           auxiliary_platform == other.auxiliary_platform &&
           auxiliary_base_platform == other.auxiliary_base_platform &&
           canonical_cpu_info == other.canonical_cpu_info &&
           named_evidence == other.named_evidence;
  }
};

absl::StatusOr<std::pair<bool, uint64_t>> ReadAuxiliaryValue(
    unsigned long type) {
  errno = 0;
  const unsigned long value = ::getauxval(type);
  if (value == 0 && errno != 0) {
    if (errno == ENOENT) return std::pair<bool, uint64_t>{false, 0};
    return absl::ErrnoToStatus(errno,
                               "Cannot read Linux auxiliary-vector evidence");
  }
  return std::pair<bool, uint64_t>{true, static_cast<uint64_t>(value)};
}

absl::StatusOr<std::string> ReadAuxiliaryString(unsigned long type) {
  ABSL_ASSIGN_OR_RETURN(const auto value, ReadAuxiliaryValue(type));
  if (!value.first) return std::string();
  if (value.second == 0 ||
      value.second > std::numeric_limits<uintptr_t>::max()) {
    return absl::FailedPreconditionError(
        "Linux auxiliary-vector string pointer is invalid.");
  }
  const char* text = reinterpret_cast<const char*>(
      static_cast<uintptr_t>(value.second));
  const size_t size = ::strnlen(text, kMaximumAuxiliaryStringBytes + 1);
  if (size == 0 || size > kMaximumAuxiliaryStringBytes) {
    return absl::FailedPreconditionError(
        "Linux auxiliary-vector string is empty or unterminated.");
  }
  return std::string(text, size);
}

absl::StatusOr<std::string> UtsField(const char* field, size_t capacity,
                                     absl::string_view name) {
  const size_t size = ::strnlen(field, capacity);
  if (size == 0 || size == capacity) {
    return absl::FailedPreconditionError(
        absl::StrCat("uname returned an empty or unterminated ", name,
                     " field."));
  }
  return std::string(field, size);
}

absl::StatusOr<LinuxPlatformEvidence> CaptureLinuxPlatformEvidence() {
  utsname name = {};
  if (::uname(&name) != 0) {
    return absl::ErrnoToStatus(errno, "Cannot read Linux uname evidence");
  }

  LinuxPlatformEvidence evidence;
  ABSL_ASSIGN_OR_RETURN(evidence.uname_fields[0],
                        UtsField(name.sysname, sizeof(name.sysname),
                                 "sysname"));
  ABSL_ASSIGN_OR_RETURN(evidence.uname_fields[1],
                        UtsField(name.release, sizeof(name.release),
                                 "release"));
  ABSL_ASSIGN_OR_RETURN(evidence.uname_fields[2],
                        UtsField(name.version, sizeof(name.version),
                                 "version"));
  ABSL_ASSIGN_OR_RETURN(evidence.uname_fields[3],
                        UtsField(name.machine, sizeof(name.machine),
                                 "machine"));

  // `sched_getaffinity` reports the effective mask after inherited, cgroup,
  // and Android EngineFactory restrictions have all been applied. Requested
  // thread counts or device labels cannot distinguish a successfully pinned
  // exact worker from one whose affinity operation failed.
  cpu_set_t affinity;
  CPU_ZERO(&affinity);
  if (::sched_getaffinity(0, sizeof(affinity), &affinity) != 0) {
    return absl::ErrnoToStatus(
        errno, "Cannot read effective Linux CPU affinity evidence");
  }
  for (uint32_t cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (CPU_ISSET(static_cast<int>(cpu), &affinity)) {
      evidence.effective_cpu_affinity.push_back(cpu);
    }
  }
  if (evidence.effective_cpu_affinity.empty()) {
    return absl::FailedPreconditionError(
        "Linux reported an empty effective CPU affinity mask.");
  }

  const long configured_page_size = ::sysconf(_SC_PAGESIZE);
  if (configured_page_size <= 0) {
    return absl::FailedPreconditionError(
        "Linux runtime identity cannot determine the page size.");
  }
  evidence.page_size = static_cast<uint64_t>(configured_page_size);
  ABSL_ASSIGN_OR_RETURN(const auto auxiliary_page_size,
                        ReadAuxiliaryValue(AT_PAGESZ));
  if (!auxiliary_page_size.first ||
      auxiliary_page_size.second != evidence.page_size) {
    return absl::FailedPreconditionError(
        "Linux page-size evidence is missing or inconsistent.");
  }
  ABSL_ASSIGN_OR_RETURN(const auto hardware_capabilities,
                        ReadAuxiliaryValue(AT_HWCAP));
  if (!hardware_capabilities.first) {
    return absl::UnimplementedError(
        "Linux CPU identity lacks AT_HWCAP evidence.");
  }
  evidence.hardware_capabilities = hardware_capabilities.second;
#if defined(AT_HWCAP2)
  ABSL_ASSIGN_OR_RETURN(const auto hardware_capabilities_2,
                        ReadAuxiliaryValue(AT_HWCAP2));
  evidence.has_hardware_capabilities_2 = hardware_capabilities_2.first;
  evidence.hardware_capabilities_2 = hardware_capabilities_2.second;
#endif
#if defined(AT_PLATFORM)
  ABSL_ASSIGN_OR_RETURN(evidence.auxiliary_platform,
                        ReadAuxiliaryString(AT_PLATFORM));
#endif
#if defined(AT_BASE_PLATFORM)
  ABSL_ASSIGN_OR_RETURN(evidence.auxiliary_base_platform,
                        ReadAuxiliaryString(AT_BASE_PLATFORM));
#endif

  ABSL_ASSIGN_OR_RETURN(
      const std::string cpu_info,
      ReadBoundedFile("/proc/cpuinfo", kMaximumProcEvidenceBytes));
  ABSL_ASSIGN_OR_RETURN(evidence.canonical_cpu_info,
                        CanonicalizeCpuInfo(cpu_info));

  constexpr std::array<absl::string_view, 8> kHardwareEvidencePaths = {
      "/sys/devices/virtual/dmi/id/product_name",
      "/sys/devices/virtual/dmi/id/product_version",
      "/sys/devices/virtual/dmi/id/board_name",
      "/sys/firmware/devicetree/base/model",
      "/sys/devices/soc0/family",
      "/sys/devices/soc0/machine",
      "/sys/devices/soc0/soc_id",
      "/sys/devices/soc0/revision",
  };
  for (absl::string_view path : kHardwareEvidencePaths) {
    absl::StatusOr<std::string> value =
        ReadBoundedFile(path, kMaximumPlatformEvidenceBytes);
    if (!value.ok()) {
      if (absl::IsNotFound(value.status()) ||
          absl::IsPermissionDenied(value.status())) {
        continue;
      }
      return value.status();
    }
    std::string normalized = NormalizeEvidenceText(*value);
    if (!normalized.empty()) {
      evidence.named_evidence.emplace_back(std::string(path),
                                           std::move(normalized));
    }
  }

#if defined(__ANDROID__)
  constexpr std::array<const char*, 8> kAndroidProperties = {
      "ro.build.fingerprint", "ro.build.version.incremental",
      "ro.product.device",    "ro.product.board",
      "ro.hardware",          "ro.board.platform",
      "ro.soc.manufacturer",  "ro.soc.model",
  };
  bool has_build_fingerprint = false;
  bool has_hardware_property = false;
  for (const char* property : kAndroidProperties) {
    std::array<char, PROP_VALUE_MAX> value = {};
    const int size = ::__system_property_get(property, value.data());
    if (size < 0 || static_cast<size_t>(size) >= value.size()) {
      return absl::FailedPreconditionError(
          "Android system-property evidence has invalid bounds.");
    }
    if (size == 0) continue;
    evidence.named_evidence.emplace_back(
        absl::StrCat("android-property:", property),
        std::string(value.data(), static_cast<size_t>(size)));
    has_build_fingerprint =
        has_build_fingerprint || std::strcmp(property, "ro.build.fingerprint") == 0;
    has_hardware_property =
        has_hardware_property ||
        std::strcmp(property, "ro.build.version.incremental") != 0 &&
            std::strcmp(property, "ro.build.fingerprint") != 0;
  }
  if (!has_build_fingerprint || !has_hardware_property) {
    return absl::UnimplementedError(
        "Android exact CPU identity lacks build or hardware properties.");
  }
#else
  absl::StatusOr<std::string> os_release =
      ReadBoundedFile("/etc/os-release", kMaximumPlatformEvidenceBytes);
  if (!os_release.ok() && absl::IsNotFound(os_release.status())) {
    os_release = ReadBoundedFile("/usr/lib/os-release",
                                 kMaximumPlatformEvidenceBytes);
  }
  if (os_release.ok()) {
    evidence.named_evidence.emplace_back("linux-os-release",
                                         std::move(*os_release));
  } else if (!absl::IsNotFound(os_release.status()) &&
             !absl::IsPermissionDenied(os_release.status())) {
    return os_release.status();
  }
#endif
  std::sort(evidence.named_evidence.begin(), evidence.named_evidence.end());
  return evidence;
}

struct ProcMapEntry {
  uintptr_t start = 0;
  uintptr_t end = 0;
  std::array<char, 4> permissions = {};
  uint64_t file_offset = 0;
  uint32_t device_major = 0;
  uint32_t device_minor = 0;
  uint64_t inode = 0;
  std::string path;
};

absl::StatusOr<std::vector<ProcMapEntry>> ReadProcMaps() {
  ABSL_ASSIGN_OR_RETURN(
      const std::string contents,
      ReadBoundedFile("/proc/self/maps", kMaximumProcEvidenceBytes));
  std::vector<ProcMapEntry> mappings;
  size_t cursor = 0;
  while (cursor < contents.size()) {
    const size_t line_end = contents.find('\n', cursor);
    const size_t bounded_end =
        line_end == std::string::npos ? contents.size() : line_end;
    const std::string line = contents.substr(cursor, bounded_end - cursor);
    unsigned long long start = 0;
    unsigned long long end = 0;
    unsigned long long file_offset = 0;
    unsigned int device_major = 0;
    unsigned int device_minor = 0;
    unsigned long long inode = 0;
    std::array<char, 5> permissions = {};
    int path_offset = 0;
    const int fields = std::sscanf(
        line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %n", &start, &end,
        permissions.data(), &file_offset, &device_major, &device_minor, &inode,
        &path_offset);
    if (fields != 7 || path_offset < 0 ||
        static_cast<size_t>(path_offset) > line.size() || start >= end ||
        start > std::numeric_limits<uintptr_t>::max() ||
        end > std::numeric_limits<uintptr_t>::max() ||
        std::strlen(permissions.data()) != 4) {
      return absl::FailedPreconditionError(
          "Linux reported a malformed /proc/self/maps entry.");
    }
    std::string path = line.substr(static_cast<size_t>(path_offset));
    while (!path.empty() && path.front() == ' ') path.erase(path.begin());
    ProcMapEntry entry{
        .start = static_cast<uintptr_t>(start),
        .end = static_cast<uintptr_t>(end),
        .permissions = {permissions[0], permissions[1], permissions[2],
                        permissions[3]},
        .file_offset = static_cast<uint64_t>(file_offset),
        .device_major = device_major,
        .device_minor = device_minor,
        .inode = static_cast<uint64_t>(inode),
        .path = std::move(path),
    };
    if (!mappings.empty() && mappings.back().end > entry.start) {
      return absl::FailedPreconditionError(
          "Linux reported overlapping process mappings.");
    }
    mappings.push_back(std::move(entry));
    if (line_end == std::string::npos) break;
    cursor = line_end + 1;
  }
  if (mappings.empty()) {
    return absl::FailedPreconditionError(
        "Linux reported an empty process map.");
  }
  return mappings;
}

bool IsAnonymousOrMutableImagePath(absl::string_view path) {
  return path.empty() || path.front() != '/' || path.front() == '[' ||
         path.starts_with("/memfd:") || path.find(" (deleted)") !=
                                                absl::string_view::npos;
}

struct ElfBackingIdentity {
  bool initialized = false;
  uint32_t device_major = 0;
  uint32_t device_minor = 0;
  uint64_t inode = 0;
  uint64_t container_offset_bias = 0;
  std::string path;

  bool operator==(const ElfBackingIdentity& other) const {
    return initialized == other.initialized &&
           device_major == other.device_major &&
           device_minor == other.device_minor && inode == other.inode &&
           container_offset_bias == other.container_offset_bias &&
           path == other.path;
  }
};

struct ElfMapSpan {
  uintptr_t start = 0;
  uintptr_t end = 0;
  std::array<char, 4> permissions = {};
  uint64_t file_offset = 0;

  bool operator==(const ElfMapSpan& other) const {
    return start == other.start && end == other.end &&
           permissions == other.permissions &&
           file_offset == other.file_offset;
  }
  bool operator<(const ElfMapSpan& other) const {
    if (start != other.start) return start < other.start;
    if (end != other.end) return end < other.end;
    if (permissions != other.permissions) {
      return permissions < other.permissions;
    }
    return file_offset < other.file_offset;
  }
};

struct ElfRangeRequirement {
  uintptr_t address = 0;
  uint64_t size = 0;
  uint64_t elf_file_offset = 0;
  bool require_executable = false;
  bool require_immutable = false;
  const char* description = nullptr;
};

const ProcMapEntry* FindProcMap(const std::vector<ProcMapEntry>& mappings,
                                uintptr_t address) {
  auto next = std::upper_bound(
      mappings.begin(), mappings.end(), address,
      [](uintptr_t target, const ProcMapEntry& mapping) {
        return target < mapping.start;
      });
  if (next == mappings.begin()) return nullptr;
  --next;
  return address >= next->start && address < next->end ? &*next : nullptr;
}

absl::Status ValidateMappedElfRange(
    const std::vector<ProcMapEntry>& mappings,
    const ElfRangeRequirement& requirement, ElfBackingIdentity* backing,
    std::vector<ElfMapSpan>* spans) {
  if (requirement.size == 0 || requirement.description == nullptr) {
    return absl::FailedPreconditionError(
        "Loaded ELF identity received an empty range requirement.");
  }
  if (requirement.elf_file_offset >
      std::numeric_limits<uint64_t>::max() - (requirement.size - 1)) {
    return absl::FailedPreconditionError(
        absl::StrCat(requirement.description,
                     " ELF file range overflows."));
  }
  if (requirement.size > std::numeric_limits<uintptr_t>::max() -
                             requirement.address) {
    return absl::FailedPreconditionError(
        absl::StrCat(requirement.description, " range overflows."));
  }
  const uintptr_t end =
      requirement.address + static_cast<uintptr_t>(requirement.size);
  uintptr_t cursor = requirement.address;
  while (cursor < end) {
    const ProcMapEntry* mapping = FindProcMap(mappings, cursor);
    if (mapping == nullptr) {
      return absl::FailedPreconditionError(
          absl::StrCat(requirement.description,
                       " is not completely mapped."));
    }
    if (mapping->permissions[0] != 'r' ||
        (requirement.require_executable &&
         mapping->permissions[2] != 'x') ||
        (requirement.require_immutable &&
         mapping->permissions[1] == 'w') ||
        mapping->permissions[3] != 'p') {
      return absl::UnimplementedError(
          absl::StrCat(requirement.description,
                       " is unreadable, writable, non-executable, or shared."));
    }
    if (mapping->inode == 0 ||
        (mapping->device_major == 0 && mapping->device_minor == 0) ||
        IsAnonymousOrMutableImagePath(mapping->path)) {
      return absl::UnimplementedError(
          absl::StrCat(requirement.description,
                       " is anonymous, deleted, or not file-backed."));
    }
    const uint64_t mapping_delta = cursor - mapping->start;
    if (mapping->file_offset >
        std::numeric_limits<uint64_t>::max() - mapping_delta) {
      return absl::FailedPreconditionError(
          absl::StrCat(requirement.description,
                       " mapped file offset overflows."));
    }
    const uint64_t actual_file_offset =
        mapping->file_offset + mapping_delta;
    const uint64_t range_delta = cursor - requirement.address;
    if (requirement.elf_file_offset >
        std::numeric_limits<uint64_t>::max() - range_delta) {
      return absl::FailedPreconditionError(
          absl::StrCat(requirement.description,
                       " ELF file offset overflows."));
    }
    const uint64_t expected_elf_offset =
        requirement.elf_file_offset + range_delta;
    if (!backing->initialized) {
      if (actual_file_offset < expected_elf_offset) {
        return absl::FailedPreconditionError(
            absl::StrCat(requirement.description,
                         " has an invalid container offset."));
      }
      backing->initialized = true;
      backing->device_major = mapping->device_major;
      backing->device_minor = mapping->device_minor;
      backing->inode = mapping->inode;
      backing->container_offset_bias =
          actual_file_offset - expected_elf_offset;
      backing->path = mapping->path;
    }
    if (mapping->device_major != backing->device_major ||
        mapping->device_minor != backing->device_minor ||
        mapping->inode != backing->inode || mapping->path != backing->path ||
        expected_elf_offset >
            std::numeric_limits<uint64_t>::max() -
                backing->container_offset_bias ||
        actual_file_offset !=
            backing->container_offset_bias + expected_elf_offset) {
      return absl::FailedPreconditionError(
          absl::StrCat(requirement.description,
                       " has ambiguous or inconsistent file backing."));
    }
    const uintptr_t span_end = std::min(end, mapping->end);
    spans->push_back(ElfMapSpan{
        .start = cursor,
        .end = span_end,
        .permissions = mapping->permissions,
        .file_offset = actual_file_offset,
    });
    cursor = span_end;
  }
  return absl::OkStatus();
}

absl::StatusOr<uintptr_t> LoadedElfAddress(ElfW(Addr) load_bias,
                                          ElfW(Addr) virtual_address) {
  if (load_bias > std::numeric_limits<uintptr_t>::max() ||
      virtual_address >
          std::numeric_limits<uintptr_t>::max() -
              static_cast<uintptr_t>(load_bias)) {
    return absl::FailedPreconditionError(
        "Loaded ELF virtual address overflows.");
  }
  return static_cast<uintptr_t>(load_bias) +
         static_cast<uintptr_t>(virtual_address);
}

struct MeasuredElfImage {
  Hash256 digest;
  uintptr_t load_bias = 0;
  std::string loader_name;
  ElfBackingIdentity backing;
  std::vector<ElfMapSpan> spans;
  uint16_t machine = 0;
  uint8_t elf_class = 0;
  uint8_t elf_data = 0;

  bool operator==(const MeasuredElfImage& other) const {
    return digest == other.digest && load_bias == other.load_bias &&
           loader_name == other.loader_name && backing == other.backing &&
           spans == other.spans && machine == other.machine &&
           elf_class == other.elf_class && elf_data == other.elf_data;
  }
};

bool ElfImageContainsAnchor(const dl_phdr_info& image,
                            uintptr_t code_anchor) {
  if (image.dlpi_phdr == nullptr || image.dlpi_phnum == 0) return false;
  for (ElfW(Half) index = 0; index < image.dlpi_phnum; ++index) {
    const ElfW(Phdr)& segment = image.dlpi_phdr[index];
    if (segment.p_type != PT_LOAD || (segment.p_flags & PF_X) == 0 ||
        segment.p_memsz == 0) {
      continue;
    }
    absl::StatusOr<uintptr_t> address =
        LoadedElfAddress(image.dlpi_addr, segment.p_vaddr);
    if (!address.ok() ||
        segment.p_memsz > std::numeric_limits<uintptr_t>::max() - *address) {
      continue;
    }
    const uintptr_t end =
        *address + static_cast<uintptr_t>(segment.p_memsz);
    if (code_anchor >= *address && code_anchor < end) return true;
  }
  return false;
}

absl::Status ValidateNoElfTextRelocations(
    const dl_phdr_info& image,
    const std::vector<ProcMapEntry>& mappings,
    ElfBackingIdentity* backing, std::vector<ElfMapSpan>* spans,
    std::vector<ElfRangeRequirement>* requirements) {
  const ElfW(Phdr)* dynamic_segment = nullptr;
  for (ElfW(Half) index = 0; index < image.dlpi_phnum; ++index) {
    const ElfW(Phdr)& segment = image.dlpi_phdr[index];
    if (segment.p_type != PT_DYNAMIC) continue;
    if (dynamic_segment != nullptr) {
      return absl::FailedPreconditionError(
          "Loaded ELF image has ambiguous dynamic metadata.");
    }
    dynamic_segment = &segment;
  }
  if (dynamic_segment == nullptr) return absl::OkStatus();
  if (dynamic_segment->p_filesz == 0 ||
      dynamic_segment->p_filesz > 1024 * 1024 ||
      dynamic_segment->p_filesz % sizeof(ElfW(Dyn)) != 0) {
    return absl::FailedPreconditionError(
        "Loaded ELF image has invalid dynamic metadata bounds.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const uintptr_t address,
      LoadedElfAddress(image.dlpi_addr, dynamic_segment->p_vaddr));
  const ElfRangeRequirement requirement{
      .address = address,
      .size = dynamic_segment->p_filesz,
      .elf_file_offset = dynamic_segment->p_offset,
      .require_executable = false,
      .require_immutable = true,
      .description = "Loaded ELF dynamic metadata",
  };
  ABSL_RETURN_IF_ERROR(
      ValidateMappedElfRange(mappings, requirement, backing, spans));
  requirements->push_back(requirement);

  const auto* entries = reinterpret_cast<const ElfW(Dyn)*>(address);
  const size_t entry_count =
      static_cast<size_t>(dynamic_segment->p_filesz / sizeof(ElfW(Dyn)));
  bool found_end = false;
  for (size_t index = 0; index < entry_count; ++index) {
    const auto tag = entries[index].d_tag;
    if (tag == DT_NULL) {
      found_end = true;
      break;
    }
    if (tag == DT_TEXTREL ||
        (tag == DT_FLAGS &&
         (entries[index].d_un.d_val & DF_TEXTREL) != 0)) {
      return absl::UnimplementedError(
          "Loaded ELF image requires executable-text relocation.");
    }
  }
  if (!found_end) {
    return absl::FailedPreconditionError(
        "Loaded ELF dynamic metadata lacks a terminator.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> ReadElfBuildId(
    const dl_phdr_info& image,
    const std::vector<ProcMapEntry>& mappings,
    ElfBackingIdentity* backing, std::vector<ElfMapSpan>* spans,
    std::vector<ElfRangeRequirement>* requirements) {
  std::optional<std::string> build_id;
  for (ElfW(Half) index = 0; index < image.dlpi_phnum; ++index) {
    const ElfW(Phdr)& segment = image.dlpi_phdr[index];
    if (segment.p_type != PT_NOTE || segment.p_filesz == 0) continue;
    if (segment.p_filesz > 1024 * 1024 ||
        segment.p_filesz > std::numeric_limits<size_t>::max()) {
      return absl::FailedPreconditionError(
          "Loaded ELF note segment has invalid bounds.");
    }
    ABSL_ASSIGN_OR_RETURN(
        const uintptr_t address,
        LoadedElfAddress(image.dlpi_addr, segment.p_vaddr));
    const ElfRangeRequirement requirement{
        .address = address,
        .size = segment.p_filesz,
        .elf_file_offset = segment.p_offset,
        .require_executable = false,
        .require_immutable = true,
        .description = "Loaded ELF note segment",
    };
    ABSL_RETURN_IF_ERROR(
        ValidateMappedElfRange(mappings, requirement, backing, spans));
    requirements->push_back(requirement);

    const auto* bytes = reinterpret_cast<const uint8_t*>(address);
    const size_t size = static_cast<size_t>(segment.p_filesz);
    size_t offset = 0;
    while (offset < size) {
      if (size - offset < sizeof(ElfW(Nhdr))) {
        return absl::FailedPreconditionError(
            "Loaded ELF note segment has a truncated note header.");
      }
      ElfW(Nhdr) note = {};
      std::memcpy(&note, bytes + offset, sizeof(note));
      offset += sizeof(note);
      const auto align_note_size = [](uint64_t value)
          -> absl::StatusOr<size_t> {
        if (value > std::numeric_limits<size_t>::max() - 3) {
          return absl::FailedPreconditionError(
              "Loaded ELF note size overflows.");
        }
        return (static_cast<size_t>(value) + 3) & ~size_t{3};
      };
      ABSL_ASSIGN_OR_RETURN(const size_t name_size,
                            align_note_size(note.n_namesz));
      ABSL_ASSIGN_OR_RETURN(const size_t descriptor_size,
                            align_note_size(note.n_descsz));
      if (name_size > size - offset ||
          descriptor_size > size - offset - name_size) {
        return absl::FailedPreconditionError(
            "Loaded ELF note payload is out of bounds.");
      }
      const uint8_t* name = bytes + offset;
      const uint8_t* descriptor = bytes + offset + name_size;
      if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
          std::memcmp(name, "GNU\0", 4) == 0) {
        // Exact profiles cannot use a short collision-prone label as the
        // immutable linker commitment for load-bearing non-code content.
        if (note.n_descsz < 16 || note.n_descsz > 64 ||
            build_id.has_value()) {
          return absl::FailedPreconditionError(
              "Loaded ELF image has an invalid or ambiguous GNU build ID.");
        }
        build_id = std::string(
            reinterpret_cast<const char*>(descriptor), note.n_descsz);
      }
      offset += name_size + descriptor_size;
    }
  }
  if (!build_id.has_value()) {
    return absl::UnimplementedError(
        "Loaded ELF image lacks an immutable GNU build ID.");
  }
  return std::move(*build_id);
}

absl::StatusOr<MeasuredElfImage> MeasureElfImage(
    const dl_phdr_info& image, uintptr_t code_anchor) {
  if (image.dlpi_phdr == nullptr || image.dlpi_phnum == 0 ||
      image.dlpi_name == nullptr) {
    return absl::FailedPreconditionError(
        "The ELF loader returned incomplete image evidence.");
  }

  struct ExecutableSegment {
    ElfW(Phdr) header;
    uintptr_t address;
  };
  std::vector<ExecutableSegment> executable_segments;
  const ElfW(Phdr)* header_segment = nullptr;
  size_t anchor_matches = 0;
  uint64_t total_executable_bytes = 0;
  for (ElfW(Half) index = 0; index < image.dlpi_phnum; ++index) {
    const ElfW(Phdr)& segment = image.dlpi_phdr[index];
    if (segment.p_type != PT_LOAD) continue;
    if (segment.p_filesz > segment.p_memsz ||
        segment.p_vaddr >
            std::numeric_limits<ElfW(Addr)>::max() - segment.p_memsz ||
        (segment.p_align > 1 &&
         ((segment.p_align & (segment.p_align - 1)) != 0 ||
          (segment.p_vaddr & (segment.p_align - 1)) !=
              (segment.p_offset & (segment.p_align - 1))))) {
      return absl::FailedPreconditionError(
          "Loaded ELF image has invalid load-segment bounds.");
    }
    if (segment.p_offset == 0 && segment.p_filesz >= sizeof(ElfW(Ehdr))) {
      if (header_segment != nullptr) {
        return absl::FailedPreconditionError(
            "Loaded ELF image has ambiguous header segments.");
      }
      header_segment = &segment;
    }
    if ((segment.p_flags & PF_X) == 0) continue;
    // Treat the ELF program-header flags as the image's maximum declared
    // segment permissions, then independently require the live VMA to be
    // private RX below. A PF_W executable is never admitted even if the
    // loader happened to remove write permission before this measurement.
    if ((segment.p_flags & PF_R) == 0 || (segment.p_flags & PF_W) != 0 ||
        segment.p_filesz == 0 || segment.p_filesz != segment.p_memsz ||
        segment.p_memsz > kMaximumElfExecutableBytes) {
      return absl::UnimplementedError(
          "Loaded ELF image has unreadable, writable, anonymous-tail, or "
          "oversized executable content.");
    }
    if (segment.p_memsz >
        kMaximumElfExecutableBytes - total_executable_bytes) {
      return absl::ResourceExhaustedError(
          "Loaded ELF executable content exceeds the measurement bound.");
    }
    total_executable_bytes += segment.p_memsz;
    ABSL_ASSIGN_OR_RETURN(
        const uintptr_t address,
        LoadedElfAddress(image.dlpi_addr, segment.p_vaddr));
    if (segment.p_memsz >
        std::numeric_limits<uintptr_t>::max() - address) {
      return absl::FailedPreconditionError(
          "Loaded ELF executable range overflows.");
    }
    const uintptr_t end = address + static_cast<uintptr_t>(segment.p_memsz);
    if (code_anchor >= address && code_anchor < end) ++anchor_matches;
    executable_segments.push_back(
        ExecutableSegment{.header = segment, .address = address});
  }
  if (header_segment == nullptr || executable_segments.empty() ||
      anchor_matches != 1) {
    return absl::FailedPreconditionError(
        "Loaded ELF image lacks an unambiguous header and anchored "
        "executable segment.");
  }
  std::sort(executable_segments.begin(), executable_segments.end(),
            [](const ExecutableSegment& left,
               const ExecutableSegment& right) {
              return left.address < right.address;
            });
  for (size_t index = 1; index < executable_segments.size(); ++index) {
    const ExecutableSegment& previous = executable_segments[index - 1];
    const ExecutableSegment& current = executable_segments[index];
    if (previous.header.p_memsz > current.address - previous.address) {
      return absl::FailedPreconditionError(
          "Loaded ELF image has overlapping executable segments.");
    }
  }

  ABSL_ASSIGN_OR_RETURN(const std::vector<ProcMapEntry> mappings,
                        ReadProcMaps());
  ElfBackingIdentity backing;
  std::vector<ElfMapSpan> spans;
  std::vector<ElfRangeRequirement> requirements;
  requirements.reserve(executable_segments.size() + 3);
  for (const ExecutableSegment& segment : executable_segments) {
    requirements.push_back(ElfRangeRequirement{
        .address = segment.address,
        .size = segment.header.p_memsz,
        .elf_file_offset = segment.header.p_offset,
        .require_executable = true,
        .require_immutable = true,
        .description = "Loaded ELF executable segment",
    });
    ABSL_RETURN_IF_ERROR(ValidateMappedElfRange(
        mappings, requirements.back(), &backing, &spans));
  }

  ABSL_ASSIGN_OR_RETURN(
      const uintptr_t header_address,
      LoadedElfAddress(image.dlpi_addr, header_segment->p_vaddr));
  const ElfRangeRequirement elf_header_requirement{
      .address = header_address,
      .size = sizeof(ElfW(Ehdr)),
      .elf_file_offset = 0,
      .require_executable = false,
      .require_immutable = true,
      .description = "Loaded ELF header",
  };
  ABSL_RETURN_IF_ERROR(ValidateMappedElfRange(
      mappings, elf_header_requirement, &backing, &spans));
  requirements.push_back(elf_header_requirement);
  const auto* header = reinterpret_cast<const ElfW(Ehdr)*>(header_address);
  if (std::memcmp(header->e_ident, ELFMAG, SELFMAG) != 0 ||
      header->e_ident[EI_VERSION] != EV_CURRENT ||
      header->e_version != EV_CURRENT ||
      (header->e_type != ET_DYN && header->e_type != ET_EXEC) ||
      header->e_machine == EM_NONE ||
      header->e_ehsize != sizeof(ElfW(Ehdr)) ||
      header->e_phentsize != sizeof(ElfW(Phdr)) ||
      header->e_phnum != image.dlpi_phnum) {
    return absl::FailedPreconditionError(
        "Loaded ELF header does not match native loader metadata.");
  }
#if __SIZEOF_POINTER__ == 8
  constexpr uint8_t kNativeElfClass = ELFCLASS64;
#else
  constexpr uint8_t kNativeElfClass = ELFCLASS32;
#endif
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  constexpr uint8_t kNativeElfData = ELFDATA2LSB;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  constexpr uint8_t kNativeElfData = ELFDATA2MSB;
#else
#error Unsupported native byte order for exact ELF identity.
#endif
  if (header->e_ident[EI_CLASS] != kNativeElfClass ||
      header->e_ident[EI_DATA] != kNativeElfData) {
    return absl::UnimplementedError(
        "Loaded ELF image class or byte order is unsupported.");
  }
  const uint64_t program_headers_size =
      static_cast<uint64_t>(image.dlpi_phnum) * sizeof(ElfW(Phdr));
  if (header->e_phoff >
          std::numeric_limits<uint64_t>::max() - program_headers_size ||
      header->e_phoff + program_headers_size > header_segment->p_filesz ||
      header->e_phoff >
          std::numeric_limits<uintptr_t>::max() - header_address ||
      reinterpret_cast<uintptr_t>(image.dlpi_phdr) !=
          header_address + static_cast<uintptr_t>(header->e_phoff)) {
    return absl::FailedPreconditionError(
        "Loaded ELF program-header table has invalid bounds or location.");
  }
  const ElfRangeRequirement program_headers_requirement{
      .address = reinterpret_cast<uintptr_t>(image.dlpi_phdr),
      .size = program_headers_size,
      .elf_file_offset = header->e_phoff,
      .require_executable = false,
      .require_immutable = true,
      .description = "Loaded ELF program headers",
  };
  ABSL_RETURN_IF_ERROR(ValidateMappedElfRange(
      mappings, program_headers_requirement, &backing, &spans));
  requirements.push_back(program_headers_requirement);
  ABSL_RETURN_IF_ERROR(ValidateNoElfTextRelocations(
      image, mappings, &backing, &spans, &requirements));
  ABSL_ASSIGN_OR_RETURN(
      const std::string build_id,
      ReadElfBuildId(image, mappings, &backing, &spans, &requirements));

  Sha256Hasher image_hasher;
  HashFrame("LITERT_LM_LOADED_ELF_IMAGE_V1", &image_hasher);
  HashFrame(build_id, &image_hasher);
  HashFrame(absl::string_view(reinterpret_cast<const char*>(header),
                              sizeof(*header)),
            &image_hasher);
  HashU64(image.dlpi_phnum, &image_hasher);
  HashFrame(absl::string_view(
                reinterpret_cast<const char*>(image.dlpi_phdr),
                static_cast<size_t>(program_headers_size)),
            &image_hasher);
  HashU64(executable_segments.size(), &image_hasher);
  for (const ExecutableSegment& segment : executable_segments) {
    HashU64(segment.header.p_vaddr, &image_hasher);
    HashU64(segment.header.p_offset, &image_hasher);
    HashU64(segment.header.p_filesz, &image_hasher);
    HashU64(segment.header.p_memsz, &image_hasher);
    HashU32(segment.header.p_flags, &image_hasher);
    HashU64(segment.header.p_align, &image_hasher);
    image_hasher.Update(absl::string_view(
        reinterpret_cast<const char*>(segment.address),
        static_cast<size_t>(segment.header.p_memsz)));
  }
  const Hash256 digest = image_hasher.Finalize();

  // Re-read mapping metadata after touching every executable byte. This makes
  // concurrent unmap, remap, mprotect, deletion, and backing-file replacement
  // a failed measurement instead of a potentially reusable identity.
  ABSL_ASSIGN_OR_RETURN(const std::vector<ProcMapEntry> mappings_after,
                        ReadProcMaps());
  ElfBackingIdentity backing_after;
  std::vector<ElfMapSpan> spans_after;
  for (const ElfRangeRequirement& requirement : requirements) {
    ABSL_RETURN_IF_ERROR(ValidateMappedElfRange(
        mappings_after, requirement, &backing_after, &spans_after));
  }
  std::sort(spans.begin(), spans.end());
  spans.erase(std::unique(spans.begin(), spans.end()), spans.end());
  std::sort(spans_after.begin(), spans_after.end());
  spans_after.erase(std::unique(spans_after.begin(), spans_after.end()),
                    spans_after.end());
  if (!(backing == backing_after) || spans != spans_after) {
    return absl::UnavailableError(
        "Loaded ELF mappings changed during identity measurement.");
  }
  return MeasuredElfImage{
      .digest = digest,
      .load_bias = static_cast<uintptr_t>(image.dlpi_addr),
      .loader_name = image.dlpi_name,
      .backing = std::move(backing),
      .spans = std::move(spans),
      .machine = header->e_machine,
      .elf_class = header->e_ident[EI_CLASS],
      .elf_data = header->e_ident[EI_DATA],
  };
}

struct ElfMeasurementContext {
  uintptr_t code_anchor = 0;
  size_t anchored_images = 0;
  absl::Status status;
  std::optional<MeasuredElfImage> measured;
};

int MeasureAnchoredElfImageCallback(dl_phdr_info* image, size_t image_size,
                                    void* opaque_context) {
  auto* context = static_cast<ElfMeasurementContext*>(opaque_context);
  if (!context->status.ok()) return 1;
  if (image == nullptr ||
      image_size < offsetof(dl_phdr_info, dlpi_phnum) +
                       sizeof(image->dlpi_phnum)) {
    context->status = absl::FailedPreconditionError(
        "The ELF loader returned a truncated image descriptor.");
    return 1;
  }
  if (!ElfImageContainsAnchor(*image, context->code_anchor)) return 0;
  ++context->anchored_images;
  if (context->anchored_images != 1) {
    context->status = absl::FailedPreconditionError(
        "Loaded code anchor has ambiguous ELF image ownership.");
    return 1;
  }
  absl::StatusOr<MeasuredElfImage> measured =
      MeasureElfImage(*image, context->code_anchor);
  if (!measured.ok()) {
    context->status = measured.status();
    return 1;
  }
  context->measured = std::move(*measured);
  return 0;
}

absl::StatusOr<MeasuredElfImage> MeasureElfImageContainingAnchorOnce(
    uintptr_t code_anchor) {
  if (code_anchor == 0) {
    return absl::FailedPreconditionError(
        "Cannot measure a zero loaded-image code anchor.");
  }
  ElfMeasurementContext context{
      .code_anchor = code_anchor,
      .status = absl::OkStatus(),
  };
  const int result =
      ::dl_iterate_phdr(MeasureAnchoredElfImageCallback, &context);
  if (!context.status.ok()) return context.status;
  if (result != 0) {
    return absl::UnavailableError(
        "The ELF loader interrupted image measurement.");
  }
  if (context.anchored_images != 1 || !context.measured.has_value()) {
    return absl::FailedPreconditionError(
        "No loaded file-backed ELF image contains the requested code "
        "anchor.");
  }
  return std::move(*context.measured);
}

void HashLinuxPlatformEvidence(const LinuxPlatformEvidence& evidence,
                               Sha256Hasher* hasher) {
  constexpr std::array<absl::string_view, 4> kUnameNames = {
      "uname.sysname", "uname.release", "uname.version", "uname.machine"};
  HashFrame("LITERT_LM_LINUX_PLATFORM_EVIDENCE_V1", hasher);
  for (size_t index = 0; index < kUnameNames.size(); ++index) {
    HashFrame(kUnameNames[index], hasher);
    HashFrame(evidence.uname_fields[index], hasher);
  }
  HashU64(evidence.effective_cpu_affinity.size(), hasher);
  for (uint32_t cpu : evidence.effective_cpu_affinity) {
    HashU32(cpu, hasher);
  }
  HashU64(evidence.page_size, hasher);
  HashU64(evidence.hardware_capabilities, hasher);
  HashU32(evidence.has_hardware_capabilities_2 ? 1 : 0, hasher);
  HashU64(evidence.hardware_capabilities_2, hasher);
  HashFrame(evidence.auxiliary_platform, hasher);
  HashFrame(evidence.auxiliary_base_platform, hasher);
  HashFrame(evidence.canonical_cpu_info, hasher);
  HashU64(evidence.named_evidence.size(), hasher);
  for (const auto& [name, value] : evidence.named_evidence) {
    HashFrame(name, hasher);
    HashFrame(value, hasher);
  }
}

absl::StatusOr<Hash256> MeasureLinuxCpuRuntimeArtifact(
    const SessionHandoffRuntimeProfile& profile) {
  if (profile.runtime_code_anchor == 0 || profile.canonical_profile.empty()) {
    return absl::FailedPreconditionError(
        "Loaded CPU runtime profile lacks measurable evidence.");
  }
  for (int attempt = 0; attempt < 3; ++attempt) {
    ABSL_ASSIGN_OR_RETURN(const LinuxPlatformEvidence first_platform,
                          CaptureLinuxPlatformEvidence());
    ABSL_ASSIGN_OR_RETURN(
        const MeasuredElfImage first_image,
        MeasureElfImageContainingAnchorOnce(profile.runtime_code_anchor));
    ABSL_ASSIGN_OR_RETURN(
        const MeasuredElfImage second_image,
        MeasureElfImageContainingAnchorOnce(profile.runtime_code_anchor));
    ABSL_ASSIGN_OR_RETURN(const LinuxPlatformEvidence second_platform,
                          CaptureLinuxPlatformEvidence());
    if (!(first_image == second_image) || !(first_platform == second_platform)) {
      continue;
    }

    Sha256Hasher hasher;
    HashFrame(kRuntimeArtifactDomain, &hasher);
    HashFrame("LINUX_ANDROID_ELF_CPU_V1", &hasher);
    HashU32(static_cast<uint32_t>(profile.runtime_class), &hasher);
    HashFrame(profile.canonical_profile, &hasher);
    hasher.Update(absl::string_view(
        reinterpret_cast<const char*>(first_image.digest.bytes.data()),
        first_image.digest.bytes.size()));
    HashLinuxPlatformEvidence(first_platform, &hasher);
    return hasher.Finalize();
  }
  return absl::UnavailableError(
      "Loaded ELF image or Linux platform evidence changed during identity "
      "measurement.");
}

#endif  // defined(__linux__) || defined(__ANDROID__)

}  // namespace

absl::StatusOr<Hash256> MeasureLoadedRuntimeArtifact(
    const SessionHandoffRuntimeProfile& profile) {
#if defined(__APPLE__)
  switch (profile.runtime_class) {
    case SessionHandoffRuntimeClass::kLiteRtCpu:
      return MeasureAppleCpuRuntimeArtifact(profile);
    case SessionHandoffRuntimeClass::kLiteRtMetal:
      return MeasureAppleMetalRuntimeArtifact(profile);
    default:
      return absl::UnimplementedError(
          "Loaded runtime/delegate identity class is unsupported.");
  }
#elif defined(__linux__) || defined(__ANDROID__)
  if (profile.runtime_class != SessionHandoffRuntimeClass::kLiteRtCpu) {
    return absl::UnimplementedError(
        "Loaded runtime/delegate identity class is unsupported.");
  }
  return MeasureLinuxCpuRuntimeArtifact(profile);
#else
  if (profile.runtime_class != SessionHandoffRuntimeClass::kLiteRtCpu) {
    return absl::UnimplementedError(
        "Loaded runtime/delegate identity class is unsupported.");
  }
  return absl::UnimplementedError(
      "Exact loaded LiteRT runtime artifact measurement is currently "
      "implemented only for Apple and ELF Linux/Android CPU processes.");
#endif
}

}  // namespace litert::lm
