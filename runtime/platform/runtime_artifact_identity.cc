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
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/platform/hash/sha256_hasher.h"

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
    "LITERT_LM_LOADED_RUNTIME_ARTIFACT_V2";

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

bool IsAppleSystemImagePath(absl::string_view path) {
  constexpr std::array<absl::string_view, 4> kPrefixes = {
      "/usr/lib/",
      "/System/Library/",
      "/System/Volumes/Preboot/Cryptexes/OS/usr/lib/",
      "/System/Volumes/Preboot/Cryptexes/OS/System/Library/",
  };
  for (absl::string_view prefix : kPrefixes) {
    if (absl::StartsWith(path, prefix)) return true;
  }
  return false;
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

absl::StatusOr<std::vector<Hash256>> MeasureRelevantImages(
    uintptr_t runtime_anchor) {
  for (int attempt = 0; attempt < 3; ++attempt) {
    const uint32_t image_count = ::_dyld_image_count();
    if (image_count == 0) {
      return absl::FailedPreconditionError(
          "dyld reported no loaded runtime images.");
    }
    std::vector<Hash256> digests;
    digests.reserve(image_count);
    std::vector<const mach_header*> image_headers;
    image_headers.reserve(image_count);
    std::vector<std::string> image_names;
    image_names.reserve(image_count);
    bool found_runtime_anchor = false;
    for (uint32_t index = 0; index < image_count; ++index) {
      const char* image_name = ::_dyld_get_image_name(index);
      if (image_name == nullptr || image_name[0] == '\0') {
        return absl::FailedPreconditionError(
            "dyld reported a loaded image without a classification path.");
      }
      const mach_header* image_header = ::_dyld_get_image_header(index);
      image_headers.push_back(image_header);
      image_names.emplace_back(image_name);
      ABSL_ASSIGN_OR_RETURN(
          MeasuredImage measured,
          MeasureImage(image_header, runtime_anchor,
                       index == 0 ||
                           !IsAppleSystemImagePath(image_names.back())));
      found_runtime_anchor =
          found_runtime_anchor || measured.contains_runtime_anchor;
      if (measured.has_digest) {
        digests.push_back(measured.digest);
      }
    }
    if (::_dyld_image_count() != image_count) continue;
    bool image_set_is_stable = true;
    for (uint32_t index = 0; index < image_count; ++index) {
      const char* image_name = ::_dyld_get_image_name(index);
      if (::_dyld_get_image_header(index) != image_headers[index] ||
          image_name == nullptr || image_names[index] != image_name) {
        image_set_is_stable = false;
        break;
      }
    }
    if (!image_set_is_stable) continue;
    if (!found_runtime_anchor) {
      return absl::FailedPreconditionError(
          "No measured loaded image contains the LiteRT runtime code anchor.");
    }
    if (digests.empty()) {
      return absl::FailedPreconditionError(
          "No loaded runtime/delegate image evidence was measured.");
    }
    std::sort(digests.begin(), digests.end());
    return digests;
  }
  return absl::UnavailableError(
      "Loaded runtime image set changed during identity measurement.");
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
  ABSL_ASSIGN_OR_RETURN(std::vector<Hash256> image_digests,
                        MeasureRelevantImages(profile.runtime_code_anchor));

  constexpr std::array<absl::string_view, 7> kPlatformEvidence = {
      "kern.osversion", "kern.osrelease", "hw.model",     "hw.machine",
      "hw.cputype",     "hw.cpusubtype",  "hw.cpufamily",
  };
  Sha256Hasher hasher;
  HashFrame(kRuntimeArtifactDomain, &hasher);
  HashU32(static_cast<uint32_t>(profile.runtime_class), &hasher);
  HashFrame(profile.canonical_profile, &hasher);
  HashU32(static_cast<uint32_t>(image_digests.size()), &hasher);
  for (const Hash256& image_digest : image_digests) {
    hasher.Update(absl::string_view(
        reinterpret_cast<const char*>(image_digest.bytes.data()),
        image_digest.bytes.size()));
  }
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
  // Also bind non-system libraries already loaded at compilation time. This
  // covers ML Drift support images while the two anchored digests above prove
  // which runtime and selected accelerator images are actually in use.
  ABSL_ASSIGN_OR_RETURN(
      std::vector<Hash256> dependency_image_digests,
      MeasureRelevantImages(profile.runtime_code_anchor));
  ABSL_ASSIGN_OR_RETURN(
      std::string metal_device_identity,
      DeriveMacOsMetalDeviceIdentity(metal.metal_device,
                                     metal.metal_command_queue));

  constexpr std::array<absl::string_view, 7> kPlatformEvidence = {
      "kern.osversion", "kern.osrelease", "hw.model",     "hw.machine",
      "hw.cputype",     "hw.cpusubtype",  "hw.cpufamily",
  };
  Sha256Hasher hasher;
  HashFrame("LITERT_LM_LOADED_METAL_RUNTIME_ARTIFACT_V1", &hasher);
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
  HashU32(static_cast<uint32_t>(dependency_image_digests.size()), &hasher);
  for (const Hash256& image_digest : dependency_image_digests) {
    hasher.Update(absl::string_view(
        reinterpret_cast<const char*>(image_digest.bytes.data()),
        image_digest.bytes.size()));
  }
  for (absl::string_view name : kPlatformEvidence) {
    ABSL_ASSIGN_OR_RETURN(std::string value, ReadSysctl(name));
    HashFrame(name, &hasher);
    HashFrame(value, &hasher);
  }
  return hasher.Finalize();
}

#endif  // defined(__APPLE__)

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
#else
  if (profile.runtime_class != SessionHandoffRuntimeClass::kLiteRtCpu) {
    return absl::UnimplementedError(
        "Loaded runtime/delegate identity class is unsupported.");
  }
  return absl::UnimplementedError(
      "Exact loaded LiteRT runtime artifact measurement is currently "
      "implemented only for Apple CPU processes.");
#endif
}

}  // namespace litert::lm
