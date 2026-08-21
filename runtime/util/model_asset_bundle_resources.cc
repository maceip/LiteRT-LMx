/* Copyright 2022 The MediaPipe Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "runtime/util/model_asset_bundle_resources.h"

#include <memory>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/util/memory_mapped_file.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/util/scoped_file.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/zip_utils.h"

namespace litert::lm {

namespace {

absl::StatusOr<Hash256> HashMappedArtifact(MemoryMappedFile* mapped_file,
                                           uint64_t expected_size) {
  if (mapped_file == nullptr) {
    return absl::InvalidArgumentError("Mapped model artifact is null.");
  }
  const uint64_t current_size = mapped_file->length();
  if (current_size != expected_size) {
    return absl::FailedPreconditionError(absl::StrCat(
        "The retained model artifact size changed from ", expected_size,
        " to ", current_size, " bytes."));
  }
  if (expected_size > std::numeric_limits<size_t>::max()) {
    return absl::ResourceExhaustedError(
        "Model artifact exceeds addressable memory for hashing.");
  }
  if (expected_size != 0 && mapped_file->data() == nullptr) {
    return absl::FailedPreconditionError(
        "The retained model artifact has no readable bytes.");
  }
  Sha256Hasher hasher;
  hasher.Update(absl::string_view(
      static_cast<const char*>(mapped_file->data()),
      static_cast<size_t>(expected_size)));
  return hasher.Finalize();
}

}  // namespace

/* static */
absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
ModelAssetBundleResources::Create(
    const std::string& tag,
    std::shared_ptr<ScopedFile> model_asset_bundle_file) {
  if (!model_asset_bundle_file || !model_asset_bundle_file->IsValid()) {
    return absl::InvalidArgumentError(
        "The model asset bundle file is not valid.");
  }

  ABSL_ASSIGN_OR_RETURN(
      auto mapped_model_asset_bundle_file,
      MemoryMappedFile::Create(model_asset_bundle_file->file()));

  const uint64_t model_artifact_size =
      mapped_model_asset_bundle_file->length();
  ABSL_ASSIGN_OR_RETURN(Hash256 model_artifact_hash,
                        HashMappedArtifact(
                            mapped_model_asset_bundle_file.get(),
                            model_artifact_size));
  absl::string_view data(
      reinterpret_cast<const char*>(mapped_model_asset_bundle_file->data()),
      static_cast<size_t>(mapped_model_asset_bundle_file->length()));

  ABSL_ASSIGN_OR_RETURN(auto files, ExtractFilesfromZipFile(data));

  return absl::WrapUnique(new ModelAssetBundleResources(
      tag, std::move(mapped_model_asset_bundle_file),
      std::move(model_asset_bundle_file), std::move(files),
      model_artifact_hash));
}

// static
absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
ModelAssetBundleResources::Create(
    const std::string& tag,
    std::shared_ptr<MemoryMappedFile> model_asset_bundle_file) {
  if (!model_asset_bundle_file) {
    return absl::InvalidArgumentError(
        "The model asset bundle file is not valid.");
  }

  const uint64_t model_artifact_size = model_asset_bundle_file->length();
  ABSL_ASSIGN_OR_RETURN(Hash256 model_artifact_hash,
                        HashMappedArtifact(model_asset_bundle_file.get(),
                                           model_artifact_size));
  absl::string_view data(
      reinterpret_cast<const char*>(model_asset_bundle_file->data()),
      static_cast<size_t>(model_asset_bundle_file->length()));

  ABSL_ASSIGN_OR_RETURN(auto files, ExtractFilesfromZipFile(data));

  return absl::WrapUnique(new ModelAssetBundleResources(
      tag, std::move(model_asset_bundle_file),
      /*retained_model_asset_bundle_file=*/nullptr, std::move(files),
      model_artifact_hash));
}

// static
absl::StatusOr<std::unique_ptr<ModelAssetBundleResources>>
ModelAssetBundleResources::Create(const std::string& tag,
                                  ScopedFile&& model_asset_bundle_file) {
  return Create(
      tag, std::make_shared<ScopedFile>(std::move(model_asset_bundle_file)));
}

ModelAssetBundleResources::ModelAssetBundleResources(
    std::string tag,
    std::shared_ptr<MemoryMappedFile> mapped_model_asset_bundle_file,
    std::shared_ptr<ScopedFile> retained_model_asset_bundle_file,
    absl::flat_hash_map<std::string, OffsetAndSize> files,
    Hash256 model_artifact_hash)
    : tag_(std::move(tag)),
      mapped_model_asset_bundle_file_(
          std::move(mapped_model_asset_bundle_file)),
      retained_model_asset_bundle_file_(
          std::move(retained_model_asset_bundle_file)),
      model_artifact_size_(mapped_model_asset_bundle_file_->length()),
      model_artifact_hash_(model_artifact_hash),
      files_(std::move(files)) {}

absl::Status ModelAssetBundleResources::VerifyModelArtifactHash() const {
  ABSL_RETURN_IF_ERROR(VerifyModelArtifactSize());
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 current_hash,
      HashMappedArtifact(mapped_model_asset_bundle_file_.get(),
                         model_artifact_size_));
  if (current_hash != model_artifact_hash_) {
    return absl::FailedPreconditionError(
        "The retained model artifact bytes changed after initialization.");
  }
  return absl::OkStatus();
}

absl::Status ModelAssetBundleResources::VerifyModelArtifactSize() const {
  if (mapped_model_asset_bundle_file_ == nullptr) {
    return absl::FailedPreconditionError(
        "The retained model artifact mapping is unavailable.");
  }
  uint64_t current_size;
  if (retained_model_asset_bundle_file_ != nullptr) {
    if (!retained_model_asset_bundle_file_->IsValid()) {
      return absl::FailedPreconditionError(
          "The retained model artifact descriptor is no longer valid.");
    }
    ABSL_ASSIGN_OR_RETURN(current_size,
                          retained_model_asset_bundle_file_->GetSize());
  } else {
    // A caller-supplied MemoryMappedFile does not expose its backing file.
    // Its mapped extent is the only cheap size evidence available here.
    current_size = mapped_model_asset_bundle_file_->length();
  }
  if (current_size != model_artifact_size_) {
    return absl::FailedPreconditionError(absl::StrCat(
        "The retained model artifact size changed from ",
        model_artifact_size_, " to ", current_size, " bytes."));
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::string_view> ModelAssetBundleResources::GetFile(
    absl::string_view filename) const {
  auto it = files_.find(filename);
  if (it != files_.end()) {
    return absl::string_view(
        reinterpret_cast<const char*>(mapped_model_asset_bundle_file_->data()) +
            it->second.offset,
        it->second.size);
  }
  return absl::NotFoundError(
      absl::StrCat("No file with name: ", filename,
                   ". All files in the model asset bundle are: ",
                   absl::StrJoin(ListFiles(), ", ")));
}

std::vector<absl::string_view> ModelAssetBundleResources::ListFiles() const {
  std::vector<absl::string_view> file_names;
  file_names.reserve(files_.size());
  for (const auto& [file_name, _] : files_) {
    file_names.push_back(file_name);
  }
  return file_names;
}
}  // namespace litert::lm
