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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CORRECTION_DIGEST_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CORRECTION_DIGEST_H_

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/dpm_event_log.h"
#include "runtime/platform/hash/hasher.h"

namespace litert::lm {

// Returns the nonzero digest for an empty correction lineage bound to this
// immutable log/case identity.
absl::StatusOr<Hash256> InitialDPMCorrectionDigest(
    absl::string_view log_id, absl::string_view case_id);

// Advances an existing lineage digest by exactly one canonical correction
// event. The event's index and exact durable fields are included in the chain.
absl::StatusOr<Hash256> AdvanceDPMCorrectionDigest(
    const Hash256& previous, const DPMEvent& correction);

// Derives the identity of the active immutable correction lineage directly
// from the raw authoritative log. Projection providers may report this value,
// but they do not get to choose it. The digest deliberately ignores ordinary
// turn events so a compatible checkpoint remains usable as the log grows; any
// newly appended correction changes the digest and invalidates descendants.
absl::StatusOr<Hash256> ComputeDPMCorrectionDigest(
    const DPMLogSnapshot& snapshot);

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_CORRECTION_DIGEST_H_
