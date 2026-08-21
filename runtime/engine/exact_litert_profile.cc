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

#include "runtime/engine/exact_litert_profile.h"

#include "absl/status/status.h"  // from @com_google_absl

namespace litert::lm {

absl::Status ValidateExactLiteRtProfileAssertion(
    const ExactLiteRtProfile& derived,
    const ExactLiteRtProfileAssertion& assertion) {
  if (assertion.expected_profile_id.has_value() &&
      *assertion.expected_profile_id != derived.profile_id) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the Engine-derived exact LiteRT "
        "profile identifier.");
  }
  if (assertion.expected_backend.has_value() &&
      *assertion.expected_backend != derived.backend) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the loaded exact LiteRT backend.");
  }
  if (assertion.expected_session_identity.has_value() &&
      *assertion.expected_session_identity != derived.session_identity) {
    return absl::FailedPreconditionError(
        "Caller assertion does not match the Engine-derived session "
        "identity.");
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
