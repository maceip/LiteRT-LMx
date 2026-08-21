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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_DPM_AGENT_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_DPM_AGENT_H_

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "runtime/dpm/capsule_restore_admission.h"
#include "runtime/dpm/dpm_engine.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/session_handoff.h"

namespace litert::lm {

// Production DPMAgentRuntime backed by a loaded LiteRT-LM Engine. The Engine
// remains caller-owned and must outlive this adapter and all sessions created
// through it. The adapter owns the fully-resolved DPM SessionConfig and the
// authoritative identity derived by that loaded Engine.
class EngineDPMAgentRuntime final : public DPMAgentRuntime {
 public:
  // `expected_identity` is an optional assertion only. It is compared with
  // the Engine-derived identity and can never provide the authoritative value.
  static absl::StatusOr<std::unique_ptr<EngineDPMAgentRuntime>> Create(
      Engine* engine, SessionConfig session_config,
      std::optional<SessionHandoffIdentity> expected_identity = std::nullopt);
  // Enables the independent CapsuleRestore guarantee for the exact same
  // resolved stateful SessionConfig used by this live parent runtime.
  static absl::StatusOr<std::unique_ptr<EngineDPMAgentRuntime>> Create(
      Engine* engine, SessionConfig session_config,
      CapsuleRestoreAdmissionBinding capsule_restore_admission,
      std::optional<SessionHandoffIdentity> expected_identity = std::nullopt);

  const SessionHandoffIdentity& GetSessionHandoffIdentity() const override {
    return session_handoff_identity_;
  }

  absl::Status ValidateSupport() const override;
  absl::Status ValidateSessionHandoffSupport() const override;
  absl::StatusOr<std::optional<Hash256>>
  GetCapsuleRestoreAdmissionRecordId() const override;
  absl::StatusOr<std::optional<Hash256>>
  GetSessionHandoffCapabilityId() const override;
  absl::StatusOr<std::optional<SessionHandoffCapability>>
  GetSessionHandoffCapability() const override;
  absl::StatusOr<std::optional<CapsuleRestoreOperationalCoverage>>
  GetCapsuleRestoreOperationalCoverage() const override;

  absl::StatusOr<std::unique_ptr<Engine::Session>> CreateSession() override;

  absl::StatusOr<DPMAgentGenerationOutcome> Generate(
      Engine::Session* session,
      const DPMAgentGenerationRequest& request) override;

 private:
  static absl::StatusOr<std::unique_ptr<EngineDPMAgentRuntime>> CreateInternal(
      Engine* engine, SessionConfig session_config,
      std::optional<CapsuleRestoreAdmissionBinding>
          capsule_restore_admission,
      std::optional<SessionHandoffIdentity> expected_identity);

  EngineDPMAgentRuntime(
      Engine* engine, SessionConfig resolved_session_config,
      SessionHandoffIdentity session_handoff_identity,
      std::optional<CapsuleRestoreAdmissionBinding>
          capsule_restore_admission,
      std::optional<ExactLiteRtProfile> capsule_restore_profile,
      std::optional<Hash256> capsule_restore_admission_record_id,
      std::optional<SessionHandoffCapability> session_handoff_capability,
      std::optional<CapsuleRestoreOperationalCoverage>
          capsule_restore_operational_coverage)
      : engine_(engine),
        resolved_session_config_(std::move(resolved_session_config)),
        session_handoff_identity_(session_handoff_identity),
        capsule_restore_admission_(std::move(capsule_restore_admission)),
        capsule_restore_profile_(std::move(capsule_restore_profile)),
        capsule_restore_admission_record_id_(
            capsule_restore_admission_record_id),
        session_handoff_capability_(
            std::move(session_handoff_capability)),
        capsule_restore_operational_coverage_(
            std::move(capsule_restore_operational_coverage)) {}

  absl::Status ValidateSession(const Engine::Session& session) const;
  // Generation/profile support is deliberately separate from capsule
  // support. CanonicalWinnerReplay can generate and publish a winner even
  // when the resolved session cannot export a complete handoff envelope.
  absl::Status ValidateRuntimeSupport() const;
  absl::Status ProbeSessionHandoffSupport() const;
  absl::StatusOr<AuthenticatedCapsuleRestoreAdmission>
  ResolveCurrentCapsuleRestoreAdmission() const;

  Engine* const engine_;
  const SessionConfig resolved_session_config_;
  const SessionHandoffIdentity session_handoff_identity_;
  const std::optional<CapsuleRestoreAdmissionBinding>
      capsule_restore_admission_;
  const std::optional<ExactLiteRtProfile> capsule_restore_profile_;
  const std::optional<Hash256> capsule_restore_admission_record_id_;
  const std::optional<SessionHandoffCapability>
      session_handoff_capability_;
  const std::optional<CapsuleRestoreOperationalCoverage>
      capsule_restore_operational_coverage_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_DPM_ENGINE_DPM_AGENT_H_
