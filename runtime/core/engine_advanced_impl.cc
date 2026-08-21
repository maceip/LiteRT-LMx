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

#include <atomic>
#include <cstdint>
#include <future>  // NOLINT(build/c++11)
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_macros.h"  // from @litert
#include "runtime/components/model_resources.h"
#include "runtime/core/exact_litert_profile_builder.h"
#include "runtime/core/session_handoff_identity.h"
#include "runtime/core/session_advanced.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/audio_executor_utils.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/litert_compiled_model_executor_utils.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/llm_litert_compiled_model_executor_factory.h"
#include "runtime/executor/session_handoff_runtime.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/executor/vision_executor_utils.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/serial_execution_manager.h"
#include "runtime/framework/resource_management/threaded_execution_manager.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/platform/runtime_artifact_identity.h"
#include "runtime/util/litert_util.h"
#include "runtime/util/logging.h"
#include "runtime/util/status_macros.h"  // NOLINT

#if defined(LITERT_LM_DEBUGGER_ENABLED)
#include "runtime/util/runtime_debugger.h"
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)

namespace litert::lm {

struct LoadedRuntimeIdentity {
  SessionHandoffRuntimeClass runtime_class;
  Hash256 runtime_artifact_hash;
  std::string canonical_profile;
  ExactLiteRtLogitsFrameContract logits_frame;
  uint32_t cpu_thread_count = 0;
  int32_t prefill_chunk_size = 0;
  uint32_t metal_corun_evidence = 0;
  std::string canonical_metal_policy;
};

class EngineAdvancedImpl : public Engine {
 public:
  ~EngineAdvancedImpl() override {
    auto status = WaitUntilDone(Engine::kDefaultTimeout);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to wait for engine to finish: " << status;
    }

    if (living_sessions_ > 0) {
      ABSL_LOG(ERROR) << "EngineAdvancedImpl destructed with "
                      << living_sessions_ << " living sessions!";
    }

    execution_manager_.reset();
    owned_env_.reset();
    tokenizer_.reset();
    litert_model_resources_.reset();
  }

  static absl::StatusOr<std::unique_ptr<Engine>> Create(
      EngineSettings engine_settings, absl::string_view input_prompt_as_hint);

  EngineAdvancedImpl(EngineSettings engine_settings,
                     std::unique_ptr<ModelResources> litert_model_resources,
                     std::unique_ptr<OwnedEnvironment> owned_env,
                     std::unique_ptr<Tokenizer> tokenizer,
                     std::unique_ptr<ExecutionManager> execution_manager,
                     std::optional<BenchmarkInfo> benchmark_info,
                     Hash256 model_artifact_hash,
                     absl::Status model_artifact_post_load_status,
                     absl::StatusOr<LoadedExactLiteRtEvidence>
                         loaded_exact_litert_evidence,
                     absl::StatusOr<ExactLiteRtLogitsFrameContract>
                         loaded_exact_logits_frame_contract,
                     absl::StatusOr<Hash256>
                         loaded_complete_session_handoff_state_inventory_hash,
                     absl::StatusOr<LoadedRuntimeIdentity>
                         loaded_runtime_identity)
      : engine_settings_(std::move(engine_settings)),
        litert_model_resources_(std::move(litert_model_resources)),
        owned_env_(std::move(owned_env)),
        tokenizer_(std::move(tokenizer)),
        execution_manager_(std::move(execution_manager)),
        benchmark_info_(std::move(benchmark_info)),
        model_artifact_hash_(model_artifact_hash),
        model_artifact_post_load_status_(
            std::move(model_artifact_post_load_status)),
        loaded_exact_litert_evidence_(
            std::move(loaded_exact_litert_evidence)),
        loaded_exact_logits_frame_contract_(
            std::move(loaded_exact_logits_frame_contract)),
        loaded_complete_session_handoff_state_inventory_hash_(
            std::move(
                loaded_complete_session_handoff_state_inventory_hash)),
        loaded_runtime_identity_(std::move(loaded_runtime_identity)) {}

  // Method to create the Session.
  absl::StatusOr<std::unique_ptr<Session>> CreateSession(
      const SessionConfig& session_config) override {
    std::optional<BenchmarkInfo> session_benchmark_info;
    if (benchmark_info_.has_value()) {
      // Each session will have its own benchmark info, which will be populated
      // with the session-specific information.
      session_benchmark_info = benchmark_info_;
      ABSL_RETURN_IF_ERROR(session_benchmark_info->TimeInitPhaseStart(
          BenchmarkInfo::InitPhase::kSession));
    }

    SessionConfig config = session_config;
    // TODO(b/418794726): Move this logics to be part of the SessionConfig
    // class.
    ABSL_RETURN_IF_ERROR(config.MaybeUpdateAndValidate(engine_settings_));

    std::optional<SessionHandoffIdentity> session_handoff_identity;
    absl::StatusOr<SessionHandoffIdentity> identity =
        ResolveSessionHandoffIdentity(config);
    if (identity.ok()) {
      session_handoff_identity = *identity;
    }
    std::optional<ExactLiteRtLogitsFrameContract> exact_logits_frame_contract;
    if (loaded_exact_logits_frame_contract_.ok()) {
      exact_logits_frame_contract = *loaded_exact_logits_frame_contract_;
    }

    if (litert_model_resources_ == nullptr) {
      return absl::FailedPreconditionError(
          "Model resources are not initialized.");
    }

    ABSL_ASSIGN_OR_RETURN(
        auto session,
        SessionAdvanced::CreateWithEngineOwnedIdentity(
            execution_manager_, tokenizer_.get(), config,
            std::move(session_benchmark_info), &living_sessions_,
            std::move(session_handoff_identity),
            std::move(exact_logits_frame_contract)));

    if (benchmark_info_.has_value()) {
      auto session_benchmark_info_or = session->GetMutableBenchmarkInfo();
      if (session_benchmark_info_or.ok()) {
        ABSL_RETURN_IF_ERROR(
            session_benchmark_info_or.value()->TimeInitPhaseEnd(
                BenchmarkInfo::InitPhase::kSession));
      }
    }
    return session;
  }
  absl::Status WaitUntilDone(absl::Duration timeout) override {
    return execution_manager_->WaitUntilAllDone(timeout);
  }

  const EngineSettings& GetEngineSettings() const override {
    return engine_settings_;
  }

  const Tokenizer& GetTokenizer() const override { return *tokenizer_; }

  absl::StatusOr<SessionHandoffIdentity> ResolveSessionHandoffIdentity(
      const SessionConfig& session_config) const override {
    if (model_artifact_hash_ == Hash256{}) {
      return absl::FailedPreconditionError(
          "The exact retained model artifact was not measurable.");
    }
    if (!model_artifact_post_load_status_.ok()) {
      return model_artifact_post_load_status_;
    }
    if (litert_model_resources_ == nullptr) {
      return absl::FailedPreconditionError(
          "The retained model resources are unavailable.");
    }
    // Full hashing is deliberately confined to initialization and post-load
    // admission: repeating a multi-GB digest on every DPM turn would make the
    // product path unusable. Later checks reject descriptor-size drift, or
    // mapped-extent drift when a caller supplied only a mapping. The retained
    // source must remain unmodified for the complete Engine lifetime; this is
    // a cooperative invariant, not hostile-alias protection.
    ABSL_RETURN_IF_ERROR(litert_model_resources_->VerifyModelArtifactSize());
    if (!loaded_runtime_identity_.ok()) {
      return loaded_runtime_identity_.status();
    }
    if (loaded_runtime_identity_->runtime_artifact_hash == Hash256{} ||
        loaded_runtime_identity_->canonical_profile.empty()) {
      return absl::FailedPreconditionError(
          "The loaded runtime/delegate evidence is incomplete.");
    }
    SessionConfig resolved = session_config;
    ABSL_RETURN_IF_ERROR(resolved.MaybeUpdateAndValidate(engine_settings_));
    ABSL_ASSIGN_OR_RETURN(
        Hash256 inference_profile_hash,
        DeriveSessionHandoffInferenceProfileHash(
            engine_settings_, resolved,
            loaded_runtime_identity_->canonical_profile,
            tokenizer_->GetTokenizerType()));
    if (inference_profile_hash == Hash256{}) {
      return absl::FailedPreconditionError(
          "The resolved inference profile hash is zero.");
    }
    return SessionHandoffIdentity{
        .model_artifact_hash = model_artifact_hash_,
        .runtime_artifact_hash =
            loaded_runtime_identity_->runtime_artifact_hash,
        .inference_profile_hash = inference_profile_hash,
    };
  }

  ExactLiteRtProfileCapability GetExactLiteRtProfileCapability()
      const override {
    uint32_t engine_evidence = 0;
    const bool retained_model_is_current =
        model_artifact_hash_ != Hash256{} &&
        model_artifact_post_load_status_.ok() &&
        litert_model_resources_ != nullptr &&
        litert_model_resources_->VerifyModelArtifactSize().ok();
    if (retained_model_is_current) {
      engine_evidence |=
          ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kModelArtifact);
    }
    if (retained_model_is_current && loaded_exact_litert_evidence_.ok()) {
      engine_evidence |=
          ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kTokenizerContract) |
          ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kLiteRtModelBytecode);
    }
    if (loaded_runtime_identity_.ok()) {
      const Backend configured_backend =
          engine_settings_.GetMainExecutorSettings().GetBackend();
      if (configured_backend == Backend::CPU &&
          loaded_runtime_identity_->runtime_class ==
              SessionHandoffRuntimeClass::kLiteRtCpu) {
        engine_evidence |=
            ExactLiteRtEvidenceBit(
                ExactLiteRtEvidence::kRuntimeAndDelegateBinary) |
            ExactLiteRtEvidenceBit(
                ExactLiteRtEvidence::kOperatingSystemAndDevice) |
            ExactLiteRtEvidenceBit(
                ExactLiteRtEvidence::kCompilationPrecisionAndQuantization) |
            ExactLiteRtEvidenceBit(
                ExactLiteRtEvidence::kExecutionShapeThreadingAndChunking);
      } else if (configured_backend == Backend::GPU &&
                 loaded_runtime_identity_->runtime_class ==
                     SessionHandoffRuntimeClass::kLiteRtMetal) {
        engine_evidence |=
            ExactLiteRtEvidenceBit(
                ExactLiteRtEvidence::kRuntimeAndDelegateBinary) |
            ExactLiteRtEvidenceBit(
                ExactLiteRtEvidence::kOperatingSystemAndDevice) |
            ExactLiteRtEvidenceBit(
                ExactLiteRtEvidence::kMetalDeviceAndFamily) |
            ExactLiteRtEvidenceBit(
                ExactLiteRtEvidence::kSelectedMetalDelegate);
        const uint32_t corun =
            loaded_runtime_identity_->metal_corun_evidence;
        if ((corun & MetalCoRunEvidenceBit(
                         MetalCoRunEvidence::kFixedPrefillSchedule)) != 0) {
          engine_evidence |= ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kFixedPrefillSchedule);
        }
        if ((corun & MetalCoRunEvidenceBit(
                         MetalCoRunEvidence::kFixedShapeDecode)) != 0) {
          engine_evidence |= ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kFixedShapeDecode);
        }
        if ((corun & MetalCoRunEvidenceBit(
                         MetalCoRunEvidence::kFixedPrefillSchedule)) != 0 &&
            (corun & MetalCoRunEvidenceBit(
                         MetalCoRunEvidence::kFixedShapeDecode)) != 0) {
          engine_evidence |= ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kExecutionShapeThreadingAndChunking);
        }
        if ((corun & MetalCoRunEvidenceBit(
                         MetalCoRunEvidence::kAdaptiveSplitKvDisabled)) != 0) {
          engine_evidence |= ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kAdaptiveSplitKvDisabled);
        }
        if ((corun & MetalCoRunEvidenceBit(
                         MetalCoRunEvidence::kQuiescentExecution)) != 0) {
          engine_evidence |= ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kQuiescentGpuExecution);
        }
        if ((corun & MetalCoRunEvidenceBit(
                         MetalCoRunEvidence::kCompleteSessionAndResetState)) !=
            0) {
          engine_evidence |= ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kCompleteGpuSessionAndResetState);
        }
        if ((corun & MetalCoRunEvidenceBit(
                         MetalCoRunEvidence::kSelectedKernelPipeline)) != 0) {
          engine_evidence |= ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kSelectedMetalKernelPipeline);
          // Effective precision/quantization/compilation flags are only proven
          // together with the selected compiled pipeline. Requested GpuOptions
          // can be downgraded by the Metal delegate and are not sufficient.
          engine_evidence |= ExactLiteRtEvidenceBit(
              ExactLiteRtEvidence::kCompilationPrecisionAndQuantization);
        }
      }
    }
    return DescribeExactLiteRtProfileCapability(
        engine_settings_.GetMainExecutorSettings().GetBackend(),
        engine_evidence);
  }

  absl::StatusOr<ExactLiteRtProfile> ResolveExactLiteRtProfile(
      const SessionConfig& session_config,
      const ExactLiteRtProfileAssertion& assertion) const override {
    const ExactLiteRtProfileCapability capability =
        GetExactLiteRtProfileCapability();
    switch (capability.availability) {
      case ExactLiteRtProfileAvailability::kCandidateDerivationAvailable:
        break;
      case ExactLiteRtProfileAvailability::kMetalEvidenceNotImplemented:
        return absl::UnimplementedError(
            "Exact LiteRT GPU profiles require executor-derived proof of the "
            "selected Metal delegate/plugin binary, MTLDevice, Metal family, "
            "OS build, and pinned GPU execution policy. A configured GPU "
            "label is not sufficient.");
      case ExactLiteRtProfileAvailability::kMetalPolicyEvidenceUnavailable: {
        // Capability discovery intentionally leaves session identity and the
        // concrete sampler contract unresolved. Report only engine-scoped
        // Metal blockers here so callers see the hooks that actually prevent
        // this loaded executor from becoming a candidate.
        constexpr uint32_t kSessionScopedEvidence =
            ExactLiteRtEvidenceBit(
                ExactLiteRtEvidence::kStableCpuGreedySampler) |
            ExactLiteRtEvidenceBit(ExactLiteRtEvidence::kSessionIdentity);
        return absl::UnimplementedError(absl::StrCat(
            "The loaded Metal executor is not an ExactRegeneration "
            "candidate. Missing concrete CoRun evidence: ",
            DescribeMissingExactLiteRtEvidence(
                capability.missing_evidence & ~kSessionScopedEvidence),
            ". Pinned LiteRT must expose the selected attention Split-KV "
            "policy and effective compiled Metal pipeline/precision flags; "
            "LiteRT-LM must enforce quiescent fixed-shape decode and inventory "
            "the complete GPU session capsule before these bits can be "
            "bound."));
      }
      case ExactLiteRtProfileAvailability::kNpuUnimplemented:
        return absl::UnimplementedError(
            "Exact LiteRT NPU profiles remain unimplemented until concrete "
            "compiler-plugin, device, topology, and execution-state evidence "
            "is enumerated.");
      case ExactLiteRtProfileAvailability::kRuntimeEvidenceUnavailable:
        if (litert_model_resources_ == nullptr) {
          return absl::FailedPreconditionError(
              "The retained model resources are unavailable.");
        }
        if (absl::Status status =
                litert_model_resources_->VerifyModelArtifactSize();
            !status.ok()) {
          return status;
        }
        if (!model_artifact_post_load_status_.ok()) {
          return model_artifact_post_load_status_;
        }
        if (!loaded_exact_litert_evidence_.ok()) {
          return loaded_exact_litert_evidence_.status();
        }
        if (!loaded_runtime_identity_.ok()) {
          return loaded_runtime_identity_.status();
        }
        return absl::FailedPreconditionError(
            "The loaded Engine lacks complete exact LiteRT evidence.");
      default:
        return absl::UnimplementedError(
            "The loaded Engine backend cannot derive an exact LiteRT "
            "profile.");
    }

    if (litert_model_resources_ == nullptr) {
      return absl::FailedPreconditionError(
          "The retained model resources are unavailable.");
    }
    // Exact-profile resolution is expected to happen once at worker/session
    // admission, so pay the full retained-artifact rehash here. Later handoff
    // checks use the cheaper size guard, but a same-size mutation must not be
    // able to mint a new exact profile.
    ABSL_RETURN_IF_ERROR(litert_model_resources_->VerifyModelArtifactHash());
    if (!model_artifact_post_load_status_.ok()) {
      return model_artifact_post_load_status_;
    }
    if (!loaded_exact_litert_evidence_.ok()) {
      return loaded_exact_litert_evidence_.status();
    }
    if (!loaded_runtime_identity_.ok()) {
      return loaded_runtime_identity_.status();
    }

    SessionConfig resolved = session_config;
    ABSL_RETURN_IF_ERROR(resolved.MaybeUpdateAndValidate(engine_settings_));
    ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity session_identity,
                          ResolveSessionHandoffIdentity(resolved));
    absl::StatusOr<ExactLiteRtProfile> derived_profile =
        engine_settings_.GetMainExecutorSettings().GetBackend() == Backend::GPU
            ? DeriveExactLiteRtMetalProfile(
                  engine_settings_, resolved, *loaded_exact_litert_evidence_,
                  loaded_runtime_identity_->runtime_class,
                  loaded_runtime_identity_->canonical_profile,
                  loaded_runtime_identity_->metal_corun_evidence,
                  loaded_runtime_identity_->canonical_metal_policy,
                  loaded_runtime_identity_->logits_frame,
                  loaded_runtime_identity_->cpu_thread_count,
                  loaded_runtime_identity_->prefill_chunk_size,
                  model_artifact_hash_,
                  loaded_runtime_identity_->runtime_artifact_hash,
                  session_identity)
            : DeriveExactLiteRtCpuProfile(
                  engine_settings_, resolved, *loaded_exact_litert_evidence_,
                  loaded_runtime_identity_->runtime_class,
                  loaded_runtime_identity_->canonical_profile,
                  loaded_runtime_identity_->logits_frame,
                  loaded_runtime_identity_->cpu_thread_count,
                  loaded_runtime_identity_->prefill_chunk_size,
                  model_artifact_hash_,
                  loaded_runtime_identity_->runtime_artifact_hash,
                  session_identity);
    ABSL_ASSIGN_OR_RETURN(ExactLiteRtProfile profile,
                          std::move(derived_profile));
    ABSL_RETURN_IF_ERROR(
        ValidateExactLiteRtProfileAssertion(profile, assertion));
    return profile;
  }

  absl::StatusOr<SessionHandoffCapability>
  ResolveSessionHandoffCapability(
      const SessionConfig& session_config,
      const SessionHandoffCapabilityAssertion& assertion) const override {
    SessionConfig resolved = session_config;
    ABSL_RETURN_IF_ERROR(resolved.MaybeUpdateAndValidate(engine_settings_));
    switch (resolved.GetMemoryStrategy()) {
      case SessionConfig::MemoryStrategy::kStateful:
        break;
      case SessionConfig::MemoryStrategy::kStatelessDeterministicProjection:
        return absl::UnimplementedError(
            "Session handoff capability is unavailable for stateless "
            "deterministic-projection sessions; capsule capture and restore "
            "require a stateful session.");
      default:
        return absl::InvalidArgumentError(
            "Session handoff capability received an unknown memory "
            "strategy.");
    }

    const Backend configured_backend =
        engine_settings_.GetMainExecutorSettings().GetBackend();
    if (configured_backend == Backend::NPU) {
      return absl::UnimplementedError(
          "Session handoff capability remains unimplemented for NPU until "
          "the compiler plugin, device-native continuation state, sampler, "
          "and all backend buffers are authoritatively inventoried.");
    }
    if (configured_backend != Backend::CPU &&
        configured_backend != Backend::GPU) {
      return absl::UnimplementedError(
          "The loaded Engine backend has no session handoff capability.");
    }
    if (!loaded_complete_session_handoff_state_inventory_hash_.ok()) {
      if (configured_backend == Backend::GPU) {
        return absl::UnimplementedError(absl::StrCat(
            "Session handoff capability is unavailable for Metal because "
            "the loaded executor has no complete GPU/native continuation "
            "state inventory: ",
            loaded_complete_session_handoff_state_inventory_hash_.status()
                .message()));
      }
      return loaded_complete_session_handoff_state_inventory_hash_.status();
    }
    if (*loaded_complete_session_handoff_state_inventory_hash_ == Hash256{}) {
      return absl::FailedPreconditionError(
          "The executor-derived complete session-state inventory hash is "
          "zero.");
    }

    // Capability derivation depends on an exact profile, never the reverse.
    // This keeps cold exact-profile identity available even for a runtime that
    // cannot yet export a complete capsule.
    ABSL_ASSIGN_OR_RETURN(
        const ExactLiteRtProfile exact_profile,
        ResolveExactLiteRtProfile(resolved, ExactLiteRtProfileAssertion{}));
    ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity session_identity,
                          ResolveSessionHandoffIdentity(resolved));
    if (exact_profile.session_identity != session_identity) {
      return absl::FailedPreconditionError(
          "Engine-derived exact profile and session handoff identity "
          "disagree.");
    }
    if (configured_backend == Backend::CPU &&
        exact_profile.backend != ExactLiteRtBackend::kCpu) {
      return absl::FailedPreconditionError(
          "Loaded CPU Engine derived a non-CPU exact profile.");
    }
    if (configured_backend == Backend::GPU &&
        exact_profile.backend != ExactLiteRtBackend::kMetalGpu) {
      return absl::FailedPreconditionError(
          "Loaded GPU Engine did not derive a concrete Metal profile.");
    }

    SessionHandoffCapability capability{
        .version = SessionHandoffCapability::kFormatVersion,
        .session_identity = session_identity,
        .exact_profile_id = exact_profile.profile_id,
        .backend = exact_profile.backend,
        .complete_state_inventory_hash =
            *loaded_complete_session_handoff_state_inventory_hash_,
        .capsule_codec_contract_hash =
            GetSessionHandoffCapsuleCodecContractHash(),
    };
    ABSL_ASSIGN_OR_RETURN(
        capability.capability_id,
        ComputeSessionHandoffCapabilityId(capability));
    ABSL_RETURN_IF_ERROR(ValidateSessionHandoffCapability(capability));
    ABSL_RETURN_IF_ERROR(
        ValidateSessionHandoffCapabilityAssertion(capability, assertion));
    return capability;
  }

  absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const override {
    return GetAudioExecutorPropertiesFromModelResources(
        *litert_model_resources_);
  }

  absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const override {
    return GetVisionExecutorPropertiesFromModelResources(
        *litert_model_resources_);
  }

 private:
  // Stored engine settings.
  EngineSettings engine_settings_;

  // Model resources, which must outlive `executor_`.
  std::unique_ptr<ModelResources> litert_model_resources_;

  // Owned environment, which must outlive `executor_`.
  std::unique_ptr<OwnedEnvironment> owned_env_;

  // Tokenizer shared by all sessions.
  std::unique_ptr<Tokenizer> tokenizer_;

  // Execution manager for the engine. All additional pointers to this object
  // must be weak pointers. The ultimate ownership of this object is in the
  // EngineAdvancedImpl.
  std::shared_ptr<ExecutionManager> execution_manager_;

  // Counter for living sessions.
  std::atomic<int> living_sessions_{0};

  // Benchmark info for the engine.
  std::optional<BenchmarkInfo> benchmark_info_;

  // Exact artifact identity retained from model/resource initialization.
  const Hash256 model_artifact_hash_;

  // Result of rehashing the retained source after model compilation and
  // tokenizer loading. Failure disables exact handoff identity without
  // changing ordinary Engine creation or inference.
  const absl::Status model_artifact_post_load_status_;

  // Ordered tokenizer and exact embedded LiteRT model bytes measured after
  // the executor and tokenizer have both finished loading.
  const absl::StatusOr<LoadedExactLiteRtEvidence>
      loaded_exact_litert_evidence_;

  // Narrow executor-owned capture contract. It is independent of session
  // capsule eligibility and broader exact-profile admission.
  const absl::StatusOr<ExactLiteRtLogitsFrameContract>
      loaded_exact_logits_frame_contract_;

  // Captured separately from loaded runtime/profile identity so failure to
  // inventory a complete capsule cannot disable cold exact-profile
  // derivation. In particular, Metal remains a profile-evidence problem and a
  // capsule-inventory problem rather than silently inheriting CPU support.
  const absl::StatusOr<Hash256>
      loaded_complete_session_handoff_state_inventory_hash_;

  // Measured runtime/delegate evidence and canonical concrete executor
  // profile. Unsupported platforms/backends retain the failure status so
  // ordinary inference remains available while capsule mode fails closed.
  const absl::StatusOr<LoadedRuntimeIdentity> loaded_runtime_identity_;
};

// Method to create Engine.
absl::StatusOr<std::unique_ptr<Engine>> EngineAdvancedImpl::Create(
    EngineSettings engine_settings, absl::string_view input_prompt_as_hint) {
  std::optional<BenchmarkInfo> benchmark_info =
      engine_settings.IsBenchmarkEnabled()
          ? std::make_optional<BenchmarkInfo>(
                engine_settings.GetBenchmarkParams().value())
          : std::nullopt;

  const auto& advanced_settings =
      engine_settings.GetMainExecutorSettings().GetAdvancedSettings();
  const bool is_npu =
      engine_settings.GetMainExecutorSettings().GetBackend() == Backend::NPU;
  // Magic-number replacement mutates the model flatbuffer in place.
  const bool enable_file_backed_model_loading =
      is_npu && advanced_settings &&
      !advanced_settings->configure_magic_numbers;

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseStart(BenchmarkInfo::InitPhase::kTotal));
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kModelAssets));
  }
  const auto& model_assets =
      engine_settings.GetMutableMainExecutorSettings().GetModelAssets();
  ABSL_ASSIGN_OR_RETURN(auto model_resources,
                        BuildLiteRtCompiledModelResources(
                            model_assets, enable_file_backed_model_loading,
                            /*enable_file_backed_for_aot_npu=*/is_npu));
  const Hash256 model_artifact_hash =
      model_resources->GetModelArtifactHash();
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
        BenchmarkInfo::InitPhase::kModelAssets));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }

  ABSL_ASSIGN_OR_RETURN(auto* llm_metadata, model_resources->GetLlmMetadata());
  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseEnd(
        BenchmarkInfo::InitPhase::kLlmMetadata));
  }
  bool hasLlmModelType = llm_metadata->has_llm_model_type();
  absl::Duration tokenizer_duration = absl::ZeroDuration();
  // This lambda is used to create the tokenizer asynchronously if the model
  // type is available, such that the tokenizer can be created in parallel with
  // the executor.
  auto create_tokenizer =
      [&tokenizer_duration,
       &model_resources]() -> absl::StatusOr<std::unique_ptr<Tokenizer>> {
    absl::Time start_time = absl::Now();
    ABSL_ASSIGN_OR_RETURN(std::unique_ptr<Tokenizer> tokenizer,
                          model_resources->GetTokenizer());
    tokenizer_duration = absl::Now() - start_time;
    return std::move(tokenizer);
  };

  const auto& main_executor_settings =
      engine_settings.GetMainExecutorSettings();

  std::future<absl::StatusOr<std::unique_ptr<Tokenizer>>> tokenizer_future;
  std::unique_ptr<Tokenizer> tokenizer;
  if (!hasLlmModelType) {
    ABSL_VLOG(1)
        << "Legacy model files don't have LlmModelType, loading tokenizer now";
    ABSL_ASSIGN_OR_RETURN(tokenizer, create_tokenizer());
    // Update and load the parameters from the model file and convert the
    // tokens to ids.
    ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
        tokenizer.get(), llm_metadata, input_prompt_as_hint,
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteAudioEncoderHw),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteAudioEncoderHw)));
  } else {
    // If the model type is available, wait for the tokenizer to be created
    // after the model is loaded.
    ABSL_VLOG(1) << "New model files have LlmModelType, loading tokenizer "
                    "asynchronously";

    if (engine_settings.GetParallelFileSectionLoading()) {
      // Launch the tokenizer creation in a separate thread in parallel with the
      // model loading.
      tokenizer_future = std::async(std::launch::async, create_tokenizer);
    } else {
      // Launch the tokenizer creation in the same thread.
      tokenizer_future = std::async(std::launch::deferred, create_tokenizer);
    }

    ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
        nullptr, llm_metadata, input_prompt_as_hint,
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteAudioEncoderHw),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteAudioEncoderHw)));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->TimeInitPhaseStart(
        BenchmarkInfo::InitPhase::kExecutor));
  }

  std::unique_ptr<OwnedEnvironment> owned_env;
  {
    ABSL_ASSIGN_OR_RETURN(
        auto temp_owned_env,
        CreateEnvironment(engine_settings, model_resources.get()));
    owned_env = std::make_unique<OwnedEnvironment>(std::move(temp_owned_env));
  }

  std::unique_ptr<LlmExecutor> executor;

  // Engine-scoped Debugger handle enabled via LITERT_LM_DEBUGGER_ENABLED=1.
  // Defaults to nullptr for zero-overhead in standard production Release
  // builds.
  std::shared_ptr<RuntimeDebugger> runtime_debugger = nullptr;
#if defined(LITERT_LM_DEBUGGER_ENABLED)
  runtime_debugger =
      RuntimeDebugger::Create(main_executor_settings.GetCacheDir());
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)

  switch (main_executor_settings.GetBackend()) {
    default: {
      ABSL_ASSIGN_OR_RETURN(executor, CreateLlmLiteRtCompiledModelExecutor(
                                          main_executor_settings,
                                          owned_env->env, *model_resources));
#if defined(LITERT_LM_DEBUGGER_ENABLED)
      if (main_executor_settings.GetBackend() == Backend::CPU ||
          main_executor_settings.GetBackend() == Backend::GPU) {
        if (auto* litert_executor =
                dynamic_cast<LlmLiteRtCompiledModelExecutorBase*>(
                    executor.get())) {
          if (runtime_debugger != nullptr) {
            litert_executor->UpdatePreGraphRunCallback(
                runtime_debugger->CreatePreGraphRunCallback());
            litert_executor->UpdatePostGraphRunCallback(
                runtime_debugger->CreatePostGraphRunCallback());
          }
        }
      }
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)
    }
  };

  std::unique_ptr<VisionExecutorSettings> vision_executor_settings_ptr;
  if (engine_settings.GetVisionExecutorSettings().has_value()) {
    vision_executor_settings_ptr = std::make_unique<VisionExecutorSettings>(
        std::move(engine_settings.GetVisionExecutorSettings().value()));
    if (vision_executor_settings_ptr->GetAdapterBackend() != Backend::CPU) {
      ABSL_LOG(WARNING) << "Vision adapter backend is not CPU, which may cause "
                           "precision loss.";
    }
  }

  std::unique_ptr<AudioExecutorSettings> audio_executor_settings_ptr;
  if (engine_settings.GetAudioExecutorSettings().has_value()) {
    audio_executor_settings_ptr = std::make_unique<AudioExecutorSettings>(
        std::move(engine_settings.GetAudioExecutorSettings().value()));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kExecutor));
  }

  if (hasLlmModelType) {
    // Now load the tokenizer and update the engine settings.
    ABSL_ASSIGN_OR_RETURN(tokenizer, tokenizer_future.get());
    ABSL_RETURN_IF_ERROR(engine_settings.MaybeUpdateAndValidate(
        tokenizer.get(), llm_metadata, input_prompt_as_hint,
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelBackendConstraint(
            ModelType::kTfLiteAudioEncoderHw),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLitePrefillDecode),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteVisionEncoder),
        model_resources->GetTFLiteModelPreferActivationType(
            ModelType::kTfLiteAudioEncoderHw)));
    // As we load the tokenizer asynchronously, we need to update the executor
    // settings after the tokenizer is loaded.
    ABSL_RETURN_IF_ERROR(executor->UpdateExecutorSettings(
        engine_settings.GetMainExecutorSettings()));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(benchmark_info->InitPhaseRecord(
        BenchmarkInfo::InitPhase::kTokenizer, tokenizer_duration));
  }

  // Capture the narrow immutable logits contract separately from session
  // capsule/runtime-profile evidence. Cold exact decode can collect evidence
  // even when handoff is not supported, but the loaded tokenizer and logits
  // vocabulary must still agree.
  absl::StatusOr<ExactLiteRtLogitsFrameContract>
      loaded_exact_logits_frame_contract =
          executor->GetExactLiteRtLogitsFrameContract();
  const int tokenizer_vocabulary_size = tokenizer->GetVocabSize();
  if (tokenizer_vocabulary_size <= 0) {
    loaded_exact_logits_frame_contract = absl::FailedPreconditionError(
        "Loaded tokenizer has no positive exact-decode vocabulary.");
  } else if (loaded_exact_logits_frame_contract.ok() &&
             loaded_exact_logits_frame_contract->vocabulary_size !=
                 static_cast<uint32_t>(tokenizer_vocabulary_size)) {
    loaded_exact_logits_frame_contract = absl::FailedPreconditionError(
        "Loaded decode logits vocabulary does not match the loaded "
        "tokenizer vocabulary.");
  }

  // Capsule inventory is intentionally measured through a separate hook from
  // exact-profile runtime evidence. Its dynamic-buffer schema is capacity
  // invariant, so this load-time commitment remains authoritative after a
  // supported KV-capacity growth; each LRTST001 capsule separately binds the
  // resulting concrete extent. A failure is retained for fail-closed
  // capability discovery without disabling ordinary or cold exact inference.
  absl::StatusOr<Hash256>
      loaded_complete_session_handoff_state_inventory_hash =
          executor->GetCompleteSessionHandoffStateInventoryHash();

  // Capture the ordered tokenizer contract and the exact embedded LiteRT
  // prefill/decode bytecode after all lazy model/tokenizer reads have
  // completed. Failure is retained as exact-profile capability evidence and
  // does not disable ordinary inference.
  absl::StatusOr<LoadedExactLiteRtEvidence> loaded_exact_litert_evidence =
      DeriveLoadedExactLiteRtEvidence(model_artifact_hash, *tokenizer,
                                      *model_resources);

  // Detect persistent drift after every identity-related lazy model/tokenizer
  // read. This does not claim atomic protection from a concurrently mutable
  // caller alias; exact modes require that retained source to remain unchanged
  // for the Engine lifetime.
  absl::Status model_artifact_post_load_status =
      model_resources->VerifyModelArtifactHash();

  absl::StatusOr<LoadedRuntimeIdentity> loaded_runtime_identity = [&]()
      -> absl::StatusOr<LoadedRuntimeIdentity> {
    ABSL_ASSIGN_OR_RETURN(
        SessionHandoffRuntimeProfile runtime_profile,
        executor->GetSessionHandoffRuntimeProfile());
    if (runtime_profile.logits_vocabulary_size <= 0) {
      return absl::FailedPreconditionError(
          "Loaded executor has no measurable logits vocabulary.");
    }
    if (runtime_profile.logits_batch_size != 1 ||
        runtime_profile.logits_sequence_size != 1 ||
        runtime_profile.logits_frame_byte_count == 0) {
      return absl::FailedPreconditionError(
          "Loaded executor has no exact batch-one, one-position logits frame "
          "contract.");
    }
    if (runtime_profile.cpu_thread_count == 0 ||
        runtime_profile.prefill_chunk_size == 0 ||
        runtime_profile.prefill_chunk_size < -1) {
      return absl::FailedPreconditionError(
          "Loaded executor has no valid retained threading and prefill "
          "contract.");
    }
    ExactLiteRtLogitsElementType exact_logits_element_type =
        ExactLiteRtLogitsElementType::kUnsupported;
    uint64_t logits_element_byte_count = 0;
    switch (runtime_profile.logits_element_type) {
      case SessionHandoffLogitsElementType::kFloat16:
        exact_logits_element_type = ExactLiteRtLogitsElementType::kFloat16;
        logits_element_byte_count = 2;
        break;
      case SessionHandoffLogitsElementType::kFloat32:
        exact_logits_element_type = ExactLiteRtLogitsElementType::kFloat32;
        logits_element_byte_count = 4;
        break;
      default:
        return absl::UnimplementedError(
            "Loaded executor logits element type has no exact frame "
            "contract.");
    }
    const uint64_t expected_logits_frame_byte_count =
        static_cast<uint64_t>(runtime_profile.logits_vocabulary_size) *
        logits_element_byte_count;
    if (runtime_profile.logits_frame_byte_count !=
        expected_logits_frame_byte_count) {
      return absl::FailedPreconditionError(
          "Loaded executor logits frame byte count is inconsistent with its "
          "runtime-derived shape and element type.");
    }
    const int tokenizer_vocabulary_size = tokenizer->GetVocabSize();
    if (tokenizer_vocabulary_size <= 0) {
      return absl::FailedPreconditionError(
          "Loaded tokenizer has no positive vocabulary size.");
    }
    if (runtime_profile.logits_vocabulary_size != tokenizer_vocabulary_size) {
      return absl::FailedPreconditionError(
          "Loaded decode logits vocabulary does not match the loaded "
          "tokenizer vocabulary.");
    }
    ABSL_ASSIGN_OR_RETURN(
        Hash256 runtime_artifact_hash,
        MeasureLoadedRuntimeArtifact(runtime_profile));
    if (runtime_artifact_hash == Hash256{}) {
      return absl::FailedPreconditionError(
          "Measured loaded runtime/delegate artifact hash is zero.");
    }
    uint32_t metal_corun_evidence = 0;
    std::string canonical_metal_policy;
    if (runtime_profile.runtime_class ==
        SessionHandoffRuntimeClass::kLiteRtMetal) {
      if (!runtime_profile.metal_corun.has_value() ||
          runtime_profile.metal_corun->canonical_policy.empty()) {
        return absl::FailedPreconditionError(
            "Concrete Metal runtime class lacks CoRun evidence.");
      }
      metal_corun_evidence =
          runtime_profile.metal_corun->derived_evidence;
      canonical_metal_policy =
          runtime_profile.metal_corun->canonical_policy;
    } else if (runtime_profile.metal_corun.has_value()) {
      return absl::FailedPreconditionError(
          "Non-Metal runtime unexpectedly supplied Metal CoRun evidence.");
    }
    return LoadedRuntimeIdentity{
        .runtime_class = runtime_profile.runtime_class,
        .runtime_artifact_hash = runtime_artifact_hash,
        .canonical_profile = std::move(runtime_profile.canonical_profile),
        .logits_frame =
            ExactLiteRtLogitsFrameContract{
                .element_type = exact_logits_element_type,
                .batch_size = static_cast<uint32_t>(
                    runtime_profile.logits_batch_size),
                .sequence_size = static_cast<uint32_t>(
                    runtime_profile.logits_sequence_size),
                .vocabulary_size = static_cast<uint32_t>(
                    runtime_profile.logits_vocabulary_size),
                .byte_count = runtime_profile.logits_frame_byte_count,
            },
        .cpu_thread_count = runtime_profile.cpu_thread_count,
        .prefill_chunk_size = runtime_profile.prefill_chunk_size,
        .metal_corun_evidence = metal_corun_evidence,
        .canonical_metal_policy = std::move(canonical_metal_policy),
    };
  }();
  std::unique_ptr<ExecutionManager> execution_manager;
  if (!engine_settings.GetSingleThreadedExecution()) {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        ThreadedExecutionManager::Create(
            tokenizer.get(), model_resources.get(), std::move(executor),
            std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env,
            /*audio_executor=*/nullptr, runtime_debugger));
  } else {
    ABSL_ASSIGN_OR_RETURN(
        execution_manager,
        SerialExecutionManager::Create(
            tokenizer.get(), model_resources.get(), std::move(executor),
            std::move(vision_executor_settings_ptr),
            std::move(audio_executor_settings_ptr), &owned_env->env,
            /*audio_executor=*/nullptr, runtime_debugger));
  }

  if (benchmark_info.has_value()) {
    ABSL_RETURN_IF_ERROR(
        benchmark_info->TimeInitPhaseEnd(BenchmarkInfo::InitPhase::kTotal));
  }

  auto llm_impl = std::make_unique<EngineAdvancedImpl>(
      std::move(engine_settings), std::move(model_resources),
      std::move(owned_env), std::move(tokenizer), std::move(execution_manager),
      std::move(benchmark_info), model_artifact_hash,
      std::move(model_artifact_post_load_status),
      std::move(loaded_exact_litert_evidence),
      std::move(loaded_exact_logits_frame_contract),
      std::move(loaded_complete_session_handoff_state_inventory_hash),
      std::move(loaded_runtime_identity));

  return llm_impl;
};

LITERT_LM_REGISTER_ENGINE(
    EngineFactory::EngineType::kAdvancedLiteRTCompiledModel,
    [](EngineSettings settings, absl::string_view input_prompt_as_hint) {
      return EngineAdvancedImpl::Create(std::move(settings),
                                        input_prompt_as_hint);
    });

}  // namespace litert::lm
