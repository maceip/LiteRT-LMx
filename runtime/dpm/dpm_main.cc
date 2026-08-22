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

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>

#include "absl/base/log_severity.h"     // from @com_google_absl
#include "absl/flags/flag.h"            // from @com_google_absl
#include "absl/flags/parse.h"           // from @com_google_absl
#include "absl/log/globals.h"           // from @com_google_absl
#include "absl/status/status.h"         // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"       // from @com_google_absl
#include "absl/strings/str_cat.h"       // from @com_google_absl
#include "absl/strings/string_view.h"   // from @com_google_absl
#include "nlohmann/json.hpp"            // from @nlohmann_json
#include "runtime/dpm/canonical_replay_catalog.h"
#include "runtime/dpm/deterministic_dpm_fact_map_projector.h"
#include "runtime/dpm/dpm_agent_replay_runtime.h"
#include "runtime/dpm/dpm_engine.h"
#include "runtime/dpm/dpm_fact_map.h"
#include "runtime/dpm/dpm_replay_mode.h"
#include "runtime/dpm/engine_dpm_agent.h"
#include "runtime/dpm/filesystem_dpm_event_log.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_factory.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/executor/llm_executor_settings.h"
#include "runtime/executor/session_handoff_runtime.h"
#include "runtime/platform/hash/hasher.h"
#include "runtime/platform/runtime_artifact_identity.h"

ABSL_FLAG(std::string, model_path, "",
          "Path to the .litertlm model used by the real LiteRT-LM Engine.");
ABSL_FLAG(std::string, storage_root, "",
          "Private durable root for one DPM log and replay catalog.");
ABSL_FLAG(std::string, log_id, "", "Immutable DPM log identity.");
ABSL_FLAG(std::string, case_id, "", "Immutable DPM case identity.");
ABSL_FLAG(std::string, operation_id, "", "Idempotent turn identity.");
ABSL_FLAG(std::string, input, "", "Untyped UTF-8 user input.");
ABSL_FLAG(std::string, fact_id, "",
          "Typed dpm.factmap fact ID to set instead of --input.");
ABSL_FLAG(std::string, fact_value, "", "Value paired with --fact_id.");
ABSL_FLAG(std::string, catalog_key_id, "local-demo-key",
          "Non-secret identifier for replay-catalog key rotation.");
ABSL_FLAG(int, max_output_tokens, 32,
          "Maximum number of greedy agent-decision tokens.");
ABSL_FLAG(int, num_cpu_threads, 0,
          "LiteRT CPU worker count; zero keeps the model default.");
ABSL_FLAG(bool, enable_ynnpack, false,
          "Delegate supported CPU operations to YNNPACK before XNNPACK.");
ABSL_FLAG(int64_t, timestamp_us, 0,
          "Optional positive deterministic input timestamp in microseconds.");
ABSL_FLAG(int64_t, response_timestamp_us, 0,
          "Optional positive deterministic response timestamp.");
ABSL_FLAG(bool, recover_only, false,
          "Recover an already committed operation without loading a model.");

namespace litert::lm {
namespace {

constexpr absl::string_view kCatalogKeyEnvironment =
    "LITERT_LMX_DPM_CATALOG_KEY";
constexpr absl::string_view kDPMCompositionImageProfile =
    "LITERT_LMX_DPM_COMPOSITION_IMAGE_V1";

void DPMCompositionImageAnchor() {}

bool IsDecisionInputKind(DPMEvent::Kind kind) {
  return kind == DPMEvent::Kind::kUser || kind == DPMEvent::Kind::kTool ||
         kind == DPMEvent::Kind::kInternal;
}

int64_t NowMicros() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

absl::StatusOr<std::filesystem::path> NormalizePath(
    const std::filesystem::path& path, absl::string_view description) {
  if (path.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat(description, " path is empty."));
  }
  std::error_code error;
  std::filesystem::path absolute = std::filesystem::absolute(path, error);
  if (error) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to resolve ", description, " path: ", error.message()));
  }
  std::filesystem::path canonical =
      std::filesystem::weakly_canonical(absolute, error);
  if (error) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unable to normalize ", description, " path: ", error.message()));
  }
  return canonical;
}

absl::Status EnsurePrivateDirectory(const std::filesystem::path& path) {
  std::error_code error;
  const std::filesystem::file_status prior =
      std::filesystem::symlink_status(path, error);
  if (error && error != std::errc::no_such_file_or_directory) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unable to inspect DPM storage directory: ", error.message()));
  }
  const bool existed = !error && std::filesystem::exists(prior);
  if (existed && (std::filesystem::is_symlink(prior) ||
                  !std::filesystem::is_directory(prior))) {
    return absl::FailedPreconditionError(
        "DPM storage path must be a non-symlink directory.");
  }
  if (!existed) {
    error.clear();
    std::filesystem::create_directories(path, error);
    if (error) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Unable to create DPM storage directory: ", error.message()));
    }
    std::filesystem::permissions(path, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace, error);
    if (error) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Unable to make DPM storage private: ", error.message()));
    }
  }
  error.clear();
  const std::filesystem::file_status current =
      std::filesystem::status(path, error);
  if (error || !std::filesystem::is_directory(current)) {
    return absl::FailedPreconditionError(
        "DPM storage directory could not be revalidated.");
  }
  constexpr std::filesystem::perms kForbidden =
      std::filesystem::perms::group_all | std::filesystem::perms::others_all;
  if ((current.permissions() & kForbidden) != std::filesystem::perms::none) {
    return absl::PermissionDeniedError(
        absl::StrCat("DPM storage directory must be owner-only (chmod 700): ",
                     path.string()));
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> MeasureDPMImplementation() {
  auto anchor_function = &DPMCompositionImageAnchor;
  uintptr_t code_anchor = 0;
  static_assert(sizeof(anchor_function) == sizeof(code_anchor));
  std::memcpy(&code_anchor, &anchor_function, sizeof(code_anchor));
  if (code_anchor == 0) {
    return absl::FailedPreconditionError(
        "DPM composition image has no live code anchor.");
  }
  return MeasureLoadedRuntimeArtifactForReproducibility(
      SessionHandoffRuntimeProfile{
          .runtime_class = SessionHandoffRuntimeClass::kLiteRtCpu,
          .canonical_profile = std::string(kDPMCompositionImageProfile),
          .runtime_code_anchor = code_anchor,
      });
}

absl::StatusOr<std::string> CatalogAuthenticationKey() {
  const char* value = std::getenv(std::string(kCatalogKeyEnvironment).c_str());
  if (value == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Set ", kCatalogKeyEnvironment,
                     " to a stable secret containing 32 to 4096 bytes."));
  }
  std::string key(value);
  if (key.size() < 32 || key.size() > 4096) {
    return absl::InvalidArgumentError(absl::StrCat(
        kCatalogKeyEnvironment, " must contain 32 to 4096 bytes."));
  }
  return key;
}

absl::Status ValidateFactId(absl::string_view id) {
  if (id.empty() || id.size() > 128) {
    return absl::InvalidArgumentError(
        "A typed fact ID must contain 1 to 128 bytes.");
  }
  for (unsigned char byte : id) {
    const bool alpha_numeric = (byte >= 'a' && byte <= 'z') ||
                               (byte >= 'A' && byte <= 'Z') ||
                               (byte >= '0' && byte <= '9');
    if (!alpha_numeric && byte != '.' && byte != '_' && byte != ':' &&
        byte != '-') {
      return absl::InvalidArgumentError(
          "A typed fact ID may contain only letters, digits, '.', '_', ':', "
          "and '-'.");
    }
  }
  return absl::OkStatus();
}

std::string BuildTypedFactPayload(absl::string_view id, absl::string_view value,
                                  uint64_t source_event) {
  nlohmann::ordered_json fact = nlohmann::ordered_json::object();
  fact["id"] = id;
  fact["value"] = value;
  fact["source_event"] = source_event;
  fact["valid_from"] = source_event;
  fact["retracted_at"] = nullptr;
  fact["prev"] = nullptr;
  nlohmann::ordered_json delta = nlohmann::ordered_json::object();
  delta["schema_id"] = kDPMToolDeltaSchemaId;
  delta["facts"] = nlohmann::ordered_json::array({std::move(fact)});
  delta["retract"] = nlohmann::ordered_json::array();
  return delta.dump();
}

const DPMEvent* FindInputEvent(const DPMLogSnapshot& snapshot,
                               absl::string_view operation_id) {
  const DPMEvent* found = nullptr;
  for (const DPMEvent& event : snapshot.events) {
    if (event.operation_id == operation_id && IsDecisionInputKind(event.kind)) {
      if (found != nullptr) return nullptr;
      found = &event;
    }
  }
  return found;
}

bool HasCommittedResponse(const DPMLogSnapshot& snapshot,
                          absl::string_view operation_id) {
  for (const DPMEvent& event : snapshot.events) {
    if (event.operation_id == operation_id &&
        event.kind == DPMEvent::Kind::kModelTurn &&
        event.turn_receipt.has_value()) {
      return true;
    }
  }
  return false;
}

absl::StatusOr<DPMTurnRequest> PrepareTypedFactTurn(
    DPMEventLog* log, absl::string_view operation_id, absl::string_view case_id,
    absl::string_view fact_id, absl::string_view fact_value,
    std::optional<int64_t> requested_timestamp,
    std::optional<int64_t> response_timestamp, const DPMFactMapLimits& limits) {
  ABSL_RETURN_IF_ERROR(ValidateFactId(fact_id));
  if (fact_value.empty() || fact_value.size() > limits.max_value_bytes) {
    return absl::InvalidArgumentError(
        "A typed fact value must be nonempty and within max_value_bytes.");
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<DPMEventLogOperationLease> lease,
                        log->AcquireOperationLease());
  if (lease == nullptr) {
    return absl::InternalError("DPM event log returned a null lease.");
  }
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot snapshot, log->Snapshot());
  if (snapshot.case_id != case_id) {
    return absl::FailedPreconditionError(
        "Typed fact case does not match the immutable DPM log case.");
  }

  const DPMEvent* existing = FindInputEvent(snapshot, operation_id);
  const uint64_t source_event =
      existing == nullptr ? snapshot.generation : existing->index;
  DPMTurnRequest request{
      .operation_id = std::string(operation_id),
      .case_id = std::string(case_id),
      .payload = BuildTypedFactPayload(fact_id, fact_value, source_event),
      .kind = DPMEvent::Kind::kTool,
      .timestamp_us = requested_timestamp,
      .response_timestamp_us = response_timestamp,
  };
  if (existing != nullptr) return request;

  if (!snapshot.events.empty() &&
      IsDecisionInputKind(snapshot.events.back().kind)) {
    return absl::FailedPreconditionError(
        "Another DPM operation has an incomplete tail turn.");
  }
  DPMEvent event{
      .index = source_event,
      .kind = DPMEvent::Kind::kTool,
      .timestamp_us = requested_timestamp.value_or(NowMicros()),
      .operation_id = std::string(operation_id),
      .case_id = std::string(case_id),
      .payload = request.payload,
  };
  if (event.timestamp_us <= 0) {
    return absl::InvalidArgumentError("Typed fact timestamp must be positive.");
  }
  request.timestamp_us = event.timestamp_us;

  // Validate the exact typed payload and resulting bounded state before the
  // durable append. This prevents a malformed CLI convenience request from
  // leaving a permanently unrecoverable input tail.
  DPMLogSnapshot proposed = snapshot;
  proposed.events.push_back(event);
  proposed.generation = proposed.events.size();
  ABSL_RETURN_IF_ERROR(
      FoldDeterministicDPMEvents(EmptyDPMFactMap(), proposed, /*range_start=*/0,
                                 proposed.events.size(), limits)
          .status());
  event.index = 0;
  ABSL_ASSIGN_OR_RETURN(
      DPMAppendResult appended,
      log->AppendIfGeneration(std::move(event), snapshot.generation));
  if (appended.event_index != source_event) {
    return absl::DataLossError(
        "Typed fact append returned a different authoritative event index.");
  }
  return request;
}

absl::StatusOr<DPMTurnRequest> RecoverRequest(DPMEventLog* log,
                                              absl::string_view operation_id,
                                              absl::string_view case_id) {
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot snapshot, log->Snapshot());
  if (snapshot.case_id != case_id) {
    return absl::FailedPreconditionError(
        "Recovery case does not match the immutable DPM log case.");
  }
  const DPMEvent* input = FindInputEvent(snapshot, operation_id);
  if (input == nullptr || !HasCommittedResponse(snapshot, operation_id)) {
    return absl::NotFoundError(
        "Recover-only requires an already committed operation ID.");
  }
  return DPMTurnRequest{
      .operation_id = input->operation_id,
      .case_id = input->case_id,
      .payload = input->payload,
      .kind = input->kind,
  };
}

absl::StatusOr<const DPMTurnReceipt*> FindReceipt(
    const DPMLogSnapshot& snapshot, const DPMTurnResult& result) {
  if (result.response_event_index >= snapshot.events.size()) {
    return absl::DataLossError(
        "DPM result response index is outside the durable log.");
  }
  const DPMEvent& response = snapshot.events[result.response_event_index];
  if (!response.turn_receipt.has_value()) {
    return absl::DataLossError("DPM response has no durable receipt.");
  }
  return &*response.turn_receipt;
}

absl::Status PrintResult(const DPMTurnResult& result,
                         const DPMLogSnapshot& snapshot,
                         const std::filesystem::path& storage_root,
                         bool operation_was_committed, bool engine_was_loaded,
                         std::optional<Hash256> implementation_hash,
                         absl::string_view runtime_identity_assurance) {
  ABSL_ASSIGN_OR_RETURN(const DPMTurnReceipt* receipt,
                        FindReceipt(snapshot, result));
  nlohmann::ordered_json output = nlohmann::ordered_json::object();
  output["status"] = "ok";
  output["storage_root"] = storage_root.string();
  output["operation_id"] = receipt->operation_id;
  output["input_event_index"] = result.input_event_index;
  output["response_event_index"] = result.response_event_index;
  output["projected_memory"] =
      nlohmann::ordered_json::parse(result.projected_memory);
  output["decision_output"] = result.decision_output;
  output["decision_token_ids"] = result.decision_token_ids;

  nlohmann::ordered_json identity = nlohmann::ordered_json::object();
  identity["model_artifact_sha256"] =
      receipt->agent_session_identity.model_artifact_hash.ToHex();
  identity["runtime_artifact_sha256"] =
      receipt->agent_session_identity.runtime_artifact_hash.ToHex();
  identity["runtime_identity_assurance_this_invocation"] =
      runtime_identity_assurance;
  identity["inference_profile_sha256"] =
      receipt->agent_session_identity.inference_profile_hash.ToHex();
  identity["dpm_reducer_contract_sha256"] =
      GetDeterministicDPMFactMapReducerContractHash().ToHex();
  identity["dpm_projector_config_sha256"] =
      receipt->projection_manifest.config_hash.ToHex();
  if (implementation_hash.has_value()) {
    identity["measured_dpm_implementation_sha256"] =
        implementation_hash->ToHex();
    identity["dpm_implementation_identity_assurance_this_invocation"] =
        "reproducibility-grade-loaded-image";
  } else {
    identity["measured_dpm_implementation_sha256"] = nullptr;
    identity["dpm_implementation_identity_assurance_this_invocation"] =
        "not-observed-recovery-only";
  }
  output["identity"] = std::move(identity);

  nlohmann::ordered_json execution = nlohmann::ordered_json::object();
  execution["litert_engine_loaded_this_invocation"] = engine_was_loaded;
  execution["committed_log_recovery"] = operation_was_committed;
  execution["model_inference_executed_this_invocation"] =
      engine_was_loaded && !operation_was_committed &&
      result.agent_producing_session_matched_output;
  execution["agent_reused_authenticated_winner"] =
      result.agent_reused_canonical_winner;
  execution["projection_replay_mode"] =
      DPMReplayModeToString(result.projection_replay_mode);
  execution["agent_replay_mode"] =
      DPMReplayModeToString(result.agent_replay_mode);
  execution["projection_manifest_sha256"] =
      result.projection_manifest_hash.ToHex();
  execution["projection_evidence_sha256"] =
      result.projection_execution_evidence_hash.ToHex();
  execution["agent_evidence_sha256"] =
      result.agent_execution_evidence_hash.ToHex();
  execution["source_event_count"] =
      receipt->projection_manifest.source_event_count;
  execution["source_prefix_sha256"] =
      receipt->projection_manifest.source_prefix_hash.ToHex();
  execution["canonical_agent_input_bytes"] =
      receipt->canonical_agent_input.size();
  execution["agent_request_sha256"] = receipt->agent_request_hash.ToHex();
  execution["agent_transcript_sha256"] = receipt->agent_transcript_hash.ToHex();
  output["execution"] = std::move(execution);

  output["claim"] = {
      {"structured_memory",
       "byte-deterministic for identical canonical log bytes, reducer "
       "contract/configuration, and bound artifact identities"},
      {"agent_output",
       "authenticated publish-once replay for this canonical request"},
      {"not_claimed",
       "independent deterministic regeneration of arbitrary model output"},
  };
  std::cout << output.dump(2) << std::endl;
  return absl::OkStatus();
}

std::optional<int64_t> PositiveFlagTimestamp(int64_t value) {
  return value == 0 ? std::nullopt : std::optional<int64_t>(value);
}

absl::Status MainHelper(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::SetMinLogLevel(absl::LogSeverityAtLeast::kError);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kError);

  const std::string storage_flag = absl::GetFlag(FLAGS_storage_root);
  const std::string log_id = absl::GetFlag(FLAGS_log_id);
  const std::string case_id = absl::GetFlag(FLAGS_case_id);
  const std::string operation_id = absl::GetFlag(FLAGS_operation_id);
  if (storage_flag.empty() || log_id.empty() || case_id.empty() ||
      operation_id.empty()) {
    return absl::InvalidArgumentError(
        "--storage_root, --log_id, --case_id, and --operation_id are "
        "required.");
  }
  const int64_t timestamp = absl::GetFlag(FLAGS_timestamp_us);
  const int64_t response_timestamp = absl::GetFlag(FLAGS_response_timestamp_us);
  if (timestamp < 0 || response_timestamp < 0) {
    return absl::InvalidArgumentError(
        "Explicit DPM timestamps must be positive or zero when omitted.");
  }

  ABSL_ASSIGN_OR_RETURN(std::filesystem::path storage_root,
                        NormalizePath(storage_flag, "DPM storage root"));
  ABSL_RETURN_IF_ERROR(EnsurePrivateDirectory(storage_root));
  const std::filesystem::path log_root = storage_root / "event-log";
  const std::filesystem::path catalog_root = storage_root / "replay-catalog";
  ABSL_RETURN_IF_ERROR(EnsurePrivateDirectory(log_root));
  ABSL_RETURN_IF_ERROR(EnsurePrivateDirectory(catalog_root));
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<FilesystemDPMEventLog> log,
      FilesystemDPMEventLog::Create(log_root, log_id, case_id));

  if (absl::GetFlag(FLAGS_recover_only)) {
    ABSL_ASSIGN_OR_RETURN(DPMTurnRequest request,
                          RecoverRequest(log.get(), operation_id, case_id));
    DPMEngine recovery_engine(log.get(), /*projection_provider=*/nullptr,
                              /*agent_runtime=*/nullptr,
                              /*checkpoint_repository=*/nullptr,
                              DPMEngineConfig{});
    ABSL_ASSIGN_OR_RETURN(DPMTurnResult result,
                          recovery_engine.RunTurn(request));
    ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot final_snapshot, log->Snapshot());
    return PrintResult(result, final_snapshot, storage_root,
                       /*operation_was_committed=*/true,
                       /*engine_was_loaded=*/false, std::nullopt,
                       "not-observed-recovery-only");
  }

  const std::string input = absl::GetFlag(FLAGS_input);
  const std::string fact_id = absl::GetFlag(FLAGS_fact_id);
  const std::string fact_value = absl::GetFlag(FLAGS_fact_value);
  const bool has_fact = !fact_id.empty() || !fact_value.empty();
  if (input.empty() == !has_fact ||
      (has_fact && (fact_id.empty() || fact_value.empty()))) {
    return absl::InvalidArgumentError(
        "Provide exactly one of --input or the --fact_id/--fact_value pair.");
  }
  const std::string model_path = absl::GetFlag(FLAGS_model_path);
  if (model_path.empty()) {
    return absl::InvalidArgumentError(
        "--model_path is required outside recover-only mode.");
  }
  const int max_output_tokens = absl::GetFlag(FLAGS_max_output_tokens);
  if (max_output_tokens <= 0 ||
      static_cast<uint64_t>(max_output_tokens) > kMaximumDPMGenerationTokens) {
    return absl::InvalidArgumentError(
        "--max_output_tokens must be between 1 and 65,536.");
  }
  ABSL_ASSIGN_OR_RETURN(std::string catalog_key, CatalogAuthenticationKey());
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<LocalFilesystemCanonicalWinnerReplayCatalog> catalog,
      LocalFilesystemCanonicalWinnerReplayCatalog::Create(
          catalog_root,
          CanonicalReplayAuthentication{
              .key_id = absl::GetFlag(FLAGS_catalog_key_id),
              .authentication_key = std::move(catalog_key),
          },
          DPMReplayMode::kCanonicalWinnerReplay));

  ABSL_ASSIGN_OR_RETURN(const Hash256 implementation_hash,
                        MeasureDPMImplementation());

  ABSL_ASSIGN_OR_RETURN(ModelAssets model_assets,
                        ModelAssets::Create(model_path));
  ABSL_ASSIGN_OR_RETURN(
      EngineSettings engine_settings,
      EngineSettings::CreateDefault(std::move(model_assets), Backend::CPU));
  auto& executor_settings = engine_settings.GetMutableMainExecutorSettings();
  // The current measured runtime identity deliberately refuses opaque
  // compilation-cache artifacts. Keep this demo path cache-free rather than
  // claiming an identity that omits their influence.
  executor_settings.SetCacheDir(":nocache");
  executor_settings.SetDisableWeightCache(true);
  executor_settings.SetDisableProgramCache(true);
  ABSL_ASSIGN_OR_RETURN(CpuConfig cpu_settings,
                        executor_settings.MutableBackendConfig<CpuConfig>());
  const int num_cpu_threads = absl::GetFlag(FLAGS_num_cpu_threads);
  if (num_cpu_threads < 0) {
    return absl::InvalidArgumentError("--num_cpu_threads cannot be negative.");
  }
  if (num_cpu_threads > 0) {
    cpu_settings.number_of_threads = num_cpu_threads;
  }
  cpu_settings.enable_ynnpack = absl::GetFlag(FLAGS_enable_ynnpack);
  executor_settings.SetBackendConfig(cpu_settings);
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<Engine> engine,
      EngineFactory::CreateDefault(std::move(engine_settings)));
  const absl::Status strong_runtime_identity =
      engine->ValidateStrongRuntimeArtifactIdentity();
  const absl::string_view runtime_identity_assurance =
      strong_runtime_identity.ok() ? "kernel-validated-loaded-image"
                                   : "reproducibility-grade-loaded-image";
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<EngineDPMAgentRuntime> inference_runtime,
      EngineDPMAgentRuntime::Create(engine.get(),
                                    SessionConfig::CreateDefault()));
  const SessionHandoffIdentity runtime_identity =
      inference_runtime->GetSessionHandoffIdentity();
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<CanonicalWinnerDPMAgentRuntime> agent_runtime,
      CanonicalWinnerDPMAgentRuntime::Create(inference_runtime.get(),
                                             catalog.get()));
  DeterministicDPMFactMapProjectorConfig projector_config{
      .implementation_artifact_hash = implementation_hash,
      .replay_token_bound = static_cast<uint32_t>(max_output_tokens),
  };
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<DeterministicDPMFactMapProjector> projector,
      DeterministicDPMFactMapProjector::Create(
          log.get(), catalog.get(), runtime_identity, projector_config));
  DPMEngine dpm_engine(
      log.get(), projector.get(), agent_runtime.get(),
      /*checkpoint_repository=*/nullptr,
      DPMEngineConfig{.max_decision_tokens = max_output_tokens});
  // This is intentionally before the CLI's typed-fact convenience append.
  // Unsupported model/runtime settings therefore leave no durable input tail.
  ABSL_RETURN_IF_ERROR(dpm_engine.GetCapabilities().status());

  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot before, log->Snapshot());
  const bool operation_was_committed =
      HasCommittedResponse(before, operation_id);
  DPMTurnRequest request;
  if (has_fact) {
    ABSL_ASSIGN_OR_RETURN(
        request,
        PrepareTypedFactTurn(log.get(), operation_id, case_id, fact_id,
                             fact_value, PositiveFlagTimestamp(timestamp),
                             PositiveFlagTimestamp(response_timestamp),
                             projector_config.limits));
  } else {
    request = DPMTurnRequest{
        .operation_id = operation_id,
        .case_id = case_id,
        .payload = input,
        .kind = DPMEvent::Kind::kUser,
        .timestamp_us = PositiveFlagTimestamp(timestamp),
        .response_timestamp_us = PositiveFlagTimestamp(response_timestamp),
    };
  }
  ABSL_ASSIGN_OR_RETURN(DPMTurnResult result, dpm_engine.RunTurn(request));
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot final_snapshot, log->Snapshot());
  return PrintResult(result, final_snapshot, storage_root,
                     operation_was_committed,
                     /*engine_was_loaded=*/true, implementation_hash,
                     runtime_identity_assurance);
}

}  // namespace
}  // namespace litert::lm

int main(int argc, char** argv) {
  const absl::Status status = litert::lm::MainHelper(argc, argv);
  if (!status.ok()) {
    std::cerr << "DPM error: " << status << std::endl;
    return 1;
  }
  return 0;
}
