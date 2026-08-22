// Copyright 2025 The ODML Authors.
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

#include "runtime/framework/resource_management/threaded_execution_manager.h"

#include <atomic>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @com_google_absl
#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/constrained_decoding/constraint.h"
#include "runtime/components/constrained_decoding/no_repeat_ngram_config.h"
#include "runtime/components/constrained_decoding/repetition_penalty_config.h"
#include "runtime/components/constrained_decoding/suppress_tokens_config.h"
#include "runtime/components/model_resources.h"
#include "runtime/components/sampler.h"
#include "runtime/components/sampler_factory.h"
#include "runtime/components/stop_token_detector.h"
#include "runtime/core/tasks.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/executor/audio_executor.h"
#include "runtime/executor/audio_executor_settings.h"
#include "runtime/executor/llm_executor.h"
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/executor/vision_executor_settings.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/framework/resource_management/resource_manager.h"
#include "runtime/framework/threadpool.h"
#include "runtime/util/convert_tensor_buffer.h"
#include "runtime/util/executor_data_util.h"
#include "runtime/util/status_macros.h"
#include "runtime/util/tensor_buffer_util.h"

#if defined(LITERT_LM_DEBUGGER_ENABLED)
#include "runtime/util/runtime_debugger.h"
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)

namespace litert::lm {

ThreadedExecutionManager::ThreadedExecutionManager(
    Tokenizer* absl_nonnull tokenizer,
    std::unique_ptr<ResourceManager> absl_nonnull resource_manager,
    ::litert::Environment* absl_nullable litert_env,
    std::shared_ptr<RuntimeDebugger> absl_nullable runtime_debugger)
    : tokenizer_(std::move(tokenizer)),
      resource_manager_(std::move(resource_manager)),
      litert_env_(litert_env),
      runtime_debugger_(std::move(runtime_debugger)) {
  execution_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"execution_thread_pool",
                                   /*max_num_threads=*/1);
  callback_thread_pool_ =
      std::make_unique<ThreadPool>(/*name_prefix=*/"callback_thread_pool",
                                   /*max_num_threads=*/1);
}

ThreadedExecutionManager::~ThreadedExecutionManager() {
  WaitUntilAllDone(Engine::kDefaultTimeout).IgnoreError();
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);

    if (!session_lookup_.empty()) {
      ABSL_LOG(ERROR) << "Not all sessions are released before the execution "
                         "manager is destroyed.";
    }

    absl::erase_if(task_lookup_, [](const auto& kv) {
      return IsTaskEndState(kv.second.task_state);
    });

    if (!task_lookup_.empty()) {
      ABSL_LOG(ERROR)
          << "Not all tasks are done before the execution manager is "
             "destroyed.";
      for (const auto& [task_id, task_info] : task_lookup_) {
        ABSL_LOG(ERROR) << "Task " << task_id
                        << " is not done. State: " << task_info.task_state;
      }
    }
  }

  // Workaround to avoid nonnull warning when releasing the resources.
  std::unique_ptr<ThreadPool> execution_thread_pool =
      std::move(execution_thread_pool_);
  execution_thread_pool.reset();

  std::unique_ptr<ThreadPool> callback_thread_pool =
      std::move(callback_thread_pool_);
  callback_thread_pool.reset();

  std::unique_ptr<ResourceManager> resource_manager =
      std::move(resource_manager_);
  resource_manager.reset();
}

absl::StatusOr<SessionId> ThreadedExecutionManager::RegisterNewSession(
    SessionConfig session_config, std::optional<BenchmarkInfo> benchmark_info) {
  switch (session_config.GetMemoryStrategy()) {
    case SessionConfig::MemoryStrategy::kStateful:
      break;
    case SessionConfig::MemoryStrategy::kStatelessDeterministicProjection:
      ABSL_RETURN_IF_ERROR(
          resource_manager_->ValidateDeterministicProjectionSupport());
      break;
    default:
      return absl::InvalidArgumentError("Unknown session memory strategy.");
  }
  ABSL_ASSIGN_OR_RETURN(
      auto context_handler,
      resource_manager_->CreateContextHandler(session_config));
  std::unique_ptr<Sampler> sampler;
  if (session_config.UseExternalSampler()) {
    if (session_config.GetSamplerBackend() != Backend::CPU) {
      return absl::InvalidArgumentError(
          "External sampler currently only supports CPU backend.");
    }
    ABSL_ASSIGN_OR_RETURN(
        sampler,
        CreateSampler(
            session_config.GetSamplerBackend(),
            session_config.GetNumOutputCandidates(),
            session_config.GetSamplerParams(),
            litert_env_ == nullptr
                ? std::nullopt
                : std::optional<
                      std::reference_wrapper<const ::litert::Environment>>(
                      *litert_env_)));
  }
  auto stop_token_detector = std::make_unique<StopTokenDetector>(1);
  for (const auto& stop_token_sequence : session_config.GetStopTokenIds()) {
    auto status =
        stop_token_detector->AddStopTokenSequence(stop_token_sequence);
    if (!status.ok()) {
      ABSL_LOG(ERROR) << "Failed to add stop token sequence: " << status;
    }
  }
  SessionId session_id = next_session_id_.fetch_add(1);
  auto session_info = std::make_shared<SessionInfo>(SessionInfo{
      .session_config = std::move(session_config),
      .context_handler = std::move(context_handler),
      .sampler = std::move(sampler),
      .stop_token_detector = std::move(stop_token_detector),
      .benchmark_info = std::move(benchmark_info),
  });
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (session_lookup_.contains(session_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Session ", session_id, " already exists in session list."));
    }
    if (session_info->session_config.AudioModalityEnabled()) {
      ABSL_RETURN_IF_ERROR(resource_manager_->TryLoadingAudioExecutor());
    }
    if (session_info->session_config.VisionModalityEnabled()) {
      ABSL_RETURN_IF_ERROR(resource_manager_->TryLoadingVisionExecutor());
    }
    session_lookup_.insert({session_id, std::move(session_info)});
#if defined(LITERT_LM_DEBUGGER_ENABLED)
    if (runtime_debugger_ != nullptr) {
      runtime_debugger_->RegisterDebugSession(session_id);
    }
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)
  }
  return session_id;
}

absl::Status ThreadedExecutionManager::ReleaseSession(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  if (!session_lookup_.at(session_id)->active_tasks.empty() ||
      session_lookup_.at(session_id)->handoff_in_progress ||
      session_lookup_.at(session_id)
          ->deterministic_projection_reset_owner.has_value()) {
    return absl::FailedPreconditionError(
        "Cannot release a session with active tasks, during handoff, or "
        "during deterministic projection reset.");
  }
  // If the session is the only session and it is audio modality enabled, we
  // need to reset the audio executor, so the streaming audio executor would
  // have a clean state for the next usage.
  if (session_lookup_.at(session_id)->session_config.AudioModalityEnabled() &&
      session_lookup_.size() == 1) {
    ABSL_ASSIGN_OR_RETURN(auto audio_executor,
                          resource_manager_->AcquireAudioExecutor());
    audio_executor->Reset().IgnoreError();
  }
  absl::erase_if(task_lookup_, [session_id](const auto& kv) {
    return kv.second.session_id == session_id;
  });
#if defined(LITERT_LM_DEBUGGER_ENABLED)
  if (runtime_debugger_ != nullptr) {
    runtime_debugger_->UnregisterDebugSession(session_id);
  }
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)
  session_lookup_.erase(session_id);
  if (session_lookup_.empty()) {
    resource_manager_->ResetCurrentHandler();
  }
  return absl::OkStatus();
}

absl::Status ThreadedExecutionManager::CancelAllTasksInSession(
    SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  PoisonSessionHandoff(*session_lookup_.at(session_id),
                       "Session cancellation was requested.");
  for (TaskId task_id : session_lookup_.at(session_id)->active_tasks) {
    task_lookup_.at(task_id).cancelled->store(true);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<const SessionInfo>>
ThreadedExecutionManager::GetSessionInfo(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  return session_lookup_.at(session_id);
}

absl::StatusOr<BenchmarkInfo*>
ThreadedExecutionManager::GetMutableBenchmarkInfo(SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  if (!session_lookup_.at(session_id)->benchmark_info.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " does not have benchmark info."));
  }
  return &session_lookup_.at(session_id)->benchmark_info.value();
}

absl::StatusOr<ExecutorSessionSnapshot>
ThreadedExecutionManager::ExportSessionSnapshot(
    SessionId session_id, const absl::flat_hash_set<TaskId>& boundary_tasks) {
  ExecutorSessionSnapshot result;
  absl::Status status = ExportSessionSnapshotTo(
      session_id, boundary_tasks,
      [&result](const ExecutorSessionSnapshot& metadata,
                const StateInterface& state) -> absl::Status {
        result = metadata;
        absl::StatusOr<std::string> serialized = state.Serialize();
        if (!serialized.ok()) return serialized.status();
        result.serialized_state = std::move(*serialized);
        return absl::OkStatus();
      });
  if (!status.ok()) return status;
  return result;
}

absl::Status ThreadedExecutionManager::ExportSessionSnapshotTo(
    SessionId session_id, const absl::flat_hash_set<TaskId>& boundary_tasks,
    absl::FunctionRef<absl::Status(const ExecutorSessionSnapshot&,
                                   const StateInterface&)>
        consumer) {
  std::shared_ptr<SessionInfo> session_info;
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (!session_lookup_.contains(session_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Session ", session_id, " not found in session list."));
    }
    session_info = session_lookup_.at(session_id);
    if (session_info->handoff_poisoned) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Session handoff is permanently unavailable: ",
          session_info->handoff_poison_reason));
    }
    if (!session_info->active_tasks.empty() ||
        session_info->handoff_in_progress ||
        session_info->deterministic_projection_reset_owner.has_value()) {
      return absl::FailedPreconditionError(
          "Session must be quiescent before handoff export.");
    }
    for (TaskId task_id : boundary_tasks) {
      if (!task_lookup_.contains(task_id) ||
          task_lookup_.at(task_id).session_id != session_id ||
          (task_lookup_.at(task_id).task_state != TaskState::kDone &&
           task_lookup_.at(task_id).task_state !=
               TaskState::kMaxNumTokensReached)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Session handoff boundary task did not complete successfully: ",
            task_id));
      }
    }
    session_info->handoff_in_progress = true;
  }
  absl::Cleanup clear_handoff = [this, session_id, session_info] {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (session_lookup_.contains(session_id) &&
        session_lookup_.at(session_id) == session_info) {
      session_info->handoff_in_progress = false;
    }
  };
  return resource_manager_->ExportSessionSnapshotTo(
      session_info->context_handler, session_info->last_prefill_token_id,
      consumer);
}

absl::StatusOr<std::vector<std::vector<int>>>
ThreadedExecutionManager::GetExactProcessedTokenHistory(
    SessionId session_id,
    const absl::flat_hash_set<TaskId>& boundary_tasks) {
  std::shared_ptr<SessionInfo> session_info;
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (!session_lookup_.contains(session_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Session ", session_id, " not found in session list."));
    }
    session_info = session_lookup_.at(session_id);
    if (!session_info->active_tasks.empty() ||
        session_info->handoff_in_progress ||
        session_info->deterministic_projection_reset_owner.has_value()) {
      return absl::FailedPreconditionError(
          "Session must be quiescent before exact token-history capture.");
    }
    for (TaskId task_id : boundary_tasks) {
      if (!task_lookup_.contains(task_id) ||
          task_lookup_.at(task_id).session_id != session_id ||
          (task_lookup_.at(task_id).task_state != TaskState::kDone &&
           task_lookup_.at(task_id).task_state !=
               TaskState::kMaxNumTokensReached)) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Exact token-history boundary task did not complete successfully: ",
            task_id));
      }
    }
    // Reuse the session-wide quiescence guard so task creation and release
    // cannot race the executor/context read. No handoff capability is queried.
    session_info->handoff_in_progress = true;
  }
  absl::Cleanup clear_guard = [this, session_id, session_info] {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (session_lookup_.contains(session_id) &&
        session_lookup_.at(session_id) == session_info) {
      session_info->handoff_in_progress = false;
    }
  };
  return resource_manager_->GetExactProcessedTokenHistory(
      session_info->context_handler);
}

absl::Status ThreadedExecutionManager::ImportSessionSnapshot(
    SessionId session_id, const ExecutorSessionSnapshot& snapshot) {
  StringByteSource serialized_state(snapshot.serialized_state);
  ExecutorSessionSnapshot metadata = snapshot;
  metadata.serialized_state.clear();
  return ImportSessionSnapshotFrom(session_id, metadata, serialized_state);
}

absl::Status ThreadedExecutionManager::ImportSessionSnapshotFrom(
    SessionId session_id, const ExecutorSessionSnapshot& snapshot,
    const ByteSource& serialized_state) {
  std::shared_ptr<SessionInfo> session_info;
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (!session_lookup_.contains(session_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Session ", session_id, " not found in session list."));
    }
    session_info = session_lookup_.at(session_id);
    if (session_info->handoff_poisoned) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Session handoff is permanently unavailable: ",
          session_info->handoff_poison_reason));
    }
    if (!session_info->active_tasks.empty() ||
        session_info->handoff_in_progress ||
        session_info->deterministic_projection_reset_owner.has_value()) {
      return absl::FailedPreconditionError(
          "Session must be quiescent before handoff import.");
    }
    if (session_info->session_config.GetMemoryStrategy() ==
        SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
      return absl::UnimplementedError(
          "Stateless deterministic-projection sessions do not admit session "
          "handoff; use a stateful session for authenticated capsule "
          "restore.");
    }
    session_info->handoff_in_progress = true;
  }
  absl::Cleanup clear_handoff = [this, session_id, session_info] {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (session_lookup_.contains(session_id) &&
        session_lookup_.at(session_id) == session_info) {
      session_info->handoff_in_progress = false;
    }
  };
  absl::Status status = resource_manager_->ImportSessionSnapshotFrom(
      session_info->context_handler, snapshot, serialized_state);
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (session_lookup_.contains(session_id) &&
        session_lookup_.at(session_id) == session_info && status.ok()) {
      session_info->last_prefill_token_id = snapshot.last_prefill_token_id;
    }
  }
  return status;
}

absl::StatusOr<TaskId> ThreadedExecutionManager::GetNewTaskId() {
  return next_task_id_.fetch_add(1);
}

absl::Status ThreadedExecutionManager::CreateTask(
    SessionId session_id, TaskId task_id,
    absl::AnyInvocable<void()> absl_nonnull task,
    absl::flat_hash_set<TaskId> dependent_tasks,
    bool potentially_mutating,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Session ", session_id, " not found in session list. Task ", task_id,
        " cannot be created."));
  }
  if (session_lookup_.at(session_id)->handoff_in_progress) {
    return absl::FailedPreconditionError(
        "Cannot create a task while session handoff is in progress.");
  }
  if (task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " already exists in task list."));
  }

  TaskState task_state = TaskState::kCreated;
  for (auto it = dependent_tasks.begin(); it != dependent_tasks.end();) {
    TaskId dep_task_id = *it;

    bool erase_dependency = false;
    auto task_it = task_lookup_.find(dep_task_id);
    if (task_it == task_lookup_.end()) {
      if (dep_task_id >= next_task_id_.load()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Dependency task ", dep_task_id, " is invalid."));
      }
      erase_dependency = true;
    } else {
      TaskInfo& dep_task_info = task_it->second;

      if (IsTaskEndState(dep_task_info.task_state)) {
        switch (dep_task_info.task_state) {
          case TaskState::kFailed:
            ABSL_FALLTHROUGH_INTENDED;
          case TaskState::kDependentTaskFailed:
            task_state = TaskState::kDependentTaskFailed;
            break;
          case TaskState::kCancelled:
            ABSL_FALLTHROUGH_INTENDED;
          case TaskState::kDependentTaskCancelled:
            if (task_state != TaskState::kDependentTaskFailed) {
              task_state = TaskState::kDependentTaskCancelled;
            }
            break;
          case TaskState::kDone:
            break;
          case TaskState::kMaxNumTokensReached:
            if (task_state == TaskState::kCreated) {
              task_state = TaskState::kMaxNumTokensReached;
            }
            break;
          default:
            return absl::InvalidArgumentError(
                absl::StrCat("Dependency task ", dep_task_id,
                             " is in end state ", dep_task_info.task_state,
                             " but not in Done or Cancelled or Failed state."));
        }
        erase_dependency = true;
      } else if (dep_task_info.task_state == TaskState::kLastCallbackQueued) {
        erase_dependency = true;
      } else {
        // Dependency task is not finished, so this new task must follow it.
        dep_task_info.following_tasks.insert(task_id);
      }
    }

    // `erase()` will invalidate `it`, so advance `it` first.
    auto copy_it = it++;
    if (erase_dependency) {
      dependent_tasks.erase(copy_it);
    }
  }

  const bool defer_lifecycle_callbacks =
      session_lookup_.at(session_id)->session_config.GetMemoryStrategy() ==
      SessionConfig::MemoryStrategy::kStatelessDeterministicProjection;
  if (defer_lifecycle_callbacks && IsTaskEndState(task_state)) {
    // Preserve terminal dependency semantics while moving callback delivery
    // out from under both the manager mutex and SessionAdvanced's mutex.
    const TaskState terminal_state = task_state;
    task_state = TaskState::kCreated;
    potentially_mutating = false;
    for (TaskId dependency_id : dependent_tasks) {
      auto dependency = task_lookup_.find(dependency_id);
      if (dependency != task_lookup_.end()) {
        dependency->second.following_tasks.erase(task_id);
      }
    }
    dependent_tasks.clear();
    task = [this, task_id, terminal_state]() mutable {
      auto task_info_or = StartTask(task_id);
      if (!task_info_or.ok()) {
        FinishTaskAndLogErrors(task_id, task_info_or.status(),
                               [](absl::StatusOr<Responses>) {});
        return;
      }
      auto [session_info, cancelled, callback] =
          std::move(task_info_or.value());
      if (session_info == nullptr) return;
      (void)cancelled;
      FinishTaskAndLogErrors(task_id, Responses(terminal_state),
                             std::move(callback));
    };
  }

  if (!IsTaskEndState(task_state)) {
    session_lookup_.at(session_id)->active_tasks.insert(task_id);
  }

  TaskInfo task_info;
  task_info.session_id = session_id;
  task_info.task_state = task_state;
  task_info.task = std::move(task);
  task_info.dependent_tasks = std::move(dependent_tasks);
  task_info.potentially_mutating = potentially_mutating;
  task_info.defer_lifecycle_callbacks = defer_lifecycle_callbacks;
  task_info.cancelled = cancelled;
  task_info.callback = std::move(callback);
  task_lookup_.insert({task_id, std::move(task_info)});

  if (!task_lookup_.at(task_id).defer_lifecycle_callbacks) {
    task_lookup_.at(task_id).callback(Responses(task_state));
  }

  // If there are no dependency tasks, we can queue the task immediately.
  // Otherwise, the task will be queued when all dependency tasks are done.
  if (task_state == TaskState::kCreated &&
      task_lookup_.at(task_id).dependent_tasks.empty()) {
    ABSL_RETURN_IF_ERROR(QueueTask(task_id));
  }
  return absl::OkStatus();
}

absl::Status ThreadedExecutionManager::QueueTask(TaskId task_id) {
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  if (task_lookup_.at(task_id).task_state != TaskState::kCreated) {
    auto error_status = absl::FailedPreconditionError(
        absl::StrCat("Task ", task_id, " is not in Created state."));
    if (!task_lookup_.at(task_id).defer_lifecycle_callbacks) {
      task_lookup_.at(task_id).callback(error_status);
    }
    return error_status;
  }
  if (!task_lookup_.at(task_id).dependent_tasks.empty()) {
    auto error_status = absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " has dependent tasks not finished."));
    if (!task_lookup_.at(task_id).defer_lifecycle_callbacks) {
      task_lookup_.at(task_id).callback(error_status);
    }
    return error_status;
  }

  auto task = std::move(task_lookup_.at(task_id).task);

  if (execution_thread_pool_ != nullptr) {
    ABSL_RETURN_IF_ERROR(execution_thread_pool_->Schedule(std::move(task)));
  } else {
    ABSL_LOG(ERROR) << "Execution thread pool is null, skipping task: "
                    << task_id;
  }

  if (!task_lookup_.at(task_id).defer_lifecycle_callbacks) {
    task_lookup_.at(task_id).callback(Responses(TaskState::kQueued));
  }
  ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kQueued));

  return absl::OkStatus();
}

absl::StatusOr<
    std::tuple<std::shared_ptr<SessionInfo>, std::shared_ptr<std::atomic<bool>>,
               absl::AnyInvocable<void(absl::StatusOr<Responses>)>>>
ThreadedExecutionManager::StartTask(TaskId task_id) {
  std::shared_ptr<SessionInfo> session_info;
  std::shared_ptr<std::atomic<bool>> cancelled;
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback;
  bool defer_lifecycle_callbacks = false;
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (!task_lookup_.contains(task_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Task ", task_id, " not found in task list."));
    }
    // If the task is cancelled, we don't need to start it.
    if (task_lookup_.at(task_id).task_state == TaskState::kCancelled) {
      return std::make_tuple(nullptr, nullptr, nullptr);
    }
    if (task_lookup_.at(task_id).callback == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Task ", task_id, " has no callback."));
    }
    if (task_lookup_.at(task_id).task_state != TaskState::kQueued) {
      auto error_status = absl::FailedPreconditionError(
          absl::StrCat("Task ", task_id, " is not in Queued state."));
      if (!task_lookup_.at(task_id).defer_lifecycle_callbacks) {
        task_lookup_.at(task_id).callback(error_status);
      }
      return error_status;
    }
    defer_lifecycle_callbacks =
        task_lookup_.at(task_id).defer_lifecycle_callbacks;
    if (!defer_lifecycle_callbacks) {
      task_lookup_.at(task_id).callback(Responses(TaskState::kProcessing));
    }
    ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kProcessing));
    task_lookup_.at(task_id).execution_started = true;

    if (!session_lookup_.contains(task_lookup_.at(task_id).session_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Session ", task_lookup_.at(task_id).session_id,
                       " not found in session list."));
    }
    session_info = session_lookup_.at(task_lookup_.at(task_id).session_id);
    cancelled = task_lookup_.at(task_id).cancelled;
    callback = std::move(task_lookup_.at(task_id).callback);
  }
  if (defer_lifecycle_callbacks) {
    callback(Responses(TaskState::kCreated));
    callback(Responses(TaskState::kQueued));
    callback(Responses(TaskState::kProcessing));
  }
  return std::make_tuple(std::move(session_info), std::move(cancelled),
                         std::move(callback));
}

absl::Status ThreadedExecutionManager::FinishTask(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  auto invoke_callback_and_return =
      [&](absl::Status status) ABSL_EXCLUSIVE_LOCKS_REQUIRED(
          session_and_task_lookup_mutex_) -> absl::Status {
    ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kFailed));
    if (callback_thread_pool_ == nullptr) {
      return absl::InternalError(
          "Callback thread pool is null; failure callback was not queued.");
    }
    absl::Status schedule_status = callback_thread_pool_->Schedule(
        [callback = std::move(callback), status]() mutable {
          callback(status);
        });
    if (!schedule_status.ok()) return schedule_status;
    return status;
  };
  {
    absl::MutexLock lock(session_and_task_lookup_mutex_);
    if (!task_lookup_.contains(task_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Task ", task_id, " not found in task list."));
    }
    if (task_lookup_.at(task_id).task_state != TaskState::kProcessing) {
      auto error_status = absl::FailedPreconditionError(
          absl::StrCat("Task ", task_id, " is not in Processing state."));
      return invoke_callback_and_return(error_status);
    }
    TaskInfo& completing_task = task_lookup_.at(task_id);
    const bool task_cancelled =
        responses.ok() &&
        (responses->GetTaskState() == TaskState::kCancelled ||
         responses->GetTaskState() == TaskState::kDependentTaskCancelled);
    const bool task_failed =
        !responses.ok() ||
        (responses->GetTaskState() == TaskState::kFailed ||
         responses->GetTaskState() == TaskState::kDependentTaskFailed);
    if (task_cancelled) {
      PoisonSessionHandoff(*session_lookup_.at(completing_task.session_id),
                           absl::StrCat("Task ", task_id,
                                        " was cancelled."));
    } else if (task_failed && completing_task.potentially_mutating &&
               completing_task.execution_started) {
      PoisonSessionHandoff(
          *session_lookup_.at(completing_task.session_id),
          absl::StrCat("Potentially mutating task ", task_id,
                       " failed after execution started."));
    }
    if (task_cancelled || task_failed) {
      auto following_waiting_tasks = FollowingWaitingTasks(task_id);
      if (!following_waiting_tasks.ok()) {
        return invoke_callback_and_return(following_waiting_tasks.status());
      }
      auto status = UpdateAllTasksToState(
          following_waiting_tasks.value(),
          task_cancelled ? TaskState::kDependentTaskCancelled
                         : TaskState::kDependentTaskFailed);
      if (!status.ok()) {
        return invoke_callback_and_return(status);
      }
    } else if (responses->GetTaskState() == TaskState::kDone ||
               responses->GetTaskState() == TaskState::kMaxNumTokensReached) {
      for (TaskId following_task_id :
           task_lookup_.at(task_id).following_tasks) {
        if (!task_lookup_.contains(following_task_id)) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " not found in task list."));
          return invoke_callback_and_return(error_status);
        }
        if (IsTaskEndState(task_lookup_.at(following_task_id).task_state)) {
          continue;
        }
        if (task_lookup_.at(following_task_id).task_state !=
            TaskState::kCreated) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " is not in Created state. Task state: ",
                           task_lookup_.at(following_task_id).task_state));
          return invoke_callback_and_return(error_status);
        }
        if (!task_lookup_.at(following_task_id)
                 .dependent_tasks.contains(task_id)) {
          auto error_status = absl::InvalidArgumentError(
              absl::StrCat("Following task ", following_task_id,
                           " does not depend on task ", task_id));
          return invoke_callback_and_return(error_status);
        }
        task_lookup_.at(following_task_id).dependent_tasks.erase(task_id);
        if (task_lookup_.at(following_task_id).dependent_tasks.empty()) {
          ABSL_RETURN_IF_ERROR(QueueTask(following_task_id));
        }
      }
    } else if (!IsTaskEndState(responses->GetTaskState())) {
      return invoke_callback_and_return(absl::InvalidArgumentError(absl::StrCat(
          "Expected task state for responses to be end state, but got ",
          responses->GetTaskState())));
    }

    TaskState next_task_state =
        responses.ok() ? responses->GetTaskState() : TaskState::kFailed;
    if (callback_thread_pool_ != nullptr) {
      absl::Status schedule_status = callback_thread_pool_->Schedule(
          [callback = std::move(callback), responses = std::move(responses),
           task_id = task_id, next_task_state = std::move(next_task_state),
           this]() mutable {
            callback(std::move(responses));
            absl::MutexLock lock(session_and_task_lookup_mutex_);
            auto status = UpdateTaskState(task_id, next_task_state);
            if (!status.ok()) {
              ABSL_LOG(ERROR) << "Failed to update task state: " << status
                              << " with task id: " << task_id;
            }
          });
      if (!schedule_status.ok()) {
        ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kFailed));
        return schedule_status;
      }
      ABSL_RETURN_IF_ERROR(
          UpdateTaskState(task_id, TaskState::kLastCallbackQueued));
    } else {
      absl::Status error_status = absl::InternalError(
          "Callback thread pool is null; final callback was not queued.");
      ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, TaskState::kFailed));
      return error_status;
    }
  }

  if (callback_thread_pool_ != nullptr) {
    // TODO b/476205457 - Consider to use a asynchronous approach to handle the
    // callback, and remove this WaitUntilDone.
    ABSL_RETURN_IF_ERROR(
        callback_thread_pool_->WaitUntilDone(absl::Seconds(10)));
  }

  return absl::OkStatus();
}

void ThreadedExecutionManager::FinishTaskAndLogErrors(
    TaskId task_id, absl::StatusOr<Responses> responses,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> absl_nonnull callback) {
  auto status = FinishTask(task_id, std::move(responses), std::move(callback));
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Failed to finish task: " << status
                    << " with task id: " << task_id;
  }
}

absl::StatusOr<absl::flat_hash_set<TaskId>>
ThreadedExecutionManager::FollowingWaitingTasks(TaskId task_id) {
  absl::flat_hash_set<TaskId> following_waiting_tasks;
  for (TaskId following_task_id : task_lookup_.at(task_id).following_tasks) {
    if (!task_lookup_.contains(following_task_id)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Following task ", following_task_id, " not found in task list."));
    }
    if (!task_lookup_.at(following_task_id).dependent_tasks.contains(task_id)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Following task ", following_task_id,
                       " does not depend on task ", task_id));
    }
    if (!IsTaskEndState(task_lookup_.at(following_task_id).task_state)) {
      following_waiting_tasks.insert(following_task_id);
      ABSL_ASSIGN_OR_RETURN(auto next_following_waiting_tasks,
                            FollowingWaitingTasks(following_task_id));
      following_waiting_tasks.insert(next_following_waiting_tasks.begin(),
                                     next_following_waiting_tasks.end());
    }
  }
  return following_waiting_tasks;
}

absl::Status ThreadedExecutionManager::BeginDeterministicProjectionReset(
    SessionId session_id, TaskId task_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  SessionInfo& session_info = *session_lookup_.at(session_id);
  if (session_info.deterministic_projection_reset_owner.has_value()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Session already has a deterministic projection reset owner: ",
        *session_info.deterministic_projection_reset_owner));
  }
  session_info.deterministic_projection_reset_owner = task_id;
  return absl::OkStatus();
}

void ThreadedExecutionManager::EndDeterministicProjectionReset(
    SessionId session_id, TaskId task_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) return;
  SessionInfo& session_info = *session_lookup_.at(session_id);
  if (session_info.deterministic_projection_reset_owner == task_id) {
    session_info.deterministic_projection_reset_owner.reset();
  }
}

absl::Status ThreadedExecutionManager::SetDeterministicProjectionReady(
    SessionId session_id, bool ready) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  session_lookup_.at(session_id)->deterministic_projection_ready = ready;
  return absl::OkStatus();
}

absl::Status ThreadedExecutionManager::ValidateDeterministicProjectionReady(
    SessionId session_id) {
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  if (!session_lookup_.contains(session_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Session ", session_id, " not found in session list."));
  }
  if (!session_lookup_.at(session_id)->deterministic_projection_ready) {
    return absl::FailedPreconditionError(
        "Stateless session has no successfully initialized projection epoch.");
  }
  return absl::OkStatus();
}

void ThreadedExecutionManager::PoisonSessionHandoff(
    SessionInfo& session_info, absl::string_view reason) {
  session_info.deterministic_projection_ready = false;
  if (session_info.handoff_poisoned) return;
  session_info.handoff_poisoned = true;
  session_info.handoff_poison_reason = std::string(reason);
}

absl::Status ThreadedExecutionManager::UpdateTaskState(TaskId task_id,
                                                       TaskState task_state) {
  if (!task_lookup_.contains(task_id)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Task ", task_id, " not found in task list."));
  }
  TaskInfo& task_info = task_lookup_.at(task_id);
  if (!IsTaskEndState(task_info.task_state) &&
      IsTaskEndState(task_state)) {
    SessionId session_id = task_info.session_id;
    if (session_lookup_.contains(session_id) &&
        session_lookup_.at(session_id)->active_tasks.contains(task_id)) {
      SessionInfo& session_info = *session_lookup_.at(session_id);
      if (task_state == TaskState::kCancelled ||
          task_state == TaskState::kDependentTaskCancelled) {
        PoisonSessionHandoff(
            session_info,
            absl::StrCat("Task ", task_id, " was cancelled."));
      } else if ((task_state == TaskState::kFailed ||
                  task_state == TaskState::kDependentTaskFailed) &&
                 task_info.potentially_mutating &&
                 task_info.execution_started) {
        PoisonSessionHandoff(
            session_info,
            absl::StrCat("Potentially mutating task ", task_id,
                         " failed after execution started."));
      }
      session_info.active_tasks.erase(task_id);
    } else {
      return absl::InternalError(absl::StrCat(
          "Task ", task_id, " is not in active tasks of session ", session_id));
    }
  }
  task_info.task_state = task_state;
  return absl::OkStatus();
}

absl::Status ThreadedExecutionManager::UpdateAllTasksToState(
    const absl::flat_hash_set<TaskId>& task_ids, TaskState task_state) {
  for (TaskId task_id : task_ids) {
    task_lookup_.at(task_id).dependent_tasks.clear();
    ABSL_RETURN_IF_ERROR(UpdateTaskState(task_id, task_state));
    if (task_lookup_.at(task_id).callback) {
      if (callback_thread_pool_ == nullptr) {
        return absl::InternalError(
            "Callback thread pool is null; dependent callback was not "
            "queued.");
      }
      auto callback = std::move(task_lookup_.at(task_id).callback);
      ABSL_RETURN_IF_ERROR(callback_thread_pool_->Schedule(
          [callback = std::move(callback), task_state]() mutable {
            callback(Responses(task_state));
          }));
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<ExecutorInputs>
ThreadedExecutionManager::ProcessAndCombineContents(
    const std::vector<InputData>& preprocessed_contents,
    std::optional<BenchmarkInfo>& benchmark_info) {
  std::vector<int> combined_token_ids;
  std::vector<ExecutorVisionData> all_image_data;
  std::vector<ExecutorAudioData> all_audio_data;
  for (const auto& preprocessed_content : preprocessed_contents) {
    if (const auto* input_text =
            std::get_if<InputText>(&preprocessed_content)) {
      ABSL_ASSIGN_OR_RETURN(const auto* token_ids,
                            input_text->GetPreprocessedTextTensor());
      if (token_ids == nullptr) {
        return absl::InvalidArgumentError(
            "Token IDs is null in preprocessed_contents.");
      }
      LITERT_ASSIGN_OR_RETURN(auto ids_buffer_span,
                              ReferTensorBufferAsSpan<int>(*token_ids));
      combined_token_ids.insert(combined_token_ids.end(),
                                ids_buffer_span.begin(), ids_buffer_span.end());
    } else if (const auto* input_image =
                   std::get_if<InputImage>(&preprocessed_content)) {
      if (benchmark_info.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("vision_executor"));
      }
      ExecutorVisionData single_image_data;
      if (input_image->IsTensorBuffer()) {
        ABSL_ASSIGN_OR_RETURN(auto tensor_buffer,
                              input_image->GetPreprocessedImageTensor());
        ABSL_ASSIGN_OR_RETURN(auto vision_executor,
                              resource_manager_->AcquireVisionExecutor());
        ABSL_ASSIGN_OR_RETURN(single_image_data,
                              vision_executor->Encode(*tensor_buffer));
      } else if (input_image->IsTensorBufferMap()) {
        ABSL_ASSIGN_OR_RETURN(auto tensor_buffer_map,
                              input_image->GetPreprocessedImageTensorMap());
        ABSL_ASSIGN_OR_RETURN(auto vision_executor,
                              resource_manager_->AcquireVisionExecutor());
        ABSL_ASSIGN_OR_RETURN(single_image_data,
                              vision_executor->Encode(*tensor_buffer_map));
      } else {
        return absl::FailedPreconditionError(
            "Image tensor or tensor map is null in preprocessed_contents.");
      }
      if (benchmark_info.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("vision_executor"));
      }
      ABSL_ASSIGN_OR_RETURN(auto embeddings_ptr,
                            single_image_data.GetEmbeddingsPtr());
      ABSL_ASSIGN_OR_RETURN(const auto& dimensions,
                            TensorBufferDims(*embeddings_ptr));
      // The last two dimensions are [..., image_token_num, model_dimension].
      const int image_token_num = dimensions.at(dimensions.size() - 2);
      combined_token_ids.insert(combined_token_ids.end(), image_token_num,
                                ExecutorVisionData::kSpecialToken);
      all_image_data.push_back(std::move(single_image_data));
    } else if (const auto* input_image_end =
                   std::get_if<InputImageEnd>(&preprocessed_content)) {
      combined_token_ids.push_back(ExecutorVisionData::kEndToken);
    } else if (const auto* input_audio =
                   std::get_if<InputAudio>(&preprocessed_content)) {
      ABSL_ASSIGN_OR_RETURN(const auto* spectrogram_tensor,
                            input_audio->GetPreprocessedAudioTensor());
      if (benchmark_info.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("audio_executor"));
      }
      ABSL_ASSIGN_OR_RETURN(auto audio_executor,
                            resource_manager_->AcquireAudioExecutor());
      ABSL_ASSIGN_OR_RETURN(auto single_audio_data,
                            audio_executor->Encode(*spectrogram_tensor));
      if (benchmark_info.has_value()) {
        ABSL_RETURN_IF_ERROR(benchmark_info->TimeMarkDelta("audio_executor"));
      }
      const int num_audio_tokens = single_audio_data.GetValidTokens();
      if (num_audio_tokens > 0) {
        all_audio_data.push_back(std::move(single_audio_data));
        combined_token_ids.insert(combined_token_ids.end(), num_audio_tokens,
                                  ExecutorAudioData::kSpecialToken);
      }
    } else if (const auto* input_audio_end =
                   std::get_if<InputAudioEnd>(&preprocessed_content)) {
      // We allow audio end token even if the audio executor is not
      // available.
      auto audio_executor = resource_manager_->AcquireAudioExecutor();
      if (audio_executor.ok()) {
        // Flush any remaining buffered spectrogram frames from streaming
        // Encode() calls.
        auto flushed_audio_data = (*audio_executor)->Flush();
        if (flushed_audio_data.ok()) {
          const int flushed_tokens = flushed_audio_data->GetValidTokens();
          if (flushed_tokens > 0) {
            all_audio_data.push_back(std::move(*flushed_audio_data));
            combined_token_ids.insert(combined_token_ids.end(), flushed_tokens,
                                      ExecutorAudioData::kSpecialToken);
          }
        } else if (!absl::IsUnimplemented(flushed_audio_data.status())) {
          return flushed_audio_data.status();
        }
      }
      combined_token_ids.push_back(ExecutorAudioData::kEndToken);
    } else {
      return absl::InvalidArgumentError(
          "Unsupported input type in preprocessed_contents.");
    }
  }

  if (combined_token_ids.empty()) {
    return absl::InvalidArgumentError(
        "No token IDs found in preprocessed_contents.");
  }

  std::optional<ExecutorVisionData> combined_image_data = std::nullopt;
  if (!all_image_data.empty()) {
    ABSL_ASSIGN_OR_RETURN(combined_image_data,
                          CombineExecutorVisionData(all_image_data));
  }
  std::optional<ExecutorAudioData> combined_audio_data = std::nullopt;
  if (!all_audio_data.empty()) {
    ABSL_ASSIGN_OR_RETURN(combined_audio_data,
                          CombineExecutorAudioData(all_audio_data));
  }

  last_prefill_token_id_ = combined_token_ids.back();

  ABSL_ASSIGN_OR_RETURN(auto token_ids_buffer,
                        tokenizer_->TokenIdsToTensorBuffer(combined_token_ids));

  ExecutorInputs inputs(ExecutorTextData(std::move(token_ids_buffer)),
                        std::move(combined_image_data),
                        std::move(combined_audio_data));
  return inputs;
}

absl::StatusOr<std::unique_ptr<ThreadedExecutionManager>>
ThreadedExecutionManager::Create(
    Tokenizer* absl_nonnull tokenizer,
    ModelResources* absl_nullable model_resources,
    std::unique_ptr<LlmExecutor> absl_nonnull llm_executor,
    std::unique_ptr<VisionExecutorSettings> absl_nullable
    vision_executor_settings,
    std::unique_ptr<AudioExecutorSettings> absl_nullable
    audio_executor_settings,
    ::litert::Environment* absl_nullable litert_env,
    std::unique_ptr<AudioExecutor> absl_nullable audio_executor,
    std::shared_ptr<RuntimeDebugger> absl_nullable runtime_debugger) {
  ABSL_ASSIGN_OR_RETURN(
      auto resource_manager,
      ResourceManager::Create(model_resources, std::move(llm_executor),
                              std::move(vision_executor_settings),
                              std::move(audio_executor_settings), litert_env,
                              std::move(audio_executor)));
  return absl::WrapUnique(new ThreadedExecutionManager(
      tokenizer, std::move(resource_manager), litert_env,
      std::move(runtime_debugger)));
}

absl::Status ThreadedExecutionManager::WaitUntilDone(TaskId task_id,
                                                     absl::Duration timeout) {
  auto task_done = [this, task_id]() {
    session_and_task_lookup_mutex_.AssertReaderHeld();
    return task_lookup_.contains(task_id) &&
           IsTaskEndState(task_lookup_.at(task_id).task_state);
  };
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return session_and_task_lookup_mutex_.AwaitWithTimeout(
             absl::Condition(&task_done), timeout)
             ? absl::OkStatus()
             : absl::DeadlineExceededError(absl::StrCat(
                   "Task ", task_id, " did not complete within the timeout of ",
                   absl::FormatDuration(timeout), "."));
}

absl::Status ThreadedExecutionManager::WaitUntilSessionDone(
    SessionId session_id, absl::Duration timeout) {
  auto session_done = [this, session_id]() {
    session_and_task_lookup_mutex_.AssertReaderHeld();
    return session_lookup_.contains(session_id) &&
           session_lookup_.at(session_id)->active_tasks.empty();
  };
  absl::MutexLock lock(session_and_task_lookup_mutex_);
  return session_and_task_lookup_mutex_.AwaitWithTimeout(
             absl::Condition(&session_done), timeout)
             ? absl::OkStatus()
             : absl::DeadlineExceededError(
                   absl::StrCat("Session ", session_id,
                                " did not complete within the timeout of ",
                                absl::FormatDuration(timeout), "."));
}

absl::Status ThreadedExecutionManager::WaitUntilAllDone(
    absl::Duration timeout) {
  ABSL_RETURN_IF_ERROR(execution_thread_pool_->WaitUntilDone(timeout));
  ABSL_RETURN_IF_ERROR(callback_thread_pool_->WaitUntilDone(timeout));
  return absl::OkStatus();
}

absl::Status ThreadedExecutionManager::AddPrefillTask(
    SessionId session_id, TaskId task_id, std::vector<InputData> inputs,
    absl::flat_hash_set<TaskId> dep_tasks, PrefillBoundary boundary,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  switch (boundary) {
    case PrefillBoundary::kStartProjection:
    case PrefillBoundary::kContinueSession:
      break;
    default:
      return absl::InvalidArgumentError("Unknown prefill boundary.");
  }
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, session_id, boundary,
               inputs = std::move(inputs)]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    const bool stateless_projection_session =
        session_info->session_config.GetMemoryStrategy() ==
        SessionConfig::MemoryStrategy::kStatelessDeterministicProjection;
    const bool starts_stateless_projection =
        stateless_projection_session &&
        boundary == PrefillBoundary::kStartProjection;
    if (starts_stateless_projection) {
      absl::Status readiness_status =
          SetDeterministicProjectionReady(session_id, false);
      if (!readiness_status.ok()) {
        FinishTaskAndLogErrors(task_id, readiness_status,
                               std::move(callback));
        return;
      }
    } else if (stateless_projection_session) {
      absl::Status readiness_status =
          ValidateDeterministicProjectionReady(session_id);
      if (!readiness_status.ok()) {
        FinishTaskAndLogErrors(task_id, readiness_status,
                               std::move(callback));
        return;
      }
    }

    if (cancelled != nullptr && cancelled->load()) {
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    auto executor_inputs =
        ProcessAndCombineContents(inputs, session_info->benchmark_info);
    if (executor_inputs.ok() &&
        session_info->session_config.GetAudioEmbeddingsCallback() != nullptr) {
      auto audio_data_status = executor_inputs->GetAudioDataPtr();
      if (audio_data_status.ok() && *audio_data_status != nullptr) {
        (*session_info->session_config.GetAudioEmbeddingsCallback())(
            **audio_data_status);
      }
    }
    if (!executor_inputs.ok()) {
      if (executor_inputs.status().message() ==
              "No token IDs found in preprocessed_contents." &&
          session_info->session_config.AudioModalityEnabled()) {
        {
          auto audio_executor = resource_manager_->AcquireAudioExecutor();
          if (!audio_executor.ok()) {
            FinishTaskAndLogErrors(task_id, audio_executor.status(),
                                   std::move(callback));
            return;
          }
          auto audio_executor_properties =
              (*audio_executor)->GetAudioExecutorProperties();
          if (!audio_executor_properties.ok()) {
            audio_executor.value().reset();
            FinishTaskAndLogErrors(task_id, audio_executor_properties.status(),
                                   std::move(callback));
            return;
          }
          if (!audio_executor_properties->is_streaming_model) {
            audio_executor.value().reset();
            FinishTaskAndLogErrors(task_id, executor_inputs.status(),
                                   std::move(callback));
            return;
          }
        }
        ABSL_VLOG(1)
            << "Input audio chunk is smaller than the audio encoder input "
               "size. The input audio chunk is buffered and will be processed "
               "together with the next input audio chunk. Skipping prefill.";
        // We allow empty input for streaming audio use case, so we mark the
        // task as done.
        FinishTaskAndLogErrors(task_id, Responses(TaskState::kDone),
                               std::move(callback));
        return;
      }
      FinishTaskAndLogErrors(task_id, executor_inputs.status(),
                             std::move(callback));
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    // Context switching and all state mutation begin only after preprocessing
    // and user callbacks have completed.
    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      FinishTaskAndLogErrors(task_id, llm_executor.status(),
                             std::move(callback));
      return;
    }

    bool owns_projection_reset = false;
    auto finish_after_executor_unlock =
        [&](absl::StatusOr<Responses> result) mutable {
          llm_executor.value().reset();
          if (owns_projection_reset) {
            EndDeterministicProjectionReset(session_id, task_id);
            owns_projection_reset = false;
          }
          FinishTaskAndLogErrors(task_id, std::move(result),
                                 std::move(callback));
        };

    if (cancelled != nullptr && cancelled->load()) {
      finish_after_executor_unlock(Responses(TaskState::kCancelled));
      return;
    }

    if (starts_stateless_projection) {
      absl::Status owner_status =
          BeginDeterministicProjectionReset(session_id, task_id);
      if (!owner_status.ok()) {
        finish_after_executor_unlock(owner_status);
        return;
      }
      owns_projection_reset = true;
      absl::Status reset_status =
          llm_executor.value()->ResetForDeterministicProjection();
      if (!reset_status.ok()) {
        finish_after_executor_unlock(reset_status);
        return;
      }
      session_info->last_prefill_token_id = 0;
      last_prefill_token_id_ = 0;
      if (cancelled != nullptr && cancelled->load()) {
        finish_after_executor_unlock(Responses(TaskState::kCancelled));
        return;
      }
    }

#if defined(LITERT_LM_DEBUGGER_ENABLED)
    if (runtime_debugger_ != nullptr) {
      runtime_debugger_->SetActiveDebugSession(session_id);
    }
#else
    // Suppress -Wunused-variable for non-debugger builds.
    (void)session_id;
#endif

    auto responses =
        Tasks::Prefill(*llm_executor.value(), *executor_inputs,
                       /*wait_for_completion=*/true,
                       /*benchmark_info=*/session_info->benchmark_info);
    if (!responses.ok()) {
      finish_after_executor_unlock(responses.status());
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    } else {
      // Keep track of the last_prefill_token_id after prefill is done.
      auto processed_tokens = llm_executor.value()->GetProcessedTokens();
      if (!processed_tokens.ok()) {
        finish_after_executor_unlock(processed_tokens.status());
        return;
      }
      auto current_step = llm_executor.value()->GetCurrentStep();
      if (!current_step.ok()) {
        finish_after_executor_unlock(current_step.status());
        return;
      }
      if (current_step.value() <= 0) {
        finish_after_executor_unlock(absl::FailedPreconditionError(
            "Prefill completed without a positive current step."));
        return;
      }
      std::vector<int> last_step_tokens =
          processed_tokens.value()->GetTokenAtStep(current_step.value() - 1);
      if (last_step_tokens.size() != 1) {
        finish_after_executor_unlock(absl::FailedPreconditionError(
            "Prefill did not produce exactly one last-token candidate."));
        return;
      }
      session_info->last_prefill_token_id = last_step_tokens.front();
      last_prefill_token_id_ = session_info->last_prefill_token_id;
      if (starts_stateless_projection) {
        absl::Status readiness_status =
            SetDeterministicProjectionReady(session_id, true);
        if (!readiness_status.ok()) {
          finish_after_executor_unlock(readiness_status);
          return;
        }
      }
    }

    finish_after_executor_unlock(std::move(responses));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    /*potentially_mutating=*/true, cancelled,
                    std::move(callback));
}

absl::Status ThreadedExecutionManager::AddDecodeTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    RepetitionPenaltyConfig repetition_penalty_config,
    NoRepeatNgramConfig no_repeat_ngram_config,
    SuppressTokensConfig suppress_tokens_config,
    Constraint* absl_nullable constraint,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    int max_output_tokens, std::optional<int> thinking_token_budget,
    std::vector<int> thinking_start_token_ids,
    std::vector<int> thinking_end_token_ids,
    std::shared_ptr<ExactLiteRtDecodeCapture> exact_litert_decode_capture) {
#if defined(LITERT_LM_DEBUGGER_ENABLED)
  if (exact_litert_decode_capture != nullptr && runtime_debugger_ != nullptr) {
    return absl::UnimplementedError(
        "Exact decode does not support runtime debugger callbacks.");
  }
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

#if defined(LITERT_LM_DEBUGGER_ENABLED)
  if (runtime_debugger_ != nullptr) {
    callback = [this, session_id, cb = std::move(callback)](
                   absl::StatusOr<Responses> responses) mutable {
      if (responses.ok() && runtime_debugger_ != nullptr) {
        runtime_debugger_->ObserveTokens(session_id, *responses);
      }
      cb(std::move(responses));
    };
  }
#endif  // defined(LITERT_LM_DEBUGGER_ENABLED)

  auto task = [this, task_id, session_id,
               repetition_penalty_config = std::move(repetition_penalty_config),
               no_repeat_ngram_config = std::move(no_repeat_ngram_config),
               suppress_tokens_config = std::move(suppress_tokens_config),
               constraint, cancelled, max_output_tokens, thinking_token_budget,
               thinking_start_token_ids = std::move(thinking_start_token_ids),
               thinking_end_token_ids =
                   std::move(thinking_end_token_ids),
               exact_litert_decode_capture =
                   std::move(exact_litert_decode_capture)]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    if (session_info->session_config.GetMemoryStrategy() ==
        SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
      absl::Status readiness_status =
          ValidateDeterministicProjectionReady(session_id);
      if (!readiness_status.ok()) {
        FinishTaskAndLogErrors(task_id, readiness_status,
                               std::move(callback));
        return;
      }
    }

    if (cancelled != nullptr && cancelled->load()) {
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      FinishTaskAndLogErrors(task_id, llm_executor.status(),
                             std::move(callback));
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    auto num_output_candidates =
        session_info->session_config.GetNumOutputCandidates();
    session_info->stop_token_detector->ResetBatch(num_output_candidates);
    std::optional<Sampler*> optional_sampler = std::nullopt;
    std::optional<litert::TensorBuffer> decoded_ids_buffer = std::nullopt;
      if (session_info->sampler != nullptr) {
        optional_sampler = session_info->sampler.get();
        std::vector<int> decoded_ids(num_output_candidates,
                                     session_info->last_prefill_token_id);
        auto decoded_ids_buffer_or =
            CopyToTensorBuffer<int>(decoded_ids, {num_output_candidates, 1});
        if (!decoded_ids_buffer_or.HasValue()) {
          llm_executor.value().reset();
          FinishTaskAndLogErrors(
              task_id,
              absl::InternalError(decoded_ids_buffer_or.Error().Message()),
              std::move(callback));
          return;
        }
        decoded_ids_buffer = std::move(decoded_ids_buffer_or.Value());
      }

#if defined(LITERT_LM_DEBUGGER_ENABLED)
    if (runtime_debugger_ != nullptr) {
      runtime_debugger_->SetActiveDebugSession(session_id);
    }
#else
    // Suppress -Wunused-variable for non-debugger builds.
    (void)session_id;
#endif

    auto responses = Tasks::Decode(
        *llm_executor.value(), *tokenizer_, *session_info->stop_token_detector,
        num_output_candidates, session_info->benchmark_info, optional_sampler,
        std::move(repetition_penalty_config), std::move(no_repeat_ngram_config),
        std::move(suppress_tokens_config), constraint,
        std::move(decoded_ids_buffer), callback, cancelled.get(),
        max_output_tokens, thinking_token_budget, thinking_end_token_ids,
        thinking_start_token_ids, std::move(exact_litert_decode_capture));
    if (!responses.ok() && absl::IsCancelled(responses.status())) {
      responses = Responses(TaskState::kCancelled);
    }

    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    }

    llm_executor.value().reset();
    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    /*potentially_mutating=*/true, cancelled,
                    std::move(callback));
}

absl::Status ThreadedExecutionManager::AddCloneSessionTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    SessionId cloned_session_id,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, session_id, cloned_session_id]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    absl::StatusOr<Responses> result = Responses(TaskState::kDone);
    [&] {
      std::shared_ptr<const SessionInfo> original_session_info;
      {
        absl::MutexLock lock(session_and_task_lookup_mutex_);
        if (!session_lookup_.contains(session_id)) {
          result = absl::InvalidArgumentError(absl::StrCat(
              "Session ", session_id, " not found in session list."));
          return;
        }
        original_session_info = session_lookup_.at(session_id);
      }

      auto cloned_context_handler_or = resource_manager_->CloneContextHandler(
          original_session_info->context_handler);
      if (!cloned_context_handler_or.ok()) {
        result = cloned_context_handler_or.status();
        return;
      }

      std::unique_ptr<Sampler> cloned_sampler;
      if (original_session_info->sampler != nullptr) {
        auto sampler = CreateSampler(
            original_session_info->session_config.GetSamplerBackend(),
            original_session_info->session_config.GetNumOutputCandidates(),
            original_session_info->session_config.GetSamplerParams());
        if (!sampler.ok()) {
          result = sampler.status();
          return;
        }
        cloned_sampler = std::move(*sampler);
      }

      auto cloned_stop_token_detector = std::make_unique<StopTokenDetector>(1);
      for (const auto& stop_token_sequence :
           original_session_info->session_config.GetStopTokenIds()) {
        auto status = cloned_stop_token_detector->AddStopTokenSequence(
            stop_token_sequence);
        if (!status.ok()) {
          result = status;
          return;
        }
      }

      {
        absl::MutexLock lock(session_and_task_lookup_mutex_);
        if (!session_lookup_.contains(cloned_session_id)) {
          result = absl::InvalidArgumentError(
              absl::StrCat("Cloned session ", cloned_session_id,
                           " not found in session list."));
          return;
        }
        session_lookup_.at(cloned_session_id)->session_config =
            original_session_info->session_config;
        session_lookup_.at(cloned_session_id)->context_handler =
            std::move(cloned_context_handler_or.value());
        session_lookup_.at(cloned_session_id)->sampler =
            std::move(cloned_sampler);
        session_lookup_.at(cloned_session_id)->last_prefill_token_id =
            original_session_info->last_prefill_token_id;
        session_lookup_.at(cloned_session_id)->stop_token_detector =
            std::move(cloned_stop_token_detector);
        session_lookup_.at(cloned_session_id)->benchmark_info =
            original_session_info->benchmark_info;
        session_lookup_.at(cloned_session_id)
            ->deterministic_projection_ready =
            original_session_info->deterministic_projection_ready;
        if (original_session_info->handoff_poisoned) {
          PoisonSessionHandoff(
              *session_lookup_.at(cloned_session_id),
              absl::StrCat("Cloned from a handoff-poisoned session: ",
                           original_session_info->handoff_poison_reason));
        }
      }
    }();

    if (cancelled != nullptr && cancelled->load()) {
      result = Responses(TaskState::kCancelled);
    }

    FinishTaskAndLogErrors(task_id, result, std::move(callback));
    return;
  };

  return CreateTask(cloned_session_id, task_id, std::move(task),
                    std::move(dep_tasks), /*potentially_mutating=*/true,
                    cancelled, std::move(callback));
}

absl::Status ThreadedExecutionManager::AddTextScoringTask(
    SessionId session_id, TaskId task_id, absl::flat_hash_set<TaskId> dep_tasks,
    const std::vector<absl::string_view>& target_text, bool store_token_lengths,
    std::shared_ptr<std::atomic<bool>> absl_nonnull cancelled,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (callback == nullptr) {
    callback = [](absl::StatusOr<Responses> responses) {};
  }

  auto task = [this, task_id, session_id, target_text,
               store_token_lengths]() mutable -> void {
    auto task_info = StartTask(task_id);
    if (!task_info.ok()) {
      FinishTaskAndLogErrors(task_id, task_info.status(),
                             [](absl::StatusOr<Responses> responses) {});
      return;
    }
    auto [session_info, cancelled, callback] = std::move(task_info.value());
    // If the session info is nullptr, it means the task is cancelled before it
    // is started.
    if (session_info == nullptr) {
      return;
    }

    if (session_info->session_config.GetMemoryStrategy() ==
        SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
      absl::Status readiness_status =
          ValidateDeterministicProjectionReady(session_id);
      if (!readiness_status.ok()) {
        FinishTaskAndLogErrors(task_id, readiness_status,
                               std::move(callback));
        return;
      }
    }

    if (cancelled != nullptr && cancelled->load()) {
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    auto llm_executor = resource_manager_->AcquireExecutorWithContextHandler(
        session_info->context_handler);
    if (!llm_executor.ok()) {
      FinishTaskAndLogErrors(task_id, llm_executor.status(),
                             std::move(callback));
      return;
    }

    if (cancelled != nullptr && cancelled->load()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(task_id, Responses(TaskState::kCancelled),
                             std::move(callback));
      return;
    }

    const int num_output_candidates =
        session_info->session_config.GetNumOutputCandidates();
    std::vector<int> decoded_ids(num_output_candidates,
                                 session_info->last_prefill_token_id);
    auto decoded_ids_buffer =
        CopyToTensorBuffer<int>(decoded_ids, {num_output_candidates, 1});
    if (!decoded_ids_buffer.HasValue()) {
      llm_executor.value().reset();
      FinishTaskAndLogErrors(
          task_id, absl::InternalError(decoded_ids_buffer.Error().Message()),
          std::move(callback));
      return;
    }

    // TODO(b/435040163): Handle the temperature. Should it be calculated from
    // the sampler or the sampler parameters? For now, hardcode it to 1.0f for
    // testing.
    auto temperature = 1.0f;
    auto responses = Tasks::Score(
        *llm_executor.value(), *tokenizer_, target_text, temperature,
        std::move(decoded_ids_buffer.Value()), store_token_lengths);

    if (cancelled != nullptr && cancelled->load()) {
      responses = Responses(TaskState::kCancelled);
    }

    llm_executor.value().reset();
    FinishTaskAndLogErrors(task_id, std::move(responses), std::move(callback));
    return;
  };

  return CreateTask(session_id, task_id, std::move(task), std::move(dep_tasks),
                    /*potentially_mutating=*/true, cancelled,
                    std::move(callback));
}

absl::StatusOr<int> ThreadedExecutionManager::GetCurrentStep(
    const SessionInfo& session_info) {
  ABSL_ASSIGN_OR_RETURN(auto llm_executor,
                        resource_manager_->AcquireExecutorWithContextHandler(
                            session_info.context_handler));
  return llm_executor->GetCurrentStep();
}

absl::Status ThreadedExecutionManager::SetCurrentStep(
    const SessionInfo& session_info, int target_step) {
  if (session_info.session_config.GetMemoryStrategy() ==
      SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
    return absl::FailedPreconditionError(
        "Step-only rewind is not supported for deterministic projection "
        "sessions.");
  }
  ABSL_ASSIGN_OR_RETURN(auto llm_executor,
                        resource_manager_->AcquireExecutorWithContextHandler(
                            session_info.context_handler));
  ABSL_ASSIGN_OR_RETURN(int current_step, llm_executor->GetCurrentStep());
  if (target_step > current_step) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Target step is greater than the current step: ", current_step));
  }
  return llm_executor->SetCurrentStep(target_step);
}

absl::StatusOr<AudioExecutorProperties>
ThreadedExecutionManager::GetAudioExecutorProperties() const {
  return resource_manager_->GetAudioExecutorProperties();
}

absl::StatusOr<VisionExecutorProperties>
ThreadedExecutionManager::GetVisionExecutorProperties() const {
  return resource_manager_->GetVisionExecutorProperties();
}

}  // namespace litert::lm
