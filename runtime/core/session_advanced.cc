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

#include "runtime/core/session_advanced.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/container/flat_hash_set.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/memory/memory.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/core/session_handoff_codec.h"
#include "runtime/core/session_utils.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/framework/resource_management/execution_manager.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/proto/sampler_params.pb.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "support/tokenizer/tokenizer.h"

#if defined(LITERT_LM_DEBUGGER_ENABLED)
#include "runtime/util/runtime_debugger.h"
#endif

namespace litert::lm {
namespace {

using TaskController = SessionInterface::TaskController;

constexpr std::array<char, 8> kDpmTokenBytesMagic = {
    'D', 'P', 'M', 'T', 'O', 'K', '0', '1'};
constexpr uint32_t kDpmTokenEncodingVersion = 1;
constexpr uint32_t kMaximumWitnessTokenIds = 1'000'000;
constexpr size_t kHandoffDigestChunkSize = 2 * 1024 * 1024;

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

std::array<char, 4> EncodeU32(uint32_t value) {
  return {
      static_cast<char>((value >> 24) & 0xff),
      static_cast<char>((value >> 16) & 0xff),
      static_cast<char>((value >> 8) & 0xff),
      static_cast<char>(value & 0xff),
  };
}

// Hashes exactly the bytes successfully accepted by the downstream sink. A
// null downstream is the import path's deliberately allocation-free,
// digest-only re-export target.
class HandoffWitnessByteSink final : public ByteSink {
 public:
  explicit HandoffWitnessByteSink(ByteSink* downstream)
      : downstream_(downstream) {}

  absl::Status Append(absl::string_view bytes) override {
    if (finished_) {
      return absl::FailedPreconditionError(
          "Session handoff witness sink was already finalized.");
    }
    if (bytes.size() >
        std::numeric_limits<uint64_t>::max() - emitted_size_) {
      return absl::ResourceExhaustedError(
          "Session handoff witness byte count overflows uint64.");
    }
    if (bytes.size() >
        kMaximumSessionHandoffEnvelopeBytes - emitted_size_) {
      return absl::ResourceExhaustedError(
          "Session handoff envelope exceeds the supported transport limit.");
    }
    if (downstream_ != nullptr) {
      ABSL_RETURN_IF_ERROR(downstream_->Append(bytes));
    }
    hasher_.Update(bytes);
    emitted_size_ += bytes.size();
    return absl::OkStatus();
  }

  absl::StatusOr<Hash256> Finish() {
    if (finished_) {
      return absl::FailedPreconditionError(
          "Session handoff witness sink was finalized more than once.");
    }
    finished_ = true;
    const Hash256 digest = hasher_.Finalize();
    if (IsZeroHash(digest) || emitted_size_ == 0) {
      return absl::InternalError(
          "Session handoff witness sink produced empty evidence.");
    }
    return digest;
  }

  uint64_t emitted_size() const { return emitted_size_; }

 private:
  ByteSink* downstream_;
  Sha256Hasher hasher_;
  uint64_t emitted_size_ = 0;
  bool finished_ = false;
};

absl::StatusOr<Hash256> HashByteSource(const ByteSource& source) {
  const uint64_t source_size = source.Size();
  if (source_size == 0) {
    return absl::DataLossError(
        "Session handoff byte source is empty.");
  }
  if (source_size > kMaximumSessionHandoffEnvelopeBytes) {
    return absl::ResourceExhaustedError(
        "Session handoff byte source exceeds the supported transport limit.");
  }
  Sha256Hasher hasher;
  std::string buffer(
      static_cast<size_t>(
          std::min<uint64_t>(source_size, kHandoffDigestChunkSize)),
      '\0');
  uint64_t offset = 0;
  while (offset < source_size) {
    const size_t count = static_cast<size_t>(
        std::min<uint64_t>(buffer.size(), source_size - offset));
    ABSL_RETURN_IF_ERROR(source.ReadAt(
        offset, absl::MakeSpan(buffer).subspan(0, count)));
    hasher.Update(absl::string_view(buffer.data(), count));
    offset += count;
  }
  if (source.Size() != source_size) {
    return absl::AbortedError(
        "Session handoff byte source changed while computing its digest.");
  }
  const Hash256 digest = hasher.Finalize();
  if (IsZeroHash(digest)) {
    return absl::DataLossError(
        "Session handoff byte source has no canonical digestable envelope.");
  }
  return digest;
}

absl::StatusOr<Hash256> ComputeProcessedHistoryTokenBytesHash(
    const ExecutorSessionSnapshot& snapshot) {
  const auto& tokens = snapshot.processed_tokens;
  if (tokens.processed_token_ids.size() != 1) {
    return absl::UnimplementedError(
        "Session continuation witness requires exactly one processed-token "
        "history.");
  }
  if (!tokens.pending_token_ids.empty() &&
      tokens.pending_token_ids.size() != 1) {
    return absl::DataLossError(
        "Session continuation witness has inconsistent pending-token "
        "history.");
  }
  const size_t processed_count = tokens.processed_token_ids.front().size();
  const size_t pending_count = tokens.pending_token_ids.empty() ? 0 : 1;
  if (processed_count > kMaximumWitnessTokenIds - pending_count) {
    return absl::ResourceExhaustedError(
        "Session continuation witness exceeds the DPMTOK01 token limit.");
  }
  const size_t total_count = processed_count + pending_count;
  if (snapshot.current_step < 0 ||
      static_cast<size_t>(snapshot.current_step) != total_count) {
    return absl::DataLossError(
        "Session continuation witness step differs from its complete token "
        "history.");
  }
  Sha256Hasher hasher;
  hasher.Update(absl::string_view(kDpmTokenBytesMagic.data(),
                                 kDpmTokenBytesMagic.size()));
  const std::array<char, 4> version = EncodeU32(kDpmTokenEncodingVersion);
  hasher.Update(absl::string_view(version.data(), version.size()));
  const std::array<char, 4> count =
      EncodeU32(static_cast<uint32_t>(total_count));
  hasher.Update(absl::string_view(count.data(), count.size()));
  // The live export boundary and transactional executor import separately
  // validate every ID against the loaded logits vocabulary. This layer owns
  // only the product-canonical nonnegative-int32 DPMTOK01 representation; it
  // must not substitute a tokenizer vocabulary for the loaded executor shape.
  const auto append_token = [&](int token_id) -> absl::Status {
    if (token_id < 0 ||
        static_cast<uint64_t>(token_id) >
            static_cast<uint64_t>((std::numeric_limits<int32_t>::max)())) {
      return absl::DataLossError(
          "Session continuation witness contains a noncanonical token ID.");
    }
    const std::array<char, 4> encoded =
        EncodeU32(static_cast<uint32_t>(token_id));
    hasher.Update(absl::string_view(encoded.data(), encoded.size()));
    return absl::OkStatus();
  };
  for (int token_id : tokens.processed_token_ids.front()) {
    ABSL_RETURN_IF_ERROR(append_token(token_id));
  }
  if (!tokens.pending_token_ids.empty()) {
    ABSL_RETURN_IF_ERROR(append_token(tokens.pending_token_ids.front()));
  }
  const Hash256 history_hash = hasher.Finalize();
  if (IsZeroHash(history_hash)) {
    return absl::InternalError(
        "Canonical DPMTOK01 history produced a zero digest.");
  }
  return history_hash;
}

absl::StatusOr<SessionContinuationStateWitness> BuildContinuationWitness(
    const SessionHandoffSnapshot& snapshot,
    const SessionHandoffIdentity& identity, absl::string_view key_id,
    const Hash256& processed_history_token_bytes_hash,
    const Hash256& envelope_hash, uint64_t envelope_size) {
  if (snapshot.executor.current_step < 0 ||
      static_cast<uint64_t>(snapshot.executor.current_step) >
          static_cast<uint64_t>(
              (std::numeric_limits<int32_t>::max)())) {
    return absl::DataLossError(
        "Session continuation witness step is outside int32.");
  }
  SessionContinuationStateWitness witness;
  witness.session_identity = identity;
  witness.phase = snapshot.phase;
  witness.current_step = static_cast<int32_t>(snapshot.executor.current_step);
  witness.ran_decode = snapshot.executor.ran_decode;
  witness.processed_history_token_bytes_hash =
      processed_history_token_bytes_hash;
  witness.envelope_hash = envelope_hash;
  witness.envelope_size = envelope_size;
  witness.key_id.assign(key_id.data(), key_id.size());
  ABSL_ASSIGN_OR_RETURN(
      witness.witness_id,
      ComputeSessionContinuationStateWitnessId(witness));
  ABSL_RETURN_IF_ERROR(
      ValidateSessionContinuationStateWitness(witness));
  return witness;
}

absl::Status ValidateSessionHandoffConfig(const SessionConfig& config) {
  if (config.UseExternalSampler()) {
    return absl::UnimplementedError(
        "Session handoff does not support external sampler state.");
  }
  if (config.GetSamplerBackend() != Backend::CPU) {
    return absl::UnimplementedError(
        "Session handoff currently supports only the CPU sampler backend.");
  }
  if (config.GetNumOutputCandidates() != 1) {
    return absl::UnimplementedError(
        "Session handoff currently supports exactly one output candidate.");
  }
  if (config.AudioModalityEnabled() || config.VisionModalityEnabled() ||
      config.GetAudioEmbeddingsCallback() != nullptr) {
    return absl::UnimplementedError(
        "Session handoff currently supports text-only sessions.");
  }
  if (config.GetScopedLoraFile() != nullptr ||
      config.GetAudioScopedLoraFile() != nullptr) {
    return absl::UnimplementedError(
        "Session handoff does not support LoRA state.");
  }
  return absl::OkStatus();
}

absl::Status ValidateExactDecodeConfig(
    const SessionConfig& config,
    const std::optional<BenchmarkInfo>& benchmark_info,
    int max_output_tokens) {
  if (config.UseExternalSampler()) {
    return absl::UnimplementedError(
        "Exact decode does not support an external sampler.");
  }
  if (config.GetSamplerBackend() != Backend::CPU ||
      config.GetSamplerParams().type() !=
          proto::SamplerParameters::GREEDY ||
      config.GetSamplerParams().backend() !=
          proto::SamplerParameters::CPU) {
    return absl::UnimplementedError(
        "Exact decode requires the explicit CPU GREEDY min-index sampler.");
  }
  if (config.GetNumOutputCandidates() != 1) {
    return absl::UnimplementedError(
        "Exact decode requires exactly one output candidate.");
  }
  if (config.GetSuppressTokensConfig().enabled()) {
    return absl::UnimplementedError(
        "Exact decode does not support session-level token suppression.");
  }
  if (config.GetApplyPromptTemplateInSession()) {
    return absl::UnimplementedError(
        "Exact decode requires the worker to prefill an already-canonical "
        "prompt without hidden prompt-template tail tokens.");
  }
  if (config.AudioModalityEnabled() || config.VisionModalityEnabled() ||
      config.GetAudioEmbeddingsCallback() != nullptr) {
    return absl::UnimplementedError(
        "Exact decode supports only ordinary text sessions.");
  }
  if (config.GetScopedLoraFile() != nullptr ||
      config.GetAudioScopedLoraFile() != nullptr) {
    return absl::UnimplementedError(
        "Exact decode does not support LoRA state.");
  }
  if (benchmark_info.has_value()) {
    return absl::UnimplementedError(
        "Exact decode does not support benchmark-controlled execution.");
  }
  if (max_output_tokens <= 0 ||
      max_output_tokens >
          static_cast<int>(kMaximumExactLiteRtDecodeFrames)) {
    return absl::InvalidArgumentError(
        "Exact decode max_output_tokens is outside the admitted range.");
  }
  return absl::OkStatus();
}

}  // namespace

// static
absl::StatusOr<std::unique_ptr<SessionAdvanced>> SessionAdvanced::Create(
    std::weak_ptr<ExecutionManager> execution_manager,
    support::Tokenizer* absl_nonnull tokenizer,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    std::atomic<int>* living_sessions_count) {
  return CreateInternal(execution_manager, tokenizer, session_config,
                        std::move(benchmark_info), living_sessions_count,
                        /*session_handoff_identity=*/std::nullopt,
                        /*exact_litert_logits_frame_contract=*/std::nullopt);
}

// static
absl::StatusOr<std::unique_ptr<SessionAdvanced>>
SessionAdvanced::CreateWithEngineOwnedIdentity(
    std::weak_ptr<ExecutionManager> execution_manager,
    support::Tokenizer* absl_nonnull tokenizer,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    std::atomic<int>* living_sessions_count,
    std::optional<SessionHandoffIdentity> session_handoff_identity,
    std::optional<ExactLiteRtLogitsFrameContract>
        exact_litert_logits_frame_contract) {
  return CreateInternal(execution_manager, tokenizer, session_config,
                        std::move(benchmark_info), living_sessions_count,
                        std::move(session_handoff_identity),
                        std::move(exact_litert_logits_frame_contract));
}

// static
absl::StatusOr<std::unique_ptr<SessionAdvanced>>
SessionAdvanced::CreateInternal(
    std::weak_ptr<ExecutionManager> execution_manager,
    support::Tokenizer* absl_nonnull tokenizer,
    const SessionConfig& session_config,
    std::optional<BenchmarkInfo> benchmark_info,
    std::atomic<int>* living_sessions_count,
    std::optional<SessionHandoffIdentity> session_handoff_identity,
    std::optional<ExactLiteRtLogitsFrameContract>
        exact_litert_logits_frame_contract) {
  auto execution_manager_lock = execution_manager.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ABSL_ASSIGN_OR_RETURN(auto session_id,
                        execution_manager_lock->RegisterNewSession(
                            session_config, benchmark_info));
  ABSL_ASSIGN_OR_RETURN(auto session_info_,
                        execution_manager_lock->GetSessionInfo(session_id));
  return absl::WrapUnique(new SessionAdvanced(
      session_id, execution_manager, tokenizer, session_info_,
      /*session_state=*/SessionState::kFresh,
      /*last_task_ids=*/{}, living_sessions_count,
      std::move(session_handoff_identity),
      std::move(exact_litert_logits_frame_contract)));
}

absl::Status SessionAdvanced::RunPrefill(
    const std::vector<InputData>& contents) {
  absl::Status status = absl::OkStatus();
  ABSL_ASSIGN_OR_RETURN(
      auto task_controller,
      RunPrefillAsync(contents, [&status](absl::StatusOr<Responses> responses) {
        status = responses.status();
      }));
  ABSL_RETURN_IF_ERROR(task_controller->WaitUntilDone(Engine::kDefaultTimeout));
  return status;
}

absl::StatusOr<std::unique_ptr<TaskController>>
SessionAdvanced::RunPrefillAsync(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  if (contents.empty()) {
    return absl::InvalidArgumentError("Input is empty.");
  }
  absl::MutexLock lock(mutex_);
  auto cancelled = std::make_shared<std::atomic<bool>>(false);

  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  std::vector<InputData> preprocessed_contents;
  if (session_info_->benchmark_info.has_value() &&
      session_info_->benchmark_info->GetBenchmarkParams().num_prefill_tokens() >
          0) {
    ABSL_ASSIGN_OR_RETURN(
        preprocessed_contents,
        PreprocessContents(contents, session_info_->session_config, *tokenizer_,
                           session_info_->benchmark_info));
  } else {
    const bool is_first_turn =
        session_state_ == SessionState::kFresh ||
        session_info_->session_config.GetMemoryStrategy() ==
            SessionConfig::MemoryStrategy::
                kStatelessDeterministicProjection;
    ContentType content_type;
    if (session_info_->session_config.GetApplyPromptTemplateInSession()) {
      content_type = (is_first_turn || session_state_ == SessionState::kDecoded)
                         ? ContentType::kFirst
                         : ContentType::kMiddle;
    } else {
      content_type = ContentType::kNA;
    }
    ABSL_ASSIGN_OR_RETURN(std::vector<InputData> templated_contents,
                          ApplyPromptTemplates(contents, content_type,
                                               session_info_->session_config,
                                               *tokenizer_, is_first_turn));
    ABSL_ASSIGN_OR_RETURN(
        preprocessed_contents,
        PreprocessContents(templated_contents, session_info_->session_config,
                           *tokenizer_, session_info_->benchmark_info));
  }
  ABSL_ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());
  ABSL_RETURN_IF_ERROR(execution_manager_lock->AddPrefillTask(
      session_id_, task_id, std::move(preprocessed_contents), last_task_ids_,
      PrefillBoundary::kStartProjection, cancelled, std::move(callback)));
  session_state_ = SessionState::kPrefilled;
  last_task_ids_ = {task_id};

  return std::make_unique<AdvancedTaskController>(task_id, cancelled,
                                                  execution_manager_);
}

absl::StatusOr<std::unique_ptr<TaskController>>
SessionAdvanced::PrefillPreprocessedContents(
    std::vector<InputData> preprocessed_contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  absl::MutexLock lock(mutex_);
  auto cancelled = std::make_shared<std::atomic<bool>>(false);
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  ABSL_ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());
  ABSL_RETURN_IF_ERROR(execution_manager_lock->AddPrefillTask(
      session_id_, task_id, std::move(preprocessed_contents), last_task_ids_,
      PrefillBoundary::kStartProjection, cancelled, std::move(callback)));
  session_state_ = SessionState::kPrefilled;
  last_task_ids_ = {task_id};

  return std::make_unique<AdvancedTaskController>(task_id, cancelled,
                                                  execution_manager_);
}

absl::StatusOr<Responses> SessionAdvanced::RunDecode() {
  return RunDecode(DecodeConfig::CreateDefault());
}

absl::StatusOr<Responses> SessionAdvanced::RunDecode(
    const DecodeConfig& decode_config) {
  return RunDecodeBlockingInternal(decode_config, nullptr);
}

absl::StatusOr<Responses> SessionAdvanced::RunDecodeBlockingInternal(
    const DecodeConfig& decode_config,
    std::shared_ptr<ExactLiteRtDecodeCapture> exact_litert_decode_capture) {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  absl::StatusOr<Responses> collected_responses;
  int num_candidates = session_info_->session_config.GetNumOutputCandidates();
  collected_responses =
      Responses(TaskState::kCreated,
                /*response_texts=*/std::vector<std::string>(num_candidates),
                /*scores=*/std::vector<float>(num_candidates),
                /*token_lengths=*/{},
                /*token_ids=*/std::vector<std::vector<int>>(num_candidates));
  int num_decode_tokens = 0;
  auto decode_sync_callback = [&collected_responses, &num_decode_tokens](
                                  absl::StatusOr<Responses> responses) {
    if (!collected_responses.ok()) return;
    if (!responses.ok()) {
      collected_responses = responses.status();
      return;
    }
    collected_responses->SetTaskState(responses->GetTaskState());
    // If the task is not completed and there is no text or score, we can
    // return early.
    if (!IsTaskEndState(responses->GetTaskState()) &&
        responses->GetTexts().empty() && responses->GetScores().empty()) {
      return;
    }
    // Accumulating the scores if it is provided.
    if (collected_responses->GetMutableScores().size() ==
        responses->GetScores().size()) {
      for (int i = 0; i < responses->GetScores().size(); ++i) {
        collected_responses->GetMutableScores()[i] += responses->GetScores()[i];
      }
    }
    // Accumulating the texts.
    if (collected_responses->GetMutableTexts().size() ==
        responses->GetTexts().size()) {
      num_decode_tokens += 1;
      for (int i = 0; i < responses->GetTexts().size(); ++i) {
        collected_responses->GetMutableTexts()[i] += responses->GetTexts()[i];
      }
    } else if (!responses->GetTexts().empty()) {
      collected_responses = absl::InternalError(
          absl::StrCat("Decode responses size mismatch: ",
                       collected_responses->GetTexts().size(), " vs ",
                       responses->GetTexts().size()));
    }
    // Accumulating the token IDs.
    if (collected_responses->GetMutableTokenIds().size() ==
        responses->GetTokenIds().size()) {
      for (int i = 0; i < responses->GetTokenIds().size(); ++i) {
        collected_responses->GetMutableTokenIds()[i].insert(
            collected_responses->GetMutableTokenIds()[i].end(),
            responses->GetTokenIds()[i].begin(),
            responses->GetTokenIds()[i].end());
      }
    }
    // Normalizing the scores by the number of decode tokens if the task is
    // completed.
    if (IsTaskEndState(responses->GetTaskState())) {
      for (int i = 0; i < responses->GetScores().size(); ++i) {
        collected_responses->GetMutableScores()[i] /=
            std::max(1, num_decode_tokens);
      }
    }
  };

  ABSL_ASSIGN_OR_RETURN(
      auto task_controller,
      RunDecodeAsyncInternal(std::move(decode_sync_callback), decode_config,
                             std::move(exact_litert_decode_capture)));
  ABSL_RETURN_IF_ERROR(task_controller->WaitUntilDone(Engine::kDefaultTimeout));
  return collected_responses;
}

absl::StatusOr<ExactLiteRtDecodeResult> SessionAdvanced::RunExactDecode(
    int max_output_tokens) {
  ABSL_RETURN_IF_ERROR(ValidateExactDecodeConfig(
      session_info_->session_config, session_info_->benchmark_info,
      max_output_tokens));
  if (!exact_litert_logits_frame_contract_.has_value()) {
    return absl::FailedPreconditionError(
        "The loaded Engine did not provide an exact logits frame contract.");
  }
  {
    absl::MutexLock lock(mutex_);
    if (session_state_ != SessionState::kPrefilled) {
      return absl::FailedPreconditionError(
          "Exact decode requires a successfully prefilled session.");
    }
  }

  // The executor-owned history is checked on both sides of decode so captured
  // IDs cannot diverge from the continuation state actually committed by the
  // model. Unlike Responses, this history includes a terminating stop/EOS ID.
  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::vector<int>> history_before,
      GetExactProcessedTokenHistory());
  if (history_before.size() != 1) {
    return absl::DataLossError(
        "Exact decode pre-state does not contain exactly one token history.");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<ExactLiteRtDecodeCapture> capture,
      ExactLiteRtDecodeCapture::Create(*exact_litert_logits_frame_contract_,
                                       max_output_tokens));
  DecodeConfig decode_config = DecodeConfig::CreateDefault();
  decode_config.SetMaxOutputTokens(max_output_tokens);
  ABSL_ASSIGN_OR_RETURN(
      Responses responses,
      RunDecodeBlockingInternal(decode_config, capture));
  if (responses.GetTaskState() == TaskState::kCancelled ||
      responses.GetTaskState() == TaskState::kDependentTaskCancelled) {
    return absl::CancelledError(
        "Exact decode was cancelled; partial evidence is not admissible.");
  }
  if (responses.GetTaskState() != TaskState::kDone &&
      responses.GetTaskState() != TaskState::kMaxNumTokensReached) {
    return absl::DataLossError(
        "Exact decode did not end in an admissible completed task state.");
  }
  ABSL_ASSIGN_OR_RETURN(ExactLiteRtDecodeEvidence evidence,
                        capture->Finish());

  ABSL_ASSIGN_OR_RETURN(
      std::vector<std::vector<int>> history_after,
      GetExactProcessedTokenHistory());
  if (history_after.size() != 1 ||
      history_after.front().size() < history_before.front().size()) {
    return absl::DataLossError(
        "Exact decode post-state has an inconsistent token history.");
  }
  if (!std::equal(history_before.front().begin(),
                  history_before.front().end(),
                  history_after.front().begin())) {
    return absl::DataLossError(
        "Exact decode modified the pre-existing token-history prefix.");
  }
  const size_t generated_count =
      history_after.front().size() - history_before.front().size();
  if (generated_count != evidence.sampled_token_ids.size() ||
      evidence.sampled_token_ids.size() != evidence.logits_frames.size()) {
    return absl::DataLossError(
        "Exact decode token-history, sampled-token, and logits-frame counts "
        "differ.");
  }
  for (size_t index = 0; index < generated_count; ++index) {
    const int committed_token =
        history_after.front()[history_before.front().size() + index];
    if (committed_token != evidence.sampled_token_ids[index]) {
      return absl::DataLossError(
          "Exact decode captured token IDs differ from committed executor "
          "history.");
    }
  }
  return ExactLiteRtDecodeResult(std::move(responses), std::move(evidence));
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionAdvanced::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return RunDecodeAsync(std::move(callback), DecodeConfig::CreateDefault());
}

absl::StatusOr<std::unique_ptr<TaskController>> SessionAdvanced::RunDecodeAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  return RunDecodeAsyncInternal(std::move(callback), decode_config, nullptr);
}

absl::StatusOr<std::unique_ptr<TaskController>>
SessionAdvanced::RunDecodeAsyncInternal(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config,
    std::shared_ptr<ExactLiteRtDecodeCapture> exact_litert_decode_capture) {
  absl::MutexLock lock(mutex_);
  if (session_state_ != SessionState::kPrefilled) {
    return absl::InternalError("Session is not prefilled yet.");
  }

  auto cancelled = std::make_shared<std::atomic<bool>>(false);

  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  // We need to do a last prefill before initializing the decode, to make sure
  // the prompt is correctly set up for decode.
  if (session_info_->session_config.GetApplyPromptTemplateInSession()) {
    std::vector<InputData> contents;
    contents.emplace_back(InputText(""));
    ABSL_ASSIGN_OR_RETURN(
        std::vector<InputData> templated_contents,
        ApplyPromptTemplates(contents, ContentType::kLast,
                             session_info_->session_config, *tokenizer_,
                             /*is_first_turn=*/false));
    if (!templated_contents.empty()) {
      ABSL_ASSIGN_OR_RETURN(
          std::vector<InputData> preprocessed_contents,
          PreprocessContents(templated_contents, session_info_->session_config,
                             *tokenizer_, session_info_->benchmark_info));
      auto noop_callback = [](absl::StatusOr<Responses> responses) {};
      ABSL_ASSIGN_OR_RETURN(auto task_id,
                            execution_manager_lock->GetNewTaskId());
      ABSL_RETURN_IF_ERROR(execution_manager_lock->AddPrefillTask(
          session_id_, task_id, std::move(preprocessed_contents),
          last_task_ids_, PrefillBoundary::kContinueSession, cancelled,
          std::move(noop_callback)));
      last_task_ids_ = {task_id};
    }
  }
  session_state_ = SessionState::kDecoded;

  ABSL_ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());

  ABSL_RETURN_IF_ERROR(execution_manager_lock->AddDecodeTask(
      session_id_, task_id, last_task_ids_,
      decode_config.GetRepetitionPenaltyConfig(),
      decode_config.GetNoRepeatNgramConfig(),
      decode_config.GetSuppressTokensConfig().value_or(
          session_info_->session_config.GetSuppressTokensConfig()),
      decode_config.GetConstraint(), cancelled, std::move(callback),
      decode_config.GetMaxOutputTokens().value_or(
          session_info_->session_config.GetMaxOutputTokens()),
      decode_config.GetThinkingTokenBudget(),
      decode_config.GetThinkingStartTokenIds(),
      decode_config.GetThinkingEndTokenIds(),
      std::move(exact_litert_decode_capture)));

  last_task_ids_ = {task_id};

  return std::make_unique<AdvancedTaskController>(task_id, cancelled,
                                                  execution_manager_);
}

absl::StatusOr<Responses> SessionAdvanced::RunTextScoring(
    const std::vector<absl::string_view>& target_text,
    bool store_token_lengths) {
  if (target_text.size() != 1) {
    // Batch scoring is not supported yet.
    return absl::InvalidArgumentError("Target text size should be 1.");
  }
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  absl::StatusOr<Responses> collected_responses;
  auto scoring_sync_callback =
      [&collected_responses](absl::StatusOr<Responses> responses) {
        collected_responses = std::move(responses);
      };

  ABSL_ASSIGN_OR_RETURN(
      auto task_controller,
      RunTextScoringAsync(target_text, std::move(scoring_sync_callback),
                          store_token_lengths));
  ABSL_RETURN_IF_ERROR(task_controller->WaitUntilDone(Engine::kDefaultTimeout));
  return collected_responses;
}

absl::StatusOr<std::unique_ptr<SessionInterface::TaskController>>
SessionAdvanced::RunTextScoringAsync(
    const std::vector<absl::string_view>& target_text,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    bool store_token_lengths) {
  absl::MutexLock lock(mutex_);
  if (target_text.size() != 1) {
    return absl::InvalidArgumentError("Target text size should be 1.");
  }
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  auto cancelled = std::make_shared<std::atomic<bool>>(false);
  ABSL_ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());
  ABSL_RETURN_IF_ERROR(execution_manager_lock->AddTextScoringTask(
      session_id_, task_id, last_task_ids_, target_text, store_token_lengths,
      cancelled, std::move(callback)));

  return std::make_unique<AdvancedTaskController>(task_id, cancelled,
                                                  execution_manager_);
}

absl::StatusOr<Responses> SessionAdvanced::GenerateContent(
    const std::vector<InputData>& contents) {
  ABSL_RETURN_IF_ERROR(RunPrefill(contents));
  return RunDecode();
}

absl::Status SessionAdvanced::GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  return GenerateContentStream(contents, std::move(callback),
                               DecodeConfig::CreateDefault());
}

absl::Status SessionAdvanced::GenerateContentStream(
    const std::vector<InputData>& contents,
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
    const DecodeConfig& decode_config) {
  absl::AnyInvocable<void(absl::StatusOr<Responses>)> prefill_callback =
      [this, decode_config, stream_callback = std::move(callback)](
          absl::StatusOr<Responses> prefill_responses) mutable {
        if (!prefill_responses.ok()) {
          stream_callback(prefill_responses.status());
          return;
        }
        if (prefill_responses->GetTaskState() == TaskState::kDone) {
          auto decode_task_controller =
              RunDecodeAsync(std::move(stream_callback), decode_config);
          if (!decode_task_controller.ok()) {
            ABSL_LOG(ERROR) << "Failed to start decode task: "
                            << decode_task_controller.status();
          }
        } else if (IsTaskEndState(prefill_responses->GetTaskState())) {
          stream_callback(absl::CancelledError(
              "Prefill task finished in cancelled state."));
        }
      };

  ABSL_ASSIGN_OR_RETURN(auto task_controller,
                        RunPrefillAsync(contents, std::move(prefill_callback)));

  return absl::OkStatus();
}

absl::StatusOr<BenchmarkInfo> SessionAdvanced::GetBenchmarkInfo() {
  absl::MutexLock lock(mutex_);
  if (session_info_->benchmark_info.has_value()) {
    return session_info_->benchmark_info.value();
  }
  return absl::InternalError(
      "Benchmark is not enabled. Please make sure the BenchmarkParams is set "
      "in the EngineSettings.");
}

absl::StatusOr<BenchmarkInfo*> SessionAdvanced::GetMutableBenchmarkInfo() {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  return execution_manager_lock->GetMutableBenchmarkInfo(session_id_);
}

absl::StatusOr<std::unique_ptr<SessionInterface>> SessionAdvanced::Clone() {
  absl::Status status = absl::OkStatus();
  std::unique_ptr<SessionInterface> session;
  {
    absl::MutexLock lock(mutex_);
    ABSL_ASSIGN_OR_RETURN(
        session,
        CloneAsyncLocked([&status](absl::StatusOr<Responses> responses) {
          status = responses.status();
        }));
  }
  ABSL_RETURN_IF_ERROR(session->WaitUntilDone());
  ABSL_RETURN_IF_ERROR(status);
  return session;
}

absl::StatusOr<std::unique_ptr<SessionInterface>> SessionAdvanced::CloneAsync(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  absl::MutexLock lock(mutex_);
  return CloneAsyncLocked(std::move(callback));
}

absl::StatusOr<std::unique_ptr<SessionInterface>>
SessionAdvanced::CloneAsyncLocked(
    absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }

  ABSL_ASSIGN_OR_RETURN(auto task_id, execution_manager_lock->GetNewTaskId());

  ABSL_ASSIGN_OR_RETURN(
      auto session_id,
      execution_manager_lock->RegisterNewSession(
          session_info_->session_config, session_info_->benchmark_info));

  ABSL_RETURN_IF_ERROR(execution_manager_lock->AddCloneSessionTask(
      session_id_, task_id, last_task_ids_, session_id,
      std::make_shared<std::atomic<bool>>(false), std::move(callback)));

  last_task_ids_ = {task_id};

  ABSL_ASSIGN_OR_RETURN(auto session_info,
                        execution_manager_lock->GetSessionInfo(session_id));

  return absl::WrapUnique(new SessionAdvanced(session_id, execution_manager_,
                                              tokenizer_, session_info,
                                              session_state_, last_task_ids_,
                                              /*living_sessions_count=*/nullptr,
                                              session_handoff_identity_,
                                              exact_litert_logits_frame_contract_));
}

SessionAdvanced::~SessionAdvanced() {
  WaitUntilDone().IgnoreError();
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    ABSL_LOG(ERROR) << "Execution manager is not available.";
    return;
  }
  auto status = execution_manager_lock->ReleaseSession(session_id_);
  if (!status.ok()) {
    ABSL_LOG(ERROR) << "Error occurred when releasing session: " << status;
  }
  if (living_sessions_count_) {
    (*living_sessions_count_)--;
  }
};

absl::Status SessionAdvanced::SaveCheckpoint(absl::string_view label) {
  absl::MutexLock lock(mutex_);
  if (session_info_->session_config.GetMemoryStrategy() ==
      SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
    return absl::FailedPreconditionError(
        "Step-only checkpoints are not complete deterministic projection "
        "capsules.");
  }
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ABSL_ASSIGN_OR_RETURN(int current_step,
                        execution_manager_lock->GetCurrentStep(*session_info_));
  checkpoint_map_[label] = {current_step, session_state_, last_task_ids_};
  return absl::OkStatus();
}

absl::Status SessionAdvanced::RewindToCheckpoint(absl::string_view label) {
  absl::MutexLock lock(mutex_);

  if (session_info_->session_config.GetMemoryStrategy() ==
      SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
    return absl::FailedPreconditionError(
        "Step-only rewind is not supported for deterministic projection "
        "sessions.");
  }

  // Look up the checkpoint step.
  auto it = checkpoint_map_.find(label);
  if (it == checkpoint_map_.end()) {
    return absl::NotFoundError(absl::StrCat("Checkpoint not found: ", label));
  }
  int target_step = it->second.step;
  session_state_ = it->second.state;
  // Restore the task dependencies. This is necessary to prevent task-dependency
  // errors when a task is failed before the rewind. Otherwise, if the task
  // failed, because of the dependency chain, all tasks after the rewind point
  // would be considered as failed then finished immediately.
  last_task_ids_ = it->second.last_task_ids;

  // Remove all checkpoints after the current step.
  absl::erase_if(checkpoint_map_, [target_step](const auto& pair) {
    return pair.second.step > target_step;
  });

  // Get the execution manager and set the current step.
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  return execution_manager_lock->SetCurrentStep(*session_info_, target_step);
}

absl::Status SessionAdvanced::RewindToStep(int step) {
  absl::MutexLock lock(mutex_);
  if (session_info_->session_config.GetMemoryStrategy() ==
      SessionConfig::MemoryStrategy::kStatelessDeterministicProjection) {
    return absl::FailedPreconditionError(
        "Step-only rewind is not supported for deterministic projection "
        "sessions.");
  }
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  if (step > 0) {
    session_state_ = SessionState::kPrefilled;
  } else {
    session_state_ = SessionState::kFresh;
  }

  // Break dependency chain on raw step rewind. This is necessary to prevent
  // task-dependency errors when a task is failed before the rewind. Otherwise,
  // if the task failed, because of the dependency chain, all tasks after the
  // rewind point would be considered as failed then finished immediately.
  last_task_ids_.clear();

  // Remove all checkpoints after the current step.
  absl::erase_if(checkpoint_map_,
                 [step](const auto& pair) { return pair.second.step > step; });

  return execution_manager_lock->SetCurrentStep(*session_info_, step);
}

absl::StatusOr<int> SessionAdvanced::GetCurrentStep() const {
  auto execution_manager_lock = execution_manager_.lock();
  if (execution_manager_lock == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  return execution_manager_lock->GetCurrentStep(*session_info_);
}

absl::StatusOr<SessionHandoffIdentity>
SessionAdvanced::GetSessionHandoffIdentity() const {
  if (!session_handoff_identity_.has_value()) {
    return absl::FailedPreconditionError(
        "Session was not created with an engine-derived handoff identity.");
  }
  return *session_handoff_identity_;
}

absl::StatusOr<std::vector<std::vector<int>>>
SessionAdvanced::GetExactProcessedTokenHistory() const {
  auto execution_manager = execution_manager_.lock();
  if (execution_manager == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ABSL_RETURN_IF_ERROR(execution_manager->WaitUntilSessionDone(
      session_id_, Engine::kDefaultTimeout));

  absl::MutexLock lock(mutex_);
  return execution_manager->GetExactProcessedTokenHistory(session_id_,
                                                           last_task_ids_);
}

absl::StatusOr<std::string> SessionAdvanced::ExportHandoff(
    const SessionHandoffOptions& options) {
  std::string envelope;
  StringByteSink sink(&envelope);
  ABSL_ASSIGN_OR_RETURN(
      SessionContinuationStateWitness witness,
      ExportHandoffToWithWitness(options, &sink));
  (void)witness;
  return envelope;
}

absl::Status SessionAdvanced::ExportHandoffTo(
    const SessionHandoffOptions& options, ByteSink* sink) {
  absl::StatusOr<SessionContinuationStateWitness> witness =
      ExportHandoffToWithWitness(options, sink);
  return witness.ok() ? absl::OkStatus() : witness.status();
}

absl::StatusOr<SessionContinuationStateWitness>
SessionAdvanced::ExportHandoffToWithWitness(
    const SessionHandoffOptions& options, ByteSink* sink) {
  if (sink == nullptr) {
    return absl::InvalidArgumentError(
        "Session handoff output sink must not be null.");
  }
  auto execution_manager = execution_manager_.lock();
  if (execution_manager == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  // Let already-running asynchronous work finish before taking the session
  // mutex. The manager guard then keeps the complete synchronous stream
  // quiescent; sinks must not re-enter this session.
  ABSL_RETURN_IF_ERROR(execution_manager->WaitUntilSessionDone(
      session_id_, Engine::kDefaultTimeout));

  absl::MutexLock lock(mutex_);
  ABSL_RETURN_IF_ERROR(
      ValidateSessionHandoffConfig(session_info_->session_config));
  if (!checkpoint_map_.empty()) {
    return absl::UnimplementedError(
        "Session handoff does not preserve rewind checkpoints.");
  }
  SessionHandoffPhase phase;
  switch (session_state_) {
    case SessionState::kFresh:
      phase = SessionHandoffPhase::kFresh;
      break;
    case SessionState::kPrefilled:
      phase = SessionHandoffPhase::kPrefilled;
      break;
    case SessionState::kDecoded:
      phase = SessionHandoffPhase::kDecoded;
      break;
    default:
      return absl::InternalError(
          "Session handoff encountered an unknown live session phase.");
  }
  return ExportHandoffToWithWitnessLocked(execution_manager, phase, options,
                                          sink);
}

absl::StatusOr<SessionContinuationStateWitness>
SessionAdvanced::ExportHandoffToWithWitnessLocked(
    const std::shared_ptr<ExecutionManager>& execution_manager,
    SessionHandoffPhase phase, const SessionHandoffOptions& options,
    ByteSink* sink) {
  if (execution_manager == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateSessionHandoffConfig(session_info_->session_config));
  if (!checkpoint_map_.empty()) {
    return absl::UnimplementedError(
        "Session handoff does not preserve rewind checkpoints.");
  }
  ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity authoritative_identity,
                        GetSessionHandoffIdentity());

  HandoffWitnessByteSink witness_sink(sink);
  SessionHandoffSnapshot snapshot;
  snapshot.phase = phase;
  std::optional<SessionContinuationStateWitness> witness;
  bool consumer_invoked = false;
  ABSL_RETURN_IF_ERROR(execution_manager->ExportSessionSnapshotTo(
      session_id_, last_task_ids_,
      [&](const ExecutorSessionSnapshot& executor_snapshot,
          const StateInterface& state) -> absl::Status {
        if (consumer_invoked) {
          return absl::InternalError(
              "Session handoff snapshot consumer was invoked more than once.");
        }
        consumer_invoked = true;
        snapshot.executor = executor_snapshot;
        ABSL_ASSIGN_OR_RETURN(
            const Hash256 processed_history_token_bytes_hash,
            ComputeProcessedHistoryTokenBytesHash(snapshot.executor));
        ABSL_RETURN_IF_ERROR(EncodeSessionHandoffTo(
            snapshot, state, authoritative_identity, options, &witness_sink));
        ABSL_ASSIGN_OR_RETURN(const Hash256 envelope_hash,
                              witness_sink.Finish());
        ABSL_ASSIGN_OR_RETURN(
            witness,
            BuildContinuationWitness(snapshot, authoritative_identity,
                                     options.key_id,
                                     processed_history_token_bytes_hash,
                                     envelope_hash,
                                     witness_sink.emitted_size()));
        return absl::OkStatus();
      }));
  if (!consumer_invoked || !witness.has_value()) {
    return absl::InternalError(
        "Session handoff export completed without live-state evidence.");
  }
  ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(*witness));
  return *witness;
}

absl::Status SessionAdvanced::ImportHandoff(
    absl::string_view envelope, const SessionHandoffOptions& expected) {
  StringByteSource source(envelope);
  absl::StatusOr<SessionContinuationStateWitness> witness =
      ImportHandoffFromWithWitness(source, expected);
  return witness.ok() ? absl::OkStatus() : witness.status();
}

absl::Status SessionAdvanced::ImportHandoffFrom(
    const ByteSource& envelope, const SessionHandoffOptions& expected) {
  absl::StatusOr<SessionContinuationStateWitness> witness =
      ImportHandoffFromWithWitness(envelope, expected);
  return witness.ok() ? absl::OkStatus() : witness.status();
}

absl::StatusOr<SessionContinuationStateWitness>
SessionAdvanced::ImportHandoffFromWithWitness(
    const ByteSource& envelope, const SessionHandoffOptions& expected) {
  ABSL_ASSIGN_OR_RETURN(SessionHandoffIdentity authoritative_identity,
                        GetSessionHandoffIdentity());
  const uint64_t incoming_envelope_size = envelope.Size();
  ABSL_ASSIGN_OR_RETURN(const Hash256 incoming_envelope_hash,
                        HashByteSource(envelope));
  // Authenticate and validate identity, structure, PRNG, and token semantics
  // before touching the target session.
  ABSL_ASSIGN_OR_RETURN(DecodedSessionHandoff decoded,
                        DecodeSessionHandoffFrom(
                            envelope, authoritative_identity, expected));
  SessionHandoffSnapshot& snapshot = decoded.snapshot;
  if (envelope.Size() != incoming_envelope_size) {
    return absl::AbortedError(
        "Session handoff byte source changed while it was authenticated.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 incoming_processed_history_token_bytes_hash,
      ComputeProcessedHistoryTokenBytesHash(snapshot.executor));
  ABSL_ASSIGN_OR_RETURN(
      const SessionContinuationStateWitness incoming_witness,
      BuildContinuationWitness(snapshot, authoritative_identity,
                               expected.key_id,
                               incoming_processed_history_token_bytes_hash,
                               incoming_envelope_hash,
                               incoming_envelope_size));
  ByteSourceView serialized_state(&envelope, decoded.serialized_state_offset,
                                  decoded.serialized_state_size);
  if (serialized_state.Size() != decoded.serialized_state_size) {
    return absl::DataLossError(
        "Decoded session handoff state range is no longer readable.");
  }

  auto execution_manager = execution_manager_.lock();
  if (execution_manager == nullptr) {
    return absl::FailedPreconditionError("Execution manager is not available.");
  }
  ABSL_RETURN_IF_ERROR(execution_manager->WaitUntilSessionDone(
      session_id_, Engine::kDefaultTimeout));

  absl::MutexLock lock(mutex_);
  ABSL_RETURN_IF_ERROR(
      ValidateSessionHandoffConfig(session_info_->session_config));
  if (session_state_ != SessionState::kFresh || !last_task_ids_.empty() ||
      !checkpoint_map_.empty()) {
    return absl::FailedPreconditionError(
        "Session handoff import target must be a new, fresh session.");
  }
  ABSL_RETURN_IF_ERROR(execution_manager->ImportSessionSnapshotFrom(
      session_id_, snapshot.executor, serialized_state));

  // Executor import is the transactional commit point. Keep the public
  // session phase coherent with that committed live state even if a later
  // integrity postcondition detects a backend defect and returns an error.
  switch (snapshot.phase) {
    case SessionHandoffPhase::kFresh:
      session_state_ = SessionState::kFresh;
      break;
    case SessionHandoffPhase::kPrefilled:
      session_state_ = SessionState::kPrefilled;
      break;
    case SessionHandoffPhase::kDecoded:
      session_state_ = SessionState::kDecoded;
      break;
  }
  last_task_ids_.clear();

  // The incoming digest is used only as an expected commitment. Re-export the
  // committed target through the complete canonical encoder and an
  // allocation-free digest-only sink, then require exact evidence equality.
  // This proves the returned witness came from the live target rather than
  // being copied from the source envelope.
  ABSL_ASSIGN_OR_RETURN(
      const SessionContinuationStateWitness committed_witness,
      ExportHandoffToWithWitnessLocked(execution_manager, snapshot.phase,
                                       expected, /*sink=*/nullptr));
  if (committed_witness != incoming_witness) {
    return absl::DataLossError(
        "Committed session continuation state differs from the authenticated "
        "handoff source.");
  }
  return committed_witness;
}

std::optional<SessionDebugInfo> SessionAdvanced::GetSessionDebugInfo() const {
#if defined(LITERT_LM_DEBUGGER_ENABLED)
  SessionDebugInfo debug_info;
  debug_info.capture_dir = RuntimeDebugger::GetSessionCaptureDir(session_id_);
  return debug_info;
#else
  return std::nullopt;
#endif
}

}  // namespace litert::lm
