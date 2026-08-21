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

#include "runtime/executor/llm_executor_processed_tokens.h"

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl

namespace litert::lm {

// tokens_ must have at least one element.
ProcessedTokens::ProcessedTokens() : tokens_(1) {}

int ProcessedTokens::TokenCount() const {
  return GetStep() + (HasPendingInputToken() ? 1 : 0);
}

absl::Status ProcessedTokens::ReduceTokenCandidates(size_t index) {
  if (index >= tokens_.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("index must be less than tokens_.size(), got ", index,
                     " vs ", tokens_.size()));
  }
  if (index > 0) {
    std::swap(tokens_[0], tokens_[index]);
  }
  tokens_.resize(1);
  return absl::OkStatus();
}

absl::Status ProcessedTokens::BroadcastTokenCandidates(size_t size) {
  if (tokens_.size() != 1) {
    return absl::FailedPreconditionError(
        "ExpandTokenCandidates called with tokens_.size() != 1.");
  }
  if (size < tokens_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("size must be greater than or equal to 1, got ", size));
  }

  tokens_.reserve(size);
  for (size_t i = 1; i < size; ++i) {
    tokens_.push_back(tokens_[0]);
  }
  return absl::OkStatus();
}

ProcessedTokens::StepAndToken ProcessedTokens::GetNextUnprocessedToken() const {
  return StepAndToken{.step = GetStep(), .token = GetPendingInputToken()};
}

void ProcessedTokens::AddProcessedTokens(const std::vector<int>& token_ids) {
  for (auto& t : tokens_) {
    t.token_ids.insert(t.token_ids.end(), token_ids.begin(), token_ids.end());
  }
}

absl::Status ProcessedTokens::AddPendingInputToken(
    const std::vector<std::shared_ptr<TokenData>>& token) {
  if (token.size() != tokens_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Token size must be equal to tokens_.size(), got ",
                     token.size(), " vs ", tokens_.size()));
  }
  if (HasPendingInputToken()) {
    return absl::AlreadyExistsError(
        "AddPendingInputToken called with an existing pending token.");
  }
  for (int i = 0; i < token.size(); ++i) {
    tokens_[i].pending_input_token = token[i];
  }
  return absl::OkStatus();
}

absl::Status ProcessedTokens::RollBackToStep(int new_step) {
  if (new_step < 0) {
    return absl::InternalError(
        absl::StrCat("new_step must be non-negative, got ", new_step));
  }
  if (new_step > TokenCount()) {
    return absl::InternalError(absl::StrCat(
        "new_step must be less than or equal to TokenCount(), got ", new_step,
        " vs ", TokenCount()));
  }

  if (new_step == TokenCount()) {
    return absl::OkStatus();
  }

  for (auto& t : tokens_) {
    t.token_ids.resize(new_step);
    t.pending_input_token = nullptr;
  }
  return absl::OkStatus();
}

std::vector<int> ProcessedTokens::GetTokenAtStep(int step) const {
  std::vector<int> token;
  if (step < 0 || step >= TokenCount()) {
    return token;
  }

  token.reserve(tokens_.size());
  if (step == GetStep() && HasPendingInputToken()) {
    for (const auto& t : tokens_) {
      token.push_back(t.pending_input_token->id());
    }
    return token;
  }

  for (const auto& t : tokens_) {
    token.push_back(t.token_ids[step]);
  }
  return token;
}

absl::Status ProcessedTokens::MarkPendingInputTokenAsProcessed() {
  if (!HasPendingInputToken()) {
    return absl::NotFoundError(
        "MarkPendingInputTokenAsProcessed called with no pending token.");
  }

  for (auto& t : tokens_) {
    t.token_ids.push_back(t.pending_input_token->id());
    t.pending_input_token = nullptr;
  }
  return absl::OkStatus();
}

std::vector<std::vector<int>> ProcessedTokens::GetCopyOfTokens() const {
  std::vector<std::vector<int>> copy_of_tokens;
  copy_of_tokens.reserve(tokens_.size());
  for (const auto& t : tokens_) {
    copy_of_tokens.push_back(t.token_ids);
    if (t.pending_input_token) {
      copy_of_tokens.back().push_back(t.pending_input_token->id());
    }
  }
  return copy_of_tokens;
}

const std::vector<int>& ProcessedTokens::GetTokensUnsafe() const {
  ABSL_CHECK_EQ(tokens_[0].pending_input_token, nullptr);
  return tokens_[0].token_ids;
}

void ProcessedTokens::InvalidatePendingInputToken() {
  for (auto& t : tokens_) {
    t.pending_input_token = nullptr;
  }
}

absl::StatusOr<ProcessedTokens::Snapshot> ProcessedTokens::ExportSnapshot()
    const {
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffSupport());
  Snapshot snapshot;
  snapshot.processed_token_ids.reserve(tokens_.size());
  if (tokens_[0].pending_input_token != nullptr) {
    snapshot.pending_token_ids.reserve(tokens_.size());
  }
  for (const Tokens& tokens : tokens_) {
    snapshot.processed_token_ids.push_back(tokens.token_ids);
    if (tokens.pending_input_token != nullptr) {
      snapshot.pending_token_ids.push_back(tokens.pending_input_token->id());
    }
  }
  return snapshot;
}

absl::Status ProcessedTokens::ValidateSessionHandoffSupport() const {
  if (tokens_.empty()) {
    return absl::InternalError("ProcessedTokens has no token candidates.");
  }
  const size_t processed_count = tokens_[0].token_ids.size();
  const bool has_pending = tokens_[0].pending_input_token != nullptr;
  const size_t pending_count = has_pending ? 1 : 0;
  if (processed_count >
      static_cast<size_t>(std::numeric_limits<int>::max()) - pending_count) {
    return absl::ResourceExhaustedError(
        "Processed token history exceeds the supported step range.");
  }
  for (const Tokens& tokens : tokens_) {
    if (tokens.token_ids.size() != processed_count) {
      return absl::FailedPreconditionError(
          "Processed token candidates have different lengths.");
    }
    if ((tokens.pending_input_token != nullptr) != has_pending) {
      return absl::FailedPreconditionError(
          "Processed token candidates disagree about pending-token state.");
    }
    if (has_pending) {
      if (!tokens.pending_input_token->embedding().empty() ||
          !tokens.pending_input_token->per_layer_embedding().empty()) {
        return absl::UnimplementedError(
            "Session handoff does not support pending token embeddings.");
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ProcessedTokens::ValidateTokenIds(int vocabulary_size) const {
  ABSL_RETURN_IF_ERROR(ValidateSessionHandoffSupport());
  if (vocabulary_size <= 0) {
    return absl::FailedPreconditionError(
        "Loaded decode vocabulary must be positive.");
  }
  for (const Tokens& tokens : tokens_) {
    for (int token_id : tokens.token_ids) {
      if (token_id < 0 || token_id >= vocabulary_size) {
        return absl::InvalidArgumentError(
            "Live session token ID is outside the loaded vocabulary.");
      }
    }
    if (tokens.pending_input_token != nullptr &&
        (tokens.pending_input_token->id() < 0 ||
         tokens.pending_input_token->id() >= vocabulary_size)) {
      return absl::InvalidArgumentError(
          "Live pending token ID is outside the loaded vocabulary.");
    }
  }
  return absl::OkStatus();
}

absl::Status ProcessedTokens::ValidateSnapshot(const Snapshot& snapshot) {
  if (snapshot.processed_token_ids.empty()) {
    return absl::InvalidArgumentError(
        "Processed token snapshot must contain at least one candidate.");
  }
  const size_t processed_count = snapshot.processed_token_ids[0].size();
  for (const auto& candidate : snapshot.processed_token_ids) {
    if (candidate.size() != processed_count) {
      return absl::InvalidArgumentError(
          "Processed token snapshot candidates have different lengths.");
    }
  }
  if (!snapshot.pending_token_ids.empty() &&
      snapshot.pending_token_ids.size() !=
          snapshot.processed_token_ids.size()) {
    return absl::InvalidArgumentError(
        "Pending token count must be zero or equal the candidate count.");
  }
  const size_t pending_count = snapshot.pending_token_ids.empty() ? 0 : 1;
  if (processed_count >
      static_cast<size_t>(std::numeric_limits<int>::max()) - pending_count) {
    return absl::ResourceExhaustedError(
        "Processed token snapshot exceeds the supported step range.");
  }
  return absl::OkStatus();
}

absl::Status ProcessedTokens::ValidateSnapshotTokenIds(
    const Snapshot& snapshot, int vocabulary_size) {
  ABSL_RETURN_IF_ERROR(ValidateSnapshot(snapshot));
  if (vocabulary_size <= 0) {
    return absl::FailedPreconditionError(
        "Loaded decode vocabulary must be positive.");
  }
  const auto validate_token_id =
      [vocabulary_size](int token_id) -> absl::Status {
    if (token_id < 0 || token_id >= vocabulary_size) {
      return absl::InvalidArgumentError(
          "Session handoff token ID is outside the loaded vocabulary.");
    }
    return absl::OkStatus();
  };
  for (const auto& candidate : snapshot.processed_token_ids) {
    for (int token_id : candidate) {
      ABSL_RETURN_IF_ERROR(validate_token_id(token_id));
    }
  }
  for (int token_id : snapshot.pending_token_ids) {
    ABSL_RETURN_IF_ERROR(validate_token_id(token_id));
  }
  return absl::OkStatus();
}

absl::StatusOr<ProcessedTokens> ProcessedTokens::FromSnapshot(
    const Snapshot& snapshot) {
  ABSL_RETURN_IF_ERROR(ValidateSnapshot(snapshot));

  ProcessedTokens restored;
  restored.tokens_.clear();
  restored.tokens_.reserve(snapshot.processed_token_ids.size());
  for (size_t i = 0; i < snapshot.processed_token_ids.size(); ++i) {
    Tokens tokens;
    tokens.token_ids = snapshot.processed_token_ids[i];
    if (!snapshot.pending_token_ids.empty()) {
      tokens.pending_input_token =
          std::make_shared<TokenData>(snapshot.pending_token_ids[i]);
    }
    restored.tokens_.push_back(std::move(tokens));
  }
  return restored;
}

int ProcessedTokens::GetStep() const { return tokens_[0].token_ids.size(); }

bool ProcessedTokens::HasPendingInputToken() const {
  return tokens_[0].pending_input_token != nullptr;
}

std::vector<std::shared_ptr<TokenData>> ProcessedTokens::GetPendingInputToken()
    const {
  std::vector<std::shared_ptr<TokenData>> token;
  if (HasPendingInputToken()) {
    token.reserve(tokens_.size());
    for (const auto& t : tokens_) {
      token.push_back(t.pending_input_token);
    }
  }
  return token;
}

}  // namespace litert::lm
