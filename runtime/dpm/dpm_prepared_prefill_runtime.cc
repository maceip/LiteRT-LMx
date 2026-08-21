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

#include "runtime/dpm/dpm_prepared_prefill_runtime.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/match.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/fresh_worker_protocol.h"
#include "runtime/engine/engine.h"
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/io_types.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kUtf8SourceChunkDomain =
    "litert-lmx-dpm-prefill-utf8-source-chunk-v1";
constexpr absl::string_view kTokenSourceChunkDomain =
    "litert-lmx-dpm-prefill-token-source-chunk-v1";

bool IsZeroHash(const Hash256& hash) { return hash == Hash256{}; }

void AppendU32(uint32_t value, std::string* output) {
  for (int shift = 24; shift >= 0; shift -= 8) {
    output->push_back(static_cast<char>((value >> shift) & 0xff));
  }
}

absl::StatusOr<Hash256> HashSourceText(absl::string_view text) {
  if (text.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError(
        "Prepared DPM UTF-8 source chunk exceeds its canonical length field.");
  }
  std::string canonical;
  AppendU32(static_cast<uint32_t>(text.size()), &canonical);
  canonical.append(text.data(), text.size());
  Sha256Hasher hasher;
  hasher.Update(kUtf8SourceChunkDomain);
  hasher.Update(canonical);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "Prepared DPM UTF-8 source hashing produced an empty digest.");
  }
  return result;
}

absl::StatusOr<Hash256> HashSourceTokenIds(
    const std::vector<int>& token_ids) {
  if (token_ids.size() > std::numeric_limits<uint32_t>::max()) {
    return absl::ResourceExhaustedError(
        "Prepared DPM token source exceeds its canonical count field.");
  }
  std::string canonical;
  AppendU32(static_cast<uint32_t>(token_ids.size()), &canonical);
  for (int token_id : token_ids) {
    AppendU32(static_cast<uint32_t>(token_id), &canonical);
  }
  Sha256Hasher hasher;
  hasher.Update(kTokenSourceChunkDomain);
  hasher.Update(canonical);
  const Hash256 result = hasher.Finalize();
  if (IsZeroHash(result)) {
    return absl::InternalError(
        "Prepared DPM token source hashing produced an empty digest.");
  }
  return result;
}

bool IsValidUtf8(absl::string_view text) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(text.data());
  size_t index = 0;
  while (index < text.size()) {
    const uint8_t lead = bytes[index++];
    if (lead <= 0x7f) continue;
    size_t continuation_count = 0;
    uint8_t minimum_second = 0x80;
    uint8_t maximum_second = 0xbf;
    if (lead >= 0xc2 && lead <= 0xdf) {
      continuation_count = 1;
    } else if (lead >= 0xe0 && lead <= 0xef) {
      continuation_count = 2;
      if (lead == 0xe0) minimum_second = 0xa0;
      if (lead == 0xed) maximum_second = 0x9f;
    } else if (lead >= 0xf0 && lead <= 0xf4) {
      continuation_count = 3;
      if (lead == 0xf0) minimum_second = 0x90;
      if (lead == 0xf4) maximum_second = 0x8f;
    } else {
      return false;
    }
    if (continuation_count > text.size() - index ||
        bytes[index] < minimum_second || bytes[index] > maximum_second) {
      return false;
    }
    ++index;
    for (size_t offset = 1; offset < continuation_count; ++offset, ++index) {
      if (bytes[index] < 0x80 || bytes[index] > 0xbf) return false;
    }
  }
  return true;
}

absl::StatusOr<std::vector<int32_t>> CanonicalizeTokenIds(
    const std::vector<int>& token_ids, uint32_t vocabulary_size) {
  if (token_ids.empty()) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill resolved an empty token segment.");
  }
  std::vector<int32_t> canonical;
  canonical.reserve(token_ids.size());
  for (int token_id : token_ids) {
    if (token_id < 0 ||
        static_cast<uint32_t>(token_id) >= vocabulary_size) {
      return absl::InvalidArgumentError(
          "Prepared DPM prefill resolved an out-of-vocabulary token.");
    }
    canonical.push_back(static_cast<int32_t>(token_id));
  }
  return canonical;
}

absl::StatusOr<std::vector<int32_t>> ResolveTextTokenIds(
    absl::string_view text, support::Tokenizer* tokenizer,
    int start_token_id, absl::string_view bos_string,
    uint32_t vocabulary_size) {
  if (text.empty() || !IsValidUtf8(text) || tokenizer == nullptr) {
    return absl::InvalidArgumentError(
        "Prepared DPM text chunk is empty, invalid UTF-8, or has no "
        "tokenizer.");
  }
  bool begins_with_bos = false;
  if (!bos_string.empty() && absl::StartsWith(text, bos_string)) {
    text.remove_prefix(bos_string.size());
    begins_with_bos = true;
  }
  ABSL_ASSIGN_OR_RETURN(std::vector<int> token_ids,
                        tokenizer->TextToTokenIds(text));
  if (begins_with_bos) token_ids.insert(token_ids.begin(), start_token_id);
  return CanonicalizeTokenIds(token_ids, vocabulary_size);
}

absl::StatusOr<std::vector<int32_t>> InspectSingleHistory(
    const Engine::Session& session, uint64_t* step) {
  if (step == nullptr) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill history inspection requires a step output.");
  }
  ABSL_ASSIGN_OR_RETURN(const int current_step, session.GetCurrentStep());
  ABSL_ASSIGN_OR_RETURN(
      const std::vector<std::vector<int>> histories,
      session.GetExactProcessedTokenHistory());
  if (current_step < 0 || histories.size() != 1 ||
      histories.front().size() != static_cast<size_t>(current_step)) {
    return absl::DataLossError(
        "Prepared DPM prefill session step differs from its single exact "
        "history.");
  }
  if (static_cast<uint64_t>(current_step) >
      kMaximumDPMPreparedPrefillTokenIds) {
    return absl::ResourceExhaustedError(
        "Prepared DPM prefill start history exceeds its admitted bound.");
  }
  std::vector<int32_t> canonical;
  canonical.reserve(histories.front().size());
  for (int token_id : histories.front()) {
    if (token_id < 0) {
      return absl::DataLossError(
          "Prepared DPM prefill exact history contains a negative token.");
    }
    canonical.push_back(static_cast<int32_t>(token_id));
  }
  *step = static_cast<uint64_t>(current_step);
  return canonical;
}

absl::StatusOr<Hash256> HashCanonicalHistory(
    const std::vector<int32_t>& token_ids) {
  ABSL_ASSIGN_OR_RETURN(const std::string token_bytes,
                        EncodeFreshWorkerTokenIds(token_ids));
  Sha256Hasher hasher;
  const Hash256 digest = hasher.OneShot(token_bytes);
  if (IsZeroHash(digest)) {
    return absl::InternalError(
        "Prepared DPM prefill history produced an empty digest.");
  }
  return digest;
}

absl::Status ValidateRestoredStartWitness(
    const DPMPreparedPrefillStart& start,
    const SessionHandoffIdentity& session_identity, uint64_t current_step,
    const Hash256& history_hash) {
  if (!start.restore_checkpoint_id.has_value() ||
      IsZeroHash(*start.restore_checkpoint_id) ||
      !start.restored_state_witness.has_value()) {
    return absl::InvalidArgumentError(
        "Own-position DPM prefill preparation lacks a checkpoint or import "
        "witness.");
  }
  const SessionContinuationStateWitness& witness =
      *start.restored_state_witness;
  ABSL_RETURN_IF_ERROR(ValidateSessionContinuationStateWitness(witness));
  if (witness.session_identity != session_identity ||
      witness.phase != SessionHandoffPhase::kDecoded ||
      !witness.ran_decode || witness.current_step <= 0 ||
      static_cast<uint64_t>(witness.current_step) != current_step ||
      witness.processed_history_token_bytes_hash != history_hash) {
    return absl::FailedPreconditionError(
        "Own-position DPM prefill import witness differs from its live "
        "session start state.");
  }
  return absl::OkStatus();
}

absl::Status ValidateExecutionStart(
    const Engine::Session& session, const DPMPreparedPrefillPlan& plan,
    const std::optional<SessionContinuationStateWitness>& restored_witness,
    std::vector<int32_t>* history) {
  if (history == nullptr) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill execution requires history storage.");
  }
  ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity session_identity,
                        session.GetSessionHandoffIdentity());
  if (session_identity != plan.session_identity) {
    return absl::FailedPreconditionError(
        "Prepared DPM prefill plan identity differs from its live session.");
  }
  uint64_t current_step = 0;
  ABSL_ASSIGN_OR_RETURN(*history,
                        InspectSingleHistory(session, &current_step));
  ABSL_ASSIGN_OR_RETURN(const Hash256 history_hash,
                        HashCanonicalHistory(*history));
  if (current_step != plan.start_step ||
      history->size() != plan.start_history_token_count ||
      history_hash != plan.start_history_token_bytes_hash) {
    return absl::FailedPreconditionError(
        "Prepared DPM prefill plan start history differs from its live "
        "session.");
  }
  if (plan.start_kind == DPMPreparedPrefillStartKind::kFreshSession) {
    if (restored_witness.has_value()) {
      return absl::InvalidArgumentError(
          "Fresh prepared DPM prefill received a restore witness.");
    }
    return absl::OkStatus();
  }
  if (!restored_witness.has_value()) {
    return absl::InvalidArgumentError(
        "Own-position prepared DPM prefill execution lacks its live import "
        "witness.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateSessionContinuationStateWitness(*restored_witness));
  if (!plan.start_state_witness_id.has_value() ||
      restored_witness->witness_id != *plan.start_state_witness_id ||
      restored_witness->session_identity != plan.session_identity ||
      restored_witness->phase != SessionHandoffPhase::kDecoded ||
      !restored_witness->ran_decode || restored_witness->current_step < 0 ||
      static_cast<uint64_t>(restored_witness->current_step) !=
          plan.start_step ||
      restored_witness->processed_history_token_bytes_hash !=
          plan.start_history_token_bytes_hash) {
    return absl::FailedPreconditionError(
        "Prepared DPM prefill restore witness differs from its plan or live "
        "session.");
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<DPMPreparedPrefillPlan> PrepareDPMEnginePrefillPlan(
    Engine* engine, Engine::Session* session,
    const std::vector<DPMAgentGenerationRequest::PrefillChunk>& source_chunks,
    const Hash256& logical_agent_request_hash,
    const DPMPreparedPrefillStart& start) {
  if (engine == nullptr || session == nullptr || source_chunks.empty() ||
      source_chunks.size() > kMaximumDPMPreparedPrefillCalls ||
      IsZeroHash(logical_agent_request_hash)) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill requires a loaded Engine, live session, bounded "
        "source chunks, and logical request hash.");
  }
  const SessionConfig& session_config = session->GetSessionConfig();
  if (session_config.GetApplyPromptTemplateInSession() ||
      session_config.GetMemoryStrategy() !=
          SessionConfig::MemoryStrategy::kStateful) {
    return absl::FailedPreconditionError(
        "Prepared DPM agent prefill requires a stateful session without "
        "prompt templates.");
  }
  ABSL_ASSIGN_OR_RETURN(const SessionHandoffIdentity session_identity,
                        session->GetSessionHandoffIdentity());
  ABSL_ASSIGN_OR_RETURN(
      const SessionHandoffIdentity engine_identity,
      engine->ResolveSessionHandoffIdentity(session_config));
  if (engine_identity != session_identity) {
    return absl::FailedPreconditionError(
        "Prepared DPM prefill session identity differs from its loaded "
        "Engine.");
  }
  const int vocabulary_size = engine->GetTokenizer().GetVocabSize();
  if (vocabulary_size <= 0) {
    return absl::FailedPreconditionError(
        "Prepared DPM prefill tokenizer has no measurable vocabulary.");
  }
  uint64_t start_step = 0;
  ABSL_ASSIGN_OR_RETURN(std::vector<int32_t> start_history,
                        InspectSingleHistory(*session, &start_step));
  for (int32_t token_id : start_history) {
    if (static_cast<uint32_t>(token_id) >=
        static_cast<uint32_t>(vocabulary_size)) {
      return absl::DataLossError(
          "Prepared DPM prefill start history contains an out-of-vocabulary "
          "token.");
    }
  }
  ABSL_ASSIGN_OR_RETURN(const Hash256 start_history_hash,
                        HashCanonicalHistory(start_history));

  switch (start.kind) {
    case DPMPreparedPrefillStartKind::kFreshSession:
      if (start_step != 0 || !start_history.empty() ||
          start.restore_checkpoint_id.has_value() ||
          start.restored_state_witness.has_value()) {
        return absl::FailedPreconditionError(
            "Fresh prepared DPM prefill did not receive a truly fresh "
            "session.");
      }
      break;
    case DPMPreparedPrefillStartKind::kOwnPositionRestore:
      ABSL_RETURN_IF_ERROR(ValidateRestoredStartWitness(
          start, session_identity, start_step, start_history_hash));
      break;
    default:
      return absl::InvalidArgumentError(
          "Prepared DPM prefill has an unknown start kind.");
  }

  const int start_token_id = session_config.GetStartTokenId();
  support::Tokenizer& tokenizer =
      const_cast<support::Tokenizer&>(engine->GetTokenizer());
  std::string bos_string;
  if (start_token_id >= 0) {
    const std::vector<int> bos_token_ids = {start_token_id};
    ABSL_ASSIGN_OR_RETURN(bos_string,
                          tokenizer.TokenIdsToText(bos_token_ids));
  }

  DPMPreparedPrefillPlan plan;
  plan.session_identity = session_identity;
  plan.logical_agent_request_hash = logical_agent_request_hash;
  plan.start_kind = start.kind;
  plan.restore_checkpoint_id = start.restore_checkpoint_id;
  if (start.restored_state_witness.has_value()) {
    plan.start_state_witness_id =
        start.restored_state_witness->witness_id;
  }
  plan.start_step = start_step;
  plan.start_history_token_count = start_history.size();
  plan.start_history_token_bytes_hash = start_history_hash;
  plan.vocabulary_size = static_cast<uint32_t>(vocabulary_size);
  plan.calls.reserve(source_chunks.size());

  uint64_t position = start_step;
  bool first_call = true;
  for (const DPMAgentGenerationRequest::PrefillChunk& source :
       source_chunks) {
    DPMPreparedPrefillCall call;
    call.start_token_position = position;
    if (first_call &&
        start.kind == DPMPreparedPrefillStartKind::kFreshSession &&
        !bos_string.empty()) {
      if (start_token_id < 0 || start_token_id >= vocabulary_size) {
        return absl::FailedPreconditionError(
            "Prepared DPM prefill resolved an invalid Engine BOS token.");
      }
      call.token_segments.push_back(DPMPreparedPrefillSegment{
          .token_ids = {static_cast<int32_t>(start_token_id)}});
    }

    switch (source.encoding) {
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text: {
        if (!source.token_ids.empty()) {
          return absl::InvalidArgumentError(
              "Prepared DPM text source also contains exact token IDs.");
        }
        call.source_encoding =
            DPMPreparedPrefillSourceEncoding::kUtf8Text;
        ABSL_ASSIGN_OR_RETURN(call.source_chunk_hash,
                              HashSourceText(source.text));
        ABSL_ASSIGN_OR_RETURN(
            std::vector<int32_t> resolved,
            ResolveTextTokenIds(
                source.text, &tokenizer, start_token_id, bos_string,
                static_cast<uint32_t>(vocabulary_size)));
        call.token_segments.push_back(
            DPMPreparedPrefillSegment{.token_ids = std::move(resolved)});
        break;
      }
      case DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds: {
        if (!source.text.empty()) {
          return absl::InvalidArgumentError(
              "Prepared DPM token source also contains UTF-8 text.");
        }
        call.source_encoding =
            DPMPreparedPrefillSourceEncoding::kExactTokenIds;
        ABSL_ASSIGN_OR_RETURN(
            std::vector<int32_t> resolved,
            CanonicalizeTokenIds(
                source.token_ids, static_cast<uint32_t>(vocabulary_size)));
        ABSL_ASSIGN_OR_RETURN(call.source_chunk_hash,
                              HashSourceTokenIds(source.token_ids));
        call.token_segments.push_back(
            DPMPreparedPrefillSegment{.token_ids = std::move(resolved)});
        break;
      }
      default:
        return absl::InvalidArgumentError(
            "Prepared DPM prefill source has an unknown encoding.");
    }
    if (IsZeroHash(call.source_chunk_hash)) {
      return absl::InternalError(
          "Prepared DPM prefill source hashing produced an empty digest.");
    }
    uint64_t call_tokens = 0;
    for (const DPMPreparedPrefillSegment& segment : call.token_segments) {
      call_tokens += segment.token_ids.size();
    }
    if (call_tokens == 0 ||
        call_tokens > kMaximumDPMPreparedPrefillTokenIds - position) {
      return absl::ResourceExhaustedError(
          "Prepared DPM prefill call exceeds the exact-history bound.");
    }
    position += call_tokens;
    call.end_token_position = position;
    plan.calls.push_back(std::move(call));
    first_call = false;
  }
  plan.end_step = position;
  ABSL_ASSIGN_OR_RETURN(plan.canonical_source_chunks_hash,
                        ComputeDPMPreparedPrefillSourceChunksHash(plan.calls));
  ABSL_ASSIGN_OR_RETURN(
      plan.resolved_token_plan_hash,
      ComputeDPMPreparedPrefillResolvedTokenPlanHash(plan));
  ABSL_ASSIGN_OR_RETURN(
      plan.shape_schedule_hash,
      ComputeDPMPreparedPrefillShapeScheduleHash(plan));
  ABSL_ASSIGN_OR_RETURN(plan.plan_id,
                        ComputeDPMPreparedPrefillPlanId(plan));
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(plan));
  return plan;
}

absl::Status ExecuteDPMEnginePrefillPlan(
    Engine::Session* session, const DPMPreparedPrefillPlan& plan,
    const std::optional<SessionContinuationStateWitness>&
        restored_state_witness) {
  if (session == nullptr) {
    return absl::InvalidArgumentError(
        "Prepared DPM prefill execution requires a live session.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMPreparedPrefillPlan(plan));
  std::vector<int32_t> expected_history;
  ABSL_RETURN_IF_ERROR(ValidateExecutionStart(
      *session, plan, restored_state_witness, &expected_history));
  expected_history.reserve(plan.end_step);

  for (const DPMPreparedPrefillCall& call : plan.calls) {
    std::vector<InputData> preprocessed_contents;
    preprocessed_contents.reserve(call.token_segments.size());
    for (const DPMPreparedPrefillSegment& segment : call.token_segments) {
      std::vector<int> token_ids;
      token_ids.reserve(segment.token_ids.size());
      for (int32_t token_id : segment.token_ids) {
        token_ids.push_back(static_cast<int>(token_id));
        expected_history.push_back(token_id);
      }
      ABSL_ASSIGN_OR_RETURN(
          auto token_buffer,
          support::Tokenizer::TokenIdsToTensorBuffer(token_ids));
      preprocessed_contents.emplace_back(InputText(std::move(token_buffer)));
    }

    absl::Status completion = absl::UnknownError(
        "Prepared DPM prefill callback did not run.");
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<Engine::Session::TaskController> task,
        session->PrefillPreprocessedContents(
            std::move(preprocessed_contents),
            [&completion](absl::StatusOr<Responses> responses) {
              completion = responses.status();
            }));
    if (task == nullptr) {
      return absl::InternalError(
          "Prepared DPM prefill returned a null task controller.");
    }
    ABSL_RETURN_IF_ERROR(task->WaitUntilDone(Engine::kDefaultTimeout));
    ABSL_RETURN_IF_ERROR(completion);
    ABSL_ASSIGN_OR_RETURN(const int current_step, session->GetCurrentStep());
    if (current_step < 0 ||
        static_cast<uint64_t>(current_step) != call.end_token_position) {
      return absl::DataLossError(
          "Prepared DPM prefill call did not end at its resolved own "
          "position.");
    }
  }

  uint64_t final_step = 0;
  ABSL_ASSIGN_OR_RETURN(const std::vector<int32_t> final_history,
                        InspectSingleHistory(*session, &final_step));
  if (final_step != plan.end_step || final_history != expected_history) {
    return absl::DataLossError(
        "Prepared DPM prefill execution did not produce its exact planned "
        "history.");
  }
  return absl::OkStatus();
}

}  // namespace litert::lm
