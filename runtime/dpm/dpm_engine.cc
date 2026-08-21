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

#include "runtime/dpm/dpm_engine.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_append.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/dpm/correction_digest.h"
#include "runtime/dpm/dpm_agent_replay_runtime.h"
#include "runtime/dpm/dpm_projection_manifest.h"
#include "runtime/dpm/dpm_projection_prompt.h"
#include "runtime/dpm/dpm_receipt_validation.h"
#include "runtime/platform/hash/sha256_hasher.h"
#include "runtime/util/byte_stream.h"

namespace litert::lm {
namespace {

constexpr absl::string_view kTranscriptDomain = "DPM_AGENT_TRANSCRIPT_V1";
constexpr absl::string_view kAgentRequestDomain = "DPM_AGENT_REQUEST_V2";
constexpr size_t kMaximumCheckpointKeyIdSize = 1024;

bool IsZeroHash(const Hash256& hash) {
  for (uint8_t byte : hash.bytes) {
    if (byte != 0) return false;
  }
  return true;
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

bool ContainsControlByte(absl::string_view text) {
  for (unsigned char byte : text) {
    if (byte < 0x20 || byte == 0x7f) return true;
  }
  return false;
}

Hash256 Sha256(absl::string_view bytes) {
  Sha256Hasher hasher;
  hasher.Update(bytes);
  return hasher.Finalize();
}

void UpdateU64(uint64_t value, Sha256Hasher* hasher) {
  std::array<char, 8> encoded{};
  for (int i = 7; i >= 0; --i) {
    encoded[7 - i] = static_cast<char>((value >> (i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(encoded.data(), encoded.size()));
}

void UpdateU32(uint32_t value, Sha256Hasher* hasher) {
  std::array<char, 4> encoded{};
  for (int i = 3; i >= 0; --i) {
    encoded[3 - i] = static_cast<char>((value >> (i * 8)) & 0xff);
  }
  hasher->Update(absl::string_view(encoded.data(), encoded.size()));
}

void UpdateBytesFrame(char tag, absl::string_view bytes,
                      Sha256Hasher* hasher) {
  hasher->Update(absl::string_view(&tag, 1));
  UpdateU64(bytes.size(), hasher);
  hasher->Update(bytes);
}

void UpdateTokenFrame(const std::vector<int>& token_ids,
                      Sha256Hasher* hasher) {
  constexpr char kTag = 'T';
  hasher->Update(absl::string_view(&kTag, 1));
  UpdateU64(token_ids.size(), hasher);
  for (int token_id : token_ids) {
    UpdateU32(static_cast<uint32_t>(static_cast<int32_t>(token_id)), hasher);
  }
}

Hash256 SnapshotDigest(const Sha256Hasher& hasher) {
  Sha256Hasher copy = hasher;
  return copy.Finalize();
}

struct TranscriptCursor {
  std::optional<Hash256> correction_digest;
  Sha256Hasher hasher;
};

void ResetTranscriptHasher(const Hash256& correction_digest,
                           TranscriptCursor* cursor) {
  cursor->correction_digest = correction_digest;
  cursor->hasher = Sha256Hasher();
  cursor->hasher.Update(kTranscriptDomain);
  const absl::string_view digest_bytes(
      reinterpret_cast<const char*>(correction_digest.bytes.data()),
      correction_digest.bytes.size());
  UpdateBytesFrame('C', digest_bytes, &cursor->hasher);
}

Hash256 ComputeLogicalAgentRequestHash(
    const TranscriptCursor& transcript_before_turn,
    absl::string_view canonical_agent_input, uint32_t max_decision_tokens,
    const Hash256& correction_digest,
    const SessionHandoffIdentity& agent_identity) {
  TranscriptCursor logical_prefix = transcript_before_turn;
  if (!logical_prefix.correction_digest.has_value() ||
      *logical_prefix.correction_digest != correction_digest) {
    ResetTranscriptHasher(correction_digest, &logical_prefix);
  }
  const Hash256 prefix_hash = SnapshotDigest(logical_prefix.hasher);
  Sha256Hasher hasher;
  hasher.Update(kAgentRequestDomain);
  UpdateBytesFrame(
      'C',
      absl::string_view(
          reinterpret_cast<const char*>(correction_digest.bytes.data()),
          correction_digest.bytes.size()),
      &hasher);
  UpdateBytesFrame(
      'T',
      absl::string_view(reinterpret_cast<const char*>(prefix_hash.bytes.data()),
                        prefix_hash.bytes.size()),
      &hasher);
  UpdateBytesFrame(
      'M', absl::string_view(reinterpret_cast<const char*>(
                                 agent_identity.model_artifact_hash.bytes.data()),
                             agent_identity.model_artifact_hash.bytes.size()),
      &hasher);
  UpdateBytesFrame(
      'R', absl::string_view(reinterpret_cast<const char*>(
                                 agent_identity.runtime_artifact_hash.bytes.data()),
                             agent_identity.runtime_artifact_hash.bytes.size()),
      &hasher);
  UpdateBytesFrame(
      'P', absl::string_view(reinterpret_cast<const char*>(
                                 agent_identity.inference_profile_hash.bytes.data()),
                             agent_identity.inference_profile_hash.bytes.size()),
      &hasher);
  constexpr char kLimitTag = 'L';
  hasher.Update(absl::string_view(&kLimitTag, 1));
  UpdateU32(max_decision_tokens, &hasher);
  UpdateBytesFrame('I', canonical_agent_input, &hasher);
  return hasher.Finalize();
}

absl::Status ValidateSnapshotShape(const DPMLogSnapshot& snapshot) {
  if (snapshot.log_id.empty() || snapshot.case_id.empty() ||
      snapshot.log_id.size() > kMaximumDPMProjectionIdentityBytes ||
      snapshot.case_id.size() > kMaximumDPMProjectionIdentityBytes) {
    return absl::DataLossError(
        "DPM log snapshot has missing or over-bound log/case identity.");
  }
  if (!IsValidUtf8(snapshot.log_id) || !IsValidUtf8(snapshot.case_id) ||
      ContainsControlByte(snapshot.log_id) ||
      ContainsControlByte(snapshot.case_id)) {
    return absl::DataLossError(
        "DPM log snapshot identities must be UTF-8 without control bytes.");
  }
  if (snapshot.generation != snapshot.events.size()) {
    return absl::DataLossError(
        "DPM log snapshot generation does not match its event count.");
  }
  if (IsZeroHash(snapshot.prefix_hash)) {
    return absl::DataLossError(
        "DPM log snapshot has no cryptographic prefix hash.");
  }
  if (snapshot.prefix_hashes.size() != snapshot.events.size() + 1 ||
      snapshot.prefix_hashes.empty() ||
      snapshot.prefix_hashes.back() != snapshot.prefix_hash) {
    return absl::DataLossError(
        "DPM log snapshot has no complete immutable prefix-proof index.");
  }
  for (const Hash256& prefix : snapshot.prefix_hashes) {
    if (IsZeroHash(prefix)) {
      return absl::DataLossError(
          "DPM log snapshot prefix-proof index contains an empty digest.");
    }
  }
  for (uint64_t i = 0; i < snapshot.events.size(); ++i) {
    if (snapshot.events[i].index != i ||
        snapshot.events[i].case_id != snapshot.case_id) {
      return absl::DataLossError(
          "DPM log snapshot contains non-contiguous or cross-case events.");
    }
    if (snapshot.events[i].operation_id.empty() ||
        snapshot.events[i].operation_id.size() >
            kMaximumDPMEventOperationIdBytes ||
        snapshot.events[i].payload.size() > kMaximumDPMEventPayloadBytes ||
        ContainsControlByte(snapshot.events[i].operation_id) ||
        !IsValidUtf8(snapshot.events[i].operation_id) ||
        !IsValidUtf8(snapshot.events[i].case_id) ||
        !IsValidUtf8(snapshot.events[i].payload)) {
      return absl::DataLossError(
          "DPM log snapshot contains invalid or over-bound event fields.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<Hash256> SnapshotPrefixHashAt(
    const DPMLogSnapshot& snapshot, uint64_t event_count) {
  if (event_count > snapshot.generation ||
      snapshot.prefix_hashes.size() != snapshot.events.size() + 1 ||
      snapshot.prefix_hashes.empty() ||
      snapshot.prefix_hashes.back() != snapshot.prefix_hash) {
    return absl::DataLossError(
        "DPM snapshot cannot authenticate the requested raw-log prefix.");
  }
  const Hash256 prefix =
      snapshot.prefix_hashes[static_cast<size_t>(event_count)];
  if (IsZeroHash(prefix)) {
    return absl::DataLossError(
        "DPM snapshot contains an empty raw-log prefix proof.");
  }
  return prefix;
}

bool IsDecisionInputKind(DPMEvent::Kind kind) {
  return kind == DPMEvent::Kind::kUser || kind == DPMEvent::Kind::kTool ||
         kind == DPMEvent::Kind::kInternal;
}

absl::StatusOr<Hash256> ComputeReceiptAgentReplayRequestHash(
    const DPMLogSnapshot& snapshot, const DPMEvent& response,
    const DPMTurnReceipt& receipt);

absl::Status ValidateInputEvent(const DPMEvent& event,
                                const DPMTurnRequest& request) {
  if (event.kind != request.kind || event.operation_id != request.operation_id ||
      event.case_id != request.case_id || event.payload != request.payload) {
    return absl::AlreadyExistsError(
        "DPM operation id is already bound to different input bytes.");
  }
  if (request.timestamp_us.has_value() &&
      event.timestamp_us != *request.timestamp_us) {
    return absl::AlreadyExistsError(
        "DPM operation id is already bound to a different timestamp.");
  }
  if (event.turn_receipt.has_value()) {
    return absl::DataLossError(
        "DPM input event unexpectedly contains a model-turn receipt.");
  }
  if (event.timestamp_us <= 0) {
    return absl::DataLossError(
        "DPM input event has a non-positive timestamp.");
  }
  return absl::OkStatus();
}

absl::Status ValidateReceiptAndAdvance(const DPMLogSnapshot& snapshot,
                                       const DPMEvent& response,
                                       TranscriptCursor* transcript) {
  if (response.kind != DPMEvent::Kind::kModelTurn ||
      !response.turn_receipt.has_value()) {
    return absl::DataLossError(
        "DPM model event is missing its authoritative turn receipt.");
  }
  const DPMTurnReceipt& receipt = *response.turn_receipt;
  if (receipt.format_version != DPMTurnReceipt::kLegacyFormatVersion &&
      receipt.format_version != DPMTurnReceipt::kFormatVersion) {
    return absl::FailedPreconditionError(
        "Unsupported DPM turn receipt version.");
  }
  if (receipt.operation_id.empty() ||
      receipt.operation_id != response.operation_id ||
      receipt.response_event_index != response.index ||
      receipt.decision_output != response.payload) {
    return absl::DataLossError(
        "DPM model event and turn receipt disagree.");
  }
  if (receipt.input_event_index >= response.index ||
      receipt.input_event_index >= snapshot.events.size()) {
    return absl::DataLossError(
        "DPM turn receipt names an invalid input event.");
  }
  const DPMEvent& input = snapshot.events[receipt.input_event_index];
  if (!IsDecisionInputKind(input.kind) ||
      input.operation_id != receipt.operation_id || input.turn_receipt ||
      input.case_id != response.case_id ||
      receipt.response_event_index != receipt.input_event_index + 1 ||
      response.timestamp_us < input.timestamp_us) {
    return absl::DataLossError(
        "DPM turn receipt is not bound to its authoritative input event.");
  }
  if (receipt.canonical_agent_input.empty() ||
      receipt.canonical_agent_input.size() >
          kMaximumDPMCanonicalAgentInputBytes ||
      receipt.projected_memory.empty() ||
      receipt.projected_memory.size() > kMaximumDPMEventPayloadBytes ||
      receipt.decision_token_ids.empty() ||
      IsZeroHash(receipt.agent_session_identity.model_artifact_hash) ||
      IsZeroHash(receipt.agent_session_identity.runtime_artifact_hash) ||
      IsZeroHash(receipt.agent_session_identity.inference_profile_hash) ||
      receipt.max_decision_tokens == 0 ||
      receipt.max_decision_tokens > kMaximumDPMGenerationTokens ||
      IsZeroHash(receipt.agent_request_hash) ||
      IsZeroHash(receipt.agent_transcript_hash)) {
    return absl::DataLossError(
        "DPM turn receipt is missing required reconstruction data.");
  }
  if (!IsValidUtf8(receipt.operation_id) ||
      !IsValidUtf8(receipt.projected_memory) ||
      !IsValidUtf8(receipt.canonical_agent_input) ||
      !IsValidUtf8(receipt.decision_output)) {
    return absl::DataLossError(
        "DPM turn receipt contains non-UTF-8 reconstruction fields.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(
      receipt.projection_manifest));
  if (receipt.projection_manifest.log_id != snapshot.log_id ||
      receipt.projection_manifest.case_id != snapshot.case_id ||
      receipt.projection_manifest.source_event_count != response.index ||
      receipt.projection_manifest.input_event_index !=
          receipt.input_event_index ||
      ComputeCanonicalDPMProjectionOutputHash(receipt.projected_memory) !=
          receipt.projection_manifest.output_hash) {
    return absl::DataLossError(
        "DPM turn receipt projection manifest is not bound to its raw-log "
        "source and projected-memory bytes.");
  }
  if (receipt.projection_manifest.baseline_manifest_hash.has_value()) {
    const uint64_t baseline_response_index =
        receipt.projection_manifest.event_range_start;
    if (baseline_response_index >= response.index ||
        baseline_response_index >= snapshot.events.size()) {
      return absl::DataLossError(
          "DPM projection baseline does not name a prior response event.");
    }
    const DPMEvent& baseline_response =
        snapshot.events[baseline_response_index];
    if (baseline_response.kind != DPMEvent::Kind::kModelTurn ||
        !baseline_response.turn_receipt.has_value()) {
      return absl::DataLossError(
          "DPM projection baseline does not name an authoritative receipt.");
    }
    const DPMProjectionManifest& baseline =
        baseline_response.turn_receipt->projection_manifest;
    if (baseline.source_event_count != baseline_response_index ||
        baseline.manifest_hash !=
            *receipt.projection_manifest.baseline_manifest_hash ||
        baseline.output_hash !=
            *receipt.projection_manifest.baseline_output_hash ||
        baseline.correction_digest !=
            receipt.projection_manifest.correction_digest ||
        baseline.config_hash != receipt.projection_manifest.config_hash ||
        baseline.runtime_identity !=
            receipt.projection_manifest.runtime_identity ||
        baseline.replay_mode != receipt.projection_manifest.replay_mode ||
        baseline.exact_profile_id !=
            receipt.projection_manifest.exact_profile_id) {
      return absl::DataLossError(
          "DPM projection baseline hashes do not identify the compatible "
          "prior authoritative receipt.");
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMTurnReceiptReplayEvidence(receipt));
  ABSL_ASSIGN_OR_RETURN(
      const Hash256 expected_agent_replay_request_hash,
      ComputeReceiptAgentReplayRequestHash(snapshot, response, receipt));
  if (expected_agent_replay_request_hash !=
      receipt.agent_replay_request_hash) {
    return absl::DataLossError(
        "DPM agent replay request hash is not derived from the "
        "authoritative full correction-epoch transcript.");
  }
  if (receipt.decision_token_ids.size() > receipt.max_decision_tokens) {
    return absl::DataLossError(
        "DPM turn receipt exceeds its committed decision-token limit.");
  }
  for (int token_id : receipt.decision_token_ids) {
    if (token_id < 0 ||
        static_cast<int64_t>(token_id) >
            std::numeric_limits<int32_t>::max()) {
      return absl::DataLossError(
          "DPM turn receipt contains a non-canonical decision token id.");
    }
  }

  if (ComputeLogicalAgentRequestHash(
          *transcript, receipt.canonical_agent_input,
          receipt.max_decision_tokens,
          receipt.projection_manifest.correction_digest,
          receipt.agent_session_identity) != receipt.agent_request_hash) {
    return absl::DataLossError(
        "DPM turn receipt agent request hash is not canonical.");
  }

  if (!transcript->correction_digest.has_value() ||
      *transcript->correction_digest !=
          receipt.projection_manifest.correction_digest) {
    ResetTranscriptHasher(receipt.projection_manifest.correction_digest,
                          transcript);
  }
  UpdateBytesFrame('I', receipt.canonical_agent_input, &transcript->hasher);
  UpdateTokenFrame(receipt.decision_token_ids, &transcript->hasher);
  UpdateBytesFrame('O', receipt.decision_output, &transcript->hasher);
  if (SnapshotDigest(transcript->hasher) != receipt.agent_transcript_hash) {
    return absl::DataLossError(
        "DPM turn receipt transcript hash chain is invalid.");
  }
  return absl::OkStatus();
}

struct OperationRecord {
  const DPMEvent* input = nullptr;
  const DPMEvent* response = nullptr;
  const DPMEvent* correction = nullptr;
  std::optional<Hash256> correction_digest_after;
};

struct AuthoritativeLogIndex {
  std::unordered_map<std::string, OperationRecord> operations;
  const DPMEvent* pending_input = nullptr;
  Hash256 correction_digest;
  std::optional<SessionHandoffIdentity> agent_identity;
  TranscriptCursor transcript;
};

absl::StatusOr<AuthoritativeLogIndex> ValidateAndIndexAuthoritativeLog(
    const DPMLogSnapshot& snapshot) {
  ABSL_RETURN_IF_ERROR(ValidateSnapshotShape(snapshot));
  AuthoritativeLogIndex index;
  ABSL_ASSIGN_OR_RETURN(
      index.correction_digest,
      InitialDPMCorrectionDigest(snapshot.log_id, snapshot.case_id));
  for (const DPMEvent& event : snapshot.events) {
    if (event.operation_id.empty() || event.timestamp_us <= 0) {
      return absl::DataLossError(
          "Authoritative DPM event has no operation id or timestamp.");
    }
    OperationRecord& operation = index.operations[event.operation_id];
    if (event.kind == DPMEvent::Kind::kCorrection) {
      if (event.payload.empty() || event.turn_receipt.has_value() ||
          operation.input != nullptr || operation.response != nullptr ||
          operation.correction != nullptr) {
        return absl::DataLossError(
            "Authoritative DPM correction operation is malformed or "
            "duplicated.");
      }
      operation.correction = &event;
      ABSL_ASSIGN_OR_RETURN(
          index.correction_digest,
          AdvanceDPMCorrectionDigest(index.correction_digest, event));
      operation.correction_digest_after = index.correction_digest;
      continue;
    }
    if (IsDecisionInputKind(event.kind)) {
      if (event.payload.empty() || event.turn_receipt.has_value() ||
          operation.input != nullptr || operation.response != nullptr ||
          operation.correction != nullptr) {
        return absl::DataLossError(
            "Authoritative DPM turn input is malformed or duplicated.");
      }
      operation.input = &event;
      continue;
    }
    if (event.kind != DPMEvent::Kind::kModelTurn) {
      return absl::DataLossError("Authoritative DPM event has unknown kind.");
    }
    if (operation.correction != nullptr || operation.response != nullptr) {
      return absl::DataLossError(
          "Authoritative DPM operation has duplicate response identity.");
    }
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &index.transcript));
    const DPMTurnReceipt& receipt = *event.turn_receipt;
    if (receipt.projection_manifest.correction_digest !=
        index.correction_digest) {
      return absl::DataLossError(
          "DPM turn receipt correction digest is not derived from the raw "
          "log.");
    }
    if (!index.agent_identity.has_value()) {
      index.agent_identity = receipt.agent_session_identity;
    } else if (*index.agent_identity != receipt.agent_session_identity) {
      return absl::DataLossError(
          "A DPM log cannot mix agent model/runtime/profile identities.");
    }
    if (operation.input == nullptr ||
        operation.input->index != receipt.input_event_index ||
        event.index != operation.input->index + 1) {
      return absl::DataLossError(
          "DPM model response does not immediately follow its input.");
    }
    operation.response = &event;
  }

  for (const auto& entry : index.operations) {
    const OperationRecord& operation = entry.second;
    if (operation.correction != nullptr) continue;
    if (operation.input == nullptr) {
      return absl::DataLossError(
          "DPM operation contains a response without an input.");
    }
    if (operation.response != nullptr) continue;
    if (index.pending_input != nullptr ||
        operation.input->index + 1 != snapshot.events.size()) {
      return absl::DataLossError(
          "DPM log contains an interleaved or multiple incomplete turn.");
    }
    index.pending_input = operation.input;
  }
  return index;
}

DPMAgentGenerationRequest::PrefillChunk TextChunk(
    absl::string_view value) {
  DPMAgentGenerationRequest::PrefillChunk chunk;
  chunk.encoding =
      DPMAgentGenerationRequest::PrefillChunk::Encoding::kUtf8Text;
  chunk.text.assign(value.data(), value.size());
  return chunk;
}

DPMAgentGenerationRequest::PrefillChunk TokenChunk(
    const std::vector<int>& value) {
  DPMAgentGenerationRequest::PrefillChunk chunk;
  chunk.encoding =
      DPMAgentGenerationRequest::PrefillChunk::Encoding::kTokenIds;
  chunk.token_ids = value;
  return chunk;
}

absl::StatusOr<Hash256> ComputeAgentReplayRequestHash(
    const DPMAgentExecutionRequest& logical_request) {
  ABSL_ASSIGN_OR_RETURN(std::string encoded,
                        EncodeDPMAgentExecutionRequest(logical_request));
  DPMCanonicalReplayRequest replay_request{
      .stage = DPMReplayStage::kAgentDecision,
      .max_output_tokens = logical_request.max_output_tokens,
      .request_contract_version = std::string(kDPMAgentReplayContractVersion),
      .canonical_payload = std::move(encoded),
  };
  return ComputeDPMCanonicalReplayRequestHash(replay_request);
}

DPMCheckpointWorkerPrefillMode ToCheckpointWorkerPrefillMode(
    FreshWorkerPrefillMode mode) {
  switch (mode) {
    case FreshWorkerPrefillMode::kFullCanonicalPrefill:
      return DPMCheckpointWorkerPrefillMode::kFullCanonicalPrefill;
    case FreshWorkerPrefillMode::kOwnPositionCapsuleDelta:
      return DPMCheckpointWorkerPrefillMode::kOwnPositionCapsuleDelta;
  }
  return DPMCheckpointWorkerPrefillMode::kNone;
}

bool ExactWorkerCheckpointProvenanceEqual(
    const DPMExactWorkerCheckpointProvenance& left,
    const DPMExactWorkerCheckpointProvenance& right) {
  return left.run_index == right.run_index &&
         left.execution_plan_hash == right.execution_plan_hash &&
         left.request_envelope_hash == right.request_envelope_hash &&
         left.result_envelope_hash == right.result_envelope_hash &&
         left.transient_envelope_size == right.transient_envelope_size &&
         left.transient_envelope_hash == right.transient_envelope_hash &&
         left.output_evidence_hash == right.output_evidence_hash;
}

absl::StatusOr<Hash256> ComputeReceiptAgentReplayRequestHash(
    const DPMLogSnapshot& snapshot, const DPMEvent& response,
    const DPMTurnReceipt& receipt) {
  if (response.index >= snapshot.events.size() ||
      &snapshot.events[response.index] != &response) {
    return absl::DataLossError(
        "DPM agent replay receipt is not located at its response index.");
  }
  std::vector<DPMAgentGenerationRequest::PrefillChunk> full_chunks;
  for (const DPMEvent& prior : snapshot.events) {
    if (prior.index >= response.index) break;
    if (prior.kind != DPMEvent::Kind::kModelTurn ||
        !prior.turn_receipt.has_value()) {
      continue;
    }
    const DPMTurnReceipt& prior_receipt = *prior.turn_receipt;
    if (prior_receipt.projection_manifest.correction_digest !=
        receipt.projection_manifest.correction_digest) {
      full_chunks.clear();
      continue;
    }
    full_chunks.push_back(TextChunk(prior_receipt.canonical_agent_input));
    full_chunks.push_back(TokenChunk(prior_receipt.decision_token_ids));
  }
  full_chunks.push_back(TextChunk(receipt.canonical_agent_input));
  DPMAgentExecutionRequest logical_request{
      .logical_agent_request_hash = receipt.agent_request_hash,
      .correction_digest = receipt.projection_manifest.correction_digest,
      .max_output_tokens = receipt.max_decision_tokens,
      .full_canonical_prefill_chunks = std::move(full_chunks),
  };
  return ComputeAgentReplayRequestHash(logical_request);
}

void AppendTextSection(absl::string_view name, absl::string_view bytes,
                       std::string* output) {
  absl::StrAppend(output, name, " ", bytes.size(), "\n");
  output->append(bytes.data(), bytes.size());
  output->push_back('\n');
}

absl::Status ValidateProjectionOutcome(
    const DPMProjectionOutcome& projection,
    const DPMProjectionRequest& request) {
  if (projection.projected_memory.empty() ||
      projection.projected_memory.size() > kMaximumDPMEventPayloadBytes ||
      !IsValidUtf8(projection.projected_memory)) {
    return absl::FailedPreconditionError(
        "DPM projection provider returned empty, over-bound, or non-UTF-8 "
        "projected memory.");
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMProjectionManifest(projection.manifest));
  if (projection.manifest.log_id != request.log.log_id ||
      projection.manifest.case_id != request.log.case_id ||
      projection.manifest.case_id != request.case_id ||
      projection.manifest.source_event_count != request.log.events.size() ||
      projection.manifest.source_prefix_hash != request.log.prefix_hash ||
      projection.manifest.input_event_index != request.input_event_index ||
      ComputeCanonicalDPMProjectionOutputHash(projection.projected_memory) !=
          projection.manifest.output_hash) {
    return absl::FailedPreconditionError(
        "DPM projection provider returned an identity not bound to the "
        "requested raw-log prefix and projected-memory bytes.");
  }
  return absl::OkStatus();
}

absl::Status ValidateAgentOutcome(
    const DPMAgentGenerationOutcome& outcome, uint32_t max_decision_tokens) {
  if (outcome.decision_output.size() > kMaximumDPMEventPayloadBytes ||
      !IsValidUtf8(outcome.decision_output)) {
    return absl::DataLossError(
        "DPM agent returned over-bound or non-UTF-8 decision bytes.");
  }
  if (outcome.decision_token_ids.empty()) {
    return absl::FailedPreconditionError(
        "DPM agent did not return the exact decision token ids.");
  }
  if (outcome.decision_token_ids.size() > max_decision_tokens) {
    return absl::FailedPreconditionError(
        "DPM agent exceeded the authoritative decision-token limit.");
  }
  for (int token_id : outcome.decision_token_ids) {
    if (token_id < 0 ||
        static_cast<int64_t>(token_id) >
            std::numeric_limits<int32_t>::max()) {
      return absl::DataLossError(
          "DPM agent returned a non-canonical decision token id.");
    }
  }
  return absl::OkStatus();
}

uint64_t CountCommittedTurns(const DPMLogSnapshot& snapshot) {
  uint64_t count = 0;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind == DPMEvent::Kind::kModelTurn && event.turn_receipt) {
      ++count;
    }
  }
  return count;
}

}  // namespace

int64_t SystemDPMClock::NowMicros() const {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

DPMEngine::DPMEngine(DPMEventLog* log,
                     DPMProjectionProvider* projection_provider,
                     DPMAgentReplayRuntime* agent_runtime,
                     DPMSessionCheckpointRepository* checkpoint_repository,
                     DPMEngineConfig config, DPMClock* clock)
    : log_(log),
      projection_provider_(projection_provider),
      agent_runtime_(agent_runtime),
      checkpoint_repository_(checkpoint_repository),
      config_(std::move(config)),
      clock_(clock == nullptr ? &system_clock_ : clock) {}

absl::Status DPMEngine::ValidateConfiguration() const {
  if (log_ == nullptr || projection_provider_ == nullptr ||
      agent_runtime_ == nullptr || clock_ == nullptr) {
    return absl::InvalidArgumentError(
        "DPMEngine requires log, projection, agent, and clock dependencies.");
  }
  if (config_.max_decision_tokens <= 0 ||
      static_cast<uint64_t>(config_.max_decision_tokens) >
          kMaximumDPMGenerationTokens) {
    return absl::InvalidArgumentError(
        "DPM decision token limit must be between 1 and 65,536.");
  }
  ABSL_RETURN_IF_ERROR(projection_provider_->ValidateSupport());
  ABSL_RETURN_IF_ERROR(agent_runtime_->ValidateSupport());
  ABSL_RETURN_IF_ERROR(agent_runtime_->ValidateGenerationLimit(
      static_cast<uint32_t>(config_.max_decision_tokens)));
  const DPMReplayMode agent_replay_mode = agent_runtime_->GetReplayMode();
  ABSL_RETURN_IF_ERROR(ValidateDPMReplayMode(agent_replay_mode));
  ABSL_ASSIGN_OR_RETURN(const std::optional<Hash256> exact_profile_id,
                        agent_runtime_->GetExactProfileId());
  if ((agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay &&
       exact_profile_id.has_value()) ||
      (agent_replay_mode == DPMReplayMode::kExactRegeneration &&
       (!exact_profile_id.has_value() || IsZeroHash(*exact_profile_id)))) {
    return absl::FailedPreconditionError(
        "DPM agent replay mode and derived exact profile disagree.");
  }
  const SessionHandoffIdentity& identity =
      agent_runtime_->GetSessionHandoffIdentity();
  if (IsZeroHash(identity.model_artifact_hash) ||
      IsZeroHash(identity.runtime_artifact_hash) ||
      IsZeroHash(identity.inference_profile_hash)) {
    return absl::FailedPreconditionError(
        "DPM agent runtime has not resolved its loaded session identity.");
  }
  const bool uses_session_checkpoints =
      config_.restore_session_checkpoints ||
      config_.checkpoint_interval_turns != 0;
  if (!uses_session_checkpoints) {
    return absl::OkStatus();
  }
  if (checkpoint_repository_ == nullptr) {
    return absl::InvalidArgumentError(
        "Enabled DPM checkpoint restore/capture requires a repository.");
  }
  if (config_.checkpoint_key_id.empty() ||
      config_.checkpoint_key_id.size() > kMaximumCheckpointKeyIdSize) {
    return absl::InvalidArgumentError(
        "DPM checkpoint key id must contain 1 to 1024 bytes.");
  }
  if (config_.checkpoint_authentication_key.size() < 32) {
    return absl::InvalidArgumentError(
        "DPM checkpoint authentication key must contain at least 32 bytes.");
  }
  ABSL_RETURN_IF_ERROR(agent_runtime_->ValidateSessionHandoffSupport());
  if (agent_replay_mode == DPMReplayMode::kExactRegeneration) {
    ABSL_ASSIGN_OR_RETURN(
        const std::optional<Hash256> capsule_restore_admission_id,
        agent_runtime_->GetCapsuleRestoreAdmissionRecordId());
    ABSL_ASSIGN_OR_RETURN(
        const std::optional<Hash256> session_handoff_capability_id,
        agent_runtime_->GetSessionHandoffCapabilityId());
    if (!capsule_restore_admission_id.has_value() ||
        IsZeroHash(*capsule_restore_admission_id) ||
        !session_handoff_capability_id.has_value() ||
        IsZeroHash(*session_handoff_capability_id)) {
      return absl::FailedPreconditionError(
          "Exact DPM checkpoints require current CapsuleRestore admission "
          "and an Engine-derived session-handoff capability.");
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<DPMTurnResult>>
DPMEngine::RecoverCommittedTurn(const DPMLogSnapshot& snapshot,
                                const DPMTurnRequest& request) const {
  const DPMEvent* committed = nullptr;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kModelTurn || !event.turn_receipt ||
        event.turn_receipt->operation_id != request.operation_id) {
      continue;
    }
    if (committed != nullptr) {
      return absl::DataLossError(
          "DPM log contains multiple committed turns for one operation id.");
    }
    committed = &event;
  }
  if (committed == nullptr) {
    return std::optional<DPMTurnResult>();
  }

  TranscriptCursor transcript;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &transcript));
    if (event.index == committed->index) break;
  }

  const DPMTurnReceipt& receipt = *committed->turn_receipt;
  ABSL_RETURN_IF_ERROR(
      ValidateInputEvent(snapshot.events[receipt.input_event_index], request));
  if (request.response_timestamp_us.has_value() &&
      committed->timestamp_us != *request.response_timestamp_us) {
    return absl::AlreadyExistsError(
        "DPM operation id is already bound to a different response "
        "timestamp.");
  }
  DPMTurnResult result;
  result.projected_memory = receipt.projected_memory;
  result.decision_output = receipt.decision_output;
  result.decision_token_ids = receipt.decision_token_ids;
  result.input_event_index = receipt.input_event_index;
  result.response_event_index = receipt.response_event_index;
  result.projection_manifest_hash =
      receipt.projection_manifest.manifest_hash;
  result.projection_replay_mode = receipt.projection_manifest.replay_mode;
  result.projection_execution_evidence_hash =
      receipt.projection_manifest.execution_evidence_hash;
  result.projection_exact_profile_id =
      receipt.projection_manifest.exact_profile_id;
  result.projection_exact_profile_admission_record_id =
      receipt.projection_manifest.exact_profile_admission_record_id;
  result.agent_replay_mode = receipt.agent_replay_mode;
  result.agent_execution_evidence_hash =
      receipt.agent_execution_evidence_hash;
  result.agent_exact_profile_id = receipt.agent_exact_profile_id;
  result.agent_exact_profile_admission_record_id =
      receipt.agent_exact_profile_admission_record_id;
  result.agent_exact_output_evidence_hash =
      receipt.agent_exact_output_evidence_hash;
  result.agent_exact_logit_frame_count =
      receipt.agent_exact_logit_frame_count;
  result.agent_reused_canonical_winner =
      receipt.agent_reused_canonical_winner;
  result.agent_producing_session_matched_output =
      receipt.agent_producing_session_matched_output;
  result.agent_worker_capsule_matched_output =
      receipt.checkpoint_capture_origin ==
          DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker &&
      receipt.agent_exact_worker_checkpoint_provenance.has_value();
  result.session_checkpoint_id = receipt.session_checkpoint_id;
  result.restored_from_session_checkpoint_id =
      receipt.restored_from_session_checkpoint_id;
  result.checkpoint_capture_origin = receipt.checkpoint_capture_origin;
  result.agent_worker_prefill_mode = receipt.agent_worker_prefill_mode;
  result.agent_physical_execution_plan_hash =
      receipt.agent_physical_execution_plan_hash;
  result.agent_capsule_restore_admission_record_id =
      receipt.agent_capsule_restore_admission_record_id;
  result.agent_exact_worker_checkpoint_provenance =
      receipt.agent_exact_worker_checkpoint_provenance;
  result.restored_session_checkpoint =
      receipt.restored_from_session_checkpoint_id.has_value();
  result.recovered_committed_turn = true;
  return std::optional<DPMTurnResult>(std::move(result));
}

absl::Status DPMEngine::ValidateRestoreCandidate(
    const DPMLogSnapshot& current,
    const DPMProjectionOutcome& projection,
    const RestoreCandidate& candidate) const {
  ABSL_RETURN_IF_ERROR(
      ValidateDPMSessionCheckpointArtifact(candidate.artifact));
  const DPMSessionCheckpointDescriptor& descriptor =
      candidate.artifact.descriptor;
  const DPMTurnReceipt& receipt = candidate.receipt;

  const bool legacy_receipt =
      receipt.format_version == DPMTurnReceipt::kLegacyFormatVersion;
  const bool legacy_descriptor =
      descriptor.format_version ==
      DPMSessionCheckpointDescriptor::kLegacyFormatVersion;
  if (legacy_receipt != legacy_descriptor) {
    return absl::FailedPreconditionError(
        "DPM checkpoint receipt and descriptor provenance versions differ.");
  }

  if (!receipt.session_checkpoint_id ||
      *receipt.session_checkpoint_id != descriptor.descriptor_id ||
      descriptor.log_id != current.log_id ||
      descriptor.log_id != receipt.projection_manifest.log_id ||
      descriptor.stage != DPMSessionCheckpointStage::kAgentDecision ||
      descriptor.response_event_index != receipt.response_event_index ||
      descriptor.source_event_count != receipt.response_event_index ||
      descriptor.source_event_count !=
          receipt.projection_manifest.source_event_count ||
      descriptor.source_prefix_hash !=
          receipt.projection_manifest.source_prefix_hash ||
      descriptor.response_event_index >= current.events.size() ||
      descriptor.response_event_index >=
          projection.manifest.input_event_index) {
    return absl::FailedPreconditionError(
        "DPM checkpoint descriptor is not attached to this log receipt.");
  }
  const DPMEvent& response = current.events[descriptor.response_event_index];
  if (!response.turn_receipt ||
      response.turn_receipt->session_checkpoint_id !=
          receipt.session_checkpoint_id ||
      response.turn_receipt->operation_id != receipt.operation_id) {
    return absl::FailedPreconditionError(
        "DPM checkpoint receipt is no longer authoritative.");
  }
  if (descriptor.replay_mode != receipt.agent_replay_mode) {
    return absl::FailedPreconditionError(
        "DPM checkpoint replay mode differs from its authoritative receipt.");
  }
  if (receipt.format_version == DPMTurnReceipt::kFormatVersion) {
    const bool matching_worker_provenance =
        descriptor.worker_provenance.has_value() ==
            receipt.agent_exact_worker_checkpoint_provenance.has_value() &&
        (!descriptor.worker_provenance.has_value() ||
         ExactWorkerCheckpointProvenanceEqual(
             *descriptor.worker_provenance,
             *receipt.agent_exact_worker_checkpoint_provenance));
    if (descriptor.capture_origin != receipt.checkpoint_capture_origin ||
        descriptor.restored_from_checkpoint_id !=
            receipt.restored_from_session_checkpoint_id ||
        descriptor.worker_prefill_mode !=
            receipt.agent_worker_prefill_mode ||
        descriptor.execution_plan_hash !=
            receipt.agent_physical_execution_plan_hash ||
        descriptor.exact_profile_id != receipt.agent_exact_profile_id ||
        descriptor.exact_profile_admission_record_id !=
            receipt.agent_exact_profile_admission_record_id.value_or(
                Hash256{}) ||
        descriptor.capsule_restore_admission_record_id !=
            receipt.agent_capsule_restore_admission_record_id.value_or(
                Hash256{}) ||
        descriptor.exact_request_execution_evidence_id !=
            (receipt.agent_replay_mode == DPMReplayMode::kExactRegeneration
                 ? receipt.agent_execution_evidence_hash
                 : Hash256{}) ||
        descriptor.exact_output_evidence_hash !=
            receipt.agent_exact_output_evidence_hash.value_or(Hash256{}) ||
        !matching_worker_provenance) {
      return absl::FailedPreconditionError(
          "DPM checkpoint descriptor and version 4 receipt provenance "
          "disagree.");
    }
  }
  if (descriptor.projection_request_hash !=
          receipt.projection_manifest.request_hash ||
      descriptor.projection_manifest_hash !=
          receipt.projection_manifest.manifest_hash ||
      descriptor.correction_digest !=
          receipt.projection_manifest.correction_digest ||
      descriptor.agent_request_hash != receipt.agent_request_hash ||
      descriptor.agent_transcript_hash != receipt.agent_transcript_hash ||
      descriptor.session_identity != receipt.agent_session_identity ||
      descriptor.correction_digest !=
          projection.manifest.correction_digest ||
      !(descriptor.session_identity ==
        agent_runtime_->GetSessionHandoffIdentity()) ||
      descriptor.key_id != config_.checkpoint_key_id) {
    return absl::FailedPreconditionError(
        "DPM checkpoint artifact is incompatible with the active turn.");
  }
  if (descriptor.replay_mode == DPMReplayMode::kExactRegeneration) {
    ABSL_ASSIGN_OR_RETURN(const std::optional<Hash256> current_profile_id,
                          agent_runtime_->GetExactProfileId());
    ABSL_ASSIGN_OR_RETURN(
        const std::optional<Hash256> current_profile_admission_id,
        agent_runtime_->GetExactProfileAdmissionRecordId());
    ABSL_ASSIGN_OR_RETURN(
        const std::optional<Hash256> current_capsule_restore_admission_id,
        agent_runtime_->GetCapsuleRestoreAdmissionRecordId());
    if (!current_profile_id.has_value() ||
        !current_profile_admission_id.has_value() ||
        !current_capsule_restore_admission_id.has_value() ||
        descriptor.exact_profile_id != current_profile_id ||
        descriptor.exact_profile_admission_record_id !=
            *current_profile_admission_id ||
        descriptor.capsule_restore_admission_record_id !=
            *current_capsule_restore_admission_id) {
      return absl::FailedPreconditionError(
          "DPM exact checkpoint does not match the currently derived and "
          "admitted exact profile and CapsuleRestore capability.");
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      Hash256 prefix_hash,
      SnapshotPrefixHashAt(current, descriptor.source_event_count));
  if (prefix_hash != descriptor.source_prefix_hash) {
    return absl::FailedPreconditionError(
        "DPM checkpoint raw-log prefix has changed.");
  }

  TranscriptCursor transcript;
  bool found = false;
  Hash256 candidate_transcript_hash;
  for (const DPMEvent& event : current.events) {
    if (found && event.kind == DPMEvent::Kind::kCorrection) {
      // Corrections are immutable epoch boundaries. Even a hypothetical hash
      // collision must not permit a descendant capsule to cross one.
      return absl::FailedPreconditionError(
          "DPM correction event invalidates the selected checkpoint "
          "descendant interval.");
    }
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(current, event, &transcript));
    if (event.index == descriptor.response_event_index) {
      found = true;
      candidate_transcript_hash = SnapshotDigest(transcript.hasher);
      continue;
    }
    if (found && event.turn_receipt->projection_manifest.correction_digest !=
                     projection.manifest.correction_digest) {
      return absl::FailedPreconditionError(
          "DPM correction epoch changed after the checkpoint.");
    }
  }
  if (!found || candidate_transcript_hash != descriptor.agent_transcript_hash) {
    return absl::FailedPreconditionError(
        "DPM checkpoint transcript prefix is not present in this log.");
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<DPMEngine::RestoreCandidate>>
DPMEngine::FindRestoreCandidate(
    const DPMLogSnapshot& current,
    const DPMProjectionOutcome& projection) const {
  if (!config_.restore_session_checkpoints ||
      checkpoint_repository_ == nullptr) {
    return std::optional<RestoreCandidate>();
  }
  for (auto it = current.events.rbegin(); it != current.events.rend(); ++it) {
    if (it->kind == DPMEvent::Kind::kCorrection) {
      // The newest correction is the immutable lower bound of the active
      // checkpoint interval. Every older capsule belongs to an invalidated
      // lineage, so do not scan or touch its repository objects.
      break;
    }
    if (it->kind != DPMEvent::Kind::kModelTurn || !it->turn_receipt ||
        !it->turn_receipt->session_checkpoint_id) {
      continue;
    }
    const DPMTurnReceipt& receipt = *it->turn_receipt;
    if (receipt.response_event_index >=
            projection.manifest.input_event_index ||
        receipt.agent_replay_mode != agent_runtime_->GetReplayMode() ||
        receipt.projection_manifest.correction_digest !=
            projection.manifest.correction_digest ||
        receipt.agent_session_identity !=
            agent_runtime_->GetSessionHandoffIdentity()) {
      // Receipt metadata is immutable and authoritative. Filter invalidated
      // epochs and incompatible runtime branches before touching disposable
      // checkpoint storage; an older compatible interval may still exist.
      continue;
    }
    absl::StatusOr<DPMSessionCheckpointArtifact> artifact =
        checkpoint_repository_->Get(*receipt.session_checkpoint_id);
    if (!artifact.ok()) {
      // Checkpoints are authenticated disposable accelerators. Missing or
      // unavailable storage must not prevent reconstruction from the log.
      continue;
    }
    RestoreCandidate candidate{receipt, std::move(*artifact)};
    if (ValidateRestoreCandidate(current, projection, candidate).ok()) {
      return std::optional<RestoreCandidate>(std::move(candidate));
    }
  }
  return std::optional<RestoreCandidate>();
}

absl::StatusOr<
    std::vector<DPMAgentGenerationRequest::PrefillChunk>>
DPMEngine::BuildFullTranscriptChunks(
    const DPMLogSnapshot& snapshot,
    absl::string_view current_agent_input,
    const Hash256& current_correction_digest) const {
  std::vector<DPMAgentGenerationRequest::PrefillChunk> chunks;
  TranscriptCursor transcript;
  bool in_current_epoch = false;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &transcript));
    if (event.turn_receipt->projection_manifest.correction_digest !=
        current_correction_digest) {
      chunks.clear();
      in_current_epoch = false;
      continue;
    }
    if (!in_current_epoch) {
      chunks.clear();
      in_current_epoch = true;
    }
    chunks.push_back(TextChunk(event.turn_receipt->canonical_agent_input));
    if (!event.turn_receipt->decision_token_ids.empty()) {
      chunks.push_back(TokenChunk(event.turn_receipt->decision_token_ids));
    }
  }
  chunks.push_back(TextChunk(current_agent_input));
  return chunks;
}

absl::StatusOr<
    std::vector<DPMAgentGenerationRequest::PrefillChunk>>
DPMEngine::BuildDeltaTranscriptChunks(
    const DPMLogSnapshot& snapshot, uint64_t restored_response_event_index,
    absl::string_view current_agent_input,
    const Hash256& current_correction_digest) const {
  if (restored_response_event_index >= snapshot.events.size() ||
      snapshot.events[restored_response_event_index].kind !=
          DPMEvent::Kind::kModelTurn ||
      !snapshot.events[restored_response_event_index].turn_receipt) {
    return absl::FailedPreconditionError(
        "DPM restore point is not a model-turn event in this log.");
  }
  std::vector<DPMAgentGenerationRequest::PrefillChunk> chunks;
  const Hash256 restored_correction_digest =
      snapshot.events[restored_response_event_index]
          .turn_receipt->projection_manifest.correction_digest;
  if (restored_correction_digest != current_correction_digest) {
    return absl::FailedPreconditionError(
        "DPM restore point is outside the active correction epoch.");
  }
  TranscriptCursor transcript;
  for (const DPMEvent& event : snapshot.events) {
    if (event.index > restored_response_event_index &&
        event.kind == DPMEvent::Kind::kCorrection) {
      return absl::FailedPreconditionError(
          "DPM correction event invalidates the selected checkpoint "
          "descendant interval.");
    }
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &transcript));
    if (event.index <= restored_response_event_index) continue;
    if (event.turn_receipt->projection_manifest.correction_digest !=
        restored_correction_digest) {
      return absl::FailedPreconditionError(
          "DPM correction epoch changed after the selected restore point.");
    }
    chunks.push_back(TextChunk(event.turn_receipt->canonical_agent_input));
    if (!event.turn_receipt->decision_token_ids.empty()) {
      chunks.push_back(TokenChunk(event.turn_receipt->decision_token_ids));
    }
  }
  chunks.push_back(TextChunk(current_agent_input));
  return chunks;
}

absl::StatusOr<std::string> DPMEngine::BuildCanonicalAgentInput(
    const DPMTurnRequest& request, const DPMLogSnapshot& source_snapshot,
    uint64_t input_event_index,
    const DPMProjectionOutcome& projection) const {
  ABSL_RETURN_IF_ERROR(ValidateSnapshotShape(source_snapshot));
  if (input_event_index >= source_snapshot.events.size()) {
    return absl::InvalidArgumentError(
        "DPM agent input names an event outside its source snapshot.");
  }
  if (request.payload.size() > kMaximumDPMEventPayloadBytes ||
      projection.projected_memory.size() > kMaximumDPMEventPayloadBytes ||
      request.case_id.size() > kMaximumDPMProjectionIdentityBytes) {
    return absl::ResourceExhaustedError(
        "DPM canonical agent input field exceeds its product bound.");
  }
  std::string input;
  input.reserve(request.payload.size() + projection.projected_memory.size() +
                request.case_id.size() + source_snapshot.log_id.size() + 512);
  input.append("DPM_AGENT_INPUT_V1\n");
  AppendTextSection("LOG_ID_BYTES", source_snapshot.log_id, &input);
  absl::StrAppend(&input, "SOURCE_GENERATION ", source_snapshot.generation,
                  "\nSOURCE_PREFIX_SHA256 ",
                  source_snapshot.prefix_hash.ToHex(),
                  "\nINPUT_EVENT_INDEX ", input_event_index,
                  "\nINPUT_KIND ", static_cast<int>(request.kind), "\n");
  AppendTextSection("CASE_ID_BYTES", request.case_id, &input);
  absl::StrAppend(&input, "PROJECTION_REQUEST_SHA256 ",
                  projection.manifest.request_hash.ToHex(),
                  "\nPROJECTION_MANIFEST_SHA256 ",
                  projection.manifest.manifest_hash.ToHex(),
                  "\nCORRECTION_SET_SHA256 ",
                  projection.manifest.correction_digest.ToHex(), "\n");
  AppendTextSection("PROJECTED_MEMORY_BYTES", projection.projected_memory,
                    &input);
  AppendTextSection("REQUEST_BYTES", request.payload, &input);
  input.append("END_DPM_AGENT_INPUT\n");
  if (input.size() > kMaximumDPMCanonicalAgentInputBytes) {
    return absl::ResourceExhaustedError(
        "DPM canonical agent input exceeds its derived product bound.");
  }
  if (!IsValidUtf8(input)) {
    return absl::FailedPreconditionError(
        "DPM projection produced an agent input that is not valid UTF-8.");
  }
  return input;
}

absl::StatusOr<Hash256> DPMEngine::ComputeTranscriptHash(
    const DPMLogSnapshot& snapshot, absl::string_view current_agent_input,
    absl::string_view current_decision,
    const std::vector<int>& current_decision_token_ids,
    const Hash256& current_correction_digest) const {
  TranscriptCursor transcript;
  for (const DPMEvent& event : snapshot.events) {
    if (event.kind != DPMEvent::Kind::kModelTurn) continue;
    ABSL_RETURN_IF_ERROR(
        ValidateReceiptAndAdvance(snapshot, event, &transcript));
  }
  if (!transcript.correction_digest.has_value() ||
      *transcript.correction_digest != current_correction_digest) {
    ResetTranscriptHasher(current_correction_digest, &transcript);
  }
  UpdateBytesFrame('I', current_agent_input, &transcript.hasher);
  UpdateTokenFrame(current_decision_token_ids, &transcript.hasher);
  UpdateBytesFrame('O', current_decision, &transcript.hasher);
  return transcript.hasher.Finalize();
}

absl::StatusOr<DPMSessionCheckpointArtifact>
DPMEngine::CaptureProducingSession(
    Engine::Session* session, const DPMLogSnapshot& source_snapshot,
    uint64_t response_event_index,
    const DPMProjectionOutcome& projection,
    const Hash256& agent_request_hash, const Hash256& transcript_hash,
    const std::optional<Hash256>& restored_from_checkpoint_id,
    int64_t created_unix_micros) const {
  if (session == nullptr) {
    return absl::InvalidArgumentError(
        "Cannot capture a null DPM producing session.");
  }
  ABSL_ASSIGN_OR_RETURN(SessionHandoffIdentity session_identity,
                        session->GetSessionHandoffIdentity());
  if (session_identity != agent_runtime_->GetSessionHandoffIdentity()) {
    return absl::FailedPreconditionError(
        "DPM producing session identity does not match its loaded runtime.");
  }
  SessionHandoffOptions options;
  options.key_id = config_.checkpoint_key_id;
  options.authentication_key = config_.checkpoint_authentication_key;
  options.expected_identity = agent_runtime_->GetSessionHandoffIdentity();

  DPMSessionCheckpointArtifact artifact;
  StringByteSink sink(&artifact.authenticated_envelope);
  ABSL_RETURN_IF_ERROR(session->ExportHandoffTo(options, &sink));

  DPMSessionCheckpointDescriptor& descriptor = artifact.descriptor;
  descriptor.log_id = source_snapshot.log_id;
  descriptor.stage = DPMSessionCheckpointStage::kAgentDecision;
  descriptor.source_event_count = source_snapshot.events.size();
  descriptor.source_prefix_hash = source_snapshot.prefix_hash;
  descriptor.response_event_index = response_event_index;
  descriptor.projection_request_hash = projection.manifest.request_hash;
  descriptor.projection_manifest_hash = projection.manifest.manifest_hash;
  descriptor.correction_digest = projection.manifest.correction_digest;
  descriptor.agent_request_hash = agent_request_hash;
  descriptor.agent_transcript_hash = transcript_hash;
  descriptor.session_identity = session_identity;
  descriptor.key_id = options.key_id;
  descriptor.envelope_hash = Sha256(artifact.authenticated_envelope);
  descriptor.envelope_size = artifact.authenticated_envelope.size();
  descriptor.created_unix_micros = created_unix_micros;
  descriptor.restored_from_checkpoint_id = restored_from_checkpoint_id;
  ABSL_ASSIGN_OR_RETURN(descriptor.descriptor_id,
                        ComputeDPMSessionCheckpointId(descriptor));
  ABSL_RETURN_IF_ERROR(ValidateDPMSessionCheckpointArtifact(artifact));
  return artifact;
}

absl::StatusOr<DPMSessionCheckpointArtifact>
DPMEngine::BuildExactWorkerCheckpointArtifact(
    absl::string_view authenticated_envelope,
    const ExactRegenerationDPMAgentPhysicalExecution& physical_execution,
    const DPMLogSnapshot& source_snapshot, uint64_t response_event_index,
    const DPMProjectionOutcome& projection,
    const Hash256& agent_request_hash, const Hash256& transcript_hash,
    const Hash256& capsule_restore_admission_record_id,
    int64_t created_unix_micros) const {
  ABSL_ASSIGN_OR_RETURN(
      const std::optional<Hash256> current_capsule_restore_admission_id,
      agent_runtime_->GetCapsuleRestoreAdmissionRecordId());
  if (!current_capsule_restore_admission_id.has_value() ||
      *current_capsule_restore_admission_id !=
          capsule_restore_admission_record_id) {
    return absl::AbortedError(
        "CapsuleRestore admission changed before exact checkpoint "
        "publication.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateExactRegenerationDPMAgentPhysicalExecution(
          physical_execution,
          static_cast<uint32_t>(config_.max_decision_tokens)));
  const DPMAgentReplayExecution& replay =
      physical_execution.replay_execution;
  if (replay.mode != DPMReplayMode::kExactRegeneration ||
      !replay.exact_profile_id.has_value() ||
      !replay.exact_profile_admission_record_id.has_value() ||
      !replay.exact_output_evidence_hash.has_value() ||
      IsZeroHash(capsule_restore_admission_record_id) ||
      !physical_execution.run_zero_transient_producing_capsule_evidence
           .has_value() ||
      !physical_execution.durable_producing_capsule_evidence.has_value() ||
      physical_execution.request_evidence.runs.empty()) {
    return absl::FailedPreconditionError(
        "Exact DPM checkpoint capture lacks admitted run-zero producing "
        "capsule evidence.");
  }
  const FreshWorkerProducingCapsuleEvidence& transient =
      *physical_execution.run_zero_transient_producing_capsule_evidence;
  const FreshWorkerDurableProducingCapsuleEvidence& durable =
      *physical_execution.durable_producing_capsule_evidence;
  const ExactRegenerationRunEvidence& run_zero =
      physical_execution.request_evidence.runs.front();
  if (physical_execution.capture_run_policy !=
          ExactRegenerationCaptureRunPolicy::kRunZeroOnly ||
      run_zero.run_index != 0 ||
      durable.session_identity != agent_runtime_->GetSessionHandoffIdentity() ||
      transient.session_identity != durable.session_identity ||
      durable.key_id != config_.checkpoint_key_id ||
      durable.envelope_size != authenticated_envelope.size() ||
      durable.envelope_hash != Sha256(authenticated_envelope) ||
      durable.output_evidence_hash != *replay.exact_output_evidence_hash ||
      transient.output_evidence_hash != *replay.exact_output_evidence_hash) {
    return absl::DataLossError(
        "Exact DPM checkpoint bytes or producing-worker evidence disagree "
        "with the consensus output and loaded runtime.");
  }

  DPMSessionCheckpointArtifact artifact;
  artifact.authenticated_envelope.assign(authenticated_envelope.data(),
                                         authenticated_envelope.size());
  DPMSessionCheckpointDescriptor& descriptor = artifact.descriptor;
  descriptor.log_id = source_snapshot.log_id;
  descriptor.stage = DPMSessionCheckpointStage::kAgentDecision;
  descriptor.source_event_count = source_snapshot.events.size();
  descriptor.source_prefix_hash = source_snapshot.prefix_hash;
  descriptor.response_event_index = response_event_index;
  descriptor.projection_request_hash = projection.manifest.request_hash;
  descriptor.projection_manifest_hash = projection.manifest.manifest_hash;
  descriptor.correction_digest = projection.manifest.correction_digest;
  descriptor.agent_request_hash = agent_request_hash;
  descriptor.agent_transcript_hash = transcript_hash;
  descriptor.session_identity = durable.session_identity;
  descriptor.key_id = durable.key_id;
  descriptor.envelope_hash = durable.envelope_hash;
  descriptor.envelope_size = durable.envelope_size;
  descriptor.created_unix_micros = created_unix_micros;
  descriptor.replay_mode = DPMReplayMode::kExactRegeneration;
  descriptor.capture_origin =
      DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker;
  descriptor.restored_from_checkpoint_id =
      physical_execution.restored_checkpoint_id;
  descriptor.exact_profile_id = replay.exact_profile_id;
  descriptor.exact_profile_admission_record_id =
      *replay.exact_profile_admission_record_id;
  descriptor.capsule_restore_admission_record_id =
      capsule_restore_admission_record_id;
  descriptor.exact_request_execution_evidence_id =
      physical_execution.request_evidence.evidence_id;
  descriptor.worker_prefill_mode =
      ToCheckpointWorkerPrefillMode(physical_execution.prefill_mode);
  descriptor.execution_plan_hash =
      physical_execution.physical_execution_plan_hash;
  descriptor.exact_output_evidence_hash =
      *replay.exact_output_evidence_hash;
  descriptor.worker_provenance = DPMExactWorkerCheckpointProvenance{
      .run_index = run_zero.run_index,
      .execution_plan_hash = physical_execution.physical_execution_plan_hash,
      .request_envelope_hash = run_zero.request_envelope_hash,
      .result_envelope_hash = run_zero.result_envelope_hash,
      .transient_envelope_size = transient.transient_envelope_size,
      .transient_envelope_hash = transient.transient_envelope_hash,
      .output_evidence_hash = transient.output_evidence_hash,
  };
  ABSL_ASSIGN_OR_RETURN(descriptor.descriptor_id,
                        ComputeDPMSessionCheckpointId(descriptor));
  ABSL_RETURN_IF_ERROR(ValidateDPMSessionCheckpointArtifact(artifact));
  return artifact;
}

absl::StatusOr<DPMTurnResult> DPMEngine::RunTurn(
    const DPMTurnRequest& request) {
  absl::MutexLock turn_lock(turn_mutex_);
  // A committed operation is recoverable from the authoritative log without
  // an inference runtime, projector, clock, or checkpoint repository. Keep
  // that idempotent recovery path ahead of all inference-support preflight.
  if (log_ == nullptr) {
    return absl::InvalidArgumentError(
        "DPM RunTurn requires an authoritative event log.");
  }
  if (request.operation_id.empty() || request.case_id.empty() ||
      request.payload.empty()) {
    return absl::InvalidArgumentError(
        "DPM turn requires non-empty operation id, case id, and payload.");
  }
  if (!IsValidUtf8(request.operation_id) || !IsValidUtf8(request.case_id) ||
      !IsValidUtf8(request.payload)) {
    return absl::InvalidArgumentError(
        "DPM operation id, case id, and input payload must be valid UTF-8.");
  }
  if (request.operation_id.size() > kMaximumDPMEventOperationIdBytes ||
      request.case_id.size() > kMaximumDPMProjectionIdentityBytes ||
      request.payload.size() > kMaximumDPMEventPayloadBytes ||
      ContainsControlByte(request.operation_id) ||
      ContainsControlByte(request.case_id)) {
    return absl::InvalidArgumentError(
        "DPM operation/case identities or input payload exceed product "
        "bounds, or an identity contains a control byte.");
  }
  if (!IsDecisionInputKind(request.kind)) {
    return absl::InvalidArgumentError(
        "DPM RunTurn accepts only user, tool, or internal input events; "
        "corrections use AppendCorrection.");
  }
  if ((request.timestamp_us.has_value() && *request.timestamp_us <= 0) ||
      (request.response_timestamp_us.has_value() &&
       *request.response_timestamp_us <= 0)) {
    return absl::InvalidArgumentError(
        "DPM turn timestamps must be positive when supplied.");
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<DPMEventLogOperationLease> operation,
                        log_->AcquireOperationLease());
  if (operation == nullptr) {
    return absl::InternalError("DPM event log returned a null operation lease.");
  }
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot initial, log_->Snapshot());
  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex initial_index,
                        ValidateAndIndexAuthoritativeLog(initial));
  if (initial.case_id != request.case_id) {
    return absl::FailedPreconditionError(
        "DPM turn case id does not match the immutable log case.");
  }
  ABSL_ASSIGN_OR_RETURN(Hash256 initial_prefix,
                        SnapshotPrefixHashAt(initial, initial.generation));
  if (initial_prefix != initial.prefix_hash) {
    return absl::AbortedError(
        "DPM log changed while acquiring its authoritative snapshot.");
  }

  std::optional<uint64_t> existing_input_index;
  auto current_operation =
      initial_index.operations.find(request.operation_id);
  if (current_operation != initial_index.operations.end()) {
    const OperationRecord& record = current_operation->second;
    if (record.correction != nullptr) {
      return absl::AlreadyExistsError(
          "DPM operation id is already bound to a correction.");
    }
    if (record.response != nullptr) {
      ABSL_ASSIGN_OR_RETURN(std::optional<DPMTurnResult> recovered,
                            RecoverCommittedTurn(initial, request));
      if (!recovered.has_value()) {
        return absl::DataLossError(
            "Indexed committed DPM operation could not be recovered.");
      }
      return std::move(*recovered);
    }
    if (record.input == nullptr) {
      return absl::DataLossError(
          "DPM operation index contains neither input nor response.");
    }
    ABSL_RETURN_IF_ERROR(ValidateInputEvent(*record.input, request));
    existing_input_index = record.input->index;
  }
  if (initial_index.pending_input != nullptr &&
      (!existing_input_index.has_value() ||
       initial_index.pending_input->index != *existing_input_index)) {
    return absl::FailedPreconditionError(
        "Another DPM operation has an incomplete tail turn that must be "
        "retried first.");
  }

  // From this point the operation either needs a new input append or must
  // retry an existing recoverable input. Both paths require the complete
  // projection, agent, and checkpoint capability preflight before any new
  // mutation or inference work.
  ABSL_RETURN_IF_ERROR(ValidateConfiguration());
  const SessionHandoffIdentity& loaded_identity =
      agent_runtime_->GetSessionHandoffIdentity();
  if (initial_index.agent_identity.has_value() &&
      *initial_index.agent_identity != loaded_identity) {
    return absl::FailedPreconditionError(
        "DPM log agent identity does not match the loaded engine profile.");
  }

  DPMLogSnapshot source_snapshot;
  uint64_t input_event_index = 0;
  if (existing_input_index.has_value()) {
    input_event_index = *existing_input_index;
    source_snapshot = std::move(initial);
  } else {
    DPMEvent input;
    input.kind = request.kind;
    input.timestamp_us = request.timestamp_us.value_or(clock_->NowMicros());
    if (input.timestamp_us <= 0) {
      return absl::InvalidArgumentError(
          "DPM input timestamp must be positive.");
    }
    input.operation_id = request.operation_id;
    input.case_id = request.case_id;
    input.payload = request.payload;
    ABSL_ASSIGN_OR_RETURN(
        DPMAppendResult appended,
        log_->AppendIfGeneration(std::move(input), initial.generation));
    source_snapshot = std::move(appended.snapshot);
    input_event_index = appended.event_index;
    if (input_event_index >= source_snapshot.events.size()) {
      return absl::DataLossError(
          "DPM log append did not return its authoritative input event.");
    }
    ABSL_RETURN_IF_ERROR(ValidateInputEvent(
        source_snapshot.events[input_event_index], request));
  }

  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex source_index,
                        ValidateAndIndexAuthoritativeLog(source_snapshot));
  if (source_snapshot.case_id != request.case_id ||
      source_index.pending_input == nullptr ||
      source_index.pending_input->index != input_event_index ||
      input_event_index + 1 != source_snapshot.events.size()) {
    return absl::DataLossError(
        "DPM input append did not produce the sole recoverable tail turn.");
  }
  if (source_index.agent_identity.has_value() &&
      *source_index.agent_identity != loaded_identity) {
    return absl::FailedPreconditionError(
        "DPM log agent identity changed before generation.");
  }
  ABSL_ASSIGN_OR_RETURN(
      Hash256 source_prefix,
      SnapshotPrefixHashAt(source_snapshot, source_snapshot.generation));
  if (source_prefix != source_snapshot.prefix_hash) {
    return absl::AbortedError(
        "DPM log changed after publishing the recoverable input.");
  }

  DPMProjectionRequest projection_request;
  projection_request.log = source_snapshot;
  projection_request.input_event_index = input_event_index;
  projection_request.case_id = request.case_id;
  ABSL_ASSIGN_OR_RETURN(DPMProjectionOutcome projection,
                        projection_provider_->Project(projection_request));
  ABSL_RETURN_IF_ERROR(
      ValidateProjectionOutcome(projection, projection_request));
  if (projection.manifest.correction_digest !=
      source_index.correction_digest) {
    return absl::FailedPreconditionError(
        "DPM projection correction digest is not derived from the raw log.");
  }
  const DPMReplayMode agent_replay_mode = agent_runtime_->GetReplayMode();
  if (projection.manifest.replay_mode != agent_replay_mode) {
    return absl::FailedPreconditionError(
        "DPM projection and agent stages must use the same named replay "
        "guarantee.");
  }
  ABSL_ASSIGN_OR_RETURN(
      std::string canonical_agent_input,
      BuildCanonicalAgentInput(request, source_snapshot, input_event_index,
                               projection));
  const uint32_t max_decision_tokens =
      static_cast<uint32_t>(config_.max_decision_tokens);
  const Hash256 agent_request_hash = ComputeLogicalAgentRequestHash(
      source_index.transcript, canonical_agent_input, max_decision_tokens,
      projection.manifest.correction_digest, loaded_identity);

  ABSL_ASSIGN_OR_RETURN(
      std::vector<DPMAgentGenerationRequest::PrefillChunk> full_chunks,
      BuildFullTranscriptChunks(source_snapshot, canonical_agent_input,
                                projection.manifest.correction_digest));
  DPMAgentExecutionRequest logical_generation_request{
      .logical_agent_request_hash = agent_request_hash,
      .correction_digest = projection.manifest.correction_digest,
      .max_output_tokens = max_decision_tokens,
      .full_canonical_prefill_chunks = full_chunks,
  };
  ABSL_RETURN_IF_ERROR(
      ValidateDPMAgentExecutionRequest(logical_generation_request));

  if (source_snapshot.events.size() >=
      std::numeric_limits<uint64_t>::max()) {
    return absl::ResourceExhaustedError(
        "DPM log cannot address another response event.");
  }
  const uint64_t response_event_index = source_snapshot.events.size();
  if (response_event_index != input_event_index + 1) {
    return absl::DataLossError(
        "DPM response would not immediately follow its authoritative input.");
  }
  const int64_t response_timestamp =
      request.response_timestamp_us.value_or(clock_->NowMicros());
  if (response_timestamp <= 0 ||
      response_timestamp <
          source_snapshot.events[input_event_index].timestamp_us) {
    return absl::InvalidArgumentError(
        "DPM response timestamp must be positive and not precede its input.");
  }
  const uint64_t turn_number = CountCommittedTurns(source_snapshot) + 1;
  const bool capture_milestone =
      config_.checkpoint_interval_turns != 0 &&
      turn_number % config_.checkpoint_interval_turns == 0;

  std::optional<RestoreCandidate> restore_candidate;
  if (config_.restore_session_checkpoints) {
    ABSL_ASSIGN_OR_RETURN(restore_candidate,
                          FindRestoreCandidate(source_snapshot, projection));
  }

  std::optional<Hash256> capsule_restore_admission_record_id;
  if (agent_replay_mode == DPMReplayMode::kExactRegeneration &&
      (restore_candidate.has_value() || capture_milestone)) {
    ABSL_ASSIGN_OR_RETURN(capsule_restore_admission_record_id,
                          agent_runtime_->GetCapsuleRestoreAdmissionRecordId());
    if (!capsule_restore_admission_record_id.has_value() ||
        IsZeroHash(*capsule_restore_admission_record_id)) {
      return absl::FailedPreconditionError(
          "Exact DPM checkpoint use requires a currently authenticated "
          "CapsuleRestore admission record.");
    }
  }

  std::unique_ptr<Engine::Session> session;
  auto validate_created_session =
      [&loaded_identity](Engine::Session* candidate) -> absl::Status {
    if (candidate == nullptr) {
      return absl::InternalError("DPM agent runtime returned a null session.");
    }
    absl::StatusOr<SessionHandoffIdentity> candidate_identity =
        candidate->GetSessionHandoffIdentity();
    if (!candidate_identity.ok()) return candidate_identity.status();
    if (*candidate_identity != loaded_identity) {
      return absl::FailedPreconditionError(
          "DPM session identity does not match its loaded agent runtime.");
    }
    return absl::OkStatus();
  };

  DPMAgentGenerationRequest generation_request;
  generation_request.max_output_tokens = config_.max_decision_tokens;
  bool restored = false;
  std::optional<Hash256> restored_from_checkpoint_id;
  std::optional<ExactRegenerationDPMAgentPhysicalExecution>
      exact_physical_execution;
  std::string exact_staged_capsule;
  DPMAgentReplayExecution agent_execution;
  std::function<absl::StatusOr<ExactRegenerationDPMAgentPhysicalExecution>(
      const RestoreCandidate*, bool, std::string*)>
      run_exact_attempt;
  if (agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay) {
    struct PreparedWinnerSession {
      std::unique_ptr<Engine::Session> session;
      DPMAgentGenerationRequest generation_request;
      std::optional<Hash256> restored_checkpoint_id;
    };
    auto prepare_winner_session =
        [&](bool allow_restore)
        -> absl::StatusOr<PreparedWinnerSession> {
      PreparedWinnerSession prepared;
      prepared.generation_request.max_output_tokens =
          config_.max_decision_tokens;
      prepared.generation_request.canonical_prefill_chunks = full_chunks;
      ABSL_ASSIGN_OR_RETURN(prepared.session,
                            agent_runtime_->CreateSession());
      ABSL_RETURN_IF_ERROR(validate_created_session(prepared.session.get()));
      if (!allow_restore || !restore_candidate.has_value()) return prepared;

      SessionHandoffOptions options;
      options.key_id = config_.checkpoint_key_id;
      options.authentication_key = config_.checkpoint_authentication_key;
      options.expected_identity = loaded_identity;
      StringByteSource source(
          restore_candidate->artifact.authenticated_envelope);
      absl::Status import_status =
          prepared.session->ImportHandoffFrom(source, options);
      if (import_status.ok()) {
        absl::StatusOr<std::vector<DPMAgentGenerationRequest::PrefillChunk>>
            delta_chunks = BuildDeltaTranscriptChunks(
                source_snapshot,
                restore_candidate->artifact.descriptor.response_event_index,
                canonical_agent_input,
                projection.manifest.correction_digest);
        if (delta_chunks.ok()) {
          prepared.generation_request.canonical_prefill_chunks =
              std::move(*delta_chunks);
          prepared.restored_checkpoint_id =
              restore_candidate->artifact.descriptor.descriptor_id;
          return prepared;
        }
      }

      // Even though import is transactional, discard its target. This also
      // covers a structurally valid capsule whose exact delta boundary can no
      // longer be reconstructed from the authoritative log.
      prepared.session.reset();
      ABSL_ASSIGN_OR_RETURN(prepared.session,
                            agent_runtime_->CreateSession());
      ABSL_RETURN_IF_ERROR(validate_created_session(prepared.session.get()));
      prepared.generation_request.canonical_prefill_chunks = full_chunks;
      return prepared;
    };

    ABSL_ASSIGN_OR_RETURN(PreparedWinnerSession prepared,
                          prepare_winner_session(true));
    session = std::move(prepared.session);
    generation_request = std::move(prepared.generation_request);
    restored_from_checkpoint_id = prepared.restored_checkpoint_id;
    restored = restored_from_checkpoint_id.has_value();

    absl::StatusOr<DPMAgentReplayExecution> generated =
        agent_runtime_->Generate(session.get(), generation_request,
                                 logical_generation_request);
    if (!generated.ok() && restored) {
      // A structurally valid capsule can still fail when its first dynamic
      // continuation is bound. Discard the mutated target and reconstruct the
      // same logical request from the authoritative log in a fresh session.
      ABSL_ASSIGN_OR_RETURN(prepared, prepare_winner_session(false));
      session = std::move(prepared.session);
      generation_request = std::move(prepared.generation_request);
      generated = agent_runtime_->Generate(
          session.get(), generation_request, logical_generation_request);
      restored = false;
      restored_from_checkpoint_id.reset();
    }
    if (!generated.ok()) return generated.status();
    agent_execution = std::move(*generated);
    if (!agent_execution.producing_session_matches_output) {
      const DPMAgentReplayExecution selected_winner = agent_execution;
      session.reset();
      restored = false;
      restored_from_checkpoint_id.reset();

      // A catalog hit or lost publish race has no session corresponding to the
      // selected bytes. Only checkpoint milestones need to rematerialize one.
      // The direct runtime call bypasses the catalog and accepts the new live
      // session only if its complete canonical output byte-matches the winner.
      if (capture_milestone) {
        absl::Status rematerialization_status =
            absl::FailedPreconditionError(
                "WinnerReplay rematerialization did not start.");
        absl::StatusOr<PreparedWinnerSession> rematerialization_session =
            prepare_winner_session(true);
        if (rematerialization_session.ok()) {
          absl::StatusOr<DPMAgentReplayExecution> rematerialized =
              agent_runtime_->RematerializeCanonicalWinner(
                  rematerialization_session->session.get(),
                  rematerialization_session->generation_request,
                  logical_generation_request, selected_winner);
          if (!rematerialized.ok() &&
              rematerialization_session->restored_checkpoint_id.has_value()) {
            // A restored live attempt may fail or disagree. Discard it and
            // perform one full-log rematerialization in a brand-new session.
            rematerialization_session = prepare_winner_session(false);
            if (rematerialization_session.ok()) {
              rematerialized =
                  agent_runtime_->RematerializeCanonicalWinner(
                      rematerialization_session->session.get(),
                      rematerialization_session->generation_request,
                      logical_generation_request, selected_winner);
            }
          }
          if (rematerialized.ok()) {
            session = std::move(rematerialization_session->session);
            generation_request =
                std::move(rematerialization_session->generation_request);
            restored_from_checkpoint_id =
                rematerialization_session->restored_checkpoint_id;
            restored = restored_from_checkpoint_id.has_value();
            agent_execution = std::move(*rematerialized);
            rematerialization_status = absl::OkStatus();
          } else {
            rematerialization_status = rematerialized.status();
          }
        } else {
          rematerialization_status = rematerialization_session.status();
        }
        if (!rematerialization_status.ok() &&
            config_.require_checkpoint_at_milestone) {
          return rematerialization_status;
        }
      }
    }
  } else {
    ABSL_ASSIGN_OR_RETURN(
        const Hash256 replay_request_hash,
        ComputeAgentReplayRequestHash(logical_generation_request));

    // Every invocation of this helper is a brand-new N-process attempt. The
    // caller owns and clears the staging string so a failed restored or capture
    // attempt cannot leak any run-zero bytes into its fallback.
    run_exact_attempt =
        [&, replay_request_hash](const RestoreCandidate* selected_checkpoint,
            bool capture_producing_capsule,
            std::string* staged_capsule)
        -> absl::StatusOr<ExactRegenerationDPMAgentPhysicalExecution> {
      if (staged_capsule == nullptr) {
        return absl::InvalidArgumentError(
            "Exact DPM execution requires caller-owned capsule staging.");
      }
      staged_capsule->clear();

      FreshWorkerExecutionPlan execution_plan;
      execution_plan.logical_replay_request_hash = replay_request_hash;
      if (selected_checkpoint != nullptr) {
        const DPMSessionCheckpointDescriptor& descriptor =
            selected_checkpoint->artifact.descriptor;
        ABSL_ASSIGN_OR_RETURN(
            std::vector<DPMAgentGenerationRequest::PrefillChunk> delta_chunks,
            BuildDeltaTranscriptChunks(
                source_snapshot, descriptor.response_event_index,
                canonical_agent_input, projection.manifest.correction_digest));
        DPMAgentDeltaExecutionRequest delta_request{
            .logical_agent_request_hash = agent_request_hash,
            .correction_digest = projection.manifest.correction_digest,
            .restore_checkpoint_id = descriptor.descriptor_id,
            .restored_response_event_index =
                descriptor.response_event_index,
            .restored_agent_transcript_hash =
                descriptor.agent_transcript_hash,
            .max_output_tokens = max_decision_tokens,
            .canonical_delta_prefill_chunks = std::move(delta_chunks),
        };
        ABSL_ASSIGN_OR_RETURN(
            execution_plan.canonical_execution_payload,
            EncodeDPMAgentDeltaExecutionRequest(delta_request));
        execution_plan.prefill_mode =
            FreshWorkerPrefillMode::kOwnPositionCapsuleDelta;
        execution_plan.restore_checkpoint_id = descriptor.descriptor_id;
        execution_plan.session_identity = descriptor.session_identity;
        execution_plan.restore_durable_envelope_hash =
            descriptor.envelope_hash;
        execution_plan.restore_durable_envelope_size =
            descriptor.envelope_size;
      }
      execution_plan.capture_producing_capsule =
          capture_producing_capsule;
      ABSL_ASSIGN_OR_RETURN(
          execution_plan.plan_hash,
          ComputeFreshWorkerExecutionPlanHash(execution_plan));

      SessionHandoffOptions restore_options;
      SessionHandoffOptions capture_options;
      std::optional<StringByteSource> restore_source;
      std::optional<StringByteSink> capture_sink;
      ExactRegenerationExecutionInput physical_input{
          .execution_plan = execution_plan,
      };
      if (selected_checkpoint != nullptr) {
        restore_options.key_id = config_.checkpoint_key_id;
        restore_options.authentication_key =
            config_.checkpoint_authentication_key;
        restore_options.expected_identity = loaded_identity;
        restore_source.emplace(
            selected_checkpoint->artifact.authenticated_envelope);
        physical_input.durable_restore_source = &*restore_source;
        physical_input.durable_restore_options = &restore_options;
      }
      if (capture_producing_capsule) {
        capture_options.key_id = config_.checkpoint_key_id;
        capture_options.authentication_key =
            config_.checkpoint_authentication_key;
        capture_options.expected_identity = loaded_identity;
        capture_sink.emplace(staged_capsule);
        physical_input.staging_capture_destination = &*capture_sink;
        physical_input.staging_capture_options = &capture_options;
      }

      ABSL_ASSIGN_OR_RETURN(
          ExactRegenerationDPMAgentPhysicalExecution physical,
          agent_runtime_->GeneratePhysicalExact(logical_generation_request,
                                                physical_input));
      ABSL_RETURN_IF_ERROR(
          ValidateExactRegenerationDPMAgentPhysicalExecution(
              physical, max_decision_tokens));
      if (physical.physical_execution_plan_hash != execution_plan.plan_hash ||
          physical.restored_checkpoint_id !=
              execution_plan.restore_checkpoint_id) {
        return absl::DataLossError(
            "Exact DPM worker execution changed its parent-selected physical "
            "checkpoint plan.");
      }
      return physical;
    };

    const bool capture_exact_session = capture_milestone;
    absl::StatusOr<ExactRegenerationDPMAgentPhysicalExecution>
        generated_exact =
            restore_candidate.has_value()
                ? run_exact_attempt(&*restore_candidate,
                                    capture_exact_session,
                                    &exact_staged_capsule)
                : run_exact_attempt(nullptr, capture_exact_session,
                                    &exact_staged_capsule);
    if (!generated_exact.ok() && restore_candidate.has_value()) {
      // A failed restored attempt is never repaired in place. Discard run-zero
      // staging and execute the entire logical request again in a new set of
      // cold processes from the authoritative log.
      generated_exact = run_exact_attempt(nullptr, capture_exact_session,
                                          &exact_staged_capsule);
    }
    if (!generated_exact.ok() && capture_exact_session &&
        !config_.require_checkpoint_at_milestone) {
      // Optional capture is not allowed to turn a partially captured worker
      // attempt into the decision. A clean all-worker full-prefill attempt
      // without capture becomes the sole request evidence.
      generated_exact =
          run_exact_attempt(nullptr, false, &exact_staged_capsule);
    }
    if (!generated_exact.ok()) return generated_exact.status();
    exact_physical_execution = std::move(*generated_exact);
    agent_execution = exact_physical_execution->replay_execution;
    restored_from_checkpoint_id =
        exact_physical_execution->restored_checkpoint_id;
    restored = restored_from_checkpoint_id.has_value();
  }

  struct MaterializedAgentDecision {
    DPMAgentGenerationOutcome outcome;
    std::optional<Hash256> exact_output_evidence_hash;
    uint32_t exact_logit_frame_count = 0;
    Hash256 transcript_hash;
  };
  auto materialize_agent_decision =
      [&](const DPMAgentReplayExecution& execution)
      -> absl::StatusOr<MaterializedAgentDecision> {
    ABSL_RETURN_IF_ERROR(
        ValidateDPMAgentReplayExecution(execution, max_decision_tokens));
    if (execution.mode != agent_replay_mode) {
      return absl::DataLossError(
          "DPM agent execution returned evidence for another replay mode.");
    }
    MaterializedAgentDecision material;
    material.outcome.decision_output = execution.decision_output;
    material.outcome.decision_token_ids = execution.decision_token_ids;
    ABSL_RETURN_IF_ERROR(
        ValidateAgentOutcome(material.outcome, max_decision_tokens));
    if (execution.mode == DPMReplayMode::kExactRegeneration) {
      if (execution.exact_logit_frames.size() >
          std::numeric_limits<uint32_t>::max()) {
        return absl::ResourceExhaustedError(
            "Exact agent logits evidence exceeds the receipt count domain.");
      }
      DPMAgentDecisionEnvelope decision_envelope{
          .decision_output = execution.decision_output,
          .canonical_token_bytes = execution.exact_token_bytes,
      };
      ABSL_ASSIGN_OR_RETURN(
          const std::string canonical_decision_envelope,
          EncodeDPMAgentDecisionEnvelope(decision_envelope));
      material.exact_output_evidence_hash =
          ComputeFreshWorkerOutputEvidenceHash(
              canonical_decision_envelope, execution.exact_token_bytes,
              execution.exact_logit_frames);
      if (!execution.exact_output_evidence_hash.has_value() ||
          *execution.exact_output_evidence_hash !=
              *material.exact_output_evidence_hash) {
        return absl::DataLossError(
            "Exact agent runtime output-evidence commitment disagrees with "
            "the canonical decision, token, and ordered logits evidence.");
      }
      material.exact_logit_frame_count =
          static_cast<uint32_t>(execution.exact_logit_frames.size());
    }
    ABSL_ASSIGN_OR_RETURN(
        material.transcript_hash,
        ComputeTranscriptHash(source_snapshot, canonical_agent_input,
                              material.outcome.decision_output,
                              material.outcome.decision_token_ids,
                              projection.manifest.correction_digest));
    return material;
  };
  ABSL_ASSIGN_OR_RETURN(MaterializedAgentDecision materialized_decision,
                        materialize_agent_decision(agent_execution));

  std::optional<Hash256> checkpoint_id;
  std::optional<DPMExactWorkerCheckpointProvenance>
      exact_checkpoint_provenance;
  if (capture_milestone) {
    if (agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay &&
        agent_execution.producing_session_matches_output && session != nullptr) {
      absl::StatusOr<DPMSessionCheckpointArtifact> artifact =
          CaptureProducingSession(
              session.get(), source_snapshot, response_event_index, projection,
              agent_request_hash, materialized_decision.transcript_hash,
              restored_from_checkpoint_id, response_timestamp);
      if (artifact.ok()) {
        absl::Status put_status = checkpoint_repository_->Put(*artifact);
        if (put_status.ok()) {
          checkpoint_id = artifact->descriptor.descriptor_id;
        } else if (config_.require_checkpoint_at_milestone) {
          return put_status;
        }
      } else if (config_.require_checkpoint_at_milestone) {
        return artifact.status();
      }
    } else if (agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay) {
      if (config_.require_checkpoint_at_milestone) {
        return absl::FailedPreconditionError(
            "DPM checkpoint milestone selected a WinnerReplay catalog result "
            "without its live producing parent session.");
      }
    } else if (exact_physical_execution.has_value() &&
               exact_physical_execution->capture_run_policy ==
                   ExactRegenerationCaptureRunPolicy::kRunZeroOnly) {
      if (!capsule_restore_admission_record_id.has_value()) {
        return absl::FailedPreconditionError(
            "Exact DPM capture lost its CapsuleRestore admission binding.");
      }
      absl::Status publish_status;
      absl::StatusOr<DPMSessionCheckpointArtifact> artifact =
          BuildExactWorkerCheckpointArtifact(
              exact_staged_capsule, *exact_physical_execution, source_snapshot,
              response_event_index, projection, agent_request_hash,
              materialized_decision.transcript_hash,
              *capsule_restore_admission_record_id, response_timestamp);
      if (artifact.ok()) {
        publish_status = checkpoint_repository_->Put(*artifact);
        if (publish_status.ok()) {
          checkpoint_id = artifact->descriptor.descriptor_id;
          exact_checkpoint_provenance = artifact->descriptor.worker_provenance;
        }
      } else {
        publish_status = artifact.status();
      }
      if (!publish_status.ok()) {
        if (config_.require_checkpoint_at_milestone) return publish_status;

        // Durable publication is part of optional capture. If it fails, the
        // captured attempt is discarded as a whole and a new all-process full
        // prefill without capture becomes the decision evidence.
        exact_staged_capsule.clear();
        ABSL_ASSIGN_OR_RETURN(
            ExactRegenerationDPMAgentPhysicalExecution clean_full_execution,
            run_exact_attempt(nullptr, false, &exact_staged_capsule));
        exact_physical_execution = std::move(clean_full_execution);
        agent_execution = exact_physical_execution->replay_execution;
        restored = false;
        restored_from_checkpoint_id.reset();
        ABSL_ASSIGN_OR_RETURN(
            materialized_decision,
            materialize_agent_decision(agent_execution));
      }
    } else if (config_.require_checkpoint_at_milestone) {
      return absl::FailedPreconditionError(
          "Required exact DPM checkpoint capture produced no authenticated "
          "run-zero capsule.");
    }
  }

  DPMAgentGenerationOutcome& agent_outcome = materialized_decision.outcome;
  const std::optional<Hash256>& agent_exact_output_evidence_hash =
      materialized_decision.exact_output_evidence_hash;
  const uint32_t agent_exact_logit_frame_count =
      materialized_decision.exact_logit_frame_count;
  const Hash256& transcript_hash = materialized_decision.transcript_hash;

  DPMTurnReceipt receipt;
  receipt.operation_id = request.operation_id;
  receipt.input_event_index = input_event_index;
  receipt.response_event_index = response_event_index;
  receipt.projection_manifest = projection.manifest;
  receipt.agent_session_identity = loaded_identity;
  receipt.max_decision_tokens = max_decision_tokens;
  receipt.agent_request_hash = agent_request_hash;
  receipt.agent_replay_mode = agent_execution.mode;
  receipt.agent_replay_request_hash = agent_execution.replay_request_hash;
  receipt.agent_execution_evidence_hash =
      agent_execution.execution_evidence_hash;
  receipt.agent_exact_profile_id = agent_execution.exact_profile_id;
  receipt.agent_exact_profile_admission_record_id =
      agent_execution.exact_profile_admission_record_id;
  receipt.agent_exact_output_evidence_hash =
      agent_exact_output_evidence_hash;
  receipt.agent_exact_logit_frame_count = agent_exact_logit_frame_count;
  receipt.agent_reused_canonical_winner =
      agent_execution.reused_canonical_winner;
  receipt.agent_producing_session_matched_output =
      agent_execution.producing_session_matches_output;
  receipt.projected_memory = projection.projected_memory;
  receipt.canonical_agent_input = canonical_agent_input;
  receipt.decision_output = agent_outcome.decision_output;
  receipt.decision_token_ids = agent_outcome.decision_token_ids;
  receipt.agent_transcript_hash = transcript_hash;
  receipt.session_checkpoint_id = checkpoint_id;
  receipt.restored_from_session_checkpoint_id =
      restored_from_checkpoint_id;
  if (checkpoint_id.has_value()) {
    receipt.checkpoint_capture_origin =
        agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay
            ? DPMCheckpointCaptureOrigin::kLiveParentSession
            : DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker;
  }
  if (exact_physical_execution.has_value()) {
    receipt.agent_worker_prefill_mode = ToCheckpointWorkerPrefillMode(
        exact_physical_execution->prefill_mode);
    receipt.agent_physical_execution_plan_hash =
        exact_physical_execution->physical_execution_plan_hash;
    if (restored_from_checkpoint_id.has_value() || checkpoint_id.has_value()) {
      if (!capsule_restore_admission_record_id.has_value()) {
        return absl::InternalError(
            "Exact DPM capsule use lost its authenticated admission ID.");
      }
      ABSL_ASSIGN_OR_RETURN(
          const std::optional<Hash256> current_capsule_admission_id,
          agent_runtime_->GetCapsuleRestoreAdmissionRecordId());
      if (current_capsule_admission_id !=
          capsule_restore_admission_record_id) {
        return absl::AbortedError(
            "CapsuleRestore admission changed before DPM receipt commit.");
      }
      receipt.agent_capsule_restore_admission_record_id =
          capsule_restore_admission_record_id;
    }
    if (checkpoint_id.has_value()) {
      if (!exact_checkpoint_provenance.has_value()) {
        return absl::InternalError(
            "Exact DPM checkpoint publication lost run-zero provenance.");
      }
      receipt.agent_exact_worker_checkpoint_provenance =
          exact_checkpoint_provenance;
    }
  }
  ABSL_RETURN_IF_ERROR(ValidateDPMTurnReceiptReplayEvidence(receipt));

  DPMEvent response;
  response.kind = DPMEvent::Kind::kModelTurn;
  response.timestamp_us = response_timestamp;
  response.operation_id = request.operation_id;
  response.case_id = request.case_id;
  response.payload = agent_outcome.decision_output;
  response.turn_receipt = std::move(receipt);
  ABSL_ASSIGN_OR_RETURN(
      DPMAppendResult committed,
      log_->AppendIfGeneration(std::move(response), source_snapshot.generation));
  if (committed.event_index != response_event_index ||
      committed.event_index >= committed.snapshot.events.size()) {
    return absl::DataLossError(
        "DPM log violated contiguous response-index publication.");
  }
  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex committed_index,
                        ValidateAndIndexAuthoritativeLog(committed.snapshot));
  auto committed_operation =
      committed_index.operations.find(request.operation_id);
  if (committed.snapshot.case_id != request.case_id ||
      committed_index.pending_input != nullptr ||
      committed_operation == committed_index.operations.end() ||
      committed_operation->second.response == nullptr ||
      committed_operation->second.response->index != response_event_index) {
    return absl::DataLossError(
        "DPM response commit did not close exactly one authoritative turn.");
  }
  ABSL_ASSIGN_OR_RETURN(
      Hash256 committed_prefix,
      SnapshotPrefixHashAt(committed.snapshot,
                           committed.snapshot.generation));
  if (committed_prefix != committed.snapshot.prefix_hash) {
    return absl::AbortedError(
        "DPM log changed after committing the authoritative response.");
  }

  DPMTurnResult result;
  result.projected_memory = std::move(projection.projected_memory);
  result.decision_output = std::move(agent_outcome.decision_output);
  result.decision_token_ids = std::move(agent_outcome.decision_token_ids);
  result.input_event_index = input_event_index;
  result.response_event_index = committed.event_index;
  result.projection_manifest_hash = projection.manifest.manifest_hash;
  result.projection_replay_mode = projection.manifest.replay_mode;
  result.projection_execution_evidence_hash =
      projection.manifest.execution_evidence_hash;
  result.projection_exact_profile_id = projection.manifest.exact_profile_id;
  result.projection_exact_profile_admission_record_id =
      projection.manifest.exact_profile_admission_record_id;
  result.agent_replay_mode = agent_execution.mode;
  result.agent_execution_evidence_hash =
      agent_execution.execution_evidence_hash;
  result.agent_exact_profile_id = agent_execution.exact_profile_id;
  result.agent_exact_profile_admission_record_id =
      agent_execution.exact_profile_admission_record_id;
  result.agent_exact_output_evidence_hash =
      agent_execution.exact_output_evidence_hash;
  result.agent_exact_logit_frame_count = agent_exact_logit_frame_count;
  result.agent_reused_canonical_winner =
      agent_execution.reused_canonical_winner;
  result.agent_producing_session_matched_output =
      agent_execution.producing_session_matches_output;
  result.agent_worker_capsule_matched_output =
      checkpoint_id.has_value() && exact_checkpoint_provenance.has_value();
  result.session_checkpoint_id = checkpoint_id;
  result.restored_from_session_checkpoint_id =
      restored_from_checkpoint_id;
  if (checkpoint_id.has_value()) {
    result.checkpoint_capture_origin =
        agent_replay_mode == DPMReplayMode::kCanonicalWinnerReplay
            ? DPMCheckpointCaptureOrigin::kLiveParentSession
            : DPMCheckpointCaptureOrigin::kAuthenticatedFreshWorker;
  }
  if (exact_physical_execution.has_value()) {
    result.agent_worker_prefill_mode = ToCheckpointWorkerPrefillMode(
        exact_physical_execution->prefill_mode);
    result.agent_physical_execution_plan_hash =
        exact_physical_execution->physical_execution_plan_hash;
    if (restored_from_checkpoint_id.has_value() || checkpoint_id.has_value()) {
      result.agent_capsule_restore_admission_record_id =
          capsule_restore_admission_record_id;
    }
    result.agent_exact_worker_checkpoint_provenance =
        exact_checkpoint_provenance;
  }
  result.restored_session_checkpoint =
      restored_from_checkpoint_id.has_value();
  return result;
}

absl::StatusOr<DPMCorrectionResult> DPMEngine::AppendCorrection(
    const DPMCorrectionRequest& request) {
  absl::MutexLock turn_lock(turn_mutex_);
  if (log_ == nullptr || clock_ == nullptr) {
    return absl::InvalidArgumentError(
        "DPM correction ingestion requires log and clock dependencies.");
  }
  if (request.operation_id.empty() || request.case_id.empty() ||
      request.canonical_payload.empty()) {
    return absl::InvalidArgumentError(
        "DPM correction requires non-empty operation id, case id, and "
        "canonical payload.");
  }
  if (!IsValidUtf8(request.operation_id) || !IsValidUtf8(request.case_id) ||
      !IsValidUtf8(request.canonical_payload)) {
    return absl::InvalidArgumentError(
        "DPM correction operation id, case id, and payload must be valid "
        "UTF-8.");
  }
  if (request.operation_id.size() > kMaximumDPMEventOperationIdBytes ||
      request.case_id.size() > kMaximumDPMProjectionIdentityBytes ||
      request.canonical_payload.size() > kMaximumDPMEventPayloadBytes ||
      ContainsControlByte(request.operation_id) ||
      ContainsControlByte(request.case_id)) {
    return absl::InvalidArgumentError(
        "DPM correction operation/case identities or payload exceed product "
        "bounds, or an identity contains a control byte.");
  }
  if (request.timestamp_us.has_value() && *request.timestamp_us <= 0) {
    return absl::InvalidArgumentError(
        "DPM correction timestamp must be positive when supplied.");
  }

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<DPMEventLogOperationLease> operation,
                        log_->AcquireOperationLease());
  if (operation == nullptr) {
    return absl::InternalError("DPM event log returned a null operation lease.");
  }
  ABSL_ASSIGN_OR_RETURN(DPMLogSnapshot snapshot, log_->Snapshot());
  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex index,
                        ValidateAndIndexAuthoritativeLog(snapshot));
  if (snapshot.case_id != request.case_id) {
    return absl::FailedPreconditionError(
        "DPM correction case id does not match the immutable log case.");
  }
  ABSL_ASSIGN_OR_RETURN(Hash256 prefix,
                        SnapshotPrefixHashAt(snapshot, snapshot.generation));
  if (prefix != snapshot.prefix_hash) {
    return absl::AbortedError(
        "DPM log changed while acquiring its correction snapshot.");
  }

  auto existing = index.operations.find(request.operation_id);
  if (existing != index.operations.end()) {
    const OperationRecord& record = existing->second;
    if (record.correction == nullptr || record.input != nullptr ||
        record.response != nullptr ||
        !record.correction_digest_after.has_value() ||
        record.correction->case_id != request.case_id ||
        record.correction->payload != request.canonical_payload) {
      return absl::AlreadyExistsError(
          "DPM operation id is already bound to different authoritative "
          "bytes.");
    }
    if (request.timestamp_us.has_value() &&
        record.correction->timestamp_us != *request.timestamp_us) {
      return absl::AlreadyExistsError(
          "DPM correction operation id is already bound to a different "
          "timestamp.");
    }
    DPMCorrectionResult result;
    result.correction_event_index = record.correction->index;
    result.correction_digest = *record.correction_digest_after;
    result.recovered_existing_correction = true;
    return result;
  }
  if (index.pending_input != nullptr) {
    return absl::FailedPreconditionError(
        "DPM correction cannot cross an incomplete tail turn; retry that "
        "operation first.");
  }

  DPMEvent correction;
  correction.kind = DPMEvent::Kind::kCorrection;
  correction.timestamp_us = request.timestamp_us.value_or(clock_->NowMicros());
  if (correction.timestamp_us <= 0) {
    return absl::InvalidArgumentError(
        "DPM correction timestamp must be positive.");
  }
  correction.operation_id = request.operation_id;
  correction.case_id = request.case_id;
  correction.payload = request.canonical_payload;
  ABSL_ASSIGN_OR_RETURN(
      DPMAppendResult appended,
      log_->AppendIfGeneration(std::move(correction), snapshot.generation));
  ABSL_ASSIGN_OR_RETURN(AuthoritativeLogIndex appended_index,
                        ValidateAndIndexAuthoritativeLog(appended.snapshot));
  auto committed = appended_index.operations.find(request.operation_id);
  if (appended.event_index >= appended.snapshot.events.size() ||
      appended.snapshot.case_id != request.case_id ||
      appended_index.pending_input != nullptr ||
      committed == appended_index.operations.end() ||
      committed->second.correction == nullptr ||
      !committed->second.correction_digest_after.has_value() ||
      committed->second.correction->index != appended.event_index) {
    return absl::DataLossError(
        "DPM correction append was not published as one authoritative event.");
  }

  DPMCorrectionResult result;
  result.correction_event_index = appended.event_index;
  result.correction_digest = *committed->second.correction_digest_after;
  return result;
}

}  // namespace litert::lm
