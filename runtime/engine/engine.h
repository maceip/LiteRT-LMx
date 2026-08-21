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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/attributes.h"  // from @com_google_absl
#include "absl/functional/any_invocable.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/engine/engine_settings.h"
#include "runtime/engine/exact_litert_decode.h"
#include "runtime/engine/exact_litert_profile.h"
#include "runtime/engine/io_types.h"
#include "runtime/engine/session_handoff.h"
#include "runtime/engine/session_handoff_capability.h"
#include "runtime/util/byte_stream.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::lm {

// Engine is the interface for the LLM runtime. It is responsible for
// - Initializing the LLM model and related resources, e.g. tokenizer,
//   embedder, etc.
// - Providing the APIs to create the Session.
//
// The Session is responsible for hosting the internal state (e.g. conversation
// history) of each separate interaction with LLM. It is created by the Engine
// and is responsible for:
// - Generating content from the input prompt/query.
// - Running the prefill and decode processes.
//
// Example usage:
//   // Create the model assets.
//   auto model_assets = ModelAssets::Create(model_path);
//   CHECK_OK(model_assets);
//
//   // Create the engine.
//   auto engine = Engine::CreateEngine(EngineSettings::CreateDefault(
//       model_assets, litert::lm::Backend::CPU));
//   CHECK_OK(engine);
//
//   // Create the session.
//   auto session = engine->CreateSession(SessionConfig::CreateDefault());
//   CHECK_OK(session);
//
//   // Run generate content.
//   auto responses = (*session)->GenerateContent({InputText("What's the tallest
//   building in the world?")});
//   CHECK_OK(responses);
//
//   // Print the response.
//   std::cout << *responses << std::endl;
// SessionInterface is responsible for hosting the internal state (e.g.
// conversation history) of each separate interaction with LLM. It is created
// by the Engine and is responsible for:
// - Generating content from the input prompt/query.
// - Running the prefill and decode processes.
class SessionInterface {
 public:
  // The TaskController is responsible for controlling the async task
  // execution.
  class TaskController {
   public:
    TaskController() = default;

    // The TaskController is not copyable. This is to avoid
    // the user from accidentally copying the TaskController and calling the
    // CancelProcess function multiple times.
    TaskController(const TaskController&) = delete;
    TaskController& operator=(const TaskController&) = delete;

    // The TaskController is movable.
    TaskController(TaskController&&) = default;
    TaskController& operator=(TaskController&&) = default;

    // The TaskController destructor.
    virtual ~TaskController() = default;

    // Waits until all the tasks are done or the timeout is reached. The
    // function will return error if the timeout is reached.
    virtual absl::Status WaitUntilDone(absl::Duration timeout) {
      return absl::UnimplementedError("Not implemented.");
    };

    // Cancels the ongoing inference process. Note that if this function is
    // called after the inference process is done, the function will be a
    // no-op.
    virtual absl::Status Cancel() {
      return absl::UnimplementedError("Not implemented.");
    };
  };

  virtual ~SessionInterface() = default;

  // High-level API to generate content from the input prompt/query. This
  // function will handle the prefill and decode processes internally and
  // the usage is similar to the Gemini Text Generation API
  // (https://ai.google.dev/gemini-api/docs/text-generation).
  //
  // DEPRECATED: Prefer using the Conversation API (Conversation::SendMessage /
  // SendMessageStream) for chat and context management. For fine-grained
  // control over execution steps or single-turn inference, use RunPrefill and
  // RunDecode.
  // - contents: The input data for generation.
  ABSL_DEPRECATED(
      "Prefer Conversation API for chat/context management, or RunPrefill and "
      "RunDecode for fine-grained execution control.")
  virtual absl::StatusOr<Responses> GenerateContent(
      const std::vector<InputData>& contents) = 0;

  // This is a not blocking call and the function will return right away. The
  // result will be streamed through the callback.
  //
  // DEPRECATED: Prefer using the Conversation API (Conversation::SendMessage /
  // SendMessageStream) for chat and context management. For fine-grained
  // control over execution steps or single-turn inference, use RunPrefill and
  // RunDecode.
  //
  // - contents: The input data for generation.
  // - callback: Callback to receive streamed results.
  //   Note:
  //     - If the generation is done successfully, the callback will be
  //       called with empty responses to signal the completion.
  //     - If there is an error during the streaming process, the callback
  //       will be called with the error status and no further results will be
  //       sent.
  //     - If the generation is cancelled, the callback will be called
  //       with a Cancellation error.
  ABSL_DEPRECATED(
      "Prefer Conversation API for chat/context management, or RunPrefill and "
      "RunDecode for fine-grained execution control.")
  virtual absl::Status GenerateContentStream(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) = 0;

  // Same as above, but with a custom decode config.
  //
  // DEPRECATED: Prefer using the Conversation API (Conversation::SendMessage /
  // SendMessageStream) for chat and context management. For fine-grained
  // control over execution steps or single-turn inference, use RunPrefill and
  // RunDecode.
  // - decode_config: configuration for the model decode process.
  ABSL_DEPRECATED(
      "Prefer Conversation API for chat/context management, or RunPrefill and "
      "RunDecode for fine-grained execution control.")
  virtual absl::Status GenerateContentStream(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      const DecodeConfig& decode_config) = 0;

  // Scores the target text after the prefill process is done. This function
  // will only run the decode process to fetch the decode output logits, which
  // is used to calculate the target text's score and update the model memory
  // using the target_text tokens.
  // This function should be called after the prefill process is done.
  // - target_text: The target text to score.
  // - store_token_lengths: Whether to store the token lengths of the target
  //   texts in `Responses`.
  // - returns: This function returns the score associated with the target
  // text after the model has been prefilled. The returned score is the sum of
  // the negative log probability of seeing the target text during decode.
  virtual absl::StatusOr<Responses> RunTextScoring(
      const std::vector<absl::string_view>& target_text,
      bool store_token_lengths) = 0;

  // Similar to the above RunTextScoring function, but this is a not blocking
  // call and the function will return right away. The processing status will
  // be signaled through the callback.
  // - target_text: The target text to score.
  // - callback: Callback to receive the scoring results.
  // - store_token_lengths: Whether to store the token lengths of the target
  //   texts in `Responses`.
  virtual absl::StatusOr<std::unique_ptr<TaskController>> RunTextScoringAsync(
      const std::vector<absl::string_view>& target_text,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      bool store_token_lengths) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Adds the input prompt/query to the model for starting the prefilling
  // process. Note that the user can break down their prompt/query into
  // multiple chunks and call this function multiple times.
  //
  // This is a blocking call and the function will return when the prefill
  // process is done.
  virtual absl::Status RunPrefill(const std::vector<InputData>& contents) = 0;

  // This is a not blocking call and the function will return right away. The
  // processing status will be signaled through the callback.
  virtual absl::StatusOr<std::unique_ptr<TaskController>> RunPrefillAsync(
      const std::vector<InputData>& contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Similar to RunPrefillAsync, but accepts preprocessed (e.g., tokenized)
  // contents.
  // This is a non-blocking call and the function will return right away. The
  // processing status will be signaled through the callback.
  // - preprocessed_contents: The preprocessed input data.
  // - callback: Callback to receive the prefill results.
  virtual absl::StatusOr<std::unique_ptr<TaskController>>
  PrefillPreprocessedContents(
      std::vector<InputData> preprocessed_contents,
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Starts the decoding process for the model to predict the response based
  // on the input prompt/query added after using RunPrefill* functions.
  // This is a blocking call and the function will return when the decoding
  // process is done.
  virtual absl::StatusOr<Responses> RunDecode() = 0;

  // Same as above, but with a custom decode config.
  // - decode_config: configuration for the model decode process.
  virtual absl::StatusOr<Responses> RunDecode(
      const DecodeConfig& decode_config) = 0;

  // Runs the narrow, blocking exact-evidence path. The implementation accepts
  // only a loaded compiled LiteRT executor with an Engine-derived exact
  // logits contract, one explicit CPU GREEDY min-index sampler, and no logits
  // modifiers, speculative decoding, benchmark/debug callbacks, or
  // multimodal/LoRA state. Evidence includes stop/EOS IDs that ordinary
  // Responses may filter from visible output. Backend-specific profile
  // admission remains separate and may reject this evidence path.
  virtual absl::StatusOr<ExactLiteRtDecodeResult> RunExactDecode(
      int max_output_tokens) {
    (void)max_output_tokens;
    return absl::UnimplementedError("Exact LiteRT decode is not available.");
  }

  // Startes the decoding process for the model to predict the response based
  // on the input prompt/query added after using RunPrefill* functions.
  // This is a not blocking call and the function will return right away. The
  // result will be streamed through the callback.
  // - callback: Callback to receive streamed results.
  virtual absl::StatusOr<std::unique_ptr<TaskController>> RunDecodeAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Same as above, but with a custom decode config.
  // - decode_config: configuration for the model decode process.
  virtual absl::StatusOr<std::unique_ptr<TaskController>> RunDecodeAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback,
      const DecodeConfig& decode_config) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Returns the benchmark info for the session. Returns error if the
  // benchmark is not enabled.
  virtual absl::StatusOr<BenchmarkInfo> GetBenchmarkInfo() = 0;

  // Returns the mutable benchmark info for the session. Returns error if the
  // benchmark is not enabled.
  virtual absl::StatusOr<BenchmarkInfo*> GetMutableBenchmarkInfo() = 0;

  // Cancels the ongoing inference process. Note that if this function is
  // called, the inference process will return with a kCancelled error.
  //
  // NOTE: Reusing the session after calling CancelProcess() is neither
  // recommended nor supported. Calling CancelProcess() leaves the session
  // state poisoned, and subsequent operations may fail or behave incorrectly.
  virtual void CancelProcess() {
    ABSL_LOG(FATAL) << "CancelProcess is not implemented.";
  }

  // Waits until all the tasks are done or the default timeout is reached.
  virtual absl::Status WaitUntilDone() = 0;

  // Clones the session.
  // The cloned session have all the settings and context
  // of the original session up to the point that the clone function is
  // called.
  // - callback: Callback to when the streamed results.
  //
  // Example usage:
  //   Session session1 = engine->CreateSession(...);
  //   session1->Prefill("What is the tallest building ");
  //   Session session2 = session1->Clone();
  //   session1->Prefill("in the world?");
  //   session1->Decode();
  //   session2->Prefill("in France?");
  //   session2->Decode();
  virtual absl::StatusOr<std::unique_ptr<SessionInterface>> Clone() {
    return absl::UnimplementedError("Not implemented.");
  };

  // Clones the session asynchronously.
  // The cloned session have all the settings and context
  // of the original session up to the point that the clone function is
  // called.
  // - callback: Callback to when the streamed results.
  //
  // Example usage:
  //   Session session1 = engine->CreateSession(...);
  //   session1->RunPrefillAsync("What is the tallest building ", ...);
  //   Session session2 = session1->CloneAsync(...);
  //   session1->RunPrefillAsync("in the world?", ...);
  //   session1->RunDecodeAsync(...);
  //   session2->RunPrefillAsync("in France?", ...);
  //   session2->RunDecodeAsync(...);
  virtual absl::StatusOr<std::unique_ptr<SessionInterface>> CloneAsync(
      absl::AnyInvocable<void(absl::StatusOr<Responses>)> callback) {
    return absl::UnimplementedError("Not implemented.");
  };
  // Save the current step with the name `label`. You can later rewind to this
  // checkpoint using `RewindToCheckpoint(label)`. If the checkpoint name
  // already exists, the step number will be overwritten.
  virtual absl::Status SaveCheckpoint(absl::string_view label) {
    return absl::UnimplementedError("SaveCheckpoint not implemented.");
  }

  // Rewinds the session to the given checkpoint. Checkpoints after the
  // restored step will be removed. Returns an error if the checkpoint name
  // does not exist.
  virtual absl::Status RewindToCheckpoint(absl::string_view label) {
    return absl::UnimplementedError("RewindToCheckpoint not implemented.");
  }

  // Rewinds the session to a specific step number.
  virtual absl::Status RewindToStep(int step) {
    return absl::UnimplementedError("RewindToStep not implemented.");
  }

  // Get the current step of the session.
  virtual absl::StatusOr<int> GetCurrentStep() const {
    return absl::UnimplementedError("GetCurrentStep not implemented.");
  }

  // Exports a canonical authenticated snapshot for exact continuation by a
  // fresh stateful session with the same model, runtime build, and inference
  // profile. Stateless deterministic-projection sessions intentionally do not
  // admit capsule handoff.
  virtual absl::StatusOr<std::string> ExportHandoff(
      const SessionHandoffOptions& options) {
    std::string envelope;
    StringByteSink sink(&envelope);
    absl::StatusOr<SessionContinuationStateWitness> witness =
        ExportHandoffToWithWitness(options, &sink);
    if (!witness.ok()) return witness.status();
    return envelope;
  }

  // Streams the envelope without a second full KV-state allocation and
  // returns runtime-derived evidence over the exact bytes actually accepted
  // by the sink. Implementations that cannot expose a complete live-state
  // witness fail closed rather than synthesizing one from caller metadata.
  virtual absl::StatusOr<SessionContinuationStateWitness>
  ExportHandoffToWithWitness(const SessionHandoffOptions& options,
                             ByteSink* sink) {
    (void)options;
    (void)sink;
    return absl::UnimplementedError(
        "Session handoff export witness is not available.");
  }

  // Compatibility adapter that discards the independently validated witness.
  virtual absl::Status ExportHandoffTo(const SessionHandoffOptions& options,
                                       ByteSink* sink) {
    absl::StatusOr<SessionContinuationStateWitness> witness =
        ExportHandoffToWithWitness(options, sink);
    return witness.ok() ? absl::OkStatus() : witness.status();
  }

  // Imports into a fresh compatible stateful session after authenticating and
  // validating the complete envelope.
  virtual absl::Status ImportHandoff(absl::string_view envelope,
                                     const SessionHandoffOptions& expected) {
    StringByteSource source(envelope);
    absl::StatusOr<SessionContinuationStateWitness> witness =
        ImportHandoffFromWithWitness(source, expected);
    return witness.ok() ? absl::OkStatus() : witness.status();
  }

  // Authenticates and transactionally imports the source, then independently
  // re-exports the committed live target into a digest-only sink. The returned
  // witness therefore cannot be copied from or supplied by incoming bytes.
  // Authentication, decoding, and state-load failures leave the fresh target
  // unchanged. A failure of the post-commit live-state re-export is an
  // integrity failure; callers must discard that target session.
  virtual absl::StatusOr<SessionContinuationStateWitness>
  ImportHandoffFromWithWitness(
      const ByteSource& envelope, const SessionHandoffOptions& expected) {
    (void)envelope;
    (void)expected;
    return absl::UnimplementedError(
        "Session handoff import witness is not available.");
  }

  // Compatibility adapter that discards the independently recomputed witness.
  virtual absl::Status ImportHandoffFrom(
      const ByteSource& envelope, const SessionHandoffOptions& expected) {
    absl::StatusOr<SessionContinuationStateWitness> witness =
        ImportHandoffFromWithWitness(envelope, expected);
    return witness.ok() ? absl::OkStatus() : witness.status();
  }

  // Get the reference to the session config for the session.
  virtual const SessionConfig& GetSessionConfig() const = 0;

  // Returns the engine-derived identity owned by this concrete session.
  // Implementations that cannot bind the session to measured model,
  // runtime/delegate, and resolved-profile evidence fail closed.
  virtual absl::StatusOr<SessionHandoffIdentity> GetSessionHandoffIdentity()
      const {
    return absl::UnimplementedError(
        "Engine-derived session handoff identity is not available.");
  }

  // Returns each candidate's exact executor token history, including a
  // backend pending token when present. Unlike Responses::GetTokenIds(), this
  // history is not filtered for stop sequences or detokenizer output.
  virtual absl::StatusOr<std::vector<std::vector<int>>>
  GetExactProcessedTokenHistory() const {
    return absl::UnimplementedError(
        "Exact processed-token history is not available.");
  }

  // Returns the debug info for the session, or nullopt if unsupported
  // or debugger is disabled.
  virtual std::optional<SessionDebugInfo> GetSessionDebugInfo() const {
    return std::nullopt;
  }
};

// EngineT is the templated interface for the LLM runtime.
//
// By templating on `SessionT`, this interface allows custom implementations
// of the engine to yield specialized session types that don't necessarily
// adhere to or inherit from the default `SessionInterface`.
//
// This is particularly useful for advanced or custom use cases where consumers
// need access to specialized methods or extended state tracking not exposed by
// standard interfaces. It allows users to interact with custom sessions
// directly without having to perform dynamic casting or type erasure on the
// returned session pointer.
template <typename SessionT>
class EngineT {
 public:
  virtual ~EngineT() = default;

  using Session = SessionT;

  // Method to create the Session.
  virtual absl::StatusOr<std::unique_ptr<SessionT>> CreateSession(
      const SessionConfig& session_config) = 0;

  // Waits until the engine is done with all the tasks. The function will
  // return error if the timeout is reached.
  virtual absl::Status WaitUntilDone(absl::Duration timeout) {
    return absl::UnimplementedError("Not implemented.");
  }

  // Returns the EngineSettings currently used by the engine.
  virtual const EngineSettings& GetEngineSettings() const = 0;

  // Get the reference to the tokenizer for the engine.
  virtual const support::Tokenizer& GetTokenizer() const = 0;

  // Derives the authoritative handoff identity from the exact artifacts and
  // concrete runtime already loaded by this engine plus a fully-resolved copy
  // of `session_config`. The default preserves source compatibility while
  // failing closed for engines that do not own measurable identity evidence.
  virtual absl::StatusOr<SessionHandoffIdentity>
  ResolveSessionHandoffIdentity(const SessionConfig& session_config) const {
    (void)session_config;
    return absl::UnimplementedError(
        "Engine-derived session handoff identity is not available.");
  }

  // Reports whether this loaded Engine can derive an ExactLiteRtProfile.
  // Candidate availability is not ExactRegeneration admission; admission
  // requires a separate independent cold-process equality record.
  virtual ExactLiteRtProfileCapability GetExactLiteRtProfileCapability()
      const {
    return {};
  }

  // Derives an exact-profile identity from the already-loaded Engine and a
  // fully-resolved SessionConfig. `assertion` can only reject the derived
  // result; it cannot supply model, backend, runtime, or profile identity.
  virtual absl::StatusOr<ExactLiteRtProfile> ResolveExactLiteRtProfile(
      const SessionConfig& session_config,
      const ExactLiteRtProfileAssertion& assertion = {}) const {
    (void)session_config;
    (void)assertion;
    return absl::UnimplementedError(
        "Engine-derived exact LiteRT profile is not available.");
  }

  // Derives the complete session-capsule capability from the loaded Engine,
  // a fully-resolved SessionConfig, the exact profile for that configuration,
  // and executor-owned state inventory evidence. Assertions can only reject
  // the result; they cannot supply or upgrade capability evidence.
  virtual absl::StatusOr<SessionHandoffCapability>
  ResolveSessionHandoffCapability(
      const SessionConfig& session_config,
      const SessionHandoffCapabilityAssertion& assertion = {}) const {
    (void)session_config;
    (void)assertion;
    return absl::UnimplementedError(
        "Engine-derived session handoff capability is not available.");
  }

  // Get the audio model properties for the session. This is only available
  // if the engine is created with audio modality enabled.
  virtual absl::StatusOr<AudioExecutorProperties> GetAudioExecutorProperties()
      const = 0;

  // Get the vision model properties for the session. This is only available
  // if the engine is created with vision modality enabled.
  virtual absl::StatusOr<VisionExecutorProperties> GetVisionExecutorProperties()
      const = 0;

  // Default timeout duration for the engine/session processes.
  static constexpr absl::Duration kDefaultTimeout = absl::Minutes(10);
};

// Default Engine implementation using the standard SessionInterface.
//
// This alias maintains backward compatibility for existing code that expects a
// non-templated `Engine` type.
//
// DESIGN PATTERN:
// - Most standard use cases should use this `Engine` alias and the
//   corresponding
//   `SessionInterface` (accessible as `Engine::Session`).
// - Advanced users requiring custom Session APIs can instantiate `EngineT` with
//   their custom Session type (e.g., `EngineT<MyCustomSession>`). This allows
//   extending the Session capability without altering the core Engine interface
//   or resort to runtime type checks/casting.
using Engine = EngineT<SessionInterface>;

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_ENGINE_ENGINE_H_
