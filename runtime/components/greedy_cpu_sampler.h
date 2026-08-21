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

#ifndef THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_GREEDY_CPU_SAMPLER_H_
#define THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_GREEDY_CPU_SAMPLER_H_

#include <memory>
#include <random>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/components/sampler.h"
#include "runtime/proto/sampler_params.pb.h"

namespace litert::lm {

// A deterministic CPU argmax sampler. It consumes no random state and resolves
// equal maxima by choosing the lowest vocabulary index.
class GreedyCpuSampler final : public Sampler {
 public:
  static absl::StatusOr<std::unique_ptr<GreedyCpuSampler>> Create(
      int batch_size, int sequence_size,
      const proto::SamplerParameters& sampler_params);

  absl::Status SampleToIdAndScoreBuffer(
      const TensorBuffer& logits_tensor, TensorBuffer& ids_tensor,
      TensorBuffer* scores_tensor) override;

  absl::Status UpdateConfig(
      const proto::SamplerParameters& sampler_params, int batch_size,
      std::shared_ptr<std::default_random_engine> rand_gen) override;

 private:
  GreedyCpuSampler(int batch_size, int sequence_size)
      : batch_size_(batch_size), sequence_size_(sequence_size) {}

  int batch_size_;
  const int sequence_size_;
};

}  // namespace litert::lm

#endif  // THIRD_PARTY_ODML_LITERT_LM_RUNTIME_COMPONENTS_GREEDY_CPU_SAMPLER_H_
