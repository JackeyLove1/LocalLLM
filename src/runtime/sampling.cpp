#include "localllm/runtime/sampling.h"

#include <limits>

#include "localllm/common/status.h"

namespace localllm {

std::int64_t SampleNextToken(const torch::Tensor& logits, const SamplingParams& params) {
  LOCALLLM_CHECK(logits.dim() == 1, "SampleNextToken expects a 1D logits tensor.");
  torch::Tensor scores = logits.to(torch::kFloat32).clone();

  if (params.temperature <= 0.0) {
    return scores.argmax().item<std::int64_t>();
  }

  scores = scores / params.temperature;
  if (params.top_k > 0 && params.top_k < scores.size(0)) {
    const auto topk = torch::topk(scores, params.top_k);
    const auto threshold = std::get<0>(topk).min().item<float>();
    scores = scores.masked_fill(scores < threshold, -std::numeric_limits<float>::infinity());
  }

  torch::Tensor probabilities = torch::softmax(scores, -1);
  if (params.top_p < 1.0) {
    auto sorted = torch::sort(probabilities, -1, true);
    torch::Tensor sorted_probs = std::get<0>(sorted);
    torch::Tensor sorted_indices = std::get<1>(sorted);
    torch::Tensor cumulative = torch::cumsum(sorted_probs, -1);
    torch::Tensor to_remove = cumulative > params.top_p;
    to_remove.index_put_({0}, false);
    sorted_probs = sorted_probs.masked_fill(to_remove, 0.0);
    probabilities = torch::zeros_like(probabilities).scatter(0, sorted_indices, sorted_probs);
    probabilities = probabilities / probabilities.sum();
  }

  torch::manual_seed(static_cast<std::int64_t>(params.seed));
  const auto sampled = torch::multinomial(probabilities, 1);
  return sampled.item<std::int64_t>();
}

}  // namespace localllm
