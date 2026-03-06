#pragma once

#include <cstdint>
#include <vector>

#include <torch/torch.h>

#include "localllm/model/model_weights.h"
#include "localllm/model/qwen_config.h"
#include "localllm/runtime/sampling.h"

namespace localllm {

struct ExecutorResult {
  torch::Tensor logits;
};

struct GenerationResult {
  std::vector<std::int64_t> prompt_token_ids;
  std::vector<std::int64_t> generated_token_ids;
  std::vector<std::int64_t> all_token_ids;
  std::string generated_text;
};

class BpeTokenizer;

class QwenExecutor {
 public:
  QwenExecutor(QwenConfig config, ModelWeights weights);

  ExecutorResult Prefill(const std::vector<std::int64_t>& prompt_token_ids);
  ExecutorResult DecodeNext(std::int64_t token_id);
  GenerationResult Generate(
      const std::vector<std::int64_t>& prompt_token_ids,
      const BpeTokenizer& tokenizer,
      const SamplingParams& params);

 private:
  struct LayerCache {
    torch::Tensor key;
    torch::Tensor value;
  };

  torch::Tensor ForwardChunk(const std::vector<std::int64_t>& token_ids, bool append_to_cache);
  torch::Tensor ApplyRmsNorm(const torch::Tensor& x, const torch::Tensor& weight, double eps) const;
  torch::Tensor Linear(const torch::Tensor& x, const torch::Tensor& weight) const;
  std::pair<torch::Tensor, torch::Tensor> ApplyRotary(
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& positions) const;
  torch::Tensor BuildCausalMask(std::int64_t query_length, std::int64_t key_length, std::int64_t past_length) const;
  void ResetState();

  QwenConfig config_;
  ModelWeights weights_;
  std::vector<LayerCache> cache_;
  std::vector<std::int64_t> prompt_token_ids_;
  std::vector<std::int64_t> generated_token_ids_;
};

}  // namespace localllm
