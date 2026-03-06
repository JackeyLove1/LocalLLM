#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace localllm {

struct SamplingParams {
  std::int64_t max_new_tokens = 64;
  double temperature = 0.0;
  double top_p = 1.0;
  std::int64_t top_k = 0;
  std::uint64_t seed = 42;
  bool stream = true;
  bool print_token_ids = false;
  std::vector<std::string> stop_strings;
};

std::int64_t SampleNextToken(const torch::Tensor& logits, const SamplingParams& params);

}  // namespace localllm
