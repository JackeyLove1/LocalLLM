#pragma once

#include <cstdint>
#include <filesystem>

namespace localllm {

struct QwenConfig {
  std::int64_t vocab_size = 0;
  std::int64_t hidden_size = 0;
  std::int64_t intermediate_size = 0;
  std::int64_t num_hidden_layers = 0;
  std::int64_t num_attention_heads = 0;
  std::int64_t num_key_value_heads = 0;
  std::int64_t head_dim = 0;
  std::int64_t max_position_embeddings = 0;
  std::int64_t bos_token_id = -1;
  std::int64_t eos_token_id = -1;
  double rms_norm_eps = 1e-6;
  double rope_theta = 1000000.0;
  bool tie_word_embeddings = true;
  bool attention_bias = false;
};

QwenConfig LoadConfig(const std::filesystem::path& model_dir);

}  // namespace localllm
