#pragma once

#include <filesystem>
#include <vector>

#include <torch/torch.h>

#include "localllm/model/qwen_config.h"

namespace localllm {

struct AttentionWeights {
  torch::Tensor q_proj;
  torch::Tensor k_proj;
  torch::Tensor v_proj;
  torch::Tensor o_proj;
  torch::Tensor q_norm;
  torch::Tensor k_norm;
};

struct MlpWeights {
  torch::Tensor gate_proj;
  torch::Tensor up_proj;
  torch::Tensor down_proj;
};

struct LayerWeights {
  torch::Tensor input_layernorm;
  torch::Tensor post_attention_layernorm;
  AttentionWeights attention;
  MlpWeights mlp;
};

struct ModelWeights {
  torch::Tensor embed_tokens;
  torch::Tensor norm;
  torch::Tensor lm_head;
  std::vector<LayerWeights> layers;
  torch::Device device = torch::kCPU;
  torch::ScalarType dtype = torch::kFloat32;
};

ModelWeights LoadWeights(
    const std::filesystem::path& model_dir, const QwenConfig& config, const torch::Device& device);

}  // namespace localllm
