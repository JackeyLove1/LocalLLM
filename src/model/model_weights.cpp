#include "localllm/model/model_weights.h"

#include "localllm/common/logging.h"
#include "localllm/common/status.h"
#include "localllm/model/safetensors_loader.h"

namespace localllm {

namespace {

std::string ShapeToString(const torch::Tensor& tensor) {
  std::string text = "[";
  const auto sizes = tensor.sizes();
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) {
      text += ", ";
    }
    text += std::to_string(sizes[i]);
  }
  text += "]";
  return text;
}

void CheckShape(
    const std::string& name, const torch::Tensor& tensor, const std::vector<std::int64_t>& expected_shape) {
  LOCALLLM_CHECK(
      tensor.sizes().vec() == expected_shape,
      "Unexpected tensor shape for " + name + ": got " + ShapeToString(tensor));
}

torch::ScalarType ResolveWeightDtype(const torch::Device& device) {
  return device.is_cuda() ? torch::kBFloat16 : torch::kFloat32;
}

std::string DeviceToString(const torch::Device& device) {
  return device.is_cuda() ? "cuda" : "cpu";
}

std::string ScalarTypeToString(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kBFloat16:
      return "bfloat16";
    case torch::kFloat32:
      return "float32";
    default:
      return "unknown";
  }
}

}  // namespace

ModelWeights LoadWeights(
    const std::filesystem::path& model_dir, const QwenConfig& config, const torch::Device& device) {
  const auto weights_path = model_dir / "model.safetensors";
  const auto weight_dtype = ResolveWeightDtype(device);
  LOG(INFO) << "Loading weights from " << weights_path.string()
            << " on device=" << DeviceToString(device)
            << " with dtype=" << ScalarTypeToString(weight_dtype);
  SafeTensorFile safetensors(weights_path);

  ModelWeights weights;
  weights.device = device;
  weights.dtype = weight_dtype;
  weights.embed_tokens =
      safetensors.LoadTensor("model.embed_tokens.weight", weights.dtype, device);
  weights.norm = safetensors.LoadTensor("model.norm.weight", weights.dtype, device);
  weights.lm_head = config.tie_word_embeddings
                        ? weights.embed_tokens
                        : safetensors.LoadTensor("lm_head.weight", weights.dtype, device);

  CheckShape("model.embed_tokens.weight", weights.embed_tokens, {config.vocab_size, config.hidden_size});
  CheckShape("model.norm.weight", weights.norm, {config.hidden_size});
  CheckShape("lm_head.weight", weights.lm_head, {config.vocab_size, config.hidden_size});

  weights.layers.reserve(config.num_hidden_layers);
  for (std::int64_t layer_index = 0; layer_index < config.num_hidden_layers; ++layer_index) {
    const std::string prefix = "model.layers." + std::to_string(layer_index);

    LayerWeights layer;
    layer.input_layernorm =
        safetensors.LoadTensor(prefix + ".input_layernorm.weight", weights.dtype, device);
    layer.post_attention_layernorm =
        safetensors.LoadTensor(prefix + ".post_attention_layernorm.weight", weights.dtype, device);
    layer.attention.q_norm = safetensors.LoadTensor(prefix + ".self_attn.q_norm.weight", weights.dtype, device);
    layer.attention.k_norm = safetensors.LoadTensor(prefix + ".self_attn.k_norm.weight", weights.dtype, device);
    layer.attention.q_proj = safetensors.LoadTensor(prefix + ".self_attn.q_proj.weight", weights.dtype, device);
    layer.attention.k_proj = safetensors.LoadTensor(prefix + ".self_attn.k_proj.weight", weights.dtype, device);
    layer.attention.v_proj = safetensors.LoadTensor(prefix + ".self_attn.v_proj.weight", weights.dtype, device);
    layer.attention.o_proj = safetensors.LoadTensor(prefix + ".self_attn.o_proj.weight", weights.dtype, device);
    layer.mlp.gate_proj = safetensors.LoadTensor(prefix + ".mlp.gate_proj.weight", weights.dtype, device);
    layer.mlp.up_proj = safetensors.LoadTensor(prefix + ".mlp.up_proj.weight", weights.dtype, device);
    layer.mlp.down_proj = safetensors.LoadTensor(prefix + ".mlp.down_proj.weight", weights.dtype, device);

    CheckShape(prefix + ".input_layernorm.weight", layer.input_layernorm, {config.hidden_size});
    CheckShape(prefix + ".post_attention_layernorm.weight", layer.post_attention_layernorm, {config.hidden_size});
    CheckShape(
        prefix + ".self_attn.q_proj.weight",
        layer.attention.q_proj,
        {config.num_attention_heads * config.head_dim, config.hidden_size});
    CheckShape(
        prefix + ".self_attn.k_proj.weight",
        layer.attention.k_proj,
        {config.num_key_value_heads * config.head_dim, config.hidden_size});
    CheckShape(
        prefix + ".self_attn.v_proj.weight",
        layer.attention.v_proj,
        {config.num_key_value_heads * config.head_dim, config.hidden_size});
    CheckShape(
        prefix + ".self_attn.o_proj.weight",
        layer.attention.o_proj,
        {config.hidden_size, config.num_attention_heads * config.head_dim});
    CheckShape(prefix + ".self_attn.q_norm.weight", layer.attention.q_norm, {config.head_dim});
    CheckShape(prefix + ".self_attn.k_norm.weight", layer.attention.k_norm, {config.head_dim});
    CheckShape(prefix + ".mlp.gate_proj.weight", layer.mlp.gate_proj, {config.intermediate_size, config.hidden_size});
    CheckShape(prefix + ".mlp.up_proj.weight", layer.mlp.up_proj, {config.intermediate_size, config.hidden_size});
    CheckShape(prefix + ".mlp.down_proj.weight", layer.mlp.down_proj, {config.hidden_size, config.intermediate_size});

    weights.layers.push_back(std::move(layer));
  }

  LOG(INFO) << "Finished loading weights: layers=" << weights.layers.size()
            << ", device=" << DeviceToString(weights.device)
            << ", dtype=" << ScalarTypeToString(weights.dtype);
  return weights;
}

}  // namespace localllm
