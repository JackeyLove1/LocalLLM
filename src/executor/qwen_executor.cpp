#include "localllm/executor/qwen_executor.h"

#include <cmath>
#include <iostream>

#include "localllm/common/logging.h"
#include "localllm/common/status.h"
#include "localllm/tokenizer/bpe_tokenizer.h"

namespace localllm {

namespace {

torch::Tensor RotateHalf(const torch::Tensor& tensor) {
  const auto split = tensor.size(-1) / 2;
  auto first = tensor.slice(-1, 0, split);
  auto second = tensor.slice(-1, split);
  return torch::cat({-second, first}, -1);
}

torch::Tensor RepeatKvHeads(const torch::Tensor& tensor, std::int64_t groups) {
  if (groups == 1) {
    return tensor;
  }
  return tensor.repeat_interleave(groups, 0);
}

}  // namespace

QwenExecutor::QwenExecutor(QwenConfig config, ModelWeights weights)
    : config_(std::move(config)), weights_(std::move(weights)) {
  ResetState();
}

void QwenExecutor::ResetState() {
  cache_.assign(static_cast<std::size_t>(config_.num_hidden_layers), LayerCache{});
  prompt_token_ids_.clear();
  generated_token_ids_.clear();
}

torch::Tensor QwenExecutor::ApplyRmsNorm(
    const torch::Tensor& x, const torch::Tensor& weight, double eps) const {
  auto hidden = x.to(torch::kFloat32);
  const auto variance = hidden.pow(2).mean(-1, true);
  hidden = hidden * torch::rsqrt(variance + eps);
  return hidden * weight.to(torch::kFloat32);
}

torch::Tensor QwenExecutor::Linear(const torch::Tensor& x, const torch::Tensor& weight) const {
  return torch::matmul(x, weight.to(torch::kFloat32).transpose(0, 1));
}

std::pair<torch::Tensor, torch::Tensor> QwenExecutor::ApplyRotary(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& positions) const {
  auto inv_freq =
      torch::arange(0, config_.head_dim, 2, torch::TensorOptions().dtype(torch::kFloat32)) /
      static_cast<float>(config_.head_dim);
  inv_freq = torch::pow(torch::tensor(config_.rope_theta, torch::kFloat32), -inv_freq);
  auto freqs = torch::outer(positions.to(torch::kFloat32), inv_freq);
  auto emb = torch::cat({freqs, freqs}, -1);
  auto cos = emb.cos().unsqueeze(0);
  auto sin = emb.sin().unsqueeze(0);
  return {
      query * cos + RotateHalf(query) * sin,
      key * cos + RotateHalf(key) * sin,
  };
}

torch::Tensor QwenExecutor::BuildCausalMask(
    std::int64_t query_length, std::int64_t key_length, std::int64_t past_length) const {
  auto query_positions =
      torch::arange(past_length, past_length + query_length, torch::TensorOptions().dtype(torch::kInt64)).unsqueeze(1);
  auto key_positions = torch::arange(0, key_length, torch::TensorOptions().dtype(torch::kInt64)).unsqueeze(0);
  return (key_positions > query_positions).to(torch::kBool);
}

torch::Tensor QwenExecutor::ForwardChunk(const std::vector<std::int64_t>& token_ids, bool append_to_cache) {
  LOCALLLM_CHECK(!token_ids.empty(), "ForwardChunk requires at least one token.");
  auto token_tensor = torch::tensor(token_ids, torch::TensorOptions().dtype(torch::kInt64));
  auto hidden_states = weights_.embed_tokens.index_select(0, token_tensor).to(torch::kFloat32);

  for (std::int64_t layer_index = 0; layer_index < config_.num_hidden_layers; ++layer_index) {
    auto& layer = weights_.layers[static_cast<std::size_t>(layer_index)];
    auto& cache = cache_[static_cast<std::size_t>(layer_index)];

    const auto residual = hidden_states;
    auto normed = ApplyRmsNorm(hidden_states, layer.input_layernorm, config_.rms_norm_eps);

    auto q = Linear(normed, layer.attention.q_proj)
                 .view({static_cast<std::int64_t>(token_ids.size()), config_.num_attention_heads, config_.head_dim})
                 .permute({1, 0, 2})
                 .contiguous();
    auto k = Linear(normed, layer.attention.k_proj)
                 .view({static_cast<std::int64_t>(token_ids.size()), config_.num_key_value_heads, config_.head_dim})
                 .permute({1, 0, 2})
                 .contiguous();
    auto v = Linear(normed, layer.attention.v_proj)
                 .view({static_cast<std::int64_t>(token_ids.size()), config_.num_key_value_heads, config_.head_dim})
                 .permute({1, 0, 2})
                 .contiguous();

    q = ApplyRmsNorm(q, layer.attention.q_norm, config_.rms_norm_eps);
    k = ApplyRmsNorm(k, layer.attention.k_norm, config_.rms_norm_eps);

    const auto past_length = cache.key.defined() ? cache.key.size(1) : 0;
    auto positions = torch::arange(
        past_length,
        past_length + static_cast<std::int64_t>(token_ids.size()),
        torch::TensorOptions().dtype(torch::kInt64));
    std::tie(q, k) = ApplyRotary(q, k, positions);

    torch::Tensor key_states = k;
    torch::Tensor value_states = v;
    if (append_to_cache) {
      if (cache.key.defined()) {
        cache.key = torch::cat({cache.key, k}, 1);
        cache.value = torch::cat({cache.value, v}, 1);
      } else {
        cache.key = k.clone();
        cache.value = v.clone();
      }
      key_states = cache.key;
      value_states = cache.value;
    }

    key_states = RepeatKvHeads(key_states, config_.num_attention_heads / config_.num_key_value_heads);
    value_states = RepeatKvHeads(value_states, config_.num_attention_heads / config_.num_key_value_heads);
    auto attn_scores =
        torch::matmul(q, key_states.transpose(1, 2)) / std::sqrt(static_cast<double>(config_.head_dim));

    const auto total_key_length = key_states.size(1);
    const auto mask = BuildCausalMask(
                          static_cast<std::int64_t>(token_ids.size()), total_key_length, past_length)
                          .unsqueeze(0);
    attn_scores = attn_scores.masked_fill(mask, -1e30f);
    const auto attn_probs = torch::softmax(attn_scores, -1);
    auto attn_output = torch::matmul(attn_probs, value_states)
                           .permute({1, 0, 2})
                           .contiguous()
                           .view({static_cast<std::int64_t>(token_ids.size()),
                                  config_.num_attention_heads * config_.head_dim});
    attn_output = Linear(attn_output, layer.attention.o_proj);
    hidden_states = residual + attn_output;

    const auto mlp_residual = hidden_states;
    auto mlp_hidden = ApplyRmsNorm(hidden_states, layer.post_attention_layernorm, config_.rms_norm_eps);
    const auto gate = torch::silu(Linear(mlp_hidden, layer.mlp.gate_proj));
    const auto up = Linear(mlp_hidden, layer.mlp.up_proj);
    const auto down = Linear(gate * up, layer.mlp.down_proj);
    hidden_states = mlp_residual + down;
  }

  hidden_states = ApplyRmsNorm(hidden_states, weights_.norm, config_.rms_norm_eps);
  return Linear(hidden_states, weights_.lm_head);
}

ExecutorResult QwenExecutor::Prefill(const std::vector<std::int64_t>& prompt_token_ids) {
  LogInfo("Starting prefill with " + std::to_string(prompt_token_ids.size()) + " prompt tokens.");
  ResetState();
  prompt_token_ids_ = prompt_token_ids;
  return ExecutorResult{ForwardChunk(prompt_token_ids, true)};
}

ExecutorResult QwenExecutor::DecodeNext(std::int64_t token_id) {
  generated_token_ids_.push_back(token_id);
  return ExecutorResult{ForwardChunk({token_id}, true)};
}

GenerationResult QwenExecutor::Generate(
    const std::vector<std::int64_t>& prompt_token_ids,
    const BpeTokenizer& tokenizer,
    const SamplingParams& params) {
  LogInfo(
      "Entering generation loop: max_new_tokens=" + std::to_string(params.max_new_tokens) +
      ", temperature=" + std::to_string(params.temperature) +
      ", top_p=" + std::to_string(params.top_p) +
      ", top_k=" + std::to_string(params.top_k) +
      ", stream=" + std::string(params.stream ? "true" : "false"));
  auto logits = Prefill(prompt_token_ids).logits;
  std::string generated_text;
  bool stop_requested = false;

  for (std::int64_t step = 0; step < params.max_new_tokens && !stop_requested; ++step) {
    const auto last_logits = logits.index({logits.size(0) - 1});
    const auto next_token = SampleNextToken(last_logits, params);
    const auto token_text = tokenizer.Decode({next_token}, false);
    generated_text += token_text;

    if (params.stream) {
      std::cout << token_text << std::flush;
      if (params.print_token_ids) {
        LogInfo("Generated token id: " + std::to_string(next_token));
      }
    }

    if (next_token == config_.eos_token_id) {
      generated_token_ids_.push_back(next_token);
      LogInfo("Stopping generation after EOS token at step " + std::to_string(step + 1) + ".");
      break;
    }
    for (const auto& stop : params.stop_strings) {
      if (!stop.empty() && generated_text.size() >= stop.size() &&
          generated_text.compare(generated_text.size() - stop.size(), stop.size(), stop) == 0) {
        stop_requested = true;
        LogInfo("Stopping generation after stop string match at step " + std::to_string(step + 1) + ".");
        break;
      }
    }
    logits = DecodeNext(next_token).logits;
  }

  std::vector<std::int64_t> all = prompt_token_ids_;
  all.insert(all.end(), generated_token_ids_.begin(), generated_token_ids_.end());
  LogInfo(
      "Generation completed: prompt_tokens=" + std::to_string(prompt_token_ids_.size()) +
      ", generated_tokens=" + std::to_string(generated_token_ids_.size()) +
      ", total_tokens=" + std::to_string(all.size()));
  return GenerationResult{prompt_token_ids_, generated_token_ids_, all, generated_text};
}

}  // namespace localllm
