#include "localllm/model/qwen_config.h"

#include "json11.hpp"

#include "localllm/common/file_util.h"
#include "localllm/common/logging.h"
#include "localllm/common/status.h"

namespace localllm {

QwenConfig LoadConfig(const std::filesystem::path& model_dir) {
  const auto config_text = ReadTextFile(model_dir / "config.json");
  std::string error;
  const auto json = json11::Json::parse(config_text, error);
  LOCALLLM_CHECK(error.empty(), "Failed to parse config.json: " + error);

  QwenConfig config;
  config.vocab_size = json["vocab_size"].int_value();
  config.hidden_size = json["hidden_size"].int_value();
  config.intermediate_size = json["intermediate_size"].int_value();
  config.num_hidden_layers = json["num_hidden_layers"].int_value();
  config.num_attention_heads = json["num_attention_heads"].int_value();
  config.num_key_value_heads = json["num_key_value_heads"].int_value();
  config.head_dim = json["head_dim"].int_value();
  config.max_position_embeddings = json["max_position_embeddings"].int_value();
  config.bos_token_id = json["bos_token_id"].is_null() ? -1 : json["bos_token_id"].int_value();
  config.eos_token_id = json["eos_token_id"].int_value();
  config.rms_norm_eps = json["rms_norm_eps"].number_value();
  config.rope_theta = json["rope_theta"].number_value();
  config.tie_word_embeddings = json["tie_word_embeddings"].bool_value();
  config.attention_bias = json["attention_bias"].bool_value();

  LOCALLLM_CHECK(config.hidden_size > 0, "Invalid hidden_size in config.json");
  LOCALLLM_CHECK(config.num_hidden_layers > 0, "Invalid num_hidden_layers in config.json");
  LOCALLLM_CHECK(config.num_attention_heads > 0, "Invalid num_attention_heads in config.json");
  LOCALLLM_CHECK(config.num_key_value_heads > 0, "Invalid num_key_value_heads in config.json");
  LOCALLLM_CHECK(config.head_dim > 0, "Invalid head_dim in config.json");
  LogInfo(
      "Loaded model config: vocab_size=" + std::to_string(config.vocab_size) +
      ", hidden_size=" + std::to_string(config.hidden_size) +
      ", layers=" + std::to_string(config.num_hidden_layers) +
      ", attention_heads=" + std::to_string(config.num_attention_heads) +
      ", kv_heads=" + std::to_string(config.num_key_value_heads) +
      ", head_dim=" + std::to_string(config.head_dim));
  return config;
}

}  // namespace localllm
