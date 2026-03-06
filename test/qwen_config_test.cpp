#include <filesystem>

#include <gtest/gtest.h>

#include "localllm/model/qwen_config.h"

namespace localllm {
namespace {

TEST(QwenConfigTest, LoadsExpectedFields) {
  const auto config = LoadConfig(std::filesystem::path("models/Qwen3-0.6B"));
  EXPECT_EQ(config.vocab_size, 151936);
  EXPECT_EQ(config.hidden_size, 1024);
  EXPECT_EQ(config.intermediate_size, 3072);
  EXPECT_EQ(config.num_hidden_layers, 28);
  EXPECT_EQ(config.num_attention_heads, 16);
  EXPECT_EQ(config.num_key_value_heads, 8);
  EXPECT_EQ(config.head_dim, 128);
  EXPECT_EQ(config.eos_token_id, 151645);
  EXPECT_TRUE(config.tie_word_embeddings);
}

}  // namespace
}  // namespace localllm
