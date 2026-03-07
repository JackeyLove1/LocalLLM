#include "localllm/model/qwen_config.h"

#include <filesystem>
#include <memory>

#include <gtest/gtest.h>

namespace localllm {
namespace {

constexpr char kModelDir[] = "models/Qwen3-0.6B";

class QwenConfigTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    config_ = std::make_unique<QwenConfig>(LoadConfig(std::filesystem::path(kModelDir)));
  }

  static void TearDownTestSuite() { config_.reset(); }

  const QwenConfig& config() const { return *config_; }

 private:
  static std::unique_ptr<QwenConfig> config_;
};

std::unique_ptr<QwenConfig> QwenConfigTest::config_;

TEST_F(QwenConfigTest, LoadsCoreDimensions) {
  EXPECT_EQ(config().vocab_size, 151936);
  EXPECT_EQ(config().hidden_size, 1024);
  EXPECT_EQ(config().intermediate_size, 3072);
  EXPECT_EQ(config().num_hidden_layers, 28);
  EXPECT_EQ(config().max_position_embeddings, 40960);
}

TEST_F(QwenConfigTest, LoadsAttentionLayout) {
  EXPECT_EQ(config().num_attention_heads, 16);
  EXPECT_EQ(config().num_key_value_heads, 8);
  EXPECT_EQ(config().head_dim, 128);
  EXPECT_FALSE(config().attention_bias);
}

TEST_F(QwenConfigTest, LoadsSpecialTokensAndNumericalFields) {
  EXPECT_EQ(config().bos_token_id, 151643);
  EXPECT_EQ(config().eos_token_id, 151645);
  EXPECT_DOUBLE_EQ(config().rms_norm_eps, 1e-6);
  EXPECT_DOUBLE_EQ(config().rope_theta, 1000000.0);
  EXPECT_TRUE(config().tie_word_embeddings);
}

}  // namespace
}  // namespace localllm
