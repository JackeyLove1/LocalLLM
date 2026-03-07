#include "localllm/executor/qwen_executor.h"
#include "localllm/model/model_weights.h"
#include "localllm/model/qwen_config.h"
#include "localllm/tokenizer/bpe_tokenizer.h"

#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace localllm {
namespace {

constexpr char kModelDir[] = "models/Qwen3-0.6B";

class QwenExecutorSmokeTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    const auto model_dir = std::filesystem::path(kModelDir);
    config_ = std::make_unique<QwenConfig>(LoadConfig(model_dir));
    tokenizer_ = std::make_unique<BpeTokenizer>(BpeTokenizer::LoadFromDir(model_dir));
    weights_ = std::make_unique<ModelWeights>(LoadWeights(model_dir, *config_, torch::kCPU));
  }

  static void TearDownTestSuite() {
    weights_.reset();
    tokenizer_.reset();
    config_.reset();
  }

  QwenExecutor CreateExecutor() const { return QwenExecutor(*config_, *weights_); }

  const QwenConfig& config() const { return *config_; }
  const BpeTokenizer& tokenizer() const { return *tokenizer_; }

  std::vector<std::int64_t> EncodePrompt(std::string_view prompt) const {
    return tokenizer().Encode(std::string(prompt));
  }

 private:
  static std::unique_ptr<QwenConfig> config_;
  static std::unique_ptr<BpeTokenizer> tokenizer_;
  static std::unique_ptr<ModelWeights> weights_;
};

std::unique_ptr<QwenConfig> QwenExecutorSmokeTest::config_;
std::unique_ptr<BpeTokenizer> QwenExecutorSmokeTest::tokenizer_;
std::unique_ptr<ModelWeights> QwenExecutorSmokeTest::weights_;

TEST_F(QwenExecutorSmokeTest, PrefillReturnsLogitsWithVocabularyWidth) {
  auto executor = CreateExecutor();
  const auto prompt_ids = EncodePrompt("Hello");
  ASSERT_FALSE(prompt_ids.empty());

  const auto prefill = executor.Prefill(prompt_ids);
  EXPECT_EQ(
      prefill.logits.sizes().vec(),
      std::vector<std::int64_t>({static_cast<std::int64_t>(prompt_ids.size()), config().vocab_size}));
}

TEST_F(QwenExecutorSmokeTest, GenerateReturnsSingleTokenOnCpu) {
  auto executor = CreateExecutor();
  const auto prompt_ids = EncodePrompt("Hello");
  ASSERT_FALSE(prompt_ids.empty());

  SamplingParams params;
  params.max_new_tokens = 1;
  params.stream = false;
  params.temperature = 0.0;
  const auto generation = executor.Generate(prompt_ids, tokenizer(), params);

  EXPECT_EQ(generation.prompt_token_ids, prompt_ids);
  EXPECT_EQ(generation.generated_token_ids.size(), 1U);
  EXPECT_EQ(generation.all_token_ids.size(), prompt_ids.size() + 1U);
  EXPECT_FALSE(generation.generated_text.empty());
}

}  // namespace
}  // namespace localllm
