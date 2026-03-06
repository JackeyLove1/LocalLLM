#include <filesystem>

#include <gtest/gtest.h>

#include "localllm/executor/qwen_executor.h"
#include "localllm/model/model_weights.h"
#include "localllm/model/qwen_config.h"
#include "localllm/tokenizer/bpe_tokenizer.h"

namespace localllm {
namespace {

TEST(QwenExecutorSmokeTest, RunsSingleTokenGenerationOnCpu) {
  const auto model_dir = std::filesystem::path("models/Qwen3-0.6B");
  const auto config = LoadConfig(model_dir);
  const auto tokenizer = BpeTokenizer::LoadFromDir(model_dir);
  auto weights = LoadWeights(model_dir, config, torch::kCPU);
  QwenExecutor executor(config, std::move(weights));

  const auto prompt_ids = tokenizer.Encode("Hello");
  const auto prefill = executor.Prefill(prompt_ids);
  EXPECT_EQ(prefill.logits.sizes().vec(), std::vector<std::int64_t>({1, config.vocab_size}));

  SamplingParams params;
  params.max_new_tokens = 1;
  params.stream = false;
  params.temperature = 0.0;
  const auto generation = executor.Generate(prompt_ids, tokenizer, params);
  EXPECT_EQ(generation.prompt_token_ids, prompt_ids);
  EXPECT_EQ(generation.generated_token_ids.size(), 1U);
  EXPECT_FALSE(generation.generated_text.empty());
}

}  // namespace
}  // namespace localllm
