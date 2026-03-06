#include <filesystem>
#include <vector>

#include <gtest/gtest.h>

#include "localllm/tokenizer/bpe_tokenizer.h"

namespace localllm {
namespace {

TEST(BpeTokenizerTest, EncodesExpectedEnglishTokens) {
  const auto tokenizer = BpeTokenizer::LoadFromDir(std::filesystem::path("models/Qwen3-0.6B"));
  EXPECT_EQ(tokenizer.Encode("Hello world"), std::vector<std::int64_t>({9707, 1879}));
  EXPECT_EQ(tokenizer.Encode("Hello\nworld"), std::vector<std::int64_t>({9707, 198, 14615}));
}

TEST(BpeTokenizerTest, SupportsSpecialTokensAndRoundTrip) {
  const auto tokenizer = BpeTokenizer::LoadFromDir(std::filesystem::path("models/Qwen3-0.6B"));
  const auto token_ids = tokenizer.Encode("<|im_start|>user\nHi<|im_end|>");
  EXPECT_EQ(token_ids, std::vector<std::int64_t>({151644, 872, 198, 13048, 151645}));
  EXPECT_EQ(tokenizer.Decode(token_ids, false), "<|im_start|>user\nHi<|im_end|>");
}

TEST(BpeTokenizerTest, HandlesChineseRoundTrip) {
  const auto tokenizer = BpeTokenizer::LoadFromDir(std::filesystem::path("models/Qwen3-0.6B"));
  const std::string text = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95\x8C";
  const auto token_ids = tokenizer.Encode(text);
  EXPECT_FALSE(token_ids.empty());
  EXPECT_EQ(tokenizer.Decode(token_ids, false), text);
}

}  // namespace
}  // namespace localllm
