#include "localllm/tokenizer/bpe_tokenizer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

namespace localllm {
namespace {

constexpr char kTokenizerModelDir[] = "models/Qwen3-0.6B";

class BpeTokenizerTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    tokenizer_ = std::make_unique<BpeTokenizer>(
        BpeTokenizer::LoadFromDir(std::filesystem::path(kTokenizerModelDir)));
    ASSERT_NE(tokenizer_.get(), nullptr);
  }

  static void TearDownTestSuite() { tokenizer_.reset(); }

  const BpeTokenizer& tokenizer() const { return GetSharedTokenizer(); }

 private:
  static const BpeTokenizer& GetSharedTokenizer() { return *tokenizer_; }

  static std::unique_ptr<BpeTokenizer> tokenizer_;
};

std::unique_ptr<BpeTokenizer> BpeTokenizerTest::tokenizer_;

TEST_F(BpeTokenizerTest, EncodeBasicPrompt) {
  const auto res = tokenizer().Encode("Hello world");
  EXPECT_GT(res.size(), 0);
}

TEST_F(BpeTokenizerTest, EncodesExpectedEnglishTokens) {
  EXPECT_EQ(tokenizer().Encode("Hello world"), std::vector<std::int64_t>({9707, 1879}));
  EXPECT_EQ(tokenizer().Encode("Hello\nworld"), std::vector<std::int64_t>({9707, 198, 14615}));
}

TEST_F(BpeTokenizerTest, SupportsSpecialTokensAndRoundTrip) {
  const auto token_ids = tokenizer().Encode("<|im_start|>user\nHi<|im_end|>");
  EXPECT_EQ(token_ids, std::vector<std::int64_t>({151644, 872, 198, 13048, 151645}));
  EXPECT_EQ(tokenizer().Decode(token_ids, false), "<|im_start|>user\nHi<|im_end|>");
}

TEST_F(BpeTokenizerTest, HandlesChineseRoundTrip) {
  const std::string text = "\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C\xE4\xB8\x96\xE7\x95\x8C";
  const auto token_ids = tokenizer().Encode(text);
  ASSERT_FALSE(token_ids.empty());
  EXPECT_EQ(tokenizer().Decode(token_ids, false), text);
}

TEST_F(BpeTokenizerTest, HandlesEnglishContractionsWithoutDuplicatingCharacters) {
  const std::string text = "it's I'm I'd can't";
  const auto token_ids = tokenizer().Encode(text);
  ASSERT_FALSE(token_ids.empty());
  EXPECT_EQ(tokenizer().Decode(token_ids, false), text);
}

TEST_F(BpeTokenizerTest, HandlesEmptyInputForEncodeAndDecode) {
  EXPECT_TRUE(tokenizer().Encode("").empty());
  EXPECT_TRUE(tokenizer().Decode({}, false).empty());
  EXPECT_TRUE(tokenizer().Decode({}, true).empty());
}

TEST_F(BpeTokenizerTest, KeepsAdjacentSpecialTokensAsAtomicTokens) {
  const auto token_ids = tokenizer().Encode("<|im_start|><|im_end|>");
  EXPECT_EQ(token_ids, std::vector<std::int64_t>({151644, 151645}));
  EXPECT_EQ(tokenizer().Decode(token_ids, false), "<|im_start|><|im_end|>");
  EXPECT_EQ(tokenizer().Decode(token_ids, true), "");
}

TEST_F(BpeTokenizerTest, SkipsSpecialTokensWithoutDroppingNormalText) {
  const std::string text = "A<|im_start|><|im_end|>B";
  const auto token_ids = tokenizer().Encode(text);
  ASSERT_EQ(std::count(token_ids.begin(), token_ids.end(), 151644), 1);
  ASSERT_EQ(std::count(token_ids.begin(), token_ids.end(), 151645), 1);
  EXPECT_EQ(tokenizer().Decode(token_ids, false), text);
  EXPECT_EQ(tokenizer().Decode(token_ids, true), "AB");
}

TEST_F(BpeTokenizerTest, AppliesChatTemplateWithExpectedSpecialTokenCounts) {
  const std::string templated = tokenizer().ApplyChatTemplate("Hello");
  EXPECT_EQ(templated, "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n");

  const auto token_ids = tokenizer().Encode(templated);
  ASSERT_EQ(std::count(token_ids.begin(), token_ids.end(), 151644), 2);
  ASSERT_EQ(std::count(token_ids.begin(), token_ids.end(), 151645), 1);
  EXPECT_EQ(tokenizer().Decode(token_ids, false), templated);
  EXPECT_EQ(tokenizer().Decode(token_ids, true), "user\nHello\nassistant\n");
}

TEST(BpeTokenizerMinimalModelTest, AppliesMergePriorityOnByteLevelSymbols) {
  const auto temp_dir = std::filesystem::path("build") / "testdata" /
                        "localllm_bpe_tokenizer_minimal_model";
  std::error_code error_code;
  ASSERT_TRUE(std::filesystem::exists(temp_dir) ||
              std::filesystem::create_directories(temp_dir, error_code));

  {
    std::ofstream vocab_file(temp_dir / "vocab.json");
    ASSERT_TRUE(vocab_file.is_open());
    vocab_file << R"({
  "a": 0,
  "b": 1,
  "c": 2,
  "ab": 3,
  "bc": 4,
  "abc": 5
})";
  }
  {
    std::ofstream merges_file(temp_dir / "merges.txt");
    ASSERT_TRUE(merges_file.is_open());
    merges_file << "a b\nab c\nb c\n";
  }
  {
    std::ofstream tokenizer_config_file(temp_dir / "tokenizer_config.json");
    ASSERT_TRUE(tokenizer_config_file.is_open());
    tokenizer_config_file << R"({
  "added_tokens_decoder": {}
})";
  }

  const auto tokenizer = BpeTokenizer::LoadFromDir(temp_dir);
  EXPECT_EQ(tokenizer.Encode("abc"), std::vector<std::int64_t>({5}));
  EXPECT_EQ(tokenizer.Decode({5}, false), "abc");
}

}  // namespace
}  // namespace localllm
