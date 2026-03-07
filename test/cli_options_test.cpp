#include "localllm/app/cli_options.h"

#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "localllm/common/status.h"

DECLARE_string(model_dir);
DECLARE_string(prompt);
DECLARE_bool(chat);
DECLARE_int64(max_new_tokens);
DECLARE_double(temperature);
DECLARE_double(top_p);
DECLARE_int64(top_k);
DECLARE_uint64(seed);
DECLARE_bool(stream);
DECLARE_bool(print_token_ids);
DECLARE_string(stop);

namespace localllm {
namespace {

TEST(CliOptionsTest, BuildsOptionsFromFlags) {
  gflags::FlagSaver flag_saver;

  FLAGS_model_dir = "models/Qwen3-0.6B";
  FLAGS_prompt = "test prompt";
  FLAGS_chat = true;
  FLAGS_max_new_tokens = 8;
  FLAGS_temperature = 0.7;
  FLAGS_top_p = 0.9;
  FLAGS_top_k = 32;
  FLAGS_seed = 123;
  FLAGS_stream = false;
  FLAGS_print_token_ids = true;
  FLAGS_stop = "END, STOP";

  std::istringstream input_stream;
  const CliOptions options = ParseCliOptions(input_stream);

  EXPECT_EQ(options.model_dir, std::filesystem::path("models/Qwen3-0.6B"));
  EXPECT_EQ(options.prompt, "test prompt");
  EXPECT_TRUE(options.chat_mode);
  EXPECT_EQ(options.sampling.max_new_tokens, 8);
  EXPECT_DOUBLE_EQ(options.sampling.temperature, 0.7);
  EXPECT_DOUBLE_EQ(options.sampling.top_p, 0.9);
  EXPECT_EQ(options.sampling.top_k, 32);
  EXPECT_EQ(options.sampling.seed, 123U);
  EXPECT_FALSE(options.sampling.stream);
  EXPECT_TRUE(options.sampling.print_token_ids);
  EXPECT_EQ(options.sampling.stop_strings, std::vector<std::string>({"END", "STOP"}));
}

TEST(CliOptionsTest, ReadsPromptFromInputWhenPromptFlagIsEmpty) {
  gflags::FlagSaver flag_saver;

  FLAGS_prompt.clear();
  std::istringstream input_stream("prompt from stdin");

  const CliOptions options = ParseCliOptions(input_stream);

  EXPECT_EQ(options.prompt, "prompt from stdin");
}

TEST(CliOptionsTest, RejectsMissingPromptWhenFlagAndInputAreEmpty) {
  gflags::FlagSaver flag_saver;

  FLAGS_prompt.clear();
  std::istringstream input_stream;

  EXPECT_THROW(static_cast<void>(ParseCliOptions(input_stream)), StatusError);
}

}  // namespace
}  // namespace localllm
