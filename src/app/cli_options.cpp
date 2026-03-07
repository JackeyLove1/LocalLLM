#include "localllm/app/cli_options.h"

#include <iostream>
#include <sstream>
#include <string>

#include <gflags/gflags.h>

#include "localllm/common/logging.h"
#include "localllm/common/status.h"

DEFINE_string(model_dir, "models/Qwen3-0.6B", "Path to the model directory.");
DEFINE_string(prompt, "", "Prompt text. If empty, the prompt is read from stdin.");
DEFINE_bool(chat, false, "Apply the tokenizer chat template to the prompt.");
DEFINE_int64(max_new_tokens, 64, "Maximum number of generated tokens.");
DEFINE_double(temperature, 0.0, "Sampling temperature.");
DEFINE_double(top_p, 1.0, "Top-p sampling threshold.");
DEFINE_int64(top_k, 0, "Top-k sampling threshold. Zero disables top-k filtering.");
DEFINE_uint64(seed, 42, "Random seed used by sampling.");
DEFINE_bool(stream, true, "Stream generated tokens to stdout as they are produced.");
DEFINE_bool(print_token_ids, false, "Log generated token ids while streaming.");
DEFINE_string(stop, "", "Comma-separated stop strings.");

namespace {

bool ValidateMaxNewTokens(const char*, std::int64_t value) {
  return value > 0;
}

bool ValidateTemperature(const char*, double value) {
  return value >= 0.0;
}

bool ValidateTopP(const char*, double value) {
  return value > 0.0 && value <= 1.0;
}

bool ValidateTopK(const char*, std::int64_t value) {
  return value >= 0;
}

const bool kMaxNewTokensValid = gflags::RegisterFlagValidator(&FLAGS_max_new_tokens, &ValidateMaxNewTokens);
const bool kTemperatureValid = gflags::RegisterFlagValidator(&FLAGS_temperature, &ValidateTemperature);
const bool kTopPValid = gflags::RegisterFlagValidator(&FLAGS_top_p, &ValidateTopP);
const bool kTopKValid = gflags::RegisterFlagValidator(&FLAGS_top_k, &ValidateTopK);

}  // namespace

namespace localllm {

namespace {

void ValidateRegisteredFlagValidators() {
  LOCALLLM_CHECK(kMaxNewTokensValid, "Failed to register validator for --max_new_tokens.");
  LOCALLLM_CHECK(kTemperatureValid, "Failed to register validator for --temperature.");
  LOCALLLM_CHECK(kTopPValid, "Failed to register validator for --top_p.");
  LOCALLLM_CHECK(kTopKValid, "Failed to register validator for --top_k.");
}

std::string TrimAsciiWhitespace(std::string text) {
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1);
}

std::vector<std::string> ParseStopStrings(const std::string& flag_value) {
  std::vector<std::string> stop_strings;
  std::size_t offset = 0;
  while (offset <= flag_value.size()) {
    const auto delimiter = flag_value.find(',', offset);
    const auto token = TrimAsciiWhitespace(flag_value.substr(offset, delimiter - offset));
    if (!token.empty()) {
      stop_strings.push_back(token);
    }
    if (delimiter == std::string::npos) {
      break;
    }
    offset = delimiter + 1;
  }
  return stop_strings;
}

}  // namespace

CliOptions ParseCliOptions() { return ParseCliOptions(std::cin); }

CliOptions ParseCliOptions(std::istream& input) {
  ValidateRegisteredFlagValidators();

  CliOptions options;
  options.model_dir = FLAGS_model_dir;
  options.prompt = FLAGS_prompt;
  options.chat_mode = FLAGS_chat;
  options.sampling.max_new_tokens = FLAGS_max_new_tokens;
  options.sampling.temperature = FLAGS_temperature;
  options.sampling.top_p = FLAGS_top_p;
  options.sampling.top_k = FLAGS_top_k;
  options.sampling.seed = FLAGS_seed;
  options.sampling.stream = FLAGS_stream;
  options.sampling.print_token_ids = FLAGS_print_token_ids;
  options.sampling.stop_strings = ParseStopStrings(FLAGS_stop);

  bool prompt_from_stdin = false;
  if (options.prompt.empty()) {
    prompt_from_stdin = true;
    std::ostringstream input_buffer;
    input_buffer << input.rdbuf();
    options.prompt = input_buffer.str();
  }
  LOCALLLM_CHECK(!options.prompt.empty(), "Prompt is required. Use --prompt or pipe stdin.");
  LOG(INFO) << "CLI flags parsed: model_dir=" << options.model_dir.string()
            << ", prompt_source=" << (prompt_from_stdin ? "stdin" : "argument")
            << ", prompt_chars=" << options.prompt.size()
            << ", chat_mode=" << (options.chat_mode ? "true" : "false")
            << ", stream=" << (options.sampling.stream ? "true" : "false")
            << ", max_new_tokens=" << options.sampling.max_new_tokens
            << ", temperature=" << options.sampling.temperature
            << ", top_p=" << options.sampling.top_p
            << ", top_k=" << options.sampling.top_k
            << ", stop_strings=" << options.sampling.stop_strings.size();
  return options;
}

}  // namespace localllm
