#include "localllm/app/cli_options.h"

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "localllm/common/logging.h"
#include "localllm/common/status.h"

namespace localllm {

namespace {

double ParseDouble(const char* value, const std::string& name) {
  char* end = nullptr;
  const double parsed = std::strtod(value, &end);
  LOCALLLM_CHECK(end != value, "Failed to parse numeric CLI option: " + name);
  return parsed;
}

std::int64_t ParseInt64(const char* value, const std::string& name) {
  char* end = nullptr;
  const auto parsed = std::strtoll(value, &end, 10);
  LOCALLLM_CHECK(end != value, "Failed to parse integer CLI option: " + name);
  return parsed;
}

}  // namespace

CliOptions ParseCliOptions(int argc, char** argv) {
  CliOptions options;
  options.model_dir = "models/Qwen3-0.6B";
  bool prompt_from_stdin = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const auto require_value = [&](const std::string& name) -> const char* {
      LOCALLLM_CHECK(i + 1 < argc, "Missing value for CLI option: " + name);
      return argv[++i];
    };

    if (arg == "--model-dir") {
      options.model_dir = require_value(arg);
    } else if (arg == "--prompt") {
      options.prompt = require_value(arg);
    } else if (arg == "--max-new-tokens") {
      options.sampling.max_new_tokens = ParseInt64(require_value(arg), arg);
    } else if (arg == "--temperature") {
      options.sampling.temperature = ParseDouble(require_value(arg), arg);
    } else if (arg == "--top-p") {
      options.sampling.top_p = ParseDouble(require_value(arg), arg);
    } else if (arg == "--top-k") {
      options.sampling.top_k = ParseInt64(require_value(arg), arg);
    } else if (arg == "--seed") {
      options.sampling.seed = static_cast<std::uint64_t>(ParseInt64(require_value(arg), arg));
    } else if (arg == "--chat") {
      options.chat_mode = true;
    } else if (arg == "--no-stream") {
      options.sampling.stream = false;
    } else if (arg == "--print-token-ids") {
      options.sampling.print_token_ids = true;
    } else if (arg == "--stop") {
      options.sampling.stop_strings.push_back(require_value(arg));
    } else {
      throw StatusError("Unknown CLI option: " + arg);
    }
  }

  if (options.prompt.empty()) {
    prompt_from_stdin = true;
    std::ostringstream input;
    input << std::cin.rdbuf();
    options.prompt = input.str();
  }
  LOCALLLM_CHECK(!options.prompt.empty(), "Prompt is required. Use --prompt or pipe stdin.");
  LogInfo(
      "CLI options parsed: model_dir=" + options.model_dir.string() +
      ", prompt_source=" + std::string(prompt_from_stdin ? "stdin" : "argument") +
      ", prompt_chars=" + std::to_string(options.prompt.size()) +
      ", chat_mode=" + std::string(options.chat_mode ? "true" : "false") +
      ", stream=" + std::string(options.sampling.stream ? "true" : "false") +
      ", max_new_tokens=" + std::to_string(options.sampling.max_new_tokens) +
      ", temperature=" + std::to_string(options.sampling.temperature) +
      ", top_p=" + std::to_string(options.sampling.top_p) +
      ", top_k=" + std::to_string(options.sampling.top_k) +
      ", stop_strings=" + std::to_string(options.sampling.stop_strings.size()));
  return options;
}

}  // namespace localllm
