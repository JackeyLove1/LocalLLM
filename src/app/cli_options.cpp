#include "localllm/app/cli_options.h"

#include <cstdlib>
#include <iostream>
#include <sstream>

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
    std::ostringstream input;
    input << std::cin.rdbuf();
    options.prompt = input.str();
  }
  LOCALLLM_CHECK(!options.prompt.empty(), "Prompt is required. Use --prompt or pipe stdin.");
  return options;
}

}  // namespace localllm
