#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "localllm/runtime/sampling.h"

namespace localllm {

struct CliOptions {
  std::filesystem::path model_dir;
  std::string prompt;
  bool chat_mode = false;
  SamplingParams sampling;
};

CliOptions ParseCliOptions(int argc, char** argv);

}  // namespace localllm
