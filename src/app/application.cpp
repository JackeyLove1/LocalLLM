#include "localllm/app/application.h"

#include <torch/torch.h>

#include "localllm/common/logging.h"
#include "localllm/executor/qwen_executor.h"
#include "localllm/model/model_weights.h"
#include "localllm/model/qwen_config.h"
#include "localllm/tokenizer/bpe_tokenizer.h"

namespace localllm {

int RunApplication(const CliOptions& options) {
  LogInfo("Loading model config from " + (options.model_dir / "config.json").string());
  const auto config = LoadConfig(options.model_dir);
  LogInfo("Loading tokenizer from " + options.model_dir.string());
  auto tokenizer = BpeTokenizer::LoadFromDir(options.model_dir);
  LogInfo("Tokenizer loaded successfully.");

  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
    LogInfo("CUDA device detected, attempting CUDA execution.");
  } else {
    LogWarn("CUDA is not available in the current libtorch build. Falling back to CPU.");
  }

  LogInfo("Loading model weights.");
  auto weights = LoadWeights(options.model_dir, config, device);
  LogInfo("Model weights loaded.");
  QwenExecutor executor(config, std::move(weights));

  std::string prompt = options.prompt;
  if (options.chat_mode) {
    LogInfo("Applying chat template to prompt.");
    prompt = tokenizer.ApplyChatTemplate(prompt);
  }
  const auto prompt_token_ids = tokenizer.Encode(prompt);
  LogInfo("Prompt tokens: " + std::to_string(prompt_token_ids.size()));

  LogInfo(
      "Starting generation: max_new_tokens=" + std::to_string(options.sampling.max_new_tokens) +
      ", stream=" + std::string(options.sampling.stream ? "true" : "false"));
  const auto result = executor.Generate(prompt_token_ids, tokenizer, options.sampling);
  LogInfo(
      "Generation finished: generated_tokens=" + std::to_string(result.generated_token_ids.size()) +
      ", total_tokens=" + std::to_string(result.all_token_ids.size()) +
      ", output_chars=" + std::to_string(result.generated_text.size()));
  if (!options.sampling.stream) {
    std::cout << result.generated_text << '\n';
  } else {
    std::cout << '\n';
  }
  return 0;
}

}  // namespace localllm
