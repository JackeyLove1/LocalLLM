#include "localllm/app/application.h"

#include <torch/torch.h>

#include "localllm/common/logging.h"
#include "localllm/executor/qwen_executor.h"
#include "localllm/model/model_weights.h"
#include "localllm/model/qwen_config.h"
#include "localllm/tokenizer/bpe_tokenizer.h"

namespace localllm {

int RunApplication(const CliOptions& options) {
  const auto config = LoadConfig(options.model_dir);
  auto tokenizer = BpeTokenizer::LoadFromDir(options.model_dir);

  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
    LogInfo("CUDA device detected, attempting CUDA execution.");
  } else {
    LogWarn("CUDA is not available in the current libtorch build. Falling back to CPU.");
  }

  auto weights = LoadWeights(options.model_dir, config, device);
  QwenExecutor executor(config, std::move(weights));

  std::string prompt = options.prompt;
  if (options.chat_mode) {
    prompt = tokenizer.ApplyChatTemplate(prompt);
  }
  const auto prompt_token_ids = tokenizer.Encode(prompt);
  LogInfo("Prompt tokens: " + std::to_string(prompt_token_ids.size()));

  const auto result = executor.Generate(prompt_token_ids, tokenizer, options.sampling);
  if (!options.sampling.stream) {
    std::cout << result.generated_text << '\n';
  } else {
    std::cout << '\n';
  }
  return 0;
}

}  // namespace localllm
