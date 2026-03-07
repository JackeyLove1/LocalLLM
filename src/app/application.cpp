#include "localllm/app/application.h"

#include <torch/torch.h>

#include "localllm/common/logging.h"
#include "localllm/executor/qwen_executor.h"
#include "localllm/model/model_weights.h"
#include "localllm/model/qwen_config.h"
#include "localllm/tokenizer/bpe_tokenizer.h"

namespace localllm {

int RunApplication(const CliOptions& options) {
  LOG(INFO) << "Loading model config from " << (options.model_dir / "config.json").string();
  const auto config = LoadConfig(options.model_dir);
  LOG(INFO) << "Loading tokenizer from " << options.model_dir.string();
  auto tokenizer = BpeTokenizer::LoadFromDir(options.model_dir);
  LOG(INFO) << "Tokenizer loaded successfully.";

  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
    LOG(INFO) << "CUDA device detected, attempting CUDA execution.";
  } else {
    LOG(WARNING) << "CUDA is not available in the current libtorch build. Falling back to CPU.";
  }

  LOG(INFO) << "Loading model weights.";
  auto weights = LoadWeights(options.model_dir, config, device);
  LOG(INFO) << "Model weights loaded.";
  QwenExecutor executor(config, std::move(weights));

  std::string prompt = options.prompt;
  if (options.chat_mode) {
    LOG(INFO) << "Applying chat template to prompt.";
    prompt = tokenizer.ApplyChatTemplate(prompt);
  }
  const auto prompt_token_ids = tokenizer.Encode(prompt);
  LOG(INFO) << "Prompt tokens: " << prompt_token_ids.size();

  LOG(INFO) << "Starting generation: max_new_tokens=" << options.sampling.max_new_tokens
            << ", stream=" << (options.sampling.stream ? "true" : "false");
  const auto result = executor.Generate(prompt_token_ids, tokenizer, options.sampling);
  LOG(INFO) << "Generation finished: generated_tokens=" << result.generated_token_ids.size()
            << ", total_tokens=" << result.all_token_ids.size()
            << ", output_chars=" << result.generated_text.size();
  if (!options.sampling.stream) {
    std::cout << result.generated_text << '\n';
  } else {
    std::cout << '\n';
  }
  return 0;
}

}  // namespace localllm
