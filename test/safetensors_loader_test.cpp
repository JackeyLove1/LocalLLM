#include <filesystem>

#include <gtest/gtest.h>

#include "localllm/model/safetensors_loader.h"

namespace localllm {
namespace {

TEST(SafeTensorsLoaderTest, ParsesMetadataAndLoadsTensor) {
  SafeTensorFile file(std::filesystem::path("models/Qwen3-0.6B/model.safetensors"));
  EXPECT_GT(file.entries().size(), 300U);

  const auto& embed = file.GetEntry("model.embed_tokens.weight");
  EXPECT_EQ(embed.dtype, "BF16");
  EXPECT_EQ(embed.shape[0], 151936);
  EXPECT_EQ(embed.shape[1], 1024);

  const auto tensor = file.LoadTensor("model.norm.weight", torch::kFloat32, torch::kCPU);
  EXPECT_EQ(tensor.sizes().vec(), std::vector<std::int64_t>({1024}));
  EXPECT_EQ(tensor.dtype(), torch::kFloat32);
}

}  // namespace
}  // namespace localllm
