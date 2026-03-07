#include "localllm/model/safetensors_loader.h"

#include <filesystem>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

namespace localllm {
namespace {

constexpr char kTensorFile[] = "models/Qwen3-0.6B/model.safetensors";

class SafeTensorsLoaderTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    file_ = std::make_unique<SafeTensorFile>(std::filesystem::path(kTensorFile));
  }

  static void TearDownTestSuite() { file_.reset(); }

  const SafeTensorFile& file() const { return *file_; }

 private:
  static std::unique_ptr<SafeTensorFile> file_;
};

std::unique_ptr<SafeTensorFile> SafeTensorsLoaderTest::file_;

TEST_F(SafeTensorsLoaderTest, ParsesMetadataForKnownTensor) {
  EXPECT_GT(file().header_size(), 0U);
  EXPECT_GT(file().entries().size(), 300U);

  const auto& embed = file().GetEntry("model.embed_tokens.weight");
  EXPECT_EQ(embed.dtype, "BF16");
  ASSERT_GE(embed.shape.size(), 2U);
  EXPECT_EQ(embed.shape[0], 151936);
  EXPECT_EQ(embed.shape[1], 1024);
}

TEST_F(SafeTensorsLoaderTest, LoadsTensorIntoRequestedDtype) {
  const auto tensor = file().LoadTensor("model.norm.weight", torch::kFloat32, torch::kCPU);
  EXPECT_EQ(tensor.sizes().vec(), std::vector<std::int64_t>({1024}));
  EXPECT_EQ(tensor.dtype(), torch::kFloat32);
  EXPECT_EQ(tensor.device(), torch::Device(torch::kCPU));
}

}  // namespace
}  // namespace localllm
