#include <vector>

#include <gtest/gtest.h>

#include "localllm/runtime/sampling.h"

namespace localllm {
namespace {

TEST(SamplingTest, GreedySamplingReturnsArgmaxWhenTemperatureZero) {
  auto logits = torch::tensor({0.1F, 0.7F, 0.2F});
  SamplingParams params;
  params.temperature = 0.0;
  EXPECT_EQ(SampleNextToken(logits, params), 1);
}

TEST(SamplingTest, TopKSamplingRespectsCandidateWindow) {
  auto logits = torch::tensor({0.0F, 1.0F, 2.0F, 3.0F});
  SamplingParams params;
  params.temperature = 1.0;
  params.top_k = 1;
  params.seed = 123;
  EXPECT_EQ(SampleNextToken(logits, params), 3);
}

}  // namespace
}  // namespace localllm
