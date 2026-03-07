#include "localllm/runtime/sampling.h"

#include <vector>

#include <gtest/gtest.h>

namespace localllm {
namespace {

SamplingParams MakeSamplingParams() {
  SamplingParams params;
  params.stream = false;
  return params;
}

TEST(SamplingTest, GreedySamplingReturnsArgmaxWhenTemperatureZero) {
  auto logits = torch::tensor({0.1F, 0.7F, 0.2F});
  auto params = MakeSamplingParams();
  params.temperature = 0.0;

  EXPECT_EQ(SampleNextToken(logits, params), 1);
}

TEST(SamplingTest, TopKSamplingRespectsCandidateWindow) {
  auto logits = torch::tensor({0.0F, 1.0F, 2.0F, 3.0F});
  auto params = MakeSamplingParams();
  params.temperature = 1.0;
  params.top_k = 1;
  params.seed = 123;

  EXPECT_EQ(SampleNextToken(logits, params), 3);
}

TEST(SamplingTest, TopPSamplingRemovesLowProbabilityTail) {
  auto logits = torch::tensor({0.0F, 1.0F, 2.0F, 3.0F});
  auto params = MakeSamplingParams();
  params.temperature = 1.0;
  params.top_p = 0.1;
  params.seed = 123;

  EXPECT_EQ(SampleNextToken(logits, params), 3);
}

}  // namespace
}  // namespace localllm
