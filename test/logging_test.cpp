#include "localllm/common/logging.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

namespace localllm {
namespace {

TEST(LoggingTest, InitializesOnceAndAcceptsRepeatedCalls) {
  InitializeLogging("localllm_tests");
  InitializeLogging("");
  InitializeLogging(nullptr);
  InitializeLogging("localllm_tests");

  LOG(INFO) << "logging test info";
  LOG(WARNING) << "logging test warning";
  LOG(ERROR) << "logging test error";

  SUCCEED();
}

}  // namespace
}  // namespace localllm
