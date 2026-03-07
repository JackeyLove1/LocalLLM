#include "localllm/common/logging.h"

#include <gtest/gtest.h>

namespace localllm {
namespace {

TEST(LoggingTest, InitializesOnceAndAcceptsRepeatedCalls) {
  InitializeLogging("localllm_tests");
  InitializeLogging("");
  InitializeLogging(nullptr);
  InitializeLogging("localllm_tests");

  LogInfo("logging test info");
  LogWarn("logging test warning");
  LogError("logging test error");

  SUCCEED();
}

}  // namespace
}  // namespace localllm
