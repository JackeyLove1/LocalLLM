#include <gtest/gtest.h>

#include "localllm/common/logging.h"

namespace localllm {
namespace {

TEST(LoggingTest, InitializeLoggingIsIdempotent) {
  InitializeLogging("localllm_tests");
  InitializeLogging("localllm_tests");

  LogInfo("logging test info");
  LogWarn("logging test warning");
  LogError("logging test error");

  SUCCEED();
}

}  // namespace
}  // namespace localllm
