#include "localllm/common/logging.h"

#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  localllm::InitializeLogging(argc > 0 ? argv[0] : "localllm_tests");
  return RUN_ALL_TESTS();
}
