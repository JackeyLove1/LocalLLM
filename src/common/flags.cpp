#include "localllm/common/flags.h"

#include <mutex>

#include <gflags/gflags.h>

namespace localllm {

void InitializeCommandLineFlags(int* argc, char*** argv) {
  static std::once_flag init_once;
  std::call_once(init_once, [argc, argv]() {
    gflags::SetUsageMessage("LocalLLM command line flags");
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);
  });
}

}  // namespace localllm
