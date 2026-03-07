#include "localllm/common/logging.h"

#include <cstdlib>
#include <mutex>

#include <glog/logging.h>

namespace localllm {

namespace {

void ConfigureLogging(const char* argv0) {
  FLAGS_logtostderr = true;
  FLAGS_alsologtostderr = false;
  FLAGS_logtostdout = false;
  FLAGS_colorlogtostderr = true;
  FLAGS_stderrthreshold = 0;
  google::InitGoogleLogging(argv0);
  std::atexit([]() { google::ShutdownGoogleLogging(); });
}

void EnsureLoggingInitialized(const char* argv0 = "localllm") {
  static std::once_flag init_once;
  const char* program_name = (argv0 != nullptr && argv0[0] != '\0') ? argv0 : "localllm";
  std::call_once(init_once, [program_name]() { ConfigureLogging(program_name); });
}

}  // namespace

void InitializeLogging(const char* argv0) { EnsureLoggingInitialized(argv0); }

}  // namespace localllm
