#include <iostream>

#include "localllm/app/application.h"
#include "localllm/app/cli_options.h"
#include "localllm/common/logging.h"
#include "localllm/common/status.h"

int main(int argc, char** argv) {
  try {
    localllm::InitializeLogging(argc > 0 ? argv[0] : "localllm");
    LOG(INFO) << "Starting LocalLLM.";
    const auto options = localllm::ParseCliOptions(argc, argv);
    return localllm::RunApplication(options);
  } catch (const localllm::StatusError& error) {
    LOG(ERROR) << "StatusError: " << error.what();
    return 1;
  } catch (const std::exception& error) {
    LOG(ERROR) << "Unhandled exception: " << error.what();
    return 1;
  }
}
