#include <iostream>

#include "localllm/app/application.h"
#include "localllm/app/cli_options.h"
#include "localllm/common/status.h"

int main(int argc, char** argv) {
  try {
    const auto options = localllm::ParseCliOptions(argc, argv);
    return localllm::RunApplication(options);
  } catch (const localllm::StatusError& error) {
    std::cerr << "[error] " << error.what() << '\n';
    return 1;
  } catch (const std::exception& error) {
    std::cerr << "[fatal] " << error.what() << '\n';
    return 1;
  }
}
