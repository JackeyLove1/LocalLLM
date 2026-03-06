#include "localllm/common/logging.h"

#include <iostream>

namespace localllm {

void LogInfo(const std::string& message) { std::cerr << "[INFO] " << message << '\n'; }

void LogWarn(const std::string& message) { std::cerr << "[WARN] " << message << '\n'; }

}  // namespace localllm
