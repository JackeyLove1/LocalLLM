#pragma once

#include <string>

namespace localllm {

void InitializeLogging(const char* argv0);
void LogInfo(const std::string& message);
void LogWarn(const std::string& message);
void LogError(const std::string& message);

}  // namespace localllm
