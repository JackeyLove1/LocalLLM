#pragma once

#include <stdexcept>
#include <string>

namespace localllm {

class StatusError : public std::runtime_error {
 public:
  explicit StatusError(const std::string& message) : std::runtime_error(message) {}
};

}  // namespace localllm

#define LOCALLLM_CHECK(condition, message) \
  do {                                     \
    if (!(condition)) {                    \
      throw ::localllm::StatusError(message); \
    }                                      \
  } while (false)
