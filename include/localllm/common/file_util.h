#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace localllm {

std::string ReadTextFile(const std::filesystem::path& path);
std::vector<std::uint8_t> ReadBinaryFile(const std::filesystem::path& path);
std::vector<std::uint8_t> ReadBinaryRange(
    const std::filesystem::path& path, std::size_t offset, std::size_t size);

}  // namespace localllm
