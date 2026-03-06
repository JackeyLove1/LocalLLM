#include "localllm/common/file_util.h"

#include <fstream>

#include "localllm/common/status.h"

namespace localllm {

std::string ReadTextFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  LOCALLLM_CHECK(stream.good(), "Failed to open file: " + path.string());
  return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::vector<std::uint8_t> ReadBinaryFile(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  LOCALLLM_CHECK(stream.good(), "Failed to open file: " + path.string());
  const auto size = static_cast<std::size_t>(stream.tellg());
  stream.seekg(0, std::ios::beg);

  std::vector<std::uint8_t> bytes(size);
  stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  LOCALLLM_CHECK(stream.good(), "Failed to read file: " + path.string());
  return bytes;
}

std::vector<std::uint8_t> ReadBinaryRange(
    const std::filesystem::path& path, std::size_t offset, std::size_t size) {
  std::ifstream stream(path, std::ios::binary);
  LOCALLLM_CHECK(stream.good(), "Failed to open file: " + path.string());
  stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
  LOCALLLM_CHECK(stream.good(), "Failed to seek file: " + path.string());

  std::vector<std::uint8_t> bytes(size);
  stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size));
  LOCALLLM_CHECK(stream.good(), "Failed to read range from file: " + path.string());
  return bytes;
}

}  // namespace localllm
