#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include <torch/torch.h>

namespace localllm {

struct SafeTensorEntry {
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::size_t data_begin = 0;
  std::size_t data_end = 0;
};

class SafeTensorFile {
 public:
  explicit SafeTensorFile(std::filesystem::path path);

  const std::unordered_map<std::string, SafeTensorEntry>& entries() const { return entries_; }
  const SafeTensorEntry& GetEntry(const std::string& name) const;

  torch::Tensor LoadTensor(
      const std::string& name, torch::ScalarType target_dtype, const torch::Device& device) const;

  std::size_t header_size() const { return header_size_; }

 private:
  torch::Tensor LoadBFloat16Tensor(
      const SafeTensorEntry& entry, torch::ScalarType target_dtype, const torch::Device& device) const;

  std::filesystem::path path_;
  std::unordered_map<std::string, SafeTensorEntry> entries_;
  std::size_t header_size_ = 0;
  std::size_t data_offset_ = 0;
};

}  // namespace localllm
