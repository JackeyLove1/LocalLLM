#include "localllm/model/safetensors_loader.h"

#include <c10/util/BFloat16.h>

#include <cstring>

#include "json11.hpp"

#include "localllm/common/file_util.h"
#include "localllm/common/status.h"

namespace localllm {

namespace {

std::size_t ReadHeaderSize(const std::vector<std::uint8_t>& bytes) {
  LOCALLLM_CHECK(bytes.size() >= sizeof(std::uint64_t), "Invalid safetensors header.");
  std::uint64_t value = 0;
  std::memcpy(&value, bytes.data(), sizeof(std::uint64_t));
  return static_cast<std::size_t>(value);
}

std::vector<std::int64_t> ParseShape(const json11::Json& json_shape) {
  std::vector<std::int64_t> shape;
  shape.reserve(json_shape.array_items().size());
  for (const auto& dim : json_shape.array_items()) {
    shape.push_back(dim.int_value());
  }
  return shape;
}

}  // namespace

SafeTensorFile::SafeTensorFile(std::filesystem::path path) : path_(std::move(path)) {
  const auto header_prefix = ReadBinaryRange(path_, 0, 8);
  header_size_ = ReadHeaderSize(header_prefix);
  data_offset_ = 8 + header_size_;
  const auto header_bytes = ReadBinaryRange(path_, 8, header_size_);
  const std::string header_text(reinterpret_cast<const char*>(header_bytes.data()), header_bytes.size());

  std::string error;
  const auto header_json = json11::Json::parse(header_text, error);
  LOCALLLM_CHECK(error.empty(), "Failed to parse safetensors header: " + error);

  for (const auto& [key, value] : header_json.object_items()) {
    if (key == "__metadata__") {
      continue;
    }

    const auto offsets = value["data_offsets"].array_items();
    LOCALLLM_CHECK(offsets.size() == 2, "Invalid data_offsets for tensor: " + key);

    SafeTensorEntry entry;
    entry.dtype = value["dtype"].string_value();
    entry.shape = ParseShape(value["shape"]);
    entry.data_begin = static_cast<std::size_t>(offsets[0].int_value());
    entry.data_end = static_cast<std::size_t>(offsets[1].int_value());
    entries_.emplace(key, std::move(entry));
  }
}

const SafeTensorEntry& SafeTensorFile::GetEntry(const std::string& name) const {
  const auto it = entries_.find(name);
  LOCALLLM_CHECK(it != entries_.end(), "Missing tensor in safetensors: " + name);
  return it->second;
}

torch::Tensor SafeTensorFile::LoadBFloat16Tensor(
    const SafeTensorEntry& entry, torch::ScalarType target_dtype, const torch::Device& device) const {
  const auto size = entry.data_end - entry.data_begin;
  const auto bytes = ReadBinaryRange(path_, data_offset_ + entry.data_begin, size);

  torch::Tensor tensor =
      torch::empty(entry.shape, torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));
  std::memcpy(tensor.data_ptr<c10::BFloat16>(), bytes.data(), bytes.size());

  if (target_dtype != torch::kBFloat16) {
    tensor = tensor.to(target_dtype);
  }
  if (device != torch::kCPU) {
    tensor = tensor.to(device);
  }
  return tensor.contiguous();
}

torch::Tensor SafeTensorFile::LoadTensor(
    const std::string& name, torch::ScalarType target_dtype, const torch::Device& device) const {
  const auto& entry = GetEntry(name);
  if (entry.dtype == "BF16") {
    return LoadBFloat16Tensor(entry, target_dtype, device);
  }
  if (entry.dtype == "F32") {
    const auto size = entry.data_end - entry.data_begin;
    const auto bytes = ReadBinaryRange(path_, data_offset_ + entry.data_begin, size);
    torch::Tensor tensor =
        torch::empty(entry.shape, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    std::memcpy(tensor.data_ptr<float>(), bytes.data(), bytes.size());
    if (target_dtype != torch::kFloat32) {
      tensor = tensor.to(target_dtype);
    }
    if (device != torch::kCPU) {
      tensor = tensor.to(device);
    }
    return tensor.contiguous();
  }

  throw StatusError("Unsupported tensor dtype in safetensors: " + entry.dtype);
}

}  // namespace localllm
