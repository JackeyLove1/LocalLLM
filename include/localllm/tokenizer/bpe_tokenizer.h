#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace localllm {

class BpeTokenizer {
 public:
  static BpeTokenizer LoadFromDir(const std::filesystem::path& model_dir);

  std::vector<std::int64_t> Encode(const std::string& text) const;
  std::string Decode(const std::vector<std::int64_t>& token_ids, bool skip_special_tokens = false) const;
  std::string ApplyChatTemplate(const std::string& user_prompt) const;

  std::int64_t TokenToId(const std::string& token) const;
  bool IsSpecialToken(std::string_view token) const;

 private:
  struct PretokenizedPiece {
    std::string text;
    bool is_special = false;
  };

  std::vector<PretokenizedPiece> SplitSpecialPieces(const std::string& text) const;
  std::vector<std::string> Pretokenize(const std::string& text) const;
  std::vector<std::string> RunBpe(const std::string& piece) const;

  std::vector<std::string> special_tokens_;
  std::unordered_map<std::string, std::int64_t> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::string, int> merge_ranks_;
  std::unordered_map<std::string, std::uint8_t> unicode_to_byte_;
  std::string byte_to_unicode_[256];
};

}  // namespace localllm
