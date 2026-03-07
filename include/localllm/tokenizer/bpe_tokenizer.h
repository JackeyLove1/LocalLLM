#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
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

  struct MergePairKey {
    std::string_view first;
    std::string_view second;
    bool operator==(const MergePairKey& o) const {
      return first == o.first && second == o.second;
    }
  };
  struct MergePairKeyHash {
    std::size_t operator()(const MergePairKey& k) const {
      return std::hash<std::string_view>{}(k.first) ^
             (std::hash<std::string_view>{}(k.second) << 1);
    }
  };

  std::vector<std::string> special_tokens_;
  std::unordered_set<std::string_view> special_tokens_set_;
  std::unordered_map<std::string, std::int64_t> vocab_;
  std::vector<std::string> id_to_token_;
  std::vector<std::pair<std::string, std::string>> merge_pairs_;
  std::unordered_map<MergePairKey, int, MergePairKeyHash> merge_ranks_;
  std::unordered_map<std::string, std::uint8_t> unicode_to_byte_;
  std::string byte_to_unicode_[256];
};

}  // namespace localllm
