#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_set>
#include <unordered_map>
#include <vector>

namespace localllm {

class BpeTokenizer {
 public:
  static BpeTokenizer LoadFromDir(const std::filesystem::path& model_dir);

  std::vector<std::int64_t> Encode(const std::string& text) const;
  std::string Decode(const std::vector<std::int64_t>& token_ids,
                     bool skip_special_tokens = false) const;
  std::string ApplyChatTemplate(const std::string& user_prompt) const;

  std::int64_t TokenToId(const std::string& token) const;
  bool IsSpecialToken(std::string_view token) const;

 private:
  static constexpr std::int64_t kInvalidTokenId = -1;
  static constexpr int kInvalidRank = std::numeric_limits<int>::max();

  struct MergeValue {
    std::int64_t merged_token_id = kInvalidTokenId;
    int rank = kInvalidRank;
  };

  struct SpecialTrieNode {
    std::array<std::unique_ptr<SpecialTrieNode>, 256> next = {};
    std::int64_t token_id = kInvalidTokenId;
  };

  struct SpecialMatch {
    bool matched = false;
    std::int64_t token_id = kInvalidTokenId;
    std::size_t end = 0;
  };

  struct Symbol {
    std::int64_t token_id = kInvalidTokenId;
    int prev = -1;
    int next = -1;
    bool alive = true;
  };

  struct PairCandidate {
    int rank = kInvalidRank;
    int left_index = -1;
    int right_index = -1;
    std::int64_t left_token_snapshot = kInvalidTokenId;
    std::int64_t right_token_snapshot = kInvalidTokenId;
    std::int64_t merged_token_id = kInvalidTokenId;

    bool operator<(const PairCandidate& other) const { return rank > other.rank; }
  };

  SpecialMatch MatchSpecial(std::string_view text, std::size_t position) const;
  std::size_t FindNextSpecialStart(std::string_view text, std::size_t position) const;
  void EncodeOrdinaryText(std::string_view text, std::vector<std::int64_t>* token_ids) const;
  std::vector<std::string> Pretokenize(std::string_view text) const;
  std::vector<std::int64_t> RunBpe(std::string_view piece) const;
  void TryPushPair(const std::vector<Symbol>& symbols, int left_index, int right_index,
                   std::priority_queue<PairCandidate>* candidates) const;
  bool IsStillValid(const std::vector<Symbol>& symbols, const PairCandidate& candidate) const;
  static std::uint64_t MakePairKey(std::int64_t left_token_id, std::int64_t right_token_id);

  std::vector<std::string> special_tokens_;
  std::unordered_set<std::string_view> special_tokens_set_;
  std::unordered_map<std::string, std::int64_t> vocab_;
  std::vector<std::string> id_to_token_;
  std::unordered_map<std::uint64_t, MergeValue> merge_table_;
  std::unordered_map<std::string, std::uint8_t> unicode_to_byte_;
  std::array<std::int64_t, 256> byte_token_ids_ = {};
  std::string byte_to_unicode_[256];
  std::unique_ptr<SpecialTrieNode> special_root_;
};

}  // namespace localllm
