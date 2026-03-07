#include "localllm/tokenizer/bpe_tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <string>

#include "json11.hpp"
#include "localllm/common/file_util.h"
#include "localllm/common/status.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace localllm {

namespace {

struct CodepointSpan {
  char32_t codepoint = 0;
  std::size_t begin = 0;
  std::size_t end = 0;
};

std::vector<CodepointSpan> DecodeUtf8Spans(const std::string& text) {
  std::vector<CodepointSpan> spans;
  spans.reserve(text.size());
  for (std::size_t i = 0; i < text.size();) {
    const unsigned char byte = static_cast<unsigned char>(text[i]);
    std::size_t length = 1;
    char32_t value = 0;
    if ((byte & 0x80U) == 0) {
      value = byte;
    } else if ((byte & 0xE0U) == 0xC0U) {
      length = 2;
      value = (byte & 0x1FU) << 6;
      value |= static_cast<unsigned char>(text[i + 1]) & 0x3FU;
    } else if ((byte & 0xF0U) == 0xE0U) {
      length = 3;
      value = (byte & 0x0FU) << 12;
      value |= (static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 6;
      value |= static_cast<unsigned char>(text[i + 2]) & 0x3FU;
    } else {
      length = 4;
      value = (byte & 0x07U) << 18;
      value |= (static_cast<unsigned char>(text[i + 1]) & 0x3FU) << 12;
      value |= (static_cast<unsigned char>(text[i + 2]) & 0x3FU) << 6;
      value |= static_cast<unsigned char>(text[i + 3]) & 0x3FU;
    }
    spans.push_back(CodepointSpan{value, i, i + length});
    i += length;
  }
  return spans;
}

std::string EncodeUtf8(char32_t codepoint) {
  std::string out;
  if (codepoint <= 0x7F) {
    out.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | ((codepoint >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | ((codepoint >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | ((codepoint >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
  return out;
}

std::string NormalizeNfc(const std::string& text) { return text; }

bool IsAsciiAlpha(char32_t cp) { return (cp >= U'a' && cp <= U'z') || (cp >= U'A' && cp <= U'Z'); }

bool IsAsciiDigit(char32_t cp) { return cp >= U'0' && cp <= U'9'; }

bool IsNewline(char32_t cp) { return cp == U'\n' || cp == U'\r'; }

bool IsWhitespace(char32_t cp) { return cp == U' ' || cp == U'\t' || cp == 0x3000 || cp == 0x00A0; }

bool IsLetter(char32_t cp) {
  if (IsAsciiAlpha(cp)) {
    return true;
  }
  if (cp > 127 && !IsNewline(cp) && !IsWhitespace(cp) && !IsAsciiDigit(cp)) {
    return !(cp >= 0x2000 && cp <= 0x206F) && !(cp >= 0x3000 && cp <= 0x303F);
  }
  return false;
}

bool IsPunctuation(char32_t cp) {
  if (cp <= 127) {
    return std::ispunct(static_cast<unsigned char>(cp)) != 0;
  }
  return !IsLetter(cp) && !IsAsciiDigit(cp) && !IsWhitespace(cp) && !IsNewline(cp);
}

bool StartsWithCaseInsensitive(const std::string& text, std::size_t offset,
                               const std::string& suffix) {
  if (offset + suffix.size() > text.size()) {
    return false;
  }
  for (std::size_t i = 0; i < suffix.size(); ++i) {
    const char lhs = static_cast<char>(std::tolower(static_cast<unsigned char>(text[offset + i])));
    if (lhs != suffix[i]) {
      return false;
    }
  }
  return true;
}

std::array<std::string, 256> BuildByteToUnicode() {
  std::array<std::string, 256> table;
  std::array<bool, 256> used = {};
  std::vector<int> byte_values;
  std::vector<int> code_points;
  for (int b = 33; b <= 126; ++b) {
    byte_values.push_back(b);
    code_points.push_back(b);
    used[static_cast<std::size_t>(b)] = true;
  }
  for (int b = 161; b <= 172; ++b) {
    byte_values.push_back(b);
    code_points.push_back(b);
    used[static_cast<std::size_t>(b)] = true;
  }
  for (int b = 174; b <= 255; ++b) {
    byte_values.push_back(b);
    code_points.push_back(b);
    used[static_cast<std::size_t>(b)] = true;
  }
  int extra = 0;
  for (int b = 0; b <= 255; ++b) {
    if (!used[static_cast<std::size_t>(b)]) {
      byte_values.push_back(b);
      code_points.push_back(256 + extra);
      ++extra;
    }
  }
  for (std::size_t i = 0; i < byte_values.size(); ++i) {
    table[static_cast<std::size_t>(byte_values[i])] =
        EncodeUtf8(static_cast<char32_t>(code_points[i]));
  }
  return table;
}

std::vector<std::string> ParseMerges(const std::string& text) {
  std::vector<std::string> merges;
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    const auto line_end = text.find('\n', cursor);
    auto line =
        text.substr(cursor, line_end == std::string::npos ? std::string::npos : line_end - cursor);
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty() && line[0] != '#') {
      merges.push_back(line);
    }
    if (line_end == std::string::npos) {
      break;
    }
    cursor = line_end + 1;
  }
  return merges;
}

}  // namespace

BpeTokenizer BpeTokenizer::LoadFromDir(const std::filesystem::path& model_dir) {
  BpeTokenizer tokenizer;
  tokenizer.byte_token_ids_.fill(kInvalidTokenId);
  tokenizer.special_root_ = std::make_unique<SpecialTrieNode>();

  const auto vocab_text = ReadTextFile(model_dir / "vocab.json");
  const auto merges_text = ReadTextFile(model_dir / "merges.txt");
  const auto tokenizer_config_text = ReadTextFile(model_dir / "tokenizer_config.json");

  std::string error;
  const auto vocab_json = json11::Json::parse(vocab_text, error);
  LOCALLLM_CHECK(error.empty(), "Failed to parse vocab.json: " + error);
  error.clear();
  const auto tokenizer_config = json11::Json::parse(tokenizer_config_text, error);
  LOCALLLM_CHECK(error.empty(), "Failed to parse tokenizer_config.json: " + error);

  for (const auto& [token, id_json] : vocab_json.object_items()) {
    const auto id = static_cast<std::int64_t>(id_json.int_value());
    tokenizer.vocab_[token] = id;
    if (static_cast<std::size_t>(id) >= tokenizer.id_to_token_.size()) {
      tokenizer.id_to_token_.resize(static_cast<std::size_t>(id) + 1);
    }
    tokenizer.id_to_token_[static_cast<std::size_t>(id)] = token;
  }

  for (const auto& [id_text, token_info] :
       tokenizer_config["added_tokens_decoder"].object_items()) {
    const auto id = static_cast<std::int64_t>(std::stoll(id_text));
    const auto token = token_info["content"].string_value();
    tokenizer.vocab_[token] = id;
    if (static_cast<std::size_t>(id) >= tokenizer.id_to_token_.size()) {
      tokenizer.id_to_token_.resize(static_cast<std::size_t>(id) + 1);
    }
    tokenizer.id_to_token_[static_cast<std::size_t>(id)] = token;
  }

  const auto byte_to_unicode = BuildByteToUnicode();
  for (std::size_t i = 0; i < byte_to_unicode.size(); ++i) {
    tokenizer.byte_to_unicode_[i] = byte_to_unicode[i];
    tokenizer.unicode_to_byte_[byte_to_unicode[i]] = static_cast<std::uint8_t>(i);

    const auto byte_token_it = tokenizer.vocab_.find(tokenizer.byte_to_unicode_[i]);
    if (byte_token_it != tokenizer.vocab_.end()) {
      tokenizer.byte_token_ids_[i] = byte_token_it->second;
    }
  }

  const auto merge_lines = ParseMerges(merges_text);
  tokenizer.merge_table_.reserve(merge_lines.size());
  int rank = 0;
  for (const auto& line : merge_lines) {
    const auto space_pos = line.find(' ');
    LOCALLLM_CHECK(space_pos != std::string::npos, "Invalid merge line: " + line);

    const auto left_piece = line.substr(0, space_pos);
    const auto right_piece = line.substr(space_pos + 1);
    const auto merged_piece = left_piece + right_piece;

    const auto left_it = tokenizer.vocab_.find(left_piece);
    LOCALLLM_CHECK(left_it != tokenizer.vocab_.end(), "Merge token missing in vocabulary.");
    const auto right_it = tokenizer.vocab_.find(right_piece);
    LOCALLLM_CHECK(right_it != tokenizer.vocab_.end(), "Merge token missing in vocabulary.");
    const auto merged_it = tokenizer.vocab_.find(merged_piece);
    LOCALLLM_CHECK(merged_it != tokenizer.vocab_.end(), "Merged token missing in vocabulary.");

    tokenizer.merge_table_[MakePairKey(left_it->second, right_it->second)] =
        MergeValue{merged_it->second, rank++};
  }

  for (const auto& [_, token_info] : tokenizer_config["added_tokens_decoder"].object_items()) {
    tokenizer.special_tokens_.push_back(token_info["content"].string_value());
  }
  for (const auto& t : tokenizer.special_tokens_) {
    tokenizer.special_tokens_set_.insert(t);

    SpecialTrieNode* current = tokenizer.special_root_.get();
    for (const unsigned char ch : t) {
      auto& next = current->next[ch];
      if (!next) {
        next = std::make_unique<SpecialTrieNode>();
      }
      current = next.get();
    }
    current->token_id = tokenizer.TokenToId(t);
  }
  return tokenizer;
}

BpeTokenizer::SpecialMatch BpeTokenizer::MatchSpecial(std::string_view text,
                                                      std::size_t position) const {
  const SpecialTrieNode* current = special_root_.get();
  if (current == nullptr) {
    return {};
  }

  std::int64_t best_token_id = kInvalidTokenId;
  std::size_t best_end = position;
  for (std::size_t i = position; i < text.size(); ++i) {
    const auto ch = static_cast<unsigned char>(text[i]);
    if (!current->next[ch]) {
      break;
    }
    current = current->next[ch].get();
    if (current->token_id != kInvalidTokenId) {
      best_token_id = current->token_id;
      best_end = i + 1;
    }
  }

  if (best_token_id == kInvalidTokenId) {
    return {};
  }
  return SpecialMatch{true, best_token_id, best_end};
}

std::size_t BpeTokenizer::FindNextSpecialStart(std::string_view text, std::size_t position) const {
  const SpecialTrieNode* current = special_root_.get();
  if (current == nullptr) {
    return text.size();
  }

  for (std::size_t i = position; i < text.size(); ++i) {
    const auto ch = static_cast<unsigned char>(text[i]);
    if (!current->next[ch]) {
      continue;
    }
    if (MatchSpecial(text, i).matched) {
      return i;
    }
  }
  return text.size();
}

std::vector<std::string> BpeTokenizer::Pretokenize(std::string_view text) const {
  const auto normalized = NormalizeNfc(std::string(text));
  const auto spans = DecodeUtf8Spans(normalized);
  std::vector<std::string> pieces;
  for (std::size_t i = 0; i < spans.size();) {
    const auto& span = spans[i];
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'s")) {
      pieces.push_back(normalized.substr(span.begin, 2));
      i += 2;
      continue;
    }
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'t")) {
      pieces.push_back(normalized.substr(span.begin, 2));
      i += 2;
      continue;
    }
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'re")) {
      pieces.push_back(normalized.substr(span.begin, 3));
      i += 3;
      continue;
    }
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'ve")) {
      pieces.push_back(normalized.substr(span.begin, 3));
      i += 3;
      continue;
    }
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'m")) {
      pieces.push_back(normalized.substr(span.begin, 2));
      i += 2;
      continue;
    }
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'ll")) {
      pieces.push_back(normalized.substr(span.begin, 3));
      i += 3;
      continue;
    }
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'d")) {
      pieces.push_back(normalized.substr(span.begin, 2));
      i += 2;
      continue;
    }

    if (IsWhitespace(span.codepoint) && i + 1 < spans.size() &&
        IsPunctuation(spans[i + 1].codepoint)) {
      std::size_t j = i + 1;
      while (j < spans.size() && IsPunctuation(spans[j].codepoint)) {
        ++j;
      }
      while (j < spans.size() && IsNewline(spans[j].codepoint)) {
        ++j;
      }
      pieces.push_back(normalized.substr(span.begin, spans[j - 1].end - span.begin));
      i = j;
      continue;
    }

    if (span.codepoint == U' ' && i + 1 < spans.size() && IsLetter(spans[i + 1].codepoint)) {
      std::size_t j = i + 1;
      while (j < spans.size() && IsLetter(spans[j].codepoint)) {
        ++j;
      }
      pieces.push_back(normalized.substr(span.begin, spans[j - 1].end - span.begin));
      i = j;
      continue;
    }

    if (IsLetter(span.codepoint)) {
      std::size_t j = i + 1;
      while (j < spans.size() && IsLetter(spans[j].codepoint)) {
        ++j;
      }
      pieces.push_back(normalized.substr(span.begin, spans[j - 1].end - span.begin));
      i = j;
      continue;
    }

    if (IsAsciiDigit(span.codepoint)) {
      pieces.push_back(normalized.substr(span.begin, span.end - span.begin));
      ++i;
      continue;
    }

    if (IsWhitespace(span.codepoint)) {
      std::size_t j = i;
      while (j < spans.size() && IsWhitespace(spans[j].codepoint)) {
        ++j;
      }
      if (j < spans.size() && IsNewline(spans[j].codepoint)) {
        while (j < spans.size() && IsNewline(spans[j].codepoint)) {
          ++j;
        }
      }
      pieces.push_back(normalized.substr(span.begin, spans[j - 1].end - span.begin));
      i = j;
      continue;
    }

    if (IsNewline(span.codepoint)) {
      std::size_t j = i + 1;
      while (j < spans.size() && IsNewline(spans[j].codepoint)) {
        ++j;
      }
      pieces.push_back(normalized.substr(span.begin, spans[j - 1].end - span.begin));
      i = j;
      continue;
    }

    if (IsPunctuation(span.codepoint)) {
      std::size_t j = i + 1;
      while (j < spans.size() && IsPunctuation(spans[j].codepoint)) {
        ++j;
      }
      while (j < spans.size() && IsNewline(spans[j].codepoint)) {
        ++j;
      }
      pieces.push_back(normalized.substr(span.begin, spans[j - 1].end - span.begin));
      i = j;
      continue;
    }

    pieces.push_back(normalized.substr(span.begin, span.end - span.begin));
    ++i;
  }
  return pieces;
}

void BpeTokenizer::TryPushPair(const std::vector<Symbol>& symbols, int left_index, int right_index,
                               std::priority_queue<PairCandidate>* candidates) const {
  LOCALLLM_CHECK(candidates != nullptr, "Candidate queue must not be null.");
  if (left_index < 0 || right_index < 0) {
    return;
  }
  if (left_index >= static_cast<int>(symbols.size()) ||
      right_index >= static_cast<int>(symbols.size())) {
    return;
  }

  const auto& left = symbols[static_cast<std::size_t>(left_index)];
  const auto& right = symbols[static_cast<std::size_t>(right_index)];
  if (!left.alive || !right.alive) {
    return;
  }
  if (left.next != right_index || right.prev != left_index) {
    return;
  }

  const auto it = merge_table_.find(MakePairKey(left.token_id, right.token_id));
  if (it == merge_table_.end()) {
    return;
  }

  candidates->push(PairCandidate{
      it->second.rank,
      left_index,
      right_index,
      left.token_id,
      right.token_id,
      it->second.merged_token_id,
  });
}

bool BpeTokenizer::IsStillValid(const std::vector<Symbol>& symbols,
                                const PairCandidate& candidate) const {
  if (candidate.left_index < 0 || candidate.right_index < 0) {
    return false;
  }
  if (candidate.left_index >= static_cast<int>(symbols.size()) ||
      candidate.right_index >= static_cast<int>(symbols.size())) {
    return false;
  }

  const auto& left = symbols[static_cast<std::size_t>(candidate.left_index)];
  const auto& right = symbols[static_cast<std::size_t>(candidate.right_index)];
  if (!left.alive || !right.alive) {
    return false;
  }
  if (left.next != candidate.right_index || right.prev != candidate.left_index) {
    return false;
  }
  if (left.token_id != candidate.left_token_snapshot ||
      right.token_id != candidate.right_token_snapshot) {
    return false;
  }
  return true;
}

std::vector<std::int64_t> BpeTokenizer::RunBpe(std::string_view piece) const {
  if (piece.empty()) {
    return {};
  }

  std::vector<Symbol> symbols;
  symbols.reserve(piece.size());
  for (const unsigned char byte : piece) {
    const auto token_id = byte_token_ids_[byte];
    LOCALLLM_CHECK(token_id != kInvalidTokenId, "Missing byte token for input piece.");

    Symbol symbol;
    symbol.token_id = token_id;
    symbol.prev = symbols.empty() ? -1 : static_cast<int>(symbols.size()) - 1;
    symbol.next = -1;
    if (!symbols.empty()) {
      symbols.back().next = static_cast<int>(symbols.size());
    }
    symbols.push_back(symbol);
  }

  std::priority_queue<PairCandidate> candidates;
  for (int i = 0; i + 1 < static_cast<int>(symbols.size()); ++i) {
    TryPushPair(symbols, i, i + 1, &candidates);
  }

  while (!candidates.empty()) {
    const auto candidate = candidates.top();
    candidates.pop();
    if (!IsStillValid(symbols, candidate)) {
      continue;
    }

    auto& left = symbols[static_cast<std::size_t>(candidate.left_index)];
    auto& right = symbols[static_cast<std::size_t>(candidate.right_index)];

    left.token_id = candidate.merged_token_id;
    right.alive = false;

    left.next = right.next;
    if (right.next != -1) {
      symbols[static_cast<std::size_t>(right.next)].prev = candidate.left_index;
    }

    if (left.prev != -1) {
      TryPushPair(symbols, left.prev, candidate.left_index, &candidates);
    }
    if (left.next != -1) {
      TryPushPair(symbols, candidate.left_index, left.next, &candidates);
    }
  }

  std::vector<std::int64_t> token_ids;
  token_ids.reserve(symbols.size());
  for (int index = 0; index != -1; index = symbols[static_cast<std::size_t>(index)].next) {
    const auto& symbol = symbols[static_cast<std::size_t>(index)];
    if (symbol.alive) {
      token_ids.push_back(symbol.token_id);
    }
  }
  return token_ids;
}

void BpeTokenizer::EncodeOrdinaryText(std::string_view text,
                                      std::vector<std::int64_t>* token_ids) const {
  LOCALLLM_CHECK(token_ids != nullptr, "Token id output must not be null.");
  if (text.empty()) {
    return;
  }

  for (const auto& pretoken : Pretokenize(text)) {
    const auto piece_token_ids = RunBpe(pretoken);
    token_ids->insert(token_ids->end(), piece_token_ids.begin(), piece_token_ids.end());
  }
}

std::vector<std::int64_t> BpeTokenizer::Encode(const std::string& text) const {
  std::vector<std::int64_t> token_ids;
  token_ids.reserve(std::min(text.size() / 2 + 8, static_cast<std::size_t>(4096)));

  std::size_t position = 0;
  while (position < text.size()) {
    const auto special_match = MatchSpecial(text, position);
    if (special_match.matched) {
      token_ids.push_back(special_match.token_id);
      position = special_match.end;
      continue;
    }

    const auto next_special = FindNextSpecialStart(text, position);
    if (next_special > position) {
      EncodeOrdinaryText(std::string_view(text).substr(position, next_special - position),
                         &token_ids);
    }
    position = next_special;
  }
  return token_ids;
}

std::string BpeTokenizer::Decode(const std::vector<std::int64_t>& token_ids,
                                 bool skip_special_tokens) const {
  std::string text;
  text.reserve(token_ids.size() * 4);
  std::vector<std::uint8_t> bytes;
  auto flush_bytes = [&]() {
    if (!bytes.empty()) {
      text.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      bytes.clear();
    }
  };

  for (const auto token_id : token_ids) {
    LOCALLLM_CHECK(token_id >= 0, "Negative token id is invalid.");
    LOCALLLM_CHECK(static_cast<std::size_t>(token_id) < id_to_token_.size(),
                   "Token id out of range.");
    const auto& token = id_to_token_[static_cast<std::size_t>(token_id)];
    if (IsSpecialToken(token)) {
      flush_bytes();
      if (!skip_special_tokens) {
        text += token;
      }
      continue;
    }
    for (const auto& span : DecodeUtf8Spans(token)) {
      const auto symbol = token.substr(span.begin, span.end - span.begin);
      const auto it = unicode_to_byte_.find(symbol);
      LOCALLLM_CHECK(it != unicode_to_byte_.end(),
                     "Unknown byte-level token fragment during decode.");
      bytes.push_back(it->second);
    }
  }
  flush_bytes();
  return text;
}

std::string BpeTokenizer::ApplyChatTemplate(const std::string& user_prompt) const {
  return "<|im_start|>user\n" + user_prompt + "<|im_end|>\n<|im_start|>assistant\n";
}

std::int64_t BpeTokenizer::TokenToId(const std::string& token) const {
  const auto it = vocab_.find(token);
  LOCALLLM_CHECK(it != vocab_.end(), "Unknown token in vocabulary: " + token);
  return it->second;
}

bool BpeTokenizer::IsSpecialToken(std::string_view token) const {
  return special_tokens_set_.find(token) != special_tokens_set_.end();
}

std::uint64_t BpeTokenizer::MakePairKey(std::int64_t left_token_id, std::int64_t right_token_id) {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(left_token_id)) << 32U) |
         static_cast<std::uint32_t>(right_token_id);
}

}  // namespace localllm
