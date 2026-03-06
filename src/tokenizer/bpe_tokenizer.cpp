#include "localllm/tokenizer/bpe_tokenizer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>

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

std::string NormalizeNfc(const std::string& text) {
  return text;
}

bool IsAsciiAlpha(char32_t cp) {
  return (cp >= U'a' && cp <= U'z') || (cp >= U'A' && cp <= U'Z');
}

bool IsAsciiDigit(char32_t cp) { return cp >= U'0' && cp <= U'9'; }

bool IsNewline(char32_t cp) { return cp == U'\n' || cp == U'\r'; }

bool IsWhitespace(char32_t cp) {
  return cp == U' ' || cp == U'\t' || cp == 0x3000 || cp == 0x00A0;
}

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

bool StartsWithCaseInsensitive(const std::string& text, std::size_t offset, const std::string& suffix) {
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
  std::vector<int> byte_values;
  std::vector<int> code_points;
  for (int b = 33; b <= 126; ++b) {
    byte_values.push_back(b);
    code_points.push_back(b);
  }
  for (int b = 161; b <= 172; ++b) {
    byte_values.push_back(b);
    code_points.push_back(b);
  }
  for (int b = 174; b <= 255; ++b) {
    byte_values.push_back(b);
    code_points.push_back(b);
  }
  int extra = 0;
  for (int b = 0; b <= 255; ++b) {
    if (std::find(byte_values.begin(), byte_values.end(), b) == byte_values.end()) {
      byte_values.push_back(b);
      code_points.push_back(256 + extra);
      ++extra;
    }
  }
  for (std::size_t i = 0; i < byte_values.size(); ++i) {
    table[static_cast<std::size_t>(byte_values[i])] = EncodeUtf8(static_cast<char32_t>(code_points[i]));
  }
  return table;
}

std::vector<std::string> ParseMerges(const std::string& text) {
  std::vector<std::string> merges;
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    const auto line_end = text.find('\n', cursor);
    auto line = text.substr(cursor, line_end == std::string::npos ? std::string::npos : line_end - cursor);
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
  const auto vocab_text = ReadTextFile(model_dir / "vocab.json");
  const auto merges_text = ReadTextFile(model_dir / "merges.txt");
  const auto tokenizer_config_text = ReadTextFile(model_dir / "tokenizer_config.json");

  std::string error;
  const auto vocab_json = json11::Json::parse(vocab_text, error);
  LOCALLLM_CHECK(error.empty(), "Failed to parse vocab.json: " + error);
  error.clear();
  const auto tokenizer_config = json11::Json::parse(tokenizer_config_text, error);
  LOCALLLM_CHECK(error.empty(), "Failed to parse tokenizer_config.json: " + error);

  tokenizer.id_to_token_.resize(vocab_json.object_items().size());
  for (const auto& [token, id_json] : vocab_json.object_items()) {
    const auto id = static_cast<std::int64_t>(id_json.int_value());
    tokenizer.vocab_[token] = id;
    if (static_cast<std::size_t>(id) >= tokenizer.id_to_token_.size()) {
      tokenizer.id_to_token_.resize(static_cast<std::size_t>(id) + 1);
    }
    tokenizer.id_to_token_[static_cast<std::size_t>(id)] = token;
  }

  for (const auto& [id_text, token_info] : tokenizer_config["added_tokens_decoder"].object_items()) {
    const auto id = static_cast<std::int64_t>(std::stoll(id_text));
    const auto token = token_info["content"].string_value();
    tokenizer.vocab_[token] = id;
    if (static_cast<std::size_t>(id) >= tokenizer.id_to_token_.size()) {
      tokenizer.id_to_token_.resize(static_cast<std::size_t>(id) + 1);
    }
    tokenizer.id_to_token_[static_cast<std::size_t>(id)] = token;
  }

  int rank = 0;
  for (const auto& line : ParseMerges(merges_text)) {
    tokenizer.merge_ranks_[line] = rank++;
  }

  for (const auto& [_, token_info] : tokenizer_config["added_tokens_decoder"].object_items()) {
    tokenizer.special_tokens_.push_back(token_info["content"].string_value());
  }
  std::sort(
      tokenizer.special_tokens_.begin(),
      tokenizer.special_tokens_.end(),
      [](const std::string& lhs, const std::string& rhs) { return lhs.size() > rhs.size(); });

  const auto byte_to_unicode = BuildByteToUnicode();
  for (std::size_t i = 0; i < byte_to_unicode.size(); ++i) {
    tokenizer.byte_to_unicode_[i] = byte_to_unicode[i];
    tokenizer.unicode_to_byte_[byte_to_unicode[i]] = static_cast<std::uint8_t>(i);
  }
  return tokenizer;
}

std::vector<BpeTokenizer::PretokenizedPiece> BpeTokenizer::SplitSpecialPieces(const std::string& text) const {
  std::vector<PretokenizedPiece> pieces;
  std::size_t cursor = 0;
  while (cursor < text.size()) {
    bool matched_special = false;
    for (const auto& token : special_tokens_) {
      if (text.compare(cursor, token.size(), token) == 0) {
        pieces.push_back(PretokenizedPiece{token, true});
        cursor += token.size();
        matched_special = true;
        break;
      }
    }
    if (matched_special) {
      continue;
    }
    std::size_t next_special = std::string::npos;
    for (const auto& token : special_tokens_) {
      const auto position = text.find(token, cursor);
      if (position != std::string::npos) {
        next_special = std::min(next_special, position);
      }
    }
    const auto end = next_special == std::string::npos ? text.size() : next_special;
    pieces.push_back(PretokenizedPiece{text.substr(cursor, end - cursor), false});
    cursor = end;
  }
  return pieces;
}

std::vector<std::string> BpeTokenizer::Pretokenize(const std::string& text) const {
  const auto normalized = NormalizeNfc(text);
  const auto spans = DecodeUtf8Spans(normalized);
  std::vector<std::string> pieces;
  for (std::size_t i = 0; i < spans.size();) {
    const auto& span = spans[i];
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'s")) {
      pieces.push_back(normalized.substr(span.begin, 2));
      ++i;
      continue;
    }
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'t")) {
      pieces.push_back(normalized.substr(span.begin, 2));
      ++i;
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
      ++i;
      continue;
    }
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'ll")) {
      pieces.push_back(normalized.substr(span.begin, 3));
      i += 3;
      continue;
    }
    if (span.codepoint == U'\'' && StartsWithCaseInsensitive(normalized, span.begin, "'d")) {
      pieces.push_back(normalized.substr(span.begin, 2));
      ++i;
      continue;
    }

    if (IsWhitespace(span.codepoint) && i + 1 < spans.size() && IsPunctuation(spans[i + 1].codepoint)) {
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

std::vector<std::string> BpeTokenizer::RunBpe(const std::string& piece) const {
  std::vector<std::string> symbols;
  symbols.reserve(piece.size());
  for (const unsigned char byte : piece) {
    symbols.push_back(byte_to_unicode_[byte]);
  }
  while (symbols.size() > 1) {
    int best_rank = std::numeric_limits<int>::max();
    std::size_t best_index = symbols.size();
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      const std::string pair = symbols[i] + " " + symbols[i + 1];
      const auto it = merge_ranks_.find(pair);
      if (it != merge_ranks_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_index = i;
      }
    }
    if (best_index == symbols.size()) {
      break;
    }
    symbols[best_index] += symbols[best_index + 1];
    symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
  }
  return symbols;
}

std::vector<std::int64_t> BpeTokenizer::Encode(const std::string& text) const {
  std::vector<std::int64_t> token_ids;
  for (const auto& piece : SplitSpecialPieces(text)) {
    if (piece.text.empty()) {
      continue;
    }
    if (piece.is_special) {
      token_ids.push_back(TokenToId(piece.text));
      continue;
    }
    for (const auto& pretoken : Pretokenize(piece.text)) {
      for (const auto& token : RunBpe(pretoken)) {
        token_ids.push_back(TokenToId(token));
      }
    }
  }
  return token_ids;
}

std::string BpeTokenizer::Decode(const std::vector<std::int64_t>& token_ids, bool skip_special_tokens) const {
  std::string text;
  std::vector<std::uint8_t> bytes;
  auto flush_bytes = [&]() {
    if (!bytes.empty()) {
      text.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      bytes.clear();
    }
  };

  for (const auto token_id : token_ids) {
    LOCALLLM_CHECK(token_id >= 0, "Negative token id is invalid.");
    LOCALLLM_CHECK(static_cast<std::size_t>(token_id) < id_to_token_.size(), "Token id out of range.");
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
      LOCALLLM_CHECK(it != unicode_to_byte_.end(), "Unknown byte-level token fragment during decode.");
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
  return std::find(special_tokens_.begin(), special_tokens_.end(), token) != special_tokens_.end();
}

}  // namespace localllm
