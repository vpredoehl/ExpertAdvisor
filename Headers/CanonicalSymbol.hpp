#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace EA::CanonicalSymbol
{

inline std::string Normalize(std::string_view raw)
{
    const auto isSpace = [](unsigned char ch) { return std::isspace(ch) != 0; };

    size_t begin = 0;
    while (begin < raw.size() && isSpace(static_cast<unsigned char>(raw[begin])))
        ++begin;

    size_t end = raw.size();
    while (end > begin && isSpace(static_cast<unsigned char>(raw[end - 1])))
        --end;

    std::string normalized;
    normalized.reserve(end - begin);
    for (size_t i = begin; i < end; ++i)
    {
        const unsigned char ch = static_cast<unsigned char>(raw[i]);
        if (std::isalnum(ch) != 0 || ch == '_')
            normalized.push_back(static_cast<char>(std::tolower(ch)));
        else
            throw std::invalid_argument("invalid symbol character in '" + std::string(raw) + "'");
    }

    if (normalized.empty())
        throw std::invalid_argument("symbol must not be empty");

    return normalized;
}

inline std::optional<std::string> TryNormalize(std::string_view raw)
{
    try
    {
        return Normalize(raw);
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }
}

inline bool Equals(std::string_view lhs, std::string_view rhs)
{
    return Normalize(lhs) == Normalize(rhs);
}

} // namespace EA::CanonicalSymbol
