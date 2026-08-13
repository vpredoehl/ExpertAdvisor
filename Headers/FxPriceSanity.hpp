#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <optional>
#include <string>
#include <string_view>

namespace EA::FxPriceSanity
{

struct Bounds
{
    double lower = 0.2;
    double upper = 2.0;
};

inline std::string LowerAscii(std::string_view value)
{
    std::string out;
    out.reserve(value.size());
    for (unsigned char ch : value)
        out.push_back(static_cast<char>(std::tolower(ch)));
    return out;
}

inline std::optional<std::string> ExtractSymbol(std::string_view context)
{
    const std::string lower = LowerAscii(context);
    size_t searchPos = 0;
    while (true)
    {
        const size_t rmpPos = lower.find("rmp", searchPos);
        if (rmpPos == std::string::npos)
            return std::nullopt;

        const size_t end = rmpPos + 3;
        size_t begin = rmpPos;
        while (begin > 0)
        {
            const unsigned char ch = static_cast<unsigned char>(lower[begin - 1]);
            if (std::isalnum(ch) == 0 && ch != '_')
                break;
            --begin;
        }

        const std::string symbol = lower.substr(begin, end - begin);
        if (symbol.size() >= 6)
            return symbol;

        searchPos = end;
    }
}

inline bool IsJpySymbol(std::string_view symbolOrContext)
{
    const std::string lower = LowerAscii(symbolOrContext);
    return lower.find("jpy") != std::string::npos;
}

inline Bounds BoundsForSymbol(std::string_view symbolOrContext)
{
    if (IsJpySymbol(symbolOrContext))
        return Bounds{20.0, 300.0};
    return Bounds{0.2, 2.0};
}

inline bool IsSanePrice(double value, const Bounds& bounds)
{
    return std::isfinite(value) && value >= bounds.lower && value <= bounds.upper;
}

inline bool IsSanePrice(double value, std::string_view symbolOrContext)
{
    return IsSanePrice(value, BoundsForSymbol(symbolOrContext));
}

} // namespace EA::FxPriceSanity
