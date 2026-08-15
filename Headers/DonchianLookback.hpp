#ifndef DonchianLookback_hpp
#define DonchianLookback_hpp

#include <cstddef>
#include <stdexcept>
#include <string>

// This bound keeps malformed CLI/database configuration from causing an
// unbounded feature scan while remaining far beyond normal candle windows.
inline constexpr std::size_t kDefaultDonchianLookback = 20;
inline constexpr std::size_t kMaximumDonchianLookback = 10000;

inline std::size_t ValidateDonchianLookback(std::size_t lookback)
{
    if (lookback == 0 || lookback > kMaximumDonchianLookback)
        throw std::invalid_argument("donchian_lookback_must_be_between_1_and_10000");
    return lookback;
}

inline int DonchianLookbackDatabaseValue(std::size_t lookback)
{
    return static_cast<int>(ValidateDonchianLookback(lookback));
}

inline std::size_t ParseDonchianLookback(const std::string& text)
{
    if (text.empty())
        throw std::invalid_argument("invalid_donchian_lookback");
    std::size_t consumed = 0;
    unsigned long long value = 0;
    try
    {
        value = std::stoull(text, &consumed, 10);
    }
    catch (const std::exception&)
    {
        throw std::invalid_argument("invalid_donchian_lookback");
    }
    if (consumed != text.size() || value > kMaximumDonchianLookback)
        throw std::invalid_argument("invalid_donchian_lookback");
    return ValidateDonchianLookback(static_cast<std::size_t>(value));
}

#endif /* DonchianLookback_hpp */
