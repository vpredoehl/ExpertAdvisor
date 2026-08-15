#ifndef Donchian20Mode_hpp
#define Donchian20Mode_hpp

#include <stdexcept>
#include <string>

enum class Donchian20Mode
{
    Enabled,
    ZeroAblation
};

inline constexpr Donchian20Mode kDefaultDonchian20Mode =
    Donchian20Mode::Enabled;

inline const char* Donchian20ModeText(Donchian20Mode mode)
{
    switch (mode)
    {
        case Donchian20Mode::Enabled: return "enabled";
        case Donchian20Mode::ZeroAblation: return "zero_ablation";
    }
    throw std::invalid_argument("unsupported Donchian-20 mode");
}

inline Donchian20Mode ParseDonchian20Mode(const std::string& text)
{
    if (text == "enabled") return Donchian20Mode::Enabled;
    if (text == "zero_ablation") return Donchian20Mode::ZeroAblation;
    throw std::invalid_argument("invalid Donchian-20 mode");
}

#endif
