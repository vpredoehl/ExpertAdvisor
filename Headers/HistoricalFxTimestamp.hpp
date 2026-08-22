#pragma once

#include "PricePoint.hpp"

#include <string>

namespace EA::HistoricalFxTimestamp
{

// Parse the historical DAT_NT/RMP civil timestamp convention.
//
// Source timestamps are naive America/New_York civil timestamps.
// The resulting PriceTP is an absolute UTC instant.
//
// Returns false for malformed timestamps and for DST civil times that are
// nonexistent or ambiguous.
bool ParseNewYorkCivilTimestamp(
    const char* text,
    PriceTP& out);

inline bool ParseNewYorkCivilTimestamp(
    const std::string& text,
    PriceTP& out)
{
    return ParseNewYorkCivilTimestamp(
        text.c_str(),
        out);
}

} // namespace EA::HistoricalFxTimestamp
