#pragma once

#include "AuthoritativeEconomicEventCandidate.hpp"

#include <vector>

namespace EA::EconomicCalendar
{

// Validates all candidates before any write transaction is opened and returns
// them in deterministic causal order.
std::vector<AuthoritativeEconomicEventCandidate>
ValidateAndOrderEconomicEventCandidates(
    std::vector<AuthoritativeEconomicEventCandidate> candidates);

} // namespace EA::EconomicCalendar
