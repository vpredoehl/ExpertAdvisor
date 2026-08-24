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

// The only currently evidenced same-agency/family/time exception is the pair
// of distinct date-only FOMC statements published on 2014-09-17.
bool IsPermittedEconomicEventTimestampCoexistence(
    const AuthoritativeEconomicEventCandidate& left,
    const AuthoritativeEconomicEventCandidate& right);

} // namespace EA::EconomicCalendar
