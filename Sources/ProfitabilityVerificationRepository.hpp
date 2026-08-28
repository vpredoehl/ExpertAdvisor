#pragma once

#include "ProfitabilityVerification.hpp"

#include <pqxx/pqxx>

namespace EA::ProfitabilityVerification
{

EvidenceResult LoadAndVerifyExactFinalEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId);

} // namespace EA::ProfitabilityVerification
