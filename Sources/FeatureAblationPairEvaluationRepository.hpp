#pragma once

#include "FeatureAblationPairEvaluation.hpp"

#include <pqxx/pqxx>

namespace EA::FeatureAblationPairEvaluation
{

// Loads through the existing exact paired-evidence repository, then augments
// it with feature-ablation-specific scientific configuration. The caller owns
// the read-only transaction and no persistence path is exposed.
ArmEvidence LoadAuthoritativeArmEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId);

} // namespace EA::FeatureAblationPairEvaluation
