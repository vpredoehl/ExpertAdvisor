#pragma once

#include "ProfitabilityVerification.hpp"

#include <pqxx/pqxx>

namespace EA::ProfitabilityVerification
{

EvidenceResult LoadAndVerifyExactFinalEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId);

CampaignProfitabilityShadowSource LoadCampaignProfitabilityShadowSource(
    pqxx::transaction_base& transaction,
    long long rankingSnapshotId);

CampaignProfitabilityCoverageAudit LoadCampaignProfitabilityCoverageAudit(
    pqxx::transaction_base& transaction,
    long long rankingSnapshotId);

} // namespace EA::ProfitabilityVerification
