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

CampaignProfitabilityTemporalFeasibilityAudit
LoadCampaignProfitabilityTemporalFeasibilityAudit(
    pqxx::transaction_base& transaction);

CampaignProfitabilityTemporalCohort LoadCampaignProfitabilityTemporalCohort(
    pqxx::transaction_base& transaction,
    long long rankingSnapshotId);

CampaignProfitabilityOutcomePreparation
LoadCampaignProfitabilityOutcomePreparation(
    pqxx::transaction_base& transaction,
    const std::string& currentDate,
    const std::string& artifactPath = kPhase12ArtifactPath);

CampaignProfitabilityOutcomeJob LoadCampaignProfitabilityOutcomeExecutionJob(
    pqxx::transaction_base& transaction,
    const std::string& cohortHash,
    long long sourceExperimentId,
    long long sourceModelId,
    const std::string& outcomeStart,
    const std::string& outcomeEnd,
    const std::string& jobHash,
    const std::string& currentDate,
    const std::string& artifactPath = kPhase12ArtifactPath);

CampaignProfitabilityOutcomePersistResult
PersistCampaignProfitabilityOutcomeIdempotently(
    pqxx::transaction_base& transaction,
    const CampaignProfitabilityOutcomePersistRequest& request);

} // namespace EA::ProfitabilityVerification
