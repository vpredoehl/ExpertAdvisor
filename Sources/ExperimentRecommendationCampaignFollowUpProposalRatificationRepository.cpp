#include "ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.hpp"

#include "ExperimentRecommendationCampaignFollowUpProposalReviewRepository.hpp"

#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

using Persisted = PersistedRecommendationCampaignFollowUpProposalRatification;

std::string RatificationColumns()
{
    return
        "recommendation_campaign_follow_up_ratification_event_id,"
        "ratification_contract_version,"
        "recommendation_campaign_follow_up_proposal_review_event_id,"
        "review_contract_version,review_identity_canonical,"
        "review_identity_hash,review_decision,reviewer_identity,"
        "recommendation_campaign_follow_up_proposal_id,"
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash,ratification_authority_role,"
        "ratification_decision,ratifier_identity,"
        "ratification_basis,ratification_identity_canonical,ratification_identity_hash,"
        "created_at::text AS created_at";
}

void ValidateReviewBinding(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignFollowUpProposalRatification& ratification)
{
    const auto persistedReview =
        FindRecommendationCampaignFollowUpProposalReview(
            transaction, ratification.reviewEventId);
    if (!persistedReview)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_review_not_found");
    const auto& review = persistedReview->review;
    if (review.decision !=
        RecommendationCampaignFollowUpProposalReviewDecision::approved)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_review_not_eligible");
    if (review.identity.contractVersion != ratification.reviewContractVersion ||
        review.identity.canonicalText != ratification.reviewCanonicalText ||
        review.identity.hash != ratification.reviewIdentityHash ||
        review.followUpProposalId != ratification.followUpProposalId ||
        review.proposalContractVersion != ratification.proposalContractVersion ||
        review.proposalCanonicalText != ratification.proposalCanonicalText ||
        review.proposalIdentityHash != ratification.proposalIdentityHash ||
        review.decision != ratification.reviewDecision ||
        review.reviewerIdentity != ratification.reviewerIdentity)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_review_mismatch");
    if (ratification.ratifierIdentity == review.reviewerIdentity)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_separation_of_duties_violation");
}

Persisted MapRatification(
    pqxx::transaction_base& transaction,
    const pqxx::row& row)
{
    try
    {
        const long long ratificationEventId = row[
            "recommendation_campaign_follow_up_ratification_event_id"]
                                                  .as<long long>();
        const std::string createdAt =
            row["created_at"].as<std::string>();
        if (ratificationEventId <= 0 || createdAt.empty())
            throw std::runtime_error("persisted_ratification_metadata_invalid");
        auto ratification = BuildRecommendationCampaignFollowUpProposalRatification(
            row["ratification_contract_version"].as<int>(),
            row["recommendation_campaign_follow_up_proposal_review_event_id"]
                .as<long long>(),
            row["review_contract_version"].as<int>(),
            row["review_identity_canonical"].as<std::string>(),
            row["review_identity_hash"].as<std::string>(),
            row["recommendation_campaign_follow_up_proposal_id"]
                .as<long long>(),
            row["proposal_contract_version"].as<int>(),
            row["proposal_identity_canonical"].as<std::string>(),
            row["proposal_identity_hash"].as<std::string>(),
            RecommendationCampaignFollowUpProposalReviewDecisionFromText(
                row["review_decision"].as<std::string>()),
            row["reviewer_identity"].as<std::string>(),
            row["ratification_authority_role"].as<std::string>(),
            RecommendationCampaignFollowUpProposalRatificationDecisionFromText(
                row["ratification_decision"].as<std::string>()),
            row["ratifier_identity"].as<std::string>(),
            row["ratification_basis"].as<std::string>());
        if (ratification.identity.canonicalText !=
                row["ratification_identity_canonical"].as<std::string>() ||
            ratification.identity.hash !=
                row["ratification_identity_hash"].as<std::string>())
            throw std::runtime_error("ratification_identity_mismatch");
        ValidateReviewBinding(transaction, ratification);
        return Persisted(
            ratificationEventId, std::move(ratification), createdAt);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            "invalid_persisted_recommendation_campaign_follow_up_proposal_ratification:" +
            std::string(error.what()));
    }
}

std::optional<Persisted> FindByReviewEventId(
    pqxx::transaction_base& transaction,
    long long reviewEventId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + RatificationColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_ratification_event "
        "WHERE recommendation_campaign_follow_up_proposal_review_event_id=$1 "
        "LIMIT 2;",
        pqxx::params{reviewEventId});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_duplicate");
    return MapRatification(transaction, rows.one_row());
}

} // namespace

PersistedRecommendationCampaignFollowUpProposalRatification::
    PersistedRecommendationCampaignFollowUpProposalRatification(
        long long ratificationEventIdValue,
        RecommendationCampaignFollowUpProposalRatification ratificationValue,
        std::string createdAtValue)
    : ratificationEventId(ratificationEventIdValue),
      ratification(std::move(ratificationValue)),
      createdAt(std::move(createdAtValue))
{
}

RecommendationCampaignFollowUpProposalRatificationPersistResult::
    RecommendationCampaignFollowUpProposalRatificationPersistResult(
        RecommendationCampaignFollowUpProposalRatificationPersistOutcome
            outcomeValue,
        PersistedRecommendationCampaignFollowUpProposalRatification persistedValue)
    : outcome(outcomeValue), persisted(std::move(persistedValue))
{
}

std::string RecommendationCampaignFollowUpProposalRatificationPersistOutcomeText(
    RecommendationCampaignFollowUpProposalRatificationPersistOutcome outcome)
{
    switch (outcome)
    {
        case RecommendationCampaignFollowUpProposalRatificationPersistOutcome::
                recorded:
            return "recorded";
        case RecommendationCampaignFollowUpProposalRatificationPersistOutcome::
                existingIdentical:
            return "existing_identical";
    }
    throw std::invalid_argument(
        "recommendation_campaign_follow_up_proposal_ratification_persist_outcome_invalid");
}

bool RecommendationCampaignFollowUpProposalRatificationSchemaExists(
    pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return RecommendationCampaignFollowUpProposalRatificationSchemaExists(
        transaction);
}

bool RecommendationCampaignFollowUpProposalRatificationSchemaExists(
    pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass('experiment_recommendation_campaign_follow_up_"
        "ratification_event') IS NOT NULL;")
        .one_row()[0].as<bool>();
}

RecommendationCampaignFollowUpProposalRatificationPersistResult
PersistRecommendationCampaignFollowUpProposalRatification(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignFollowUpProposalRatification& ratification)
{
    ValidateRecommendationCampaignFollowUpProposalRatification(ratification);
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, "
        "6064001007004009));",
        pqxx::params{std::to_string(ratification.reviewEventId)});
    ValidateReviewBinding(transaction, ratification);
    if (auto existing = FindByReviewEventId(
            transaction, ratification.reviewEventId))
    {
        if (existing->ratification != ratification)
            throw std::runtime_error(
                "recommendation_campaign_follow_up_proposal_ratification_conflict");
        return {
            RecommendationCampaignFollowUpProposalRatificationPersistOutcome::
                existingIdentical,
            std::move(*existing)};
    }

    const long long ratificationEventId = transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_follow_up_"
        "ratification_event (ratification_contract_version,"
        "recommendation_campaign_follow_up_proposal_review_event_id,"
        "review_contract_version,review_identity_canonical,"
        "review_identity_hash,review_decision,reviewer_identity,"
        "recommendation_campaign_follow_up_proposal_id,"
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash,ratification_authority_role,"
        "ratification_decision,ratifier_identity,"
        "ratification_basis,ratification_identity_canonical,ratification_identity_hash) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17) "
        "RETURNING recommendation_campaign_follow_up_ratification_event_id;",
        pqxx::params{ratification.identity.contractVersion,
            ratification.reviewEventId, ratification.reviewContractVersion,
            ratification.reviewCanonicalText, ratification.reviewIdentityHash,
            RecommendationCampaignFollowUpProposalReviewDecisionText(
                ratification.reviewDecision),
            ratification.reviewerIdentity, ratification.followUpProposalId,
            ratification.proposalContractVersion,
            ratification.proposalCanonicalText, ratification.proposalIdentityHash,
            ratification.ratificationAuthorityRole,
            RecommendationCampaignFollowUpProposalRatificationDecisionText(
                ratification.decision),
            ratification.ratifierIdentity, ratification.ratificationBasis,
            ratification.identity.canonicalText, ratification.identity.hash})
                                        .one_row()[0].as<long long>();
    const auto persisted = FindRecommendationCampaignFollowUpProposalRatification(
        transaction, ratificationEventId);
    if (!persisted)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_insert_missing");
    return {
        RecommendationCampaignFollowUpProposalRatificationPersistOutcome::recorded,
        std::move(*persisted)};
}

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatification(
    pqxx::connection& connection,
    long long ratificationEventId)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationCampaignFollowUpProposalRatification(
        transaction, ratificationEventId);
}

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatification(
    pqxx::transaction_base& transaction,
    long long ratificationEventId)
{
    if (ratificationEventId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_ratification_event_id_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + RatificationColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_ratification_event "
        "WHERE recommendation_campaign_follow_up_ratification_event_id=$1;",
        pqxx::params{ratificationEventId});
    if (rows.empty()) return std::nullopt;
    return MapRatification(transaction, rows.one_row());
}

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
    pqxx::connection& connection,
    long long reviewEventId)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
        transaction, reviewEventId);
}

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatificationByReviewEventId(
    pqxx::transaction_base& transaction,
    long long reviewEventId)
{
    if (reviewEventId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_ratification_review_event_id_invalid");
    return FindByReviewEventId(transaction, reviewEventId);
}

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatificationByProposalId(
    pqxx::connection& connection,
    long long followUpProposalId)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationCampaignFollowUpProposalRatificationByProposalId(
        transaction, followUpProposalId);
}

std::optional<PersistedRecommendationCampaignFollowUpProposalRatification>
FindRecommendationCampaignFollowUpProposalRatificationByProposalId(
    pqxx::transaction_base& transaction,
    long long followUpProposalId)
{
    if (followUpProposalId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_ratification_proposal_id_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + RatificationColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_ratification_event "
        "WHERE recommendation_campaign_follow_up_proposal_id=$1 LIMIT 2;",
        pqxx::params{followUpProposalId});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_ratification_duplicate");
    return MapRatification(transaction, rows.one_row());
}

std::vector<PersistedRecommendationCampaignFollowUpProposalRatification>
ListRecommendationCampaignFollowUpProposalRatifications(
    pqxx::connection& connection,
    int limit)
{
    pqxx::read_transaction transaction{connection};
    return ListRecommendationCampaignFollowUpProposalRatifications(
        transaction, limit);
}

std::vector<PersistedRecommendationCampaignFollowUpProposalRatification>
ListRecommendationCampaignFollowUpProposalRatifications(
    pqxx::transaction_base& transaction,
    int limit)
{
    if (limit <= 0 ||
        limit >
            kMaximumRecommendationCampaignFollowUpProposalRatificationListLimit)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_ratification_list_limit_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + RatificationColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_ratification_event "
        "ORDER BY recommendation_campaign_follow_up_ratification_event_id "
        "DESC LIMIT $1;",
        pqxx::params{limit});
    std::vector<PersistedRecommendationCampaignFollowUpProposalRatification>
        ratifications;
    ratifications.reserve(rows.size());
    for (const auto& row : rows)
        ratifications.push_back(MapRatification(transaction, row));
    return ratifications;
}

} // namespace EA::ExperimentRecommendation
