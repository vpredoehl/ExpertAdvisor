#include "ExperimentRecommendationCampaignFollowUpProposalReviewRepository.hpp"

#include "ExperimentRecommendationCampaignFollowUpProposalRepository.hpp"

#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

using Persisted = PersistedRecommendationCampaignFollowUpProposalReview;

std::string ReviewColumns()
{
    return
        "recommendation_campaign_follow_up_proposal_review_event_id,"
        "review_contract_version,"
        "recommendation_campaign_follow_up_proposal_id,"
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash,review_decision,reviewer_identity,"
        "reason_text,review_identity_canonical,review_identity_hash,"
        "created_at::text AS created_at";
}

void ValidateProposalBinding(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignFollowUpProposalReview& review)
{
    const auto persistedProposal =
        FindRecommendationCampaignFollowUpProposal(
            transaction, review.followUpProposalId);
    if (!persistedProposal)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_review_proposal_not_found");
    const auto& proposal = persistedProposal->proposal;
    if (proposal.identity.contractVersion != review.proposalContractVersion ||
        proposal.identity.canonicalText != review.proposalCanonicalText ||
        proposal.identity.hash != review.proposalIdentityHash)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_review_proposal_mismatch");
}

Persisted MapReview(
    pqxx::transaction_base& transaction,
    const pqxx::row& row)
{
    try
    {
        const long long reviewEventId = row[
            "recommendation_campaign_follow_up_proposal_review_event_id"]
                                                .as<long long>();
        const std::string createdAt =
            row["created_at"].as<std::string>();
        if (reviewEventId <= 0 || createdAt.empty())
            throw std::runtime_error("persisted_review_metadata_invalid");
        auto review = BuildRecommendationCampaignFollowUpProposalReview(
            row["review_contract_version"].as<int>(),
            row["recommendation_campaign_follow_up_proposal_id"]
                .as<long long>(),
            row["proposal_contract_version"].as<int>(),
            row["proposal_identity_canonical"].as<std::string>(),
            row["proposal_identity_hash"].as<std::string>(),
            RecommendationCampaignFollowUpProposalReviewDecisionFromText(
                row["review_decision"].as<std::string>()),
            row["reviewer_identity"].as<std::string>(),
            row["reason_text"].as<std::string>());
        if (review.identity.canonicalText !=
                row["review_identity_canonical"].as<std::string>() ||
            review.identity.hash !=
                row["review_identity_hash"].as<std::string>())
            throw std::runtime_error("review_identity_mismatch");
        ValidateProposalBinding(transaction, review);
        return Persisted(
            reviewEventId, std::move(review), createdAt);
    }
    catch (const std::exception& error)
    {
        throw std::runtime_error(
            "invalid_persisted_recommendation_campaign_follow_up_proposal_review:" +
            std::string(error.what()));
    }
}

std::optional<Persisted> FindByProposalId(
    pqxx::transaction_base& transaction,
    long long followUpProposalId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ReviewColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_proposal_review_event "
        "WHERE recommendation_campaign_follow_up_proposal_id=$1 LIMIT 2;",
        pqxx::params{followUpProposalId});
    if (rows.empty()) return std::nullopt;
    if (rows.size() != 1)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_review_duplicate");
    return MapReview(transaction, rows.one_row());
}

} // namespace

PersistedRecommendationCampaignFollowUpProposalReview::
    PersistedRecommendationCampaignFollowUpProposalReview(
        long long reviewEventIdValue,
        RecommendationCampaignFollowUpProposalReview reviewValue,
        std::string createdAtValue)
    : reviewEventId(reviewEventIdValue),
      review(std::move(reviewValue)),
      createdAt(std::move(createdAtValue))
{
}

RecommendationCampaignFollowUpProposalReviewPersistResult::
    RecommendationCampaignFollowUpProposalReviewPersistResult(
        RecommendationCampaignFollowUpProposalReviewPersistOutcome
            outcomeValue,
        PersistedRecommendationCampaignFollowUpProposalReview persistedValue)
    : outcome(outcomeValue), persisted(std::move(persistedValue))
{
}

std::string RecommendationCampaignFollowUpProposalReviewPersistOutcomeText(
    RecommendationCampaignFollowUpProposalReviewPersistOutcome outcome)
{
    switch (outcome)
    {
        case RecommendationCampaignFollowUpProposalReviewPersistOutcome::
                recorded:
            return "recorded";
        case RecommendationCampaignFollowUpProposalReviewPersistOutcome::
                existingIdentical:
            return "existing_identical";
    }
    throw std::invalid_argument(
        "recommendation_campaign_follow_up_proposal_review_persist_outcome_invalid");
}

bool RecommendationCampaignFollowUpProposalReviewSchemaExists(
    pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return RecommendationCampaignFollowUpProposalReviewSchemaExists(
        transaction);
}

bool RecommendationCampaignFollowUpProposalReviewSchemaExists(
    pqxx::transaction_base& transaction)
{
    return transaction.exec(
        "SELECT to_regclass('experiment_recommendation_campaign_follow_up_"
        "proposal_review_event') IS NOT NULL;")
        .one_row()[0].as<bool>();
}

RecommendationCampaignFollowUpProposalReviewPersistResult
PersistRecommendationCampaignFollowUpProposalReview(
    pqxx::transaction_base& transaction,
    const RecommendationCampaignFollowUpProposalReview& review)
{
    ValidateRecommendationCampaignFollowUpProposalReview(review);
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, "
        "6063001007003009));",
        pqxx::params{std::to_string(review.followUpProposalId)});
    ValidateProposalBinding(transaction, review);
    if (auto existing = FindByProposalId(
            transaction, review.followUpProposalId))
    {
        if (existing->review != review)
            throw std::runtime_error(
                "recommendation_campaign_follow_up_proposal_review_conflict");
        return {
            RecommendationCampaignFollowUpProposalReviewPersistOutcome::
                existingIdentical,
            std::move(*existing)};
    }

    const long long reviewEventId = transaction.exec(
        "INSERT INTO experiment_recommendation_campaign_follow_up_proposal_"
        "review_event (review_contract_version,"
        "recommendation_campaign_follow_up_proposal_id,"
        "proposal_contract_version,proposal_identity_canonical,"
        "proposal_identity_hash,review_decision,reviewer_identity,"
        "reason_text,review_identity_canonical,review_identity_hash) VALUES "
        "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING "
        "recommendation_campaign_follow_up_proposal_review_event_id;",
        pqxx::params{review.identity.contractVersion,
            review.followUpProposalId, review.proposalContractVersion,
            review.proposalCanonicalText, review.proposalIdentityHash,
            RecommendationCampaignFollowUpProposalReviewDecisionText(
                review.decision),
            review.reviewerIdentity, review.reasonText,
            review.identity.canonicalText, review.identity.hash})
                                        .one_row()[0].as<long long>();
    const auto persisted = FindRecommendationCampaignFollowUpProposalReview(
        transaction, reviewEventId);
    if (!persisted)
        throw std::runtime_error(
            "recommendation_campaign_follow_up_proposal_review_insert_missing");
    return {
        RecommendationCampaignFollowUpProposalReviewPersistOutcome::recorded,
        std::move(*persisted)};
}

std::optional<PersistedRecommendationCampaignFollowUpProposalReview>
FindRecommendationCampaignFollowUpProposalReview(
    pqxx::connection& connection,
    long long reviewEventId)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationCampaignFollowUpProposalReview(
        transaction, reviewEventId);
}

std::optional<PersistedRecommendationCampaignFollowUpProposalReview>
FindRecommendationCampaignFollowUpProposalReview(
    pqxx::transaction_base& transaction,
    long long reviewEventId)
{
    if (reviewEventId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_review_event_id_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + ReviewColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_proposal_review_event "
        "WHERE recommendation_campaign_follow_up_proposal_review_event_id=$1;",
        pqxx::params{reviewEventId});
    if (rows.empty()) return std::nullopt;
    return MapReview(transaction, rows.one_row());
}

std::optional<PersistedRecommendationCampaignFollowUpProposalReview>
FindRecommendationCampaignFollowUpProposalReviewByProposalId(
    pqxx::connection& connection,
    long long followUpProposalId)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationCampaignFollowUpProposalReviewByProposalId(
        transaction, followUpProposalId);
}

std::optional<PersistedRecommendationCampaignFollowUpProposalReview>
FindRecommendationCampaignFollowUpProposalReviewByProposalId(
    pqxx::transaction_base& transaction,
    long long followUpProposalId)
{
    if (followUpProposalId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_review_proposal_id_invalid");
    return FindByProposalId(transaction, followUpProposalId);
}

std::vector<PersistedRecommendationCampaignFollowUpProposalReview>
ListRecommendationCampaignFollowUpProposalReviews(
    pqxx::connection& connection,
    int limit)
{
    pqxx::read_transaction transaction{connection};
    return ListRecommendationCampaignFollowUpProposalReviews(
        transaction, limit);
}

std::vector<PersistedRecommendationCampaignFollowUpProposalReview>
ListRecommendationCampaignFollowUpProposalReviews(
    pqxx::transaction_base& transaction,
    int limit)
{
    if (limit <= 0 ||
        limit >
            kMaximumRecommendationCampaignFollowUpProposalReviewListLimit)
        throw std::invalid_argument(
            "recommendation_campaign_follow_up_proposal_review_list_limit_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + ReviewColumns() + " FROM "
        "experiment_recommendation_campaign_follow_up_proposal_review_event "
        "ORDER BY recommendation_campaign_follow_up_proposal_review_event_id "
        "DESC LIMIT $1;",
        pqxx::params{limit});
    std::vector<PersistedRecommendationCampaignFollowUpProposalReview> reviews;
    reviews.reserve(rows.size());
    for (const auto& row : rows)
        reviews.push_back(MapReview(transaction, row));
    return reviews;
}

} // namespace EA::ExperimentRecommendation
