#include "ExperimentRecommendationConversionProposalReviewRepository.hpp"

#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row, const char* column)
{
    const pqxx::field field = row[column];
    if (field.is_null()) return std::nullopt;
    return field.as<Value>();
}

std::string DecisionColumns()
{
    return
        "recommendation_conversion_review_decision_id,"
        "recommendation_conversion_proposal_id,decision,decision_request_id,"
        "operator_identity,reason_text,decided_at::text AS decided_at,"
        "created_at::text AS created_at";
}

PersistedRecommendationConversionProposalReviewDecision MapDecision(
    const pqxx::row& row)
{
    PersistedRecommendationConversionProposalReviewDecision value;
    value.reviewDecisionId =
        row["recommendation_conversion_review_decision_id"].as<long long>();
    value.proposalId =
        row["recommendation_conversion_proposal_id"].as<long long>();
    const auto decision = ParseRecommendationConversionProposalReviewDecision(
        row["decision"].as<std::string>());
    if (!decision)
        throw std::runtime_error(
            "invalid_persisted_conversion_proposal_review_decision");
    value.decision = *decision;
    value.requestId = row["decision_request_id"].as<std::string>();
    value.operatorIdentity =
        OptionalValue<std::string>(row, "operator_identity");
    value.reasonText = OptionalValue<std::string>(row, "reason_text");
    value.decidedAt = row["decided_at"].as<std::string>();
    value.createdAt = row["created_at"].as<std::string>();
    try
    {
        RecommendationConversionProposalReviewRequest request;
        request.proposalId = value.proposalId;
        request.decision = value.decision;
        request.requestId = value.requestId;
        request.operatorIdentity = value.operatorIdentity;
        request.reasonText = value.reasonText;
        const auto normalized =
            NormalizeRecommendationConversionProposalReviewRequest(request);
        if (normalized.requestId != value.requestId ||
            normalized.operatorIdentity != value.operatorIdentity ||
            normalized.reasonText != value.reasonText)
            throw std::runtime_error("noncanonical_persisted_review_text");
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(
            "invalid_persisted_conversion_proposal_review_decision");
    }
    return value;
}

RecommendationConversionProposalReviewDisposition DispositionFor(
    RecommendationConversionProposalReviewDecision decision)
{
    return decision == RecommendationConversionProposalReviewDecision::approve
        ? RecommendationConversionProposalReviewDisposition::approved
        : RecommendationConversionProposalReviewDisposition::rejected;
}

bool Matches(
    const PersistedRecommendationConversionProposalReviewDecision& persisted,
    const RecommendationConversionProposalReviewRequest& request)
{
    return persisted.proposalId == request.proposalId &&
           persisted.decision == request.decision &&
           persisted.requestId == request.requestId &&
           persisted.operatorIdentity == request.operatorIdentity &&
           persisted.reasonText == request.reasonText;
}

void ValidateListLimit(int limit)
{
    if (limit <= 0 ||
        limit > kMaximumRecommendationConversionProposalReviewListLimit)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_review_limit_out_of_range");
}

} // namespace

std::string RecommendationConversionProposalReviewPersistOutcomeText(
    RecommendationConversionProposalReviewPersistOutcome outcome)
{
    switch (outcome)
    {
        case RecommendationConversionProposalReviewPersistOutcome::recorded:
            return "recorded";
        case RecommendationConversionProposalReviewPersistOutcome::
            existingIdentical:
            return "existing_identical";
        case RecommendationConversionProposalReviewPersistOutcome::proposalNotFound:
            return "proposal_not_found";
    }
    throw std::invalid_argument(
        "invalid_recommendation_conversion_proposal_review_persist_outcome");
}

bool RecommendationConversionProposalReviewSchemaExists(
    pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return transaction.exec(
        "SELECT to_regclass("
        "'experiment_recommendation_conversion_review_decision') IS NOT NULL;")
        .one_row()[0].as<bool>();
}

void LockRecommendationConversionProposalReviewSequence(
    pqxx::transaction_base& transaction,
    long long proposalId)
{
    if (proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_review_id_invalid");
    // The proposal is immutable and the runtime role intentionally has no
    // UPDATE privilege, so a row-level FOR UPDATE lock is unavailable. This
    // transaction-scoped lock serializes review decisions and conversion for
    // one proposal without making advisory-lock hash equality authoritative.
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, "
        "-7046029254386353131));",
        pqxx::params{
            "recommendation_conversion_proposal_review_sequence_v1:" +
            std::to_string(proposalId)});
}

RecommendationConversionProposalReviewPersistResult
RecordRecommendationConversionProposalReviewDecision(
    pqxx::connection& connection,
    const RecommendationConversionProposalReviewRequest& request)
{
    const RecommendationConversionProposalReviewRequest normalized =
        NormalizeRecommendationConversionProposalReviewRequest(request);
    pqxx::work transaction{connection};
    LockRecommendationConversionProposalReviewSequence(
        transaction, normalized.proposalId);
    const pqxx::result proposal = transaction.exec(
        "SELECT recommendation_conversion_proposal_id FROM "
        "experiment_recommendation_conversion_proposal WHERE "
        "recommendation_conversion_proposal_id=$1;",
        pqxx::params{normalized.proposalId});
    if (proposal.empty())
    {
        transaction.commit();
        return {
            RecommendationConversionProposalReviewPersistOutcome::proposalNotFound,
            std::nullopt};
    }

    const pqxx::result inserted = transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_review_decision ("
        "recommendation_conversion_proposal_id,decision,decision_request_id,"
        "operator_identity,reason_text) VALUES ($1,$2,$3,$4,$5) "
        "ON CONFLICT (recommendation_conversion_proposal_id,"
        "decision_request_id) DO NOTHING RETURNING " + DecisionColumns() + ";",
        pqxx::params{
            normalized.proposalId,
            RecommendationConversionProposalReviewDecisionText(
                normalized.decision),
            normalized.requestId, normalized.operatorIdentity,
            normalized.reasonText});
    if (!inserted.empty())
    {
        auto persisted = MapDecision(inserted.one_row());
        transaction.commit();
        return {RecommendationConversionProposalReviewPersistOutcome::recorded,
                std::move(persisted)};
    }

    const pqxx::result existing = transaction.exec(
        "SELECT " + DecisionColumns() + " FROM "
        "experiment_recommendation_conversion_review_decision WHERE "
        "recommendation_conversion_proposal_id=$1 AND decision_request_id=$2;",
        pqxx::params{normalized.proposalId, normalized.requestId});
    if (existing.empty())
        throw std::runtime_error(
            "recommendation_conversion_proposal_review_retry_missing");
    auto persisted = MapDecision(existing.one_row());
    if (!Matches(persisted, normalized))
        throw std::runtime_error(
            "recommendation_conversion_proposal_review_request_conflict");
    transaction.commit();
    return {
        RecommendationConversionProposalReviewPersistOutcome::existingIdentical,
        std::move(persisted)};
}

std::optional<PersistedRecommendationConversionProposalReviewDecision>
FindRecommendationConversionProposalReviewDecision(
    pqxx::connection& connection,
    long long reviewDecisionId)
{
    if (reviewDecisionId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_review_decision_id_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + DecisionColumns() + " FROM "
        "experiment_recommendation_conversion_review_decision WHERE "
        "recommendation_conversion_review_decision_id=$1;",
        pqxx::params{reviewDecisionId});
    if (rows.empty()) return std::nullopt;
    return MapDecision(rows.one_row());
}

std::vector<PersistedRecommendationConversionProposalReviewDecision>
ListRecommendationConversionProposalReviewDecisions(
    pqxx::connection& connection,
    long long proposalId,
    int limit)
{
    if (proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_review_id_invalid");
    ValidateListLimit(limit);
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT " + DecisionColumns() + " FROM "
        "experiment_recommendation_conversion_review_decision WHERE "
        "recommendation_conversion_proposal_id=$1 ORDER BY "
        "recommendation_conversion_review_decision_id ASC LIMIT $2;",
        pqxx::params{proposalId, limit});
    std::vector<PersistedRecommendationConversionProposalReviewDecision> values;
    values.reserve(rows.size());
    for (const pqxx::row& row : rows) values.push_back(MapDecision(row));
    return values;
}

std::optional<RecommendationConversionProposalCurrentReview>
GetRecommendationConversionProposalCurrentReview(
    pqxx::connection& connection,
    long long proposalId)
{
    pqxx::read_transaction transaction{connection};
    return GetRecommendationConversionProposalCurrentReview(
        transaction, proposalId);
}

std::optional<RecommendationConversionProposalCurrentReview>
GetRecommendationConversionProposalCurrentReview(
    pqxx::transaction_base& transaction,
    long long proposalId)
{
    if (proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_proposal_review_id_invalid");
    const pqxx::result proposal = transaction.exec(
        "SELECT 1 FROM experiment_recommendation_conversion_proposal WHERE "
        "recommendation_conversion_proposal_id=$1;",
        pqxx::params{proposalId});
    if (proposal.empty()) return std::nullopt;
    const pqxx::result rows = transaction.exec(
        "SELECT " + DecisionColumns() + " FROM "
        "experiment_recommendation_conversion_review_decision WHERE "
        "recommendation_conversion_proposal_id=$1 ORDER BY "
        "recommendation_conversion_review_decision_id DESC LIMIT 1;",
        pqxx::params{proposalId});
    RecommendationConversionProposalCurrentReview current;
    if (!rows.empty())
    {
        current.latestDecision = MapDecision(rows.one_row());
        current.disposition = DispositionFor(current.latestDecision->decision);
    }
    return current;
}

std::vector<RecommendationConversionProposalReviewSummary>
ListRecommendationConversionProposalsByReviewDisposition(
    pqxx::connection& connection,
    std::optional<RecommendationConversionProposalReviewDisposition> disposition,
    int limit)
{
    ValidateListLimit(limit);
    const std::optional<std::string> dispositionText = disposition
        ? std::optional<std::string>{
              RecommendationConversionProposalReviewDispositionText(*disposition)}
        : std::nullopt;
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT p.recommendation_conversion_proposal_id,p.recommendation_id,"
        "p.source_experiment_id,p.conversion_identity_hash,"
        "d.recommendation_conversion_review_decision_id,d.decision,"
        "d.decision_request_id,d.operator_identity,d.reason_text,"
        "d.decided_at::text AS decided_at,d.created_at::text AS created_at,"
        "CASE WHEN d.decision='approve' THEN 'approved' WHEN "
        "d.decision='reject' THEN 'rejected' ELSE 'pending_review' END AS "
        "review_disposition FROM experiment_recommendation_conversion_proposal p "
        "LEFT JOIN LATERAL (SELECT * FROM "
        "experiment_recommendation_conversion_review_decision WHERE "
        "recommendation_conversion_proposal_id="
        "p.recommendation_conversion_proposal_id ORDER BY "
        "recommendation_conversion_review_decision_id DESC LIMIT 1) d ON true "
        "WHERE ($1::text IS NULL OR CASE WHEN d.decision='approve' THEN "
        "'approved' WHEN d.decision='reject' THEN 'rejected' ELSE "
        "'pending_review' END=$1) ORDER BY "
        "p.recommendation_conversion_proposal_id ASC LIMIT $2;",
        pqxx::params{dispositionText, limit});
    std::vector<RecommendationConversionProposalReviewSummary> values;
    values.reserve(rows.size());
    for (const pqxx::row& row : rows)
    {
        RecommendationConversionProposalReviewSummary value;
        value.proposalId =
            row["recommendation_conversion_proposal_id"].as<long long>();
        value.recommendationId = row["recommendation_id"].as<long long>();
        value.sourceExperimentId = row["source_experiment_id"].as<long long>();
        value.conversionIdentityHash =
            row["conversion_identity_hash"].as<std::string>();
        const auto parsed = ParseRecommendationConversionProposalReviewDisposition(
            row["review_disposition"].as<std::string>());
        if (!parsed)
            throw std::runtime_error(
                "invalid_persisted_conversion_proposal_review_disposition");
        value.currentReview.disposition = *parsed;
        if (!row["recommendation_conversion_review_decision_id"].is_null())
            value.currentReview.latestDecision = MapDecision(row);
        values.push_back(std::move(value));
    }
    return values;
}

} // namespace EA::ExperimentRecommendation
