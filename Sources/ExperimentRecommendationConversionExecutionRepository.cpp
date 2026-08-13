#include "ExperimentRecommendationConversionExecutionRepository.hpp"

#include "ExperimentRecommendationConversionProposalReviewRepository.hpp"
#include "ExperimentRecommendationRepository.hpp"

#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string ExecutionColumns()
{
    return
        "recommendation_conversion_execution_id,"
        "recommendation_conversion_proposal_id,"
        "recommendation_conversion_review_decision_id,experiment_id,"
        "execution_contract_version,authorization_decision,"
        "execution_identity_canonical,"
        "execution_identity_hash,created_at::text AS created_at";
}

PersistedRecommendationConversionExecution MapExecution(const pqxx::row& row)
{
    PersistedRecommendationConversionExecution value;
    value.executionId =
        row["recommendation_conversion_execution_id"].as<long long>();
    value.proposalId =
        row["recommendation_conversion_proposal_id"].as<long long>();
    value.reviewDecisionId =
        row["recommendation_conversion_review_decision_id"].as<long long>();
    value.experimentId = row["experiment_id"].as<long long>();
    value.executionContractVersion =
        row["execution_contract_version"].as<int>();
    const auto authorizationDecision =
        ParseRecommendationConversionProposalReviewDecision(
            row["authorization_decision"].as<std::string>());
    if (!authorizationDecision)
        throw std::runtime_error(
            "invalid_persisted_recommendation_conversion_authorization");
    value.authorizationDecision = *authorizationDecision;
    value.executionIdentityCanonical =
        row["execution_identity_canonical"].as<std::string>();
    value.executionIdentityHash =
        row["execution_identity_hash"].as<std::string>();
    value.createdAt = row["created_at"].as<std::string>();
    if (value.executionId <= 0 || value.proposalId <= 0 ||
        value.reviewDecisionId <= 0 || value.experimentId <= 0 ||
        value.executionContractVersion !=
            kRecommendationConversionExecutionContractVersion ||
        value.authorizationDecision !=
            RecommendationConversionProposalReviewDecision::approve ||
        value.executionIdentityCanonical.empty() ||
        value.executionIdentityHash !=
            RecommendationCanonicalHash(value.executionIdentityCanonical) ||
        value.createdAt.empty())
        throw std::runtime_error(
            "invalid_persisted_recommendation_conversion_execution");
    return value;
}

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

std::string BuildExecutionIdentityImpl(
    const PersistedRecommendationConversionProposal& proposal,
    long long reviewDecisionId)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_conversion_execution_v1"
        << ";contract_version="
        << kRecommendationConversionExecutionContractVersion
        << ";proposal_id=" << proposal.proposalId
        << ";review_decision_id=" << reviewDecisionId
        << ";authorization_decision=approve"
        << ";proposal_conversion_identity="
        << LengthText(proposal.proposal.conversionIdentityCanonical)
        << ";proposed_invocation="
        << LengthText(proposal.proposal.proposedInvocationCanonical);
    return out.str();
}

std::optional<PersistedRecommendationConversionExecution> FindByProposal(
    pqxx::transaction_base& transaction,
    long long proposalId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ExecutionColumns() + " FROM "
        "experiment_recommendation_conversion_execution WHERE "
        "recommendation_conversion_proposal_id=$1;",
        pqxx::params{proposalId});
    if (rows.empty()) return std::nullopt;
    return MapExecution(rows.one_row());
}

ExperimentInvocationConfiguration LoadExperimentInvocation(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT symbol,prediction_horizon,c_next_threshold,core_lr_mult,"
        "head_lr_mult,target_epochs,checkpoint_interval,"
        "to_char(train_start AT TIME ZONE $2,'YYYY-MM-DD') AS train_start,"
        "to_char(train_end AT TIME ZONE $2,'YYYY-MM-DD') AS train_end,"
        "to_char(infer_start AT TIME ZONE $2,'YYYY-MM-DD') AS infer_start,"
        "to_char(infer_end AT TIME ZONE $2,'YYYY-MM-DD') AS infer_end,"
        "resume_model_id FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId, kRecommendationDateTimeZone});
    if (rows.empty())
        throw std::runtime_error(
            "recommendation_conversion_execution_experiment_missing");
    const pqxx::row row = rows.one_row();
    ExperimentInvocationConfiguration invocation;
    invocation.configuration.symbol = row["symbol"].as<std::string>();
    invocation.configuration.predictionHorizon =
        row["prediction_horizon"].as<int>();
    invocation.configuration.labelThreshold =
        row["c_next_threshold"].as<double>();
    if (!row["core_lr_mult"].is_null())
        invocation.configuration.coreLrMult =
            row["core_lr_mult"].as<double>();
    if (!row["head_lr_mult"].is_null())
        invocation.configuration.headLrMult =
            row["head_lr_mult"].as<double>();
    invocation.configuration.targetEpochs = row["target_epochs"].as<int>();
    invocation.configuration.trainStartDate =
        row["train_start"].as<std::string>();
    invocation.configuration.trainEndDate =
        row["train_end"].as<std::string>();
    if (!row["infer_start"].is_null())
        invocation.configuration.inferStartDate =
            row["infer_start"].as<std::string>();
    if (!row["infer_end"].is_null())
        invocation.configuration.inferEndDate =
            row["infer_end"].as<std::string>();
    invocation.checkpointInterval = row["checkpoint_interval"].as<int>();
    if (!row["resume_model_id"].is_null())
        invocation.resumeModelId = row["resume_model_id"].as<long long>();
    return invocation;
}

void ValidateExisting(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationConversionExecution& execution,
    const PersistedRecommendationConversionProposal& proposal)
{
    const std::string canonical =
        BuildExecutionIdentityImpl(proposal, execution.reviewDecisionId);
    if (execution.proposalId != proposal.proposalId ||
        execution.executionIdentityCanonical != canonical ||
        execution.executionIdentityHash != RecommendationCanonicalHash(canonical))
        throw std::runtime_error(
            "inconsistent_persisted_recommendation_conversion_execution");
    const pqxx::result authorization = transaction.exec(
        "SELECT decision FROM "
        "experiment_recommendation_conversion_review_decision WHERE "
        "recommendation_conversion_review_decision_id=$1 AND "
        "recommendation_conversion_proposal_id=$2;",
        pqxx::params{execution.reviewDecisionId, execution.proposalId});
    if (authorization.empty() ||
        authorization.one_row()["decision"].as<std::string>() != "approve")
        throw std::runtime_error(
            "invalid_persisted_recommendation_conversion_authorization");
    const auto invocation = LoadExperimentInvocation(
        transaction, execution.experimentId);
    if (BuildRecommendationInvocationIdentity(invocation).canonicalText !=
        proposal.proposal.proposedInvocationCanonical)
        throw std::runtime_error(
            "inconsistent_persisted_recommendation_conversion_experiment");
}

long long InsertPausedExperiment(
    pqxx::transaction_base& transaction,
    const ExperimentInvocationConfiguration& invocation)
{
    const auto& configuration = invocation.configuration;
    return transaction.exec(
        "INSERT INTO experiment (symbol,prediction_horizon,c_next_threshold,"
        "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start,train_end,infer_start,infer_end,resume_model_id,"
        "duplicate_nonce,status,phase,invocation_mode,updated_at) VALUES ("
        "$1,$2,$3,$4,$5,$6,$7,"
        "($8::date::timestamp AT TIME ZONE $13),"
        "($9::date::timestamp AT TIME ZONE $13),"
        "CASE WHEN $10::text IS NULL THEN NULL ELSE "
        "($10::date::timestamp AT TIME ZONE $13) END,"
        "CASE WHEN $11::text IS NULL THEN NULL ELSE "
        "($11::date::timestamp AT TIME ZONE $13) END,"
        "$12,0,'paused','train','recommendation_conversion',now()) "
        "RETURNING experiment_id;",
        pqxx::params{
            configuration.symbol, configuration.predictionHorizon,
            configuration.labelThreshold, configuration.coreLrMult,
            configuration.headLrMult, configuration.targetEpochs,
            invocation.checkpointInterval, configuration.trainStartDate,
            configuration.trainEndDate, configuration.inferStartDate,
            configuration.inferEndDate, invocation.resumeModelId,
            kRecommendationDateTimeZone})
        .one_row()[0].as<long long>();
}

} // namespace

std::string BuildRecommendationConversionExecutionIdentityCanonical(
    const PersistedRecommendationConversionProposal& proposal,
    long long reviewDecisionId)
{
    return BuildExecutionIdentityImpl(proposal, reviewDecisionId);
}

std::string RecommendationConversionExecutionOutcomeText(
    RecommendationConversionExecutionOutcome outcome)
{
    switch (outcome)
    {
        case RecommendationConversionExecutionOutcome::created:
            return "created";
        case RecommendationConversionExecutionOutcome::existingIdentical:
            return "existing_identical";
        case RecommendationConversionExecutionOutcome::proposalNotFound:
            return "proposal_not_found";
        case RecommendationConversionExecutionOutcome::pendingReview:
            return "pending_review";
        case RecommendationConversionExecutionOutcome::rejected:
            return "rejected";
    }
    throw std::invalid_argument(
        "invalid_recommendation_conversion_execution_outcome");
}

bool RecommendationConversionExecutionSchemaExists(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return transaction.exec(
        "SELECT to_regclass("
        "'experiment_recommendation_conversion_execution') IS NOT NULL;")
        .one_row()[0].as<bool>();
}

RecommendationConversionExecutionResult
ExecuteApprovedRecommendationConversionProposalInTransaction(
    pqxx::transaction_base& transaction,
    long long proposalId)
{
    if (proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_execution_proposal_id_invalid");
    const pqxx::result proposalRow = transaction.exec(
        "SELECT recommendation_conversion_proposal_id FROM "
        "experiment_recommendation_conversion_proposal WHERE "
        "recommendation_conversion_proposal_id=$1;",
        pqxx::params{proposalId});
    if (proposalRow.empty())
    {
        return {RecommendationConversionExecutionOutcome::proposalNotFound,
                std::nullopt};
    }
    const auto proposal = FindRecommendationConversionProposal(
        transaction, proposalId);
    if (!proposal)
        throw std::runtime_error(
            "recommendation_conversion_execution_locked_proposal_missing");
    if (auto existing = FindByProposal(transaction, proposalId))
    {
        ValidateExisting(transaction, *existing, *proposal);
        return {RecommendationConversionExecutionOutcome::existingIdentical,
                std::move(existing)};
    }
    const auto review = GetRecommendationConversionProposalCurrentReview(
        transaction, proposalId);
    if (!review)
        throw std::runtime_error(
            "recommendation_conversion_execution_review_state_missing");
    if (!review->latestDecision)
        return {RecommendationConversionExecutionOutcome::pendingReview,
                std::nullopt};
    if (review->disposition !=
        RecommendationConversionProposalReviewDisposition::approved)
        return {RecommendationConversionExecutionOutcome::rejected,
                std::nullopt};
    const long long reviewDecisionId = review->latestDecision->reviewDecisionId;
    const long long experimentId = InsertPausedExperiment(
        transaction, proposal->proposal.proposedInvocation);
    const std::string canonical = BuildExecutionIdentityImpl(
        *proposal, reviewDecisionId);
    auto execution = MapExecution(transaction.exec(
        "INSERT INTO experiment_recommendation_conversion_execution ("
        "recommendation_conversion_proposal_id,"
        "recommendation_conversion_review_decision_id,experiment_id,"
        "execution_contract_version,authorization_decision,"
        "execution_identity_canonical,execution_identity_hash) VALUES "
        "($1,$2,$3,$4,'approve',$5,$6) RETURNING " +
            ExecutionColumns() + ";",
        pqxx::params{
            proposalId, reviewDecisionId, experimentId,
            kRecommendationConversionExecutionContractVersion, canonical,
            RecommendationCanonicalHash(canonical)})
        .one_row());
    return {RecommendationConversionExecutionOutcome::created,
            std::move(execution)};
}

RecommendationConversionExecutionResult
ExecuteApprovedRecommendationConversionProposal(
    pqxx::connection& connection,
    long long proposalId)
{
    if (proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_execution_proposal_id_invalid");
    try
    {
        pqxx::work transaction{connection};
        LockRecommendationConversionProposalReviewSequence(
            transaction, proposalId);
        auto result = ExecuteApprovedRecommendationConversionProposalInTransaction(
            transaction, proposalId);
        transaction.commit();
        return result;
    }
    catch (const pqxx::unique_violation&)
    {
        throw std::runtime_error(
            "recommendation_conversion_execution_experiment_identity_conflict");
    }
}

std::optional<PersistedRecommendationConversionExecution>
FindRecommendationConversionExecution(
    pqxx::connection& connection,
    long long executionId)
{
    pqxx::read_transaction transaction{connection};
    return FindRecommendationConversionExecution(transaction, executionId);
}

std::optional<PersistedRecommendationConversionExecution>
FindRecommendationConversionExecution(
    pqxx::transaction_base& transaction,
    long long executionId)
{
    if (executionId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_execution_id_invalid");
    const pqxx::result rows = transaction.exec(
        "SELECT " + ExecutionColumns() + " FROM "
        "experiment_recommendation_conversion_execution WHERE "
        "recommendation_conversion_execution_id=$1;",
        pqxx::params{executionId});
    if (rows.empty()) return std::nullopt;
    return MapExecution(rows.one_row());
}

void ValidatePersistedRecommendationConversionExecution(
    pqxx::transaction_base& transaction,
    const PersistedRecommendationConversionExecution& execution)
{
    const auto proposal = FindRecommendationConversionProposal(
        transaction, execution.proposalId);
    if (!proposal)
        throw std::runtime_error(
            "recommendation_conversion_execution_proposal_missing");
    ValidateExisting(transaction, execution, *proposal);
}

std::optional<PersistedRecommendationConversionExecution>
FindRecommendationConversionExecutionByProposal(
    pqxx::connection& connection,
    long long proposalId)
{
    if (proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_execution_proposal_id_invalid");
    pqxx::read_transaction transaction{connection};
    return FindByProposal(transaction, proposalId);
}

std::optional<PersistedRecommendationConversionExecution>
FindRecommendationConversionExecutionByProposal(
    pqxx::transaction_base& transaction,
    long long proposalId)
{
    if (proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_execution_proposal_id_invalid");
    return FindByProposal(transaction, proposalId);
}

} // namespace EA::ExperimentRecommendation
