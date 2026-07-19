#include "ExperimentRecommendationConversionActivationRepository.hpp"

#include "ExperimentRecommendationConversionExecutionRepository.hpp"

#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string ActivationColumns()
{
    return
        "recommendation_conversion_activation_id,"
        "recommendation_conversion_execution_id,"
        "recommendation_conversion_proposal_id,"
        "recommendation_conversion_review_decision_id,experiment_id,"
        "activation_contract_version,previous_status,previous_phase,"
        "resulting_status,resulting_phase,activation_identity_canonical,"
        "activation_identity_hash,created_at::text AS created_at";
}

PersistedRecommendationConversionActivation MapActivation(const pqxx::row& row)
{
    PersistedRecommendationConversionActivation value;
    value.activationId =
        row["recommendation_conversion_activation_id"].as<long long>();
    value.executionId =
        row["recommendation_conversion_execution_id"].as<long long>();
    value.proposalId =
        row["recommendation_conversion_proposal_id"].as<long long>();
    value.reviewDecisionId =
        row["recommendation_conversion_review_decision_id"].as<long long>();
    value.experimentId = row["experiment_id"].as<long long>();
    value.activationContractVersion =
        row["activation_contract_version"].as<int>();
    value.previousStatus = row["previous_status"].as<std::string>();
    value.previousPhase = row["previous_phase"].as<std::string>();
    value.resultingStatus = row["resulting_status"].as<std::string>();
    value.resultingPhase = row["resulting_phase"].as<std::string>();
    value.activationIdentityCanonical =
        row["activation_identity_canonical"].as<std::string>();
    value.activationIdentityHash =
        row["activation_identity_hash"].as<std::string>();
    value.createdAt = row["created_at"].as<std::string>();
    if (value.activationId <= 0 || value.executionId <= 0 ||
        value.proposalId <= 0 || value.reviewDecisionId <= 0 ||
        value.experimentId <= 0 ||
        value.activationContractVersion !=
            kRecommendationConversionActivationContractVersion ||
        value.previousStatus !=
            kRecommendationConversionActivationPreviousStatus ||
        value.previousPhase !=
            kRecommendationConversionActivationPreviousPhase ||
        value.resultingStatus !=
            kRecommendationConversionActivationResultingStatus ||
        value.resultingPhase !=
            kRecommendationConversionActivationResultingPhase ||
        value.activationIdentityCanonical.empty() ||
        value.activationIdentityHash !=
            RecommendationCanonicalHash(value.activationIdentityCanonical) ||
        value.createdAt.empty())
        throw std::runtime_error(
            "invalid_persisted_recommendation_conversion_activation");
    return value;
}

std::optional<PersistedRecommendationConversionActivation> FindByColumn(
    pqxx::transaction_base& transaction,
    const char* column,
    long long id)
{
    const pqxx::result rows = transaction.exec(
        "SELECT " + ActivationColumns() + " FROM "
        "experiment_recommendation_conversion_activation WHERE " + column +
        "=$1;",
        pqxx::params{id});
    if (rows.empty()) return std::nullopt;
    return MapActivation(rows.one_row());
}

RecommendationConversionActivationIdentity ExpectedIdentity(
    const PersistedRecommendationConversionExecution& execution)
{
    return BuildRecommendationConversionActivationIdentity({
        execution.executionId,
        execution.proposalId,
        execution.reviewDecisionId,
        execution.experimentId,
        execution.executionIdentityHash});
}

bool MatchesExecution(
    const PersistedRecommendationConversionActivation& activation,
    const PersistedRecommendationConversionExecution& execution,
    const RecommendationConversionActivationIdentity& identity)
{
    return activation.executionId == execution.executionId &&
        activation.proposalId == execution.proposalId &&
        activation.reviewDecisionId == execution.reviewDecisionId &&
        activation.experimentId == execution.experimentId &&
        activation.activationIdentityCanonical == identity.canonicalText &&
        activation.activationIdentityHash == identity.hash;
}

struct ExperimentActivationState
{
    std::string status;
    std::string phase;
    std::optional<int> workerPid;
    std::optional<std::string> currentOperation;
    std::optional<int> currentEpoch;
    std::optional<std::string> invocationMode;
    long long duplicateNonce = -1;
    bool started = false;
    bool workerStarted = false;
    bool completed = false;
    bool hasExitCode = false;
    bool hasError = false;
    bool hasLastModel = false;
};

ExperimentActivationState LockExperiment(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT status,phase,worker_pid,current_operation,current_epoch,"
        "invocation_mode,duplicate_nonce,started_at IS NOT NULL AS started,"
        "worker_started_at IS NOT NULL AS worker_started,"
        "completed_at IS NOT NULL AS completed,exit_code IS NOT NULL AS "
        "has_exit_code,error_message IS NOT NULL AS has_error,"
        "last_model_id IS NOT NULL AS has_last_model FROM experiment WHERE "
        "experiment_id=$1 FOR UPDATE;",
        pqxx::params{experimentId});
    if (rows.empty())
        throw std::runtime_error(
            "recommendation_conversion_activation_experiment_missing");
    const pqxx::row row = rows.one_row();
    ExperimentActivationState state;
    state.status = row["status"].as<std::string>();
    state.phase = row["phase"].as<std::string>();
    if (!row["worker_pid"].is_null())
        state.workerPid = row["worker_pid"].as<int>();
    if (!row["current_operation"].is_null())
        state.currentOperation = row["current_operation"].as<std::string>();
    if (!row["current_epoch"].is_null())
        state.currentEpoch = row["current_epoch"].as<int>();
    if (!row["invocation_mode"].is_null())
        state.invocationMode = row["invocation_mode"].as<std::string>();
    state.duplicateNonce = row["duplicate_nonce"].as<long long>();
    state.started = row["started"].as<bool>();
    state.workerStarted = row["worker_started"].as<bool>();
    state.completed = row["completed"].as<bool>();
    state.hasExitCode = row["has_exit_code"].as<bool>();
    state.hasError = row["has_error"].as<bool>();
    state.hasLastModel = row["has_last_model"].as<bool>();
    return state;
}

std::optional<std::string> InvalidStateReason(
    const ExperimentActivationState& state)
{
    if (state.status != kRecommendationConversionActivationPreviousStatus)
        return "activation_requires_paused_status";
    if (state.phase != kRecommendationConversionActivationPreviousPhase)
        return "activation_requires_train_phase";
    if (state.workerPid) return "activation_requires_null_worker_pid";
    if (state.currentOperation)
        return "activation_requires_null_current_operation";
    if (state.currentEpoch) return "activation_requires_null_current_epoch";
    if (!state.invocationMode ||
        *state.invocationMode != "recommendation_conversion")
        return "activation_requires_conversion_invocation_mode";
    if (state.duplicateNonce != 0)
        return "activation_requires_zero_duplicate_nonce";
    if (state.started) return "activation_requires_null_started_at";
    if (state.workerStarted)
        return "activation_requires_null_worker_started_at";
    if (state.completed) return "activation_requires_null_completed_at";
    if (state.hasExitCode) return "activation_requires_null_exit_code";
    if (state.hasError) return "activation_requires_null_error_message";
    if (state.hasLastModel) return "activation_requires_null_last_model_id";
    return std::nullopt;
}

void LockActivationSequence(
    pqxx::transaction_base& transaction,
    long long executionId)
{
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, "
        "4354685564936845355));",
        pqxx::params{
            "recommendation_conversion_activation_sequence_v1:" +
            std::to_string(executionId)});
}

} // namespace

std::string RecommendationConversionActivationOutcomeText(
    RecommendationConversionActivationOutcome outcome)
{
    switch (outcome)
    {
        case RecommendationConversionActivationOutcome::activated:
            return "activated";
        case RecommendationConversionActivationOutcome::existingIdentical:
            return "existing_identical";
        case RecommendationConversionActivationOutcome::notFound:
            return "not_found";
        case RecommendationConversionActivationOutcome::invalidState:
            return "invalid_state";
        case RecommendationConversionActivationOutcome::conflict:
            return "conflict";
    }
    throw std::invalid_argument(
        "invalid_recommendation_conversion_activation_outcome");
}

bool RecommendationConversionActivationSchemaExists(
    pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return transaction.exec(
        "SELECT to_regclass("
        "'experiment_recommendation_conversion_activation') IS NOT NULL;")
        .one_row()[0].as<bool>();
}

RecommendationConversionActivationResult
ActivateRecommendationConversionExecution(
    pqxx::connection& connection,
    long long executionId)
{
    if (executionId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_activation_execution_id_invalid");
    try
    {
        pqxx::work transaction{connection};
        LockActivationSequence(transaction, executionId);
        const auto execution = FindRecommendationConversionExecution(
            transaction, executionId);
        if (!execution)
        {
            transaction.commit();
            return {RecommendationConversionActivationOutcome::notFound,
                    std::nullopt,
                    "conversion_execution_not_found"};
        }

        const ExperimentActivationState state = LockExperiment(
            transaction, execution->experimentId);
        ValidatePersistedRecommendationConversionExecution(
            transaction, *execution);
        const auto identity = ExpectedIdentity(*execution);
        if (auto existing = FindByColumn(
                transaction,
                "recommendation_conversion_execution_id",
                executionId))
        {
            if (!MatchesExecution(*existing, *execution, identity))
            {
                transaction.commit();
                return {RecommendationConversionActivationOutcome::conflict,
                        std::move(existing),
                        "activation_identity_conflict"};
            }
            transaction.commit();
            return {
                RecommendationConversionActivationOutcome::existingIdentical,
                std::move(existing),
                {}};
        }

        if (const auto reason = InvalidStateReason(state))
        {
            transaction.commit();
            return {RecommendationConversionActivationOutcome::invalidState,
                    std::nullopt,
                    *reason};
        }

        auto activation = MapActivation(transaction.exec(
            "INSERT INTO experiment_recommendation_conversion_activation ("
            "recommendation_conversion_execution_id,"
            "recommendation_conversion_proposal_id,"
            "recommendation_conversion_review_decision_id,experiment_id,"
            "activation_contract_version,previous_status,previous_phase,"
            "resulting_status,resulting_phase,activation_identity_canonical,"
            "activation_identity_hash) VALUES ($1,$2,$3,$4,$5,'paused',"
            "'train','pending','train',$6,$7) RETURNING " +
                ActivationColumns() + ";",
            pqxx::params{
                execution->executionId, execution->proposalId,
                execution->reviewDecisionId, execution->experimentId,
                kRecommendationConversionActivationContractVersion,
                identity.canonicalText, identity.hash})
            .one_row());

        const pqxx::result updated = transaction.exec(
            "UPDATE experiment SET status='pending',phase='train',"
            "updated_at=now() WHERE experiment_id=$1 AND status='paused' AND "
            "phase='train' AND worker_pid IS NULL AND current_operation IS "
            "NULL AND current_epoch IS NULL AND invocation_mode="
            "'recommendation_conversion' AND started_at IS NULL AND "
            "duplicate_nonce=0 AND "
            "worker_started_at IS NULL AND completed_at IS NULL AND "
            "exit_code IS NULL AND error_message IS NULL AND last_model_id "
            "IS NULL RETURNING experiment_id;",
            pqxx::params{execution->experimentId});
        if (updated.empty())
            throw std::runtime_error(
                "recommendation_conversion_activation_state_changed");
        transaction.commit();
        return {RecommendationConversionActivationOutcome::activated,
                std::move(activation),
                {}};
    }
    catch (const pqxx::unique_violation&)
    {
        return {RecommendationConversionActivationOutcome::conflict,
                std::nullopt,
                "activation_uniqueness_conflict"};
    }
}

std::optional<PersistedRecommendationConversionActivation>
FindRecommendationConversionActivation(
    pqxx::connection& connection,
    long long activationId)
{
    if (activationId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_activation_id_invalid");
    pqxx::read_transaction transaction{connection};
    return FindByColumn(
        transaction, "recommendation_conversion_activation_id", activationId);
}

std::optional<PersistedRecommendationConversionActivation>
FindRecommendationConversionActivationByExecution(
    pqxx::connection& connection,
    long long executionId)
{
    if (executionId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_activation_execution_id_invalid");
    pqxx::read_transaction transaction{connection};
    return FindByColumn(
        transaction, "recommendation_conversion_execution_id", executionId);
}

std::optional<PersistedRecommendationConversionActivation>
FindRecommendationConversionActivationByExperiment(
    pqxx::connection& connection,
    long long experimentId)
{
    if (experimentId <= 0)
        throw std::invalid_argument(
            "recommendation_conversion_activation_experiment_id_invalid");
    pqxx::read_transaction transaction{connection};
    return FindByColumn(transaction, "experiment_id", experimentId);
}

} // namespace EA::ExperimentRecommendation
