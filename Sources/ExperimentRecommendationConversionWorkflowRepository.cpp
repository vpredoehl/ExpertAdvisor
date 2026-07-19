#include "ExperimentRecommendationConversionWorkflowRepository.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::optional<std::string> OptionalText(
    const pqxx::row& row,
    const char* column)
{
    if (row[column].is_null()) return std::nullopt;
    return row[column].as<std::string>();
}

std::string WorkflowQuery()
{
    return R"SQL(
SELECT
    p.recommendation_conversion_proposal_id AS proposal_id,
    p.recommendation_id,
    p.source_experiment_id,
    p.conversion_contract_version AS proposal_contract_version,
    p.conversion_identity_canonical AS proposal_identity_canonical,
    p.conversion_identity_hash AS proposal_identity_hash,
    p.created_at::text AS proposal_created_at,
    latest.recommendation_conversion_review_decision_id AS latest_review_id,
    latest.recommendation_conversion_proposal_id AS latest_review_proposal_id,
    latest.decision AS latest_review_decision,
    latest.decided_at::text AS latest_review_decided_at,
    execution.recommendation_conversion_execution_id AS execution_id,
    execution.recommendation_conversion_proposal_id AS execution_proposal_id,
    execution.recommendation_conversion_review_decision_id AS execution_review_id,
    execution.experiment_id AS execution_experiment_id,
    execution.execution_contract_version,
    execution.authorization_decision,
    execution.execution_identity_canonical,
    execution.execution_identity_hash,
    execution.created_at::text AS execution_created_at,
    execution_authorization.recommendation_conversion_review_decision_id AS authorization_review_id,
    execution_authorization.recommendation_conversion_proposal_id AS authorization_proposal_id,
    execution_authorization.decision AS authorization_decision_value,
    activation.recommendation_conversion_activation_id AS activation_id,
    activation.recommendation_conversion_execution_id AS activation_execution_id,
    activation.recommendation_conversion_proposal_id AS activation_proposal_id,
    activation.recommendation_conversion_review_decision_id AS activation_review_id,
    activation.experiment_id AS activation_experiment_id,
    activation.activation_contract_version,
    activation.previous_status,
    activation.previous_phase,
    activation.resulting_status,
    activation.resulting_phase,
    activation.activation_identity_canonical,
    activation.activation_identity_hash,
    activation.created_at::text AS activation_created_at,
    experiment.experiment_id AS current_experiment_id,
    experiment.status AS experiment_status,
    experiment.phase AS experiment_phase,
    experiment.worker_pid,
    experiment.current_operation,
    experiment.current_epoch,
    experiment.updated_at::text AS experiment_updated_at,
    (SELECT count(*) FROM experiment_recommendation_conversion_execution ec
      WHERE ec.recommendation_conversion_proposal_id =
            p.recommendation_conversion_proposal_id) AS execution_count,
    (SELECT count(*) FROM experiment_recommendation_conversion_activation ac
      WHERE ac.recommendation_conversion_proposal_id =
            p.recommendation_conversion_proposal_id) AS activation_count
FROM experiment_recommendation_conversion_proposal p
LEFT JOIN LATERAL (
    SELECT r.*
    FROM experiment_recommendation_conversion_review_decision r
    WHERE r.recommendation_conversion_proposal_id =
          p.recommendation_conversion_proposal_id
    ORDER BY r.recommendation_conversion_review_decision_id DESC
    LIMIT 1
) latest ON true
LEFT JOIN LATERAL (
    SELECT e.*
    FROM experiment_recommendation_conversion_execution e
    WHERE e.recommendation_conversion_proposal_id =
          p.recommendation_conversion_proposal_id
    ORDER BY e.recommendation_conversion_execution_id
    LIMIT 1
) execution ON true
LEFT JOIN experiment_recommendation_conversion_review_decision execution_authorization
  ON execution_authorization.recommendation_conversion_review_decision_id =
     execution.recommendation_conversion_review_decision_id
LEFT JOIN LATERAL (
    SELECT a.*
    FROM experiment_recommendation_conversion_activation a
    WHERE a.recommendation_conversion_proposal_id =
          p.recommendation_conversion_proposal_id
    ORDER BY a.recommendation_conversion_activation_id
    LIMIT 1
) activation ON true
LEFT JOIN experiment
  ON experiment.experiment_id = execution.experiment_id
)SQL";
}

RecommendationConversionWorkflowView MapWorkflow(const pqxx::row& row)
{
    RecommendationConversionWorkflowView view;
    RecommendationConversionWorkflowFacts facts;
    view.proposalId = row["proposal_id"].as<long long>();
    view.recommendationId = row["recommendation_id"].as<long long>();
    view.sourceExperimentId = row["source_experiment_id"].as<long long>();
    view.proposalContractVersion =
        row["proposal_contract_version"].as<int>();
    view.proposalIdentityCanonical =
        row["proposal_identity_canonical"].as<std::string>();
    view.proposalIdentityHash =
        row["proposal_identity_hash"].as<std::string>();
    view.proposalCreatedAt = row["proposal_created_at"].as<std::string>();

    facts.proposalId = view.proposalId;
    facts.proposalContractVersion = view.proposalContractVersion;
    facts.proposalIdentityCanonical = view.proposalIdentityCanonical;
    facts.proposalIdentityHash = view.proposalIdentityHash;
    facts.executionCount = row["execution_count"].as<long long>();
    facts.activationCount = row["activation_count"].as<long long>();

    if (!row["latest_review_id"].is_null())
    {
        view.latestReview = RecommendationConversionWorkflowReviewFact{
            row["latest_review_id"].as<long long>(),
            row["latest_review_proposal_id"].as<long long>(),
            row["latest_review_decision"].as<std::string>()};
        view.latestReviewDecidedAt = OptionalText(
            row, "latest_review_decided_at");
        facts.latestReview = view.latestReview;
    }

    if (!row["execution_id"].is_null())
    {
        view.execution = RecommendationConversionWorkflowExecutionFact{
            row["execution_id"].as<long long>(),
            row["execution_proposal_id"].as<long long>(),
            row["execution_review_id"].as<long long>(),
            row["execution_experiment_id"].as<long long>(),
            row["execution_contract_version"].as<int>(),
            row["authorization_decision"].as<std::string>(),
            row["execution_identity_canonical"].as<std::string>(),
            row["execution_identity_hash"].as<std::string>()};
        view.executionCreatedAt = OptionalText(row, "execution_created_at");
        facts.execution = view.execution;
    }
    if (!row["authorization_review_id"].is_null())
    {
        view.executionReview = RecommendationConversionWorkflowReviewFact{
            row["authorization_review_id"].as<long long>(),
            row["authorization_proposal_id"].as<long long>(),
            row["authorization_decision_value"].as<std::string>()};
        facts.executionReview = view.executionReview;
    }

    if (!row["activation_id"].is_null())
    {
        view.activation = RecommendationConversionWorkflowActivationFact{
            row["activation_id"].as<long long>(),
            row["activation_execution_id"].as<long long>(),
            row["activation_proposal_id"].as<long long>(),
            row["activation_review_id"].as<long long>(),
            row["activation_experiment_id"].as<long long>(),
            row["activation_contract_version"].as<int>(),
            row["previous_status"].as<std::string>(),
            row["previous_phase"].as<std::string>(),
            row["resulting_status"].as<std::string>(),
            row["resulting_phase"].as<std::string>(),
            row["activation_identity_canonical"].as<std::string>(),
            row["activation_identity_hash"].as<std::string>()};
        view.activationCreatedAt = OptionalText(row, "activation_created_at");
        facts.activation = view.activation;
    }

    if (!row["current_experiment_id"].is_null())
    {
        RecommendationConversionWorkflowExperimentFact experiment;
        experiment.experimentId =
            row["current_experiment_id"].as<long long>();
        experiment.status = row["experiment_status"].as<std::string>();
        experiment.phase = row["experiment_phase"].as<std::string>();
        if (!row["worker_pid"].is_null())
            experiment.workerPid = row["worker_pid"].as<int>();
        if (!row["current_operation"].is_null())
            experiment.currentOperation =
                row["current_operation"].as<std::string>();
        if (!row["current_epoch"].is_null())
            experiment.currentEpoch = row["current_epoch"].as<int>();
        view.experiment = experiment;
        view.experimentUpdatedAt = OptionalText(row, "experiment_updated_at");
        facts.experiment = experiment;
    }

    view.derivation = DeriveRecommendationConversionWorkflow(facts);
    return view;
}

std::optional<RecommendationConversionWorkflowView> FindInTransaction(
    pqxx::transaction_base& transaction,
    long long proposalId)
{
    const pqxx::result rows = transaction.exec(
        WorkflowQuery() + " WHERE p.recommendation_conversion_proposal_id=$1;",
        pqxx::params{proposalId});
    if (rows.empty()) return std::nullopt;
    return MapWorkflow(rows.one_row());
}

void ValidatePositiveId(long long id, const char* message)
{
    if (id <= 0) throw std::invalid_argument(message);
}

void ValidateLimit(int limit)
{
    if (limit <= 0 || limit > kMaximumRecommendationConversionWorkflowListLimit)
        throw std::invalid_argument(
            "recommendation_conversion_workflow_limit_invalid");
}

} // namespace

bool RecommendationConversionWorkflowSchemasExist(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return transaction.exec(R"SQL(
SELECT to_regclass('experiment_recommendation_conversion_proposal') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_review_decision') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_execution') IS NOT NULL
   AND to_regclass('experiment_recommendation_conversion_activation') IS NOT NULL
   AND to_regclass('experiment') IS NOT NULL;
)SQL").one_row()[0].as<bool>();
}

std::optional<RecommendationConversionWorkflowView>
FindRecommendationConversionWorkflow(
    pqxx::connection& connection,
    long long proposalId)
{
    ValidatePositiveId(
        proposalId, "recommendation_conversion_workflow_proposal_id_invalid");
    pqxx::read_transaction transaction{connection};
    return FindInTransaction(transaction, proposalId);
}

std::optional<RecommendationConversionWorkflowView>
FindRecommendationConversionWorkflowByExecution(
    pqxx::connection& connection,
    long long executionId)
{
    ValidatePositiveId(
        executionId, "recommendation_conversion_workflow_execution_id_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT recommendation_conversion_proposal_id FROM "
        "experiment_recommendation_conversion_execution WHERE "
        "recommendation_conversion_execution_id=$1;",
        pqxx::params{executionId});
    if (rows.empty()) return std::nullopt;
    return FindInTransaction(transaction, rows.one_row()[0].as<long long>());
}

std::optional<RecommendationConversionWorkflowView>
FindRecommendationConversionWorkflowByExperiment(
    pqxx::connection& connection,
    long long experimentId)
{
    ValidatePositiveId(
        experimentId, "recommendation_conversion_workflow_experiment_id_invalid");
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT recommendation_conversion_proposal_id FROM "
        "experiment_recommendation_conversion_execution WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (rows.empty()) return std::nullopt;
    return FindInTransaction(transaction, rows.one_row()[0].as<long long>());
}

std::vector<RecommendationConversionWorkflowView>
ListRecommendationConversionWorkflows(
    pqxx::connection& connection,
    int candidateLimit)
{
    ValidateLimit(candidateLimit);
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        WorkflowQuery() +
            " ORDER BY p.recommendation_conversion_proposal_id DESC LIMIT $1;",
        pqxx::params{candidateLimit});
    std::vector<RecommendationConversionWorkflowView> views;
    views.reserve(rows.size());
    for (const pqxx::row& row : rows) views.push_back(MapWorkflow(row));
    return views;
}

std::vector<RecommendationConversionWorkflowView>
ListRecommendationConversionWorkflowsForRecommendations(
    pqxx::transaction_base& transaction,
    const std::vector<long long>& recommendationIds)
{
    if (recommendationIds.size() >
        static_cast<std::size_t>(
            kMaximumRecommendationConversionWorkflowListLimit))
        throw std::invalid_argument(
            "recommendation_conversion_workflow_recommendation_limit_exceeded");
    if (recommendationIds.empty()) return {};

    std::vector<long long> ids = recommendationIds;
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::invalid_argument(
            "recommendation_conversion_workflow_recommendation_duplicate");

    pqxx::params parameters;
    std::ostringstream predicates;
    predicates << " WHERE p.recommendation_id IN (";
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        ValidatePositiveId(
            ids[i], "recommendation_conversion_workflow_recommendation_id_invalid");
        if (i != 0) predicates << ',';
        predicates << '$' << (i + 1);
        parameters.append(ids[i]);
    }
    predicates << ") ORDER BY p.recommendation_id,"
                  "p.recommendation_conversion_proposal_id;";

    const pqxx::result rows = transaction.exec(
        WorkflowQuery() + predicates.str(), parameters);
    std::vector<RecommendationConversionWorkflowView> views;
    views.reserve(rows.size());
    for (const pqxx::row& row : rows) views.push_back(MapWorkflow(row));
    return views;
}

std::vector<RecommendationConversionWorkflowView>
ListRecommendationConversionWorkflowsForProposals(
    pqxx::transaction_base& transaction,
    const std::vector<long long>& proposalIds)
{
    if (proposalIds.size() >
        static_cast<std::size_t>(
            kMaximumRecommendationConversionWorkflowListLimit))
        throw std::invalid_argument(
            "recommendation_conversion_workflow_proposal_limit_exceeded");
    if (proposalIds.empty()) return {};

    std::vector<long long> ids = proposalIds;
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end())
        throw std::invalid_argument(
            "recommendation_conversion_workflow_proposal_duplicate");

    pqxx::params parameters;
    std::ostringstream predicates;
    predicates << " WHERE p.recommendation_conversion_proposal_id IN (";
    for (std::size_t i = 0; i < ids.size(); ++i)
    {
        ValidatePositiveId(
            ids[i], "recommendation_conversion_workflow_proposal_id_invalid");
        if (i != 0) predicates << ',';
        predicates << '$' << (i + 1);
        parameters.append(ids[i]);
    }
    predicates << ") ORDER BY p.recommendation_conversion_proposal_id;";

    const pqxx::result rows = transaction.exec(
        WorkflowQuery() + predicates.str(), parameters);
    std::vector<RecommendationConversionWorkflowView> views;
    views.reserve(rows.size());
    for (const pqxx::row& row : rows) views.push_back(MapWorkflow(row));
    return views;
}

} // namespace EA::ExperimentRecommendation
