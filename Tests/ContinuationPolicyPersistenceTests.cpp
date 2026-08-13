#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>

#include <pqxx/pqxx>

#include "../Sources/ContinuationPolicyPersistence.hpp"

using namespace EA::ExperimentScheduler;

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

} // namespace

int main()
{
    const std::string connectionString =
        "hostaddr=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=pqxx dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM");
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ ONLY;");
    assert(ContinuationPolicySchemaExists(transaction));

    const pqxx::result queuedSources = transaction.exec(
        "SELECT d.source_experiment_id, d.source_checkpoint_eval_id, "
        "d.policy_revision, d.policy_hash, d.queued_experiment_id "
        "FROM experiment_continuation_decision d "
        "JOIN experiment source ON source.experiment_id = d.source_experiment_id "
        "JOIN experiment child ON child.experiment_id = d.queued_experiment_id "
        "WHERE d.decision = 'continuation_queued' "
        "AND source.continuation_policy_target_epochs = d.target_epochs "
        "ORDER BY d.source_experiment_id LIMIT 1;");
    assert(!queuedSources.empty());
    const long long queuedSourceId =
        queuedSources[0]["source_experiment_id"].as<long long>();
    const std::optional<ContinuationPolicyConfig> directlyLoaded =
        FindContinuationPolicyConfig(transaction, queuedSourceId);
    assert(directlyLoaded.has_value());
    const pqxx::row transactionIdBefore =
        transaction.exec("SELECT txid_current_if_assigned();").one_row();
    const ContinuationAutoPreflightLookup queued =
        LoadAutomaticContinuationPreflightReadOnly(transaction, queuedSourceId);
    const pqxx::row transactionIdAfter =
        transaction.exec("SELECT txid_current_if_assigned();").one_row();
    assert(transaction.exec("SHOW transaction_read_only;").one_row()[0].as<std::string>() ==
           "on");
    assert(transactionIdBefore[0].is_null() == transactionIdAfter[0].is_null());
    if (!transactionIdBefore[0].is_null())
    {
        assert(transactionIdBefore[0].as<unsigned long long>() ==
               transactionIdAfter[0].as<unsigned long long>());
    }
    assert(ContinuationPolicySemanticCanonicalText(*directlyLoaded) ==
           ContinuationPolicySemanticCanonicalText(queued.currentPolicy));
    assert(queued.currentPolicy.sourceExperimentId == queuedSourceId);
    assert(queued.persistedIdentity.has_value());
    const PersistedContinuationIdentity& persisted = *queued.persistedIdentity;
    assert(persisted.sourceExperimentId == queuedSourceId);
    assert(persisted.queuedChildExists);
    assert(persisted.queuedExperimentId ==
           queuedSources[0]["queued_experiment_id"].as<long long>());
    assert(persisted.policyRevision ==
           queuedSources[0]["policy_revision"].as<long long>());
    assert(persisted.policyHash ==
           queuedSources[0]["policy_hash"].as<std::string>());
    if (queuedSources[0]["source_checkpoint_eval_id"].is_null())
    {
        assert(!persisted.sourceCheckpointEvalId.has_value());
        assert(persisted.sourceAnalysisScope == "final");
    }
    else
    {
        assert(persisted.sourceCheckpointEvalId ==
               queuedSources[0]["source_checkpoint_eval_id"].as<long long>());
        assert(persisted.sourceAnalysisScope == "checkpoint");
    }

    const pqxx::result noDecisionSources = transaction.exec(
        "SELECT e.experiment_id FROM experiment e "
        "LEFT JOIN experiment_continuation_decision d "
        "  ON d.source_experiment_id = e.experiment_id "
        "WHERE d.continuation_decision_id IS NULL "
        "ORDER BY e.experiment_id LIMIT 1;");
    assert(!noDecisionSources.empty());
    const ContinuationAutoPreflightLookup noDecision =
        LoadAutomaticContinuationPreflightReadOnly(
            transaction,
            noDecisionSources[0]["experiment_id"].as<long long>());
    assert(!noDecision.persistedIdentity.has_value());
    assert(!noDecision.satisfaction.alreadySatisfied);
    assert(noDecision.satisfaction.reason ==
           "no_persisted_decision_for_current_target");

    if (queued.satisfaction.alreadySatisfied)
    {
        const std::string fields =
            FormatAutomaticContinuationSatisfiedFields(queued, true);
        assert(fields.find(",source_experiment_id=" +
                           std::to_string(queuedSourceId)) != std::string::npos);
        assert(fields.find(",current_policy_hash=") != std::string::npos);
        assert(fields.find(",persisted_decision_policy_hash=") !=
               std::string::npos);
        assert(fields.find(",dry_run=1") != std::string::npos);
    }

    transaction.commit();

    // The manual/queue path uses the same mapper through an explicitly locking
    // entry point. Roll back after proving that lookup remains available.
    pqxx::connection lockingConnection{connectionString};
    pqxx::work lockingTransaction{lockingConnection};
    lockingTransaction.exec("SET TRANSACTION READ WRITE;");
    const std::optional<ContinuationPolicyConfig> locked =
        LockContinuationPolicyConfigForUpdate(
            lockingTransaction,
            queuedSourceId);
    assert(locked.has_value());
    assert(ContinuationPolicySemanticCanonicalText(*locked) ==
           ContinuationPolicySemanticCanonicalText(*directlyLoaded));
    lockingTransaction.abort();

    pqxx::connection writeConnection{connectionString};
    pqxx::work writeTransaction{writeConnection};
    writeTransaction.exec("SET TRANSACTION READ WRITE;");
    bool rejectedWriteCapablePreflight = false;
    try
    {
        (void)LoadAutomaticContinuationPreflightReadOnly(
            writeTransaction,
            queuedSourceId);
    }
    catch (const std::logic_error& error)
    {
        rejectedWriteCapablePreflight =
            std::string{error.what()} ==
            "automatic_continuation_preflight_requires_read_only_transaction";
    }
    assert(rejectedWriteCapablePreflight);
    writeTransaction.abort();

    std::ifstream schedulerSource("Sources/ExperimentScheduler.cpp");
    assert(schedulerSource.is_open());
    const std::string schedulerText{
        std::istreambuf_iterator<char>(schedulerSource),
        std::istreambuf_iterator<char>()};
    assert(schedulerText.find("decision_source_experiment_id") == std::string::npos);
    assert(schedulerText.find("evidence_changed_after_decision") == std::string::npos);
    assert(schedulerText.find("MapPersistedContinuationIdentity") == std::string::npos);
    assert(schedulerText.find("PersistedContinuationIdentity persisted") ==
           std::string::npos);

    std::cout << "ContinuationPolicyPersistenceTests passed\n";
    return 0;
}
