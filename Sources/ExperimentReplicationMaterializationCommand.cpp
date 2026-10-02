#include "ExperimentReplicationMaterialization.hpp"
#include "ExperimentReplicationMaterializationRepository.hpp"

#include "CanonicalSymbol.hpp"
#include "CausalFibonacciStructuralFeatureConfiguration.hpp"
#include "ExperimentComparisonIdentity.hpp"
#include "SchedulerCore/SemanticWorkerRegistry.hpp"
#include "SchedulerCore/TrainingWorkerSelection.hpp"

#include "ExperimentReplicationPlanningPostgres.hpp"

#include <algorithm>
#include <pqxx/pqxx>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentReplicationMaterialization
{
namespace
{

class PostgresFreshExperimentInserter final : public ExperimentInserter
{
public:
    explicit PostgresFreshExperimentInserter(
        pqxx::transaction_base& transaction)
        : transaction_(transaction)
    {
    }

    long long InsertFreshPausedExperiment(
        const Planning::ProposedExperimentSpecification& specification)
        override
    {
        return InsertFreshPausedReplicationExperiment(
            transaction_, specification);
    }

private:
    pqxx::transaction_base& transaction_;
};

class PostgresLayout11ConfluenceInserter final :
    public Layout11ConfluenceExperimentInserter
{
public:
    explicit PostgresLayout11ConfluenceInserter(pqxx::transaction_base& transaction)
        : transaction_(transaction) {}

    long long InsertFreshPausedLayout11ConfluenceExperiment(
        const Layout11ConfluenceArm& arm) override
    {
        return EA::ExperimentReplicationMaterialization::
            InsertFreshPausedLayout11ConfluenceExperiment(
                transaction_, arm.templateExperimentId,
                arm.freshInitializationSeed, arm.featureAblationMask);
    }
private:
    pqxx::transaction_base& transaction_;
};

class NoopLayout11ConfluenceInserter final : public Layout11ConfluenceExperimentInserter
{
public:
    long long InsertFreshPausedLayout11ConfluenceExperiment(
        const Layout11ConfluenceArm&) override
    {
        throw std::logic_error("layout11_confluence_plan_attempted_insert");
    }
};

} // namespace

namespace
{

using Arm = EA::ExperimentPairComparison::ArmResultSet;

std::map<std::string, std::string> ConfiguredIdentity(const Arm& arm)
{
    std::map<std::string, std::string> result;
    for (const auto& field : arm.scientificIdentity)
    {
        const bool configured = std::find(
            EA::ExperimentComparisonIdentity::kConfiguredScientificIdentityFields.begin(),
            EA::ExperimentComparisonIdentity::kConfiguredScientificIdentityFields.end(),
            field.name) !=
            EA::ExperimentComparisonIdentity::kConfiguredScientificIdentityFields.end();
        if (!configured)
            continue;
        if (field.name.empty() || !field.value ||
            !result.emplace(field.name, *field.value).second)
            throw std::invalid_argument(
                "cross_symbol_source_configured_identity_unavailable_or_duplicated");
    }
    if (result.size() !=
        EA::ExperimentComparisonIdentity::kConfiguredScientificIdentityFields.size())
        throw std::invalid_argument("cross_symbol_source_configured_identity_catalog_incomplete");
    return result;
}

std::string Required(const std::map<std::string, std::string>& identity,
                     const char* name)
{
    const auto found = identity.find(name);
    if (found == identity.end())
        throw std::invalid_argument(
            std::string{"cross_symbol_source_identity_missing:"} + name);
    return found->second;
}

void ValidateHistoricalLayout9(const std::map<std::string, std::string>& identity)
{
    if (Required(identity, "configured_model_input_semantic_layout_version") != "9" ||
        Required(identity, "configured_model_input_width") != "103")
        throw std::invalid_argument(
            "cross_symbol_source_requires_layout9_width103");
    (void)Required(identity, "feature_ablation_mask");
    (void)Required(identity, "fresh_initialization_seed");
    (void)Required(identity, "economic_calendar_snapshot_id");
    (void)Required(identity, "training_objective_canonical");
}

Arm MakeProposed(const Arm& source, const std::string& target)
{
    Arm proposed = source;
    std::size_t replaced = 0;
    for (auto& field : proposed.scientificIdentity)
        if (field.name == "symbol")
        {
            field.value = target;
            ++replaced;
        }
    if (replaced != 1)
        throw std::invalid_argument("cross_symbol_source_symbol_identity_invalid");
    return proposed;
}

void RenderIdentity(std::ostream& output,
                    const std::map<std::string, std::string>& identity)
{
    for (const auto& [name, value] : identity)
        output << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION_IDENTITY"
               << ",field=" << name << ",value=" << std::quoted(value) << '\n';
}

void RenderWorker(std::ostream& output, const char* phase,
                  const EA::Scheduler::SemanticWorkerSelection& selection,
                  const EA::Scheduler::SemanticWorkerRegistry& registry)
{
    output << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION_WORKER"
           << ",phase=" << phase
           << ",selected=" << (selection.selected ? "true" : "false")
           << ",rule=" << selection.reason
           << ",diagnostic=" << selection.diagnostic;
    if (selection.selected)
    {
        const auto* worker = registry.findByCanonicalExecutable(
            selection.canonicalExecutablePath);
        if (worker == nullptr)
            throw std::invalid_argument("cross_symbol_selected_worker_unregistered");
        output << ",canonical_executable_path=" << worker->canonicalExecutablePath
               << ",canonical_manifest_path=" << worker->canonicalManifestPath
               << ",executable_sha256=" << worker->sha256
               << ",source_commit=" << worker->sourceCommit
               << ",runtime_identity=" << worker->runtimeIdentity;
    }
    output << '\n';
}

int RunCrossSymbol(pqxx::transaction_base& transaction,
                   const CrossSymbolCommand& command, bool apply,
                   std::ostream& output, std::ostream& errors)
{
    try
    {
        if (command.sourceExperimentId <= 0)
            throw std::invalid_argument("cross_symbol_source_experiment_id_invalid");
        const std::string target = EA::CanonicalSymbol::Normalize(command.targetSymbol);
        // This is the individual LSTM Fibonacci capability boundary; it is
        // deliberately not the frozen TrainingSymbols() sweep universe.
        (void)EA::CausalFibonacciStructuralFeatureConfiguration::Configuration{}
            .FibonacciConfigurationForSymbol(target);

        EA::Scheduler::SemanticWorkerRegistryLoadRequest registryRequest;
        registryRequest.registryPath = command.semanticWorkerRegistryPath;
        const auto registry = EA::Scheduler::SemanticWorkerRegistry::Load(registryRequest);
        const Planning::PostgresPlanningSource source{transaction};
        const Arm sourceArm = EA::ExperimentPairComparison::MakeArmResultSet(
            source.Load(command.sourceExperimentId));
        const auto sourceIdentity = ConfiguredIdentity(sourceArm);
        ValidateHistoricalLayout9(sourceIdentity);
        const std::string sourceSymbol = Required(sourceIdentity, "symbol");
        const Arm proposed = MakeProposed(sourceArm, target);
        const auto proposedIdentity = ConfiguredIdentity(proposed);
        for (const auto& [field, sourceValue] : sourceIdentity)
            if (field != "symbol" &&
                Required(proposedIdentity, field.c_str()) != sourceValue)
                throw std::invalid_argument(
                    "cross_symbol_unexpected_configured_identity_difference:" + field);
        if (sourceSymbol == target)
            throw std::invalid_argument("cross_symbol_target_matches_source");

        const EA::Scheduler::PersistedWorkerSemanticIdentity persisted{
            static_cast<std::size_t>(103), 9, false};
        const auto train = registry.selectTrainingReferenceWorker(
            persisted, EA::Scheduler::RequiredTrainingWorkerCapabilities(
                           Required(proposedIdentity, "feature_ablation_mask")));
        const auto infer = registry.selectInferenceWorker(persisted);
        const auto equivalent = source.FindEquivalent(proposed);

        output << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION"
               << ",mode=" << (apply ? "materialize" : "preview")
               << ",source_experiment_id=" << command.sourceExperimentId
               << ",source_symbol=" << sourceSymbol
               << ",target_symbol=" << target
               << ",intentional_difference=symbol"
               << ",created_status=paused,created_phase=train"
               << ",worker_process_action=none,semantic_worker_publication=none"
               << ",registry_path=" << registry.canonicalRegistryPath()
               << ",registry_schema_version=" << registry.schemaVersion() << '\n';
        RenderIdentity(output, proposedIdentity);
        RenderWorker(output, "train", train, registry);
        RenderWorker(output, "infer", infer, registry);
        output << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION_DUPLICATE"
               << ",state=" << Planning::EquivalentExperimentStateText(equivalent.state)
               << ",ids=";
        for (std::size_t i = 0; i < equivalent.experimentIds.size(); ++i)
            output << (i ? "|" : "") << equivalent.experimentIds[i];
        output << ",reason=" << (equivalent.reason.empty() ? "none" : equivalent.reason)
               << '\n';

        if (!train.selected || !infer.selected)
            throw std::invalid_argument("cross_symbol_historical_worker_contract_unresolved");
        if (equivalent.state != Planning::EquivalentExperimentState::NoEquivalentExperimentFound)
            throw std::invalid_argument("cross_symbol_equivalent_target_exists_or_ambiguous");
        if (!apply)
        {
            output << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION_RESULT"
                   << ",state=previewed,mutations=0,queued=false,started=false\n";
            return 0;
        }
        const long long created = InsertFreshPausedCrossSymbolExperiment(
            transaction, command.sourceExperimentId, target);
        if (created <= 0) throw std::runtime_error("cross_symbol_created_id_invalid");
        output << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION_RESULT"
               << ",state=materialized,experiment_id=" << created
               << ",transaction=committed,queued=false,started=false\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION_RESULT"
               << ",state=" << (apply ? "not_materialized" : "not_previewed")
               << ",reason=" << error.what()
               << ",mutations=0,queued=false,started=false\n";
        return 3;
    }
}

} // namespace

int RunMaterializationCommand(const std::string& connectionString,
                              const MaterializationCommand& command,
                              std::ostream& output,
                              std::ostream& errors)
{
    std::ostringstream stagedOutput;
    std::ostringstream stagedErrors;
    bool transactionStarted = false;
    bool commitAttempted = false;
    bool commitSucceeded = false;
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transactionStarted = true;
        transaction.exec("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;");
        // SHARE ROW EXCLUSIVE conflicts with every INSERT/UPDATE/DELETE table
        // lock. Thus existing and future writers cannot bypass this boundary,
        // even when they do not participate in an advisory-lock convention.
        transaction.exec(
            "LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE;");

        EA::Scheduler::SemanticWorkerRegistryLoadRequest registryRequest;
        registryRequest.registryPath = command.semanticWorkerRegistryPath;
        const auto registry = EA::Scheduler::SemanticWorkerRegistry::Load(
            registryRequest);

        const Planning::PostgresPlanningSource source{transaction};
        PostgresFreshExperimentInserter inserter{transaction};
        const int result = RunMaterializationInTransaction(
            command, source, source, inserter, stagedOutput, stagedErrors,
            &registry);
        if (result == 0)
        {
            commitAttempted = true;
            transaction.commit();
            commitSucceeded = true;
        }
        else
        {
            // Publish deterministic validation/equivalence failure output only
            // after the write transaction has actually been aborted.
            transaction.abort();
        }
        output << stagedOutput.str();
        errors << stagedErrors.str();
        return result;
    }
    catch (const pqxx::in_doubt_error& error)
    {
        errors << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
               << ",state=materialization_outcome_unknown"
               << ",reason=database_commit_outcome_unknown"
               << ",detail=";
        std::string detail = error.what();
        for (char& character : detail)
            if (!((character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') ||
                  character == '_' || character == '-' || character == '.'))
                character = '_';
        errors << (detail.empty() ? "EMPTY" : detail)
               << ",pair_count=0,experiment_count=0"
               << ",transaction=commit_outcome_unknown"
               << ",queued=false,started=false,exit_code=2\n";
        return 2;
    }
    catch (const std::exception& error)
    {
        errors << "CONTROLLED_REPLICATION_WAVE_MATERIALIZATION_RESULT"
               << ",state="
               << (commitSucceeded ? "materialized_output_failure" :
                   "not_materialized")
               << ",reason=database_insert_commit_or_output_failure"
               << ",detail=";
        std::string detail = error.what();
        for (char& character : detail)
            if (!((character >= 'a' && character <= 'z') ||
                  (character >= 'A' && character <= 'Z') ||
                  (character >= '0' && character <= '9') ||
                  character == '_' || character == '-' || character == '.'))
                character = '_';
        errors << (detail.empty() ? "EMPTY" : detail)
               << ",pair_count=0,experiment_count=0"
               << ",transaction="
               << (!transactionStarted ? "not_started" :
                   (commitSucceeded ? "committed" :
                    (commitAttempted ? "commit_failed" : "rolled_back")))
               << ",queued=false,started=false"
               << ",exit_code=2\n";
        return 2;
    }
}

int RunCrossSymbolPreviewCommand(const std::string& connectionString,
                                 const CrossSymbolCommand& command,
                                 std::ostream& output,
                                 std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
        const int result = RunCrossSymbol(transaction, command, false, output, errors);
        transaction.commit();
        return result;
    }
    catch (const std::exception& error)
    {
        errors << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION_RESULT"
               << ",state=not_previewed,reason=" << error.what()
               << ",mutations=0,queued=false,started=false\n";
        return 2;
    }
}

int RunLayout11ConfluencePlanCommand(const std::string& connectionString,
                                     const Layout11ConfluenceCommand& command,
                                     std::ostream& output,
                                     std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
        EA::Scheduler::SemanticWorkerRegistryLoadRequest registryRequest;
        registryRequest.registryPath = command.semanticWorkerRegistryPath;
        const auto registry = EA::Scheduler::SemanticWorkerRegistry::Load(registryRequest);
        const Planning::PostgresPlanningSource source{transaction};
        NoopLayout11ConfluenceInserter inserter;
        const int result = RunLayout11ConfluenceMaterializationInTransaction(
            command, source, source, inserter, output, errors, &registry, false);
        transaction.commit();
        return result;
    }
    catch (const std::exception& error)
    {
        errors << "LAYOUT11_CONFLUENCE_REPLICATION_RESULT,state=not_planned,reason="
               << error.what() << ",mutations=0,queued=false,started=false\n";
        return 2;
    }
}

int RunLayout11ConfluenceMaterializationCommand(
    const std::string& connectionString, const Layout11ConfluenceCommand& command,
    std::ostream& output, std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;");
        transaction.exec("LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE;");
        EA::Scheduler::SemanticWorkerRegistryLoadRequest registryRequest;
        registryRequest.registryPath = command.semanticWorkerRegistryPath;
        const auto registry = EA::Scheduler::SemanticWorkerRegistry::Load(registryRequest);
        const Planning::PostgresPlanningSource source{transaction};
        PostgresLayout11ConfluenceInserter inserter{transaction};
        const int result = RunLayout11ConfluenceMaterializationInTransaction(
            command, source, source, inserter, output, errors, &registry, true);
        if (result == 0) transaction.commit();
        else transaction.abort();
        return result;
    }
    catch (const pqxx::in_doubt_error& error)
    {
        errors << "LAYOUT11_CONFLUENCE_REPLICATION_RESULT,state=materialization_outcome_unknown,reason="
               << error.what() << ",queued=false,started=false\n";
        return 2;
    }
    catch (const std::exception& error)
    {
        errors << "LAYOUT11_CONFLUENCE_REPLICATION_RESULT,state=not_materialized,reason="
               << error.what() << ",queued=false,started=false\n";
        return 2;
    }
}

int RunCrossSymbolMaterializationCommand(const std::string& connectionString,
                                         const CrossSymbolCommand& command,
                                         std::ostream& output,
                                         std::ostream& errors)
{
    try
    {
        pqxx::connection connection{connectionString};
        pqxx::work transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;");
        transaction.exec("LOCK TABLE experiment IN SHARE ROW EXCLUSIVE MODE;");
        const int result = RunCrossSymbol(transaction, command, true, output, errors);
        if (result == 0) transaction.commit();
        else transaction.abort();
        return result;
    }
    catch (const pqxx::in_doubt_error& error)
    {
        errors << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION_RESULT"
               << ",state=materialization_outcome_unknown,reason=" << error.what()
               << ",queued=false,started=false\n";
        return 2;
    }
    catch (const std::exception& error)
    {
        errors << "CROSS_SYMBOL_HISTORICAL_MATERIALIZATION_RESULT"
               << ",state=not_materialized,reason=" << error.what()
               << ",queued=false,started=false\n";
        return 2;
    }
}

} // namespace EA::ExperimentReplicationMaterialization
