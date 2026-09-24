#include "ExperimentReplicationMaterialization.hpp"
#include "ExperimentReplicationMaterializationRepository.hpp"

#include "ExperimentReplicationPlanningPostgres.hpp"

#include <pqxx/pqxx>
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

        const Planning::PostgresPlanningSource source{transaction};
        PostgresFreshExperimentInserter inserter{transaction};
        const int result = RunMaterializationInTransaction(
            command, source, source, inserter, stagedOutput, stagedErrors);
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

} // namespace EA::ExperimentReplicationMaterialization
