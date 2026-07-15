#include <barrier>
#include <chrono>
#include <cstdlib>
#include <future>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "../Sources/ExperimentRecommendationRepository.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

std::string ExperimentStateDigest(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return transaction.exec(
        "SELECT count(*)::text || ':' || md5(COALESCE(string_agg("
        "concat_ws('|',experiment_id::text,status,phase,"
        "COALESCE(worker_pid::text,'NULL'),COALESCE(current_operation,'NULL'),"
        "updated_at::text),'#' ORDER BY experiment_id),'')) FROM experiment;")
        .one_row()[0].as<std::string>();
}

void Cleanup(pqxx::connection& connection,
             const std::vector<long long>& scanIds)
{
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    for (const long long scanId : scanIds)
    {
        transaction.exec(
            "DELETE FROM experiment_recommendation "
            "WHERE recommendation_scan_id=$1;",
            pqxx::params{scanId});
        transaction.exec(
            "DELETE FROM experiment_recommendation_scan "
            "WHERE recommendation_scan_id=$1;",
            pqxx::params{scanId});
    }
    transaction.commit();
}

} // namespace

int main()
{
    const std::string connectionString =
        "hostaddr=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=pqxx dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM");
    pqxx::connection connection{connectionString};
    const std::string before = ExperimentStateDigest(connection);

    RecommendationPolicy policy;
    policy.policyVersion = 1000000000 + static_cast<int>(
        std::chrono::steady_clock::now().time_since_epoch().count() %
        100000000);
    policy.maximumPredictedNeutralProportion.reset();
    std::vector<long long> scanIds;

    try
    {
        const auto loaded = LoadRecommendationSources(connection, {});
        std::optional<RecommendationSource> chosenSource;
        std::optional<GeneratedRecommendationCandidate> chosenCandidate;
        for (const auto& item : loaded)
        {
            if (!item.source ||
                !EvaluateRecommendationSource(policy, *item.source).eligible)
                continue;
            const auto generated =
                GenerateRecommendationCandidates(policy, *item.source);
            for (const auto& candidate : generated.candidates)
            {
                pqxx::read_transaction transaction{connection};
                if (FindExperimentDuplicate(
                        transaction, candidate.semanticIdentity,
                        candidate.invocationIdentity, policy).type ==
                    RecommendationDuplicateType::noDuplicate)
                {
                    chosenSource = *item.source;
                    chosenCandidate = candidate;
                    break;
                }
            }
            if (chosenCandidate) break;
        }
        Require(chosenSource.has_value() && chosenCandidate.has_value(),
                "no persistence-test candidate available");

        // Avoid colliding with any pre-existing canonical policy identity.
        // A race after this read is harmless: the test will fail without
        // deleting a row owned by another scan.
        bool unusedPolicyIdentity = false;
        for (int attempt = 0; attempt < 100; ++attempt)
        {
            pqxx::read_transaction transaction{connection};
            if (FindRecommendationDuplicate(
                    transaction, chosenCandidate->semanticIdentity,
                    chosenCandidate->invocationIdentity, policy).kind ==
                PersistedRecommendationMatchKind::none)
            {
                unusedPolicyIdentity = true;
                break;
            }
            ++policy.policyVersion;
        }
        Require(unusedPolicyIdentity,
                "could not allocate unused test policy identity");

        RecommendationScanRequest scanRequest;
        scanRequest.policy = policy;
        const long long firstScan = BeginRecommendationScan(connection, scanRequest);
        scanIds.push_back(firstScan);
        const long long secondScan = BeginRecommendationScan(connection, scanRequest);
        scanIds.push_back(secondScan);

        std::barrier startTogether{3};
        const auto persist = [&](long long scanId) {
            pqxx::connection concurrentConnection{connectionString};
            RecommendationPersistenceRequest request;
            request.recommendationScanId = scanId;
            request.policy = policy;
            request.source = *chosenSource;
            request.candidate = *chosenCandidate;
            request.generationOrdinal = 1;
            request.structuralRank = 1;
            startTogether.arrive_and_wait();
            return PersistRecommendationIdempotently(
                concurrentConnection, request);
        };

        std::future<RecommendationPersistResult> first =
            std::async(std::launch::async, persist, firstScan);
        std::future<RecommendationPersistResult> second =
            std::async(std::launch::async, persist, secondScan);
        startTogether.arrive_and_wait();
        const RecommendationPersistResult firstResult = first.get();
        const RecommendationPersistResult secondResult = second.get();
        const int created =
            (firstResult.outcome == RecommendationPersistOutcome::created) +
            (secondResult.outcome == RecommendationPersistOutcome::created);
        const int existing =
            (firstResult.outcome ==
                RecommendationPersistOutcome::activeRecommendation) +
            (secondResult.outcome ==
                RecommendationPersistOutcome::activeRecommendation);
        Require(created == 1 && existing == 1,
                "concurrent persistence was not idempotent");
        Require(firstResult.recommendationId && secondResult.recommendationId &&
                    firstResult.recommendationId ==
                        secondResult.recommendationId,
                "concurrent persistence returned different rows");

        {
            pqxx::read_transaction transaction{connection};
            const long long activeCount = transaction.exec(
                "SELECT count(*) FROM experiment_recommendation "
                "WHERE semantic_configuration_canonical=$1 AND policy_canonical=$2 "
                "AND status IN ('proposed','approved');",
                pqxx::params{chosenCandidate->semanticIdentity.canonicalText,
                             RecommendationPolicyCanonicalText(policy)})
                .one_row()[0].as<long long>();
            Require(activeCount == 1,
                    "more than one active recommendation persisted");
        }

        RecommendationScanCounters firstCounters;
        firstCounters.recommendationsCreated =
            firstResult.outcome == RecommendationPersistOutcome::created ? 1 : 0;
        firstCounters.recommendationsAlreadyExisting =
            firstResult.outcome ==
                    RecommendationPersistOutcome::activeRecommendation ? 1 : 0;
        CompleteRecommendationScan(connection, firstScan, firstCounters);
        RecommendationScanCounters secondCounters;
        secondCounters.recommendationsCreated =
            secondResult.outcome == RecommendationPersistOutcome::created ? 1 : 0;
        secondCounters.recommendationsAlreadyExisting =
            secondResult.outcome ==
                    RecommendationPersistOutcome::activeRecommendation ? 1 : 0;
        CompleteRecommendationScan(connection, secondScan, secondCounters);

        Require(FindRecommendation(
                    connection, *firstResult.recommendationId).has_value(),
                "persisted recommendation detail missing");
        const auto completedScan = FindRecommendationScan(connection, firstScan);
        Require(completedScan && completedScan->status == "completed",
                "completed scan was not finalized");

        const long long failedScan = BeginRecommendationScan(connection, scanRequest);
        scanIds.push_back(failedScan);
        FailRecommendationScan(
            connection, failedScan, RecommendationScanCounters{}, "");
        const auto failed = FindRecommendationScan(connection, failedScan);
        Require(failed && failed->status == "failed" &&
                    failed->errorMessage ==
                        std::optional<std::string>{
                            "unknown_recommendation_scan_failure"},
                "failed scan did not persist a nonempty diagnostic");
        bool duplicateFinalizationRejected = false;
        try
        {
            CompleteRecommendationScan(
                connection, failedScan, RecommendationScanCounters{});
        }
        catch (const std::runtime_error& error)
        {
            duplicateFinalizationRejected =
                std::string{error.what()} == "recommendation_scan_not_running";
        }
        Require(duplicateFinalizationRejected,
                "duplicate scan finalization was not rejected");
        Require(ExperimentStateDigest(connection) == before,
                "recommendation persistence modified experiment state");
    }
    catch (...)
    {
        Cleanup(connection, scanIds);
        throw;
    }

    Cleanup(connection, scanIds);
    Require(ExperimentStateDigest(connection) == before,
            "test cleanup modified experiment state");
    return 0;
}
