#include "CausalSurpriseObservabilityRepository.hpp"
#include "CausalSurpriseObservabilityService.hpp"

#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>

#include <pqxx/pqxx>

namespace Observability = EA::CausalSurpriseObservability;

namespace
{

std::string RequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        throw std::runtime_error(std::string{name} + "_required");
    return value;
}

std::string ConnectionString()
{
    return "hostaddr=" + RequiredEnvironment("LSTM_DB_HOST") +
        " user=" + RequiredEnvironment("LSTM_DB_USER") +
        " dbname=" + RequiredEnvironment("LSTM_TEST_DB_NAME");
}

} // namespace

int main()
{
    pqxx::connection connection{ConnectionString()};
    {
        pqxx::work write{connection};
        write.exec(R"SQL(
CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY,
    symbol text NOT NULL,
    prediction_horizon integer NOT NULL,
    train_start timestamptz NOT NULL,
    train_end timestamptz NOT NULL,
    infer_start timestamptz,
    infer_end timestamptz,
    model_input_width integer,
    model_input_semantic_layout_version integer,
    donchian20_mode text NOT NULL,
    feature_warmup_scope text NOT NULL,
    donchian_lookback integer NOT NULL,
    feature_ablation_mask text NOT NULL
);
INSERT INTO experiment VALUES
(619,'EURUSDRMP',4,'2010-01-01','2025-01-01','2025-01-01','2026-01-01',
 77,7,'enabled','full_history_warmup',20,''),
(620,'EURUSDRMP',4,'2010-01-01','2025-01-01','2025-01-01','2026-01-01',
 77,7,'enabled','full_history_warmup',20,
 'causal_first_release_surprise_available,causal_first_release_surprise'),
(500,'eurusdrmp',4,'2010-01-01','2025-01-01',NULL,NULL,
 75,5,'zero_ablation','legacy_cold_boundary',30,''),
(501,'eurusdrmp',4,'2010-01-01','2025-01-01',NULL,NULL,
 NULL,NULL,'enabled','legacy_cold_boundary',20,'');
)SQL");
        write.commit();
    }

    {
        pqxx::read_transaction read{connection};
        const auto current = Observability::LoadExperimentContext(read, 619);
        assert(current.experimentId == 619);
        assert(current.symbol == "eurusdrmp");
        assert(current.predictionHorizon == 4);
        assert(current.modelInputWidth == 77);
        assert(current.modelInputSemanticLayoutVersion == 7);
        assert(current.featureAblationMask.empty());
        assert(current.featureWarmupScope ==
               EA::FeatureWarmupScope::FullHistoryWarmup);

        // Historical width/layout identities remain representable, and no
        // model/checkpoint/final-inference relation exists in this fixture.
        const auto historical =
            Observability::LoadExperimentContext(read, 500);
        assert(historical.modelInputWidth == 75);
        assert(historical.modelInputSemanticLayoutVersion == 5);
        assert(historical.featureWarmupScope ==
               EA::FeatureWarmupScope::LegacyColdBoundary);
        assert(historical.donchianLookback == 30);

        const auto preIdentity =
            Observability::LoadExperimentContext(read, 501);
        assert(!preIdentity.modelInputWidth);
        assert(!preIdentity.modelInputSemanticLayoutVersion);

        bool unknownFailedClosed = false;
        try
        {
            (void)Observability::LoadExperimentContext(read, 999999);
        }
        catch (const std::runtime_error& error)
        {
            unknownFailedClosed = std::string{error.what()}.find(
                "experiment_not_found") != std::string::npos;
        }
        assert(unknownFailedClosed);

        const auto range = Observability::ResolveRanges(
            current, Observability::Scope::train).front();
        const auto result = Observability::Evaluate(
            current, Observability::Scope::train,
            {{range, {}, 0, {}}});
        const std::string rendered = Observability::Render(result);
        assert(rendered.find("model_input_width=77") != std::string::npos);
        assert(rendered.find(
            "model_input_semantic_layout_version=7") !=
            std::string::npos);
        assert(rendered.find("total_feature_rows=0") != std::string::npos);
        assert(rendered.find("read_only=true,software_success=true") !=
               std::string::npos);
        const std::string gaps =
            Observability::RenderGapAttribution(result);
        assert(gaps.find("CAUSAL_SURPRISE_GAP_ATTRIBUTION") !=
               std::string::npos);
        assert(gaps.find("total_feature_rows=0") != std::string::npos);
        assert(gaps.find("terminal_partition_matches_total=true") !=
               std::string::npos);
        assert(gaps.find("read_only=true,software_success=true") !=
               std::string::npos);

        const auto ablation =
            Observability::LoadExperimentContext(read, 620);
        const auto ablationResult = Observability::Evaluate(
            ablation, Observability::Scope::train,
            {{range, {}, 0, {}}});
        assert(result.attributionIdentity ==
               ablationResult.attributionIdentity);
        assert(Observability::CompareUpstream(
            result, ablationResult).coverageMatches);
    }

    // The production command uses this transaction type for all experiment
    // and event inspection. PostgreSQL itself rejects mutation through it.
    {
        pqxx::read_transaction read{connection};
        bool mutationRejected = false;
        try
        {
            read.exec("INSERT INTO experiment SELECT * FROM experiment "
                      "WHERE false;");
        }
        catch (const pqxx::sql_error&)
        {
            mutationRejected = true;
        }
        assert(mutationRejected);
    }

    {
        std::ostringstream output;
        std::ostringstream errors;
        const int status = Observability::RunCommand(
            ConnectionString(), "dbname=unused", 999999,
            Observability::Scope::train, output, errors);
        assert(status == 3);
        assert(output.str().empty());
        assert(errors.str().find("experiment_not_found") !=
               std::string::npos);
        assert(errors.str().find("read_only=true") != std::string::npos);
    }

    {
        std::ostringstream output;
        std::ostringstream errors;
        const int status = Observability::RunGapAttributionCommand(
            ConnectionString(), "dbname=unused", 999999,
            Observability::Scope::train, output, errors);
        assert(status == 3);
        assert(output.str().empty());
        assert(errors.str().find("CAUSAL_SURPRISE_GAP_ATTRIBUTION_FAILED") !=
               std::string::npos);
        assert(errors.str().find("read_only=true") != std::string::npos);
    }

    std::cout << "Causal surprise observability repository tests passed\n";
    return 0;
}
