#include <cassert>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "EconomicEventFeatures.hpp"
#include "EconomicEventRepository.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

PriceTP At(std::int64_t seconds)
{
    return PriceTP{std::chrono::seconds{seconds}};
}

bool Near(float actual, double expected)
{
    return std::abs(static_cast<double>(actual) - expected) <= 1.0e-6;
}

void AssertNoAuthoritativeSurprise(const EconomicEventFeatureValues& values)
{
    assert(values.authoritativeInitialHasSurprise == 0.0F);
    assert(values.authoritativeInitialSurprise == 0.0F);
    assert(values.authoritativeInitialSurpriseAbs == 0.0F);
    assert(values.authoritativeInitialSurpriseDirection == 0.0F);
}

} // namespace

int main()
{
    pqxx::connection connection{
        "host=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=" + EnvironmentOr("LSTM_DB_USER", "pqxx") +
        " dbname=" + EnvironmentOr("LSTM_DB_NAME", "invalid")};

    {
        pqxx::work write{connection};
        write.exec(R"SQL(
            INSERT INTO economic_event (
                currency,event_family,event_timestamp_utc,source_agency,
                source_event_id,source_url,reference_period,event_importance,
                historical_time_confidence,source_release_date,
                source_release_time,source_timezone
            ) VALUES
                ('USD','PCE','2024-01-01 00:00:00+00','BEA',
                 'bea:pce:ready','https://www.bea.gov/news/ready',
                 'December 2023',3,'exact','2024-01-01','00:00:00','UTC'),
                ('USD','PCE','2024-01-02 00:00:00+00','BEA',
                 'bea:pce:missing-consensus',
                 'https://www.bea.gov/news/missing-consensus',
                 'January 2024',3,'exact','2024-01-02','00:00:00','UTC'),
                ('USD','PCE','2024-01-03 00:00:00+00','BEA',
                 'bea:pce:incompatible','https://www.bea.gov/news/incompatible',
                 'February 2024',3,'exact','2024-01-03','00:00:00','UTC'),
                ('USD','CPI','2024-01-04 00:00:00+00','BLS',
                 'bls:cpi:compatibility','https://www.bls.gov/cpi',
                 'December 2023',3,'exact','2024-01-04','00:00:00','UTC');

            INSERT INTO economic_event_consensus (
                economic_event_id,consensus_source,source_report_id,
                source_event_id,source_observation_id,source_event_name,
                source_period,source_priority,source_timestamp_epoch,
                source_date,source_release_date,source_artifact_path,
                candidate_classification,match_rule,semantic_contract,
                provider_provenance,forecast_raw,forecast_parse_status,
                forecast_value_kind,forecast_value_low,
                forecast_canonical_value_low,forecast_unit,forecast_scale,
                forecast_qualifier,previous_parse_status,actual_parse_status
            )
            SELECT economic_event_id,'OANDA',694,
                   CASE source_event_id
                       WHEN 'bea:pce:ready' THEN 1001
                       WHEN 'bea:pce:incompatible' THEN 1002
                       ELSE 1003
                   END,
                   'oanda:' || source_event_id,event_family,
                   reference_period,3,
                   EXTRACT(EPOCH FROM event_timestamp_utc)::bigint,
                   event_timestamp_utc AT TIME ZONE 'UTC',
                   source_release_date,'fixture/oanda.csv',
                   'oanda_populated_initial','fixture_match',
                   'oanda_economic_consensus_candidate_v1',
                   '{"provider":"OANDA"}',
                   CASE source_event_id
                       WHEN 'bea:pce:ready' THEN '0.3% m/m'
                       WHEN 'bea:pce:incompatible' THEN '0.4% m/m'
                       ELSE '0.2% m/m'
                   END,
                   'parsed','scalar',
                   CASE source_event_id
                       WHEN 'bea:pce:ready' THEN 0.3
                       WHEN 'bea:pce:incompatible' THEN 0.4
                       ELSE 0.2
                   END,
                   CASE source_event_id
                       WHEN 'bea:pce:ready' THEN 0.3
                       WHEN 'bea:pce:incompatible' THEN 0.4
                       ELSE 0.2
                   END,
                   'percent',1,'m/m','missing','missing'
            FROM economic_event
            WHERE source_event_id IN (
                'bea:pce:ready','bea:pce:incompatible',
                'bls:cpi:compatibility');

            INSERT INTO economic_event_release_actual (
                economic_event_id,source_agency,source_observation_id,
                publication_state,revision_sequence,available_at,retrieved_at,
                source_url,source_artifact_path,source_artifact_sha256,
                semantic_contract,source_provenance,actual_raw,
                actual_value_kind,actual_value_low,
                actual_canonical_value_low,actual_unit,actual_scale,
                actual_qualifier
            )
            SELECT economic_event_id,'BEA','bea:pce:ready:initial',
                   'initial',0,'2024-01-01 00:05:00+00',
                   '2026-08-30 00:00:00+00',source_url,
                   'fixture/bea-pce-ready.html',repeat('a',64),
                   'bea_current_dollar_pce_mom_percent_v1',
                   '{"provider":"BEA"}','increased 0.5 percent',
                   'scalar',0.5,0.5,'percent',1,'m/m'
            FROM economic_event WHERE source_event_id='bea:pce:ready';

            INSERT INTO economic_event_release_actual (
                economic_event_id,source_agency,source_observation_id,
                publication_state,revision_sequence,available_at,retrieved_at,
                source_url,source_artifact_path,source_artifact_sha256,
                semantic_contract,source_provenance,actual_raw,
                actual_value_kind,actual_value_low,
                actual_canonical_value_low,actual_unit,actual_scale,
                actual_qualifier
            )
            SELECT economic_event_id,'BEA','bea:pce:ready:revision:1',
                   'revision',1,'2024-01-01 01:00:00+00',
                   '2026-08-30 00:00:00+00',source_url,
                   'fixture/bea-pce-revision.html',repeat('b',64),
                   'bea_current_dollar_pce_revision_v1',
                   '{"provider":"BEA"}','revised to 9.9 percent',
                   'scalar',9.9,9.9,'percent',1,'m/m'
            FROM economic_event WHERE source_event_id='bea:pce:ready';

            INSERT INTO economic_event_release_actual (
                economic_event_id,source_agency,source_observation_id,
                publication_state,revision_sequence,available_at,retrieved_at,
                source_url,source_artifact_path,source_artifact_sha256,
                semantic_contract,source_provenance,actual_raw,
                actual_value_kind,actual_value_low,
                actual_canonical_value_low,actual_unit,actual_scale,
                actual_qualifier
            )
            SELECT economic_event_id,'BEA',
                   'bea:pce:missing-consensus:initial','initial',0,
                   event_timestamp_utc,'2026-08-30 00:00:00+00',source_url,
                   'fixture/bea-pce-missing.html',repeat('c',64),
                   'bea_current_dollar_pce_mom_percent_v1',
                   '{"provider":"BEA"}','increased 0.7 percent',
                   'scalar',0.7,0.7,'percent',1,'m/m'
            FROM economic_event
            WHERE source_event_id='bea:pce:missing-consensus';

            INSERT INTO economic_event_release_actual (
                economic_event_id,source_agency,source_observation_id,
                publication_state,revision_sequence,available_at,retrieved_at,
                source_url,source_artifact_path,source_artifact_sha256,
                semantic_contract,source_provenance,actual_raw,
                actual_value_kind,actual_value_low,
                actual_canonical_value_low,actual_unit,actual_scale,
                actual_qualifier
            )
            SELECT economic_event_id,'BEA','bea:pce:incompatible:initial',
                   'initial',0,event_timestamp_utc,
                   '2026-08-30 00:00:00+00',source_url,
                   'fixture/bea-pce-incompatible.html',repeat('d',64),
                   'fixture_incompatible_count_v1','{"provider":"BEA"}',
                   '5','scalar',5,5,'count',1,'m/m'
            FROM economic_event WHERE source_event_id='bea:pce:incompatible';
        )SQL");
        write.commit();
    }

    pqxx::read_transaction read{connection};

    // SQL itself excludes an initial actual at the exact information upper
    // bound. The pre-release feature observation therefore cannot see it.
    const auto before = LoadEconomicEventsForFeatureRange(
        read, "USD", "2023-12-31 23:45:00+00", "2024-01-01 00:05:00+00");
    assert(before.size() == 1);
    assert(before.front().selectedConsensus);
    assert(!before.front().releaseActual);
    EconomicEventFeatureEngine beforeEngine{before};
    const auto beforeValues =
        beforeEngine.AdvanceCompletedBar(At(1'704'066'300));
    assert(beforeValues.relevantEventHasConsensus == 1.0F);
    AssertNoAuthoritativeSurprise(beforeValues);

    // Once the upper bound is after available_at, the certified initial is
    // loaded and becomes the only actual used by the feature.
    const auto after = LoadEconomicEventsForFeatureRange(
        read, "USD", "2023-12-31 23:45:00+00", "2024-01-01 00:15:00+00");
    assert(after.size() == 1);
    assert(after.front().releaseActual);
    assert(after.front().releaseActual->actual.canonicalValueLow == 0.5);
    EconomicEventFeatureEngine afterEngine{after};
    const auto afterValues =
        afterEngine.AdvanceCompletedBar(At(1'704'067'200));
    assert(afterValues.authoritativeInitialHasSurprise == 1.0F);
    assert(Near(afterValues.authoritativeInitialSurprise, 0.02));
    assert(Near(afterValues.authoritativeInitialSurpriseAbs, 0.02));
    assert(afterValues.authoritativeInitialSurpriseDirection == 1.0F);

    // A query made after revision availability still returns revision zero.
    // The later 9.9 revision cannot rewrite the historical 0.5 feature value.
    const auto postRevision = LoadEconomicEventsForFeatureRange(
        read, "USD", "2023-12-31 23:45:00+00", "2024-01-01 02:00:00+00");
    assert(postRevision.size() == 1);
    assert(postRevision.front().releaseActual);
    assert(postRevision.front().releaseActual->sourceObservationId ==
           "bea:pce:ready:initial");
    assert(postRevision.front().releaseActual->actual.canonicalValueLow == 0.5);

    // Repeated repository lookup and independent feature evaluation are byte-
    // deterministic for the same persisted snapshot and observation cutoff.
    const auto repeated = LoadEconomicEventsForFeatureRange(
        read, "USD", "2023-12-31 23:45:00+00", "2024-01-01 00:15:00+00");
    EconomicEventFeatureEngine firstDeterministic{after};
    EconomicEventFeatureEngine secondDeterministic{repeated};
    assert(firstDeterministic.AdvanceCompletedBar(At(1'704'067'200)).Ordered() ==
           secondDeterministic.AdvanceCompletedBar(At(1'704'067'200)).Ordered());

    // Missing consensus and incompatible canonical units both fail closed.
    const auto missing = LoadEconomicEventsForFeatureRange(
        read, "USD", "2024-01-01 23:45:00+00", "2024-01-02 00:15:00+00");
    EconomicEventFeatureEngine missingEngine{missing};
    const auto missingValues =
        missingEngine.AdvanceCompletedBar(At(1'704'153'600));
    assert(missingValues.relevantEventHasConsensus == 0.0F);
    AssertNoAuthoritativeSurprise(missingValues);

    const auto incompatible = LoadEconomicEventsForFeatureRange(
        read, "USD", "2024-01-02 23:45:00+00", "2024-01-03 00:15:00+00");
    EconomicEventFeatureEngine incompatibleEngine{incompatible};
    const auto incompatibleValues =
        incompatibleEngine.AdvanceCompletedBar(At(1'704'240'000));
    assert(incompatibleValues.relevantEventHasConsensus == 1.0F);
    AssertNoAuthoritativeSurprise(incompatibleValues);
    assert(incompatibleEngine.Diagnostics().incompatibleInitialActualRowCount == 1);

    // Existing occurrence, recency, and consensus behavior is unchanged for an
    // event with no authoritative actual.
    const auto compatibility = LoadEconomicEventsForFeatureRange(
        read, "USD", "2024-01-03 23:45:00+00", "2024-01-04 00:15:00+00");
    EconomicEventFeatureEngine compatibilityEngine{compatibility};
    const auto compatibilityValues =
        compatibilityEngine.AdvanceCompletedBar(At(1'704'326'400));
    assert(compatibilityValues.inflationEvent == 1.0F);
    assert(compatibilityValues.inflationRecencyDecay > 0.0F);
    assert(compatibilityValues.relevantEventHasConsensus == 1.0F);
    assert(Near(compatibilityValues.relevantEventConsensusLow, 0.02));
    AssertNoAuthoritativeSurprise(compatibilityValues);

    return 0;
}
