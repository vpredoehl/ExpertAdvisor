#include <algorithm>
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

} // namespace

int main()
{
    pqxx::connection connection{
        "host=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=" + EnvironmentOr("LSTM_DB_USER", "pqxx") +
        " dbname=" + EnvironmentOr("LSTM_DB_NAME", "invalid")};

    // A completely absent feature corpus is a source/history failure, not a
    // legitimate no-event interval whose semantic value happens to be zero.
    {
        pqxx::read_transaction emptyRead{connection};
        bool unavailableRejected = false;
        try
        {
            (void)LoadEconomicEventsForFeatureRange(
                emptyRead, "USD", "2023-11-14 00:00:00+00",
                "2023-11-15 00:00:00+00");
        }
        catch (const std::runtime_error& error)
        {
            unavailableRejected =
                std::string{error.what()} ==
                "economic_event_feature_source_history_unavailable:USD";
        }
        assert(unavailableRejected);
    }

    {
        pqxx::work write{connection};
        write.exec(
            "INSERT INTO economic_event "
            "(currency,event_family,event_timestamp_utc,source_agency,"
            "source_event_id,source_url,event_importance,"
            "historical_time_confidence) VALUES "
            "('USD','CPI','2023-11-14 19:00:00+00','BLS','old-cpi',"
            "'https://example.test/old-cpi',3,'exact'),"
            "('USD','CPI','2023-11-14 21:13:20+00','BLS','seed-cpi',"
            "'https://example.test/seed-cpi',3,'exact'),"
            "('USD','EMPLOYMENT','2023-11-14 20:43:20+00','BLS',"
            "'seed-employment','https://example.test/seed-employment',3,'exact'),"
            "('USD','WEEKLY_CLAIMS','2023-11-14 21:43:20+00','DOL_ETA',"
            "'seed-claims','https://example.test/seed-claims',3,'reconstructed'),"
            "('USD','GDP','2023-11-14 22:18:20+00','BEA','range-gdp',"
            "'https://example.test/range-gdp',3,'exact'),"
            "('USD','JOLTS','2023-11-14 22:28:20+00','BLS','range-jolts',"
            "'https://example.test/range-jolts',3,'exact'),"
            "('USD','FOMC','2023-11-14 23:30:00+00','FEDERAL_RESERVE',"
            "'after-range','https://example.test/after-range',3,'exact');");
        write.exec(
            "INSERT INTO economic_event_consensus ("
            "economic_event_id,consensus_source,source_report_id,source_event_id,"
            "source_observation_id,source_event_name,source_period,source_priority,"
            "source_timestamp_epoch,source_date,source_release_date,"
            "source_artifact_path,candidate_classification,match_rule,"
            "semantic_contract,provider_provenance,forecast_raw,"
            "forecast_parse_status,forecast_value_kind,forecast_value_low,"
            "forecast_canonical_value_low,forecast_unit,forecast_scale,"
            "previous_parse_status,actual_raw,actual_parse_status,"
            "actual_value_kind,actual_value_low,actual_canonical_value_low,"
            "actual_unit,actual_scale) SELECT economic_event_id,'OANDA',690,"
            "9001,'oanda:event:9001','GDP (Annualized) - pre..','Q3',3,"
            "1700000300,'2023-11-14 22:18','2023-11-14',"
            "'fixture/oanda.csv','oanda_populated_initial','fixture_match',"
            "'oanda_economic_consensus_candidate_v1','{\"provider\":\"OANDA\"}',"
            "'2.0%','parsed','scalar',2.0,2.0,'percent',1,'missing',"
            "'2.5%','parsed','scalar',2.5,2.5,'percent',1 "
            "FROM economic_event WHERE source_event_id='range-gdp';");
        write.exec(
            "INSERT INTO economic_event_consensus ("
            "economic_event_id,consensus_source,source_event_id,"
            "source_observation_id,source_event_name,source_release_date,"
            "source_artifact_path,source_artifact_sha256,"
            "candidate_classification,match_rule,semantic_contract,"
            "provider_provenance,forecast_raw,forecast_parse_status,"
            "forecast_value_kind,forecast_value_low,"
            "forecast_canonical_value_low,forecast_unit,forecast_scale,"
            "previous_parse_status,actual_parse_status) SELECT economic_event_id,"
            "'MYFXBOOK',-9002,'myfxbook:event:9002:date:2023-11-14:series:1:observation:1',"
            "'JOLTS Job Openings','2023-11-14','fixture/myfxbook.json',"
            "repeat('a',64),'myfxbook_jolts_gap_fill','fixture_match',"
            "'myfxbook_consensus_observation_v1',"
            "'{\"provider\":\"MYFXBOOK\"}','4530','parsed','scalar',"
            "4530,4530000,'count',1000,'missing','missing' "
            "FROM economic_event WHERE source_event_id='range-jolts';");
        write.exec(
            "INSERT INTO economic_event_release_actual ("
            "economic_event_id,source_agency,source_observation_id,"
            "publication_state,revision_sequence,available_at,retrieved_at,"
            "source_url,source_artifact_path,source_artifact_sha256,"
            "semantic_contract,source_provenance,actual_raw,actual_value_kind,"
            "actual_value_low,actual_canonical_value_low,actual_unit,"
            "actual_scale) SELECT economic_event_id,'BLS','bls:jolts:initial',"
            "'initial',0,event_timestamp_utc + interval '1 microsecond',"
            "'2026-08-29 00:00:00+00',"
            "'https://www.bls.gov/news.release/jolts.nr0.htm',"
            "'fixture/bls-jolts-initial.html',repeat('b',64),"
            "'bls_jolts_initial_actual_v1','{\"provider\":\"BLS\"}',"
            "'5000','scalar',5000,5000000,'count',1000 "
            "FROM economic_event WHERE source_event_id='range-jolts';");
        write.exec(
            "INSERT INTO economic_event_release_actual ("
            "economic_event_id,source_agency,source_observation_id,"
            "publication_state,revision_sequence,available_at,retrieved_at,"
            "source_url,source_artifact_path,source_artifact_sha256,"
            "semantic_contract,source_provenance,actual_raw,actual_value_kind,"
            "actual_value_low,actual_canonical_value_low,actual_unit,"
            "actual_scale) SELECT economic_event_id,'BLS','bls:jolts:revision:1',"
            "'revision',1,event_timestamp_utc + interval '1 day',"
            "'2026-08-29 00:00:00+00',"
            "'https://www.bls.gov/news.release/jolts.nr0.htm',"
            "'fixture/bls-jolts-revision.html',repeat('c',64),"
            "'bls_jolts_revision_actual_v1','{\"provider\":\"BLS\"}',"
            "'5100','scalar',5100,5100000,'count',1000 "
            "FROM economic_event WHERE source_event_id='range-jolts';");
        // Production ingestion writes the legacy migration-088 row and the
        // provider-neutral migration-090 observation in one transaction.
        write.exec(
            "INSERT INTO economic_event_actual_observation ("
            "economic_event_id,source_name,source_role,source_native_event_id,"
            "source_observation_id,evidence_key,observation_kind,"
            "revision_sequence,source_publication_at,"
            "source_publication_time_status,observed_at,ingested_at,"
            "availability_proof,source_url,source_artifact_path,"
            "source_artifact_sha256,semantic_contract,source_provenance,"
            "actual_raw,actual_value_kind,actual_value_low,actual_value_high,"
            "actual_canonical_value_low,actual_canonical_value_high,"
            "actual_unit,actual_scale,actual_qualifier,legacy_release_actual_id"
            ") SELECT a.economic_event_id,a.source_agency,'authoritative',"
            "e.source_event_id,a.source_observation_id,"
            "'range-test:' || a.economic_event_release_actual_id::text,"
            "a.publication_state,a.revision_sequence,a.available_at,'exact',"
            "a.retrieved_at,a.imported_at,'source_publication',a.source_url,"
            "a.source_artifact_path,a.source_artifact_sha256,"
            "a.semantic_contract,a.source_provenance,a.actual_raw,"
            "a.actual_value_kind,a.actual_value_low,a.actual_value_high,"
            "a.actual_canonical_value_low,a.actual_canonical_value_high,"
            "a.actual_unit,a.actual_scale,a.actual_qualifier,"
            "a.economic_event_release_actual_id "
            "FROM economic_event_release_actual a "
            "JOIN economic_event e USING (economic_event_id) "
            "ORDER BY a.economic_event_release_actual_id;");
        write.commit();
    }

    // Corpus availability is independent of interval occupancy. Once USD
    // history exists, a range before the first event remains a legitimate
    // event-free interval.
    {
        pqxx::read_transaction eventFreeRead{connection};
        const auto eventFree = LoadEconomicEventsForFeatureRange(
            eventFreeRead, "USD", "2020-01-01 00:00:00+00",
            "2020-01-02 00:00:00+00");
        assert(eventFree.empty());
    }

    // The actual store is immutable, rejects non-authoritative source agency,
    // and keeps revision sequence causally ordered.
    for (const std::string& statement : {
        "UPDATE economic_event_release_actual SET actual_raw='changed' "
        "WHERE source_observation_id='bls:jolts:initial'",
        "INSERT INTO economic_event_release_actual (economic_event_id,"
        "source_agency,source_observation_id,publication_state,"
        "revision_sequence,available_at,retrieved_at,source_url,"
        "source_artifact_path,source_artifact_sha256,semantic_contract,"
        "source_provenance,actual_raw,actual_value_kind,actual_value_low,"
        "actual_canonical_value_low,actual_unit,actual_scale) SELECT "
        "economic_event_id,'BEA','wrong-agency','revision',2,"
        "event_timestamp_utc + interval '2 days','2026-08-29 00:00:00+00',"
        "'https://www.bls.gov/example','fixture/wrong',repeat('d',64),"
        "'fixture_v1','{}','1','scalar',1,1,'count',1 FROM economic_event "
        "WHERE source_event_id='range-jolts'",
        "INSERT INTO economic_event_release_actual (economic_event_id,"
        "source_agency,source_observation_id,publication_state,"
        "revision_sequence,available_at,retrieved_at,source_url,"
        "source_artifact_path,source_artifact_sha256,semantic_contract,"
        "source_provenance,actual_raw,actual_value_kind,actual_value_low,"
        "actual_canonical_value_low,actual_unit,actual_scale) SELECT "
        "economic_event_id,'BLS','out-of-order','revision',2,"
        "event_timestamp_utc + interval '12 hours','2026-08-29 00:00:00+00',"
        "'https://www.bls.gov/example','fixture/order',repeat('e',64),"
        "'fixture_v1','{}','1','scalar',1,1,'count',1 FROM economic_event "
        "WHERE source_event_id='range-jolts'"})
    {
        bool rejected = false;
        try
        {
            pqxx::work invalid{connection};
            invalid.exec(statement);
            invalid.commit();
        }
        catch (const pqxx::sql_error&)
        {
            rejected = true;
        }
        assert(rejected);
    }

    pqxx::read_transaction read{connection};
    const auto halfOpenEvents = LoadEconomicEvents(
        read, "USD", "2023-11-14 22:18:20+00",
        "2023-11-14 22:28:20+00");
    assert(halfOpenEvents.size() == 1);
    assert(halfOpenEvents.front().sourceEventId ==
           std::optional<std::string>{"range-gdp"});
    assert(halfOpenEvents.front().selectedConsensus);
    assert(halfOpenEvents.front().selectedConsensus->provider == "OANDA");
    assert(!halfOpenEvents.front().selectedConsensus
                ->unprovenProviderActual);
    assert(!halfOpenEvents.front().releaseActual);

    // The base observation really does contain a populated historical OANDA
    // actual. Schema 082 deliberately does not expose it through the selected
    // view, so presence alone cannot activate surprise in runtime features.
    const pqxx::row persistedOanda = read.exec(
        "SELECT actual_parse_status, actual_canonical_value_low "
        "FROM economic_event_consensus "
        "WHERE source_event_id = 9001;").one_row();
    assert(persistedOanda["actual_parse_status"].as<std::string>() ==
           "parsed");
    assert(persistedOanda["actual_canonical_value_low"].as<double>() ==
           2.5);

    const auto events = LoadEconomicEventsForFeatureRange(
        read, "USD", "2023-11-14 22:13:20+00",
        "2023-11-14 22:43:20+00");

    // The old CPI row is superseded within its canonical stream. The loader
    // returns exactly the three prior stream seeds and two in-range rows.
    assert(events.size() == 5);
    assert(events[0].sourceEventId == std::optional<std::string>{"seed-employment"});
    assert(events[1].sourceEventId == std::optional<std::string>{"seed-cpi"});
    assert(events[2].sourceEventId == std::optional<std::string>{"seed-claims"});
    assert(events[2].historicalTimeConfidence == "reconstructed");
    assert(!events[2].selectedConsensus);
    assert(events[3].sourceEventId == std::optional<std::string>{"range-gdp"});
    assert(events[4].sourceEventId == std::optional<std::string>{"range-jolts"});
    assert(events[3].selectedConsensus);
    assert(events[3].selectedConsensus->provider == "OANDA");
    assert(!events[3].selectedConsensus->unprovenProviderActual);
    assert(events[4].selectedConsensus);
    assert(events[4].selectedConsensus->provider == "MYFXBOOK");
    assert(!events[4].selectedConsensus->unprovenProviderActual);
    assert(events[4].selectedConsensus->forecast.canonicalValueLow == 4530000.0);
    assert(events[4].releaseActual);
    assert(events[4].releaseActual->actual.canonicalValueLow == 5000000.0);
    assert(events[4].releaseActual->availableAtUnixMicros % 1000000LL == 1);
    assert(events[4].releaseActual->sourceAgency == "BLS");
    assert(events[4].releaseActual->sourceObservationId ==
           "bls:jolts:initial");
    assert(events[4].releaseActual->sourceArtifactSha256 ==
           std::string(64, 'b'));
    assert(read.query_value<long long>(
        "SELECT count(*) FROM economic_event_release_actual") == 2);
    assert(read.query_value<double>(
        "SELECT actual_canonical_value_low "
        "FROM economic_event_feature_release_actual") == 5000000.0);
    assert(read.query_value<std::string>(
        "SELECT source_observation_id "
        "FROM economic_event_feature_release_actual") ==
           "bls:jolts:initial");

    // The range loader calls the PIT function once at its maximum cutoff and
    // carries the proved timestamp for per-bar inclusive gating. Immediately
    // before publication it matches an empty direct PIT result; at the exact
    // microsecond it matches the same deterministic first-release row.
    const auto immediatelyBeforePit = LoadEconomicEventsForFeatureRange(
        read, "USD", "2023-11-14 22:13:20+00",
        "2023-11-14 22:28:20+00");
    assert(immediatelyBeforePit.back().sourceEventId ==
           std::optional<std::string>{"range-jolts"});
    assert(immediatelyBeforePit.back().firstReleaseActualState ==
           EconomicEventFirstReleaseActualState::notYetAvailable);
    assert(!immediatelyBeforePit.back().firstReleaseActual);
    assert(read.query_value<long long>(
        "SELECT count(*) FROM economic_event_first_release_actual_at("
        "'2023-11-14 22:28:20+00'::timestamptz) WHERE economic_event_id=("
        "SELECT economic_event_id FROM economic_event WHERE "
        "source_event_id='range-jolts')") == 0);

    const auto exactlyAtPit = LoadEconomicEventsForFeatureRange(
        read, "USD", "2023-11-14 22:13:20+00",
        "2023-11-14 22:28:20.000001+00");
    assert(exactlyAtPit.back().sourceEventId ==
           std::optional<std::string>{"range-jolts"});
    assert(exactlyAtPit.back().firstReleaseActualState ==
           EconomicEventFirstReleaseActualState::provenFirstRelease);
    assert(exactlyAtPit.back().firstReleaseActual);
    assert(exactlyAtPit.back().firstReleaseActual->actual.canonicalValueLow ==
           read.query_value<double>(
               "SELECT first_release_actual_value_low FROM "
               "economic_event_first_release_actual_at("
               "'2023-11-14 22:28:20.000001+00'::timestamptz) WHERE "
               "economic_event_id=(SELECT economic_event_id FROM "
               "economic_event WHERE source_event_id='range-jolts')"));

    // Even after revision publication, bulk retrieval continues to expose the
    // immutable first release, while the audit-only canonical view advances.
    const auto afterRevisionPit = LoadEconomicEventsForFeatureRange(
        read, "USD", "2023-11-14 22:13:20+00",
        "2023-11-16 00:00:00+00");
    const auto joltsAfterRevision = std::find_if(
        afterRevisionPit.begin(), afterRevisionPit.end(),
        [](const EconomicEvent& event)
        {
            return event.sourceEventId ==
                std::optional<std::string>{"range-jolts"};
        });
    assert(joltsAfterRevision != afterRevisionPit.end());
    assert(joltsAfterRevision->firstReleaseActual);
    assert(joltsAfterRevision->firstReleaseActual->actual.canonicalValueLow ==
           5000000.0);
    assert(read.query_value<double>(
        "SELECT canonical_value_low FROM economic_event_first_release_actual "
        "WHERE economic_event_id=(SELECT economic_event_id FROM economic_event "
        "WHERE source_event_id='range-jolts')") == 5100000.0);

    constexpr std::int64_t firstBarStart = 1'700'000'000;
    EconomicEventFeatureEngine engine{events};
    const auto first = engine.AdvanceCompletedBar(At(firstBarStart));
    assert(first.inflationEvent == 0.0F);
    assert(first.employmentEvent == 0.0F);
    assert(first.growthEvent == 1.0F);
    assert(first.relevantEventHasConsensus == 1.0F);
    assert(Near(first.relevantEventConsensusLow, 0.453));
    assert(first.releasedEventHasSurprise == 0.0F);
    assert(first.releasedEventSurprise == 0.0F);
    assert(first.releasedEventSurpriseAbs == 0.0F);
    assert(first.releasedEventSurpriseDirection == 0.0F);
    assert(first.authoritativeInitialHasSurprise == 0.0F);
    assert(first.authoritativeInitialSurprise == 0.0F);
    assert(first.authoritativeInitialSurpriseAbs == 0.0F);
    assert(first.authoritativeInitialSurpriseDirection == 0.0F);
    assert(first.causalFirstReleaseSurpriseAvailable == 0.0F);
    assert(first.causalFirstReleaseSurprise == 0.0F);
    assert(Near(first.inflationRecencyDecay,
                std::exp(-4500.0 / 86400.0)));
    assert(Near(first.employmentRecencyDecay,
                std::exp(-2700.0 / 86400.0)));

    const auto second =
        engine.AdvanceCompletedBar(At(firstBarStart + 900));
    assert(second.employmentEvent == 1.0F);
    assert(second.relevantEventHasConsensus == 1.0F);
    assert(Near(second.relevantEventConsensusLow, 0.453));
    assert(second.releasedEventHasSurprise == 0.0F);
    assert(second.releasedEventSurprise == 0.0F);
    assert(second.releasedEventSurpriseAbs == 0.0F);
    assert(second.releasedEventSurpriseDirection == 0.0F);
    assert(second.authoritativeInitialHasSurprise == 1.0F);
    assert(Near(second.authoritativeInitialSurprise, 0.047));
    assert(Near(second.authoritativeInitialSurpriseAbs, 0.047));
    assert(second.authoritativeInitialSurpriseDirection == 1.0F);
    assert(second.causalFirstReleaseSurpriseAvailable == 1.0F);
    assert(Near(second.causalFirstReleaseSurprise, 0.047));

    return 0;
}
