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
            "'{\"provider\":\"MYFXBOOK\"}','4530000','parsed','scalar',"
            "4530000,4530000,'count',1,'missing','missing' "
            "FROM economic_event WHERE source_event_id='range-jolts';");
        write.commit();
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

    return 0;
}
