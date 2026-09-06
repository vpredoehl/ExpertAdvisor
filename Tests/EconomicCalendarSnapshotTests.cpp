#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

#include "EconomicEventRepository.hpp"

using namespace EA::EconomicCalendar;

namespace
{

std::string RequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        throw std::runtime_error(std::string{"missing environment: "} + name);
    return value;
}

const EconomicEvent& RequireSourceEvent(
    const std::vector<EconomicEvent>& events,
    const std::string& sourceEventId)
{
    const auto found = std::find_if(
        events.begin(), events.end(), [&sourceEventId](const EconomicEvent& event)
        {
            return event.sourceEventId == sourceEventId;
        });
    assert(found != events.end());
    return *found;
}

template <typename Operation>
void RequireFailure(Operation operation)
{
    bool failed = false;
    try
    {
        operation();
    }
    catch (const std::exception&)
    {
        failed = true;
    }
    assert(failed);
}

void SeedEvents(pqxx::work& write, bool reverseOrder)
{
    const std::string prefix = R"SQL(
        INSERT INTO economic_event(
            currency,event_family,event_timestamp_utc,source_agency,
            source_event_id,source_url,reference_period,event_importance,
            historical_time_confidence,source_release_date,
            source_release_time,source_timezone) VALUES )SQL";
    const std::string cpi = R"SQL(
        ('USD','CPI','2024-01-01 13:30:00+00','BLS','phase9:cpi:prior',
         'https://www.bls.gov/cpi','December 2023',3,'exact','2024-01-01',
         '08:30:00','America/New_York'))SQL";
    const std::string claimsA = R"SQL(
        ('USD','WEEKLY_CLAIMS','2024-01-04 12:30:00+00','DOL_ETA',
         'phase9:claims:a','https://oui.doleta.gov/press/a.pdf',
         'week ending 2023-12-30',3,'exact','2024-01-04','08:30:00',
         'America/New_York'))SQL";
    const std::string pce = R"SQL(
        ('USD','PCE','2024-01-05 13:30:00+00','BEA','phase9:pce:range',
         'https://www.bea.gov/news/pce','December 2023',3,'exact','2024-01-05',
         '08:30:00','America/New_York'))SQL";
    const std::string claimsB = R"SQL(
        ('USD','WEEKLY_CLAIMS','2024-01-11 12:30:00+00','DOL_ETA',
         'phase9:claims:b','https://oui.doleta.gov/press/b.pdf',
         'week ending 2024-01-06',3,'exact','2024-01-11','08:30:00',
         'America/New_York'))SQL";
    const std::vector<std::string> rows = reverseOrder
        ? std::vector<std::string>{claimsB, pce, claimsA, cpi}
        : std::vector<std::string>{cpi, claimsA, pce, claimsB};
    for (const std::string& row : rows)
        write.exec(prefix + row + ";");
}

void InsertHistoricalConsensus(
    pqxx::work& write,
    const std::string& sourceEventId,
    long long providerEventId,
    long long forecast,
    const std::string& retrievedAt)
{
    write.exec(R"SQL(
        INSERT INTO economic_event_consensus(
            economic_event_id,consensus_source,source_event_id,
            source_observation_id,source_event_name,source_release_date,
            source_artifact_path,source_artifact_sha256,
            candidate_classification,match_rule,semantic_contract,
            provider_provenance,provider_observed_at,forecast_available_at,
            source_retrieved_at,forecast_availability_proof,forecast_raw,
            forecast_parse_status,forecast_value_kind,forecast_value_low,
            forecast_canonical_value_low,forecast_unit,forecast_scale,
            previous_parse_status,actual_parse_status)
        SELECT economic_event_id,'MYFXBOOK',$2::bigint,
               'phase9:consensus:' || source_event_id,event_family,
               source_release_date,'fixture/phase9-consensus.json',repeat('a',64),
               'myfxbook_weekly_claims_pre_release_snapshot','phase9_exact',
               'myfxbook_weekly_claims_pre_release_snapshot_v1',
               '{"provider":"MYFXBOOK","phase":9}',
               event_timestamp_utc - interval '6 hours',
               event_timestamp_utc - interval '6 hours',$4::timestamptz,
               'internet_archive_pre_release_capture',$3::text,'parsed',
               'scalar',$3::numeric,$3::numeric,'count',1,'missing','missing'
        FROM economic_event WHERE source_event_id=$1;
    )SQL", pqxx::params{sourceEventId, providerEventId, forecast, retrievedAt});
}

void InsertReleaseActual(pqxx::work& write)
{
    write.exec(R"SQL(
        INSERT INTO economic_event_release_actual(
            economic_event_id,source_agency,source_observation_id,
            publication_state,revision_sequence,available_at,retrieved_at,
            source_url,source_artifact_path,source_artifact_sha256,
            semantic_contract,source_provenance,actual_raw,actual_value_kind,
            actual_value_low,actual_canonical_value_low,actual_unit,
            actual_scale)
        SELECT economic_event_id,'DOL_ETA','phase9:release:a','initial',0,
               event_timestamp_utc + interval '5 minutes',
               '2026-09-05 18:00:00+00',source_url,'fixture/phase9-a.pdf',
               repeat('b',64),'dol_eta_weekly_claims_v1',
               '{"provider":"DOL_ETA"}','240000','scalar',240000,240000,
               'count',1
        FROM economic_event WHERE source_event_id='phase9:claims:a';
    )SQL");
}

void InsertFirstRelease(
    pqxx::work& write,
    const std::string& sourceEventId,
    long long actual)
{
    write.exec(R"SQL(
        INSERT INTO economic_event_actual_observation(
            economic_event_id,source_name,source_role,source_native_event_id,
            source_observation_id,evidence_key,observation_kind,
            revision_sequence,source_publication_at,
            source_publication_time_status,observed_at,availability_proof,
            source_url,source_artifact_path,source_artifact_sha256,
            semantic_contract,source_provenance,actual_raw,actual_value_kind,
            actual_value_low,actual_canonical_value_low,actual_unit,
            actual_scale)
        SELECT economic_event_id,source_agency,'authoritative',source_event_id,
               'phase9:actual:' || source_event_id,
               'phase9:evidence:' || source_event_id,'initial',0,
               event_timestamp_utc + interval '5 minutes','exact',
               '2026-09-05 18:00:00+00','source_publication',source_url,
               'fixture/phase9-actual.pdf',repeat('c',64),
               'dol_eta_weekly_claims_v1','{"provider":"DOL_ETA"}',
               $2::text,'scalar',$2::numeric,$2::numeric,'count',1
        FROM economic_event WHERE source_event_id=$1;
    )SQL", pqxx::params{sourceEventId, actual});
}

void SeedBaseline(
    pqxx::connection& connection,
    bool reverseOrder,
    const std::string& retrievedAt)
{
    pqxx::work write{connection};
    SeedEvents(write, reverseOrder);
    InsertHistoricalConsensus(
        write, "phase9:claims:a", 9001, 235000, retrievedAt);
    InsertReleaseActual(write);
    InsertFirstRelease(write, "phase9:claims:a", 240000);
    write.commit();
}

EconomicCalendarSnapshotReport CreateSnapshot(pqxx::connection& connection)
{
    pqxx::work write{connection};
    write.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const auto report = CreateOrReuseEconomicCalendarSnapshot(
        write, "phase9_test", std::optional<std::string>{"focused test"}, false);
    write.commit();
    return report;
}

long long InsertExperiment(
    pqxx::work& write,
    const std::string& symbol,
    const std::optional<EconomicCalendarSnapshotIdentity>& snapshot)
{
    return write.exec(R"SQL(
        INSERT INTO experiment(
            symbol,prediction_horizon,c_next_threshold,target_epochs,
            checkpoint_interval,train_start,train_end,status,phase,
            duplicate_nonce,donchian20_mode,feature_warmup_scope,
            donchian_lookback,feature_ablation_mask,model_input_width,
            model_input_semantic_layout_version,
            economic_calendar_snapshot_id,economic_calendar_snapshot_hash)
        VALUES($1,4,0.0008,1,0,'2024-01-01','2024-02-01','pending','train',
               0,'enabled','full_history_warmup',20,'',77,6,$2,$3)
        RETURNING experiment_id;
    )SQL", pqxx::params{
        symbol,
        snapshot ? std::optional<long long>{snapshot->snapshotId} : std::nullopt,
        snapshot ? std::optional<std::string>{snapshot->contentHash} : std::nullopt})
        .one_row()[0].as<long long>();
}

void TestHashOnly(
    pqxx::connection& runtimeConnection,
    pqxx::connection& adminConnection,
    bool reverseOrder)
{
    SeedBaseline(
        adminConnection, reverseOrder,
        reverseOrder ? "2026-09-06 00:00:00+00" : "2026-09-05 00:00:00+00");
    pqxx::work read{runtimeConnection};
    read.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    read.exec("SET TRANSACTION READ ONLY;");
    const auto report = CreateOrReuseEconomicCalendarSnapshot(
        read, "phase9_hash_test", std::nullopt, true);
    assert(report.dryRun);
    assert(!report.snapshotId);
    std::cout << "HASH=" << report.contentHash << '\n';
}

void TestFull(
    pqxx::connection& connection,
    pqxx::connection& adminConnection)
{
    SeedBaseline(adminConnection, false, "2026-09-05 00:00:00+00");

    {
        pqxx::work dry{connection};
        dry.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
        dry.exec("SET TRANSACTION READ ONLY;");
        const auto report = CreateOrReuseEconomicCalendarSnapshot(
            dry, "phase9_dry_run", std::nullopt, true);
        assert(report.dryRun && !report.snapshotId);
        assert(report.canonicalEventCount == 4);
        assert(report.selectedConsensusCount == 1);
        assert(report.releaseActualCount == 1);
        assert(report.provenFirstReleaseActualCount == 1);
        assert(report.provenanceUnavailableCount == 3);
    }
    {
        pqxx::work rollback{connection};
        rollback.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
        const auto report = CreateOrReuseEconomicCalendarSnapshot(
            rollback, "phase9_rollback", std::nullopt, false);
        assert(report.snapshotId);
        rollback.abort();
    }
    {
        pqxx::read_transaction read{connection};
        assert(read.query_value<long long>(
            "SELECT count(*) FROM economic_calendar_snapshot") == 0);
    }

    const auto s1 = CreateSnapshot(connection);
    assert(s1.snapshotId && !s1.reused && !s1.dryRun);
    const EconomicCalendarSnapshotIdentity s1Identity{*s1.snapshotId,
                                                       s1.contentHash};
    const auto reused = CreateSnapshot(connection);
    assert(reused.snapshotId == s1.snapshotId);
    assert(reused.contentHash == s1.contentHash);
    assert(reused.reused);

    {
        pqxx::read_transaction read{connection};
        const auto before = LoadEconomicEventsForFeatureRange(
            read, "USD", "2024-01-03 00:00:00+00",
            "2024-01-04 12:34:59+00", s1Identity);
        const auto& claims = RequireSourceEvent(before, "phase9:claims:a");
        assert(claims.selectedConsensus);
        assert(claims.firstReleaseActualState ==
               EconomicEventFirstReleaseActualState::notYetAvailable);
        assert(!claims.firstReleaseActual);
        assert(!claims.releaseActual);

        const auto exact = LoadEconomicEventsForFeatureRange(
            read, "USD", "2024-01-03 00:00:00+00",
            "2024-01-04 12:35:00+00", s1Identity);
        const auto& exactClaims = RequireSourceEvent(exact, "phase9:claims:a");
        assert(exactClaims.firstReleaseActualState ==
               EconomicEventFirstReleaseActualState::provenFirstRelease);
        assert(exactClaims.firstReleaseActual);
        assert(!exactClaims.releaseActual);

        const auto after = LoadEconomicEventsForFeatureRange(
            read, "USD", "2024-01-03 00:00:00+00",
            "2024-01-04 12:35:00.000001+00", s1Identity);
        assert(RequireSourceEvent(after, "phase9:claims:a").releaseActual);

        const auto consensusBefore = LoadEconomicEventsForFeatureRange(
            read, "USD", "2024-01-03 00:00:00+00",
            "2024-01-04 12:29:59+00", s1Identity);
        assert(std::none_of(
            consensusBefore.begin(), consensusBefore.end(),
            [](const EconomicEvent& event)
            { return event.sourceEventId == "phase9:claims:a"; }));
        const auto consensusAt = LoadEconomicEventsForFeatureRange(
            read, "USD", "2024-01-03 00:00:00+00",
            "2024-01-04 12:30:00+00", s1Identity);
        assert(RequireSourceEvent(consensusAt, "phase9:claims:a")
                   .selectedConsensus);

        const auto range = LoadEconomicEventsForFeatureRange(
            read, "USD", "2024-01-03 00:00:00+00",
            "2024-01-11 12:30:00+00", s1Identity);
        assert(range.size() == 4);
        assert(range.front().sourceEventId == "phase9:cpi:prior");
        assert(range.back().sourceEventId == "phase9:claims:b");
        const auto noPrior = LoadEconomicEventsForFeatureRange(
            read, "USD", "2023-12-01 00:00:00+00",
            "2023-12-02 00:00:00+00", s1Identity);
        assert(noPrior.empty());
    }

    {
        pqxx::work immutable{connection};
        RequireFailure([&]
        {
            immutable.exec(
                "UPDATE economic_calendar_snapshot_event SET currency='EUR' "
                "WHERE economic_calendar_snapshot_id=$1",
                pqxx::params{*s1.snapshotId});
        });
    }
    {
        pqxx::work immutable{connection};
        RequireFailure([&]
        {
            immutable.exec(
                "DELETE FROM economic_calendar_snapshot_event WHERE "
                "economic_calendar_snapshot_id=$1",
                pqxx::params{*s1.snapshotId});
        });
    }
    {
        pqxx::work immutable{connection};
        RequireFailure([&]
        {
            immutable.exec(
                "INSERT INTO economic_calendar_snapshot_event SELECT * FROM "
                "economic_calendar_snapshot_event WHERE "
                "economic_calendar_snapshot_id=$1 LIMIT 1",
                pqxx::params{*s1.snapshotId});
        });
    }
    {
        pqxx::work immutable{connection};
        RequireFailure([&]
        {
            immutable.exec(
                "UPDATE economic_calendar_snapshot SET content_hash="
                "'fnv1a64:0000000000000000' WHERE "
                "economic_calendar_snapshot_id=$1",
                pqxx::params{*s1.snapshotId});
        });
    }

    long long boundExperimentId = 0;
    long long boundModelId = 0;
    {
        pqxx::work write{connection};
        boundExperimentId = InsertExperiment(
            write, "phase9bound", s1Identity);
        boundModelId = write.exec(
            "INSERT INTO model(name,experiment_id) VALUES('phase9-bound',$1) "
            "RETURNING model_id", pqxx::params{boundExperimentId})
            .one_row()[0].as<long long>();
        write.commit();
    }
    {
        pqxx::read_transaction read{connection};
        assert(LoadExperimentEconomicCalendarSnapshot(read, boundExperimentId)
                   ->snapshotId == *s1.snapshotId);
        assert(LoadModelEconomicCalendarSnapshot(read, boundModelId)
                   ->contentHash == s1.contentHash);
        const long long legacyExperimentId = read.query_value<long long>(
            "SELECT experiment_id FROM experiment WHERE symbol='phase9legacy'");
        const long long legacyModelId = read.query_value<long long>(
            "SELECT model_id FROM model WHERE name='phase9-legacy-model'");
        assert(!LoadExperimentEconomicCalendarSnapshot(
            read, legacyExperimentId));
        assert(!LoadModelEconomicCalendarSnapshot(read, legacyModelId));
        RequireFailure([&]
        {
            (void)InspectEconomicCalendarSnapshot(
                read, {*s1.snapshotId, "fnv1a64:0000000000000000"});
        });
        RequireFailure([&]
        {
            (void)InspectEconomicCalendarSnapshot(
                read, {9'999'999, "fnv1a64:0000000000000000"});
        });
    }

    {
        pqxx::work write{adminConnection};
        write.exec(R"SQL(
            INSERT INTO economic_event(
                currency,event_family,event_timestamp_utc,source_agency,
                source_event_id,source_url,reference_period,event_importance,
                historical_time_confidence,source_release_date,
                source_release_time,source_timezone)
            VALUES('USD','GDP','2024-01-06 13:30:00+00','BEA',
                   'phase9:gdp:later','https://www.bea.gov/news/gdp',
                   'Q4 2023',3,'exact','2024-01-06','08:30:00',
                   'America/New_York');
        )SQL");
        InsertHistoricalConsensus(
            write, "phase9:claims:b", 9002, 250000,
            "2026-09-05 00:00:00+00");
        InsertFirstRelease(write, "phase9:claims:b", 255000);
        write.commit();
    }

    {
        pqxx::read_transaction read{connection};
        const auto frozen = LoadEconomicEventsForFeatureRange(
            read, "USD", "2024-01-03 00:00:00+00",
            "2024-01-11 12:35:00+00", s1Identity);
        assert(!RequireSourceEvent(frozen, "phase9:claims:b")
                    .selectedConsensus);
        assert(RequireSourceEvent(frozen, "phase9:claims:b")
                   .firstReleaseActualState ==
               EconomicEventFirstReleaseActualState::provenanceUnavailable);
        assert(std::none_of(
            frozen.begin(), frozen.end(), [](const EconomicEvent& event)
            { return event.sourceEventId == "phase9:gdp:later"; }));

        const auto live = LoadEconomicEventsForFeatureRange(
            read, "USD", "2024-01-03 00:00:00+00",
            "2024-01-11 12:35:00+00");
        assert(RequireSourceEvent(live, "phase9:claims:b").selectedConsensus);
        assert(RequireSourceEvent(live, "phase9:claims:b")
                   .firstReleaseActualState ==
               EconomicEventFirstReleaseActualState::provenFirstRelease);
        assert(std::any_of(
            live.begin(), live.end(), [](const EconomicEvent& event)
            { return event.sourceEventId == "phase9:gdp:later"; }));
    }

    const auto s2 = CreateSnapshot(connection);
    assert(s2.snapshotId && *s2.snapshotId != *s1.snapshotId);
    assert(s2.contentHash != s1.contentHash);
    {
        pqxx::read_transaction read{connection};
        const auto resumeIdentity =
            LoadModelEconomicCalendarSnapshot(read, boundModelId);
        assert(resumeIdentity);
        assert(resumeIdentity->snapshotId == *s1.snapshotId);
        assert(resumeIdentity->contentHash == s1.contentHash);
        const auto updated = LoadEconomicEventsForFeatureRange(
            read, "USD", "2024-01-03 00:00:00+00",
            "2024-01-11 12:35:00+00",
            EconomicCalendarSnapshotIdentity{*s2.snapshotId, s2.contentHash});
        assert(RequireSourceEvent(updated, "phase9:claims:b")
                   .selectedConsensus);
        assert(RequireSourceEvent(updated, "phase9:claims:b")
                   .firstReleaseActual);
        assert(std::any_of(
            updated.begin(), updated.end(), [](const EconomicEvent& event)
            { return event.sourceEventId == "phase9:gdp:later"; }));
    }

    long long creatingId = 0;
    {
        pqxx::work write{connection};
        creatingId = write.exec(
            "INSERT INTO economic_calendar_snapshot("
            "hash_contract_version,content_hash,created_by) VALUES("
            "1,'fnv1a64:2222222222222222','phase9_test') RETURNING "
            "economic_calendar_snapshot_id")
            .one_row()[0].as<long long>();
        write.commit();
    }
    {
        pqxx::read_transaction read{connection};
        RequireFailure([&]
        {
            (void)InspectEconomicCalendarSnapshot(
                read, {creatingId, "fnv1a64:2222222222222222"});
        });
    }

    std::cout << "ECONOMIC_CALENDAR_SNAPSHOT_FULL=PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
    pqxx::connection connection{
        "host=" + RequiredEnvironment("LSTM_DB_HOST") +
        " user=" + RequiredEnvironment("LSTM_DB_USER") +
        " dbname=" + RequiredEnvironment("LSTM_TEST_DB_NAME")};
    pqxx::connection adminConnection{
        "host=" + RequiredEnvironment("LSTM_DB_HOST") +
        " user=" + RequiredEnvironment("LSTM_DB_ADMIN_USER") +
        " dbname=" + RequiredEnvironment("LSTM_TEST_DB_NAME")};
    if (argc == 3 && std::string{argv[1]} == "--hash-order")
    {
        TestHashOnly(
            connection, adminConnection, std::string{argv[2]} == "reverse");
        return 0;
    }
    TestFull(connection, adminConnection);
    return 0;
}
