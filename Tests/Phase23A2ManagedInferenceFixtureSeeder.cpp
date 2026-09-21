#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <pqxx/pqxx>

#include "EconomicEventRepository.hpp"

namespace
{
std::string RequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0')
        throw std::runtime_error(std::string{"missing environment: "} + name);
    return value;
}

std::string ConnectionString()
{
    const char* host = std::getenv("LSTM_DB_HOST");
    const char* user = std::getenv("LSTM_DB_USER");
    return "hostaddr=" + std::string{host && *host ? host : "127.0.0.1"} +
        " gssencmode=disable user=" +
        std::string{user && *user ? user : "pqxx"} + " dbname=" +
        RequiredEnvironment("LSTM_DB_NAME");
}

void InsertMatrix(pqxx::work& transaction, long long modelId,
                  const std::string& name, int rows, int cols,
                  const std::string& values)
{
    transaction.exec(
        "INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) "
        "SELECT $1,$2,$3::integer,$4::integer,(i-1)/$4::integer,"
        "(i-1)%$4::integer,value "
        "FROM unnest($5::double precision[]) WITH ORDINALITY AS v(value,i);",
        pqxx::params{modelId, name, rows, cols, values});
}

void InsertAsciiMatrix(pqxx::work& transaction, long long modelId,
                       const std::string& name, const std::string& value)
{
    std::string values{"{"};
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        if (index != 0) values += ',';
        values += std::to_string(static_cast<unsigned char>(value[index]));
    }
    values += '}';
    InsertMatrix(transaction, modelId, name, 1, static_cast<int>(value.size()),
                 values);
}

void SeedEconomicHistory(pqxx::work& transaction)
{
    transaction.exec(R"SQL(
        INSERT INTO economic_event(
            currency,event_family,event_timestamp_utc,source_agency,
            source_event_id,source_url,reference_period,event_importance,
            historical_time_confidence,source_release_date,
            source_release_time,source_timezone)
        VALUES
            ('USD','CPI','2023-12-01 13:30:00+00','BLS',
             'phase23a2:cpi:prior','https://www.bls.gov/cpi','November 2023',
             3,'exact','2023-12-01','08:30:00','America/New_York'),
            ('USD','WEEKLY_CLAIMS','2024-01-04 12:30:00+00','DOL_ETA',
             'phase23a2:claims:a','https://oui.doleta.gov/claims/a.pdf',
             'week ending 2023-12-30',3,'exact','2024-01-04','08:30:00',
             'America/New_York'),
            ('USD','PCE','2024-01-05 13:30:00+00','BEA',
             'phase23a2:pce','https://www.bea.gov/news/pce','December 2023',
             3,'exact','2024-01-05','08:30:00','America/New_York'),
            ('USD','WEEKLY_CLAIMS','2024-01-11 12:30:00+00','DOL_ETA',
             'phase23a2:claims:b','https://oui.doleta.gov/claims/b.pdf',
             'week ending 2024-01-06',3,'exact','2024-01-11','08:30:00',
             'America/New_York');
    )SQL");
    transaction.exec(R"SQL(
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
        SELECT economic_event_id,'MYFXBOOK',230201,
               'phase23a2:consensus',event_family,source_release_date,
               'fixture/phase23a2-consensus.json',repeat('a',64),
               'myfxbook_weekly_claims_pre_release_snapshot','phase23a2_exact',
               'myfxbook_weekly_claims_pre_release_snapshot_v1',
               '{"provider":"MYFXBOOK","phase":23}',
               event_timestamp_utc-interval '6 hours',
               event_timestamp_utc-interval '6 hours','2026-01-01 00:00:00+00',
               'internet_archive_pre_release_capture','235000','parsed','scalar',235000,
               235000,'count',1,'missing','missing'
        FROM economic_event WHERE source_event_id='phase23a2:claims:a';
    )SQL");
    transaction.exec(R"SQL(
        INSERT INTO economic_event_release_actual(
            economic_event_id,source_agency,source_observation_id,
            publication_state,revision_sequence,available_at,retrieved_at,
            source_url,source_artifact_path,source_artifact_sha256,
            semantic_contract,source_provenance,actual_raw,actual_value_kind,
            actual_value_low,actual_canonical_value_low,actual_unit,actual_scale)
        SELECT economic_event_id,'DOL_ETA','phase23a2:release','initial',0,
               event_timestamp_utc+interval '5 minutes','2026-01-01 00:00:00+00',
               source_url,'fixture/phase23a2-release.pdf',repeat('b',64),
               'dol_eta_weekly_claims_v1','{"provider":"DOL_ETA"}',
               '240000','scalar',240000,240000,'count',1
        FROM economic_event WHERE source_event_id='phase23a2:claims:a';
    )SQL");
    transaction.exec(R"SQL(
        INSERT INTO economic_event_actual_observation(
            economic_event_id,source_name,source_role,source_native_event_id,
            source_observation_id,evidence_key,observation_kind,
            revision_sequence,source_publication_at,
            source_publication_time_status,observed_at,availability_proof,
            source_url,source_artifact_path,source_artifact_sha256,
            semantic_contract,source_provenance,actual_raw,actual_value_kind,
            actual_value_low,actual_canonical_value_low,actual_unit,actual_scale)
        SELECT economic_event_id,source_agency,'authoritative',source_event_id,
               'phase23a2:actual','phase23a2:evidence','initial',0,
               event_timestamp_utc+interval '5 minutes','exact',
               '2026-01-01 00:00:00+00','source_publication',source_url,
               'fixture/phase23a2-actual.pdf',repeat('c',64),
               'dol_eta_weekly_claims_v1','{"provider":"DOL_ETA"}',
               '240000','scalar',240000,240000,'count',1
        FROM economic_event WHERE source_event_id='phase23a2:claims:a';
    )SQL");
}
} // namespace

int main()
{
    try
    {
        pqxx::connection connection{ConnectionString()};
        {
            pqxx::work write{connection};
            SeedEconomicHistory(write);
            write.commit();
        }

        EA::EconomicCalendar::EconomicCalendarSnapshotReport snapshot;
        {
            pqxx::work write{connection};
            write.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
            snapshot = EA::EconomicCalendar::CreateOrReuseEconomicCalendarSnapshot(
                write, "phase23a2_fixture", "deterministic managed inference fixture",
                false);
            write.commit();
        }
        if (!snapshot.snapshotId)
            throw std::runtime_error("phase23a2_snapshot_not_materialized");

        pqxx::work write{connection};
        const long long experimentId = write.exec(
            "INSERT INTO experiment(symbol,prediction_horizon,c_next_threshold,"
            "target_epochs,checkpoint_interval,train_start,train_end,infer_start,"
            "infer_end,status,phase,current_operation,duplicate_nonce,"
            "donchian20_mode,feature_warmup_scope,donchian_lookback,"
            "feature_ablation_mask,model_input_width,"
            "model_input_semantic_layout_version,economic_calendar_snapshot_id,"
            "economic_calendar_snapshot_hash) VALUES("
            "'phase23a2audrmp',4,0.0008,20,20,'2024-01-01','2024-01-05',"
            "'2024-01-05','2024-01-06','pending','infer','infer',2302,"
            "'enabled','full_history_warmup',20,'',77,7,$1,$2) "
            "RETURNING experiment_id;",
            pqxx::params{*snapshot.snapshotId, snapshot.contentHash})
            .one_row()[0].as<long long>();
        const long long modelId = write.exec(
            "INSERT INTO model(name,experiment_id) VALUES("
            "'phase23a2-deterministic-model',$1) RETURNING model_id;",
            pqxx::params{experimentId}).one_row()[0].as<long long>();

        // PostgreSQL's generator makes the 78x4 LSTM gate matrix explicit
        // without importing any production model content.
        write.exec("INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) "
                   "SELECT $1,'param',78,4,(i-1)/4,(i-1)%4,0.0 "
                   "FROM generate_series(1,312) i;", pqxx::params{modelId});
        InsertMatrix(write, modelId, "bias", 1, 4, "{0,0,0,0}");
        InsertMatrix(write, modelId, "model_meta", 1, 3, "{1,77,1}");
        InsertMatrix(write, modelId, "model_input_semantics_meta", 1, 2,
                     "{1,7}");
        InsertMatrix(write, modelId, "target_meta", 1, 6, "{2,1,0,0,0,1}");
        InsertMatrix(write, modelId, "returnHeadDirWeight", 1, 3, "{0,0,0}");
        InsertMatrix(write, modelId, "returnHeadDirBias", 1, 3, "{0,0,1}");
        InsertMatrix(write, modelId, "train_config_meta", 1, 14,
                     "{1,4,0.0008,30,1,1,1,1,1,1,20,1,1,1}");
        InsertAsciiMatrix(write, modelId, "train_symbol_meta", "phase23a2audrmp");
        InsertAsciiMatrix(write, modelId, "donchian20_mode_meta", "enabled");
        InsertAsciiMatrix(write, modelId, "feature_warmup_scope_meta",
                          "full_history_warmup");
        InsertAsciiMatrix(write, modelId, "donchian_lookback_meta", "20");
        write.exec("UPDATE experiment SET last_model_id=$2 WHERE experiment_id=$1;",
                   pqxx::params{experimentId, modelId});
        write.commit();

        std::cout << "PHASE23A2_FIXTURE"
                  << ",experiment_id=" << experimentId
                  << ",model_id=" << modelId
                  << ",snapshot_id=" << *snapshot.snapshotId
                  << ",snapshot_hash=" << snapshot.contentHash
                  << std::endl;
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "PHASE23A2_FIXTURE_FAILED,error=" << error.what() << std::endl;
        return 1;
    }
}
