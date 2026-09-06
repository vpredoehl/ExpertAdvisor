#include "EconomicEventRepository.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace EA::EconomicCalendar
{
namespace
{

std::string Hex64(std::uint64_t value)
{
    std::ostringstream output;
    output << std::hex << std::nouppercase << std::setfill('0')
           << std::setw(16) << value;
    return output.str();
}

std::string Fnv1a64(std::string_view text)
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char byte : text)
    {
        hash ^= byte;
        hash *= UINT64_C(1099511628211);
    }
    return "fnv1a64:" + Hex64(hash);
}

void AppendCanonicalField(std::string& canonical, std::string_view value)
{
    canonical.append(std::to_string(value.size()));
    canonical.push_back(':');
    canonical.append(value);
    canonical.push_back(';');
}

std::string CanonicalRowsQuery(bool snapshot)
{
    const std::string eventFields =
        "e.currency,e.event_family,"
        "ROUND(EXTRACT(EPOCH FROM e.event_timestamp_utc)*1000000)::bigint,"
        "e.source_agency,e.source_event_id,e.source_url,e.reference_period,"
        "e.event_importance,e.historical_time_confidence,"
        "to_char(e.source_release_date,'YYYY-MM-DD'),"
        "to_char(e.source_release_time,'HH24:MI:SS.US'),e.source_timezone";
    const std::string eventRows = snapshot
        ? "SELECT e.* FROM "
          "economic_calendar_snapshot_event e WHERE "
          "e.economic_calendar_snapshot_id=$1"
        : "SELECT row_number() OVER (ORDER BY "
          "jsonb_build_array(" + eventFields + ")::text COLLATE \"C\") "
          "AS canonical_event_order,e.* FROM economic_event e";
    const std::string consensus = snapshot
        ? "economic_calendar_snapshot_consensus c JOIN event_rows e ON "
          "e.economic_calendar_snapshot_id=c.economic_calendar_snapshot_id "
          "AND e.economic_event_id=c.economic_event_id"
        : "economic_event_selected_consensus c JOIN event_rows e "
          "USING(economic_event_id)";
    const std::string release = snapshot
        ? "economic_calendar_snapshot_release_actual a JOIN event_rows e ON "
          "e.economic_calendar_snapshot_id=a.economic_calendar_snapshot_id "
          "AND e.economic_event_id=a.economic_event_id"
        : "economic_event_feature_release_actual a JOIN event_rows e "
          "USING(economic_event_id)";
    const std::string first = snapshot
        ? "economic_calendar_snapshot_first_release_actual f JOIN event_rows e ON "
          "e.economic_calendar_snapshot_id=f.economic_calendar_snapshot_id "
          "AND e.economic_event_id=f.economic_event_id"
        : "economic_event_first_release_actual f JOIN event_rows e "
          "USING(economic_event_id)";

    return
        "WITH event_rows AS (" + eventRows + "),canonical_rows AS ("
        "SELECT 'event'::text AS row_kind, jsonb_build_array("
        "e.canonical_event_order," + eventFields +
        ")::text AS canonical FROM event_rows e" +
        " UNION ALL SELECT 'consensus',jsonb_build_array("
        "e.canonical_event_order,c.consensus_value_low,"
        "c.consensus_value_high,c.consensus_value_kind,c.consensus_unit,"
        "c.consensus_scale,c.consensus_qualifier,c.consensus_source,"
        "c.source_observation_id,to_char(c.source_release_date,'YYYY-MM-DD'),"
        "c.source_artifact_path,c.source_artifact_sha256,"
        "c.candidate_classification,c.match_rule,c.semantic_contract,"
        "c.provider_provenance,ROUND(EXTRACT(EPOCH FROM "
        "c.provider_observed_at)*1000000)::bigint,"
        "ROUND(EXTRACT(EPOCH FROM c.forecast_available_at)*1000000)::bigint,"
        "c.forecast_availability_proof)::text FROM " + consensus +
        " UNION ALL SELECT 'release_actual',jsonb_build_array("
        "e.canonical_event_order,"
        "ROUND(EXTRACT(EPOCH FROM a.available_at)*1000000)::bigint,"
        "a.actual_value_kind,a.actual_canonical_value_low,"
        "a.actual_canonical_value_high,a.actual_unit,a.actual_scale,"
        "a.actual_qualifier,a.source_agency,a.source_observation_id,"
        "a.source_artifact_path,a.source_artifact_sha256,a.semantic_contract,"
        "a.source_provenance)::text FROM " + release +
        " UNION ALL SELECT 'first_release',jsonb_build_array("
        "e.canonical_event_order,f.provenance_state,f.selection_reason,"
        "f.observation_count,f.observed_semantic_value_count,"
        "f.earliest_candidate_count,f.possible_first_semantic_value_count,"
        "f.first_release_value_kind,f.first_release_value_low,"
        "f.first_release_value_high,f.first_release_unit,"
        "f.first_release_scale,f.first_release_qualifier,"
        "ROUND(EXTRACT(EPOCH FROM f.proven_available_at)*1000000)::bigint,"
        "f.first_release_source,f.first_release_source_native_event_id,"
        "f.first_release_source_observation_id,"
        "f.first_release_evidence_key)::text FROM " + first +
        ") SELECT row_kind,canonical FROM canonical_rows "
        "ORDER BY row_kind COLLATE \"C\",canonical COLLATE \"C\";";
}

pqxx::result CanonicalRows(
    pqxx::transaction_base& transaction,
    const std::optional<long long>& snapshotId)
{
    if (snapshotId)
        return transaction.exec(
            CanonicalRowsQuery(true), pqxx::params{*snapshotId});
    return transaction.exec(CanonicalRowsQuery(false));
}

std::string SnapshotContentHash(const pqxx::result& rows)
{
    std::string canonical = "economic-calendar-snapshot-v1;";
    for (const pqxx::row& row : rows)
    {
        AppendCanonicalField(canonical, row[0].as<std::string>());
        AppendCanonicalField(canonical, row[1].as<std::string>());
    }
    return Fnv1a64(canonical);
}

EconomicCalendarSnapshotReport BuildLiveSnapshotReport(
    pqxx::transaction_base& transaction)
{
    EconomicCalendarSnapshotReport report;
    report.contentHash = SnapshotContentHash(CanonicalRows(transaction, {}));
    const pqxx::row counts = transaction.exec(
        "SELECT (SELECT count(*) FROM economic_event),"
        "(SELECT count(*) FROM economic_event_selected_consensus),"
        "(SELECT count(*) FROM economic_event_feature_release_actual),"
        "(SELECT count(*) FROM economic_event_first_release_actual "
        " WHERE provenance_state='proven_first_release'),"
        "(SELECT count(*) FROM economic_event_first_release_actual "
        " WHERE provenance_state='provenance_unavailable'),"
        "(SELECT count(*) FROM economic_event_first_release_actual "
        " WHERE provenance_state='ambiguous'),"
        "COALESCE((SELECT jsonb_object_agg(k,n ORDER BY k) FROM ("
        " SELECT source_agency||'/'||event_family AS k,count(*) AS n"
        " FROM economic_event GROUP BY source_agency,event_family) q),"
        "'{}'::jsonb)::text;").one_row();
    report.canonicalEventCount = counts[0].as<long long>();
    report.selectedConsensusCount = counts[1].as<long long>();
    report.releaseActualCount = counts[2].as<long long>();
    report.provenFirstReleaseActualCount = counts[3].as<long long>();
    report.provenanceUnavailableCount = counts[4].as<long long>();
    report.ambiguousFirstReleaseCount = counts[5].as<long long>();
    report.sourceFamilyCountsJson = counts[6].as<std::string>();
    return report;
}

EconomicCalendarSnapshotReport BuildStoredSnapshotReport(
    pqxx::transaction_base& transaction,
    long long snapshotId)
{
    EconomicCalendarSnapshotReport report;
    report.snapshotId = snapshotId;
    report.contentHash = SnapshotContentHash(
        CanonicalRows(transaction, snapshotId));
    const pqxx::row counts = transaction.exec(
        "SELECT (SELECT count(*) FROM economic_calendar_snapshot_event"
        " WHERE economic_calendar_snapshot_id=$1),"
        "(SELECT count(*) FROM economic_calendar_snapshot_consensus"
        " WHERE economic_calendar_snapshot_id=$1),"
        "(SELECT count(*) FROM economic_calendar_snapshot_release_actual"
        " WHERE economic_calendar_snapshot_id=$1),"
        "(SELECT count(*) FROM economic_calendar_snapshot_first_release_actual"
        " WHERE economic_calendar_snapshot_id=$1 AND"
        " provenance_state='proven_first_release'),"
        "(SELECT count(*) FROM economic_calendar_snapshot_first_release_actual"
        " WHERE economic_calendar_snapshot_id=$1 AND"
        " provenance_state='provenance_unavailable'),"
        "(SELECT count(*) FROM economic_calendar_snapshot_first_release_actual"
        " WHERE economic_calendar_snapshot_id=$1 AND"
        " provenance_state='ambiguous'),"
        "COALESCE((SELECT jsonb_object_agg(k,n ORDER BY k) FROM ("
        " SELECT source_agency||'/'||event_family AS k,count(*) AS n"
        " FROM economic_calendar_snapshot_event"
        " WHERE economic_calendar_snapshot_id=$1"
        " GROUP BY source_agency,event_family) q),'{}'::jsonb)::text;",
        pqxx::params{snapshotId}).one_row();
    report.canonicalEventCount = counts[0].as<long long>();
    report.selectedConsensusCount = counts[1].as<long long>();
    report.releaseActualCount = counts[2].as<long long>();
    report.provenFirstReleaseActualCount = counts[3].as<long long>();
    report.provenanceUnavailableCount = counts[4].as<long long>();
    report.ambiguousFirstReleaseCount = counts[5].as<long long>();
    report.sourceFamilyCountsJson = counts[6].as<std::string>();
    return report;
}

void RequireSameReport(
    const EconomicCalendarSnapshotReport& expected,
    const EconomicCalendarSnapshotReport& actual)
{
    if (actual.contentHash != expected.contentHash ||
        actual.canonicalEventCount != expected.canonicalEventCount ||
        actual.selectedConsensusCount != expected.selectedConsensusCount ||
        actual.releaseActualCount != expected.releaseActualCount ||
        actual.provenFirstReleaseActualCount !=
            expected.provenFirstReleaseActualCount ||
        actual.provenanceUnavailableCount !=
            expected.provenanceUnavailableCount ||
        actual.ambiguousFirstReleaseCount !=
            expected.ambiguousFirstReleaseCount ||
        actual.sourceFamilyCountsJson != expected.sourceFamilyCountsJson)
    {
        throw std::runtime_error(
            "economic_calendar_snapshot_materialization_mismatch");
    }
}

template <typename Value>
std::optional<Value> OptionalValue(
    const pqxx::row& row,
    const char* column)
{
    const pqxx::field field = row[column];

    if (field.is_null())
        return std::nullopt;

    return field.as<Value>();
}

void ValidateCurrency(const std::string& currency)
{
    if (currency.size() != 3)
        throw std::invalid_argument(
            "economic_event_currency_must_be_three_characters");

    if (!std::all_of(
            currency.begin(),
            currency.end(),
            [](unsigned char value)
            {
                return std::isupper(value) != 0;
            }))
    {
        throw std::invalid_argument(
            "economic_event_currency_must_be_uppercase");
    }
}

EconomicEvent MapEconomicEvent(const pqxx::row& row)
{
    EconomicEvent event;

    event.economicEventId =
        row["economic_event_id"].as<long long>();

    event.currency =
        row["currency"].as<std::string>();

    event.eventFamily =
        row["event_family"].as<std::string>();

    event.eventTimestampUnixMicros =
        row["event_timestamp_unix_micros"]
            .as<std::int64_t>();

    event.eventTimestampUtc =
        row["event_timestamp_utc_text"]
            .as<std::string>();

    event.sourceAgency =
        row["source_agency"].as<std::string>();

    event.sourceEventId =
        OptionalValue<std::string>(
            row,
            "source_event_id");

    event.sourceUrl =
        row["source_url"].as<std::string>();

    event.referencePeriod =
        OptionalValue<std::string>(
            row,
            "reference_period");

    event.eventImportance =
        row["event_importance"].as<int>();

    event.historicalTimeConfidence =
        row["historical_time_confidence"]
            .as<std::string>();

    event.sourceReleaseDate =
        OptionalValue<std::string>(
            row,
            "source_release_date");

    event.sourceReleaseTime =
        OptionalValue<std::string>(
            row,
            "source_release_time");

    event.sourceTimezone =
        OptionalValue<std::string>(
            row,
            "source_timezone");

    if (!row["consensus_value_low"].is_null())
    {
        EconomicEventSelectedConsensus selected;
        selected.forecast.valueKind =
            row["consensus_value_kind"].as<std::string>();
        selected.forecast.canonicalValueLow =
            row["consensus_value_low"].as<double>();
        selected.forecast.canonicalValueHigh =
            OptionalValue<double>(row, "consensus_value_high");
        selected.forecast.unit =
            row["consensus_unit"].as<std::string>();
        selected.forecast.scale =
            row["consensus_scale"].as<double>();
        selected.forecast.qualifier =
            OptionalValue<std::string>(row, "consensus_qualifier");
        selected.provider =
            row["consensus_source"].as<std::string>();

        event.selectedConsensus = std::move(selected);
    }

    if (!row["release_actual_value_low"].is_null())
    {
        EconomicEventReleaseActual actual;
        actual.actual.valueKind =
            row["release_actual_value_kind"].as<std::string>();
        actual.actual.canonicalValueLow =
            row["release_actual_value_low"].as<double>();
        actual.actual.canonicalValueHigh = OptionalValue<double>(
            row, "release_actual_value_high");
        actual.actual.unit =
            row["release_actual_unit"].as<std::string>();
        actual.actual.scale =
            row["release_actual_scale"].as<double>();
        actual.actual.qualifier = OptionalValue<std::string>(
            row, "release_actual_qualifier");
        actual.availableAtUnixMicros =
            row["release_actual_available_at_unix_micros"]
                .as<std::int64_t>();
        actual.sourceAgency =
            row["release_actual_source_agency"].as<std::string>();
        actual.sourceObservationId =
            row["release_actual_source_observation_id"].as<std::string>();
        actual.sourceArtifactPath =
            row["release_actual_source_artifact_path"].as<std::string>();
        actual.sourceArtifactSha256 =
            row["release_actual_source_artifact_sha256"].as<std::string>();
        actual.semanticContract =
            row["release_actual_semantic_contract"].as<std::string>();
        event.releaseActual = std::move(actual);
    }

    const std::string pitState =
        row["pit_first_release_provenance_state"].as<std::string>();
    if (pitState == "provenance_unavailable")
    {
        event.firstReleaseActualState =
            EconomicEventFirstReleaseActualState::provenanceUnavailable;
    }
    else if (pitState == "ambiguous")
    {
        event.firstReleaseActualState =
            EconomicEventFirstReleaseActualState::ambiguous;
    }
    else if (pitState == "not_yet_available")
    {
        event.firstReleaseActualState =
            EconomicEventFirstReleaseActualState::notYetAvailable;
    }
    else if (pitState == "proven_first_release")
    {
        event.firstReleaseActualState =
            EconomicEventFirstReleaseActualState::provenFirstRelease;
    }
    else
    {
        throw std::runtime_error(
            "economic_event_first_release_actual_unknown_pit_state:" +
            pitState);
    }

    event.firstReleaseActualSelectionReason =
        row["first_release_selection_reason"].as<std::string>();

    if (!row["pit_first_release_actual_value_low"].is_null())
    {
        EconomicEventFirstReleaseActual actual;
        actual.actual.valueKind = row[
            "pit_first_release_actual_value_kind"].as<std::string>();
        actual.actual.canonicalValueLow = row[
            "pit_first_release_actual_value_low"].as<double>();
        actual.actual.canonicalValueHigh = OptionalValue<double>(
            row, "pit_first_release_actual_value_high");
        actual.actual.unit = row[
            "pit_first_release_actual_unit"].as<std::string>();
        actual.actual.scale = row[
            "pit_first_release_actual_scale"].as<double>();
        actual.actual.qualifier = OptionalValue<std::string>(
            row, "pit_first_release_actual_qualifier");
        actual.provenAvailableAtUnixMicros = row[
            "pit_first_release_proven_available_at_unix_micros"]
                .as<std::int64_t>();
        actual.sourceName =
            row["pit_first_release_source_name"].as<std::string>();
        actual.sourceNativeEventId = OptionalValue<std::string>(
            row, "pit_first_release_source_native_event_id");
        actual.sourceObservationId = row[
            "pit_first_release_source_observation_id"].as<std::string>();
        actual.evidenceKey =
            row["pit_first_release_evidence_key"].as<std::string>();
        event.firstReleaseActual = std::move(actual);
    }

    if ((event.firstReleaseActualState ==
             EconomicEventFirstReleaseActualState::provenFirstRelease) !=
        event.firstReleaseActual.has_value())
    {
        throw std::runtime_error(
            "economic_event_first_release_actual_pit_state_value_mismatch");
    }

    return event;
}

std::string EconomicEventProjection(
    std::string_view eventRelation,
    std::string_view consensusRelation,
    std::string_view releaseActualRelation,
    std::string_view firstReleaseActualRelation,
    std::string_view firstReleaseAssessmentRelation,
    std::string_view eventIdExpression = {})
{
    const std::string prefix = eventRelation.empty()
        ? std::string{}
        : std::string{eventRelation} + ".";
    const std::string consensusPrefix = consensusRelation.empty()
        ? std::string{}
        : std::string{consensusRelation} + ".";
    const std::string releaseActualPrefix = releaseActualRelation.empty()
        ? std::string{}
        : std::string{releaseActualRelation} + ".";
    const std::string firstReleaseActualPrefix =
        firstReleaseActualRelation.empty()
            ? std::string{}
            : std::string{firstReleaseActualRelation} + ".";
    const std::string firstReleaseAssessmentPrefix =
        firstReleaseAssessmentRelation.empty()
            ? std::string{}
            : std::string{firstReleaseAssessmentRelation} + ".";

    const std::string projectedEventId = eventIdExpression.empty()
        ? prefix + "economic_event_id"
        : std::string{eventIdExpression};

    return
        projectedEventId + " AS economic_event_id, " +
        prefix + "currency, " +
        prefix + "event_family, " +
        "ROUND(EXTRACT(EPOCH FROM " + prefix +
            "event_timestamp_utc) * 1000000)::bigint "
            "AS event_timestamp_unix_micros, " +
        "to_char(" + prefix +
            "event_timestamp_utc AT TIME ZONE 'UTC', "
            "'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"') "
            "AS event_timestamp_utc_text, " +
        prefix + "source_agency, " +
        prefix + "source_event_id, " +
        prefix + "source_url, " +
        prefix + "reference_period, " +
        prefix + "event_importance, " +
        prefix + "historical_time_confidence, " +
        prefix + "source_release_date::text AS source_release_date, " +
        prefix + "source_release_time::text AS source_release_time, " +
        prefix + "source_timezone, " +
        consensusPrefix + "consensus_value_low, " +
        consensusPrefix + "consensus_value_high, " +
        consensusPrefix + "consensus_value_kind, " +
        consensusPrefix + "consensus_unit, " +
        consensusPrefix + "consensus_scale, " +
        consensusPrefix + "consensus_qualifier, " +
        consensusPrefix + "consensus_source, " +
        "ROUND(EXTRACT(EPOCH FROM " + releaseActualPrefix +
            "available_at) * 1000000)::bigint "
            "AS release_actual_available_at_unix_micros, " +
        releaseActualPrefix + "actual_value_kind "
            "AS release_actual_value_kind, " +
        releaseActualPrefix + "actual_canonical_value_low "
            "AS release_actual_value_low, " +
        releaseActualPrefix + "actual_canonical_value_high "
            "AS release_actual_value_high, " +
        releaseActualPrefix + "actual_unit AS release_actual_unit, " +
        releaseActualPrefix + "actual_scale AS release_actual_scale, " +
        releaseActualPrefix +
            "actual_qualifier AS release_actual_qualifier, " +
        releaseActualPrefix +
            "source_agency AS release_actual_source_agency, " +
        releaseActualPrefix +
            "source_observation_id AS release_actual_source_observation_id, " +
        releaseActualPrefix +
            "source_artifact_path AS release_actual_source_artifact_path, " +
        releaseActualPrefix +
            "source_artifact_sha256 AS release_actual_source_artifact_sha256, " +
        releaseActualPrefix +
            "semantic_contract AS release_actual_semantic_contract, " +
        firstReleaseActualPrefix + "first_release_actual_value_kind "
            "AS pit_first_release_actual_value_kind, " +
        firstReleaseActualPrefix + "first_release_actual_value_low "
            "AS pit_first_release_actual_value_low, " +
        firstReleaseActualPrefix + "first_release_actual_value_high "
            "AS pit_first_release_actual_value_high, " +
        firstReleaseActualPrefix + "first_release_actual_unit "
            "AS pit_first_release_actual_unit, " +
        firstReleaseActualPrefix + "first_release_actual_scale "
            "AS pit_first_release_actual_scale, " +
        firstReleaseActualPrefix + "first_release_actual_qualifier "
            "AS pit_first_release_actual_qualifier, " +
        "ROUND(EXTRACT(EPOCH FROM " + firstReleaseActualPrefix +
            "proven_available_at) * 1000000)::bigint "
            "AS pit_first_release_proven_available_at_unix_micros, " +
        "CASE WHEN " + firstReleaseActualPrefix +
            "economic_event_id IS NOT NULL THEN " +
            firstReleaseActualPrefix + "provenance_state "
            "WHEN " + firstReleaseAssessmentPrefix +
            "provenance_state = 'proven_first_release' "
            "THEN 'not_yet_available' ELSE " +
            firstReleaseAssessmentPrefix + "provenance_state END "
            "AS pit_first_release_provenance_state, " +
        firstReleaseAssessmentPrefix + "selection_reason "
            "AS first_release_selection_reason, " +
        firstReleaseActualPrefix + "source_name "
            "AS pit_first_release_source_name, " +
        firstReleaseActualPrefix + "source_native_event_id "
            "AS pit_first_release_source_native_event_id, " +
        firstReleaseActualPrefix + "source_observation_id "
            "AS pit_first_release_source_observation_id, " +
        firstReleaseActualPrefix + "evidence_key "
            "AS pit_first_release_evidence_key ";
}

std::vector<EconomicEvent> MapEconomicEvents(const pqxx::result& rows)
{
    std::vector<EconomicEvent> events;
    events.reserve(rows.size());

    for (const pqxx::row& row : rows)
        events.push_back(MapEconomicEvent(row));

    return events;
}

} // namespace


bool SameEconomicCalendarSnapshotIdentity(
    const std::optional<EconomicCalendarSnapshotIdentity>& lhs,
    const std::optional<EconomicCalendarSnapshotIdentity>& rhs)
{
    if (lhs.has_value() != rhs.has_value()) return false;
    if (!lhs) return true;
    return lhs->snapshotId == rhs->snapshotId &&
           lhs->contentHash == rhs->contentHash;
}


bool EconomicEventSchemaExists(
    pqxx::transaction_base& transaction)
{
    const pqxx::result rows = transaction.exec(
        "SELECT to_regclass("
        "'public.economic_event'"
        ") IS NOT NULL AS exists;");

    return
        rows.size() == 1 &&
        rows.one_row()["exists"].as<bool>();
}

EconomicCalendarSnapshotReport InspectEconomicCalendarSnapshot(
    pqxx::transaction_base& transaction,
    const EconomicCalendarSnapshotIdentity& identity)
{
    if (identity.snapshotId <= 0 || identity.contentHash.empty())
        throw std::invalid_argument(
            "economic_calendar_snapshot_identity_invalid");
    const pqxx::result headers = transaction.exec(
        "SELECT snapshot_state,hash_contract_version,content_hash,"
        "canonical_event_count,selected_consensus_count,release_actual_count,"
        "proven_first_release_actual_count,provenance_unavailable_count,"
        "ambiguous_first_release_count,source_family_counts::text "
        "FROM economic_calendar_snapshot "
        "WHERE economic_calendar_snapshot_id=$1;",
        pqxx::params{identity.snapshotId});
    if (headers.size() != 1)
        throw std::runtime_error("economic_calendar_snapshot_missing");
    const pqxx::row header = headers.one_row();
    if (header[0].as<std::string>() != "finalized")
        throw std::runtime_error("economic_calendar_snapshot_not_finalized");
    if (header[1].as<int>() != kEconomicCalendarSnapshotHashContractVersion)
        throw std::runtime_error(
            "economic_calendar_snapshot_hash_contract_unsupported");
    if (header[2].as<std::string>() != identity.contentHash)
        throw std::runtime_error("economic_calendar_snapshot_hash_mismatch");

    EconomicCalendarSnapshotReport stored = BuildStoredSnapshotReport(
        transaction, identity.snapshotId);
    EconomicCalendarSnapshotReport expected;
    expected.snapshotId = identity.snapshotId;
    expected.contentHash = header[2].as<std::string>();
    expected.canonicalEventCount = header[3].as<long long>();
    expected.selectedConsensusCount = header[4].as<long long>();
    expected.releaseActualCount = header[5].as<long long>();
    expected.provenFirstReleaseActualCount = header[6].as<long long>();
    expected.provenanceUnavailableCount = header[7].as<long long>();
    expected.ambiguousFirstReleaseCount = header[8].as<long long>();
    expected.sourceFamilyCountsJson = header[9].as<std::string>();
    RequireSameReport(expected, stored);
    return stored;
}

EconomicCalendarSnapshotReport CreateOrReuseEconomicCalendarSnapshot(
    pqxx::work& transaction,
    const std::string& createdBy,
    const std::optional<std::string>& creationNote,
    bool dryRun)
{
    if (createdBy.empty())
        throw std::invalid_argument(
            "economic_calendar_snapshot_created_by_required");
    EconomicCalendarSnapshotReport live = BuildLiveSnapshotReport(transaction);
    live.dryRun = dryRun;
    if (dryRun) return live;

    const pqxx::result existing = transaction.exec(
        "SELECT economic_calendar_snapshot_id FROM economic_calendar_snapshot "
        "WHERE content_hash=$1;", pqxx::params{live.contentHash});
    if (!existing.empty())
    {
        const long long snapshotId = existing.one_row()[0].as<long long>();
        EconomicCalendarSnapshotReport reused = InspectEconomicCalendarSnapshot(
            transaction, {snapshotId, live.contentHash});
        RequireSameReport(live, reused);
        reused.reused = true;
        return reused;
    }

    const long long snapshotId = transaction.exec(
        "INSERT INTO economic_calendar_snapshot("
        "hash_contract_version,content_hash,created_by,creation_note) "
        "VALUES($1,$2,$3,$4) RETURNING economic_calendar_snapshot_id;",
        pqxx::params{kEconomicCalendarSnapshotHashContractVersion,
                     live.contentHash, createdBy, creationNote})
        .one_row()[0].as<long long>();

    transaction.exec(
        "INSERT INTO economic_calendar_snapshot_event("
        "economic_calendar_snapshot_id,economic_event_id,canonical_event_order,currency,"
        "event_family,event_timestamp_utc,source_agency,source_event_id,"
        "source_url,reference_period,event_importance,"
        "historical_time_confidence,source_release_date,source_release_time,"
        "source_timezone) SELECT $1,economic_event_id,row_number() OVER ("
        "ORDER BY jsonb_build_array(currency,event_family,"
        "ROUND(EXTRACT(EPOCH FROM event_timestamp_utc)*1000000)::bigint,"
        "source_agency,source_event_id,source_url,reference_period,"
        "event_importance,historical_time_confidence,"
        "to_char(source_release_date,'YYYY-MM-DD'),"
        "to_char(source_release_time,'HH24:MI:SS.US'),source_timezone)"
        "::text COLLATE \"C\"),currency,event_family,"
        "event_timestamp_utc,source_agency,source_event_id,source_url,"
        "reference_period,event_importance,historical_time_confidence,"
        "source_release_date,source_release_time,source_timezone "
        "FROM economic_event ORDER BY economic_event_id;",
        pqxx::params{snapshotId});
    transaction.exec(
        "INSERT INTO economic_calendar_snapshot_consensus("
        "economic_calendar_snapshot_id,economic_event_id,"
        "economic_event_consensus_id,consensus_value_low,"
        "consensus_value_high,consensus_value_kind,consensus_unit,"
        "consensus_scale,consensus_qualifier,consensus_source,source_report_id,"
        "source_event_id,source_observation_id,source_release_date,"
        "source_artifact_path,source_artifact_sha256,candidate_classification,"
        "match_rule,semantic_contract,provider_provenance,provider_observed_at,"
        "forecast_available_at,source_retrieved_at,"
        "forecast_availability_proof) SELECT $1,economic_event_id,"
        "economic_event_consensus_id,consensus_value_low,consensus_value_high,"
        "consensus_value_kind,consensus_unit,consensus_scale,"
        "consensus_qualifier,consensus_source,source_report_id,source_event_id,"
        "source_observation_id,source_release_date,source_artifact_path,"
        "source_artifact_sha256,candidate_classification,match_rule,"
        "semantic_contract,provider_provenance,provider_observed_at,"
        "forecast_available_at,source_retrieved_at,"
        "forecast_availability_proof FROM economic_event_selected_consensus "
        "ORDER BY economic_event_id;", pqxx::params{snapshotId});
    transaction.exec(
        "INSERT INTO economic_calendar_snapshot_release_actual("
        "economic_calendar_snapshot_id,economic_event_id,"
        "economic_event_release_actual_id,available_at,actual_value_kind,"
        "actual_canonical_value_low,actual_canonical_value_high,actual_unit,"
        "actual_scale,actual_qualifier,source_agency,source_observation_id,"
        "source_artifact_path,source_artifact_sha256,semantic_contract,"
        "source_provenance) SELECT $1,economic_event_id,"
        "economic_event_release_actual_id,available_at,actual_value_kind,"
        "actual_canonical_value_low,actual_canonical_value_high,actual_unit,"
        "actual_scale,actual_qualifier,source_agency,source_observation_id,"
        "source_artifact_path,source_artifact_sha256,semantic_contract,"
        "source_provenance FROM economic_event_feature_release_actual "
        "ORDER BY economic_event_id;", pqxx::params{snapshotId});
    transaction.exec(
        "INSERT INTO economic_calendar_snapshot_first_release_actual("
        "economic_calendar_snapshot_id,economic_event_id,provenance_state,"
        "selection_reason,observation_count,observed_semantic_value_count,"
        "earliest_candidate_count,possible_first_semantic_value_count,"
        "first_release_observation_id,first_release_value_kind,"
        "first_release_value_low,first_release_value_high,first_release_unit,"
        "first_release_scale,first_release_qualifier,proven_available_at,"
        "first_release_source,first_release_source_native_event_id,"
        "first_release_source_observation_id,first_release_evidence_key) "
        "SELECT $1,economic_event_id,provenance_state,selection_reason,"
        "observation_count,observed_semantic_value_count,"
        "earliest_candidate_count,possible_first_semantic_value_count,"
        "first_release_observation_id,first_release_value_kind,"
        "first_release_value_low,first_release_value_high,first_release_unit,"
        "first_release_scale,first_release_qualifier,proven_available_at,"
        "first_release_source,first_release_source_native_event_id,"
        "first_release_source_observation_id,first_release_evidence_key "
        "FROM economic_event_first_release_actual ORDER BY economic_event_id;",
        pqxx::params{snapshotId});

    EconomicCalendarSnapshotReport stored = BuildStoredSnapshotReport(
        transaction, snapshotId);
    RequireSameReport(live, stored);
    transaction.exec(
        "UPDATE economic_calendar_snapshot SET snapshot_state='finalized',"
        "finalized_at=clock_timestamp(),canonical_event_count=$2,"
        "selected_consensus_count=$3,release_actual_count=$4,"
        "proven_first_release_actual_count=$5,"
        "provenance_unavailable_count=$6,ambiguous_first_release_count=$7,"
        "source_family_counts=$8::jsonb "
        "WHERE economic_calendar_snapshot_id=$1 AND snapshot_state='creating';",
        pqxx::params{snapshotId, stored.canonicalEventCount,
                     stored.selectedConsensusCount, stored.releaseActualCount,
                     stored.provenFirstReleaseActualCount,
                     stored.provenanceUnavailableCount,
                     stored.ambiguousFirstReleaseCount,
                     stored.sourceFamilyCountsJson});
    stored.dryRun = false;
    return stored;
}

std::optional<EconomicCalendarSnapshotIdentity>
LoadExperimentEconomicCalendarSnapshot(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT economic_calendar_snapshot_id,economic_calendar_snapshot_hash "
        "FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (rows.size() != 1)
        throw std::runtime_error(
            "experiment_not_found_for_economic_calendar_snapshot");
    if (rows.one_row()[0].is_null() && rows.one_row()[1].is_null())
        return std::nullopt;
    if (rows.one_row()[0].is_null() || rows.one_row()[1].is_null())
        throw std::runtime_error(
            "experiment_economic_calendar_snapshot_identity_incomplete");
    EconomicCalendarSnapshotIdentity identity{
        rows.one_row()[0].as<long long>(),
        rows.one_row()[1].as<std::string>()};
    (void)InspectEconomicCalendarSnapshot(transaction, identity);
    return identity;
}

std::optional<EconomicCalendarSnapshotIdentity>
LoadModelEconomicCalendarSnapshot(
    pqxx::transaction_base& transaction,
    long long modelId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT m.economic_calendar_snapshot_id,"
        "m.economic_calendar_snapshot_hash,e.economic_calendar_snapshot_id,"
        "e.economic_calendar_snapshot_hash FROM model m LEFT JOIN experiment e "
        "ON e.experiment_id=m.experiment_id WHERE m.model_id=$1;",
        pqxx::params{modelId});
    if (rows.size() != 1)
        throw std::runtime_error(
            "model_not_found_for_economic_calendar_snapshot");
    const pqxx::row row = rows.one_row();
    const bool modelNull = row[0].is_null() && row[1].is_null();
    const bool experimentNull = row[2].is_null() && row[3].is_null();
    if (row[0].is_null() != row[1].is_null() ||
        row[2].is_null() != row[3].is_null())
        throw std::runtime_error(
            "economic_calendar_snapshot_identity_incomplete");
    if (modelNull && experimentNull) return std::nullopt;
    if (modelNull != experimentNull ||
        row[0].as<long long>() != row[2].as<long long>() ||
        row[1].as<std::string>() != row[3].as<std::string>())
        throw std::runtime_error(
            "model_experiment_economic_calendar_snapshot_mismatch");
    EconomicCalendarSnapshotIdentity identity{
        row[0].as<long long>(), row[1].as<std::string>()};
    (void)InspectEconomicCalendarSnapshot(transaction, identity);
    return identity;
}


std::vector<EconomicEvent> LoadEconomicEvents(
    pqxx::transaction_base& transaction,
    const std::string& currency,
    const std::string& startUtc,
    const std::string& endUtc)
{
    ValidateCurrency(currency);

    //
    // Half-open range:
    //
    //     start <= event < end
    //
    // This matches the interval convention used by the importer and avoids
    // overlap when adjacent windows are queried.
    //
    // Both the numeric epoch representation and the UTC text representation
    // are derived explicitly from timestamptz. Therefore neither result
    // depends on the active PostgreSQL session timezone.
    //
    const pqxx::result rows = transaction.exec(
        "SELECT " + EconomicEventProjection("e", "c", "a", "p", "f") +
        "FROM economic_event e "
        "LEFT JOIN economic_event_selected_consensus c "
        "USING (economic_event_id) "
        "LEFT JOIN economic_event_feature_release_actual a ON "
        "a.economic_event_id = e.economic_event_id "
        "AND a.available_at < $3::timestamptz "
        "JOIN economic_event_first_release_actual f ON "
        "f.economic_event_id = e.economic_event_id "
        "LEFT JOIN economic_event_first_release_actual_at($3::timestamptz) p "
        "ON p.economic_event_id = e.economic_event_id "
        "WHERE e.currency = $1 "
        "AND e.event_timestamp_utc >= $2::timestamptz "
        "AND e.event_timestamp_utc < $3::timestamptz "
        "ORDER BY "
        "e.event_timestamp_utc ASC, "
        "e.economic_event_id ASC;",
        pqxx::params{
            currency,
            startUtc,
            endUtc});

    return MapEconomicEvents(rows);
}


std::vector<EconomicEvent> LoadEconomicEventsForFeatureRange(
    pqxx::transaction_base& transaction,
    const std::string& currency,
    const std::string& startUtc,
    const std::string& endUtc,
    const std::optional<EconomicCalendarSnapshotIdentity>& snapshot)
{
    ValidateCurrency(currency);

    if (snapshot)
    {
        (void)InspectEconomicCalendarSnapshot(transaction, *snapshot);
        const pqxx::result sourceAvailability = transaction.exec(
            "SELECT 1 FROM economic_calendar_snapshot_event "
            "WHERE economic_calendar_snapshot_id=$1 AND currency=$2 LIMIT 1;",
            pqxx::params{snapshot->snapshotId, currency});
        if (sourceAvailability.empty())
        {
            throw std::runtime_error(
                "economic_event_snapshot_feature_source_history_unavailable:" +
                currency);
        }

        const pqxx::result rows = transaction.exec(
            "WITH prior_canonical_stream AS ("
            "SELECT DISTINCT ON (source_agency,event_family) economic_event_id "
            "FROM economic_calendar_snapshot_event "
            "WHERE economic_calendar_snapshot_id=$1 AND currency=$2 "
            "AND event_timestamp_utc<$3::timestamptz "
            "ORDER BY source_agency,event_family,event_timestamp_utc DESC,"
            "canonical_event_order DESC),selected_event AS ("
            "SELECT e.* FROM economic_calendar_snapshot_event e "
            "JOIN prior_canonical_stream p USING(economic_event_id) "
            "WHERE e.economic_calendar_snapshot_id=$1 UNION ALL "
            "SELECT e.* FROM economic_calendar_snapshot_event e "
            "WHERE e.economic_calendar_snapshot_id=$1 AND e.currency=$2 "
            "AND e.event_timestamp_utc>=$3::timestamptz "
            "AND e.event_timestamp_utc<=$4::timestamptz),pit_actual AS ("
            "SELECT economic_event_id,first_release_value_kind "
            "AS first_release_actual_value_kind,first_release_value_low "
            "AS first_release_actual_value_low,first_release_value_high "
            "AS first_release_actual_value_high,first_release_unit "
            "AS first_release_actual_unit,first_release_scale "
            "AS first_release_actual_scale,first_release_qualifier "
            "AS first_release_actual_qualifier,proven_available_at,"
            "provenance_state,first_release_source AS source_name,"
            "first_release_source_native_event_id AS source_native_event_id,"
            "first_release_source_observation_id AS source_observation_id,"
            "first_release_evidence_key AS evidence_key "
            "FROM economic_calendar_snapshot_first_release_actual "
            "WHERE economic_calendar_snapshot_id=$1 "
            "AND provenance_state='proven_first_release' "
            "AND proven_available_at<=$4::timestamptz) SELECT " +
            EconomicEventProjection(
                "e", "c", "a", "p", "f", "e.canonical_event_order") +
            "FROM selected_event e LEFT JOIN "
            "economic_calendar_snapshot_consensus c ON "
            "c.economic_calendar_snapshot_id=$1 AND "
            "c.economic_event_id=e.economic_event_id LEFT JOIN "
            "economic_calendar_snapshot_release_actual a ON "
            "a.economic_calendar_snapshot_id=$1 AND "
            "a.economic_event_id=e.economic_event_id "
            "AND a.available_at<$4::timestamptz JOIN "
            "economic_calendar_snapshot_first_release_actual f ON "
            "f.economic_calendar_snapshot_id=$1 AND "
            "f.economic_event_id=e.economic_event_id LEFT JOIN pit_actual p "
            "ON p.economic_event_id=e.economic_event_id "
            "ORDER BY e.event_timestamp_utc ASC,"
            "e.canonical_event_order ASC;",
            pqxx::params{snapshot->snapshotId, currency, startUtc, endUtc});
        return MapEconomicEvents(rows);
    }

    // An event-free interval is a valid feature state, but an entirely absent
    // currency corpus is not. Fail before constructing Tensor rows so a
    // missing/unloaded source cannot masquerade as legitimate zero-valued
    // economic-event features. Missing relations, views, privileges, and query
    // failures continue to surface as their original database errors.
    const pqxx::result sourceAvailability = transaction.exec(
        "SELECT 1 FROM economic_event WHERE currency = $1 LIMIT 1;",
        pqxx::params{currency});
    if (sourceAvailability.empty())
    {
        throw std::runtime_error(
            "economic_event_feature_source_history_unavailable:" + currency);
    }

    // DISTINCT ON seeds one latest row for every authoritative canonical
    // stream. Several canonical streams may map to one model family; retaining
    // each stream's latest prior row lets the shared C++ mapper select the
    // truly latest model-family timestamp without duplicating that mapping in
    // SQL. The disjoint UNION ALL then includes every event required while
    // target bars are processed.  The release-actual join is also bounded by
    // the range's information upper bound.  This prevents an authoritative
    // observation that was not yet knowable anywhere in the requested range
    // from crossing the repository boundary.  The chronological feature
    // engine retains the same strict check for each individual completed bar.
    const pqxx::result rows = transaction.exec(
        "WITH prior_canonical_stream AS ("
        "SELECT DISTINCT ON (source_agency, event_family) "
        "economic_event_id "
        "FROM economic_event "
        "WHERE currency = $1 "
        "AND event_timestamp_utc < $2::timestamptz "
        "ORDER BY source_agency, event_family, "
        "event_timestamp_utc DESC, economic_event_id DESC"
        "), selected_event AS ("
        "SELECT e.* FROM economic_event e "
        "JOIN prior_canonical_stream p USING (economic_event_id) "
        "UNION ALL "
        "SELECT e.* FROM economic_event e "
        "WHERE e.currency = $1 "
        "AND e.event_timestamp_utc >= $2::timestamptz "
        "AND e.event_timestamp_utc <= $3::timestamptz"
        ") SELECT " + EconomicEventProjection("e", "c", "a", "p", "f") +
        "FROM selected_event e "
        "LEFT JOIN economic_event_selected_consensus c "
        "USING (economic_event_id) "
        "LEFT JOIN economic_event_feature_release_actual a ON "
        "a.economic_event_id = e.economic_event_id "
        "AND a.available_at < $3::timestamptz "
        "JOIN economic_event_first_release_actual f ON "
        "f.economic_event_id = e.economic_event_id "
        "LEFT JOIN economic_event_first_release_actual_at($3::timestamptz) p "
        "ON p.economic_event_id = e.economic_event_id "
        "ORDER BY e.event_timestamp_utc ASC, e.economic_event_id ASC;",
        pqxx::params{
            currency,
            startUtc,
            endUtc});

    return MapEconomicEvents(rows);
}

} // namespace EA::EconomicCalendar
