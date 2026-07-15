#include "ExperimentRecommendationRepository.hpp"

#include "CanonicalSymbol.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row, const char* column)
{
    const pqxx::field field = row[column];
    if (field.is_null()) return std::nullopt;
    return field.as<Value>();
}

bool TableExists(pqxx::transaction_base& transaction, const char* table)
{
    return !transaction.exec(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = $1 LIMIT 1;",
        pqxx::params{table}).empty();
}

ExperimentInvocationConfiguration MapExperimentInvocation(
    const pqxx::row& row,
    const std::string& prefix)
{
    const auto name = [&](const char* suffix) { return prefix + suffix; };
    if (!row[name("date_mapping_valid")].as<bool>())
        throw std::invalid_argument("ambiguous_experiment_date_mapping");

    ExperimentInvocationConfiguration invocation;
    invocation.configuration.symbol =
        row[name("symbol")].as<std::string>();
    invocation.configuration.predictionHorizon =
        row[name("prediction_horizon")].as<int>();
    invocation.configuration.labelThreshold =
        row[name("label_threshold")].as<double>();
    invocation.configuration.coreLrMult =
        OptionalValue<double>(row, name("core_lr_mult").c_str());
    invocation.configuration.headLrMult =
        OptionalValue<double>(row, name("head_lr_mult").c_str());
    invocation.configuration.targetEpochs =
        row[name("target_epochs")].as<int>();
    invocation.configuration.trainStartDate =
        row[name("train_start_date")].as<std::string>();
    invocation.configuration.trainEndDate =
        row[name("train_end_date")].as<std::string>();
    invocation.configuration.inferStartDate =
        OptionalValue<std::string>(row, name("infer_start_date").c_str());
    invocation.configuration.inferEndDate =
        OptionalValue<std::string>(row, name("infer_end_date").c_str());
    invocation.checkpointInterval =
        row[name("checkpoint_interval")].as<int>();
    invocation.resumeModelId =
        OptionalValue<long long>(row, name("resume_model_id").c_str());
    return BuildRecommendationInvocationIdentity(invocation).invocation;
}

RecommendationScanCounters MapScanCounters(const pqxx::row& row)
{
    RecommendationScanCounters counters;
    counters.sourcesScanned = row["sources_scanned"].as<int>();
    counters.sourcesEligible = row["sources_eligible"].as<int>();
    counters.sourcesSkipped = row["sources_skipped"].as<int>();
    counters.candidatesGenerated = row["candidates_generated"].as<int>();
    counters.candidatesRejected = row["candidates_rejected"].as<int>();
    counters.duplicatesExistingExperiment =
        row["duplicates_existing_experiment"].as<int>();
    counters.duplicatesTerminalExperiment =
        row["duplicates_terminal_experiment"].as<int>();
    counters.duplicatesActiveRecommendation =
        row["duplicates_active_recommendation"].as<int>();
    counters.duplicatesHistoricalRecommendation =
        row["duplicates_historical_recommendation"].as<int>();
    counters.hashCollisions = row["hash_collisions"].as<int>();
    counters.recommendationsCreated =
        row["recommendations_created"].as<int>();
    counters.recommendationsAlreadyExisting =
        row["recommendations_already_existing"].as<int>();
    counters.persistenceErrors = row["persistence_errors"].as<int>();
    return counters;
}

PersistedRecommendationSummary MapRecommendationSummary(const pqxx::row& row)
{
    PersistedRecommendationSummary summary;
    summary.recommendationId = row["recommendation_id"].as<long long>();
    summary.recommendationScanId =
        row["recommendation_scan_id"].as<long long>();
    summary.status = row["status"].as<std::string>();
    summary.sourceExperimentId = row["source_experiment_id"].as<long long>();
    summary.sourceModelId = OptionalValue<long long>(row, "source_model_id");
    summary.sourceAnalysisId = OptionalValue<long long>(row, "source_analysis_id");
    summary.sourceSymbol = row["source_symbol"].as<std::string>();
    summary.sourcePredictionHorizon =
        row["source_prediction_horizon"].as<int>();
    summary.changedParameter = row["changed_parameter"].as<std::string>();
    summary.sourceValueCanonical =
        row["source_value_canonical"].as<std::string>();
    summary.proposedValueCanonical =
        row["proposed_value_canonical"].as<std::string>();
    summary.semanticHash = row["semantic_hash"].as<std::string>();
    summary.invocationHash = row["invocation_hash"].as<std::string>();
    summary.policyHash = row["policy_hash"].as<std::string>();
    summary.generationOrdinal = row["generation_ordinal"].as<int>();
    summary.structuralRank = row["structural_rank"].as<int>();
    summary.reason = row["reason"].as<std::string>();
    summary.createdAt = row["created_at"].as<std::string>();
    return summary;
}

PersistedRecommendationScanSummary MapScanSummary(const pqxx::row& row)
{
    PersistedRecommendationScanSummary summary;
    summary.recommendationScanId =
        row["recommendation_scan_id"].as<long long>();
    summary.status = row["status"].as<std::string>();
    summary.policyHash = row["policy_hash"].as<std::string>();
    summary.policyVersion = row["policy_version"].as<int>();
    summary.symbolFilter = OptionalValue<std::string>(row, "symbol_filter");
    summary.horizonFilter = OptionalValue<int>(row, "horizon_filter");
    summary.sourceExperimentFilter =
        OptionalValue<long long>(row, "source_experiment_filter");
    summary.requestedMaximum = OptionalValue<int>(row, "requested_maximum");
    summary.counters = MapScanCounters(row);
    summary.startedAt = row["started_at"].as<std::string>();
    summary.completedAt = OptionalValue<std::string>(row, "completed_at");
    summary.errorMessage = OptionalValue<std::string>(row, "error_message");
    return summary;
}

std::string ScanCounterAssignments()
{
    return
        "sources_scanned=$3, sources_eligible=$4, sources_skipped=$5, "
        "candidates_generated=$6, candidates_rejected=$7, "
        "duplicates_existing_experiment=$8, duplicates_terminal_experiment=$9, "
        "duplicates_active_recommendation=$10, duplicates_historical_recommendation=$11, "
        "hash_collisions=$12, recommendations_created=$13, "
        "recommendations_already_existing=$14, persistence_errors=$15";
}

pqxx::params ScanCompletionParams(
    long long scanId,
    const std::string& errorMessage,
    const RecommendationScanCounters& counters)
{
    return pqxx::params{
        scanId, errorMessage, counters.sourcesScanned,
        counters.sourcesEligible, counters.sourcesSkipped,
        counters.candidatesGenerated, counters.candidatesRejected,
        counters.duplicatesExistingExperiment,
        counters.duplicatesTerminalExperiment,
        counters.duplicatesActiveRecommendation,
        counters.duplicatesHistoricalRecommendation,
        counters.hashCollisions, counters.recommendationsCreated,
        counters.recommendationsAlreadyExisting, counters.persistenceErrors};
}

} // namespace

bool RecommendationSchemaExists(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return TableExists(transaction, "experiment_recommendation_scan") &&
           TableExists(transaction, "experiment_recommendation");
}

long long BeginRecommendationScan(
    pqxx::connection& connection,
    const RecommendationScanRequest& request)
{
    const std::string policyCanonical =
        RecommendationPolicyCanonicalText(request.policy);
    const std::string policyHash = RecommendationPolicyHash(request.policy);
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const pqxx::result rows = transaction.exec(
        "INSERT INTO experiment_recommendation_scan ("
        "status, policy_canonical, policy_hash, policy_version, symbol_filter, "
        "horizon_filter, source_experiment_filter, requested_maximum) "
        "VALUES ('running',$1,$2,$3,$4,$5,$6,$7) "
        "RETURNING recommendation_scan_id;",
        pqxx::params{policyCanonical, policyHash, request.policy.policyVersion,
                     request.filters.symbol, request.filters.predictionHorizon,
                     request.filters.sourceExperimentId,
                     request.requestedMaximum});
    const long long scanId = rows.one_row()[0].as<long long>();
    transaction.commit();
    return scanId;
}

std::vector<RecommendationSourceLoadResult> LoadRecommendationSources(
    pqxx::connection& connection,
    const RecommendationSourceFilters& filters)
{
    pqxx::read_transaction transaction{connection};
    return LoadRecommendationSources(transaction, filters);
}

std::vector<RecommendationSourceLoadResult> LoadRecommendationSources(
    pqxx::transaction_base& transaction,
    const RecommendationSourceFilters& filters)
{
    const pqxx::result rows = transaction.exec(
        "SELECT e.experiment_id AS experiment_id, "
        "e.symbol AS source_symbol, e.prediction_horizon AS source_prediction_horizon, "
        "e.c_next_threshold AS source_label_threshold, "
        "e.core_lr_mult AS source_core_lr_mult, e.head_lr_mult AS source_head_lr_mult, "
        "e.target_epochs AS source_target_epochs, e.checkpoint_interval AS source_checkpoint_interval, "
        "to_char(e.train_start AT TIME ZONE 'America/Chicago','YYYY-MM-DD') AS source_train_start_date, "
        "to_char(e.train_end AT TIME ZONE 'America/Chicago','YYYY-MM-DD') AS source_train_end_date, "
        "CASE WHEN e.infer_start IS NULL THEN NULL ELSE to_char(e.infer_start AT TIME ZONE 'America/Chicago','YYYY-MM-DD') END AS source_infer_start_date, "
        "CASE WHEN e.infer_end IS NULL THEN NULL ELSE to_char(e.infer_end AT TIME ZONE 'America/Chicago','YYYY-MM-DD') END AS source_infer_end_date, "
        "e.resume_model_id AS source_resume_model_id, e.last_model_id AS source_model_id, "
        "((e.train_start AT TIME ZONE 'America/Chicago') = date_trunc('day', e.train_start AT TIME ZONE 'America/Chicago') "
        " AND (e.train_end AT TIME ZONE 'America/Chicago') = date_trunc('day', e.train_end AT TIME ZONE 'America/Chicago') "
        " AND (e.infer_start IS NULL OR (e.infer_start AT TIME ZONE 'America/Chicago') = date_trunc('day', e.infer_start AT TIME ZONE 'America/Chicago')) "
        " AND (e.infer_end IS NULL OR (e.infer_end AT TIME ZONE 'America/Chicago') = date_trunc('day', e.infer_end AT TIME ZONE 'America/Chicago'))) AS source_date_mapping_valid, "
        "a.analysis_id AS source_analysis_id, a.analysis_status AS source_analysis_status, "
        "a.leader_score AS source_leader_score, a.infer_accuracy AS source_infer_accuracy, "
        "CASE WHEN COALESCE(a.pred_down_count,0)+COALESCE(a.pred_neutral_count,0)+COALESCE(a.pred_up_count,0) > 0 "
        " THEN a.pred_neutral_count::double precision / "
        "      (COALESCE(a.pred_down_count,0)+COALESCE(a.pred_neutral_count,0)+COALESCE(a.pred_up_count,0))::double precision "
        " ELSE NULL END AS source_predicted_neutral_proportion, "
        "COALESCE(a.pred_down_count,0)+COALESCE(a.pred_neutral_count,0)+COALESCE(a.pred_up_count,0) AS source_evidence_count "
        "FROM experiment e "
        "LEFT JOIN experiment_analysis_result a "
        " ON a.experiment_id=e.experiment_id "
        "AND a.model_id=e.last_model_id "
        "AND COALESCE(a.analysis_scope,'final')='final' "
        "WHERE e.status='completed' AND e.phase='done' "
        "AND ($1::text IS NULL OR lower(btrim(e.symbol))=$1) "
        "AND ($2::integer IS NULL OR e.prediction_horizon=$2) "
        "AND ($3::bigint IS NULL OR e.experiment_id=$3) "
        "ORDER BY e.experiment_id ASC;",
        pqxx::params{filters.symbol, filters.predictionHorizon,
                     filters.sourceExperimentId});

    std::vector<RecommendationSourceLoadResult> results;
    results.reserve(rows.size());
    for (const pqxx::row& row : rows)
    {
        RecommendationSourceLoadResult loaded;
        loaded.experimentId = row["experiment_id"].as<long long>();
        if (row["source_model_id"].is_null())
        {
            loaded.skipReason = "missing_final_model";
            results.push_back(std::move(loaded));
            continue;
        }
        if (row["source_analysis_id"].is_null())
        {
            loaded.skipReason = "missing_final_analysis";
            results.push_back(std::move(loaded));
            continue;
        }
        if (row["source_analysis_status"].as<std::string>() != "completed")
        {
            loaded.skipReason = "final_analysis_not_completed";
            results.push_back(std::move(loaded));
            continue;
        }
        try
        {
            RecommendationSource source;
            source.experimentId = loaded.experimentId;
            source.modelId = OptionalValue<long long>(row, "source_model_id");
            source.analysisId = OptionalValue<long long>(row, "source_analysis_id");
            source.invocation = MapExperimentInvocation(row, "source_");
            source.leaderScore = OptionalValue<double>(row, "source_leader_score");
            source.inferenceAccuracy = OptionalValue<double>(row, "source_infer_accuracy");
            source.predictedNeutralProportion =
                OptionalValue<double>(row, "source_predicted_neutral_proportion");
            source.evidenceCount = row["source_evidence_count"].as<long long>();
            loaded.source = std::move(source);
        }
        catch (const std::exception& error)
        {
            loaded.skipReason = std::string{"source_mapping_error:"} + error.what();
        }
        results.push_back(std::move(loaded));
    }
    return results;
}

ExperimentDuplicateMatch FindExperimentDuplicate(
    pqxx::transaction_base& transaction,
    const RecommendationCandidateIdentity& semanticIdentity,
    const RecommendationInvocationIdentity& invocationIdentity,
    const RecommendationPolicy& policy)
{
    const EffectiveExperimentConfiguration& candidate =
        semanticIdentity.configuration;
    const pqxx::result rows = transaction.exec(
        "SELECT e.experiment_id AS candidate_experiment_id, e.status AS candidate_status, "
        "e.symbol AS candidate_symbol, e.prediction_horizon AS candidate_prediction_horizon, "
        "e.c_next_threshold AS candidate_label_threshold, e.core_lr_mult AS candidate_core_lr_mult, "
        "e.head_lr_mult AS candidate_head_lr_mult, e.target_epochs AS candidate_target_epochs, "
        "e.checkpoint_interval AS candidate_checkpoint_interval, "
        "to_char(e.train_start AT TIME ZONE 'America/Chicago','YYYY-MM-DD') AS candidate_train_start_date, "
        "to_char(e.train_end AT TIME ZONE 'America/Chicago','YYYY-MM-DD') AS candidate_train_end_date, "
        "CASE WHEN e.infer_start IS NULL THEN NULL ELSE to_char(e.infer_start AT TIME ZONE 'America/Chicago','YYYY-MM-DD') END AS candidate_infer_start_date, "
        "CASE WHEN e.infer_end IS NULL THEN NULL ELSE to_char(e.infer_end AT TIME ZONE 'America/Chicago','YYYY-MM-DD') END AS candidate_infer_end_date, "
        "e.resume_model_id AS candidate_resume_model_id, "
        "((e.train_start AT TIME ZONE 'America/Chicago') = date_trunc('day', e.train_start AT TIME ZONE 'America/Chicago') "
        " AND (e.train_end AT TIME ZONE 'America/Chicago') = date_trunc('day', e.train_end AT TIME ZONE 'America/Chicago') "
        " AND (e.infer_start IS NULL OR (e.infer_start AT TIME ZONE 'America/Chicago') = date_trunc('day', e.infer_start AT TIME ZONE 'America/Chicago')) "
        " AND (e.infer_end IS NULL OR (e.infer_end AT TIME ZONE 'America/Chicago') = date_trunc('day', e.infer_end AT TIME ZONE 'America/Chicago'))) AS candidate_date_mapping_valid "
        "FROM experiment e "
        "WHERE lower(btrim(e.symbol))=$1 AND e.prediction_horizon=$2 "
        "AND e.target_epochs=$3 "
        "ORDER BY CASE WHEN e.status IN ('failed','cancelled') THEN 1 ELSE 0 END, e.experiment_id ASC;",
        pqxx::params{candidate.symbol, candidate.predictionHorizon,
                     candidate.targetEpochs});

    std::optional<ExperimentDuplicateMatch> terminal;
    for (const pqxx::row& row : rows)
    {
        try
        {
            const ExperimentInvocationConfiguration existingInvocation =
                MapExperimentInvocation(row, "candidate_");
            const RecommendationCandidateIdentity existingSemantic =
                BuildRecommendationCandidateIdentity(
                    existingInvocation.configuration);
            if (existingSemantic.canonicalText != semanticIdentity.canonicalText)
                continue;
            const RecommendationInvocationIdentity existingExact =
                BuildRecommendationInvocationIdentity(existingInvocation);
            ExperimentDuplicateMatch match;
            match.experimentId = row["candidate_experiment_id"].as<long long>();
            match.experimentStatus = row["candidate_status"].as<std::string>();
            match.semanticExactMatch = true;
            match.invocationExactMatch =
                existingExact.canonicalText == invocationIdentity.canonicalText;
            if (match.experimentStatus == "failed" ||
                match.experimentStatus == "cancelled")
            {
                match.type = policy.terminalExperimentsAreDuplicates
                    ? RecommendationDuplicateType::excludedTerminalExperiment
                    : RecommendationDuplicateType::noDuplicate;
                if (!terminal) terminal = match;
                continue;
            }
            match.type = RecommendationDuplicateType::existingExperiment;
            return match;
        }
        catch (const std::exception& error)
        {
            throw std::runtime_error(
                "experiment_duplicate_mapping_error:experiment_id=" +
                std::to_string(
                    row["candidate_experiment_id"].as<long long>()) +
                ":" + error.what());
        }
    }
    return terminal.value_or(ExperimentDuplicateMatch{});
}

RecommendationDuplicateMatch FindRecommendationDuplicate(
    pqxx::transaction_base& transaction,
    const RecommendationCandidateIdentity& semanticIdentity,
    const RecommendationInvocationIdentity& invocationIdentity,
    const RecommendationPolicy& policy)
{
    const std::string policyCanonical = RecommendationPolicyCanonicalText(policy);
    const pqxx::result rows = transaction.exec(
        "SELECT recommendation_id, status, semantic_configuration_canonical, "
        "semantic_hash, invocation_configuration_canonical, invocation_hash, "
        "policy_canonical, policy_hash "
        "FROM experiment_recommendation "
        "WHERE semantic_hash=$1 OR semantic_configuration_canonical=$2 "
        "ORDER BY CASE WHEN status IN ('proposed','approved') THEN 0 ELSE 1 END, "
        "recommendation_id ASC;",
        pqxx::params{semanticIdentity.hash, semanticIdentity.canonicalText});

    RecommendationDuplicateMatch match;
    const std::string policyHash = RecommendationPolicyHash(policy);
    for (const pqxx::row& row : rows)
    {
        const std::string existingCanonical =
            row["semantic_configuration_canonical"].as<std::string>();
        const std::string existingHash = row["semantic_hash"].as<std::string>();
        if (existingHash == semanticIdentity.hash &&
            existingCanonical != semanticIdentity.canonicalText)
        {
            match.hashCollisions.push_back({
                "semantic", RecommendationCandidateHashCollision{
                    existingHash, existingCanonical,
                    semanticIdentity.canonicalText}});
        }
        const std::string existingInvocationCanonical =
            row["invocation_configuration_canonical"].as<std::string>();
        const std::string existingInvocationHash =
            row["invocation_hash"].as<std::string>();
        if (existingInvocationHash == invocationIdentity.hash &&
            existingInvocationCanonical != invocationIdentity.canonicalText)
        {
            match.hashCollisions.push_back({
                "invocation", RecommendationCandidateHashCollision{
                    existingInvocationHash, existingInvocationCanonical,
                    invocationIdentity.canonicalText}});
        }
        const std::string existingPolicyCanonical =
            row["policy_canonical"].as<std::string>();
        const std::string existingPolicyHash =
            row["policy_hash"].as<std::string>();
        if (existingPolicyHash == policyHash &&
            existingPolicyCanonical != policyCanonical)
        {
            match.hashCollisions.push_back({
                "policy", RecommendationCandidateHashCollision{
                    existingPolicyHash, existingPolicyCanonical,
                    policyCanonical}});
        }
        const bool semanticExact =
            existingCanonical == semanticIdentity.canonicalText;
        const bool policyExact =
            existingPolicyCanonical == policyCanonical;
        if (!semanticExact || !policyExact || match.recommendationId)
            continue;
        match.recommendationId = row["recommendation_id"].as<long long>();
        match.status = row["status"].as<std::string>();
        match.semanticExactMatch = true;
        match.policyExactMatch = true;
        match.invocationExactMatch =
            existingInvocationCanonical == invocationIdentity.canonicalText;
        match.kind = (match.status == "proposed" || match.status == "approved")
            ? PersistedRecommendationMatchKind::active
            : PersistedRecommendationMatchKind::historical;
    }
    return match;
}

RecommendationPersistResult PersistRecommendationIdempotently(
    pqxx::connection& connection,
    const RecommendationPersistenceRequest& request)
{
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const std::string policyCanonical =
        RecommendationPolicyCanonicalText(request.policy);
    const std::string lockIdentity =
        request.candidate.semanticIdentity.canonicalText + "\n" + policyCanonical;
    transaction.exec(
        "SELECT pg_advisory_xact_lock(hashtextextended($1,0));",
        pqxx::params{lockIdentity});

    RecommendationPersistResult result;
    result.experimentMatch = FindExperimentDuplicate(
        transaction, request.candidate.semanticIdentity,
        request.candidate.invocationIdentity, request.policy);
    if (result.experimentMatch.type == RecommendationDuplicateType::existingExperiment)
    {
        result.outcome = RecommendationPersistOutcome::existingExperiment;
        transaction.commit();
        return result;
    }
    if (result.experimentMatch.type ==
        RecommendationDuplicateType::excludedTerminalExperiment)
    {
        result.outcome = RecommendationPersistOutcome::terminalExperiment;
        transaction.commit();
        return result;
    }

    result.recommendationMatch = FindRecommendationDuplicate(
        transaction, request.candidate.semanticIdentity,
        request.candidate.invocationIdentity, request.policy);
    if (result.recommendationMatch.kind == PersistedRecommendationMatchKind::active)
    {
        result.outcome = RecommendationPersistOutcome::activeRecommendation;
        result.recommendationId = result.recommendationMatch.recommendationId;
        transaction.commit();
        return result;
    }
    if (result.recommendationMatch.kind == PersistedRecommendationMatchKind::historical)
    {
        result.outcome = RecommendationPersistOutcome::historicalRecommendation;
        result.recommendationId = result.recommendationMatch.recommendationId;
        transaction.commit();
        return result;
    }

    if (!request.source.leaderScore || !request.source.inferenceAccuracy ||
        request.source.evidenceCount <= 0)
        throw std::invalid_argument("recommendation_persistence_source_evidence_invalid");

    const std::string sourceValue =
        CanonicalRecommendationDouble(request.candidate.sourceValue);
    const std::string proposedValue =
        CanonicalRecommendationDouble(request.candidate.proposedValue);
    const pqxx::result inserted = transaction.exec(
        "INSERT INTO experiment_recommendation ("
        "recommendation_scan_id,status,source_experiment_id,source_model_id,source_analysis_id,"
        "source_symbol,source_prediction_horizon,source_leader_score,source_infer_accuracy,"
        "source_predicted_neutral_proportion,source_evidence_count,changed_parameter,"
        "source_value_canonical,proposed_value_canonical,absolute_delta,relative_delta,horizon_delta,"
        "semantic_configuration_canonical,semantic_hash,invocation_configuration_canonical,invocation_hash,"
        "policy_canonical,policy_hash,generation_ordinal,structural_rank,duplicate_type,reason) "
        "VALUES ($1,'proposed',$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,'no_duplicate','single_parameter_neighborhood') "
        "ON CONFLICT (semantic_configuration_canonical,policy_canonical) "
        "WHERE status IN ('proposed','approved') "
        "AND semantic_configuration_canonical IS NOT NULL "
        "AND policy_canonical IS NOT NULL DO NOTHING "
        "RETURNING recommendation_id;",
        pqxx::params{
            request.recommendationScanId, request.source.experimentId,
            request.source.modelId, request.source.analysisId,
            request.source.invocation.configuration.symbol,
            request.source.invocation.configuration.predictionHorizon,
            *request.source.leaderScore, *request.source.inferenceAccuracy,
            request.source.predictedNeutralProportion, request.source.evidenceCount,
            RecommendationMutationParameterText(request.candidate.parameter),
            sourceValue, proposedValue, request.candidate.absoluteDelta,
            request.candidate.relativeDelta, request.candidate.horizonDelta,
            request.candidate.semanticIdentity.canonicalText,
            request.candidate.semanticIdentity.hash,
            request.candidate.invocationIdentity.canonicalText,
            request.candidate.invocationIdentity.hash,
            policyCanonical, RecommendationPolicyHash(request.policy),
            request.generationOrdinal, request.structuralRank});
    if (!inserted.empty())
    {
        result.recommendationId = inserted.one_row()[0].as<long long>();
        result.outcome = RecommendationPersistOutcome::created;
    }
    else
    {
        result.recommendationMatch = FindRecommendationDuplicate(
            transaction, request.candidate.semanticIdentity,
            request.candidate.invocationIdentity, request.policy);
        if (result.recommendationMatch.kind !=
                PersistedRecommendationMatchKind::active ||
            !result.recommendationMatch.recommendationId)
        {
            throw std::runtime_error(
                "active_recommendation_conflict_without_matching_row");
        }
        result.recommendationId = result.recommendationMatch.recommendationId;
        result.outcome = RecommendationPersistOutcome::activeRecommendation;
    }
    transaction.commit();
    return result;
}

void CompleteRecommendationScan(
    pqxx::connection& connection,
    long long scanId,
    const RecommendationScanCounters& counters)
{
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const pqxx::result updated = transaction.exec(
        "UPDATE experiment_recommendation_scan SET status='completed', "
        "completed_at=now(), error_message=NULLIF($2::text,''), updated_at=now(), " +
        ScanCounterAssignments() +
        " WHERE recommendation_scan_id=$1 AND status='running';",
        ScanCompletionParams(scanId, "", counters));
    if (updated.affected_rows() != 1)
        throw std::runtime_error("recommendation_scan_not_running");
    transaction.commit();
}

void FailRecommendationScan(
    pqxx::connection& connection,
    long long scanId,
    const RecommendationScanCounters& counters,
    const std::string& errorMessage)
{
    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");
    const std::string persistedError = errorMessage.empty()
        ? "unknown_recommendation_scan_failure"
        : errorMessage;
    const pqxx::result updated = transaction.exec(
        "UPDATE experiment_recommendation_scan SET status='failed', "
        "completed_at=now(), error_message=$2, updated_at=now(), " +
        ScanCounterAssignments() +
        " WHERE recommendation_scan_id=$1 AND status='running';",
        ScanCompletionParams(scanId, persistedError, counters));
    if (updated.affected_rows() != 1)
        throw std::runtime_error("recommendation_scan_not_running");
    transaction.commit();
}

std::vector<PersistedRecommendationSummary> ListRecommendations(
    pqxx::connection& connection,
    const RecommendationListFilters& filters)
{
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT recommendation_id,recommendation_scan_id,status,source_experiment_id,"
        "source_model_id,source_analysis_id,source_symbol,source_prediction_horizon,"
        "changed_parameter,source_value_canonical,proposed_value_canonical,semantic_hash,"
        "invocation_hash,policy_hash,generation_ordinal,structural_rank,reason,created_at::text AS created_at "
        "FROM experiment_recommendation "
        "WHERE recommendation_scan_id IS NOT NULL "
        "AND ($1::text IS NULL OR status=$1) "
        "AND ($2::text IS NULL OR source_symbol=$2) "
        "AND ($3::integer IS NULL OR source_prediction_horizon=$3) "
        "AND ($4::bigint IS NULL OR recommendation_scan_id=$4) "
        "ORDER BY recommendation_id DESC LIMIT $5;",
        pqxx::params{filters.status, filters.symbol, filters.predictionHorizon,
                     filters.recommendationScanId, filters.limit});
    std::vector<PersistedRecommendationSummary> summaries;
    summaries.reserve(rows.size());
    for (const pqxx::row& row : rows)
        summaries.push_back(MapRecommendationSummary(row));
    return summaries;
}

std::optional<PersistedRecommendationDetail> FindRecommendation(
    pqxx::connection& connection,
    long long recommendationId)
{
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT recommendation_id,recommendation_scan_id,status,source_experiment_id,"
        "source_model_id,source_analysis_id,source_symbol,source_prediction_horizon,"
        "changed_parameter,source_value_canonical,proposed_value_canonical,semantic_hash,"
        "invocation_hash,policy_hash,generation_ordinal,structural_rank,reason,created_at::text AS created_at,"
        "source_leader_score,source_infer_accuracy,source_predicted_neutral_proportion,source_evidence_count,"
        "absolute_delta,relative_delta,horizon_delta,semantic_configuration_canonical,"
        "invocation_configuration_canonical,policy_canonical,duplicate_type,matched_experiment_id,"
        "matched_recommendation_id,approved_experiment_id,rejected_at::text AS rejected_at,"
        "rejected_reason,expired_at::text AS expired_at "
        "FROM experiment_recommendation WHERE recommendation_id=$1 "
        "AND recommendation_scan_id IS NOT NULL;",
        pqxx::params{recommendationId});
    if (rows.empty()) return std::nullopt;
    PersistedRecommendationDetail detail;
    static_cast<PersistedRecommendationSummary&>(detail) =
        MapRecommendationSummary(rows.one_row());
    const pqxx::row row = rows.one_row();
    detail.sourceLeaderScore = row["source_leader_score"].as<double>();
    detail.sourceInferAccuracy = row["source_infer_accuracy"].as<double>();
    detail.sourcePredictedNeutralProportion =
        OptionalValue<double>(row, "source_predicted_neutral_proportion");
    detail.sourceEvidenceCount = row["source_evidence_count"].as<long long>();
    detail.absoluteDelta = row["absolute_delta"].as<double>();
    detail.relativeDelta = OptionalValue<double>(row, "relative_delta");
    detail.horizonDelta = OptionalValue<int>(row, "horizon_delta");
    detail.semanticConfigurationCanonical =
        row["semantic_configuration_canonical"].as<std::string>();
    detail.invocationConfigurationCanonical =
        row["invocation_configuration_canonical"].as<std::string>();
    detail.policyCanonical = row["policy_canonical"].as<std::string>();
    detail.duplicateType = row["duplicate_type"].as<std::string>();
    detail.matchedExperimentId = OptionalValue<long long>(row, "matched_experiment_id");
    detail.matchedRecommendationId =
        OptionalValue<long long>(row, "matched_recommendation_id");
    detail.approvedExperimentId =
        OptionalValue<long long>(row, "approved_experiment_id");
    detail.rejectedAt = OptionalValue<std::string>(row, "rejected_at");
    detail.rejectedReason = OptionalValue<std::string>(row, "rejected_reason");
    detail.expiredAt = OptionalValue<std::string>(row, "expired_at");
    return detail;
}

std::vector<PersistedRecommendationScanSummary> ListRecommendationScans(
    pqxx::connection& connection,
    int limit)
{
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT recommendation_scan_id,status,policy_hash,policy_version,symbol_filter,"
        "horizon_filter,source_experiment_filter,requested_maximum,sources_scanned,"
        "sources_eligible,sources_skipped,candidates_generated,candidates_rejected,"
        "duplicates_existing_experiment,duplicates_terminal_experiment,"
        "duplicates_active_recommendation,duplicates_historical_recommendation,"
        "hash_collisions,recommendations_created,recommendations_already_existing,"
        "persistence_errors,started_at::text AS started_at,completed_at::text AS completed_at,error_message "
        "FROM experiment_recommendation_scan ORDER BY recommendation_scan_id DESC LIMIT $1;",
        pqxx::params{limit});
    std::vector<PersistedRecommendationScanSummary> summaries;
    summaries.reserve(rows.size());
    for (const pqxx::row& row : rows)
        summaries.push_back(MapScanSummary(row));
    return summaries;
}

std::optional<PersistedRecommendationScanDetail> FindRecommendationScan(
    pqxx::connection& connection,
    long long scanId)
{
    pqxx::read_transaction transaction{connection};
    const pqxx::result rows = transaction.exec(
        "SELECT recommendation_scan_id,status,policy_hash,policy_version,policy_canonical,"
        "symbol_filter,horizon_filter,source_experiment_filter,requested_maximum,sources_scanned,"
        "sources_eligible,sources_skipped,candidates_generated,candidates_rejected,"
        "duplicates_existing_experiment,duplicates_terminal_experiment,"
        "duplicates_active_recommendation,duplicates_historical_recommendation,"
        "hash_collisions,recommendations_created,recommendations_already_existing,"
        "persistence_errors,started_at::text AS started_at,completed_at::text AS completed_at,error_message "
        "FROM experiment_recommendation_scan WHERE recommendation_scan_id=$1;",
        pqxx::params{scanId});
    if (rows.empty()) return std::nullopt;
    PersistedRecommendationScanDetail detail;
    static_cast<PersistedRecommendationScanSummary&>(detail) =
        MapScanSummary(rows.one_row());
    detail.policyCanonical = rows.one_row()["policy_canonical"].as<std::string>();
    return detail;
}

std::string PersistedRecommendationMatchKindText(
    PersistedRecommendationMatchKind kind)
{
    switch (kind)
    {
        case PersistedRecommendationMatchKind::none: return "none";
        case PersistedRecommendationMatchKind::active: return "active_recommendation";
        case PersistedRecommendationMatchKind::historical: return "historical_recommendation";
    }
    return "invalid_recommendation_match_kind";
}

std::string RecommendationPersistOutcomeText(
    RecommendationPersistOutcome outcome)
{
    switch (outcome)
    {
        case RecommendationPersistOutcome::created: return "created";
        case RecommendationPersistOutcome::existingExperiment: return "existing_experiment";
        case RecommendationPersistOutcome::terminalExperiment: return "excluded_terminal_experiment";
        case RecommendationPersistOutcome::activeRecommendation: return "active_recommendation";
        case RecommendationPersistOutcome::historicalRecommendation: return "historical_recommendation";
    }
    return "invalid_recommendation_persist_outcome";
}

} // namespace EA::ExperimentRecommendation
