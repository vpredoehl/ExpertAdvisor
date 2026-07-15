#include <cassert>
#include <cstdlib>
#include <optional>
#include <string>

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

long long InsertExperiment(pqxx::transaction_base& transaction,
                           const std::string& symbol,
                           const std::string& status,
                           const std::string& phase,
                           const std::string& trainStart,
                           const std::optional<std::string>& inferStart,
                           long long lastModelId = 987654321)
{
    const pqxx::result rows = transaction.exec(
        "INSERT INTO experiment (symbol,prediction_horizon,c_next_threshold,"
        "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start,train_end,infer_start,infer_end,status,phase,last_model_id,"
        "duplicate_nonce) VALUES ($1,12,0.001,1.0,5.0,120,20,"
        "$2::timestamptz,'2025-01-01 00:00:00 America/Chicago',"
        "$3::timestamptz,CASE WHEN $3::text IS NULL THEN NULL ELSE "
        "'2026-01-01 00:00:00 America/Chicago'::timestamptz END,$4,$5,$6,"
        "floor(random()*900000000)::bigint+100000000) RETURNING experiment_id;",
        pqxx::params{symbol, trainStart, inferStart, status, phase,
                     lastModelId});
    return rows.one_row()[0].as<long long>();
}

void InsertFinalAnalysis(pqxx::transaction_base& transaction,
                         long long experimentId,
                         const std::string& status,
                         long long modelId = 987654321)
{
    transaction.exec(
        "INSERT INTO experiment_analysis_result (experiment_id,model_id,"
        "symbol,prediction_horizon,target_epochs,infer_accuracy,leader_score,"
        "pred_down_count,pred_neutral_count,pred_up_count,analysis_status,"
        "analysis_scope) VALUES ($1,$2,'step3_fixture',12,120,0.7,0.8,"
        "20,30,50,$3,'final');",
        pqxx::params{experimentId, modelId, status});
}

const RecommendationSourceLoadResult& FindLoaded(
    const std::vector<RecommendationSourceLoadResult>& values,
    long long id)
{
    for (const auto& value : values)
        if (value.experimentId == id) return value;
    assert(false && "fixture source not returned");
    return values.front();
}

long long InsertRecommendation(
    pqxx::transaction_base& transaction,
    long long scanId,
    const RecommendationSource& source,
    const RecommendationCandidateIdentity& semantic,
    const RecommendationInvocationIdentity& invocation,
    const RecommendationPolicy& policy,
    const std::string& status,
    const std::string& semanticHash)
{
    const pqxx::result rows = transaction.exec(
        "INSERT INTO experiment_recommendation (recommendation_scan_id,status,"
        "source_experiment_id,source_model_id,source_analysis_id,source_symbol,"
        "source_prediction_horizon,source_leader_score,source_infer_accuracy,"
        "source_predicted_neutral_proportion,source_evidence_count,changed_parameter,"
        "source_value_canonical,proposed_value_canonical,absolute_delta,"
        "semantic_configuration_canonical,semantic_hash,"
        "invocation_configuration_canonical,invocation_hash,policy_canonical,"
        "policy_hash,generation_ordinal,structural_rank,duplicate_type,reason,"
        "rejected_at,rejected_reason,expired_at) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'core_lr_mult','1','2',1,"
        "$12,$13,$14,$15,$16,$17,1,1,'no_duplicate','test',"
        "CASE WHEN $2='rejected' THEN now() ELSE NULL END,"
        "CASE WHEN $2='rejected' THEN 'test_rejection' ELSE NULL END,"
        "CASE WHEN $2='expired' THEN now() ELSE NULL END) "
        "RETURNING recommendation_id;",
        pqxx::params{
            scanId, status, source.experimentId, source.modelId,
            source.analysisId, source.invocation.configuration.symbol,
            source.invocation.configuration.predictionHorizon,
            *source.leaderScore, *source.inferenceAccuracy,
            source.predictedNeutralProportion, source.evidenceCount,
            semantic.canonicalText, semanticHash, invocation.canonicalText,
            invocation.hash, RecommendationPolicyCanonicalText(policy),
            RecommendationPolicyHash(policy)});
    return rows.one_row()[0].as<long long>();
}

} // namespace

int main()
{
    const std::string connectionString =
        "hostaddr=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=pqxx dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM");
    pqxx::connection connection{connectionString};
    assert(RecommendationSchemaExists(connection));

    pqxx::work transaction{connection};
    transaction.exec("SET TRANSACTION READ WRITE;");

    const pqxx::result hardeningConstraints = transaction.exec(
        "SELECT conname FROM pg_constraint WHERE conname IN ("
        "'experiment_recommendation_scan_source_filter_positive_check',"
        "'experiment_recommendation_scan_failed_error_nonempty_check',"
        "'experiment_recommendation_status_metadata_symmetric_check');");
    assert(hardeningConstraints.size() == 3);
    bool emptyFailureRejected = false;
    try
    {
        pqxx::subtransaction invalid{transaction, "invalid_failed_scan"};
        invalid.exec(
            "INSERT INTO experiment_recommendation_scan (status,"
            "policy_canonical,policy_hash,policy_version,completed_at,"
            "error_message) VALUES ('failed','test','test',1,now(),'');");
        invalid.commit();
    }
    catch (const pqxx::check_violation&)
    {
        emptyFailureRejected = true;
    }
    assert(emptyFailureRejected);

    const long long validId = InsertExperiment(
        transaction, "step3_fixture", "completed", "done",
        "2010-01-01 00:00:00 America/Chicago",
        std::optional<std::string>{"2025-01-01 00:00:00 America/Chicago"});
    InsertFinalAnalysis(transaction, validId, "completed");

    const long long nullableInferId = InsertExperiment(
        transaction, "step3_null_infer", "completed", "done",
        "2010-01-01 00:00:00 America/Chicago", std::nullopt);
    InsertFinalAnalysis(transaction, nullableInferId, "completed");

    const long long missingAnalysisId = InsertExperiment(
        transaction, "step3_missing_analysis", "completed", "done",
        "2010-01-01 00:00:00 America/Chicago", std::nullopt);
    const long long incompleteAnalysisId = InsertExperiment(
        transaction, "step3_incomplete_analysis", "completed", "done",
        "2010-01-01 00:00:00 America/Chicago", std::nullopt);
    InsertFinalAnalysis(transaction, incompleteAnalysisId, "failed");

    const long long ambiguousDateId = InsertExperiment(
        transaction, "step3_ambiguous_date", "completed", "done",
        "2010-01-01 12:00:00 America/Chicago", std::nullopt);
    InsertFinalAnalysis(transaction, ambiguousDateId, "completed");

    const long long winterDateId = InsertExperiment(
        transaction, "step3_winter_date", "completed", "done",
        "2024-01-15 00:00:00 America/Chicago", std::nullopt);
    InsertFinalAnalysis(transaction, winterDateId, "completed");
    const long long summerDateId = InsertExperiment(
        transaction, "step3_summer_date", "completed", "done",
        "2024-07-15 00:00:00 America/Chicago", std::nullopt);
    InsertFinalAnalysis(transaction, summerDateId, "completed");
    const long long dstDateId = InsertExperiment(
        transaction, "step3_dst_date", "completed", "done",
        "2024-03-10 00:00:00 America/Chicago", std::nullopt);
    InsertFinalAnalysis(transaction, dstDateId, "completed");
    const long long staleAnalysisId = InsertExperiment(
        transaction, "step3_stale_analysis", "completed", "done",
        "2010-01-01 00:00:00 America/Chicago", std::nullopt, 987654322);
    InsertFinalAnalysis(transaction, staleAnalysisId, "completed", 987654321);

    RecommendationSourceFilters filters;
    const auto loaded = LoadRecommendationSources(transaction, filters);
    const auto& valid = FindLoaded(loaded, validId);
    assert(valid.source.has_value());
    assert(valid.source->invocation.configuration.symbol == "step3_fixture");
    assert(valid.source->invocation.configuration.trainStartDate ==
           "2010-01-01");
    assert(valid.source->invocation.configuration.inferStartDate ==
           std::optional<std::string>{"2025-01-01"});
    assert(valid.source->predictedNeutralProportion ==
           std::optional<double>{0.3});
    assert(valid.source->evidenceCount == 100);

    const auto& nullable = FindLoaded(loaded, nullableInferId);
    assert(nullable.source.has_value());
    assert(!nullable.source->invocation.configuration.inferStartDate);
    assert(!nullable.source->invocation.configuration.inferEndDate);
    assert(FindLoaded(loaded, missingAnalysisId).skipReason ==
           "missing_final_analysis");
    assert(FindLoaded(loaded, incompleteAnalysisId).skipReason ==
           "final_analysis_not_completed");
    assert(FindLoaded(loaded, ambiguousDateId).skipReason ==
           "source_mapping_error:ambiguous_experiment_date_mapping");
    assert(FindLoaded(loaded, staleAnalysisId).skipReason ==
           "missing_final_analysis");
    assert(FindLoaded(loaded, winterDateId).source->invocation.configuration
               .trainStartDate == "2024-01-15");
    assert(FindLoaded(loaded, summerDateId).source->invocation.configuration
               .trainStartDate == "2024-07-15");
    assert(FindLoaded(loaded, dstDateId).source->invocation.configuration
               .trainStartDate == "2024-03-10");

    // Explicit Chicago conversion is independent of the active session zone.
    transaction.exec("SET LOCAL TIME ZONE 'UTC';");
    const auto utcLoaded = LoadRecommendationSources(transaction, {});
    transaction.exec("SET LOCAL TIME ZONE 'Asia/Tokyo';");
    const auto tokyoLoaded = LoadRecommendationSources(transaction, {});
    assert(FindLoaded(utcLoaded, winterDateId).source->invocation.configuration
               .trainStartDate ==
           FindLoaded(tokyoLoaded, winterDateId).source->invocation.configuration
               .trainStartDate);
    assert(FindLoaded(utcLoaded, summerDateId).source->invocation.configuration
               .trainStartDate ==
           FindLoaded(tokyoLoaded, summerDateId).source->invocation.configuration
               .trainStartDate);

    EffectiveExperimentConfiguration ambiguousConfiguration;
    ambiguousConfiguration.symbol = "step3_ambiguous_date";
    ambiguousConfiguration.predictionHorizon = 12;
    ambiguousConfiguration.labelThreshold = 0.001;
    ambiguousConfiguration.coreLrMult = 1.0;
    ambiguousConfiguration.headLrMult = 5.0;
    ambiguousConfiguration.targetEpochs = 120;
    ambiguousConfiguration.trainStartDate = "2010-01-01";
    ambiguousConfiguration.trainEndDate = "2025-01-01";
    ExperimentInvocationConfiguration ambiguousInvocation;
    ambiguousInvocation.configuration = ambiguousConfiguration;
    ambiguousInvocation.checkpointInterval = 20;
    bool duplicateMappingFailed = false;
    try
    {
        FindExperimentDuplicate(
            transaction,
            BuildRecommendationCandidateIdentity(ambiguousConfiguration),
            BuildRecommendationInvocationIdentity(ambiguousInvocation),
            RecommendationPolicy{});
    }
    catch (const std::runtime_error& error)
    {
        duplicateMappingFailed =
            std::string{error.what()}.find(
                "experiment_duplicate_mapping_error:experiment_id=") == 0;
    }
    assert(duplicateMappingFailed);

    filters.symbol = "step3_fixture";
    const auto filtered = LoadRecommendationSources(transaction, filters);
    assert(filtered.size() == 1);
    assert(filtered.front().experimentId == validId);
    filters.predictionHorizon = 24;
    assert(LoadRecommendationSources(transaction, filters).empty());
    filters.predictionHorizon.reset();
    filters.sourceExperimentId = validId;
    assert(LoadRecommendationSources(transaction, filters).size() == 1);

    const RecommendationSource& source = *valid.source;
    const RecommendationCandidateIdentity semantic =
        BuildRecommendationCandidateIdentity(source.invocation.configuration);
    const RecommendationInvocationIdentity invocation =
        BuildRecommendationInvocationIdentity(source.invocation);
    RecommendationPolicy policy;
    const ExperimentDuplicateMatch existing = FindExperimentDuplicate(
        transaction, semantic, invocation, policy);
    assert(existing.type == RecommendationDuplicateType::existingExperiment);
    assert(existing.experimentId == std::optional<long long>{validId});
    assert(existing.semanticExactMatch);
    assert(existing.invocationExactMatch);

    ExperimentInvocationConfiguration operationalVariant = source.invocation;
    operationalVariant.checkpointInterval += 1;
    operationalVariant.resumeModelId = 123456789;
    const ExperimentDuplicateMatch semanticDespiteInvocation =
        FindExperimentDuplicate(
            transaction, semantic,
            BuildRecommendationInvocationIdentity(operationalVariant), policy);
    assert(semanticDespiteInvocation.type ==
           RecommendationDuplicateType::existingExperiment);
    assert(!semanticDespiteInvocation.invocationExactMatch);

    transaction.exec("UPDATE experiment SET status='failed' WHERE experiment_id=$1;",
                     pqxx::params{validId});
    const ExperimentDuplicateMatch terminal = FindExperimentDuplicate(
        transaction, semantic, invocation, policy);
    assert(terminal.type ==
           RecommendationDuplicateType::excludedTerminalExperiment);
    policy.terminalExperimentsAreDuplicates = false;
    const ExperimentDuplicateMatch allowedTerminal = FindExperimentDuplicate(
        transaction, semantic, invocation, policy);
    assert(allowedTerminal.type == RecommendationDuplicateType::noDuplicate);
    assert(allowedTerminal.experimentId == std::optional<long long>{validId});

    // Active and historical recommendation matching uses canonical semantic
    // and policy text. Hash equality alone records a collision, not equality.
    transaction.exec(
        "UPDATE experiment SET status='completed' WHERE experiment_id=$1;",
        pqxx::params{validId});
    const long long scanId = transaction.exec(
        "INSERT INTO experiment_recommendation_scan (status,policy_canonical,"
        "policy_hash,policy_version) VALUES ('running',$1,$2,$3) "
        "RETURNING recommendation_scan_id;",
        pqxx::params{RecommendationPolicyCanonicalText(policy),
                     RecommendationPolicyHash(policy), policy.policyVersion})
        .one_row()[0].as<long long>();
    EffectiveExperimentConfiguration proposalConfiguration =
        source.invocation.configuration;
    proposalConfiguration.coreLrMult = 2.0;
    const RecommendationCandidateIdentity proposalSemantic =
        BuildRecommendationCandidateIdentity(proposalConfiguration);
    ExperimentInvocationConfiguration proposalInvocation = source.invocation;
    proposalInvocation.configuration = proposalConfiguration;
    const RecommendationInvocationIdentity proposalExact =
        BuildRecommendationInvocationIdentity(proposalInvocation);
    const long long activeId = InsertRecommendation(
        transaction, scanId, source, proposalSemantic, proposalExact, policy,
        "proposed", proposalSemantic.hash);
    const RecommendationDuplicateMatch active = FindRecommendationDuplicate(
        transaction, proposalSemantic, proposalExact, policy);
    assert(active.kind == PersistedRecommendationMatchKind::active);
    assert(active.recommendationId == std::optional<long long>{activeId});
    assert(active.semanticExactMatch && active.invocationExactMatch &&
           active.policyExactMatch);

    bool strayStatusMetadataRejected = false;
    try
    {
        pqxx::subtransaction invalid{transaction, "invalid_status_metadata"};
        invalid.exec(
            "UPDATE experiment_recommendation SET expired_at=now() "
            "WHERE recommendation_id=$1;",
            pqxx::params{activeId});
        invalid.commit();
    }
    catch (const pqxx::check_violation&)
    {
        strayStatusMetadataRejected = true;
    }
    assert(strayStatusMetadataRejected);

    transaction.exec(
        "UPDATE experiment_recommendation SET status='rejected',"
        "rejected_at=now(),rejected_reason='test_rejection' "
        "WHERE recommendation_id=$1;",
        pqxx::params{activeId});
    const RecommendationDuplicateMatch historical = FindRecommendationDuplicate(
        transaction, proposalSemantic, proposalExact, policy);
    assert(historical.kind == PersistedRecommendationMatchKind::historical);
    assert(historical.recommendationId == std::optional<long long>{activeId});

    EffectiveExperimentConfiguration collisionConfiguration =
        proposalConfiguration;
    collisionConfiguration.labelThreshold = 0.002;
    const RecommendationCandidateIdentity collisionSemantic =
        BuildRecommendationCandidateIdentity(collisionConfiguration);
    ExperimentInvocationConfiguration collisionInvocation = source.invocation;
    collisionInvocation.configuration = collisionConfiguration;
    const RecommendationInvocationIdentity collisionExact =
        BuildRecommendationInvocationIdentity(collisionInvocation);
    RecommendationPolicy collisionPolicy = policy;
    collisionPolicy.policyVersion += 1;
    InsertRecommendation(
        transaction, scanId, source, collisionSemantic, collisionExact,
        collisionPolicy, "rejected", proposalSemantic.hash);
    const RecommendationDuplicateMatch withCollision =
        FindRecommendationDuplicate(
            transaction, proposalSemantic, proposalExact, policy);
    assert(!withCollision.hashCollisions.empty());
    assert(withCollision.hashCollisions.front().identityKind == "semantic");

    // All fixtures, including their analysis rows, are transaction-local.
    transaction.abort();

    const auto scans = ListRecommendationScans(connection, 10);
    if (!scans.empty())
        assert(FindRecommendationScan(
            connection, scans.front().recommendationScanId));
    const auto recommendations = ListRecommendations(
        connection, RecommendationListFilters{});
    if (!recommendations.empty())
    {
        const auto detail = FindRecommendation(
            connection, recommendations.front().recommendationId);
        assert(detail.has_value());
        assert(!detail->semanticConfigurationCanonical.empty());
        assert(!detail->invocationConfigurationCanonical.empty());
        assert(!detail->policyCanonical.empty());
    }

    return 0;
}
