#include <barrier>
#include <cassert>
#include <cstdlib>
#include <future>
#include <optional>
#include <string>
#include <stdexcept>
#include <utility>
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

RecommendationScoringInput Input(long long recommendationId,
                                 long long experimentId)
{
    RecommendationScoringInput input;
    input.recommendationId = recommendationId;
    input.sourceExperimentId = experimentId;
    input.sourcePredictionHorizon = 12;
    input.sourceRankWithinGroup = 1;
    input.sourceLeaderScore = 0.75;
    input.sourceInferenceAccuracy = 0.70;
    input.sourcePredictedNeutralProportion = 0.30;
    input.sourceEvidenceCount = 100;
    input.changedParameter = kCoreLrMult;
    input.sourceValueCanonical = "1";
    input.proposedValueCanonical = "1.25";
    input.absoluteDelta = 0.25;
    input.relativeDelta = 0.25;
    input.generationOrdinal = 1;
    input.structuralRank = 1;
    input.semanticCanonicalText = "step4_fixture_semantic";
    input.invocationCanonicalText = "step4_fixture_invocation";
    input.recommendationPolicyCanonicalText = "step4_fixture_policy";
    input.duplicateType = "no_duplicate";
    input.recommendationStatus = "proposed";
    return input;
}

void RequireRetryMismatch(
    pqxx::connection& connection,
    long long runId,
    const RankedRecommendationScore& ranked)
{
    bool mismatch = false;
    try
    {
        (void)PersistRecommendationScore(connection, {runId, ranked});
    }
    catch (const std::runtime_error& error)
    {
        mismatch = std::string{error.what()} ==
            "recommendation_score_retry_mismatch";
    }
    assert(mismatch);
}

bool SourceRankInsertRejected(
    pqxx::work& transaction,
    long long scanId,
    long long experimentId,
    long long analysisId,
    const std::optional<int>& sourceRank,
    const std::string& suffix)
{
    try
    {
        pqxx::subtransaction attempt{transaction, "source_rank_" + suffix};
        attempt.exec(
            "INSERT INTO experiment_recommendation (recommendation_scan_id,status,"
            "source_experiment_id,source_analysis_id,source_symbol,"
            "source_prediction_horizon,source_leader_score,source_infer_accuracy,"
            "source_evidence_count,changed_parameter,source_value_canonical,"
            "proposed_value_canonical,absolute_delta,"
            "semantic_configuration_canonical,semantic_hash,"
            "invocation_configuration_canonical,invocation_hash,"
            "policy_canonical,policy_hash,source_rank,generation_ordinal,"
            "structural_rank,duplicate_type,reason) VALUES "
            "($1,'proposed',$2,$3,'step4_source_rank_fixture',12,0.75,0.70,100,"
            "'core_lr_mult','1','1.25',0.25,$4,'fnv1a64:rank_semantic',"
            "$5,'fnv1a64:rank_invocation',$6,'fnv1a64:rank_policy',$7,1,1,"
            "'no_duplicate','source_rank_constraint_test');",
            pqxx::params{scanId, experimentId, analysisId,
                         "source_rank_semantic_" + suffix,
                         "source_rank_invocation_" + suffix,
                         "source_rank_policy_" + suffix, sourceRank});
        attempt.commit();
        return false;
    }
    catch (const pqxx::check_violation& error)
    {
        assert(error.sqlstate() == "23514");
        return true;
    }
}

void VerifyLegacyRankTriggerSemantics(pqxx::work& transaction)
{
    transaction.exec(
        "CREATE TEMP TABLE source_rank_trigger_fixture ("
        "recommendation_scan_id bigint,source_rank integer,note text);");
    transaction.exec(
        "INSERT INTO source_rank_trigger_fixture VALUES "
        "(NULL,NULL,'legacy_unrelated'),"
        "(NULL,NULL,'legacy_transition'),"
        "(NULL,NULL,'legacy_untouched');");
    transaction.exec(
        "CREATE TRIGGER source_rank_trigger_fixture_enforce "
        "BEFORE INSERT OR UPDATE OF source_rank,recommendation_scan_id "
        "ON source_rank_trigger_fixture FOR EACH ROW EXECUTE FUNCTION "
        "enforce_experiment_recommendation_source_rank();");
    transaction.exec(
        "UPDATE source_rank_trigger_fixture SET note='legacy_updated' "
        "WHERE note='legacy_unrelated';");
    assert(transaction.exec(
        "SELECT count(*) FROM source_rank_trigger_fixture "
        "WHERE note='legacy_updated' AND recommendation_scan_id IS NULL "
        "AND source_rank IS NULL;").one_row()[0].as<int>() == 1);

    const auto rejected = [&](const std::optional<int>& rank,
                              const std::string& name) {
        try
        {
            pqxx::subtransaction attempt{transaction, name};
            attempt.exec(
                "INSERT INTO source_rank_trigger_fixture VALUES (1,$1,'new');",
                pqxx::params{rank});
            attempt.commit();
            return false;
        }
        catch (const pqxx::check_violation& error)
        {
            assert(error.sqlstate() == "23514");
            return true;
        }
    };
    assert(rejected(std::nullopt, "temp_rank_missing"));
    assert(rejected(0, "temp_rank_zero"));
    assert(rejected(-1, "temp_rank_negative"));
    assert(!rejected(1, "temp_rank_positive"));

    const auto updateRejected = [&](const std::string& statement,
                                    const std::string& name) {
        try
        {
            pqxx::subtransaction attempt{transaction, name};
            attempt.exec(statement);
            attempt.commit();
            return false;
        }
        catch (const pqxx::check_violation& error)
        {
            assert(error.sqlstate() == "23514");
            return true;
        }
    };
    const auto requireState = [&](const std::string& note,
                                  const std::optional<long long>& scanId,
                                  const std::optional<int>& sourceRank) {
        assert(transaction.exec(
            "SELECT count(*) FROM source_rank_trigger_fixture "
            "WHERE note=$1 AND recommendation_scan_id IS NOT DISTINCT FROM "
            "$2::bigint AND source_rank IS NOT DISTINCT FROM $3::integer;",
            pqxx::params{note, scanId, sourceRank})
            .one_row()[0].as<int>() == 1);
    };

    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=1 "
        "WHERE note='legacy_transition';", "temp_attach_null_rank"));
    requireState("legacy_transition", std::nullopt, std::nullopt);
    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=1,"
        "source_rank=0 WHERE note='legacy_transition';",
        "temp_attach_zero_rank"));
    requireState("legacy_transition", std::nullopt, std::nullopt);
    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=1,"
        "source_rank=-1 WHERE note='legacy_transition';",
        "temp_attach_negative_rank"));
    requireState("legacy_transition", std::nullopt, std::nullopt);

    transaction.exec(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=1,"
        "source_rank=7 WHERE note='legacy_transition';");
    requireState("legacy_transition", 1, 7);

    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET source_rank=NULL "
        "WHERE note='legacy_transition';", "temp_rank_clear"));
    requireState("legacy_transition", 1, 7);
    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET source_rank=0 "
        "WHERE note='legacy_transition';", "temp_rank_zero_update"));
    requireState("legacy_transition", 1, 7);
    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET source_rank=-1 "
        "WHERE note='legacy_transition';", "temp_rank_negative_update"));
    requireState("legacy_transition", 1, 7);

    transaction.exec(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=2 "
        "WHERE note='new';");
    requireState("new", 2, 1);
    transaction.exec(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=3,"
        "source_rank=8 WHERE note='new';");
    requireState("new", 3, 8);

    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=1,"
        "source_rank=NULL WHERE note='new';",
        "temp_reassociate_null_rank"));
    requireState("new", 3, 8);
    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=1,"
        "source_rank=0 WHERE note='new';",
        "temp_reassociate_zero_rank"));
    requireState("new", 3, 8);
    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=1,"
        "source_rank=-1 WHERE note='new';",
        "temp_reassociate_negative_rank"));
    requireState("new", 3, 8);

    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=NULL "
        "WHERE note='new';", "temp_detach_positive_rank"));
    requireState("new", 3, 8);
    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=NULL,"
        "source_rank=NULL WHERE note='new';", "temp_detach_null_rank"));
    requireState("new", 3, 8);
    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=NULL,"
        "source_rank=0 WHERE note='new';", "temp_detach_zero_rank"));
    requireState("new", 3, 8);
    assert(updateRejected(
        "UPDATE source_rank_trigger_fixture SET recommendation_scan_id=NULL,"
        "source_rank=-1 WHERE note='new';", "temp_detach_negative_rank"));
    requireState("new", 3, 8);

    requireState("legacy_untouched", std::nullopt, std::nullopt);
}

void Cleanup(pqxx::connection& connection, long long experimentId,
             long long scanId, long long recommendationId)
{
    pqxx::work tx{connection};
    tx.exec("SET TRANSACTION READ WRITE;");
    tx.exec(
        "DELETE FROM experiment_recommendation_score_component WHERE "
        "recommendation_score_id IN (SELECT s.recommendation_score_id FROM "
        "experiment_recommendation_score s JOIN experiment_recommendation r "
        "ON r.recommendation_id=s.recommendation_id "
        "WHERE r.recommendation_scan_id=$1);", pqxx::params{scanId});
    tx.exec(
        "DELETE FROM experiment_recommendation_score WHERE recommendation_id "
        "IN (SELECT recommendation_id FROM experiment_recommendation "
        "WHERE recommendation_scan_id=$1);", pqxx::params{scanId});
    tx.exec(
        "DELETE FROM experiment_recommendation_score_run WHERE "
        "recommendation_id_filter=$1;", pqxx::params{recommendationId});
    tx.exec("DELETE FROM experiment_recommendation WHERE recommendation_scan_id=$1;",
            pqxx::params{scanId});
    tx.exec("DELETE FROM experiment_recommendation_scan WHERE recommendation_scan_id=$1;",
            pqxx::params{scanId});
    tx.exec("DELETE FROM experiment_analysis_result WHERE experiment_id=$1;",
            pqxx::params{experimentId});
    tx.exec("DELETE FROM experiment WHERE experiment_id=$1;",
            pqxx::params{experimentId});
    tx.commit();
}
} // namespace

int main()
{
    const std::string connectionString =
        "hostaddr=" + EnvironmentOr("LSTM_DB_HOST", "127.0.0.1") +
        " user=pqxx dbname=" + EnvironmentOr("LSTM_DB_NAME", "LSTM");
    pqxx::connection connection{connectionString};
    assert(RecommendationScoringSchemaExists(connection));
    {
        pqxx::read_transaction inspect{connection};
        assert(inspect.exec(
            "SELECT 1 FROM pg_indexes WHERE indexname="
            "'experiment_recommendation_score_run_rank_idx';").size() == 1);
        assert(inspect.exec(
            "SELECT 1 FROM pg_constraint WHERE conrelid="
            "'experiment_recommendation_score'::regclass AND contype='u';")
            .size() >= 2);
        assert(inspect.exec(
            "SELECT 1 FROM pg_constraint WHERE conrelid="
            "'experiment_recommendation_score_component'::regclass "
            "AND contype='f';").size() == 1);
        assert(inspect.exec(
            "SELECT 1 FROM pg_trigger WHERE tgname="
            "'experiment_recommendation_source_rank_enforce_trigger' "
            "AND tgrelid='experiment_recommendation'::regclass "
            "AND NOT tgisinternal;").size() == 1);
    }

    long long experimentId = -1;
    long long analysisId = -1;
    long long scanId = -1;
    long long recommendationId = -1;
    {
        pqxx::work tx{connection};
        tx.exec("SET TRANSACTION READ WRITE;");
        VerifyLegacyRankTriggerSemantics(tx);
        experimentId = tx.exec(
            "INSERT INTO experiment (symbol,prediction_horizon,c_next_threshold,"
            "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
            "train_start,train_end,status,phase,duplicate_nonce) VALUES "
            "('step4_scoring_fixture',12,0.001,1,5,120,20,"
            "'2010-01-01 America/Chicago'::timestamptz,"
            "'2025-01-01 America/Chicago'::timestamptz,'completed','done',"
            "floor(random()*900000000)::bigint+100000000) RETURNING experiment_id;")
            .one_row()[0].as<long long>();
        analysisId = tx.exec(
            "INSERT INTO experiment_analysis_result (experiment_id,model_id,"
            "symbol,prediction_horizon,target_epochs,infer_accuracy,leader_score,"
            "pred_down_count,pred_neutral_count,pred_up_count,analysis_status,"
            "analysis_scope) VALUES ($1,987654399,'step4_scoring_fixture',12,"
            "120,0.70,0.75,20,30,50,'completed','final') RETURNING analysis_id;",
            pqxx::params{experimentId}).one_row()[0].as<long long>();
        scanId = tx.exec(
            "INSERT INTO experiment_recommendation_scan (status,policy_canonical,"
            "policy_hash,policy_version,completed_at) VALUES "
            "('completed','step4_fixture_policy','fnv1a64:fixture',1,now()) "
            "RETURNING recommendation_scan_id;")
            .one_row()[0].as<long long>();
        assert(SourceRankInsertRejected(
            tx, scanId, experimentId, analysisId, std::nullopt, "missing"));
        assert(SourceRankInsertRejected(
            tx, scanId, experimentId, analysisId, 0, "zero"));
        assert(SourceRankInsertRejected(
            tx, scanId, experimentId, analysisId, -1, "negative"));
        assert(!SourceRankInsertRejected(
            tx, scanId, experimentId, analysisId, 1, "positive"));
        recommendationId = tx.exec(
            "INSERT INTO experiment_recommendation (recommendation_scan_id,status,"
            "source_experiment_id,source_analysis_id,source_symbol,source_prediction_horizon,"
            "source_leader_score,source_infer_accuracy,"
            "source_predicted_neutral_proportion,source_evidence_count,"
            "changed_parameter,source_value_canonical,proposed_value_canonical,"
            "absolute_delta,relative_delta,semantic_configuration_canonical,"
            "semantic_hash,invocation_configuration_canonical,invocation_hash,"
            "policy_canonical,policy_hash,source_rank,generation_ordinal,"
            "structural_rank,duplicate_type,reason) VALUES "
            "($1,'proposed',$2,$3,'step4_scoring_fixture',12,0.75,0.70,0.30,100,"
            "'core_lr_mult','1','1.25',0.25,0.25,'step4_fixture_semantic',"
            "'fnv1a64:semantic','step4_fixture_invocation','fnv1a64:invocation',"
            "'step4_fixture_policy','fnv1a64:policy',1,1,1,'no_duplicate',"
            "'step4_fixture') RETURNING recommendation_id;",
            pqxx::params{scanId, experimentId, analysisId}).one_row()[0].as<long long>();
        tx.exec(
            "INSERT INTO experiment_recommendation_score_run (status,"
            "scoring_policy_canonical,scoring_policy_hash,scoring_version,"
            "recommendation_status_filter,recommendation_id_filter,completed_at) "
            "VALUES ('completed','deliberate_collision_fixture',$1,1,"
            "'proposed',$2,now());",
            pqxx::params{
                RecommendationScoringPolicyHash(RecommendationScoringPolicy{}),
                recommendationId});
        tx.commit();
    }

    try
    {
        std::string experimentStateBefore;
        std::string recommendationStateBefore;
        {
            pqxx::read_transaction snapshot{connection};
            experimentStateBefore = snapshot.exec(
                "SELECT status||'|'||phase||'|'||coalesce(worker_pid::text,'NULL')"
                "||'|'||coalesce(current_operation,'NULL')||'|'||updated_at::text "
                "FROM experiment WHERE experiment_id=$1;",
                pqxx::params{experimentId}).one_row()[0].as<std::string>();
            recommendationStateBefore = snapshot.exec(
                "SELECT status||'|'||updated_at::text FROM "
                "experiment_recommendation WHERE recommendation_id=$1;",
                pqxx::params{recommendationId}).one_row()[0].as<std::string>();
        }
        RecommendationScoreRunRequest runRequest;
        runRequest.filters.recommendationId = recommendationId;
        assert(FindRecommendationScoringPolicyHashCollision(
            connection, runRequest.policy) ==
            std::optional<std::string>{"deliberate_collision_fixture"});
        const long long runId = BeginRecommendationScoreRun(connection, runRequest);
        const auto loaded = LoadRecommendationsForScoring(
            connection, runRequest.filters);
        assert(loaded.size() == 1 && loaded.front().input);
        assert(loaded.front().input->sourceRankWithinGroup == 1);
        assert(loaded.front().input->absoluteDelta == 0.25);
        assert(loaded.front().input->relativeDelta ==
               std::optional<double>{0.25});
        assert(DeriveRecommendationScoringDistance(*loaded.front().input) ==
               0.25);

        RecommendationScoringInput input = Input(recommendationId, experimentId);
        RecommendationScoreResult score =
            ScoreExperimentRecommendation(runRequest.policy, input);
        RankedRecommendationScore ranked{input, score, 1, 1, 1};
        std::barrier start{3};
        const auto persist = [&]() {
            pqxx::connection concurrent{connectionString};
            start.arrive_and_wait();
            return PersistRecommendationScore(concurrent, {runId, ranked});
        };
        auto first = std::async(std::launch::async, persist);
        auto second = std::async(std::launch::async, persist);
        start.arrive_and_wait();
        const auto firstResult = first.get();
        const auto secondResult = second.get();
        assert(firstResult.recommendationScoreId ==
               secondResult.recommendationScoreId);
        assert(firstResult.created != secondResult.created);

        const auto identicalRetry = PersistRecommendationScore(
            connection, {runId, ranked});
        assert(!identicalRetry.created);
        assert(identicalRetry.recommendationScoreId ==
               firstResult.recommendationScoreId);

        RankedRecommendationScore changed = ranked;
        changed.score.rawPositiveScore += 0.001;
        RequireRetryMismatch(connection, runId, changed);
        changed = ranked;
        changed.score.structuralDistance += 0.001;
        RequireRetryMismatch(connection, runId, changed);
        changed = ranked;
        changed.score.reasonCode += "_changed";
        RequireRetryMismatch(connection, runId, changed);
        changed = ranked;
        changed.score.explanationSummary += " changed";
        RequireRetryMismatch(connection, runId, changed);
        changed = ranked;
        changed.score.components.pop_back();
        RequireRetryMismatch(connection, runId, changed);
        changed = ranked;
        std::swap(changed.score.components[0], changed.score.components[1]);
        RequireRetryMismatch(connection, runId, changed);

        const auto RequireComponentMismatch = [&](const auto& mutate) {
            RankedRecommendationScore componentChanged = ranked;
            mutate(componentChanged.score.components.front());
            RequireRetryMismatch(connection, runId, componentChanged);
        };
        RequireComponentMismatch([](auto& value) {
            value.componentName = "inference_accuracy";
        });
        RequireComponentMismatch([](auto& value) {
            value.reasonCode += "_changed";
        });
        RequireComponentMismatch([](auto& value) {
            value.inputCanonical += "_changed";
        });
        RequireComponentMismatch([](auto& value) {
            value.normalizedValue += 0.001;
        });
        RequireComponentMismatch([](auto& value) {
            value.weight += 0.001;
        });
        RequireComponentMismatch([](auto& value) {
            value.weightedContribution += 0.001;
        });
        RequireComponentMismatch([](auto& value) {
            value.penalty = !value.penalty;
        });
        RequireComponentMismatch([](auto& value) {
            value.explanation += " changed";
        });

        const auto detail = FindRecommendationScore(
            connection, firstResult.recommendationScoreId);
        assert(detail && detail->components.size() == 9);
        assert(detail->scoreRank == 1 && detail->tieGroup == 1);
        RecommendationScoreRunCounters counters;
        counters.recommendationsConsidered = 1;
        counters.recommendationsScored = 1;
        CompleteRecommendationScoreRun(connection, runId, counters);
        const auto run = FindRecommendationScoreRun(connection, runId);
        assert(run && run->status == "completed");

        const long long secondRun = BeginRecommendationScoreRun(connection, runRequest);
        const auto secondHistory = PersistRecommendationScore(
            connection, {secondRun, ranked});
        assert(secondHistory.recommendationScoreId !=
               firstResult.recommendationScoreId);
        const auto secondDetail = FindRecommendationScore(
            connection, secondHistory.recommendationScoreId);
        assert(secondDetail && detail);
        assert(secondDetail->finalScore == detail->finalScore);
        assert(secondDetail->rawPositiveScore == detail->rawPositiveScore);
        assert(secondDetail->rawPenaltyScore == detail->rawPenaltyScore);
        assert(secondDetail->rawTotalScore == detail->rawTotalScore);
        assert(secondDetail->structuralDistance == detail->structuralDistance);
        assert(secondDetail->scoreRank == detail->scoreRank);
        assert(secondDetail->tieGroup == detail->tieGroup);
        assert(secondDetail->rankingOrdinal == detail->rankingOrdinal);
        assert(secondDetail->components.size() == detail->components.size());
        for (std::size_t index = 0; index < detail->components.size(); ++index)
        {
            const auto& lhs = secondDetail->components[index];
            const auto& rhs = detail->components[index];
            assert(lhs.componentName == rhs.componentName);
            assert(lhs.reasonCode == rhs.reasonCode);
            assert(lhs.inputCanonical == rhs.inputCanonical);
            assert(lhs.normalizedValue == rhs.normalizedValue);
            assert(lhs.weight == rhs.weight);
            assert(lhs.weightedContribution == rhs.weightedContribution);
            assert(lhs.penalty == rhs.penalty);
            assert(lhs.explanation == rhs.explanation);
        }
        CompleteRecommendationScoreRun(connection, secondRun, counters);
        const auto scores = ListRecommendationScores(connection,
            RecommendationScoringFilters{.recommendationId = recommendationId});
        assert(scores.size() == 2);

        const long long failedRun = BeginRecommendationScoreRun(connection, runRequest);
        FailRecommendationScoreRun(connection, failedRun, {}, "");
        const auto failed = FindRecommendationScoreRun(connection, failedRun);
        assert(failed && failed->status == "failed");
        assert(failed->errorMessage ==
               std::optional<std::string>{
                   "unknown_recommendation_score_run_failure"});
        bool duplicateFinalizationRejected = false;
        try
        {
            CompleteRecommendationScoreRun(connection, failedRun, {});
        }
        catch (const std::runtime_error&)
        {
            duplicateFinalizationRejected = true;
        }
        assert(duplicateFinalizationRejected);

        pqxx::read_transaction snapshot{connection};
        const std::string experimentStateAfter = snapshot.exec(
            "SELECT status||'|'||phase||'|'||coalesce(worker_pid::text,'NULL')"
            "||'|'||coalesce(current_operation,'NULL')||'|'||updated_at::text "
            "FROM experiment WHERE experiment_id=$1;",
            pqxx::params{experimentId}).one_row()[0].as<std::string>();
        const std::string recommendationStateAfter = snapshot.exec(
            "SELECT status||'|'||updated_at::text FROM "
            "experiment_recommendation WHERE recommendation_id=$1;",
            pqxx::params{recommendationId}).one_row()[0].as<std::string>();
        assert(experimentStateAfter == experimentStateBefore);
        assert(recommendationStateAfter == recommendationStateBefore);
    }
    catch (...)
    {
        Cleanup(connection, experimentId, scanId, recommendationId);
        throw;
    }
    Cleanup(connection, experimentId, scanId, recommendationId);
    return 0;
}
