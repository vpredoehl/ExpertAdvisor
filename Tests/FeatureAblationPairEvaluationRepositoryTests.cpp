#include "FeatureAblationPairEvaluationRepository.hpp"
#include "FeatureAblationPairEvaluationService.hpp"
#include "FeatureAblationReplicationEvaluationService.hpp"

#include "FeatureAblation.hpp"
#include "InferenceProfitabilityRepository.hpp"
#include "TrainingObjective.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace Feature = EA::FeatureAblationPairEvaluation;
namespace Replication = EA::FeatureAblationReplicationEvaluation;
namespace Objective = EA::TrainingObjective;
namespace Profitability = EA::InferenceProfitability;

namespace
{

constexpr double kExperimentThreshold = 0.0008;
const double kModelThreshold =
    static_cast<double>(static_cast<float>(kExperimentThreshold));

struct Ids
{
    long long experiment;
    long long model;
    long long inference;
    long long analysis;
};

std::string RequiredEnvironment(const char* name)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
        throw std::runtime_error(std::string{"missing_environment:"} + name);
    return value;
}

void InsertMatrixRow(pqxx::transaction_base& transaction,
                     long long modelId,
                     const std::string& name,
                     const std::vector<double>& values)
{
    for (std::size_t index = 0; index < values.size(); ++index)
        transaction.exec(
            "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,n_rows,"
            "n_cols,value) VALUES($1,$2,0,$3,1,$4,$5);",
            pqxx::params{modelId, name, static_cast<int>(index),
                         static_cast<int>(values.size()), values[index]});
}

void InsertAscii(pqxx::transaction_base& transaction,
                 long long modelId,
                 const std::string& name,
                 const std::string& value)
{
    std::vector<double> encoded;
    encoded.reserve(value.size());
    for (const unsigned char character : value)
        encoded.push_back(static_cast<double>(character));
    InsertMatrixRow(transaction, modelId, name, encoded);
}

void InsertExperiment(pqxx::transaction_base& transaction,
                      const Ids& ids,
                      bool control,
                      bool complete = true,
                      std::optional<std::string> maskOverride = std::nullopt,
                      int modelInputWidth = 71,
                      int modelInputLayoutVersion = 4,
                      std::optional<long long> snapshotId = std::nullopt,
                      std::optional<std::string> snapshotHash = std::nullopt)
{
    const auto objective = Objective::Legacy();
    const std::string canonical = Objective::CanonicalText(objective);
    const std::string hash = Objective::DeterministicHash(canonical);
    const std::string mask = maskOverride.value_or(
        control ? std::string(EA::kEconomicEventConsensusAblationMaskText)
                : "");
    transaction.exec(
        "INSERT INTO experiment("
        "experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
        "train_start,train_end,infer_start,infer_end,status,phase,last_model_id,"
        "resume_model_id,duplicate_nonce,donchian20_mode,donchian_lookback,"
        "feature_warmup_scope,feature_ablation_mask,resume_expand_input_width,"
        "git_commit,git_branch,git_dirty,build_config,compiler_version,"
        "schema_version,scheduler_version,binary_name,training_objective_id,"
        "training_objective_version,loss_definition_version,"
        "training_objective_canonical,training_objective_hash,"
        "auxiliary_loss_mode,auxiliary_loss_coefficient,"
        "target_clipping_definition,objective_normalization_identity,"
        "model_input_width,model_input_semantic_layout_version,"
        "economic_calendar_snapshot_id,economic_calendar_snapshot_hash) VALUES("
        "$1,'audchfrmp',4,$2,119.75,25,80,20,"
        "'2010-01-01','2025-01-01','2025-01-01','2026-01-01',$3,$4,$5,"
        "NULL,$1,'enabled',20,'legacy_cold_boundary',$6,false,"
        "'b5b925234367ede13528f6bdbfa0ae328ecc5103','phase6',false,"
        "'Release','AppleClang-test','079','scheduler-test','LSTM_Release',"
        "$7,1,1,$8,$9,'disabled',0,'none',"
        "'weighted_loss_sum_by_weight_sum_gradients__calculate_batch_return_by_example_count_v1',"
        "$10,$11,$12,$13);",
        pqxx::params{ids.experiment, kExperimentThreshold,
                     complete ? "completed" : "running",
                     complete ? "done" : "train",
                     complete ? std::optional<long long>{ids.model}
                              : std::nullopt,
                     mask, objective.objectiveIdentifier, canonical, hash,
                     modelInputWidth, modelInputLayoutVersion,
                     snapshotId, snapshotHash});
    if (!complete) return;

    transaction.exec(
        "INSERT INTO model(model_id,experiment_id) VALUES($1,$2);",
        pqxx::params{ids.model, ids.experiment});
    InsertMatrixRow(transaction, ids.model, "train_config_meta", {
        1.0, 4.0, kModelThreshold, 64.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 80.0, 119.75, 25.0, 25.0});
    InsertMatrixRow(transaction, ids.model, "target_meta",
                    {0.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    InsertMatrixRow(transaction, ids.model, "model_meta",
                    {1.0, static_cast<double>(modelInputWidth), 64.0});
    InsertMatrixRow(transaction, ids.model, "model_input_semantics_meta",
                    {1.0, static_cast<double>(modelInputLayoutVersion)});
    InsertMatrixRow(transaction, ids.model, "optimizer_meta",
                    {1.0, 1.0, 100.0, 0.0, 0.0});
    InsertAscii(transaction, ids.model, "train_symbol_meta", "audchfrmp");
    InsertAscii(transaction, ids.model, "train_range_meta",
                "2010-01-01|2025-01-01");
    transaction.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,n_rows,"
        "n_cols,value) VALUES($1,'param',0,0,$2,256,0);",
        pqxx::params{ids.model, modelInputWidth + 64});
    InsertAscii(transaction, ids.model, "training_objective_canonical_meta",
                canonical);
    InsertAscii(transaction, ids.model, "training_objective_hash_meta", hash);
}

void InsertFinalClassificationEvidence(
    pqxx::transaction_base& transaction,
    const Ids& ids,
    double accuracy,
    std::optional<double> persistedAnalysisAccuracy = std::nullopt)
{
    transaction.exec(
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,symbol,prediction_horizon,"
        "threshold_logret,window_size,label_rule_id,target_type,from_date,"
        "to_date,completed_epochs,accuracy,accept_model,pred_down,pred_neutral,"
        "pred_up) VALUES($1,$2,'completed','final','audchfrmp',4,$3,64,1,0,"
        "'2025-01-01','2026-01-01',80,$4,true,0.25,0.35,0.40);",
        pqxx::params{ids.inference, ids.model, kModelThreshold, accuracy});
    transaction.exec(
        "INSERT INTO experiment_analysis_result("
        "analysis_id,experiment_id,model_id,analysis_scope,analysis_status,"
        "infer_accuracy,accept_accuracy,accept_rate,leader_score,"
        "pred_down_count,pred_neutral_count,pred_up_count,accept_count) VALUES("
        "$1,$2,$3,'final','completed',$4,$5,0.65,$6,25,35,40,65);",
        pqxx::params{ids.analysis, ids.experiment, ids.model,
                     persistedAnalysisAccuracy.value_or(accuracy),
                     accuracy + 0.05, accuracy * 0.9});
}

void InsertProfitabilityEvidence(pqxx::transaction_base& transaction,
                                 const Ids& ids,
                                 double aggregateReturn)
{
    Profitability::Accumulator accumulator;
    accumulator.Observe(Profitability::kNeutralClass, 100.0f, 110.0f);
    accumulator.Observe(
        Profitability::kUpClass, 100.0f,
        static_cast<float>(100.0 * std::exp(aggregateReturn)));
    Profitability::ObservationRequest request;
    request.provenance.experimentId = ids.experiment;
    request.provenance.modelId = ids.model;
    request.provenance.inferenceEvalResultId = ids.inference;
    request.provenance.scope = Profitability::Scope::finalInference;
    request.provenance.inferenceStart = "2025-01-01";
    request.provenance.inferenceEnd = "2026-01-01";
    request.statistics = accumulator.statistics();
    request.sourceContentHash = accumulator.SourceContentHash();
    (void)Profitability::PersistObservationIdempotently(transaction, request);
}

void InsertFinalEvidence(pqxx::transaction_base& transaction,
                         const Ids& ids,
                         double accuracy,
                         double aggregateReturn)
{
    InsertFinalClassificationEvidence(transaction, ids, accuracy);
    InsertProfitabilityEvidence(transaction, ids, aggregateReturn);
}

void InsertCheckpointOnlyEvidence(pqxx::transaction_base& transaction,
                                  const Ids& ids)
{
    const long long checkpointEvalId = ids.inference + 5000000;
    transaction.exec(
        "INSERT INTO experiment_checkpoint_eval("
        "checkpoint_eval_id,parent_experiment_id,checkpoint_model_id) "
        "VALUES($1,$2,$3);",
        pqxx::params{checkpointEvalId, ids.experiment, ids.model});
    transaction.exec(
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,"
        "completed_epochs,accuracy,accept_model,pred_down,pred_neutral,pred_up) "
        "VALUES($1,$2,'completed','checkpoint',$3,$4,'audchfrmp',4,$5,64,1,0,"
        "'2025-01-01','2026-01-01',80,0.60,true,0.25,0.35,0.40);",
        pqxx::params{ids.inference, ids.model, checkpointEvalId,
                     ids.experiment, kModelThreshold});
}

void InsertSecondExactFinalInference(pqxx::transaction_base& transaction,
                                     const Ids& ids)
{
    transaction.exec(
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,symbol,prediction_horizon,"
        "threshold_logret,window_size,label_rule_id,target_type,from_date,"
        "to_date,completed_epochs,accuracy,accept_model,pred_down,pred_neutral,"
        "pred_up) VALUES($1,$2,'completed','final','audchfrmp',4,$3,64,1,0,"
        "'2025-01-01','2026-01-01',80,0.62,true,0.25,0.35,0.40);",
        pqxx::params{ids.inference + 9000000, ids.model, kModelThreshold});
}

std::string DatabaseDigest(pqxx::connection& connection)
{
    pqxx::read_transaction transaction{connection};
    return transaction.exec(
        "SELECT md5("
        "(SELECT COALESCE(string_agg(row_to_json(e)::text,'|' ORDER BY "
        "experiment_id),'') FROM experiment e) || '#' || "
        "(SELECT COALESCE(string_agg(row_to_json(m)::text,'|' ORDER BY "
        "model_id),'') FROM model m) || '#' || "
        "(SELECT COALESCE(string_agg(row_to_json(i)::text,'|' ORDER BY id),"
        "'') FROM inference_eval_result i) || '#' || "
        "(SELECT COALESCE(string_agg(row_to_json(a)::text,'|' ORDER BY "
        "analysis_id),'') FROM experiment_analysis_result a) || '#' || "
        "(SELECT COALESCE(string_agg(row_to_json(p)::text,'|' ORDER BY "
        "profitability_observation_id),'') FROM "
        "inference_profitability_observation p));").one_row()[0]
        .as<std::string>();
}

} // namespace

int main()
{
    const std::string databaseName =
        RequiredEnvironment("EA_CONSENSUS_EFFICACY_DB_NAME");
    assert(std::regex_match(
        databaseName,
        std::regex{"^ea_consensus_efficacy_[A-Za-z0-9_]+$"}));
    const std::string connectionString =
        "host=" + RequiredEnvironment("EA_CONSENSUS_EFFICACY_DB_HOST") +
        " port=" + RequiredEnvironment("EA_CONSENSUS_EFFICACY_DB_PORT") +
        " user=" + RequiredEnvironment("EA_CONSENSUS_EFFICACY_DB_USER") +
        " dbname=" + databaseName;
    pqxx::connection connection{connectionString};
    {
        pqxx::work fixture{connection};
        const Ids control{990601, 1990601, 2990601, 3990601};
        const Ids treatment{990602, 1990602, 2990602, 3990602};
        const Ids pendingControl{990611, 1990611, 2990611, 3990611};
        const Ids pendingTreatment{990612, 1990612, 2990612, 3990612};
        const Ids ambiguousControl{990621, 1990621, 2990621, 3990621};
        const Ids ambiguousTreatment{990622, 1990622, 2990622, 3990622};
        const Ids checkpointControl{990631, 1990631, 2990631, 3990631};
        const Ids checkpointTreatment{990632, 1990632, 2990632, 3990632};
        const Ids noProfitControl{990641, 1990641, 2990641, 3990641};
        const Ids noProfitTreatment{990642, 1990642, 2990642, 3990642};
        const Ids incompatibleControl{990651, 1990651, 2990651, 3990651};
        const Ids incompatibleTreatment{990652, 1990652, 2990652, 3990652};
        const Ids accuracyMismatchControl{990661, 1990661, 2990661, 3990661};
        const Ids accuracyMismatchTreatment{990662, 1990662, 2990662, 3990662};
        const Ids surpriseControl{990619, 1990619, 2990619, 3990619};
        const Ids surpriseAblation{990620, 1990620, 2990620, 3990620};
        const Ids snapshotSurpriseControl{
            990671, 1990671, 2990671, 3990671};
        const Ids snapshotSurpriseAblation{
            990672, 1990672, 2990672, 3990672};

        InsertExperiment(fixture, control, true);
        InsertExperiment(fixture, treatment, false);
        InsertFinalClassificationEvidence(fixture, control, 0.6100004, 0.61);
        InsertProfitabilityEvidence(fixture, control, 0.05);
        InsertFinalClassificationEvidence(
            fixture, treatment, 0.6400004, 0.64);
        InsertProfitabilityEvidence(fixture, treatment, 0.08);

        InsertExperiment(fixture, pendingControl, true, false);
        InsertExperiment(fixture, pendingTreatment, false, false);

        InsertExperiment(fixture, ambiguousControl, true);
        InsertExperiment(fixture, ambiguousTreatment, false);
        InsertFinalEvidence(fixture, ambiguousControl, 0.61, 0.05);
        InsertSecondExactFinalInference(fixture, ambiguousControl);
        InsertFinalEvidence(fixture, ambiguousTreatment, 0.64, 0.08);

        InsertExperiment(fixture, checkpointControl, true);
        InsertExperiment(fixture, checkpointTreatment, false);
        InsertCheckpointOnlyEvidence(fixture, checkpointControl);
        InsertFinalEvidence(fixture, checkpointTreatment, 0.64, 0.08);

        InsertExperiment(fixture, noProfitControl, true);
        InsertExperiment(fixture, noProfitTreatment, false);
        InsertFinalClassificationEvidence(fixture, noProfitControl, 0.61);
        InsertFinalEvidence(fixture, noProfitTreatment, 0.64, 0.08);

        InsertExperiment(fixture, incompatibleControl, true);
        InsertExperiment(fixture, incompatibleTreatment, false);
        fixture.exec(
            "UPDATE experiment SET feature_warmup_scope='full_history_warmup' "
            "WHERE experiment_id=$1;",
            pqxx::params{incompatibleTreatment.experiment});

        InsertExperiment(fixture, accuracyMismatchControl, true);
        InsertExperiment(fixture, accuracyMismatchTreatment, false);
        InsertFinalClassificationEvidence(
            fixture, accuracyMismatchControl, 0.6100006, 0.61);
        InsertProfitabilityEvidence(fixture, accuracyMismatchControl, 0.05);
        InsertFinalClassificationEvidence(
            fixture, accuracyMismatchTreatment, 0.6400004, 0.64);
        InsertProfitabilityEvidence(fixture, accuracyMismatchTreatment, 0.08);

        InsertExperiment(fixture, surpriseControl, false, true,
                         std::string{}, 77, 7);
        InsertExperiment(
            fixture, surpriseAblation, false, true,
            std::string(EA::kCausalEconomicEventSurpriseAblationMaskText),
            77, 7);
        InsertFinalEvidence(fixture, surpriseControl, 0.64, 0.08);
        InsertFinalEvidence(fixture, surpriseAblation, 0.61, 0.05);
        InsertExperiment(fixture, snapshotSurpriseControl, false, true,
                         std::string{}, 77, 7, 1,
                         "fnv1a64:67610f94f5c8e7cc");
        InsertExperiment(
            fixture, snapshotSurpriseAblation, false, true,
            std::string(EA::kCausalEconomicEventSurpriseAblationMaskText),
            77, 7, 1, "fnv1a64:67610f94f5c8e7cc");
        InsertFinalEvidence(fixture, snapshotSurpriseControl, 0.63, 0.07);
        InsertFinalEvidence(fixture, snapshotSurpriseAblation, 0.60, 0.04);
        fixture.commit();
    }

    const std::string before = DatabaseDigest(connection);
    std::ostringstream output;
    std::ostringstream errors;
    const int completeExit = Feature::RunComparisonCommand(
        connectionString, {{990601, 990602}}, output, errors);
    assert(completeExit == 0);
    assert(errors.str().empty());
    assert(output.str().find("disposition=comparable_complete") !=
           std::string::npos);
    assert(output.str().find("control_experiment_id=990602") !=
           std::string::npos);
    assert(output.str().find("ablation_experiment_id=990601") !=
           std::string::npos);
    assert(output.str().find(
               "expected_ablation_mask=relevant_event_has_consensus,") !=
           std::string::npos);
    assert(output.str().find(
               "metric=inference_accuracy,control=0.64000000000000001,") !=
           std::string::npos);
    assert(output.str().find(
               "control_minus_ablation=0.030000000000000027") !=
           std::string::npos);
    assert(output.str().find("control_minus_ablation=") !=
           std::string::npos);
    assert(output.str().find("read_only=true") != std::string::npos);
    assert(DatabaseDigest(connection) == before);

    output.str("");
    output.clear();
    errors.str("");
    errors.clear();
    const int surpriseExit = Feature::RunComparisonCommand(
        connectionString,
        {{990619, 990620},
         std::string(EA::kCausalEconomicEventSurpriseAblationMaskText)},
        output, errors);
    assert(surpriseExit == 0);
    assert(errors.str().empty());
    assert(output.str().find("control_experiment_id=990619") !=
           std::string::npos);
    assert(output.str().find("ablation_experiment_id=990620") !=
           std::string::npos);
    assert(output.str().find(
               "expected_ablation_mask=causal_first_release_surprise_available,"
               "causal_first_release_surprise") != std::string::npos);
    assert(output.str().find("model_input_width=77") != std::string::npos);
    assert(output.str().find(
               "model_input_semantic_layout_version=7") !=
           std::string::npos);
    assert(output.str().find(
               "delta_sign_convention=control_minus_ablation") !=
           std::string::npos);
    assert(output.str().find("economic_calendar_snapshot_id=NULL") !=
           std::string::npos);
    assert(DatabaseDigest(connection) == before);

    output.str("");
    output.clear();
    errors.str("");
    errors.clear();
    Replication::ComparisonCommand surpriseReplication;
    surpriseReplication.experimentIdPairs = {
        {990619, 990620}, {990671, 990672}};
    surpriseReplication.expectedAblationMask =
        std::string(EA::kCausalEconomicEventSurpriseAblationMaskText);
    const int surpriseReplicationExit = Replication::RunComparisonCommand(
        connectionString, surpriseReplication, output, errors);
    assert(surpriseReplicationExit == 0);
    assert(errors.str().empty());
    assert(output.str().find(
               "expected_ablation_mask=causal_first_release_surprise_available,"
               "causal_first_release_surprise") != std::string::npos);
    assert(output.str().find(
               "ordinal=1,control_experiment_id=990619,") !=
           std::string::npos);
    assert(output.str().find(
               "control_economic_calendar_snapshot_id=NULL") !=
           std::string::npos);
    assert(output.str().find(
               "ordinal=2,control_experiment_id=990671,") !=
           std::string::npos);
    assert(output.str().find(
               "control_economic_calendar_snapshot_id=1") !=
           std::string::npos);
    assert(output.str().find(
               "distinct_economic_calendar_corpus_count=2") !=
           std::string::npos);
    assert(output.str().find(
               "economic_calendar_snapshot_is_treatment=false") !=
           std::string::npos);
    assert(DatabaseDigest(connection) == before);

    // Different resume_model_id values are valid for feature-ablation pairs
    // only when each identifies that arm's own checkpoint at the same epoch.
    {
        pqxx::read_transaction read{connection};
        auto resumedControl =
            Feature::LoadAuthoritativeArmEvidence(read, 990601);
        auto resumedTreatment =
            Feature::LoadAuthoritativeArmEvidence(read, 990602);

        resumedControl.authoritative.configuration.resumeModelId = 9100601;
        resumedTreatment.authoritative.configuration.resumeModelId = 9100602;
        resumedControl.resumeCheckpointProvenance =
            Feature::ResumeCheckpointProvenance{
                9100601, 990601, true, 60};
        resumedTreatment.resumeCheckpointProvenance =
            Feature::ResumeCheckpointProvenance{
                9100602, 990602, true, 60};

        const auto matchedResume =
            Feature::CompareLegacyConsensusPair(resumedControl, resumedTreatment);
        assert(matchedResume.disposition ==
               Feature::Disposition::ComparableComplete);
        assert(matchedResume.invalidReasons.empty());

        resumedTreatment.resumeCheckpointProvenance->checkpointEpoch = 40;
        const auto mismatchedEpoch =
            Feature::CompareLegacyConsensusPair(resumedControl, resumedTreatment);
        assert(mismatchedEpoch.disposition ==
               Feature::Disposition::IncompatibleConfiguration);
        assert(std::find(
                   mismatchedEpoch.invalidReasons.begin(),
                   mismatchedEpoch.invalidReasons.end(),
                   "resume_checkpoint_epoch_mismatch") !=
               mismatchedEpoch.invalidReasons.end());

        resumedTreatment.resumeCheckpointProvenance->checkpointEpoch = 60;
        resumedTreatment.resumeCheckpointProvenance->
            ownExperimentCheckpoint = false;
        const auto unverifiedResume =
            Feature::CompareLegacyConsensusPair(resumedControl, resumedTreatment);
        assert(unverifiedResume.disposition ==
               Feature::Disposition::IncompatibleConfiguration);
        assert(std::find(
                   unverifiedResume.invalidReasons.begin(),
                   unverifiedResume.invalidReasons.end(),
                   "resume_model_id_mismatch") !=
               unverifiedResume.invalidReasons.end());

        resumedTreatment.authoritative.configuration.resumeModelId =
            std::nullopt;
        resumedTreatment.resumeCheckpointProvenance = std::nullopt;
        const auto freshVsResumed =
            Feature::CompareLegacyConsensusPair(resumedControl, resumedTreatment);
        assert(freshVsResumed.disposition ==
               Feature::Disposition::IncompatibleConfiguration);
        assert(std::find(
                   freshVsResumed.invalidReasons.begin(),
                   freshVsResumed.invalidReasons.end(),
                   "resume_model_id_mismatch") !=
               freshVsResumed.invalidReasons.end());
    }
    assert(DatabaseDigest(connection) == before);

    output.str("");
    output.clear();
    errors.str("");
    errors.clear();
    const int accuracyMismatchExit = Feature::RunComparisonCommand(
        connectionString, {{990661, 990662}}, output, errors);
    assert(accuracyMismatchExit == 3);
    assert(errors.str().empty());
    assert(output.str().find(
               "ablation_inference_analysis_accuracy_mismatch") !=
           std::string::npos);
    assert(DatabaseDigest(connection) == before);

    output.str("");
    output.clear();
    errors.str("");
    errors.clear();
    Replication::ComparisonCommand replicationCommand;
    replicationCommand.experimentIdPairs = {
        {990601, 990602}, {990611, 990612},
        {990631, 990632}, {990641, 990642}};
    const int replicationExit = Replication::RunComparisonCommand(
        connectionString, replicationCommand, output, errors);
    assert(replicationExit == 4);
    assert(errors.str().empty());
    assert(output.str().find(
               "replication_decision=insufficient_evidence") !=
           std::string::npos);
    assert(output.str().find(
               "productionization_software_ready=true") !=
           std::string::npos);
    assert(output.str().find(
               "productionization_action=await_replication") !=
           std::string::npos);
    assert(output.str().find("ordinal=1,control_experiment_id=990601") !=
           std::string::npos);
    assert(output.str().find("ordinal=4,control_experiment_id=990641") !=
           std::string::npos);
    assert(output.str().find("activation_performed=false") !=
           std::string::npos);
    assert(DatabaseDigest(connection) == before);

    output.str("");
    output.clear();
    errors.str("");
    errors.clear();
    const int incompleteExit = Feature::RunComparisonCommand(
        connectionString, {{990611, 990612}}, output, errors);
    assert(incompleteExit == 4);
    assert(errors.str().empty());
    assert(output.str().find("disposition=comparable_incomplete") !=
           std::string::npos);
    assert(DatabaseDigest(connection) == before);

    output.str("");
    output.clear();
    errors.str("");
    errors.clear();
    const int ambiguousExit = Feature::RunComparisonCommand(
        connectionString, {{990621, 990622}}, output, errors);
    assert(ambiguousExit == 3);
    assert(output.str().empty());
    assert(errors.str().find("disposition=ambiguous_final_inference") !=
           std::string::npos);
    assert(errors.str().find("final_inference_ambiguous") !=
           std::string::npos);
    assert(DatabaseDigest(connection) == before);

    output.str("");
    output.clear();
    errors.str("");
    errors.clear();
    const int checkpointOnlyExit = Feature::RunComparisonCommand(
        connectionString, {{990631, 990632}}, output, errors);
    assert(checkpointOnlyExit == 4);
    assert(errors.str().empty());
    assert(output.str().find("disposition=missing_final_inference") !=
           std::string::npos);
    assert(output.str().find("role=ablation,inference_result_id=NULL") !=
           std::string::npos);
    assert(DatabaseDigest(connection) == before);

    output.str("");
    output.clear();
    errors.str("");
    errors.clear();
    const int noProfitExit = Feature::RunComparisonCommand(
        connectionString, {{990641, 990642}}, output, errors);
    assert(noProfitExit == 4);
    assert(errors.str().empty());
    assert(output.str().find(
               "disposition=profitability_evidence_unavailable") !=
           std::string::npos);
    assert(output.str().find(
               "ablation_final_profitability_evidence_unavailable") !=
           std::string::npos);
    assert(DatabaseDigest(connection) == before);

    output.str("");
    output.clear();
    errors.str("");
    errors.clear();
    const int incompatibleExit = Feature::RunComparisonCommand(
        connectionString, {{990651, 990652}}, output, errors);
    assert(incompatibleExit == 3);
    assert(errors.str().empty());
    assert(output.str().find("disposition=incompatible_configuration") !=
           std::string::npos);
    assert(output.str().find("feature_warmup_scope_mismatch") !=
           std::string::npos);
    assert(DatabaseDigest(connection) == before);

    std::cout << "FeatureAblationPairEvaluationRepositoryTests passed\n";
    return 0;
}
