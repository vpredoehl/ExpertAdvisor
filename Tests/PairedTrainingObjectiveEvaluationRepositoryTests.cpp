#include "InferenceProfitabilityRepository.hpp"
#include "PairedTrainingObjectiveEvaluation.hpp"
#include "PairedTrainingObjectiveEvaluationRepository.hpp"
#include "PairedTrainingObjectiveEvaluationService.hpp"
#include "TrainingObjective.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace Pair = EA::PairedTrainingObjectiveEvaluation;
namespace Objective = EA::TrainingObjective;
namespace Profitability = EA::InferenceProfitability;

namespace
{

constexpr double kExperimentThreshold = 0.0008;
const double kModelThreshold =
    static_cast<double>(static_cast<float>(kExperimentThreshold));

struct FixtureIds
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

template <typename Value>
std::optional<Value> OptionalValue(const pqxx::row& row,
                                   const char* column)
{
    if (row[column].is_null()) return std::nullopt;
    return row[column].as<Value>();
}

bool Has(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool LoadFailsWith(pqxx::transaction_base& transaction,
                   long long experimentId,
                   const std::string& reason)
{
    try
    {
        (void)Pair::LoadAuthoritativeArmEvidence(transaction, experimentId);
    }
    catch (const Pair::EvidenceLoadError& error)
    {
        return error.reason().find(reason) != std::string::npos;
    }
    catch (const std::invalid_argument& error)
    {
        return std::string{error.what()}.find(reason) != std::string::npos;
    }
    return false;
}

void InsertMatrixRow(pqxx::transaction_base& transaction,
                     long long modelId,
                     const std::string& name,
                     const std::vector<double>& values)
{
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        transaction.exec(
            "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,n_rows,"
            "n_cols,value) VALUES($1,$2,0,$3,1,$4,$5);",
            pqxx::params{modelId, name, static_cast<int>(index),
                         static_cast<int>(values.size()), values[index]});
    }
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

void ReplaceAscii(pqxx::transaction_base& transaction,
                  long long modelId,
                  const std::string& name,
                  const std::string& value)
{
    transaction.exec(
        "DELETE FROM matrix WHERE model_id=$1 AND param_name=$2;",
        pqxx::params{modelId, name});
    InsertAscii(transaction, modelId, name, value);
}

void InsertExperiment(pqxx::transaction_base& transaction,
                      const FixtureIds& ids,
                      const Objective::Configuration& objective)
{
    const std::string canonical = Objective::CanonicalText(objective);
    const std::string hash = Objective::DeterministicHash(canonical);
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
        "regression_target_definition,regression_normalization_identity,"
        "robust_loss_definition,robust_loss_delta,target_clipping_definition,"
        "objective_normalization_identity) VALUES("
        "$1,'synthetic_phase4d_symbol',6,$2,120,25,80,20,"
        "'2020-01-01','2025-01-01','2025-01-02','2026-01-01',"
        "'completed','done',$3,NULL,$1,'enabled',20,'full_history_warmup','',"
        "false,'syntheticcommit','phase6',false,'Release','AppleClang-test',"
        "'079','scheduler-test','LSTM_Release',$4,$5,$6,$7,$8,$9,$10,"
        "$11,$12,$13,$14,$15,$16);",
        pqxx::params{
            ids.experiment,
            kExperimentThreshold,
            ids.model,
            objective.objectiveIdentifier,
            objective.objectiveVersion,
            objective.lossDefinitionVersion,
            canonical,
            hash,
            std::string{Objective::AuxiliaryLossModeText(
                objective.auxiliaryLossMode)},
            objective.auxiliaryLossCoefficient,
            objective.regressionTargetDefinition,
            objective.regressionNormalizationIdentity,
            objective.robustLossDefinition,
            objective.robustLossDelta,
            objective.targetClippingDefinition,
            std::string{"weighted_loss_sum_by_weight_sum_gradients__"
                        "calculate_batch_return_by_example_count_v1"}});

    transaction.exec(
        "INSERT INTO model(model_id,experiment_id) VALUES($1,$2);",
        pqxx::params{ids.model, ids.experiment});

    InsertMatrixRow(transaction, ids.model, "train_config_meta", {
        1.0, 6.0, kModelThreshold, 64.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 80.0, 120.0, 25.0, 2.5});
    InsertMatrixRow(transaction, ids.model, "target_meta",
                    {1.0, 1.0, 0.0, 0.0, 0.0, 1.0});
    InsertMatrixRow(transaction, ids.model, "model_meta", {1.0, 50.0, 64.0});
    InsertMatrixRow(transaction, ids.model, "model_input_semantics_meta",
                    {1.0, 1.0});
    InsertMatrixRow(transaction, ids.model, "optimizer_meta",
                    {1.0, 1.0, 100.0, 0.0, 0.0});
    InsertAscii(transaction, ids.model, "train_symbol_meta",
                "synthetic_phase4d_symbol");
    InsertAscii(transaction, ids.model, "train_range_meta",
                "2020-01-01|2025-01-01");
    transaction.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,n_rows,"
        "n_cols,value) VALUES($1,'param',0,0,114,256,0);",
        pqxx::params{ids.model});
    InsertAscii(transaction, ids.model, "training_objective_canonical_meta",
                canonical);
    InsertAscii(transaction, ids.model, "training_objective_hash_meta", hash);
    InsertAscii(transaction, ids.model, "feature_warmup_scope_meta",
                "full_history_warmup");
    InsertAscii(transaction, ids.model, "donchian20_mode_meta", "enabled");
    InsertAscii(transaction, ids.model, "donchian_lookback_meta", "20");

}

void InsertInference(pqxx::transaction_base& transaction,
                     const FixtureIds& ids,
                     long long inferenceId,
                     const std::string& scope = "final",
                     const std::optional<long long>& checkpointId = std::nullopt,
                     const std::string& fromDate = "2025-01-02",
                     const std::string& toDate = "2026-01-01")
{
    const std::optional<long long> parent =
        scope == "checkpoint"
            ? std::optional<long long>{ids.experiment}
            : std::nullopt;
    transaction.exec(
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,"
        "completed_epochs,accuracy,accept_model,pred_down,pred_neutral,pred_up) "
        "VALUES($1,$2,'completed',$3,$4,$5,'synthetic_phase4d_symbol',6,$6,"
        "64,1,1,$7,$8,80,0.70,true,0.20,0.30,0.50);",
        pqxx::params{inferenceId, ids.model, scope, checkpointId, parent,
                     kModelThreshold, fromDate, toDate});
}

void InsertAnalysis(pqxx::transaction_base& transaction,
                    const FixtureIds& ids)
{
    transaction.exec(
        "INSERT INTO experiment_analysis_result("
        "analysis_id,experiment_id,model_id,analysis_scope,analysis_status,"
        "infer_accuracy,accept_accuracy,accept_rate,leader_score) "
        "VALUES($1,$2,$3,'final','completed',0.70,0.75,0.70,0.63);",
        pqxx::params{ids.analysis, ids.experiment, ids.model});
}

Profitability::Observation PersistObservation(
    pqxx::transaction_base& transaction,
    const FixtureIds& ids,
    long long inferenceId,
    Profitability::Scope scope,
    const std::optional<long long>& checkpointId,
    const Profitability::Accumulator& accumulator,
    const std::string& fromDate = "2025-01-02",
    const std::string& toDate = "2026-01-01")
{
    Profitability::ObservationRequest request;
    request.provenance.experimentId = ids.experiment;
    request.provenance.modelId = ids.model;
    request.provenance.inferenceEvalResultId = inferenceId;
    request.provenance.scope = scope;
    request.provenance.checkpointEvalId = checkpointId;
    request.provenance.inferenceStart = fromDate;
    request.provenance.inferenceEnd = toDate;
    request.statistics = accumulator.statistics();
    request.sourceContentHash = accumulator.SourceContentHash();
    return Profitability::PersistObservationIdempotently(transaction, request)
        .observation;
}


Pair::ArmEvidence LoadArm(pqxx::transaction_base& transaction,
                          long long experimentId)
{
    return Pair::LoadAuthoritativeArmEvidence(transaction, experimentId);
}

Pair::MaterialityPolicy Policy()
{
    Pair::MaterialityPolicy policy;
    policy.minimumProfitabilityImprovement = 0.01;
    policy.maximumProfitabilityWorsening = 0.01;
    policy.classification = Pair::ClassificationDegradationPolicy{
        0.02, 0.02, 0.05, 0.02, 0.05};
    return policy;
}

Pair::ComparisonResult Compare(const Pair::ArmEvidence& control,
                               const Pair::ArmEvidence& treatment)
{
    return Pair::Compare(control, treatment, Policy());
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
        "(SELECT COALESCE(string_agg(row_to_json(x)::text,'|' ORDER BY "
        "model_id,param_name,row_idx,col_idx),'') FROM matrix x) || '#' || "
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
        RequiredEnvironment("EA_PHASE4D_VERIFY_DB_NAME");
    assert(std::regex_match(
        databaseName,
        std::regex{"^ea_phase4d_loader_[A-Za-z0-9_]+$"}));
    const std::string connectionString =
        "host=" + RequiredEnvironment("EA_PHASE4D_VERIFY_DB_HOST") +
        " port=" + RequiredEnvironment("EA_PHASE4D_VERIFY_DB_PORT") +
        " user=" + RequiredEnvironment("EA_PHASE4D_VERIFY_DB_USER") +
        " dbname=" + databaseName;
    pqxx::connection connection{connectionString};
    pqxx::work transaction{connection};

    const FixtureIds controlIds{990001, 1990001, 2990001, 4990001};
    const FixtureIds treatmentIds{990002, 1990002, 2990002, 4990002};
    InsertExperiment(transaction, controlIds, Objective::Legacy());
    InsertExperiment(
        transaction, treatmentIds, Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, controlIds, controlIds.inference);
    InsertInference(transaction, treatmentIds, treatmentIds.inference);
    InsertAnalysis(transaction, controlIds);
    InsertAnalysis(transaction, treatmentIds);

    Profitability::Accumulator controlReturns;
    controlReturns.Observe(Profitability::kNeutralClass, 100.0f, 110.0f);
    controlReturns.Observe(Profitability::kUpClass, 100.0f, 105.0f);
    Profitability::Accumulator treatmentReturns;
    treatmentReturns.Observe(Profitability::kNeutralClass, 100.0f, 110.0f);
    treatmentReturns.Observe(Profitability::kUpClass, 100.0f, 120.0f);
    const auto controlObservation = PersistObservation(
        transaction, controlIds, controlIds.inference,
        Profitability::Scope::finalInference, std::nullopt, controlReturns);
    const auto treatmentObservation = PersistObservation(
        transaction, treatmentIds, treatmentIds.inference,
        Profitability::Scope::finalInference, std::nullopt, treatmentReturns);

    const Pair::ArmEvidence control = LoadArm(transaction, controlIds.experiment);
    const Pair::ArmEvidence treatment =
        LoadArm(transaction, treatmentIds.experiment);
    assert(control.finalModelId == controlIds.model);
    assert(treatment.finalModelId == treatmentIds.model);
    assert(control.materializedModelObjectives.size() == 1);
    assert(treatment.materializedModelObjectives.size() == 1);
    assert(control.materializedModelObjectives[0].isFinalModel);
    assert(treatment.materializedModelObjectives[0].isFinalModel);
    assert(control.configuration.threshold == kExperimentThreshold);
    assert(treatment.configuration.threshold == kExperimentThreshold);
    assert(treatment.classification);
    assert(treatment.classification->threshold == kModelThreshold);
    assert(treatment.classification->threshold !=
           treatment.configuration.threshold);
    assert(std::fabs(treatment.classification->threshold -
                     treatment.configuration.threshold) <= 1.0e-7);
    assert(treatment.profitability);
    assert(treatment.profitability->observationId ==
           treatmentObservation.observationId);
    assert(treatment.profitability->inferenceResultId == treatmentIds.inference);
    assert(treatment.profitability->inferenceScope == "final");
    assert(!treatment.profitability->checkpointEvalId);
    assert(treatment.profitability->actionableCount == 1);
    assert(treatment.profitability
               ->averageTerminalHorizonLogReturnPerActionablePrediction);
    assert(treatment.classification->inferenceAccuracy == 0.70);
    assert(treatment.classification->acceptAccuracy == 0.75);
    assert(treatment.classification->acceptRate == 0.70);
    assert(treatment.classification->leaderScore == 0.63);
    assert(control.profitability->observationId == controlObservation.observationId);
    const auto positive = Compare(control, treatment);
    assert(positive.disposition == Pair::Disposition::Promising);
    assert(positive.invalidReasons.empty());
    assert(positive.incompleteReasons.empty());

    // Reversed role assignment is supported and deterministic; roles are not
    // silently reordered by objective identity or experiment id.
    const auto reversedFirst = Compare(treatment, control);
    const auto reversedSecond = Compare(treatment, control);
    assert(reversedFirst.disposition == reversedSecond.disposition);
    assert(reversedFirst.aggregateProfitability.treatmentMinusControl ==
           reversedSecond.aggregateProfitability.treatmentMinusControl);

    // Missing identities are distinct loader failures, while an existing but
    // unfinished arm is a scientific INCOMPLETE result.
    assert(LoadFailsWith(transaction, 99999991, "not_found"));
    const FixtureIds incompleteIds{990020, 1990020, 2990020, 4990020};
    InsertExperiment(transaction, incompleteIds,
                     Objective::ProfitabilityAuxiliary());
    transaction.exec(
        "UPDATE experiment SET status='running',phase='train' "
        "WHERE experiment_id=$1;", pqxx::params{incompleteIds.experiment});
    const auto incompleteArm = LoadArm(transaction, incompleteIds.experiment);
    assert(Compare(control, incompleteArm).disposition ==
           Pair::Disposition::Incomplete);

    const FixtureIds missingModelIds{990021, 1990021, 2990021, 4990021};
    InsertExperiment(transaction, missingModelIds,
                     Objective::ProfitabilityAuxiliary());
    transaction.exec(
        "UPDATE experiment SET last_model_id=NULL WHERE experiment_id=$1;",
        pqxx::params{missingModelIds.experiment});
    const auto missingModel = LoadArm(transaction, missingModelIds.experiment);
    assert(!missingModel.finalModelId);
    assert(Compare(control, missingModel).disposition ==
           Pair::Disposition::Incomplete);

    const FixtureIds missingInferenceIds{990022, 1990022, 2990022, 4990022};
    InsertExperiment(transaction, missingInferenceIds,
                     Objective::ProfitabilityAuxiliary());
    InsertAnalysis(transaction, missingInferenceIds);
    const auto missingInference =
        LoadArm(transaction, missingInferenceIds.experiment);
    assert(!missingInference.classification);
    assert(Compare(control, missingInference).disposition ==
           Pair::Disposition::Incomplete);

    const FixtureIds missingAnalysisIds{990023, 1990023, 2990023, 4990023};
    InsertExperiment(transaction, missingAnalysisIds,
                     Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, missingAnalysisIds,
                    missingAnalysisIds.inference);
    PersistObservation(transaction, missingAnalysisIds,
                       missingAnalysisIds.inference,
                       Profitability::Scope::finalInference, std::nullopt,
                       treatmentReturns);
    const auto missingAnalysis = LoadArm(transaction, missingAnalysisIds.experiment);
    assert(!missingAnalysis.classification);
    assert(!missingAnalysis.profitability);
    assert(Compare(control, missingAnalysis).disposition ==
           Pair::Disposition::Incomplete);

    const FixtureIds checkpointInferenceOnlyIds{
        990024, 1990024, 2990024, 4990024};
    InsertExperiment(transaction, checkpointInferenceOnlyIds,
                     Objective::ProfitabilityAuxiliary());
    const long long checkpointInferenceOnlyCheckpointId = 5990024;
    transaction.exec(
        "INSERT INTO experiment_checkpoint_eval("
        "checkpoint_eval_id,parent_experiment_id,checkpoint_model_id) "
        "VALUES($1,$2,$3);",
        pqxx::params{checkpointInferenceOnlyCheckpointId,
                     checkpointInferenceOnlyIds.experiment,
                     checkpointInferenceOnlyIds.model});
    InsertInference(transaction, checkpointInferenceOnlyIds,
                    checkpointInferenceOnlyIds.inference, "checkpoint",
                    checkpointInferenceOnlyCheckpointId);
    const auto checkpointInferenceOnly =
        LoadArm(transaction, checkpointInferenceOnlyIds.experiment);
    assert(!checkpointInferenceOnly.classification);

    // Same-objective experiments remain a scientific INVALID_COMPARISON, not
    // a loader failure.
    const FixtureIds sameObjectiveIds{990025, 1990025, 2990025, 4990025};
    InsertExperiment(transaction, sameObjectiveIds, Objective::Legacy());
    InsertInference(transaction, sameObjectiveIds, sameObjectiveIds.inference);
    InsertAnalysis(transaction, sameObjectiveIds);
    PersistObservation(transaction, sameObjectiveIds, sameObjectiveIds.inference,
                       Profitability::Scope::finalInference, std::nullopt,
                       controlReturns);
    assert(Has(Compare(control,
                       LoadArm(transaction, sameObjectiveIds.experiment))
                   .invalidReasons,
               "training_objective_same"));

    // Checkpoint profitability cannot fill a missing final observation.
    const FixtureIds checkpointOnlyIds{990011, 1990011, 2990011, 4990011};
    InsertExperiment(transaction, checkpointOnlyIds,
                     Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, checkpointOnlyIds, checkpointOnlyIds.inference);
    InsertAnalysis(transaction, checkpointOnlyIds);
    const long long checkpointId = 5990011;
    const long long checkpointInferenceId = 2990111;
    transaction.exec(
        "INSERT INTO experiment_checkpoint_eval("
        "checkpoint_eval_id,parent_experiment_id,checkpoint_model_id) "
        "VALUES($1,$2,$3);",
        pqxx::params{checkpointId, checkpointOnlyIds.experiment,
                     checkpointOnlyIds.model});
    InsertInference(transaction, checkpointOnlyIds, checkpointInferenceId,
                    "checkpoint", checkpointId);
    PersistObservation(transaction, checkpointOnlyIds, checkpointInferenceId,
                       Profitability::Scope::checkpointInference, checkpointId,
                       treatmentReturns);
    const auto checkpointOnly =
        LoadArm(transaction, checkpointOnlyIds.experiment);
    assert(checkpointOnly.classification);
    assert(!checkpointOnly.profitability);
    const auto checkpointOnlyResult = Compare(control, checkpointOnly);
    assert(checkpointOnlyResult.disposition == Pair::Disposition::Incomplete);
    assert(Has(checkpointOnlyResult.incompleteReasons,
               "treatment_final_profitability_missing"));

    // A final inference row plus only a differently linked final observation
    // remains incomplete; there is no recency or model-only fallback.
    const FixtureIds noProfitIds{990012, 1990012, 2990012, 4990012};
    InsertExperiment(transaction, noProfitIds,
                     Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, noProfitIds, noProfitIds.inference);
    InsertAnalysis(transaction, noProfitIds);
    const long long wrongContextInferenceId = 2991012;
    InsertInference(transaction, noProfitIds, wrongContextInferenceId,
                    "final", std::nullopt, "2025-02-01", "2026-02-01");
    PersistObservation(transaction, noProfitIds, wrongContextInferenceId,
                       Profitability::Scope::finalInference, std::nullopt,
                       treatmentReturns, "2025-02-01", "2026-02-01");
    const auto noProfit = LoadArm(transaction, noProfitIds.experiment);
    assert(noProfit.classification);
    assert(!noProfit.profitability);
    assert(Compare(control, noProfit).disposition ==
           Pair::Disposition::Incomplete);

    // More than one exact final inference row fails closed without choosing
    // the newest or smallest identity.
    const FixtureIds ambiguousIds{990013, 1990013, 2990013, 4990013};
    InsertExperiment(transaction, ambiguousIds,
                     Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, ambiguousIds, ambiguousIds.inference);
    InsertInference(transaction, ambiguousIds, 2991013);
    InsertAnalysis(transaction, ambiguousIds);
    assert(LoadFailsWith(transaction, ambiguousIds.experiment,
                         "final_inference_ambiguous"));

    // Zero actionable predictions preserve aggregate zero and NULL average.
    const FixtureIds zeroIds{990014, 1990014, 2990014, 4990014};
    InsertExperiment(transaction, zeroIds, Objective::ProfitabilityAuxiliary());
    InsertInference(transaction, zeroIds, zeroIds.inference);
    InsertAnalysis(transaction, zeroIds);
    Profitability::Accumulator zeroReturns;
    zeroReturns.Observe(Profitability::kNeutralClass, 100.0f, 120.0f);
    PersistObservation(transaction, zeroIds, zeroIds.inference,
                       Profitability::Scope::finalInference, std::nullopt,
                       zeroReturns);
    const auto zero = LoadArm(transaction, zeroIds.experiment);
    assert(zero.profitability);
    assert(zero.profitability->actionableCount == 0);
    assert(zero.profitability->aggregateTerminalHorizonLogReturnSum == 0.0);
    assert(!zero.profitability
                ->averageTerminalHorizonLogReturnPerActionablePrediction);
    const auto zeroResult = Compare(control, zero);
    assert(zeroResult.invalidReasons.empty());
    assert(zeroResult.incompleteReasons.empty());

    // Hostile persisted experiment hash is rejected by the evaluator.  The
    // migration constraint is deliberately dropped only inside this test
    // transaction; the transaction is aborted below.
    transaction.exec(
        "ALTER TABLE experiment DROP CONSTRAINT "
        "experiment_training_objective_provenance_check;");
    transaction.exec(
        "UPDATE experiment SET training_objective_hash="
        "'fnv1a64:0000000000000000' WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    assert(LoadFailsWith(transaction, treatmentIds.experiment,
                         "experiment_objective_provenance_mismatch"));
    const std::string auxiliaryCanonical =
        Objective::CanonicalText(Objective::ProfitabilityAuxiliary());
    const std::string auxiliaryHash =
        Objective::DeterministicHash(auxiliaryCanonical);
    transaction.exec(
        "UPDATE experiment SET training_objective_canonical=$2,"
        "training_objective_hash=$3 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment, auxiliaryCanonical, auxiliaryHash});

    // A numerically close but noncanonical objective text remains invalid even
    // with its matching hash; canonical representation is authoritative.
    std::string closeCanonical = auxiliaryCanonical;
    const std::string needle = "auxiliary_loss_coefficient=0.1;";
    const std::size_t coefficient = closeCanonical.find(needle);
    assert(coefficient != std::string::npos);
    closeCanonical.replace(coefficient, needle.size(),
                           "auxiliary_loss_coefficient=0.10000000000000001;");
    transaction.exec(
        "UPDATE experiment SET training_objective_canonical=$2,"
        "training_objective_hash=$3 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment, closeCanonical,
                     Objective::DeterministicHash(closeCanonical)});
    assert(LoadFailsWith(transaction, treatmentIds.experiment, ""));
    transaction.exec(
        "UPDATE experiment SET training_objective_canonical=$2,"
        "training_objective_hash=$3 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment, auxiliaryCanonical, auxiliaryHash});

    const std::string legacyCanonical =
        Objective::CanonicalText(Objective::Legacy());
    const std::string legacyHash = Objective::DeterministicHash(legacyCanonical);
    ReplaceAscii(transaction, treatmentIds.model,
                 "training_objective_canonical_meta", legacyCanonical);
    ReplaceAscii(transaction, treatmentIds.model,
                 "training_objective_hash_meta", legacyHash);
    const auto modelMismatch = LoadArm(transaction, treatmentIds.experiment);
    const auto modelMismatchResult = Compare(control, modelMismatch);
    assert(modelMismatchResult.disposition == Pair::Disposition::InvalidComparison);
    assert(Has(modelMismatchResult.invalidReasons,
               "treatment_experiment_model_objective_mismatch"));
    ReplaceAscii(transaction, treatmentIds.model,
                 "training_objective_canonical_meta", auxiliaryCanonical);
    ReplaceAscii(transaction, treatmentIds.model,
                 "training_objective_hash_meta", auxiliaryHash);

    transaction.exec(
        "UPDATE experiment SET symbol='synthetic_other_symbol' "
        "WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "symbol_mismatch"));
    transaction.exec(
        "UPDATE experiment SET symbol='synthetic_phase4d_symbol' "
        "WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});

    transaction.exec(
        "UPDATE experiment SET prediction_horizon=12 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    transaction.exec(
        "UPDATE matrix SET value=12 WHERE model_id=$1 "
        "AND param_name='train_config_meta' AND col_idx=1;",
        pqxx::params{treatmentIds.model});
    transaction.exec(
        "UPDATE inference_eval_result SET prediction_horizon=12 "
        "WHERE id=$1;", pqxx::params{treatmentIds.inference});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "prediction_horizon_mismatch"));
    transaction.exec(
        "UPDATE experiment SET prediction_horizon=6 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    transaction.exec(
        "UPDATE matrix SET value=6 WHERE model_id=$1 "
        "AND param_name='train_config_meta' AND col_idx=1;",
        pqxx::params{treatmentIds.model});
    transaction.exec(
        "UPDATE inference_eval_result SET prediction_horizon=6 "
        "WHERE id=$1;", pqxx::params{treatmentIds.inference});

    transaction.exec(
        "UPDATE experiment SET c_next_threshold=0.001 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    transaction.exec(
        "UPDATE matrix SET value=0.001 WHERE model_id=$1 "
        "AND param_name='train_config_meta' AND col_idx=2;",
        pqxx::params{treatmentIds.model});
    transaction.exec(
        "UPDATE inference_eval_result SET threshold_logret=0.001 "
        "WHERE id=$1;", pqxx::params{treatmentIds.inference});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "threshold_mismatch"));
    transaction.exec(
        "UPDATE experiment SET c_next_threshold=$2 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment, kExperimentThreshold});
    transaction.exec(
        "UPDATE matrix SET value=$2 WHERE model_id=$1 "
        "AND param_name='train_config_meta' AND col_idx=2;",
        pqxx::params{treatmentIds.model, kModelThreshold});
    transaction.exec(
        "UPDATE inference_eval_result SET threshold_logret=$2 WHERE id=$1;",
        pqxx::params{treatmentIds.inference, kModelThreshold});

    transaction.exec(
        "UPDATE experiment SET c_next_threshold=$2 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment,
                     kExperimentThreshold + 1.0000001e-7});
    assert(LoadFailsWith(transaction, treatmentIds.experiment,
                         "final_model_training_context_mismatch"));
    transaction.exec(
        "UPDATE experiment SET c_next_threshold=$2 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment, kExperimentThreshold});

    transaction.exec(
        "UPDATE experiment SET core_lr_mult=121 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "core_lr_mismatch"));
    transaction.exec(
        "UPDATE experiment SET core_lr_mult=120 WHERE experiment_id=$1;",
        pqxx::params{treatmentIds.experiment});

    transaction.exec(
        "UPDATE matrix SET value=2.5001 WHERE model_id=$1 "
        "AND param_name='train_config_meta' AND row_idx=0 AND col_idx=13;",
        pqxx::params{treatmentIds.model});
    assert(Has(Compare(control, LoadArm(transaction, treatmentIds.experiment))
                   .invalidReasons,
               "persisted_head_bias_lr_mismatch"));
    transaction.exec(
        "UPDATE matrix SET value=2.5 WHERE model_id=$1 "
        "AND param_name='train_config_meta' AND row_idx=0 AND col_idx=13;",
        pqxx::params{treatmentIds.model});

    // The Phase 1 trigger rejects a profitability row whose declared
    // experiment/model provenance does not own its inference result.
    bool wrongInferenceLinkRejected = false;
    try
    {
        pqxx::subtransaction hostile{transaction, "wrong_inference_link"};
        Profitability::ObservationRequest request;
        request.provenance.experimentId = treatmentIds.experiment;
        request.provenance.modelId = treatmentIds.model;
        request.provenance.inferenceEvalResultId = controlIds.inference;
        request.provenance.scope = Profitability::Scope::finalInference;
        request.provenance.inferenceStart = "2025-01-02";
        request.provenance.inferenceEnd = "2026-01-01";
        request.statistics = treatmentReturns.statistics();
        request.sourceContentHash = "fnv1a64:0000000000000001";
        (void)Profitability::PersistObservationIdempotently(hostile, request);
        hostile.commit();
    }
    catch (const pqxx::sql_error&)
    {
        wrongInferenceLinkRejected = true;
    }
    assert(wrongInferenceLinkRejected);

    // Commit synthetic fixtures solely so the production service can read
    // them through a separate libpqxx read_transaction.  The enclosing shell
    // owns and drops the isolated cluster/database.
    transaction.commit();

    const std::string before = DatabaseDigest(connection);
    Pair::ComparisonCommand command;
    command.experimentIds = {controlIds.experiment, treatmentIds.experiment};
    command.policy = Policy();
    std::ostringstream firstOutput;
    std::ostringstream firstErrors;
    assert(Pair::RunComparisonCommand(
               connectionString, command, firstOutput, firstErrors) == 0);
    assert(firstErrors.str().empty());
    assert(firstOutput.str().find("TRAINING_OBJECTIVE_PAIR_COMPARISON") !=
           std::string::npos);
    assert(firstOutput.str().find("disposition=PROMISING") !=
           std::string::npos);
    std::ostringstream secondOutput;
    std::ostringstream secondErrors;
    assert(Pair::RunComparisonCommand(
               connectionString, command, secondOutput, secondErrors) == 0);
    assert(firstOutput.str() == secondOutput.str());
    assert(secondErrors.str().empty());

    Pair::ComparisonCommand missingControl = command;
    missingControl.experimentIds.first = 99999991;
    std::ostringstream missingControlOutput;
    std::ostringstream missingControlErrors;
    assert(Pair::RunComparisonCommand(
               connectionString, missingControl, missingControlOutput,
               missingControlErrors) == 3);
    assert(missingControlOutput.str().empty());
    assert(missingControlErrors.str().find("not_found") != std::string::npos);

    Pair::ComparisonCommand missingTreatment = command;
    missingTreatment.experimentIds.second = 99999992;
    std::ostringstream missingTreatmentOutput;
    std::ostringstream missingTreatmentErrors;
    assert(Pair::RunComparisonCommand(
               connectionString, missingTreatment, missingTreatmentOutput,
               missingTreatmentErrors) == 3);
    assert(missingTreatmentOutput.str().empty());
    assert(missingTreatmentErrors.str().find("not_found") !=
           std::string::npos);

    const std::string after = DatabaseDigest(connection);
    assert(before == after);
    std::cout << "paired_training_objective_repository_tests_passed\n";
    return 0;
}
