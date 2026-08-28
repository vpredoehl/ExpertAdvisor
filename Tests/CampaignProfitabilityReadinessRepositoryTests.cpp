#include "../Sources/ExperimentRecommendation.hpp"
#include "../Sources/InferenceProfitabilityRepository.hpp"
#include "../Sources/ProfitabilityVerificationService.hpp"

#include <cassert>
#include <cstdlib>
#include <optional>
#include <sstream>
#include <string>

#include <pqxx/pqxx>

namespace Profitability = EA::InferenceProfitability;
namespace Verification = EA::ProfitabilityVerification;
namespace Recommendation = EA::ExperimentRecommendation;

namespace
{

std::string Environment(const char* name)
{
    const char* value = std::getenv(name);
    assert(value && *value);
    return value;
}

std::string ConnectionString(const std::string& user)
{
    return "hostaddr=" + Environment("LSTM_DB_HOST") +
        " user=" + user + " dbname=" + Environment("LSTM_DB_NAME");
}

void InsertContext(pqxx::transaction_base& transaction,
                   long long experimentId,
                   long long modelId,
                   long long resultId)
{
    transaction.exec(
        "INSERT INTO experiment("
        "experiment_id,symbol,prediction_horizon,c_next_threshold,"
        "infer_start,infer_end,last_model_id,target_epochs,status,phase,"
        "updated_at) VALUES($1,'EURUSDRMP',5,0.001,'2025-01-01',"
        "'2025-12-31',$2,100,'completed','analyze',now());",
        pqxx::params{experimentId, modelId});
    transaction.exec(
        "INSERT INTO model(model_id,experiment_id) VALUES($1,$2);",
        pqxx::params{modelId, experimentId});
    transaction.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "SELECT $1,'train_config_meta',0,ordinality-1,value "
        "FROM unnest(ARRAY[1,5,0.001,64,1,1,1,1,1,1,100,1,1,1]"
        "::double precision[]) WITH ORDINALITY AS config(value,ordinality);",
        pqxx::params{modelId});
    transaction.exec(
        "INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value) "
        "VALUES($1,'target_meta',0,0,1);",
        pqxx::params{modelId});
    transaction.exec(
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,"
        "completed_epochs,accept_model) VALUES("
        "$1,$2,'completed','final',NULL,NULL,'EURUSDRMP',5,0.001,64,1,1,"
        "'2025-01-01','2025-12-31',100,true);",
        pqxx::params{resultId, modelId});
}

Profitability::PersistResult Persist(pqxx::transaction_base& transaction,
                                     long long experimentId,
                                     long long modelId,
                                     long long resultId,
                                     bool positive)
{
    Profitability::Accumulator values;
    values.Observe(
        positive ? Profitability::kUpClass : Profitability::kDownClass,
        100.0f,
        110.0f);
    Profitability::ObservationRequest request;
    request.provenance.experimentId = experimentId;
    request.provenance.modelId = modelId;
    request.provenance.inferenceEvalResultId = resultId;
    request.provenance.scope = Profitability::Scope::finalInference;
    request.provenance.inferenceStart = "2025-01-01";
    request.provenance.inferenceEnd = "2025-12-31";
    request.statistics = values.statistics();
    request.sourceContentHash = values.SourceContentHash();
    return Profitability::PersistObservationIdempotently(transaction, request);
}

void InsertSnapshot(pqxx::transaction_base& transaction,
                    long long snapshotId,
                    int memberCount)
{
    const std::string identity =
        "profitability-readiness-snapshot-" + std::to_string(snapshotId);
    const std::string policy = "profitability-readiness-ranking-policy";
    const std::string scoring = "profitability-readiness-scoring-semantic";
    const std::string evaluation =
        "profitability-readiness-evaluation-semantic";
    transaction.exec(
        "INSERT INTO experiment_recommendation_ranking_snapshot VALUES("
        "$1,'completed',$2,$3,$4,$5,1,'verified_homogeneous',$6,$7,1,"
        "$8,$9,1,1,1,'verified_homogeneous',$10);",
        pqxx::params{
            snapshotId,
            identity, Recommendation::RecommendationCanonicalHash(identity),
            policy, Recommendation::RecommendationCanonicalHash(policy),
            scoring, Recommendation::RecommendationCanonicalHash(scoring),
            evaluation, Recommendation::RecommendationCanonicalHash(evaluation),
            memberCount});
}

void InsertRecommendation(
    pqxx::transaction_base& transaction,
    long long recommendationId,
    long long experimentId,
    long long modelId,
    long long resultId,
    const std::optional<Profitability::Observation>& observation,
    const std::string& unavailableReason,
    bool frozenMismatch = false)
{
    const std::string semantic =
        "profitability-readiness-semantic-" +
        std::to_string(recommendationId);
    const std::string invocation =
        "profitability-readiness-invocation-" +
        std::to_string(recommendationId);
    transaction.exec(
        "INSERT INTO experiment_recommendation("
        "recommendation_id,source_experiment_id,source_model_id,source_symbol,"
        "source_prediction_horizon,changed_parameter,source_leader_score,"
        "source_infer_accuracy,source_predicted_neutral_proportion,"
        "semantic_configuration_canonical,semantic_hash,"
        "invocation_configuration_canonical,invocation_hash,"
        "final_profitability_provenance_version,"
        "source_final_inference_eval_result_id,"
        "source_final_profitability_observation_id,"
        "source_final_profitability_unavailable_reason,"
        "source_final_profitability_inference_scope,"
        "source_final_profitability_inference_start,"
        "source_final_profitability_inference_end,"
        "source_final_profitability_actionable_count,"
        "source_final_profitability_aggregate_return,"
        "source_final_profitability_average_return,"
        "source_final_profitability_metric_definition_hash,"
        "source_final_profitability_source_content_hash,"
        "source_final_profitability_observation_identity_hash) VALUES("
        "$1,$2,$3,'EURUSDRMP',5,'core_lr_mult',0.9,0.8,0.1,$4,$5,$6,$7,"
        "1,$8,$9,$10,'final',$11,$12,$13,$14,$15,$16,$17,$18);",
        pqxx::params{
            recommendationId, experimentId, modelId,
            semantic, Recommendation::RecommendationCanonicalHash(semantic),
            invocation, Recommendation::RecommendationCanonicalHash(invocation),
            resultId,
            observation
                ? std::optional<long long>{observation->observationId}
                : std::nullopt,
            unavailableReason.empty()
                ? std::optional<std::string>{}
                : std::optional<std::string>{unavailableReason},
            observation
                ? std::optional<std::string>{
                      observation->provenance.inferenceStart}
                : std::nullopt,
            observation
                ? std::optional<std::string>{observation->provenance.inferenceEnd}
                : std::nullopt,
            observation
                ? std::optional<long long>{static_cast<long long>(
                      observation->statistics.actionableCount)}
                : std::nullopt,
            observation
                ? std::optional<double>{observation->statistics
                      .aggregateTerminalHorizonLogReturnSum}
                : std::nullopt,
            observation
                ? observation
                      ->averageTerminalHorizonLogReturnPerActionablePrediction
                : std::nullopt,
            observation
                ? std::optional<std::string>{observation->metricDefinitionHash}
                : std::nullopt,
            observation
                ? std::optional<std::string>{observation->sourceContentHash}
                : std::nullopt,
            observation
                ? std::optional<std::string>{
                      frozenMismatch
                          ? Profitability::DeterministicHash("mismatch")
                          : observation->observationIdentityHash}
                : std::nullopt});
}

void InsertMember(pqxx::transaction_base& transaction,
                  long long memberId,
                  long long snapshotId,
                  long long recommendationId,
                  long long experimentId,
                  long long modelId,
                  int rank,
                  double score)
{
    transaction.exec(
        "INSERT INTO experiment_recommendation_ranking_member VALUES("
        "$1,$2,$3,$4,$5,$6,'advisory_ready',$7,'EURUSDRMP',5,"
        "'core_lr_mult');",
        pqxx::params{memberId, snapshotId, recommendationId, experimentId,
                     modelId, rank, score});
}

void Setup()
{
    pqxx::connection connection{
        ConnectionString(Environment("LSTM_DB_ADMIN_USER"))};
    pqxx::work transaction{connection};
    InsertContext(transaction, 1, 10, 101);
    InsertContext(transaction, 2, 20, 201);
    InsertContext(transaction, 3, 30, 301);
    const auto positive = Persist(transaction, 1, 10, 101, true).observation;
    const auto negative = Persist(transaction, 2, 20, 201, false).observation;

    InsertSnapshot(transaction, 7, 3);
    InsertRecommendation(transaction, 1001, 1, 10, 101, positive, "");
    InsertRecommendation(transaction, 1002, 2, 20, 201, negative, "");
    InsertRecommendation(
        transaction, 1003, 3, 30, 301, std::nullopt,
        "no_profitability_observation");
    InsertMember(transaction, 7001, 7, 1002, 2, 20, 1, 0.99);
    InsertMember(transaction, 7002, 7, 1001, 1, 10, 2, 0.50);
    InsertMember(transaction, 7003, 7, 1003, 3, 30, 3, 0.75);

    InsertSnapshot(transaction, 8, 1);
    InsertRecommendation(
        transaction, 1004, 1, 10, 101, positive, "", true);
    InsertMember(transaction, 8001, 8, 1004, 1, 10, 1, 0.90);
    transaction.commit();
}

void Verify()
{
    const std::string connectionString =
        ConnectionString(Environment("LSTM_DB_USER"));
    std::ostringstream output;
    std::ostringstream errors;
    assert(Verification::RunCampaignReadinessCommand(
               connectionString, 7, output, errors) == 4);
    const std::string rendered = output.str();
    assert(errors.str().empty());
    assert(rendered.find(
        "CAMPAIGN_PROFITABILITY_COVERAGE,ranking_snapshot_id=7,"
        "candidate_count=3,exact_final_profitability_count=2,"
        "missing_profitability_count=1") != std::string::npos);
    assert(rendered.find("profitability_sign=positive") != std::string::npos);
    assert(rendered.find("profitability_sign=negative") != std::string::npos);
    assert(rendered.find("profitability_sign=unavailable") != std::string::npos);
    assert(rendered.find(
        "ranking_member_id=7002,recommendation_id=1001") != std::string::npos);
    assert(rendered.find(
        "current_rank=2,current_score=0.5") != std::string::npos);
    assert(rendered.find("profitability_shadow_rank=1,rank_delta=1,"
                         "live_rank_authoritative=true") != std::string::npos);
    assert(rendered.find("live_profitability_weight=0") != std::string::npos);
    assert(rendered.find(
        "live_profitability_score_contribution=0") != std::string::npos);
    assert(rendered.find("current_rank_changed=false") != std::string::npos);
    assert(rendered.find(
        "campaign_profitability_contract_ready=true") != std::string::npos);
    assert(rendered.find(
        "profitability_shadow_ranking_ready=true") != std::string::npos);
    assert(rendered.find(
        "campaign_profitability_activation_performed=false") !=
           std::string::npos);
    assert(rendered.find("live_ranking_changed=false,read_only=true") !=
           std::string::npos);

    output.str({});
    output.clear();
    assert(Verification::RunCampaignReadinessCommand(
               connectionString, 8, output, errors) == 3);
    assert(output.str().find(
        "profitability_evidence_status=invalid_provenance") !=
           std::string::npos);
    assert(output.str().find(
        "campaign_profitability_contract_ready=false") !=
           std::string::npos);
    assert(output.str().find(
        "campaign_profitability_action=blocked_evidence_contract") !=
           std::string::npos);
    assert(output.str().find(
        "campaign_profitability_activation_performed=false") !=
           std::string::npos);

    pqxx::connection verifyConnection{connectionString};
    pqxx::read_transaction transaction{verifyConnection};
    assert(transaction.exec("SHOW transaction_read_only;")
               .one_row()[0].as<std::string>() == "on");
    assert(transaction.exec(
        "SELECT count(*) FROM experiment_recommendation;")
               .one_row()[0].as<int>() == 4);
    assert(transaction.exec(
        "SELECT count(*) FROM experiment_recommendation_ranking_member;")
               .one_row()[0].as<int>() == 4);
    const pqxx::row sentinel = transaction.exec(
        "SELECT last_value,is_called FROM campaign_read_sentinel;").one_row();
    assert(sentinel["last_value"].as<long long>() == 1);
    assert(!sentinel["is_called"].as<bool>());
}

} // namespace

int main(int argc, char** argv)
{
    assert(argc == 2);
    const std::string mode = argv[1];
    if (mode == "setup")
        Setup();
    else
    {
        assert(mode == "verify");
        Verify();
    }
}
