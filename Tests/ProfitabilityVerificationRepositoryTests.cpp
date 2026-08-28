#include "../Sources/InferenceProfitabilityRepository.hpp"
#include "../Sources/ProfitabilityVerificationRepository.hpp"

#include <cassert>
#include <cstdlib>
#include <string>

#include <pqxx/pqxx>

namespace Profitability = EA::InferenceProfitability;
namespace Verification = EA::ProfitabilityVerification;

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
                   long long modelId)
{
    transaction.exec(
        "INSERT INTO experiment(experiment_id,symbol,prediction_horizon,"
        "c_next_threshold,infer_start,infer_end,last_model_id) "
        "VALUES($1,'USDJPYRMP',5,0.001,'2025-01-01','2025-12-31',$2);",
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
        "INSERT INTO experiment_analysis_result("
        "experiment_analysis_result_id,experiment_id,model_id) "
        "VALUES($1,$2,$3);",
        pqxx::params{9000 + experimentId, experimentId, modelId});
}

void InsertInferenceResult(pqxx::transaction_base& transaction,
                           long long resultId,
                           long long modelId,
                           const std::string& scope = "final",
                           const std::optional<long long>& checkpointId =
                               std::nullopt,
                           const std::optional<long long>& parentId =
                               std::nullopt)
{
    transaction.exec(
        "INSERT INTO inference_eval_result("
        "id,model_id,status,inference_scope,checkpoint_eval_id,"
        "parent_experiment_id,symbol,prediction_horizon,threshold_logret,"
        "window_size,label_rule_id,target_type,from_date,to_date,"
        "completed_epochs,accept_model) "
        "VALUES($1,$2,'completed',$3,$4,$5,'USDJPYRMP',5,0.001,64,1,1,"
        "'2025-01-01','2025-12-31',100,true);",
        pqxx::params{resultId, modelId, scope, checkpointId, parentId});
}

Profitability::ObservationRequest Request(long long experimentId,
                                          long long modelId,
                                          long long resultId,
                                          const Profitability::Accumulator& data,
                                          Profitability::Scope scope =
                                              Profitability::Scope::finalInference,
                                          const std::optional<long long>& checkpointId =
                                              std::nullopt)
{
    Profitability::ObservationRequest request;
    request.provenance.experimentId = experimentId;
    request.provenance.modelId = modelId;
    request.provenance.inferenceEvalResultId = resultId;
    request.provenance.scope = scope;
    request.provenance.checkpointEvalId = checkpointId;
    request.provenance.inferenceStart = "2025-01-01";
    request.provenance.inferenceEnd = "2025-12-31";
    request.statistics = data.statistics();
    request.sourceContentHash = data.SourceContentHash();
    return request;
}

void Setup()
{
    pqxx::connection connection{ConnectionString(Environment("LSTM_DB_ADMIN_USER"))};
    pqxx::work transaction{connection};
    for (long long experimentId = 1; experimentId <= 5; ++experimentId)
        InsertContext(transaction, experimentId, experimentId * 10);

    InsertInferenceResult(transaction, 101, 10);
    InsertInferenceResult(transaction, 201, 20);
    transaction.exec(
        "INSERT INTO experiment_checkpoint_eval("
        "checkpoint_eval_id,parent_experiment_id,checkpoint_model_id) "
        "VALUES(220,2,20);");
    InsertInferenceResult(transaction, 202, 20, "checkpoint", 220, 2);
    InsertInferenceResult(transaction, 401, 40);
    InsertInferenceResult(transaction, 402, 40);
    InsertInferenceResult(transaction, 501, 50);

    Profitability::Accumulator positive;
    positive.Observe(Profitability::kUpClass, 100.0f, 110.0f);
    positive.Observe(Profitability::kNeutralClass, 100.0f, 100.0f);
    (void)Profitability::PersistObservationIdempotently(
        transaction, Request(1, 10, 101, positive));

    Profitability::Accumulator checkpoint;
    checkpoint.Observe(Profitability::kDownClass, 100.0f, 110.0f);
    (void)Profitability::PersistObservationIdempotently(
        transaction,
        Request(2, 20, 202, checkpoint,
                Profitability::Scope::checkpointInference, 220));

    Profitability::Accumulator first;
    first.Observe(Profitability::kUpClass, 100.0f, 105.0f);
    Profitability::Accumulator second;
    second.Observe(Profitability::kUpClass, 100.0f, 106.0f);
    (void)Profitability::PersistObservationIdempotently(
        transaction, Request(5, 50, 501, first));
    (void)Profitability::PersistObservationIdempotently(
        transaction, Request(5, 50, 501, second));
    transaction.commit();
}

void Verify()
{
    pqxx::connection connection{ConnectionString(Environment("LSTM_DB_USER"))};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    assert(transaction.exec("SHOW transaction_read_only;")
               .one_row()[0].as<std::string>() == "on");

    const auto valid = Verification::LoadAndVerifyExactFinalEvidence(
        transaction, 1);
    assert(valid.state == Verification::EvidenceState::valid);
    assert(valid.observation);
    const auto repeated = Verification::LoadAndVerifyExactFinalEvidence(
        transaction, 1);
    assert(valid.evidenceIdentityCanonical == repeated.evidenceIdentityCanonical);
    assert(valid.evidenceIdentityHash == repeated.evidenceIdentityHash);

    const auto checkpointCannotSubstitute =
        Verification::LoadAndVerifyExactFinalEvidence(transaction, 2);
    assert(checkpointCannotSubstitute.state ==
           Verification::EvidenceState::unavailable);
    assert(!checkpointCannotSubstitute.observation);
    assert(checkpointCannotSubstitute.reason ==
           "no_profitability_observation");

    const auto ambiguousFinal =
        Verification::LoadAndVerifyExactFinalEvidence(transaction, 4);
    assert(ambiguousFinal.state == Verification::EvidenceState::ambiguous);
    assert(ambiguousFinal.reason == "ambiguous_final_inference_result");

    const auto ambiguousObservation =
        Verification::LoadAndVerifyExactFinalEvidence(transaction, 5);
    assert(ambiguousObservation.state ==
           Verification::EvidenceState::ambiguous);
    assert(ambiguousObservation.reason ==
           "ambiguous_profitability_observation");

    const auto missingExperiment =
        Verification::LoadAndVerifyExactFinalEvidence(transaction, 999);
    assert(missingExperiment.state == Verification::EvidenceState::incomplete);
    assert(missingExperiment.reason == "experiment_not_found");
    assert(Verification::ExitCode({valid, checkpointCannotSubstitute}) == 4);
    assert(Verification::ExitCode({valid, ambiguousFinal}) == 3);
    assert(Verification::ExitCode({valid, ambiguousObservation}) == 3);
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
