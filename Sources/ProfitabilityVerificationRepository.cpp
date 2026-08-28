#include "ProfitabilityVerificationRepository.hpp"

#include <stdexcept>

namespace EA::ProfitabilityVerification
{
namespace
{

EvidenceResult StateOnly(long long experimentId,
                         const std::optional<long long>& modelId,
                         const std::optional<long long>& inferenceResultId,
                         const std::string& inferenceStart,
                         const std::string& inferenceEnd,
                         EvidenceState state,
                         const std::string& reason)
{
    ExpectedFinalEvidence expected;
    expected.experimentId = experimentId;
    expected.modelId = modelId.value_or(-1);
    expected.inferenceEvalResultId = inferenceResultId.value_or(-1);
    expected.inferenceStart = inferenceStart;
    expected.inferenceEnd = inferenceEnd;
    EvidenceResult result = ValidateExactFinalObservation(expected, std::nullopt);
    result.state = state;
    result.reason = reason;

    // Rebuild the deterministic identity with the requested state/reason by
    // validating an intentionally incomplete expectation, then binding the
    // public diagnostic fields below. State-only results never claim a valid
    // observation identity.
    std::string canonical = "profitability_exact_final_evidence_state_v1;";
    canonical += "experiment_id=" + std::to_string(experimentId) + ";";
    canonical += "model_id=" +
        (modelId ? std::to_string(*modelId) : "NULL") + ";";
    canonical += "inference_eval_result_id=" +
        (inferenceResultId ? std::to_string(*inferenceResultId) : "NULL") + ";";
    canonical += "inference_start=" +
        (inferenceStart.empty() ? "NULL" : inferenceStart) + ";";
    canonical += "inference_end=" +
        (inferenceEnd.empty() ? "NULL" : inferenceEnd) + ";";
    canonical += "state=" + EvidenceStateText(state) + ";reason=" + reason;
    result.evidenceIdentityCanonical = std::move(canonical);
    result.evidenceIdentityHash = InferenceProfitability::DeterministicHash(
        result.evidenceIdentityCanonical);
    result.experimentId = experimentId;
    result.finalModelId = modelId;
    result.finalInferenceEvalResultId = inferenceResultId;
    return result;
}

} // namespace

EvidenceResult LoadAndVerifyExactFinalEvidence(
    pqxx::transaction_base& transaction,
    long long experimentId)
{
    if (experimentId <= 0)
        throw std::invalid_argument("invalid_profitability_experiment_id");

    const pqxx::result experiments = transaction.exec(R"SQL(
SELECT e.experiment_id,e.last_model_id,
       e.infer_start::date::text AS inference_start,
       e.infer_end::date::text AS inference_end
FROM experiment e
WHERE e.experiment_id=$1
)SQL", pqxx::params{experimentId});
    if (experiments.empty())
        return StateOnly(experimentId, std::nullopt, std::nullopt, {}, {},
                         EvidenceState::incomplete, "experiment_not_found");
    if (experiments.size() != 1)
        return StateOnly(experimentId, std::nullopt, std::nullopt, {}, {},
                         EvidenceState::ambiguous,
                         "ambiguous_experiment_identity");

    const pqxx::row row = experiments.one_row();
    const std::optional<long long> modelId = row["last_model_id"].is_null()
        ? std::nullopt
        : std::optional<long long>{row["last_model_id"].as<long long>()};
    const std::string inferenceStart = row["inference_start"].is_null()
        ? "" : row["inference_start"].as<std::string>();
    const std::string inferenceEnd = row["inference_end"].is_null()
        ? "" : row["inference_end"].as<std::string>();
    if (!modelId || inferenceStart.empty() || inferenceEnd.empty())
        return StateOnly(experimentId, modelId, std::nullopt,
                         inferenceStart, inferenceEnd,
                         EvidenceState::incomplete,
                         "final_inference_context_incomplete");

    const auto finalResult =
        InferenceProfitability::ResolveExactFinalInferenceResult(
            transaction, experimentId, *modelId);
    using FinalStatus =
        InferenceProfitability::ExactFinalInferenceResultStatus;
    if (finalResult.status != FinalStatus::available)
    {
        const EvidenceState state =
            finalResult.status == FinalStatus::ambiguousFinalInferenceResult
                ? EvidenceState::ambiguous
            : finalResult.status == FinalStatus::finalInferenceContextMismatch
                ? EvidenceState::invalidProvenance
                : EvidenceState::incomplete;
        return StateOnly(
            experimentId, modelId, finalResult.inferenceEvalResultId,
            inferenceStart, inferenceEnd, state,
            InferenceProfitability::ExactFinalInferenceResultStatusText(
                finalResult.status));
    }

    if (!InferenceProfitability::SchemaExists(transaction))
        throw std::runtime_error("profitability_schema_unavailable");

    InferenceProfitability::AuthoritativeObservationSelector selector;
    selector.experimentId = experimentId;
    selector.modelId = *modelId;
    selector.inferenceEvalResultId = *finalResult.inferenceEvalResultId;
    selector.scope = InferenceProfitability::Scope::finalInference;
    selector.checkpointEvalId.reset();
    selector.metricDefinitionCanonical =
        InferenceProfitability::kMetricDefinitionCanonical;
    selector.metricDefinitionHash =
        InferenceProfitability::MetricDefinitionHash();
    const auto selected =
        InferenceProfitability::SelectAuthoritativeObservation(
            transaction, selector);
    using ObservationStatus =
        InferenceProfitability::AuthoritativeObservationStatus;
    if (selected.status != ObservationStatus::available)
    {
        const EvidenceState state =
            selected.status == ObservationStatus::noObservation
                ? EvidenceState::unavailable
            : selected.status == ObservationStatus::ambiguousObservation
                ? EvidenceState::ambiguous
            : selected.status == ObservationStatus::metricDefinitionMismatch
                ? EvidenceState::invalidMetricDefinition
                : EvidenceState::invalidProvenance;
        return StateOnly(
            experimentId, modelId, finalResult.inferenceEvalResultId,
            inferenceStart, inferenceEnd, state,
            InferenceProfitability::AuthoritativeObservationStatusText(
                selected.status));
    }

    ExpectedFinalEvidence expected;
    expected.experimentId = experimentId;
    expected.modelId = *modelId;
    expected.inferenceEvalResultId = *finalResult.inferenceEvalResultId;
    expected.inferenceStart = inferenceStart;
    expected.inferenceEnd = inferenceEnd;
    return ValidateExactFinalObservation(expected, selected.observation);
}

} // namespace EA::ProfitabilityVerification
