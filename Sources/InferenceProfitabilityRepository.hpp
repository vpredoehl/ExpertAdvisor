#pragma once

#include "InferenceProfitability.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::InferenceProfitability
{

enum class Scope
{
    finalInference,
    checkpointInference
};

std::string ScopeText(Scope scope);

struct ObservationProvenance
{
    std::optional<long long> experimentId;
    long long modelId = -1;
    long long inferenceEvalResultId = -1;
    Scope scope = Scope::finalInference;
    std::optional<long long> checkpointEvalId;
    std::string inferenceStart;
    std::string inferenceEnd;
};

struct ObservationRequest
{
    ObservationProvenance provenance;
    Statistics statistics;
    std::string sourceContentHash;
    std::string metricDefinitionCanonical = kMetricDefinitionCanonical;
};

struct Observation
{
    long long observationId = -1;
    ObservationProvenance provenance;
    Statistics statistics;
    std::optional<double>
        averageTerminalHorizonLogReturnPerActionablePrediction;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
    std::string sourceContentHash;
    std::string observationIdentityCanonical;
    std::string observationIdentityHash;
    std::string createdAt;
};

struct PersistResult
{
    Observation observation;
    bool created = false;
};

struct ObservationSelector
{
    long long inferenceEvalResultId = -1;
    Scope scope = Scope::finalInference;
    std::optional<long long> checkpointEvalId;
    std::string metricDefinitionHash;
    std::string sourceContentHash;
};

// Selects the single current-definition observation for exact continuation
// provenance. Source content is deliberately not guessed: more than one
// matching immutable source observation is ambiguous and therefore
// unavailable to the consumer.
struct AuthoritativeObservationSelector
{
    std::optional<long long> experimentId;
    long long modelId = -1;
    long long inferenceEvalResultId = -1;
    Scope scope = Scope::finalInference;
    std::optional<long long> checkpointEvalId;
    std::string metricDefinitionCanonical;
    std::string metricDefinitionHash;
};

enum class AuthoritativeObservationStatus
{
    available,
    noObservation,
    ambiguousObservation,
    metricDefinitionMismatch,
    provenanceMismatch
};

std::string AuthoritativeObservationStatusText(
    AuthoritativeObservationStatus status);

struct AuthoritativeObservationSelection
{
    AuthoritativeObservationStatus status =
        AuthoritativeObservationStatus::noObservation;
    std::optional<Observation> observation;
};

enum class ExactFinalInferenceResultStatus
{
    available,
    noExactFinalInferenceResult,
    ambiguousFinalInferenceResult,
    finalInferenceContextMismatch
};

std::string ExactFinalInferenceResultStatusText(
    ExactFinalInferenceResultStatus status);

struct ExactFinalInferenceResultSelection
{
    ExactFinalInferenceResultStatus status =
        ExactFinalInferenceResultStatus::noExactFinalInferenceResult;
    std::optional<long long> inferenceEvalResultId;
    std::optional<bool> acceptModel;
};

bool SchemaExists(pqxx::transaction_base& transaction);

std::string BuildObservationIdentityCanonical(
    const ObservationRequest& request);

PersistResult PersistObservationIdempotently(
    pqxx::transaction_base& transaction,
    const ObservationRequest& request);

std::vector<Observation> LoadObservations(
    pqxx::transaction_base& transaction,
    const ObservationSelector& selector);

AuthoritativeObservationSelection SelectAuthoritativeObservation(
    pqxx::transaction_base& transaction,
    const AuthoritativeObservationSelector& selector);

// Resolves the exact completed FINAL inference row produced for a scheduler
// continuation source. The experiment/model relationship and the persisted
// model inference configuration are part of the authority boundary. No
// recency fallback is permitted when the exact identity is absent or
// ambiguous.
ExactFinalInferenceResultSelection ResolveExactFinalInferenceResult(
    pqxx::transaction_base& transaction,
    long long sourceExperimentId,
    long long finalModelId);

std::optional<Observation> LoadObservationById(
    pqxx::transaction_base& transaction,
    long long observationId);

} // namespace EA::InferenceProfitability
