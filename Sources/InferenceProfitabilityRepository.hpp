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

bool SchemaExists(pqxx::transaction_base& transaction);

std::string BuildObservationIdentityCanonical(
    const ObservationRequest& request);

PersistResult PersistObservationIdempotently(
    pqxx::transaction_base& transaction,
    const ObservationRequest& request);

std::vector<Observation> LoadObservations(
    pqxx::transaction_base& transaction,
    const ObservationSelector& selector);

std::optional<Observation> LoadObservationById(
    pqxx::transaction_base& transaction,
    long long observationId);

} // namespace EA::InferenceProfitability
