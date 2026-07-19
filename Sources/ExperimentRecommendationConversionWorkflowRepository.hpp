#pragma once

#include "ExperimentRecommendationConversionWorkflow.hpp"

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ExperimentRecommendation
{

struct RecommendationConversionWorkflowView
{
    long long proposalId = -1;
    long long recommendationId = -1;
    long long sourceExperimentId = -1;
    int proposalContractVersion = 0;
    std::string proposalIdentityCanonical;
    std::string proposalIdentityHash;
    std::string proposalCreatedAt;
    std::optional<RecommendationConversionWorkflowReviewFact> latestReview;
    std::optional<std::string> latestReviewDecidedAt;
    std::optional<RecommendationConversionWorkflowReviewFact>
        executionReview;
    std::optional<RecommendationConversionWorkflowExecutionFact> execution;
    std::optional<std::string> executionCreatedAt;
    std::optional<RecommendationConversionWorkflowActivationFact> activation;
    std::optional<std::string> activationCreatedAt;
    std::optional<RecommendationConversionWorkflowExperimentFact> experiment;
    std::optional<std::string> experimentUpdatedAt;
    RecommendationConversionWorkflowDerivation derivation;
};

bool RecommendationConversionWorkflowSchemasExist(
    pqxx::connection& connection);

std::optional<RecommendationConversionWorkflowView>
FindRecommendationConversionWorkflow(
    pqxx::connection& connection,
    long long proposalId);

std::optional<RecommendationConversionWorkflowView>
FindRecommendationConversionWorkflowByExecution(
    pqxx::connection& connection,
    long long executionId);

std::optional<RecommendationConversionWorkflowView>
FindRecommendationConversionWorkflowByExperiment(
    pqxx::connection& connection,
    long long experimentId);

std::vector<RecommendationConversionWorkflowView>
ListRecommendationConversionWorkflows(
    pqxx::connection& connection,
    int candidateLimit = kDefaultRecommendationConversionWorkflowListLimit);

} // namespace EA::ExperimentRecommendation
