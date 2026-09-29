#include "ControlledReplicationStudySpecificationService.hpp"

#include "FeatureAblationPairEvaluationRepository.hpp"
#include "PairedTrainingObjectiveEvaluationRepository.hpp"

#include <pqxx/pqxx>

#include <algorithm>
#include <ostream>
#include <map>
#include <sstream>
#include <string_view>

namespace EA::ControlledReplicationStudy
{
namespace
{

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') || character == '_' ||
            character == '-' || character == '.';
        if (!safe) character = '_';
    }
    return value.empty() ? "EMPTY" : value;
}

class PostgresEvidenceSource final :
    public ExperimentPairComparison::EvidenceSource
{
public:
    explicit PostgresEvidenceSource(pqxx::transaction_base& transaction)
        : transaction_(transaction)
    {
    }

    FeatureAblationPairEvaluation::ArmEvidence Load(
        long long experimentId) const override
    {
        try
        {
            return FeatureAblationPairEvaluation::
                LoadAuthoritativeArmEvidence(transaction_, experimentId);
        }
        catch (const PairedTrainingObjectiveEvaluation::EvidenceLoadError& error)
        {
            throw ExperimentPairComparison::EvidenceUnavailableError(
                error.reason());
        }
    }

private:
    pqxx::transaction_base& transaction_;
};

class CachingEvidenceSource final :
    public ExperimentPairComparison::EvidenceSource
{
public:
    explicit CachingEvidenceSource(const PostgresEvidenceSource& source)
        : source_(source)
    {
    }

    FeatureAblationPairEvaluation::ArmEvidence Load(
        long long experimentId) const override
    {
        const auto found = cache_.find(experimentId);
        if (found != cache_.end()) return found->second;
        auto loaded = source_.Load(experimentId);
        cache_.emplace(experimentId, loaded);
        return loaded;
    }

private:
    const PostgresEvidenceSource& source_;
    mutable std::map<long long, FeatureAblationPairEvaluation::ArmEvidence>
        cache_;
};

std::string ContextDimensionValue(
    const FeatureAblationPairEvaluation::ArmEvidence& evidence,
    std::string_view key)
{
    if (key == "symbol") return evidence.authoritative.configuration.symbol;
    if (key == "prediction_horizon")
        return std::to_string(
            evidence.authoritative.configuration.predictionHorizon);
    throw std::invalid_argument("study_specification_undeclared_context_dimension");
}

void ValidateAgainstEvidence(const Specification& specification,
                             const CachingEvidenceSource& source)
{
    std::map<std::string, std::string> contextDimensionValues;
    for (const auto& context : specification.contexts)
    {
        for (const auto& [key, expected] : context.dimensions)
        {
            const auto inserted = contextDimensionValues.emplace(key, expected);
            if (!inserted.second && inserted.first->second != expected &&
                std::find(specification.allowedContextDimensions.begin(),
                          specification.allowedContextDimensions.end(), key) ==
                    specification.allowedContextDimensions.end())
                throw std::invalid_argument("study_specification_undeclared_context_difference");
        }
        for (const auto& pair : context.pairs)
        {
            const auto armA = source.Load(pair.armA);
            const auto armB = source.Load(pair.armB);
            if (armA.extended.freshInitializationSeed != pair.requestedSeed ||
                armB.extended.freshInitializationSeed != pair.requestedSeed)
                throw std::invalid_argument("study_specification_seed_mismatch");
            if (armA.authoritative.configuration.featureAblationMask !=
                    specification.controlValue ||
                armB.authoritative.configuration.featureAblationMask !=
                    specification.treatmentValue)
                throw std::invalid_argument("study_specification_intervention_mismatch");
            for (const auto& [key, expected] : context.dimensions)
            {
                if (ContextDimensionValue(armA, key) != expected ||
                    ContextDimensionValue(armB, key) != expected)
                    throw std::invalid_argument("study_specification_context_identity_mismatch");
            }
        }
    }
}

} // namespace

int RunValidateCommand(const std::string& path,
                       std::ostream& output,
                       std::ostream& errors)
{
    try
    {
        const Specification specification = Load(path);
        const ValidationResult result = Validate(specification);
        if (!result.valid)
        {
            errors << "CONTROLLED_REPLICATION_STUDY_VALIDATION_FAILED"
                   << ",reason=" << MachineText(result.reason)
                   << ",read_only=true,exit_code=3\n";
            return 3;
        }
        output << "CONTROLLED_REPLICATION_STUDY_VALIDATION"
               << ",study_identifier=" << specification.studyIdentifier
               << ",version=" << specification.version
               << ",study_type=" << specification.studyType
               << ",identity_hash=" << result.identityHash
               << ",context_count=" << specification.contexts.size()
               << ",aggregation_policy=" << specification.aggregationPolicy
               << ",statistical_independence=not_inferred"
               << ",subjective_winner=NONE"
               << ",outcomes_inspected=false"
               << ",read_only=true,state=valid\n";
        for (std::size_t index = 0; index < specification.contexts.size(); ++index)
            output << "CONTROLLED_REPLICATION_STUDY_CONTEXT"
                   << ",ordinal=" << specification.contexts[index].ordinal
                   << ",pair_count=" << specification.contexts[index].pairs.size()
                   << ",argument_order_preserved=true\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "CONTROLLED_REPLICATION_STUDY_VALIDATION_FAILED"
               << ",reason=" << MachineText(error.what())
               << ",read_only=true,exit_code=3\n";
        return 3;
    }
}

int RunCompareCommand(const std::string& connectionString,
                      const std::string& path,
                      std::ostream& output,
                      std::ostream& errors)
{
    try
    {
        const Specification specification = Load(path);
        const ValidationResult validation = Validate(specification);
        if (!validation.valid)
        {
            errors << "CONTROLLED_REPLICATION_STUDY_COMPARISON_FAILED"
                   << ",reason=" << MachineText(validation.reason)
                   << ",read_only=true,exit_code=3\n";
            return 3;
        }
        const auto command = MakeFamilyComparisonCommand(specification);
        pqxx::connection connection{connectionString};
        pqxx::read_transaction transaction{connection};
        transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
        const PostgresEvidenceSource source{transaction};
        const CachingEvidenceSource cachedSource{source};
        ValidateAgainstEvidence(specification, cachedSource);
        const auto report =
            ExperimentReplicationComparison::EvaluateFamilyComparison(
                command, cachedSource);
        output << "CONTROLLED_REPLICATION_STUDY_COMPARISON"
               << ",study_identifier=" << specification.studyIdentifier
               << ",version=" << specification.version
               << ",identity_hash=" << validation.identityHash
               << ",frozen_membership=true"
               << ",outcomes_inspected=true"
               << ",statistical_independence=not_inferred"
               << ",subjective_winner=NONE"
               << ",read_only=true\n"
               << ExperimentReplicationComparison::RenderFamilyReport(report);
        return 0;
    }
    catch (const ExperimentPairComparison::EvidenceUnavailableError& error)
    {
        errors << "CONTROLLED_REPLICATION_STUDY_COMPARISON_FAILED"
               << ",reason=" << MachineText(error.reason())
               << ",read_only=true,exit_code=3\n";
        return 3;
    }
    catch (const std::exception& error)
    {
        errors << "CONTROLLED_REPLICATION_STUDY_COMPARISON_FAILED"
               << ",reason=" << MachineText(error.what())
               << ",read_only=true,exit_code=3\n";
        return 3;
    }
}

} // namespace EA::ControlledReplicationStudy
