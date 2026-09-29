#pragma once

#include "ExperimentReplicationComparisonService.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace EA::ControlledReplicationStudy
{

inline constexpr int kSpecificationVersion = 1;
inline constexpr const char* kStudyType =
    "controlled_replication_cross_context";

struct PairSpec
{
    long long armA = 0;
    long long armB = 0;
    unsigned int requestedSeed = 0;
};

struct ContextSpec
{
    std::string ordinal;
    std::vector<std::pair<std::string, std::string>> dimensions;
    std::vector<PairSpec> pairs;
};

struct Specification
{
    int version = kSpecificationVersion;
    std::string studyIdentifier;
    std::string studyType = kStudyType;
    std::string interventionField;
    std::string controlSemantics;
    std::string controlValue;
    std::string treatmentValue;
    std::string replicationDimension;
    std::vector<std::string> allowedContextDimensions;
    std::vector<std::string> requiredConfiguredIdentityFields;
    std::vector<std::string> requiredExecutionProvenanceFields;
    std::string aggregationPolicy;
    std::string statisticalIndependence = "not_inferred";
    std::string subjectiveWinner = "NONE";
    std::string freezeTimestamp;
    std::vector<std::string> exclusions;
    std::vector<ContextSpec> contexts;
    std::string identityHash;
};

struct ValidationResult
{
    bool valid = false;
    std::string reason;
    std::string canonical;
    std::string identityHash;
};

// Canonical semantic identity. The timestamp and all declared membership are
// included; completed producer-attempt provenance is deliberately absent.
std::string Canonicalize(const Specification& specification);
std::string IdentityHash(const Specification& specification);
std::string Render(const Specification& specification);

Specification Parse(const std::string& artifact);
Specification Load(const std::string& path);
ValidationResult Validate(const Specification& specification);
ValidationResult ValidateArtifact(const std::string& artifact);

// Converts a validated manifest into the existing family-comparison command.
// Context order and pair order are retained exactly; no raw-pair flattening is
// performed.
EA::ExperimentReplicationComparison::FamilyComparisonCommand
MakeFamilyComparisonCommand(const Specification& specification);

} // namespace EA::ControlledReplicationStudy
