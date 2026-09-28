#pragma once

#include <optional>
#include <string>
#include <vector>

#include <pqxx/pqxx>

namespace EA::ControlledExperimentFamily {

struct Arm {
  long long id = 0;
  int ordinal = 0;
  std::string key;
  std::string displayRole;
  std::optional<std::string> baselineKey;
  std::optional<std::string> comparisonDirection;
  std::string declaredDifferenceCanonical;
  std::string declaredDifferenceHash;
  std::string featureAblationMask;
  std::string featureAblationMaskHash;
  std::string overrideCanonical;
  std::string overrideHash;
};

struct Cell {
  long long id = 0;
  int ordinal = 0;
  std::string symbol;
  int predictionHorizon = 0;
  unsigned int freshInitializationSeed = 0;
  std::string identityCanonical;
  std::string identityHash;
};

struct Member {
  long long id = 0;
  int ordinal = 0;
  int cellOrdinal = 0;
  int armOrdinal = 0;
  std::string plannedIdentityCanonical;
  std::string plannedIdentityHash;
  int hashCollisionOrdinal = 0;
  std::optional<long long> experimentId;
};

struct ExecutionRequirement {
  long long id = 0;
  std::string phase;
  int semanticLayoutVersion = 0;
  int modelInputWidth = 0;
  std::string semanticWorkerRole;
  std::string requiredCapabilitiesCanonical;
  std::string requiredCapabilitiesHash;
  std::string sourceCommit;
  std::string executableSha256;
  std::string runtimeIdentity;
  std::optional<std::string> canonicalManifestPath;
  std::optional<std::string> manifestSha256;
};

struct Version {
  long long id = 0;
  long long familyId = 0;
  int ordinal = 0;
  std::optional<long long> supersedesVersionId;
  int specificationContractVersion = 0;
  int planningContractVersion = 0;
  std::string specificationIdentityCanonical;
  std::string specificationIdentityHash;
  int specificationHashCollisionOrdinal = 0;
  std::string planIdentityCanonical;
  std::string planIdentityHash;
  int planHashCollisionOrdinal = 0;
  int expectedCellCount = 0;
  int expectedMemberCount = 0;
  int expectedArmCount = 0;
  std::string trainStart;
  std::string trainEnd;
  std::string inferStart;
  std::string inferEnd;
  std::string predictionThreshold;
  int targetEpochs = 0;
  int modelInputWidth = 0;
  int modelInputSemanticLayoutVersion = 0;
  std::string featureWarmupScope;
  std::string donchian20Mode;
  int donchianLookback = 0;
  int trainingContractVersion = 0;
  std::string trainingContractCanonical;
  std::string trainingContractHash;
  std::string targetGenerationContractCanonical;
  std::string targetGenerationContractHash;
  std::string trainingObjectiveId;
  std::string trainingObjectiveCanonical;
  std::string trainingObjectiveHash;
  std::string checkpointContractCanonical;
  std::string checkpointContractHash;
  std::string continuationMode;
  std::optional<std::string> continuationPolicyCanonical;
  std::optional<std::string> continuationPolicyHash;
  long long economicCalendarSnapshotId = 0;
  std::string economicCalendarSnapshotHash;
  std::string finalEvidenceContractCanonical;
  std::string finalEvidenceContractHash;
  std::string initialSchedulerPriority;
  std::string createdBy;
  std::string creationReason;
  std::string renderedSpecificationSnapshot;
};

struct FamilyGraph {
  long long familyId = 0;
  std::string familyKey;
  std::string familyCreatedBy;
  Version version;
  std::vector<Arm> arms;
  std::vector<Cell> cells;
  std::vector<Member> members;
  std::vector<ExecutionRequirement> executionRequirements;
};

bool SchemaExists(pqxx::transaction_base &transaction);
FamilyGraph CreateFamilyGraph(pqxx::transaction_base &transaction,
                              const FamilyGraph &declaration);
std::optional<FamilyGraph> LoadFamilyGraph(pqxx::transaction_base &transaction,
                                           long long versionId);
std::optional<FamilyGraph>
LoadFamilyGraphByPlanIdentity(pqxx::transaction_base &transaction,
                              const std::string &planIdentityCanonical);

} // namespace EA::ControlledExperimentFamily
