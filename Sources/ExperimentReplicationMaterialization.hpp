#pragma once

#include "ExperimentReplicationPlanningService.hpp"

#include <iosfwd>
#include <string>

namespace EA::ExperimentReplicationMaterialization
{

namespace Planning = ExperimentReplicationPlanning;

inline constexpr int kMaterializerVersion = 2;

struct MaterializationCommand : Planning::PlanningCommand
{
    bool allowExistingEquivalent = false;
};

// A deliberately narrow, frozen Layout-11 study constructor.  The template is
// a completed empty-mask Layout-10 control whose non-layout configuration is
// revalidated and then cloned; it is never modified.
struct Layout11ConfluenceCommand
{
    long long templateExperimentId = 0;
    std::vector<unsigned int> requestedSeeds;
    std::string semanticWorkerRegistryPath =
        "Builds/SemanticWorkers/registry.json";
};

struct Layout11ConfluenceArm
{
    long long templateExperimentId = 0;
    unsigned int freshInitializationSeed = 0;
    std::string role;
    std::string featureAblationMask;
    ExperimentPairComparison::ArmResultSet proposed;
};

class Layout11ConfluenceExperimentInserter
{
public:
    virtual ~Layout11ConfluenceExperimentInserter() = default;
    virtual long long InsertFreshPausedLayout11ConfluenceExperiment(
        const Layout11ConfluenceArm& arm) = 0;
};

int RunLayout11ConfluenceMaterializationInTransaction(
    const Layout11ConfluenceCommand& command,
    const ExperimentPairComparison::EvidenceSource& evidence,
    const Planning::EquivalentExperimentSource& equivalents,
    Layout11ConfluenceExperimentInserter& inserter,
    std::ostream& output,
    std::ostream& errors,
    const EA::Scheduler::SemanticWorkerRegistry* registry = nullptr,
    bool apply = true);

int RunLayout11ConfluencePlanCommand(const std::string& connectionString,
                                     const Layout11ConfluenceCommand& command,
                                     std::ostream& output,
                                     std::ostream& errors);
int RunLayout11ConfluenceMaterializationCommand(
    const std::string& connectionString,
    const Layout11ConfluenceCommand& command,
    std::ostream& output,
    std::ostream& errors);

// A single historical source may be transported to one canonical target
// symbol.  This is intentionally separate from controlled replication, whose
// only intervention is the seed.
struct CrossSymbolCommand
{
    long long sourceExperimentId = 0;
    std::string targetSymbol;
    std::string semanticWorkerRegistryPath =
        "Builds/SemanticWorkers/registry.json";
};

// Preview opens a repeatable-read transaction and performs no INSERT. Apply
// serializes against experiment writers and creates one paused/train record.
int RunCrossSymbolPreviewCommand(const std::string& connectionString,
                                 const CrossSymbolCommand& command,
                                 std::ostream& output,
                                 std::ostream& errors);
int RunCrossSymbolMaterializationCommand(const std::string& connectionString,
                                         const CrossSymbolCommand& command,
                                         std::ostream& output,
                                         std::ostream& errors);

class ExperimentInserter
{
public:
    virtual ~ExperimentInserter() = default;
    virtual long long InsertFreshPausedExperiment(
        const Planning::ProposedExperimentSpecification& specification) = 0;
};

// Executes against an already established transaction/serialization boundary.
// A nonzero return performs no inserts for validation/equivalence failures.
// Insert exceptions propagate so the owning transaction necessarily rolls back.
int RunMaterializationInTransaction(
    const MaterializationCommand& command,
    const ExperimentPairComparison::EvidenceSource& evidence,
    const Planning::EquivalentExperimentSource& equivalents,
    ExperimentInserter& inserter,
    std::ostream& output,
    std::ostream& errors,
    const EA::Scheduler::SemanticWorkerRegistry* registry = nullptr);

// Owns one PostgreSQL write transaction.  It takes a table lock before source
// reload, preflight, equivalence checks, or inserts and commits only a complete
// successful wave. Database failures retain the scheduler CLI exit-2 contract.
int RunMaterializationCommand(const std::string& connectionString,
                              const MaterializationCommand& command,
                              std::ostream& output,
                              std::ostream& errors);

} // namespace EA::ExperimentReplicationMaterialization
