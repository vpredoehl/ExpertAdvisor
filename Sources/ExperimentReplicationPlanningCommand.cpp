#include "ExperimentReplicationPlanningService.hpp"
#include "ExperimentReplicationPlanningPostgres.hpp"

#include "FeatureAblationPairEvaluationRepository.hpp"
#include "PairedTrainingObjectiveEvaluationRepository.hpp"

#include <algorithm>
#include <map>
#include <pqxx/pqxx>
#include <set>

namespace EA::ExperimentReplicationPlanning
{
namespace
{

using IdentityMap = std::map<std::string, std::optional<std::string>>;

IdentityMap Identities(const ExperimentPairComparison::ArmResultSet& arm)
{
    IdentityMap result;
    for (const auto& field : arm.scientificIdentity)
        result.emplace(field.name, field.value);
    return result;
}

enum class IdentityMatch
{
    Exact,
    Different,
    Undetermined
};

bool IsConfigurationField(const std::string& name)
{
    return name != "training_execution_identity" &&
        name != "inference_execution_identity";
}

IdentityMatch CompareConfiguredIdentity(
    const ExperimentPairComparison::ArmResultSet& proposed,
    const ExperimentPairComparison::ArmResultSet& candidate)
{
    const IdentityMap left = Identities(proposed);
    const IdentityMap right = Identities(candidate);
    bool unavailable = false;
    std::set<std::string> names;
    for (const auto& [name, unused] : left)
    {
        (void)unused;
        if (IsConfigurationField(name)) names.insert(name);
    }
    for (const auto& [name, unused] : right)
    {
        (void)unused;
        if (IsConfigurationField(name)) names.insert(name);
    }
    for (const std::string& name : names)
    {
        const auto lhs = left.find(name);
        const auto rhs = right.find(name);
        if (lhs == left.end() || rhs == right.end() ||
            !lhs->second || !rhs->second)
        {
            unavailable = true;
            continue;
        }
        if (*lhs->second != *rhs->second) return IdentityMatch::Different;
    }
    return unavailable ? IdentityMatch::Undetermined : IdentityMatch::Exact;
}

std::string RequiredIdentity(
    const ExperimentPairComparison::ArmResultSet& arm,
    const std::string& name)
{
    const IdentityMap identity = Identities(arm);
    const auto found = identity.find(name);
    if (found == identity.end() || !found->second || *found->second == "NULL")
        throw std::invalid_argument(
            "equivalence_lookup_identity_unavailable:" + name);
    return *found->second;
}

} // namespace

PostgresPlanningSource::PostgresPlanningSource(
    pqxx::transaction_base& transaction)
    : transaction_(transaction)
{
}

FeatureAblationPairEvaluation::ArmEvidence PostgresPlanningSource::Load(
    long long experimentId) const
{
    try
    {
        return FeatureAblationPairEvaluation::LoadAuthoritativeArmEvidence(
            transaction_, experimentId);
    }
    catch (const PairedTrainingObjectiveEvaluation::EvidenceLoadError& error)
    {
        throw ExperimentPairComparison::EvidenceUnavailableError(
            error.reason());
    }
}

EquivalentExperimentResult PostgresPlanningSource::FindEquivalent(
    const ExperimentPairComparison::ArmResultSet& proposedArm) const
{
        const std::string symbol = RequiredIdentity(proposedArm, "symbol");
        const std::string seed = RequiredIdentity(
            proposedArm, "fresh_initialization_seed");
        const std::string mask = RequiredIdentity(
            proposedArm, "feature_ablation_mask");
        const pqxx::result rows = transaction_.exec(
            "SELECT experiment_id FROM experiment "
            "WHERE symbol=$1 AND fresh_initialization_seed=$2::bigint "
            "AND feature_ablation_mask=$3 AND status <> 'cancelled' "
            "ORDER BY experiment_id;",
            pqxx::params{symbol, seed, mask});

        EquivalentExperimentResult result;
        std::vector<long long> undetermined;
        for (const pqxx::row& row : rows)
        {
            const long long experimentId = row[0].as<long long>();
            try
            {
                const auto candidate = ExperimentPairComparison::MakeArmResultSet(
                    FeatureAblationPairEvaluation::LoadAuthoritativeArmEvidence(
                        transaction_, experimentId));
                const IdentityMatch match = CompareConfiguredIdentity(
                    proposedArm, candidate);
                if (match == IdentityMatch::Exact)
                    result.experimentIds.push_back(experimentId);
                else if (match == IdentityMatch::Undetermined)
                    undetermined.push_back(experimentId);
            }
            catch (const PairedTrainingObjectiveEvaluation::EvidenceLoadError&)
            {
                undetermined.push_back(experimentId);
            }
        }

        if (result.experimentIds.size() == 1 && undetermined.empty())
            result.state = EquivalentExperimentState::EquivalentExperimentFound;
        else if (result.experimentIds.empty() && undetermined.empty())
            result.state =
                EquivalentExperimentState::NoEquivalentExperimentFound;
        else
        {
            result.state =
                EquivalentExperimentState::EquivalentExperimentAmbiguous;
            result.experimentIds.insert(result.experimentIds.end(),
                                        undetermined.begin(),
                                        undetermined.end());
            std::sort(result.experimentIds.begin(), result.experimentIds.end());
            result.reason = undetermined.empty()
                ? "multiple_exact_equivalents"
                : "candidate_identity_evidence_unavailable";
        }
        return result;
}

int RunPlanningCommand(const std::string& connectionString,
                       const PlanningCommand& command,
                       std::ostream& output,
                       std::ostream& errors)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const PostgresPlanningSource source{transaction};
    return RunPlanningCommand(command, source, source, output, errors);
}

} // namespace EA::ExperimentReplicationPlanning
