#include "ExperimentPairComparisonService.hpp"

#include "FeatureAblationPairEvaluationRepository.hpp"
#include "PairedTrainingObjectiveEvaluationRepository.hpp"

#include <pqxx/pqxx>

namespace EA::ExperimentPairComparison
{
namespace
{

class PostgresEvidenceSource final : public EvidenceSource
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
            throw EvidenceUnavailableError(error.reason());
        }
    }

private:
    pqxx::transaction_base& transaction_;
};

} // namespace

int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors)
{
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const PostgresEvidenceSource source{transaction};
    return RunComparisonCommand(command, source, output, errors);
}

} // namespace EA::ExperimentPairComparison
