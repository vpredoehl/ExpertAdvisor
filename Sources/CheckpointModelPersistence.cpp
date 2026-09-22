#include "CheckpointModelPersistence.hpp"

#include "LstmRuntimeLogging.hpp"
#include "PgModelIO.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <pqxx/pqxx>

namespace EA::CheckpointModelPersistence
{
namespace
{

std::string LstmDbConnectionString()
{
    const char* host = std::getenv("LSTM_DB_HOST");
    const char* database = std::getenv("LSTM_DB_NAME");
    return "hostaddr=" +
           std::string{
               host != nullptr && *host != '\0'
                   ? host
                   : "127.0.0.1"} +
           " gssencmode=disable user=pqxx dbname=" +
           std::string{
               database != nullptr && *database != '\0'
                   ? database
                   : "LSTM"};
}

void DiagnosticMessage(const char* message)
{
    if (EA::RuntimeDiagnosticLoggingEnabled())
        std::cout << message << std::endl;
}

std::string CheckpointBaseModelName(
    const EA::LaunchArgs& launchArgs,
    bool resumed,
    const std::string& rawPriceTableName)
{
    if (resumed)
        return launchArgs.newModelName.value_or(
            rawPriceTableName + "-resume-model");
    return launchArgs.newModelName.value_or(
        rawPriceTableName + "-model");
}

std::string EpochCheckpointModelName(
    const std::string& baseModelName,
    std::size_t completedEpoch)
{
    std::ostringstream oss;
    oss << baseModelName
        << "_epoch"
        << std::setw(3)
        << std::setfill('0')
        << completedEpoch;
    return oss.str();
}

bool ModelNameExists(pqxx::work& w, const std::string& modelName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM model WHERE name = $1 LIMIT 1;",
        modelName);
    return !r.empty();
}

std::string UniqueModelName(
    pqxx::work& w,
    const std::string& desiredName)
{
    if (!ModelNameExists(w, desiredName))
        return desiredName;

    for (int suffix = 1; suffix <= 9999; ++suffix)
    {
        std::ostringstream candidate;
        candidate << desiredName
                  << "_dup"
                  << std::setw(3)
                  << std::setfill('0')
                  << suffix;
        if (!ModelNameExists(w, candidate.str()))
            return candidate.str();
    }

    throw std::runtime_error(
        "unable to allocate unique model name for checkpoint '" +
        desiredName + "'");
}

} // namespace

