#include "SchedulerOperationalReadModel.hpp"

#include <pqxx/pqxx>

#include <stdexcept>

namespace EA::SchedulerCore
{

SchedulerOperationalReadModel::SchedulerOperationalReadModel(
    std::string connectionString)
    : connectionString_(std::move(connectionString))
{
}

int SchedulerOperationalReadModel::withReadOnlySnapshot(
    const std::function<int(pqxx::read_transaction&)>& consumer) const
{
    pqxx::connection connection{connectionString_};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY;");
    return consumer(transaction);
}

} // namespace EA::SchedulerCore
