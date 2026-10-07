#pragma once

#include <optional>
#include <pqxx/pqxx>

#include "EconomicEventRepository.hpp"
#include "LaunchArguments.hpp"
#include "PgModelIO.hpp"

namespace EA::RuntimeEconomicCalendarIdentity
{
std::optional<EconomicCalendar::EconomicCalendarSnapshotIdentity> Resolve(
    pqxx::transaction_base& transaction, const LaunchArgs& launchArgs);
std::optional<EconomicCalendar::EconomicCalendarSnapshotIdentity> FromMaterialization(
    const DBIO::PgModelIO::PersistedModelMaterialization& persisted);
}
