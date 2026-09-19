#pragma once

#include "Donchian20Mode.hpp"
#include "EconomicEventRepository.hpp"
#include "FeatureWarmupScope.hpp"
#include "Tensor.hpp"

#include <cstddef>
#include <optional>
#include <string>

namespace EA::ModelInputPreparation
{
// Value-only database settings. Connections and transactions remain private to
// Prepare so callers cannot retain data borrowed from the read transactions.
struct DatabaseConnectionSettings
{
    std::string forexConnectionString;
    std::string lstmConnectionString;
};

struct Request
{
    std::string symbol;
    std::string outputStart;
    std::string outputEnd;
    EA::FeatureWarmupScope warmupScope;
    Donchian20Mode donchian20Mode;
    std::size_t donchianLookback;
    std::optional<EA::EconomicCalendar::EconomicCalendarSnapshotIdentity>
        calendarSnapshot;
};

struct Result
{
    Tensor tensor;
    std::size_t logicalOutputStartIndex = 0;
    std::optional<EA::EconomicCalendar::EconomicCalendarSnapshotIdentity>
        calendarSnapshot;
};

// Materializes market and economic-event inputs into an owned Tensor.
// Snapshot lineage selection and model-input-width policy deliberately remain
// with the caller.
Result Prepare(const Request& request,
               const DatabaseConnectionSettings& connectionSettings);
}
