#pragma once

#include <optional>
#include <string>

#include <functional>
#include <iosfwd>
#include <pqxx/pqxx>

namespace EA::SchedulerCore
{

class SchedulerOperationalReadModel;

int PrintObserverSchedulerStatus(SchedulerOperationalReadModel& readModel,
                                 std::ostream& output,
                                 std::ostream& error);
int PrintObserverExperimentStatus(SchedulerOperationalReadModel& readModel,
                                  std::optional<long long> experimentId,
                                  std::ostream& output,
                                  std::ostream& error);
int PrintObserverSchedulerEvidence(SchedulerOperationalReadModel& readModel,
                                   std::ostream& output,
                                   std::ostream& error);
int PrintObserverExperimentEvidence(SchedulerOperationalReadModel& readModel,
                                    long long experimentId,
                                    std::ostream& output,
                                    std::ostream& error);

// This deliberately exposes observation values only.  It has no claim,
// lifecycle, worker-attempt, control, or generic SQL surface.  The private
// transaction bridge is available solely to the authoritative status service.
class SchedulerOperationalReadModel
{
public:
    explicit SchedulerOperationalReadModel(std::string connectionString);

private:
    friend int PrintObserverSchedulerStatus(SchedulerOperationalReadModel&,
                                            std::ostream&, std::ostream&);
    friend int PrintObserverExperimentStatus(SchedulerOperationalReadModel&,
                                             std::optional<long long>,
                                             std::ostream&, std::ostream&);
    friend int PrintObserverSchedulerEvidence(SchedulerOperationalReadModel&,
                                              std::ostream&, std::ostream&);
    friend int PrintObserverExperimentEvidence(SchedulerOperationalReadModel&,
                                               long long,
                                               std::ostream&, std::ostream&);

    int withReadOnlySnapshot(
        const std::function<int(pqxx::read_transaction&)>& consumer) const;

    std::string connectionString_;
};

} // namespace EA::SchedulerCore
