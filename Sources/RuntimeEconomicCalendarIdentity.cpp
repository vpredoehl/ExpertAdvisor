#include "RuntimeEconomicCalendarIdentity.hpp"

#include <stdexcept>

namespace EA::RuntimeEconomicCalendarIdentity
{
std::optional<EconomicCalendar::EconomicCalendarSnapshotIdentity> Resolve(
    pqxx::transaction_base& transaction, const LaunchArgs& launchArgs)
{
    std::optional<long long> experimentId = launchArgs.schedulerExperimentId;
    if (launchArgs.schedulerCheckpointEvalId)
    {
        const pqxx::result rows = transaction.exec(
            "SELECT COALESCE(parent_experiment_id,experiment_id) FROM "
            "experiment_checkpoint_eval WHERE checkpoint_eval_id=$1;",
            pqxx::params{*launchArgs.schedulerCheckpointEvalId});
        if (rows.size() != 1)
            throw std::runtime_error("checkpoint_eval_not_found_for_economic_calendar_snapshot");
        experimentId = rows.one_row()[0].as<long long>();
    }
    std::optional<EconomicCalendar::EconomicCalendarSnapshotIdentity> experimentSnapshot;
    if (experimentId)
        experimentSnapshot = EconomicCalendar::LoadExperimentEconomicCalendarSnapshot(transaction, *experimentId);
    std::optional<long long> sourceModelId = launchArgs.resumeModelId;
    if (!sourceModelId) sourceModelId = launchArgs.modelId;
    std::optional<EconomicCalendar::EconomicCalendarSnapshotIdentity> modelSnapshot;
    if (sourceModelId)
        modelSnapshot = EconomicCalendar::LoadModelEconomicCalendarSnapshot(transaction, *sourceModelId);
    if (experimentSnapshot.has_value() != modelSnapshot.has_value() && experimentId && sourceModelId)
        throw std::runtime_error("runtime_economic_calendar_snapshot_lineage_mismatch");
    if (experimentSnapshot && modelSnapshot &&
        (experimentSnapshot->snapshotId != modelSnapshot->snapshotId ||
         experimentSnapshot->contentHash != modelSnapshot->contentHash))
        throw std::runtime_error("runtime_economic_calendar_snapshot_identity_conflict");
    return experimentSnapshot ? experimentSnapshot : modelSnapshot;
}

std::optional<EconomicCalendar::EconomicCalendarSnapshotIdentity> FromMaterialization(
    const DBIO::PgModelIO::PersistedModelMaterialization& persisted)
{
    const bool hasId = persisted.identity.economicCalendarSnapshotId.has_value();
    const bool hasHash = persisted.identity.economicCalendarSnapshotHash.has_value();
    if (hasId != hasHash)
        throw std::runtime_error("economic_calendar_snapshot_identity_incomplete");
    if (!hasId) return std::nullopt;
    return EconomicCalendar::EconomicCalendarSnapshotIdentity{
        *persisted.identity.economicCalendarSnapshotId,
        *persisted.identity.economicCalendarSnapshotHash};
}
}
