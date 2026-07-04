#pragma once

#include <iosfwd>
#include <optional>
#include <string>

#include <pqxx/pqxx>

namespace EA::RunMetadata
{

constexpr const char* kSchedulerVersion = "lstm-experiment-framework-1";

struct Snapshot
{
    std::string gitCommit = "unknown";
    std::string gitBranch = "unknown";
    std::optional<bool> gitDirty;
    std::string buildConfig;
    std::string compilerVersion;
    std::string schedulerVersion = kSchedulerVersion;
    std::string binaryName;
    std::string invocationMode;
};

Snapshot Capture(const std::string& binaryName,
                 const std::string& invocationMode);

std::string CurrentUtcTimestamp();
std::string SqlNullableBool(const std::optional<bool>& value);

bool ColumnExists(pqxx::work& w,
                  const std::string& tableName,
                  const std::string& columnName);
bool ExperimentRunMetadataColumnsExist(pqxx::work& w);
std::string CurrentSchemaVersion(pqxx::work& w);

void AppendRunMetadataColumns(std::ostringstream& sql);
void AppendRunMetadataValues(std::ostringstream& sql,
                             pqxx::work& w,
                             const Snapshot& metadata,
                             const std::string& schemaVersion);
void BackfillMissingExperimentRunMetadata(pqxx::work& w,
                                          const std::string& binaryName,
                                          const std::string& invocationMode);

} // namespace EA::RunMetadata
