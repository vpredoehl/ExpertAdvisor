#include "RunMetadata.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iomanip>
#include <sstream>

#include <pqxx/pqxx>

namespace EA::RunMetadata
{
namespace
{

std::string Trim(std::string value)
{
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r' ||
                              value.back() == ' ' || value.back() == '\t'))
        value.pop_back();
    size_t first = 0;
    while (first < value.size() &&
           (value[first] == ' ' || value[first] == '\t' ||
            value[first] == '\n' || value[first] == '\r'))
        ++first;
    return value.substr(first);
}

std::optional<std::string> CaptureCommandOutput(const char* command)
{
    FILE* pipe = popen(command, "r");
    if (!pipe)
        return std::nullopt;

    std::array<char, 256> buffer{};
    std::ostringstream out;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr)
        out << buffer.data();
    const int rc = pclose(pipe);
    if (rc != 0)
        return std::nullopt;
    return Trim(out.str());
}

bool TableExists(pqxx::work& w, const std::string& tableName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name = $1 LIMIT 1;",
        tableName);
    return !r.empty();
}

} // namespace

Snapshot Capture(const std::string& binaryName,
                 const std::string& invocationMode)
{
    Snapshot snapshot;
    snapshot.binaryName = binaryName;
    snapshot.invocationMode = invocationMode;

#if defined(NDEBUG)
    snapshot.buildConfig = "Release";
#else
    snapshot.buildConfig = "Debug";
#endif

#if defined(__clang_version__)
    snapshot.compilerVersion = __clang_version__;
#elif defined(__VERSION__)
    snapshot.compilerVersion = __VERSION__;
#else
    snapshot.compilerVersion = "unknown";
#endif

    if (const auto commit = CaptureCommandOutput("git rev-parse HEAD 2>/dev/null");
        commit.has_value() && !commit->empty())
    {
        snapshot.gitCommit = *commit;
    }
    if (const auto branch = CaptureCommandOutput("git branch --show-current 2>/dev/null");
        branch.has_value() && !branch->empty())
    {
        snapshot.gitBranch = *branch;
    }
    if (const auto status = CaptureCommandOutput("git status --porcelain 2>/dev/null");
        status.has_value())
    {
        snapshot.gitDirty = !status->empty();
    }

    return snapshot;
}

std::string CurrentUtcTimestamp()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return out.str();
}

std::string SqlNullableBool(const std::optional<bool>& value)
{
    if (!value.has_value())
        return "NULL";
    return *value ? "TRUE" : "FALSE";
}

bool ColumnExists(pqxx::work& w,
                  const std::string& tableName,
                  const std::string& columnName)
{
    pqxx::result r = w.exec_params(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = $1 AND column_name = $2 LIMIT 1;",
        tableName,
        columnName);
    return !r.empty();
}

bool ExperimentRunMetadataColumnsExist(pqxx::work& w)
{
    return ColumnExists(w, "experiment", "git_commit") &&
           ColumnExists(w, "experiment", "git_branch") &&
           ColumnExists(w, "experiment", "git_dirty") &&
           ColumnExists(w, "experiment", "build_config") &&
           ColumnExists(w, "experiment", "compiler_version") &&
           ColumnExists(w, "experiment", "schema_version") &&
           ColumnExists(w, "experiment", "scheduler_version") &&
           ColumnExists(w, "experiment", "binary_name") &&
           ColumnExists(w, "experiment", "invocation_mode") &&
           ColumnExists(w, "experiment", "run_metadata_captured_at");
}

std::string CurrentSchemaVersion(pqxx::work& w)
{
    if (!TableExists(w, "schema_migrations"))
        return "unknown";
    pqxx::result rows = w.exec(
        "SELECT version FROM schema_migrations "
        "ORDER BY applied_at DESC, version DESC LIMIT 1;");
    if (rows.empty() || rows[0][0].is_null())
        return "unknown";
    return rows[0][0].as<std::string>();
}

void AppendRunMetadataColumns(std::ostringstream& sql)
{
    sql << ", git_commit, git_branch, git_dirty, build_config, compiler_version, "
        << "schema_version, scheduler_version, binary_name, invocation_mode, "
        << "run_metadata_captured_at";
}

void AppendRunMetadataValues(std::ostringstream& sql,
                             pqxx::work& w,
                             const Snapshot& metadata,
                             const std::string& schemaVersion)
{
    sql << ","
        << w.quote(metadata.gitCommit) << ","
        << w.quote(metadata.gitBranch) << ","
        << SqlNullableBool(metadata.gitDirty) << ","
        << w.quote(metadata.buildConfig) << ","
        << w.quote(metadata.compilerVersion) << ","
        << w.quote(schemaVersion) << ","
        << w.quote(metadata.schedulerVersion) << ","
        << w.quote(metadata.binaryName) << ","
        << w.quote(metadata.invocationMode) << ","
        << "now()";
}

void BackfillMissingExperimentRunMetadata(pqxx::work& w,
                                          const std::string& binaryName,
                                          const std::string& invocationMode)
{
    if (!ExperimentRunMetadataColumnsExist(w))
        return;

    const Snapshot metadata = Capture(binaryName, invocationMode);
    const std::string schemaVersion = CurrentSchemaVersion(w);
    w.exec(
        "UPDATE experiment SET "
        "git_commit = COALESCE(git_commit, " + w.quote(metadata.gitCommit) + "), "
        "git_branch = COALESCE(git_branch, " + w.quote(metadata.gitBranch) + "), "
        "git_dirty = COALESCE(git_dirty, " + SqlNullableBool(metadata.gitDirty) + "), "
        "build_config = COALESCE(build_config, " + w.quote(metadata.buildConfig) + "), "
        "compiler_version = COALESCE(compiler_version, " + w.quote(metadata.compilerVersion) + "), "
        "schema_version = COALESCE(schema_version, " + w.quote(schemaVersion) + "), "
        "scheduler_version = COALESCE(scheduler_version, " + w.quote(metadata.schedulerVersion) + "), "
        "binary_name = COALESCE(binary_name, " + w.quote(metadata.binaryName) + "), "
        "invocation_mode = COALESCE(invocation_mode, " + w.quote(invocationMode) + "), "
        "run_metadata_captured_at = COALESCE(run_metadata_captured_at, now()), "
        "updated_at = updated_at "
        "WHERE run_metadata_captured_at IS NULL "
        "AND status IN ('pending', 'running');");
}

} // namespace EA::RunMetadata
