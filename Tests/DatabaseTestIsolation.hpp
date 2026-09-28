#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <pqxx/pqxx>

namespace EA::Test
{

struct DisposablePostgresTarget
{
    std::string database;
    std::string host;
    std::string port;
    std::string adminUser;
    std::string runtimeUser;
};

inline std::string EnvironmentOr(const char* name, const char* fallback)
{
    const char* value = std::getenv(name);
    return value && *value ? value : fallback;
}

inline bool IsClearlyDisposableDatabaseName(const std::string& value)
{
    if (value.empty()) return false;

    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(),
        [](unsigned char character) {
            return static_cast<char>(std::tolower(character));
        });
    if (lower == "lstm") return false;
    if (!std::all_of(value.begin(), value.end(), [](unsigned char character) {
            return std::isalnum(character) || character == '_' ||
                character == '-';
        }))
        return false;

    return lower.starts_with("ea_") ||
        lower.find("test") != std::string::npos ||
        lower.find("tmp") != std::string::npos ||
        lower.find("disposable") != std::string::npos;
}

inline DisposablePostgresTarget RequireDisposablePostgresTarget()
{
    const char* configuredDatabase = std::getenv("LSTM_TEST_DB_NAME");
    if (!configuredDatabase || !*configuredDatabase)
        throw std::runtime_error("LSTM_TEST_DB_NAME_required");
    if (!IsClearlyDisposableDatabaseName(configuredDatabase))
        throw std::runtime_error(
            "clearly_disposable_non_LSTM_database_required");

    return {configuredDatabase,
        EnvironmentOr("LSTM_TEST_DB_HOST", "127.0.0.1"),
        EnvironmentOr("LSTM_TEST_DB_PORT", "5432"),
        EnvironmentOr("LSTM_TEST_DB_ADMIN_USER",
            EnvironmentOr("USER", "vjp").c_str()),
        EnvironmentOr("LSTM_TEST_DB_USER", "pqxx")};
}

inline void RequireConnectedDisposablePostgresTarget(
    pqxx::connection& connection, const DisposablePostgresTarget& target)
{
    pqxx::read_transaction transaction{connection};
    const std::string connected =
        transaction.query_value<std::string>("SELECT current_database();");
    if (connected != target.database ||
        !IsClearlyDisposableDatabaseName(connected))
        throw std::runtime_error("connected_disposable_database_required");
}

} // namespace EA::Test
