#include <cassert>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "DatabaseTestIsolation.hpp"

int main()
{
    unsetenv("LSTM_TEST_DB_NAME");
    bool missingRejected = false;
    try {
        (void)EA::Test::RequireDisposablePostgresTarget();
    } catch (const std::runtime_error &error) {
        missingRejected = std::string{error.what()} ==
            "LSTM_TEST_DB_NAME_required";
    }
    assert(missingRejected);

    setenv("LSTM_TEST_DB_NAME", "LSTM", 1);
    bool productionRejected = false;
    try {
        (void)EA::Test::RequireDisposablePostgresTarget();
    } catch (const std::runtime_error &error) {
        productionRejected = std::string{error.what()} ==
            "clearly_disposable_non_LSTM_database_required";
    }
    assert(productionRejected);

    setenv("LSTM_TEST_DB_NAME", "ea_isolation_test_123", 1);
    const auto target = EA::Test::RequireDisposablePostgresTarget();
    assert(target.database == "ea_isolation_test_123");
    assert(target.host == "127.0.0.1");
    assert(target.port == "5432");
    assert(target.runtimeUser == "pqxx");
    assert(EA::Test::IsClearlyDisposableDatabaseName(target.database));
    assert(!EA::Test::IsClearlyDisposableDatabaseName("LSTM"));
    return 0;
}
