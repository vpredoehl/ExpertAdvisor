#include "RuntimeDatabaseConnection.hpp"

#include <cstdlib>

namespace EA::RuntimeDatabaseConnection
{
std::string ForexConnectionString()
{
    const char* host = std::getenv("FOREX_DB_HOST");
    const char* database = std::getenv("FOREX_DB_NAME");
    return "hostaddr=" +
           std::string{host != nullptr && *host != '\0' ? host : "127.0.0.1"} +
           " gssencmode=disable user=pqxx dbname=" +
           std::string{database != nullptr && *database != '\0' ? database : "forex"};
}

std::string LstmConnectionString()
{
    const char* host = std::getenv("LSTM_DB_HOST");
    const char* database = std::getenv("LSTM_DB_NAME");
    return "hostaddr=" +
           std::string{host != nullptr && *host != '\0' ? host : "127.0.0.1"} +
           " gssencmode=disable user=pqxx dbname=" +
           std::string{database != nullptr && *database != '\0' ? database : "LSTM"};
}
}
