#include "RunMetadata.hpp"

#include <cassert>
#include <optional>

int main()
{
    // SQL booleans are deliberately three-valued: unknown must never become
    // false in persisted scientific run metadata.
    assert(EA::RunMetadata::SqlNullableBool(std::nullopt) == "NULL");
    assert(EA::RunMetadata::SqlNullableBool(false) == "FALSE");
    assert(EA::RunMetadata::SqlNullableBool(true) == "TRUE");
    return 0;
}
