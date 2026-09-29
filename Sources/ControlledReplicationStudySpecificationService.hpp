#pragma once

#include "ControlledReplicationStudySpecification.hpp"

#include <iosfwd>
#include <string>

namespace EA::ControlledReplicationStudy
{

int RunValidateCommand(const std::string& path,
                       std::ostream& output,
                       std::ostream& errors);

int RunCompareCommand(const std::string& connectionString,
                      const std::string& path,
                      std::ostream& output,
                      std::ostream& errors);

} // namespace EA::ControlledReplicationStudy
