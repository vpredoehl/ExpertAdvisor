#pragma once

#include "CausalSurpriseObservability.hpp"

#include <iosfwd>
#include <string>

namespace EA::CausalSurpriseObservability
{

std::string Render(const Result& result);

int RunCommand(const std::string& lstmConnectionString,
               const std::string& forexConnectionString,
               long long experimentId,
               Scope scope,
               std::ostream& output,
               std::ostream& errors);

} // namespace EA::CausalSurpriseObservability
