#pragma once
#include "LSTM.hpp"
#include <initializer_list>

// Linked only by the isolated numerical fixture. No production observer exists.
namespace EA::Testing
{
void RecordLSTMNumericalMatrices(const char* stage,
    std::initializer_list<const EA::LSTM::EAMatrix*> matrices);
}
