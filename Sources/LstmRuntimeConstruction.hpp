#pragma once
#include <cstddef>
#include <optional>
#include "LSTM.hpp"
namespace EA::LstmRuntimeConstruction
{
LSTM CreateLstmForRuntimeLogLevel(const ::Tensor&, std::size_t, float, float,
                                  LSTM::TargetType, std::optional<std::size_t>,
                                  FeatureAblationMask, unsigned int);
}
