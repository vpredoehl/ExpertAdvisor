#include "LstmRuntimeConstruction.hpp"
#include "RuntimeLogging.hpp"
namespace EA::LstmRuntimeConstruction
{
LSTM CreateLstmForRuntimeLogLevel(const ::Tensor& tensor, std::size_t hiddenSize,
                                  float initialLongTerm, float initialShortTerm,
                                  LSTM::TargetType targetType,
                                  std::optional<std::size_t> modelInputWidth,
                                  FeatureAblationMask ablationMask,
                                  unsigned int freshInitializationSeed)
{
    RuntimeLogging::ScopedDiagnosticCoutSilencer silence;
    return LSTM { tensor, hiddenSize, initialLongTerm, initialShortTerm, targetType,
                  modelInputWidth, std::move(ablationMask), freshInitializationSeed };
}
}
