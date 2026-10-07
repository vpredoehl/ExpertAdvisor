#include "LstmHotspotProfileFinalizer.hpp"

#include "LSTM.hpp"

#include <iostream>
#include <utility>

namespace EA
{
LstmHotspotProfileFinalizer::LstmHotspotProfileFinalizer(
    bool enabled, std::optional<std::string> outputPath)
    : enabled_(enabled), outputPath_(std::move(outputPath)) {}

LstmHotspotProfileFinalizer::~LstmHotspotProfileFinalizer()
{
    if (!enabled_) return;
    LSTM::PrintHotspotProfileSummary();
    if (outputPath_)
    {
        const bool wrote = LSTM::WriteHotspotProfileReport(*outputPath_);
        std::cout << "LSTM_PROFILE_REPORT"
                  << ",path=" << *outputPath_
                  << ",written=" << (wrote ? 1 : 0) << std::endl;
    }
}
}
