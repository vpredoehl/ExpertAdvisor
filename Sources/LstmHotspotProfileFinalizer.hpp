#pragma once

#include <optional>
#include <string>

namespace EA
{
class LstmHotspotProfileFinalizer
{
public:
    LstmHotspotProfileFinalizer(bool enabled, std::optional<std::string> outputPath);
    ~LstmHotspotProfileFinalizer();
    LstmHotspotProfileFinalizer(const LstmHotspotProfileFinalizer&) = delete;
    LstmHotspotProfileFinalizer& operator=(const LstmHotspotProfileFinalizer&) = delete;
private:
    bool enabled_;
    std::optional<std::string> outputPath_;
};
}
