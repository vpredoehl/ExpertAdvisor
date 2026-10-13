#pragma once

#include "CausalFibonacciTrendlineHistoricalEvaluation.hpp"
#include <bit>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>

namespace EA::FibonacciResearch::TrendlineHistorical
{
// Research execution only. The canonical loader has already converted civil
// timestamps and validated source volume; its Candle contract is UTC/OHLC.
// No SQL, detector state, volume reinterpretation or synthesized bars live here.
class ResearchCandleSnapshot final
{
public:
    inline static constexpr std::string_view kEncoding = "research-canonical-candle-binary64-be-v1";
    struct Budgets
    {
        std::uint64_t maxRows = 1000000, maxBytes = 64 * 1024 * 1024;
        std::uint64_t reserveDiskBytes = 64 * 1024 * 1024;
    };
    ResearchCandleSnapshot(const std::filesystem::path& parent, std::int64_t start,
                          std::int64_t end, std::uint64_t expected, Budgets budgets)
        : start_(start), end_(end), expected_(expected), budgets_(budgets)
    {
        static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559);
        if (start < kOrigin || start >= end || end > kFinalStart || !expected ||
            expected > budgets.maxRows || budgets.maxBytes < kOverhead ||
            expected > (budgets.maxBytes - kOverhead) / kRecordBytes)
            throw std::invalid_argument("research candle snapshot range/row/byte budget");
        const auto base = std::filesystem::canonical(parent);
        const auto available = std::filesystem::space(base).available;
        const auto required = expected * kRecordBytes + kOverhead;
        if (available < required || available - required < budgets.reserveDiskBytes)
            throw std::runtime_error("research candle snapshot disk budget");
        std::string pattern = (base / "ea_fib_candles.XXXXXX").string();
        if (!mkdtemp(pattern.data())) throw std::runtime_error("cannot own candle snapshot directory");
        owned_ = pattern; path_ = owned_ / "candles.bin";
        try
        {
            output_.exceptions(std::ios::badbit | std::ios::failbit);
            output_.open(path_, std::ios::binary | std::ios::trunc);
            Write("EAFIBCS1");
            WriteWord(static_cast<std::uint64_t>(start_));
            WriteWord(static_cast<std::uint64_t>(end_));
            WriteWord(expected_);
        }
        catch (...) { Cleanup(); throw; }
    }
    ~ResearchCandleSnapshot() { Cleanup(); }
    ResearchCandleSnapshot(const ResearchCandleSnapshot&) = delete;
    ResearchCandleSnapshot& operator=(const ResearchCandleSnapshot&) = delete;
    void Append(const TG1A::Candle& candle)
    {
        if (sealed_ || rows_ >= expected_) throw std::runtime_error("snapshot sealed or excess rows");
        CheckCandle(candle, last_);
        if (rows_ % 4096 == 0)
        {
            const auto free = std::filesystem::space(owned_).available;
            const auto remaining = (expected_ - rows_) * kRecordBytes + 64;
            if (free < remaining || free - remaining < budgets_.reserveDiskBytes)
                throw std::runtime_error("research candle snapshot disk exhaustion");
        }
        WriteWord(static_cast<std::uint64_t>(candle.timestamp));
        for (double price : {candle.open, candle.high, candle.low, candle.close})
            WriteWord(std::bit_cast<std::uint64_t>(price));
        AddCanonical(canonical_, candle); last_ = candle.timestamp; ++rows_;
    }
    void Seal()
    {
        if (sealed_ || rows_ != expected_) throw std::runtime_error("snapshot row-count mismatch");
        canonicalDigest_ = canonical_.Digest();
        Write(hash_.Digest()); // Header + all bit-exact records; trailer is outside the digest.
        output_.flush(); output_.close();
        if (chmod(path_.c_str(), S_IRUSR) != 0) throw std::runtime_error("snapshot cannot be immutable");
        sealed_ = true;
        Validate();
    }
    void Validate() const { Read({}); }
    void Replay(const std::function<void(const TG1A::Candle&)>& consumer) const
    {
        if (!consumer) throw std::invalid_argument("empty snapshot replay consumer");
        Validate(); // No detector advances until every input byte is verified.
        Read(consumer); // Revalidate while replaying; failure prevents COMPLETE.
    }
    const std::filesystem::path& Path() const { return path_; }
    std::uint64_t Rows() const { return rows_; }
    std::uint64_t Bytes() const { return expected_ * kRecordBytes + kOverhead; }
    const std::string& CanonicalDigest() const { return canonicalDigest_; }
private:
    inline static constexpr std::uint64_t kRecordBytes = 40, kOverhead = 32 + 64;
    std::int64_t start_, end_;
    std::uint64_t expected_, rows_ = 0;
    Budgets budgets_;
    std::filesystem::path owned_, path_;
    std::ofstream output_;
    Sha256 hash_, canonical_;
    std::optional<std::int64_t> last_;
    std::string canonicalDigest_;
    bool sealed_ = false;
    static void AddCanonical(Sha256& hash, const TG1A::Candle& c)
    {
        hash.Add(std::to_string(c.timestamp) + "," + Number(c.open) + "," + Number(c.high) + "," +
                 Number(c.low) + "," + Number(c.close) + "\n");
    }
    void CheckCandle(const TG1A::Candle& c, std::optional<std::int64_t> last) const
    {
        if (c.timestamp < start_ || c.timestamp >= end_ || (last && c.timestamp <= *last))
            throw std::runtime_error("snapshot candle ordering/half-open bounds");
        for (double p : {c.open, c.high, c.low, c.close})
            if (!std::isfinite(p)) throw std::runtime_error("snapshot nonfinite OHLC");
    }
    void Write(std::string_view bytes)
    {
        output_.write(bytes.data(), static_cast<std::streamsize>(bytes.size())); hash_.Add(bytes);
    }
    void WriteWord(std::uint64_t word)
    {
        std::array<char, 8> bytes{};
        for (std::size_t i = 0; i < 8; ++i) bytes[i] = static_cast<char>(word >> (56 - i * 8));
        Write({bytes.data(), bytes.size()});
    }
    void Read(const std::function<void(const TG1A::Candle&)>& consumer) const
    {
        if (!sealed_ || std::filesystem::is_symlink(path_) ||
            !std::filesystem::is_regular_file(path_) || std::filesystem::file_size(path_) != Bytes())
            throw std::runtime_error("snapshot unsealed/truncated/excess bytes");
        std::ifstream input(path_, std::ios::binary); input.exceptions(std::ios::badbit | std::ios::failbit);
        Sha256 hash, canonical;
        const auto read = [&](char* bytes, std::size_t n)
        { input.read(bytes, static_cast<std::streamsize>(n)); hash.Add({bytes, n}); };
        const auto word = [&]()
        {
            std::array<char, 8> bytes{}; read(bytes.data(), bytes.size()); std::uint64_t v = 0;
            for (unsigned char b : bytes) v = (v << 8) | b;
            return v;
        };
        std::array<char, 8> magic{}; read(magic.data(), magic.size());
        if (std::string_view(magic.data(), magic.size()) != "EAFIBCS1" ||
            word() != static_cast<std::uint64_t>(start_) || word() != static_cast<std::uint64_t>(end_) ||
            word() != expected_) throw std::runtime_error("snapshot header mismatch");
        std::optional<std::int64_t> last;
        for (std::uint64_t row = 0; row < expected_; ++row)
        {
            TG1A::Candle c{}; c.timestamp = std::bit_cast<std::int64_t>(word());
            c.open = std::bit_cast<double>(word()); c.high = std::bit_cast<double>(word());
            c.low = std::bit_cast<double>(word()); c.close = std::bit_cast<double>(word());
            CheckCandle(c, last); AddCanonical(canonical, c); last = c.timestamp;
            if (consumer) consumer(c);
        }
        std::array<char, 64> trailer{}; input.read(trailer.data(), trailer.size());
        if (std::string_view(trailer.data(), trailer.size()) != hash.Digest() ||
            canonical.Digest() != canonicalDigest_) throw std::runtime_error("snapshot checksum mismatch");
    }
    void Cleanup() noexcept
    {
        // Only the one file and empty directory owned by mkdtemp. Never recursive.
        try { if (output_.is_open()) output_.close(); } catch (...) {}
        std::error_code error;
        if (!owned_.empty() && path_.parent_path() == owned_ &&
            owned_.filename().string().starts_with("ea_fib_candles.") &&
            !std::filesystem::is_symlink(owned_, error))
        { std::filesystem::remove(path_, error); std::filesystem::remove(owned_, error); }
    }
};
} // namespace EA::FibonacciResearch::TrendlineHistorical
