#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <MetaNN/metal/metal_matmul.h>
#include "MetalForwardAffine.hpp"
#include "ModelInputContract.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <vector>

using EA::MetalForwardAffine::Memory;
using EA::MetalForwardAffine::SynchronizationStats;

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }

namespace
{
constexpr size_t fixtureHiddenSize = 64; // Current production default in Params.hpp.
constexpr size_t productionK = EA::kCurrentModelInputWidth + fixtureHiddenSize;
constexpr size_t productionN = 4 * fixtureHiddenSize;
void Require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Fixture
{
    size_t m, k, n;
    Memory backingA, backingB, backingBias, backingBase, backingCombined;
    Memory a, b, bias, base, combined;
    Fixture(size_t rows, size_t inner, size_t cols, size_t offset, float scale)
        : m(rows), k(inner), n(cols), backingA(offset + m*k + 4),
          backingB(offset + k*n + 4), backingBias(offset + n + 4),
          backingBase(offset + m*n + 4), backingCombined(offset + m*n + 4),
          a(backingA.Shift(offset)), b(backingB.Shift(offset)),
          bias(backingBias.Shift(offset)), base(backingBase.Shift(offset)),
          combined(backingCombined.Shift(offset))
    {
        for (auto* mem : {&backingA, &backingB, &backingBias, &backingBase, &backingCombined})
            std::fill(mem->MutableRawMemory(), mem->MutableRawMemory()+mem->Size(), -12345.0f);
        for (size_t i=0; i<m*k; ++i) a.MutableRawMemory()[i] = float(int(i%23)-11) * scale / 16.0f;
        for (size_t i=0; i<k*n; ++i) b.MutableRawMemory()[i] = float(int(i%19)-9) * scale / 32.0f;
        for (size_t i=0; i<n; ++i) bias.MutableRawMemory()[i] = float(int(i%7)-3) * scale / 8.0f;
    }
    void Run(bool optimized, SynchronizationStats* stats=nullptr)
    {
        if (optimized) EA::MetalForwardAffine::CombinedMatMulBias(a,b,bias,combined,m,k,n,stats);
        else MetaNN::NSMetalMatMul::MatMulBias(a,b,bias,base,m,k,n);
    }
    void Equal() const
    {
        size_t mismatches=0;
        double maxAbs=0;
        for (size_t i=0; i<m*n; ++i)
        {
            const float x=base.RawMemory()[i], y=combined.RawMemory()[i];
            Require(std::isfinite(x) && std::isfinite(y), "nonfinite result");
            if (std::memcmp(&x,&y,sizeof(float)) != 0) ++mismatches;
            maxAbs=std::max(maxAbs,std::fabs(double(x)-double(y)));
        }
        if (mismatches) std::cerr << "mismatches=" << mismatches << ",max_abs=" << maxAbs << '\n';
        Require(mismatches==0,"GEMM+bias bitwise mismatch");
        Require(std::memcmp(backingBase.RawMemory(),backingCombined.RawMemory(),
                            backingBase.Size()*sizeof(float))==0,"output guards differ");
        const size_t offset=base.Offset();
        for (size_t i=0; i<offset; ++i) Require(backingCombined.RawMemory()[i]==-12345.0f,"prefix guard overwritten");
        for (size_t i=offset+m*n; i<backingCombined.Size(); ++i)
            Require(backingCombined.RawMemory()[i]==-12345.0f,"suffix guard overwritten");
    }
};

void MatrixTests()
{
    for (const auto dims : {std::array<size_t,3>{1,productionK,productionN}, {128,productionK,productionN},
                            {17,productionK,productionN}, {3,5,7}, {7,96,64}})
        for (const size_t offset : {size_t{0},size_t{4},size_t{12}})
            for (const float scale : {1.0f,1.0e-12f,1.0e12f})
            {
                Fixture f(dims[0],dims[1],dims[2],offset,scale);
                for (size_t iteration=0; iteration<8; ++iteration)
                {
                    // CPU changes happen only after the preceding synchronous return.
                    f.a.MutableRawMemory()[0] = float(iteration+1)*scale/16.0f;
                    SynchronizationStats stats;
                    f.Run(false); f.Run(true,&stats); f.Equal();
                    Require(stats.submissions==1 && stats.blockingWaits==1 &&
                            stats.successfulCompletions==1,"synchronization count");
                }
            }
    // A sequential producer/consumer chain, with immediate host validation.
    Fixture f(3,7,7,4,1.0f);
    for (size_t iteration=0; iteration<16; ++iteration)
    {
        f.Run(false); f.Run(true); f.Equal();
        std::copy(f.combined.RawMemory(),f.combined.RawMemory()+21,f.a.MutableRawMemory());
    }
    SynchronizationStats stats{99,99,99};
    EA::MetalForwardAffine::CombinedMatMulBias(f.a,f.b,f.bias,f.combined,0,7,7,&stats);
    Require(stats.submissions==0 && stats.blockingWaits==0,"zero-size no-op");
    auto rejected = [&](size_t m, size_t k, size_t n, Memory& output) {
        stats={};
        try { EA::MetalForwardAffine::CombinedMatMulBias(f.a,f.b,f.bias,output,m,k,n,&stats); }
        catch (const std::runtime_error&) {
            Require(stats.submissions==0 && stats.blockingWaits==0,"invalid input submitted");
            return;
        }
        throw std::runtime_error("invalid range/alias was accepted");
    };
    rejected(100,7,7,f.combined);
    rejected(3,7,7,f.a);
    rejected(SIZE_MAX,7,7,f.combined);
    // Also exercise the actual runtime wrapper; separate processes cover selection.
    f.Run(false);
    EA::MetalForwardAffine::ForwardMatMulBias(f.a,f.b,f.bias,f.combined,3,7,7,true);
    f.Equal();
    std::cout << "PASS matrix cases=45 repeated_operations=360 sequential_operations=16 bitwise_equal=1\n";
}

double Trial(Fixture& f, bool combined, size_t iterations)
{
    const auto start=std::chrono::steady_clock::now();
    for (size_t i=0; i<iterations; ++i) f.Run(combined);
    return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/double(iterations);
}

void Report(const std::vector<double>& samples, const char* path, size_t m)
{
    auto sorted=samples;
    std::sort(sorted.begin(),sorted.end());
    const double mean=std::accumulate(samples.begin(),samples.end(),0.0)/double(samples.size());
    double variance=0;
    for (double x:samples) variance+=(x-mean)*(x-mean);
    std::cout << "LATENCY,path=" << path << ",m=" << m << ",median_us="
        << (sorted[5]+sorted[6])/2 << ",mean_us=" << mean << ",stddev_us="
        << std::sqrt(variance/double(samples.size())) << ",min_us=" << sorted.front()
        << ",max_us=" << sorted.back() << '\n';
}

void Benchmark()
{
    id<MTLDevice> device=MTLCreateSystemDefaultDevice();
    std::cout << "HARDWARE,device=" << device.name.UTF8String << ",trials=12,iterations=100,warmup=32\n";
    for (size_t m : {size_t{1},size_t{17},size_t{128}})
    {
        Fixture f(m,productionK,productionN,4,1.0f);
        for (size_t i=0; i<32; ++i) { f.Run(false); f.Run(true); }
        const NSUInteger memoryBefore=device.currentAllocatedSize;
        std::vector<double> baseline,combined;
        for (size_t trial=0; trial<12; ++trial)
        {
            if (trial%2==0) { baseline.push_back(Trial(f,false,100)); combined.push_back(Trial(f,true,100)); }
            else { combined.push_back(Trial(f,true,100)); baseline.push_back(Trial(f,false,100)); }
            f.Equal();
            std::cout << "TRIAL,m=" << m << ",index=" << trial << ",baseline_us=" << baseline.back()
                      << ",combined_us=" << combined.back() << '\n';
        }
        Report(baseline,"metann",m); Report(combined,"combined",m);
        std::sort(baseline.begin(),baseline.end()); std::sort(combined.begin(),combined.end());
        struct rusage usage{};
        Require(getrusage(RUSAGE_SELF,&usage)==0,"getrusage failed");
        std::cout << "PERFORMANCE,m=" << m << ",k=" << productionK << ",n=" << productionN << ",speedup="
            << (baseline[5]+baseline[6])/(combined[5]+combined[6])
            << ",baseline_submissions=2400,baseline_waits=2400,combined_submissions=1200,combined_waits=1200"
            << ",device_bytes_before=" << memoryBefore << ",device_bytes_after=" << device.currentAllocatedSize
            << ",process_peak_rss_bytes=" << usage.ru_maxrss << '\n';
    }
}
}

int main(int argc, char** argv)
{
    @autoreleasepool
    {
        try
        {
            std::cout << std::setprecision(10);
            if (argc==2 && std::string(argv[1])=="--benchmark") Benchmark();
            else if (argc==2 && std::string(argv[1])=="--selection")
            {
                const char* path = EA::MetalForwardAffine::SelectedPathName();
                std::cout << "SELECTION,path=" << path << '\n';
            }
            else { Require(argc==1,"unknown test argument"); MatrixTests(); }
        }
        catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    }
}
