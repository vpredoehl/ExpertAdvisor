#if !__has_feature(objc_arc)
#error "MetalForwardAffine.mm requires Objective-C ARC"
#endif

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "MetalForwardAffine.hpp"
#include <MetaNN/metal/metal_matmul.h>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace EA::MetalForwardAffine
{
namespace
{
enum class Path { MetaNN, Combined };

Path SelectedPath()
{
    static const Path path = [] {
        const char* value = std::getenv("EA_LSTM_FORWARD_AFFINE");
        if (value == nullptr || std::string(value) == "metann") return Path::MetaNN;
        if (std::string(value) == "combined") return Path::Combined;
        throw std::runtime_error("EA_LSTM_FORWARD_AFFINE must be metann or combined");
    }();
    return path;
}

id<MTLDevice> Device()
{
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device || !MPSSupportsMTLDevice(device))
        throw std::runtime_error("MetalForwardAffine: MPS device unavailable");
    return device;
}

id<MTLCommandQueue> Queue()
{
    static id<MTLCommandQueue> queue = [Device() newCommandQueue];
    if (!queue) throw std::runtime_error("MetalForwardAffine: command queue unavailable");
    return queue;
}

id<MTLComputePipelineState> BiasPipeline()
{
    // Load the exact existing compiled kernel; do not duplicate/recompile it.
    static id<MTLComputePipelineState> pipeline = [] {
        id<MTLLibrary> library = [Device() newDefaultLibrary];
        id<MTLFunction> function = [library newFunctionWithName:@"add_row_bias_f32"];
        if (!function)
            throw std::runtime_error("MetalForwardAffine: add_row_bias_f32 unavailable in default.metallib");
        NSError* error = nil;
        id<MTLComputePipelineState> result = [Device() newComputePipelineStateWithFunction:function error:&error];
        if (!result)
            throw std::runtime_error(std::string("MetalForwardAffine: bias pipeline: ") +
                                     (error ? error.localizedDescription.UTF8String : "unknown error"));
        return result;
    }();
    return pipeline;
}

MPSMatrixMultiplication* Kernel(size_t m, size_t k, size_t n)
{
    static std::mutex mutex;
    static std::map<std::tuple<size_t, size_t, size_t>, MPSMatrixMultiplication*> cache;
    const std::lock_guard<std::mutex> lock(mutex);
    const auto key = std::make_tuple(m, k, n);
    if (const auto it = cache.find(key); it != cache.end()) return it->second;
    MPSMatrixMultiplication* kernel = [[MPSMatrixMultiplication alloc]
        initWithDevice:Device() transposeLeft:NO transposeRight:NO
        resultRows:m resultColumns:n interiorColumns:k alpha:1.0 beta:0.0];
    if (!kernel) throw std::runtime_error("MetalForwardAffine: MPS kernel unavailable");
    cache.emplace(key, kernel);
    return kernel;
}

size_t Product(size_t a, size_t b)
{
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error("MetalForwardAffine: size overflow");
    return a * b;
}

struct BufferRange
{
    id<MTLBuffer> buffer;
    size_t offset;
    size_t bytes;
};

BufferRange Range(const Memory& memory, size_t elements)
{
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)memory.NativeHandle();
    const size_t offset = Product(memory.Offset(), sizeof(float));
    const size_t bytes = Product(elements, sizeof(float));
    if (!buffer || elements > memory.Size() || offset > buffer.length || bytes > buffer.length - offset)
        throw std::runtime_error("MetalForwardAffine: invalid buffer range");
    if (buffer.device != Device() || buffer.storageMode != MTLStorageModeShared ||
        buffer.hazardTrackingMode != MTLHazardTrackingModeTracked)
        throw std::runtime_error("MetalForwardAffine: expected shared tracked buffer on the MPS device");
    return {buffer, offset, bytes};
}

bool Overlaps(const BufferRange& a, const BufferRange& b)
{
    return a.buffer == b.buffer && a.offset < b.offset + b.bytes && b.offset < a.offset + a.bytes;
}

MPSMatrix* Matrix(const BufferRange& range, size_t rows, size_t cols)
{
    MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
        columns:cols rowBytes:Product(cols, sizeof(float)) dataType:MPSDataTypeFloat32];
    MPSMatrix* matrix = [[MPSMatrix alloc] initWithBuffer:range.buffer
        offset:range.offset descriptor:descriptor];
    if (!matrix) throw std::runtime_error("MetalForwardAffine: MPS matrix unavailable");
    return matrix;
}
}

const char* SelectedPathName()
{
    return SelectedPath() == Path::Combined ? "combined" : "metann";
}

void ForwardMatMulBias(const Memory& a, const Memory& b, const Memory& bias,
                       Memory& c, size_t m, size_t k, size_t n, bool diagnosticLogging)
{
    const auto path = SelectedPath();
    if (diagnosticLogging)
    {
        static std::once_flag printed;
        std::call_once(printed, [path] {
            std::cout << "DIAG_LSTM_FORWARD_AFFINE,path="
                      << (path == Path::Combined ? "combined" : "metann") << '\n';
        });
    }
    if (path == Path::Combined) CombinedMatMulBias(a, b, bias, c, m, k, n);
    else MetaNN::NSMetalMatMul::MatMulBias(a, b, bias, c, m, k, n);
}

void CombinedMatMulBias(const Memory& a, const Memory& b, const Memory& bias,
                        Memory& c, size_t m, size_t k, size_t n, SynchronizationStats* stats)
{
    if (stats) *stats = {};
    if (m == 0 || k == 0 || n == 0) return; // Match MetaNN's no-work contract.
    if (m > UINT32_MAX || n > UINT32_MAX || Product(m, n) > UINT32_MAX)
        throw std::runtime_error("MetalForwardAffine: bias kernel dimensions exceed uint32 range");

    // Shared C++ owners also prevent allocator-pool reuse while work is pending.
    const Memory ownerA = a, ownerB = b, ownerBias = bias, ownerC = c;
    @autoreleasepool
    {
        const auto ra = Range(ownerA, Product(m, k));
        const auto rb = Range(ownerB, Product(k, n));
        const auto rc = Range(ownerC, Product(m, n));
        const auto rBias = Range(ownerBias, n);
        if (Overlaps(rc, ra) || Overlaps(rc, rb) || Overlaps(rc, rBias))
            throw std::runtime_error("MetalForwardAffine: output overlaps an input");

        MPSMatrix* matA = Matrix(ra, m, k);
        MPSMatrix* matB = Matrix(rb, k, n);
        MPSMatrix* matC = Matrix(rc, m, n);
        MPSMatrixMultiplication* kernel = Kernel(m, k, n);
        id<MTLComputePipelineState> pipeline = BiasPipeline();
        id<MTLCommandBuffer> command = [Queue() commandBuffer];
        if (!command) throw std::runtime_error("MetalForwardAffine: command buffer unavailable");
        [kernel encodeToCommandBuffer:command leftMatrix:matA rightMatrix:matB resultMatrix:matC];

        // A separate serial encoder preserves GEMM-write -> bias-read/write
        // ordering through automatic tracked-resource hazard synchronization.
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) throw std::runtime_error("MetalForwardAffine: bias encoder unavailable");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:rc.buffer offset:rc.offset atIndex:0];
        [encoder setBuffer:rBias.buffer offset:rBias.offset atIndex:1];
        const uint32_t mm = static_cast<uint32_t>(m), nn = static_cast<uint32_t>(n);
        [encoder setBytes:&mm length:sizeof(mm) atIndex:2];
        [encoder setBytes:&nn length:sizeof(nn) atIndex:3];
        const NSUInteger tw = pipeline.threadExecutionWidth;
        NSUInteger th = pipeline.maxTotalThreadsPerThreadgroup / tw;
        if (th == 0) th = 1;
        [encoder dispatchThreads:MTLSizeMake(n, m, 1)
            threadsPerThreadgroup:MTLSizeMake(tw, th, 1)];
        [encoder endEncoding];

        [command commit];
        if (stats) ++stats->submissions;
        [command waitUntilCompleted];
        if (stats) ++stats->blockingWaits;
        if (command.status != MTLCommandBufferStatusCompleted || command.error != nil)
            throw std::runtime_error(std::string("MetalForwardAffine: command failed: ") +
                (command.error ? command.error.localizedDescription.UTF8String : "incomplete command buffer"));
        if (stats) ++stats->successfulCompletions;
    }
}
}
