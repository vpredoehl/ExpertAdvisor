#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <iostream>

namespace
{

bool ResolvePipeline(id<MTLDevice> device,
                     id<MTLLibrary> library,
                     NSString* functionName,
                     const char* libraryName)
{
    if (library == nil)
    {
        std::cerr << "METAL_RUNTIME_LIBRARY_MISSING,library="
                  << libraryName << '\n';
        return false;
    }
    id<MTLFunction> function = [library newFunctionWithName:functionName];
    if (function == nil)
    {
        std::cerr << "METAL_RUNTIME_FUNCTION_MISSING,library="
                  << libraryName << ",function="
                  << functionName.UTF8String << '\n';
        return false;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil)
    {
        std::cerr << "METAL_RUNTIME_PIPELINE_FAILED,library="
                  << libraryName << ",function="
                  << functionName.UTF8String << ",error="
                  << (error == nil ? "unknown" : error.localizedDescription.UTF8String)
                  << '\n';
        return false;
    }
    return true;
}

} // namespace

int main()
{
    @autoreleasepool
    {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil)
        {
            std::cerr << "METAL_RUNTIME_DEVICE_UNAVAILABLE\n";
            return 2;
        }

        id<MTLLibrary> defaultLibrary = [device newDefaultLibrary];
        if (!ResolvePipeline(
                device, defaultLibrary, @"add_row_bias_f32", "default.metallib"))
            return 3;

        NSURL* metannUrl = [[NSBundle mainBundle]
            URLForResource:@"MetaNN" withExtension:@"metallib"];
        if (metannUrl == nil)
        {
            std::cerr << "METAL_RUNTIME_LIBRARY_MISSING,library=MetaNN.metallib\n";
            return 4;
        }
        NSError* error = nil;
        id<MTLLibrary> metannLibrary =
            [device newLibraryWithURL:metannUrl error:&error];
        if (metannLibrary == nil)
        {
            std::cerr << "METAL_RUNTIME_LIBRARY_LOAD_FAILED,library=MetaNN.metallib,error="
                      << (error == nil
                              ? "unknown"
                              : error.localizedDescription.UTF8String)
                      << '\n';
            return 5;
        }
        if (!ResolvePipeline(
                device, metannLibrary, @"vector_add_f32", "MetaNN.metallib"))
            return 6;

        std::cout << "MetalRuntimeResolutionTests passed\n";
        return 0;
    }
}
