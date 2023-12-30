#include <cuda_runtime.h>
#include <fmt/format.h>

int main() {
    cudaDeviceProp prop{};
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        fmt::print("Device Number: {}\n", i);
        fmt::print("  Device name: {}\n", prop.name);
        fmt::print("  Maximum threads per block: {}\n", prop.maxThreadsPerBlock);
        fmt::print("  Maximum block dimensions: {} x {} x {}\n",
                   prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        fmt::print("  Maximum grid dimensions: {} x {} x {}\n",
                   prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    return 0;
}
