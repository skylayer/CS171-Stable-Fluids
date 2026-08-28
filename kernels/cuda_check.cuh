//
// Error checking for the CUDA runtime calls that used to go unchecked.
//

#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include <cstdio>
#include <cstdlib>

/* A failed cudaMalloc used to return a null pointer that the next kernel
 * dereferenced, so the first symptom was an unrelated crash somewhere down the
 * pipeline. Report where it actually happened instead. */
// clang-format off
#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t err_ = (call);                                                      \
        if (err_ != cudaSuccess) {                                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__, __LINE__,        \
                         cudaGetErrorString(err_), #call);                                    \
            std::abort();                                                                     \
        }                                                                                     \
    } while (0)

/* Kernel launches fail asynchronously: cudaGetLastError picks up a bad launch
 * configuration, the synchronize picks up a fault inside the kernel. */
#define CUDA_CHECK_KERNEL()                                                                   \
    do {                                                                                      \
        CUDA_CHECK(cudaGetLastError());                                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                                  \
    } while (0)
// clang-format on

#endif // CUDA_CHECK_CUH
