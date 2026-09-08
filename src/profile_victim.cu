#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                           \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)

__global__
void victim_kernel(const float* __restrict__ input,
                   float* __restrict__ output,
                   size_t n,
                   int iterations)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    float x = input[idx];

    #pragma unroll 1
    for (int i = 0; i < iterations; ++i) {
        x = x * 1.000001f + 0.000001f;
        x = x * 0.999999f + 0.000002f;
    }

    output[idx] = x;
}

int main()
{
    constexpr size_t VICTIM_MIB = 1024;
    constexpr int THREADS = 256;
    constexpr int ITERATIONS = 4;
    constexpr int RUNS = 20;

    const size_t elements =
        VICTIM_MIB * 1024ULL * 1024ULL / sizeof(float);

    const size_t bytes = elements * sizeof(float);

    float* input = nullptr;
    float* output = nullptr;

    CUDA_CHECK(cudaMalloc(&input, bytes));
    CUDA_CHECK(cudaMalloc(&output, bytes));

    CUDA_CHECK(cudaMemset(input, 0, bytes));

    const int blocks =
        static_cast<int>((elements + THREADS - 1) / THREADS);

    // Warmup
    for (int i = 0; i < 5; ++i) {
        victim_kernel<<<blocks, THREADS>>>(
            input, output, elements, ITERATIONS);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Profiling target
    for (int i = 0; i < RUNS; ++i) {
        victim_kernel<<<blocks, THREADS>>>(
            input, output, elements, ITERATIONS);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(input));
    CUDA_CHECK(cudaFree(output));

    return 0;
}
