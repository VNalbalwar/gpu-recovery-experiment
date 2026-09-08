#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                          \
            std::exit(EXIT_FAILURE);                         \
        }                                                     \
    } while (0)

__device__ __forceinline__
unsigned long long global_timer_ns()
{
    unsigned long long t;

    asm volatile(
        "mov.u64 %0, %%globaltimer;"
        : "=l"(t));

    return t;
}

__global__
void bandwidth_interferer(float* __restrict__ buffer,
                          size_t n,
                          unsigned long long duration_ns)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned long long start = global_timer_ns();

    float x = 0.5f;

    while (global_timer_ns() - start < duration_ns) {
        size_t pos = (idx * 4096ULL) % n;

        x += buffer[pos];
        buffer[pos] = x;

        idx += gridDim.x * blockDim.x;

        if (idx >= n)
            idx %= n;
    }
}

int main()
{
    constexpr int THREADS = 256;
    constexpr int BLOCKS = 8;
    constexpr int WARMUP_RUNS = 3;
    constexpr int TARGET_RUNS = 20;

    constexpr size_t INTERFERER_ELEMENTS =
        256ULL * 1024ULL * 1024ULL; // 1 GiB

    constexpr unsigned long long DURATION_NS =
        100ULL * 1'000'000ULL; // 100 ms

    float* buffer = nullptr;

    CUDA_CHECK(cudaMalloc(
        &buffer,
        INTERFERER_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaMemset(
        buffer,
        1,
        INTERFERER_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());

    // Warmup
    for (int i = 0; i < WARMUP_RUNS; ++i) {
        bandwidth_interferer<<<BLOCKS, THREADS>>>(
            buffer,
            INTERFERER_ELEMENTS,
            DURATION_NS);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Target runs
    for (int i = 0; i < TARGET_RUNS; ++i) {
        bandwidth_interferer<<<BLOCKS, THREADS>>>(
            buffer,
            INTERFERER_ELEMENTS,
            DURATION_NS);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(buffer));

    return 0;
}
