#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

#define CUDA_CHECK(call)                                  \
    do                                                    \
    {                                                     \
        cudaError_t err = (call);                         \
        if (err != cudaSuccess)                           \
        {                                                 \
            std::cerr << "CUDA error at " << __FILE__     \
                      << ":" << __LINE__ << " -> "        \
                      << cudaGetErrorString(err) << "\n"; \
            std::exit(EXIT_FAILURE);                      \
        }                                                 \
    } while (0)

__device__ __forceinline__ unsigned long long global_timer_ns()
{
    unsigned long long t;

    asm volatile(
        "mov.u64 %0, %%globaltimer;"
        : "=l"(t));

    return t;
}

__global__ void l2_interferer(
    float *__restrict__ buffer,
    size_t elements,
    unsigned long long duration_ns)
{
    const size_t tid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    float x = 1.0f;

    const unsigned long long start = global_timer_ns();

    while (global_timer_ns() - start < duration_ns)
    {
        size_t index =
            (tid * 32ULL) % elements;

#pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            float value;

            asm volatile(
                "ld.global.cg.f32 %0, [%1];"
                : "=f"(value)
                : "l"(&buffer[index]));

            x += value;
            x = x * 1.000001f + 0.000001f;

            asm volatile(
                "st.global.cg.f32 [%0], %1;"
                :
                : "l"(&buffer[index]), "f"(x));

            index += 32ULL;

            if (index >= elements)
                index -= elements;
        }
    }
}

int main(int argc, char **argv)
{
    constexpr int THREADS = 256;

    double duration_ms = 100.0;

    if (argc >= 2)
        duration_ms = std::stod(argv[1]);

    const int blocks =
        (argc >= 3) ? std::stoi(argv[2]) : 24;

    // 16 MiB working set.
    // Large enough to exceed L1 substantially while remaining
    // within the GPU's L2/cache hierarchy working range.
    constexpr size_t ELEMENTS =
        16ULL * 1024ULL * 1024ULL / sizeof(float);

    const unsigned long long duration_ns =
        static_cast<unsigned long long>(
            duration_ms * 1'000'000.0);

    std::cout
        << "============================================\n"
        << " L2 Interferer\n"
        << "============================================\n"
        << "Duration:       " << duration_ms << " ms\n"
        << "Blocks:         " << blocks << "\n"
        << "Threads/block:  " << THREADS << "\n"
        << "Working set:    16 MiB\n"
        << "============================================\n";

    float *buffer = nullptr;

    CUDA_CHECK(cudaMalloc(
        &buffer,
        ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaMemset(
        buffer,
        1,
        ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());

    l2_interferer<<<blocks, THREADS>>>(
        buffer,
        ELEMENTS,
        duration_ns);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(buffer));
    CUDA_CHECK(cudaDeviceReset());

    std::cout << "Done.\n";

    return EXIT_SUCCESS;
}
