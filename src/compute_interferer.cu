#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS 256
#define BLOCKS 8
#define DURATION_MS 100

__device__ __forceinline__ unsigned long long global_timer_ns()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void compute_interferer(unsigned long long duration_ns)
{
    // Multiple independent accumulators help expose instruction-level
    // parallelism and keep the FP32 pipelines busy.
    float x0 = 1.0f;
    float x1 = 2.0f;
    float x2 = 3.0f;
    float x3 = 4.0f;
    float x4 = 5.0f;
    float x5 = 6.0f;
    float x6 = 7.0f;
    float x7 = 8.0f;

    const unsigned long long start = global_timer_ns();

    while (global_timer_ns() - start < duration_ns)
    {
#pragma unroll
        for (int i = 0; i < 32; ++i)
        {
            x0 = fmaf(x0, 1.000001f, x1);
            x1 = fmaf(x1, 0.999999f, x2);
            x2 = fmaf(x2, 1.000002f, x3);
            x3 = fmaf(x3, 0.999998f, x4);

            x4 = fmaf(x4, 1.000003f, x5);
            x5 = fmaf(x5, 0.999997f, x6);
            x6 = fmaf(x6, 1.000004f, x7);
            x7 = fmaf(x7, 0.999996f, x0);
        }
    }

    // Keep the computation observable.
    if (x0 == -1.0f)
        printf("%f\n", x0);
}

int main()
{
    const unsigned long long duration_ns =
        static_cast<unsigned long long>(DURATION_MS) * 1000000ULL;

    printf("Compute Interferer\n");
    printf("Blocks: %d\n", BLOCKS);
    printf("Threads: %d\n", THREADS);
    printf("Duration: %d ms\n", DURATION_MS);

    compute_interferer<<<BLOCKS, THREADS>>>(duration_ns);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed: %s\n",
                cudaGetErrorString(err));
        return 1;
    }

    printf("Done.\n");

    return 0;
}
