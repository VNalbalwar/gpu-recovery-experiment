#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>

#define WARMUP_SAMPLES 30
#define BASELINE_SAMPLES 100
#define INTERFERENCE_TARGET_SAMPLES 100
#define RECOVERY_SAMPLES 100

#define THREADS 256

#define DEFAULT_INTERFERENCE_MS 100
#define DEFAULT_VICTIM_MEMORY_MIB 64
#define DEFAULT_INTERFERER_BLOCKS 8
#define VICTIM_ITERATIONS 4

#define CHECK_CUDA(call)                                                   \
    do                                                                    \
    {                                                                     \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__device__ __forceinline__ unsigned long long global_timer_ns()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}


// -------------------------------------------------------------------------
// Victim kernel
// -------------------------------------------------------------------------
__global__ void victim_kernel(
    const float *__restrict__ input,
    float *__restrict__ output,
    size_t n,
    int iterations)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n)
        return;

    float x = input[idx];

#pragma unroll 1
    for (int i = 0; i < iterations; ++i)
    {
        x = x * 1.000001f + 0.000001f;
        x = x * 0.999999f + 0.000002f;
    }

    output[idx] = x;
}


// -------------------------------------------------------------------------
// Compute-focused interferer
// -------------------------------------------------------------------------
__global__ void compute_interferer(
    unsigned long long duration_ns)
{
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

    // Prevent the compiler from treating the calculation as dead code.
    if (x0 == -1.0f)
        printf("%f\n", x0);
}


// -------------------------------------------------------------------------
// Timing helper
// -------------------------------------------------------------------------
float run_victim_sample(
    float *input,
    float *output,
    size_t elements,
    int iterations,
    cudaStream_t stream,
    cudaEvent_t start_event,
    cudaEvent_t stop_event)
{
    const int blocks =
        static_cast<int>((elements + THREADS - 1) / THREADS);

    CHECK_CUDA(cudaEventRecord(start_event, stream));

    victim_kernel<<<blocks, THREADS, 0, stream>>>(
        input,
        output,
        elements,
        iterations);

    CHECK_CUDA(cudaEventRecord(stop_event, stream));
    CHECK_CUDA(cudaEventSynchronize(stop_event));

    float elapsed_ms = 0.0f;

    CHECK_CUDA(cudaEventElapsedTime(
        &elapsed_ms,
        start_event,
        stop_event));

    return elapsed_ms;
}


// -------------------------------------------------------------------------
// Statistics
// -------------------------------------------------------------------------
float median(std::vector<float> values)
{
    if (values.empty())
        return 0.0f;

    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n % 2 == 0)
        return (values[n / 2 - 1] + values[n / 2]) * 0.5f;

    return values[n / 2];
}


float percentile(
    std::vector<float> values,
    double p)
{
    if (values.empty())
        return 0.0f;

    std::sort(values.begin(), values.end());

    double index = p * static_cast<double>(values.size() - 1);

    size_t lower = static_cast<size_t>(index);
    size_t upper = lower + 1;

    if (upper >= values.size())
        return values.back();

    double fraction = index - static_cast<double>(lower);

    return static_cast<float>(
        values[lower] * (1.0 - fraction) +
        values[upper] * fraction);
}


void print_statistics(
    const char *name,
    const std::vector<float> &samples)
{
    printf(
        "%s:\n"
        "  median = %.4f ms\n"
        "  P50    = %.4f ms\n"
        "  P95    = %.4f ms\n"
        "  P99    = %.4f ms\n",
        name,
        median(samples),
        percentile(samples, 0.50),
        percentile(samples, 0.95),
        percentile(samples, 0.99));
}


// -------------------------------------------------------------------------
// Main experiment
// -------------------------------------------------------------------------
int main(int argc, char **argv)
{
    int interference_ms = DEFAULT_INTERFERENCE_MS;
    const char *output_file = "results/compute_100ms.csv";
    int victim_memory_mib = DEFAULT_VICTIM_MEMORY_MIB;
    int interferer_blocks = DEFAULT_INTERFERER_BLOCKS;

    if (argc >= 2)
        interference_ms = atoi(argv[1]);

    if (argc >= 3)
        output_file = argv[2];

    if (argc >= 4)
        victim_memory_mib = atoi(argv[3]);

    if (argc >= 5)
        interferer_blocks = atoi(argv[4]);

    if (interference_ms <= 0)
    {
        fprintf(stderr, "Interference duration must be > 0 ms\n");
        return EXIT_FAILURE;
    }

    if (victim_memory_mib <= 0)
    {
        fprintf(stderr, "Victim memory must be > 0 MiB\n");
        return EXIT_FAILURE;
    }

    if (interferer_blocks < 1 || interferer_blocks > 64)
    {
        fprintf(stderr, "Interferer blocks must be between 1 and 64\n");
        return EXIT_FAILURE;
    }


    // ---------------------------------------------------------------------
    // Experiment configuration
    // ---------------------------------------------------------------------

    const size_t victim_bytes =
        static_cast<size_t>(victim_memory_mib) *
        1024ULL * 1024ULL;

    const size_t victim_elements =
        victim_bytes / sizeof(float);

    const unsigned long long interference_duration_ns =
        static_cast<unsigned long long>(interference_ms) *
        1000000ULL;


    printf("========================================\n");
    printf("Compute Interference Recovery Experiment\n");
    printf("========================================\n");
    printf("Interference: %d ms\n", interference_ms);
    printf("Victim memory: %d MiB\n", victim_memory_mib);
    printf("Interferer blocks: %d\n", interferer_blocks);
    printf("Threads/block: %d\n", THREADS);
    printf("Victim iterations: %d\n", VICTIM_ITERATIONS);
    printf("========================================\n");


    // ---------------------------------------------------------------------
    // Allocate memory
    // ---------------------------------------------------------------------

    float *d_input = nullptr;
    float *d_output = nullptr;

    CHECK_CUDA(cudaMalloc(
        &d_input,
        victim_bytes));

    CHECK_CUDA(cudaMalloc(
        &d_output,
        victim_bytes));

    CHECK_CUDA(cudaMemset(
        d_input,
        0,
        victim_bytes));


    // ---------------------------------------------------------------------
    // Streams and events
    // ---------------------------------------------------------------------

    cudaStream_t victim_stream;
    cudaStream_t interference_stream;

    CHECK_CUDA(cudaStreamCreateWithFlags(
        &victim_stream,
        cudaStreamNonBlocking));

    CHECK_CUDA(cudaStreamCreateWithFlags(
        &interference_stream,
        cudaStreamNonBlocking));


    cudaEvent_t victim_start_event;
    cudaEvent_t victim_stop_event;

    cudaEvent_t interference_start_event;
    cudaEvent_t interference_end_event;

    CHECK_CUDA(cudaEventCreate(&victim_start_event));
    CHECK_CUDA(cudaEventCreate(&victim_stop_event));

    CHECK_CUDA(cudaEventCreate(&interference_start_event));
    CHECK_CUDA(cudaEventCreate(&interference_end_event));


    // ---------------------------------------------------------------------
    // Warm-up
    // ---------------------------------------------------------------------

    printf("\nWarming up...\n");

    for (int i = 0; i < WARMUP_SAMPLES; ++i)
    {
        run_victim_sample(
            d_input,
            d_output,
            victim_elements,
            VICTIM_ITERATIONS,
            victim_stream,
            victim_start_event,
            victim_stop_event);
    }


    // ---------------------------------------------------------------------
    // Baseline
    // ---------------------------------------------------------------------

    printf("Collecting baseline samples...\n");

    std::vector<float> baseline_samples;
    baseline_samples.reserve(BASELINE_SAMPLES);

    for (int i = 0; i < BASELINE_SAMPLES; ++i)
    {
        float latency = run_victim_sample(
            d_input,
            d_output,
            victim_elements,
            VICTIM_ITERATIONS,
            victim_stream,
            victim_start_event,
            victim_stop_event);

        baseline_samples.push_back(latency);
    }


    // ---------------------------------------------------------------------
    // Interference
    // ---------------------------------------------------------------------

    printf("Starting compute interference...\n");

    std::vector<float> interference_samples;
    interference_samples.reserve(INTERFERENCE_TARGET_SAMPLES);

    CHECK_CUDA(cudaEventRecord(
        interference_start_event,
        interference_stream));

    compute_interferer<<<
        interferer_blocks,
        THREADS,
        0,
        interference_stream>>>(
            interference_duration_ns);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaEventRecord(
        interference_end_event,
        interference_stream));


    // Collect victim samples while interference is active.

    while (static_cast<int>(interference_samples.size()) <
           INTERFERENCE_TARGET_SAMPLES)
    {
        float latency = run_victim_sample(
            d_input,
            d_output,
            victim_elements,
            VICTIM_ITERATIONS,
            victim_stream,
            victim_start_event,
            victim_stop_event);

        interference_samples.push_back(latency);

        cudaError_t query =
            cudaEventQuery(interference_end_event);

        if (query == cudaSuccess)
            break;

        if (query != cudaErrorNotReady)
        {
            CHECK_CUDA(query);
        }
    }

    CHECK_CUDA(cudaEventSynchronize(
        interference_end_event));


    // ---------------------------------------------------------------------
    // Recovery
    // ---------------------------------------------------------------------

    printf("Interference ended. Collecting recovery samples...\n");

    // Ensure recovery launches occur after the GPU-side interference
    // completion event.
    CHECK_CUDA(cudaStreamWaitEvent(
        victim_stream,
        interference_end_event,
        0));


    std::vector<float> recovery_samples;
    std::vector<double> recovery_times_ms;

    recovery_samples.reserve(RECOVERY_SAMPLES);
    recovery_times_ms.reserve(RECOVERY_SAMPLES);


    for (int i = 0; i < RECOVERY_SAMPLES; ++i)
    {
        // Record the victim start event after waiting for the
        // interference-end event.
        CHECK_CUDA(cudaEventRecord(
            victim_start_event,
            victim_stream));

        const int blocks =
            static_cast<int>(
                (victim_elements + THREADS - 1) / THREADS);

        victim_kernel<<<
            blocks,
            THREADS,
            0,
            victim_stream>>>(
                d_input,
                d_output,
                victim_elements,
                VICTIM_ITERATIONS);

        CHECK_CUDA(cudaEventRecord(
            victim_stop_event,
            victim_stream));

        CHECK_CUDA(cudaEventSynchronize(
            victim_stop_event));


        float latency_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(
            &latency_ms,
            victim_start_event,
            victim_stop_event));


        float elapsed_since_interference_ms = 0.0f;

        CHECK_CUDA(cudaEventElapsedTime(
            &elapsed_since_interference_ms,
            interference_end_event,
            victim_start_event));


        recovery_samples.push_back(latency_ms);
        recovery_times_ms.push_back(
            static_cast<double>(
                elapsed_since_interference_ms));
    }


    // ---------------------------------------------------------------------
    // Print statistics
    // ---------------------------------------------------------------------

    printf("\n========================================\n");
    printf("Experiment Results\n");
    printf("========================================\n");

    print_statistics(
        "Baseline",
        baseline_samples);

    print_statistics(
        "Interference",
        interference_samples);

    print_statistics(
        "Recovery",
        recovery_samples);


    const float baseline_median =
        median(baseline_samples);

    const float interference_median =
        median(interference_samples);

    const float recovery_median =
        median(recovery_samples);


    printf("\nDerived metrics:\n");

    printf(
        "  Interference slowdown: %.3fx\n",
        interference_median / baseline_median);

    printf(
        "  Recovery / baseline:   %.3fx\n",
        recovery_median / baseline_median);

    printf(
        "  First recovery sample:  %.4f ms\n",
        recovery_samples.front());

    printf(
        "  First recovery timestamp: %.4f ms\n",
        recovery_times_ms.front());

    printf(
        "  Interference samples: %zu\n",
        interference_samples.size());


    // ---------------------------------------------------------------------
    // CSV output
    // ---------------------------------------------------------------------

    FILE *fp = fopen(output_file, "w");

    if (!fp)
    {
        perror("Failed to open output file");
        return EXIT_FAILURE;
    }

    fprintf(
        fp,
        "sample,phase,victim_latency_ms,"
        "recovery_index,time_since_interference_end_ms\n");


    for (size_t i = 0; i < baseline_samples.size(); ++i)
    {
        fprintf(
            fp,
            "%zu,baseline,%.9f,,\n",
            i,
            baseline_samples[i]);
    }


    for (size_t i = 0; i < interference_samples.size(); ++i)
    {
        fprintf(
            fp,
            "%zu,interference,%.9f,,\n",
            i,
            interference_samples[i]);
    }


    for (size_t i = 0; i < recovery_samples.size(); ++i)
    {
        fprintf(
            fp,
            "%zu,recovery,%.9f,%zu,%.9f\n",
            i,
            recovery_samples[i],
            i,
            recovery_times_ms[i]);
    }

    fclose(fp);


    // ---------------------------------------------------------------------
    // Cleanup
    // ---------------------------------------------------------------------

    CHECK_CUDA(cudaEventDestroy(victim_start_event));
    CHECK_CUDA(cudaEventDestroy(victim_stop_event));

    CHECK_CUDA(cudaEventDestroy(interference_start_event));
    CHECK_CUDA(cudaEventDestroy(interference_end_event));

    CHECK_CUDA(cudaStreamDestroy(victim_stream));
    CHECK_CUDA(cudaStreamDestroy(interference_stream));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));


    printf("\nResults written to: %s\n", output_file);

    return EXIT_SUCCESS;
}
