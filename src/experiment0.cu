#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " -> " << cudaGetErrorString(err) << "\n";          \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


// ============================================================
// Victim kernel
// ============================================================
//
// A memory-accessing workload whose execution latency is
// measured for every individual invocation.
//
// ============================================================

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


// ============================================================
// DRAM bandwidth interferer
// ============================================================
//
// Runs for duration_ns according to the GPU global timer.
//
// The 1 GiB allocation is much larger than the RTX 4060
// Laptop GPU's L2 cache, so accesses cannot simply reside
// entirely inside L2.
//
// ============================================================

__device__ __forceinline__
unsigned long long global_timer_ns()
{
    unsigned long long t;

    asm volatile(
        "mov.u64 %0, %%globaltimer;"
        : "=l"(t)
    );

    return t;
}


__global__
void bandwidth_interferer(float* __restrict__ buffer,
                          size_t n,
                          unsigned long long duration_ns)
{
    size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned long long start =
        global_timer_ns();

    float x = 0.5f;

    while (global_timer_ns() - start < duration_ns) {

        size_t pos = idx;

        /*
         * Spread accesses through the large allocation.
         */
        pos = (pos * 4096ULL) % n;

        x += buffer[pos];

        buffer[pos] = x;

        idx +=
            gridDim.x * blockDim.x;

        if (idx >= n)
            idx %= n;
    }
}


// ============================================================
// Statistics
// ============================================================

double median(std::vector<double> values)
{
    if (values.empty())
        return 0.0;

    std::sort(values.begin(), values.end());

    const size_t n = values.size();

    if (n % 2 == 0)
        return (values[n / 2 - 1] +
                values[n / 2]) / 2.0;

    return values[n / 2];
}


double percentile(std::vector<double> values,
                  double p)
{
    if (values.empty())
        return 0.0;

    std::sort(values.begin(), values.end());

    const double position =
        p * static_cast<double>(values.size() - 1);

    const size_t lower =
        static_cast<size_t>(position);

    const size_t upper =
        lower + 1;

    if (upper >= values.size())
        return values.back();

    const double fraction =
        position - lower;

    return values[lower] * (1.0 - fraction) +
           values[upper] * fraction;
}


// ============================================================
// Measure one victim invocation
// ============================================================
//
// The synchronization occurs OUTSIDE the CUDA event interval.
//
// Therefore victim_latency_ms measures:
//
//     start event
//          ↓
//     victim kernel
//          ↓
//     stop event
//
// and not the host synchronization overhead.
//
// ============================================================

float run_victim_sample(
    const float* input,
    float* output,
    size_t n,
    int iterations,
    int blocks,
    cudaStream_t stream,
    cudaEvent_t start_event,
    cudaEvent_t stop_event)
{
    CUDA_CHECK(cudaEventRecord(
        start_event,
        stream));

    victim_kernel<<<
        blocks,
        256,
        0,
        stream
    >>>(
        input,
        output,
        n,
        iterations
    );

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(
        stop_event,
        stream));

    /*
     * Wait until this victim invocation is complete.
     *
     * This is intentional: it establishes a true temporal
     * sequence between victim samples.
     */
    CUDA_CHECK(cudaEventSynchronize(
        stop_event));

    float elapsed_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms,
        start_event,
        stop_event));

    return elapsed_ms;
}


// ============================================================
// Main
// ============================================================

int main(int argc, char** argv)
{
    // --------------------------------------------------------
    // Configuration
    // --------------------------------------------------------

    constexpr int WARMUP_SAMPLES = 30;

    constexpr int BASELINE_SAMPLES = 100;

    constexpr int INTERFERENCE_SAMPLES = 100;

    constexpr int RECOVERY_SAMPLES = 100;

    constexpr int THREADS = 256;

    /*
     * 16M floats = 64 MiB.
     */
    constexpr size_t VICTIM_ELEMENTS =
        16ULL * 1024ULL * 1024ULL;

    /*
     * 256M floats = 1 GiB.
     *
     * The RTX 4060 Laptop GPU has 8 GiB VRAM,
     * so this leaves plenty of headroom.
     */
    constexpr size_t INTERFERER_ELEMENTS =
        256ULL * 1024ULL * 1024ULL;

    constexpr int VICTIM_ITERATIONS = 4;

    /*
     * Start conservatively.
     *
     * We want measurable memory contention while still
     * allowing the victim to execute.
     */
    constexpr int INTERFERER_BLOCKS = 8;

    double interference_ms = 10.0;

    std::string output_file =
        "results/experiment0_10ms.csv";

    /*
     * Usage:
     *
     * ./experiment0 10 results/test.csv
     */
    if (argc >= 2)
        interference_ms = std::stod(argv[1]);

    if (argc >= 3)
        output_file = argv[2];


    // --------------------------------------------------------
    // GPU setup
    // --------------------------------------------------------

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDeviceProperties(
        &prop,
        0));

    std::cout
        << "\n============================================\n"
        << " Experiment 0: Transient DRAM Interference\n"
        << "============================================\n\n";

    std::cout
        << "GPU:              "
        << prop.name
        << "\n";

    std::cout
        << "SM count:         "
        << prop.multiProcessorCount
        << "\n";

    std::cout
        << "Global memory:    "
        << std::fixed
        << std::setprecision(2)
        << prop.totalGlobalMem /
           (1024.0 * 1024.0 * 1024.0)
        << " GiB\n";

    std::cout
        << "Interference:     "
        << interference_ms
        << " ms\n";

    std::cout
        << "Interferer grid:  "
        << INTERFERER_BLOCKS
        << " blocks x "
        << THREADS
        << " threads\n";

    std::cout
        << "Output:           "
        << output_file
        << "\n\n";


    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------

    std::cout
        << "Allocating GPU memory...\n";

    float* victim_input = nullptr;

    float* victim_output = nullptr;

    float* interferer_buffer = nullptr;

    CUDA_CHECK(cudaMalloc(
        &victim_input,
        VICTIM_ELEMENTS *
        sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &victim_output,
        VICTIM_ELEMENTS *
        sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &interferer_buffer,
        INTERFERER_ELEMENTS *
        sizeof(float)));


    // --------------------------------------------------------
    // Initialize buffers
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemset(
        victim_input,
        1,
        VICTIM_ELEMENTS *
        sizeof(float)));

    CUDA_CHECK(cudaMemset(
        victim_output,
        0,
        VICTIM_ELEMENTS *
        sizeof(float)));

    CUDA_CHECK(cudaMemset(
        interferer_buffer,
        1,
        INTERFERER_ELEMENTS *
        sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // Create independent streams
    // --------------------------------------------------------

    cudaStream_t victim_stream;

    cudaStream_t interferer_stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &victim_stream,
        cudaStreamNonBlocking));

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &interferer_stream,
        cudaStreamNonBlocking));


    // --------------------------------------------------------
    // CUDA events
    // --------------------------------------------------------

    cudaEvent_t start_event;

    cudaEvent_t stop_event;

    CUDA_CHECK(cudaEventCreate(
        &start_event));

    CUDA_CHECK(cudaEventCreate(
        &stop_event));


    // --------------------------------------------------------
    // Victim launch configuration
    // --------------------------------------------------------

    const int victim_blocks =
        static_cast<int>(
            (VICTIM_ELEMENTS +
             THREADS - 1) /
            THREADS);


    // --------------------------------------------------------
    // Warm-up
    // --------------------------------------------------------

    std::cout
        << "Warming up GPU...\n";

    for (int i = 0;
         i < WARMUP_SAMPLES;
         ++i)
    {
        run_victim_sample(
            victim_input,
            victim_output,
            VICTIM_ELEMENTS,
            VICTIM_ITERATIONS,
            victim_blocks,
            victim_stream,
            start_event,
            stop_event);
    }


    // --------------------------------------------------------
    // Let the GPU settle
    // --------------------------------------------------------

    std::cout
        << "Settling GPU...\n";

    CUDA_CHECK(cudaDeviceSynchronize());

    std::this_thread::sleep_for(
        std::chrono::milliseconds(500));

    CUDA_CHECK(cudaDeviceSynchronize());


    // --------------------------------------------------------
    // Result storage
    // --------------------------------------------------------

    std::vector<double> baseline;
    std::vector<double> interference;
    std::vector<double> recovery;

    baseline.reserve(BASELINE_SAMPLES);

    interference.reserve(
        INTERFERENCE_SAMPLES);

    recovery.reserve(
        RECOVERY_SAMPLES);


    // --------------------------------------------------------
    // CSV
    // --------------------------------------------------------

    std::ofstream csv(output_file);

    if (!csv.is_open()) {

        std::cerr
            << "ERROR: Could not open "
            << output_file
            << "\n";

        return EXIT_FAILURE;
    }

    csv
        << "sample,"
        << "phase,"
        << "victim_latency_ms\n";


    // --------------------------------------------------------
    // BASELINE
    // --------------------------------------------------------

    std::cout
        << "\n--------------------------------------------\n"
        << " Phase 1: BASELINE\n"
        << "--------------------------------------------\n";

    for (int i = 0;
         i < BASELINE_SAMPLES;
         ++i)
    {
        const float latency =
            run_victim_sample(
                victim_input,
                victim_output,
                VICTIM_ELEMENTS,
                VICTIM_ITERATIONS,
                victim_blocks,
                victim_stream,
                start_event,
                stop_event);

        baseline.push_back(latency);

        csv
            << i
            << ",baseline,"
            << std::setprecision(9)
            << latency
            << "\n";
    }


    // --------------------------------------------------------
    // START INTERFERENCE
    // --------------------------------------------------------

    const unsigned long long duration_ns =
        static_cast<unsigned long long>(
            interference_ms *
            1'000'000.0);

    std::cout
        << "\n>>> STARTING "
        << interference_ms
        << " ms DRAM INTERFERENCE\n";


    CUDA_CHECK(cudaEventRecord(
        start_event,
        interferer_stream));

    /*
     * The interferer is launched on a different stream.
     */
    bandwidth_interferer<<<
        INTERFERER_BLOCKS,
        THREADS,
        0,
        interferer_stream
    >>>(
        interferer_buffer,
        INTERFERER_ELEMENTS,
        duration_ns
    );

    CUDA_CHECK(cudaGetLastError());


    // --------------------------------------------------------
    // INTERFERENCE
    // --------------------------------------------------------

    std::cout
        << "\n--------------------------------------------\n"
        << " Phase 2: INTERFERENCE\n"
        << "--------------------------------------------\n";

    for (int i = 0;
         i < INTERFERENCE_SAMPLES;
         ++i)
    {
        const float latency =
            run_victim_sample(
                victim_input,
                victim_output,
                VICTIM_ELEMENTS,
                VICTIM_ITERATIONS,
                victim_blocks,
                victim_stream,
                start_event,
                stop_event);

        interference.push_back(latency);

        csv
            << BASELINE_SAMPLES + i
            << ",interference,"
            << std::setprecision(9)
            << latency
            << "\n";
    }


    // --------------------------------------------------------
    // IMPORTANT:
    //
    // Wait until the interferer has ACTUALLY FINISHED.
    // --------------------------------------------------------

    CUDA_CHECK(cudaStreamSynchronize(
        interferer_stream));

    std::cout
        << "\n>>> INTERFERENCE FINISHED\n";


    // --------------------------------------------------------
    // RECOVERY
    // --------------------------------------------------------

    std::cout
        << "\n--------------------------------------------\n"
        << " Phase 3: RECOVERY\n"
        << "--------------------------------------------\n";

    for (int i = 0;
         i < RECOVERY_SAMPLES;
         ++i)
    {
        const float latency =
            run_victim_sample(
                victim_input,
                victim_output,
                VICTIM_ELEMENTS,
                VICTIM_ITERATIONS,
                victim_blocks,
                victim_stream,
                start_event,
                stop_event);

        recovery.push_back(latency);

        csv
            << BASELINE_SAMPLES +
               INTERFERENCE_SAMPLES +
               i
            << ",recovery,"
            << std::setprecision(9)
            << latency
            << "\n";
    }


    csv.close();


    // --------------------------------------------------------
    // Statistics
    // --------------------------------------------------------

    const double baseline_median =
        median(baseline);

    const double interference_median =
        median(interference);

    const double recovery_median =
        median(recovery);


    std::cout
        << "\n============================================\n"
        << " Experiment 0 Summary\n"
        << "============================================\n";

    std::cout
        << "Baseline median:      "
        << baseline_median
        << " ms\n";

    std::cout
        << "Interference median:  "
        << interference_median
        << " ms\n";

    std::cout
        << "Recovery median:      "
        << recovery_median
        << " ms\n";


    if (baseline_median > 0.0) {

        std::cout
            << "Interference slowdown:"
            << " "
            << interference_median /
               baseline_median
            << "x\n";

        std::cout
            << "Recovery / baseline:  "
            << recovery_median /
               baseline_median
            << "x\n";
    }


    // --------------------------------------------------------
    // Percentiles
    // --------------------------------------------------------

    std::cout
        << "\nP50 / P95 / P99\n";

    std::cout
        << "Baseline:     "
        << percentile(baseline, 0.50)
        << " / "
        << percentile(baseline, 0.95)
        << " / "
        << percentile(baseline, 0.99)
        << " ms\n";

    std::cout
        << "Interference: "
        << percentile(interference, 0.50)
        << " / "
        << percentile(interference, 0.95)
        << " / "
        << percentile(interference, 0.99)
        << " ms\n";

    std::cout
        << "Recovery:     "
        << percentile(recovery, 0.50)
        << " / "
        << percentile(recovery, 0.95)
        << " / "
        << percentile(recovery, 0.99)
        << " ms\n";


    std::cout
        << "\nResults saved to:\n"
        << output_file
        << "\n";

    std::cout
        << "============================================\n";


    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    cudaEventDestroy(start_event);

    cudaEventDestroy(stop_event);

    cudaStreamDestroy(victim_stream);

    cudaStreamDestroy(interferer_stream);

    cudaFree(victim_input);

    cudaFree(victim_output);

    cudaFree(interferer_buffer);

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
