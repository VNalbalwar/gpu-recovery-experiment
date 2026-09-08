#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
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
        return (values[n / 2 - 1] + values[n / 2]) / 2.0;

    return values[n / 2];
}

double percentile(std::vector<double> values, double p)
{
    if (values.empty())
        return 0.0;

    std::sort(values.begin(), values.end());

    const double position = p * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(position);
    const size_t upper = lower + 1;

    if (upper >= values.size())
        return values.back();

    const double fraction = position - lower;

    return values[lower] * (1.0 - fraction) +
           values[upper] * fraction;
}

// ============================================================
// Measure one victim invocation
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
    CUDA_CHECK(cudaEventRecord(start_event, stream));

    victim_kernel<<<blocks, 256, 0, stream>>>(
        input,
        output,
        n,
        iterations);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop_event, stream));

    // Synchronize outside the measured CUDA-event interval.
    CUDA_CHECK(cudaEventSynchronize(stop_event));

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

    // We target this many observations during interference,
    // but never continue collecting after the completion event.
    constexpr int INTERFERENCE_TARGET_SAMPLES = 100;

    constexpr int RECOVERY_SAMPLES = 100;
    constexpr int THREADS = 256;

    constexpr size_t VICTIM_ELEMENTS =
        16ULL * 1024ULL * 1024ULL; // 64 MiB

    constexpr size_t INTERFERER_ELEMENTS =
        256ULL * 1024ULL * 1024ULL; // 1 GiB

    constexpr int VICTIM_ITERATIONS = 4;
    constexpr int INTERFERER_BLOCKS = 8;

    double interference_ms = 50.0;

    std::string output_file =
        "results/experiment0_50ms.csv";

    if (argc >= 2)
        interference_ms = std::stod(argv[1]);

    if (argc >= 3)
        output_file = argv[2];

    if (interference_ms <= 0.0) {
        std::cerr << "ERROR: interference duration must be > 0 ms.\n";
        return EXIT_FAILURE;
    }

    // --------------------------------------------------------
    // GPU setup
    // --------------------------------------------------------

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout
        << "\n============================================\n"
        << " Experiment 0: Transient DRAM Interference\n"
        << "============================================\n\n"
        << "GPU:              " << prop.name << "\n"
        << "SM count:         " << prop.multiProcessorCount << "\n"
        << "Global memory:    " << std::fixed << std::setprecision(2)
        << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
        << " GiB\n"
        << "Interference:     " << interference_ms << " ms\n"
        << "Interferer grid:  " << INTERFERER_BLOCKS
        << " blocks x " << THREADS << " threads\n"
        << "Output:           " << output_file << "\n\n";

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------

    std::cout << "Allocating GPU memory...\n";

    float* victim_input = nullptr;
    float* victim_output = nullptr;
    float* interferer_buffer = nullptr;

    CUDA_CHECK(cudaMalloc(
        &victim_input,
        VICTIM_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &victim_output,
        VICTIM_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaMalloc(
        &interferer_buffer,
        INTERFERER_ELEMENTS * sizeof(float)));

    // --------------------------------------------------------
    // Initialize buffers
    // --------------------------------------------------------

    CUDA_CHECK(cudaMemset(
        victim_input,
        1,
        VICTIM_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaMemset(
        victim_output,
        0,
        VICTIM_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaMemset(
        interferer_buffer,
        1,
        INTERFERER_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // Create streams
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
    // Events
    // --------------------------------------------------------

    cudaEvent_t victim_start_event;
    cudaEvent_t victim_stop_event;
    cudaEvent_t interference_start_event;
    cudaEvent_t interference_end_event;

    CUDA_CHECK(cudaEventCreate(&victim_start_event));
    CUDA_CHECK(cudaEventCreate(&victim_stop_event));
    CUDA_CHECK(cudaEventCreate(&interference_start_event));
    CUDA_CHECK(cudaEventCreate(&interference_end_event));

    const int victim_blocks =
        static_cast<int>(
            (VICTIM_ELEMENTS + THREADS - 1) / THREADS);

    // --------------------------------------------------------
    // Warm-up
    // --------------------------------------------------------

    std::cout << "Warming up GPU...\n";

    for (int i = 0; i < WARMUP_SAMPLES; ++i) {
        run_victim_sample(
            victim_input,
            victim_output,
            VICTIM_ELEMENTS,
            VICTIM_ITERATIONS,
            victim_blocks,
            victim_stream,
            victim_start_event,
            victim_stop_event);
    }

    // --------------------------------------------------------
    // Settle
    // --------------------------------------------------------

    std::cout << "Settling GPU...\n";

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
    interference.reserve(INTERFERENCE_TARGET_SAMPLES);
    recovery.reserve(RECOVERY_SAMPLES);

    // --------------------------------------------------------
    // CSV
    // --------------------------------------------------------
    // recovery_index is -1 outside recovery and starts at 0 for
    // the FIRST victim invocation after interference_end_event.
    // The row's sample index remains globally ordered.
    // --------------------------------------------------------

    std::ofstream csv(output_file);

    if (!csv.is_open()) {
        std::cerr
            << "ERROR: Could not open "
            << output_file
            << "\n";
        return EXIT_FAILURE;
    }

    csv << "sample,phase,victim_latency_ms,recovery_index\n";

    // --------------------------------------------------------
    // BASELINE
    // --------------------------------------------------------

    std::cout
        << "\n--------------------------------------------\n"
        << " Phase 1: BASELINE\n"
        << "--------------------------------------------\n";

    for (int i = 0; i < BASELINE_SAMPLES; ++i) {
        const float latency = run_victim_sample(
            victim_input,
            victim_output,
            VICTIM_ELEMENTS,
            VICTIM_ITERATIONS,
            victim_blocks,
            victim_stream,
            victim_start_event,
            victim_stop_event);

        baseline.push_back(latency);

        csv << i
            << ",baseline,"
            << std::setprecision(9)
            << latency
            << ",-1\n";
    }

    // --------------------------------------------------------
    // START INTERFERENCE
    // --------------------------------------------------------

    const unsigned long long duration_ns =
        static_cast<unsigned long long>(
            interference_ms * 1'000'000.0);

    std::cout
        << "\n>>> STARTING "
        << interference_ms
        << " ms DRAM INTERFERENCE\n";

    // This marker is queued immediately before the interferer.
    CUDA_CHECK(cudaEventRecord(
        interference_start_event,
        interferer_stream));

    bandwidth_interferer<<<
        INTERFERER_BLOCKS,
        THREADS,
        0,
        interferer_stream
    >>>(
        interferer_buffer,
        INTERFERER_ELEMENTS,
        duration_ns);

    CUDA_CHECK(cudaGetLastError());

    // Queued immediately after the interferer kernel.
    // Completion of this event is the recovery boundary.
    CUDA_CHECK(cudaEventRecord(
        interference_end_event,
        interferer_stream));

    CUDA_CHECK(cudaEventSynchronize(
        interference_start_event));

    // --------------------------------------------------------
    // INTERFERENCE
    // --------------------------------------------------------

    std::cout
        << "\n--------------------------------------------\n"
        << " Phase 2: INTERFERENCE\n"
        << "--------------------------------------------\n";

    int interference_sample_count = 0;

    for (int i = 0;
         i < INTERFERENCE_TARGET_SAMPLES;
         ++i)
    {
        // Do not launch another victim after the GPU-side
        // completion marker has already passed.
        cudaError_t status =
            cudaEventQuery(interference_end_event);

        if (status == cudaSuccess)
            break;

        if (status != cudaErrorNotReady)
            CUDA_CHECK(status);

        const float latency = run_victim_sample(
            victim_input,
            victim_output,
            VICTIM_ELEMENTS,
            VICTIM_ITERATIONS,
            victim_blocks,
            victim_stream,
            victim_start_event,
            victim_stop_event);

        interference.push_back(latency);
        ++interference_sample_count;

        csv << BASELINE_SAMPLES + i
            << ",interference,"
            << std::setprecision(9)
            << latency
            << ",-1\n";
    }

    // Establish the exact end boundary before recovery sample 0.
    CUDA_CHECK(cudaEventSynchronize(interference_end_event));

    std::cout
        << "Interference victim samples collected: "
        << interference_sample_count
        << "\n"
        << ">>> INTERFERENCE FINISHED\n";

    if (interference.empty()) {
        std::cerr
            << "ERROR: No victim samples were collected while the "
               "interferer was active. Try a longer interference duration.\n";

        csv.close();

        CUDA_CHECK(cudaEventDestroy(victim_start_event));
        CUDA_CHECK(cudaEventDestroy(victim_stop_event));
        CUDA_CHECK(cudaEventDestroy(interference_start_event));
        CUDA_CHECK(cudaEventDestroy(interference_end_event));
        CUDA_CHECK(cudaStreamDestroy(victim_stream));
        CUDA_CHECK(cudaStreamDestroy(interferer_stream));
        CUDA_CHECK(cudaFree(victim_input));
        CUDA_CHECK(cudaFree(victim_output));
        CUDA_CHECK(cudaFree(interferer_buffer));
        CUDA_CHECK(cudaDeviceReset());
        return EXIT_FAILURE;
    }

    // --------------------------------------------------------
    // RECOVERY
    // --------------------------------------------------------

    std::cout
        << "\n--------------------------------------------\n"
        << " Phase 3: RECOVERY\n"
        << "--------------------------------------------\n";

    for (int i = 0; i < RECOVERY_SAMPLES; ++i) {
        const float latency = run_victim_sample(
            victim_input,
            victim_output,
            VICTIM_ELEMENTS,
            VICTIM_ITERATIONS,
            victim_blocks,
            victim_stream,
            victim_start_event,
            victim_stop_event);

        recovery.push_back(latency);

        csv << BASELINE_SAMPLES +
               interference_sample_count +
               i
            << ",recovery,"
            << std::setprecision(9)
            << latency
            << ","
            << i
            << "\n";
    }

    csv.close();

    // --------------------------------------------------------
    // Statistics
    // --------------------------------------------------------

    const double baseline_median = median(baseline);
    const double interference_median = median(interference);
    const double recovery_median = median(recovery);

    std::cout
        << "\n============================================\n"
        << " Experiment 0 Summary\n"
        << "============================================\n"
        << "Baseline median:      " << baseline_median << " ms\n"
        << "Interference median:  " << interference_median << " ms\n"
        << "Recovery median:      " << recovery_median << " ms\n";

    if (baseline_median > 0.0) {
        std::cout
            << "Interference slowdown: "
            << interference_median / baseline_median
            << "x\n"
            << "Recovery / baseline:   "
            << recovery_median / baseline_median
            << "x\n";
    }

    std::cout
        << "\nP50 / P95 / P99\n"
        << "Baseline:     "
        << percentile(baseline, 0.50) << " / "
        << percentile(baseline, 0.95) << " / "
        << percentile(baseline, 0.99) << " ms\n"
        << "Interference: "
        << percentile(interference, 0.50) << " / "
        << percentile(interference, 0.95) << " / "
        << percentile(interference, 0.99) << " ms\n"
        << "Recovery:     "
        << percentile(recovery, 0.50) << " / "
        << percentile(recovery, 0.95) << " / "
        << percentile(recovery, 0.99) << " ms\n";

    std::cout
        << "\nResults saved to:\n"
        << output_file
        << "\n"
        << "============================================\n";

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventDestroy(victim_start_event));
    CUDA_CHECK(cudaEventDestroy(victim_stop_event));
    CUDA_CHECK(cudaEventDestroy(interference_start_event));
    CUDA_CHECK(cudaEventDestroy(interference_end_event));

    CUDA_CHECK(cudaStreamDestroy(victim_stream));
    CUDA_CHECK(cudaStreamDestroy(interferer_stream));

    CUDA_CHECK(cudaFree(victim_input));
    CUDA_CHECK(cudaFree(victim_output));
    CUDA_CHECK(cudaFree(interferer_buffer));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
