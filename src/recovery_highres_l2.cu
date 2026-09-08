#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                   \
    do                                                                     \
    {                                                                      \
        cudaError_t err = (call);                                          \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " -> " << cudaGetErrorString(err) << "\n";        \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// ============================================================
// Configuration
// ============================================================

constexpr int THREADS = 256;
constexpr int VICTIM_BLOCKS = 8;

constexpr int WARMUP_SAMPLES = 30;
constexpr int BASELINE_SAMPLES = 100;
constexpr int RECOVERY_SAMPLES = 100;

constexpr int VICTIM_ITERATIONS = 4;

// Number of repeated operations used to make the probe longer.
// We will calibrate this experimentally.
constexpr int DEFAULT_WORK_REPETITIONS = 32;


// ============================================================
// Victim kernel
// ============================================================

__global__ void probe_victim(
    const float *__restrict__ input,
    float *__restrict__ output,
    int work_repetitions)
{
    const int tid =
        blockIdx.x * blockDim.x + threadIdx.x;

    const int total_threads =
        VICTIM_BLOCKS * THREADS;

    float x = input[tid];

#pragma unroll 1
    for (int r = 0; r < work_repetitions; ++r)
    {
#pragma unroll 1
        for (int i = 0; i < VICTIM_ITERATIONS; ++i)
        {
            x = x * 1.000001f + 0.000001f;
            x = x * 0.999999f + 0.000002f;
        }

        // Keep the compiler from eliminating the work.
        x += input[(tid + r) % total_threads] * 0.0000001f;
    }

    output[tid] = x;
}

// ============================================================
// GPU global timer
// ============================================================

__device__ __forceinline__ unsigned long long global_timer_ns()
{
    unsigned long long t;

    asm volatile(
        "mov.u64 %0, %%globaltimer;"
        : "=l"(t));

    return t;
}

// ============================================================
// L2 interferer
//
// Same basic L2 interferer used by experiment_l2.cu.
// 16 MiB working set + cache-global loads/stores.
// ============================================================

constexpr size_t L2_INTERFERER_ELEMENTS =
    16ULL * 1024ULL * 1024ULL / sizeof(float);

__global__ void l2_interferer(
    float *__restrict__ buffer,
    size_t elements,
    unsigned long long duration_ns)
{
    const size_t tid =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    float x = 1.0f;

    const unsigned long long start =
        global_timer_ns();

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

// ============================================================
// Median
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

// ============================================================
// Main
// ============================================================

int main(int argc, char **argv)
{
    // --------------------------------------------------------
    // Usage:
    //
    // ./recovery_highres_l2 \
    //     <interference_ms> \
    //     <output_file> \
    //     <work_repetitions> \
    //     <interferer_blocks>
    //
    // Example:
    //
    // ./build/recovery_highres_l2 \
    //     100 \
    //     results/hr_l2_100ms.csv \
    //     32 \
    //     24
    // --------------------------------------------------------

    double interference_ms = 100.0;

    std::string output_file =
        "results/hr_l2_100ms.csv";

    int work_repetitions =
        DEFAULT_WORK_REPETITIONS;

    int interferer_blocks = 24;

    if (argc >= 2)
        interference_ms = std::stod(argv[1]);

    if (argc >= 3)
        output_file = argv[2];

    if (argc >= 4)
        work_repetitions = std::stoi(argv[3]);

    if (argc >= 5)
        interferer_blocks = std::stoi(argv[4]);

    if (interference_ms <= 0.0)
    {
        std::cerr
            << "ERROR: interference duration must be > 0 ms.\n";
        return EXIT_FAILURE;
    }

    if (work_repetitions <= 0 ||
        work_repetitions > 10000)
    {
        std::cerr
            << "ERROR: work repetitions must be between "
               "1 and 10000.\n";
        return EXIT_FAILURE;
    }

    if (interferer_blocks <= 0 ||
        interferer_blocks > 32)
    {
        std::cerr
            << "ERROR: interferer blocks must be between "
               "1 and 32.\n";
        return EXIT_FAILURE;
    }

    // --------------------------------------------------------
    // GPU setup
    // --------------------------------------------------------

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};

    CUDA_CHECK(
        cudaGetDeviceProperties(&prop, 0));

    constexpr int total_threads =
        VICTIM_BLOCKS * THREADS;

    std::cout
        << "\n============================================\n"
        << " High-Resolution L2 Recovery Experiment\n"
        << "============================================\n\n"
        << "GPU:                 " << prop.name << "\n"
        << "SM count:            "
        << prop.multiProcessorCount << "\n"
        << "Victim blocks:       "
        << VICTIM_BLOCKS << "\n"
        << "Victim threads:      "
        << THREADS << "\n"
        << "Total victim threads:"
        << total_threads << "\n"
        << "Work repetitions:    "
        << work_repetitions << "\n"
        << "Victim iterations:   "
        << VICTIM_ITERATIONS << "\n"
        << "Interference:        "
        << interference_ms << " ms\n"
        << "Interferer grid:     "
        << interferer_blocks
        << " blocks x "
        << THREADS
        << " threads\n"
        << "Recovery probes:     "
        << RECOVERY_SAMPLES << "\n"
        << "Output:              "
        << output_file << "\n\n";

    // --------------------------------------------------------
    // Allocate victim memory.
    //
    // Only 8 blocks are used, so this is intentionally tiny.
    // --------------------------------------------------------

    const size_t victim_bytes =
        static_cast<size_t>(total_threads) *
        sizeof(float);

    float *victim_input = nullptr;
    float *victim_output = nullptr;
    float *interferer_buffer = nullptr;

    CUDA_CHECK(cudaMalloc(
        &victim_input,
        victim_bytes));

    CUDA_CHECK(cudaMalloc(
        &victim_output,
        victim_bytes));

    CUDA_CHECK(cudaMalloc(
        &interferer_buffer,
        L2_INTERFERER_ELEMENTS *
            sizeof(float)));

    CUDA_CHECK(cudaMemset(
        victim_input,
        1,
        victim_bytes));

    CUDA_CHECK(cudaMemset(
        victim_output,
        0,
        victim_bytes));

    CUDA_CHECK(cudaMemset(
        interferer_buffer,
        1,
        L2_INTERFERER_ELEMENTS *
            sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // Streams
    // --------------------------------------------------------

    cudaStream_t victim_stream;
    cudaStream_t interferer_stream;
    cudaStream_t recovery_stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &victim_stream,
        cudaStreamNonBlocking));

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &interferer_stream,
        cudaStreamNonBlocking));

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &recovery_stream,
        cudaStreamNonBlocking));

    // --------------------------------------------------------
    // Warmup
    // --------------------------------------------------------

    cudaEvent_t warmup_start;
    cudaEvent_t warmup_stop;

    CUDA_CHECK(cudaEventCreate(
        &warmup_start));

    CUDA_CHECK(cudaEventCreate(
        &warmup_stop));

    std::cout
        << "Warming up GPU...\n";

    for (int i = 0;
         i < WARMUP_SAMPLES;
         ++i)
    {
        CUDA_CHECK(cudaEventRecord(
            warmup_start,
            victim_stream));

        probe_victim<<<
            VICTIM_BLOCKS,
            THREADS,
            0,
            victim_stream>>>(
                victim_input,
                victim_output,
                work_repetitions);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(
            warmup_stop,
            victim_stream));

        CUDA_CHECK(cudaEventSynchronize(
            warmup_stop));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // --------------------------------------------------------
    // Baseline measurement
    // --------------------------------------------------------

    std::vector<cudaEvent_t> baseline_start(
        BASELINE_SAMPLES);

    std::vector<cudaEvent_t> baseline_stop(
        BASELINE_SAMPLES);

    for (int i = 0;
         i < BASELINE_SAMPLES;
         ++i)
    {
        CUDA_CHECK(cudaEventCreate(
            &baseline_start[i]));

        CUDA_CHECK(cudaEventCreate(
            &baseline_stop[i]));
    }

    std::cout
        << "\n--------------------------------------------\n"
        << " Phase 1: BASELINE\n"
        << "--------------------------------------------\n";

    for (int i = 0;
         i < BASELINE_SAMPLES;
         ++i)
    {
        CUDA_CHECK(cudaEventRecord(
            baseline_start[i],
            victim_stream));

        probe_victim<<<
            VICTIM_BLOCKS,
            THREADS,
            0,
            victim_stream>>>(
                victim_input,
                victim_output,
                work_repetitions);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(
            baseline_stop[i],
            victim_stream));
    }

    CUDA_CHECK(cudaEventSynchronize(
        baseline_stop.back()));

    std::vector<double> baseline_latency_us;

    baseline_latency_us.reserve(
        BASELINE_SAMPLES);

    for (int i = 0;
         i < BASELINE_SAMPLES;
         ++i)
    {
        float elapsed_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms,
            baseline_start[i],
            baseline_stop[i]));

        baseline_latency_us.push_back(
            static_cast<double>(elapsed_ms) *
            1000.0);
    }

    const double baseline_median_us =
        median(baseline_latency_us);

    std::cout
        << "Baseline median: "
        << std::fixed
        << std::setprecision(3)
        << baseline_median_us
        << " us\n";

    // --------------------------------------------------------
    // Interference events
    // --------------------------------------------------------

    cudaEvent_t interference_start_event;
    cudaEvent_t interference_end_event;

    CUDA_CHECK(cudaEventCreate(
        &interference_start_event));

    CUDA_CHECK(cudaEventCreate(
        &interference_end_event));

    const unsigned long long duration_ns =
        static_cast<unsigned long long>(
            interference_ms *
            1'000'000.0);

    // --------------------------------------------------------
    // Start interference.
    //
    // IMPORTANT:
    // No host synchronization happens here.
    // --------------------------------------------------------

    std::cout
        << "\n>>> QUEUING "
        << interference_ms
        << " ms L2 INTERFERENCE\n";

    CUDA_CHECK(cudaEventRecord(
        interference_start_event,
        interferer_stream));

    l2_interferer<<<
        interferer_blocks,
        THREADS,
        0,
        interferer_stream>>>(
        interferer_buffer,
        L2_INTERFERER_ELEMENTS,
        duration_ns);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(
        interference_end_event,
        interferer_stream));

    // --------------------------------------------------------
    // GPU-side dependency:
    //
    // interference end
    //       ↓
    // recovery stream
    //
    // The CPU does NOT wait here.
    // --------------------------------------------------------

    CUDA_CHECK(cudaStreamWaitEvent(
        recovery_stream,
        interference_end_event,
        0));

    // --------------------------------------------------------
    // Recovery events
    // --------------------------------------------------------

    std::vector<cudaEvent_t> recovery_start(
        RECOVERY_SAMPLES);

    std::vector<cudaEvent_t> recovery_stop(
        RECOVERY_SAMPLES);

    for (int i = 0;
         i < RECOVERY_SAMPLES;
         ++i)
    {
        CUDA_CHECK(cudaEventCreate(
            &recovery_start[i]));

        CUDA_CHECK(cudaEventCreate(
            &recovery_stop[i]));
    }

    // --------------------------------------------------------
    // Queue ALL recovery probes.
    //
    // No CPU synchronization occurs between probes.
    // --------------------------------------------------------

    std::cout
        << ">>> QUEUING "
        << RECOVERY_SAMPLES
        << " HIGH-RES RECOVERY PROBES\n";

    for (int i = 0;
         i < RECOVERY_SAMPLES;
         ++i)
    {
        CUDA_CHECK(cudaEventRecord(
            recovery_start[i],
            recovery_stream));

        probe_victim<<<
            VICTIM_BLOCKS,
            THREADS,
            0,
            recovery_stream>>>(
                victim_input,
                victim_output,
                work_repetitions);

        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(
            recovery_stop[i],
            recovery_stream));
    }

    // --------------------------------------------------------
    // Only NOW wait for completion.
    // --------------------------------------------------------

    CUDA_CHECK(cudaEventSynchronize(
        recovery_stop.back()));

    std::cout
        << ">>> RECOVERY PROBES FINISHED\n";

    // --------------------------------------------------------
    // Measure actual interference duration
    // --------------------------------------------------------

    float interference_duration_measured_ms =
        0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &interference_duration_measured_ms,
        interference_start_event,
        interference_end_event));

    // --------------------------------------------------------
    // Open CSV
    // --------------------------------------------------------

    std::ofstream csv(output_file);

    if (!csv.is_open())
    {
        std::cerr
            << "ERROR: Could not open "
            << output_file << "\n";

        return EXIT_FAILURE;
    }

    csv
        << "sample,phase,victim_latency_us,"
           "time_since_interference_end_us\n";

    // --------------------------------------------------------
    // Write baseline
    // --------------------------------------------------------

    for (int i = 0;
         i < BASELINE_SAMPLES;
         ++i)
    {
        csv
            << i
            << ",baseline,"
            << std::setprecision(9)
            << baseline_latency_us[i]
            << ",-1\n";
    }

    // --------------------------------------------------------
    // Write recovery
    // --------------------------------------------------------

    std::vector<double> recovery_latency_us;
    std::vector<double> recovery_time_us;

    recovery_latency_us.reserve(
        RECOVERY_SAMPLES);

    recovery_time_us.reserve(
        RECOVERY_SAMPLES);

    for (int i = 0;
         i < RECOVERY_SAMPLES;
         ++i)
    {
        float elapsed_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &elapsed_ms,
            recovery_start[i],
            recovery_stop[i]));

        float since_end_ms = 0.0f;

        CUDA_CHECK(cudaEventElapsedTime(
            &since_end_ms,
            interference_end_event,
            recovery_start[i]));

        const double latency_us =
            static_cast<double>(elapsed_ms) *
            1000.0;

        const double since_end_us =
            static_cast<double>(since_end_ms) *
            1000.0;

        recovery_latency_us.push_back(
            latency_us);

        recovery_time_us.push_back(
            since_end_us);

        csv
            << BASELINE_SAMPLES + i
            << ",recovery,"
            << std::setprecision(9)
            << latency_us
            << ","
            << since_end_us
            << "\n";
    }

    csv.close();

    // --------------------------------------------------------
    // Summary
    // --------------------------------------------------------

    const double recovery_median_us =
        median(recovery_latency_us);

    std::cout
        << "\n============================================\n"
        << " High-Resolution Recovery Summary\n"
        << "============================================\n"
        << "Victim blocks:              "
        << VICTIM_BLOCKS << "\n"
        << "Victim threads/block:       "
        << THREADS << "\n"
        << "Work repetitions:           "
        << work_repetitions << "\n"
        << "Interference duration:      "
        << interference_duration_measured_ms
        << " ms\n"
        << "Baseline median:            "
        << baseline_median_us
        << " us\n"
        << "Recovery median:            "
        << recovery_median_us
        << " us\n"
        << "Recovery / baseline:        "
        << recovery_median_us /
               baseline_median_us
        << "x\n"
        << "First recovery latency:     "
        << recovery_latency_us.front()
        << " us\n"
        << "First recovery timestamp:   "
        << recovery_time_us.front()
        << " us\n"
        << "Last recovery timestamp:    "
        << recovery_time_us.back()
        << " us\n"
        << "Results saved to:            "
        << output_file
        << "\n"
        << "============================================\n";

    // --------------------------------------------------------
    // Cleanup
    // --------------------------------------------------------

    for (int i = 0;
         i < BASELINE_SAMPLES;
         ++i)
    {
        CUDA_CHECK(cudaEventDestroy(
            baseline_start[i]));

        CUDA_CHECK(cudaEventDestroy(
            baseline_stop[i]));
    }

    for (int i = 0;
         i < RECOVERY_SAMPLES;
         ++i)
    {
        CUDA_CHECK(cudaEventDestroy(
            recovery_start[i]));

        CUDA_CHECK(cudaEventDestroy(
            recovery_stop[i]));
    }

    CUDA_CHECK(cudaEventDestroy(
        warmup_start));

    CUDA_CHECK(cudaEventDestroy(
        warmup_stop));

    CUDA_CHECK(cudaEventDestroy(
        interference_start_event));

    CUDA_CHECK(cudaEventDestroy(
        interference_end_event));

    CUDA_CHECK(cudaStreamDestroy(
        victim_stream));

    CUDA_CHECK(cudaStreamDestroy(
        interferer_stream));

    CUDA_CHECK(cudaStreamDestroy(
        recovery_stream));

    CUDA_CHECK(cudaFree(
        victim_input));

    CUDA_CHECK(cudaFree(
        victim_output));

    CUDA_CHECK(cudaFree(
        interferer_buffer));

    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
