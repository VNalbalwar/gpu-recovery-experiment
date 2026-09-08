#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__    \
                      << " -> " << cudaGetErrorString(err) << "\n";         \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// ============================================================
// Experimental configuration
// ============================================================

constexpr int THREADS = 256;
constexpr int VICTIM_BLOCKS = 8;

constexpr int WARMUP_SAMPLES = 30;
constexpr int BASELINE_SAMPLES = 50;
constexpr int RECOVERY_SAMPLES = 200;

constexpr int TOTAL_TRIALS = 20;
constexpr int INTERFERENCE_TRIALS = 10;
constexpr int CONTROL_TRIALS = 10;

constexpr int VICTIM_ITERATIONS = 4;
constexpr int DEFAULT_WORK_REPETITIONS = 64;

constexpr double INTERFERENCE_MS = 100.0;
constexpr int INTERFERER_BLOCKS = 24;

// Conservative between-trial isolation interval.
// This is an experimental protocol parameter, not a claimed
// hardware recovery time.
constexpr int COOLDOWN_MS = 5000;

constexpr size_t L2_INTERFERER_ELEMENTS =
    16ULL * 1024ULL * 1024ULL / sizeof(float);

// ============================================================
// Victim
// ============================================================

__global__ void probe_victim(
    const float *__restrict__ input,
    float *__restrict__ output,
    int work_repetitions)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    const int total_threads = VICTIM_BLOCKS * THREADS;

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
// ============================================================

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
        size_t index = (tid * 32ULL) % elements;

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

// ============================================================
// One victim probe
// ============================================================

double run_probe(
    cudaStream_t stream,
    cudaEvent_t start,
    cudaEvent_t stop,
    float *input,
    float *output,
    int work_repetitions)
{
    CUDA_CHECK(cudaEventRecord(start, stream));

    probe_victim<<<
        VICTIM_BLOCKS,
        THREADS,
        0,
        stream>>>(
        input,
        output,
        work_repetitions);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, stream));

    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms,
        start,
        stop));

    return static_cast<double>(elapsed_ms) * 1000.0;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char **argv)
{
    std::string output_file = "results/hr_l2_extended.csv";

    int work_repetitions = DEFAULT_WORK_REPETITIONS;

    if (argc >= 2)
        output_file = argv[1];

    if (argc >= 3)
        work_repetitions = std::stoi(argv[2]);

    if (work_repetitions <= 0 || work_repetitions > 10000)
    {
        std::cerr
            << "ERROR: work repetitions must be between 1 and 10000.\n";
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaSetDevice(0));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    constexpr int total_threads = VICTIM_BLOCKS * THREADS;

    std::cout
        << "\n============================================\n"
        << " Extended High-Resolution L2 Recovery Experiment\n"
        << "============================================\n\n"
        << "GPU:                 " << prop.name << "\n"
        << "SM count:            " << prop.multiProcessorCount << "\n"
        << "Victim blocks:       " << VICTIM_BLOCKS << "\n"
        << "Threads/block:       " << THREADS << "\n"
        << "Work repetitions:    " << work_repetitions << "\n"
        << "Interferer blocks:   " << INTERFERER_BLOCKS << "\n"
        << "Interference:        " << INTERFERENCE_MS << " ms\n"
        << "Baseline probes:     " << BASELINE_SAMPLES << "\n"
        << "Recovery probes:     " << RECOVERY_SAMPLES << "\n"
        << "Total trials:        " << TOTAL_TRIALS << "\n"
        << "Cooldown:            " << COOLDOWN_MS << " ms\n"
        << "Output:              " << output_file << "\n\n";

    const size_t victim_bytes =
        static_cast<size_t>(total_threads) * sizeof(float);

    float *victim_input = nullptr;
    float *victim_output = nullptr;
    float *interferer_buffer = nullptr;

    CUDA_CHECK(cudaMalloc(&victim_input, victim_bytes));
    CUDA_CHECK(cudaMalloc(&victim_output, victim_bytes));
    CUDA_CHECK(cudaMalloc(
        &interferer_buffer,
        L2_INTERFERER_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaMemset(victim_input, 1, victim_bytes));
    CUDA_CHECK(cudaMemset(victim_output, 0, victim_bytes));
    CUDA_CHECK(cudaMemset(
        interferer_buffer,
        1,
        L2_INTERFERER_ELEMENTS * sizeof(float)));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaStream_t victim_stream;
    cudaStream_t interferer_stream;
    cudaStream_t recovery_stream;

    CUDA_CHECK(cudaStreamCreateWithFlags(
        &victim_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &interferer_stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(
        &recovery_stream, cudaStreamNonBlocking));

    cudaEvent_t probe_start;
    cudaEvent_t probe_stop;
    cudaEvent_t interference_start;
    cudaEvent_t interference_end;

    CUDA_CHECK(cudaEventCreate(&probe_start));
    CUDA_CHECK(cudaEventCreate(&probe_stop));
    CUDA_CHECK(cudaEventCreate(&interference_start));
    CUDA_CHECK(cudaEventCreate(&interference_end));

    std::cout << "Warming up GPU...\n";

    for (int i = 0; i < WARMUP_SAMPLES; ++i)
    {
        run_probe(
            victim_stream,
            probe_start,
            probe_stop,
            victim_input,
            victim_output,
            work_repetitions);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> conditions;
    conditions.reserve(TOTAL_TRIALS);

    for (int i = 0; i < INTERFERENCE_TRIALS; ++i)
        conditions.push_back(1);

    for (int i = 0; i < CONTROL_TRIALS; ++i)
        conditions.push_back(0);

    std::mt19937 rng(20260909);
    std::shuffle(conditions.begin(), conditions.end(), rng);

    std::ofstream csv(output_file);

    if (!csv)
    {
        std::cerr
            << "ERROR: could not open output file: "
            << output_file << "\n";
        return EXIT_FAILURE;
    }

    csv
        << "trial_id,"
        << "condition,"
        << "probe_id,"
        << "phase,"
        << "victim_latency_us,"
        << "baseline_median_us,"
        << "normalized_latency,"
        << "time_since_interference_end_us\n";

    for (int trial = 0; trial < TOTAL_TRIALS; ++trial)
    {
        const bool interference = conditions[trial] == 1;
        const char *condition_name =
            interference ? "interference" : "control";

        std::cout
            << "\n============================================\n"
            << " Trial " << (trial + 1) << "/" << TOTAL_TRIALS
            << " : " << condition_name << "\n"
            << "============================================\n";

        std::vector<double> baseline;
        baseline.reserve(BASELINE_SAMPLES);

        for (int i = 0; i < BASELINE_SAMPLES; ++i)
        {
            baseline.push_back(run_probe(
                victim_stream,
                probe_start,
                probe_stop,
                victim_input,
                victim_output,
                work_repetitions));
        }

        const double baseline_median = median(baseline);

        std::cout
            << "Baseline median: "
            << std::fixed << std::setprecision(3)
            << baseline_median << " us\n";

        const unsigned long long duration_ns =
            static_cast<unsigned long long>(
                INTERFERENCE_MS * 1'000'000.0);

        if (interference)
        {
            std::cout << "Starting 100 ms L2 interference...\n";

            CUDA_CHECK(cudaEventRecord(
                interference_start,
                interferer_stream));

            l2_interferer<<<
                INTERFERER_BLOCKS,
                THREADS,
                0,
                interferer_stream>>>(
                interferer_buffer,
                L2_INTERFERER_ELEMENTS,
                duration_ns);

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(
                interference_end,
                interferer_stream));
        }
        else
        {
            std::cout << "Running control condition...\n";

            CUDA_CHECK(cudaEventRecord(
                interference_start,
                interferer_stream));

            CUDA_CHECK(cudaEventRecord(
                interference_end,
                interferer_stream));
        }

        // Critical GPU-side dependency: no host synchronization before
        // recovery probes. This preserves the timing relationship between
        // interference completion and recovery sampling.
        CUDA_CHECK(cudaStreamWaitEvent(
            recovery_stream,
            interference_end,
            0));

        std::cout
            << "Running " << RECOVERY_SAMPLES
            << " recovery probes...\n";

        std::vector<cudaEvent_t> recovery_start(RECOVERY_SAMPLES);
        std::vector<cudaEvent_t> recovery_stop(RECOVERY_SAMPLES);

        for (int i = 0; i < RECOVERY_SAMPLES; ++i)
        {
            CUDA_CHECK(cudaEventCreate(&recovery_start[i]));
            CUDA_CHECK(cudaEventCreate(&recovery_stop[i]));
        }

        for (int i = 0; i < RECOVERY_SAMPLES; ++i)
        {
            CUDA_CHECK(cudaEventRecord(
                recovery_start[i], recovery_stream));

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
                recovery_stop[i], recovery_stream));
        }

        CUDA_CHECK(cudaEventSynchronize(recovery_stop.back()));

        std::vector<double> recovery_latencies;
        recovery_latencies.reserve(RECOVERY_SAMPLES);

        for (int i = 0; i < RECOVERY_SAMPLES; ++i)
        {
            float elapsed_ms = 0.0f;
            float since_end_ms = 0.0f;

            CUDA_CHECK(cudaEventElapsedTime(
                &elapsed_ms,
                recovery_start[i],
                recovery_stop[i]));

            CUDA_CHECK(cudaEventElapsedTime(
                &since_end_ms,
                interference_end,
                recovery_start[i]));

            const double latency_us =
                static_cast<double>(elapsed_ms) * 1000.0;

            const double since_end_us =
                static_cast<double>(since_end_ms) * 1000.0;

            const double normalized =
                latency_us / baseline_median;

            recovery_latencies.push_back(latency_us);

            csv
                << trial << ","
                << condition_name << ","
                << i << ",recovery,"
                << std::setprecision(9)
                << latency_us << ","
                << baseline_median << ","
                << normalized << ","
                << since_end_us << "\n";
        }

        csv.flush();

        const double recovery_median = median(recovery_latencies);

        std::cout
            << "Recovery median:  "
            << recovery_median << " us\n"
            << "Recovery ratio:   "
            << recovery_median / baseline_median << "x\n"
            << "Observation end:  "
            << "~";

        if (!recovery_latencies.empty())
        {
            float last_since_end_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(
                &last_since_end_ms,
                interference_end,
                recovery_start.back()));
            std::cout << static_cast<double>(last_since_end_ms) * 1000.0;
        }

        std::cout << " us after interference end\n";

        for (int i = 0; i < RECOVERY_SAMPLES; ++i)
        {
            CUDA_CHECK(cudaEventDestroy(recovery_start[i]));
            CUDA_CHECK(cudaEventDestroy(recovery_stop[i]));
        }

        if (trial + 1 < TOTAL_TRIALS)
        {
            std::cout
                << "Cooldown: " << COOLDOWN_MS << " ms...\n";

            std::this_thread::sleep_for(
                std::chrono::milliseconds(COOLDOWN_MS));
        }
    }

    CUDA_CHECK(cudaEventDestroy(probe_start));
    CUDA_CHECK(cudaEventDestroy(probe_stop));
    CUDA_CHECK(cudaEventDestroy(interference_start));
    CUDA_CHECK(cudaEventDestroy(interference_end));

    CUDA_CHECK(cudaStreamDestroy(victim_stream));
    CUDA_CHECK(cudaStreamDestroy(interferer_stream));
    CUDA_CHECK(cudaStreamDestroy(recovery_stream));

    CUDA_CHECK(cudaFree(victim_input));
    CUDA_CHECK(cudaFree(victim_output));
    CUDA_CHECK(cudaFree(interferer_buffer));

    csv.close();

    std::cout
        << "\n============================================\n"
        << " Experiment complete\n"
        << "============================================\n"
        << "Results written to: " << output_file << "\n";

    return EXIT_SUCCESS;
}
