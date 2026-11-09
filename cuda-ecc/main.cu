#include <cuda_runtime.h>

#include "ldpc.h"
#include "ldpc_dataset.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t err = (call);                                                                                      \
        if (err != cudaSuccess)                                                                                        \
        {                                                                                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__);                  \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

constexpr int THREADS_PER_BLOCK = 256;

static unsigned int blocks_for(int threads)
{
    return static_cast<unsigned int>((threads + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
}

// blockIdx.y is the frame; x spans that frame's checks and bits. Threads below matrix_rows flag a failed parity
// check; threads below message_size write the hard decision, which stops changing once the frame is done.
__global__ void check_syndrome(const float* current_message_belief, const int* tanner_bits_to_check, int bits_per_check,
                               int message_size, int matrix_rows, uint8_t* decoded, const int* done, int* unsatisfied)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int frame_index = blockIdx.y;
    current_message_belief += static_cast<size_t>(frame_index) * message_size;
    if (index < matrix_rows)
    {
        int parity = 0;
        for (int i = 0; i < bits_per_check; i++)
        {
            parity ^= current_message_belief[tanner_bits_to_check[index * bits_per_check + i]] < 0.0f;
        }
        if (parity)
        {
            unsatisfied[frame_index] = 1;
        }
    }
    if (index < message_size && !done[frame_index])
    {
        decoded[static_cast<size_t>(frame_index) * message_size + index] = current_message_belief[index] < 0.0f ? 1 : 0;
    }
}

// One thread per frame. Runs after check_syndrome, so every block has finished reading done and writing unsatisfied.
__global__ void finish_iteration(int frames, int iteration, int* done, int* unsatisfied, int* iterations_used)
{
    int frame_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (frame_index < frames)
    {
        if (!done[frame_index] && !unsatisfied[frame_index])
        {
            done[frame_index] = 1;
            iterations_used[frame_index] = iteration;
        }
        unsatisfied[frame_index] = 0;
    }
}

static void usage()
{
    fprintf(stderr, "usage: ldpc_cuda [dataset.bin] [--max-iterations 50] [--batch 128]\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
    std::string dataset_path = "dataset.bin";
    int max_iterations = 50;
    int batch = 128;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--max-iterations") && i + 1 < argc)
            max_iterations = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch") && i + 1 < argc)
            batch = atoi(argv[++i]);
        else if (argv[i][0] != '-')
            dataset_path = argv[i];
        else
            usage();
    }
    if (max_iterations <= 0 || batch <= 0)
    {
        usage();
    }

    LdpcDataset dataset;
    std::vector<int> tanner_checks_per_bit;
    int max_checks_per_bit = 0;
    try
    {
        dataset = load_dataset(dataset_path);
        tanner_checks_per_bit = make_tanner_checks_per_bit(dataset.tanner_bits_to_check, dataset.message_size,
                                                           dataset.bits_per_check, max_checks_per_bit);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }

    const int n = dataset.message_size;
    const int m = dataset.matrix_rows;
    const int bits_per_check = dataset.bits_per_check;
    const unsigned int check_blocks = blocks_for(m);
    const unsigned int bit_blocks = blocks_for(n);
    const unsigned int syndrome_blocks = blocks_for(std::max(n, m));

    // minimums is a dense m x n matrix per frame, so cap the batch at half of the free GPU memory,
    // and at 65535, the largest gridDim.y.
    const size_t frame_minimums = static_cast<size_t>(m) * n;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t frames_that_fit = free_bytes / 2 / (frame_minimums * sizeof(float));
    if (frames_that_fit == 0)
    {
        fprintf(stderr, "error: one frame's %d x %d minimums matrix does not fit in GPU memory\n", m, n);
        return EXIT_FAILURE;
    }
    batch = static_cast<int>(
        std::min<size_t>({static_cast<size_t>(batch), static_cast<size_t>(dataset.frames), frames_that_fit, 65535}));

    int* d_tanner_bits_to_check = nullptr;
    int* d_tanner_checks_per_bit = nullptr;
    float* d_noisy_message = nullptr;
    float* d_current_message_belief = nullptr;
    float* d_minimums = nullptr;
    uint8_t* d_decoded = nullptr;
    int* d_done = nullptr;
    int* d_unsatisfied = nullptr;
    int* d_iterations = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tanner_bits_to_check, dataset.tanner_bits_to_check.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_tanner_checks_per_bit, tanner_checks_per_bit.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_noisy_message, static_cast<size_t>(batch) * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_current_message_belief, static_cast<size_t>(batch) * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_minimums, batch * frame_minimums * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_decoded, static_cast<size_t>(batch) * n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_done, batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_unsatisfied, batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_iterations, batch * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tanner_bits_to_check, dataset.tanner_bits_to_check.data(),
                          dataset.tanner_bits_to_check.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tanner_checks_per_bit, tanner_checks_per_bit.data(),
                          tanner_checks_per_bit.size() * sizeof(int), cudaMemcpyHostToDevice));
    // generate_minimums_per_check only ever writes the (check, bit) entries that are edges of H,
    // so zeroing once leaves every other entry at 0 for all frames and iterations.
    CUDA_CHECK(cudaMemset(d_minimums, 0, batch * frame_minimums * sizeof(float)));

    std::vector<uint8_t> decoded(static_cast<size_t>(dataset.frames) * n);
    std::vector<int> iterations(dataset.frames);
    std::vector<int> done(batch);
    const std::vector<int> no_iterations_yet(batch, max_iterations);

    // Frames [first, first + count) are decoded together: each kernel launch covers all of them via blockIdx.y.
    // Frames that finish early keep being updated until the whole batch is done, but their output is frozen.
    auto decode_batch = [&](int first, int count) {
        CUDA_CHECK(cudaMemset(d_minimums, 0, batch * frame_minimums * sizeof(float)));
        size_t samples = static_cast<size_t>(count) * n;
        CUDA_CHECK(cudaMemcpy(d_noisy_message, &dataset.noisy_messages[static_cast<size_t>(first) * n],
                              samples * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(
            cudaMemcpy(d_current_message_belief, d_noisy_message, samples * sizeof(float), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_done, 0, count * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_unsatisfied, 0, count * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_iterations, no_iterations_yet.data(), count * sizeof(int), cudaMemcpyHostToDevice));

        const unsigned int frames = static_cast<unsigned int>(count);
        for (int iteration = 1; iteration <= max_iterations; iteration++)
        {
            generate_minimums_per_check<<<dim3(check_blocks, frames), THREADS_PER_BLOCK>>>(
                d_current_message_belief, n, d_tanner_bits_to_check, bits_per_check, m, d_minimums);
            generate_check_per_bit<<<dim3(bit_blocks, frames), THREADS_PER_BLOCK>>>(
                d_current_message_belief, d_noisy_message, n, d_tanner_checks_per_bit, max_checks_per_bit, m,
                d_minimums);
            check_syndrome<<<dim3(syndrome_blocks, frames), THREADS_PER_BLOCK>>>(
                d_current_message_belief, d_tanner_bits_to_check, bits_per_check, n, m, d_decoded, d_done,
                d_unsatisfied);
            finish_iteration<<<blocks_for(count), THREADS_PER_BLOCK>>>(count, iteration, d_done, d_unsatisfied,
                                                                       d_iterations);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(done.data(), d_done, count * sizeof(int), cudaMemcpyDeviceToHost));
            if (std::all_of(done.begin(), done.begin() + count, [](int d) { return d != 0; }))
            {
                break;
            }
        }

        CUDA_CHECK(cudaMemcpy(&decoded[static_cast<size_t>(first) * n], d_decoded, samples * sizeof(uint8_t),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&iterations[first], d_iterations, count * sizeof(int), cudaMemcpyDeviceToHost));
    };

    // Untimed warm-up so module loading and first-launch costs stay out of the measurement.
    decode_batch(0, batch);

    auto start = std::chrono::steady_clock::now();
    for (int first = 0; first < dataset.frames; first += batch)
    {
        decode_batch(first, std::min(batch, dataset.frames - first));
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    char decoder_name[128];
    snprintf(decoder_name, sizeof(decoder_name), "CUDA (ldpc.cu kernels), %d frames per launch", batch);
    print_report(decoder_name, dataset, max_iterations, decoded, iterations, seconds);

    CUDA_CHECK(cudaFree(d_tanner_bits_to_check));
    CUDA_CHECK(cudaFree(d_tanner_checks_per_bit));
    CUDA_CHECK(cudaFree(d_noisy_message));
    CUDA_CHECK(cudaFree(d_current_message_belief));
    CUDA_CHECK(cudaFree(d_minimums));
    CUDA_CHECK(cudaFree(d_decoded));
    CUDA_CHECK(cudaFree(d_done));
    CUDA_CHECK(cudaFree(d_unsatisfied));
    CUDA_CHECK(cudaFree(d_iterations));
    return 0;
}
