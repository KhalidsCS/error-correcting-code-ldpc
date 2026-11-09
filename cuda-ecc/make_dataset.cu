#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ldpc_dataset.h"

#include <chrono>
#include <cmath>
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

// BPSK-modulate each bit (0 -> +1, 1 -> -1) and add N(0, sigma^2) noise, one thread per bit.
__global__ void generate_awgn(const uint8_t* message, size_t message_size, float sigma, unsigned long long seed,
                              float* noisy_message)
{
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < message_size)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, index, 0, &state);
        float signed_bit = message[index] == 1 ? -1.0f : 1.0f;
        noisy_message[index] = signed_bit + sigma * curand_normal(&state);
    }
}

static void usage()
{
    fprintf(stderr, "usage: make_dataset [--out dataset.bin] [--frames 10000] [--n 1020] [--checks-per-bit 3]\n"
                    "                    [--bits-per-check 6] [--snr 2.0] [--seed 1]\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char** argv)
{
    std::string out_path = "dataset.bin";
    int frames = 10000;
    int message_size = 1020;
    int checks_per_bit = 3;
    int bits_per_check = 6;
    float snr_db = 2.0f;
    unsigned long long seed = 1;
    for (int i = 1; i < argc; i++)
    {
        if (i + 1 >= argc)
        {
            usage();
        }
        const char* value = argv[++i];
        if (!strcmp(argv[i - 1], "--out"))
            out_path = value;
        else if (!strcmp(argv[i - 1], "--frames"))
            frames = atoi(value);
        else if (!strcmp(argv[i - 1], "--n"))
            message_size = atoi(value);
        else if (!strcmp(argv[i - 1], "--checks-per-bit"))
            checks_per_bit = atoi(value);
        else if (!strcmp(argv[i - 1], "--bits-per-check"))
            bits_per_check = atoi(value);
        else if (!strcmp(argv[i - 1], "--snr"))
            snr_db = static_cast<float>(atof(value));
        else if (!strcmp(argv[i - 1], "--seed"))
            seed = strtoull(value, nullptr, 10);
        else
            usage();
    }
    if (frames <= 0)
    {
        usage();
    }

    try
    {
        LdpcDataset dataset;
        dataset.message_size = message_size;
        dataset.bits_per_check = bits_per_check;
        dataset.checks_per_bit = checks_per_bit;
        dataset.frames = frames;
        dataset.snr_db = snr_db;
        dataset.tanner_bits_to_check = make_regular_tanner(message_size, checks_per_bit, bits_per_check, seed);
        dataset.matrix_rows = static_cast<int>(dataset.tanner_bits_to_check.size()) / bits_per_check;
        dataset.codewords = encode_random_codewords(dataset.tanner_bits_to_check, message_size, bits_per_check, frames,
                                                    seed + 1, dataset.information_bits);

        // Noise variance for unit-energy BPSK: sigma^2 = 1 / (2 * R * Eb/N0).
        float rate = static_cast<float>(dataset.information_bits) / message_size;
        float sigma = sqrtf(1.0f / (2.0f * rate * powf(10.0f, snr_db / 10.0f)));

        size_t samples = dataset.codewords.size();
        uint8_t* d_codewords = nullptr;
        float* d_noisy_messages = nullptr;
        CUDA_CHECK(cudaMalloc(&d_codewords, samples * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_noisy_messages, samples * sizeof(float)));
        CUDA_CHECK(
            cudaMemcpy(d_codewords, dataset.codewords.data(), samples * sizeof(uint8_t), cudaMemcpyHostToDevice));

        auto start = std::chrono::steady_clock::now();
        constexpr int AWGN_BLOCK_SIZE = 256;
        unsigned int awgn_blocks = static_cast<unsigned int>((samples + AWGN_BLOCK_SIZE - 1) / AWGN_BLOCK_SIZE);
        generate_awgn<<<awgn_blocks, AWGN_BLOCK_SIZE>>>(d_codewords, samples, sigma, seed + 2, d_noisy_messages);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        double awgn_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        dataset.noisy_messages.resize(samples);
        CUDA_CHECK(cudaMemcpy(dataset.noisy_messages.data(), d_noisy_messages, samples * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_codewords));
        CUDA_CHECK(cudaFree(d_noisy_messages));

        save_dataset(out_path, dataset);
        printf("wrote %s: %d frames, n=%d k=%d m=%d, (%d,%d)-regular, Eb/N0 %.2f dB, sigma %.4f\n", out_path.c_str(),
               frames, message_size, dataset.information_bits, dataset.matrix_rows, checks_per_bit, bits_per_check,
               snr_db, sigma);
        printf("AWGN for %zu samples generated on the GPU in %.3f ms\n", samples, awgn_ms);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return 0;
}
