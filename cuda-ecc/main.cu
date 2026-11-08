#include <math.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "ldpc.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do                                                                          \
    {                                                                           \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess)                                                 \
        {                                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err), __FILE__, __LINE__);               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// (7, 4) Hamming code on an AWGN channel, same setup as ldpc_prototype.py.
constexpr int DATA_SIZE = 7;
constexpr int MATRIX_ROWS = 3;
constexpr int BITS_PER_CHECK = 4;
constexpr float SNR_DB = 10.0f;
constexpr float RATE = 4.0f / 7.0f;
constexpr int MAX_ITERATIONS = 100;

const int PARITY_CHECK_MATRIX[MATRIX_ROWS][DATA_SIZE] = {
    {1, 1, 0, 1, 1, 0, 0},
    {1, 0, 1, 1, 0, 1, 0},
    {0, 1, 1, 1, 0, 0, 1},
};

// BPSK-modulate each bit (0 -> +1, 1 -> -1) and add N(0, sigma^2) noise, one thread per bit.
__global__ void generate_awgn(const int* message, size_t message_size, float sigma,
                              unsigned long long seed, float* noisy_message)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < message_size)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(seed, index, 0, &state);
        float signed_bit = message[index] == 1 ? -1.0f : 1.0f;
        noisy_message[index] = signed_bit + sigma * curand_normal(&state);
    }
}

static bool syndrome_is_zero(const std::vector<int>& bits)
{
    for (int r = 0; r < MATRIX_ROWS; r++)
    {
        int parity = 0;
        for (int c = 0; c < DATA_SIZE; c++)
        {
            parity ^= PARITY_CHECK_MATRIX[r][c] & bits[c];
        }
        if (parity != 0)
        {
            return false;
        }
    }
    return true;
}

static void print_bits(const char* label, const std::vector<int>& bits)
{
    printf("%s[", label);
    for (int i = 0; i < DATA_SIZE; i++)
    {
        printf(i ? " %d" : "%d", bits[i]);
    }
    printf("]\n");
}

int main()
{
    // Tanner graph, check side, flattened: row i holds the bit indices in check i.
    std::vector<int> tanner_bits_to_check;
    for (int r = 0; r < MATRIX_ROWS; r++)
    {
        for (int c = 0; c < DATA_SIZE; c++)
        {
            if (PARITY_CHECK_MATRIX[r][c] == 1)
            {
                tanner_bits_to_check.push_back(c);
            }
        }
    }

    // BPSK (0 -> +1, 1 -> -1) plus Gaussian noise with sigma^2 = 1 / (2 * R * Eb/N0).
    std::vector<int> message = {1, 0, 1, 0, 1, 0, 1};
    float snr = powf(10.0f, SNR_DB / 10.0f);
    float sigma = sqrtf(1.0f / (2.0f * RATE * snr));
    unsigned long long seed = (static_cast<unsigned long long>(std::random_device{}()) << 32) | std::random_device{}();

    int* d_message = nullptr;
    float* d_noisy_message = nullptr;
    float* d_current_message_belief = nullptr;
    float* d_minimums = nullptr;
    int* d_tanner_bits_to_check = nullptr;
    size_t message_bytes = DATA_SIZE * sizeof(float);
    size_t minimums_bytes = MATRIX_ROWS * DATA_SIZE * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_message, DATA_SIZE * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_noisy_message, message_bytes));
    CUDA_CHECK(cudaMalloc(&d_current_message_belief, message_bytes));
    CUDA_CHECK(cudaMalloc(&d_minimums, minimums_bytes));
    CUDA_CHECK(cudaMalloc(&d_tanner_bits_to_check, tanner_bits_to_check.size() * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_message, message.data(), DATA_SIZE * sizeof(int), cudaMemcpyHostToDevice));
    constexpr int AWGN_BLOCK_SIZE = 256;
    int awgn_blocks = (DATA_SIZE + AWGN_BLOCK_SIZE - 1) / AWGN_BLOCK_SIZE;
    generate_awgn<<<awgn_blocks, AWGN_BLOCK_SIZE>>>(d_message, DATA_SIZE, sigma, seed, d_noisy_message);
    CUDA_CHECK(cudaGetLastError());

    std::vector<float> noisy_message(DATA_SIZE);
    CUDA_CHECK(cudaMemcpy(noisy_message.data(), d_noisy_message, message_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(d_current_message_belief, d_noisy_message, message_bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_tanner_bits_to_check, tanner_bits_to_check.data(),
                          tanner_bits_to_check.size() * sizeof(int), cudaMemcpyHostToDevice));

    std::vector<float> current_message_belief(DATA_SIZE);
    std::vector<int> output_message(DATA_SIZE);
    int iterations = 0;
    while (iterations < MAX_ITERATIONS)
    {
        // Bits outside a check never get written by the kernel, so they must contribute 0.
        CUDA_CHECK(cudaMemset(d_minimums, 0, minimums_bytes));
        generate_minimums_per_check<<<1, MATRIX_ROWS>>>(d_current_message_belief, DATA_SIZE,
                                                         d_tanner_bits_to_check, BITS_PER_CHECK,
                                                         MATRIX_ROWS, d_minimums);
        CUDA_CHECK(cudaGetLastError());

        // belief = noisy_message + check-to-bit corrections, as in the prototype's llr().
        CUDA_CHECK(cudaMemcpy(d_current_message_belief, d_noisy_message, message_bytes, cudaMemcpyDeviceToDevice));
        generate_check_per_bit<<<1, DATA_SIZE>>>(d_current_message_belief, DATA_SIZE, MATRIX_ROWS, d_minimums);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(current_message_belief.data(), d_current_message_belief, message_bytes,
                              cudaMemcpyDeviceToHost));
        for (int i = 0; i < DATA_SIZE; i++)
        {
            output_message[i] = current_message_belief[i] < 0.0f ? 1 : 0;
        }
        iterations++;
        if (syndrome_is_zero(output_message))
        {
            break;
        }
    }

    printf("noisy message:  [");
    for (int i = 0; i < DATA_SIZE; i++)
    {
        printf(i ? " %.3f" : "%.3f", noisy_message[i]);
    }
    printf("]\n");
    printf("final belief:   [");
    for (int i = 0; i < DATA_SIZE; i++)
    {
        printf(i ? " %.3f" : "%.3f", current_message_belief[i]);
    }
    printf("]\n");
    print_bits("sent message:   ", message);
    print_bits("decoded message:", output_message);
    printf("iterations: %d\n", iterations);
    printf(output_message == message ? "Message decoded successfully\n" : "Message decoded incorrectly\n");

    CUDA_CHECK(cudaFree(d_message));
    CUDA_CHECK(cudaFree(d_noisy_message));
    CUDA_CHECK(cudaFree(d_current_message_belief));
    CUDA_CHECK(cudaFree(d_minimums));
    CUDA_CHECK(cudaFree(d_tanner_bits_to_check));
    return 0;
}
