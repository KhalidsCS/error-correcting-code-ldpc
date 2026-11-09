#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "ldpc.h"

__global__ void generate_minimums_per_check(const float* current_message_belief, size_t message_size,
                                            int* tanner_bits_to_check, size_t tanner_btc_row_size, size_t matrix_rows,
                                            float* minimums)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int frame_index = blockIdx.y;
    if (index < matrix_rows)
    {
        float min_value = INFINITY;
        float second_min_value = INFINITY;
        short sign = 1;
        for (int i = 0; i < tanner_btc_row_size; i++)
        {
            int bit_to_check = tanner_bits_to_check[i + tanner_btc_row_size * index];
            float belief_without_echo =
                current_message_belief[bit_to_check + frame_index * message_size] -
                minimums[bit_to_check + message_size * index + frame_index * message_size * matrix_rows];
            short current_sign = decide_sign(belief_without_echo);

            update_minimums(min_value, second_min_value, belief_without_echo);
            sign *= current_sign;
        }
        for (int i = 0; i < tanner_btc_row_size; i++)
        {
            int bit_to_check = tanner_bits_to_check[i + tanner_btc_row_size * index];
            float belief_without_echo =
                current_message_belief[bit_to_check + frame_index * message_size] -
                minimums[bit_to_check + message_size * index + frame_index * message_size * matrix_rows];
            short current_sign = decide_sign(belief_without_echo);
            minimums[bit_to_check + message_size * index + frame_index * message_size * matrix_rows] =
                0.75f * sign * current_sign * decide_minimums(min_value, second_min_value, belief_without_echo);
        }
    }
}
__global__ void generate_check_per_bit(float* current_message_belief, const float* noisy_message,
                                       const size_t message_size, const int* tanner_checks_per_bit,
                                       const size_t max_checks_per_bit, const size_t matrix_rows, const float* minimums)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int frame_index = blockIdx.y;
    if (index < message_size)
    {
        current_message_belief[index + frame_index * message_size] = noisy_message[index + frame_index * message_size];
        for (size_t i = 0; i < max_checks_per_bit; i++)
        {
            int check_index = tanner_checks_per_bit[index * max_checks_per_bit + i];
            current_message_belief[index + frame_index * message_size] +=
                minimums[message_size * check_index + index + frame_index * message_size * matrix_rows];
        }
    }
}
__device__ __forceinline__ void update_minimums(float& min_value, float& second_min_value, float noisy_message)
{
    second_min_value = fminf(second_min_value, fmaxf(fabs(noisy_message), min_value));
    min_value = fminf(fabs(noisy_message), min_value);
}
__device__ __forceinline__ float decide_minimums(float& min_value, float& second_min_value,
                                                 float current_message_belief)
{
    if (fabs(current_message_belief) == min_value)
    {
        return second_min_value;
    }
    else
    {
        return min_value;
    }
}
__device__ __forceinline__ short decide_sign(const float& current_message_belief)
{
    short sign;
    (current_message_belief > 0) ? sign = 1 : sign = -1;
    return sign;
}
