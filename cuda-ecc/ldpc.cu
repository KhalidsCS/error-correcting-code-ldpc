#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "ldpc.h"

__global__ void generate_minimums_per_check(const float* current_message_belief, size_t message_size,
                                            int* tanner_bits_to_check, size_t tanner_btc_row_size, size_t matrix_rows,
                                            float* minimums)
{
    int index = threadIdx.x;
    if (index < matrix_rows)
    {
        float min_value = INFINITY;
        float second_min_value = INFINITY;
        short sign = 1;
        for (int i = 0; i < tanner_btc_row_size; i++)
        {
            int bit_to_check = tanner_bits_to_check[i + index * tanner_btc_row_size];
            float noisy_message = current_message_belief[bit_to_check];
            sign *= decide_sign(noisy_message);
            update_minimums(min_value, second_min_value, noisy_message);
        }
        for (int i = 0; i < tanner_btc_row_size; i++)
        {

            int bit_to_check = tanner_bits_to_check[i + tanner_btc_row_size * index];
            short current_sign = decide_sign(current_message_belief[bit_to_check]);
            minimums[bit_to_check + message_size * index] =
                0.75f * sign * current_sign *
                decide_minimums(min_value, second_min_value, current_message_belief[bit_to_check]);
        }
    }
}
__global__ void generate_check_per_bit(float* current_message_belief, size_t message_size, size_t matrix_rows,
                                       float* minimums)
{
    int index = threadIdx.x;
    if (index < message_size)
    {

        for (int i = 0; i < matrix_rows; i++)
        {
            current_message_belief[index] += minimums[message_size * i + index];
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