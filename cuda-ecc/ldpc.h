#pragma once

#include <cuda_runtime.h>
#include <stddef.h>

__global__ void generate_minimums_per_check(const float* current_message_belief, size_t message_size,
     int* tanner_bits_to_check, size_t tanner_btc_row_size, size_t matrix_rows, float* minimums);
__global__ void generate_check_per_bit(float* current_message_belief, size_t message_size, size_t matrix_rows, float* minimums);

__device__ __forceinline__ void update_minimums(float& min_value, float& second_min_value, float noisy_message);
__device__ __forceinline__ float decide_minimums(float& min_value, float& second_min_value, float current_message_belief);
__device__ __forceinline__ short decide_sign(const float& current_message_belief);
