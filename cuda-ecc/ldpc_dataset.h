#pragma once

#include <cstdint>
#include <string>
#include <vector>

// A regular LDPC code plus a batch of noisy BPSK frames, shared by the CPU and CUDA decoders.
struct LdpcDataset
{
    int message_size = 0;     // n, bits per codeword
    int matrix_rows = 0;      // m, parity checks
    int bits_per_check = 0;   // row weight of H
    int checks_per_bit = 0;   // column weight of H
    int information_bits = 0; // k = n - rank(H)
    int frames = 0;
    float snr_db = 0.0f;
    std::vector<int> tanner_bits_to_check; // matrix_rows rows of bits_per_check bit indices
    std::vector<uint8_t> codewords;        // frames * message_size sent bits
    std::vector<float> noisy_messages;     // frames * message_size received values
};

// Gallager construction: checks_per_bit bands, each a random column permutation of a band whose
// row i covers bits [i * bits_per_check, (i + 1) * bits_per_check).
std::vector<int> make_regular_tanner(int message_size, int checks_per_bit, int bits_per_check, uint64_t seed);

// Bit side of the Tanner graph: row b lists, in increasing order, the checks that contain bit b.
// Every row has max_checks_per_bit entries, so every bit must be in the same number of checks.
std::vector<int> make_tanner_checks_per_bit(const std::vector<int>& tanner_bits_to_check, int message_size,
                                            int bits_per_check, int& max_checks_per_bit);

// Random codewords of the code whose parity checks are the Tanner rows. Sets information_bits.
std::vector<uint8_t> encode_random_codewords(const std::vector<int>& tanner_bits_to_check, int message_size,
                                             int bits_per_check, int frames, uint64_t seed, int& information_bits);

void save_dataset(const std::string& path, const LdpcDataset& dataset);
LdpcDataset load_dataset(const std::string& path);

void print_report(const char* decoder, const LdpcDataset& dataset, int max_iterations,
                  const std::vector<uint8_t>& decoded, const std::vector<int>& iterations, double seconds);
