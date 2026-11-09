#include "ldpc_dataset.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

// Single-threaded min-sum decoder. The update and stopping rules are the same as the CUDA path
// (ldpc.cu kernels plus the syndrome check in main.cu), and bit sums run in the same check order,
// so both decoders produce bit-identical output for the same dataset.
namespace
{

void update_minimums(float& min_value, float& second_min_value, float noisy_message)
{
    second_min_value = std::fmin(second_min_value, std::fmax(std::fabs(noisy_message), min_value));
    min_value = std::fmin(std::fabs(noisy_message), min_value);
}

float decide_minimums(float min_value, float second_min_value, float current_message_belief)
{
    return std::fabs(current_message_belief) == min_value ? second_min_value : min_value;
}

short decide_sign(float current_message_belief)
{
    return current_message_belief > 0 ? 1 : -1;
}

class CpuDecoder
{
  public:
    explicit CpuDecoder(const LdpcDataset& dataset)
        : message_size_(dataset.message_size), matrix_rows_(dataset.matrix_rows),
          bits_per_check_(dataset.bits_per_check), tanner_bits_to_check_(dataset.tanner_bits_to_check),
          check_to_bit_(tanner_bits_to_check_.size()), current_message_belief_(message_size_),
          bit_edge_offsets_(message_size_ + 1, 0), bit_edges_(tanner_bits_to_check_.size())
    {
        // Bit side of the Tanner graph: for each bit, its edges in increasing check order.
        for (int bit : tanner_bits_to_check_)
        {
            bit_edge_offsets_[bit + 1]++;
        }
        for (int bit = 0; bit < message_size_; bit++)
        {
            bit_edge_offsets_[bit + 1] += bit_edge_offsets_[bit];
        }
        std::vector<int> next(bit_edge_offsets_.begin(), bit_edge_offsets_.end() - 1);
        for (int edge = 0; edge < static_cast<int>(tanner_bits_to_check_.size()); edge++)
        {
            bit_edges_[next[tanner_bits_to_check_[edge]]++] = edge;
        }
    }

    // Returns the number of iterations run; decoded gets the hard decision (1 where the belief is negative).
    int decode(const float* noisy_message, uint8_t* decoded, int max_iterations)
    {
        std::copy(noisy_message, noisy_message + message_size_, current_message_belief_.begin());
        std::fill(check_to_bit_.begin(), check_to_bit_.end(), 0.0f);
        for (int iteration = 1; iteration <= max_iterations; iteration++)
        {
            for (int check = 0; check < matrix_rows_; check++)
            {
                const int* bits = &tanner_bits_to_check_[check * bits_per_check_];
                float* messages = &check_to_bit_[check * bits_per_check_];
                float min_value = std::numeric_limits<float>::infinity();
                float second_min_value = std::numeric_limits<float>::infinity();
                short sign = 1;
                // Each check sees the bit's belief minus its own message from the last iteration (no echo).
                for (int i = 0; i < bits_per_check_; i++)
                {
                    float belief_without_echo = current_message_belief_[bits[i]] - messages[i];
                    sign *= decide_sign(belief_without_echo);
                    update_minimums(min_value, second_min_value, belief_without_echo);
                }
                for (int i = 0; i < bits_per_check_; i++)
                {
                    float belief_without_echo = current_message_belief_[bits[i]] - messages[i];
                    short current_sign = decide_sign(belief_without_echo);
                    messages[i] =
                        0.75f * sign * current_sign * decide_minimums(min_value, second_min_value, belief_without_echo);
                }
            }

            for (int bit = 0; bit < message_size_; bit++)
            {
                float belief = noisy_message[bit];
                for (int e = bit_edge_offsets_[bit]; e < bit_edge_offsets_[bit + 1]; e++)
                {
                    belief += check_to_bit_[bit_edges_[e]];
                }
                current_message_belief_[bit] = belief;
                decoded[bit] = belief < 0.0f ? 1 : 0;
            }

            if (syndrome_is_zero(decoded))
            {
                return iteration;
            }
        }
        return max_iterations;
    }

  private:
    bool syndrome_is_zero(const uint8_t* decoded) const
    {
        for (int check = 0; check < matrix_rows_; check++)
        {
            int parity = 0;
            for (int i = 0; i < bits_per_check_; i++)
            {
                parity ^= decoded[tanner_bits_to_check_[check * bits_per_check_ + i]];
            }
            if (parity != 0)
            {
                return false;
            }
        }
        return true;
    }

    int message_size_;
    int matrix_rows_;
    int bits_per_check_;
    std::vector<int> tanner_bits_to_check_;
    std::vector<float> check_to_bit_; // one message per edge, laid out like tanner_bits_to_check_
    std::vector<float> current_message_belief_;
    std::vector<int> bit_edge_offsets_;
    std::vector<int> bit_edges_;
};

void usage()
{
    fprintf(stderr, "usage: ldpc_cpu [dataset.bin] [--max-iterations 50]\n");
    exit(EXIT_FAILURE);
}

} // namespace

int main(int argc, char** argv)
{
    std::string dataset_path = "dataset.bin";
    int max_iterations = 50;
    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--max-iterations") && i + 1 < argc)
            max_iterations = atoi(argv[++i]);
        else if (argv[i][0] != '-')
            dataset_path = argv[i];
        else
            usage();
    }
    if (max_iterations <= 0)
    {
        usage();
    }

    try
    {
        LdpcDataset dataset = load_dataset(dataset_path);
        size_t n = static_cast<size_t>(dataset.message_size);
        std::vector<uint8_t> decoded(static_cast<size_t>(dataset.frames) * n);
        std::vector<int> iterations(dataset.frames);
        CpuDecoder decoder(dataset);

        auto start = std::chrono::steady_clock::now();
        for (int f = 0; f < dataset.frames; f++)
        {
            iterations[f] = decoder.decode(&dataset.noisy_messages[f * n], &decoded[f * n], max_iterations);
        }
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        print_report("C++ on the CPU, single thread", dataset, max_iterations, decoded, iterations, seconds);
    }
    catch (const std::exception& e)
    {
        fprintf(stderr, "error: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return 0;
}
