#include "ldpc_dataset.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace
{

constexpr char MAGIC[4] = {'L', 'D', 'P', 'C'};
constexpr int32_t VERSION = 1;

int parity64(uint64_t x)
{
    x ^= x >> 32;
    x ^= x >> 16;
    x ^= x >> 8;
    x ^= x >> 4;
    x ^= x >> 2;
    x ^= x >> 1;
    return static_cast<int>(x & 1);
}

template <typename T> void write_value(std::ofstream& out, const T& value)
{
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T> void write_array(std::ofstream& out, const std::vector<T>& values)
{
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
}

template <typename T> void read_value(std::ifstream& in, T& value)
{
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T> void read_array(std::ifstream& in, std::vector<T>& values, size_t count)
{
    values.resize(count);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
}

} // namespace

std::vector<int> make_regular_tanner(int message_size, int checks_per_bit, int bits_per_check, uint64_t seed)
{
    if (message_size <= 0 || checks_per_bit <= 0 || bits_per_check <= 0 || message_size % bits_per_check != 0)
    {
        throw std::runtime_error("message size must be a positive multiple of bits per check");
    }
    int band_rows = message_size / bits_per_check;
    std::mt19937_64 rng(seed);
    std::vector<int> columns(message_size);
    std::iota(columns.begin(), columns.end(), 0);

    std::vector<int> tanner_bits_to_check;
    tanner_bits_to_check.reserve(static_cast<size_t>(band_rows) * checks_per_bit * bits_per_check);
    for (int band = 0; band < checks_per_bit; band++)
    {
        if (band > 0)
        {
            std::shuffle(columns.begin(), columns.end(), rng);
        }
        for (int row = 0; row < band_rows; row++)
        {
            std::vector<int> bits(columns.begin() + row * bits_per_check, columns.begin() + (row + 1) * bits_per_check);
            std::sort(bits.begin(), bits.end());
            tanner_bits_to_check.insert(tanner_bits_to_check.end(), bits.begin(), bits.end());
        }
    }
    return tanner_bits_to_check;
}

std::vector<int> make_tanner_checks_per_bit(const std::vector<int>& tanner_bits_to_check, int message_size,
                                            int bits_per_check, int& max_checks_per_bit)
{
    int matrix_rows = static_cast<int>(tanner_bits_to_check.size()) / bits_per_check;
    std::vector<int> degree(message_size, 0);
    for (int bit : tanner_bits_to_check)
    {
        degree[bit]++;
    }
    max_checks_per_bit = *std::max_element(degree.begin(), degree.end());
    if (*std::min_element(degree.begin(), degree.end()) != max_checks_per_bit)
    {
        throw std::runtime_error("bits are in different numbers of checks; tanner_checks_per_bit has no padding value");
    }

    std::vector<int> tanner_checks_per_bit(static_cast<size_t>(message_size) * max_checks_per_bit);
    std::vector<int> filled(message_size, 0);
    for (int check = 0; check < matrix_rows; check++)
    {
        for (int i = 0; i < bits_per_check; i++)
        {
            int bit = tanner_bits_to_check[check * bits_per_check + i];
            tanner_checks_per_bit[static_cast<size_t>(bit) * max_checks_per_bit + filled[bit]++] = check;
        }
    }
    return tanner_checks_per_bit;
}

std::vector<uint8_t> encode_random_codewords(const std::vector<int>& tanner_bits_to_check, int message_size,
                                             int bits_per_check, int frames, uint64_t seed, int& information_bits)
{
    int matrix_rows = static_cast<int>(tanner_bits_to_check.size()) / bits_per_check;
    int words = (message_size + 63) / 64;

    // H as packed bit rows, reduced to row echelon form over GF(2).
    std::vector<uint64_t> h(static_cast<size_t>(matrix_rows) * words, 0);
    for (int r = 0; r < matrix_rows; r++)
    {
        for (int i = 0; i < bits_per_check; i++)
        {
            int bit = tanner_bits_to_check[r * bits_per_check + i];
            h[static_cast<size_t>(r) * words + bit / 64] ^= 1ull << (bit % 64);
        }
    }
    std::vector<int> pivots;
    int rank = 0;
    for (int col = 0; col < message_size && rank < matrix_rows; col++)
    {
        int word = col / 64;
        uint64_t mask = 1ull << (col % 64);
        int pivot = -1;
        for (int r = rank; r < matrix_rows; r++)
        {
            if (h[static_cast<size_t>(r) * words + word] & mask)
            {
                pivot = r;
                break;
            }
        }
        if (pivot < 0)
        {
            continue;
        }
        if (pivot != rank)
        {
            std::swap_ranges(h.begin() + static_cast<size_t>(pivot) * words,
                             h.begin() + static_cast<size_t>(pivot + 1) * words,
                             h.begin() + static_cast<size_t>(rank) * words);
        }
        for (int r = 0; r < matrix_rows; r++)
        {
            if (r != rank && (h[static_cast<size_t>(r) * words + word] & mask))
            {
                for (int w = 0; w < words; w++)
                {
                    h[static_cast<size_t>(r) * words + w] ^= h[static_cast<size_t>(rank) * words + w];
                }
            }
        }
        pivots.push_back(col);
        rank++;
    }
    information_bits = message_size - rank;

    // Non-pivot bits are free: draw them at random, then each reduced row fixes its pivot bit.
    std::vector<uint8_t> is_pivot(message_size, 0);
    for (int p : pivots)
    {
        is_pivot[p] = 1;
    }
    std::mt19937_64 rng(seed);
    std::vector<uint8_t> codewords(static_cast<size_t>(frames) * message_size, 0);
    std::vector<uint64_t> packed(words);
    for (int f = 0; f < frames; f++)
    {
        uint8_t* codeword = &codewords[static_cast<size_t>(f) * message_size];
        std::fill(packed.begin(), packed.end(), 0);
        uint64_t random_bits = 0;
        int bits_left = 0;
        for (int bit = 0; bit < message_size; bit++)
        {
            if (is_pivot[bit])
            {
                continue;
            }
            if (bits_left == 0)
            {
                random_bits = rng();
                bits_left = 64;
            }
            codeword[bit] = static_cast<uint8_t>(random_bits & 1);
            random_bits >>= 1;
            bits_left--;
            packed[bit / 64] |= static_cast<uint64_t>(codeword[bit]) << (bit % 64);
        }
        for (int r = 0; r < rank; r++)
        {
            uint64_t acc = 0;
            for (int w = 0; w < words; w++)
            {
                acc ^= h[static_cast<size_t>(r) * words + w] & packed[w];
            }
            codeword[pivots[r]] = static_cast<uint8_t>(parity64(acc));
        }

        for (int r = 0; r < matrix_rows; r++)
        {
            int parity = 0;
            for (int i = 0; i < bits_per_check; i++)
            {
                parity ^= codeword[tanner_bits_to_check[r * bits_per_check + i]];
            }
            if (parity != 0)
            {
                throw std::runtime_error("encoder produced a word that fails a parity check");
            }
        }
    }
    return codewords;
}

void save_dataset(const std::string& path, const LdpcDataset& dataset)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("cannot open " + path + " for writing");
    }
    out.write(MAGIC, sizeof(MAGIC));
    write_value(out, VERSION);
    write_value(out, static_cast<int32_t>(dataset.message_size));
    write_value(out, static_cast<int32_t>(dataset.matrix_rows));
    write_value(out, static_cast<int32_t>(dataset.bits_per_check));
    write_value(out, static_cast<int32_t>(dataset.checks_per_bit));
    write_value(out, static_cast<int32_t>(dataset.information_bits));
    write_value(out, static_cast<int32_t>(dataset.frames));
    write_value(out, dataset.snr_db);
    write_array(out, dataset.tanner_bits_to_check);
    write_array(out, dataset.codewords);
    write_array(out, dataset.noisy_messages);
    if (!out)
    {
        throw std::runtime_error("failed while writing " + path);
    }
}

LdpcDataset load_dataset(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("cannot open " + path + " (run make_dataset first)");
    }
    char magic[4] = {};
    int32_t version = 0;
    in.read(magic, sizeof(magic));
    read_value(in, version);
    if (!in || !std::equal(magic, magic + 4, MAGIC) || version != VERSION)
    {
        throw std::runtime_error(path + " is not an LDPC dataset (version " + std::to_string(VERSION) + ")");
    }

    int32_t fields[6] = {};
    for (int32_t& field : fields)
    {
        read_value(in, field);
    }
    LdpcDataset dataset;
    dataset.message_size = fields[0];
    dataset.matrix_rows = fields[1];
    dataset.bits_per_check = fields[2];
    dataset.checks_per_bit = fields[3];
    dataset.information_bits = fields[4];
    dataset.frames = fields[5];
    read_value(in, dataset.snr_db);

    size_t samples = static_cast<size_t>(dataset.frames) * dataset.message_size;
    read_array(in, dataset.tanner_bits_to_check, static_cast<size_t>(dataset.matrix_rows) * dataset.bits_per_check);
    read_array(in, dataset.codewords, samples);
    read_array(in, dataset.noisy_messages, samples);
    if (!in)
    {
        throw std::runtime_error(path + " is truncated");
    }
    return dataset;
}

void print_report(const char* decoder, const LdpcDataset& dataset, int max_iterations,
                  const std::vector<uint8_t>& decoded, const std::vector<int>& iterations, double seconds)
{
    size_t n = static_cast<size_t>(dataset.message_size);
    long long channel_bit_errors = 0;
    long long bit_errors = 0;
    long long total_iterations = 0;
    int frame_errors = 0;
    // FNV-1a over every decoded bit and iteration count, so two decoders can be checked for identical output.
    uint64_t checksum = 1469598103934665603ull;
    constexpr uint64_t FNV_PRIME = 1099511628211ull;
    for (int f = 0; f < dataset.frames; f++)
    {
        int errors = 0;
        for (size_t i = 0; i < n; i++)
        {
            size_t at = static_cast<size_t>(f) * n + i;
            channel_bit_errors += (dataset.noisy_messages[at] < 0.0f) != (dataset.codewords[at] == 1);
            errors += decoded[at] != dataset.codewords[at];
            checksum = (checksum ^ decoded[at]) * FNV_PRIME;
        }
        bit_errors += errors;
        frame_errors += errors > 0;
        total_iterations += iterations[f];
        checksum = (checksum ^ static_cast<uint64_t>(iterations[f])) * FNV_PRIME;
    }

    double coded_bits = static_cast<double>(dataset.frames) * n;
    printf("decoder:         %s\n", decoder);
    printf("code:            n=%d k=%d m=%d, (%d,%d)-regular, Eb/N0 %.2f dB\n", dataset.message_size,
           dataset.information_bits, dataset.matrix_rows, dataset.checks_per_bit, dataset.bits_per_check,
           dataset.snr_db);
    printf("frames:          %d, max iterations %d\n", dataset.frames, max_iterations);
    printf("decode time:     %.3f ms (%.2f us/frame, %.2f Mbit/s)\n", seconds * 1e3, seconds * 1e6 / dataset.frames,
           coded_bits / seconds / 1e6);
    printf("avg iterations:  %.3f\n", static_cast<double>(total_iterations) / dataset.frames);
    printf("channel BER:     %.3e (hard decision before decoding)\n", channel_bit_errors / coded_bits);
    printf("decoded BER:     %.3e (%lld bit errors)\n", bit_errors / coded_bits, bit_errors);
    printf("decoded FER:     %.3e (%d frame errors)\n", static_cast<double>(frame_errors) / dataset.frames,
           frame_errors);
    printf("output checksum: %016llx\n", static_cast<unsigned long long>(checksum));
}
