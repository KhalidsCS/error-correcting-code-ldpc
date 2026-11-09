# CUDA LDPC decoder

Min-sum LDPC decoding on an AWGN channel, first as a Python prototype and then as matching CUDA and C++ implementations that can be compared on the same frames.

The decoder is a scaled min-sum update on the Tanner graph. Each check finds the two smallest-magnitude incoming beliefs, multiplies by the product of the other bits' signs and a 0.75 scale factor, and writes that correction back onto the bits. Iteration stops when every parity check is satisfied, or after a maximum of 50 iterations.

## Layout

| Path | Role |
|------|------|
| `ldpc_prototype.py` | First decoder: (7, 4) Hamming code, one word at a time |
| `cuda-ecc/ldpc.cu` | CUDA kernels: check-to-bit min-sum, then bit-to-check sum |
| `cuda-ecc/main.cu` | CUDA driver: batched launches, syndrome check, early stop |
| `cuda-ecc/ldpc_cpu.cpp` | Single-thread C++ decoder with the same update rule |
| `cuda-ecc/make_dataset.cu` | Regular Gallager codes, random codewords, GPU AWGN |
| `cuda-ecc/ldpc_dataset.{h,cpp}` | Shared dataset format and Tanner-graph helpers |

Both compiled decoders load the same `dataset.bin` and print the same report: decode time, average iterations, BER/FER, and a checksum over every decoded bit and iteration count. Matching checksums mean the two paths produced bit-identical output.

## Decoder

Each frame is a noisy BPSK word (`0 → +1`, `1 → -1`). Beliefs start at the received values.

1. **Check nodes** (`generate_minimums_per_check`). For each check, subtract that check's previous message from each connected bit (no echo of the check's own last opinion). Track the smallest and second-smallest magnitudes and the product of signs. Write a new check-to-bit message: `0.75 × (product of other signs) × (the other bit's min magnitude)`.
2. **Bit nodes** (`generate_check_per_bit`). Reset each bit to the channel value and add every incoming check-to-bit message.
3. **Hard decision.** Negative belief → 1, otherwise 0. Stop when `H x = 0`, else repeat.

The first version fed each check the bit's full belief, so a check's own previous message came back as if it were new information. Subtracting that message (the echo) is what made the error rate drop from tens of percent of frames to a few percent or zero on the same datasets.

On the GPU, `blockIdx.x` is the bit or check inside a frame and `blockIdx.y` is the frame in the batch, so one launch covers every frame in the batch.

## Dataset

`make_dataset` builds a regular Gallager Tanner graph (`checks_per_bit` bands, each a permutation of a banded identity), encodes random valid codewords, and adds Gaussian noise on the GPU with cuRAND.

Default: 10,000 frames of a (3, 6)-regular code with `n = 1020`, `k = 512`, `m = 510`, at 2 dB Eb/N0. Noise variance is `σ² = 1 / (2 R Eb/N0)` with `R = k/n`.

```
make_dataset [--out dataset.bin] [--frames 10000] [--n 1020]
             [--checks-per-bit 3] [--bits-per-check 6] [--snr 2.0] [--seed 1]
```

## Build and run

From an x64 Visual Studio developer prompt (so `cl.exe` is on PATH), in `cuda-ecc`:

```
cmake -S . -B build -G Ninja
cmake --build build
cd build
.\make_dataset.exe
.\ldpc_cpu.exe
.\ldpc_cuda.exe
```

Optional flags for the decoders: `[dataset.bin] [--max-iterations 50]`. `ldpc_cuda` also takes `--batch 128`.

On Windows, a plain PowerShell has no `cl.exe`. Use the "Developer PowerShell (VS 2022 x64)" profile, or load the VS Build Tools environment first.

## Results

Measured on an RTX 4070 Laptop GPU against a single-thread C++ decoder. Both paths produced the same checksum, BER, FER, and iteration count on each dataset.

| Dataset | C++ | CUDA | Speedup | FER | Avg iterations |
|---------|-----|------|---------|-----|----------------|
| n=1020, 2 dB, 10k frames | 6.03 s | 1.24 s | 4.9× | 2.0% | 12.0 |
| n=1020, 3 dB, 10k frames | 2.70 s | 0.35 s | 7.7× | 0 | 5.4 |
| n=4098, 2.5 dB, 1k frames | 1.59 s | 0.35 s | 4.5× | 0 | 7.9 |

Before the echo fix, the same 2 dB dataset had a 58% frame error rate and 38 average iterations. After it: 2.0% FER and 12 iterations.

CUDA is faster on throughput (time to finish the whole dataset). The C++ decoder is faster per individual frame (~0.6 ms vs a 128-frame CUDA batch of ~16 ms at 2 dB), because it stops each frame as soon as the syndrome is zero. A CUDA batch keeps iterating until its slowest frame finishes.

## What is still open

- Finished frames in a CUDA batch still run until the slowest frame in that batch is done. Returning early when `done[frame]` is set would reclaim that work, especially at higher SNR.
- `minimums` is stored as a dense `m × n` matrix per frame. Storing one value per Tanner edge would shrink it by about the reciprocal of the check degree (170× on the default code) and allow larger codes and batches.
