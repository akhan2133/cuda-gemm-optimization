# CUDA GEMM Optimization

CUDA-based matrix multiplication optimization project developed as part of **CMPT 479: GPU Architecture and Programming Fall 2026** at Simon Fraser University.

This project explores GPU performance optimization techniques through multiple implementations of General Matrix Multiplication (GEMM), focusing on memory access efficiency, data reuse, and execution performance.

## Implementations

The project explores four CUDA kernel implementations:

- **Naive GEMM:** Baseline matrix multiplication using one thread per output element.
- **Shared-Memory Tiling:** Uses shared memory to improve data reuse and reduce global memory accesses.
- **Vectorized GEMM:** Uses `float4` memory accesses and computes four output elements per thread.
- **Double-Buffered GEMM:** Uses alternating shared-memory buffers to investigate synchronization overhead.

## Performance Evaluation

Benchmarked FP32 matrix multiplication on 20,000 × 20,000 matrices using NVIDIA GPU hardware.

| Implementation | Kernel Time | Speedup |
|---|---:|---:|
| Naive GEMM | 21,951.4 ms | 1.00× |
| Shared-Memory Tiling | 16,945.8 ms | 1.30× |
| Vectorized GEMM | 6,859.02 ms | 3.20× |
| Double-Buffered GEMM | 7,016.41 ms | 3.13× |

*Speedups are calculated relative to the naive kernel using a 32 × 32 thread-block configuration. Results reflect the fastest measured configuration for each optimized implementation.*

The vectorized implementation achieved the best measured performance, with a **3.20× speedup** over the baseline.

Double buffering improved performance for some tile configurations but introduced additional shared-memory usage, illustrating the tradeoffs involved in GPU optimization.

## Technologies

- CUDA C++
- NVIDIA GPU architecture
- Shared memory and memory coalescing
- Vectorized memory access (`float4`)
- CUDA Events for kernel benchmarking
- Linux development environment

## Source Code Availability

This project was completed as coursework for CMPT 479 at Simon Fraser University.

The source code is maintained in a private repository in accordance with the course's academic integrity policy and is not publicly distributed.

I am happy to discuss the implementation, optimization techniques, and performance findings during technical interviews. Any demonstration of private source code is subject to instructor approval.
