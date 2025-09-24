#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gpuerrors.h"
#include "htt.h"   // assumes k_const is declared here (e.g., __constant__ float k_const;)

#define tx threadIdx.x
#define ty threadIdx.y
#define bx blockIdx.x
#define by blockIdx.y
#define bdx blockDim.x
#define bdy blockDim.y

// --- 1D texture bound to linear memory; tex1Dfetch uses integer indices ---
texture<float, cudaTextureType1D, cudaReadModeElementType> texIn;

// 5-point stencil using texture cache; safe at borders (clamped indices)
__global__ void stencil1D(float* __restrict__ out, const unsigned int N)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    unsigned int idx = y * N + x;

    // Clamp neighbors to avoid OOB reads/writes
    unsigned int a = (x > 0)       ? idx - 1 : idx;      // left
    unsigned int b = (x + 1 < N)   ? idx + 1 : idx;      // right
    unsigned int c = (y > 0)       ? idx - N : idx;      // up
    unsigned int d = (y + 1 < N)   ? idx + N : idx;      // down

    float c0 = tex1Dfetch(texIn, idx);
    float l  = tex1Dfetch(texIn, a);
    float r  = tex1Dfetch(texIn, b);
    float u  = tex1Dfetch(texIn, c);
    float dn = tex1Dfetch(texIn, d);

    out[idx] = c0 + k_const * (l + r + u + dn - 4.0f * c0);
}

void gpuKernel(float* ad, float* cd, const unsigned int N, const unsigned int M)
{
    // Bind 1D texture to *linear* memory (device pointer `ad`)
    const size_t bytes = size_t(N) * size_t(N) * sizeof(float);
    HANDLE_ERROR(cudaBindTexture(0, texIn, ad, bytes));

    // Block/grid config: (M==12||13) -> (128,2), else -> (64,4)
    const bool bigX = (M == 12 || M == 13);
    dim3 block(bigX ? 128 : 64, bigX ? 2 : 4);
    dim3 grid( (N + block.x - 1) / block.x,
               (N + block.y - 1) / block.y );

    // Launch
    stencil1D<<<grid, block>>>(cd, N);
    HANDLE_ERROR(cudaGetLastError());

    // Unbind texture
    HANDLE_ERROR(cudaUnbindTexture(texIn));
}
