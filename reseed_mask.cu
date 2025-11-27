// reseed_mask.cu
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void reseed_mask_kernel(float* __restrict__ mask,
                                   int nb, int sy, int sx, int seed)
{
    const int nc = 3; // mask[...,3]
    unsigned long long total = (unsigned long long)nb * sy * sx * nc;
    unsigned long long idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // cuRAND state per element
    curandStatePhilox4_32_10_t st;
    curand_init((unsigned long long)seed, idx, 0, &st);

    // Write one float in (0,1]; OK as mask
    mask[idx] = curand_uniform(&st);
}

extern "C" float* reseed_mask(float* mask,
                              int nb, int sy, int sx, int seed)
{
    const int nc = 3;
    const unsigned long long total = (unsigned long long)nb * sy * sx * nc;
    if (!total) return mask;

    const int tpb = 256;
    const int blocks = (int)((total + tpb - 1) / tpb);

    reseed_mask_kernel<<<blocks, tpb>>>(mask, nb, sy, sx, seed);
    // Keep this while debugging so your SaC print sees fresh data:
    //cudaDeviceSynchronize();

    return mask; // same pointer → SaC won’t D→H
}
