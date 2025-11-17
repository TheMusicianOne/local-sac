// reseed_mask.cu
#include <cuda_runtime.h>

// Fast PCG random number generator - much faster than curand
// https://www.pcg-random.org/
__device__ inline unsigned int pcg_hash(unsigned int seed)
{
    unsigned int state = seed * 747796405u + 2891336453u;
    unsigned int word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

__device__ inline float pcg_float(unsigned int seed)
{
    unsigned int h = pcg_hash(seed);
    return (float)h / 4294967296.0f;  // Map to [0,1)
}

__global__ void reseed_mask_kernel(float* __restrict__ mask,
                                   int nb, int sy, int sx)
{
    const int nc = 3;
    unsigned long long total = (unsigned long long)nb * sy * sx * nc;
    unsigned long long idx   = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // Mix clock, grid position, and index for unique seed
    unsigned long long time_seed = clock64();
    unsigned int seed = (unsigned int)(time_seed + idx * 0x9e3779b9u + blockIdx.x * 0x517cc1b7u);
    
    // Generate random float in (0,1]
    mask[idx] = pcg_float(seed);
}

extern "C" float* reseed_mask(float* mask, int nb, int sy, int sx)
{
    const int nc = 3;
    const unsigned long long total = (unsigned long long)nb * sy * sx * nc;
    if (!total) return mask;
    
    const int tpb = 256;
    const int blocks = (int)((total + tpb - 1) / tpb);
    
    reseed_mask_kernel<<<blocks, tpb>>>(mask, nb, sy, sx);
    
    return mask;
}