// volumes_update.cu
#include <cuda_runtime.h>
extern "C" {

__global__ void k_volumes_update(const int* __restrict__ pixels,
                                 const int* __restrict__ next,
                                 int* __restrict__ out,
                                 int nb, int sy, int sx, int nc)
{
    int b = blockIdx.z;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb || y >= sy || x >= sx) return;

    const int plane = sy * sx;
    const int idx   = b * plane + y * sx + x;

    int oldc = pixels[idx];
    int newc = next[idx];

    // Update counts (includes c==0 same as your SaC reference version)
    if (0 <= oldc && oldc < nc) atomicAdd(&out[b * nc + oldc], -1);
    if (0 <= newc && newc < nc) atomicAdd(&out[b * nc + newc], +1);
}

// SaC => (pixels, next, out, nb, sy, sx, nc)
int* volumes_update(const int* pixels_dev,
                    const int* next_dev,
                    int* out_dev,
                    int nb, int sy, int sx, int nc)
{
    // clear output
    cudaMemset(out_dev, 0, size_t(nb) * size_t(nc) * sizeof(int));

    dim3 block(32, 8, 1);
    dim3 grid((sx + block.x - 1)/block.x,
              (sy + block.y - 1)/block.y,
              nb);

    k_volumes_update<<<grid, block>>>(pixels_dev, next_dev, out_dev, nb, sy, sx, nc);
    // (check errors in your build if you like)
    return out_dev;  // in-place: return == out
}

} // extern "C"
