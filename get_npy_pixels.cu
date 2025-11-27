// get_npy_pixels.cu
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <cctype>

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)        \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)

// -----------------------------------------------------------------------------
// Minimal loader for 2D int32 npy of known shape (FS, FS)
// -----------------------------------------------------------------------------
static std::vector<int> load_npy_2d_int32(const char* filename, int FS) {
    std::ifstream f(filename, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("Could not open file: ") + filename);

    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY")
        throw std::runtime_error("Not a .npy file (bad magic)");

    char major = 0, minor = 0;
    f.read(&major, 1);
    f.read(&minor, 1);

    unsigned short header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);

    std::string header(header_len, ' ');
    f.read(&header[0], header_len);

    if (header.find("<i4") == std::string::npos)
        throw std::runtime_error("dtype must be int32 <i4");

    if (header.find("'fortran_order': True") != std::string::npos ||
        header.find("\"fortran_order\": true") != std::string::npos)
        throw std::runtime_error("Fortran-order arrays not supported");

    std::size_t pos_shape = header.find("shape");
    if (pos_shape == std::string::npos)
        throw std::runtime_error("No 'shape' in header");

    std::size_t pos_open  = header.find('(', pos_shape);
    std::size_t pos_close = header.find(')', pos_open);
    if (pos_open == std::string::npos || pos_close == std::string::npos)
        throw std::runtime_error("Failed to parse shape");

    std::string shape_str = header.substr(pos_open + 1,
                                          pos_close - pos_open - 1);
    shape_str.erase(std::remove_if(shape_str.begin(), shape_str.end(), ::isspace),
                    shape_str.end());
    if (!shape_str.empty() && shape_str.back() == ',')
        shape_str.pop_back();

    std::size_t comma_pos = shape_str.find(',');
    if (comma_pos == std::string::npos)
        throw std::runtime_error("Expected 2D shape (FS,FS)");

    std::string dim0_str = shape_str.substr(0, comma_pos);
    std::string dim1_str = shape_str.substr(comma_pos + 1);

    int dim0 = std::stoi(dim0_str);
    int dim1 = std::stoi(dim1_str);
    if (dim0 != FS || dim1 != FS)
        throw std::runtime_error("Shape mismatch in npy: expected (FS,FS)");

    std::size_t total = static_cast<std::size_t>(FS) * FS;
    std::vector<int> data(total);
    f.read(reinterpret_cast<char*>(data.data()), total * sizeof(int));
    if (!f)
        throw std::runtime_error("Failed to read array data");

    return data;
}

// -----------------------------------------------------------------------------
// Tiling kernel for 2D base → (BS,FS,FS)
// -----------------------------------------------------------------------------
__global__ void tile_2d_kernel(const int* base, int* out, int BS, int FS) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BS * FS * FS;
    if (idx >= total) return;

    int base_size = FS * FS;
    int base_idx  = idx % base_size;  // tile for each batch
    out[idx] = base[base_idx];
}

// -----------------------------------------------------------------------------
// SaC-facing function (2D → (BS,FS,FS)):
//
// SaC call:
//   load_field3d_from_npy(pixels, BS, FS, ['p','\0'])
//
// C view:
//   BS -> BS
//   FS -> FS (field_size = 512)
//   filename -> "p\0"
//
// Loads "p_FS.npy" with shape (FS,FS), tiles into (BS,FS,FS) in 'field'.
// -----------------------------------------------------------------------------
extern "C"
int* load_field3d_from_npy(int* field,
                           int BS, int FS,
                           char* filename)
{
    if (BS <= 0 || FS <= 0)
        return field;

    char base_char = filename[0]; // 'p'
    std::string full = "init_data/" +
                   std::string(1, base_char) +
                   "_" + std::to_string(FS) + ".npy";

    std::vector<int> host_vec;
    try {
        host_vec = load_npy_2d_int32(full.c_str(), FS);
    } catch (const std::exception& e) {
        std::cerr << "Error loading " << full << ": " << e.what() << "\n";
        return field;
    }

    const std::size_t base_elems = static_cast<std::size_t>(FS) * FS;
    int* d_base = nullptr;
    CUDA_CHECK(cudaMalloc(&d_base, base_elems * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_base, host_vec.data(),
                          base_elems * sizeof(int),
                          cudaMemcpyHostToDevice));

    int total   = BS * FS * FS;
    int tpb     = 256;
    int blocks  = (total + tpb - 1) / tpb;

    tile_2d_kernel<<<blocks, tpb>>>(d_base, field, BS, FS);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_base));

    return field;
}
