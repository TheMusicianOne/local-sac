// get_npy.cu
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
// Minimal loader for 1D int32 npy of known length FS
// -----------------------------------------------------------------------------
static std::vector<int> load_npy_1d_int32(const char* filename, int FS) {
    std::ifstream f(filename, std::ios::binary);
    if (!f)
        throw std::runtime_error(std::string("Could not open file: ") + filename);

    char magic[6];
    f.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY")
        throw std::runtime_error("Not a .npy file");

    char major = 0, minor = 0;
    f.read(&major, 1);
    f.read(&minor, 1);

    unsigned short header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 2);

    std::string header(header_len, ' ');
    f.read(&header[0], header_len);

    if (header.find("<i4") == std::string::npos)
        throw std::runtime_error("dtype must be int32 <i4");

    // fortran_order check (not critical for 1D, but sanity)
    if (header.find("'fortran_order': True") != std::string::npos ||
        header.find("\"fortran_order\": true") != std::string::npos)
        throw std::runtime_error("Fortran-order not supported");

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

    int parsed_len = std::stoi(shape_str);
    if (parsed_len != FS)
        throw std::runtime_error("Shape mismatch in npy");

    std::vector<int> data(FS);
    f.read(reinterpret_cast<char*>(data.data()), FS * sizeof(int));
    if (!f)
        throw std::runtime_error("Failed to read array data");

    return data;
}

// -----------------------------------------------------------------------------
// Tiling kernel: out[b*FS + i] = base[i]
// -----------------------------------------------------------------------------
__global__ void tile_kernel(const int* base, int* out, int BS, int FS) {
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    int total = BS * FS;
    if (idx >= total) return;
    int i = idx % FS;
    out[idx] = base[i];
}

// -----------------------------------------------------------------------------
// SaC-facing function (1D → (BS,FS)):
//
// SaC call:
//   load_field_from_npy(field, BS, NCPO, FS, ['v','\0'])
//
// C view:
//   BS -> BS
//   FS -> FS (length of 1D array, NCPO)
//   NC -> FS (unused here)
//   filename -> "v\0" or "c\0"
//
// Loads filename[0] + "_" + FS + ".npy", shape (FS,).
// Tiles it BS times into 'field' (size BS*FS ints).
// -----------------------------------------------------------------------------
extern "C"
int* load_field_from_npy(int* field,
                         int BS, int FS, int NC,
                         char* filename)
{
    // BS: batches
    // FS: field size (512) - not used here
    // NC: length of 1D array (NCPO = 2369)

    if (BS <= 0 || NC <= 0)
        return field;

    char base_char = filename[0]; // 'v' or 'c'

    // Use NC as the suffix, because arrays are length NCPO:
    std::string full = "init_data/" +
                    std::string(1, base_char) +
                   "_" + std::to_string(FS) + ".npy";

    std::vector<int> host_vec;
    try {
        host_vec = load_npy_1d_int32(full.c_str(), NC);
    } catch (const std::exception& e) {
        std::cerr << "Error loading " << full << ": " << e.what() << "\n";
        return field;
    }

    int* d_base = nullptr;
    CUDA_CHECK(cudaMalloc(&d_base, NC * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_base, host_vec.data(),
                          NC * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Tile into field: shape (BS, NCPO)
    int total  = BS * NC;
    int tpb    = 256;
    int blocks = (total + tpb - 1) / tpb;

    // Here, "FS" of the kernel is actually NC = inner length
    tile_kernel<<<blocks, tpb>>>(d_base, field, BS, NC);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_base));

    return field;
}
