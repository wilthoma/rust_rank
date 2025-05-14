#ifndef CUDANTT_H
#define CUDANTT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>
#include "modular_linalg.h"
#include "ntt.h"

using u64 = uint64_t;

// Example prime and primitive root for NTT
const u64 p = 2305843009146585089ULL;     // = 15 * 2^27 + 1, 31-bit prime
const u64 root = 3;          // primitive root modulo p


__device__ u64 dmod_add(u64 a, u64 b, u64 p) {
    u64 s = a + b;
    return s >= p ? s - p : s;
}

__device__ u64 dmod_sub(u64 a, u64 b, u64 p) {
    return a >= b ? a - b : p + a - b;
}

__device__ __forceinline__ u64 dmod_mul(u64 a, u64 b, u64 p) {

    u64 lo = a * b;
    u64 hi = __umul64hi(a, b);
    u64 res;

    //__uint128_t product = static_cast<__uint128_t>(a) * b;
    u64 cc = (lo << 3) >> 3; // lowest 61 bits 
    u64 bbh = (hi << 32) >> 29; // bits 64-95
    u64 bbl = lo >> 61; // bits 61-63
    u64 bb = bbh | bbl; // bits 61-95
    u64 aa = hi >> 32; // bits 96-127
    return dmod_add(dmod_sub(dmod_add(cc, bb << 26,p), bb,p), dmod_sub(dmod_add(aa << 26, p - aa,p), aa << 35,p),p);


    // uint64_t cc = static_cast<uint64_t>((product << 67) >> 67); // lowest 61 bits
    // uint64_t bb = static_cast<uint64_t>((product << 32) >> 93); // bits 61-95
    // uint64_t aa = static_cast<uint64_t>(product >> 96);         // bits 96-127
    // return mod_add(mod_sub(mod_add(cc, bb << 26), bb), mod_sub(mod_add(aa << 26, PRIME<T>() - aa), aa << 35));


    // u64 lo = a * b;
    // u64 hi = __umul64hi(a, b);
    // u64 res;

    // asm("{\n\t"
    //     ".reg .u64 t1, t2;\n\t"
    //     "mul.lo.u64 t1, %1, %2;\n\t"
    //     "mul.hi.u64 t2, %1, %2;\n\t"
    //     "mad.wide.u32 %0, 0, 0, t1;\n\t"
    //     "rem.u64 %0, %0, %3;\n\t"
    //     "}\n"
    //     : "=l"(res)
    //     : "l"(a), "l"(b), "l"(p));

    // return res;
}

__global__ void bit_reverse_kernel(u64* data, u64 n, u64 logn) {
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    u64 rev = 0;
    for (u64 i = 0; i < logn; ++i)
        rev |= ((tid >> i) & 1) << (logn - 1 - i);
    if (tid < rev) {
        u64 tmp = data[tid];
        data[tid] = data[rev];
        data[rev] = tmp;
    }
}

__global__ void ntt_stage_kernel(u64* data, const u64* twiddles, u64 n, u64 step, u64 p) {
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    u64 m = step * 2;
    if (tid >= n / 2) return;

    u64 i = (tid / step) * m + (tid % step);
    u64 j = i + step;
    u64 w = twiddles[(n / m) * (tid % step)];

    u64 u = data[i];
    u64 v = dmod_mul(data[j], w, p);

    data[i] = dmod_add(u, v, p);
    data[j] = dmod_sub(u, v, p);
}

__global__ void scale_kernel(u64* data, u64 n, u64 ninv, u64 p) {
    u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = dmod_mul(data[tid], ninv, p);
    }
}



// u64 modinv(u64 a, u64 p) {
//     u64 t = 0, newt = 1;
//     u64 r = p, newr = a;

//     while (newr != 0) {
//         u64 q = r / newr;
//         u64 tmp = newt;
//         newt = t - q * newt; t = tmp;

//         tmp = newr;
//         newr = r - q * newr; r = tmp;
//     }
//     return (t + p) % p;
// }

void generate_twiddles(std::vector<u64>& tw, u64 n, u64 root, u64 p, bool inverse = false) {
    u64 r = inverse ? modinv(root, p) : root;
    tw[0] = 1;
    for (u64 i = 1; i < n / 2; ++i) {
        tw[i] = (__uint128_t)tw[i - 1] * r % p;
    }
}

void ntt_cuda(std::vector<u64>& h_data, bool inverse = false) {
    u64 n = h_data.size();
    if ((n & (n - 1)) != 0) {
        throw std::invalid_argument("n must be a power of two");
    }

    u64 logn = std::log2(n);

    std::vector<u64> h_twiddles(n / 2);
    generate_twiddles(h_twiddles, n, root, p, inverse);

    u64 *d_data = nullptr, *d_twiddles = nullptr;
    cudaMalloc(&d_data, n * sizeof(u64));
    cudaMalloc(&d_twiddles, (n / 2) * sizeof(u64));
    cudaMemcpy(d_data, h_data.data(), n * sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_twiddles, h_twiddles.data(), (n / 2) * sizeof(u64), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    bit_reverse_kernel<<<numBlocks, threadsPerBlock>>>(d_data, n, logn);
    cudaDeviceSynchronize();

    for (u64 step = 1; step < n; step *= 2) {
        u64 threads = n / 2;
        u64 blocks = (threads + threadsPerBlock - 1) / threadsPerBlock;
        ntt_stage_kernel<<<blocks, threadsPerBlock>>>(d_data, d_twiddles, n, step, p);
        cudaDeviceSynchronize();
    }

    if (inverse) {
        u64 ninv = modinv(n, p);
        scale_kernel<<<numBlocks, threadsPerBlock>>>(d_data, n, ninv, p);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_data.data(), d_data, n * sizeof(u64), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_twiddles);
}


void test_ntt_cuda_inv() {
    size_t n = 1 << 20; // e.g., 2^20 = 1 million
    std::vector<u64> data(n);
    for (u64 i = 0; i < n; ++i)
        data[i] = i % p;
    std::vector<u64> data_copy = data; // Copy for verification

    ntt_cuda(data, false); // Forward NTT
    ntt_cuda(data, true);  // Inverse NTT (should restore original)


    // Check if the original data is restored
    assert(data == data_copy && "NTT followed by inverse NTT should return the original data");
    std::cout << "Cuda NTT and inverse NTT test passed!" << std::endl;
}

void test_ntt_cuda_same_as_ntt() {
    size_t n = 1 << 20; // e.g., 2^20 = 1 million
    std::vector<u64> data(n);
    for (u64 i = 0; i < n; ++i)
        data[i] = i % p;

    std::vector<u64> data_copy = data; // Copy for verification

    ntt_cuda(data, false); // Forward NTT
    ntt(data_copy, false); // Forward NTT on CPU

    // Check if the results are the same
    assert(data == data_copy && "Cuda NTT and CPU NTT should produce the same result");
    std::cout << "Cuda NTT and CPU NTT test passed!" << std::endl;
}

void test_modmul_cuda() {
    size_t n = 1 << 20; // e.g., 2^20 = 1 million
    std::vector<u64> a(n), b(n), host_result(n), device_result(n);

    // Generate random integers between 0 and p-1
    for (size_t i = 0; i < n; ++i) {
        a[i] = rand() % p;
        b[i] = rand() % p;
    }

    // Host-side multiplication mod p
    for (size_t i = 0; i < n; ++i) {
        host_result[i] = (static_cast<__uint128_t>(a[i]) * b[i]) % p;
    }

    // Device-side multiplication mod p
    u64 *d_a = nullptr, *d_b = nullptr, *d_result = nullptr;
    cudaMalloc(&d_a, n * sizeof(u64));
    cudaMalloc(&d_b, n * sizeof(u64));
    cudaMalloc(&d_result, n * sizeof(u64));

    cudaMemcpy(d_a, a.data(), n * sizeof(u64), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(u64), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    auto modmul_kernel = [] __global__(u64* a, u64* b, u64* result, u64 n, u64 p) {
        u64 tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid < n) {
            result[tid] = dmod_mul(a[tid], b[tid], p);
        }
    };

    modmul_kernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_result, n, p);
    cudaDeviceSynchronize();

    cudaMemcpy(device_result.data(), d_result, n * sizeof(u64), cudaMemcpyDeviceToHost);

    // Compare results
    assert(host_result == device_result && "Host and device results should match");
    std::cout << "Cuda modmul test passed!" << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
}

oid bit_reverse_host(std::vector<u64>& data, u64 n, u64 logn) {
    for (u64 i = 0; i < n; ++i) {
        u64 rev = bit_reverse(i, logn);
        // for (u64 j = 0; j < logn; ++j) {
        //     rev |= ((i >> j) & 1) << (logn - 1 - j);
        // }
        if (i < rev) {
            std::swap(data[i], data[rev]);
        }
    }
}

void test_bit_reverse_cuda() {
    size_t n = 1 << 20; // e.g., 2^20 = 1 million
    u64 logn = std::log2(n);

    // Generate test data
    std::vector<u64> host_data(n);
    for (u64 i = 0; i < n; ++i) {
        host_data[i] = i;
    }

    // Perform bit reversal on the host
    std::vector<u64> host_result = host_data;
    bit_reverse_host(host_result, n, logn);

    // Perform bit reversal on the device
    u64* d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(u64));
    cudaMemcpy(d_data, host_data.data(), n * sizeof(u64), cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    bit_reverse_kernel<<<numBlocks, threadsPerBlock>>>(d_data, n, logn);
    cudaDeviceSynchronize();

    std::vector<u64> device_result(n);
    cudaMemcpy(device_result.data(), d_data, n * sizeof(u64), cudaMemcpyDeviceToHost);

    // Compare results
    assert(host_result == device_result && "Host and device bit reversal results should match");
    std::cout << "Cuda bit reverse test passed!" << std::endl;

    cudaFree(d_data);
}

#endif // CUDANTT_H