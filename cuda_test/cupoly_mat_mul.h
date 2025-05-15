#ifndef CUPOLY_MAT_MUL_H
#define CUPOLY_MAT_MUL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>
#include "poly_mat_mul.h"
#include "ntt.h"
#include "cudantt.h"

using namespace std;

std::vector<u64> poly_mul_fft_cuda(const std::vector<u64>& p1, const std::vector<u64>& p2) {
    size_t n = p1.size();
    size_t m = p2.size();
    size_t nlenres = n + m - 1;
    // find next highest power of two
    size_t nlenres_adj = 1;
    while (nlenres_adj < nlenres) {
        nlenres_adj <<= 1;
    }
    // pad with zeros
    std::vector<u64> fa(nlenres_adj, 0);
    std::vector<u64> fb(nlenres_adj, 0);
    std::copy(p1.begin(), p1.end(), fa.begin());
    std::copy(p2.begin(), p2.end(), fb.begin());
    // perform ntt
    auto start = high_resolution_clock::now();
    ntt_cuda(fa, false);
    ntt_cuda(fb, false);
    auto durationntt = high_resolution_clock::now() - start;
    // multiply
    std::vector<u64> fresult(nlenres_adj, 0);
    for (size_t i = 0; i < nlenres_adj; ++i) {
        fresult[i] = mod_mul(fa[i], fb[i]);
    }
    // perform inverse ntt
    // start = high_resolution_clock::now();
    ntt_cuda(fresult, true);
    // auto durationmul = high_resolution_clock::now() - start;
    // resize result
    fresult.resize(nlenres);
    return fresult;
}

vector<vector<vector<u64>>> cupoly_mat_mul_fft1(const vector<vector<vector<u64>>>& a, const vector<vector<vector<u64>>>& b) {
    size_t m = a.size();
    size_t n = a[0].size();
    size_t k = b[0].size();
    if (n != b.size()) {
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }

    size_t nlena = a[0][0].size();
    size_t nlenb = b[0][0].size();
    size_t nlenres = nlena + nlenb - 1;

    size_t nlenres_adj = 1;
    while (nlenres_adj < nlenres) {
        nlenres_adj <<= 1;
    }

    vector<u64> fa(m*n*nlenres_adj, 0);
    vector<u64> fb(n*k*nlenres_adj, 0);
    vector<u64> fresult(m*k*nlenres_adj, 0);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < nlena; ++l) {
                fa[l*m*n + i*n + j] = a[i][j][l];
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < nlenb; ++l) {
                fb[l*n*k + i*k + j] = b[i][j][l];
            }
        }
    }

    // auto start = high_resolution_clock::now();
    // matrix_ntt_parallel(fa, false);
    // matrix_ntt_parallel(fb, false);
    // auto durationntt = high_resolution_clock::now() - start;
    ntt_cuda_colwise(fa, m*n, false);
    ntt_cuda_colwise(fb, n*k, false);

    // start = high_resolution_clock::now();
    // multiply
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t r = 0; r < n; ++r) {
                for (size_t l = 0; l < nlenres_adj; ++l) {
                    fresult[l*m*k + i*k+j] = mod_add(fresult[l*m*k + i*k+j],mod_mul(fa[l*m*n+i*n+r], fb[l*n*k+r*k+j]));
                }
            }
        }
    }


    // start = high_resolution_clock::now();
    // matrix_ntt_parallel(fresult, true);
    // durationntt += high_resolution_clock::now() - start;
    ntt_cuda_colwise(fresult, m*k, true);

    // resize result
    vector<vector<vector<u64>>> hresult(m, vector<vector<u64>>(k, vector<u64>(nlenres, 0)));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < nlenres; ++l) {
                hresult[i][j][l] = fresult[l*m*k + i*k+j];
            }
        }
    }

    return hresult;

}

// Kernel for matrix multiplication
// __global__ void vmatmul_kernel(u64* dA, u64* dB, u64* dC, int m, int n, int k, int deg, u64 p) {
//     u64 dg = blockIdx.x * blockDim.x + threadIdx.x;
//     u64 row = blockIdx.z * blockDim.z + threadIdx.z;
//     u64 col = blockIdx.y * blockDim.y + threadIdx.y;

//     if (row < m && col < k && dg < deg) {
//         u64 sum = 0;
//         for (int r = 0; r < n; ++r) {
//             sum = dmod_add(sum, dmod_mul(dA[m * n * dg + row * n + r], dB[n * k * dg + r * k + col],p),p);
//         }
//         dC[m*k*dg + row * k + col] = sum;
//     }
// }
__global__ void vmatmul_kernel(u64* dA, u64* dB, u64* dC, int m, int n, int k, int deg, u64 p) {
    u64 dg = blockIdx.x * blockDim.x + threadIdx.x;
    u64 row = blockIdx.z * blockDim.z + threadIdx.z;
    u64 col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < m && col < k && dg < deg) {
        u64 sum = 0;
        u64 A_offset = dg * m * n + row * n;
        u64 B_offset = dg * n * k + col;
        for (int r = 0; r < n; ++r) {
            sum = dmod_add(sum, dmod_mul(dA[A_offset + r], dB[B_offset + r * k], p), p);
        }
        dC[dg * m * k + row * k + col] = sum;
    }
}

void cuda_vmatmul(u64* dA, u64* dB, u64* dC, int m, int n, int k, int deg) {
    // initialize 
    // cudaMemset(dC, 0, m * k * deg * sizeof(u64)); CUCHECK

    // Kernel launch parameters
    dim3 blockSize(16, 8, 8);
    dim3 gridSize((deg + blockSize.x - 1) / blockSize.x, (k + blockSize.y - 1) / blockSize.y, (m + blockSize.z - 1) / blockSize.z);

    // Launch the kernel
    vmatmul_kernel<<<gridSize, blockSize>>>(dA, dB, dC, m, n, k, deg, PRIME<u64>()); CUCHECK
}

// vector<u64> vecvecvec_to_vec(const vector<vector<vector<u64>>>& vec) {
//     size_t m = vec.size();
//     size_t n = vec[0].size();
//     size_t k = vec[0][0].size();
    

//     return result;
// }

// u64* vecvecvec_to_cuda(const vector<vector<vector<u64>>>& vec) {
//     u64* d_vec;
//     size_t size = m * n * k * deg * sizeof(u64);
//     cudaMalloc(&d_vec, size); CUCHECK

//     // Copy data to device
//     vector<u64> h_vec(m * n * k * deg);

    

vector<vector<vector<u64>>> cupoly_mat_mul_fft2(const vector<vector<vector<u64>>>& a, const vector<vector<vector<u64>>>& b) {
    size_t m = a.size();
    size_t n = a[0].size();
    size_t k = b[0].size();
    if (n != b.size()) {
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }

    size_t nlena = a[0][0].size();
    size_t nlenb = b[0][0].size();
    size_t nlenres = nlena + nlenb - 1;

    size_t nlenres_adj = 1;
    while (nlenres_adj < nlenres) {
        nlenres_adj <<= 1;
    }

    vector<u64> fa(m*n*nlenres_adj, 0);
    vector<u64> fb(n*k*nlenres_adj, 0);
    vector<u64> fresult(m*k*nlenres_adj, 0);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < nlena; ++l) {
                fa[l*m*n + i*n + j] = a[i][j][l];
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < nlenb; ++l) {
                fb[l*n*k + i*k + j] = b[i][j][l];
            }
        }
    }

    // auto start = high_resolution_clock::now();
    // matrix_ntt_parallel(fa, false);
    // matrix_ntt_parallel(fb, false);
    // auto durationntt = high_resolution_clock::now() - start;
    u64 *dfa, *dfb, *dresult; 
    CHECK_CUDA(cudaMalloc(&dfa, fa.size() * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&dfb, fb.size() * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&dresult, fresult.size() * sizeof(u64)));
    CHECK_CUDA(cudaMemcpy(dfa, fa.data(), fa.size() * sizeof(u64), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dfb, fb.data(), fb.size() * sizeof(u64), cudaMemcpyHostToDevice));

    ntt_cuda_colwise_gpu(dfa, nlenres_adj, m*n, false);
    ntt_cuda_colwise_gpu(dfb, nlenres_adj, n*k, false);


    // CHECK_CUDA(cudaMemset(dresult, 0, m*k*nlenres_adj * sizeof(u64)));
    cuda_vmatmul(dfa, dfb, dresult, m, n, k, nlenres_adj);

    // start = high_resolution_clock::now();
    // matrix_ntt_parallel(fresult, true);
    // durationntt += high_resolution_clock::now() - start;
    ntt_cuda_colwise_gpu(dresult, nlenres_adj, m*k, true);
    CHECK_CUDA(cudaMemcpy(fresult.data(), dresult, fresult.size() * sizeof(u64), cudaMemcpyDeviceToHost));

    // resize result
    vector<vector<vector<u64>>> hresult(m, vector<vector<u64>>(k, vector<u64>(nlenres, 0)));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < nlenres; ++l) {
                hresult[i][j][l] = fresult[l*m*k + i*k+j];
            }
        }
    }

    CHECK_CUDA(cudaFree(dfa));
    CHECK_CUDA(cudaFree(dfb));
    CHECK_CUDA(cudaFree(dresult));

    return hresult;

}

vector<vector<vector<u64>>> cupoly_mat_mul_fft3(const vector<vector<vector<u64>>>& a, const vector<vector<vector<u64>>>& b) {
    size_t m = a.size();
    size_t n = a[0].size();
    size_t k = b[0].size();
    if (n != b.size()) {
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }

    size_t nlena = a[0][0].size();
    size_t nlenb = b[0][0].size();
    size_t nlenres = nlena + nlenb - 1;

    vector<u64> fa(m*n*nlena, 0);
    vector<u64> fb(n*k*nlenb, 0);
    vector<u64> fresult(m*k*nlen_res, 0);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < nlena; ++l) {
                fa[l*m*n + i*n + j] = a[i][j][l];
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < nlenb; ++l) {
                fb[l*n*k + i*k + j] = b[i][j][l];
            }
        }
    }

    // auto start = high_resolution_clock::now();
    // matrix_ntt_parallel(fa, false);
    // matrix_ntt_parallel(fb, false);
    // auto durationntt = high_resolution_clock::now() - start;
    u64 *dfa, *dfb, *dresult; 
    CHECK_CUDA(cudaMalloc(&dfa, fa.size() * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&dfb, fb.size() * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&dresult, fresult.size() * sizeof(u64)));
    CHECK_CUDA(cudaMemcpy(dfa, fa.data(), fa.size() * sizeof(u64), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dfb, fb.data(), fb.size() * sizeof(u64), cudaMemcpyHostToDevice));

    cupoly_mat_mul_fft_gpu(dfa, dfb, dresult, m,n,k, nlena, nlenb, 0, nlenres); 
    CHECK_CUDA(cudaMemcpy(fresult.data(), dresult, fresult.size() * sizeof(u64), cudaMemcpyDeviceToHost));

    // resize result
    vector<vector<vector<u64>>> hresult(m, vector<vector<u64>>(k, vector<u64>(nlenres, 0)));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < nlenres; ++l) {
                hresult[i][j][l] = fresult[l*m*k + i*k+j];
            }
        }
    }

    CHECK_CUDA(cudaFree(dfa));
    CHECK_CUDA(cudaFree(dfb));
    CHECK_CUDA(cudaFree(dresult));

    return hresult;

}

void cupoly_mat_mul_fft_gpu(u64* da, u64* db, u64* dresult, size_t m, size_t n, size_t k, size_t len_a, size_t len_b, size_t res_start_deg, size_t max_len_res) {
    
    size_t nlenres = len_a + len_b - 1;

    size_t nlenres_adj = 1;
    while (nlenres_adj < nlenres) {
        nlenres_adj <<= 1;
    }


    // auto start = high_resolution_clock::now();
    // matrix_ntt_parallel(fa, false);
    // matrix_ntt_parallel(fb, false);
    // auto durationntt = high_resolution_clock::now() - start;
    u64 *dfa, *dfb, *dfresult; 
    CHECK_CUDA(cudaMalloc(&dfa, m * n * nlenres_adj * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&dfb, n * k * nlenres_adj * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&dfresult, m * k * nlenres_adj * sizeof(u64)));
    // copy from cuda buffer da to dfa
    CHECK_CUDA(cudaMemcpy(dfa, da, m * n * len_a * sizeof(u64), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(dfb, db, n * k * len_b * sizeof(u64), cudaMemcpyDeviceToDevice));

    ntt_cuda_colwise_gpu(dfa, nlenres_adj, m*n, false);
    ntt_cuda_colwise_gpu(dfb, nlenres_adj, n*k, false);


    // CHECK_CUDA(cudaMemset(dresult, 0, m*k*nlenres_adj * sizeof(u64)));
    cuda_vmatmul(dfa, dfb, dfresult, m, n, k, nlenres_adj);

    ntt_cuda_colwise_gpu(dfresult, nlenres_adj, m*k, true);

    size_t actual_len_res = min(max_len_res, nlenres-res_start_deg);

    CHECK_CUDA(cudaMemcpy(dresult, dfresult+m*k*res_start_deg, m * k * actual_len_res * sizeof(u64), cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaFree(dfa));
    CHECK_CUDA(cudaFree(dfb));
    CHECK_CUDA(cudaFree(dfresult));
}

template<typename T>
__global__ void modp_kernel(T *device_matrix, int matrix_size, T p) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < matrix_size) {
        device_matrix[index] = device_matrix[index] % p;
    }
}

template<typename T>
inline void modp_buffer(T*d_data, size_t size, T prime) {
    modp_kernel<<<((size + 255) / 256), 256>>>(d_data, size, prime);
    CHECK_CUDA(cudaGetLastError());
}


vector<vector<vector<u64>>> cupoly_mat_mul_fft1b(const vector<vector<vector<u64>>>& a, const vector<vector<vector<u64>>>& b) {
    size_t m = a.size();
    size_t n = a[0].size();
    size_t k = b[0].size();
    if (n != b.size()) {
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }

    size_t nlena = a[0][0].size();
    size_t nlenb = b[0][0].size();
    size_t nlenres = nlena + nlenb - 1;

    size_t nlenres_adj = 1;
    while (nlenres_adj < nlenres) {
        nlenres_adj <<= 1;
    }

    vector<u64> fa(m*n*nlenres_adj, 0);
    vector<u64> fb(n*k*nlenres_adj, 0);
    vector<u64> fresult(m*k*nlenres_adj, 0);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < nlena; ++l) {
                fa[l*m*n + i*n + j] = a[i][j][l];
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < nlenb; ++l) {
                fb[l*n*k + i*k + j] = b[i][j][l];
            }
        }
    }

    // auto start = high_resolution_clock::now();
    // matrix_ntt_parallel(fa, false);
    // matrix_ntt_parallel(fb, false);
    // auto durationntt = high_resolution_clock::now() - start;
    u64 *dfa, *dfb, *dresult; 
    CHECK_CUDA(cudaMalloc(&dfa, fa.size() * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&dfb, fb.size() * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&dresult, fresult.size() * sizeof(u64)));
    CHECK_CUDA(cudaMemcpy(dfa, fa.data(), fa.size() * sizeof(u64), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dfb, fb.data(), fb.size() * sizeof(u64), cudaMemcpyHostToDevice));

    ntt_cuda_colwise_gpu(dfa, nlenres_adj, m*n, false);
    ntt_cuda_colwise_gpu(dfb, nlenres_adj, n*k, false);

    // copy back to fa, fb
    CHECK_CUDA(cudaMemcpy(fa.data(), dfa, fa.size() * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(fb.data(), dfb, fb.size() * sizeof(u64), cudaMemcpyDeviceToHost));

    // start = high_resolution_clock::now();
    // degree-wise matrix multiply
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t r = 0; r < n; ++r) {
                for (size_t l = 0; l < nlenres_adj; ++l) {
                    fresult[l*m*k + i*k+j] = mod_add(fresult[l*m*k + i*k+j],mod_mul(fa[l*m*n+i*n+r], fb[l*n*k+r*k+j]));
                }
            }
        }
    }
    CHECK_CUDA(cudaMemcpy(dresult, fresult.data(), fresult.size() * sizeof(u64), cudaMemcpyHostToDevice));
    // cuda_vmatmul(dfa, dfb, dresult, m, n, k, nlenres_adj);

    // start = high_resolution_clock::now();
    // matrix_ntt_parallel(fresult, true);
    // durationntt += high_resolution_clock::now() - start;
    ntt_cuda_colwise_gpu(dresult, nlenres_adj, m*k, true);
    CHECK_CUDA(cudaMemcpy(fresult.data(), dresult, fresult.size() * sizeof(u64), cudaMemcpyDeviceToHost));

    // resize result
    vector<vector<vector<u64>>> hresult(m, vector<vector<u64>>(k, vector<u64>(nlenres, 0)));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < nlenres; ++l) {
                hresult[i][j][l] = fresult[l*m*k + i*k+j];
            }
        }
    }
    
    CHECK_CUDA(cudaFree(dfa));
    CHECK_CUDA(cudaFree(dfb));
    CHECK_CUDA(cudaFree(dresult));

    return hresult;

}

//************** TESTS *********************/


void test_cuda_poly_mul_methods() {
    // Generate two random polynomials
    size_t degree1 = 40; // Example degree
    size_t degree2 = 100; // Example degree
    std::vector<u64> poly1(degree1);
    std::vector<u64> poly2(degree2);

    // Fill polynomials with random coefficients
    for (size_t i = 0; i < degree1; ++i) {
        poly1[i] = rand() % 1000; // Random coefficients in range [0, 999]
    }
    for (size_t i = 0; i < degree2; ++i) {
        poly2[i] = rand() % 1000;
    }

    // Compute results using both methods
    auto result_cuda = poly_mul_fft_cuda(poly1, poly2);
    // auto result_cpu = poly_mul_fft(poly1, poly2);
    auto result_cpu = poly_mul_naive(poly1, poly2, PRIME<u64>());

    // Compare results
    assert(result_cuda.size() == result_cpu.size());
    for (size_t i = 0; i < result_cuda.size(); ++i) {
        if (result_cuda[i] != result_cpu[i]) {
            std::cerr << "Mismatch at index " << i << ": cuda=" << result_cuda[i]
                      << ", cpu=" << result_cpu[i] << std::endl;
            assert(false);
        }
    }
    std::cout << "Test passed: poly_mul_cuda and poly_mul_cpu agree." << std::endl;
}

void test_cuda_matrix_poly_mul_methods() {
    // Generate random 3D matrices of polynomials
    size_t m = 3; // Rows of matrix A
    size_t n = 4; // Columns of matrix A / Rows of matrix B
    size_t k = 2; // Columns of matrix B
    size_t poly_degree_a = 10; // Degree of polynomials in matrix A
    size_t poly_degree_b = 15; // Degree of polynomials in matrix B

    vector<vector<vector<u64>>> matrix_a(m, vector<vector<u64>>(n, vector<u64>(poly_degree_a)));
    vector<vector<vector<u64>>> matrix_b(n, vector<vector<u64>>(k, vector<u64>(poly_degree_b)));

    // Fill matrices with random polynomial coefficients
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < poly_degree_a; ++l) {
                matrix_a[i][j][l] = rand() % 1000; // Random coefficients in range [0, 999]
            }
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < poly_degree_b; ++l) {
                matrix_b[i][j][l] = rand() % 1000;
            }
        }
    }

    // Compute results using both methods
    // auto result_cuda = cupoly_mat_mul_fft1(matrix_a, matrix_b);
    auto result_cuda = cupoly_mat_mul_fft3(matrix_a, matrix_b);
    auto result_cpu = poly_mat_mul_fft(matrix_a, matrix_b, 0, poly_degree_b);

    // Compare results
    assert(result_cuda.size() == result_cpu.size());
    for (size_t i = 0; i < result_cuda.size(); ++i) {
        assert(result_cuda[i].size() == result_cpu[i].size());
        for (size_t j = 0; j < result_cuda[i].size(); ++j) {
            assert(result_cuda[i][j].size() == result_cpu[i][j].size());
            for (size_t l = 0; l < result_cuda[i][j].size(); ++l) {
                if (result_cuda[i][j][l] != result_cpu[i][j][l]) {
                    std::cerr << "Mismatch at matrix[" << i << "][" << j << "][" << l
                              << "]: cuda=" << result_cuda[i][j][l]
                              << ", cpu=" << result_cpu[i][j][l] << std::endl;
                    assert(false);
                }
            }
        }
    }
    std::cout << "Test passed: cupoly_mat_mul_fft1 and poly_mat_mul_fft agree." << std::endl;
}


void test_cuda_vmat_mul() {
    // Generate random 3D matrices of polynomials
    size_t m = 2; // Rows of matrix A
    size_t n = 2; // Columns of matrix A / Rows of matrix B
    size_t k = 2; // Columns of matrix B
    size_t poly_degree = 3;

    vector<u64> matrix_a(m * n * poly_degree, 0);
    vector<u64> matrix_b(n * k * poly_degree, 0);
    vector<u64> matrix_c(m * k * poly_degree, 0);
    vector<u64> matrix_d(m * k * poly_degree, 0);

    // Fill matrices with random polynomial coefficients
    for (int i=0;i<matrix_a.size();i++) {
        matrix_a[i] = rand() % 5; // Random coefficients in range [0, 999]
    }
    for (int i=0;i<matrix_b.size();i++) {
        matrix_b[i] = rand() % 5;
    }
    u64 *fa, *fb, *fc;
    CHECK_CUDA(cudaMalloc(&fa, matrix_a.size() * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&fb, matrix_b.size() * sizeof(u64)));
    CHECK_CUDA(cudaMalloc(&fc, matrix_c.size() * sizeof(u64)));
    CHECK_CUDA(cudaMemcpy(fa, matrix_a.data(), matrix_a.size() * sizeof(u64), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(fb, matrix_b.data(), matrix_b.size() * sizeof(u64), cudaMemcpyHostToDevice));
    cuda_vmatmul(fa, fb, fc, m, n, k, poly_degree);
    CHECK_CUDA(cudaMemcpy(matrix_c.data(), fc, matrix_c.size() * sizeof(u64), cudaMemcpyDeviceToHost));

    // manual multiply
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t r = 0; r < n; ++r) {
                for (size_t l = 0; l < poly_degree; ++l) {
                    matrix_d[l*m*k + i*k+j] = mod_add(matrix_d[l*m*k + i*k+j],mod_mul(matrix_a[l*m*n+i*n+r], matrix_b[l*n*k+r*k+j]));
                }
            }
        }
    }
    // display results
    for (int dg=0; dg<poly_degree;dg++) {
        std::cout << "Degree: " << dg << std::endl; 
        std::cout << "Matrix A: " << std::endl;
        for (int i=0;i<m;i++) {
            for (int j=0;j<n;j++) {
                cout << matrix_a[dg*m*n + i*n+j] << " ";
            }
            cout << endl;
        }
        std::cout << "Matrix B: " << std::endl;
        for (int i=0;i<n;i++) {
            for (int j=0;j<k;j++) {
                cout << matrix_b[dg*n*k + i*k+j] << " ";
            }
            cout << endl;
        }
        std::cout << "Matrix C (CUDA): " << std::endl;
        for (int i=0;i<m;i++) {
            for (int j=0;j<k;j++) {
                cout << matrix_c[dg*m*k + i*k+j] << " ";
            }
            cout << endl;
        }
        std::cout << "Matrix C (manual): " << std::endl;
        for (int i=0;i<m;i++) {
            for (int j=0;j<k;j++) {
                cout << matrix_d[dg*m*k + i*k+j] << " ";
            }
            cout << endl;
        }
    }

}

#endif // CUPOLY_MAT_MUL_H