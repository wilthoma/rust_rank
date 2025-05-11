/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>  // Include fstream for file input
#include <chrono>
// #include "cublas_utils.h"
#include <random>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include "include/CLI11.hpp"
#include "matrices.h"
#include "cuda_helpers.h"


const int THESMALLPRIME = 3323;



#define USE_TIC 0 // turn on/off the timing

typedef int32_t myfloat;


using namespace std;

int mod_inv(int a, myfloat mod) {
    int b = mod, u = 1, v = 0;
    while (b) {
        int t = a / b;
        a -= t * b; swap(a, b);
        u -= t * v; swap(u, v);
    }
    return (u % mod + mod) % mod;
}

vector<myfloat> berlekamp_massey(const vector<myfloat>& s, myfloat mod) {
    vector<myfloat> C = {1}, B = {1};
    int L = 0, m = 1, b = 1;

    for (int n = 0; n < (int)s.size(); n++) {
        int d = s[n];
        for (int i = 1; i <= L; i++)
            d = (d + 1LL * C[i] * s[n - i]) % mod;
        if (d == 0) {
            m++;
            continue;
        }
        vector<myfloat> T = C;
        int coef = 1LL * d * mod_inv(b, mod) % mod;
        if ((int)C.size() < (int)(B.size() + m))
            C.resize(B.size() + m);
        for (int i = 0; i < (int)B.size(); i++)
            C[i + m] = (C[i + m] - 1LL * coef * B[i] % mod + mod) % mod;
        if (2 * L <= n) {
            L = n + 1 - L;
            B = T;
            b = d;
            m = 1;
        } else {
            m++;
        }
    }
    return C;
}




// // Define your function here (e.g., increment each element)
// __device__ myfloat my_function(myfloat input) {
//     // return __int2double_rn(__double2int_rn(input) % THESMALLPRIME);
//     #if USE_DOUBLE
//         return fmod(input, 3323.0);
//     #else
//         //return fmod(input, 3323.0f);
//         return input % THESMALLPRIME;
//     #endif
// }
  
// // CUDA kernel to apply the function
// __global__ void apply_function_kernel(myfloat *device_matrix, int matrix_size) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < matrix_size) {
//         device_matrix[index] = my_function(device_matrix[index]);
//     }
// }
// __global__ void apply_function_kernel_offset(myfloat *device_matrix, int offset, int matrix_size) {
//     int index = blockIdx.x * blockDim.x + threadIdx.x;
//     if (index < matrix_size) {
//         device_matrix[index+offset] = my_function(device_matrix[index+offset]);
//     }
// }

auto tic_start_time= std::chrono::high_resolution_clock::now();


inline void tic() {
    #if USE_TIC
    // Start the timer
    cudaDeviceSynchronize();
    tic_start_time = std::chrono::high_resolution_clock::now();
    #endif
}
inline void toc(const std::string& msg = "") {
    #if USE_TIC
    // Stop the timer
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time in milliseconds
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - tic_start_time).count();
    std::cout << "Elapsed time: " << elapsed_time << " ms. " << msg << std::endl;
    #endif
}

// silent versions without cuda synchronize
auto stic_start_time= std::chrono::high_resolution_clock::now();
inline void stic() 
{
    // Start the timer
    stic_start_time = std::chrono::high_resolution_clock::now();
}
inline long long stoc() 
{
    // Stop the timer
    auto end_time = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time in milliseconds
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - stic_start_time).count();
    return elapsed_time;
}



// __global__ void csr_spmm_naive(
//     int M, int N,
//     const int *__restrict__ csr_row_ptr,
//     const int *__restrict__ csr_col_idx,
//     const myfloat *__restrict__ csr_val,
//     const myfloat *__restrict__ B, // dense matrix B [K x N]
//     myfloat *__restrict__ C        // output matrix C [M x N]
// ) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row >= M) return;

//     // Initialize output row
//     for (int j = 0; j < N; j++) {
//         C[row * N + j] = 0.0;
//     }

//     // Iterate over non-zeros in row
//     int row_start = csr_row_ptr[row];
//     int row_end = csr_row_ptr[row + 1];
//     for (int idx = row_start; idx < row_end; idx++) {
//         int col = csr_col_idx[idx];
//         myfloat val = csr_val[idx];

//         for (int j = 0; j < N; j++) {
//             C[row * N + j] += val * B[col * N + j];
//         }
//     }
// }

// __global__ void csr_spmm_2d(
//     int M, int N,
//     const int *__restrict__ csr_row_ptr,
//     const int *__restrict__ csr_col_idx,
//     const myfloat *__restrict__ csr_val,
//     const myfloat *__restrict__ B, // [K x N]
//     myfloat *__restrict__ C        // [M x N]
// ) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     int col = blockIdx.y * blockDim.y + threadIdx.y;

//     if (row >= M || col >= N) return;

//     myfloat sum = 0.0;
//     int start = csr_row_ptr[row];
//     int end = csr_row_ptr[row + 1];

//     for (int idx = start; idx < end; idx++) {
//         int k = csr_col_idx[idx];
//         myfloat a = csr_val[idx];
//         myfloat b = B[k * N + col]; // column-major access of B
//         sum += a * b;
//     }

//     C[row * N + col] = sum;
// }

// __global__ void dense_gemm_TN_chunked3D(int n, int k,  // n = n_dense_vectors, k = n_veclen
//     const myfloat* __restrict__ A,  // Transposed: A^T [n x k]
//     const myfloat* __restrict__ B,  // B [k x n]
//     myfloat* C,                    // Output: C [n x n]
//     int chunk_size)
// {
//     int chunk = blockIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.z * blockDim.x + threadIdx.x;

//     int chunk_start = chunk * chunk_size;
//     int chunk_end = min(chunk_start + chunk_size, k);

//     if (row < n && col < n) {
//         myfloat acc = 0;
//         for (int i = chunk_start; i < chunk_end; ++i) {
//             acc += A[i * n + row] * B[i * n + col];  
//             // acc += A[i * n + row] * B[i + col * k];  // A is row-major transposed
//         }

//         // Accumulate the result into global memory (C[row + col * n])
//         atomicAdd(&C[row + col * n], acc % THESMALLPRIME);
//     }
// }

// __global__ void dense_gemm_TN_chunked3D_offset(int n, int k,  // n = n_dense_vectors, k = n_veclen
//     const myfloat* __restrict__ A,  // Transposed: A^T [n x k]
//     const myfloat* __restrict__ B,  // B [k x n]
//     myfloat* C,                    // Output: [n x n] slice of C at position offset
//     int offset, // offset in C, in units of myfloat
//     int chunk_size)
// {
//     int chunk = blockIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.z * blockDim.x + threadIdx.x;

//     // TODO : check it is the correct order (not <)
//     // we only need to compute half the matrix
//     // We waste a bit of buffer, though
//     if (row > col) {
//         return;
//     }

//     int chunk_start = chunk * chunk_size;
//     int chunk_end = min(chunk_start + chunk_size, k);

//     if (row < n && col < n) {
//         myfloat acc = 0;
//         for (int i = chunk_start; i < chunk_end; ++i) {
//             acc += A[i * n + row] * B[i * n + col];  
//             // acc += A[i * n + row] * B[i + col * k];  // A is row-major transposed
//         }

//         // Accumulate the result into global memory (C[row + col * n])
//         atomicAdd(&C[row + col * n + offset], acc % THESMALLPRIME);
//     }
// }

// static std::vector<std::vector<myfloat>> hSp_list;

// int compute_and_push_sp(myfloat* dM1, myfloat* dM2, myfloat* dSp, int n_dense_vectors, int n_veclen) {
//     int Sp_size = n_dense_vectors * n_dense_vectors;
//     std::vector<myfloat> hSp(Sp_size);

//     cudaMemset(dSp, 0, n_dense_vectors * n_dense_vectors * sizeof(myfloat));

//     int num_chunks = (n_veclen + DOT_CHUNK_SIZE - 1) / DOT_CHUNK_SIZE;
//     dim3 blockDim(16, 16);  // threads per block
//     dim3 gridDim(num_chunks,
//                  (n_dense_vectors + blockDim.y - 1) / blockDim.y,
//                  (n_dense_vectors + blockDim.x - 1) / blockDim.x);
    
//     dense_gemm_TN_chunked3D<<<gridDim, blockDim>>>(
//         n_dense_vectors, n_veclen, dM1, dM2, dSp, DOT_CHUNK_SIZE
//     );

//         apply_function_kernel<<<((Sp_size + 255) / 256), 256>>>(dSp, Sp_size);

//         // Copy the device buffer dSp to a local host buffer
//         CHECK_CUDA(cudaMemcpy(hSp.data(), dSp, Sp_size * sizeof(myfloat), cudaMemcpyDeviceToHost));
    
//         // print the first 10 entries of the result
//         // std::cout << "dSp (first 10 entries): ";
//         // for (int i = 0; i < 10 && i < Sp_size; ++i) {
//         //     std::cout << hSp[i] << " ";
//         // }
//         // std::cout << std::endl;
        
//         hSp_list.push_back(hSp);

//         return 0;
// }

// int compute_and_push_bigsp(myfloat* dM1, myfloat* dM2, myfloat* dBigSp, int n_dense_vectors, int n_veclen, int &seq_position) {
//     int Sp_size = n_dense_vectors * n_dense_vectors;
//     //std::vector<myfloat> hSp(Sp_size);

//     //cudaMemset(dSp, 0, n_dense_vectors * n_dense_vectors * sizeof(myfloat));

//     int num_chunks = (n_veclen + DOT_CHUNK_SIZE - 1) / DOT_CHUNK_SIZE;
//     dim3 blockDim(16, 16);  // threads per block
//     dim3 gridDim(num_chunks,
//                  (n_dense_vectors + blockDim.y - 1) / blockDim.y,
//                  (n_dense_vectors + blockDim.x - 1) / blockDim.x);
    
//     int offset = seq_position * Sp_size; // offset in units of myfloat
//     dense_gemm_TN_chunked3D_offset<<<gridDim, blockDim>>>(
//         n_dense_vectors, n_veclen, dM1, dM2, dBigSp, offset, DOT_CHUNK_SIZE
//     );

//     apply_function_kernel_offset<<<((Sp_size + 255) / 256), 256>>>(dBigSp, offset, Sp_size);
//     seq_position++;

//     // Copy the device buffer dSp to a local host buffer
//     //CHECK_CUDA(cudaMemcpy(hSp.data(), dSp, Sp_size * sizeof(myfloat), cudaMemcpyDeviceToHost));

//     // print the first 10 entries of the result
//     // std::cout << "dSp (first 10 entries): ";
//     // for (int i = 0; i < 10 && i < Sp_size; ++i) {
//     //     std::cout << hSp[i] << " ";
//     // }
//     // std::cout << std::endl;
    
//     //hSp_list.push_back(hSp);

//     return 0;
// }

void compute_and_push_bigsp2(CudaDenseMatrix<myfloat> &B, CudaDenseMatrix<myfloat> &C, myfloat* dBigSp, int &seq_position, myfloat prime) {
    B.mTm_tri(C, dBigSp, seq_position, prime);
    seq_position++;
}






// void test_singlemat_product(CsrMatrix &matA, myfloat *dB)
// {
//     // take a product with a vector of ones and see the result
//     std::vector<myfloat> vecA(matA.numCols, 1);
//     // 

// }



void display_cuda_buffer(myfloat* d_buffer, int size, int max_elements = 10) {
    std::vector<myfloat> h_buffer(size);
    cudaMemcpy(h_buffer.data(), d_buffer, size * sizeof(myfloat), cudaMemcpyDeviceToHost);
    display_vector(h_buffer, max_elements);
}


template <typename T>
void save_wdm_file_sym(
    const std::string& wdm_filename,
    size_t n_rows,
    size_t n_cols,
    T theprime,
    const std::vector<T>& row_precond,
    const std::vector<T>& col_precond,
    const std::vector<std::vector<T>>& v_list,
    const std::vector<std::vector<T>>& curv_list,
    const std::vector<std::vector<T>>& seq_list
) {
    std::ofstream file(wdm_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + wdm_filename);
    }

    // Write the first line: m n p Nlen num_u num_v
    file << n_rows << " " << n_cols << " " << theprime << " " 
         << seq_list[0].size() << " " << v_list.size() << "\n";

    // Write the second line: row_precond
    for (size_t i = 0; i < row_precond.size(); ++i) {
        if (i > 0) file << " ";
        file << row_precond[i];
    }
    file << "\n";

    // Write the third line: col_precond
    for (size_t i = 0; i < col_precond.size(); ++i) {
        if (i > 0) file << " ";
        file << col_precond[i];
    }
    file << "\n";

    // Write the v_list
    for (const auto& vv : v_list) {
        auto v = prettify_vect(vv, theprime);
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) file << " ";
            file << v[i];
        }
        file << "\n";
    }

    // Write the curv_list
    for (const auto& curvv : curv_list) {
        auto curv = prettify_vect(curvv, theprime);
        for (size_t i = 0; i < curv.size(); ++i) {
            if (i > 0) file << " ";
            file << curv[i];
        }
        file << "\n";
    }

    // Write the seq_list
    for (const auto& seq : seq_list) {
        auto seq_pretty = prettify_vect(seq, theprime);
        for (size_t i = 0; i < seq_pretty.size(); ++i) {
            if (i > 0) file << " ";
            file << seq_pretty[i];
        }
        file << "\n";
    }

    file.close();
    if (!file) {
        throw std::runtime_error("Failed to write to file: " + wdm_filename);
    }
}

int save_all_data(
    const std::string& filename,
    int n_rows,
    int n_cols,
    int n_dense_cols,
    myfloat theprime,
    const std::vector<myfloat>& row_precond,
    const std::vector<myfloat>& col_precond,
    const std::vector<myfloat>& initial_B,
    const myfloat* dB,
    const std::vector<myfloat>& hBigSp
    //const std::vector<std::vector<myfloat>>& sp_list
) 
{

    std::cout << "Saving data to " << filename << "..." << std::endl;

    // read data from cuda buffer
    std::vector<myfloat> hB(n_cols * n_dense_cols);
    CHECK_CUDA(cudaMemcpy(hB.data(), dB, n_cols * n_dense_cols * sizeof(myfloat), cudaMemcpyDeviceToHost));
    
    std::cout << "A" << std::endl;

    // translate into vector of vectors
    std::vector<std::vector<myfloat>> cur_B = reshape_to_vector_of_vectors(hB, n_cols);
    std::cout << "A" << std::endl;
    std::vector<std::vector<myfloat>> ini_B = reshape_to_vector_of_vectors(initial_B, n_cols);
    std::cout << "A" << std::endl;
    std::vector<std::vector<myfloat>> sp_list = reshape_to_vector_of_vectors(hBigSp, n_dense_cols*n_dense_cols);
    // select upper triangular part of sp_list
    int nlen = sp_list.size(); 
    std::vector<std::vector<myfloat>> sp_list_upper;
    for (int i = 0; i < n_dense_cols; ++i) {
        for (int j = i; j < n_dense_cols; ++j) {
            std::vector<myfloat> sp_row(nlen);
            for (int k = 0; k < nlen; ++k) {
                sp_row[k] = sp_list[k][i * n_dense_cols + j];
            }
            sp_list_upper.push_back(sp_row);
        }
    }
    std::cout << "A" << std::endl;

    save_wdm_file_sym(
        filename,
        n_rows,
        n_cols,
        theprime,
        row_precond,
        col_precond,
        ini_B,
        cur_B,
        sp_list_upper
    );
    std::cout << "Data saved to " << filename << std::endl;

    return 0;
}


template <typename T>
std::tuple<uint32_t, size_t, size_t, size_t> load_wdm_file_sym(
    const std::string& wdm_filename,
    std::vector<T>& row_precond,
    std::vector<T>& col_precond,
    std::vector<std::vector<T>>& v_list,
    std::vector<std::vector<T>>& curv_list,
    std::vector<std::vector<T>>& seq_list
) {
    std::ifstream file(wdm_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + wdm_filename);
    }

    // Read the first line: m n p Nlen num_v
    size_t m, n, nlen, num_v;
    uint32_t p;
    file >> m >> n >> p >> nlen >> num_v;
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Read row_precond
    row_precond.resize(m);
    for (size_t i = 0; i < m; ++i) {
        file >> row_precond[i];
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Read col_precond
    col_precond.resize(n);
    for (size_t i = 0; i < n; ++i) {
        file >> col_precond[i];
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Read v_list
    v_list.clear();
    for (size_t i = 0; i < num_v; ++i) {
        std::vector<T> v(n);
        for (size_t j = 0; j < n; ++j) {
            file >> v[j];
        }
        v_list.push_back(std::move(v));
    }

    // Read curv_list
    curv_list.clear();
    for (size_t i = 0; i < num_v; ++i) {
        std::vector<T> curv(n);
        for (size_t j = 0; j < n; ++j) {
            file >> curv[j];
        }
        curv_list.push_back(std::move(curv));
    }

    // Read seq_list
    seq_list.clear();
    size_t seq_count = num_v * (num_v + 1) / 2;
    for (size_t i = 0; i < seq_count; ++i) {
        std::vector<T> seq(nlen);
        for (size_t j = 0; j < nlen; ++j) {
            file >> seq[j];
        }
        seq_list.push_back(std::move(seq));
    }

    // Ensure all vectors are of the correct size
    if (row_precond.size() != m) {
        throw std::runtime_error("Row preconditioner length does not match matrix rows");
    }
    if (col_precond.size() != n) {
        throw std::runtime_error("Column preconditioner length does not match matrix columns");
    }

    return std::make_tuple(p, m, n, num_v);
}

void report_progress(
    const long long elapsed,
    long long& last_report,
    int & last_nlen,
    const int nlen,
    const int max_nlen,
    const int num_v,
    const std::string& suffix = ""
) {

    double speed = static_cast<double>(nlen - last_nlen) / (elapsed - last_report) / 1000;
    double remaining = static_cast<double>(max_nlen - nlen) / speed / 1000;

    std::cout << "\rProgress: " << nlen << "/" << max_nlen
              << " | Elapsed: " << (elapsed/1000) << "s"
              << " | Throughput: " << speed << "/s (total " << speed * num_v << "/s)"
              << " | Remaining: " << remaining << "s"
              << " | " << suffix << "           " << std::flush;

    last_nlen = nlen;
    last_report = elapsed;
}

int main(int argc, char* argv[]) {
    CLI::App app{"Wiedemann sequence and rank computation"};

    std::string sms_filename, wdm_filename;
    bool overwrite = false;
    bool benchmark = false;
    bool transpose_matrix = false;

    size_t max_nlen = 0;
    size_t save_after = 200;
    size_t num_v = 1;
    myfloat pprime = 0;

    app.add_option("filename", sms_filename, "The SMS file containing the sparse matrix")
        ->required();
    app.add_option("-f", wdm_filename, "The WDM file for saving the result. If existing, progress will be loaded from there, unless -o is specified.")
        ->default_val("");
    app.add_flag("-o,--overwrite", overwrite, "Overwrite existing files");

    app.add_flag("--benchmark", benchmark, "Run matrix vector multiply benchmark, but no other computation.");
    app.add_flag("--transpose", transpose_matrix, "Transpose the matrix.");

    app.add_option("-N,--N", max_nlen, "The desired sequence length to be computed")
        ->default_val(0);

    app.add_option("-s,--saveafter", save_after, "Trigger automatic saves each s seconds.")
        ->default_val(200);

    app.add_option("-v", num_v, "The number of vectors v. The output will consist of symmetric matrices of size v x v.")
        ->default_val(1);

    app.add_option("-p,--prime", pprime, "The prime number to use for modular arithmetic")
        ->default_val(THESMALLPRIME);


    CLI11_PARSE(app, argc, argv);
    if (wdm_filename.empty()) {
        wdm_filename = sms_filename + ".wdm";
    }

    myfloat prime = pprime;

    // load matrix from file
    // if (argc < 5) {
    //     std::cerr << "Usage: " << argv[0] << " <matrix_file> <nr dense columns> <sequence length> <outfile>" << std::endl;
    //     return -1;
    // }
    //char* outfile = argv[4];
    //int seq_len = atoi(argv[3]);



    // std::vector<int> rowIndices, colIndices, csrOffsets, csrColumns, csrColumnsT, csrOffsetsT;
    // std::vector<myfloat> values, csrValues, csrValuesT;
    // int numRows, numCols, nnz;
    //auto loadStart = std::chrono::high_resolution_clock::now();
    stic();
    CooMatrix<myfloat> cooA = CooMatrix<myfloat>::from_sms_file(sms_filename, prime);
    //load_sms_matrix(argv[1], rowIndices, colIndices, values, numRows, numCols, nnz);
    ////auto loadStop = std::chrono::high_resolution_clock::now();
    // auto loadMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(loadStop - loadStart).count();
    std::cout << "Matrix loading runtime: " << stoc() << " ms" << std::endl;

    if (max_nlen == 0) {
        max_nlen = 2 * min(cooA.numRows, cooA.numCols)/num_v;
    }

    //auto convertStart = std::chrono::high_resolution_clock::now();
    stic();
    //coo_matrix_to_csr(numRows, rowIndices, colIndices, values, csrOffsets, csrColumns, csrValues);
    CsrMatrix<myfloat> A = cooA.to_csr();
    //auto convertStop = std::chrono::high_resolution_clock::now();
    //auto convertMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(convertStop - convertStart).count();
    std::cout << "COO to CSR conversion runtime: " << stoc() << " ms" << std::endl;

    std::cout << A.numRows <<"x" << A.numCols << " matrix loaded from file: " << argv[1] << " with nnz=" << A.values.size() << std::endl;

    if (A.is_csr_valid()) {
        std::cout << "CSR matrix is valid." << std::endl;
    } else {
        std::cerr << "CSR matrix is invalid." << std::endl;
        return -1;
    }

    CsrMatrix<myfloat> AT = A.transpose();

    // transpose_csr_matrix(csrOffsets, csrColumns, csrValues, numRows, numCols, csrOffsetsT, csrColumnsT, csrValuesT);
    std::cout << "CSR matrix transposed." << std::endl;
    std::cout << "Transposed matrix size: " << AT.numRows << "x" << AT.numCols << std::endl;
    std::cout << "Transposed matrix nnz: " << AT.values.size() << std::endl;


    if (AT.is_csr_valid()) {
        std::cout << "Transposed CSR matrix is valid." << std::endl;
    } else {
        std::cerr << "Transposed CSR matrix is invalid." << std::endl;
        return -1;
    }

    // Rescale the csr matrices
    std::vector<myfloat> scale_factors_rows = generate_random_vector(A.numRows, prime, true);
    std::vector<myfloat> scale_factors_cols = generate_random_vector(A.numCols, prime, true);
    A.csr_rowrescale(scale_factors_rows, prime);
    A.csr_columnrescale(scale_factors_cols, prime);
    AT.csr_rowrescale(scale_factors_cols, prime);

    // A.csr_rowrescale(numRows, numCols, csrOffsets, csrColumns, csrValues, scale_factors_rows, THESMALLPRIME);
    // csr_columnrescale(numRows, numCols, csrOffsets, csrColumns, csrValues, scale_factors_cols, THESMALLPRIME);
    // csr_rowrescale(numCols, numRows, csrOffsetsT, csrColumnsT, csrValuesT, scale_factors_cols, THESMALLPRIME);

    // std::cout<< "A";

    // Random dense matrix for multiplication
    //int denseCols = atoi(argv[2]);  // Example: Result matrix column size
    int denseCols = num_v; 
    // std::cout<< "A" << denseCols << " " <<numCols << std::endl; 
    std::vector<myfloat> h_dense = generate_random_vector(A.numCols * denseCols, prime);
    // std::vector<myfloat> h_dense(numCols * denseCols, 1);


    //std::vector<myfloat> c_dense(numRows * denseCols, 0);
    // for (int i = 0; i < numRows * denseCols; ++i) {
    //     c_dense[i] = 0; //static_cast<myfloat>(rand()) / RAND_MAX;  // Random initialization
    // }

    // std::cout<< "A";

    // std::vector<myfloat> c_result(numRows * denseCols);

    // int   A_num_rows      = numRows;
    // int   A_num_cols      = numCols;
    // int   A_nnz           = nnz;
    // int   B_num_rows      = A_num_cols;
    // int   B_num_cols      = denseCols;
    // int   ldb             = B_num_rows;
    // int   ldc             = A_num_rows;
    // // int ldd = A_num_rows;
    // int   B_size          = ldb * B_num_cols;
    // int   C_size          = ldc * B_num_cols;
    // int* hA_csrOffsets = &csrOffsets[0];
    // // std::copy(csrOffsets.begin(), csrOffsets.end(), hA_csrOffsets);
    // int*   hA_columns    = &csrColumns[0]; 
    // myfloat* hA_values     = &csrValues[0]; 
    // myfloat* hB            = &h_dense[0];
    // myfloat* hC            = &c_dense[0]; 
    // myfloat* hC_result     = &c_result[0]; 



    // std::cout<< "A";
    //--------------------------------------------------------------------------
    // Device memory management
    // int   *dA_csrOffsets, *dA_columns, *dA_csrOffsetsT, *dA_columnsT;
    // myfloat *dA_values, *dA_valuesT, *dB, *dC;
    // CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,(A_num_rows + 1) * sizeof(int)) )
    // CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    // CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(myfloat))  )
    // CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsetsT,(A_num_cols + 1) * sizeof(int)) )
    // CHECK_CUDA( cudaMalloc((void**) &dA_columnsT, A_nnz * sizeof(int))    )
    // CHECK_CUDA( cudaMalloc((void**) &dA_valuesT,  A_nnz * sizeof(myfloat))  )
    // CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(myfloat)) )
    // CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(myfloat)) )
    // // std::cout<< "B";
    // CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
    //                        (A_num_rows + 1) * sizeof(int),
    //                        cudaMemcpyHostToDevice) )
    // CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
    //                        cudaMemcpyHostToDevice) )
    // CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(myfloat),
    //                        cudaMemcpyHostToDevice) )
    // CHECK_CUDA( cudaMemcpy(dA_csrOffsetsT, &csrOffsetsT[0],
    //                         (A_num_cols + 1) * sizeof(int),
    //                         cudaMemcpyHostToDevice) )
    //  CHECK_CUDA( cudaMemcpy(dA_columnsT, &csrColumnsT[0], A_nnz * sizeof(int),
    //                         cudaMemcpyHostToDevice) )
    //  CHECK_CUDA( cudaMemcpy(dA_valuesT, &csrValuesT[0], A_nnz * sizeof(myfloat),
    //                         cudaMemcpyHostToDevice) )
    // CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(myfloat),
    //                        cudaMemcpyHostToDevice) )
    // CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(myfloat),
    //                        cudaMemcpyHostToDevice) )
                        //    std::cout<< "B";
    //--------------------------------------------------------------------------

    // size_t               bufferSize3 = 0;


    CudaCsrMatrix<myfloat> cuA = CudaCsrMatrix<myfloat>::from_host(A);
    CudaCsrMatrix<myfloat> cuAT = CudaCsrMatrix<myfloat>::from_host(AT);
    CudaDenseMatrix<myfloat> cuB = CudaDenseMatrix<myfloat>::from_host(h_dense, A.numCols, denseCols);
    CudaDenseMatrix<myfloat> cuC = CudaDenseMatrix<myfloat>::allocate(A.numRows, denseCols);
    CudaDenseMatrix<myfloat> cuD = CudaDenseMatrix<myfloat>::allocate(A.numCols, denseCols);

    // cudaEvent_t start, stop;


    
    // Create dense matrix D for the result of the second multiplication
    // myfloat *dD;
    // int D_size = A_num_cols * B_num_cols;
    // CHECK_CUDA(cudaMalloc((void**)&dD, D_size * sizeof(myfloat)));
    // CHECK_CUDA(cudaMemset(dD, 0, D_size * sizeof(myfloat)));
    // int ldd = A_num_cols; // Leading dimension of D

    

    myfloat *dSp, *dBigSp;
    int Sp_size = num_v * num_v;
    CHECK_CUDA(cudaMalloc((void**)&dSp, Sp_size * sizeof(myfloat)));
    CHECK_CUDA(cudaMemset(dSp, 0, Sp_size * sizeof(myfloat)));
    // int ldsp = B_num_cols; // Leading dimension of D
    // buffer for holding the whole sequence
    int bigSp_len = max_nlen+5; // +5 to be safe
    int bigSp_size = Sp_size * bigSp_len;
    CHECK_CUDA(cudaMalloc((void**)&dBigSp, bigSp_size * sizeof(myfloat)));
    CHECK_CUDA(cudaMemset(dBigSp, 0, bigSp_size * sizeof(myfloat)));
    
    
    // CHECK_CUDA(cudaEventCreate(&start));
    // CHECK_CUDA(cudaEventCreate(&stop));
    // CHECK_CUDA(cudaEventRecord(start, 0));

    
    int seq_position = 0; // the current write position into the output sequence buffer dBigSp

    compute_and_push_bigsp2(cuB, cuB, dBigSp, seq_position, prime);
    // compute_and_push_sp(dB, dB, dSp, B_num_cols, A_num_cols);
    // dim3 blockDim(16, 32);
    // dim3 gridDim((A_num_rows + blockDim.x - 1) / blockDim.x,
    //             (B_num_cols + blockDim.y - 1) / blockDim.y);
    // dim3 blockDimT(16, 32);
    // dim3 gridDimT((A_num_cols + blockDimT.x - 1) / blockDimT.x,
    //             (B_num_cols + blockDimT.y - 1) / blockDimT.y);

    auto computationStart = std::chrono::high_resolution_clock::now();
    long long lastSave = 0;
    long long lastReport = 0;
    long long reportInterval = 1000;
    int last_nlen = 0;

    
    for (int round=0;round<max_nlen/4;round++){
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - computationStart).count();

        if (elapsed-lastSave > save_after*1000) {
            std::cout << "Saving data after " << elapsed/1000 << " s" << std::endl;
           //TODOOO  save_all_data(wdm_filename, A.numRows, A.numCols, denseCols, prime, scale_factors_rows, scale_factors_cols, h_dense, cuB.d_data, hBigSp);
            lastSave = elapsed; // reset the timer
        }
        if (elapsed-lastReport > reportInterval) {
            report_progress(
                elapsed,
                lastReport,
                last_nlen,
                seq_position,
                max_nlen,
                num_v
            );
        }



        // if (round%10==0){
        //     std::cout << "Round " << round << std::endl;
        // } 
        // std::cout << "Round " << round << std::endl;
        // execute SpMM, multiply by A to get C

        // tic();
        // int threads_per_block = 128;
        // int blocks_per_grid = (A_num_rows + threads_per_block - 1) / threads_per_block;
        // display_cuda_buffer(dB, B_size, 10);
        // csr_spmm_naive<<<blocks_per_grid, threads_per_block>>>(
        //     A_num_rows, B_num_cols, dA_csrOffsets, dA_columns, dA_values,
        //     dB, dC
        // );
        // display_cuda_buffer(dC, C_size, 10);
        // cudaError_t err = cudaGetLastError();
        // if (err != cudaSuccess)
        //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // toc("Handcrafted...:");
        tic();

        // std::cout << "gridDim = ("<< gridDim.x<<"x"<< gridDim.y;
        // std::cout << ") blockDim = ("<< blockDim.x<<"x"<< blockDim.y<<")" << std::endl;
        // csr_spmm_2d<<<gridDim, blockDim>>>(
        //     A_num_rows, B_num_cols, dA_csrOffsets, dA_columns, dA_values,
        //     dB, dC
        // );
        
        cuA.spmm(cuB, cuC, prime);
        // display_cuda_buffer(dC, C_size, 10);
        toc("Handcrafted 2d...:");
        tic();
        // apply_function_kernel<<<((C_size + 255) / 256), 256>>>(dC, C_size);
        // toc("apply_function_kernel C");
        
        // Execute SpMM for the second multiplication (matA^T * matC -> matD)
        // tic();
        // CHECK_CUSPARSE(cusparseSpMM(handle,
        //                             CUSPARSE_OPERATION_TRANSPOSE,
        //                             CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                             &alpha, matA, matC, &beta, matD, CUDA_FMT,
        //                             CUSPARSE_TRANS_ALGO, dBuffer2));
        // CHECK_CUSPARSE(cusparseSpMM(handle,
        //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                 &alpha, matAT, matC, &beta, matD, CUDA_FMT,
        //                                 CUSPARSE_TRANS_ALGO, dBuffer2));
        // csr_spmm_2d<<<gridDimT, blockDimT>>>(
        //     A_num_cols, B_num_cols, dA_csrOffsetsT, dA_columnsT, dA_valuesT,
        //     dC, dD
        // );
        cuAT.spmm(cuC, cuD, prime);
        toc("SpMM A^T*C->D");
        // display_cuda_buffer(dD, D_size, 10);
        // tic();
        // apply_function_kernel<<<((D_size + 255) / 256), 256>>>(dD, D_size);
        // toc("apply_function_kernel D");
        
        tic();
        // compute_and_push_bigsp(dB, dD, dSp, B_num_cols, A_num_cols, seq_position);
        // compute_and_push_bigsp(dD, dD, dSp, B_num_cols, A_num_cols, seq_position);
        // compute_and_push_sp(dB, dD, dSp, B_num_cols, A_num_cols);
        // compute_and_push_sp(dD, dD, dSp, B_num_cols, A_num_cols);
        compute_and_push_bigsp2(cuB, cuD, dBigSp, seq_position, prime);
        compute_and_push_bigsp2(cuD, cuD, dBigSp, seq_position, prime);
        toc("compute_and_push_sp D");

        // Next multiply by A to get C
        // CHECK_CUSPARSE( cusparseSpMM(handle,
        //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                                 &alpha, matA, matD, &beta, matC, CUDA_FMT,
        //                                 CUSPARSE_NORMAL_ALGO, dBuffer) )
        // csr_spmm_2d<<<gridDim, blockDim>>>(
        //     A_num_rows, B_num_cols, dA_csrOffsets, dA_columns, dA_values,
        //     dD, dC
        // );
        // apply_function_kernel<<<((C_size + 255) / 256), 256>>>(dC, C_size);
        cuA.spmm(cuD, cuC, prime);
        // and by A^T to get B
        // CHECK_CUSPARSE(cusparseSpMM(handle,
        //     CUSPARSE_OPERATION_TRANSPOSE,
        //     CUSPARSE_OPERATION_NON_TRANSPOSE,
        //     &alpha, matA, matC, &beta, matB, CUDA_FMT,
        //     CUSPARSE_TRANS_ALGO, dBuffer2));
        // CHECK_CUSPARSE(cusparseSpMM(handle,
        //         CUSPARSE_OPERATION_NON_TRANSPOSE,
        //         CUSPARSE_OPERATION_NON_TRANSPOSE,
        //         &alpha, matAT, matC, &beta, matB, CUDA_FMT,
        //         CUSPARSE_TRANS_ALGO, dBuffer2));
        // csr_spmm_2d<<<gridDimT, blockDimT>>>(
        //     A_num_cols, B_num_cols, dA_csrOffsetsT, dA_columnsT, dA_valuesT,
        //     dC, dB
        // );
        // apply_function_kernel<<<((B_size + 255) / 256), 256>>>(dB, B_size);
        cuAT.spmm(cuC, cuB, prime);

        // compute_and_push_bigsp(dB, dD, dBigSp, B_num_cols, A_num_cols, seq_position);
        // compute_and_push_bigsp(dB, dB, dBigSp, B_num_cols, A_num_cols, seq_position);                               
        // compute_and_push_sp(dB, dD, dSp, B_num_cols, A_num_cols);
        // compute_and_push_sp(dB, dB, dSp, B_num_cols, A_num_cols);
        compute_and_push_bigsp2(cuB, cuD, dBigSp, seq_position, prime);
        compute_and_push_bigsp2(cuB, cuB, dBigSp, seq_position, prime);  
        
    }

    // seq_position = hSp_list.size();

    // extract the sequence from dBigSp
    std::cout << "Extracting the sequence from dBigSp" << std::endl;
    int effectivesize = seq_position * Sp_size;
    std::vector<myfloat> hBigSp(effectivesize);
    CHECK_CUDA(cudaMemcpy(hBigSp.data(), dBigSp, effectivesize * sizeof(myfloat), cudaMemcpyDeviceToHost));
    std::cout << "Done.";
    // std::cout << "hSp (first 10 entries): ";
    

    CHECK_CUDA(cudaDeviceSynchronize());
    // destroy matrix/vector descriptors

    // CHECK_CUDA(cudaEventRecord(stop, 0));
    // CHECK_CUDA(cudaEventSynchronize(stop));

    // float milliseconds = 0;
    // CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    auto computationStop = std::chrono::high_resolution_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(computationStop - computationStart).count();
    std::cout << "SpMM operation runtime: " << milliseconds << " ms" << std::endl;
    std::cout << "Total throughput: " << seq_position * num_v * 1e3 / milliseconds  << "/s." << std::endl;

    // CHECK_CUDA(cudaEventDestroy(start));
    // CHECK_CUDA(cudaEventDestroy(stop));


    // reduce mod p 
    // Call the kernel
    // auto modstart = std::chrono::high_resolution_clock::now();
    // int matrix_size = C_size;
    // apply_function_kernel<<<((matrix_size + 255) / 256), 256>>>(dC, matrix_size);
    // auto modstop = std::chrono::high_resolution_clock::now();
    // auto modMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(modstop - modstart).count();
    // std::cout << "Kernel execution runtime (mod): " << modMilliseconds << " ms" << std::endl;

    //--------------------------------------------------------------------------
    // device result check
    // auto copyStart = std::chrono::high_resolution_clock::now();
    // CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(myfloat),
    //                        cudaMemcpyDeviceToHost) )
    // auto copyStop = std::chrono::high_resolution_clock::now();
    // auto copyMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(copyStop - copyStart).count();
    // std::cout << "Device to host copy runtime: " << copyMilliseconds << " ms" << std::endl;
    
    // int correct = 1;

    // check minimal polynomials 
    // for (int i = 0; i < B_num_cols; i++) {
    //     for (int j = 0; j < B_num_cols; j++) {
    //         if (hSp_list[i][j] != hSp_list[i][j]) {
    //             std::cout << "hSp_list[" << i << "][" << j << "] = " << hSp_list[i][j] << std::endl;
    //             correct = 0; // direct floating point comparison is not reliable
    //             break;
    //         }
    //     }
    // }

    save_all_data(
        wdm_filename,
        A.numRows,
        A.numCols,
        num_v,
        prime,
        scale_factors_rows,
        scale_factors_cols,
        h_dense,
        cuB.d_data,
        hBigSp
        //hSp_list
    );

    // int slen = hSp_list.size();
    // std::vector<myfloat> oneseq(slen,0);
    // for (int i = 0; i < 1; i++) {
    //     for (int j = i; j < 1; j++) {
    //         // collect the ij entries of hSp_list
    //         for (int k = 0; k < slen; k++) {
    //             oneseq[k] = hSp_list[k][i * B_num_cols + j];
    //         }
    //         std::cout << "oneseq: ";
    //         display_vector(oneseq, 10);
    //         // compute the minimal polynomial
    //         std::vector<myfloat> coeffs = berlekamp_massey(oneseq, THESMALLPRIME);
    //         std::cout << "Poly length: " << coeffs.size() << std::endl;
    //         std::cout << "Poly coeffs: ";
    //         display_vector(coeffs, 10);
    //     }
    // }
    // for (int i = 0; i < A_num_rows; i++) {
    //     for (int j = 0; j < B_num_cols; j++) {
    //         if (hC[i + j * ldc] != hC_result[i + j * ldc]) {
    //             correct = 0; // direct floating point comparison is not reliable
    //             break;
    //         }
    //     }
    // }
    // if (correct)
    //     printf("spmm_csr_example test PASSED\n");
    // else
    //     printf("spmm_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation

    cuA.release();
    cuAT.release();
    cuB.release();
    cuC.release();
    cuD.release();
    CHECK_CUDA(cudaFree(dBigSp));
    CHECK_CUDA(cudaFree(dSp));

    return 0;
}


