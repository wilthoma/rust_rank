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
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>  // Include fstream for file input
#include <chrono>
#include <cublas_v2.h>
// #include "cublas_utils.h"


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                   \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// const int THESMALLPRIME = 3323;

#define USE_DOUBLE 1
#if USE_DOUBLE
    typedef double myfloat;
    #define CUDA_FMT CUDA_R_64F
#else
    typedef float myfloat;
    #define CUDA_FMT CUDA_R_32F
#endif
// #define CUDA_FMT CUDA_R_64F
// #define CUSPARSE_TRANS_ALGO CUSPARSE_SPMM_CSR_ALG3
#define CUSPARSE_TRANS_ALGO CUSPARSE_SPMM_ALG_DEFAULT
#define CUSPARSE_NORMAL_ALGO CUSPARSE_SPMM_CSR_ALG1

void load_sms_matrix(const std::string& filename, std::vector<int>& rowIndices, std::vector<int>& colIndices, std::vector<myfloat>& values, int& numRows, int& numCols, int& nnz) {
    std::ifstream file(filename);  // Make sure to include <fstream>
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(-1);
    }

    char arbitraryChar;
    file >> numRows >> numCols >> arbitraryChar;

    std::vector<int> tempRowIndices;
    std::vector<int> tempColIndices;
    std::vector<myfloat> tempValues;

    int row, col;
    myfloat value;
    while (file >> row >> col >> value) {
        tempRowIndices.push_back(row);
        tempColIndices.push_back(col);
        tempValues.push_back(value);
    }

    nnz = tempRowIndices.size();
    rowIndices = std::move(tempRowIndices);
    colIndices = std::move(tempColIndices);
    values = std::move(tempValues);

    file.close();
}

void coo_matrix_to_csr(int numRows, const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<myfloat>& values,
                       std::vector<int>& csrOffsets, std::vector<int>& csrColumns, std::vector<myfloat>& csrValues) {
    csrOffsets.resize(numRows + 1, 0);
    csrColumns.resize(values.size());
    csrValues.resize(values.size());

    std::vector<int> rowCount(numRows, 0);
    for (int i = 0; i < rowIndices.size(); ++i) {
        rowCount[rowIndices[i]]++;
    }

    csrOffsets[0] = 0;
    for (int i = 1; i <= numRows; ++i) {
        csrOffsets[i] = csrOffsets[i - 1] + rowCount[i - 1];
    }

    std::vector<int> tempOffsets = csrOffsets;
    for (int i = 0; i < rowIndices.size(); ++i) {
        int row = rowIndices[i];
        int destIndex = tempOffsets[row]++;
        csrColumns[destIndex] = colIndices[i];
        csrValues[destIndex] = values[i];
    }
}

// Define your function here (e.g., increment each element)
__device__ myfloat my_function(myfloat input) {
    // return __int2double_rn(__double2int_rn(input) % THESMALLPRIME);
    #if USE_DOUBLE
        return fmod(input, 3323.0);
    #else
        return fmod(input, 3323.0f);
    #endif
}
  
// CUDA kernel to apply the function
__global__ void apply_function_kernel(myfloat *device_matrix, int matrix_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < matrix_size) {
        device_matrix[index] = my_function(device_matrix[index]);
    }
}

auto tic_start_time= std::chrono::high_resolution_clock::now();


void tic() {
    // Start the timer
    cudaDeviceSynchronize();
    tic_start_time = std::chrono::high_resolution_clock::now();
}
void toc(const std::string& msg = "") {
    // Stop the timer
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    // Calculate the elapsed time in milliseconds
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - tic_start_time).count();
    std::cout << "Elapsed time: " << elapsed_time << " ms. " << msg << std::endl;
}

void transpose_csr_matrix(
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_indices,
    const std::vector<myfloat>& values,
    int n_rows,
    int n_cols,
    std::vector<int>& row_ptr_t,
    std::vector<int>& col_indices_t,
    std::vector<myfloat>& values_t) {

    // Step 1: Count non-zero elements per column
    std::vector<int> nnz_per_col(n_cols, 0);
    for (const auto& col : col_indices) {
        nnz_per_col[col]++;
    }

    // Step 2: Compute row_ptr for transposed matrix
    row_ptr_t.resize(n_cols + 1, 0);
    for (int i = 0; i < n_cols; ++i) {
        row_ptr_t[i + 1] = row_ptr_t[i] + nnz_per_col[i];
    }

    // Step 3: Prepare space for transposed values and indices
    int nnz = values.size();
    values_t.resize(nnz);
    col_indices_t.resize(nnz);

    std::vector<int> next_insert_pos = row_ptr_t;

    // Step 4: Populate the transposed matrix
    for (int row = 0; row < n_rows; ++row) {
        int start = row_ptr[row];
        int end = row_ptr[row + 1];

        for (int idx = start; idx < end; ++idx) {
            int col = col_indices[idx];
            myfloat val = values[idx];

            int insert_pos = next_insert_pos[col];
            values_t[insert_pos] = val;
            col_indices_t[insert_pos] = row;
            next_insert_pos[col]++;
        }
    }
}


__global__ void csr_spmm_naive(
    int M, int N,
    const int *__restrict__ csr_row_ptr,
    const int *__restrict__ csr_col_idx,
    const myfloat *__restrict__ csr_val,
    const myfloat *__restrict__ B, // dense matrix B [K x N]
    myfloat *__restrict__ C        // output matrix C [M x N]
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // Initialize output row
    for (int j = 0; j < N; j++) {
        C[row * N + j] = 0.0;
    }

    // Iterate over non-zeros in row
    int row_start = csr_row_ptr[row];
    int row_end = csr_row_ptr[row + 1];
    for (int idx = row_start; idx < row_end; idx++) {
        int col = csr_col_idx[idx];
        myfloat val = csr_val[idx];

        for (int j = 0; j < N; j++) {
            C[row * N + j] += val * B[col * N + j];
        }
    }
}

__global__ void csr_spmm_2d(
    int M, int N,
    const int *__restrict__ csr_row_ptr,
    const int *__restrict__ csr_col_idx,
    const myfloat *__restrict__ csr_val,
    const myfloat *__restrict__ B, // [K x N]
    myfloat *__restrict__ C        // [M x N]
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    myfloat sum = 0.0;
    int start = csr_row_ptr[row];
    int end = csr_row_ptr[row + 1];

    for (int idx = start; idx < end; idx++) {
        int k = csr_col_idx[idx];
        myfloat a = csr_val[idx];
        myfloat b = B[k * N + col]; // column-major access of B
        sum += a * b;
    }

    C[row * N + col] = sum;
}


static std::vector<std::vector<myfloat>> hSp_list;

int compute_and_push_sp(cublasHandle_t blashandle, myfloat* dM1, myfloat* dM2, myfloat* dSp, int n_dense_vectors, int n_veclen) {
    myfloat alpha           = 1.0f;
    myfloat beta            = 0.0f;
    // float alpha = 1.0f;
    // float beta = 0.0f;
    int Sp_size = n_dense_vectors * n_dense_vectors;
    std::vector<myfloat> hSp(Sp_size);

    #if USE_DOUBLE
        CUBLAS_CHECK(cublasDgemm(blashandle, CUBLAS_OP_T, CUBLAS_OP_N, n_dense_vectors, n_dense_vectors, n_veclen, &alpha, dM1, n_veclen, dM2, n_veclen, &beta, dSp, n_dense_vectors));
    #else
        CUBLAS_CHECK(cublasSgemm(blashandle, CUBLAS_OP_T, CUBLAS_OP_N, n_dense_vectors, n_dense_vectors, n_veclen, &alpha, dM1, n_veclen, dM2, n_veclen, &beta, dSp, n_dense_vectors));
    #endif

        apply_function_kernel<<<((Sp_size + 255) / 256), 256>>>(dSp, Sp_size);

        // Copy the device buffer dSp to a local host buffer
        CHECK_CUDA(cudaMemcpy(hSp.data(), dSp, Sp_size * sizeof(myfloat), cudaMemcpyDeviceToHost));
    
        // print the first 10 entries of the result
        std::cout << "dSp (first 10 entries): ";
        for (int i = 0; i < 10 && i < Sp_size; ++i) {
            std::cout << hSp[i] << " ";
        }
        std::cout << std::endl;
        
        hSp_list.push_back(hSp);

        return 0;
}

int main(int argc, char* argv[]) {

    // load matrix from file
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file> <nr dense columns>" << std::endl;
        return -1;
    }
    std::vector<int> rowIndices, colIndices, csrOffsets, csrColumns, csrColumnsT, csrOffsetsT;
    std::vector<myfloat> values, csrValues, csrValuesT;
    int numRows, numCols, nnz;
    auto loadStart = std::chrono::high_resolution_clock::now();
    load_sms_matrix(argv[1], rowIndices, colIndices, values, numRows, numCols, nnz);
    auto loadStop = std::chrono::high_resolution_clock::now();
    auto loadMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(loadStop - loadStart).count();
    std::cout << "Matrix loading runtime: " << loadMilliseconds << " ms" << std::endl;

    auto convertStart = std::chrono::high_resolution_clock::now();
    coo_matrix_to_csr(numRows, rowIndices, colIndices, values, csrOffsets, csrColumns, csrValues);
    auto convertStop = std::chrono::high_resolution_clock::now();
    auto convertMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(convertStop - convertStart).count();
    std::cout << "COO to CSR conversion runtime: " << convertMilliseconds << " ms" << std::endl;

    std::cout << numRows <<"x" << numCols << " matrix loaded from file: " << argv[1] << " with nnz=" << nnz << std::endl;

    transpose_csr_matrix(csrOffsets, csrColumns, csrValues, numRows, numCols, csrOffsetsT, csrColumnsT, csrValuesT);
    std::cout << "CSR matrix transposed." << std::endl;
    std::cout << "Transposed matrix size: " << numCols << "x" << numRows << std::endl;
    std::cout << "Transposed matrix nnz: " << csrValuesT.size() << std::endl;

    // Random dense matrix for multiplication
    int denseCols = atoi(argv[2]);  // Example: Result matrix column size
    std::vector<myfloat> h_dense(numCols * denseCols);
    for (int i = 0; i < numCols * denseCols; ++i) {
        h_dense[i] = i % 101; //static_cast<myfloat>(rand()) / RAND_MAX;  // Random initialization
    }

    std::vector<myfloat> c_dense(numRows * denseCols);
    for (int i = 0; i < numRows * denseCols; ++i) {
        c_dense[i] = 0; //static_cast<myfloat>(rand()) / RAND_MAX;  // Random initialization
    }


    std::vector<myfloat> c_result(numRows * denseCols);

    int   A_num_rows      = numRows;
    int   A_num_cols      = numCols;
    int   A_nnz           = nnz;
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = denseCols;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    // int ldd = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
    int* hA_csrOffsets = &csrOffsets[0];
    // std::copy(csrOffsets.begin(), csrOffsets.end(), hA_csrOffsets);
    int*   hA_columns    = &csrColumns[0]; 
    myfloat* hA_values     = &csrValues[0]; 
    myfloat* hB            = &h_dense[0];
    myfloat* hC            = &c_dense[0]; 
    myfloat* hC_result     = &c_result[0]; 
    myfloat alpha           = 1.0f;
    myfloat beta            = 0.0f;



    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns, *dA_csrOffsetsT, *dA_columnsT;
    myfloat *dA_values, *dA_valuesT, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,(A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(myfloat))  )
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsetsT,(A_num_cols + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columnsT, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_valuesT,  A_nnz * sizeof(myfloat))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(myfloat)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(myfloat)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(myfloat),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_csrOffsetsT, &csrOffsetsT[0],
                            (A_num_cols + 1) * sizeof(int),
                            cudaMemcpyHostToDevice) )
     CHECK_CUDA( cudaMemcpy(dA_columnsT, &csrColumnsT[0], A_nnz * sizeof(int),
                            cudaMemcpyHostToDevice) )
     CHECK_CUDA( cudaMemcpy(dA_valuesT, &csrValuesT[0], A_nnz * sizeof(myfloat),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(myfloat),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(myfloat),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseSpMatDescr_t matAT;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    void*                dBuffer2    = NULL;
    // void*                dBuffer3    = NULL;
    size_t               bufferSize = 0;
    size_t               bufferSize2 = 0;
    // size_t               bufferSize3 = 0;
    cudaEvent_t start, stop;


    cublasHandle_t blashandle;
    CUBLAS_CHECK(cublasCreate(&blashandle));

    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_FMT) )
    CHECK_CUSPARSE( cusparseCreateCsr(&matAT, A_num_cols, A_num_rows, A_nnz,
        dA_csrOffsetsT, dA_columnsT, dA_valuesT,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_FMT) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_FMT, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_FMT, CUSPARSE_ORDER_COL) )

    // Create dense matrix D for the result of the second multiplication
    myfloat *dD;
    int D_size = A_num_cols * B_num_cols;
    CHECK_CUDA(cudaMalloc((void**)&dD, D_size * sizeof(myfloat)));
    CHECK_CUDA(cudaMemset(dD, 0, D_size * sizeof(myfloat)));
    int ldd = A_num_cols; // Leading dimension of D

    // Create dense matrix descriptor for D
    cusparseDnMatDescr_t matD;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matD, A_num_cols, B_num_cols, ldd, dD,
                                        CUDA_FMT, CUSPARSE_ORDER_COL));

    myfloat *dSp;
    int Sp_size = B_num_cols * B_num_cols;
    CHECK_CUDA(cudaMalloc((void**)&dSp, Sp_size * sizeof(myfloat)));
    CHECK_CUDA(cudaMemset(dSp, 0, Sp_size * sizeof(myfloat)));
    int ldsp = B_num_cols; // Leading dimension of D

    // Create dense matrix descriptor for D
    cusparseDnMatDescr_t matSp;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matSp, B_num_cols, B_num_cols, ldsp, dSp,
                                        CUDA_FMT, CUSPARSE_ORDER_COL));
    // std::vector<myfloat> hSp(Sp_size);

    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_FMT,
        CUSPARSE_NORMAL_ALGO, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
    // CHECK_CUSPARSE( cusparseSpMM_bufferSize(
    //     handle,
    //     CUSPARSE_OPERATION_TRANSPOSE,
    //     CUSPARSE_OPERATION_NON_TRANSPOSE,
    //     &alpha, matA, matC, &beta, matD, CUDA_FMT,
    //     CUSPARSE_TRANS_ALGO, &bufferSize2) )
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matAT, matC, &beta, matD, CUDA_FMT,
        CUSPARSE_TRANS_ALGO, &bufferSize2) )
    CHECK_CUDA( cudaMalloc(&dBuffer2, bufferSize2) )

    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    
        // execute SpMM

    // execute preprocess (optional)
    CHECK_CUSPARSE( cusparseSpMM_preprocess(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_FMT,
                                 CUSPARSE_NORMAL_ALGO, dBuffer) )
    // CHECK_CUSPARSE( cusparseSpMM_preprocess(
    //                                 handle,
    //                                 CUSPARSE_OPERATION_TRANSPOSE,
    //                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                 &alpha, matA, matC, &beta, matD, CUDA_FMT,
    //                                 CUSPARSE_TRANS_ALGO, dBuffer2) )                    
    CHECK_CUSPARSE( cusparseSpMM_preprocess(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matAT, matC, &beta, matD, CUDA_FMT,
        CUSPARSE_TRANS_ALGO, dBuffer2) )    
    compute_and_push_sp(blashandle, dB, dB, dSp, B_num_cols, A_num_cols);

    for (int round=0;round<100;round++){
        std::cout << "Round " << round << std::endl;
        // execute SpMM, multiply by A to get C

        tic();
        int threads_per_block = 128;
        int blocks_per_grid = (A_num_rows + threads_per_block - 1) / threads_per_block;
        
        csr_spmm_naive<<<blocks_per_grid, threads_per_block>>>(
            A_num_rows, B_num_cols, dA_csrOffsets, dA_columns, dA_values,
            dB, dC
        );
        toc("Handcrafted...:");
        tic();
        dim3 blockDim(16, 16);
        dim3 gridDim((B_num_cols + blockDim.x - 1) / blockDim.x,
                    (A_num_rows + blockDim.y - 1) / blockDim.y);

        csr_spmm_2d<<<gridDim, blockDim>>>(
            A_num_rows, B_num_cols, dA_csrOffsets, dA_columns, dA_values,
            dB, dC
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        toc("Handcrafted 2d...:");
        tic();
        CHECK_CUSPARSE( cusparseSpMM(handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_FMT,
            CUSPARSE_NORMAL_ALGO, dBuffer) );
        toc("SpMM A*B->C");
        tic();
        apply_function_kernel<<<((C_size + 255) / 256), 256>>>(dC, C_size);
        toc("apply_function_kernel C");
        
        // Execute SpMM for the second multiplication (matA^T * matC -> matD)
        tic();
        // CHECK_CUSPARSE(cusparseSpMM(handle,
        //                             CUSPARSE_OPERATION_TRANSPOSE,
        //                             CUSPARSE_OPERATION_NON_TRANSPOSE,
        //                             &alpha, matA, matC, &beta, matD, CUDA_FMT,
        //                             CUSPARSE_TRANS_ALGO, dBuffer2));
        CHECK_CUSPARSE(cusparseSpMM(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matAT, matC, &beta, matD, CUDA_FMT,
                                        CUSPARSE_TRANS_ALGO, dBuffer2));
        toc("SpMM A^T*C->D");
        tic();
        apply_function_kernel<<<((D_size + 255) / 256), 256>>>(dD, D_size);
        toc("apply_function_kernel D");
        
        tic();
        compute_and_push_sp(blashandle, dB, dD, dSp, B_num_cols, A_num_cols);
        compute_and_push_sp(blashandle, dD, dD, dSp, B_num_cols, A_num_cols);
        toc("compute_and_push_sp D");

        // Next multiply by A to get C
        CHECK_CUSPARSE( cusparseSpMM(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, matD, &beta, matC, CUDA_FMT,
                                        CUSPARSE_NORMAL_ALGO, dBuffer) )
        apply_function_kernel<<<((C_size + 255) / 256), 256>>>(dC, C_size);
        // and by A^T to get B
        // CHECK_CUSPARSE(cusparseSpMM(handle,
        //     CUSPARSE_OPERATION_TRANSPOSE,
        //     CUSPARSE_OPERATION_NON_TRANSPOSE,
        //     &alpha, matA, matC, &beta, matB, CUDA_FMT,
        //     CUSPARSE_TRANS_ALGO, dBuffer2));
        CHECK_CUSPARSE(cusparseSpMM(handle,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha, matAT, matC, &beta, matB, CUDA_FMT,
                CUSPARSE_TRANS_ALGO, dBuffer2));
        apply_function_kernel<<<((B_size + 255) / 256), 256>>>(dB, B_size);
        
        compute_and_push_sp(blashandle, dB, dD, dSp, B_num_cols, A_num_cols);
        compute_and_push_sp(blashandle, dB, dB, dSp, B_num_cols, A_num_cols);

        // CHECK_CUBLAS(cublasGemmEx(
        //     handle,
        //     CUBLAS_OP_T, CUBLAS_OP_N, // transA = Dᵗ, transB = D
        //     B_num_cols, B_num_cols, A_num_cols,
        //     &alpha,
        //     dD, CUDA_FMT, A_num_cols,
        //     dD, CUDA_FMT, A_num_cols,
        //     &beta,
        //     dSp, CUDA_FMT, B_num_cols,
        //     CUDA_FMT,
        //     CUBLAS_GEMM_DEFAULT));                                
    
    }



    CHECK_CUDA(cudaDeviceSynchronize());
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    // Clean up the dense matrix descriptor for D
    CHECK_CUSPARSE(cusparseDestroyDnMat(matD));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matSp));

    CHECK_CUSPARSE( cusparseDestroy(handle) )

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "SpMM operation runtime: " << milliseconds << " ms" << std::endl;
    std::cout << "Total throughput: " << hSp_list.size() * B_num_cols * 1e3 / milliseconds  << "/s." << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));


    // reduce mod p 
    // Call the kernel
    auto modstart = std::chrono::high_resolution_clock::now();
    int matrix_size = C_size;
    apply_function_kernel<<<((matrix_size + 255) / 256), 256>>>(dC, matrix_size);
    auto modstop = std::chrono::high_resolution_clock::now();
    auto modMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(modstop - modstart).count();
    std::cout << "Kernel execution runtime (mod): " << modMilliseconds << " ms" << std::endl;

    //--------------------------------------------------------------------------
    // device result check
    auto copyStart = std::chrono::high_resolution_clock::now();
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(myfloat),
                           cudaMemcpyDeviceToHost) )
    auto copyStop = std::chrono::high_resolution_clock::now();
    auto copyMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(copyStop - copyStart).count();
    std::cout << "Device to host copy runtime: " << copyMilliseconds << " ms" << std::endl;
    
    int correct = 1;
    // for (int i = 0; i < A_num_rows; i++) {
    //     for (int j = 0; j < B_num_cols; j++) {
    //         if (hC[i + j * ldc] != hC_result[i + j * ldc]) {
    //             correct = 0; // direct floating point comparison is not reliable
    //             break;
    //         }
    //     }
    // }
    if (correct)
        printf("spmm_csr_example test PASSED\n");
    else
        printf("spmm_csr_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA(cudaFree(dD));
    CHECK_CUDA(cudaFree(dSp));
    return EXIT_SUCCESS;
}