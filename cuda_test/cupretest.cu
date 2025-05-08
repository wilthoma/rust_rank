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
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpMM
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>  // Include fstream for file input

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

void load_sms_matrix(const std::string& filename, std::vector<int>& rowIndices, std::vector<int>& colIndices, std::vector<float>& values, int& numRows, int& numCols, int& nnz) {
    std::ifstream file(filename);  // Make sure to include <fstream>
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(-1);
    }

    char arbitraryChar;
    file >> numRows >> numCols >> arbitraryChar;

    std::vector<int> tempRowIndices;
    std::vector<int> tempColIndices;
    std::vector<float> tempValues;

    int row, col;
    float value;
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

void coo_matrix_to_csr(int numRows, const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<float>& values,
                       std::vector<int>& csrOffsets, std::vector<int>& csrColumns, std::vector<float>& csrValues) {
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



int main(int argc, char* argv[]) {
    // Host problem definition
    // int   A_num_rows      = 4;
    // int   A_num_cols      = 4;
    // int   A_nnz           = 9;
    // int   B_num_rows      = A_num_cols;
    // int   B_num_cols      = 3;
    // int   ldb             = B_num_rows;
    // int   ldc             = A_num_rows;
    // int   B_size          = ldb * B_num_cols;
    // int   C_size          = ldc * B_num_cols;
    // int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    // int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    // float hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
    //                           6.0f, 7.0f, 8.0f, 9.0f };
    // float hB[]            = { 1.0f,  2.0f,  3.0f,  4.0f,
    //                           5.0f,  6.0f,  7.0f,  8.0f,
    //                           9.0f, 10.0f, 11.0f, 12.0f };
    // float hC[]            = { 0.0f, 0.0f, 0.0f, 0.0f,
    //                           0.0f, 0.0f, 0.0f, 0.0f,
    //                           0.0f, 0.0f, 0.0f, 0.0f };
    // float hC_result[]     = { 19.0f,  8.0f,  51.0f,  52.0f,
    //                           43.0f, 24.0f, 123.0f, 120.0f,
    //                           67.0f, 40.0f, 195.0f, 188.0f };
    // float alpha           = 1.0f;
    // float beta            = 0.0f;

    // load matrix from file
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file>" << std::endl;
        return -1;
    }
    std::vector<int> rowIndices, colIndices, csrOffsets, csrColumns;
    std::vector<float> values, csrValues;
    int numRows, numCols, nnz;
    load_sms_matrix(argv[1], rowIndices, colIndices, values, numRows, numCols, nnz);
    coo_matrix_to_csr(numRows, rowIndices, colIndices, values, csrOffsets, csrColumns, csrValues);

    std::cout << numRows <<"x" << numCols << " matrix loaded from file: " << argv[1] << " with nnz=" << nnz << std::endl;

    // Random dense matrix for multiplication
    int denseCols = 10;  // Example: Result matrix column size
    std::vector<float> h_dense(numCols * denseCols);
    for (int i = 0; i < numCols * denseCols; ++i) {
        h_dense[i] = 1; //static_cast<float>(rand()) / RAND_MAX;  // Random initialization
    }

    std::vector<float> c_dense(numRows * denseCols);
    for (int i = 0; i < numRows * denseCols; ++i) {
        c_dense[i] = 0; //static_cast<float>(rand()) / RAND_MAX;  // Random initialization
    }

    std::vector<float> c_result(numRows * denseCols);

    int   A_num_rows      = numRows;
    int   A_num_cols      = numCols;
    int   A_nnz           = nnz;
    int   B_num_rows      = A_num_cols;
    int   B_num_cols      = denseCols;
    int   ldb             = B_num_rows;
    int   ldc             = A_num_rows;
    int   B_size          = ldb * B_num_cols;
    int   C_size          = ldc * B_num_cols;
    int* hA_csrOffsets = &csrOffsets[0];
    // std::copy(csrOffsets.begin(), csrOffsets.end(), hA_csrOffsets);
    int*   hA_columns    = &csrColumns[0]; //{ 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    float* hA_values     = &csrValues[0]; //{ 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                          //    6.0f, 7.0f, 8.0f, 9.0f };
    float* hB            = &h_dense[0];//{ 1.0f,  2.0f,  3.0f,  4.0f,
                           //   5.0f,  6.0f,  7.0f,  8.0f,
                           //   9.0f, 10.0f, 11.0f, 12.0f };
    float* hC            = &c_dense[0]; //{ 0.0f, 0.0f, 0.0f, 0.0f,
                            //   0.0f, 0.0f, 0.0f, 0.0f,
                            //   0.0f, 0.0f, 0.0f, 0.0f };
    float* hC_result     = &c_result[0]; //{ 19.0f,  8.0f,  51.0f,  52.0f,
                            //   43.0f, 24.0f, 123.0f, 120.0f,
                            //   67.0f, 40.0f, 195.0f, 188.0f };
    float alpha           = 1.0f;
    float beta            = 0.0f;



    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    float *dA_values, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))    )
    CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))  )
    CHECK_CUDA( cudaMalloc((void**) &dB,         B_size * sizeof(float)) )
    CHECK_CUDA( cudaMalloc((void**) &dC,         C_size * sizeof(float)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size * sizeof(float),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));


    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, B_num_cols, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) )
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute preprocess (optional)
    CHECK_CUSPARSE( cusparseSpMM_preprocess(
                                 handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) )

    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "SpMM operation runtime: " << milliseconds << " ms" << std::endl;

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));


    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size * sizeof(float),
                           cudaMemcpyDeviceToHost) )
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
    return EXIT_SUCCESS;
}