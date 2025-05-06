#include <cuda_runtime.h>
#include <cusparse.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cassert>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __LINE__ << ": " << cudaGetErrorString(cudaGetLastError()) << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUSPARSE(call) \
    if ((call) != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE error at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    // Load sparse matrix from "matrix.txt" (Matrix Market format)
    std::ifstream fin("matrix.txt");
    if (!fin) {
        std::cerr << "Failed to open matrix.txt" << std::endl;
        return 1;
    }

    std::string line;
    // Skip comments
    do {
        std::getline(fin, line);
    } while (line[0] == '%');

    int numRows, numCols, nnz;
    std::istringstream(line) >> numRows >> numCols >> nnz;

    std::vector<int> h_rowIndices(nnz);
    std::vector<int> h_colIndices(nnz);
    std::vector<float> h_values(nnz);

    for (int i = 0; i < nnz; ++i) {
        int row, col;
        float val;
        fin >> row >> col >> val;
        h_rowIndices[i] = row - 1; // 1-based to 0-based
        h_colIndices[i] = col - 1;
        h_values[i] = val;
    }
    fin.close();

    // Convert COO to CSR
    std::vector<int> h_rowPtr(numRows + 1, 0);
    for (int i = 0; i < nnz; ++i)
        h_rowPtr[h_rowIndices[i] + 1]++;
    for (int i = 0; i < numRows; ++i)
        h_rowPtr[i + 1] += h_rowPtr[i];

    std::vector<int> h_colInd(nnz);
    std::vector<float> h_csrVals(nnz);
    std::vector<int> rowOffset = h_rowPtr;
    for (int i = 0; i < nnz; ++i) {
        int row = h_rowIndices[i];
        int dest = rowOffset[row]++;
        h_colInd[dest] = h_colIndices[i];
        h_csrVals[dest] = h_values[i];
    }

    // Dense matrix dimensions
    int denseCols = 64;
    std::vector<float> h_dense(numCols * denseCols);
    for (auto& val : h_dense)
        val = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    int *d_rowPtr, *d_colInd;
    float *d_vals, *d_dense, *d_result;
    CHECK_CUDA(cudaMalloc((void**)&d_rowPtr, (numRows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_vals, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_dense, numCols * denseCols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_result, numRows * denseCols * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_rowPtr, h_rowPtr.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, h_colInd.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_vals, h_csrVals.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_dense, h_dense.data(), numCols * denseCols * sizeof(float), cudaMemcpyHostToDevice));

    // cuSPARSE setup
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, numRows, numCols, nnz,
                                     d_rowPtr, d_colInd, d_vals,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, numCols, denseCols, denseCols,
                                       d_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, numRows, denseCols, denseCols,
                                       d_result, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    float alpha = 1.0f, beta = 0.0f;
    size_t bufferSize = 0;
    void* dBuffer = nullptr;

    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, CUDA_R_32F, CUSPARSE_MM_ALG_DEFAULT, dBuffer));

    // Copy result back
    std::vector<float> h_result(numRows * denseCols);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_result, numRows * denseCols * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Sparse x dense multiplication completed. Example result value: "
              << h_result[0] << std::endl;

    // Cleanup
    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_vals);
    cudaFree(d_dense);
    cudaFree(d_result);
    cudaFree(dBuffer);

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);

    return 0;
}
