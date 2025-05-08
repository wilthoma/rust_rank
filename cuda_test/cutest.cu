#include <iostream>
#include <vector>
#include <cassert>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <fstream>  // Include fstream for file input

// CUDA error checking macro
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
        exit(-1); \
    } \
}

// cuSPARSE error checking macro
#define CHECK_CUSPARSE(call) { \
    cusparseStatus_t err = call; \
    if (err != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE error: " << err << std::endl; \
        exit(-1); \
    } \
}

void load_sms_matrix(const std::string& filename, std::vector<int>& rowIndices, std::vector<int>& colIndices, std::vector<float>& values, int& numRows, int& numCols, int& nnz) {
    std::ifstream file(filename);  // Make sure to include <fstream>
    if (!file) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(-1);
    }

    file >> numRows >> numCols >> nnz;

    rowIndices.resize(nnz);
    colIndices.resize(nnz);
    values.resize(nnz);

    for (int i = 0; i < nnz; ++i) {
        file >> rowIndices[i] >> colIndices[i] >> values[i];
    }
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_file>" << std::endl;
        return -1;
    }

    // Load matrix
    std::vector<int> rowIndices, colIndices;
    std::vector<float> values;
    int numRows, numCols, nnz;
    load_sms_matrix(argv[1], rowIndices, colIndices, values, numRows, numCols, nnz);

    // Random dense matrix for multiplication
    int denseCols = 10;  // Example: Result matrix column size
    std::vector<float> h_dense(numCols * denseCols);
    for (int i = 0; i < numCols * denseCols; ++i) {
        h_dense[i] = 1; //static_cast<float>(rand()) / RAND_MAX;  // Random initialization
    }
    

    // Allocate device memory for input matrices and results
    float *d_dense, *d_result, *d_rowPtr, *d_colInd, *d_csrVals;
    CHECK_CUDA(cudaMalloc(&d_dense, numCols * denseCols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_result, numRows * denseCols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rowPtr, (numRows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_colInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csrVals, nnz * sizeof(float)));

    // Copy dense matrix and sparse matrix data to the device
    CHECK_CUDA(cudaMemcpy(d_dense, h_dense.data(), numCols * denseCols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rowPtr, rowIndices.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_colInd, colIndices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrVals, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice));

    // Initialize cuSPARSE handle
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Prepare cuSPARSE matrices
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, numRows, numCols, nnz, d_rowPtr, d_colInd, d_csrVals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseDnMatDescr_t matB, matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, numCols, denseCols, denseCols, d_dense, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, numRows, denseCols, denseCols, d_result, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Set alpha and beta in device memory
    float alpha = 1.0f, beta = 0.0f;
    float *d_alpha, *d_beta;
    CHECK_CUDA(cudaMalloc(&d_alpha, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_beta, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_alpha, &alpha, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_beta, &beta, sizeof(float), cudaMemcpyHostToDevice));

    // Compute the buffer size required by cusparseSpMM
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha,
        matA,
        matB,
        d_beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize
    ));

    // Allocate buffer if needed
    if (bufferSize > 0) {
        CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
    }

    // Perform sparse matrix-dense matrix multiplication
    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        d_alpha,
        matA,
        matB,
        d_beta,
        matC,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer
    ));

    // Synchronize the device to ensure all computation is done
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy the result back to host
    std::vector<float> h_result(numRows * denseCols);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_result, numRows * denseCols * sizeof(float), cudaMemcpyDeviceToHost));

    // Print a small part of the result (for debugging purposes)
    std::cout << "Result matrix (first 5 elements):" << std::endl;
    for (int i = 0; i < 5 && i < numRows * denseCols; ++i) {
        std::cout << h_result[i] << " ";
    }
    std::cout << std::endl;

    // Clean up resources
    CHECK_CUDA(cudaFree(d_dense));
    CHECK_CUDA(cudaFree(d_result));
    CHECK_CUDA(cudaFree(d_rowPtr));
    CHECK_CUDA(cudaFree(d_colInd));
    CHECK_CUDA(cudaFree(d_csrVals));
    CHECK_CUDA(cudaFree(d_alpha));
    CHECK_CUDA(cudaFree(d_beta));
    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroy(handle));
    return 0;
}
