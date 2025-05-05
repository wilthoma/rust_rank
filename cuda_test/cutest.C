#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <curand_kernel.h>

// Function to load the sparse matrix in SMS format from a file
void loadSparseMatrix(const std::string& filename, std::vector<int>& rowOffsets, std::vector<int>& colIndices, std::vector<float>& values, int& numRows, int& numCols, int& nnz) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    std::getline(file, line); // Read dimensions line
    std::istringstream(dimStream(line)) >> numRows >> numCols >> nnz;

    rowOffsets.resize(numRows + 1);
    colIndices.resize(nnz);
    values.resize(nnz);

    // Read the row offsets
    for (int i = 0; i < numRows + 1; ++i) {
        file >> rowOffsets[i];
    }

    // Read the column indices and values
    for (int i = 0; i < nnz; ++i) {
        file >> colIndices[i] >> values[i];
    }
}

// Function to generate a random dense matrix on the GPU
__global__ void generateRandomDenseMatrix(float* d_matrix, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        d_matrix[idx] = curand_uniform(&state[idx]); // Generate random values between 0 and 1
    }
}

int main() {
    int numRows, numCols, nnz;
    std::vector<int> rowOffsets, colIndices;
    std::vector<float> values;

    // Load the sparse matrix
    loadSparseMatrix("matrix.txt", rowOffsets, colIndices, values, numRows, numCols, nnz);

    // Allocate memory for the sparse matrix and dense matrix
    int* d_rowOffsets;
    int* d_colIndices;
    float* d_values;
    float* d_denseMatrix;
    float* d_resultMatrix;

    cudaMalloc(&d_rowOffsets, (numRows + 1) * sizeof(int));
    cudaMalloc(&d_colIndices, nnz * sizeof(int));
    cudaMalloc(&d_values, nnz * sizeof(float));
    cudaMalloc(&d_denseMatrix, numCols * numCols * sizeof(float));
    cudaMalloc(&d_resultMatrix, numRows * numCols * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_rowOffsets, rowOffsets.data(), (numRows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIndices, colIndices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Generate a random dense matrix on the GPU
    dim3 blocks((numCols * numCols + 255) / 256);
    dim3 threads(256);
    generateRandomDenseMatrix<<<blocks, threads>>>(d_denseMatrix, numCols, numCols);

    // Initialize cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Allocate memory for the result matrix (C = A * B)
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform the sparse-dense matrix multiplication (C = A * B)
    cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, numRows, numCols, numCols, nnz, &alpha,
                   d_values, d_rowOffsets, d_colIndices, d_denseMatrix, numCols, &beta, d_resultMatrix, numCols);

    // Copy the result back to the host
    std::vector<float> result(numRows * numCols);
    cudaMemcpy(result.data(), d_resultMatrix, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);

    // Output the result
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cout << result[i * numCols + j] << " ";
        }
        std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_rowOffsets);
    cudaFree(d_colIndices);
    cudaFree(d_values);
    cudaFree(d_denseMatrix);
    cudaFree(d_resultMatrix);
    cusparseDestroy(handle);

    return 0;
}
