
#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <vector>


#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}


#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                     \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)


template<typename T>
__global__ void csr_spmm(
    int M, int N,
    const int *__restrict__ csr_row_ptr,
    const int *__restrict__ csr_col_idx,
    const T *__restrict__ csr_val,
    const T *__restrict__ B, // [K x N]
    T *__restrict__ C        // [M x N]
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || col >= N) return;

    T sum = 0.0;
    int start = csr_row_ptr[row];
    int end = csr_row_ptr[row + 1];

    for (int idx = start; idx < end; idx++) {
        int k = csr_col_idx[idx];
        T a = csr_val[idx];
        T b = B[k * N + col]; // column-major access of B
        sum += a * b;
    }

    C[row * N + col] = sum;
}

template<typename T>
__global__ void modp_kernel(T *device_matrix, int matrix_size, T p) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < matrix_size) {
        device_matrix[index] = device_matrix[index] % p;
    }
}


template<typename T>
struct CudaCsrMatrix {
    int numRows;
    int numCols;
    int* d_rowOffsets;
    int* d_colIndices;
    T* d_values;

    static CudaCsrMatrix<T> from_host(const CsrMatrix<T>& host_matrix) {
        CudaCsrMatrix<T> cuda_matrix;
        cuda_matrix.numRows = host_matrix.numRows;
        cuda_matrix.numCols = host_matrix.numCols;

        size_t size_rowOffsets = (host_matrix.numRows + 1) * sizeof(int);
        size_t size_colIndices = host_matrix.values.size() * sizeof(int);
        size_t size_values = host_matrix.values.size() * sizeof(T);

        CHECK_CUDA(cudaMalloc((void**)&cuda_matrix.d_rowOffsets, size_rowOffsets));
        CHECK_CUDA(cudaMalloc((void**)&cuda_matrix.d_colIndices, size_colIndices));
        CHECK_CUDA(cudaMalloc((void**)&cuda_matrix.d_values, size_values));

        CHECK_CUDA(cudaMemcpy(cuda_matrix.d_rowOffsets, host_matrix.rowOffsets.data(), size_rowOffsets, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(cuda_matrix.d_colIndices, host_matrix.colIndices.data(), size_colIndices, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(cuda_matrix.d_values, host_matrix.values.data(), size_values, cudaMemcpyHostToDevice));

        return cuda_matrix;
    }

    void release() {
        CHECK_CUDA(cudaFree(d_rowOffsets));
        CHECK_CUDA(cudaFree(d_colIndices));
        CHECK_CUDA(cudaFree(d_values));
    }

    // produces the product this * B and stores the result in C
    void spmm(
        const CudaDenseMatrix<T>& B,
        CudaDenseMatrix<T>& C) {
        // Check if the dimensions are compatible
        if (numCols != dense_matrix.numRows) {
            throw std::runtime_error("Matrix dimensions do not match for SpMM.");
        }
        if (result_matrix.numRows != numRows || result_matrix.numCols != dense_matrix.numCols) {
            throw std::runtime_error("Result matrix dimensions do not match.");
        }
        dim3 blockDim(16, 32);
        dim3 gridDim((numRows + blockDim.x - 1) / blockDim.x,
                    (B.numCols + blockDim.y - 1) / blockDim.y);

        csr_spmm<<<gridDim, blockDim>>>(
            numRows,B.numCols, d_rowOffsets, d_colIndices, d_values,
            B.d_data, C.d_data
        );
    }

    
};




template<typename T>
struct CudaDenseMatrix {
    int numRows;
    int numCols;
    T* d_data;

    static CudaDenseMatrix<T> allocate(int rows, int cols, T default_value = 0) {
        CudaDenseMatrix<T> cuda_matrix;
        cuda_matrix.numRows = rows;
        cuda_matrix.numCols = cols;

        size_t size_data = rows * cols * sizeof(T);
        CHECK_CUDA(cudaMalloc((void**)&cuda_matrix.d_data, size_data));
        CHECK_CUDA(cudaMemset(cuda_matrix.d_data, default_value, size_data));

        return cuda_matrix;
    }

    static CudaDenseMatrix<T> from_host(const std::vector<T>& host_matrix, int rows, int cols) {
        CudaDenseMatrix<T> cuda_matrix;
        cuda_matrix.numRows = rows;
        cuda_matrix.numCols = cols;

        size_t size_data = rows * cols * sizeof(T);
        CHECK_CUDA(cudaMalloc((void**)&cuda_matrix.d_data, size_data));
        CHECK_CUDA(cudaMemcpy(cuda_matrix.d_data, host_matrix.data(), size_data, cudaMemcpyHostToDevice));

        return cuda_matrix;
    }

    void release() {
        CHECK_CUDA(cudaFree(d_data));
    }

    void modp(T prime) {
        int size = numRows * numCols;
        modp_kernel<<<((size + 255) / 256), 256>>>(d_data, size, prime);
    }

    // computes the upper tringular part of this^T* B and stores the desult in dC, at position offset
    void mTm_tri(const CudaDenseMatrix<T>& B, T* dC, int offset, T prime) {
        // Check if the dimensions are compatible
        if (numCols != B.numRows) {
            throw std::runtime_error("Matrix dimensions do not match for M^T * B.");
        }



    }

};



#endif // CUDA_HELPERS_H