
#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdexcept>
#include <vector>

const int DOT_CHUNK_SIZE = 64;

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        throw std::runtime_error("CUDA error");                                \
    }                                                                          \
}


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
__global__ void modp_kernel(T *device_matrix, int matrix_size, int offset, T p) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < matrix_size) {
        index += offset;
        device_matrix[index] = device_matrix[index] % p;
    }
}

template<typename T>
__global__ void dense_gemm_TN_chunked3D_offset(int n, int k,  // n = n_dense_vectors, k = n_veclen
    const T* __restrict__ A,  // Transposed: A^T [n x k]
    const T* __restrict__ B,  // B [k x n]
    T* C,                    // Output: [n x n] slice of C at position offset
    int offset, // offset in C, in units of myfloat
    int chunk_size, T prime)
{
    int chunk = blockIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.z * blockDim.x + threadIdx.x;

    // TODO : check it is the correct order (not <)
    // we only need to compute half the matrix
    // We waste a bit of buffer, though
    if (row > col) {
        return;
    }

    int chunk_start = chunk * chunk_size;
    int chunk_end = min(chunk_start + chunk_size, k);

    if (row < n && col < n) {
        T acc = 0;
        for (int i = chunk_start; i < chunk_end; ++i) {
            acc += A[i * n + row] * B[i * n + col];  
            // acc += A[i * n + row] * B[i + col * k];  // A is row-major transposed
        }

        // Accumulate the result into global memory (C[row + col * n])
        atomicAdd(&C[row + col * n + offset], acc % prime);
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
    void spmm(const CudaDenseMatrix<T>& B, CudaDenseMatrix<T>& C, T prime = 0) {
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

        //CHECK_CUDA(
            csr_spmm<<<gridDim, blockDim>>>(
            numRows,B.numCols, d_rowOffsets, d_colIndices, d_values,
            B.d_data, C.d_data);//);
        if (prime != 0) {
            C.modp(prime);
        }
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
        CHECK_CUDA(modp_kernel<<<((size + 255) / 256), 256>>>(d_data, size, 0, prime));
    }

    // computes the upper tringular part of this^T* B and stores the desult in dC, at position position
    void mTm_tri(const CudaDenseMatrix<T>& B, T* dC, int position, T prime) {
        // Check if the dimensions are compatible
        if (numRows != B.numRows || numCols != B.numCols) {
            throw std::runtime_error("Matrix dimensions do not match for M^T * B.");
        }
        int n_vec_len = numRows;
        int n_dense_vectors = numCols;
        int Sp_size = n_dense_vectors * n_dense_vectors;

        int num_chunks = (n_veclen + DOT_CHUNK_SIZE - 1) / DOT_CHUNK_SIZE;
        dim3 blockDim(16, 16);  // threads per block
        dim3 gridDim(num_chunks,
                     (n_dense_vectors + blockDim.y - 1) / blockDim.y,
                     (n_dense_vectors + blockDim.x - 1) / blockDim.x);
        
        int offset = seq_position * Sp_size; // offset in units of myfloat
        //CHECK_CUDA(
            dense_gemm_TN_chunked3D_offset<<<gridDim, blockDim>>>(
            n_dense_vectors, n_veclen, d_data, B.d_data, dC, offset, DOT_CHUNK_SIZE, prime);
        //);
    
        //CHECK_CUDA(
            modp_kernel<T><<<((Sp_size + 255) / 256), 256>>>(dC, Sp_size, offset, prime);
        //);

    }

};



#endif // CUDA_HELPERS_H