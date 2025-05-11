#ifndef MATRICES_H
#define MATRICES_H

#include <vector>
#include <iostream>
#include <fstream>
#include <random>
#include <string>


template<typename T>
struct CooMatrix;

template<typename T>
struct CsrMatrix {
    int numRows;
    int numCols;
    std::vector<int> rowOffsets;
    std::vector<int> colIndices;
    std::vector<T> values;

    CsrMatrix(int rows, int cols, const std::vector<int>& offsets, const std::vector<int>& indices, const std::vector<T>& vals)
        : numRows(rows), numCols(cols), rowOffsets(offsets), colIndices(indices), values(vals) {}

    void csr_columnrescale(
        // int numRows,
        // int numCols,
        // const std::vector<int>& csrOffsets,
        // const std::vector<int>& csrColumns,
        // std::vector<myfloat>& csrValues,
        const std::vector<T>& scale_factors, T p) 
    {
        if (scale_factors.size() != numCols) {
            std::cerr << "Error: scale_factors size does not match number of columns." << std::endl;
            return;
        }
        int nnz = values.size();
        for (int i=0;i<nnz;i++) {
            values[i] = (values[i] * scale_factors[colIndices[i]]) % p;
        }
    }
    void csr_rowrescale(
        // int numRows,
        // int numCols,
        // const std::vector<int>& csrOffsets,
        // const std::vector<int>& csrColumns,
        // std::vector<myfloat>& csrValues,
        const std::vector<T>& scale_factors,
        T p) 
    {
        if (scale_factors.size() != numRows) {
            std::cerr << "Error: scale_factors size does not match number of rows." << std::endl;
            return;
        }
        for (int row=0;row<numRows;row++) {
            int start = rowOffsets[row];
            int end = rowOffsets[row + 1];
            T scale = scale_factors[row];
            for (int idx = start; idx < end; ++idx) {
                values[idx] = (values[idx] * scale) % p;
            }
        }
    }

    bool is_csr_valid() { //int numRows, int numCols, const std::vector<int>& csrOffsets, const std::vector<int>& csrColumns, const std::vector<myfloat>& csrValues) {
        // Check if the CSR format is valid
        if (rowOffsets.size() != numRows + 1) {
            std::cerr << "Invalid CSR offsets size." << std::endl;
            return false;
        }
        if (rowOffsets[0] != 0) {
            std::cerr << "Invalid CSR offsets: first element should be 0." << std::endl;
            return false;
        }
        for (int i = 1; i <= numRows; ++i) {
            if (rowOffsets[i] < rowOffsets[i - 1]) {
                std::cerr << "Invalid CSR offsets: non-decreasing property violated." << std::endl;
                return false;
            }
        }
        for (int i = 0; i < values.size(); ++i) {
            if (colIndices[i] < 0 || colIndices[i] >= numCols) {
                std::cerr << "Invalid CSR columns: out of bounds." << std::endl;
                return false;
            }
        }
        if (values.size() != colIndices.size()) {
            std::cerr << "Invalid CSR values: size mismatch." << std::endl;
            return false;
        }
        return true;
    }

    CsrMatrix<T> transpose(){
        // const std::vector<int>& row_ptr,
        // const std::vector<int>& col_indices,
        // const std::vector<myfloat>& values,
        // int n_rows,
        // int n_cols,
        std::vector<int> row_ptr_t;
        std::vector<int> col_indices_t;
        std::vector<T> values_t;
    
        // Step 1: Count non-zero elements per column
        std::vector<int> nnz_per_col(numCols, 0);
        int nnzc = colIndices.size();
        for (int i = 0; i < nnzc; ++i) {
            int col = colIndices[i];
            // if (col <0 || col >= n_cols) {
            //     std::cerr << "Error: Column index out of bounds." <<col<< " vs " <<i << " vs "<< n_cols  << std::endl;
            //     exit(-1);
            // }
            nnz_per_col[col]++;
        }
    
        // Step 2: Compute row_ptr for transposed matrix
        std::vector<int> tmprow_ptr_t(numCols+1, 0);
        // row_ptr_t.resize(n_cols + 1, 0);
        for (int i = 0; i < numCols; ++i) {
            tmprow_ptr_t[i + 1] = tmprow_ptr_t[i] + nnz_per_col[i];
        }
    
        // Step 3: Prepare space for transposed values and indices
        int nnz = values.size();
        values_t.resize(nnz,0);
        col_indices_t.resize(nnz,0);
    
        std::vector<int> next_insert_pos = tmprow_ptr_t;
    
        // Step 4: Populate the transposed matrix
        for (int row = 0; row < numRows; ++row) {
            int start = rowOffsets[row];
            int end = rowOffsets[row + 1];
    
            for (int idx = start; idx < end; ++idx) {
                int col = colIndices[idx];
                T val = values[idx];
    
                int insert_pos = next_insert_pos[col];
                values_t[insert_pos] = val;
                col_indices_t[insert_pos] = row;
                next_insert_pos[col]++;
            }
        }
    
        row_ptr_t = std::move(tmprow_ptr_t);

        return CsrMatrix<T>(numCols, numRows, row_ptr_t, col_indices_t, values_t);
    }

    static CsrMatrix<T> from_sms_file(const std::string& filename, T prime) {
        return CooMatrix<T>::from_sms_file(filename, prime).to_csr();
    }
};

template<typename T>
struct CooMatrix {
    int numRows;
    int numCols;
    std::vector<int> rowIndices;
    std::vector<int> colIndices;
    std::vector<T> values;

    CooMatrix(int rows, int cols, const std::vector<int>& rIndices, const std::vector<int>& cIndices, const std::vector<T>& vals)
        : numRows(rows), numCols(cols), rowIndices(rIndices), colIndices(cIndices), values(vals) {}

    CsrMatrix<T> to_csr() {
        std::vector<int> csrOffsets(numRows + 1, 0);

        std::vector<int> rowCount(numRows, 0);
        for (int i = 0; i < rowIndices.size(); ++i) {
            rowCount[rowIndices[i]]++;
        }

        csrOffsets[0] = 0;
        for (int i = 1; i <= numRows; ++i) {
            csrOffsets[i] = csrOffsets[i - 1] + rowCount[i - 1];
        }

        return CsrMatrix<T>(numRows, numCols, csrOffsets, colIndices, values);
    }

    CooMatrix<T> transpose() {
        CooMatrix<T> ret = CooMatrix<T>(numCols, numRows, colIndices, rowIndices, values);
        ret.sort();
        return ret;
    }

    void sort() {
        std::vector<int> indices(rowIndices.size());
        for (int i = 0; i < rowIndices.size(); ++i) {
            indices[i] = i;
        }

        std::sort(indices.begin(), indices.end(), [this](int a, int b) {
            if (rowIndices[a] != rowIndices[b]) {
                return rowIndices[a] < rowIndices[b];
            }
            return colIndices[a] < colIndices[b];
        });

        std::vector<int> sortedRowIndices(rowIndices.size());
        std::vector<int> sortedColIndices(colIndices.size());
        std::vector<T> sortedValues(values.size());

        for (int i = 0; i < indices.size(); ++i) {
            sortedRowIndices[i] = rowIndices[indices[i]];
            sortedColIndices[i] = colIndices[indices[i]];
            sortedValues[i] = values[indices[i]];
        }

        rowIndices = std::move(sortedRowIndices);
        colIndices = std::move(sortedColIndices);
        values = std::move(sortedValues);
    }

    

    static CooMatrix<T> from_sms_file(const std::string& filename, T prime) {
        std::ifstream file(filename);  // Make sure to include <fstream>
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            exit(-1);
        }
    
        char arbitraryChar;
        int numRows, numCols;
        file >> numRows >> numCols >> arbitraryChar;
    
        std::vector<int> tempRowIndices;
        std::vector<int> tempColIndices;
        std::vector<T> tempValues;
    
        int row, col;
        int value;
        while (file >> row >> col >> value) {
            if (row>0 && col>0) {
                tempRowIndices.push_back(row-1); // sms file is 1-based, we use 0-base
                tempColIndices.push_back(col-1);
                if (value >=0 || prime ==0) 
                {
                    tempValues.push_back((T) value);
                }
                else {
                    tempValues.push_back((T) (((value % prime) + prime)%prime));
                }
            }
        }
    
        nnz = tempRowIndices.size();
        rowIndices = std::move(tempRowIndices);
        colIndices = std::move(tempColIndices);
        values = std::move(tempValues);
    
        file.close();

        return CooMatrix<T>(numRows, numCols, rowIndices, colIndices, values);
    }
};



// void load_sms_matrix(const std::string& filename, std::vector<int>& rowIndices, std::vector<int>& colIndices, std::vector<myfloat>& values, int& numRows, int& numCols, int& nnz) {
//     std::ifstream file(filename);  // Make sure to include <fstream>
//     if (!file) {
//         std::cerr << "Failed to open file: " << filename << std::endl;
//         exit(-1);
//     }

//     char arbitraryChar;
//     file >> numRows >> numCols >> arbitraryChar;

//     std::vector<int> tempRowIndices;
//     std::vector<int> tempColIndices;
//     std::vector<myfloat> tempValues;

//     int row, col;
//     myfloat value;
//     while (file >> row >> col >> value) {
//         if (row>0 && col>0) {
//             tempRowIndices.push_back(row-1);
//             tempColIndices.push_back(col-1);
//             tempValues.push_back(value);
//         }
//     }

//     nnz = tempRowIndices.size();
//     rowIndices = std::move(tempRowIndices);
//     colIndices = std::move(tempColIndices);
//     values = std::move(tempValues);

//     file.close();
// }

// void coo_matrix_to_csr(int numRows, const std::vector<int>& rowIndices, const std::vector<int>& colIndices, const std::vector<myfloat>& values,
//                        std::vector<int>& csrOffsets, std::vector<int>& csrColumns, std::vector<myfloat>& csrValues) {
//     csrOffsets.resize(numRows + 1, 0);
//     csrColumns.resize(values.size());
//     csrValues.resize(values.size());

//     std::vector<int> rowCount(numRows, 0);
//     for (int i = 0; i < rowIndices.size(); ++i) {
//         rowCount[rowIndices[i]]++;
//     }

//     csrOffsets[0] = 0;
//     for (int i = 1; i <= numRows; ++i) {
//         csrOffsets[i] = csrOffsets[i - 1] + rowCount[i - 1];
//     }

//     csrColumns = colIndices;
//     csrValues = values;
    
//     // std::vector<int> tempOffsets = csrOffsets;
//     // for (int i = 0; i < rowIndices.size(); ++i) {
//     //     int row = rowIndices[i];
//     //     int destIndex = tempOffsets[row]++;
//     //     csrColumns[destIndex] = colIndices[i];
//     //     csrValues[destIndex] = values[i];
//     // }
// }

template<typename T>
void display_vector(const std::vector<T>& vec, T prime, int max_elements = 10) {
    std::cout << "Vector: ";
    for (int i = 0; i < std::min(max_elements, (int)vec.size()); ++i) {
        std::cout << (vec[i]>=0?vec[i]:vec[i]+prime) << " ";
    }
    std::cout << std::endl;
}

template<typename T>
std::vector<std::vector<T>> reshape_to_vector_of_vectors(const std::vector<T>& input, int N) {
    if (input.size() % N != 0) {
        throw std::invalid_argument("Input size is not divisible by N");
    }

    int num_vectors = input.size() / N;
    std::vector<std::vector<T>> result(num_vectors, std::vector<T>(N));

    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < N; ++j) {
            result[i][j] = input[j * num_vectors + i];
        }
    }

    return result;
}


template<typename T>
std::vector<T> generate_random_vector(int size, T prime, bool only_nonzero=false) {
    std::vector<T> random_vector(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dis(only_nonzero? 1 : 0, prime - 1);

    for (int i = 0; i < size; ++i) {
        random_vector[i] = dis(gen);
    }

    return random_vector;
}


template <typename T>
std::vector<T> prettify_vect(const std::vector<T>& vec, T theprime) {
    std::vector<T> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = (vec[i] % theprime + theprime) % theprime;
    }
    return result;
}


#endif // MATRICES_H