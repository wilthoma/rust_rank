#ifndef MODULAR_LINALG_H
#define MODULAR_LINALG_H

#include <iostream>
#include <vector>
#include <stdexcept>
#include <tuple>
#include <optional>
#include <algorithm>
#include <cstdint>
#include <cmath>
#include <cassert>
#include <numeric>

typedef unsigned __int128 uint128_t;

template <typename T>
struct DMatrix {
    size_t rows;
    size_t cols;
    std::vector<T> data;

    DMatrix(size_t r, size_t c) : rows(r), cols(c), data(r * c, 0) {}

    T& operator()(size_t i, size_t j) {
        return data[i * cols + j];
    }

    const T& operator()(size_t i, size_t j) const {
        return data[i * cols + j];
    }

    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T entry = (*this)(i, j);
                std::cout << static_cast<long>(entry) << " ";
            }
            std::cout << std::endl;
        }
    }
    
};

// Helper function to compute modular inverse
template <typename T>
T modinv(T a, T p) {
    // Compute modular inverse using modular exponentiation
    // a^(p-2) ≡ a^(-1) (mod p) when p is prime
    auto modpow = [](T base, T exp, T mod) -> T {
        T result = 1;
        base = base % mod;
        while (exp > 0) {
            if (exp % 2 == 1) {
                result = (result * base) % mod;
            }
            base = (base * base) % mod;
            exp /= 2;
        }
        return result;
    };
    return modpow(a, p - 2, p);
}


template <typename T>
DMatrix<T> matrix_inverse(const DMatrix<T>& mat, T p) {
    size_t n = mat.rows;
    if (n != mat.cols) {
        throw std::invalid_argument("Matrix must be square");
    }

    // Augment the matrix with the identity matrix
    DMatrix<T> aug(n, 2 * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            aug(i, j) = mat(i, j) % p;
        }
        aug(i, n + i) = 1;
    }

    // Perform Gaussian elimination
    for (size_t i = 0; i < n; ++i) {
        // Find pivot
        if (aug(i, i) == 0) {
            // Try to swap with a lower row
            bool found = false;
            for (size_t j = i + 1; j < n; ++j) {
                if (aug(j, i) != 0) {
                    for (size_t k = 0; k < 2 * n; ++k) {
                        std::swap(aug(i, k), aug(j, k));
                    }
                    found = true;
                    break;
                }
            }
            if (!found) {
                throw std::runtime_error("Matrix is singular and not invertible");
            }
        }

        // Normalize pivot row
        T inv = modinv(aug(i, i), p);
        for (size_t k = 0; k < 2 * n; ++k) {
            aug(i, k) = (aug(i, k) * inv) % p;
        }

        // Eliminate other rows
        for (size_t j = 0; j < n; ++j) {
            if (j != i) {
                T factor = aug(j, i);
                for (size_t k = 0; k < 2 * n; ++k) {
                    T tmp = (aug(i, k) * factor) % p;
                    aug(j, k) = (aug(j, k) + p - tmp) % p;
                }
            }
        }
    }

    // Extract inverse matrix
    DMatrix<T> inv_mat(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            inv_mat(i, j) = aug(i, n + j);
        }
    }

    return inv_mat;
}


template <typename T>
T modular_determinant(DMatrix<T> mat, T p) {
    if (mat.rows != mat.cols) {
        throw std::invalid_argument("Matrix must be square to compute determinant mod p");
    }

    size_t n = mat.rows;
    T det = 1;
    int64_t sign = 1;

    // Reduce all entries mod p
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            mat(i, j) = (mat(i, j) % p + p) % p;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        // 1. Find pivot
        size_t pivot_row = i;
        while (pivot_row < n && mat(pivot_row, i) == 0) {
            ++pivot_row;
        }

        if (pivot_row == n) {
            // Pivot = 0 → determinant is 0
            return 0;
        }

        if (pivot_row != i) {
            // Swap rows
            for (size_t col = 0; col < n; ++col) {
                std::swap(mat(i, col), mat(pivot_row, col));
            }
            sign = -sign;
        }

        T pivot = mat(i, i);
        det = (det * pivot) % p;

        T inv = modinv(pivot, p);

        // 2. Eliminate rows below
        for (size_t row = i + 1; row < n; ++row) {
            T factor = (mat(row, i) * inv) % p;
            for (size_t col = i; col < n; ++col) {
                T sub = (factor * mat(i, col)) % p;
                mat(row, col) = (mat(row, col) + p - sub) % p;
            }
        }
    }

    if (sign == -1) {
        det = (p - det) % p;
    }

    return det;
}


template <typename T>
size_t modular_rank(DMatrix<T> mat, T p) {
    size_t nrows = mat.rows;
    size_t ncols = mat.cols;
    size_t rank = 0;

    // Reduce entries modulo p
    for (size_t i = 0; i < nrows; ++i) {
        for (size_t j = 0; j < ncols; ++j) {
            mat(i, j) = (mat(i, j) % p + p) % p;
        }
    }

    for (size_t col = 0; col < ncols; ++col) {
        // 1. Find a pivot row
        size_t pivot_row = rank;
        while (pivot_row < nrows && mat(pivot_row, col) == 0) {
            ++pivot_row;
        }

        if (pivot_row < nrows) {
            // 2. Swap pivot row into position
            if (pivot_row != rank) {
                for (size_t j = 0; j < ncols; ++j) {
                    std::swap(mat(rank, j), mat(pivot_row, j));
                }
            }

            // 3. Normalize pivot row
            T inv = modinv(mat(rank, col), p);
            for (size_t j = col; j < ncols; ++j) {
                mat(rank, j) = (mat(rank, j) * inv) % p;
            }

            // 4. Eliminate rows below
            for (size_t row = rank + 1; row < nrows; ++row) {
                T factor = mat(row, col);
                for (size_t j = col; j < ncols; ++j) {
                    T sub = (factor * mat(rank, j)) % p;
                    mat(row, j) = (mat(row, j) + p - sub) % p;
                }
            }

            ++rank;
            if (rank == nrows) {
                break;
            }
        }
    }

    return rank;
}

template <typename T>
std::tuple<DMatrix<T>, DMatrix<T>, DMatrix<T>> lsp_decomposition(const DMatrix<T>& a, T p) {
    // std::cout << "LSP decomposition started..." << std::endl;
    DMatrix<T> mat = a;
    size_t m = mat.rows;
    if (m != mat.cols) {
        throw std::invalid_argument("Matrix must be square");
    }

    DMatrix<T> l(m, m);
    for (size_t i = 0; i < m; ++i) {
        l(i, i) = 1; // Initialize L as the identity matrix
    }
    // l.print();

    std::vector<size_t> perm(m);
    for (size_t i = 0; i < m; ++i) {
        perm[i] = i;
    }
    // std::cout << "Permutation vector initialized." << std::endl;

    size_t rank = 0;

    for (size_t k = 0; k < m; ++k) {
        // Find pivot in row k (column-wise search)
        std::optional<size_t> pivot_col;
        for (size_t j = rank; j < m; ++j) {
            if (mat(k, j) % p != 0) {
                pivot_col = j;
                break;
            }
        }
        // std::cout << "Pivot column found: " << (pivot_col.has_value() ? std::to_string(pivot_col.value()) : "none") << std::endl;
        if (pivot_col.has_value()) {
            size_t j = pivot_col.value();

            // Swap columns rank <-> j in mat and perm
            for (size_t i = 0; i < m; ++i) {
                std::swap(mat(i, rank), mat(i, j));
            }
            std::swap(perm[rank], perm[j]);
            // std::cout << "Columns swapped: " << rank << " <-> " << j << std::endl;
            // Eliminate below
            T pivot_inv = modinv(mat(k, rank), p);
            for (size_t i = k + 1; i < m; ++i) {
                T factor = (mat(i, rank) * pivot_inv) % p;
                l(i, k) = factor;
                for (size_t col = rank; col < m; ++col) {
                    T sub = (factor * mat(k, col)) % p;
                    mat(i, col) = (mat(i, col) + p - sub) % p;
                }
            }
            // std::cout << "Pivot found at row " << k << ", column " << rank << std::endl;
            ++rank;
        }
    }

    // Build the permutation matrix P
    DMatrix<T> pmat(m, m);
    for (size_t i = 0; i < m; ++i) {
        pmat(i, perm[i]) = 1;
    }

    // Final modular reduction
    DMatrix<T> s(m, m);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < m; ++j) {
            s(i, j) = mat(i, j) % p;
        }
    }

    return std::make_tuple(l, s, pmat);
}

// ****** TEST CODE below ******

template <typename T>
bool is_unit_lower_triangular(const DMatrix<T>& l, T p) {
    size_t n = l.rows;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T val = l(i, j) % p;
            if (i == j && val != 1) {
                return false;
            } else if (i < j && val != 0) {
                return false;
            }
        }
    }
    return true;
}

template <typename T>
bool is_valid_s_matrix(const DMatrix<T>& s, T p) {
    size_t m = s.rows;

    // Count nonzero rows from the top
    size_t rank = 0;
    for (size_t i = 0; i < m; ++i) {
        // Check that all entries to the left are zero
        for (size_t j = 0; j < rank; ++j) {
            if (s(i, j) % p != 0) {
                return false;
            }
        }
        // If row is nonzero, then rank entry must be nonzero
        bool row_nonzero = false;
        for (size_t j = 0; j < s.cols; ++j) {
            if (s(i, j) % p != 0) {
                row_nonzero = true;
                break;
            }
        }
        if (row_nonzero) {
            if (s(i, rank) % p == 0) {
                return false;
            }
            ++rank;
        }
    }

    return true;
}

template <typename T>
void check_lsp(const DMatrix<T>& a, T p) {
    auto [l, s, pmat] = lsp_decomposition(a, p);

    // Reconstruct the original matrix
    DMatrix<T> reconstructed(a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            T sum = 0;
            for (size_t k = 0; k < a.cols; ++k) {
                for (size_t m = 0; m < a.cols; ++m) {
                    sum = (sum + l(i, k) * s(k, m) * pmat(m, j)) % p;
                }
            }
            reconstructed(i, j) = sum;
        }
    }

    // Reduce original matrix mod p
    DMatrix<T> original_modp(a.rows, a.cols);
    for (size_t i = 0; i < a.rows; ++i) {
        for (size_t j = 0; j < a.cols; ++j) {
            original_modp(i, j) = a(i, j) % p;
        }
    }

    // Assertions
    assert(original_modp.data == reconstructed.data && "Reconstructed matrix does not match original");
    assert(is_unit_lower_triangular(l, p) && "L is not unit lower triangular");
    assert(is_valid_s_matrix(s, p) && "S matrix is not in expected form");
}

void test_lsp_decomposition_small() {
    uint128_t p = 101;
    DMatrix<uint128_t> a(3, 3);
    a(0, 0) = 2; a(0, 1) = 4; a(0, 2) = 6;
    a(1, 0) = 1; a(1, 1) = 3; a(1, 2) = 5;
    a(2, 0) = 0; a(2, 1) = 0; a(2, 2) = 1;
    check_lsp(a, p);
}

void test_lsp_decomposition_larger() {
    uint128_t p = 97;
    DMatrix<uint128_t> a(5, 5);
    a(0, 0) = 10; a(0, 1) = 20; a(0, 2) = 30; a(0, 3) = 40; a(0, 4) = 50;
    a(1, 0) =  5; a(1, 1) = 10; a(1, 2) = 15; a(1, 3) = 20; a(1, 4) = 25;
    a(2, 0) =  0; a(2, 1) =  1; a(2, 2) =  0; a(2, 3) =  1; a(2, 4) =  0;
    a(3, 0) =  1; a(3, 1) =  0; a(3, 2) =  1; a(3, 3) =  0; a(3, 4) =  1;
    a(4, 0) = 99; a(4, 1) = 99; a(4, 2) = 99; a(4, 3) = 99; a(4, 4) = 99; // mod 97 => 2
    check_lsp(a, p);
}

template <typename T>
bool is_identity_mod(const DMatrix<T>& mat, T p) {
    size_t n = mat.rows;
    if (mat.cols != n) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            T val = mat(i, j) % p;
            if ((i == j && val != 1) || (i != j && val != 0)) {
                return false;
            }
        }
    }
    return true;
}

void test_modular_inverse_small_matrix() {
    uint128_t p = 97;
    DMatrix<uint128_t> a(3, 3);
    a(0, 0) = 2; a(0, 1) = 3; a(0, 2) = 1;
    a(1, 0) = 1; a(1, 1) = 1; a(1, 2) = 1;
    a(2, 0) = 3; a(2, 1) = 5; a(2, 2) = 2;

    DMatrix<uint128_t> inv = matrix_inverse(a, p);

    DMatrix<uint128_t> product(3, 3);
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            uint128_t sum = 0;
            for (size_t k = 0; k < 3; ++k) {
                sum = (sum + a(i, k) * inv(k, j)) % p;
            }
            product(i, j) = sum;
        }
    }

    assert(is_identity_mod(product, p) && "A * A⁻¹ != I mod p");
}


void test_modular_inverse_identity() {
    uint128_t p = 97;
    DMatrix<uint128_t> a(4, 4);
    for (size_t i = 0; i < 4; ++i) {
        a(i, i) = 1; // Set identity matrix
    }

    DMatrix<uint128_t> inv = matrix_inverse(a, p);

    assert(inv.data == a.data && "Inverse of identity should be identity");
}

void test_modinv_basic() {
    uint64_t p = 97;

    for (uint64_t a = 1; a < p; ++a) {
        uint64_t inv = modinv(a, p);
        assert((a * inv) % p == 1 && "modinv(a, p) * a != 1 mod p");
    }
}

void test_runall_modular_linalg() {
    std::cout << "Running modular linalg tests..." << std::endl;
    test_modular_inverse_identity();
    std::cout << "Passed modular inverse identity test." << std::endl;
    test_modinv_basic();
    std::cout << "Passed basic modular inverse test." << std::endl;
    test_lsp_decomposition_small();
    std::cout << "Passed small LSP decomposition test." << std::endl;
    test_lsp_decomposition_larger();
    std::cout << "Passed larger LSP decomposition test." << std::endl;
    test_modular_inverse_small_matrix();
    std::cout << "Passed modular inverse small matrix test." << std::endl;
    std::cout << "All modular linalg tests passed!" << std::endl;
}

#endif // MODULAR_LINALG_H