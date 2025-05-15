#ifndef SIGMA_BASIS_H
#define SIGMA_BASIS_H


#include <vector>
#include <iostream>
#include <chrono>
// #include <iomanip>
#include "poly_mat_mul.h"
#include "modular_linalg.h"
#include "ntt.h"


template<typename T>
vector<T> vecvecvec_to_vec(const vector<vector<vector<T>>>& vec) {
    size_t m = vec.size();
    size_t n = vec[0].size();
    size_t k = vec[0][0].size();
    vector<T> result(m * n * k,0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < k; ++l) {
                result[l*m*n + i*n+j] = vec[i][j][l];
            }
        }
    }
    return result;
}

template<typename T>
void display_vector(const std::vector<T>& vec, T prime, int max_elements = 10) {
    std::cout << "Vector: ";
    for (int i = 0; i < std::min(max_elements, (int)vec.size()); ++i) {
        std::cout << (vec[i]>=0?vec[i]:vec[i]+prime) << " ";
    }
    std::cout << std::endl;
}
template<typename T>
void display_vecvecvec(const std::vector<std::vector<std::vector<T>>>& vec, T prime, int max_elements = 10) {
    auto v = vecvecvec_to_vec(vec);
    display_vector<T>(v, prime, max_elements);
}

template <typename S>
std::vector<std::vector<std::vector<S>>> unit_mat(std::size_t n) {
    std::vector<std::vector<std::vector<S>>> mat(n, std::vector<std::vector<S>>(n, std::vector<S>(1, S(0))));
    for (std::size_t i = 0; i < n; ++i) {
        mat[i][i][0] = S(1);
    }
    return mat;
}


template <typename S>
void shift_trunc_in(std::vector<std::vector<std::vector<S>>>& mat, std::size_t shiftd, std::size_t truncd) {
    std::size_t m = mat.size();
    std::size_t n = mat[0].size();
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k <= truncd; ++k) {
                mat[i][j][k] = mat[i][j][k + shiftd];
            }
            mat[i][j].resize(truncd + 1, S(0));
        }
    }
}


struct ProgressData {
    std::size_t total;
    std::size_t current;
    std::chrono::time_point<std::chrono::steady_clock> start_time;

    ProgressData(std::size_t total)
        : total(total), current(0), start_time(std::chrono::steady_clock::now()) {}

    void progress_tick() {
        ++current;
        if (current % 100 == 0) {
            auto elapsed_time = std::chrono::steady_clock::now() - start_time;
            double percent = (static_cast<double>(current) / total) * 100.0;
            std::cout << "\rSigma basis progress: " << std::fixed << std::setprecision(2)
                      << percent << "% (" << current << " of " << total
                      << "), Time elapsed: " << std::chrono::duration_cast<std::chrono::seconds>(elapsed_time).count()
                      << "s    " << std::flush;
        }
    }
};


template <typename S>
std::pair<std::vector<std::vector<std::vector<S>>>, std::vector<int64_t>> M_Basis(
    const std::vector<std::vector<std::vector<S>>>& G, 
    std::vector<int64_t> delta, 
    S prime) 
{
    std::size_t m = G.size();
    assert(m > 0 && "Must have #rows > 0");
    std::size_t n = G[0].size();
    assert(m >= n && "Must have #rows >= #cols");

    // Sort delta in descending order and remember the permutation
    std::vector<std::size_t> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&delta](std::size_t a, std::size_t b) {
        return delta[a] > delta[b];
    });
    std::sort(delta.begin(), delta.end(), std::greater<int64_t>());

    // Create permutation matrix pi_m
    DMatrix<S> pi_m(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        pi_m(i,perm[i]) = S(1);
    }

    // Construct Delta matrix
    DMatrix<S> Delta(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Delta(i,j) = G[i][j][0];
        }
    }
    Delta = modular_mat_mul(pi_m, Delta, prime);

    // Augment Delta into Delta_aug
    DMatrix<S> Delta_aug(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Delta_aug(i,j) = Delta(i,j);
        }
    }

    // Perform LSP decomposition
    auto [L, S_mat, P] = lsp_decomposition(Delta_aug, prime);
    auto Linv = matrix_inverse(L, prime);

    // Construct D1 and Dx
    DMatrix<S> D1(m, m);
    DMatrix<S> Dx(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        auto row = S_mat.row(i);
        bool is_zero_row = std::all_of(row.begin(), row.end(), [](S x) { return x == S(0); });
        if (is_zero_row) {
            D1(i,i) = S(1);
        } else {
            Dx(i,i) = S(1);
        }
    }

    // Compute M1 and Mx
    auto M1 = modular_mat_mul(modular_mat_mul(D1, Linv, prime), pi_m, prime);
    auto Mx = modular_mat_mul(modular_mat_mul(Dx, Linv, prime), pi_m, prime);

    // Fill output matrix
    std::vector<std::vector<std::vector<S>>> ret(m, std::vector<std::vector<S>>(m, std::vector<S>(2, S(0))));
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            ret[i][j][0] = M1(i,j);
            ret[i][j][1] = Mx(i,j);
        }
    }

    // Update delta
    for (std::size_t i = 0; i < m; ++i) {
        if (Dx(i,i) == S(1)) {
            delta[i] -= 1;
        }
    }

    return {ret, delta};
}


template <typename S>
std::pair<std::vector<std::vector<std::vector<S>>>, std::vector<int64_t>> _PM_Basis(
    const std::vector<std::vector<std::vector<S>>>& G, 
    std::size_t d, 
    const std::vector<int64_t>& delta, 
    S seqprime, 
    ProgressData& progress) 
{
    std::size_t n = G.size();

    if (d == 0) {
        return {unit_mat<S>(n), delta};
    } else if (d == 1) {
        progress.progress_tick();
        auto [MM, mumu] = M_Basis(G, delta, seqprime);
        // cout << "_PM (" <<d<<") in:" << endl;
        // display_vecvecvec<S>(G, seqprime, 32);
        // cout << "_PM (" <<d<<") out:" << endl;
        // display_vecvecvec<S>(MM, seqprime, 32);
        return {MM, mumu};
    } else {
        auto [MM, mumu] = _PM_Basis(G, d / 2, delta, seqprime, progress);

        auto GG = poly_mat_mul_fft_red(MM, G, seqprime, 0, d + 1);
        shift_trunc_in(GG, d / 2, d / 2);

        auto [MMM, mumumu] = _PM_Basis(GG, d / 2, mumu, seqprime, progress);
        auto rret = poly_mat_mul_fft_red(MMM, MM, seqprime, 0, MM[0][0].size());
        // return {poly_mat_mul_fft_red(MMM, MM, seqprime, 0, MM[0][0].size()), mumumu};

        // cout << "_PM (" <<d<<") in:" << endl;
        // display_vecvecvec<S>(G, seqprime, 32);
        // cout << "_PM (" <<d<<") out:" << endl;
        // display_vecvecvec<S>(rret, seqprime, 32);

        return {rret, mumumu};
    }
}


template <typename S, typename T>
std::pair<std::vector<std::vector<std::vector<S>>>, std::vector<int64_t>> PM_Basis(
    std::vector<std::vector<T>>& seq, 
    std::size_t d, 
    T seqprime) 
{
    // Expand sequence to square matrix
    std::size_t nn = seq.size();
    std::size_t n = 0;
    std::size_t nlen = seq[0].size();
    for (std::size_t i = 0; i < nn; ++i) {
        if (i * (i + 1) / 2 == nn) {
            n = i;
        }
    }

    assert(n * (n + 1) / 2 == nn && "Sequence length does not match square matrix size");

    // Enlarge d to a power of 2
    std::size_t dd = 1;
    while (dd < d) {
        dd *= 2;
    }
    d = dd;

    assert(d < nlen && "Sequence too short");

    // Prepare input data. Also add a unit matrix of size nxn below the matrix
    std::vector<std::vector<std::vector<S>>> G(2 * n, std::vector<std::vector<S>>(n, std::vector<S>(nlen, S(0))));
    std::size_t ii = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            for (std::size_t k = 0; k < nlen; ++k) {
                G[i][j][k] = static_cast<S>(seq[ii][k]);
                G[j][i][k] = G[i][j][k];
            }
            ++ii;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        G[n + i][i][0] = S(1);
    }

    std::vector<int64_t> delta(2 * n, 0);

    ProgressData progress(d);
    auto [M, mu] = _PM_Basis(G, d, delta, static_cast<S>(seqprime), progress);

    return {M, mu};
}

template <typename S>
void analyze_min_generators(const std::vector<std::vector<std::vector<S>>>& poly, S p) {
    std::size_t n = poly[0].size();
    std::size_t k = poly.size();
    std::cout << "Analyzing min generators ... " << poly[0][0].size() << " found of max degree " << poly.size() << std::endl;

    std::cout << "Individual generator degrees..." << std::endl;
    for (std::size_t j = 0; j < poly[0][0].size(); ++j) {
        std::size_t deg = 0;
        for (std::size_t i = 0; i < poly.size(); ++i) {
            bool non_zero_found = false;
            for (std::size_t row = 0; row < poly[i].size(); ++row) {
                if (poly[i][row][j] != S(0)) {
                    non_zero_found = true;
                    break;
                }
            }
            if (non_zero_found) {
                deg = i;
            }
        }
        std::cout << "Generator " << j << " has degree: " << deg << std::endl;
    }

    // Estimate matrix rank
    auto tmodular_rank = [](const std::vector<std::vector<S>>& mat, S prime) {
        DMatrix<S> dmat(mat.size(), mat[0].size());
        for (std::size_t i = 0; i < mat.size(); ++i) {
            for (std::size_t j = 0; j < mat[i].size(); ++j) {
                dmat(i, j) = mat[i][j];
            }
        }

        return modular_rank(dmat, prime); // Assuming modular_matrix_rank is implemented elsewhere
    };

    std::size_t matrank = n * (k - 3) + tmodular_rank(poly[k - 1], p) + tmodular_rank(poly[0], p);
    std::cout << "Estimated matrix rank: " << matrank << std::endl;
}

template <typename S>
void analyze_pm_basis(const std::vector<std::vector<std::vector<S>>>& basis, const std::vector<int64_t>& delta, S p) {
    // Analyze the basis
    std::size_t n = basis.size();
    assert(n == basis[0].size() && "Matrix of basis vectors must be square");
    std::size_t nlen = basis[0][0].size();

    std::cout << "Analyzing min generators ... " << n << " found of max degree " << nlen << std::endl;

    std::cout << "Individual generator degrees..." << std::endl;
    for (std::size_t j = 0; j < n; ++j) {
        const auto& g = basis[j];
        std::size_t deg = 0;
        std::size_t val = std::numeric_limits<std::size_t>::max();
        for (std::size_t i = 0; i < nlen; ++i) {
            bool non_zero_found = std::any_of(g.begin(), g.end(), [i](const std::vector<S>& row) { return row[i] != S(0); });
            if (non_zero_found) {
                deg = i;
                if (val == std::numeric_limits<std::size_t>::max()) {
                    val = i;
                }
            }
        }
        std::cout << "Generator " << j << " has degree: " << deg << " and valuation " << val << " and delta " << delta[j] << std::endl;
    }

    // Extract reversed basis for original sequence
    std::size_t m = n / 2;
    std::vector<std::vector<std::vector<S>>> basisrev(m, std::vector<std::vector<S>>(m, std::vector<S>(nlen, S(0))));
    std::size_t maxdel = 0;
    for (std::size_t i = 0; i < m; ++i) {
        std::size_t del = static_cast<std::size_t>(-delta[i]);
        maxdel = std::max(maxdel, del);
        for (std::size_t j = 0; j < m; ++j) {
            for (std::size_t k = 0; k <= del; ++k) {
                basisrev[i][j][k] = basis[i][j][del - k];
            }
        }
    }

    std::vector<std::vector<std::vector<S>>> basisrev2(maxdel + 1, std::vector<std::vector<S>>(m, std::vector<S>(m, S(0))));
    std::size_t maxdeg = 0;
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            for (std::size_t k = 0; k <= maxdel; ++k) {
                basisrev2[k][i][j] = basisrev[i][j][k];
                if (basisrev[i][j][k] != S(0) && k > maxdeg) {
                    maxdeg = k;
                }
            }
        }
    }
    basisrev2.resize(maxdeg + 1);

    analyze_min_generators(basisrev2, p);
}

#endif // SIGMA_BASIS_H