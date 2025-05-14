#ifndef CUSIGMA_BASIS_H
#define CUSIGMA_BASIS_H

#include <vector>
#include <iostream>
#include <chrono>
// #include <iomanip>
#include "cupoly_mat_mul.h"
#include "modular_linalg.h"
#include "ntt.h"
#include "sigma_basis.h"


// the input is a matrix G of size m x m/2
// the output is the two DMatrices of size m x m (the constant coeff and x-coeff) and the new delta
std::tuple<DMatrix<u64>, DMatrix<u64>, std::vector<int64_t>> M_Basis2(
    const DMatrix<u64>& G, 
    std::vector<int64_t> delta, 
    u64 prime) 
{
    std::size_t m = G.rows;
    std::size_t n = G.cols;
    // assert(n > 0 && "Must have #rows > 0");
    // std::size_t n = G[0].size();
    // assert(m >= n && "Must have #rows >= #cols");

    // Sort delta in descending order and remember the permutation
    std::vector<std::size_t> perm(m);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&delta](std::size_t a, std::size_t b) {
        return delta[a] > delta[b];
    });
    std::sort(delta.begin(), delta.end(), std::greater<int64_t>());

    // Create permutation matrix pi_m
    DMatrix<u64> pi_m(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        pi_m(i,perm[i]) = 1;
    }

    // Construct Delta matrix
    DMatrix<u64> Delta(m, n);
    // for (std::size_t i = 0; i < m; ++i) {
    //     for (std::size_t j = 0; j < n; ++j) {
    //         Delta(i,j) = G[i][j][0];
    //     }
    // }
    Delta = modular_mat_mul(pi_m, G, prime);

    // Augment Delta into Delta_aug
    DMatrix<u64> Delta_aug(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            Delta_aug(i,j) = Delta(i,j);
        }
    }

    // Perform LSP decomposition
    auto [L, S_mat, P] = lsp_decomposition(Delta_aug, prime);
    auto Linv = matrix_inverse(L, prime);

    // Construct D1 and Dx
    DMatrix<u64> D1(m, m);
    DMatrix<u64> Dx(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        auto row = S_mat.row(i);
        bool is_zero_row = std::all_of(row.begin(), row.end(), [](u64 x) { return x == 0; });
        if (is_zero_row) {
            D1(i,i) = 1;
        } else {
            Dx(i,i) = 1;
        }
    }

    // Compute M1 and Mx
    auto M1 = modular_mat_mul(modular_mat_mul(D1, Linv, prime), pi_m, prime);
    auto Mx = modular_mat_mul(modular_mat_mul(Dx, Linv, prime), pi_m, prime);

    // Fill output matrix
    // std::vector<std::vector<std::vector<S>>> ret(m, std::vector<std::vector<S>>(m, std::vector<S>(2, S(0))));
    // for (std::size_t i = 0; i < m; ++i) {
    //     for (std::size_t j = 0; j < m; ++j) {
    //         ret[i][j][0] = M1(i,j);
    //         ret[i][j][1] = Mx(i,j);
    //     }
    // }

    // Update delta
    for (std::size_t i = 0; i < m; ++i) {
        if (Dx(i,i) == 1) {
            delta[i] -= 1;
        }
    }

    return {M1, Mx, delta};
}



// the input matrix G is of size nxn/2 and seq. length (at least) d,
// the output Res is of size nxn and seq. length d+1
std::vector<int64_t> _cuPM_Basis(
    u64* dG,
    u64* dRes,
    size_t n, 
    size_t d, 
    const std::vector<int64_t>& delta, 
    u64 seqprime, 
    ProgressData& progress) 
{

    if (d == 0) {
        // TODO: fill output buffer with unit matrix
        std::vector<u64> res(n * n, 0);
        for (std::size_t i = 0; i < n; ++i) {
            res[i*n+i] = 1;
        }
        CHECK_CUDA(cudaMemcpy(dRes, res.data(), res.size() * sizeof(u64), cudaMemcpyHostToDevice));
        return delta;
    } else if (d == 1) {
        progress.progress_tick();
        // extract leading coefficient
        std::vector<u64> A(n * n/2, 0);
        CHECK_CUDA(cudaMemcpy(A.data(), dG, A.size() * sizeof(u64), cudaMemcpyDeviceToHost));
        DMatrix<u64> mA(n, n/2);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n/2; ++j) {
                mA(i,j) = A[i*n/2+j];
            }
        }
        auto [MM, MMM, mumu] = M_Basis2(mA, delta, seqprime);
        // copy back to device (todo)
        vector<u64> res(n * n * 2, 0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                res[i*n+j] = MM(i,j);
                res[n*n+i*n+j] = MMM(i,j);
            }
        }
        CHECK_CUDA(cudaMemcpy(dRes, res.data(), res.size() * sizeof(u64), cudaMemcpyHostToDevice));

        return mumu;
    } else {
        u64* dout1;
        CHECK_CUDA(cudaMalloc(&dout1, n * n * (d/2+1) * sizeof(u64)));
        auto mumu = _cuPM_Basis(dG, dout1, n, d / 2, delta, seqprime, progress);

        u64* mul1;
        CHECK_CUDA(cudaMalloc(&mul1, n * n * (d/2) * sizeof(u64)));
        // auto GG = poly_mat_mul_fft_red(MM, G, seqprime, 0, d + 1);
        cupoly_mat_mul_fft_gpu(dout1, dG, mul1, n, n, n/2, d/2+1, d, d/2, d/2);
        modp_buffer(mul1, n * (n/2) * (d/2), seqprime);
        // shift_trunc_in(GG, d / 2, d / 2);

        u64* dout2;
        CHECK_CUDA(cudaMalloc(&dout2, n * n * (d/2+1) * sizeof(u64)));

        auto mumumu = _cuPM_Basis(mul1, dout2, n, d / 2, mumu, seqprime, progress);

        cupoly_mat_mul_fft_gpu(dout1, dout2, dRes, n, n, n, d/2+1, d/2+1, 0, d+1);
        modp_buffer(dRes, n * n * (d+1), seqprime);

        CHECK_CUDA(cudaFree(dout1));
        CHECK_CUDA(cudaFree(mul1));
        CHECK_CUDA(cudaFree(dout2));
        
        return mumumu;
        // return {poly_mat_mul_fft_red(MMM, MM, seqprime, 0, MM[0][0].size()), mumumu};
    }
}


template <typename T>
std::pair<std::vector<std::vector<std::vector<u64>>>, std::vector<int64_t>> cuPM_Basis(
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

    // Enlarge (=weaken) max degree estimate d to a power of 2
    std::size_t dd = 1;
    while (dd < d) {
        dd *= 2;
    }
    d = dd; 

    assert(d < nlen && "Sequence too short");

    // Prepare input data. Also add a unit matrix of size nxn below the matrix
    //std::vector<std::vector<std::vector<S>>> G(2 * n, std::vector<std::vector<S>>(n, std::vector<S>(nlen, S(0))));
    std::vector<u64> G(2 * n * n * nlen, 0);
    std::size_t ii = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i; j < n; ++j) {
            for (std::size_t k = 0; k < nlen; ++k) {
                G[k*2*n*n+i*n+j] = static_cast<u64>(seq[ii][k]);
                G[k*2*n*n+j*n+i] = G[k*2*n*n+i*n+j];
                // G[i][j][k] = static_cast<S>(seq[ii][k]);
                // G[j][i][k] = G[i][j][k];
            }
            ++ii;
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        G[n*(n + i)+i] = 1;
    }
    u64 *dG;
    CHECK_CUDA(cudaMalloc(&dG, G.size() * sizeof(u64)));
    CHECK_CUDA(cudaMemcpy(dG, G.data(), G.size() * sizeof(u64), cudaMemcpyHostToDevice));

    std::vector<int64_t> delta(2 * n, 0);

    // output buffer
    std::vector<u64> M(2 * n * 2 * n * (d+1), 0);
    u64 *dM;
    CHECK_CUDA(cudaMalloc(&dM, M.size() * sizeof(u64)));

    ProgressData progress(d);
    auto mu = _cuPM_Basis(dG, dM, 2*n, d, delta, static_cast<u64>(seqprime), progress);

    CHECK_CUDA(cudaMemcpy(M.data(), dM, M.size() * sizeof(u64), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dG));
    CHECK_CUDA(cudaFree(dM));

    // Convert M to 3D vector
    std::vector<std::vector<std::vector<u64>>> M3D(2 * n, std::vector<std::vector<u64>>(2 * n, std::vector<u64>(d+1, 0)));
    for (std::size_t i = 0; i < 2 * n; ++i) {
        for (std::size_t j = 0; j < 2 * n; ++j) {
            for (std::size_t k = 0; k < d+1; ++k) {
                M3D[i][j][k] = M[k*2*n*2*n+i*2*n+j];
            }
        }
    }
    return {M3D, mu};
}






#endif