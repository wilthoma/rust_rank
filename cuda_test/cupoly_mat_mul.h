#ifndef CUPOLY_MAT_MUL_H
#define CUPOLY_MAT_MUL_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>
#include <cmath>
#include <iostream>
#include "poly_mat_mul.h"
#include "ntt.h"
#include "cudantt.h"


std::vector<u64> poly_mul_fft_cuda(const std::vector<u64>& p1, const std::vector<u64>& p2) {
    size_t n = p1.size();
    size_t m = p2.size();
    size_t nlenres = n + m - 1;
    // find next highest power of two
    size_t nlenres_adj = 1;
    while (nlenres_adj < nlenres) {
        nlenres_adj <<= 1;
    }
    // pad with zeros
    std::vector<u64> fa(nlenres_adj, 0);
    std::vector<u64> fb(nlenres_adj, 0);
    std::copy(p1.begin(), p1.end(), fa.begin());
    std::copy(p2.begin(), p2.end(), fb.begin());
    // perform ntt
    auto start = high_resolution_clock::now();
    ntt_cuda(fa, false);
    ntt_cuda(fb, false);
    auto durationntt = high_resolution_clock::now() - start;
    // multiply
    std::vector<u64> fresult(nlenres_adj, 0);
    for (size_t i = 0; i < nlenres_adj; ++i) {
        fresult[i] = mod_mul(fa[i], fb[i]);
    }
    // perform inverse ntt
    // start = high_resolution_clock::now();
    ntt_cuda(fresult, true);
    // auto durationmul = high_resolution_clock::now() - start;
    // resize result
    fresult.resize(nlenres);
    return fresult;
}



//************** TESTS *********************/


void test_cuda_poly_mul_methods() {
    // Generate two random polynomials
    size_t degree1 = 40; // Example degree
    size_t degree2 = 100; // Example degree
    std::vector<u64> poly1(degree1);
    std::vector<u64> poly2(degree2);

    // Fill polynomials with random coefficients
    for (size_t i = 0; i < degree1; ++i) {
        poly1[i] = rand() % 1000; // Random coefficients in range [0, 999]
    }
    for (size_t i = 0; i < degree2; ++i) {
        poly2[i] = rand() % 1000;
    }

    // Compute results using both methods
    auto result_cuda = poly_mul_fft_cuda(poly1, poly2);
    // auto result_cpu = poly_mul_fft(poly1, poly2);
    auto result_cpu = poly_mul_naive(poly1, poly2, PRIME<u64>());

    // Compare results
    assert(result_cuda.size() == result_cpu.size());
    for (size_t i = 0; i < result_cuda.size(); ++i) {
        if (result_cuda[i] != result_cpu[i]) {
            std::cerr << "Mismatch at index " << i << ": cuda=" << result_cuda[i]
                      << ", cpu=" << result_cpu[i] << std::endl;
            assert(false);
        }
    }
    std::cout << "Test passed: poly_mul_cuda and poly_mul_cpu agree." << std::endl;
}



#endif // CUPOLY_MAT_MUL_H