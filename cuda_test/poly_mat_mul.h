#ifndef POLY_MAT_MUL_H
#define POLY_MAT_MUL_H


#include <vector>
#include <mutex>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <thread>
#include <future>
#include <cstdint>
#include <stdexcept>
#include <cassert>
#include "ntt.h" 

using namespace std;
using namespace chrono;


template <typename T>
std::vector<T> poly_mul_naive(std::vector<T> poly1, std::vector<T> poly2, T p) {
    size_t n = poly1.size();
    size_t m = poly2.size();
    std::vector<T> result(n + m - 1, T(0));

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            result[i + j] = (result[i + j] + poly1[i] * poly2[j]) % p;
        }
    }

    return result;
}



template <typename T>
std::vector<std::vector<std::vector<T>>> poly_mat_mul_naive(
    const std::vector<std::vector<std::vector<T>>>& a,
    const std::vector<std::vector<std::vector<T>>>& b,
    T p) {
    size_t m = a.size();
    size_t n = a[0].size();
    size_t k = b[0].size();
    if (n != b.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }
    size_t nlen = a[0][0].size();
    size_t nlenb = b[0][0].size();
    size_t nlenres = nlen + nlenb - 1;
    std::vector<std::vector<std::vector<T>>> result(m, std::vector<std::vector<T>>(k, std::vector<T>(nlenres, T(0))));
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t r = 0; r < n; ++r) {
                auto res = poly_mul_naive(a[i][r], b[r][j], p);
                for (size_t l = 0; l < nlenres; ++l) {
                    result[i][j][l] = (result[i][j][l] + res[l]) % p;
                }
            }
        }
    }
    return result;
    
}

template<typename T>
std::vector<T> poly_mul_fft(const std::vector<T>& p1, const std::vector<T>& p2) {
    size_t n = p1.size();
    size_t m = p2.size();
    size_t nlenres = n + m - 1;
    // find next highest power of two
    size_t nlenres_adj = 1;
    while (nlenres_adj < nlenres) {
        nlenres_adj <<= 1;
    }
    // pad with zeros
    std::vector<T> fa(nlenres_adj, T(0));
    std::vector<T> fb(nlenres_adj, T(0));
    std::copy(p1.begin(), p1.end(), fa.begin());
    std::copy(p2.begin(), p2.end(), fb.begin());
    // perform ntt
    auto start = high_resolution_clock::now();
    ntt(fa, false);
    ntt(fb, false);
    auto durationntt = high_resolution_clock::now() - start;
    // multiply
    std::vector<T> fresult(nlenres_adj, T(0));
    for (size_t i = 0; i < nlenres_adj; ++i) {
        fresult[i] = mod_mul(fa[i], fb[i]);
    }
    // perform inverse ntt
    start = high_resolution_clock::now();
    ntt(fresult, true);
    auto durationmul = high_resolution_clock::now() - start;
    // resize result
    fresult.resize(nlenres);
    return fresult;
}


static mutex NTT_TIME_MUTEX;
static mutex MUL_TIME_MUTEX;
static mutex NTT_TIME_L_MUTEX;
static mutex MUL_TIME_L_MUTEX;

static duration<double> NTT_TIME = duration<double>::zero();
static duration<double> MUL_TIME = duration<double>::zero();
static duration<double> NTT_TIME_L = duration<double>::zero();
static duration<double> MUL_TIME_L = duration<double>::zero();

template <typename T>
vector<vector<vector<T>>> poly_mat_mul_fft(const vector<vector<vector<T>>>& a, const vector<vector<vector<T>>>& b, size_t b_start_deg, size_t b_end_deg) {
    size_t m = a.size();
    size_t n = a[0].size();
    size_t k = b[0].size();
    if (n != b.size()) {
        throw invalid_argument("Matrix dimensions do not match for multiplication");
    }

    size_t nlena = a[0][0].size();
    size_t nlenb = min(b[0][0].size(), b_end_deg - b_start_deg);
    size_t nlenres = nlena + nlenb - 1;

    size_t nlenres_adj = 1;
    while (nlenres_adj < nlenres) {
        nlenres_adj <<= 1;
    }

    vector<vector<vector<T>>> fa(m, vector<vector<T>>(n, vector<T>(nlenres_adj, T(0))));
    vector<vector<vector<T>>> fb(n, vector<vector<T>>(k, vector<T>(nlenres_adj, T(0))));
    vector<vector<vector<T>>> fresult(m, vector<vector<T>>(k, vector<T>(nlenres_adj, T(0))));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            copy(a[i][j].begin(), a[i][j].end(), fa[i][j].begin());
        }
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            copy(b[i][j].begin() + b_start_deg, b[i][j].begin() + b_start_deg + nlenb, fb[i][j].begin());
        }
    }

    auto start = high_resolution_clock::now();
    matrix_ntt_parallel(fa, false);
    matrix_ntt_parallel(fb, false);
    auto durationntt = high_resolution_clock::now() - start;

    start = high_resolution_clock::now();
    // multiply
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t r = 0; r < n; ++r) {
                for (size_t l = 0; l < nlenres_adj; ++l) {
                    fresult[i][j][l] = mod_add(fresult[i][j][l],mod_mul(fa[i][r][l], fb[r][j][l]));
                }
            }
        }
    }





    // vector<future<void>> futures;
    // for (size_t i = 0; i < m; ++i) {
    //     futures.push_back(async(launch::async, [&, i]() {
    //         for (size_t j = 0; j < k; ++j) {
    //             for (size_t r = 0; r < n; ++r) {
    //                 for (size_t l = 0; l < nlenres_adj; ++l) {
    //                     fresult[i][j][l] += fa[i][r][l] * fb[r][j][l];
    //                 }
    //             }
    //         }
    //     }));
    // }
    // for (auto& fut : futures) {
    //     fut.get();
    // }
    auto durationmul = high_resolution_clock::now() - start;

    start = high_resolution_clock::now();
    matrix_ntt_parallel(fresult, true);
    durationntt += high_resolution_clock::now() - start;

    {
        lock_guard<mutex> lock(MUL_TIME_MUTEX);
        MUL_TIME += duration_cast<duration<double>>(durationmul);
    }
    {
        lock_guard<mutex> lock(NTT_TIME_MUTEX);
        NTT_TIME += duration_cast<duration<double>>(durationntt);
    }

    if (duration_cast<milliseconds>(durationmul).count() > 10 || duration_cast<milliseconds>(durationntt).count() > 10) {
        {
            lock_guard<mutex> lock(MUL_TIME_L_MUTEX);
            MUL_TIME_L += duration_cast<duration<double>>(durationmul);
        }
        {
            lock_guard<mutex> lock(NTT_TIME_L_MUTEX);
            NTT_TIME_L += duration_cast<duration<double>>(durationntt);
        }
    }

    if (duration_cast<milliseconds>(durationmul).count() > 200 || duration_cast<milliseconds>(durationntt).count() > 200) {
        cout << "\nDuration (Multiplication): " << duration_cast<milliseconds>(durationmul).count()
             << "ms (total " << MUL_TIME.count() << "s, " << MUL_TIME_L.count() << "s), Duration (NTT): "
             << duration_cast<milliseconds>(durationntt).count() << "ms (total " << NTT_TIME.count()
             << "s, " << NTT_TIME_L.count() << "s)" << endl;
    }

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < k; ++j) {
            fresult[i][j].resize(nlenres, T(0));
        }
    }

    return fresult;
}




template <typename T>
vector<vector<vector<T>>> poly_mat_mul_fft_red(
    const vector<vector<vector<T>>>& a,
    const vector<vector<vector<T>>>& b,
    T reducetoprime,
    size_t b_start_deg,
    size_t b_end_deg) {
    
    // Perform FFT-based polynomial matrix multiplication
    auto red = poly_mat_mul_fft(a, b, b_start_deg, b_end_deg);

    // Reduce the result modulo the smaller prime
    size_t n = red.size();
    for (size_t i = 0; i < n; ++i) {
        size_t m = red[i].size();
        for (size_t j = 0; j < m; ++j) {
            size_t k = red[i][j].size();
            for (size_t l = 0; l < k; ++l) {
                red[i][j][l] = red[i][j][l] % reducetoprime;
            }
        }
    }

    return red;
}






//********* Test code */


void test_poly_mul_naive_vs_fft() {
    using T = uint64_t; // Use 64-bit unsigned integer type
    T p = PRIME<T>(); // A large prime modulus
    size_t poly_len1 = 5, poly_len2 = 4;

    // Initialize random polynomials
    std::vector<T> poly1(poly_len1), poly2(poly_len2);
    for (size_t i = 0; i < poly_len1; ++i) {
        poly1[i] = static_cast<T>(rand()) % p;
    }
    for (size_t i = 0; i < poly_len2; ++i) {
        poly2[i] = static_cast<T>(rand()) % p;
    }

    // Compute results using both methods
    auto result_naive = poly_mul_naive(poly1, poly2, p);
    auto result_fft = poly_mul_fft(poly1, poly2);

    // Compare results
    assert(result_naive.size() == result_fft.size());
    for (size_t i = 0; i < result_naive.size(); ++i) {
        if (result_naive[i] != result_fft[i]) {
            std::cerr << "Mismatch at index " << i << ": naive=" << result_naive[i]
                      << ", fft=" << result_fft[i] << std::endl;
            assert(false);
        }
    }

    std::cout << "Test passed: poly_mul_naive and poly_mul_fft agree." << std::endl;
}




void test_poly_mat_mul_naive_vs_fft() {
    using T = uint64_t; //__uint128_t; // Use GCC/Clang's 128-bit integer type
    T p = PRIME<T>(); // A large prime modulus
    size_t m = 2, n = 2, k = 2, poly_len = 3;

    // Initialize random polynomial matrices
    std::vector<std::vector<std::vector<T>>> a(m, std::vector<std::vector<T>>(n, std::vector<T>(poly_len)));
    std::vector<std::vector<std::vector<T>>> b(n, std::vector<std::vector<T>>(k, std::vector<T>(poly_len)));

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            for (size_t l = 0; l < poly_len; ++l) {
                a[i][j][l] = static_cast<T>(rand()) % p;
            }
        }
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < k; ++j) {
            for (size_t l = 0; l < poly_len; ++l) {
                b[i][j][l] = static_cast<T>(rand()) % p;
            }
        }
    }

    // Compute results using both methods
    auto result_naive = poly_mat_mul_naive(a, b, p);
    auto result_fft = poly_mat_mul_fft(a, b, 0, poly_len);

    // Compare results
    assert(result_naive.size() == result_fft.size());
    for (size_t i = 0; i < result_naive.size(); ++i) {
        for (size_t j = 0; j < result_naive[i].size(); ++j) {
            assert(result_naive[i][j].size() == result_fft[i][j].size());
            for (size_t l = 0; l < result_naive[i][j].size(); ++l) {
                if (result_naive[i][j][l] != result_fft[i][j][l]) {
                    std::cerr << "Mismatch at (" << i << ", " << j << ", " << l << "): "
                              << "naive=" << result_naive[i][j][l] << ", fft=" << result_fft[i][j][l] << std::endl;
                    assert(false);
                }
            }
        }
    }

    std::cout << "Test passed: poly_mat_mul_naive and poly_mat_mul_fft agree." << std::endl;
}



#endif // POLY_MAT_MUL_H