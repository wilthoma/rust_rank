#ifndef NTT_H
#define NTT_H


#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>

// #include <execution>
#include <random>
#include <cassert>
#include <iostream>
#include <bit>
#include <bitset>



template <typename T>
inline T PRIME() {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return 2013265921;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return 2305843009146585089ULL;
    } else if constexpr (std::is_same_v<T, __uint128_t>) {
        return 18446744069414584321ULL;
    } else {
        return 2013265921;
        //static_assert(false, "Unsupported type for PRIME");
    }
}

template <typename T>
inline T ROOT() {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return 31;
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        return 3;
    } else if constexpr (std::is_same_v<T, __uint128_t>) {
        return 7;
    } else {
        return 3;
        //static_assert(false, "Unsupported type for ROOT");
    }
}

template <typename T>
inline T mod_add(T a, T b) {
    T result = a + b;
    if (result >= PRIME<T>()) {
        result -= PRIME<T>();
    }
    return result;
}

template <typename T>
inline T mod_sub(T a, T b) {
    if (a < b) {
        return PRIME<T>()-b+a;
    } else {
        return a-b;
    }
}

template <typename T>
T mod_mul(T a, T b) {
    if constexpr (std::is_same_v<T, uint32_t>) {
        return static_cast<T>((static_cast<uint64_t>(a) * b) % PRIME<T>());
    } else if constexpr (std::is_same_v<T, uint64_t>) {
        __uint128_t product = static_cast<__uint128_t>(a) * b;
        uint64_t cc = static_cast<uint64_t>((product << 67) >> 67); // lowest 61 bits
        uint64_t bb = static_cast<uint64_t>((product << 32) >> 93); // bits 61-95
        uint64_t aa = static_cast<uint64_t>(product >> 96);         // bits 96-127
        return mod_add(mod_sub(mod_add(cc, bb << 26), bb), mod_sub(mod_add(aa << 26, PRIME<T>() - aa), aa << 35));
    } else if constexpr (std::is_same_v<T, __uint128_t>) {
        __uint128_t product = a * b;
        __uint128_t cc = (product << 64) >> 64;
        __uint128_t bb = (product << 32) >> 96;
        __uint128_t aa = product >> 96;
        return mod_sub(mod_sub(mod_add(cc, bb << 32), bb), aa);
    } else {
        return static_cast<T>((static_cast<uint64_t>(a) * b) % PRIME<T>());
        //static_assert(false, "Unsupported type for mod_mul");
    }
}

template <typename T>
inline T mod_pow(T base, T exp) {
    T result = 1;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = mod_mul(result,base);
        }
        base = mod_mul(base,base);
        exp /= 2;
    }
    return result;
}

template <typename T>
T mod_inv(T x) {
    return mod_pow(x, PRIME<T>() - 2);
}


template <typename T>
inline T bit_reverse(T x, size_t bits) {
    T result = 0;
    for (size_t i = 0; i < bits; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

template <typename T>
void ntt(std::vector<T>& a, bool invert) {
    const T p = PRIME<T>();
    const T root = ROOT<T>();
    const size_t n = a.size();
    const size_t bits = static_cast<size_t>(std::log2(n));
    // const size_t bits = static_cast<size_t>(std::countr_zero(n));
    assert((n & (n - 1)) == 0 && "Length of input array must be a power of 2");

    // Bit reversal permutation
    for (size_t i = 0; i < n; ++i) {
        size_t j = bit_reverse(i, bits);
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    T root_pow = invert
        ? mod_pow(root, p - 1 - (p - 1) / n)
        : mod_pow(root, (p - 1) / n);

    for (size_t len = 2; len <= n; len <<= 1) {
        T wlen = mod_pow(root_pow, static_cast<T>(n / len));
        for (size_t i = 0; i < n; i += len) {
            T w = 1;
            for (size_t j = 0; j < len / 2; ++j) {
                T u = a[i + j];
                T v = mod_mul(a[i + j + len / 2],w);
                a[i + j] = mod_add(u,v);
                a[i + j + len / 2] = mod_sub(u,v);
                w = mod_mul(w, wlen);
            }
        }
    }

    if (invert) {
        T n_inv = mod_pow(static_cast<T>(n), p - 2);
        for (auto& x : a) {
            x = mod_mul(x, n_inv);
        }
    }
}





template <typename T>
void matrix_ntt_parallel(std::vector<std::vector<std::vector<T>>>& a, bool invert) {
    // Apply NTT to all elements in parallel
    for (auto& row : a) {
        std::for_each( row.begin(), row.end(), [&](std::vector<T>& x) {
            ntt(x, invert);
        });
    }
}




// Function to generate random data
template <typename T>
std::vector<T> generate_random_data(size_t len, T p) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<T> dis(0, p - 1);

    std::vector<T> data(len);
    for (size_t i = 0; i < len; ++i) {
        data[i] = dis(gen);
    }
    return data;
}

// Test function for NTT and its inverse
void test_ntt_inverse() {
    using T = uint64_t;
    const T p = PRIME<T>(); // A prime modulus
    const T root = ROOT<T>(); // A primitive root modulo p
    const size_t len = 16; // Must be a power of two

    // Generate random data
    std::vector<T> data;
    auto raw_data = generate_random_data<T>(len, p);
    for (auto val : raw_data) {
        data.emplace_back(val);
    }

    // Clone the original data
    std::vector<T> original_data = data;

    // Perform forward and inverse NTT
    ntt(data, false); // Forward NTT
    ntt(data, true);  // Inverse NTT

    // Verify that the data matches the original
    assert(data == original_data && "NTT followed by inverse NTT should return the original data");
    std::cout << "NTT and inverse NTT test passed!" << std::endl;
}


void test_modmul_agreement() {
    using T = uint64_t;

    std::vector<std::pair<T, T>> test_cases = {
        {static_cast<T>(PRIME<uint32_t>()), static_cast<T>(ROOT<uint32_t>())},
        {static_cast<T>(PRIME<uint64_t>()), static_cast<T>(ROOT<uint64_t>())},
        {static_cast<T>(PRIME<__uint128_t>()), static_cast<T>(ROOT<__uint128_t>())},
    };

    std::random_device rd;
    std::mt19937_64 rng(rd());

    for (const auto& [prime, root] : test_cases) {
        std::uniform_int_distribution<T> dist(0, prime - 1);

        for (int i = 0; i < 1000; ++i) {
            T x = dist(rng);
            T y = dist(rng);

            T expected = static_cast<T>((static_cast<__uint128_t>(x) * y) % prime);
            T result;

            if (prime == PRIME<uint32_t>()) {
                result = mod_mul(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
            } else if (prime == PRIME<uint64_t>()) {
                result = mod_mul(static_cast<uint64_t>(x), static_cast<uint64_t>(y));
            } else if (prime == PRIME<__uint128_t>()) {
                result = static_cast<T>(mod_mul(static_cast<__uint128_t>(x), static_cast<__uint128_t>(y)));
            } else {
                throw std::runtime_error("Unsupported type");
            }

            assert(result == expected && "mod_mul failed");
        }
    }

    std::cout << "mod_mul agreement test passed!" << std::endl;
}


#include <chrono>

void benchmark_ntt_u64_vs_ntt_u128() {
    using T64 = uint64_t;
    using T128 = __uint128_t;

    const T64 p_u64 = PRIME<T64>(); // A prime modulus for uint64_t
    const T64 root_u64 = ROOT<T64>(); // A primitive root modulo p_u64
    const size_t len = 1 << 21; // Must be a power of two

    const T128 p_u128 = PRIME<T128>(); // A prime modulus for __uint128_t
    const T128 root_u128 = ROOT<T128>(); // A primitive root modulo p_u128

    // Generate random data
    auto data_u64 = generate_random_data<T64>(len, p_u64 - 1);
    std::vector<T128> data_u128(data_u64.begin(), data_u64.end());

    // Benchmark NTT<uint64_t>
    auto start_u64 = std::chrono::high_resolution_clock::now();
    ntt(data_u64, false);
    auto duration_u64 = std::chrono::high_resolution_clock::now() - start_u64;

    // Benchmark NTT<__uint128_t>
    auto start_u128 = std::chrono::high_resolution_clock::now();
    ntt(data_u128, false);
    auto duration_u128 = std::chrono::high_resolution_clock::now() - start_u128;

    std::cout << "NTT<uint64_t> took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration_u64).count()
              << " ms, NTT<__uint128_t> took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration_u128).count()
              << " ms" << std::endl;

    // Again

    // Benchmark NTT<uint64_t> (inverse)
    start_u64 = std::chrono::high_resolution_clock::now();
    ntt(data_u64, true);
    duration_u64 = std::chrono::high_resolution_clock::now() - start_u64;

    // Benchmark NTT<__uint128_t> (inverse)
    start_u128 = std::chrono::high_resolution_clock::now();
    ntt(data_u128, true);
    duration_u128 = std::chrono::high_resolution_clock::now() - start_u128;

    std::cout << "Inverse NTT<uint64_t> took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration_u64).count()
              << " ms, Inverse NTT<__uint128_t> took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration_u128).count()
              << " ms" << std::endl;

    // Ensure the results are valid (not strictly necessary for benchmarking)
    std::vector<T64> original_data_u64(data_u128.begin(), data_u128.end());
    assert(data_u64 == original_data_u64 && "NTT<uint64_t> and NTT<__uint128_t> results should match");
    std::cout << "Test passed: NTT and inverse NTT results match!" << std::endl;
}

#endif // NTT_H