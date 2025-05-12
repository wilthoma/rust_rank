#ifndef NTT_H
#define NTT_H


#include <vector>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstdint>

#include <execution>
#include <random>
#include <cassert>
#include <iostream>


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
        T wlen = mod_pow(root, static_cast<T>(n / len));
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





#endif // NTT_H