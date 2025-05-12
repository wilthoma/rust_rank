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


template <typename T>
class ModMul {
public:
    static const T PRIME;
    static const T ROOT;

    T value;

    ModMul(T v = 0) : value(v) {}

    ModMul<T> mulmod(const ModMul<T>& other) const {
        return ModMul<T>((static_cast<uint64_t>(value) * other.value) % PRIME);
    }

    ModMul<T> addmod(const ModMul<T>& other) const {
        T a = value + other.value;
        return ModMul<T>(a >= PRIME ? a - PRIME : a);
    }

    ModMul<T> submod(const ModMul<T>& other) const {
        return ModMul<T>(value < other.value ? (value + PRIME) - other.value : value - other.value);
    }

    ModMul<T> powmod(T exp) const {
        ModMul<T> base = *this;
        ModMul<T> result = 1;
        T two = 2;
        while (exp > 0) {
            if (exp % two == 1) {
                result = result.mulmod(base);
            }
            base = base.mulmod(base);
            exp /= two;
        }
        return result;
    }

    ModMul<T> invmod() const {
        T two = 2;
        return powmod(PRIME - two);
    }
};

template <>
const uint32_t ModMul<uint32_t>::PRIME = 2013265921;
template <>
const uint32_t ModMul<uint32_t>::ROOT = 31;

template <>
const uint64_t ModMul<uint64_t>::PRIME = 2305843009146585089ULL;
template <>
const uint64_t ModMul<uint64_t>::ROOT = 3;

template <>
const __uint128_t ModMul<__uint128_t>::PRIME = 18446744069414584321ULL;
template <>
const __uint128_t ModMul<__uint128_t>::ROOT = 7;



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
    const T p = T::PRIME;
    const T root = T::ROOT;
    const size_t n = a.size();
    const size_t bits = static_cast<size_t>(std::log2(n));
    assert((n & (n - 1)) == 0 && "Length of input array must be a power of 2");
    const T two = T(1) + T(1);

    // Bit reversal permutation
    for (size_t i = 0; i < n; ++i) {
        size_t j = bit_reverse(i, bits);
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    T root_pow = invert
        ? root.powmod(p - T(1) - (p - T(1)) / T(n))
        : root.powmod((p - T(1)) / T(n));

    for (size_t len = 2; len <= n; len <<= 1) {
        T wlen = root_pow.powmod(T(n / len));
        for (size_t i = 0; i < n; i += len) {
            T w = T(1);
            for (size_t j = 0; j < len / 2; ++j) {
                T u = a[i + j];
                T v = a[i + j + len / 2].mulmod(w);
                a[i + j] = u.addmod(v);
                a[i + j + len / 2] = u.submod(v);
                w = w.mulmod(wlen);
            }
        }
    }

    if (invert) {
        T n_inv = T(n).powmod(p - two);
        for (auto& x : a) {
            x = x.mulmod(n_inv);
        }
    }
}





template <typename T>
void matrix_ntt_parallel(std::vector<std::vector<std::vector<T>>>& a, bool invert) {
    // Apply NTT to all elements in parallel
    for (auto& row : a) {
        std::for_each(std::execution::par, row.begin(), row.end(), [&](std::vector<T>& x) {
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
    using T = ModMul<uint64_t>;
    const uint64_t p = T::PRIME; // A prime modulus
    const uint64_t root = T::ROOT; // A primitive root modulo p
    const size_t len = 16; // Must be a power of two

    // Generate random data
    std::vector<T> data;
    auto raw_data = generate_random_data<uint64_t>(len, p);
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
}





#endif // NTT_H