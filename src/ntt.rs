use rayon::prelude::*;
use std::simd::{u64x4};
use std::simd::prelude::{SimdPartialOrd, SimdUint};
// use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use std::ops::{Add, Sub, AddAssign, SubAssign, Mul, Rem, MulAssign, Div, DivAssign};
use num_traits::{One, Zero};
use rand::distr::uniform::SampleUniform;
use std::fmt::{Display, Debug};
use std::iter::Sum;
use rand::Rng;


pub trait NTTInteger: Debug+Into<u128> + From<u32>+ MulAssign+ Div<Output = Self> + DivAssign + Sub<Output = Self> + SubAssign + Display+Sum+PartialOrd + SampleUniform + Copy + Zero + One + Rem<Output = Self> + Add<Output = Self> + AddAssign + Mul<Output = Self> + Send + Sync +'static { }
impl<T: Debug+Into<u128> + From<u32>+MulAssign+ Div<Output = Self> + DivAssign + Sub<Output = Self> + SubAssign + Display+ Sum+PartialOrd + SampleUniform + Copy + Zero + One + Rem<Output = Self> + Add<Output = Self> + AddAssign + Mul<Output = Self> + Send + Sync+'static> NTTInteger for T {}


#[inline(always)]
pub fn mod_add<T : NTTInteger>(mut a: T, b: T, p:T) -> T {
    a += b;
    if a >= p { a -= p; }
    a
}

#[inline(always)]
pub fn mod_sub<T : NTTInteger>(mut a: T, b: T, p:T) -> T {
    if a < b { a += p; }
    a - b
}

#[inline(always)]
pub fn mod_mul<T : NTTInteger>(a: T, b: T, p : T) -> T {
    (a * b) % p // !!!!! this % is responsible for 2/3 of the runtime -- optimize
}


#[inline(always)]
pub fn mod_pow<T : NTTInteger>(mut base: T, mut exp: T, p:T) -> T {
    let mut result = T::one();
    let two = T::one() + T::one();
    while exp > T::zero() {
        if exp % two == T::one() {
            result = mod_mul(result, base, p);
        }
        base = mod_mul(base, base, p);
        exp /= two;
    }
    result
}

// Modular inverse using the extended Euclidean algorithm
#[inline(always)]
pub fn modinv<T:NTTInteger>(a: T, p: T) -> T {
    let two = T::one() + T::one();
    mod_pow(a, p - two, p)
}

#[inline(always)]
pub fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

pub fn ntt<T : NTTInteger>(a: &mut [T], invert: bool, p : T, root : T) {
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    assert_eq!(n, 1 << bits, "Length of input array must be a power of 2");
    let two = T::one() + T::one();

    // Bit reversal permutation
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }

    let root = if invert {
        mod_pow(root, p - T::one() - (p - T::one()) / (T::from(n as u32)),p)
    } else {
        mod_pow(root, (p - T::one()) / (T::from(n as u32)),p)
    };

    let mut len = 2;
    while len <= n {
        let wlen = mod_pow(root, T::from((n / len) as u32),p);

        a.chunks_mut(len).for_each(|chunk| {
            let mut w = T::one();
            let (left, right) = chunk.split_at_mut(len/2);
            for (l, r) in left.iter_mut().zip(right.iter_mut()) {
                let u = *l;
                let v = mod_mul(*r, w, p);
                *l = mod_add(u, v, p);
                *r = mod_sub(u, v, p);
                w = mod_mul(w, wlen, p);
            }
        });

        len <<= 1;
    }

    if invert {
        let n_inv = mod_pow(T::from(n as u32), p - two, p);
        a.iter_mut().for_each(|x| {
            *x = mod_mul(*x, n_inv, p);
        });
    }
}


pub fn ntt_par<T : NTTInteger>(a: &mut [T], invert: bool, p : T, root : T) {
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    assert_eq!(n, 1 << bits, "Length of input array must be a power of 2");
    let two = T::one() + T::one();

    // Bit reversal permutation
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }

    let root = if invert {
        mod_pow(root, p - T::one() - (p - T::one()) / (T::from(n as u32)),p)
    } else {
        mod_pow(root, (p - T::one()) / (T::from(n as u32)),p)
    };

    let mut len = 2;
    while len <= n {
        let wlen = mod_pow(root, T::from((n / len) as u32),p);

        // Parallel over blocks
        a.par_chunks_mut(len).for_each(|chunk| {
            let mut w = T::one();
            let (left, right) = chunk.split_at_mut(len/2);
            for (l, r) in left.iter_mut().zip(right.iter_mut()) {
                let u = *l;
                let v = mod_mul(*r, w, p);
                *l = mod_add(u, v, p);
                *r = mod_sub(u, v, p);
                w = mod_mul(w, wlen, p);
            }
        });

        len <<= 1;
    }

    if invert {
        let n_inv = mod_pow(T::from(n as u32), p - two, p);
        a.par_iter_mut().for_each(|x| {
            *x = mod_mul(*x, n_inv, p);
        });
    }
}


pub fn matrix_ntt_parallel<T:NTTInteger>(a: &mut Vec<Vec<Vec<T>>>, invert: bool, p : T, root : T) {
    // apply ntt to all elements in parallel

    a.par_iter_mut().flat_map(|row| row.par_iter_mut()).for_each(|x| {
        ntt(x, invert, p, root);
    });
}


/// Modular addition for SIMD vectors
#[inline(always)]
fn simd_mod_add(a: u64x4, b: u64x4, MOD:u64) -> u64x4 {
    let sum = a + b;
    let overflow = sum.simd_ge(u64x4::splat(MOD));
    sum - overflow.select(u64x4::splat(MOD), u64x4::splat(0))
}

/// Modular subtraction for SIMD vectors
#[inline(always)]
fn simd_mod_sub(a: u64x4, b: u64x4, MOD:u64) -> u64x4 {
    let underflow = a.simd_lt(b);
    a - b + underflow.select(u64x4::splat(MOD), u64x4::splat(0))
}

/// Modular multiplication for SIMD vectors (using u128)
#[inline(always)]
fn simd_mod_mul(a: u64x4, b: u64x4, MOD:u64) -> u64x4 {
    let prod0 = (a[0] as u128 * b[0] as u128) % (MOD as u128);
    let prod1 = (a[1] as u128 * b[1] as u128) % (MOD as u128);
    let prod2 = (a[2] as u128 * b[2] as u128) % (MOD as u128);
    let prod3 = (a[3] as u128 * b[3] as u128) % (MOD as u128);
    u64x4::from_array([prod0 as u64, prod1 as u64, prod2 as u64, prod3 as u64])
}

pub fn ntt_simd(a: &mut [u64], invert: bool, p:u64, root:u64) {
    assert!(false, "SIMD NTT is not correctly implemented yet");
    let MOD = p;
    let ROOT = root;
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    assert_eq!(n, 1 << bits);

    // Bit reversal
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if i < j {
            a.swap(i, j);
        }
    }

    let root = if invert {
        mod_pow(ROOT, MOD - 1 - (MOD - 1) / (n as u64), MOD)
    } else {
        mod_pow(ROOT, (MOD - 1) / (n as u64), MOD)
    };

    let mut len = 2;
    while len <= n {
        let wlen = mod_pow(root, (n / len) as u64, MOD);

        a.par_chunks_mut(len).for_each(|chunk| {
            let mut w = 1u64;
            let (left, right) = chunk.split_at_mut(len/2);
            let (left_chunks, left_rem) = left.split_at_mut((len/2)/4 * 4);
            let (right_chunks, right_rem) = right.split_at_mut((len/2)/4 * 4);

            for (l_block, r_block) in left_chunks.chunks_exact_mut(4).zip(right_chunks.chunks_exact_mut(4)) {
                let l = u64x4::from_slice(l_block);
                let r = u64x4::from_slice(r_block);

                let w_vec = u64x4::splat(w);
                let r_twisted = simd_mod_mul(r, w_vec, MOD);

                let new_l = simd_mod_add(l, r_twisted, MOD);
                let new_r = simd_mod_sub(l, r_twisted, MOD);

                let new_l_array = new_l.to_array();
                let new_r_array = new_r.to_array();

                l_block.copy_from_slice(&new_l_array);
                r_block.copy_from_slice(&new_r_array);

                w = mod_mul(w, wlen, MOD);
            }

            // Handle leftovers
            for (l, r) in left_rem.iter_mut().zip(right_rem.iter_mut()) {
                let u = *l;
                let v = mod_mul(*r, w, MOD);
                *l = mod_add(u, v, MOD);
                *r = mod_sub(u, v, MOD);
                w = mod_mul(w, wlen, MOD);
            }
        });

        len <<= 1;
    }

    if invert {
        let n_inv = mod_pow(n as u64, MOD - 2, MOD);
        a.par_iter_mut().for_each(|x| {
            *x = mod_mul(*x, n_inv, MOD);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_data(len: usize, p: u64) -> Vec<u64> {
        let mut rng = rand::rng();
        (0..len).map(|_| rng.random_range(0..p)).collect()
    }

    #[test]
    fn test_ntt_inverse() {
        let p = 998244353; // A prime modulus
        let root = 3; // A primitive root modulo p
        let len = 16; // Must be a power of two
        let mut data = generate_random_data(len, p);

        let original_data = data.clone();
        ntt(&mut data, false, p, root); // Forward NTT
        ntt(&mut data, true, p, root); // Inverse NTT

        assert_eq!(data, original_data, "NTT followed by inverse NTT should return the original data");
    }

    // #[test]
    // fn test_ntt_simd_inverse() {
    //     let p = 998244353; // A prime modulus
    //     let root = 3; // A primitive root modulo p
    //     let len = 16; // Must be a power of two
    //     let mut data = generate_random_data(len, p);

    //     let original_data = data.clone();
    //     ntt_simd(&mut data, false, p, root); // Forward NTT (SIMD)
    //     ntt_simd(&mut data, true, p, root); // Inverse NTT (SIMD)

    //     assert_eq!(data, original_data, "NTT_SIMD followed by inverse NTT_SIMD should return the original data");
    // }

    // #[test]
    // fn test_ntt_vs_ntt_simd() {
    //     let p = 998244353; // A prime modulus
    //     let root = 3; // A primitive root modulo p
    //     let len = 16; // Must be a power of two
    //     let mut data1 = generate_random_data(len, p);
    //     let mut data2 = data1.clone();

    //     ntt(&mut data1, false, p, root); // Forward NTT
    //     ntt_simd(&mut data2, false, p, root); // Forward NTT (SIMD)

    //     assert_eq!(data1, data2, "Results of NTT and NTT_SIMD should agree");
    // }
}
