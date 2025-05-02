use bubblemath::linear_recurrence::poly_mul;
// use num_traits::Zero;
// use rayon::iter::IntoParallelIterator;

use crate::{ntt::{matrix_ntt_parallel, mod_add, mod_mul, ntt, NTTInteger, ModMul}, polynomial};
use rand::Rng;
use std::{cmp::min, time::Duration};
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::ops::Index;
use num::BigUint;
use std::time::Instant;
use std::ops::{RangeBounds, Bound};

const FFT_THRESHOLD: usize = 30; // for higher output degree, fft poly multiplication is used 

pub trait KaraMultiply: Sized {
    fn poly_mul(a: &[Self], b: &[Self], p: Self) -> Vec<Self>;
}

impl KaraMultiply for u64 {
    fn poly_mul(a: &[u64], b: &[u64], p: u64) -> Vec<u64> {
        poly_mul(a, b, p)
    }
}
impl KaraMultiply for u128 {
    fn poly_mul(a: &[u128], b: &[u128], p: u128) -> Vec<u128> {
        poly_mul128(a, b, p)
    }
}

#[inline(always)]
fn add_vects_in<T>(a : &mut Vec<T>, b : &Vec<T>, p : T) 
where T : NTTInteger {
    assert_eq!(a.len(), b.len(), "Vectors must be of the same length");
    a.iter_mut().zip(b.iter()).for_each(|(a, b)| {
        *a = mod_add(*a, *b, p);
    });
    // for i in 0..a.len() {
    //     a[i] = (a[i] + b[i]) % p;
    // }
}

pub fn poly_mat_mul_bubble<T:NTTInteger+KaraMultiply>(a: &Vec<Vec<Vec<T>>>, b: &Vec<Vec<Vec<T>>>, p: T) -> Vec<Vec<Vec<T>>> {
    let m = a.len();
    let n = a[0].len();
    let k = b[0].len();
    assert_eq!(n, b.len(), "Matrix dimensions do not match for multiplication");

    let nlena = a[0][0].len();
    let nlenb = b[0][0].len();
    let nlenres = nlena + nlenb - 1;
    let mut result = vec![vec![vec![T::zero(); nlenres]; k]; m];
    for i in 0..m {
        for j in 0..k {
            for l in 0..n {
                add_vects_in(&mut result[i][j], &KaraMultiply::poly_mul(&a[i][l], &b[l][j], p), p);
            }
        }
    }
    result

}

pub fn poly_mat_mul_bubble_red<T:NTTInteger+KaraMultiply>(a: &Vec<Vec<Vec<T>>>, b: &Vec<Vec<Vec<T>>>, p: T, b_start_deg :usize, b_end_deg : usize) -> Vec<Vec<Vec<T>>> {
    let m = a.len();
    let n = a[0].len();
    let k = b[0].len();
    assert_eq!(n, b.len(), "Matrix dimensions do not match for multiplication");

    let nlena = a[0][0].len();
    let nlenb = min(b[0][0].len(), b_end_deg-b_start_deg);
    let nlenres = nlena + nlenb - 1;
    let mut result = vec![vec![vec![T::zero(); nlenres]; k]; m];
    for i in 0..m {
        for j in 0..k {
            for l in 0..n {
                add_vects_in(&mut result[i][j], &KaraMultiply::poly_mul(&a[i][l], &b[l][j][b_start_deg..b_end_deg], p), p);
            }
        }
    }
    result
}





pub fn poly_mat_mul_red_adaptive<T : NTTInteger+KaraMultiply>(a: &Vec<Vec<Vec<T>>>, b: &Vec<Vec<Vec<T>>>, p: T, root: T, reducetoprime : T, b_start_deg :usize, b_end_deg : usize) -> Vec<Vec<Vec<T>>> {
    // selects the fastest method heuristically.
    let deg = a[0][0].len() + min(b[0][0].len(), b_end_deg-b_start_deg);
    let nr_multi = a.len() * b.len() * b[0].len();

    if nr_multi < 200 && deg < FFT_THRESHOLD {
        // use bubble method
        poly_mat_mul_bubble_red(a, b, reducetoprime, b_start_deg, b_end_deg)
    } else {
        // use fft method
        poly_mat_mul_fft_red(a, b, p, root, reducetoprime, b_start_deg, b_end_deg)
    }
}

// let mut ntt_time : Duration = Duration::new(0, 0);
// let mut mul_time : Duration = Duration::new(0, 0);

static NTT_TIME: Lazy<Mutex<Duration>> = Lazy::new(|| Mutex::new(Duration::new(0, 0)));
static MUL_TIME: Lazy<Mutex<Duration>> = Lazy::new(|| Mutex::new(Duration::new(0, 0)));

static NTT_TIME_L: Lazy<Mutex<Duration>> = Lazy::new(|| Mutex::new(Duration::new(0, 0)));
static MUL_TIME_L: Lazy<Mutex<Duration>> = Lazy::new(|| Mutex::new(Duration::new(0, 0)));


pub fn poly_mat_mul_fft<T : NTTInteger>(a: &Vec<Vec<Vec<T>>>, b: &Vec<Vec<Vec<T>>>, p: T, root: T, b_start_deg :usize, b_end_deg : usize) -> Vec<Vec<Vec<T>>> {
    let m = a.len();
    let n = a[0].len();
    let k = b[0].len();
    assert_eq!(n, b.len(), "Matrix dimensions do not match for multiplication");

    let nlena = a[0][0].len();
    let nlenb = min(b[0][0].len(), b_end_deg-b_start_deg);
    let nlenres = nlena + nlenb - 1;


    // find correct out length adjusted to power of 2
    let mut nlenres_adj: usize = 1;
    while nlenres_adj < nlenres {
        nlenres_adj <<= 1;
    }

    // adjust lengths of inputs
    let mut fa = vec![vec![vec![T::zero(); nlenres_adj];n];m];
    let mut fb = vec![vec![vec![T::zero(); nlenres_adj];k];n];
    let mut fresult = vec![vec![vec![T::zero(); nlenres_adj];k];m];

    for i in 0..m {
        for j in 0..n {
            for l in 0..nlena {
                fa[i][j][l] = a[i][j][l];
            }
        }
    }
    for i in 0..n {
        for j in 0..k {
            for l in 0..nlenb {
                fb[i][j][l] = b[i][j][b_start_deg+l];
            }
        }
    }
    // compute ffts
    let start = Instant::now();
    matrix_ntt_parallel(&mut fa, false);
    // for i in 0..m {
    //     for j in 0..n {
    //         ntt(&mut fa[i][j], false, p, root);
    //     }
    // }
    matrix_ntt_parallel(&mut fb, false);
    let mut durationntt = start.elapsed();
    // for i in 0..n {
    //     for j in 0..k {
    //         ntt(&mut fb[i][j], false, p, root);
    //     }
    // }

    // multiply matrices
    let start = Instant::now();
    use rayon::prelude::*;

    fresult.par_iter_mut().enumerate().for_each(|(i, row)| {
        row.par_iter_mut().enumerate().for_each(|(j, cell)| {
            for r in 0..n {
                for l in 0..nlenres_adj {
                    cell[l] = cell[l].addmod(fa[i][r][l].mulmod(fb[r][j][l]));
                }
            }
        });
    });

    // for i in 0..m {
    //     for j in 0..k {
    //         for r in 0..n {
    //             for l in 0..nlenres_adj {
    //                 fresult[i][j][l] = mod_add(fresult[i][j][l], mod_mul(fa[i][r][l], fb[r][j][l],p), p);
    //             }
    //         }
    //     }
    // }

    let durationmul = start.elapsed();
    // compute iffts qnd resize result
    let start = Instant::now();
    matrix_ntt_parallel(&mut fresult, true);
    durationntt += start.elapsed();

    {
        let mut mul_time_guard = MUL_TIME.lock().unwrap();
        *mul_time_guard += durationmul;
    }
    {
        let mut ntt_time_guard = NTT_TIME.lock().unwrap();
        *ntt_time_guard += durationntt;
    }
    let ntt_time = *NTT_TIME.lock().unwrap();
    let mul_time = *MUL_TIME.lock().unwrap();

    if durationmul.as_millis() > 10 || durationntt.as_millis() > 10 {
        {
            let mut mul_time_guard = MUL_TIME_L.lock().unwrap();
            *mul_time_guard += durationmul;
        }
        {
            let mut ntt_time_guard = NTT_TIME_L.lock().unwrap();
            *ntt_time_guard += durationntt;
        }
    }
    let ntt_timel = *NTT_TIME_L.lock().unwrap();
    let mul_timel = *MUL_TIME_L.lock().unwrap();

    if durationmul.as_millis() > 200 || durationntt.as_millis() > 200 {
        println!(
            "\nDuration (Multiplication): {:?} (total {:?}, {:?}) , Duration (NTT): {:?} (total {:?}, {:?})",
            durationmul, mul_time, mul_timel, durationntt, ntt_time, ntt_timel
        );
    }

    for i in 0..m {
        for j in 0..k {
            // ntt(&mut fresult[i][j], true, p, root);
            fresult[i][j].resize(nlenres, T::zero());
        }
    }

    fresult
}


/// Multiplies two matrices of polynomials using FFT and reduces the result modulo a prime.
/// p should be a large prime relative to reducetoprime and the sequence size so that no overflow can occur.
/// max_b_degree allows to restrict the coefficients of b to be considered to that length, even if the buffer is larger
pub fn poly_mat_mul_fft_red<T : NTTInteger>(a: &Vec<Vec<Vec<T>>>, b: &Vec<Vec<Vec<T>>>, p: T, root: T, reducetoprime : T, b_start_deg :usize, b_end_deg : usize) -> Vec<Vec<Vec<T>>> {
    // check whether prime is large enough for NTT
    // let mut max_power_of_two = 1;
    // let mut k = 0;
    // let deg_a = a[0][0].len();
    // let deg_b = min(b_end_deg-b_start_deg, b[0][0].len());
    // while (p.into() - 1) % (2 * max_power_of_two as u128 ) == 0 {
    //     max_power_of_two *= 2;
    //     k += 1;
    // }
    // let ntt_limit = max_power_of_two;

    // let required_size = (deg_a + deg_b).next_power_of_two();
    // assert!( required_size < ntt_limit, "Polynomials are too large for this modulus (NTT size too big) largeprime: {} sequence prime: {} degrees: {}, {}", p, reducetoprime, deg_a, deg_b);

    // // no overflow possible?
    // assert!( reducetoprime.into()*reducetoprime.into() * ((deg_a + deg_b) as u128) < p.into(), "Polynomials are too large for this modulus (overflow possible) largeprime: {} sequence prime: {} degrees: {}, {}", p, reducetoprime, deg_a, deg_b);

    // multiply
    let mut red = poly_mat_mul_fft(a, b, p, root, b_start_deg, b_end_deg);

    // reduce mod smaller prime
    let n = red.len();
    for i in 0..n {
        let m = red[i].len();
        for j in 0..m {
            let k = red[i][j].len();
            for l in 0..k {
                red[i][j][l] = red[i][j][l] % reducetoprime;
            }
        }
    }
    red
}



/// Adapted u128 version of the bubblemath code 
/// https://github.com/Bubbler-4/math-rs/blob/main/bubblemath/src/linear_recurrence.rs
/// 
pub fn poly_mul128(p1: &[u128], p2: &[u128], modulo: u128) -> Vec<u128> {
    let mut p1_adjusted: Vec<u32> = Vec::with_capacity(p1.len() * 5);
    for &p1_i in p1 {
        p1_adjusted.push((p1_i & 0xFFFFFFFF) as u32);
        p1_adjusted.push(((p1_i >> 32) & 0xFFFFFFFF) as u32); // only 64 bit are used
        p1_adjusted.push(0);
        p1_adjusted.push(0);
        p1_adjusted.push(0);
    }
    let mut p2_adjusted: Vec<u32> = Vec::with_capacity(p2.len() * 5);
    for &p2_i in p2 {
        p2_adjusted.push((p2_i & 0xFFFFFFFF) as u32);
        p2_adjusted.push(((p2_i >> 32) & 0xFFFFFFFF) as u32);
        p2_adjusted.push(0);
        p2_adjusted.push(0);
        p2_adjusted.push(0);
    }
    let p1_biguint = BigUint::new(p1_adjusted);
    let p2_biguint = BigUint::new(p2_adjusted);
    let res = (&p1_biguint * &p2_biguint).to_u32_digits();
    let mut digits: Vec<u128> = res.chunks(5).map(|chunk| {
        match chunk.len() {
            1 => chunk[0] as u128 % modulo,
            2 => (chunk[0] as u128 + ((chunk[1] as u128) << 32)) % modulo,
            3 => (chunk[0] as u128 + ((chunk[1] as u128) << 32) + ((chunk[2] as u128) << 64) ) % modulo,
             // we hard-drop the overflow u[4], I think the original bubblemath code is not correct here
            4 | 5 => (chunk[0] as u128 + ((chunk[1] as u128) << 32) + ((chunk[2] as u128) << 64) + ((chunk[3] as u128) <<96) ) % modulo,
            _ => unreachable!()
        }
    }).collect();
    digits.resize(p1.len() + p2.len() - 1, 0);
    digits
}


#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_matrix<T : NTTInteger>(m: usize, n: usize, poly_len: usize, max_value: T) -> Vec<Vec<Vec<T>>> {
        let mut rng = rand::rng();
        (0..m)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        (0..poly_len)
                            .map(|_| T::from(rng.random_range(0..(max_value.into() as u32))))
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn test_poly_mat_mul_consistency() {
        let m = 2; // Number of rows in matrix A
        let n = 4; // Number of columns in matrix A and rows in matrix B
        let k = 6; // Number of columns in matrix B
        let poly_len_a = 3; // Length of polynomials in matrix A
        let poly_len_b = 3; // Length of polynomials in matrix B
        let p = u64::PRIME; // A large prime modulus
        let root = 3; // Primitive root for NTT
        let max_value = p-1; // Maximum value for random coefficients

        // Generate random input matrices
        let a = generate_random_matrix(m, n, poly_len_a, max_value);
        let b = generate_random_matrix(n, k, poly_len_b, max_value);
        let a128 = a.iter().map(|row| row.iter().map(|poly| poly.iter().map(|&x| x as u128).collect()).collect()).collect::<Vec<Vec<Vec<u128>>>>();
        let b128 = b.iter().map(|row| row.iter().map(|poly| poly.iter().map(|&x| x as u128).collect()).collect()).collect::<Vec<Vec<Vec<u128>>>>();

        // Compute results using both methods
        // we need to use bubble for u128 to avoid overflow. u64 bubble assumes all values are <2^32
        let result_bubble = poly_mat_mul_bubble(&a128, &b128, p as u128);
        let result_fft = poly_mat_mul_fft(&a, &b, p, root, 0, poly_len_b);

        // Assert that the results are the same
        let result_bubble64 = result_bubble.iter().map(|row| row.iter().map(|poly| poly.iter().map(|&x| x as u64).collect()).collect()).collect::<Vec<Vec<Vec<u64>>>>();
        assert_eq!(result_bubble64, result_fft, "Results from poly_mat_mul_bubble and poly_mat_mul_fft do not match");
    }

    #[test]
    fn test_poly_mul128_consistency() {

        let poly_len = 5; // Length of the polynomials
        let max_value = 1_000_000_000_000_000_000; // Maximum value for random coefficients
        let modulo = 1_000_000_000_000_000_003; // A large prime modulus

        // Generate two random polynomials
        let mut rng = rand::rng();
        let p1: Vec<u128> = (0..poly_len).map(|_| rng.random_range(0..max_value)).collect();
        let p2: Vec<u128> = (0..poly_len).map(|_| rng.random_range(0..max_value)).collect();

        // Compute the result using poly_mul128
        let result_poly_mul128 = poly_mul128(&p1, &p2, modulo);

        // Compute the result using naive polynomial multiplication
        let mut result_naive = vec![0u128; p1.len() + p2.len() - 1];
        for (i, &coeff1) in p1.iter().enumerate() {
            for (j, &coeff2) in p2.iter().enumerate() {
                result_naive[i + j] = (result_naive[i + j] + coeff1 * coeff2) % modulo;
            }
        }

        // Assert that the results are the same
        assert_eq!(
            result_poly_mul128, result_naive,
            "Results from poly_mul128 and naive multiplication do not match"
        );
    }

    #[test]
    fn benchmark_poly_mul128_vs_poly_mul_fft() {
        return; // Skip this test for now
        let degrees = vec![2,2,5,10,15, 20, 50, 100, 500, 1000]; // Polynomial degrees to test
        let max_value = 1_000_000_000_000_000_000; // Maximum value for random coefficients
        let modulo = 1_000_000_000_000_000_003; // A large prime modulus
        let root = 3; // Primitive root for NTT

        for &degree in &degrees {
            // Generate two random polynomials
            let mut rng = rand::rng();
            let p1: Vec<u128> = (0..degree).map(|_| rng.random_range(0..max_value)).collect();
            let p2: Vec<u128> = (0..degree).map(|_| rng.random_range(0..max_value)).collect();

            // Benchmark poly_mul128
            let start = Instant::now();
            let _result_poly_mul128 = poly_mul128(&p1, &p2, modulo);
            let duration_poly_mul128 = start.elapsed();

            // Prepare matrices for poly_mul_fft
            let a = vec![vec![p1.clone()]];
            let b = vec![vec![p2.clone()]];

            // Benchmark poly_mul_fft
            let start = Instant::now();
            let _result_poly_mul_fft = poly_mat_mul_fft(&a, &b, modulo, root, 0, b[0][0].len());
            let duration_poly_mul_fft = start.elapsed();

            // Print results
            println!(
                "Degree: {}, poly_mul128: {:?}, poly_mul_fft: {:?}",
                degree, duration_poly_mul128, duration_poly_mul_fft
            );
        }
    }

    #[test]
    fn benchmark_poly_mat_mul_fft_vs_bubble() {
        return; // Skip this test for now
        // return; // Skip this test for now
        let matrix_sizes = vec![(2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (10,10,10), (20,20,20)]; // (m, n, k) sizes
        let degrees = vec![1,2, 5, 10, 20, 50, 100, 200, 500, 1000]; // Polynomial degrees to test
        let max_value = 1_000_000; // Maximum value for random coefficients
        let p = 998244353u128; // A large prime modulus
        let root = 3; // Primitive root for NTT

        for &(m, n, k) in &matrix_sizes {
            println!("Matrix size: {}x{}x{}", m, n, k);
            for &degree in &degrees {
                // Generate random matrices
                let a = generate_random_matrix(m, n, degree, max_value);
                let b = generate_random_matrix(n, k, degree, max_value);

                // Benchmark poly_mat_mul_bubble
                let start = Instant::now();
                let _result_bubble = poly_mat_mul_bubble(&a, &b, p);
                let duration_bubble = start.elapsed();

                // Benchmark poly_mat_mul_fft
                let start = Instant::now();
                let _result_fft = poly_mat_mul_fft(&a, &b, p, root, 0, b[0][0].len());
                let duration_fft = start.elapsed();

                // Print results
                println!(
                    "Degree: {}, Bubble: {:?}, FFT: {:?}",
                    degree, duration_bubble, duration_fft
                );
            }
        }
    }
}
