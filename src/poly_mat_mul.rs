use bubblemath::linear_recurrence::poly_mul;
// use num_traits::Zero;
// use rayon::iter::IntoParallelIterator;

use crate::ntt::{matrix_ntt_parallel, mod_add, mod_mul, ntt, NTTInteger};
use rand::Rng;


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

pub fn poly_mat_mul_bubble(a: Vec<Vec<Vec<u64>>>, b: Vec<Vec<Vec<u64>>>, p: u64) -> Vec<Vec<Vec<u64>>> {
    let m = a.len();
    let n = a[0].len();
    let k = b[0].len();
    assert_eq!(n, b.len(), "Matrix dimensions do not match for multiplication");

    let nlena = a[0][0].len();
    let nlenb = b[0][0].len();
    let nlenres = nlena + nlenb - 1;
    let mut result = vec![vec![vec![0; nlenres]; k]; m];
    for i in 0..m {
        for j in 0..k {
            for l in 0..n {
                add_vects_in(&mut result[i][j], &poly_mul(&a[i][l], &b[l][j], p), p);
            }
        }
    }
    result

}


pub fn poly_mat_mul_fft<T : NTTInteger>(a: &Vec<Vec<Vec<T>>>, b: &Vec<Vec<Vec<T>>>, p: T, root: T) -> Vec<Vec<Vec<T>>> {
    let m = a.len();
    let n = a[0].len();
    let k = b[0].len();
    assert_eq!(n, b.len(), "Matrix dimensions do not match for multiplication");

    let nlena = a[0][0].len();
    let nlenb = b[0][0].len();
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
                fb[i][j][l] = b[i][j][l];
            }
        }
    }
    // compute ffts
    matrix_ntt_parallel(&mut fa, false, p, root);
    // for i in 0..m {
    //     for j in 0..n {
    //         ntt(&mut fa[i][j], false, p, root);
    //     }
    // }
    matrix_ntt_parallel(&mut fb, false, p, root);
    // for i in 0..n {
    //     for j in 0..k {
    //         ntt(&mut fb[i][j], false, p, root);
    //     }
    // }

    // multiply matrices
    for i in 0..m {
        for j in 0..k {
            for r in 0..n {
                for l in 0..nlenres_adj {
                    fresult[i][j][l] = mod_add(fresult[i][j][l], mod_mul(fa[i][r][l], fb[r][j][l],p), p);
                }
            }
        }
    }

    // compute iffts qnd resize result
    matrix_ntt_parallel(&mut fresult, true, p, root);
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
pub fn poly_mat_mul_fft_red<T : NTTInteger>(a: &Vec<Vec<Vec<T>>>, b: &Vec<Vec<Vec<T>>>, p: T, root: T, reducetoprime : T) -> Vec<Vec<Vec<T>>> {
    // check whether prime is large enough for NTT
    let mut max_power_of_two = 1;
    let mut k = 0;
    let deg_a = a[0][0].len();
    let deg_b = b[0][0].len();
    while (p.into() - 1) % (2 * max_power_of_two as u128 ) == 0 {
        max_power_of_two *= 2;
        k += 1;
    }
    let ntt_limit = max_power_of_two;

    let required_size = (deg_a + deg_b).next_power_of_two();
    assert!( required_size < ntt_limit, "Polynomials are too large for this modulus (NTT size too big) largeprime: {} sequence prime: {} degrees: {}, {}", p, reducetoprime, deg_a, deg_b);

    // no overflow possible?
    assert!( reducetoprime.into()*reducetoprime.into() * ((deg_a + deg_b) as u128) < p.into(), "Polynomials are too large for this modulus (overflow possible) largeprime: {} sequence prime: {} degrees: {}, {}", p, reducetoprime, deg_a, deg_b);

    // multiply
    let mut red = poly_mat_mul_fft(a, b, p, root);

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


#[cfg(test)]
mod tests {
    use super::*;

    fn generate_random_matrix(m: usize, n: usize, poly_len: usize, max_value: u64) -> Vec<Vec<Vec<u64>>> {
        let mut rng = rand::rng();
        (0..m)
            .map(|_| {
                (0..n)
                    .map(|_| {
                        (0..poly_len)
                            .map(|_| rng.random_range(0..max_value))
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
        let max_value = 500000; // Maximum value for random coefficients
        let p = 998244353; // A large prime modulus
        let root = 3; // Primitive root for NTT

        // Generate random input matrices
        let a = generate_random_matrix(m, n, poly_len_a, max_value);
        let b = generate_random_matrix(n, k, poly_len_b, max_value);

        // Compute results using both methods
        let result_bubble = poly_mat_mul_bubble(a.clone(), b.clone(), p);
        let result_fft = poly_mat_mul_fft(&a, &b, p, root);

        // Assert that the results are the same
        assert_eq!(result_bubble, result_fft, "Results from poly_mat_mul_bubble and poly_mat_mul_fft do not match");
    }
}
