use crate::{matrices::GoodInteger, ntt::{mod_pow, modinv, NTTInteger, ModMul}, poly_mat_mul::poly_mat_mul_fft_red};
use nalgebra::DMatrix;
use std::{fmt::Debug, ops::{Add, Mul}, vec};




/// Computes the modular inverse of a matrix modulo `p`.
/// Panics if the matrix is not invertible.
pub fn matrix_inverse<T: NTTInteger>(mat: &DMatrix<T>, p: T) -> DMatrix<T> {
    let mat = mat.clone();
    let n = mat.nrows();
    assert_eq!(n, mat.ncols(), "Matrix must be square");

    // Augment the matrix with the identity matrix
    let mut aug = DMatrix::zeros(n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = mat[(i, j)] % p;
        }
        aug[(i, n + i)] = T::one();
    }

    // Perform Gaussian elimination
    for i in 0..n {
        // Find pivot
        if aug[(i, i)] == T::zero() {
            // Try to swap with a lower row
            let mut found = false;
            for j in (i + 1)..n {
                if aug[(j, i)] != T::zero() {
                    aug.swap_rows(i, j);
                    found = true;
                    break;
                }
            }
            if !found {
                panic!("Matrix is singular and not invertible");
            }
        }

        // Normalize pivot row
        let inv = modinv(aug[(i, i)], p);
        for k in 0..2 * n {
            aug[(i, k)] = (aug[(i, k)] * inv) % p;
        }

        // Eliminate other rows
        for j in 0..n {
            if j != i {
                let factor = aug[(j, i)];
                for k in 0..2 * n {
                    let tmp = (aug[(i, k)] * factor) % p;
                    aug[(j, k)] = (aug[(j, k)] + p - tmp) % p;
                }
            }
        }
    }

    // Extract inverse matrix
    let mut inv_mat = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            inv_mat[(i, j)] = aug[(i, n + j)];
        }
    }

    inv_mat
}




pub fn modular_determinant<T : NTTInteger + Debug>(mat: &DMatrix<T>, p: T) -> T {
    assert!(mat.is_square(), "Matrix must be square to compute determinant mod p");

    let mut mat = mat.clone();
    let n = mat.nrows();
    let mut det = T::one();
    let mut sign = 1i64;

    // Reduce all entries mod p
    for i in 0..n {
        for j in 0..n {
            mat[(i, j)] = (mat[(i, j)] % p + p) % p;
        }
    }

    for i in 0..n {
        // 1. Find pivot
        let mut pivot_row = None;
        for row in i..n {
            if mat[(row, i)] != T::zero() {
                pivot_row = Some(row);
                break;
            }
        }

        if let Some(pivot_idx) = pivot_row {
            if pivot_idx != i {
                mat.swap_rows(i, pivot_idx);
                sign = -sign;
            }

            let pivot = mat[(i, i)];
            det = (det * pivot) % p ;

            let inv = modinv(pivot, p);

            // 2. Eliminate rows below
            for row in (i + 1)..n {
                let factor = mat[(row, i)] * inv % p;
                for col in i..n {
                    let sub = factor * mat[(i, col)] % p;
                    mat[(row, col)] = (mat[(row, col)] +p - sub) % p;
                }
            }
        } else {
            // Pivot = 0 → determinant is 0
            return T::zero();
        }
    }

    if sign == -1 {
        det = (p - det) % p;
    }

    det
}


pub fn modular_rank<T : NTTInteger + Debug>(mat: &DMatrix<T>, p: T) -> usize {
    let mut mat = mat.clone();
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    let mut rank = 0;

    // Reduce entries modulo p
    for i in 0..nrows {
        for j in 0..ncols {
            mat[(i, j)] = (mat[(i, j)] % p + p) % p;
        }
    }

    for col in 0..ncols {
        // 1. Find a pivot row
        let mut pivot_row = None;
        for row in rank..nrows {
            if mat[(row, col)] != T::zero() {
                pivot_row = Some(row);
                break;
            }
        }

        if let Some(pivot_idx) = pivot_row {
            // 2. Swap pivot row into position
            mat.swap_rows(rank, pivot_idx);

            // 3. Normalize pivot row (optional for rank, but clean)
            let inv = modinv(mat[(rank, col)], p);
            for j in col..ncols {
                mat[(rank, j)] = (mat[(rank, j)] * inv) % p;
            }

            // 4. Eliminate rows below
            for row in (rank + 1)..nrows {
                let factor = mat[(row, col)];
                for j in col..ncols {
                    let sub = factor * mat[(rank, j)] % p;
                    mat[(row, j)] = (mat[(row, j)] +p - sub) %p;//.rem_euclid(p);
                }
            }

            rank += 1;
            if rank == nrows {
                break;
            }
        }
    }

    rank
}


/// Computes the LSP decomposition of a matrix modulo p.
/// LSP decomposition: A = L * S * P (mod p)
pub fn lsp_decomposition<T:NTTInteger>(a: &DMatrix<T>, p: T) -> (DMatrix<T>, DMatrix<T>, DMatrix<T>) {
    let mut a = a.clone();
    let m = a.nrows();
    assert_eq!(m, a.ncols(), "Matrix must be square");

    let mut l = DMatrix::<T>::identity(m, m);
    let mut perm = (0..m).collect::<Vec<usize>>();

    let mut rank = 0;

    for k in 0..m {
        // Find pivot in row k (column-wise search)
        let mut pivot_col = None;
        for j in rank..m {
            if a[(k, j)] % p != T::zero() {
                pivot_col = Some(j);
                break;
            }
        }

        if let Some(j) = pivot_col {
            // Swap columns k <-> j in A and perm
            a.swap_columns(rank, j);
            perm.swap(rank, j);

            // Eliminate below
            let pivot_inv = modinv(a[(k, rank)], p);
            for i in (k + 1)..m {
                let factor = (a[(i, rank)] * pivot_inv) % p;
                l[(i, k)] = factor;
                for col in rank..m {
                    let sub = (factor * a[(k, col)]) % p;
                    a[(i, col)] = (a[(i, col)] + p - sub) % p;
                }
            }

            rank += 1;
        }
    }

    // Build the permutation matrix P
    let mut pmat = DMatrix::<T>::zeros(m, m);
    for (i, &j) in perm.iter().enumerate() {
        // pmat[(i, j)] = 1;
        pmat[(j, i)] = T::one();
    }

    // Final modular reduction
    let s = a.map(|x| x % p);

    (l, s, pmat)
}


fn matrix_multiply_classic<T : NTTInteger>(a : &Vec<Vec<Vec<T>>>, b : &Vec<Vec<Vec<T>>>) -> Vec<Vec<Vec<T>>> {
    let n = a.len();
    let m = a[0].len();
    let p = b[0].len();
    let nlen  = a[0][0].len();

    let mut res : Vec<Vec<Vec<T>>> = vec![vec![vec![T::zero(); nlen]; p]; n];

    for i in 0..n {
        for j in 0..p {
            for kk in 0..nlen {
                for k in 0..m {
                    res[i][j][kk] = res[i][j][kk].addmod(a[i][k][kk].mulmod(b[k][j][kk]));
                }
            }
        }
    }
    res
}

    fn matrix_multiply_strassen<T: NTTInteger>(a: &Vec<Vec<Vec<T>>>, b: &Vec<Vec<Vec<T>>>) -> Vec<Vec<Vec<T>>> {
        let n = a.len();
        let nlen = a[0][0].len();

        if n == 1 {
            let mut res = vec![vec![vec![T::zero(); nlen]; 1]; 1];
            for i in 0..nlen {
                res[0][0][i] = a[0][0][i].mulmod(b[0][0][i]);
            }
            return res;
        }

        let half = n / 2;

        let mut a11 = vec![vec![vec![T::zero(); nlen]; half]; half];
        let mut a12 = vec![vec![vec![T::zero(); nlen]; half]; half];
        let mut a21 = vec![vec![vec![T::zero(); nlen]; half]; half];
        let mut a22 = vec![vec![vec![T::zero(); nlen]; half]; half];
        let mut b11 = vec![vec![vec![T::zero(); nlen]; half]; half];
        let mut b12 = vec![vec![vec![T::zero(); nlen]; half]; half];
        let mut b21 = vec![vec![vec![T::zero(); nlen]; half]; half];
        let mut b22 = vec![vec![vec![T::zero(); nlen]; half]; half];

        for i in 0..half {
            for j in 0..half {
                for k in 0..nlen {
                    a11[i][j][k] = a[i][j][k];
                    a12[i][j][k] = a[i][j + half][k];
                    a21[i][j][k] = a[i + half][j][k];
                    a22[i][j][k] = a[i + half][j + half][k];
                    b11[i][j][k] = b[i][j][k];
                    b12[i][j][k] = b[i][j + half][k];
                    b21[i][j][k] = b[i + half][j][k];
                    b22[i][j][k] = b[i + half][j + half][k];
                }
            }
        }

        let m1 = matrix_multiply_strassen(
            &matrix_add(&a11, &a22, nlen),
            &matrix_add(&b11, &b22, nlen),
        );
        let m2 = matrix_multiply_strassen(&matrix_add(&a21, &a22, nlen), &b11);
        let m3 = matrix_multiply_strassen(&a11, &matrix_sub(&b12, &b22, nlen));
        let m4 = matrix_multiply_strassen(&a22, &matrix_sub(&b21, &b11, nlen));
        let m5 = matrix_multiply_strassen(&matrix_add(&a11, &a12, nlen), &b22);
        let m6 = matrix_multiply_strassen(&matrix_sub(&a21, &a11, nlen), &matrix_add(&b11, &b12, nlen));
        let m7 = matrix_multiply_strassen(&matrix_sub(&a12, &a22, nlen), &matrix_add(&b21, &b22, nlen));

        let c11 = matrix_add(
            &matrix_sub(&matrix_add(&m1, &m4, nlen), &m5, nlen),
            &m7,
            nlen,
        );
        let c12 = matrix_add(&m3, &m5, nlen);
        let c21 = matrix_add(&m2, &m4, nlen);
        let c22 = matrix_add(
            &matrix_sub(&matrix_add(&m1, &m3, nlen), &m2, nlen),
            &m6,
            nlen,
        );

        let mut res = vec![vec![vec![T::zero(); nlen]; n]; n];
        for i in 0..half {
            for j in 0..half {
                for k in 0..nlen {
                    res[i][j][k] = c11[i][j][k];
                    res[i][j + half][k] = c12[i][j][k];
                    res[i + half][j][k] = c21[i][j][k];
                    res[i + half][j + half][k] = c22[i][j][k];
                }
            }
        }

        res
    }

    fn matrix_add<T: NTTInteger>(
        a: &Vec<Vec<Vec<T>>>,
        b: &Vec<Vec<Vec<T>>>,
        nlen: usize,
    ) -> Vec<Vec<Vec<T>>> {
        let n = a.len();
        let mut res = vec![vec![vec![T::zero(); nlen]; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..nlen {
                    res[i][j][k] = a[i][j][k].addmod( b[i][j][k] );
                }
            }
        }
        res
    }

    fn matrix_sub<T: NTTInteger>(
        a: &Vec<Vec<Vec<T>>>,
        b: &Vec<Vec<Vec<T>>>,
        nlen: usize,
    ) -> Vec<Vec<Vec<T>>> {
        let n = a.len();
        let mut res = vec![vec![vec![T::zero(); nlen]; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..nlen {
                    res[i][j][k] = a[i][j][k].submod(b[i][j][k]);
                }
            }
        }
        res
    }





#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use std::time::Instant;

    fn is_unit_lower_triangular(l: &DMatrix<u128>, p: u128) -> bool {
        let n = l.nrows();
        for i in 0..n {
            for j in 0..n {
                let val = l[(i, j)] % p;
                if i == j && val != 1 {
                    return false;
                } else if i < j && val != 0 {
                    return false;
                }
            }
        }
        true
    }

    fn is_valid_s_matrix(s: &DMatrix<u128>, p: u128) -> bool {
        let m = s.nrows();

        // Count nonzero rows from the top
        let mut rank = 0;
        for i in 0..m {
            // check that all entries to the left are zero
            for j in 0..rank {
                if s[(i,j)] % p != 0 {
                    return false;
                }
            } 
            // if row is nonzero, then rank entry must be nonzero
            if s.row(i).iter().any(|&x| x % p != 0) {
                if s[(i, rank)] % p == 0 {
                    return false;
                }

                rank += 1;
            }
        }

        true
    }

    fn check_lsp(a: DMatrix<u128>, p: u128) {
        let (l, s, pmat) = lsp_decomposition(&a, p);

        // println!("{} {} {} {}", a,l,s,pmat);

        let reconstructed = (l.clone() * s.clone() * pmat.clone()).map(|x| x % p);
        let original_modp = a.map(|x| x % p);

        assert_eq!(original_modp, reconstructed, "Reconstructed matrix does not match original");
        assert!(is_unit_lower_triangular(&l, p), "L is not unit lower triangular");
        assert!(is_valid_s_matrix(&s, p), "S matrix is not in expected form");
    }

    #[test]
    fn test_lsp_decomposition_small() {
        let p = 101;
        let a = DMatrix::<u128>::from_row_slice(3, 3, &[
            2, 4, 6,
            1, 3, 5,
            0, 0, 1,
        ]);
        check_lsp(a, p);
    }

    #[test]
    fn test_lsp_decomposition_larger() {
        let p = 97;
        let a = DMatrix::<u128>::from_row_slice(5, 5, &[
            10, 20, 30, 40, 50,
             5, 10, 15, 20, 25,
             0,  1,  0,  1,  0,
             1,  0,  1,  0,  1,
            99, 99, 99, 99, 99, // mod 97 => 2
        ]);
        check_lsp(a, p);
    }

    fn is_identity_mod(mat: &DMatrix<u128>, p: u128) -> bool {
        let n = mat.nrows();
        if mat.ncols() != n {
            return false;
        }
        for i in 0..n {
            for j in 0..n {
                let val = mat[(i, j)] % p;
                if (i == j && val != 1) || (i != j && val != 0) {
                    return false;
                }
            }
        }
        true
    }

    #[test]
    fn test_modular_inverse_small_matrix() {
        let p = 97;
        let a = DMatrix::<u128>::from_row_slice(3, 3, &[
            2, 3, 1,
            1, 1, 1,
            3, 5, 2,
        ]);

        let inv = matrix_inverse(&a, p);
        let product = (&a * &inv).map(|x| x % p);

        assert!(is_identity_mod(&product, p), "A * A⁻¹ != I mod p");
    }

    #[test]
    fn test_modular_inverse_identity() {
        let p = 97;
        let a = DMatrix::<u128>::identity(4, 4);
        let inv = matrix_inverse(&a, p);
        assert_eq!(inv, a, "Inverse of identity should be identity");
    }

    #[test]
    fn test_modinv_basic() {
        let p = 97u64;

        for a in 1..p {
            let inv = modinv(a, p);
            assert_eq!((a * inv) % p, 1, "modinv({a}, {p}) * {a} != 1 mod {p}");
        }
    }

    #[test]
    fn test_matrix_multiply_performance_and_correctness() {
        return; // Skip this test for now
        let n = 128; // Matrix size (must be a power of 2 for Strassen's algorithm)
        let nlen = 500; // Length of modular coefficients
        let p = u128::PRIME; // Modulus

        // Generate random matrices a and b
        let mut a = vec![vec![vec![0u128; nlen]; n]; n];
        let mut b = vec![vec![vec![0u128; nlen]; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..nlen {
                    a[i][j][k] = (i + j + k) as u128 % p;
                    b[i][j][k] = (i * j + k) as u128 % p;
                }
            }
        }

        // Measure execution time and compute result for classic multiplication
        let start_classic = Instant::now();
        let result_classic = matrix_multiply_classic(&a, &b);
        let duration_classic = start_classic.elapsed();

        // Measure execution time and compute result for Strassen's multiplication
        let start_strassen = Instant::now();
        let result_strassen = matrix_multiply_strassen(&a, &b);
        let duration_strassen = start_strassen.elapsed();

        println!(
            "Classic multiplication took: {:?}, Strassen's multiplication took: {:?}",
            duration_classic, duration_strassen
        );

        // Measure execution time and compute result for classic multiplication
        let start_classic = Instant::now();
        let result_classic = matrix_multiply_classic(&a, &b);
        let duration_classic = start_classic.elapsed();

        // Measure execution time and compute result for Strassen's multiplication
        let start_strassen = Instant::now();
        let result_strassen = matrix_multiply_strassen(&a, &b);
        let duration_strassen = start_strassen.elapsed();

        // Print execution times
        println!(
            "Classic multiplication took: {:?}, Strassen's multiplication took: {:?}",
            duration_classic, duration_strassen
        );

        // Compare results
        assert_eq!(
            result_classic, result_strassen,
            "Results of classic and Strassen's multiplication do not match"
        );


    }

    // #[test]
    // #[should_panic(expected = "attempted to invert zero")]
    // fn test_modinv_zero_panics() {
    //     let _ = modinv(0, 97u64);
    // }

    // #[test]
    // fn test_pm_basis() {

    //     let p = 998244353; // large prime modulus suitable for NTT
    //     let n = 3; // size of the square matrix A
    //     let m = 2; // size of the rectangular matrices U and V

    //     // Generate a random square matrix A
    //     let a = DMatrix::<u128>::from_row_slice(n, n, &[
    //         2, 3, 1,
    //         1, 1, 1,
    //         3, 5, 2,
    //     ]);

    //     // Generate random rectangular matrices U and V
    //     let u = DMatrix::<u128>::from_row_slice(m, n, &[
    //         1, 0, 2,
    //         0, 1, 3,
    //         // 1, 1, 1,
    //         // 2, 0, 1,
    //     ]);

    //     let v = DMatrix::<u128>::from_row_slice(n, m, &[
    //         1, 2, 0, 1,
    //         0, 1, //1, 0,
    //         //2, 0, 1, 1,
    //     ]);

    //     // Generate G as U^T * A^k * V for k = 1, 2, ...
    //     let mut g = vec![];
    //     let mut ak = a.clone();
    //     for _ in 0..n {
    //         let gk = (&u.transpose() * &ak * &v).map(|x| x % p);
    //         let gk_vec = gk.row_iter()
    //             .map(|row| row.iter().cloned().collect::<Vec<u128>>())
    //             .collect::<Vec<Vec<u128>>>();
    //         g.push(gk_vec);
    //         ak = (&ak * &a).map(|x| x % p); // Compute A^(k+1)
    //     }

    //     // Convert G into the required format for _PM_Basis
    //     let g_vec = g.iter()
    //         .map(|matrix| matrix.clone())
    //         .collect::<Vec<Vec<Vec<u128>>>>();

    //     // Call _PM_Basis
    //     let d = 3; // Example depth
    //     let delta = vec![0; n];
    //     let seqprime = 1; // Example sequence prime
    //     let root = 3; // Example root
    //     let (m_basis, mu) = _PM_Basis(&g_vec, d, &delta, seqprime, p, root);

    //     // Validate the output
    //     assert_eq!(m_basis.len(), n, "M_Basis should have the correct dimensions");
    //     assert_eq!(mu.len(), n, "Mu should have the correct dimensions");

    //     // Additional checks can be added based on expected properties of M_Basis and Mu
    // }
}
