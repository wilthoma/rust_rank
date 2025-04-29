use crate::{matrices::GoodInteger, ntt::mod_pow, ntt::modinv, poly_mat_mul::poly_mat_mul_fft_red};
use nalgebra::DMatrix;
use std::{ops::{Add, Mul}, vec};




pub fn PM_Basis<T: GoodInteger+Into<u128> >(seq : &Vec<Vec<T>>, d:usize, seqprime: T, largeprime : u128, root : u128) -> Vec<Vec<Vec<u128>>> {
    // expand sequence to square matrix
    let nn = seq.len();
    let mut n=0;
    let nlen = seq[0].len();
    for i in 0..nn {
        if i*(i+1/2) == n {
            n = i;
        }
    }

    let mut G = vec![vec![vec![0; nlen]; n]; n];
    let mut ii = 0;
    for i in 0..n {
        for j in i..n {
            for k in 0..nlen {
                G[i][j][k] = seq[ii][k].into();
                G[j][i][k] = G[i][j][k];
            }
            ii += 1;
        }
    }

    let delta = vec![0; n];

    let (M, mu) = _PM_Basis(&G, d, &delta, seqprime.into(), largeprime, root);
    
    M
}



fn _PM_Basis(G : &Vec<Vec<Vec<u128>>>, d: usize, delta : &Vec<usize>, seqprime : u128, largeprime : u128, root : u128) -> (Vec<Vec<Vec<u128>>>, Vec<usize>) {
    // must have d= 0 or d= power of 2
    let n = G.len();

    if d == 0 {
        (unit_mat(n), delta.clone())
    } else if d==1 {
        M_Basis(G, delta, seqprime)
    } else {
        let (MM, mumu) = _PM_Basis(G, d/2, delta, seqprime, largeprime, root);
        let mut GG = poly_mat_mul_fft_red(&MM,&G, largeprime, root, seqprime);
        shift_trunc_in(&mut GG, d/2);
        let (MMM, mumumu) = _PM_Basis(&GG, d/2, &mumu, seqprime, largeprime, root);
        (poly_mat_mul_fft_red(&MMM ,&MM, largeprime, root, seqprime), mumumu)
    }
}


fn unit_mat(n : usize) -> Vec<Vec<Vec<u128>>> {
    let mut mat = vec![vec![vec![0]; n]; n];
    for i in 0..n {
        mat[i][i][0] = 1;
    }
    mat
}


fn shift_trunc_in(mat : &mut Vec<Vec<Vec<u128>>>, shiftd: usize) {
    let m = mat.len();
    let n = mat[0].len();
    for i in 0..m {
        for j in 0..n {
            for k in 0..(shiftd+1) {
                mat[i][j][k] = mat[i][j][k+shiftd];
            }
            mat[i][j].resize(shiftd +1, 0);
        }
    } 
}

// fn scalar_mat_to_poly_mat(mat : DMatrix<u128>, prime : u128) -> DMatrix<UPoly> {
//     DMatrix::from_fn(mat.nrows(), mat.ncols(), |i,j| {
//         let coeff = mat[(i,j)];
//         UPoly::from(coeff, prime)
//     })
// }

// fn mat_mul(a : &DMatrix<UPoly>, b : &DMatrix<UPoly>) -> DMatrix<UPoly> {
//     let n = a.nrows();
//     let m = a.ncols();
//     let k = b.ncols();
//     assert_eq!(m, b.nrows(), "Matrix dimensions do not match for multiplication");
    
//     DMatrix::from_fn(n, k, |i,j| {
//         let mut sum = UPoly::zero(a[(0,0)].p);
//         for l in 0..m {
//             sum = sum.add(&a[(i,l)].mul(&b[(l,j)]));
//         }
//         sum
//     })
// }

// fn mat_mulf(a : &DMatrix<FTPoly>, b : &DMatrix<FTPoly>) -> DMatrix<FTPoly> {
//     let n = a.nrows();
//     let m = a.ncols();
//     let k = b.ncols();
//     assert_eq!(m, b.nrows(), "Matrix dimensions do not match for multiplication");
    
//     DMatrix::from_fn(n, k, |i,j| {
//         let mut sum = FTPoly::zero(a[(0,0)].p, a[(0,0)].root);
//         for l in 0..m {
//             sum = sum.add(a[(i,l)].mul(b[(l,j)]));
//         }
//         sum
//     })
// }

// fn mat_coeff(a : DMatrix<UPoly>, k:usize) -> DMatrix<u128> {
//     let n = a.nrows();
//     let m = a.ncols();
//     DMatrix::from_fn(n, m, |i,j| {
//         if a[(i,j)].coeffs.len() > k {
//             a[(i,j)].coeffs[k]
//         } else {
//             0
//         }
//     })
// }




pub fn M_Basis(G : &Vec<Vec<Vec<u128>>>, delta : &Vec<usize>, prime : u128) -> (Vec<Vec<Vec<u128>>>, Vec<usize>) {
    let m = G.len();
    let n = G[0].len();
    assert!(m>=n, "Must have #rows >= #cols");
    let mut delta = delta.clone(); 


    // sort delta in descending order and remember the permutation
    let mut perm = (0..n).collect::<Vec<_>>();
    perm.sort_by(|&a, &b| delta[a].cmp(&delta[b]).reverse());
    delta.sort_by(|a, b| b.cmp(a));


    let mut pi_m = DMatrix::zeros(n, n);
    for i in 0..n {
        pi_m[(i, perm[i])] = 1;
    }
    // let mut Gk_inv = pi_m.clone();
    // Gk_inv = Gk_inv.try_inverse().unwrap();

    // delta.sort_by(|a, b| b.cmp(a));

    // let Delta0 = Gk_inv * M * (G[k-1].clone());
    let mut Delta = DMatrix::from_fn(n,n,|i,j| G[i][j][0]);
    Delta = &pi_m * &Delta;
    let mut Delta_aug = DMatrix::zeros(m,m);
    for i in 0..m {
        for j in 0..n {
            Delta_aug[(i,j)] = Delta[(i,j)];
        }
    }
    
    let (L,S,P) = lsp_decomposition(Delta, prime);
    let Linv = matrix_inverse(L, prime);

    // D = D2 + x Dx
    let mut D1 = DMatrix::zeros(n, n); 
    let mut Dx = DMatrix::zeros(n, n); 
    for i in 0..n {
        if S.row(i).iter().all(|&x| x == 0) {
            D1[(i,i)]=1;
        } else {
            Dx[(i,i)]=1;
        }
    }
    
    let M1 = &D1 * &Linv * &pi_m;
    let Mx = &Dx * &Linv * &pi_m;

    // fill output again in vects
    let mut ret = vec![vec![vec![0; 2]; n]; n];
    for i in 0..n {
        for j in 0..n {
            ret[i][j][0] = M1[(i,j)];
            ret[i][j][1] = Mx[(i,j)];
        }
    }


    for i in 0..n {
        delta[i] -= Dx[(i,i)] as usize;
    }


    (ret, delta)

}


fn modinv2(a: u128, p: u128) -> u128 {
    assert!(a != 0, "attempted to invert zero");
    let mut t = 0i128;
    let mut new_t = 1i128;
    let mut r = p as i128;
    let mut new_r = a as i128;

    while new_r != 0 {
        let quotient = r / new_r;
        t = t - quotient * new_t;
        std::mem::swap(&mut t, &mut new_t);
        r = r - quotient * new_r;
        std::mem::swap(&mut r, &mut new_r);
    }

    if r > 1 {
        panic!("modular inverse does not exist");
    }
    if t < 0 {
        t += p as i128;
    }

    t as u128
}

/// Computes the LSP decomposition of a matrix modulo p.
/// LSP decomposition: A = L * S * P (mod p)
pub fn lsp_decomposition(mut a: DMatrix<u128>, p: u128) -> (DMatrix<u128>, DMatrix<u128>, DMatrix<u128>) {
    let m = a.nrows();
    assert_eq!(m, a.ncols(), "Matrix must be square");

    let mut l = DMatrix::<u128>::identity(m, m);
    let mut perm = (0..m).collect::<Vec<usize>>();

    let mut rank = 0;

    for k in 0..m {
        // Find pivot in row k (column-wise search)
        let mut pivot_col = None;
        for j in k..m {
            if a[(k, j)] % p != 0 {
                pivot_col = Some(j);
                break;
            }
        }

        if let Some(j) = pivot_col {
            // Swap columns k <-> j in A and perm
            a.swap_columns(rank, j);
            perm.swap(rank, j);

            // Eliminate below
            let pivot_inv = modinv2(a[(k, rank)], p);
            for i in (k + 1)..m {
                let factor = (a[(i, rank)] * pivot_inv) % p;
                l[(i, k)] = (p-factor)%p;
                for col in rank..m {
                    let sub = (factor * a[(k, col)]) % p;
                    a[(i, col)] = (a[(i, col)] + p - sub) % p;
                }
            }

            rank += 1;
        }
        println!{"{}: {}",k,a};
    }

    // Build the permutation matrix P
    let mut pmat = DMatrix::<u128>::zeros(m, m);
    for (i, &j) in perm.iter().enumerate() {
        // pmat[(i, j)] = 1;
        pmat[(j, i)] = 1;
    }

    // Final modular reduction
    let s = a.map(|x| x % p);

    (l, s, pmat)
}

// fn lsp_factorization(Delta : DMatrix<u128>, prime : u128) -> (DMatrix<u128>, DMatrix<u128>, DMatrix<u128>) {
//     let n = Delta.nrows();
//     let mut L = DMatrix::identity(n, n);
//     let mut S = Delta.clone();
//     let mut P = DMatrix::identity(n, n);

//     for i in 0..n {
//         for j in 0..i {
//             let factor = S[(i,j)] / S[(j,j)];
//             L[(i,j)] = factor;
//             for k in 0..n {
//                 S[(i,k)] -= factor * S[(j,k)];
//             }
//         }
//     }

//     (L,S,P)
// }




/// Computes the modular inverse of a matrix modulo `p`.
/// Panics if the matrix is not invertible.
pub fn matrix_inverse(mut mat: DMatrix<u128>, p: u128) -> DMatrix<u128> {
    let n = mat.nrows();
    assert_eq!(n, mat.ncols(), "Matrix must be square");

    // Augment the matrix with the identity matrix
    let mut aug = DMatrix::zeros(n, 2 * n);
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = mat[(i, j)] % p;
        }
        aug[(i, n + i)] = 1;
    }

    // Perform Gaussian elimination
    for i in 0..n {
        // Find pivot
        if aug[(i, i)] == 0 {
            // Try to swap with a lower row
            let mut found = false;
            for j in (i + 1)..n {
                if aug[(j, i)] != 0 {
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



#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

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
            if s.row(i).iter().any(|&x| x % p != 0) {
                rank += 1;
            }
        }

        // Check upper triangular and nonzero diagonal in nonzero rows
        for i in 0..rank {
            if s[(i, i)] % p == 0 {
                return false;
            }
            for j in 0..i {
                if s[(i, j)] % p != 0 {
                    return false;
                }
            }
        }

        // Remaining rows must be all zero
        for i in rank..m {
            if s.row(i).iter().any(|&x| x % p != 0) {
                return false;
            }
        }

        true
    }

    fn check_lsp(a: DMatrix<u128>, p: u128) {
        let (l, s, pmat) = lsp_decomposition(a.clone(), p);

        println!("{} {} {} {}", a,l,s,pmat);

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
}
