


use nalgebra::DMatrix;
use petgraph::algo::k_shortest_path;

use crate::{invariant_factor::{top_invariant_factor, vec_matrix_to_poly_matrix}, matrices::GoodInteger, ntt::{NTTInteger, modinv as modinv2}};
// use std::ops::{Add, Mul, Sub};

type Matrix<T> = DMatrix<T>;

// use std::cmp::{min, max};

/// Matrix Berlekamp-Massey algorithm
pub fn matrix_berlekamp_massey(m: &[DMatrix<i64>], delta: usize, p: i64) -> Option<Vec<DMatrix<i64>>> {
    let n = m[0].nrows();
    let mut f: Vec<DMatrix<i64>> = vec![DMatrix::zeros(n, 2 * n); 1];
    for i in 0..n {
        f[0][(i, i)] = 1;
    }
    // println!("1");

    let mut d = vec![0; n];
    d.extend(vec![1; n]);

    let mut t = -1;
    let mut beta = 1;
    let mut sigma: usize = d.iter().take(n).sum();
    let mut mu: usize = *d.iter().take(n).max().unwrap();

    while beta < delta - sigma + mu + 1 && t+2< m.len() as i64 { // todo: termination condition
        t += 1;

        if t % 100 == 0 {
            println!("Block BM progress t: {}", t);
        }
        // println!("{} {}",t, m.len());
        // print!("{:?}", d);

        let mut phi_t = DMatrix::zeros(n, 2 * n);
        for (i, mi) in m.iter().enumerate() {
            if i > t as usize { break; }
            if t as usize - i < f.len() {
                phi_t += mi * &f[t as usize - i];
                phi_t.apply(|x| *x = (*x % p + p) % p);
            }
        }
        // println!("2 {}", t);
        // MBM 5
        let (tau, new_d) = auxiliary_gaussian_elimination(&phi_t, &d, p);
        d = new_d;



        sigma = d.iter().take(n).sum();
        mu = *d.iter().take(n).max().unwrap();
        beta = *d.iter().skip(n).min().unwrap();

        if sigma >= delta + 1 {
            println!("Insufficient bound. {}, {:?}", t, d);
            return None;
        }
        // println!("3");
        f = update_f(&f, &tau, n, delta + 1, p);

                // MBM 6
                for i in n..2*n {
                    d[i] = d[i]+1;
                }
        // println!("d: {:?} f: {:?}", d, f.iter().map(|x| (x[(0,0)],x[(0,1)])).collect::<Vec<_>>());
    }

    // println!("4");
    let maxd = d.iter().take(n).max().unwrap();
    // println!("maxd: {:?}", maxd);
    let mut F = vec![DMatrix::zeros(n, n); maxd + 1];
    for j in 0..n {
        for (k, coeff) in f.iter().enumerate() {
            if k > d[j] { break; }
            for i in 0..n {
                F[d[j]-k][(i, j)] = coeff[(i, j)];
                //poly[d[j] - k][(i, 0)] = coeff[(i, j)];
            }
        }
    }
    // println!("Result........");
    // print!("{:?}", d);
    // printseries(&f, 4);
    // printseries(&F, 4);

    Some(F)
}

// /// Builds a Matrix<T> of size (N x 2N) with the coefficient of z^k from each column polynomial in f
// fn get_poly_coeff(f: &[Matrix<i64>], k: usize) -> Matrix<i64>
// {
//     let n = f[0].nrows();
//     let two_n = f.len();
//     let mut coeff_mat = Matrix::<i64>::zeros(n, two_n);

//     for j in 0..two_n {
//         let f_j = &f[j];
//         if k < f_j.ncols() {
//             // f_j is the polynomial column vector, represented as matrix
//             // and the k-th coefficient is the k-th column
//             let coeff_col = f_j.column(k);
//             for i in 0..n {
//                 coeff_mat[(i, j)] = coeff_col[i].clone();
//             }
//         }
//     }

//     coeff_mat
// }

// /// Computes discrepancy Ξ = Coeff(t; M(z) * f(z))
// fn compute_discrepancy(
//     m_series: &[Matrix<i64>], // M(z) coefficients
//     f: &[Matrix<i64>],        // vector of 2N columns, each column is a polynomial in matrix form
//     t: usize,
// ) -> Matrix<i64>
// {
//     let n = m_series[0].nrows();
//     let two_n = f.len();
//     let mut result = Matrix::<i64>::zeros(n, two_n);

//     for k in 0..=t {
//         if k >= m_series.len() {
//             break;
//         }
//         let m_k = &m_series[k]; // M_k
//         let f_tk = get_poly_coeff(f, t - k); // coefficient of z^{t-k} from f(z)
//         let product = m_k * f_tk;
//         result += product;
//     }

//     result
// }


/// Multiply two polynomials of matrices modulo z^delta
fn poly_mat_mul(a: &[DMatrix<i64>], b: &[DMatrix<i64>], delta: usize, p: i64) -> Vec<DMatrix<i64>> {
    let n = a[0].nrows();
    let nc = a[0].ncols();
    let nb = b[0].nrows();
    let nbc = b[0].ncols();
    // print!("poly_mat_mul: a: {}x{}, b: {}x{}", n, nc, nb, nbc);
    let mut result = vec![DMatrix::zeros(n, nc); delta];
    for i in 0..a.len() {
        for j in 0..b.len() {
            if i + j < delta {
                result[i + j] = &result[i + j] + &(&a[i] * &b[j]);
                result[i + j].apply(|x| *x = (*x % p + p) % p);
            }
        }
    }
    result
}

fn shift_auxiliary_part(f: &mut Vec<DMatrix<i64>>, n: usize, p: i64) {
    // println!("shift");
    let rows = f[0].nrows();
    let mut shifted = vec![DMatrix::zeros(rows, 2 * n); f.len() + 1];

    // Shift auxiliary part (columns n..2n) one degree up
    for i in 0..f.len() {
        for row in 0..rows {
            for col in 0..n {
                shifted[i + 1][(row, n + col)] = f[i][(row, n + col)] % p;
            }
        }
    }

    // Copy generator part (columns 0..n) unshifted
    for i in 0..f.len() {
        for row in 0..rows {
            for col in 0..n {
                shifted[i][(row, col)] = f[i][(row, col)] % p;
            }
        }
    }
    // println!("/shift");

    // Truncate to original length
    // shifted.truncate(f.len());
    *f = shifted;
}

/// Update MBM9 correctly: f(z) <- f(z) * tau, then shift auxiliary part by z
fn update_f(f: &Vec<DMatrix<i64>>, tau: &DMatrix<i64>, n: usize, delta: usize, p: i64) -> Vec<DMatrix<i64>> {
    // println!("update");
    let mut new_f = poly_mat_mul(f, &[tau.clone()], delta + 1, p); // +1 to allow for shift
    shift_auxiliary_part(&mut new_f, n, p);
    new_f.truncate(delta);
    // println!("/update");
    new_f
}

/// Modular inverse using the extended Euclidean algorithm
fn modinv(a: i64, p: i64) -> i64 {
    let (mut t, mut new_t) = (0, 1);
    let (mut r, mut new_r) = (p, a);

    while new_r != 0 {
        let quotient = r / new_r;
        t = t - quotient * new_t;
        std::mem::swap(&mut t, &mut new_t);
        r = r - quotient * new_r;
        std::mem::swap(&mut r, &mut new_r);
    }

    if r > 1 {
        panic!("a is not invertible");
    }
    if t < 0 {
        t += p;
    }
    t
}

/// Perform auxiliary Gaussian elimination on phi_t and return transformation matrix tau and updated degrees
fn auxiliary_gaussian_elimination(phi_t: &DMatrix<i64>, d: &[usize], p: i64) -> (DMatrix<i64>, Vec<usize>) {
    // println!("{:?}", phi_t);
    let n = phi_t.nrows();
    let mut tau = DMatrix::identity(2 * n, 2 * n);
    let mut phi = phi_t.clone();
    let mut d = d.to_vec();
    let mut a: Vec<usize> = (0..n).collect();

    // println!("a");
    for i in 0..n {
        let mut b_i: Vec<usize> = a.iter().copied().filter(|&j| phi[(i, j)] % p != 0).collect();
        b_i.push(n + i);

        if b_i.is_empty() {
            continue;
        }
        // println!("b");

        let mut l = *b_i.iter().min_by_key(|&&j| d[j]).unwrap();
        if d[l] == d[n+i] && l != n+i {
            l = n+i;
        }
        b_i.retain(|&j| j != l);

        for &j in &b_i {
            if l == n + i {
                if phi[(i, n + i)] != 0 {
                    let inv = modinv(phi[(i, n + i)], p);
                    let factor:i64 = (phi[(i, j)] * inv).rem_euclid(p);
                    for row in 0..n {
                        phi[(row, j)] = (phi[(row, j)] - factor * phi[(row, n + i)]).rem_euclid(p);
                    }
                    for row in 0..2 * n {
                        let tmp : i64 = tau[(row, j)] - factor  * tau[(row, n + i)];
                        tau[(row, j)] = tmp.rem_euclid(p);
                        // println!("{:?} ", tau);
                    }
                }
            } else if j < n + i {
                if phi[(i, l)] != 0 {
                    let inv = modinv(phi[(i, l)], p);
                    let factor = (phi[(i, j)] * inv).rem_euclid(p);
                    for row in 0..n {
                        phi[(row, j)] = (phi[(row, j)] - factor * phi[(row, l)]).rem_euclid(p);
                    }
                    for row in 0..2 * n {
                        tau[(row, j)] = (tau[(row, j)] - factor * tau[(row, l)]).rem_euclid(p);
                    }
                }
            } else if j == n + i {
                if phi[(i, n + i)] != 0 {
                    let inv = modinv(phi[(i, n + i)], p);
                    let factor = (-phi[(i, l)] * inv).rem_euclid(p);
                    for row in 0..n {
                        phi[(row, n + i)] = (phi[(row, n + i)] * factor + phi[(row, l)]).rem_euclid(p);
                    }
                    for row in 0..2 * n {
                        tau[(row, n + i)] = (tau[(row, n + i)] * factor + tau[(row, l)]).rem_euclid(p);
                    }
                    for row in 0..n {
                        phi.swap((row, l), (row, n + i));
                    }
                    for row in 0..2 * n {
                        tau.swap((row, l), (row, n + i));
                    }
                    d.swap(l, n + i);
                } else {
                    for row in 0..2 * n {
                        tau[(row, n + i)] = (tau[(row, n + i)] + tau[(row, l)]).rem_euclid(p);
                    }
                    d.swap(l, n + i);
                    a.retain(|&j| j != l);
                }
            }
        }
    }

    (tau, d)
}


// #[test]
pub fn test_matrix_berlekamp_massey_simple() {
    let p = 7;
    let n = 2;
    let delta = 4;

    // Define M0, M1, M2 in F_7
    let m0 = Matrix::from_vec(n, n, vec![
        1, 2,
        0, 3
    ]);

    let m1 = Matrix::from_vec(n, n, vec![
        4, 0,
        2, 1
    ]);

    let m2 = Matrix::from_vec(n, n, vec![
        5, 6,
        1, 2
    ]);

    let m_coeffs = vec![m0, m1, m2];

    // display error
    // let result = match matrix_berlekamp_massey(&m_coeffs, delta, p) {
    //     Ok(g) => g,
    //     Err(e) => {
    //         eprintln!("Failed to compute minpoly: {:?}", e);
    //         // triangle_counts.push(0);
    //         return;
    //     }
    // };
    let result = matrix_berlekamp_massey(&m_coeffs, delta, p).unwrap();

    println!("Matrix generator coefficients:");
    for (i, mat) in result.iter().enumerate() {
        println!("z^{}:\n{}", i, mat);
    }
}

pub fn printseries(v : &Vec<Matrix<i64>>, d : usize) {
    for i in 0..d {
        if i>=v.len() {
            break;
        }
        println!("z^{}:\n{}", i, v[i]);
    }
}


pub fn test_matrix_berlekamp_massey_simple2() {
    let p = 7;
    let n = 2;
    let delta = 6;

    // Define M0, M1, M2 in F_7
    let m0 = Matrix::from_vec(n, n, vec![
        0, 1,
        0, 0
    ]);

    let m1 = Matrix::from_vec(n, n, vec![
        1, 1,
        0, 1
    ]);

    let m2 = Matrix::from_vec(3, 3, vec![
        1, 1,0,
        0, 1,1,
        0,0,1
    ]);

    let m0_series: Vec<_> = (0..10).map(|i| m0.pow(i)).collect();
    let m1_series: Vec<_>  = (0..10).map(|i| m1.pow(i)).collect();
    let m2_series: Vec<_>  = (0..10).map(|i| m2.pow(i)).collect();

    printseries(&m0_series, 4);
    printseries(&m1_series, 4);
    printseries(&m2_series, 4);

    // display error
    // let result = match matrix_berlekamp_massey(&m_coeffs, delta, p) {
    //     Ok(g) => g,
    //     Err(e) => {
    //         eprintln!("Failed to compute minpoly: {:?}", e);
    //         // triangle_counts.push(0);
    //         return;
    //     }
    // };
    let result0 = matrix_berlekamp_massey(&m0_series, delta, p).unwrap();
    
    let result1 = matrix_berlekamp_massey(&m1_series, delta, p).unwrap();
    let result2 = matrix_berlekamp_massey(&m2_series, delta, p).unwrap();


    println!("Matrix generator coefficients:");
    for (i, mat) in result0.iter().enumerate() {
        println!("z^{}:\n{}", i, mat);
    }
    println!("Matrix generator coefficients:");
    for (i, mat) in result1.iter().enumerate() {
        println!("z^{}:\n{}", i, mat);
        
    }
    println!("Matrix generator coefficients:");

    printseries(&result2, 4);

    // 
    println!("{:?}", vec_matrix_to_poly_matrix(&result2,p));
    println!("Top invariant factors");
    let f0 = top_invariant_factor(vec_matrix_to_poly_matrix(&result0,p));
    println!("{:?}", f0);
    let f1 = top_invariant_factor(vec_matrix_to_poly_matrix(&result1,p));
    println!("{:?}", f1);

    let f2 = top_invariant_factor(vec_matrix_to_poly_matrix(&result2,p));
    println!("{:?}", f2);


}

pub fn vecvec_to_symmetric_matrix_list<T:GoodInteger+Into<i64>>(v: &Vec<Vec<T>>, n:usize) -> Vec<DMatrix<i64>> {
    assert_eq!(v.len(), (n*(n+1)/2) as usize, "v length must be n*(n+1)/2");
    let deg = v[0].len();
    let mut m = vec![DMatrix::zeros(n, n); deg];
    let mut ii =0;
    for i in 0..n {
        for j in i..n {
            for k in 0..deg {
                m[k][(i, j)] = v[ii][k].into();
                m[k][(j, i)] = v[ii][k].into();
            }
            ii += 1;
        }
    }
    m
}

pub fn is_generating_poly(poly : &Vec<Matrix<i64>>, seq : &Vec<Matrix<i64>>, p:i64) -> bool {
    let n = poly[0].nrows();
    let k = poly.len();
    let N = seq.len();
    
    for l in 0..N-k {
        let mut sum = Matrix::zeros(n, n);
        for i in 0..k {
            sum += &seq[l+i] * &poly[i];
            sum.apply(|x| *x = (*x % p + p) % p);
        }
        if sum != Matrix::zeros(n, n) {
            println!("Not generating poly {} {:?} {:?}",l, sum, seq[0]);
            return false;
        }
    }
    true
}


pub fn modular_rank(mat: &DMatrix<i64>, p: i64) -> usize {
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
            if mat[(row, col)] != 0 {
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
                    mat[(row, j)] = (mat[(row, j)] - factor * mat[(rank, j)]).rem_euclid(p);
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

pub fn modular_determinant(mat: &DMatrix<i64>, p: i64) -> i64 {
    assert!(mat.is_square(), "Matrix must be square to compute determinant mod p");

    let mut mat = mat.clone();
    let n = mat.nrows();
    let mut det = 1i64;
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
            if mat[(row, i)] != 0 {
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
            det = (det * pivot).rem_euclid(p);

            let inv = modinv(pivot, p);

            // 2. Eliminate rows below
            for row in (i + 1)..n {
                let factor = mat[(row, i)] * inv % p;
                for col in i..n {
                    mat[(row, col)] = (mat[(row, col)] - factor * mat[(i, col)]).rem_euclid(p);
                }
            }
        } else {
            // Pivot = 0 → determinant is 0
            return 0;
        }
    }

    if sign == -1 {
        det = (p - det).rem_euclid(p);
    }

    det
}

pub fn analyze_min_generators(poly : &Vec<Matrix<i64>>, p:i64)
{
    let n = poly[0].nrows();
    let k = poly.len();
    println!("Analyzing min generators ... {} found of max degree {}", poly[0].ncols(), poly.len());
    // println!("Matrix ranks by degree...");
    // for i in 0..poly.len() {
    //     println!("{}: {}", i, modular_rank(&poly[i], p));
    // }
    // println!("Matrix determinants by degree...");
    // for i in 0..poly.len() {
    //     println!("{}: {}", i, modular_determinant(&poly[i], p));
    // }
    println!("Individual generator degrees...");
    for j in 0..poly[0].ncols() {
        let mut deg = 0;
        for i in 0..poly.len() {
            if poly[i].column(j).iter().any(|&x| x != 0) {
                deg = i;
            }
        }
        println!("Generator {} has degree: {}", j, deg);
    }
    // this assumes some luck...
    let matrank = n*(k-3) + modular_rank(&poly[k-1], p) + modular_rank(&poly[0], p);
    println!("Estimated matrix rank: {}", matrank);
}