


use nalgebra::{DMatrix as Matrix, DVector, Dim};
use std::ops::{Add, Mul, Sub};

// type Matrix<T> = DMatrix<T>;

use std::cmp::{min, max};

pub fn matrix_berlekamp_massey(
    m_coeffs: &[Matrix<i64>], // M0, M1, ..., Mδ
    delta: usize,
    p: i64,
) -> Result<Vec<Matrix<i64>>, &'static str> {
    let n = m_coeffs[0].nrows();
    let mut f = Matrix::<i64>::zeros(n, 2 * n);
    for i in 0..n {
        f[(i, i)] = 1; // f ← [I 0]
    }

    let mut d = vec![0; n];
    d.extend(vec![1; n]); // d1..N ← 0; dN+1..2N ← 1

    let mut beta = 1;
    let mut sigma = 0;
    let mut mu = 0;
    let mut t: usize = 0;

    loop {
        if beta >= delta - sigma + mu + 1 {
            break;
        }

        // MBM3: increment t
        t += 1;
        if t >= m_coeffs.len() {
            return Err("Ran out of M(z) coefficients");
        }

        // MBM4: Ξ = coeff(t; M(z) * f(z))
        let mut xi = Matrix::<i64>::zeros(n, 2 * n);
        for k in 0..=t {
            if k < m_coeffs.len() {
                let m_k = &m_coeffs[k];
                let f_tk = f.clone().map(|x| x); // clone f(z)
                for j in 0..2 * n {
                    let fj = f_tk.column(j).into_owned();
                    let mut fj_tk = fj.clone();
                    //Matrix::<i64>::zeros(n, 1);
                    if t < k {
                        fj_tk.copy_from(&Matrix::<i64>::zeros(n, 1));
                    } 
                    let m_col = m_k * fj_tk;
                    for row in 0..n {
                        xi[(row, j)] = (xi[(row, j)] + m_col[(row, 0)]) % p;
                    }
                }
            }
        }

        // MBM5: Gaussian elimination
        let tau = auxiliary_gaussian_elimination(&xi, &mut d, t, n, p);

        // MBM6: dN+i += 1 for i in 1..=N
        for i in n..2 * n {
            d[i] += 1;
        }

        // MBM7: update β, μ, σ
        beta = *d[n..2 * n].iter().min().unwrap();
        mu = *d[0..n].iter().max().unwrap();
        sigma = d[0..n].iter().sum();

        // MBM8: fail if σ ≥ δ + 1
        if sigma >= delta + 1 {
            return Err("insufficient bound");
        }

        // MBM9: f(z) ← f(z) · τ · diag(IN, z·IN)
        let mut diag = Matrix::<i64>::identity(2 * n, 2 * n);
        for i in n..2 * n {
            diag[(i, i)] = 0;
        }
        let mut z_diag = Matrix::<i64>::identity(2 * n, 2 * n);
        for i in n..2 * n {
            z_diag[(i, i)] = 1; // represents multiplication by z
        }
        let new_f = f.clone() * tau * z_diag;
        f = new_f.map(|x| x % p);

        t += 1;
    }

    // MBM10–11: extract minimal matrix generator
    let mut generator = vec![];
    for j in 0..n {
        let deg = d[j];
        let mut poly = vec![Matrix::<i64>::zeros(n, n); deg + 1];
        for i in 0..=deg {
            for row in 0..n {
                let val = f[(row, j)];
                poly[deg - i][(row, j)] = val;
            }
        }
        generator.push(poly);
    }

    // Convert from Vec<Vec<Matrix>> to Vec<Matrix> of polynomial coefficients
    let max_deg = generator.iter().map(|poly| poly.len()).max().unwrap_or(0);
    let mut result = vec![Matrix::<i64>::zeros(n, n); max_deg];
    for j in 0..n {
        for (i, mat) in generator[j].iter().enumerate() {
            result[i] = (&result[i] + mat).map(|x| x % p);
        }
    }

    Ok(result)
}

/// Computes discrepancy Ξ = Coeff(t; M(z) * f(z))
fn compute_discrepancy(
    m_series: &[Matrix<i64>], // M(z) coefficients
    f: &[Matrix<i64>],        // vector of 2N columns, each column is a polynomial in matrix form
    t: usize,
) -> Matrix<i64>
{
    let n = m_series[0].nrows();
    let two_n = f.len();
    let mut result = Matrix::<i64>::zeros(n, two_n);

    for k in 0..=t {
        if k >= m_series.len() {
            break;
        }
        let m_k = &m_series[k]; // M_k
        let f_tk = get_poly_coeff(f, t - k); // coefficient of z^{t-k} from f(z)
        let product = m_k * f_tk;
        result += product;
    }

    result
}

/// Builds a Matrix<T> of size (N x 2N) with the coefficient of z^k from each column polynomial in f
fn get_poly_coeff(f: &[Matrix<i64>], k: usize) -> Matrix<i64>
{
    let n = f[0].nrows();
    let two_n = f.len();
    let mut coeff_mat = Matrix::<i64>::zeros(n, two_n);

    for j in 0..two_n {
        let f_j = &f[j];
        if k < f_j.ncols() {
            // f_j is the polynomial column vector, represented as matrix
            // and the k-th coefficient is the k-th column
            let coeff_col = f_j.column(k);
            for i in 0..n {
                coeff_mat[(i, j)] = coeff_col[i].clone();
            }
        }
    }

    coeff_mat
}


fn auxiliary_gaussian_elimination(
    xi: &Matrix<i64>, // Ξ ∈ K^{N×2N}
    d: &mut Vec<usize>, // degrees
    t: usize,
    n: usize,
    p: i64,
) -> Matrix<i64> {
    let two_n = 2 * n;
    let mut tau = Matrix::<i64>::identity(two_n, two_n);
    let mut xi = xi.clone(); // Work on a local copy

    let mut a_set: Vec<usize> = (0..n).collect();

    for i in 0..n {
        let mut beta_i: Vec<usize> = a_set
            .iter()
            .filter(|&&j| xi[(i, j)] % p != 0)
            .cloned()
            .collect();
        beta_i.push(n + i);

        if beta_i.is_empty() {
            continue;
        }

        let &l = beta_i.iter().min_by_key(|&&j| d[j]).unwrap();
        let l = if l < n && beta_i.contains(&(n + i)) { n + i } else { l };
        beta_i.retain(|&x| x != l);

        for &j in &beta_i {
            let pivot = xi[(i, l)].rem_euclid(p);
            if pivot == 0 {
                continue;
            }
            let factor = (xi[(i, j)] * modinv(pivot, p)).rem_euclid(p);

            if l == n + i {
                // GE9-10
                for row in 0..n {
                    xi[(row, j)] = (xi[(row, j)] - factor * xi[(row, l)]).rem_euclid(p);
                }
                for row in 0..two_n {
                    tau[(row, j)] = (tau[(row, j)] - factor * tau[(row, l)]).rem_euclid(p);
                }
            } else if j < n + i {
                // GE13-14
                for row in 0..n {
                    xi[(row, j)] = (xi[(row, j)] - factor * xi[(row, l)]).rem_euclid(p);
                }
                for row in 0..two_n {
                    tau[(row, j)] = (tau[(row, j)] - factor * tau[(row, l)]).rem_euclid(p);
                }
            } else if j == n + i {
                if xi[(i, n + i)] % p != 0 {
                    // GE17-20
                    let inv = modinv(xi[(i, n + i)], p);
                    let factor = (-xi[(i, l)] * inv).rem_euclid(p);
                    for row in 0..n {
                        xi[(row, n + i)] = (factor * xi[(row, n + i)] + xi[(row, l)]).rem_euclid(p);
                    }
                    for row in 0..two_n {
                        tau[(row, n + i)] = (factor * tau[(row, n + i)] + tau[(row, l)]).rem_euclid(p);
                    }
                    for row in 0..n {
                        xi.swap((row, l), (row, n + i));
                    }
                    tau.swap_columns(l, n + i);
                    d.swap(l, n + i);
                } else {
                    // GE22-24
                    for row in 0..two_n {
                        tau[(row, n + i)] = (tau[(row, n + i)] + tau[(row, l)]).rem_euclid(p);
                    }
                    d.swap(l, n + i);
                }
            }
        }

        a_set.retain(|&x| x != l);
    }

    // Final update of degrees
    for col in 0..two_n {
        if xi.column(col).iter().any(|&x| x % p != 0) {
            d[col] = t + 1;
        }
    }

    tau
}



fn modinv(a: i64, p: i64) -> i64 {
    let (mut t, mut new_t) = (0, 1);
    let (mut r, mut new_r) = (p, a.rem_euclid(p));

    while new_r != 0 {
        let quotient = r / new_r;
        t = t - quotient * new_t;
        std::mem::swap(&mut t, &mut new_t);
        r = r - quotient * new_r;
        std::mem::swap(&mut r, &mut new_r);
    }

    if r > 1 {
        panic!("modinv does not exist");
    }
    if t < 0 {
        t += p;
    }
    t
}


#[test]
fn test_matrix_berlekamp_massey_simple() {
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

    let result = matrix_berlekamp_massey(&m_coeffs, delta, p)
        .expect("Should produce a minimal generator");

    println!("Matrix generator coefficients:");
    for (i, mat) in result.iter().enumerate() {
        println!("z^{}:\n{}", i, mat);
    }
}