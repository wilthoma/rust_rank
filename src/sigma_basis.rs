use crate::{blockbmtest::analyze_min_generators, matrices::GoodInteger, modular_linalg::{lsp_decomposition, matrix_inverse, modular_determinant, modular_rank}};
use crate::{ntt::{mod_pow, modinv, NTTInteger}, poly_mat_mul::{poly_mat_mul_fft_red, poly_mat_mul_red_adaptive}};
use nalgebra::DMatrix;
use std::{fmt::Debug, ops::{Add, Mul}, vec, io::Write};




pub fn PM_Basis<T: GoodInteger+Into<u128> >(seq : &Vec<Vec<T>>, d:usize, seqprime: T, largeprime : u128, root : u128) -> (Vec<Vec<Vec<u128>>>, Vec<i128>) {
    // expand sequence to square matrix
    let nn = seq.len();
    let mut n=0;
    let nlen = seq[0].len();
    for i in 0..nn {
        if i*(i+1)/2 == nn {
            n = i;
        }
    }

    assert!(n*(n+1)/2 == nn, "Sequence length does not match square matrix size");

    // enlarge d to a power of 2
    let mut dd = 1;
    while dd < d {
        dd *= 2;
    }
    let d = dd;

    assert!(d<nlen, "Sequence too short: d={}, n={}, nlen={}",d, n, nlen);
    // assert!(d*n<nlen, "Sequence too short to reach order nd={}, d={}, n={}, nlen={}", n*d,d, n, nlen);

    // prepare input data. Also add a unit matrix of size nxn below the matrix
    let mut G = vec![vec![vec![0; nlen]; n]; 2*n];
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
    for i in 0..n{
        G[n+i][i][0] = 1;
    }

    let delta = vec![0; 2*n];

    let (M, mu) = _PM_Basis(&G, d, &delta, seqprime.into(), largeprime, root, &mut ProgressData::new(d));
    
    (M, mu)
}


fn mat_coeff(a : &Vec<Vec<Vec<u128>>>, k:usize) -> DMatrix<u128> {
    let n = a.len();
    let m = a[0].len();
    DMatrix::from_fn(n, m, |i,j| {
        if a[i][j].len() > k {
            a[i][j][k]
        } else {
            0
        }
    })
}



pub fn analyze_pm_basis(basis : &Vec<Vec<Vec<u128>>>, delta : &Vec<i128>, p:u128) {
    // analyze the basis
    // print matrix ranks and determinants
    // print individual generator degrees
    // print delta values

    // if reverse_polys {
    //     for i in 0..basis.len() {
    //         for j in 0..basis[0].len() {
    //             basis[i][j].reverse();
    //         }
    //     }
    // }

    let n = basis.len();
    assert_eq!(n, basis[0].len(), "Matrix of basis vectors must be square");
    let nlen = basis[0][0].len();

    println!("Analyzing min generators ... {} found of max degree {}", n, nlen);
    // println!("Matrix ranks by degree...");
    // for i in 0..nlen {
    //     println!("{}: {}", i, modular_rank(&mat_coeff(&basis, i), p));
    // }
    // println!("Matrix determinants by degree...");
    // for i in 0..nlen {
    //     println!("{}: {}", i, modular_determinant(&mat_coeff(&basis, i), p));
    // }
    println!("Individual generator degrees...");
    for j in 0..n {
        // the j-th generator is the j-th row
        let g = &basis[j];
        let mut deg = 0;
        let mut val = 99999999;
        for i in 0..nlen {
            if g.iter().any(|x| x[i] != 0) {
                deg = i;
                if val == 99999999 {
                    val = i;
                }
            }
        }
        println!("Generator {} has degree: {} and valuation {} and delta {}", j, deg, val, delta[j]);
    }

    // extract reversed basis for original sequence, composed of the upper left block (I hope)
    let m = n/2;
    let mut basisrev = vec![vec![vec![0; nlen]; m]; m];
    let mut maxdel = 0;
    for i in 0..m {
        let del = (-delta[i]) as usize;
        if del > maxdel {
            maxdel = del;
        }
        for j in 0..m {
            for k in 0..=del {
                basisrev[i][j][k] = basis[i][j][del-k] as i64;
            }
        }
    }

    let mut basisrev2 = vec![DMatrix::zeros(m,m); maxdel+1];
    let mut maxdeg = 0;
    for i in 0..m {
        for j in 0..m {
            for k in 0..=maxdel {
                basisrev2[k][(i,j)] = basisrev[i][j][k];
                if basisrev[i][j][k] != 0 && k> maxdeg {
                    maxdeg = k;
                }
            }
        }
    }
    basisrev2.resize(maxdeg+1, DMatrix::zeros(m,m));

    analyze_min_generators(&basisrev2, p as i64);


    // this assumes some luck...
    //let matrank = n*(nlen-3) + modular_rank(&mat_coeff(&basis, nlen-1), p) + modular_rank(&mat_coeff(&basis, 0), p);
    //println!("Estimated matrix rank: {}", matrank);

}


struct ProgressData {
    total : usize,
    current : usize,
    start_time : std::time::Instant,
}
impl ProgressData {
    fn new(total : usize) -> Self {
        ProgressData {
            total,
            current : 0,
            start_time : std::time::Instant::now(),
        }
    }
    #[inline(always)]
    fn progress_tick(&mut self) {
        self.current += 1;
        if self.current % 100 == 0 {
            let elapsed_time = self.start_time.elapsed();
            let percent = (self.current as f64 / self.total as f64) * 100.0;
            print!("\rSigma basis progress: {:.2}% ({} of {}), Time elapsed: {:?}    ", percent, self.current, self.total, elapsed_time);
            std::io::stdout().flush().unwrap();
        }
    }
}

fn _PM_Basis(G : &Vec<Vec<Vec<u128>>>, d: usize, delta : &Vec<i128>, seqprime : u128, largeprime : u128, root : u128, progress : &mut ProgressData) -> (Vec<Vec<Vec<u128>>>, Vec<i128>) {
    // must have d= 0 or d= power of 2
    let n = G.len();

    if d == 0 {
        (unit_mat(n), delta.clone())
    } else if d==1 {
        progress.progress_tick();
        M_Basis(G, delta, seqprime)
    } else {
        let (MM, mumu) = _PM_Basis(G, d/2, delta, seqprime, largeprime, root, progress);

        let start_time = std::time::Instant::now();
        // println!("startntt...");
        // let mut GG = poly_mat_mul_fft_red(&MM, &G, largeprime, root, seqprime, d*n+1);
        let mut GG = poly_mat_mul_red_adaptive(&MM, &G, largeprime, root, seqprime, d+1);
        // let elapsed_time = start_time.elapsed();
        // println!("Time taken for poly_mat_mul_fft_red: {:?}", elapsed_time);

        shift_trunc_in(&mut GG, d/2);
        // println!("startntt2...");
        let (MMM, mumumu) = _PM_Basis(&GG, d/2, &mumu, seqprime, largeprime, root, progress);
        // let start_time = std::time::Instant::now();
        let ret = (poly_mat_mul_red_adaptive(&MMM ,&MM, largeprime, root, seqprime, d+1), mumumu);
        // let ret = (poly_mat_mul_fft_red(&MMM ,&MM, largeprime, root, seqprime, d*n+1), mumumu);
        // let elapsed_time = start_time.elapsed();
        // println!("Time taken for poly_mat_mul_fft_red 2: {:?}", elapsed_time);
        ret
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




pub fn M_Basis(G : &Vec<Vec<Vec<u128>>>, delta : &Vec<i128>, prime : u128) -> (Vec<Vec<Vec<u128>>>, Vec<i128>) {
    let m = G.len();
    assert!(m>0, "Must have #rows > 0");
    let n = G[0].len();
    assert!(m>=n, "Must have #rows >= #cols");
    let mut delta = delta.clone(); 


    // sort delta in descending order and remember the permutation
    let mut perm = (0..m).collect::<Vec<_>>();
    perm.sort_by(|&a, &b| delta[a].cmp(&delta[b]).reverse());
    delta.sort_by(|a, b| b.cmp(a));


    let mut pi_m = DMatrix::zeros(m, m);
    for i in 0..m {
        pi_m[(i, perm[i])] = 1;
    }
    // let mut Gk_inv = pi_m.clone();
    // Gk_inv = Gk_inv.try_inverse().unwrap();

    // delta.sort_by(|a, b| b.cmp(a));

    // let Delta0 = Gk_inv * M * (G[k-1].clone());
    let mut Delta = DMatrix::from_fn(m,n,|i,j| G[i][j][0]);
    Delta = &pi_m * &Delta;
    let mut Delta_aug = DMatrix::zeros(m,m);
    for i in 0..m {
        for j in 0..n {
            Delta_aug[(i,j)] = Delta[(i,j)];
        }
    }
    
    let (L,S,P) = lsp_decomposition(&Delta_aug, prime);
    let Linv = matrix_inverse(&L, prime);

    // D = D2 + x Dx
    let mut D1 = DMatrix::zeros(m, m); 
    let mut Dx = DMatrix::zeros(m, m); 
    for i in 0..m {
        if S.row(i).iter().all(|&x| x == 0) {
            D1[(i,i)]=1;
        } else {
            Dx[(i,i)]=1;
        }
    }
    
    let M1 = &D1 * &Linv * &pi_m;
    let Mx = &Dx * &Linv * &pi_m;

    // fill output again in vects
    let mut ret = vec![vec![vec![0; 2]; m]; m];
    for i in 0..m {
        for j in 0..m {
            ret[i][j][0] = M1[(i,j)];
            ret[i][j][1] = Mx[(i,j)];
        }
    }


    for i in 0..m {
        // delta[i] += Dx[(i,i)] as i128;
        delta[i] -= Dx[(i,i)] as i128;
    }


    (ret, delta)

}


// fn modinv2(a: u128, p: u128) -> u128 {
//     assert!(a != 0, "attempted to invert zero");
//     let mut t = 0i128;
//     let mut new_t = 1i128;
//     let mut r = p as i128;
//     let mut new_r = a as i128;

//     while new_r != 0 {
//         let quotient = r / new_r;
//         t = t - quotient * new_t;
//         std::mem::swap(&mut t, &mut new_t);
//         r = r - quotient * new_r;
//         std::mem::swap(&mut r, &mut new_r);
//     }

//     if r > 1 {
//         panic!("modular inverse does not exist");
//     }
//     if t < 0 {
//         t += p as i128;
//     }

//     t as u128
// }


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




