use nalgebra::DMatrix;
use std::fmt;

const P: i64 = 5; // change as needed

#[derive(Clone, PartialEq, Eq)]
pub struct Poly {
    coeffs: Vec<i64>, // Coefficients mod P
}

impl Poly {
    pub fn new(mut coeffs: Vec<i64>) -> Self {
        for c in &mut coeffs {
            *c = (*c % P + P) % P;
        }
        while coeffs.last() == Some(&0) {
            coeffs.pop();
        }
        Self { coeffs }
    }

    fn zero() -> Self { Self { coeffs: vec![] } }
    fn one() -> Self { Self::new(vec![1]) }
    fn is_zero(&self) -> bool { self.coeffs.is_empty() }
    pub fn deg(&self) -> usize { self.coeffs.len().saturating_sub(1) }
    fn lc(&self) -> i64 { *self.coeffs.last().unwrap_or(&0) }

    fn scale(&self, c: i64) -> Self {
        let c = (c % P + P) % P;
        if c == 0 { return Poly::zero(); }
        Poly::new(self.coeffs.iter().map(|x| x * c).collect())
    }

    fn add(&self, other: &Self) -> Self {
        let mut res = vec![0; self.coeffs.len().max(other.coeffs.len())];
        for i in 0..res.len() {
            let a = if i < self.coeffs.len() { self.coeffs[i] } else { 0 };
            let b = if i < other.coeffs.len() { other.coeffs[i] } else { 0 };
            res[i] = (a + b).rem_euclid(P);
        }
        Poly::new(res)
    }

    fn sub(&self, other: &Self) -> Self {
        let mut res = vec![0; self.coeffs.len().max(other.coeffs.len())];
        for i in 0..res.len() {
            let a = if i < self.coeffs.len() { self.coeffs[i] } else { 0 };
            let b = if i < other.coeffs.len() { other.coeffs[i] } else { 0 };
            res[i] = (a - b).rem_euclid(P);
        }
        Poly::new(res)
    }

    fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Poly::zero();
        }
        let mut res = vec![0; self.deg() + other.deg() + 1];
        for (i, &a) in self.coeffs.iter().enumerate() {
            for (j, &b) in other.coeffs.iter().enumerate() {
                res[i + j] = (res[i + j] + a * b).rem_euclid(P);
            }
        }
        Poly::new(res)
    }

    fn div_rem(&self, other: &Self) -> (Self, Self) {
        let mut r = self.clone();
        let mut q = Poly::zero();
        if other.is_zero() {
            panic!("Divide by zero polynomial");
        }
        let inv_lc = mod_inv(other.lc(), P);
        while !r.is_zero() && r.deg() >= other.deg() {
            let d = r.deg() - other.deg();
            let coef = r.lc() * inv_lc % P;
            let mut t = vec![0; d + 1];
            t[d] = coef;
            let t_poly = Poly::new(t);
            q = q.add(&t_poly);
            r = r.sub(&t_poly.mul(other));
        }
        (q, r)
    }

    fn gcd(a: Self, b: Self) -> Self {
        let mut a = a;
        let mut b = b;
        while !b.is_zero() {
            let (_, r) = a.div_rem(&b);
            a = b;
            b = r;
        }
        // Make monic
        a.scale(mod_inv(a.lc(), P))
    }
}

fn mod_inv(a: i64, p: i64) -> i64 {
    let (mut t, mut r) = (0, p);
    let (mut new_t, mut new_r) = (1, a);
    while new_r != 0 {
        let q = r / new_r;
        t = t - q * new_t;
        r = r - q * new_r;
        std::mem::swap(&mut t, &mut new_t);
        std::mem::swap(&mut r, &mut new_r);
    }
    if r > 1 { panic!("Not invertible") }
    (t + p) % p
}

impl fmt::Debug for Poly {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.is_zero() {
            return write!(f, "0");
        }
        let mut terms = vec![];
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c == 0 { continue; }
            if i == 0 {
                terms.push(format!("{}", c));
            } else if i == 1 {
                terms.push(format!("{}z", c));
            } else {
                terms.push(format!("{}z^{}", c, i));
            }
        }
        write!(f, "{}", terms.join(" + "))
    }
}


pub fn top_invariant_factor(mut mat: DMatrix<Poly>) -> Poly {
    let (m, n) = (mat.nrows(), mat.ncols());
    let mut r = 0;

    while r < m && r < n {
        // Step 1: Find pivot with smallest degree
        let mut pivot_pos = None;
        let mut min_deg = usize::MAX;
        for i in r..m {
            for j in r..n {
                let deg = mat[(i, j)].deg();
                if !mat[(i, j)].is_zero() && deg < min_deg {
                    min_deg = deg;
                    pivot_pos = Some((i, j));
                }
            }
        }

        let (i_pivot, j_pivot) = match pivot_pos {
            Some(pos) => pos,
            None => break, // whole submatrix is zero
        };

        // Step 2: Swap rows and cols to move pivot to (r, r)
        if i_pivot != r {
            mat.swap_rows(i_pivot, r);
        }
        if j_pivot != r {
            mat.swap_columns(j_pivot, r);
        }

        // Step 3: Clear below and right using row and column operations
        for i in (r + 1)..m {
            let (q, _) = mat[(i, r)].div_rem(&mat[(r, r)]);
            let sub = q.mul(&mat[(r, r)]);
            mat[(i, r)] = mat[(i, r)].sub(&sub);
        }
        for j in (r + 1)..n {
            let (q, _) = mat[(r, j)].div_rem(&mat[(r, r)]);
            let sub = q.mul(&mat[(r, r)]);
            mat[(r, j)] = mat[(r, j)].sub(&sub);
        }

        // Step 4: Clean up rest of submatrix (r+1.., r+1..) using GCDs
        for i in (r + 1)..m {
            for j in (r + 1)..n {
                let a = mat[(r, r)].clone();
                let b = mat[(i, j)].clone();
                let g = Poly::gcd(a, b);
                mat[(i, j)] = g;
            }
        }

        r += 1;
    }

    // Top invariant factor = last nonzero diagonal entry
    for i in (0..m.min(n)).rev() {
        if !mat[(i, i)].is_zero() {
            return mat[(i, i)].clone();
        }
    }

    Poly::zero()
}

pub fn vec_matrix_to_poly_matrix(vecmats: &[DMatrix<i64>]) -> DMatrix<Poly> {
    let rows = vecmats[0].nrows();
    let cols = vecmats[0].ncols();
    let degree = vecmats.len();

    let mut result = DMatrix::from_element(rows, cols, Poly::zero());

    for (d, mat) in vecmats.iter().enumerate() {
        for i in 0..rows {
            for j in 0..cols {
                let coeff = mat[(i, j)];
                if coeff != 0 {
                    // Insert the coeff at degree d
                    let mut poly = result[(i, j)].clone();
                    if poly.coeffs.len() <= d {
                        poly.coeffs.resize(d + 1, 0);
                    }
                    poly.coeffs[d] = (poly.coeffs[d] + coeff).rem_euclid(P);
                    result[(i, j)] = Poly::new(poly.coeffs.clone());
                }
            }
        }
    }

    result
}


