use nalgebra::DMatrix;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poly {
    pub coeffs: Vec<i64>,
    pub p: i64, // modulus, part of the struct!
}

impl Poly {
    pub fn new(mut coeffs: Vec<i64>, p: i64) -> Self {
        for c in &mut coeffs {
            *c = (*c).rem_euclid(p);
        }
        while coeffs.last() == Some(&0) && coeffs.len() > 1 {
            coeffs.pop();
        }
        Poly { coeffs, p }
    }

    pub fn zero(p: i64) -> Self {
        Poly { coeffs: vec![0], p }
    }

    pub fn one(p: i64) -> Self {
        Poly { coeffs: vec![1], p }
    }

    pub fn deg(&self) -> usize {
        if self.is_zero() {
            0
        } else {
            self.coeffs.len() - 1
        }
    }

    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0)
    }

    pub fn leading_coeff(&self) -> i64 {
        *self.coeffs.last().unwrap_or(&0)
    }

    fn check_same_modulus(&self, other: &Self) {
        assert_eq!(self.p, other.p, "Polynomials must have the same modulus");
    }
}


impl Poly {
    pub fn add(&self, other: &Self) -> Self {
        self.check_same_modulus(other);
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = vec![0; n];
        for i in 0..n {
            let a = if i < self.coeffs.len() { self.coeffs[i] } else { 0 };
            let b = if i < other.coeffs.len() { other.coeffs[i] } else { 0 };
            coeffs[i] = (a + b).rem_euclid(self.p);
        }
        Poly::new(coeffs, self.p)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.check_same_modulus(other);
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = vec![0; n];
        for i in 0..n {
            let a = if i < self.coeffs.len() { self.coeffs[i] } else { 0 };
            let b = if i < other.coeffs.len() { other.coeffs[i] } else { 0 };
            coeffs[i] = (a - b).rem_euclid(self.p);
        }
        Poly::new(coeffs, self.p)
    }
}

impl Poly {
    pub fn mul(&self, other: &Self) -> Self {
        self.check_same_modulus(other);
        let mut coeffs = vec![0; self.deg() + other.deg() + 1];
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() {
                coeffs[i + j] = (coeffs[i + j] + self.coeffs[i] * other.coeffs[j]).rem_euclid(self.p);
            }
        }
        Poly::new(coeffs, self.p)
    }
}


impl Poly {
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        self.check_same_modulus(divisor);
        assert!(!divisor.is_zero(), "Division by zero polynomial");

        let mut quotient = vec![0; self.coeffs.len()];
        let mut remainder = self.clone();
        let inv = modinv(divisor.leading_coeff(), self.p);

        while !remainder.is_zero() && remainder.deg() >= divisor.deg() {
            let deg_diff = remainder.deg() - divisor.deg();
            let lead_coeff = remainder.leading_coeff() * inv % self.p;
            let mut t = vec![0; deg_diff + 1];
            t[deg_diff] = lead_coeff;
            let t_poly = Poly::new(t, self.p);
            quotient[deg_diff] = lead_coeff;

            let subtrahend = divisor.mul(&t_poly);
            remainder = remainder.sub(&subtrahend);
        }

        (Poly::new(quotient, self.p), remainder)
    }

    pub fn div_exact(&self, divisor: &Self) -> Self {
        let (q, r) = self.div_rem(divisor);
        assert!(r.is_zero(), "div_exact: remainder not zero");
        q
    }
}

impl Poly {
    pub fn gcd(mut a: Poly, mut b: Poly) -> Poly {
        a.check_same_modulus(&b);
        let p = a.p;
        while !b.is_zero() {
            let (_, r) = a.div_rem(&b);
            a = b;
            b = r;
        }
        // Make monic (optional)
        let inv = modinv(a.leading_coeff(), p);
        a.scale(inv)
    }

    pub fn lcm(a: Poly, b: Poly) -> Poly {
        a.check_same_modulus(&b);
        if a.is_zero() {
            return b;
        }
        if b.is_zero() {
            return a;
        }
        let g = Poly::gcd(a.clone(), b.clone());
        a.mul(&b).div_exact(&g)
    }

    pub fn scale(&self, scalar: i64) -> Self {
        let coeffs = self.coeffs.iter().map(|c| (c * scalar).rem_euclid(self.p)).collect();
        Poly::new(coeffs, self.p)
    }
}

fn modinv(x: i64, p: i64) -> i64 {
    let (mut a, mut b, mut u) = (x.rem_euclid(p), p, 1);
    while a != 1 && a != 0 {
        u = u * (p / a) % p;
        b %= a;
        std::mem::swap(&mut a, &mut b);
    }
    if a == 0 {
        panic!("No modular inverse exists");
    }
    (u + p) % p
}


pub fn top_invariant_factor(mut mat: DMatrix<Poly>) -> Poly {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let p = mat[(0, 0)].p; // Assume all entries share same p

    let mut rank = 0;

    for col in 0..ncols {
        // 1. Find pivot row
        let mut pivot_row = None;
        for row in rank..nrows {
            if !mat[(row, col)].is_zero() {
                pivot_row = Some(row);
                break;
            }
        }

        if let Some(pivot_row_idx) = pivot_row {
            // 2. Swap rows
            mat.swap_rows(rank, pivot_row_idx);

            // 3. Eliminate below and above
            for row in 0..nrows {
                if row != rank && !mat[(row, col)].is_zero() {
                    let a = mat[(row, col)].clone();
                    let b = mat[(rank, col)].clone();
                    let lcm = Poly::lcm(a.clone(), b.clone());
                    let m1 = lcm.div_exact(&a);
                    let m2 = lcm.div_exact(&b);
                    for j in 0..ncols {
                        mat[(row, j)] = mat[(row, j)].mul(&m1).sub(&mat[(rank, j)].mul(&m2));
                    }
                }
            }

            // 4. Eliminate right of pivot
            for j in 0..ncols {
                if j != col && !mat[(rank, j)].is_zero() {
                    let a = mat[(rank, j)].clone();
                    let b = mat[(rank, col)].clone();
                    let lcm = Poly::lcm(a.clone(), b.clone());
                    let m1 = lcm.div_exact(&a);
                    let m2 = lcm.div_exact(&b);
                    for i in 0..nrows {
                        mat[(i, j)] = mat[(i, j)].mul(&m1).sub(&mat[(i, col)].mul(&m2));
                    }
                }
            }

            rank += 1;
            if rank == nrows {
                break;
            }
        }
    }

    // Collect diagonal entries
    let mut diagonals = Vec::new();
    for i in 0..nrows.min(ncols) {
        if !mat[(i, i)].is_zero() {
            diagonals.push(mat[(i, i)].clone());
        }
    }

    if diagonals.is_empty() {
        Poly::zero(p)
    } else {
        let mut lcm = diagonals[0].clone();
        for diag in diagonals.iter().skip(1) {
            lcm = Poly::lcm(lcm, diag.clone());
        }
        lcm
    }
}

pub fn top_invariant_factor_fast(mut mat: DMatrix<Poly>) -> Poly {
    let nrows = mat.nrows();
    let ncols = mat.ncols();
    let p = mat[(0, 0)].p; // Assume all entries share same p

    let mut rank = 0;

    for col in 0..ncols {
        // 1. Find pivot row
        let mut pivot_row = None;
        for row in rank..nrows {
            if !mat[(row, col)].is_zero() {
                pivot_row = Some(row);
                break;
            }
        }

        if let Some(pivot_row_idx) = pivot_row {
            // 2. Swap pivot row to current rank position
            mat.swap_rows(rank, pivot_row_idx);

            // 3. Normalize pivot to be monic
            let pivot_lead = mat[(rank, col)].leading_coeff();
            let inv = modinv(pivot_lead, p);
            for j in col..ncols {
                mat[(rank, j)] = mat[(rank, j)].scale(inv);
            }

            // 4. Eliminate entries *below* pivot
            for row in (rank + 1)..nrows {
                if !mat[(row, col)].is_zero() {
                    let factor = mat[(row, col)].clone();
                    for j in col..ncols {
                        mat[(row, j)] = mat[(row, j)].sub(&mat[(rank, j)].mul(&factor));
                    }
                }
            }

            rank += 1;
            if rank == nrows {
                break;
            }
        }
    }

    // 5. LCM of diagonal elements
    let mut diagonals = Vec::new();
    for i in 0..nrows.min(ncols) {
        if !mat[(i, i)].is_zero() {
            diagonals.push(mat[(i, i)].clone());
        }
    }

    if diagonals.is_empty() {
        Poly::zero(p)
    } else {
        let mut lcm = diagonals[0].clone();
        for diag in diagonals.iter().skip(1) {
            lcm = Poly::lcm(lcm, diag.clone());
        }
        lcm
    }
}


pub fn vec_matrix_to_poly_matrix3(vecmats: &[DMatrix<i64>], p:i64) -> DMatrix<Poly> {
    let rows = vecmats[0].nrows();
    let cols = vecmats[0].ncols();
    let degree = vecmats.len();

    // println!("vec_matrix_to_poly_matrix3 {}", vecmats.len());

    let mut result = DMatrix::from_element(rows, cols, Poly::zero(p));

    for (d, mat) in vecmats.iter().enumerate() {
        for i in 0..rows {
            for j in 0..cols {
                // println!("vec_matrix_to_poly_matrix3 {} {} {}", d,i,j);
                let coeff = mat[(i, j)];
                if coeff != 0 {
                    // Insert the coeff at degree d
                    let mut poly = result[(i, j)].clone();
                    if poly.coeffs.len() <= d {
                        poly.coeffs.resize(d + 1, 0);
                    }
                    poly.coeffs[d] = (poly.coeffs[d] + coeff).rem_euclid(p);
                    result[(i, j)] = Poly::new(poly.coeffs.clone(), poly.p);
                }
            }
        }
    }

    result
}


pub fn poly_matrix_determinant(mut mat: DMatrix<Poly>) -> Poly {
    assert!(mat.is_square(), "Matrix must be square to compute determinant");

    let n = mat.nrows();
    let p = mat[(0, 0)].p;
    let mut det = Poly::one(p);
    let mut sign = 1;

    for i in 0..n {
        // Find pivot
        let mut pivot_row = None;
        for row in i..n {
            if !mat[(row, i)].is_zero() {
                pivot_row = Some(row);
                break;
            }
        }

        if let Some(pivot_idx) = pivot_row {
            if pivot_idx != i {
                mat.swap_rows(i, pivot_idx);
                sign = -sign; // Swapping rows flips determinant sign
            }

            let pivot = mat[(i, i)].clone();
            det = det.mul(&pivot);

            // Eliminate below
            for row in (i + 1)..n {
                if !mat[(row, i)].is_zero() {
                    let factor = mat[(row, i)].clone().div_exact(&pivot);
                    for col in i..n {
                        mat[(row, col)] = mat[(row, col)].sub(&mat[(i, col)].mul(&factor));
                    }
                }
            }
        } else {
            // Zero pivot means determinant is zero
            return Poly::zero(p);
        }
    }

    if sign == -1 {
        det = det.scale(p - 1); // Negate
    }

    det
}