
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UPoly {
    pub coeffs: Vec<u128>,
    pub p: u128, // modulus, part of the struct!
}


impl UPoly {
    pub fn new(mut coeffs: Vec<u128>, p: u128) -> Self {
        for c in &mut coeffs {
            *c = (*c).rem_euclid(p);
        }
        while coeffs.last() == Some(&0) && coeffs.len() > 1 {
            coeffs.pop();
        }
        UPoly { coeffs, p }
    }

    pub fn zero(p: u128) -> Self {
        UPoly { coeffs: vec![0], p }
    }

    pub fn one(p: u128) -> Self {
        UPoly { coeffs: vec![1], p }
    }

    pub fn from(c : u128, p: u128) -> Self {
        UPoly { coeffs: vec![c], p }
    }

    pub fn x(p: u128) -> Self {
        UPoly { coeffs: vec![0,1], p }
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

    pub fn leading_coeff(&self) -> u128 {
        *self.coeffs.last().unwrap_or(&0)
    }

    fn check_same_modulus(&self, other: &Self) {
        assert_eq!(self.p, other.p, "Polynomials must have the same modulus");
    }
}


impl UPoly {
    pub fn add(&self, other: &Self) -> Self {
        self.check_same_modulus(other);
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = vec![0; n];
        for i in 0..n {
            let a = if i < self.coeffs.len() { self.coeffs[i] } else { 0 };
            let b = if i < other.coeffs.len() { other.coeffs[i] } else { 0 };
            coeffs[i] = (a + b).rem_euclid(self.p);
        }
        UPoly::new(coeffs, self.p)
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.check_same_modulus(other);
        let n = self.coeffs.len().max(other.coeffs.len());
        let mut coeffs = vec![0; n];
        for i in 0..n {
            let a = if i < self.coeffs.len() { self.coeffs[i] } else { 0 };
            let b = if i < other.coeffs.len() { other.coeffs[i] } else { 0 };
            coeffs[i] = (a +self.p- b).rem_euclid(self.p);
        }
        UPoly::new(coeffs, self.p)
    }
}

impl UPoly {
    pub fn mul(&self, other: &Self) -> Self {
        self.check_same_modulus(other);
        let mut coeffs = vec![0; self.deg() + other.deg() + 1];
        for i in 0..self.coeffs.len() {
            for j in 0..other.coeffs.len() {
                coeffs[i + j] = (coeffs[i + j] + self.coeffs[i] * other.coeffs[j]).rem_euclid(self.p);
            }
        }
        UPoly::new(coeffs, self.p)
    }
}

