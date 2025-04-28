use std::ops::{Add, Sub, Mul};
use num_traits::Zero;

use crate::ntt::{ntt, mod_pow};
use crate::upoly::UPoly;

// Define the FTPoly struct representing a polynomial with u128 coefficients in the Fourier domain
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FTPoly {
    pub coeffs: Vec<u128>,  // Coefficients in the Fourier domain
    pub p : u128,       // Modulus
    pub root : u128, // Primitive root of unity
}

impl FTPoly {
    pub fn zero(p : u128, root: u128) -> Self {
        FTPoly { coeffs: vec![0], p : p, root : root }
    }

    pub fn one(p : u128, root: u128) -> Self {
        FTPoly::from_coeffs(vec![1], p, root)
    }
}

impl FTPoly {
    // Create a polynomial from coefficients (after NTT)
    pub fn from_coeffs(coeffs: Vec<u128>, p : u128, root : u128) -> Self {
        let mut coeffs = coeffs;
        let mut n = coeffs.len();
        
        // Enlarge the vector to the next power of 2 if necessary
        if n & (n - 1) != 0 {
            let mut new_n = 1;
            while new_n < n {
                new_n <<= 1;
            }
            coeffs.resize(new_n, 0);
            n = new_n;
        }

        // Apply NTT
        ntt(&mut coeffs, false, p, root);

        FTPoly { coeffs, p, root }
    }

    // Convert back to polynomial in original domain (using inverse NTT)
    pub fn to_poly(&self) -> Vec<u128> {
        let mut coeffs = self.coeffs.clone();
        ntt(&mut coeffs, true, self.p, self.root);  // Apply inverse NTT
        coeffs
    }
    pub fn to_upoly(&self) -> UPoly {
        UPoly::new(self.to_poly(), self.p)
    }
    
}

impl Add for FTPoly {
    type Output = FTPoly;

    fn add(self, other: FTPoly) -> FTPoly {
        let P = self.p;
        let mut coeffs = self.coeffs.clone();
        let mut n = coeffs.len();

        // Enlarge the vector if necessary
        if n != other.coeffs.len() {
            if n < other.coeffs.len() {
                coeffs.resize(other.coeffs.len(), 0);
                n = other.coeffs.len();
            } else {
                let mut other_coeffs = other.coeffs.clone();
                other_coeffs.resize(n, 0);
                return FTPoly {
                    coeffs: coeffs.iter().zip(other_coeffs.iter()).map(|(a, b)| (a + b) % P).collect(),
                    p : self.p, 
                    root : self.root
                };
            }
        }

        for (a, b) in coeffs.iter_mut().zip(other.coeffs.iter()) {
            *a = (*a + *b) % P;
        }

        FTPoly { coeffs, p : self.p, root : self.root }
    }
}

impl Sub for FTPoly {
    type Output = FTPoly;

    fn sub(self, other: FTPoly) -> FTPoly {
        let mut coeffs = self.coeffs.clone();
        let mut n = coeffs.len();
        let P = self.p;

        // Enlarge the vector if necessary
        if n != other.coeffs.len() {
            if n < other.coeffs.len() {
                coeffs.resize(other.coeffs.len(), 0);
                n = other.coeffs.len();
            } else {
                let mut other_coeffs = other.coeffs.clone();
                other_coeffs.resize(n, 0);
                return FTPoly {
                    coeffs: coeffs.iter().zip(other_coeffs.iter()).map(|(a, b)| (a + P - b) % P).collect(),
                    p : self.p,
                    root : self.root
                };
            }
        }

        for (a, b) in coeffs.iter_mut().zip(other.coeffs.iter()) {
            *a = (*a + P - *b) % P;
        }

        FTPoly { coeffs : coeffs , p : self.p, root : self.root }
    }
}

impl Mul for FTPoly {
    type Output = FTPoly;

    fn mul(self, other: FTPoly) -> FTPoly {
        let mut coeffs = self.coeffs.clone();
        let mut n = coeffs.len();
        let P = self.p;

        // Enlarge the vector if necessary
        if n != other.coeffs.len() {
            if n < other.coeffs.len() {
                coeffs.resize(other.coeffs.len(), 0);
                n = other.coeffs.len();
            } else {
                let mut other_coeffs = other.coeffs.clone();
                other_coeffs.resize(n, 0);
                return FTPoly {
                    coeffs: coeffs.iter().zip(other_coeffs.iter()).map(|(a, b)| (a * b) % P).collect(),
                    p : self.p, root : self.root
                };
            }
        }

        // Perform pointwise multiplication in the Fourier domain
        for (a, b) in coeffs.iter_mut().zip(other.coeffs.iter()) {
            *a = (*a * *b) % P;
        }

        // Convert back to original domain by performing inverse NTT
        FTPoly { coeffs : coeffs, p : self.p, root : self.root  }
    }
}


impl FTPoly {
    // Scale the polynomial by a constant factor
    pub fn scale(self, factor: u128) -> FTPoly {
        let P = self.p;
        let coeffs = self.coeffs.iter().map(|&c| (c * factor) % P).collect();
        FTPoly { coeffs, p: self.p, root: self.root }
    }
}