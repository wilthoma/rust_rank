use crate::matrices::MyInt;


fn modp(x: MyInt, P : MyInt) -> MyInt {
    let mut r = x % P;
    if r < 0 { r += P; }
    r
}

fn modinv(a: MyInt, P : MyInt) -> MyInt {
    // Compute modular inverse using extended Euclidean algorithm
    let (mut t, mut new_t) = (0, 1);
    let (mut r, mut new_r) = (P, a);
    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }
    modp(t, P)
}

fn trim(mut poly: Vec<MyInt>) -> Vec<MyInt> {
    while poly.last() == Some(&0) && poly.len() > 1 {
        poly.pop();
    }
    poly
}

fn multiply_polynomials(a: &[MyInt], b: &[MyInt], P : MyInt) -> Vec<MyInt> {
    let mut result = vec![0; a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] = modp(result[i + j] + a[i] * b[j], P);
        }
    }
    trim(result)
}

fn divide_polynomials(numer: &[MyInt], denom: &[MyInt], P : MyInt) -> Vec<MyInt> {
    let mut result = vec![0; numer.len().saturating_sub(denom.len()) + 1];
    let mut remainder = numer.to_vec();

    while remainder.len() >= denom.len() && remainder.iter().any(|&x| x != 0) {
        let coeff = modp(remainder[remainder.len() - 1] * modinv(denom[denom.len() - 1], P), P);
        let deg = remainder.len() - denom.len();
        result[deg] = coeff;

        for i in 0..denom.len() {
            let idx = deg + i;
            remainder[idx] = modp(remainder[idx] - coeff * denom[i], P);
        }

        remainder = trim(remainder);
    }

    trim(result)
}

fn polynomial_mod(a: &[MyInt], b: &[MyInt], P : MyInt) -> Vec<MyInt> {
    let mut r = a.to_vec();

    while r.len() >= b.len() && r.iter().any(|&x| x != 0) {
        let coeff = modp(r[r.len() - 1] * modinv(b[b.len() - 1], P), P);
        let deg = r.len() - b.len();
        for i in 0..b.len() {
            r[deg + i] = modp(r[deg + i] - coeff * b[i],P);
        }
        r = trim(r);
    }

    r
}

fn gcd_polynomials(mut a: Vec<MyInt>, mut b: Vec<MyInt>, P : MyInt) -> Vec<MyInt> {
    while !b.is_empty() && b.iter().any(|&x| x != 0) {
        let r = polynomial_mod(&a, &b, P);
        a = b;
        b = r;
    }

    if let Some(&lead) = a.last() {
        let inv = modinv(lead, P);
        a.iter_mut().for_each(|x| *x = modp(*x * inv, P));
    }

    trim(a)
}

fn lcm_two_polynomials(a: &[MyInt], b: &[MyInt], P : MyInt) -> Vec<MyInt> {
    let gcd = gcd_polynomials(a.to_vec(), b.to_vec(), P);
    let product = multiply_polynomials(a, b, P);
    divide_polynomials(&product, &gcd, P)
}

fn lcm_polynomials(polys: &[Vec<MyInt>], P : MyInt) -> Vec<MyInt> {
    polys.iter()
        .cloned()
        .reduce(|a, b| lcm_two_polynomials(&a, &b, P))
        .unwrap_or_else(|| vec![1])
}


pub fn block_berlekamp_massey(seq: Vec<Vec<MyInt>>, num_u:usize, num_v:usize, p:MyInt) -> Vec<MyInt> {
    if num_v >1 {
        println!("Warning: Matrix Berlekamp Massey is not yet implemented. We just take least common multiple of minimal poly of all sequences.");
    }

    // convert all sequences to u64
    let useq: Vec<Vec<u64>> = seq.iter().map(|s| s.iter().map(|&x| (if x>=0 {x} else {x+p})  as u64).collect()).collect();

    let bmres: Vec<Vec<u64>> = useq.iter().map(|s| bubblemath::linear_recurrence::berlekamp_massey(s, p as u64))
        .collect();


    // convert back to MyInt
    let bmres: Vec<Vec<MyInt>> = bmres.iter().map(|s| s.iter().map(|&x| x as MyInt).collect()).collect();
    return lcm_polynomials(&bmres, p );

}