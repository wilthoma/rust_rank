use crate::matrices::MyInt;


fn modp(x: MyInt, p : MyInt) -> MyInt {
    let mut r = x % p;
    if r < (0 as MyInt) { r += p; }
    r
}

fn modinv(a: MyInt, p : MyInt) -> MyInt {
    // Compute modular inverse using extended Euclidean algorithm
    let (mut t, mut new_t) = (0 as MyInt, 1 as MyInt);
    let (mut r, mut new_r) = (p, a);
    while new_r != (0 as MyInt) {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }
    modp(t, p)
}

fn trim(mut poly: Vec<MyInt>) -> Vec<MyInt> {
    while poly.last() == Some(&(0 as MyInt)) && poly.len() > 1 {
        poly.pop();
    }
    poly
}

fn multiply_polynomials(a: &[MyInt], b: &[MyInt], p : MyInt) -> Vec<MyInt> {
    let mut result: Vec<MyInt> = vec![0 as MyInt; a.len() + b.len() - 1];
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[i + j] = modp(result[i + j] + a[i] * b[j], p);
        }
    }
    trim(result)
}

fn divide_polynomials(numer: &[MyInt], denom: &[MyInt], p : MyInt) -> Vec<MyInt> {
    let mut result = vec![0 as MyInt; numer.len().saturating_sub(denom.len()) + 1];
    let mut remainder = numer.to_vec();

    while remainder.len() >= denom.len() && remainder.iter().any(|&x| x != (0 as MyInt)) {
        let coeff = modp(remainder[remainder.len() - 1] * modinv(denom[denom.len() - 1], p), p);
        let deg = remainder.len() - denom.len();
        result[deg] = coeff;

        for i in 0..denom.len() {
            let idx = deg + i;
            remainder[idx] = modp(remainder[idx] - coeff * denom[i], p);
        }

        remainder = trim(remainder);
    }

    trim(result)
}

fn polynomial_mod(a: &[MyInt], b: &[MyInt], p : MyInt) -> Vec<MyInt> {
    let mut r = a.to_vec();

    while r.len() >= b.len() && r.iter().any(|&x| x != (0 as MyInt)) {
        let coeff = modp(r[r.len() - 1] * modinv(b[b.len() - 1], p), p);
        let deg = r.len() - b.len();
        for i in 0..b.len() {
            r[deg + i] = modp(r[deg + i] - coeff * b[i],p);
        }
        r = trim(r);
    }

    r
}

fn gcd_polynomials(mut a: Vec<MyInt>, mut b: Vec<MyInt>, p : MyInt) -> Vec<MyInt> {
    while !b.is_empty() && b.iter().any(|&x| x != (0 as MyInt)) {
        let r = polynomial_mod(&a, &b, p);
        a = b;
        b = r;
    }

    if let Some(&lead) = a.last() {
        let inv = modinv(lead, p);
        a.iter_mut().for_each(|x| *x = modp(*x * inv, p));
    }

    trim(a)
}

fn lcm_two_polynomials(a: &[MyInt], b: &[MyInt], p : MyInt) -> Vec<MyInt> {
    let gcd = gcd_polynomials(a.to_vec(), b.to_vec(), p);
    let product = multiply_polynomials(a, b, p);
    divide_polynomials(&product, &gcd, p)
}

fn lcm_polynomials(polys: &[Vec<MyInt>], p : MyInt) -> Vec<MyInt> {
    polys.iter()
        .cloned()
        .reduce(|a, b| lcm_two_polynomials(&a, &b, p))
        .unwrap_or_else(|| vec![1 as MyInt])
}


pub fn block_berlekamp_massey(seq: Vec<Vec<MyInt>>, num_u:usize, num_v:usize, p:MyInt) -> Vec<MyInt> {
    if num_v >1 || num_u > 1{
        println!("Warning: Matrix Berlekamp Massey is not yet implemented. We just take least common multiple of minimal poly of all sequences.");
    }

    // convert all sequences to u64
    let useq: Vec<Vec<u64>> = seq.iter().map(|s| s.iter().map(|&x| (if x>=(0 as MyInt) {x} else {x+p})  as u64).collect()).collect();

    let bmres: Vec<Vec<u64>> = useq.iter().map(|s| bubblemath::linear_recurrence::berlekamp_massey(s, p as u64))
        .collect();

    for v in bmres.iter(){
        println!("BM individual length: {}", v.len())  ;
    }

    // convert back to MyInt
    let bmres: Vec<Vec<MyInt>> = bmres.iter().map(|s| s.iter().map(|&x| x as MyInt).collect()).collect();
    return lcm_polynomials(&bmres, p );

}