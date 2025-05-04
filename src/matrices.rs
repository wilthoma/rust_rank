


// use petgraph::csr::Csr;
use rayon::prelude::*;
use std::array::from_fn;
use std::fs::File;
use std::io::{self, BufRead};
use std::simd::simd_swizzle;
use std::vec;
use rand::Rng;
// use std::cmp::{Ordering, min};
use std::iter::Sum;
use image::{ImageBuffer, Luma};
use std::path::Path;
use std::time::Instant;
use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use std::ops::{Add, Sub, AddAssign, SubAssign, Mul, Rem, MulAssign, Div, DivAssign};
use num_traits::{One, Zero};
use rand::distr::uniform::SampleUniform;
use std::fmt::Display;
use itertools::Itertools; // Import Itertools for the `chunks` method

// use std::alloc::{alloc_zeroed, Layout};
// use std::ptr::NonNull;

// use core::simd::{Simd, SimdInt}; // SIMD signed integers

pub const DOT_PRODUCT_CHUNK_SIZE: usize = 100;


pub trait GoodInteger: MulAssign+ Div<Output = Self> + DivAssign + Sub<Output = Self> + SubAssign + Display+Sum+PartialOrd + SampleUniform + SimdElement + Copy + Zero + One + Rem<Output = Self> + Add<Output = Self> + AddAssign + Mul<Output = Self> + Send + Sync +'static { }
impl<T: MulAssign+ Div<Output = Self> + DivAssign + Sub<Output = Self> + SubAssign + Display+ Sum+PartialOrd + SampleUniform + SimdElement + Copy + Zero + One + Rem<Output = Self> + Add<Output = Self> + AddAssign + Mul<Output = Self> + Send + Sync+'static> GoodInteger for T {}

pub trait GoodSimd : Copy
+ Send
+ Sync
+ Add<Output = Self>
+ AddAssign
+ Mul<Output = Self>
+ Rem<Output = Self> { }
impl<T, const LANES : usize> GoodSimd for Simd<T, LANES>
where
    T: GoodInteger,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES> : Send
    + Sync
    + Add<Output = Self>
    + AddAssign
    + Mul<Output = Self>
    + Rem<Output = Self> { }

/// Sparse matrix in Compressed Sparse Row (CSR) format
#[derive(Clone, Debug)]
pub struct CsrMatrix<T>
where
    T: SimdElement + Copy + Rem<Output = T>,
{
    pub values: Vec<T>,          // Non-zero values
    pub col_indices: Vec<usize>, // Column indices of values
    pub row_ptr: Vec<usize>,     // Index in `values` where each row starts
    pub n_rows: usize,           // Number of rows
    pub n_cols: usize,           // Number of columns
}


pub fn prettify_vect<T>(v: &[T], theprime: T) -> Vec<T> 
where T:Add<Output = T> + Rem<Output = T> + Copy ,{
    // let theprime= T::from(theprime);
    v.iter().map(|&x| (x + theprime) % theprime).collect()
}

pub fn zip_simd_vector<T,const LANES : usize>(vects : &[Vec<T>]) -> Vec<Simd<T, LANES>>
where
    T: SimdElement + Copy ,
    LaneCount<LANES>: SupportedLaneCount,
{
    assert_eq!(LANES, vects.len(), "must have one vector per lane.");
    if vects.len() == 0 {
        return vec![];
    }
    let length = vects[0].len();
    (0..length)
        .map(|i| {
            let arr = std::array::from_fn(|lane| vects[lane][i]);
            Simd::from_array(arr)
        })
        .collect()
}
pub fn zip_simd_vectors<T,const LANES : usize>(vects : &[Vec<T>]) -> Vec<Vec<Simd<T, LANES>>>
where
    T: SimdElement + Copy ,
    LaneCount<LANES>: SupportedLaneCount,
{
    assert_eq!(vects.len() % LANES , 0, "number of vectors must be multiple of lane width.");
    vects.chunks(LANES).map(|vs| zip_simd_vector(vs)).collect()
}

#[inline(always)]
pub fn unzip_simd_vector<T, const LANES: usize>(vect: &[Simd<T, LANES>]) -> [Vec<T>;LANES]//Vec<Vec<T>>
where
    T: SimdElement + Copy + Zero,
    LaneCount<LANES>: SupportedLaneCount,
{    
    let length = vect.len();
    assert!(length > 0, "Length of vector must be greater than 0");

    // let start = Instant::now();
    // let res1 = vect.iter().map(|x| x.to_array()).collect::<Vec<_>>();
    // let duration = start.elapsed();
    // println!("pre-Unzipping took: {:?}", duration);

    // let start = Instant::now();
    // let res2: [Vec<T>; LANES] = std::array::from_fn(|i| {
    //     res1.iter().map(|arr| arr[i]).collect::<Vec<_>>()
    // });
    // let duration = start.elapsed();
    // println!("pre-Unzipping2 took: {:?}", duration);

    // let start = Instant::now();
    let res3: [Vec<T>; LANES] = std::array::from_fn(|i| {
        vect.iter().map(|x| { let a = x.to_array(); 
            a[i] }).collect::<Vec<_>>()
    });
    // let duration = start.elapsed();
    // println!("pre-Unzipping3 took: {:?}", duration);

    // let start = Instant::now();
    // let mut result = vec![vec![T::zero();length]; LANES];
    // for (i,v) in vect.iter().enumerate() {
    //     let arr = vect[i].to_array();
    //     for lane in 0..LANES {
    //         result[lane][i] = arr[lane];
    //     }
    // }
    // let duration = start.elapsed();
    // println!("rest-Unzipping took: {:?}", duration);
    // result
    res3
}

#[inline(always)]
pub fn unzip_simd_vectors<T, const LANES: usize>(vects: &[Vec<Simd<T, LANES>>]) -> Vec<Vec<T>>
where
    T: SimdElement + Copy+ Zero,
    LaneCount<LANES>: SupportedLaneCount,
{   
    vects.into_iter().flat_map(|v| unzip_simd_vector(v)).collect()
}

#[inline(always)]
pub fn dot_product_mod_p_parallel<T>(vec1: &[T], vec2: &[T], theprime: T) -> T 
where T: 
Add<Output = T> +Mul<Output = T> + Copy + Rem<Output = T> + AddAssign + Send + Sync + Zero,{
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    let chunk_size = DOT_PRODUCT_CHUNK_SIZE;

    vec1.par_chunks(chunk_size)
        .zip(vec2.par_chunks(chunk_size))
        .map(|(chunk1, chunk2)| {
            chunk1.iter()
                .zip(chunk2.iter())
                .fold(T::zero(), |acc, (&x, &y)| acc + (x * y))
                % theprime
        })
        .reduce(|| T::zero(), |acc, chunk_sum| (acc + chunk_sum) ) % theprime
}

#[inline(always)]
pub fn dot_product_mod_p_serial<T>(vec1: &[T], vec2: &[T], theprime: T) -> T
where T: 
Copy + Rem<Output = T> + AddAssign + Zero + Sum<T> + Mul<Output = T> + Add<Output = T>,{
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    let chunk_size = DOT_PRODUCT_CHUNK_SIZE;

    vec1.chunks(chunk_size)
        .zip(vec2.chunks(chunk_size))
        .map(|(chunk1, chunk2)| {
            chunk1.iter()
                .zip(chunk2.iter())
                .fold(T::zero(), |acc, (&x, &y)| acc + (x * y))
                % theprime
        })
        .sum::<T>() % theprime
}

pub fn dot_product_simd<T, const LANES : usize>(
    vec1: &[Simd<T, LANES>],
    vec2: &[Simd<T, LANES>],
    theprime: T,
) -> Simd<T, LANES>
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd,
    LaneCount<LANES>: SupportedLaneCount,
{
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");
    vec1.iter().zip(vec2.iter())
    .chunks(DOT_PRODUCT_CHUNK_SIZE)
    .into_iter()
    .map(|chunk| {
        chunk.fold(Simd::splat(T::zero()), |acc, (&x, &y)| acc + (x * y)) % Simd::splat(theprime)
    }).fold(Simd::splat(T::zero()), |acc, x| acc + x ) % Simd::splat(theprime)
}

pub fn dot_product_simd_naive<T, const LANES : usize>(
    vec1: &[Simd<T, LANES>],
    vec2: &[Simd<T, LANES>],
    theprime: T,
) -> Simd<T, LANES>
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd,
    LaneCount<LANES>: SupportedLaneCount,
{
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");
    let v1s = unzip_simd_vector(vec1);
    let v2s = unzip_simd_vector(vec2);

    let res: [T; LANES] = std::array::from_fn(|i|  {
        dot_product_mod_p_serial(&v1s[i], &v2s[i], theprime)
    });
    Simd::from_array(res)
}



pub trait Rotateme
{
    fn rotate(self) -> Self;
}

macro_rules! impl_rotate_simd {
    ($lanes:literal, [$($idx:expr),*]) => {
        impl<T: GoodInteger> Rotateme for Simd<T, $lanes> {
            fn rotate(self) -> Simd<T, $lanes> {
                simd_swizzle!(self, [$($idx),*])
            }
        }
    };
}

impl_rotate_simd!(1, [0]);
impl_rotate_simd!(2, [1, 0]);
impl_rotate_simd!(4, [1, 2, 3, 0]);
impl_rotate_simd!(8, [1, 2, 3, 4, 5, 6, 7, 0]);
impl_rotate_simd!(16, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]);

pub fn rotate_simd_vect<T, const LANES:usize>(
    vect: &[Simd<T, LANES>],
) -> Vec<Simd<T, LANES>>
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd + Rotateme,
    LaneCount<LANES>: SupportedLaneCount,
{
    vect.iter()
        .map(|&x| x.rotate())
        .collect()
}

pub fn rotate_simd_vect_in<T, const LANES:usize>(
    vect: &mut Vec<Simd<T, LANES>>,
)
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd + Rotateme,
    LaneCount<LANES>: SupportedLaneCount,
{
    for i in 0..vect.len() {
        vect[i] = vect[i].rotate();
    }
}

pub fn dot_products_simd<T, const LANES : usize>(
    vec1: &Vec<Simd<T, LANES>>,
    vec2: &Vec<Simd<T, LANES>>,
    theprime: T,
) -> [[T; LANES]; LANES]
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd + Rotateme,
    LaneCount<LANES>: SupportedLaneCount,
{
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");
    let mut result = [[T::zero(); LANES]; LANES];
    let mut preresult = [Simd::splat(T::zero()); LANES];

    preresult[0] = dot_product_simd(&vec1, &vec2, theprime);
    let mut v2rot = vec2.clone();
    for i in 1..LANES {
        // let start_rotate = Instant::now();
        rotate_simd_vect_in(&mut v2rot);
        // let duration_rotate = start_rotate.elapsed();
        // println!("Rotation took: {:?}", duration_rotate);
        // let start = Instant::now();   
        preresult[i] = dot_product_simd(&vec1, &v2rot, theprime);
        // let duration = start.elapsed();
        // println!("Dot product simd took: {:?}", duration);
    }

    // extract dot products from preresult
    for i in 0..LANES {
        let s = preresult[i].to_array();
        for j in 0..LANES {
            result[j][(i+j)%LANES] = s[j];
        }
    }
    result
}

pub fn dot_products_simd_naive<T, const LANES : usize>(
    vec1: &Vec<Simd<T, LANES>>,
    vec2: &Vec<Simd<T, LANES>>,
    theprime: T,
) -> [[T; LANES]; LANES]
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd + Rotateme,
    LaneCount<LANES>: SupportedLaneCount,
{
    let start = Instant::now();
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    let v1s = unzip_simd_vector(vec1);
    let v2s = unzip_simd_vector(vec2);
    let duration = start.elapsed();
    println!("Unzipping vectors took: {:?}", duration);   

    let start = Instant::now();

    let mut result: [[T; LANES]; LANES] = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let start1 = Instant::now();
            let res = dot_product_mod_p_serial(&v1s[i], &v2s[j], theprime);
            let duration1 = start1.elapsed();
            println!("Normal dot product took: {:?}", duration1);  
            res
        })
    });
    let duration = start.elapsed();
    println!("Normal dot products took: {:?}", duration);  
    result
}

pub fn dot_products_all_simd<T, const LANES : usize>(
    vects: &Vec<Vec<Simd<T, LANES>>>,
    theprime: T,
) -> Vec<T>
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd + Rotateme,
    LaneCount<LANES>: SupportedLaneCount,
{
    let m = vects.len();
    let num_v = m * LANES;

    // compute all products
    let mut prods_simd = vec![vec![ [[T::zero();LANES];LANES]; m]; m];
    for i in 0..m {
        for j in i..m {
            prods_simd[i][j] = dot_products_simd(&vects[i], &vects[j], theprime);
            prods_simd[j][i] = prods_simd[i][j];
        }
    }

    // extract individual dot products from the simd versions
    let mut ret = vec![T::zero(); num_v * (num_v+1)/2];
    let mut ii = 0;
    for i in 0..num_v {
        for j in i..num_v {
            ret[ii] = prods_simd[i/LANES][j/LANES][i%LANES][j%LANES];
            ii += 1;
        }
    }
    ret
}

pub fn dot_products_all_std<T, const LANES : usize>(
    vects: &Vec<Vec<Simd<T, LANES>>>,
    theprime: T,
) -> Vec<T>
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd + Rotateme,
    LaneCount<LANES>: SupportedLaneCount,
{
    let m = vects.len();
    let num_v = m * LANES;

    // unzip all SIMD vectors

    let start_unzip = Instant::now();
    let unzipped = vects.iter().flat_map(|v| unzip_simd_vector(v)).collect::<Vec<_>>();
    let duration_unzip = start_unzip.elapsed();
    println!("Unzipping vectors took: {:?}", duration_unzip);   
    // compute pairwise dot products
    let mut ii = 0;
    let mut ret = vec![T::zero(); num_v * (num_v+1)/2];
    for i in 0..num_v {
        for j in i..num_v {
            let vec1 = &unzipped[i];
            let vec2 = &unzipped[j];

            ret[ii] = dot_product_mod_p_serial(vec1, vec2, theprime);
            ii += 1;
        }
    }
    ret
}

pub fn dot_products_all_std2<T, const LANES : usize>(
    vects: &Vec<Vec<Simd<T, LANES>>>,
    theprime: T,
) -> Vec<T>
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd + Rotateme,
    LaneCount<LANES>: SupportedLaneCount,
{
    let m = vects.len();
    let num_v = m * LANES;

    let ijpairs = (0..num_v).into_iter()
    .flat_map(move |i| (i..num_v).into_iter().map(move |j| (i, j)));

    let dotps = ijpairs.map(|(i,j)| {
        let vec1 = &vects[i/LANES];
        let vec2 = &vects[j/LANES];
        let l1 = i%LANES;
        let l2 = j%LANES;
        vec1.chunks(DOT_PRODUCT_CHUNK_SIZE)
            .zip(vec2.chunks(DOT_PRODUCT_CHUNK_SIZE))
            .map(|(chunk1, chunk2)| {
                chunk1.iter()
                    .zip(chunk2.iter())
                    .fold(T::zero(), |acc, (&x, &y)| acc + (x.to_array()[l1] * y.to_array()[l2]))
                    % theprime
            })
        .sum::<T>() % theprime
    }).collect::<Vec<_>>();

    dotps
}


pub fn dot_products_all_std3<T, const LANES : usize>(
    vects: &Vec<Vec<Simd<T, LANES>>>,
    theprime: T,
) -> Vec<T>
where
    T: GoodInteger,
    Simd<T, LANES> : GoodSimd + Rotateme,
    LaneCount<LANES>: SupportedLaneCount,
{
    let m = vects.len();
    let num_v = m * LANES;

    let ijpairs = (0..num_v).into_iter()
    .flat_map(move |i| (i..num_v).into_iter().map(move |j| (i, j)));

    let dotps = ijpairs.map(|(i,j)| {
        let vec1 = &vects[i/LANES];
        let vec2 = &vects[j/LANES];
        let l1 = i%LANES;
        let l2 = j%LANES;
        vec1.chunks(DOT_PRODUCT_CHUNK_SIZE)
            .zip(vec2.chunks(DOT_PRODUCT_CHUNK_SIZE))
            .map(|(chunk1, chunk2)| {
                chunk1.iter()
                    .zip(chunk2.iter())
                    .fold(T::zero(), |acc, (&x, &y)| acc + (x.to_array()[l1] * y.to_array()[l2]))
                    % theprime
            })
        .sum::<T>() % theprime
    }).collect::<Vec<_>>();

    dotps
}


pub fn create_random_vector<T>(length: usize, theprime: T) -> Vec<T> 
where T : Zero + SampleUniform + PartialOrd + Copy , 
{
    let mut rng = rand::rng();
    (0..length).map(|_| T::from( rng.random_range(T::zero()..theprime)) ).collect()
}


pub fn create_random_vector_nozero<T>(length: usize, theprime: T) -> Vec<T> 
where T : One + SampleUniform + PartialOrd + Copy, 
{
    let mut rng = rand::rng();
    (0..length).map(|_| T::from( rng.random_range(T::one()..theprime)) ).collect()
}

pub fn create_random_vector_simd<T, const LANES: usize>(
    length: usize,
    theprime: T,
    // mut scalar_gen: impl FnMut(usize, T) -> Vec<T>,
) -> Vec<Simd<T, LANES>>
where
    T: SimdElement + Copy +Zero + SampleUniform + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    // Generate one scalar vector per lane
    let scalar_vectors: [Vec<T>; LANES] = std::array::from_fn(| _ | create_random_vector(length, theprime));

    (0..length)
        .map(|i| {
            let arr = std::array::from_fn(|lane| scalar_vectors[lane][i]);
            Simd::from_array(arr)
        })
        .collect()
}


impl<T> CsrMatrix<T> where
T: SimdElement
+ Copy
+ Zero
+ Rem<Output = T>
+ Send
+ Sync
+ Add<Output = T>
+ AddAssign
+ Mul<Output = T>
+ SampleUniform
+ PartialOrd,
{
    

    pub fn parallel_sparse_matvec_mul_simd<const LANES:usize>(&self, vector: &[Simd<T, LANES>], theprime: T) -> Vec<Simd<T, LANES>> 
    where
LaneCount<LANES>: SupportedLaneCount,
Simd<T, LANES>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, LANES>>
    + AddAssign
    + Mul<Output = Simd<T, LANES>>
    + Rem<Output = Simd<T, LANES>>,
    {
        assert_eq!(self.n_cols, vector.len(), "Matrix and vector dimensions must align.");
        // const ttheprime :MyInt = 27644437;
        // Parallel iterator over rows
        (0..self.n_rows).into_par_iter().map(|row| {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];


            let colis = self.col_indices[start..end].iter().map(|&col| vector[col]);
            let sum = colis
                .zip(&self.values[start..end])
                .fold(Simd::splat(T::zero()), |acc, (v, &val)| (acc + v * Simd::splat(val)) ) ;
            sum % Simd::splat(theprime)
        }).collect()
    }

    pub fn serial_sparse_matvec_mul_simd<const LANES:usize>(&self, vector: &[Simd<T, LANES>], theprime: T) -> Vec<Simd<T, LANES>> 
    where
LaneCount<LANES>: SupportedLaneCount,
Simd<T, LANES>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, LANES>>
    + AddAssign
    + Mul<Output = Simd<T, LANES>>
    + Rem<Output = Simd<T, LANES>>,
    {
        assert_eq!(self.n_cols, vector.len(), "Matrix and vector dimensions must align.");
        // const ttheprime :MyInt = 27644437;
        // Parallel iterator over rows
        (0..self.n_rows).into_iter().map(|row| {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];


            let colis = self.col_indices[start..end].iter().map(|&col| vector[col]);
            let sum = colis
                .zip(&self.values[start..end])
                .fold(Simd::splat(T::zero()), |acc, (v, &val)| (acc + v * Simd::splat(val)) ) ;
            sum % Simd::splat(theprime)
        }).collect()
    }

    pub fn load_csr_matrix_from_sms(file_path: &str, theprime : u32) -> Result<CsrMatrix<T>, Box<dyn std::error::Error>> 
    where T : TryFrom<u32> ,
    <T as TryFrom<u32>>::Error : std::fmt::Debug,
    // T as TryFrom<u32>: Debug 
    {
        let file = File::open(file_path)?;
        let reader = io::BufReader::new(file);
    
        let mut values = Vec::new();
        let mut col_indices = Vec::new();
        let mut row_ptr = Vec::new();
        let mut n_rows = 0;
        let mut n_cols = 0;
    
        row_ptr.push(0); // Start of the first row
    
        for line in reader.lines() {
            let line = line?;
            let line = line.trim();
    
            if line.starts_with('%') || line.is_empty() {
                // Skip comments or empty lines
                continue;
            }
    
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() == 3 && n_rows <= 0 && n_cols <= 0 {
                // Header line with dimensions
                n_rows = parts[0].parse()?;
                n_cols = parts[1].parse()?;
                //let _non_zeros: usize = parts[2].parse()?; // Not used directly
            } else if parts.len() == 3 {
                // Matrix entry
                let row: usize = parts[0].parse()?;
                let col: usize = parts[1].parse()?;
                let theprimei = theprime as i32;
                let mut intvalue :i32 = parts[2].parse()?;
                intvalue = intvalue % theprimei;
                if intvalue < 0 {
                    intvalue += theprimei;
                }
                let value: T = T::try_from(intvalue as u32).unwrap();

    
                if row<=0 || col<=0 {
                    continue; // Skip invalid entries
                }
    
                // Convert 1-based indexing to 0-based
                while row_ptr.len() < row {
                    row_ptr.push(values.len());
                }
    
                values.push(value);
                col_indices.push(col - 1);
            }
        }
    
        // Finalize row_ptr
        while row_ptr.len() <= n_rows {
            row_ptr.push(values.len());
        }
    
        Ok(CsrMatrix {
            values,
            col_indices,
            row_ptr,
            n_rows,
            n_cols,
        })
    }
    
    pub fn transpose(&self) -> CsrMatrix<T> {
        let mut nnz_per_col = vec![0; self.n_cols];

        // Step 1: Count non-zero elements per column
        for &col in &self.col_indices {
            nnz_per_col[col] += 1;
        }

        // Step 2: Compute row_ptr for transposed matrix
        let mut row_ptr_t = vec![0; self.n_cols + 1];
        for i in 0..self.n_cols {
            row_ptr_t[i + 1] = row_ptr_t[i] + nnz_per_col[i];
        }

        // Step 3: Prepare space for transposed values and indices
        let nnz = self.values.len();
        let mut values_t = vec![T::zero(); nnz];
        let mut col_indices_t = vec![0; nnz];

        let mut next_insert_pos = row_ptr_t.clone();

        // Step 4: Populate the transposed matrix
        for row in 0..self.n_rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            for idx in start..end {
                let col = self.col_indices[idx];
                let val = self.values[idx];

                let insert_pos = next_insert_pos[col];
                values_t[insert_pos] = val;
                col_indices_t[insert_pos] = row;
                next_insert_pos[col] += 1;
            }
        }

        CsrMatrix {
            values: values_t,
            col_indices: col_indices_t,
            row_ptr: row_ptr_t,
            n_rows: self.n_cols,
            n_cols: self.n_rows,
        }
    }

    pub fn parallel_sparse_matvec_mul(&self, vector: &[T], theprime: T) -> Vec<T> {
        assert_eq!(self.n_cols, vector.len(), "Matrix and vector dimensions must align.");
        // Parallel iterator over rows
        (0..self.n_rows).into_par_iter().map(|row| {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];


            let colis = self.col_indices[start..end].iter().map(|&col| vector[col]);
            let sum = colis
                .zip(&self.values[start..end])
                .fold(T::zero(), |acc, (v, &val)| (acc + v * val)) % theprime;
            sum
        }).collect()
    }

    pub fn serial_sparse_matvec_mul(&self, vector: &[T], theprime: T) -> Vec<T> {
        assert_eq!(self.n_cols, vector.len(), "Matrix and vector dimensions must align.");
        (0..self.n_rows).into_iter().map(|row| {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];


            let colis = self.col_indices[start..end].iter().map(|&col| vector[col]);
            let sum = colis
                .zip(&self.values[start..end])
                .fold(T::zero(), |acc, (v, &val)| (acc + v * val)) % theprime;
            sum
        }).collect()
    }

    /// Multiply a CSR matrix column-wise with the entries of a vector
    /// The existing CSR matrix is updated in-place
    pub fn scale_csr_matrix_columns(&mut self, vector: &[T], theprime: T) {
        let matrix = self;
        assert_eq!(matrix.n_cols, vector.len(), "Matrix and vector dimensions must align.");
        // let theprime = T::from(theprime);
        for i in 0..matrix.values.len() {
            let col = matrix.col_indices[i];
            matrix.values[i] = (matrix.values[i] * vector[col]) % theprime;
        }
    }
    /// Multiply a CSR matrix row-wise with the entries of a vector
    /// The existing CSR matrix is updated in-place
    pub fn scale_csr_matrix_rows(&mut self, vector: &[T], theprime: T) {
        let matrix = self;
        assert_eq!(matrix.n_rows, vector.len(), "Matrix and vector dimensions must align.");
        // let theprime = T::from(theprime);

        for row in 0..matrix.n_rows {
            let start = matrix.row_ptr[row];
            let end = matrix.row_ptr[row + 1];
            for i in start..end {
                if i>= matrix.values.len() {
                    println!("{} {} {} {}", i, matrix.values.len(), matrix.row_ptr.len(), row);
                }
                let cc =matrix.values[i];
                matrix.values[i] = (cc * vector[row]) % theprime;
            }
        }
    }

    pub fn max_nnzs(&self) -> (usize, usize) {
        // compute row and column nnz
        let mut row_nnz = vec![0; self.n_rows];
        let mut col_nnz = vec![0; self.n_cols];

        // Precompute row_nnz
        for row in 0..self.n_rows {
            row_nnz[row] = self.row_ptr[row + 1] - self.row_ptr[row];
        }

        // Precompute col_nnz
        for &col in &self.col_indices {
            col_nnz[col] += 1;
        }
        // Find the maximum nnz in rows and columns
        let max_row_nnz = *row_nnz.iter().max().unwrap_or(&0);
        let max_col_nnz = *col_nnz.iter().max().unwrap_or(&0);
        (max_row_nnz, max_col_nnz)
    }
    pub fn is_prime_valid(&self, theprime: u32, max_t_value : i128) -> bool {
        let (max_row_nnz, max_col_nnz) = self.max_nnzs();

        let prod = (theprime as i128) * (theprime as i128);
        // let max_i64_value = T::max_value(); //std::i64::MAX as i128;
        // check if prod * nnz <  std::i128::MAX for all entries
        (prod * (DOT_PRODUCT_CHUNK_SIZE as i128) < max_t_value) && // chunk_size 100 for dot products assumed
        (max_row_nnz as i128) * prod < max_t_value && 
        (max_col_nnz as i128) * prod < max_t_value
    }

    pub fn normal_simd_speedtest(&self, theprime : T, n_reps: usize) 
    where Simd<T, 1>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, 1>>
    + AddAssign
    + Mul<Output = Simd<T, 1>>
    + Rem<Output = Simd<T, 1>>,
    Simd<T, 2>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, 2>>
    + AddAssign
    + Mul<Output = Simd<T, 2>>
    + Rem<Output = Simd<T, 2>>,
    Simd<T, 4>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, 4>>
    + AddAssign
    + Mul<Output = Simd<T, 4>>
    + Rem<Output = Simd<T, 4>>,
    Simd<T, 8>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, 8>>
    + AddAssign
    + Mul<Output = Simd<T, 8>>
    + Rem<Output = Simd<T, 8>>,
    Simd<T, 16>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, 16>>
    + AddAssign
    + Mul<Output = Simd<T, 16>>
    + Rem<Output = Simd<T, 16>>,
    Simd<T, 32>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, 32>>
    + AddAssign
    + Mul<Output = Simd<T, 32>>
    + Rem<Output = Simd<T, 32>>,{
        let svector32: Vec<Simd<T, 32>> = create_random_vector_simd(self.n_cols, theprime);
        let svector16: Vec<Simd<T, 16>> = create_random_vector_simd(self.n_cols, theprime);
        let svector8: Vec<Simd<T, 8>> = create_random_vector_simd(self.n_cols, theprime);
        let svector4: Vec<Simd<T, 4>> = create_random_vector_simd(self.n_cols, theprime);
        let svector2: Vec<Simd<T, 2>> = create_random_vector_simd(self.n_cols, theprime);
        let svector1: Vec<Simd<T, 1>> = create_random_vector_simd(self.n_cols, theprime);
        let nvector = create_random_vector(self.n_cols, theprime);
        for _ in 0..n_reps {
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul(&nvector, theprime);
            let duration = start.elapsed();
            println!("Normal multiplication took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd(&svector1, theprime);
            let duration = start.elapsed();
            println!("SIMD1 multiplication took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd(&svector2, theprime);
            let duration = start.elapsed();
            println!("SIMD2 multiplication took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd(&svector4, theprime);
            let duration = start.elapsed();
            println!("SIMD4 multiplication took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd(&svector8, theprime);
            let duration = start.elapsed();
            println!("SIMD8 multiplication took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd(&svector16, theprime);
            let duration = start.elapsed();
            println!("SIMD16 multiplication took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd(&svector32, theprime);
            let duration = start.elapsed();
            println!("SIMD32 multiplication took: {:?}", duration);
        }
    }

    pub fn normal_simd_speedtest_serial(&self, theprime : T, n_reps: usize) 
    where Simd<T, 4>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, 4>>
    + AddAssign
    + Mul<Output = Simd<T, 4>>
    + Rem<Output = Simd<T, 4>>,
    Simd<T, 8>: Copy
    + Send
    + Sync
    + Add<Output = Simd<T, 8>>
    + AddAssign
    + Mul<Output = Simd<T, 8>>
    + Rem<Output = Simd<T, 8>>,{
        let svector8: Vec<Simd<T, 8>> = create_random_vector_simd(self.n_cols, theprime);
        let svector4: Vec<Simd<T, 4>> = create_random_vector_simd(self.n_cols, theprime);
        let nvector = create_random_vector(self.n_cols, theprime);
        for _ in 0..n_reps {
            let start = Instant::now();
            let _result = self.serial_sparse_matvec_mul(&nvector, theprime);
            let duration = start.elapsed();
            println!("Normal multiplication (serial) took: {:?}", duration);
            let start = Instant::now();
            let _result = self.serial_sparse_matvec_mul_simd(&svector4, theprime);
            let duration = start.elapsed();
            println!("SIMD4 multiplication (serial) took: {:?}", duration);
            let start = Instant::now();
            let _result = self.serial_sparse_matvec_mul_simd(&svector8, theprime);
            let duration = start.elapsed();
            println!("SIMD8 multiplication (serial) took: {:?}", duration);
        }
    }

    pub fn spy_plot(
        &self,
        filename: &str,
        height: u32,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Compute width based on matrix aspect ratio
        let width = ((self.n_cols as f32 / self.n_rows as f32) * height as f32).ceil() as u32;
    
        let mut img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_pixel(width, height, Luma([255u8]));
    
        let row_scale = self.n_rows as f32 / height as f32;
        let col_scale = self.n_cols as f32 / width as f32;
    
        for row in 0..self.n_rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for idx in start..end {
                let col = self.col_indices[idx];
    
                let x = (col as f32 / col_scale).floor() as u32;
                let y = (row as f32 / row_scale).floor() as u32;
    
                if x < width && y < height {
                    img.put_pixel(x, height - 1 - y, Luma([0u8]));
                }
            }
        }
    
        img.save(Path::new(filename))?;
        Ok(())
    }

    pub fn from<S>(a : &CsrMatrix<S>) -> CsrMatrix<T>
    where S: SimdElement + Copy + From<u32> + Rem<Output = S> + Send + Sync + Add<Output = S> + AddAssign + Mul<Output = S>,
    T: From<S>
    {
        CsrMatrix {
            values: a.values.iter().map(|&x| T::from(x)).collect(),
            col_indices: a.col_indices.clone(),
            row_ptr: a.row_ptr.clone(),
            n_rows: a.n_rows,
            n_cols: a.n_cols,
        }
    }


}


impl CsrMatrix<i64> {
    pub fn toi32(&self) -> CsrMatrix<i32> {
        let a = self;
        CsrMatrix {
            values: a.values.iter().map(|&x| x as i32).collect(),
            col_indices: a.col_indices.clone(),
            row_ptr: a.row_ptr.clone(),
            n_rows: a.n_rows,
            n_cols: a.n_cols,
        }
    }
}

impl CsrMatrix<u64> {
    pub fn tou32(&self) -> CsrMatrix<u32> {
        let a = self;
        CsrMatrix {
            values: a.values.iter().map(|&x| x as u32).collect(),
            col_indices: a.col_indices.clone(),
            row_ptr: a.row_ptr.clone(),
            n_rows: a.n_rows,
            n_cols: a.n_cols,
        }
    }
    pub fn tou64(&self) -> CsrMatrix<u64> {
        self.clone()
    }
    pub fn tou16(&self) -> CsrMatrix<u16> {
        let a = self;
        CsrMatrix {
            values: a.values.iter().map(|&x| x as u16).collect(),
            col_indices: a.col_indices.clone(),
            row_ptr: a.row_ptr.clone(),
            n_rows: a.n_rows,
            n_cols: a.n_cols,
        }
    }
}

impl CsrMatrix<u32> {
    pub fn tou64(&self) -> CsrMatrix<u64> {
        let a = self;
        CsrMatrix {
            values: a.values.iter().map(|&x| x as u64).collect(),
            col_indices: a.col_indices.clone(),
            row_ptr: a.row_ptr.clone(),
            n_rows: a.n_rows,
            n_cols: a.n_cols,
        }
    }
    pub fn tou32(&self) -> CsrMatrix<u32> {
        self.clone()
    }
    pub fn tou16(&self) -> CsrMatrix<u16> {
        let a = self;
        CsrMatrix {
            values: a.values.iter().map(|&x| x as u16).collect(),
            col_indices: a.col_indices.clone(),
            row_ptr: a.row_ptr.clone(),
            n_rows: a.n_rows,
            n_cols: a.n_cols,
        }
    }
}

impl CsrMatrix<u16> {
    pub fn tou64(&self) -> CsrMatrix<u64> {
        let a = self;
        CsrMatrix {
            values: a.values.iter().map(|&x| x as u64).collect(),
            col_indices: a.col_indices.clone(),
            row_ptr: a.row_ptr.clone(),
            n_rows: a.n_rows,
            n_cols: a.n_cols,
        }
    }
    pub fn tou32(&self) -> CsrMatrix<u32> {
        let a = self;
        CsrMatrix {
            values: a.values.iter().map(|&x| x as u32).collect(),
            col_indices: a.col_indices.clone(),
            row_ptr: a.row_ptr.clone(),
            n_rows: a.n_rows,
            n_cols: a.n_cols,
        }
    }
    pub fn tou16(&self) -> CsrMatrix<u16> {
        self.clone()
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_products_all_simd_vs_std() {
        const LANES: usize = 4;
        const VECTOR_LENGTH: usize = 1000;
        const THE_PRIME: u32 = 27644437;

        // Generate random SIMD vectors
        let vects: Vec<Vec<Simd<u32, LANES>>> = (0..10)
            .map(|_| create_random_vector_simd(VECTOR_LENGTH, THE_PRIME))
            .collect();

        // Compute dot products using both methods
        let simd_result = dot_products_all_simd(&vects, THE_PRIME);
        let std_result = dot_products_all_std(&vects, THE_PRIME);

        // Assert that both results are the same
        assert_eq!(simd_result, std_result, "Results from SIMD and standard methods do not match.");
    }

    #[test]
    fn benchmark_dot_products_all() {
        return; // Skip this test for now
        const LANES: usize = 4;
        const VECTOR_LENGTH: usize = 30000000;
        const THE_PRIME: u32 = 3323;

        // Generate random SIMD vectors
        let vects: Vec<Vec<Simd<u32, LANES>>> = (0..8)
            .map(|_| create_random_vector_simd(VECTOR_LENGTH, THE_PRIME))
            .collect();

        // Benchmark standard method
        let start_std = Instant::now();
        let _std_result = dot_products_all_std(&vects, THE_PRIME);
        let duration_std = start_std.elapsed();
        println!("Standard method took: {:?}", duration_std);

        // Benchmark standard method
        let start_std = Instant::now();
        let _std_result2 = dot_products_all_std2(&vects, THE_PRIME);
        let duration_std = start_std.elapsed();
        println!("Standard method2 took: {:?}", duration_std);

        // Benchmark SIMD method
        let start_simd = Instant::now();
        let _simd_result = dot_products_all_simd(&vects, THE_PRIME);
        let duration_simd = start_simd.elapsed();
        println!("SIMD method took: {:?}", duration_simd);

                // Benchmark standard method
                let start_std = Instant::now();
                let _std_result = dot_products_all_std(&vects, THE_PRIME);
                let duration_std = start_std.elapsed();
                println!("Standard method took: {:?}", duration_std);

                        // Benchmark standard method
        let start_std = Instant::now();
        let _std_result2 = dot_products_all_std2(&vects, THE_PRIME);
        let duration_std = start_std.elapsed();
        println!("Standard method2 took: {:?}", duration_std);
        
                // Benchmark SIMD method
                let start_simd = Instant::now();
                let _simd_result = dot_products_all_simd(&vects, THE_PRIME);
                let duration_simd = start_simd.elapsed();
                println!("SIMD method took: {:?}", duration_simd);

    }

    #[test]
    fn benchmark_dot_product_simd_vs_naive() {

        const LANES: usize = 8;
        const VECTOR_LENGTH: usize = 10000000;
        const THE_PRIME: u32 = 3323;

        // Generate random SIMD vectors
        let vec1: Vec<Simd<u32, LANES>> = create_random_vector_simd(VECTOR_LENGTH, THE_PRIME);
        let vec2: Vec<Simd<u32, LANES>> = create_random_vector_simd(VECTOR_LENGTH, THE_PRIME);

        // Benchmark SIMD method
        let start_simd = Instant::now();
        let _simd_result = dot_product_simd(&vec1, &vec2, THE_PRIME);
        let duration_simd = start_simd.elapsed();
        println!("SIMD method took: {:?}", duration_simd);

        // Benchmark naive SIMD method
        let start_naive = Instant::now();
        let _naive_result = dot_product_simd_naive(&vec1, &vec2, THE_PRIME);
        let duration_naive = start_naive.elapsed();
        println!("Naive SIMD method took: {:?}", duration_naive);

        // Benchmark SIMD method
        let start_simd = Instant::now();
        let _simd_result = dot_product_simd(&vec1, &vec2, THE_PRIME);
        let duration_simd = start_simd.elapsed();
        println!("SIMD method took: {:?}", duration_simd);

        // Benchmark naive SIMD method
        let start_naive = Instant::now();
        let _naive_result = dot_product_simd_naive(&vec1, &vec2, THE_PRIME);
        let duration_naive = start_naive.elapsed();
        println!("Naive SIMD method took: {:?}", duration_naive);

                // Benchmark SIMD method
                let start_simd = Instant::now();
                let _simd_result = dot_product_simd(&vec1, &vec2, THE_PRIME);
                let duration_simd = start_simd.elapsed();
                println!("SIMD method took: {:?}", duration_simd);
        
                // Benchmark naive SIMD method
                let start_naive = Instant::now();
                let _naive_result = dot_product_simd_naive(&vec1, &vec2, THE_PRIME);
                let duration_naive = start_naive.elapsed();
                println!("Naive SIMD method took: {:?}", duration_naive);

        // Assert that both results are the same
        assert_eq!(_simd_result, _naive_result, "Results from SIMD and naive SIMD methods do not match.");
    }

    #[test]
    fn benchmark_dot_products_simd_vs_naive() {
        for steps in 0..2 {
        // return;
        const LANES: usize = 4;
        const VECTOR_LENGTH: usize = 30000000;
        const THE_PRIME: u32 = 3323;

        // Generate random SIMD vectors
        let vects: Vec<Vec<Simd<u32, LANES>>> = (0..1)
            .map(|_| create_random_vector_simd(VECTOR_LENGTH, THE_PRIME))
            .collect();

        // Benchmark naive SIMD method
        let start_naive = Instant::now();
        let _naive_result = vects.iter()
            .flat_map(|v1| vects.iter().map(move |v2| dot_products_simd_naive(v1, v2, THE_PRIME)))
            .collect::<Vec<_>>();
        let duration_naive = start_naive.elapsed();
        println!("Naive SIMD method took: {:?}", duration_naive);

        // Benchmark optimized SIMD method
        let start_simd = Instant::now();
        let _simd_result = vects.iter()
            .flat_map(|v1| vects.iter().map(move |v2| dot_products_simd(v1, v2, THE_PRIME)))
            .collect::<Vec<_>>();
        let duration_simd = start_simd.elapsed();
        println!("Optimized SIMD method took: {:?}", duration_simd);

        // Assert that both results are the same
        assert_eq!(_simd_result, _naive_result, "Results from SIMD and naive SIMD methods do not match.");
    }
    }
}
