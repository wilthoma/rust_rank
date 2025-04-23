


// use petgraph::csr::Csr;
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use rand::Rng;
// use std::cmp::{Ordering, min};
use std::iter::Sum;
use image::{ImageBuffer, Luma};
use std::path::Path;
use std::time::Instant;
use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use std::ops::{Add, AddAssign, Mul, Rem};
// use num_traits::Bounded;


// use std::alloc::{alloc_zeroed, Layout};
// use std::ptr::NonNull;

// use core::simd::{Simd, SimdInt}; // SIMD signed integers

const DOT_PRODUCT_CHUNK_SIZE: usize = 100;

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


pub fn prettify_vect<T>(v: &[T], theprime: u32) -> Vec<T> 
where T:Add<Output = T> + Rem<Output = T> + Copy + From<u32>,{
    let theprime= T::from(theprime);
    v.iter().map(|&x| (x + theprime) % theprime).collect()
}


pub fn dot_product_mod_p_parallel<T>(vec1: &[T], vec2: &[T], theprime: T) -> T 
where T: 
Add<Output = T> +Mul<Output = T> + Copy + Rem<Output = T> + AddAssign + Send + Sync + From<u32>,{
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    let chunk_size = DOT_PRODUCT_CHUNK_SIZE;

    vec1.par_chunks(chunk_size)
        .zip(vec2.par_chunks(chunk_size))
        .map(|(chunk1, chunk2)| {
            chunk1.iter()
                .zip(chunk2.iter())
                .fold(T::from(0), |acc, (&x, &y)| acc + (x * y))
                % theprime
        })
        .reduce(|| T::from(0), |acc, chunk_sum| (acc + chunk_sum) ) % theprime
}

pub fn dot_product_mod_p_serial<T>(vec1: &[T], vec2: &[T], theprime: T) -> T
where T: 
Copy + Rem<Output = T> + AddAssign + From<u32> + Sum<T> + Mul<Output = T> + Add<Output = T>,{
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    let chunk_size = DOT_PRODUCT_CHUNK_SIZE;

    vec1.chunks(chunk_size)
        .zip(vec2.chunks(chunk_size))
        .map(|(chunk1, chunk2)| {
            chunk1.iter()
                .zip(chunk2.iter())
                .fold(T::from(0), |acc, (&x, &y)| acc + (x * y))
                % theprime
        })
        .sum::<T>() % theprime
}

pub fn create_random_vector<T>(length: usize, theprime: u32) -> Vec<T> 
where T : From<u32>, 
{
    let mut rng = rand::rng();
    (0..length).map(|_| T::from( rng.random_range(0..theprime)) ).collect()
}

pub fn create_random_vector_nozero<T>(length: usize, theprime: u32) -> Vec<T> 
where T : From<u32>, 
{
    let mut rng = rand::rng();
    (0..length).map(|_| T::from( rng.random_range(1..theprime)) ).collect()
}

pub fn create_random_vector_simd<T, const LANES: usize>(
    length: usize,
    theprime: u32,
    // mut scalar_gen: impl FnMut(usize, T) -> Vec<T>,
) -> Vec<Simd<T, LANES>>
where
    T: SimdElement + Copy+ From<u32>,
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
+ From<u32>
+ Rem<Output = T>
+ Send
+ Sync
+ Add<Output = T>
+ AddAssign
+ Mul<Output = T>,
{
    pub fn parallel_sparse_matvec_mul_simd2<const LANES: usize>(
        &self,
        vector: &[Simd<T, LANES>],
        theprime: T,
    ) -> Vec<Simd<T, LANES>>
    where
        LaneCount<LANES>: SupportedLaneCount,
        Simd<T, LANES>: Copy
            + Send
            + Sync
            + Add<Output = Simd<T, LANES>>
            + AddAssign
            + Mul<Output = Simd<T, LANES>>
            + Rem<Output = Simd<T, LANES>>,
        T: Copy + From<u32>,
    {
        assert_eq!(self.n_cols, vector.len(), "Matrix and vector dimensions must align.");
    
        let mut result = vec![Simd::splat(T::from(0)); self.n_rows];
    
        result
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, out)| {
                let start = self.row_ptr[row];
                let end = self.row_ptr[row + 1];
    
                let sum = self.col_indices[start..end]
                    .iter()
                    .map(|&col| vector[col])
                    .zip(&self.values[start..end])
                    .fold(Simd::splat(T::from(0)), |acc, (v, &val)| {
                        acc + v * Simd::splat(val)
                    });
    
                *out = sum % Simd::splat(theprime);
            });
    
        result
    }



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
                .fold(Simd::splat(T::from(0)), |acc, (v, &val)| (acc + v * Simd::splat(val)) ) ;
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
                .fold(Simd::splat(T::from(0)), |acc, (v, &val)| (acc + v * Simd::splat(val)) ) ;
            sum % Simd::splat(theprime)
        }).collect()
    }

    pub fn load_csr_matrix_from_sms(file_path: &str, theprime : u32) -> Result<CsrMatrix<T>, Box<dyn std::error::Error>> {
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
                let value: T = T::from(intvalue as u32);

    
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
        let mut values_t = vec![T::from(0); nnz];
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
                .fold(T::from(0), |acc, (v, &val)| (acc + v * val)) % theprime;
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
                .fold(T::from(0), |acc, (v, &val)| (acc + v * val)) % theprime;
            sum
        }).collect()
    }

    /// Multiply a CSR matrix column-wise with the entries of a vector
    /// The existing CSR matrix is updated in-place
    pub fn scale_csr_matrix_columns(&mut self, vector: &[T], theprime: u32) {
        let matrix = self;
        assert_eq!(matrix.n_cols, vector.len(), "Matrix and vector dimensions must align.");
        let theprime = T::from(theprime);
        for i in 0..matrix.values.len() {
            let col = matrix.col_indices[i];
            matrix.values[i] = (matrix.values[i] * vector[col]) % theprime;
        }
    }
    /// Multiply a CSR matrix row-wise with the entries of a vector
    /// The existing CSR matrix is updated in-place
    pub fn scale_csr_matrix_rows(&mut self, vector: &[T], theprime: u32) {
        let matrix = self;
        assert_eq!(matrix.n_rows, vector.len(), "Matrix and vector dimensions must align.");
        let theprime = T::from(theprime);

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

    pub fn normal_simd_speedtest(&self, theprime : u32, n_reps: usize) 
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
            let _result = self.parallel_sparse_matvec_mul(&nvector, T::from(theprime));
            let duration = start.elapsed();
            println!("Normal multiplication took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd(&svector4, T::from(theprime));
            let duration = start.elapsed();
            println!("SIMD4 multiplication took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd2(&svector4, T::from(theprime));
            let duration = start.elapsed();
            println!("SIMD4 multiplication2 took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd(&svector8, T::from(theprime));
            let duration = start.elapsed();
            println!("SIMD8 multiplication took: {:?}", duration);
            let start = Instant::now();
            let _result = self.parallel_sparse_matvec_mul_simd2(&svector8, T::from(theprime));
            let duration = start.elapsed();
            println!("SIMD8 multiplication2 took: {:?}", duration);
        }
    }

    pub fn normal_simd_speedtest_serial(&self, theprime : u32, n_reps: usize) 
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
            let _result = self.serial_sparse_matvec_mul(&nvector, T::from(theprime));
            let duration = start.elapsed();
            println!("Normal multiplication (serial) took: {:?}", duration);
            let start = Instant::now();
            let _result = self.serial_sparse_matvec_mul_simd(&svector4, T::from(theprime));
            let duration = start.elapsed();
            println!("SIMD4 multiplication (serial) took: {:?}", duration);
            let start = Instant::now();
            let _result = self.serial_sparse_matvec_mul_simd(&svector8, T::from(theprime));
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
}