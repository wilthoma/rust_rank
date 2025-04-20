
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use rand::Rng;
use std::cmp::{Ordering, min};
use image::{ImageBuffer, Luma};
use std::path::Path;

// use core::simd::{Simd, SimdInt}; // SIMD signed integers

pub type MyInt = i64;
// pub type MyInt = f64;

/// Sparse matrix in Compressed Sparse Row (CSR) format
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    pub values: Vec<MyInt>,      // Non-zero values
    pub col_indices: Vec<usize>, // Column indices of values
    pub row_ptr: Vec<usize>,     // Index in `values` where each row starts
    pub n_rows: usize,           // Number of rows
    pub n_cols: usize,           // Number of columns
}

// #[inline]
// fn mod_mul(a: i64, b: i64, p: i64) -> i64 {
//     let result = (a as i128 * b as i128) % p as i128;
//     ((result + p as i128) % p as i128) as i64
// }

fn modinv(a: MyInt, p: MyInt) -> MyInt {
    let (mut a, mut m) = (a.rem_euclid(p), p);
    let (mut x0, mut x1) = (0, 1);
    while a > 1 {
        let q = a / m;
        (a, m) = (m, a % m);
        (x0, x1) = (x1 - q * x0, x0);
    }
    x1.rem_euclid(p)
}

impl CsrMatrix {
    /// Transposes the CSR matrix
    pub fn transpose(&self) -> CsrMatrix {
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
        let mut values_t = vec![0 as MyInt; nnz];
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

    /// Prints matrix in triplet format for debugging
    pub fn print_triplets(&self) {
        println!("(row, col, value):");
        for row in 0..self.n_rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
            for idx in start..end {
                println!("({}, {}, {})", row, self.col_indices[idx], self.values[idx]);
            }
        }
    }

    /// Multiply a sparse CSR matrix with a dense vector in parallel
    pub fn parallel_sparse_matvec_mul2(&self, vector: &[MyInt], theprime: MyInt) -> Vec<MyInt> {
        let matrix = self;
        assert_eq!(matrix.n_cols, vector.len(), "Matrix and vector dimensions must align.");

        // Parallel iterator over rows
        let chunk_size = 512; // Tune this!
        (0..matrix.n_rows)
            .collect::<Vec<_>>() // optional: maybe even better with work-stealing
            .par_chunks(chunk_size)
            .flat_map_iter(|rows| {
                rows.iter().map(|&row| {
                    let start = matrix.row_ptr[row];
                    let end = matrix.row_ptr[row + 1];
                    let mut sum: MyInt = 0 as MyInt;
                    for i in start..end {
                        let col = matrix.col_indices[i];
                        sum = (sum + matrix.values[i] * vector[col]) % theprime;
                    }
                    sum
                })
            })
            .collect()
    }
    
    pub fn parallel_sparse_matvec_mul(&self, vector: &[MyInt], theprime: MyInt) -> Vec<MyInt> {
        assert_eq!(self.n_cols, vector.len(), "Matrix and vector dimensions must align.");
        // const ttheprime :MyInt = 27644437;
        // Parallel iterator over rows
        (0..self.n_rows).into_par_iter().map(|row| {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];


            let colis = self.col_indices[start..end].iter().map(|&col| vector[col]);
            let sum = colis
                .zip(&self.values[start..end])
                // .zip(self.values[start..end].iter())
                //.collect::<Vec<_>>().iter()
                // .map(|(v, val)| (v * *val) )
                // .map(|(v, &val)| (v * val) )
                // .sum::<MyInt>() % theprime ;
                .fold(0, |acc, (v, &val)| (acc + v * val)) % theprime;
                //.map(|(v, &val)| (v * val) )
                //.sum::<MyInt>() % theprime ;
            // % theprime 
                // .fold(0, |acc, x| (acc + x)) % theprime ;
            // let mut sum: MyInt = 0;;
            // for i in start..end {
            //     let col = self.col_indices[i];
            //     sum = (sum + self.values[i] * vector[col]) % theprime;
            // }
            sum
            //tprime.rem_of(sum)
        }).collect()
    }
    pub fn parallel_sparse_matvec_mul_unsafe(&self, vector: &[MyInt], theprime: MyInt) -> Vec<MyInt> {
        assert_eq!(self.n_cols, vector.len(), "Matrix and vector dimensions must align.");
        // Parallel iterator over rows
        (0..self.n_rows).into_par_iter().map(|row| {
            let start = unsafe { *self.row_ptr.get_unchecked(row) };
            let end = unsafe { *self.row_ptr.get_unchecked(row + 1) };

            let colis = (start..end).map(|i| unsafe { *vector.get_unchecked(*self.col_indices.get_unchecked(i)) });
            let sum = colis
                .zip(unsafe { self.values.get_unchecked(start..end) })
                .map(|(v, &val)| (v * val))
                .fold(0, |acc, x| (acc + x)) % theprime;
            sum
        }).collect()
    }

    pub fn serial_sparse_matvec_mul(&self, vector: &[MyInt], theprime: MyInt) -> Vec<MyInt> {
        assert_eq!(self.n_cols, vector.len(), "Matrix and vector dimensions must align.");

        // Parallel iterator over rows
        (0..self.n_rows).into_iter().map(|row| {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            let mut sum: MyInt = 0 as MyInt;
            for i in start..end {
                let col = self.col_indices[i];
                sum +=  self.values[i] * vector[col];
                sum %= theprime;
            }
            sum
        }).collect()
    }

    #[inline]
    pub fn serial_sparse_matvec_mul_chunk(&self, vector: &[MyInt], theprime: MyInt) -> Vec<MyInt> {
        assert_eq!(self.n_cols, vector.len(), "Matrix and vector dimensions must align.");

        // Parallel iterator over rows
        let mut result = vec![0 as MyInt; self.n_rows];
        for row in 0..self.n_rows {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];

            let mut sum: MyInt = 0 as MyInt;
            let mut i = start;
            while i + 4 <= end {
                let col1 = self.col_indices[i];
                let col2 = self.col_indices[i + 1];
                let col3 = self.col_indices[i + 2];
                let col4 = self.col_indices[i + 3];

                let a1 = self.values[i] * vector[col1];
                // sum %= theprime;
                let a2 = self.values[i + 1] * vector[col2];
                // sum %= theprime;
                let a3 = self.values[i + 2] * vector[col3];
                // sum %= theprime;
                let a4= self.values[i + 3] * vector[col4];
                sum += a1 + a2 + a3 + a4;
                sum %= theprime;

                i += 4;
            }

            while i < end {
                let col = self.col_indices[i];
                sum += self.values[i] * vector[col];
                // sum %= theprime;
                i += 1;
            }
            result[row] = sum % theprime;
        }
        result
    }



    pub fn gaussian_elimination_markowitz(&mut self, p: MyInt) {
        let r = min(self.n_rows, self.n_cols);
        let mut nsteps = 0;

        let nnz0 = self.values.len();

        let mut row_nnz = vec![0; self.n_rows];
        let mut col_nnz = vec![0; self.n_cols];
        let mut used_rows = vec![false; self.n_rows];
        let mut used_cols = vec![false; self.n_cols];

        // Precompute row_nnz
        for row in 0..self.n_rows {
            row_nnz[row] = self.row_ptr[row + 1] - self.row_ptr[row];
        }

        // Precompute col_nnz
        for &col in &self.col_indices {
            col_nnz[col] += 1;
        }

        for _ in 0..self.n_rows.min(self.n_cols) {
            println!("Elim: pivoting...");
            let Some(((pivot_row, pivot_col), score)) = self.select_markowitz_pivot(&row_nnz, &col_nnz,&used_rows, &used_cols, p) else {
                continue;
            };
            // println!("Selected pivot at row {}, column {} with score {}", pivot_row, pivot_col, score);

            // check if it is worth to reduce
            nsteps += 1;
            if score * (r-nsteps) > 2* self.values.len()-nsteps  {
                println!("Further Gaussian elimination not worth: {} * {} > {}; {} steps", score, r-nsteps,self.values.len(),  nsteps);
                return;
            }
            
            // used_rows.insert(pivot_row);
            // used_cols.insert(pivot_col);
            used_rows[pivot_row] = true;
            used_cols[pivot_col] = true;

            // Optionally zero out nnz counts to prevent reuse
            row_nnz[pivot_row] = 0;
            col_nnz[pivot_col] = 0;

            println!("Elim: {}, delta: {}", nsteps, self.values.len()-nnz0);

            let pivot_val = self.get(pivot_row, pivot_col, p);
            let inv = modinv(pivot_val, p);
            self.scale_row(pivot_row, inv, p);

            for row in 0..self.n_rows {
                if row == pivot_row {
                    continue;
                }
                let coeff = self.get(row, pivot_col, p);
                if coeff != 0 {
                    self.row_subtract(row, pivot_row, coeff, p);
                }
            }

        }
    }

    fn get(&self, row: usize, col: usize, p: MyInt) -> MyInt {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        for i in start..end {
            if self.col_indices[i] == col {
                return self.values[i].rem_euclid(p);
            }
        }
        0
    }

    fn scale_row(&mut self, row: usize, factor: MyInt, p: MyInt) {
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        for i in start..end {
            self.values[i] = (self.values[i] * factor).rem_euclid(p);
        }
    }

    fn row_subtract(&mut self, target: usize, source: usize, factor: MyInt, p: MyInt) {
        use std::collections::HashMap;
        let mut row_map: HashMap<usize, MyInt> = HashMap::new();

        let tgt_start = self.row_ptr[target];
        let tgt_end = self.row_ptr[target + 1];
        for i in tgt_start..tgt_end {
            row_map.insert(self.col_indices[i], self.values[i]);
        }

        let src_start = self.row_ptr[source];
        let src_end = self.row_ptr[source + 1];
        for i in src_start..src_end {
            let col = self.col_indices[i];
            let val = self.values[i];
            *row_map.entry(col).or_insert(0) =
                (row_map.get(&col).unwrap_or(&0) - factor * val).rem_euclid(p);
        }

        let mut new_cols = Vec::new();
        let mut new_vals = Vec::new();
        for (&col, &val) in &row_map {
            if val != 0 {
                new_cols.push(col);
                new_vals.push(val);
            }
        }

        let mut zipped: Vec<_> = new_cols.into_iter().zip(new_vals).collect();
        zipped.sort_by_key(|&(c, _)| c);

        self.col_indices.splice(tgt_start..tgt_end, zipped.iter().map(|&(c, _)| c));
        self.values.splice(tgt_start..tgt_end, zipped.iter().map(|&(_, v)| v));

        let new_len = zipped.len();
        let old_len = tgt_end - tgt_start;
        let diff = new_len as isize - old_len as isize;
        for i in (target + 1)..=self.n_rows {
            self.row_ptr[i] = (self.row_ptr[i] as isize + diff) as usize;
        }
    }

    fn select_markowitz_pivot(
        &self,
        row_nnz: &[usize],
        col_nnz: &[usize],
        used_rows: &[bool],
        used_cols: &[bool],
        p: MyInt,
    ) -> Option<((usize, usize), usize)> {
        let mut best_score = usize::MAX;
        let mut pivot = None;
    
        for row in 0..self.n_rows {
            if used_rows[row] {
                continue;
            }
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];
    
            for idx in start..end {
                let col = self.col_indices[idx];
                if used_cols[col] {
                    continue;
                }
    
                let val = self.values[idx].rem_euclid(p);
                if val == 0 {
                    continue;
                }
    
                let score = (row_nnz[row].saturating_sub(1)) * (col_nnz[col].saturating_sub(1));
                if score < best_score {
                    best_score = score;
                    pivot = Some((row, col));
                }
            }
        }
    
        pivot.map(|pos| (pos, best_score))
    }

    fn col_nnz(&self, col: usize) -> usize {
        let mut count = 0;
        for i in 0..self.n_rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            for j in start..end {
                if self.col_indices[j] == col && self.values[j] != 0 {
                    count += 1;
                    break;
                }
            }
        }
        count
    }


    // /// SIMD-enhanced CSR matrix-vector multiplication with vector preloading
    // pub fn parallel_csr_matvec_mul_simd_preload<const LANES: usize>(
    //     &self,
    //     vector: &[MyInt],
    //     theprime: MyInt
    // ) -> Vec<MyInt>
    // where
    //     core::simd::LaneCount<LANES>: core::simd::SupportedLaneCount,
    // {
    //     let matrix = self;
    //     assert_eq!(matrix.n_cols, vector.len(), "Dimension mismatch");

       
    //     (0..matrix.n_rows).into_par_iter()
    //     .map(|row| {
    //         let start = matrix.row_ptr[row];
    //         let end = matrix.row_ptr[row + 1];
    //         let nnz = end - start;

    //         // Preload relevant vector values into contiguous buffer
    //         let preload: Vec<i64> = matrix.col_indices[start..end]
    //             .iter()
    //             .map(|&col| vector[col])
    //             .collect();

    //         let mut sum = 0i64;
    //         let mut i = 0;

    //         while i + LANES <= nnz {
    //             let val_simd = Simd::<i64, LANES>::from_slice(&matrix.values[start + i..start + i + LANES]);
    //             let vec_simd = Simd::<i64, LANES>::from_slice(&preload[i..i + LANES]);

    //             // SIMD modular multiplication (cast to i128 for safety)
    //             let prod_simd = (val_simd.cast::<i128>() * vec_simd.cast::<i128>()) % Simd::splat(p as i128);

    //             // Correct negatives: (x % p + p) % p
    //             let corrected = (prod_simd + Simd::splat(p as i128)) % Simd::splat(p as i128);
                
    //             sum = (sum + corrected.reduce_sum() as i64) % p;
    //             i += LANES;
    //         }

    //         // Tail loop for remaining elements
    //         while i < nnz {
    //             sum = (sum + mod_mul(matrix.values[start + i], preload[i], p)) % p;
    //             i += 1;
    //         }

    //         // Ensure sum is in range [0, P-1]
    //         (sum + p) % p
    //     })
    //     .collect()
    // }


    /// Check whether two CSR matrices are the same
    pub fn are_csr_matrices_equal(&self, matrix2: &CsrMatrix) -> bool {
        let matrix1 = self;
        matrix1.n_rows == matrix2.n_rows &&
        matrix1.n_cols == matrix2.n_cols &&
        matrix1.values.len() == matrix2.values.len() &&
        matrix1.row_ptr.len() == matrix2.row_ptr.len() &&
        matrix1.values.iter().zip(&matrix2.values).all(|(&v1, &v2)| v1 == v2) &&
        matrix1.col_indices.iter().zip(&matrix2.col_indices).all(|(&c1, &c2)| c1 == c2) &&
        matrix1.row_ptr.iter().zip(&matrix2.row_ptr).all(|(&r1, &r2)| r1 == r2)
    }

    /// Multiply a CSR matrix column-wise with the entries of a vector
    /// The existing CSR matrix is updated in-place
    pub fn scale_csr_matrix_columns(&mut self, vector: &[MyInt], theprime: MyInt) {
        let matrix = self;
        assert_eq!(matrix.n_cols, vector.len(), "Matrix and vector dimensions must align.");

        for i in 0..matrix.values.len() {
            let col = matrix.col_indices[i];
            matrix.values[i] = (matrix.values[i] * vector[col]) % theprime;
        }
    }
    /// Multiply a CSR matrix row-wise with the entries of a vector
    /// The existing CSR matrix is updated in-place
    pub fn scale_csr_matrix_rows(&mut self, vector: &[MyInt], theprime: MyInt) {
        let matrix = self;
        assert_eq!(matrix.n_rows, vector.len(), "Matrix and vector dimensions must align.");

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



}


#[inline]
pub fn serial_sparse_matvec_mul_chunk2(A:&CsrMatrix, vector: &[MyInt], theprime: MyInt) -> Vec<MyInt> {
    assert_eq!(A.n_cols, vector.len(), "Matrix and vector dimensions must align.");

    // Parallel iterator over rows
    let mut result = vec![0 as MyInt; A.n_rows];
    for row in 0..A.n_rows {
        let start = A.row_ptr[row];
        let end: usize = A.row_ptr[row + 1];

        let mut sum: MyInt = 0 as MyInt;
        let mut i = start;
        while i + 4 <= end {
            let col1 = A.col_indices[i];
            let col2 = A.col_indices[i + 1];
            let col3 = A.col_indices[i + 2];
            let col4 = A.col_indices[i + 3];

            let a1 = A.values[i] * vector[col1];
            // sum %= theprime;
            let a2 = A.values[i + 1] * vector[col2];
            // sum %= theprime;
            let a3 = A.values[i + 2] * vector[col3];
            // sum %= theprime;
            let a4= A.values[i + 3] * vector[col4];
            sum += a1 + a2 + a3 + a4;
            sum %= theprime;

            i += 4;
        }

        while i < end {
            let col = A.col_indices[i];
            sum += A.values[i] * vector[col];
            // sum %= theprime;
            i += 1;
        }
        result[row] = sum % theprime;
    }
    result
}

/// Compute the dot product of two vectors modulo `p`
pub fn dot_product_mod_p(vec1: &[MyInt], vec2: &[MyInt], p: MyInt) -> MyInt {
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    vec1.iter()
        .zip(vec2.iter())
        .fold(0 as MyInt, |acc, (&x, &y)| (acc + (x * y) ) % p)
        // .fold(0, |acc, (&x, &y)| (acc + (x * y) % p) % p)

}

pub fn dot_product_mod_p_parallel(vec1: &[MyInt], vec2: &[MyInt], p: MyInt) -> MyInt {
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    let chunk_size = 100;

    vec1.par_chunks(chunk_size)
        .zip(vec2.par_chunks(chunk_size))
        .map(|(chunk1, chunk2)| {
            chunk1.iter()
                .zip(chunk2.iter())
                .fold(0 as MyInt, |acc, (&x, &y)| acc + (x * y))
                % p
        })
        .reduce(|| 0, |acc, chunk_sum| (acc + chunk_sum) ) % p
}



/// Load a sparse matrix in SMS format from a file
pub fn load_csr_matrix_from_sms(file_path: &str) -> Result<CsrMatrix, Box<dyn std::error::Error>> {

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
            let value: MyInt = parts[2].parse()?;

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


/// Create a random vector of given length with integer entries between 0 and `theprime`
pub fn create_random_vector(length: usize, theprime: MyInt) -> Vec<MyInt> {
    let mut rng = rand::rng();
    (0..length).map(|_| rng.random_range(0..(theprime as i64)) as MyInt).collect()
}
pub fn create_random_vector_nozero(length: usize, theprime: MyInt) -> Vec<MyInt> {
    let mut rng = rand::rng();
    (0..length).map(|_| rng.random_range(1..(theprime as i64)) as MyInt).collect()
}


pub fn reorder_csr_matrix_by_keys(
    matrix: &CsrMatrix,
    row_keys: &[usize],
    col_keys: &[usize],
) -> CsrMatrix {
    assert_eq!(matrix.n_rows, row_keys.len());
    assert_eq!(matrix.n_cols, col_keys.len());

    // Compute row and column permutations based on sorting the keys
    let mut row_order: Vec<usize> = (0..matrix.n_rows).collect();
    let mut col_order: Vec<usize> = (0..matrix.n_cols).collect();

    row_order.sort_by(|&i, &j| row_keys[i].cmp(&row_keys[j]));
    col_order.sort_by(|&i, &j| col_keys[i].cmp(&col_keys[j]));

    // Inverse maps from old index -> new index
    let mut row_inv = vec![0; matrix.n_rows];
    let mut col_inv = vec![0; matrix.n_cols];
    for (new_idx, &old_idx) in row_order.iter().enumerate() {
        row_inv[old_idx] = new_idx;
    }
    for (new_idx, &old_idx) in col_order.iter().enumerate() {
        col_inv[old_idx] = new_idx;
    }

    let mut new_values = Vec::new();
    let mut new_col_indices = Vec::new();
    let mut new_row_ptr = Vec::with_capacity(matrix.n_rows + 1);
    new_row_ptr.push(0);

    // Rebuild matrix using new row and column order
    for &old_row in &row_order {
        let start = matrix.row_ptr[old_row];
        let end = matrix.row_ptr[old_row + 1];
        let mut row_entries: Vec<(usize, MyInt)> = Vec::new();

        for idx in start..end {
            let old_col = matrix.col_indices[idx];
            let new_col = col_inv[old_col];
            row_entries.push((new_col, matrix.values[idx]));
        }

        row_entries.sort_by_key(|&(col, _)| col);

        for (col, val) in row_entries {
            new_col_indices.push(col);
            new_values.push(val);
        }

        new_row_ptr.push(new_values.len());
    }

    CsrMatrix {
        values: new_values,
        col_indices: new_col_indices,
        row_ptr: new_row_ptr,
        n_rows: matrix.n_rows,
        n_cols: matrix.n_cols,
    }
}


pub fn spy_plot(
    matrix: &CsrMatrix,
    filename: &str,
    height: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    // Compute width based on matrix aspect ratio
    let width = ((matrix.n_cols as f32 / matrix.n_rows as f32) * height as f32).ceil() as u32;

    let mut img = ImageBuffer::<Luma<u8>, Vec<u8>>::from_pixel(width, height, Luma([255u8]));

    let row_scale = matrix.n_rows as f32 / height as f32;
    let col_scale = matrix.n_cols as f32 / width as f32;

    for row in 0..matrix.n_rows {
        let start = matrix.row_ptr[row];
        let end = matrix.row_ptr[row + 1];
        for idx in start..end {
            let col = matrix.col_indices[idx];

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
