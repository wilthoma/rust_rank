
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use rand::Rng;
use std::cmp::Ordering;
use image::{ImageBuffer, Luma};
use std::path::Path;

// use core::simd::{Simd, SimdInt}; // SIMD signed integers

pub type MyInt = i64;

/// Sparse matrix in Compressed Sparse Row (CSR) format
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
        let mut values_t = vec![0; nnz];
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
                    let mut sum: MyInt = 0;
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
                .map(|(v, &val)| (v * val) )
                .fold(0, |acc, x| (acc + x)) % theprime ;
            // let mut sum: MyInt = 0;;
            // for i in start..end {
            //     let col = self.col_indices[i];
            //     sum = (sum + self.values[i] * vector[col]) % theprime;
            // }
            sum
        }).collect()
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

/// Compute the dot product of two vectors modulo `p`
pub fn dot_product_mod_p(vec1: &[MyInt], vec2: &[MyInt], p: MyInt) -> MyInt {
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    vec1.iter()
        .zip(vec2.iter())
        .fold(0, |acc, (&x, &y)| (acc + (x * y) ) % p)
        // .fold(0, |acc, (&x, &y)| (acc + (x * y) % p) % p)

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
    (0..length).map(|_| rng.random_range(0..theprime)).collect()
}
pub fn create_random_vector_nozero(length: usize, theprime: MyInt) -> Vec<MyInt> {
    let mut rng = rand::rng();
    (0..length).map(|_| rng.random_range(1..theprime)).collect()
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
