
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, BufRead};
use rand::Rng;

pub type MyInt = i64;

/// Sparse matrix in Compressed Sparse Row (CSR) format
pub struct CsrMatrix {
    pub values: Vec<MyInt>,      // Non-zero values
    pub col_indices: Vec<usize>, // Column indices of values
    pub row_ptr: Vec<usize>,     // Index in `values` where each row starts
    pub n_rows: usize,           // Number of rows
    pub n_cols: usize,           // Number of columns
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
    pub fn parallel_sparse_matvec_mul(&self, vector: &[MyInt], theprime: MyInt) -> Vec<MyInt> {
        let matrix = self;
        assert_eq!(matrix.n_cols, vector.len(), "Matrix and vector dimensions must align.");

        // Parallel iterator over rows
        (0..matrix.n_rows).into_par_iter().map(|row| {
            let start = matrix.row_ptr[row];
            let end = matrix.row_ptr[row + 1];

            let mut sum: MyInt = 0;
            for i in start..end {
                let col = matrix.col_indices[i];
                sum = (sum + matrix.values[i] * vector[col]) % theprime;
            }
            sum
        }).collect()
    }
    

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
