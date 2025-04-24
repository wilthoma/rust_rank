#[derive(Clone, Debug)]
pub struct CsrMatrixOld 
{
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

impl CsrMatrixOld {
    /// Transposes the CSR matrix
    pub fn transpose(&self) -> CsrMatrixOld {
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

        CsrMatrixOld {
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


    pub fn parallel_sparse_matvec_mul_simd8(&self, vector: &[MyIntx8], theprime: MyInt) -> Vec<MyIntx8> {
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
                .fold(MyIntx8::splat(0), |acc, (v, &val)| (acc + v * Simd::splat(val))) % Simd::splat(theprime);
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
        (0..self.n_rows).into_iter().map(|row| {
            let start = self.row_ptr[row];
            let end = self.row_ptr[row + 1];


            let colis = self.col_indices[start..end].iter().map(|&col| vector[col]);
            let sum = colis
                .zip(&self.values[start..end])
                .fold(0, |acc, (v, &val)| (acc + v * val)) % theprime;
            sum
        }).collect()
    }

    pub fn serial_sparse_matvec_mul2(&self, vector: &[MyInt], theprime: MyInt) -> Vec<MyInt> {
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
    pub fn are_csr_matrices_equal(&self, matrix2: &CsrMatrixOld) -> bool {
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

    pub fn is_prime_valid(&self, theprime: MyInt) -> bool {
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

        let prod = (theprime as i128) * (theprime as i128);
        let max_i64_value = std::i64::MAX as i128;
        // check if prod * nnz <  std::i128::MAX for all entries
        (prod * (DOT_PRODUCT_CHUNK_SIZE as i128) < max_i64_value) && // chunk_size 100 for dot products assumed
        row_nnz.iter().all( |&nnz| {
            (nnz as i128) * prod < max_i64_value
        }) && 
        col_nnz.iter().all( |&nnz| {
            (nnz as i128) * prod < max_i64_value
        })
    }

}

#[inline]
pub fn serial_sparse_matvec_mul_chunk2(A:&CsrMatrixOld, vector: &[MyInt], theprime: MyInt) -> Vec<MyInt> {
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

pub fn dot_product_mod_p_parallel_old(vec1: &[MyInt], vec2: &[MyInt], p: MyInt) -> MyInt {
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    let chunk_size = DOT_PRODUCT_CHUNK_SIZE;

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

pub fn dot_product_mod_p_serial_old(vec1: &[MyInt], vec2: &[MyInt], p: MyInt) -> MyInt {
    assert_eq!(vec1.len(), vec2.len(), "Vectors must have the same length.");

    let chunk_size = DOT_PRODUCT_CHUNK_SIZE;

    vec1.chunks(chunk_size)
        .zip(vec2.chunks(chunk_size))
        .map(|(chunk1, chunk2)| {
            chunk1.iter()
                .zip(chunk2.iter())
                .fold(0 as MyInt, |acc, (&x, &y)| acc + (x * y))
                % p
        })
        .sum::<MyInt>() % p
}


/// Load a sparse matrix in SMS format from a file
pub fn load_csr_matrix_from_sms(file_path: &str) -> Result<CsrMatrixOld, Box<dyn std::error::Error>> {

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

    Ok(CsrMatrixOld {
        values,
        col_indices,
        row_ptr,
        n_rows,
        n_cols,
    })
}



/// Create a random vector of given length with integer entries between 0 and `theprime`
pub fn create_random_vector_old(length: usize, theprime: MyInt) -> Vec<MyInt> {
    let mut rng = rand::rng();
    (0..length).map(|_| rng.random_range(0..(theprime as i64)) as MyInt).collect()
}
pub fn create_random_vector_nozero_old(length: usize, theprime: MyInt) -> Vec<MyInt> {
    let mut rng = rand::rng();
    (0..length).map(|_| rng.random_range(1..(theprime as i64)) as MyInt).collect()
}

// pub fn create_random_vector_simd(length: usize, theprime: MyInt) -> Vec<MyIntx4> {
//     let x1=create_random_vector(length, theprime);
//     let x2=create_random_vector(length, theprime);
//     let x3=create_random_vector(length, theprime);
//     let x0=create_random_vector(length, theprime);
//     x0.iter().zip(&x1).zip(&x2).zip(&x3)
//     .map(|(((a, b), c), d)| Simd::from_array([*a, *b, *c, *d]))
//     .collect()
// }
pub fn create_random_vector_simd8(length: usize, theprime: MyInt) -> Vec<MyIntx8> {
    let x1=create_random_vector_old(length, theprime);
    let x2=create_random_vector_old(length, theprime);
    let x3=create_random_vector_old(length, theprime);
    let x0=create_random_vector_old(length, theprime);
    let x4=create_random_vector_old(length, theprime);
    let x5=create_random_vector_old(length, theprime);
    let x6=create_random_vector_old(length, theprime);
    let x7=create_random_vector_old(length, theprime);
    x0.iter().zip(&x1).zip(&x2).zip(&x3).zip(&x4).zip(&x5).zip(&x6).zip(&x7)
    .map(|(((((((a, b), c), d),e),f),g),h)| Simd::from_array([*a, *b, *c, *d, *e, *f, *g, *h]))
    .collect()
}

pub fn reorder_csr_matrix_by_keys(
    matrix: &CsrMatrixOld,
    row_keys: &[usize],
    col_keys: &[usize],
) -> CsrMatrixOld {
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

    CsrMatrixOld {
        values: new_values,
        col_indices: new_col_indices,
        row_ptr: new_row_ptr,
        n_rows: matrix.n_rows,
        n_cols: matrix.n_cols,
    }
}


pub fn csr2_from_csrmatrix<T>(matrix: &CsrMatrixOld) -> CsrMatrix<T>
where 
T: SimdElement
+ Copy
+ Rem<Output = T>
+ Send
+ Sync
+From<i32>,
{
    CsrMatrix {
        values: matrix.values.iter().map(|&v| T::from(v as i32)).collect(),
        col_indices: matrix.col_indices.clone(),
        row_ptr: matrix.row_ptr.clone(),
        n_rows: matrix.n_rows,
        n_cols: matrix.n_cols,
    }
}




// wdm routines

/// Load the state from a WDM file
pub fn load_wdm_file(
    wdm_filename: &str,
    a: &CsrMatrixOld, // the matrix is only needed to sanity check the dimensions 
    row_precond: &mut Vec<MyInt>,
    col_precond: &mut Vec<MyInt>,
    u: &mut Vec<MyInt>,
    v: &mut Vec<MyInt>,
    curv: &mut Vec<MyInt>,
    seq: &mut Vec<MyInt>,
) -> Result<MyInt, Box<dyn std::error::Error>> {
    let file = File::open(wdm_filename)?;
    let mut reader = io::BufReader::new(file);

    // Read the first line: m n p Nlen
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let m: usize = parts.next().ok_or("Missing m")?.parse()?;
    let n: usize = parts.next().ok_or("Missing n")?.parse()?;
    let p: MyInt = parts.next().ok_or("Missing p")?.parse()?;
    let nlen: usize = parts.next().ok_or("Missing Nlen")?.parse()?;

    // Check if matrix dimensions match
    if m != a.n_rows || n != a.n_cols {
        return Err("Matrix dimensions do not match with content of WDM file".into());
    }

    // Read row_precond
    line.clear();
    reader.read_line(&mut line)?;
    *row_precond = line
        .split_whitespace()
        .map(|x| x.parse::<MyInt>())
        .collect::<Result<Vec<_>, _>>()?;

    // Read col_precond
    line.clear();
    reader.read_line(&mut line)?;
    *col_precond = line
        .split_whitespace()
        .map(|x| x.parse::<MyInt>())
        .collect::<Result<Vec<_>, _>>()?;

    // Read u
    line.clear();
    reader.read_line(&mut line)?;
    *u = line
        .split_whitespace()
        .map(|x| x.parse::<MyInt>())
        .collect::<Result<Vec<_>, _>>()?;

    // Read v
    line.clear();
    reader.read_line(&mut line)?;
    *v = line
        .split_whitespace()
        .map(|x| x.parse::<MyInt>())
        .collect::<Result<Vec<_>, _>>()?;

    // Read curv
    line.clear();
    reader.read_line(&mut line)?;
    *curv = line
        .split_whitespace()
        .map(|x| x.parse::<MyInt>())
        .collect::<Result<Vec<_>, _>>()?;

    // Read seq
    line.clear();
    reader.read_line(&mut line)?;
    *seq = line
        .split_whitespace()
        .take(nlen)
        .map(|x| x.parse::<MyInt>())
        .collect::<Result<Vec<_>, _>>()?;

    // Ensure all vectors are of the correct size   
    if row_precond.len() != a.n_rows {
        return Err("Row preconditioner length does not match matrix rows".into());
    }
    if col_precond.len() != a.n_cols {
        return Err("Column preconditioner length does not match matrix columns".into());
    }
    if u.len() != a.n_cols {
        return Err("u vector length does not match matrix columns".into());
    }
    if v.len() != a.n_cols {
        return Err("v vector length does not match matrix columns".into());
    }
    if curv.len() != a.n_cols {
        return Err("curv vector length does not match matrix columns".into());
    }
    if seq.len() != nlen {
        return Err("seq vector length does not match expected sequence length".into());
    }

    Ok(p) // Return the prime as the only return value
}



pub fn load_wdm_file2(
    wdm_filename: &str,
    row_precond: &mut Vec<MyInt>,
    col_precond: &mut Vec<MyInt>,
    u_list: &mut Vec<Vec<MyInt>>,
    v_list: &mut Vec<Vec<MyInt>>,
    curv_list: &mut Vec<Vec<MyInt>>,
    seq_list: &mut Vec<Vec<MyInt>>,
) -> Result<(MyInt, usize, usize, usize, usize), Box<dyn std::error::Error>> {
    let file = File::open(wdm_filename)?;
    let mut reader = io::BufReader::new(file);

    // Read the first line: m n p Nlen num_u num_v
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let m: usize = parts.next().ok_or("Missing m")?.parse()?;
    let n: usize = parts.next().ok_or("Missing n")?.parse()?;
    let p: MyInt = parts.next().ok_or("Missing p")?.parse()?;
    let nlen: usize = parts.next().ok_or("Missing Nlen")?.parse()?;
    let num_u: usize = parts.next().ok_or("Missing num_u")?.parse()?;
    let num_v: usize = parts.next().ok_or("Missing num_v")?.parse()?;

    // Check if matrix dimensions match
    // if m != a.n_rows || n != a.n_cols {
    //     return Err("Matrix dimensions do not match with content of WDM file".into());
    // }

    // Read row_precond
    line.clear();
    reader.read_line(&mut line)?;
    *row_precond = line
        .split_whitespace()
        .map(|x| x.parse::<MyInt>())
        .collect::<Result<Vec<_>, _>>()?;

    // Read col_precond
    line.clear();
    reader.read_line(&mut line)?;
    *col_precond = line
        .split_whitespace()
        .map(|x| x.parse::<MyInt>())
        .collect::<Result<Vec<_>, _>>()?;

    // Read u_list
    u_list.clear();
    for _ in 0..num_u {
        line.clear();
        reader.read_line(&mut line)?;
        let u = line
            .split_whitespace()
            .map(|x| x.parse::<MyInt>())
            .collect::<Result<Vec<_>, _>>()?;
        if u.len() != n {
            return Err("u vector length does not match matrix columns".into());
        }
        u_list.push(u);
    }

    // Read v_list
    v_list.clear();
    for _ in 0..num_v {
        line.clear();
        reader.read_line(&mut line)?;
        let v = line
            .split_whitespace()
            .map(|x| x.parse::<MyInt>())
            .collect::<Result<Vec<_>, _>>()?;
        if v.len() != n {
            return Err("v vector length does not match matrix columns".into());
        }
        v_list.push(v);
    }

    // Read curv_list
    curv_list.clear();
    for _ in 0..num_v {
        line.clear();
        reader.read_line(&mut line)?;
        let curv = line
            .split_whitespace()
            .map(|x| x.parse::<MyInt>())
            .collect::<Result<Vec<_>, _>>()?;
        if curv.len() != n {
            return Err("curv vector length does not match matrix columns".into());
        }
        curv_list.push(curv);
    }

    // Read seq_list
    seq_list.clear();
    for _ in 0..(num_u * num_v) {
        line.clear();
        reader.read_line(&mut line)?;
        let seq = line
            .split_whitespace()
            .take(nlen)
            .map(|x| x.parse::<MyInt>())
            .collect::<Result<Vec<_>, _>>()?;
        if seq.len() != nlen {
            return Err("seq vector length does not match expected sequence length".into());
        }
        seq_list.push(seq);
    }

    // Ensure all vectors are of the correct size   
    if row_precond.len() != m {
        return Err("Row preconditioner length does not match matrix rows".into());
    }
    if col_precond.len() != n {
        return Err("Column preconditioner length does not match matrix columns".into());
    }

    Ok((p,m,n,num_u, num_v)) // Return the prime as the only return value
}


/// Save the current state to the WDM file
pub fn save_wdm_file(
    wdm_filename: &str,
    a: &CsrMatrixOld,
    theprime: MyInt,
    row_precond: &[MyInt],
    col_precond: &[MyInt],
    u: &[MyInt],
    v: &[MyInt],
    curv: &[MyInt],
    seq: &[MyInt],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(wdm_filename)?;
    // Use a buffered writer for improved performance
    let mut writer = io::BufWriter::new(file);

    // Write the first line: m n p Nlen
    writeln!(writer, "{} {} {} {}", a.n_rows, a.n_cols, theprime, seq.len())?;

    // Write the second line: row_precond
    for (i, val) in row_precond.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;

    // Write the third line: col_precond
    for (i, val) in col_precond.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;

    // Write the fourth line: u
    for (i, val) in u.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;

    // Write the fifth line: v
    for (i, val) in v.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;

    // Write the sixth line: curv
    for (i, val) in curv.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;

    // Write the seventh line: seq
    for (i, val) in seq.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;

    writer.flush()?; // Ensure all data is written to the file

    Ok(())
}


pub fn save_wdm_file2(
    wdm_filename: &str,
    a: &CsrMatrixOld,
    theprime: MyInt,
    row_precond: &[MyInt],
    col_precond: &[MyInt],
    u_list: &[Vec<MyInt>],
    v_list: &[Vec<MyInt>],
    curv_list: &[Vec<MyInt>],
    seq_list: &[Vec<MyInt>],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(wdm_filename)?;
    // Use a buffered writer for improved performance
    let mut writer = io::BufWriter::new(file);

    // Write the first line: m n p Nlen num_u num_v
    writeln!(
        writer,
        "{} {} {} {} {} {}",
        a.n_rows,
        a.n_cols,
        theprime,
        seq_list[0].len(),
        u_list.len(),
        v_list.len()
    )?;

    // Write the second line: row_precond
    for (i, val) in row_precond.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;

    // Write the third line: col_precond
    for (i, val) in col_precond.iter().enumerate() {
        if i > 0 {
            write!(writer, " ")?;
        }
        write!(writer, "{}", val)?;
    }
    writeln!(writer)?;

    // Write the u_list
    for u in u_list {
        for (i, val) in u.iter().enumerate() {
            if i > 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{}", val)?;
        }
        writeln!(writer)?;
    }

    // Write the v_list
    for v in v_list {
        for (i, val) in v.iter().enumerate() {
            if i > 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{}", val)?;
        }
        writeln!(writer)?;
    }

    // Write the curv_list
    for curv in curv_list {
        for (i, val) in curv.iter().enumerate() {
            if i > 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{}", val)?;
        }
        writeln!(writer)?;
    }

    // Write the seq_list
    for seq in seq_list {
        for (i, val) in seq.iter().enumerate() {
            if i > 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{}", val)?;
        }
        writeln!(writer)?;
    }

    writer.flush()?; // Ensure all data is written to the file

    Ok(())
}




pub fn main_loop_s_mt<T>(
    a: &Arc<CsrMatrix<T>>,
    at: &Arc<CsrMatrix<T>>,
    row_precond: &[T],
    col_precond: &[T],
    curv: &mut Vec<Vec<T>>,
    v: &Vec<Vec<T>>,
    seq: &mut Vec<Vec<T>>,
    max_nlen: usize,
    wdm_filename: &str,
    theprime: T,
    save_after: usize,
    use_matvmul_parallel: bool,
    use_vecp_parallel: bool,
    deep_clone_matrix: bool,

) -> Result<(), Box<dyn std::error::Error>> 
where 
    T: Zero + SampleUniform+ PartialOrd+ Display + SimdElement + AddAssign + Sum<T> + Copy + Mul<Output = T> + Add<Output = T> + Rem<Output = T> + Send + Sync + 'static,
{

    let start = std::time::Instant::now();
    let mut last_save = start;
    let mut last_report = start;
    let mut last_nlen = seq[0].len();
    let mut curv_result  = curv.clone();
    let to_be_produced = max_nlen - seq[0].len(); 
    let num_v = curv.len();

    //let mut tmpv = vec![0; a.n_rows];
    // let num_v = v.len();
    let buffer_capacity = 10;
    let (txs, rxs) : (Vec<_>, Vec<_>) = 
        (0..num_v).map(|_| crossbeam_channel::bounded(buffer_capacity)).unzip();

    // Update the worker threads to send only one vector
    let _workers: Vec<_> = txs.into_iter().enumerate().map(|(worker_id, tx)| {
        let local_curv = curv[worker_id].clone();
        let a = if deep_clone_matrix { Arc::new(CsrMatrix::clone(&a)) } else { std::sync::Arc::clone(a) };
        let at = if deep_clone_matrix { Arc::new(CsrMatrix::clone(&at)) } else { std::sync::Arc::clone(at) };
        thread::spawn(move || {

            if use_matvmul_parallel {
                // we store curw=A^t (A^tA)^i v for the next iteration since ownership of (A^tA)^i v is lost by sending over the channel
                let mut curw = a.parallel_sparse_matvec_mul(&local_curv, theprime);

                for _ in 0..(to_be_produced/2) {
                    let vec = at.parallel_sparse_matvec_mul(&curw, theprime);
                    curw = a.parallel_sparse_matvec_mul(&vec, theprime);
                    if tx.send(vec).is_err(){
                        eprintln!("Error sending token from worker {}", worker_id);
                        return;
                    };
                }
             } else { 
                // same code, but version with serial matvecmul
                let mut curw = a.serial_sparse_matvec_mul(&local_curv, theprime);

                for _ in 0..(to_be_produced/2) {
                    let vec = at.serial_sparse_matvec_mul(&curw, theprime);
                    curw = a.serial_sparse_matvec_mul(&vec, theprime);
                    if tx.send(vec).is_err(){
                        eprintln!("Error sending token from worker {}", worker_id);
                        return;
                    };
                }
            }
            // println!("Worker {} done", worker_id);
        })
    }).collect();
    
    for _ in 0..to_be_produced/2 {
        let mut received_tokens = Vec::new();

        // First, read one token from each channel
        for rx in &rxs {
            match rx.recv() {
            Ok(vec) => {
                received_tokens.push(vec);
            }
            Err(e) => {
                eprintln!("Error receiving token: {}", e);
                return Err(Box::new(e));
            }
            }
        }
        // Now, process the received tokens
        // let start_time = std::time::Instant::now();
        let mut ii = 0;
        for i in 0..num_v {
            for j in i..num_v {
                let vec1 = &received_tokens[i];
                let vec1_prev = &curv_result[i];
                let vec2 = &received_tokens[j];

                if use_vecp_parallel {
                    seq[ii].push(dot_product_mod_p_parallel(vec1_prev, vec2, theprime));
                    seq[ii].push(dot_product_mod_p_parallel(vec1, vec2, theprime));
                }
                else {
                    seq[ii].push(dot_product_mod_p_serial(vec1_prev, vec2, theprime));
                    seq[ii].push(dot_product_mod_p_serial(vec1, vec2, theprime));
                }
                ii += 1;
            }
        }

        // println!("Time taken for dot product computations: {:?}\n", start_time.elapsed());

        // store current vectors in curv
        curv_result.clone_from_slice(&received_tokens);

        // std::thread::sleep(std::time::Duration::from_millis(1500));

        if last_save.elapsed().as_secs_f64() > save_after as f64 {
            println!("\nSaving...");
            let save_start = std::time::Instant::now();
            last_save = std::time::Instant::now();
            match save_wdm_file_sym(&wdm_filename, a.n_rows, a.n_cols, theprime, &row_precond, &col_precond, &v, &curv_result, &seq) {
                Ok(_) => {
                    println!("\nSaved state to file {} at sequence length {} (saving took {:?}).\n", wdm_filename, seq[0].len(), save_start.elapsed());
                }
                Err(e) => {
                    eprintln!("Error saving state to file {}: {}", wdm_filename, e);
                    //return Err(Box::new(e));
                }
            }
            // std::thread::sleep(std::time::Duration::from_millis(500));
            
            // println!("\nSaved state to file {} at sequence length {} (saving took {:?}).\n", wdm_filename, seq[0].len(), save_start.elapsed());
        }

        if last_report.elapsed().as_secs_f64() > REPORT_AFTER {
            // println!("Report");
            let fill_status: String = rxs.iter()
                    .map(|rx| format!("{} ", rx.len()))
                    .collect();
            report_progress(start, last_report, last_nlen, seq[0].len(), max_nlen, &format!("Channel fill status {}", fill_status));
            last_report = std::time::Instant::now();
            last_nlen = seq[0].len();
        }
    }

    *curv = curv_result;
    Ok(())

}