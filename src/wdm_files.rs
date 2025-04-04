
use std::fs::File;
use std::io::{self, BufRead, Write};

use crate::matrices::{MyInt, CsrMatrix};

/// This module contains code to read and write .wdm files.
/// These are text files used to store progress in the computation of the Wiedemann sequence, that is,
/// the sequence u^T (A^TA)^k v for k=1,2,3....
/// Structure of a .wdm file
/// 1. First line: m n p N  -- with m,n matrix size, p the prime number, Nlen the length of the Wiedemann sequence
/// 2. Second line: row_precond -- the row preconditioner
/// 3. Third line: col_precond -- the column preconditioner
/// 4. Fourth line: u -- the u vector
/// 5. Fifth line: v -- the v vector
/// 6. Sixth line: curv -- the current vector (A^TA)^{N} v
/// 7. Seventh line: seq -- the Wiedemann sequence u^T(A^TA)^{k} v for k=1,2,3...,N

/// Load the state from a WDM file
pub fn load_wdm_file(
    wdm_filename: &str,
    a: &CsrMatrix, // the matrix is only needed to sanity check the dimensions 
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

/// Save the current state to the WDM file
pub fn save_wdm_file(
    wdm_filename: &str,
    a: &CsrMatrix,
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


