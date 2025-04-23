
use std::fs::File;
use std::io::{self, BufRead, Write};
use std::fmt::Display;
// use std::process::Output;

use crate::matrices::{prettify_vect, MyInt};

/// This module contains code to read and write .wdm files.
/// These are text files used to store progress in the computation of the Wiedemann sequence, that is,
/// the sequence u^T (A^TA)^k v for k=1,2,3....
/// Structure of a .wdm file
/// 1. First line: m n p N num_v -- with m,n matrix size, p the prime number, Nlen the length of the Wiedemann sequence num_u and num_v the number of columns in the U and V matrix
/// 2. Second line: row_precond -- the row preconditioner
/// 3. Third line: col_precond -- the column preconditioner
/// 5. Next num_v lines: v's -- the v vectors, i.e., columns of the V matrix.
/// 6. Next num_v lines: curv -- the current V matrix (A^TA)^{N} V
/// 7. Next num_v x (num_v+1)/2 lines: seq -- the Wiedemann sequence M_k = V^T(A^TA)^{k} V for k=1,2,3...,N. 


pub fn load_wdm_file_sym(
    wdm_filename: &str,
    row_precond: &mut Vec<MyInt>,
    col_precond: &mut Vec<MyInt>,
    v_list: &mut Vec<Vec<MyInt>>,
    curv_list: &mut Vec<Vec<MyInt>>,
    seq_list: &mut Vec<Vec<MyInt>>,
) -> Result<(MyInt, usize, usize, usize), Box<dyn std::error::Error>> {
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
    for _ in 0..(num_v * (num_v+1)/2) {
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

    Ok((p,m,n, num_v)) // Return the prime as the only return value
}


pub fn save_wdm_file_sym<T> (
    wdm_filename: &str,
    n_rows : usize,
    n_cols : usize,
    theprime: T,
    row_precond: &[T],
    col_precond: &[T],
    v_list: &[Vec<T>],
    curv_list: &[Vec<T>],
    seq_list: &[Vec<T>], 
) -> Result<(), Box<dyn std::error::Error>>
where T: Display + std::ops::Add<Output=T> + Copy + std::ops::Mul<Output=T> + std::ops::AddAssign + std::ops::Rem<Output=T>+From<i32> 
{
    let file = File::create(wdm_filename)?;
    // Use a buffered writer for improved performance
    let mut writer = io::BufWriter::new(file);

    // Write the first line: m n p Nlen num_u num_v
    writeln!(
        writer,
        "{} {} {} {} {}",
        n_rows,
        n_cols,
        theprime,
        seq_list[0].len(),
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

    // Write the v_list
    for vv in v_list {
        let v = prettify_vect(vv, theprime);
        for (i, val) in v.iter().enumerate() {
            if i > 0 {
                write!(writer, " ")?;
            }
            write!(writer, "{}", val)?;
        }
        writeln!(writer)?;
    }

    // Write the curv_list
    for curvv in curv_list {
        let curv = prettify_vect(curvv, theprime);
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
