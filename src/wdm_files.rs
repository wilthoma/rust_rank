
use std::fs::File;
use std::io::{self, BufRead, Write};

use crate::matrices::{MyInt, CsrMatrix};

/// This module contains code to read and write .wdm files.
/// These are text files used to store progress in the computation of the Wiedemann sequence, that is,
/// the sequence u^T (A^TA)^k v for k=1,2,3....
/// Structure of a .wdm file
/// 1. First line: m n p N  num_u num_v -- with m,n matrix size, p the prime number, Nlen the length of the Wiedemann sequence num_u and num_v the number of columns in the U and V matrix
/// 2. Second line: row_precond -- the row preconditioner
/// 3. Third line: col_precond -- the column preconditioner
/// 4. Line 4 - 4+num_u-1: u's -- the u vectors, i.e., columns of the U matrix.
/// 5. Next num_v lines: v's -- the v vectors, i.e., columns of the V matrix.
/// 6. Next num_v lines: curv -- the current V matrix (A^TA)^{N} V
/// 7. Next num_u x num_v lines: seq -- the Wiedemann sequence M_k = U^T(A^TA)^{k} V for k=1,2,3...,N. The line i+j*num_u contains the elements M_k[i,j] for k=1,2,....

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


pub fn save_wdm_file2(
    wdm_filename: &str,
    a: &CsrMatrix,
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
