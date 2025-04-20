
// #![feature(portable_simd)]

mod matrices;
mod graphs;
mod wdm_files;
mod block_berlekamp_massey;
use block_berlekamp_massey::block_berlekamp_massey;
use matrices::*; //{create_random_vector, create_random_vector_nozero, load_csr_matrix_from_sms, reorder_csr_matrix_by_keys, spy_plot, CsrMatrix, MyInt};
use wdm_files::{save_wdm_file2, load_wdm_file2};
use graphs::count_triangles_in_file;
use std::io::Write;
use clap::{Arg, Command};
use std::cmp::min;
use rayon::prelude::*;

// type MyInt = i64;

const THEPRIME: MyInt = 27644437 as MyInt; // A large prime number for modular arithmetic

const REPORT_AFTER: f64 = 1.0; // seconds

fn reorder_matrix(a: &CsrMatrix, rowfilename: &str, colfilename: &str) -> CsrMatrix {
    // plot original sparsity patter
    let start_time = std::time::Instant::now();
    _ = spy_plot(a, "data/sparsity_pattern0.png", 800).unwrap();
    println!("Time taken for spy_plot: {:?}", start_time.elapsed());

    let start_time = std::time::Instant::now();
    let v1 = count_triangles_in_file(rowfilename).unwrap();
    println!("Time taken for count_triangles_in_file (gra12_9.g6): {:?}", start_time.elapsed());

    let start_time = std::time::Instant::now();
    let v2 = count_triangles_in_file(colfilename).unwrap();
    println!("Time taken for count_triangles_in_file (gra11_9.g6): {:?}", start_time.elapsed());

    let start_time = std::time::Instant::now();
    let b = reorder_csr_matrix_by_keys(&a, &v1, &v2);
    println!("Time taken for reorder_csr_matrix_by_keys: {:?}", start_time.elapsed());

    spy_plot(&b, "data/sparsity_pattern1.png", 800);

    b
}


pub fn report_progress(
    start_time: std::time::Instant,
    last_time: std::time::Instant,
    last_nlen: usize,
    nlen: usize,
    max_nlen: usize,
) {
    let elapsed = start_time.elapsed();
    let elapsed_last = last_time.elapsed();
    let speed = (nlen - last_nlen) as f64 / elapsed_last.as_secs_f64();
    let remaining = (max_nlen - nlen) as f64 / speed;
    print!(
        "\rProgress: {}/{} | Elapsed: {:?} | Throughput: {:.2}/s | Remaining: {:?}            ",
        nlen,
        max_nlen,
        elapsed,
        speed,
        std::time::Duration::from_secs_f64(remaining)
    );
    std::io::stdout().flush().unwrap();
}


pub fn main_loop_s(
    a: &CsrMatrix,
    at: &CsrMatrix,
    row_precond: &[MyInt],
    col_precond: &[MyInt],
    curv: &mut Vec<Vec<MyInt>>,
    v: &Vec<Vec<MyInt>>,
    seq: &mut Vec<Vec<MyInt>>,
    max_nlen: usize,
    wdm_filename: &str,
    theprime: MyInt,
    save_after: usize,
) -> Result<(), Box<dyn std::error::Error>> {

    let start = std::time::Instant::now();
    let mut last_save = start;
    let mut last_report = start;
    let mut last_nlen = seq[0].len();
    let mut curv_result  = curv.clone();
    //let mut tmpv = vec![0; a.n_rows];
    let num_v = v.len();

    const batch_size: usize = 8;
    let nbatches = (max_nlen + batch_size -1) / batch_size;

    for _ in seq[0].len()..nbatches {
        // Multiply the matrix A with the vector curv
        // println!(".");
        // let tmpv_result = a.parallel_sparse_matvec_mul_optimized(curv, theprime);
        // let tmpv_result = a.parallel_csr_matvec_mul_simd_preload::<8>(curv, theprime);
        // let tmpv_result = a.parallel_sparse_matvec_mul(curv, theprime);
        let buffer = curv_result.par_iter().map(|tcurv| {
            
            // produce a batch of vectors by repeated application of a and at

            // let tmpv_result = curv_result.par_iter().map(|tcurv| {
            // a.serial_sparse_matvec_mul(tcurv, theprime)
            // a.serial_sparse_matvec_mul_chunk(tcurv, theprime)
            // serial_sparse_matvec_mul_chunk2(&a,tcurv, theprime)
            // a.parallel_sparse_matvec_mul_unsafe(tcurv, theprime)
            let mut v2 = tcurv;
            (0..batch_size)
                .map(|_| {
                    let vec1 = a.parallel_sparse_matvec_mul(v2, theprime);
                    let vec2 =  at.parallel_sparse_matvec_mul(vec1, theprime);
                    v2 = &vec2;
                    (vec1, vec2)
                })
                .collect::<Vec<_>>()       

        }).collect::<Vec<_>>();

        // compute dot products
        for t in 0..batch_size {
            for ii in 0..num_v*num_v {
                let i = ii % num_v;
                let j = ii / num_v;
                // curv_result[i][j] = (curv_result[i][j] + u[i][j] * v[j][i]) % theprime;
                let (vec1a, vec1b) = &buffer[i][t];
                let (vec2a, vec2b) = &buffer[j][t];
                // wasting some time here ... should restrict to i<=j
                seq[ii].push(dot_product_mod_p(vec1a, vec2a, theprime))
                seq[ii].push(dot_product_mod_p(vec1b, vec2b, theprime))
            }
        }

        // store current vectors in curv
        for ii in 0..num_v {
            let vec2 = &buffer[ii][batch_size-1].0;
            curv_result[ii].copy_from_slice(vec2);
        }


        // ) a.parallel_sparse_matvec_mul(&curv_result, theprime);
        //tmpv.copy_from_slice(&tmpv_result);
        // println!("..");
        // Multiply the matrix At with the vector tmpv
        // let curv_result = at.parallel_sparse_matvec_mul_optimized(&tmpv_result, theprime);
        // let curv_result = at.parallel_csr_matvec_mul_simd_preload::<8>(&tmpv_result, theprime);
        // curv_result = tmpv_result.par_iter().map(|ttcurv| {
        // curv_result = tmpv_result.iter().map(|ttcurv| {
        //     // at.serial_sparse_matvec_mul(ttcurv, theprime)
        //     // at.serial_sparse_matvec_mul_chunk(ttcurv, theprime)
        //     // serial_sparse_matvec_mul_chunk2(&at, ttcurv, theprime)
        //     // at.parallel_sparse_matvec_mul_unsafe(ttcurv, theprime)
        //     at.parallel_sparse_matvec_mul(ttcurv, theprime)
        // }).collect::<Vec<_>>();

        // curv_result = at.parallel_sparse_matvec_mul(&tmpv_result, theprime);
        // curv = at.parallel_sparse_matvec_mul(&tmpv_result, theprime);
        // curv.copy_from_slice(&curv_result);
        // println!("...");
        
        // // Compute the dot product of u and curv
        // let dot_products = (0..num_u*num_v).into_iter().map(|ii| {
        //     let i = ii % num_u;
        //     let j = ii / num_u;
        //     // curv_result[i][j] = (curv_result[i][j] + u[i][j] * v[j][i]) % theprime;
        //     dot_product_mod_p(&u[i], &curv_result[j], theprime)
        // }).collect::<Vec<_>>();
        // // push to seq
        // for ii in 0..num_u*num_v {
        //     // seq[ii].push(0);
        //     seq[ii].push(dot_products[ii]);
        // }

        // let dot_product = u.iter().zip(curv.iter()).fold(0, |acc, (&ui, &curvi)| (acc + ui * curvi) % theprime);
        // seq.push(dot_product);
        // println!("....");
        if last_save.elapsed().as_secs_f64() > save_after as f64 {
            println!("\nSaving...");
            let save_start = std::time::Instant::now();
            // save_wdm_file(&wdm_filename, &a, theprime, &row_precond, &col_precond, &u, &v, &curv, &seq)?;
            save_wdm_file2(&wdm_filename, &a, theprime, &row_precond, &col_precond, &v, &v, &curv_result, &seq)?;
            last_save = std::time::Instant::now();
            println!("\nSaved state to file {} at sequence length {} (saving took {:?}).\n", wdm_filename, seq[0].len(), save_start.elapsed());
        }

        if last_report.elapsed().as_secs_f64() > REPORT_AFTER {
            // println!("Report");
            report_progress(start, last_report, last_nlen, seq[0].len(), max_nlen);
            last_report = std::time::Instant::now();
            last_nlen = seq[0].len();
        }
    }
    // copy curv_result to curv
    *curv = curv_result.clone();
    Ok(())
}

pub fn main_loop(
    a: &CsrMatrix,
    at: &CsrMatrix,
    row_precond: &[MyInt],
    col_precond: &[MyInt],
    curv: &mut Vec<Vec<MyInt>>,
    u: &Vec<Vec<MyInt>>,
    v: &Vec<Vec<MyInt>>,
    seq: &mut Vec<Vec<MyInt>>,
    max_nlen: usize,
    wdm_filename: &str,
    theprime: MyInt,
    save_after: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let mut last_save = start;
    let mut last_report = start;
    let mut last_nlen = seq[0].len();
    let mut curv_result  = curv.clone();
    //let mut tmpv = vec![0; a.n_rows];
    let num_u = u.len();
    let num_v = v.len();

    for _ in seq[0].len()..max_nlen {
        // Multiply the matrix A with the vector curv
        // println!(".");
        // let tmpv_result = a.parallel_sparse_matvec_mul_optimized(curv, theprime);
        // let tmpv_result = a.parallel_csr_matvec_mul_simd_preload::<8>(curv, theprime);
        // let tmpv_result = a.parallel_sparse_matvec_mul(curv, theprime);
        let tmpv_result = curv_result.iter().map(|tcurv| {
        // let tmpv_result = curv_result.par_iter().map(|tcurv| {
            // a.serial_sparse_matvec_mul(tcurv, theprime)
            // a.serial_sparse_matvec_mul_chunk(tcurv, theprime)
            // serial_sparse_matvec_mul_chunk2(&a,tcurv, theprime)
            // a.parallel_sparse_matvec_mul_unsafe(tcurv, theprime)
            a.parallel_sparse_matvec_mul(tcurv, theprime)
        }).collect::<Vec<_>>();

        // ) a.parallel_sparse_matvec_mul(&curv_result, theprime);
        //tmpv.copy_from_slice(&tmpv_result);
        // println!("..");
        // Multiply the matrix At with the vector tmpv
        // let curv_result = at.parallel_sparse_matvec_mul_optimized(&tmpv_result, theprime);
        // let curv_result = at.parallel_csr_matvec_mul_simd_preload::<8>(&tmpv_result, theprime);
        // curv_result = tmpv_result.par_iter().map(|ttcurv| {
        curv_result = tmpv_result.iter().map(|ttcurv| {
            // at.serial_sparse_matvec_mul(ttcurv, theprime)
            // at.serial_sparse_matvec_mul_chunk(ttcurv, theprime)
            // serial_sparse_matvec_mul_chunk2(&at, ttcurv, theprime)
            // at.parallel_sparse_matvec_mul_unsafe(ttcurv, theprime)
            at.parallel_sparse_matvec_mul(ttcurv, theprime)
        }).collect::<Vec<_>>();

        // curv_result = at.parallel_sparse_matvec_mul(&tmpv_result, theprime);
        // curv = at.parallel_sparse_matvec_mul(&tmpv_result, theprime);
        // curv.copy_from_slice(&curv_result);
        // println!("...");
        
        // Compute the dot product of u and curv
        let dot_products = (0..num_u*num_v).into_iter().map(|ii| {
            let i = ii % num_u;
            let j = ii / num_u;
            // curv_result[i][j] = (curv_result[i][j] + u[i][j] * v[j][i]) % theprime;
            dot_product_mod_p(&u[i], &curv_result[j], theprime)
        }).collect::<Vec<_>>();
        // push to seq
        for ii in 0..num_u*num_v {
            // seq[ii].push(0);
            seq[ii].push(dot_products[ii]);
        }

        // let dot_product = u.iter().zip(curv.iter()).fold(0, |acc, (&ui, &curvi)| (acc + ui * curvi) % theprime);
        // seq.push(dot_product);
        // println!("....");
        if last_save.elapsed().as_secs_f64() > save_after as f64 {
            println!("\nSaving...");
            let save_start = std::time::Instant::now();
            // save_wdm_file(&wdm_filename, &a, theprime, &row_precond, &col_precond, &u, &v, &curv, &seq)?;
            save_wdm_file2(&wdm_filename, &a, theprime, &row_precond, &col_precond, &u, &v, &curv_result, &seq)?;
            last_save = std::time::Instant::now();
            println!("\nSaved state to file {} at sequence length {} (saving took {:?}).\n", wdm_filename, seq[0].len(), save_start.elapsed());
        }

        if last_report.elapsed().as_secs_f64() > REPORT_AFTER {
            // println!("Report");
            report_progress(start, last_report, last_nlen, seq[0].len(), max_nlen);
            last_report = std::time::Instant::now();
            last_nlen = seq[0].len();
        }
    }
    // copy curv_result to curv
    *curv = curv_result.clone();
    Ok(())
}


fn main() {
    let matches = Command::new("Wiedemann sequence and rank computation")
        .version("1.0")
        .author("Thomas Willwacher")
        .about("Processes sparse matrices in SMS format")
        .arg(
            Arg::new("filename")
                .help("The SMS file containing the sparse matrix")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new("overwrite")
                .short('o')
                .long("overwrite")
                .required(false)
                .help("Overwrite existing files")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("num_threads")
                .short('t')
                .long("threads")
                .help("Number of threads to use")
                .value_parser(clap::value_parser!(usize))
                .value_name("NUM")
                .default_value("4"),
        )
        .arg(
            Arg::new("maxnlen")
                .short('N')
                .long("N")
                .help("The desired sequence length to be computed")
                .value_parser(clap::value_parser!(usize))
                .value_name("MAXNLEN")
                .default_value("0"),
        )
        .arg(
            Arg::new("saveafter")
                .short('s')
                .long("saveafter")
                .help("Trigger automatic saves each s seconds.")
                .value_parser(clap::value_parser!(usize))
                .value_name("SAVEAFTER")
                .default_value("200"),
        )
        .arg(
            Arg::new("prime")
                .short('p')
                .long("prime")
                .help("The prime number to use for modular arithmetic")
                .value_parser(clap::value_parser!(MyInt))
                .value_name("PRIME")
                .default_value("27644437"),
        )
        .arg(
            Arg::new("num_u")
                .short('u')
                .long("num_u")
                .help("The number of columns in the U matrix.")
                .value_parser(clap::value_parser!(usize))
                .value_name("NUMU")
                .default_value("1"),
        )
        .arg(
            Arg::new("num_v")
                .short('v')
                .long("num_v")
                .help("The number of columns in the V matrix.")
                .value_parser(clap::value_parser!(usize))
                .value_name("NUMV")
                .default_value("1"),
        )
        .get_matches();

    let filename = matches.get_one::<String>("filename").expect("Filename is required");
    let overwrite = *matches.get_one::<bool>("overwrite").unwrap_or(&false);
    let num_threads: usize = *matches.get_one::<usize>("num_threads").expect("Invalid number of threads");
    let mut prime: MyInt = *matches.get_one::<MyInt>("prime").unwrap_or(&THEPRIME);
    let mut max_nlen: usize = *matches.get_one::<usize>("maxnlen").unwrap_or(&0) as usize;
    let save_after: usize = *matches.get_one::<usize>("saveafter").unwrap_or(&200);
    let mut num_u: usize = *matches.get_one::<usize>("num_u").unwrap_or(&1);
    let mut num_v: usize = *matches.get_one::<usize>("num_v").unwrap_or(&1);

    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to create thread pool");
    }
    // load the matrix file

    let start_time = std::time::Instant::now();
    let mut a = load_csr_matrix_from_sms(filename).expect("Failed to load matrix");
    // let mut a = reorder_matrix(&a, "data/gra12_9.g6", "data/gra11_9.g6");
    let duration = start_time.elapsed();
    println!("Time taken to load matrix: {:?}", duration);
    println!("Loaded matrix with {} rows and {} columns", a.n_rows, a.n_cols);
    let mut row_precond: Vec<MyInt> = create_random_vector_nozero(a.n_rows, prime);
    let mut col_precond: Vec<MyInt> = create_random_vector_nozero(a.n_cols, prime);
    let mut u: Vec<Vec<MyInt>> = (0..num_u).map(|_| create_random_vector(a.n_cols, prime)).collect();
    let mut v: Vec<Vec<MyInt>> = (0..num_v).map(|_| create_random_vector(a.n_cols, prime)).collect();
    let mut curv: Vec<Vec<MyInt>> = v.clone();
    let mut seq: Vec<Vec<MyInt>> = (0..num_u*num_v).map(|_| Vec::new()).collect();

    let wdm_filename = format!("{}.wdm", filename); 
    // check if exists
    if std::path::Path::new(&wdm_filename).exists() && !overwrite {
        println!("Loading state from file {}...", &wdm_filename);
        let (tprime, tm, tn, tnum_u, tnum_v) = load_wdm_file2(&wdm_filename, &mut row_precond, &mut col_precond, &mut u, &mut v, &mut curv, &mut seq).unwrap();
        prime = tprime;
        if tm != a.n_rows || tn != a.n_cols {
            println!("Matrix dimensions do not match! {}x{} vs {}x{}", tm, tn, a.n_rows, a.n_cols);
            std::process::exit(1);
        }
        num_u = tnum_u;
        num_v = tnum_v;
        println!("Loaded state from file {} with {} entries. Note: parameteer in wdm file take precedence over those passed via command line,", &wdm_filename, seq[0].len());
    } else {

    }
    // compute desired sequence length if none was provided
    if max_nlen <=0 {
        let d = min(a.n_cols, a.n_rows) as f32;
        max_nlen =  (d/(num_u as f32) + d/(num_v as f32) + 1.0).floor() as usize; // 2* min(a.n_cols, a.n_rows);
    }

    // test: gaussian elim
    let mut AA = a.clone();
    println!("Gaussian elimination...");
    AA.gaussian_elimination_markowitz(prime);

    // Check if already done with sequence computation
    if max_nlen > 0 && seq[0].len() >= max_nlen {
        println!("Already computed {} entries, nothing to do.", seq[0].len());

    } else {
        println!{"Preconditioning matrix ..."}
        // precondition matrix
        let mut at = a.transpose(); //transpose_csr_matrix(&a);
        // let att = at.transpose(); // transpose_csr_matrix(&at);
        // if a.are_csr_matrices_equal( &att){
        //     println!("Yeah");
        //     // a.print_triplets();
        //     // at.print_triplets();
        //     // att.print_triplets();
        // } else {
        //     println!("Boooooo");
        //     // a.print_triplets();
        //     // at.print_triplets();
        //     // att.print_triplets();
        // }
        // println!("scaling a");
    
        a.scale_csr_matrix_rows(&row_precond, prime);
        a.scale_csr_matrix_columns(&col_precond, prime);
        // println!("scaling at");
        at.scale_csr_matrix_rows(&col_precond, prime); 
        println!{"Done"}
    
        println!{"Starting computation..."}
    
        // run main loop
        if let Err(e) = main_loop(&a, &at, &row_precond, &col_precond, &mut curv, &u, &v, &mut seq, max_nlen, &wdm_filename, prime, save_after) {
            eprintln!("Error in main loop: {}", e);
            std::process::exit(1);
        } else {
            save_wdm_file2(&wdm_filename, &a, prime, &row_precond, &col_precond, &u, &v, &curv, &seq).expect("Couldn't save result...");
            println!("\nSaved state to file {} at sequence length {}.", wdm_filename, seq[0].len());
        }
    }

    
    // convert seq to a vector of u64
    println!("Sequence computed, running Berlekamp-Massey...");
    // let useq: Vec<u64> = seq.iter().map(|&x| (if x>=0 {x} else {x+prime})  as u64).collect();
    let start_time = std::time::Instant::now();
    let bmres = block_berlekamp_massey(seq, num_u, num_v, prime);
    // let bmres: Vec<u64> = bubblemath::linear_recurrence::berlekamp_massey(&useq, prime as u64);
    let duration = start_time.elapsed();
    println!("Time taken for Berlekamp-Massey: {:?}", duration);
    println!("Berlekamp-Massey result: {:?}", bmres.len());
    
    // Create a random vector
    // let random_vector = create_random_vector(a.n_cols, prime);

    // // Multiply the matrix with the random vector and measure the time
    // let start_time = std::time::Instant::now();
    // let result = parallel_sparse_matvec_mul(&a, &random_vector, prime);
    // let duration = start_time.elapsed();

    // println!("Time taken for matrix-vector multiplication: {:?}", duration);
    // println!("Result vector has {} entries", result.len());

    // println!("Filename: {}", filename);
    // println!("Overwrite: {}", overwrite);

    // println!("Hello, world!");
}
