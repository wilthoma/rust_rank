
// #![feature(portable_simd)]
#![feature(portable_simd)]

mod matrices;
mod graphs;
mod wdm_files;
mod block_berlekamp_massey;
mod blockbmtest;
mod vectorstream;
mod invariant_factor;
use bubblemath::linear_recurrence::berlekamp_massey;
use invariant_factor::{top_invariant_factor, vecvec_to_symmetric_poly_matrix, vec_matrix_to_poly_matrix};
use vectorstream::*;
use blockbmtest::{matrix_berlekamp_massey, test_matrix_berlekamp_massey_simple2, vecvec_to_symmetric_matrix_list};
use block_berlekamp_massey::block_berlekamp_massey;
use matrices::*; //{create_random_vector, create_random_vector_nozero, load_csr_matrix_from_sms, reorder_csr_matrix_by_keys, spy_plot, CsrMatrix, MyInt};
use wdm_files::{save_wdm_file_sym, load_wdm_file_sym};
// use graphs::count_triangles_in_file;
use std::io::Write;
use clap::{Arg, Command};
use std::cmp::min;
use rayon::prelude::*;
use std::sync::Arc;
use crossbeam_channel;
use std::thread;
use std::ops::{Add, AddAssign, Mul, Rem};
use std::iter::Sum;
use std::fmt::Display;
use num_traits::Zero;
use rand::distr::uniform::SampleUniform;
use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};

// type MyInt = u64;
// type MyInt = u32;
type MyInt = u32;

const THEPRIME: u32 = 27644437; // A large prime number for modular arithmetic
const THESMALLPRIME : u32 = 3323;
const THETINYPRIME : u16 = 23;
// const THESMALLPRIME : u32 = 6481;
// const THESMALLPRIME : u32 = 5669;

const REPORT_AFTER: f64 = 1.0; // seconds



pub fn report_progress(
    start_time: std::time::Instant,
    last_time: std::time::Instant,
    last_nlen: usize,
    nlen: usize,
    max_nlen: usize,
    suffix: &str,
) {
    let elapsed = start_time.elapsed();
    let elapsed_last = last_time.elapsed();
    let speed = (nlen - last_nlen) as f64 / elapsed_last.as_secs_f64();
    let remaining = (max_nlen - nlen) as f64 / speed;
    print!(
        "\rProgress: {}/{} | Elapsed: {:?} | Throughput: {:.2}/s | Remaining: {:?} | {}           ",
        nlen,
        max_nlen,
        elapsed,
        speed,
        std::time::Duration::from_secs_f64(remaining),
        suffix
    );
    std::io::stdout().flush().unwrap();
}


pub fn main_loop_s_mt2<T:GoodInteger, S:VectorStream<T>>(
    stream : &mut S,
    a: &Arc<CsrMatrix<T>>,
    row_precond: &[T],
    col_precond: &[T],
    curv: &mut Vec<Vec<T>>,
    v: &Vec<Vec<T>>,
    seq: &mut Vec<Vec<T>>,
    max_nlen: usize,
    wdm_filename: &str,
    theprime: T,
    save_after: usize,
    use_vecp_parallel: bool,
) -> Result<(), Box<dyn std::error::Error>> 
{

    let start = std::time::Instant::now();
    let mut last_save = start;
    let mut last_report = start;
    let mut last_nlen = seq[0].len();
    // let mut curv_result  = curv.clone();
    let to_be_produced = max_nlen - seq[0].len(); 
    let num_v = curv.len();
    
    for _ in 0..to_be_produced/2 {
        let received_tokens = stream.next().unwrap();

        // Now, process the received tokens
        // let start_time = std::time::Instant::now();
    

        // let dotps = (0..num_v).into_iter()
        // .flat_map(move |i| (i..num_v).into_iter().map(move |j| (i, j)))
        // .collect::<Vec<(usize, usize)>>().par_iter().map(|(i, j)| {
        //     let vec1 = &received_tokens[*i];
        //     let vec1_prev = &curv[*i];
        //     let vec2 = &received_tokens[*j];

        //     if use_vecp_parallel {
        //         (dot_product_mod_p_parallel(vec1_prev, vec2, theprime), dot_product_mod_p_parallel(vec1, vec2, theprime))
        //     }
        //     else {
        //         (dot_product_mod_p_serial(vec1_prev, vec2, theprime), dot_product_mod_p_serial(vec1, vec2, theprime))
        //     }
        // }).collect::<Vec<_>>();

        // for (ii, (dot1, dot2)) in dotps.into_iter().enumerate() {
        //     seq[ii].push(dot1);
        //     seq[ii].push(dot2);
        // }


        let mut ii = 0;
        for i in 0..num_v {
            for j in i..num_v {

                let vec1 = &received_tokens[i];
                let vec1_prev = &curv[i];
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
        curv.clone_from_slice(&received_tokens);

        // std::thread::sleep(std::time::Duration::from_millis(1500));

        if last_save.elapsed().as_secs_f64() > save_after as f64 {
            println!("\nSaving...");
            let save_start = std::time::Instant::now();
            last_save = std::time::Instant::now();
            match save_wdm_file_sym(&wdm_filename, a.n_rows, a.n_cols, theprime, &row_precond, &col_precond, &v, curv, &seq) {
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
            let fill_status = stream.fill_status();
            report_progress(start, last_report, last_nlen, seq[0].len(), max_nlen, &format!("Channel fill status {}", fill_status));
            last_report = std::time::Instant::now();
            last_nlen = seq[0].len();
        }
    }

    // *curv = curv_result;
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
                .help("Size of shared threadpool to use (excluding the worker threads)")
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
            Arg::new("lanes")
                .short('l')
                .long("lanes")
                .help("The number of SIMD lanes to use (1,2,4,8 or 16).")
                .value_parser(clap::value_parser!(usize))
                .value_name("LANES")
                .default_value("1"),
        )
        .arg(
            Arg::new("prime")
                .short('p')
                .long("prime")
                .help("The prime number to use for modular arithmetic")
                .value_parser(clap::value_parser!(u32))
                .value_name("PRIME")
                //.default_value("27644437")
                ,
        )
        .arg(
            Arg::new("num_workers")
                .short('w')
                .long("num_workers")
                .help("The number of worker threads, excluding the shared thread pool. The number of columns in the V matrix is the number of worker threads times the lane count.")
                .value_parser(clap::value_parser!(usize))
                .value_name("NUMW")
                .default_value("1"),
        )
        .arg(
            Arg::new("serial_matvmul")
                .long("serial_matvmul")
                .required(false)
                .help("Use serial matrix vector multiplication (no threadpool usage)")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("parallel_dot")
                .long("parallel_dot")
                .required(false)
                .help("Use the threadpool vector-vector scalar products")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("clone")
                .long("clone")
                .required(false)
                .help("Clone the matrix for each worker thread. (Uses memory.)")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("transpose")
                .long("transpose")
                .required(false)
                .help("Transpose the matrix.")
                .action(clap::ArgAction::SetTrue)
        )
        .arg(
            Arg::new("benchmark")
                .long("benchmark")
                .required(false)
                .help("Run matrix vector multiply benchmark, but no other computation.")
                .action(clap::ArgAction::SetTrue)
        )
        .get_matches();

    let filename = matches.get_one::<String>("filename").expect("Filename is required");
    let overwrite = *matches.get_one::<bool>("overwrite").unwrap_or(&false);
    let serial_matvmul = *matches.get_one::<bool>("serial_matvmul").unwrap_or(&false);
    let parallel_dot = *matches.get_one::<bool>("parallel_dot").unwrap_or(&false);
    let benchmark = *matches.get_one::<bool>("benchmark").unwrap_or(&false);
    let deep_clone = *matches.get_one::<bool>("clone").unwrap_or(&false);
    let transpose_matrix = *matches.get_one::<bool>("transpose").unwrap_or(&false);
    let num_threads: usize = *matches.get_one::<usize>("num_threads").expect("Invalid number of threads");
    let default_prime = if (MyInt::max_value() as u128) < (THEPRIME as u128)* (THEPRIME as u128) {THESMALLPRIME} else {THEPRIME};
    let pprime: u32 = *matches.get_one::<u32>("prime").unwrap_or(&default_prime);
    let mut max_nlen: usize = *matches.get_one::<usize>("maxnlen").unwrap_or(&0) as usize;
    let save_after: usize = *matches.get_one::<usize>("saveafter").unwrap_or(&200);
    let lanes: usize = *matches.get_one::<usize>("lanes").unwrap_or(&1);
    // let mut num_u: usize = *matches.get_one::<usize>("num_u").unwrap_or(&1);
    let mut num_workers: usize = *matches.get_one::<usize>("num_workers").unwrap_or(&1);
    let num_v = num_workers * lanes;

    let mut prime = pprime as MyInt;

    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to create thread pool");
    }


    // load the matrix file -- TODO: matrix loading must be moved after the wdm loading since the prime number is needed for matrix loading
    let start_time = std::time::Instant::now();
    let mut a:CsrMatrix<MyInt> = CsrMatrix::load_csr_matrix_from_sms(filename, prime as u32).expect("Failed to load matrix");
    if transpose_matrix {
        a = a.transpose();
    }
    // let a = std::sync::Arc::new(load_csr_matrix_from_sms(filename).expect("Failed to load matrix"));
    // let mut a = reorder_matrix(&a, "data/gra12_9.g6", "data/gra11_9.g6");
    let duration = start_time.elapsed();
    println!("Time taken to load matrix: {:?}", duration);
    println!("Loaded matrix with {} rows and {} columns", a.n_rows, a.n_cols);

    let mut row_precond: Vec<MyInt> = create_random_vector_nozero(a.n_rows, prime);
    let mut col_precond: Vec<MyInt> = create_random_vector_nozero(a.n_cols, prime);
    // let mut col_precond: Vec<MyInt> = (0..a.n_cols).map(|_| 1).collect();
    // let mut row_precond: Vec<MyInt> = (0..a.n_rows).map(|_| 2).collect();

    let mut v: Vec<Vec<MyInt>> = (0..num_v).map(|_| create_random_vector(a.n_cols, prime)).collect();
    // let mut v: Vec<Vec<MyInt>> = (0..num_v).map(|_| (0..a.n_cols).map(|_| 1).collect() ).collect();
    let mut curv: Vec<Vec<MyInt>> = v.clone();
    let mut seq: Vec<Vec<MyInt>> = (0..num_v*(num_v+1)/2).map(|_| Vec::new()).collect();

    let wdm_filename = if transpose_matrix {format!("{}_t.wdm", filename) } else {format!("{}.wdm", filename)}; 
    // check if exists
    if std::path::Path::new(&wdm_filename).exists() && !overwrite {
        println!("Loading state from file {}...", &wdm_filename);
        let (tprime, tm, tn, tnum_v) = load_wdm_file_sym(&wdm_filename, &mut row_precond, &mut col_precond, &mut v, &mut curv, &mut seq).unwrap();
        prime = tprime as MyInt;
        if tm != a.n_rows || tn != a.n_cols {
            println!("Matrix dimensions do not match! {}x{} vs {}x{}", tm, tn, a.n_rows, a.n_cols);
            std::process::exit(1);
        }
        // num_u = tnum_u;
        if tnum_v != num_v {
            println!("Number of workers * lanes does not match the size of V in the saved file! {} vs {}", tnum_v, num_v);
            std::process::exit(1);
        }
        println!("Loaded state from file {} with {} entries. Note: parameteer in wdm file take precedence over those passed via command line,", &wdm_filename, seq[0].len());
    } else {

    }
    // compute desired sequence length if none was provided
    if max_nlen <=0 {
        let d = min(a.n_cols, a.n_rows) as f32;
        max_nlen = (2.0 * d/(num_v as f32) + 4.0).floor() as usize; // 2* min(a.n_cols, a.n_rows);
    }

    // check prime validity -- i.e., if with our custom simplifications we might run into overflows
    let (max_row_nnz, max_col_nnz) = a.max_nnzs();
    println!("Max row nnz: {} Max col nnz: {}", max_row_nnz, max_col_nnz);
    if a.is_prime_valid(prime as u32, MyInt::max_value() as i128) {
        println!("Prime number {} is valid, no overflows expected.", prime);
    } else {
        // compute estimate for max prime number 
        let max_nnz = std::cmp::max(std::cmp::max(max_row_nnz, max_col_nnz), matrices::DOT_PRODUCT_CHUNK_SIZE);
        let max_prime = ( (MyInt::max_value() as f64) / (max_nnz as f64) ).sqrt();
        println!("Prime number {} is not valid (must be <{}), may result in overflows. Exiting...", prime, max_prime);
        std::process::exit(1);
    }

    // test: gaussian elim
    // let mut AA = a.clone();
    // println!("Gaussian elimination...");
    // AA.gaussian_elimination_markowitz(prime);

    if benchmark
    {   
        let a = a.tou64();
        println!("Running benchmark u64...");
        a.normal_simd_speedtest( THESMALLPRIME as u64, 3);
        a.normal_simd_speedtest_serial( THESMALLPRIME as u64, 3);
        println!("Running benchmark u32...");
        let aa : CsrMatrix<u32> = a.tou32();
        aa.normal_simd_speedtest( THESMALLPRIME, 3);
        aa.normal_simd_speedtest_serial( THESMALLPRIME, 3);
        println!("Running benchmark u16...");
        let aaa : CsrMatrix<u16> = a.tou16();
        aaa.normal_simd_speedtest( THETINYPRIME, 3);
        aaa.normal_simd_speedtest_serial( THETINYPRIME, 3);
        let at = a.transpose().tou64();
        println!("Running benchmark u64 (transpose)...");
        at.normal_simd_speedtest( THESMALLPRIME as u64, 3);
        at.normal_simd_speedtest_serial( THESMALLPRIME as u64, 3);
        println!("Running benchmark u32 (transpose)...");
        let aat : CsrMatrix<u32> = at.tou32();
        aat.normal_simd_speedtest( THESMALLPRIME, 3);
        aat.normal_simd_speedtest_serial( THESMALLPRIME, 3);
        println!("Running benchmark u16 (transpose)...");
        let aaat : CsrMatrix<u16> = at.tou16();
        aaat.normal_simd_speedtest( THETINYPRIME, 3);
        aaat.normal_simd_speedtest_serial( THETINYPRIME, 3);
        return;
    }


    // Check if already done with sequence computation
    let to_be_computed = max_nlen - seq[0].len();
    if to_be_computed <= 0 {
        println!("Already computed {} sequence entries, nothing to do.", seq[0].len());

    } else {
        println!{"Preconditioning matrix ..."}
        // precondition matrix
        let mut at = a.transpose(); //transpose_csr_matrix(&a);
    
        a.scale_csr_matrix_rows(&row_precond, prime);
        a.scale_csr_matrix_columns(&col_precond, prime);
        at.scale_csr_matrix_rows(&col_precond, prime); 

        let a = std::sync::Arc::new(a);
        let at = std::sync::Arc::new(at);
        println!{"Done"}
    
        println!{"Starting computation with {} workers on {} lane(s) each...", num_workers, lanes}
        // let x : Box<&mut dyn VectorStream<MyInt>> = Box::new(&NormalVectorStream::new(&a, &at, &curv, prime, to_be_computed/2, !serial_matvmul, deep_clone));
        
        // main_loop_s_mt2( x, &a, &at, &row_precond, &col_precond, &mut curv, &v, &mut seq, max_nlen, 
            // &wdm_filename, prime, save_after, parallel_dot);
        let err;
        if lanes == 1 {
            // we don't use simd 
            let mut stream: NormalVectorStream<MyInt> = NormalVectorStream::new(&a, &at, &curv, prime, to_be_computed/2, !serial_matvmul, deep_clone);
            err = main_loop_s_mt2(&mut stream, &a, &row_precond, &col_precond, &mut curv, &v, &mut seq, max_nlen, 
                &wdm_filename, prime, save_after, parallel_dot);
        } else if lanes == 2{
            // we use simd 
            let mut stream: SimdVectorStream<MyInt, 2> = SimdVectorStream::new(&a, &at, &curv, prime, to_be_computed/2, !serial_matvmul, deep_clone);
            err = main_loop_s_mt2(&mut stream, &a, &row_precond, &col_precond, &mut curv, &v, &mut seq, max_nlen, 
                &wdm_filename, prime, save_after, parallel_dot);
        } else if lanes == 4{
            // we use simd 
            let mut stream: SimdVectorStream<MyInt, 4> = SimdVectorStream::new(&a, &at, &curv, prime, to_be_computed/2, !serial_matvmul, deep_clone);
            err = main_loop_s_mt2(&mut stream, &a, &row_precond, &col_precond, &mut curv, &v, &mut seq, max_nlen, 
                &wdm_filename, prime, save_after, parallel_dot);
        } else if lanes == 8{
            // we use simd 
            let mut stream: SimdVectorStream<MyInt, 8> = SimdVectorStream::new(&a, &at, &curv, prime, to_be_computed/2, !serial_matvmul, deep_clone);
            err = main_loop_s_mt2(&mut stream, &a, &row_precond, &col_precond, &mut curv, &v, &mut seq, max_nlen, 
                &wdm_filename, prime, save_after, parallel_dot);
        } else if lanes == 16{
            // we use simd 
            let mut stream: SimdVectorStream<MyInt, 16> = SimdVectorStream::new(&a, &at, &curv, prime, to_be_computed/2, !serial_matvmul, deep_clone);
            err = main_loop_s_mt2(&mut stream, &a, &row_precond, &col_precond, &mut curv, &v, &mut seq, max_nlen, 
                &wdm_filename, prime, save_after, parallel_dot);
        } else {
            println!("Invalid number of lanes: {}. Must be 1, 2, 4, 8 or 16.", lanes);
            std::process::exit(1);
        }

        // run main loop
        if let Err(e) = err {
            eprintln!("Error in main loop: {}", e);
            std::process::exit(1);
        } else {
            save_wdm_file_sym(&wdm_filename, a.n_rows, a.n_cols, prime, &row_precond, &col_precond, &v, &curv, &seq).expect("Couldn't save result...");
            println!("\nSaved state to file {} at sequence length {}.", wdm_filename, seq[0].len());
        }
    }

    
    // convert seq to a vector of u64
    println!("Sequence computed, running Berlekamp-Massey...");
    // let useq: Vec<u64> = seq.iter().map(|&x| (if x>=0 {x} else {x+prime})  as u64).collect();
    let start_time = std::time::Instant::now();
    let bmres = block_berlekamp_massey(seq.clone(), num_v, num_v, prime );
    // let bmres: Vec<u64> = bubblemath::linear_recurrence::berlekamp_massey(&useq, prime as u64);
    let duration = start_time.elapsed();
    println!("Time taken for Berlekamp-Massey: {:?}", duration);
    println!("Berlekamp-Massey result: {:?}", bmres.len());
    println!("First coeff: {:} Last coeff: {:}", bmres[0], bmres[bmres.len()-1]);
    println!("{:?}", bmres);

    // test_matrix_berlekamp_massey_simple();

    // test_matrix_berlekamp_massey_simple2();
    // return;
    // TODO : add leading coeff to seq!!
    let delta = seq[0].len();
    let seq2 = vecvec_to_symmetric_matrix_list(&seq, num_v);
    let res = matrix_berlekamp_massey(&seq2, delta, prime as i64).unwrap();
    let res2 = vec_matrix_to_poly_matrix(&res,  prime as i64);
    let res3 = top_invariant_factor(res2.clone());
    println!("Matrix Berlekamp-Massey result: {:}", res3.deg());
    println!("{:?}", res3);
    println!("{:?}", res2);

}
