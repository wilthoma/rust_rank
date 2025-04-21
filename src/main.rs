
// #![feature(portable_simd)]

mod matrices;
mod graphs;
mod wdm_files;
mod block_berlekamp_massey;
use block_berlekamp_massey::block_berlekamp_massey;
use image::buffer;
use matrices::*; //{create_random_vector, create_random_vector_nozero, load_csr_matrix_from_sms, reorder_csr_matrix_by_keys, spy_plot, CsrMatrix, MyInt};
use wdm_files::{save_wdm_file_sym, load_wdm_file_sym};
use graphs::count_triangles_in_file;
use std::io::Write;
use clap::{Arg, Command};
use std::cmp::min;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{self, Sender, Receiver};
use std::thread;

// type MyInt = i64;

const THEPRIME: MyInt = 27644437 as MyInt; // A large prime number for modular arithmetic

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

pub fn main_loop_s_mt(
    a: &Arc<CsrMatrix>,
    at: &Arc<CsrMatrix>,
    row_precond: &[MyInt],
    col_precond: &[MyInt],
    curv: &mut Vec<Vec<MyInt>>,
    v: &Vec<Vec<MyInt>>,
    seq: &mut Vec<Vec<MyInt>>,
    max_nlen: usize,
    wdm_filename: &str,
    theprime: MyInt,
    save_after: usize,
    use_matvmul_parallel: bool,
    use_vecp_parallel: bool,
    deep_clone_matrix: bool,

) -> Result<(), Box<dyn std::error::Error>> {

    let start = std::time::Instant::now();
    let mut last_save = start;
    let mut last_report = start;
    let mut last_nlen = seq[0].len();
    let mut curv_result  = curv.clone();
    let to_be_produced = max_nlen - seq[0].len();
    //let mut tmpv = vec![0; a.n_rows];
    let num_v = v.len();
    let buffer_capacity = 10;
    let (txs, rxs): (Vec<mpsc::SyncSender<_>>, Vec<mpsc::Receiver<_>>) = 
        (0..num_v).map(|_| mpsc::sync_channel(buffer_capacity)).unzip();

    // let a = std::sync::Arc::new(a.clone());
    // let at = std::sync::Arc::new(at.clone());
    let workers: Vec<_> = txs.into_iter().enumerate().map(|(worker_id, tx)| {
                let local_curv = curv[worker_id].clone();
                let a = if deep_clone_matrix{Arc::new(CsrMatrix::clone(&a))} else {std::sync::Arc::clone(a)};
                let at = if deep_clone_matrix{Arc::new(CsrMatrix::clone(&at))} else {std::sync::Arc::clone(at)};
                // let a = CsrMatrix::clone(&a);
                // let at = CsrMatrix::clone(&at);
                thread::spawn(move || {
                    let mut local_curv = local_curv;
                    for _ in 0..to_be_produced {
                        let vec1 = if use_matvmul_parallel {a.parallel_sparse_matvec_mul(&local_curv, theprime)}
                                             else {a.serial_sparse_matvec_mul(&local_curv, theprime)};
                        let vec2 = if use_matvmul_parallel {at.parallel_sparse_matvec_mul(&vec1, theprime)}
                                             else {at.serial_sparse_matvec_mul(&vec1, theprime)};
                        tx.send((vec1.clone(), vec2.clone())).unwrap();
                        local_curv = vec2;
                    }
                })
            }).collect();
    
    for _ in 0..to_be_produced {
        let mut received_tokens = Vec::new();

        // First, read one token from each channel
        for rx in &rxs {
            match rx.recv() {
            Ok((vec1, vec2)) => {
                received_tokens.push((vec1, vec2));
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
                let (vec1a, vec1b) = &received_tokens[i];
                let (vec2a, vec2b) = &received_tokens[j];
                if (use_vecp_parallel){
                    seq[ii].push(dot_product_mod_p_parallel(vec1a, vec2a, theprime));
                    seq[ii].push(dot_product_mod_p_parallel(vec1b, vec2b, theprime));
                }
                else {
                    seq[ii].push(dot_product_mod_p_serial(vec1a, vec2a, theprime));
                    seq[ii].push(dot_product_mod_p_serial(vec1b, vec2b, theprime));
                }
                ii += 1;
            }
        }

        // println!("Time taken for dot product computations: {:?}\n", start_time.elapsed());

        // store current vectors in curv
        for ii in 0..num_v {
            let vec2 = &received_tokens[ii].1;
            curv_result[ii].copy_from_slice(vec2);

        }
        // std::thread::sleep(std::time::Duration::from_millis(1500));

        if last_save.elapsed().as_secs_f64() > save_after as f64 {
            println!("\nSaving...");
            let save_start = std::time::Instant::now();
            // save_wdm_file(&wdm_filename, &a, theprime, &row_precond, &col_precond, &u, &v, &curv, &seq)?;
            save_wdm_file_sym(&wdm_filename, &a, theprime, &row_precond, &col_precond, &v, &curv_result, &seq)?;
            last_save = std::time::Instant::now();
            println!("\nSaved state to file {} at sequence length {} (saving took {:?}).\n", wdm_filename, seq[0].len(), save_start.elapsed());
        }

        if last_report.elapsed().as_secs_f64() > REPORT_AFTER {
            // println!("Report");
            let channel_fills: String = rxs.iter().map(|rx| {
                let count = rx.try_iter().count();
                format!("{} ", count)
            }).collect::<String>();
            report_progress(start, last_report, last_nlen, seq[0].len(), max_nlen, &format!("Channel fill {}", channel_fills));
            last_report = std::time::Instant::now();
            last_nlen = seq[0].len();
        }
    }
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
                .help("Size of threadpool to use")
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
            Arg::new("num_v")
                .short('v')
                .long("num_v")
                .help("The number of columns in the V matrix. This is also the number of worker threads.")
                .value_parser(clap::value_parser!(usize))
                .value_name("NUMV")
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
        .get_matches();

    let filename = matches.get_one::<String>("filename").expect("Filename is required");
    let overwrite = *matches.get_one::<bool>("overwrite").unwrap_or(&false);
    let serial_matvmul = *matches.get_one::<bool>("serial_matvmul").unwrap_or(&false);
    let parallel_dot = *matches.get_one::<bool>("parallel_dot").unwrap_or(&false);
    let deep_clone = *matches.get_one::<bool>("clone").unwrap_or(&false);
    let num_threads: usize = *matches.get_one::<usize>("num_threads").expect("Invalid number of threads");
    let mut prime: MyInt = *matches.get_one::<MyInt>("prime").unwrap_or(&THEPRIME);
    let mut max_nlen: usize = *matches.get_one::<usize>("maxnlen").unwrap_or(&0) as usize;
    let save_after: usize = *matches.get_one::<usize>("saveafter").unwrap_or(&200);
    // let mut num_u: usize = *matches.get_one::<usize>("num_u").unwrap_or(&1);
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
    // let a = std::sync::Arc::new(load_csr_matrix_from_sms(filename).expect("Failed to load matrix"));
    // let mut a = reorder_matrix(&a, "data/gra12_9.g6", "data/gra11_9.g6");
    let duration = start_time.elapsed();
    println!("Time taken to load matrix: {:?}", duration);
    println!("Loaded matrix with {} rows and {} columns", a.n_rows, a.n_cols);
    let mut row_precond: Vec<MyInt> = create_random_vector_nozero(a.n_rows, prime);
    let mut col_precond: Vec<MyInt> = create_random_vector_nozero(a.n_cols, prime);
    // let mut u: Vec<Vec<MyInt>> = (0..num_u).map(|_| create_random_vector(a.n_cols, prime)).collect();
    let mut v: Vec<Vec<MyInt>> = (0..num_v).map(|_| create_random_vector(a.n_cols, prime)).collect();
    let mut curv: Vec<Vec<MyInt>> = v.clone();
    let mut seq: Vec<Vec<MyInt>> = (0..num_v*(num_v+1)/2).map(|_| Vec::new()).collect();

    let wdm_filename = format!("{}.wdm", filename); 
    // check if exists
    if std::path::Path::new(&wdm_filename).exists() && !overwrite {
        println!("Loading state from file {}...", &wdm_filename);
        let (tprime, tm, tn, tnum_v) = load_wdm_file_sym(&wdm_filename, &mut row_precond, &mut col_precond, &mut v, &mut curv, &mut seq).unwrap();
        prime = tprime;
        if tm != a.n_rows || tn != a.n_cols {
            println!("Matrix dimensions do not match! {}x{} vs {}x{}", tm, tn, a.n_rows, a.n_cols);
            std::process::exit(1);
        }
        // num_u = tnum_u;
        num_v = tnum_v;
        println!("Loaded state from file {} with {} entries. Note: parameteer in wdm file take precedence over those passed via command line,", &wdm_filename, seq[0].len());
    } else {

    }
    // compute desired sequence length if none was provided
    if max_nlen <=0 {
        let d = min(a.n_cols, a.n_rows) as f32;
        max_nlen = (2.0 * d/(num_v as f32) + 4.0).floor() as usize; // 2* min(a.n_cols, a.n_rows);
    }

    // test: gaussian elim
    // let mut AA = a.clone();
    // println!("Gaussian elimination...");
    // AA.gaussian_elimination_markowitz(prime);

    // Check if already done with sequence computation
    if max_nlen > 0 && seq[0].len() >= max_nlen {
        println!("Already computed {} entries, nothing to do.", seq[0].len());

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
    
        println!{"Starting computation..."}
    
        // run main loop
        if let Err(e) = main_loop_s_mt(&a, &at, &row_precond, &col_precond, &mut curv, &v, &mut seq, max_nlen, 
                &wdm_filename, prime, save_after, !serial_matvmul, parallel_dot, deep_clone) {
            eprintln!("Error in main loop: {}", e);
            std::process::exit(1);
        } else {
            save_wdm_file_sym(&wdm_filename, &a, prime, &row_precond, &col_precond, &v, &curv, &seq).expect("Couldn't save result...");
            println!("\nSaved state to file {} at sequence length {}.", wdm_filename, seq[0].len());
        }
    }

    
    // convert seq to a vector of u64
    println!("Sequence computed, running Berlekamp-Massey...");
    // let useq: Vec<u64> = seq.iter().map(|&x| (if x>=0 {x} else {x+prime})  as u64).collect();
    let start_time = std::time::Instant::now();
    let bmres = block_berlekamp_massey(seq, num_v, num_v, prime);
    // let bmres: Vec<u64> = bubblemath::linear_recurrence::berlekamp_massey(&useq, prime as u64);
    let duration = start_time.elapsed();
    println!("Time taken for Berlekamp-Massey: {:?}", duration);
    println!("Berlekamp-Massey result: {:?}", bmres.len());
}
