mod matrices;
mod wdm_files;
use matrices::{CsrMatrix, MyInt, load_csr_matrix_from_sms, create_random_vector, create_random_vector_nozero};
use wdm_files::{save_wdm_file, load_wdm_file};

use std::io::Write;
use clap::{Arg, Command};
use std::cmp::min;

// type MyInt = i64;

const THEPRIME: MyInt = 27644437; // A large prime number for modular arithmetic

const REPORT_AFTER: f64 = 1.0; // seconds



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


pub fn main_loop(
    a: &CsrMatrix,
    at: &CsrMatrix,
    row_precond: &[MyInt],
    col_precond: &[MyInt],
    curv: &mut Vec<MyInt>,
    u: &[MyInt],
    v: &[MyInt],
    seq: &mut Vec<MyInt>,
    max_nlen: usize,
    wdm_filename: &str,
    theprime: MyInt,
    save_after: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();
    let mut last_save = start;
    let mut last_report = start;
    let mut last_nlen = seq.len();
    //let mut tmpv = vec![0; a.n_rows];

    for _ in seq.len()..max_nlen {
        // Multiply the matrix A with the vector curv
        // println!(".");
        let tmpv_result = a.parallel_sparse_matvec_mul(curv, theprime);
        //tmpv.copy_from_slice(&tmpv_result);
        // println!("..");
        // Multiply the matrix At with the vector tmpv
        let curv_result = at.parallel_sparse_matvec_mul(&tmpv_result, theprime);
        curv.copy_from_slice(&curv_result);
        // println!("...");
        // Compute the dot product of u and curv
        let dot_product = u.iter().zip(curv.iter()).fold(0, |acc, (&ui, &curvi)| (acc + ui * curvi) % theprime);
        seq.push(dot_product);
        // println!("....");
        if last_save.elapsed().as_secs_f64() > save_after as f64 {
            println!("\nSaving...");
            let save_start = std::time::Instant::now();
            save_wdm_file(&wdm_filename, &a, theprime, &row_precond, &col_precond, &u, &v, &curv, &seq)?;
            last_save = std::time::Instant::now();
            println!("\nSaved state to file {} at sequence length {} (saving took {:?}).\n", wdm_filename, seq.len(), save_start.elapsed());
        }

        if last_report.elapsed().as_secs_f64() > REPORT_AFTER {
            // println!("Report");
            report_progress(start, last_report, last_nlen, seq.len(), max_nlen);
            last_report = std::time::Instant::now();
            last_nlen = seq.len();
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
        .get_matches();

    let filename = matches.get_one::<String>("filename").expect("Filename is required");
    let overwrite = *matches.get_one::<bool>("overwrite").unwrap_or(&false);
    let num_threads: usize = *matches.get_one::<usize>("num_threads").expect("Invalid number of threads");
    let mut prime: MyInt = *matches.get_one::<MyInt>("prime").unwrap_or(&THEPRIME);
    let mut max_nlen: usize = *matches.get_one::<usize>("maxnlen").unwrap_or(&0) as usize;
    let save_after: usize = *matches.get_one::<usize>("saveafter").unwrap_or(&200);

    println!("Overwrite: {}", overwrite);

    if num_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build_global()
            .expect("Failed to create thread pool");
    }
    // load the matrix file

    let start_time = std::time::Instant::now();
    let mut a = load_csr_matrix_from_sms(filename).expect("Failed to load matrix");
    let duration = start_time.elapsed();
    println!("Time taken to load matrix: {:?}", duration);
    println!("Loaded matrix with {} rows and {} columns", a.n_rows, a.n_cols);
    if max_nlen <=0 {
        max_nlen = 2* min(a.n_cols, a.n_rows);
    }
    let mut row_precond: Vec<MyInt> = create_random_vector_nozero(a.n_rows, prime);
    let mut col_precond: Vec<MyInt> = create_random_vector_nozero(a.n_cols, prime);
    let mut u: Vec<MyInt> = create_random_vector(a.n_cols, prime);
    let mut v: Vec<MyInt> = create_random_vector(a.n_cols, prime);
    let mut curv: Vec<MyInt> = v.clone();
    let mut seq: Vec<MyInt> = Vec::new();

    let wdm_filename = format!("{}.wdm", filename); 
    // check if exists
    if std::path::Path::new(&wdm_filename).exists() && !overwrite {
        println!("Loading state from file {}...", &wdm_filename);
        prime = load_wdm_file(&wdm_filename, &a, &mut row_precond, &mut col_precond, &mut u, &mut v, &mut curv, &mut seq).unwrap();
    } else {

    }

    // Check if already done with sequence computation
    if max_nlen > 0 && seq.len() >= max_nlen {
        println!("Already computed {} entries, nothing to do.", seq.len());

    } else {
        println!{"Preconditioning matrix ..."}
        // precondition matrix
        let mut at = a.transpose(); //transpose_csr_matrix(&a);
        let att = at.transpose(); // transpose_csr_matrix(&at);
        if a.are_csr_matrices_equal( &att){
            println!("Yeah");
            // a.print_triplets();
            // at.print_triplets();
            // att.print_triplets();
        } else {
            println!("Boooooo");
            // a.print_triplets();
            // at.print_triplets();
            // att.print_triplets();
        }
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
            save_wdm_file(&wdm_filename, &a, prime, &row_precond, &col_precond, &u, &v, &curv, &seq).expect("Couldn't save result...");
            println!("\nSaved state to file {} at sequence length {}.", wdm_filename, seq.len());
        }
    }

    
    // convert seq to a vector of u64
    println!("Sequence computed, running Berlekamp-Massey...");
    let useq: Vec<u64> = seq.iter().map(|&x| (if x>=0 {x} else {x+prime})  as u64).collect();
    let start_time = std::time::Instant::now();
    let bmres = bubblemath::linear_recurrence::berlekamp_massey(&useq, prime as u64);
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
