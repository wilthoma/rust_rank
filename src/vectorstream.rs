
use crate::matrices::*;
use std::thread;
use core::simd::{LaneCount, Simd, SimdElement, SupportedLaneCount};
use std::sync::Arc;

pub trait VectorStream<T> {
    fn next(&mut self) -> Option<Vec<Vec<T>>>;
    fn fill_status(&self) -> String ;
}

pub struct NormalVectorStream<T> {
    receivers: Vec<crossbeam_channel::Receiver<Vec<T>>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl<T> VectorStream<T> for NormalVectorStream<T>
where T: GoodInteger
{
    fn next(&mut self) -> Option<Vec<Vec<T>>> {
        let mut received_tokens = Vec::new();
        for rx in &self.receivers {
            match rx.recv() {
                Ok(vec) => {
                    received_tokens.push(vec);
                }
                Err(e) => {
                    eprintln!("Error receiving token: {}", e);
                    return None;
                }
            }
        }
        Some(received_tokens)
    }
    fn fill_status(&self) -> String {
        let fill_status: String = self.receivers.iter()
            .map(|rx| format!("{} ", rx.len()))
            .collect();
        fill_status
    }
}

impl<T> NormalVectorStream<T> 
where T: GoodInteger
{
    pub fn new(a : &Arc<CsrMatrix<T>>, at : &Arc<CsrMatrix<T>>, curv : &Vec<Vec<T>>, theprime : T, to_be_produced : usize, use_matvmul_parallel : bool, deep_clone_matrix :bool) -> Self {
        let buffer_capacity = 10;
        let n_threads = curv.len();
        let (txs, rxs) : (Vec<_>, Vec<_>) = 
            (0..n_threads).map(|_| crossbeam_channel::bounded(buffer_capacity)).unzip();
        // Update the worker threads to send only one vector
        let _workers: Vec<_> = txs.into_iter().enumerate().map(|(worker_id, tx)| {
            let local_curv = curv[worker_id].clone();
            let a = if deep_clone_matrix { Arc::new(CsrMatrix::clone(a)) } else { std::sync::Arc::clone(a) };
            let at = if deep_clone_matrix { Arc::new(CsrMatrix::clone(at)) } else { std::sync::Arc::clone(at) };
            thread::spawn(move || {

                if use_matvmul_parallel {
                    // we store curw=A^t (A^tA)^i v for the next iteration since ownership of (A^tA)^i v is lost by sending over the channel
                    let mut curw = a.parallel_sparse_matvec_mul(&local_curv, theprime);

                    for _ in 0..to_be_produced {
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

                    for _ in 0..to_be_produced {
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
        NormalVectorStream {
            receivers: rxs,
            workers: _workers,
        }
    }
}

pub struct SimdVectorStream<T, const LANES: usize> 
where T: GoodInteger,
LaneCount<LANES>: SupportedLaneCount,
{
    receivers: Vec<crossbeam_channel::Receiver<Vec<Simd<T, LANES>>>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl<T, const LANES : usize> VectorStream<T> for SimdVectorStream<T, LANES>
where T: GoodInteger,
      LaneCount<LANES>: SupportedLaneCount,
{
    fn next(&mut self) -> Option<Vec<Vec<T>>> {
        let mut received_tokens = Vec::new();
        for rx in &self.receivers {
            match rx.recv() {
                Ok(vec) => {
                    received_tokens.push(vec);
                }
                Err(e) => {
                    eprintln!("Error receiving token: {}", e);
                    return None;
                }
            }
        }
        // Convert simd vectors to normal
        Some(unzip_simd_vectors(&received_tokens))
    }
    fn fill_status(&self) -> String {
        let fill_status: String = self.receivers.iter()
            .map(|rx| format!("{} ", rx.len()))
            .collect();
        fill_status
    }
}
impl<T, const LANES : usize> SimdVectorStream<T, LANES>
where T: GoodInteger, LaneCount<LANES>: SupportedLaneCount, Simd<T, LANES>: GoodSimd,
{
    pub fn new(a : &Arc<CsrMatrix<T>>, at : &Arc<CsrMatrix<T>>, curv : &Vec<Vec<T>>, theprime : T, to_be_produced : usize, use_matvmul_parallel : bool, deep_clone_matrix :bool) -> Self {
        assert_eq!(curv.len() % LANES, 0, "Number of vectors must be divisible by lane count");
        let n_threads =curv.len() / LANES; 

        let buffer_capacity = 10;
        let (txs, rxs) : (Vec<_>, Vec<_>) = 
            (0..n_threads).map(|_| crossbeam_channel::bounded(buffer_capacity)).unzip();
        let simd_curv = zip_simd_vectors::<T,LANES>(curv);
        // Update the worker threads to send only one vector
        let _workers: Vec<_> = txs.into_iter().enumerate().map(|(worker_id, tx)| {
            let local_curv = simd_curv[worker_id].clone();
            let a = if deep_clone_matrix { Arc::new(CsrMatrix::clone(a)) } else { std::sync::Arc::clone(a) };
            let at = if deep_clone_matrix { Arc::new(CsrMatrix::clone(at)) } else { std::sync::Arc::clone(at) };
            thread::spawn(move || {

                if use_matvmul_parallel {
                    // we store curw=A^t (A^tA)^i v for the next iteration since ownership of (A^tA)^i v is lost by sending over the channel
                    let mut curw = a.parallel_sparse_matvec_mul_simd(&local_curv, theprime);

                    for _ in 0..(to_be_produced) {
                        let vec = at.parallel_sparse_matvec_mul_simd(&curw, theprime);
                        curw = a.parallel_sparse_matvec_mul_simd(&vec, theprime);
                        if tx.send(vec).is_err(){
                            eprintln!("Error sending token from worker {}", worker_id);
                            return;
                        };
                    }
                } else { 
                    // same code, but version with serial matvecmul
                    let mut curw = a.serial_sparse_matvec_mul_simd(&local_curv, theprime);

                    for _ in 0..(to_be_produced) {
                        let vec = at.serial_sparse_matvec_mul_simd(&curw, theprime);
                        curw = a.serial_sparse_matvec_mul_simd(&vec, theprime);
                        if tx.send(vec).is_err(){
                            eprintln!("Error sending token from worker {}", worker_id);
                            return;
                        };
                    }
                }
                // println!("Worker {} done", worker_id);
            })
        }).collect();
        SimdVectorStream {
            receivers: rxs,
            workers: _workers,
        }
    }
}
