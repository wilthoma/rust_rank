
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <stdexcept>
#include "include/CLI11.hpp"
#include "ntt.h"
#include "poly_mat_mul.h" 
#include "modular_linalg.h"
#include "sigma_basis.h"
#include "wdmfiles.h"
#include "cudantt.h"
#include "cupoly_mat_mul.h"


using namespace std;

typedef uint32_t myint;

int main(int argc, char** argv) {
    CLI::App app{"CUDA PolyFinder"};
    string wdm_filename;
    bool runtests = false;

    app.add_option("filename", wdm_filename, "WDM file to load")->required();
    app.add_flag("-t,--test", runtests, "Run tests");
    CLI11_PARSE(app, argc, argv);
    
    if (runtests) {
        cout << "Running tests..." << endl;
        test_ntt_inverse();
        test_modmul_agreement();
        // benchmark_ntt_u64_vs_ntt_u128();
        test_poly_mul_naive_vs_fft();
        test_poly_mat_mul_naive_vs_fft();
        test_runall_modular_linalg();
        test_bit_reverse_cuda();
        test_modmul_cuda();
        //test_cudantt_small();
        test_ntt_cuda_inv();
        test_ntt_cuda_same_as_ntt();
        test_cuda_poly_mul_methods();
        return 0;
    }

    // check if wdm file exists
    if (!std::filesystem::exists(wdm_filename)) {
        cerr << "Error: WDM file not found: " << wdm_filename << endl;
        return 1;
    }
    
    cout << "Loading WDM file: " << wdm_filename << endl;
    // Load the WDM file
    vector<myint> row_precond, col_precond;
    vector<vector<myint>> v_list, curv_list, seq_list;
    uint32_t p;
    size_t m, n, nlen, num_v;
    std::tie(p, m, n, num_v) = load_wdm_file_sym<myint>(wdm_filename, row_precond, col_precond, v_list, curv_list, seq_list);
    nlen = seq_list[0].size();
    cout << "Loaded WDM file: " << wdm_filename << endl;
    cout << "p: " << p << ", m: " << m << ", n: " << n << ", num_v: " << num_v << endl;
    // print number of entries in seq list
    cout << "Number of entries in sequence(s): " << nlen << endl;

    size_t thed = 1;
    while (thed <= seq_list[0].size()) {
        thed *= 2;
    }
    thed /= 2;

    auto [pmb, del] = PM_Basis<uint64_t, myint>(seq_list, thed, p);
    analyze_pm_basis<uint64_t>(pmb, del, p);
    

    return 0;
}