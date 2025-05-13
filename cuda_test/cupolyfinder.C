
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


using namespace std;



int main(int argc, char** argv) {
    CLI::App app{"CUDA PolyFinder"};
    string wdm_filename;
    app.add_option("filename", wdm_filename, "WDM file to load")->required();
    CLI11_PARSE(app, argc, argv);
    cout << "Loading WDM file: " << wdm_filename << endl;
    // TODO

    cout << "Running tests..." << endl;
    test_ntt_inverse();
    test_modmul_agreement();
    // benchmark_ntt_u64_vs_ntt_u128();
    test_poly_mul_naive_vs_fft();
    test_poly_mat_mul_naive_vs_fft();
    test_runall_modular_linalg();

    return 0;
}