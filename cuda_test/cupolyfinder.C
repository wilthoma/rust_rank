
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <stdexcept>
#include "include/CLI11.hpp"
#include "ntt.h"


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


    return 0;
}