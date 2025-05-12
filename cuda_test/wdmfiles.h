#ifndef WDMFILES_H
#define WDMFILES_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <stdexcept>
#include <limits>


template <typename T>
std::tuple<uint32_t, size_t, size_t, size_t> load_wdm_file_sym(
    const std::string& wdm_filename,
    std::vector<T>& row_precond,
    std::vector<T>& col_precond,
    std::vector<std::vector<T>>& v_list,
    std::vector<std::vector<T>>& curv_list,
    std::vector<std::vector<T>>& seq_list
) {
    std::ifstream file(wdm_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + wdm_filename);
    }

    // Read the first line: m n p Nlen num_v
    size_t m, n, nlen, num_v;
    uint32_t p;
    file >> m >> n >> p >> nlen >> num_v;
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Read row_precond
    row_precond.resize(m);
    for (size_t i = 0; i < m; ++i) {
        file >> row_precond[i];
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Read col_precond
    col_precond.resize(n);
    for (size_t i = 0; i < n; ++i) {
        file >> col_precond[i];
    }
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); // Skip to the next line

    // Read v_list
    v_list.clear();
    for (size_t i = 0; i < num_v; ++i) {
        std::vector<T> v(n);
        for (size_t j = 0; j < n; ++j) {
            file >> v[j];
        }
        v_list.push_back(std::move(v));
    }

    // Read curv_list
    curv_list.clear();
    for (size_t i = 0; i < num_v; ++i) {
        std::vector<T> curv(n);
        for (size_t j = 0; j < n; ++j) {
            file >> curv[j];
        }
        curv_list.push_back(std::move(curv));
    }

    // Read seq_list
    seq_list.clear();
    size_t seq_count = num_v * (num_v + 1) / 2;
    for (size_t i = 0; i < seq_count; ++i) {
        std::vector<T> seq(nlen);
        for (size_t j = 0; j < nlen; ++j) {
            file >> seq[j];
        }
        seq_list.push_back(std::move(seq));
    }

    // Ensure all vectors are of the correct size
    if (row_precond.size() != m) {
        throw std::runtime_error("Row preconditioner length does not match matrix rows");
    }
    if (col_precond.size() != n) {
        throw std::runtime_error("Column preconditioner length does not match matrix columns");
    }

    return std::make_tuple(p, m, n, num_v);
}



template <typename T>
void save_wdm_file_sym(
    const std::string& wdm_filename,
    size_t n_rows,
    size_t n_cols,
    T theprime,
    const std::vector<T>& row_precond,
    const std::vector<T>& col_precond,
    const std::vector<std::vector<T>>& v_list,
    const std::vector<std::vector<T>>& curv_list,
    const std::vector<std::vector<T>>& seq_list
) {
    std::ofstream file(wdm_filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + wdm_filename);
    }

    // Write the first line: m n p Nlen num_u num_v
    file << n_rows << " " << n_cols << " " << theprime << " " 
         << seq_list[0].size() << " " << v_list.size() << "\n";

    // Write the second line: row_precond
    for (size_t i = 0; i < row_precond.size(); ++i) {
        if (i > 0) file << " ";
        file << row_precond[i];
    }
    file << "\n";

    // Write the third line: col_precond
    for (size_t i = 0; i < col_precond.size(); ++i) {
        if (i > 0) file << " ";
        file << col_precond[i];
    }
    file << "\n";

    // Write the v_list
    for (const auto& vv : v_list) {
        auto v = prettify_vect(vv, theprime);
        for (size_t i = 0; i < v.size(); ++i) {
            if (i > 0) file << " ";
            file << v[i];
        }
        file << "\n";
    }

    // Write the curv_list
    for (const auto& curvv : curv_list) {
        auto curv = prettify_vect(curvv, theprime);
        for (size_t i = 0; i < curv.size(); ++i) {
            if (i > 0) file << " ";
            file << curv[i];
        }
        file << "\n";
    }

    // Write the seq_list
    for (const auto& seq : seq_list) {
        auto seq_pretty = prettify_vect(seq, theprime);
        for (size_t i = 0; i < seq_pretty.size(); ++i) {
            if (i > 0) file << " ";
            file << seq_pretty[i];
        }
        file << "\n";
    }

    file.close();
    if (!file) {
        throw std::runtime_error("Failed to write to file: " + wdm_filename);
    }
}



#endif // WDMFILES_H