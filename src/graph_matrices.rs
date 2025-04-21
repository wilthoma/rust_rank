

pub fn reorder_matrix(a: &CsrMatrix, rowfilename: &str, colfilename: &str) -> CsrMatrix {
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
