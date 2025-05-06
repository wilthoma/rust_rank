// Constants (can be passed via uniforms)
struct Params {
    n_rows: u32,
    n_vecs: u32,
    n_cols: u32,
}

@group(0) @binding(0) var<storage, read> values: array<u32>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(3) var<storage, read> B: array<u32>;     // col-major
@group(0) @binding(4) var<storage, read_write> C: array<atomic<u32>>; // row-major
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.n_rows) {
        return;
    }

    let row_start = row_ptr[row];
    let row_end = row_ptr[row + 1];

    // For each output vector (column of B)
    for (var k = 0u; k < params.n_vecs; k = k + 1u) {
        var acc = 0u;

        for (var idx = row_start; idx < row_end; idx = idx + 1u) {
            let col = col_indices[idx];
            let a_ij = values[idx];
            let b_jk = B[col + k * params.n_cols];  // column-major
            acc = acc + a_ij * b_jk;
        }

        // C[row][k] = acc  (row-major)
        let out_index = row * params.n_vecs + k;
        atomicStore(&C[out_index], acc);
    }
}