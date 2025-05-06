use wgpu::{util::DeviceExt, wgc::pipeline};
use crate::matrices::*;
use bytemuck::{Pod, Zeroable};
use std::time::Instant;
use rand::Rng;
use std::borrow::Cow;


#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n_rows: u32,
    n_vecs: u32,
    n_cols: u32,
}


pub async fn csr_dense_mult(
    a: &CsrMatrix<u32>,
    at: &CsrMatrix<u32>,
    B: &[u32],
    n_vecs: usize,
) -> Vec<u32> 
{
    let n_rows = a.n_rows;
    let n_cols = a.n_cols;

    // Initialize wgpu
    let instance = wgpu::Instance::default();
    let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();

    // Create buffers and bind groups
    let device = &device;
    let queue = &queue;


    // Flatten B (dense input) as column-major
    let b_flat: Vec<u32> = (0..n_cols).flat_map(|j| (0..n_vecs).map(move |k| B[j + k * n_cols])).collect();

    let b_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Dense Matrix B"),
        contents: bytemuck::cast_slice(&b_flat),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // Output buffer (after first and second multiplication)
    let output_intermediate = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Intermediate Output"),
        size: (a.n_rows * n_vecs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let output_final = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Final Output"),
        size: (a.n_cols * n_vecs * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let output_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Buffer"),
        size: (a.n_cols * n_vecs * 4) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
    //     label: Some("Params Buffer"),
    //     contents: bytemuck::bytes_of(&params),
    //     usage: wgpu::BufferUsages::UNIFORM,
    // });

    // Allocate CSR buffers on GPU
    let csr_buffers = upload_csr_to_gpu(&device, &a, n_vecs);
    let csr_t_buffers = upload_csr_to_gpu(&device, &at, n_vecs);

    let csr_pipeline = create_csr_pipeline(&device, &csr_buffers, &b_buf, &output_intermediate, "src/spmm_mul.wgsl");
    let csr_t_pipeline = create_csr_pipeline(&device, &csr_t_buffers, &output_intermediate, &output_final, "src/spmm_mul.wgsl");

    // Dispatch computation
    println!("Starting shaders...");

    let start = Instant::now();
    run_csr_multiplication(&device, &queue, &csr_pipeline, a.n_rows).await;
    println!("First done...");
    run_csr_multiplication(&device, &queue, &csr_t_pipeline,at.n_rows).await;
    let duration = start.elapsed();
    println!("Inner execution time: {:?}", duration);

    // Copy result to readback buffer
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Readback Encoder") });
    encoder.copy_buffer_to_buffer(&output_final, 0, &output_readback, 0, (a.n_cols * n_vecs * 4) as u64);
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::Wait).unwrap();

    // Read buffer
    let buffer_slice = output_readback.slice(..);
    buffer_slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait).unwrap();
    let data = buffer_slice.get_mapped_range();
    let result: &[u32] = bytemuck::cast_slice(&data);

    result.to_vec()

}

struct CsrPipeline {
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
}

fn create_csr_pipeline(device: &wgpu::Device, csr_bufs: &GpuCsrBuffers,    input_buf: &wgpu::Buffer,
    output_buf: &wgpu::Buffer, shader_src: &str) -> CsrPipeline {
    // Load shader
    let shader_src = std::fs::read_to_string(shader_src).unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Matrix Shader"),
        source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&shader_src)),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Bind Group Layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry { 
                binding: 5, 
                visibility: wgpu::ShaderStages::COMPUTE, 
                ty: wgpu::BindingType::Buffer { 
                    ty: wgpu::BufferBindingType::Uniform, 
                    has_dynamic_offset: false, 
                    min_binding_size: None 
                },
                count: None 
            },

        ],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: csr_bufs.val_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: csr_bufs.col_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: csr_bufs.row_ptr_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: input_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: output_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: csr_bufs.params_buf.as_entire_binding() },
        ],
        label: Some("Matrix Bind Group"),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("CSR Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        cache: None,
        compilation_options: wgpu::PipelineCompilationOptions::default(),
    });

    // let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
    //     label: Some("Compute Encoder"),
    // });

    CsrPipeline { bind_group_layout, bind_group, pipeline: compute_pipeline }
}

async fn run_csr_multiplication(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &CsrPipeline,
    n_rows: usize,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
            label: Some("Compute Pass"), 
            timestamp_writes: None 
        });
        cpass.set_pipeline(&pipeline.pipeline);
        cpass.set_bind_group(0, &pipeline.bind_group, &[]);
        let x_groups = 65535;
        let y_groups = (n_rows as u32 + x_groups - 1) / x_groups;
        cpass.dispatch_workgroups(x_groups.min(n_rows as u32), y_groups, 1);
        // cpass.dispatch_workgroups( n_rows as u32, 1, 1);
    }
    queue.submit(Some(encoder.finish()));
    device.poll(wgpu::PollType::Wait).unwrap();
}

struct GpuCsrBuffers {
    val_buf: wgpu::Buffer,
    col_buf: wgpu::Buffer,
    row_ptr_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
}

fn upload_csr_to_gpu(device: &wgpu::Device, csr: &CsrMatrix<u32>, n_vecs : usize) -> GpuCsrBuffers {
    let val_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSR Values"),
        contents: bytemuck::cast_slice(&csr.values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let col_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSR Col Indices"),
        contents: bytemuck::cast_slice(&csr.col_indices),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let row_ptr_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSR Row Ptr"),
        contents: bytemuck::cast_slice(&csr.row_ptr),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let params = Params {
        n_rows: csr.n_rows as u32,
        n_vecs: n_vecs as u32,
        n_cols: csr.n_cols as u32,
    };
    let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("CSR Params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });


    GpuCsrBuffers { val_buf, col_buf, row_ptr_buf, params_buf }
}

#[cfg(test)]
mod tests {
    use super::*;

    

#[test]
fn test_csr_dense_mult_with_sms() {
    // Load CSR matrix from SMS file
    let p = 257u32;
    let csr = CsrMatrix::load_csr_matrix_from_sms("data/contractD12_10.txt", p).unwrap();
    let csrt = csr.transpose();
    let n_vecs = 16;

    println!("CSR Matrix loaded with {} rows and {} columns", csr.n_rows, csr.n_cols);

    // Create a dense matrix B
    let mut rng = rand::rng();
    let b: Vec<u32> = (0..csr.n_cols * n_vecs).map(|_| rng.random_range(0..p)).collect();

    let n_vecs = 4;

    // Measure execution time
    let start = Instant::now();
    let result = pollster::block_on(csr_dense_mult(&csr, &csrt, &b, n_vecs));
    let duration = start.elapsed();

    // Print the result and execution time
    // println!("Result: {:?}", result);
    println!("Execution time: {:?}", duration);
}
}
