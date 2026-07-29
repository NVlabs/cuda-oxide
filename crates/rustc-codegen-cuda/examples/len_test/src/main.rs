/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Minimal test for Rvalue::Len codegen.
//!
//! Each output element is set to the length of the input slice,
//! forcing the compiler to translate a `.len()` call on device code.
//!
//! Build and run with:
//!   cargo oxide run len_test

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn len_test(input: &[f32], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let len = input.len() as u64;
        if let Some(o) = out.get_mut(idx) {
            *o = len;
        }
    }
}

fn main() {
    println!("=== Rvalue::Len Test ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();

    println!("Input slice length: {}", N);

    let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();
    let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module.len_test(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input_dev,
            &mut out_dev,
        )
    }
    .expect("Kernel launch failed");

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    println!("Output (first 5 elements): {:?}", &out_host[0..5]);

    let mut errors = 0;
    for (i, &val) in out_host.iter().enumerate() {
        if val != N as u64 {
            if errors < 5 {
                eprintln!("  Error at [{}]: expected {}, got {}", i, N, val);
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("\n✓ SUCCESS: All {} elements report correct slice length!", N);
    } else {
        println!("\n✗ FAILED: {} errors", errors);
        std::process::exit(1);
    }
}
