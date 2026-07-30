// SPDX-License-Identifier: Apache-2.0

//! Runtime regression for pointer provenance in a direct struct constant.
//!
//! The pointer field's stored bytes contain the addend into the target
//! allocation, while the allocation's provenance table identifies the Rust
//! static being referenced. The importer must combine both pieces, materialize
//! a pointer to the corresponding device global, and preserve that pointer when
//! the struct constant is consumed by GPU code.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};

static FIRST: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

pub struct Holder {
    pub pointer: &'static [u8; 16],
    pub flag: bool,
}

const DIRECT: Holder = Holder {
    pointer: &FIRST,
    flag: true,
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn direct_struct_pointer(mut output: DisjointSlice<u8>) {
        if let Some((slot, index)) = output.get_mut_indexed() {
            let holder = DIRECT;
            *slot = holder.pointer[index.get() & 15] + holder.flag as u8;
        }
    }
}

fn main() {
    println!("=== struct_constant_provenance ===");

    const N: usize = 64;

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let mut output =
        DeviceBuffer::<u8>::zeroed(&stream, N).expect("Failed to allocate device output buffer");

    // SAFETY: the launch is one-dimensional and the output buffer contains one
    // element for every launched thread.
    unsafe {
        module.direct_struct_pointer(&stream, LaunchConfig::for_num_elems(N as u32), &mut output)
    }
    .expect("direct_struct_pointer launch failed");

    let actual = output
        .to_host_vec(&stream)
        .expect("Failed to copy device output to host");
    let expected: Vec<u8> = (0..N)
        .map(|index| FIRST[index & 15] + u8::from(DIRECT.flag))
        .collect();

    assert_eq!(
        actual, expected,
        "struct constant pointer provenance produced incorrect GPU output"
    );

    println!("PASS: struct constant pointer provenance preserved at runtime");
}
