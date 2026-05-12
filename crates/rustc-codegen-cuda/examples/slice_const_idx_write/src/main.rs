//! Reproducer + regression test for mir-importer's `Deref -> ConstantIndex`
//! 2-level projection on the LHS of an assignment.
//!
//! Before the fix:
//!   error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
//!          Translation failed: <kernel>: Compilation error: invalid input program.
//!          Unsupported construct: 2-level projection Deref ->
//!          ConstantIndex { offset: 0, min_length: 1, from_end: false } not
//!          yet implemented for assignment
//!
//! After the fix this example compiles to PTX and prints SUCCESS.
//!
//! Build + run with:
//!   cargo oxide run slice_const_idx_write
//!
//! The trigger is `out[K] = v` where `out: &mut [u32]` and `K` is a constant.
//! In MIR the place is `(*out)[ConstantIndex { offset: K, ... }]`, a Deref
//! followed by ConstantIndex. The single-level ConstantIndex arm in
//! `statement.rs` only handles array slots; this two-level form needs to
//! load the slice fat pointer, extract its data pointer, GEP to the offset,
//! and store there.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn poke_first(out: &mut [u32]) {
        let idx = thread::index_1d();
        // Only thread 0 writes — keeps the kernel single-shot for the test.
        if idx.get() == 0 {
            // The MIR LHS here is `(*out)[ConstantIndex { offset: 0, .. }]` —
            // Deref then ConstantIndex assignment.
            out[0] = 0xCAFE_u32;
            out[1] = 0xBABE_u32;
        }
    }
}

fn main() {
    println!("=== Deref -> ConstantIndex assignment reproducer ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 4;
    let mut dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .poke_first(&stream, LaunchConfig::for_num_elems(N as u32), &mut dev)
        .expect("Kernel launch failed");

    let r = dev.to_host_vec(&stream).unwrap();
    assert_eq!(r[0], 0xCAFE);
    assert_eq!(r[1], 0xBABE);
    assert_eq!(r[2], 0);
    assert_eq!(r[3], 0);

    println!("SUCCESS: Deref -> ConstantIndex assignment handled — `out[K] = v` codegen'd to PTX");
}
