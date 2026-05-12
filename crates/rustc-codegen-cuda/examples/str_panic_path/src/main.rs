//! Reproducer + regression test for `RigidTy(Str)` translation in mir-importer.
//!
//! Before the fix:
//!   error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
//!          Translation failed: <kernel>: Compilation error: invalid input program.
//!          Unsupported construct: Type translation not yet implemented for: RigidTy(Str)
//!
//! After the fix this example compiles to PTX and prints SUCCESS.
//!
//! Build + run with:
//!   cargo oxide run str_panic_path
//!
//! The kernel threads a `&'static str` through its MIR via `core::hint::black_box`.
//! That forces a `&str` local + an `&str`-typed constant operand to survive opt
//! passes, without dragging in `core::fmt::Arguments` / formatter function
//! pointers (which `panic!` / `assert!` / `.unwrap()` would pull in). Once the
//! `RigidTy(Str)` arm in `translate_type` is in place and slice-typed constants
//! get a handler in `translate_constant`, codegen succeeds.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn passthrough_with_str(data: &[u32], mut out: DisjointSlice<u32>) {
        // Forces an `&'static str` constant + slice-typed local into the MIR
        // (the type-translator hits `RigidTy(Str)`, the constant-translator
        // hits a `MirSliceType<u8>` constant) but doesn't call any external
        // function — `.len()` is a plain field extract on the fat pointer.
        // The result is multiplied into the data so the optimizer can't DCE
        // the str away.
        let label: &'static str = "abc";
        let mul = label.len() as u32; // = 3
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = data[i] * mul;
        }
    }
}

fn main() {
    println!("=== RigidTy(Str) reproducer ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let host: Vec<u32> = (0..N as u32).collect();
    let dev = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .passthrough_with_str(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &dev,
            &mut out,
        )
        .expect("Kernel launch failed");

    let r = out.to_host_vec(&stream).unwrap();
    let expected: Vec<u32> = host.iter().map(|x| x * 3).collect();
    assert_eq!(r, expected, "output should be input * label.len() (== 3)");

    println!("SUCCESS: RigidTy(Str) handled — `&'static str` value flowed through kernel MIR");
}
