//! Reproducer + regression test for mir-importer's `Aggregate(RawPtr(...))`
//! limitation.
//!
//! Before the fix:
//!   error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
//!          Translation failed: <kernel>: Compilation error: invalid input program.
//!          Unsupported construct: Aggregate kind RawPtr(Ty { ... Slice(...) }, _)
//!          not yet supported
//!
//! After the fix this example compiles to PTX and prints SUCCESS.
//!
//! Build + run with:
//!   cargo oxide run slice_range
//!
//! The trigger is `src[..half]`: in MIR this lowers via
//! `core::ptr::from_raw_parts(thin_ptr, len)` to an
//! `AggregateKind::RawPtr(Slice(u8), Not)` rvalue that packs the thin data
//! pointer and the runtime length into a fat slice pointer. The catch-all
//! arm in `Rvalue::Aggregate` currently rejects it.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn copy_head(src: &[u8], half: usize, mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            // `&src[..half]` constructs a fat raw slice pointer aggregate
            // before reborrowing as `&[u8]`. That construction is the
            // `AggregateKind::RawPtr(Slice(u8), Not)` the importer rejects.
            let sub: &[u8] = &src[..half];
            *slot = if i < sub.len() { sub[i] } else { 0 };
        }
    }
}

fn main() {
    println!("=== Aggregate(RawPtr(Slice, _)) reproducer ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 16;
    const HALF: usize = 8;
    let host: Vec<u8> = (0u8..N as u8).collect();
    let dev = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .copy_head(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &dev,
            HALF,
            &mut out,
        )
        .expect("Kernel launch failed");

    let r = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = if i < HALF { host[i] } else { 0 };
        assert_eq!(
            r[i], expected,
            "index {}: expected {}, got {}",
            i, expected, r[i]
        );
    }

    println!("SUCCESS: Aggregate(RawPtr(Slice, _)) handled — `&src[..half]` codegen'd to PTX");
}
