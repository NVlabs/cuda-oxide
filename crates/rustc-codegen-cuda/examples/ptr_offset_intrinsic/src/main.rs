//! Known-failure reproducer for the missing
//! `core::intrinsics::ptr_offset_from_unsigned` handler in mir-lower.
//!
//! ## Expected failure
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Verification failed for 'llvm module':
//!        Symbol _RINvNtCsXXX_4core10intrinsics24ptr_offset_from_unsigned... not found
//! ```
//!
//! ## What triggers it
//!
//! The kernel calls
//! `<*const u8>::offset_from_unsigned(other_ptr)`, which after
//! macro expansion bottoms out at the rustc compiler intrinsic
//! `core::intrinsics::ptr_offset_from_unsigned`. The intrinsic has
//! no MIR body (it's a compiler-builtin lowered directly by rustc's
//! codegen backend). The cuda-oxide collector skips bodyless
//! intrinsics; the call site in the kernel survives into LLVM IR as
//! a `call.uni` to a symbol nothing defines, and `llc` rejects the
//! module.
//!
//! ## What it would take to fix
//!
//! Parallel to how `convert_rust_bit_intrinsic` /
//! `convert_rust_saturating_intrinsic` /
//! `convert_rust_float_math_intrinsic` are dispatched in
//! `crates/mir-lower/src/convert/ops/call.rs` (search "from_placeholder_callee"
//! for the dispatch chain at the top of `convert`).
//!
//! The lowering for `ptr_offset_from_unsigned(p, origin)` is:
//!
//! ```text
//! %p_int     = ptrtoint i8* %p     to i64
//! %orig_int  = ptrtoint i8* %origin to i64
//! %diff      = sub i64 %p_int, %orig_int
//! %count     = udiv i64 %diff, <size_of<T>>     ; T is from intrinsic's generic arg
//! ```
//!
//! (Or, equivalently, lower to `llvm.usub.sat`-style with explicit
//! bounds checks; standard rustc lowers via `pointer_arith`.)
//!
//! A handler likely lives next to `convert_rust_bit_intrinsic` and
//! follows the same callee-name placeholder pattern. The trickiest
//! part is extracting the pointee size: the intrinsic is generic
//! over `T` (`<*const T>::offset_from_unsigned`), so the handler
//! needs to read the type argument from the monomorphized symbol
//! name (or, better, from MIR generic args if the importer
//! preserves them on the placeholder call).
//!
//! Originally surfaced from `~/vanity-miner-rs/` after the iter
//! alias-translation fix landed (commit
//! `Translate IntoIter / Item aliases and stop misclassifying
//! tuple-result calls as unit`) — `core::slice::iter::ChunksExact::next`
//! uses `offset_from_unsigned` to compute its `len_remaining`
//! bookkeeping, so any kernel that consumes a `ChunksExact` ends up
//! transitively dragging the intrinsic in.
//!
//! ## Why this is its own example
//!
//! The previous walls in this thread —
//! `examples/cross_crate_pubfn/` (MIR availability cross-crate) and
//! `examples/iter_zip_chunks_exact/` (alias type translation +
//! tuple-result call lowering) — are now regression-tested. This
//! one is the next layer down: a compiler-intrinsic dispatch gap in
//! mir-lower. Each peeled layer has revealed the next; keeping them
//! as separate reproducers means a future contributor can flip each
//! from known-failure to passing independently.
//!
//! ## Build with
//!
//!     cargo oxide build ptr_offset_intrinsic
//!
//! Expected: build error from `llc` —
//! `Symbol ..ptr_offset_from_unsigned.. not found`.
//!
//! ## What this example is NOT
//!
//! - Not about `Iterator::Item` / `IntoIterator::IntoIter` / cross-
//!   crate MIR. Those are fixed; this fails on a one-file kernel
//!   that imports nothing beyond `cuda_device`.
//! - Not about `#[device]` / `#[cuda_module]` annotations. The
//!   missing symbol is in `core`, not user code.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// FAILS: `p2.offset_from_unsigned(p1)` lowers to the
    /// `core::intrinsics::ptr_offset_from_unsigned` intrinsic, which
    /// mir-lower does not have a handler for.
    #[kernel]
    pub fn ptr_offset_demo(input: &[u8], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            let p1 = input.as_ptr();
            // SAFETY: i < input.len() is guaranteed by the launch config
            // matching the input slice length; only thread `i` reads
            // `input[i]` here.
            let p2 = unsafe { p1.add(i) };
            let diff = unsafe { p2.offset_from_unsigned(p1) };
            *slot = diff as u8;
        }
    }
}

fn main() {
    println!("=== ptr_offset_intrinsic ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 64;
    let host: Vec<u8> = (0..N as u8).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .ptr_offset_demo(&stream, LaunchConfig::for_num_elems(N as u32), &input, &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        assert_eq!(result[i], i as u8);
    }
    println!("SUCCESS: ptr_offset_from_unsigned lowered to PTX");
}
