//! Regression test for the `core::intrinsics::ptr_offset_from_unsigned`
//! lowering in mir-lower.
//!
//! ## Pre-fix failure
//!
//! Without the handler this failed at llc verification:
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Verification failed for 'llvm module':
//!        Symbol _RINvNtCsXXX_4core10intrinsics24ptr_offset_from_unsigned... not found
//! ```
//!
//! ## What triggers it
//!
//! The kernel calls `<*const u8>::offset_from_unsigned(other_ptr)`,
//! which bottoms out at the rustc compiler intrinsic
//! `core::intrinsics::ptr_offset_from_unsigned`. The intrinsic has
//! no MIR body — rustc's codegen backend normally lowers it
//! directly to ptrtoint/sub/udiv. cuda-oxide skips bodyless
//! callees, so without a handler the call survives into LLVM IR as
//! an undefined symbol.
//!
//! ## The fix
//!
//! Lives in three places, mirroring the existing
//! `convert_rust_bit_intrinsic` / `convert_rust_saturating_intrinsic` /
//! `convert_rust_float_math_intrinsic` chain:
//!
//! - `crates/dialect-mir/src/rust_intrinsics.rs`:
//!   `CALLEE_PTR_OFFSET_FROM_UNSIGNED` placeholder constant.
//! - `crates/mir-importer/src/translator/terminator/intrinsics/ptr_arith.rs`:
//!   recognizes `core::intrinsics::ptr_offset_from_unsigned`, emits
//!   the placeholder `mir.call`.
//! - `crates/mir-lower/src/convert/ops/call.rs`:
//!   `convert_rust_ptr_arith_intrinsic` lowers the placeholder to
//!
//!   ```text
//!   %self_int  = ptrtoint i8* %self   to i64
//!   %orig_int  = ptrtoint i8* %origin to i64
//!   %byte_diff = sub i64 %self_int, %orig_int
//!   %count     = udiv i64 %byte_diff, sizeof(T)   ; elided when sizeof(T) == 1
//!   ```
//!
//!   `sizeof(T)` is recovered by looking up the operand's
//!   most-recent `MirPtrType` and walking its pointee through
//!   `convert_type` + `get_type_size` — same trick
//!   `arithmetic.rs::is_signed_int_op` uses to recover signedness
//!   from pointer operands.
//!
//! Originally surfaced from `~/vanity-miner-rs/` after the iter
//! alias-translation fix landed
//! (commit `Translate IntoIter / Item aliases and stop
//! misclassifying tuple-result calls as unit`) —
//! `core::slice::iter::ChunksExact::next` uses
//! `offset_from_unsigned` for its `len_remaining` bookkeeping, so
//! any kernel that consumes a `ChunksExact` ends up transitively
//! dragging the intrinsic in. The repro here exercises the
//! intrinsic directly so the regression test is self-contained.
//!
//! ## Not yet handled
//!
//! - `core::intrinsics::ptr_offset_from` (signed variant — would
//!   use `sdiv`). Same shape; add when a workload needs it.
//!
//! ## Build with
//!
//!     cargo oxide build ptr_offset_intrinsic
//!
//! Expected: green build, PTX containing a real `sub` (and `div`
//! when pointee size > 1).
//!
//! ## What this example is NOT
//!
//! - Not about `Iterator::Item` / `IntoIterator::IntoIter` /
//!   cross-crate MIR. Those land in the previous commits' regression
//!   tests. This one is a one-file kernel that imports nothing
//!   beyond `cuda_device`.
//! - Not about `#[device]` / `#[cuda_module]` annotations. The
//!   intrinsic lives in `core`, not user code.

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
