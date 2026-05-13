//! Regression test for orphan `AddressOfOp` SSA reference on
//! runtime-indexed `&'static [T; N]` reads.
//!
//! ## Pre-fix wall
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//! failed: ... llc failed: use of undefined value '%vN'
//!   %v8 = load [5 x i64], ptr addrspace(1) %vN
//! ```
//!
//! The function references an SSA value (`%vN`) for the static's
//! address but no operation in the function defines it. The global
//! itself (`@__device_global_0 = addrspace(1) global [N x i8] c"..."`)
//! is correctly emitted at module scope; only the in-function
//! pointer-resolution instruction is missing.
//!
//! Surfaced from vanity-miner-rs's slot 76 `check_static_u64_array_lookup`,
//! which is the minimal probe for the "static-multi-byte-element"
//! family that takes down slots 2/3/11/12/41/42/71/72 (dalek
//! `Scalar52::L/R/RR` reads, base58 `DIVISORS: [u64; 5]`, and their
//! downstream cascades). Any `static T: [U; N] = [...]` with
//! runtime indexing tripped the same bug.
//!
//! ## Root cause
//!
//! [crates/dialect-llvm/src/export.rs] pre-passes op-results to
//! assign SSA names before emitting instruction text. `ConstantOp`
//! was special-cased: the pre-pass mapped its result directly to a
//! literal (e.g. `18446744073709551615`), so consumers in
//! earlier-iterated blocks resolved to the literal instead of a
//! stale `%vN`.
//!
//! `AddressOfOp` is "virtual" in the same way — its export emits
//! no instruction text, just remaps its result to `@global_name`.
//! But it WAS NOT pre-passed. mir-importer's static-translation
//! path inserts the `MirGlobalAllocOp` near the use site
//! (`prev_op`); when the use is inside a conditional chain (e.g.
//! a bounds-check-success block), the use's block sometimes comes
//! BEFORE the AddressOfOp's block in iteration order. The use was
//! exported first, got the stale `%vN` placeholder, and the
//! AddressOfOp's later overwrite of value_names never reached the
//! already-printed load.
//!
//! ## What landed
//!
//! Added an `AddressOfOp` arm to the pre-pass in `export_function`,
//! mirroring the `ConstantOp` special case: insert
//! `(result -> "@<global_name>")` into `value_names` before any
//! instruction is exported. Now every consumer prints the correct
//! global name regardless of block iteration order.
//!
//! ## Build
//!
//!     cargo oxide build static_u64_array_load

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

static STATIC_U64_TABLE: [u64; 5] = [
    0x0123_4567_89AB_CDEF,
    0xFEDC_BA98_7654_3210,
    0x1111_2222_3333_4444,
    0xAAAA_BBBB_CCCC_DDDD,
    0xDEAD_BEEF_CAFE_BABE,
];

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            let i = core::hint::black_box(3usize);
            *slot = STATIC_U64_TABLE[i];
        }
    }
}

fn main() {
    println!("=== static_u64_array_load ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 1).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(1), &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], 0xAAAA_BBBB_CCCC_DDDD);
    println!("SUCCESS: static [u64; 5] read at runtime index");
}
