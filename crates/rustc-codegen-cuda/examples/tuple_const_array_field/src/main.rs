//! Regression test for unsupported `const T: ([U; N], V) = (..)`.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Unsupported construct: Tuple constant field 0 has unsupported
//!   type MirArrayType { element_ty: ..., size: 10 }
//! ```
//!
//! Surfaced from vanity-miner-rs's `check_dynamic_index_write`
//! self-test slot, which used `const fn run_growth() -> ([u32; 10],
//! usize)` to host-compute the expected base58-divrem growth-loop
//! result, then unpacked the tuple at consumption sites. Common
//! Rust idiom: any `const fn` that needs to return more than one
//! value naturally tuples them.
//!
//! ## Root cause
//!
//! [crates/mir-importer/src/translator/rvalue.rs::translate_tuple_constant]
//! looks up each field's byte size via `constant_storage_size`,
//! which handles only integer / float / FP16 / pointer leaves.
//! For an `MirArrayType` field, the lookup returns `None` and
//! translation bails with the "unsupported type" diagnostic above.
//!
//! Symmetric to `array_of_tuple_const`: same compositional gap,
//! flipped. A complete fix needs the tuple-translator to recurse
//! into aggregate field types (array, nested tuple, ADT) by
//! splitting the constant's raw bytes against per-field layouts
//! and re-entering the appropriate constant-translation path.
//!
//! ## Workaround for downstream code today
//!
//! Split the tuple const into N parallel consts of homogeneous
//! type, each individually supported:
//!
//! ```ignore
//! const EXPECTED_LIMBS: [u32; 10] = ..;
//! const EXPECTED_COUNT: usize = ..;
//! ```
//!
//! Each leaf type already round-trips through the codegen.
//!
//! ## Build
//!
//!     cargo oxide build tuple_const_array_field   # fails today

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Reads a `const` of type `([u32; 4], u32)` and writes the
    /// array elements plus the trailing count to `out`. The kernel
    /// never actually runs today — codegen fails on the tuple
    /// constant before PTX is emitted.
    #[kernel]
    pub fn run(mut out: DisjointSlice<u32>) {
        const TABLE: ([u32; 4], u32) = ([10, 20, 30, 40], 4);

        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            *slot = match i {
                0..=3 => TABLE.0[i],
                4 => TABLE.1,
                _ => 0,
            };
        }
    }
}

fn main() {
    println!("=== tuple_const_array_field ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let mut out = DeviceBuffer::<u32>::zeroed(&stream, 5).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(5), &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    assert_eq!(result, [10, 20, 30, 40, 4]);
    println!("SUCCESS: tuple-of-array constant indexes correctly");
}
