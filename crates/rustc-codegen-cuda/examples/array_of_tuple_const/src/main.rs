//! Regression test for unsupported `const ARR: [(T, U, ...); N]`.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Unsupported construct: translate_array_constant: unsupported
//!   element type: MirTupleType { types: [Ptr<i64>, Ptr<i64>, Ptr<i64>] }
//! ```
//!
//! Surfaced from vanity-miner-rs's `check_arith_divrem_by_58_pow_5`
//! bisection slot, which bundled `(input, expected_q, expected_r)`
//! per test case with `const CASES: [(u64, u64, u64); 6] = [..]`.
//! Common Rust pattern; will recur for any consumer that uses
//! tuple-typed test-vector tables.
//!
//! ## Root cause
//!
//! [crates/mir-importer/src/translator/rvalue.rs::translate_array_constant]
//! decodes array-constant bytes only for integer / float / FP16
//! element types. Tuple element types fall through to the
//! "unsupported" branch. A complete fix needs to recursively
//! translate each per-element byte slice as a tuple value (similar
//! to `translate_tuple_constant`, but starting from raw bytes
//! instead of a `ConstOperand`), then assemble the array.
//!
//! ## Workaround for downstream code today
//!
//! Replace `const ARR: [(A, B, C); N]` with N parallel arrays of
//! the same length:
//!
//! ```ignore
//! const INPUTS:    [u64; 6] = [..];
//! const EXPECTED_Q: [u64; 6] = [..];
//! const EXPECTED_R: [u64; 6] = [..];
//! ```
//!
//! These are arrays of primitives and go through the supported
//! integer-element branch.
//!
//! ## Build
//!
//!     cargo oxide build array_of_tuple_const   # fails today

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Indexes a `[(u64, u64, u64); 4]` constant by a runtime-supplied
    /// case index, writing the three tuple fields to out[0..3]. The
    /// kernel never actually runs today — the build fails at codegen
    /// before the binary is produced.
    #[kernel]
    pub fn run(args: &[u64], mut out: DisjointSlice<u64>) {
        const CASES: [(u64, u64, u64); 4] = [
            (10, 100, 1000),
            (20, 200, 2000),
            (30, 300, 3000),
            (40, 400, 4000),
        ];

        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < 3
            && !args.is_empty()
        {
            let case_idx = (args[0] as usize) & 3;
            let (a, b, c) = CASES[case_idx];
            *slot = match i {
                0 => a,
                1 => b,
                2 => c,
                _ => 0,
            };
        }
    }
}

fn main() {
    println!("=== array_of_tuple_const ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let args = DeviceBuffer::from_host(&stream, &[2u64]).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 3).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(3), &args, &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    assert_eq!(result, [30, 300, 3000]);
    println!("SUCCESS: array-of-tuple constant indexes correctly");
}
