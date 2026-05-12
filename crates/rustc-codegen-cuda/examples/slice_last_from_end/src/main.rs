//! Known-failure repro for `ProjectionElem::ConstantIndex { from_end: true }`.
//!
//! ## Wall (current state)
//!
//! ```text
//! Compilation error: invalid input program.
//! Unsupported construct: ConstantIndex with from_end=true not yet supported
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `sec1::point::Tag::compress_y`, whose body destructures the input
//! bytes via the `[.., last]` slice pattern (or calls `slice::last()`),
//! both of which lower to `ConstantIndex { offset: 1, min_length: 1,
//! from_end: true }`.
//!
//! The `from_end` projection means "count from the back": with
//! `offset: 1` the index is `len - 1` (the last element). For arrays
//! `len` is the compile-time `N`. For slices `len` is the runtime
//! length stored in the fat pointer.
//!
//! ## Where it bails
//!
//! Three sites in `crates/mir-importer/src/translator/rvalue.rs`:
//!
//! 1. The single-projection slot path (`local[const_idx_from_end]`)
//!    around line 2968.
//! 2. The address-build helper around line 3470 (used by multi-step
//!    projections lowering to addresses).
//! 3. The iterative value-materialisation path around line 3808.
//!
//! All three bail unconditionally on `from_end=true`.
//!
//! ## What a fix needs to do
//!
//! * Array case (compile-time `N` known from the type): translate
//!   `from_end: true, offset: O` to a constant index `N - O` and use
//!   the existing path.
//! * Slice case: extract the slice's len at runtime (field 1 of the
//!   fat pointer), subtract `offset`, then GEP and load.
//!
//! ## Build with
//!
//!     cargo oxide build slice_last_from_end

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// `#[inline(never)]` shim so the `slice::last()` body's
/// `ConstantIndex { from_end: true }` projection survives to the
/// importer instead of being inlined into the caller (where the
/// optimizer can fold it).
#[inline(never)]
fn last_or_zero(s: &[u8]) -> u8 {
    match s.last() {
        Some(&v) => v,
        None => 0,
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn slice_last(input: &[u8], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && (i + 1) * 4 <= input.len()
        {
            let base = i * 4;
            let window = &input[base..base + 4];
            *slot = super::last_or_zero(window);
        }
    }
}

fn main() {
    println!("=== slice_last_from_end ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u8> = (0..(N * 4) as u8).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .slice_last(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = host[i * 4 + 3];
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: ConstantIndex from_end=true codegen'd to PTX");
}
