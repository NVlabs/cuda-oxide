//! Regression test for `ProjectionElem::ConstantIndex { from_end: true }`
//! against a slice — the MIR shape produced by `slice::last()` and the
//! `[.., last]` slice pattern.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Compilation error: invalid input program.
//! Unsupported construct: ConstantIndex with from_end=true not yet supported
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `sec1::point::Tag::compress_y`, whose body destructures the input
//! bytes via the `[.., last]` slice pattern.
//!
//! `from_end: true, offset: O` means index `len - O`. The translator's
//! iterative projection loop hits ConstantIndex AFTER the Deref step
//! has already extracted just the data pointer from the slice — the
//! length is gone and the from_end arithmetic can't be done.
//!
//! ## What landed
//!
//! `translate_place_iterative` now peeks one projection ahead in the
//! `Deref` arm. If the next projection is
//! `ConstantIndex { from_end: true }` AND the current value is a
//! `MirSliceType<T>`, the new helper
//! `apply_slice_deref_constant_index_from_end` handles both
//! projections as a unit:
//!
//! 1. extract data pointer (field 0)
//! 2. extract length (field 1)
//! 3. emit i64 constant for `offset`
//! 4. `mir.sub` for `len - offset`
//! 5. `mir.ptr_offset` for the element address
//!
//! Result is a `MirPtrType<T>` to the element — same shape every other
//! ConstantIndex transform leaves behind, so downstream projections /
//! load / store paths see the expected type.
//!
//! Other from_end bails (single-projection slot path on arrays, the
//! address-build helper, the non-Deref-prefixed iterative case) are
//! left as-is — they aren't on the path the user's code hits, and
//! handling them requires either compile-time-`N` extraction (arrays)
//! or restructuring the address-build helper to keep slice length
//! around.
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
