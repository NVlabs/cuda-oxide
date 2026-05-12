//! Regression test for `mir.field_addr` accepting tuple pointees,
//! not just struct pointees.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Verification failed:
//!   MirFieldAddrOp pointer must point to a struct type, got:
//!   mir.tuple <mir.struct <ProjectivePoint, ...>, mir.struct <Scalar, ...>>
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `k256::arithmetic::mul::lincomb`'s inner closure at
//! `mul.rs:317:43`, which closes over `&[(ProjectivePoint, Scalar)]`.
//! Inside the closure body, `&pair.0` (taking the address of the
//! first tuple element) lowers to a `mir.field_addr` whose pointee
//! is the tuple `(ProjectivePoint, Scalar)`. Both `MirStructType`
//! and `MirTupleType` are positionally-indexed aggregates; rejecting
//! tuples was a structural omission, not a semantic distinction.
//!
//! ## What landed
//!
//! Two coupled changes:
//!
//! 1. `dialect-mir/src/ops/aggregate.rs` — `MirFieldAddrOp::verify`
//!    now accepts `MirTupleType` pointees in addition to
//!    `MirStructType`. Index bounds and result-type checks share
//!    the same code path because the field-types list is computed
//!    once from whichever aggregate matched.
//!
//! 2. `mir-lower/src/convert/ops/aggregate.rs` — `convert_field_addr`
//!    dispatches on tuple vs struct. Tuples have no `mem_to_decl`
//!    reordering (declaration order == memory order), so we
//!    synthesize the identity mapping and reuse the existing
//!    GEP-emission path.
//!
//! ## Build with
//!
//!     cargo oxide build field_addr_tuple

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirrors a `(Heavy, Tag)` tuple of two non-trivial Copy types,
/// like k256's `(ProjectivePoint, Scalar)`.
#[derive(Clone, Copy)]
pub struct Heavy {
    pub a: u64,
    pub b: u64,
    pub c: u64,
}

/// External "sink" — accepts `&Heavy` by reference and returns its
/// `.a`. Marked `#[inline(never)]` so the caller MUST materialize a
/// real `&Heavy` (no inlining-away of the field_addr).
#[inline(never)]
fn heavy_a(h: &Heavy) -> u64 {
    // Tiny opaque op to keep the function from being constant-folded.
    h.a.wrapping_add(h.b).wrapping_sub(h.b)
}

/// Mirrors `k256::arithmetic::mul::lincomb`'s shape: a generic
/// function that walks `&[(Heavy, T)]` via `Iterator::for_each`,
/// closing over an accumulator. The closure body takes `&pair.0`
/// and passes it to the non-inlinable `heavy_a` sink — that's the
/// `mir.field_addr` on a tuple pointee that the verifier rejects.
/// Generic over `T` to mirror `lincomb<P, S>`.
#[inline(never)]
fn fold_pairs<T: Copy + Into<u64>>(pairs: &[(Heavy, T)]) -> u64 {
    let mut acc: u64 = 0;
    pairs.iter().for_each(|pair| {
        // `&pair.0` — address of the first tuple element passed by
        // reference to the opaque sink. Forces a real field_addr.
        let h: &Heavy = &pair.0;
        acc = acc
            .wrapping_add(heavy_a(h))
            .wrapping_add(pair.1.into());
    });
    acc
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn tuple_field_addr(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && (i + 1) * 4 <= input.len()
        {
            let base = i * 4;
            let pairs: [(Heavy, u64); 2] = [
                (
                    Heavy {
                        a: input[base],
                        b: input[base + 1],
                        c: input[base + 2],
                    },
                    input[base + 3],
                ),
                (
                    Heavy {
                        a: input[base],
                        b: input[base + 2],
                        c: input[base + 1],
                    },
                    input[base + 3],
                ),
            ];
            *slot = super::fold_pairs::<u64>(&pairs);
        }
    }
}

fn main() {
    println!("=== field_addr_tuple ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u64> = (0..(N * 4) as u64).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .tuple_field_addr(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let base = i * 4;
        // Two pairs, each contributes h.a + tail.
        let expected = host[base]
            .wrapping_add(host[base + 3])
            .wrapping_add(host[base])
            .wrapping_add(host[base + 3]);
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: field_addr on tuple pointee codegen'd to PTX");
}
