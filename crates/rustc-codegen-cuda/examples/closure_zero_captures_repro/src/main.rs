//! Known-failure repro for closure-type capture mismatch.
//!
//! ## Wall (current state)
//!
//! ```text
//! Verification failed for '...generate::closure#0':
//! MirConstructStructOp has 2 operands but struct '...generate::closure#0' has 0 fields
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `<GenericArray<u8, U33> as GenericSequence>::generate`, whose
//! body builds a `[MaybeUninit<u8>; N]` then walks it via a closure
//! that captures the slot pointer and an index pointer.
//!
//! ## Where it bails
//!
//! `crates/mir-importer/src/translator/types.rs` (line ~641):
//!
//! ```ignore
//! // Extract upvar types from substs[2] (the tuple of captured types)
//! if substs.0.len() >= 3
//!     && let GenericArgKind::Type(upvar_tuple_ty) = &substs.0[2]
//!     && let TyKind::RigidTy(RigidTy::Tuple(upvar_tys)) = upvar_tuple_ty.kind()
//! { ... }
//! ```
//!
//! This assumes the closure substs are `[closure_kind, sig, upvars]`
//! with upvars at index 2. That's true for top-level fn-local closures
//! whose parent has no generics. But for closures defined inside
//! **generic** functions/methods, the parent generics are *prepended*:
//!
//! ```text
//! substs = [parent_generic_0, ..., parent_generic_N,
//!           closure_kind, sig, upvars]
//! ```
//!
//! With `parent_generic_N >= 1`, `substs[2]` is no longer the upvar
//! tuple, the `TyKind::Tuple` match fails, `field_types` stays empty,
//! and the closure type is shaped with **0** captures. Then the
//! aggregate construction passes the actual captures and verification
//! blows up.
//!
//! The fix should read the upvar tuple from the **last** element of
//! `substs.0` (per `ClosureArgs::split` in rustc — closure kind / sig /
//! upvars are always the trailing three slots), not index 2.
//!
//! ## Build with
//!
//!     cargo oxide build closure_zero_captures_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Generic outer function. The `F` generic parameter (a callable)
/// combined with the `T` element type means the closure defined
/// **inside** this body has parent generics [T, F] before its own
/// closure-args [kind, sig, upvars]. That shifts the upvar tuple
/// past the `substs[2]` slot the type translator probes.
///
/// Mirrors the shape of `<GenericArray<u8, U33> as GenericSequence>
/// ::generate<F>`, whose body uses `Iterator::for_each` with a
/// closure capturing `f` and a `position` pointer.
#[inline(never)]
fn generate_via_for_each<T: Copy, F: FnMut(usize) -> T>(buf: &mut [T], mut f: F) {
    let mut position: usize = 0;
    // `Iterator::for_each` is a trait method — the closure is
    // monomorphized but won't be folded into its own caller body
    // the way a directly-invoked `impl FnMut` would, because it's
    // built as a real closure aggregate to match the
    // `FnMut(&mut T)` argument shape `for_each` expects.
    buf.iter_mut().for_each(|dst| {
        *dst = f(position);
        position += 1;
    });
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn closure_in_generic(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && (i + 1) * 4 <= input.len()
        {
            let base = i * 4;
            let mut buf: [u32; 4] = [0; 4];
            // The inner closure `|j| input[base + j].wrapping_add(...)`
            // becomes the `f` of `generate_via_for_each`. The
            // FOR_EACH closure inside `generate_via_for_each` is
            // the one with mismatched substs.
            super::generate_via_for_each::<u32, _>(&mut buf, |j| {
                input[base + j].wrapping_mul(31).wrapping_add(j as u32)
            });
            let mut acc: u32 = 0;
            let mut k = 0;
            while k < 4 {
                acc = acc.wrapping_add(buf[k]);
                k += 1;
            }
            *slot = acc;
        }
    }
}

fn main() {
    println!("=== closure_zero_captures_repro ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 4;
    let host: Vec<u32> = (0..(N * 4) as u32).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .closure_in_generic(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let base = i * 4;
        let mut acc: u32 = 0;
        for j in 0..4 {
            acc = acc.wrapping_add(host[base + j].wrapping_mul(31).wrapping_add(j as u32));
        }
        assert_eq!(result[i], acc, "thread {} mismatch", i);
    }
    println!("SUCCESS: closure-in-generic codegen'd to PTX");
}
