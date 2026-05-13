//! Regression test for `place.0[const_idx] = value` on a
//! newtype-wrapped array.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Unsupported construct: 2-level projection
//!   Field(0, [u64; 5]) -> ConstantIndex { offset: 0, min_length: 1,
//!     from_end: false }
//!   not yet implemented for assignment
//! ```
//!
//! Surfaced from vanity-miner-rs's bisection slot mirroring dalek's
//! `Scalar52::from_bytes`, where the constructor builds the inner
//! `[u64; 5]` slot-by-slot:
//!
//! ```ignore
//! let mut s = Scalar52::ZERO;
//! s.0[0] =   words[0]                            & mask;
//! s.0[1] = ((words[0] >> 52) | (words[1] << 12)) & mask;
//! ...
//! ```
//!
//! rustc's MIR encodes each `s.0[N] = ...` as an assignment whose
//! target Place has the 2-level projection chain
//! `[Field(0), ConstantIndex { offset: N }]`. mir-importer's
//! `translate_statement` had cases for each pair of projections
//! used in the wild (`Deref->Field`, `Field->Field`,
//! `Field->Index(local)`, `Deref->ConstantIndex`, `Deref->Index(local)`)
//! but no arm for `Field->ConstantIndex` until this fix.
//!
//! ## Root cause / fix
//!
//! Added a `Field -> ConstantIndex` arm to the place-write match in
//! [crates/mir-importer/src/translator/statement.rs]. It composes
//! the building blocks the sibling arms already use:
//!
//! * `mir.field_addr` against the local's slot — pointer to the
//!   inner array.
//! * `mir.constant` for the offset value (the only difference from
//!   the runtime-index `Field->Index(local)` sibling).
//! * `emit_array_element_store` — GEP + store, shared with the
//!   single-level `Index` write path.
//!
//! This shape is endemic to newtypes wrapping primitive arrays:
//! dalek `Scalar52` / `FieldElement51`, k256 `FieldElement5x52`,
//! any `Limb`-style accumulator built limb-by-limb.
//!
//! ## Build
//!
//!     cargo oxide build newtype_const_index_assign

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Newtype wrapping `[u64; 5]` — same shape as dalek's
/// `Scalar52(pub [u64; 5])` and similar field-element constructors.
pub struct Wrap5(pub [u64; 5]);

impl Wrap5 {
    /// Mirrors dalek's `Scalar52::from_bytes` shape: construct a
    /// zero-initialised newtype then assign each limb individually
    /// via `s.0[N] = …`. `#[inline(never)]` blocks the optimizer
    /// from fusing the struct into a direct array build.
    #[inline(never)]
    pub fn build_by_limb(seed: [u64; 5]) -> Self {
        let mut s = Wrap5([0u64; 5]);
        s.0[0] = seed[0] ^ 0xAA;
        s.0[1] = seed[1] ^ 0xBB;
        s.0[2] = seed[2] ^ 0xCC;
        s.0[3] = seed[3] ^ 0xDD;
        s.0[4] = seed[4] ^ 0xEE;
        s
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(args: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx)
            && args.len() >= 5
        {
            let seed = [args[0], args[1], args[2], args[3], args[4]];
            let s = Wrap5::build_by_limb(seed);

            let read_idx = core::hint::black_box(2usize) & 7;
            if read_idx < 5 {
                *slot = s.0[read_idx];
            }
        }
    }
}

fn main() {
    println!("=== newtype_const_index_assign ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let args_host = [11u64, 22, 33, 44, 55];
    let args = DeviceBuffer::from_host(&stream, &args_host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 1).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(1), &args, &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], 33 ^ 0xCC, "expected args[2] ^ 0xCC");
    println!("SUCCESS: newtype-wrapped array indexed-assign + read");
}
