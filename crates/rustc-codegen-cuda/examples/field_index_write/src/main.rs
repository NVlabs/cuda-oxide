//! Regression test for the 2-projection `[Field(idx), Index(_i)]`
//! writer arm — `_local.field[i] = value` against a tuple struct
//! whose field is a fixed-size array.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Compilation error: invalid input program.
//! Unsupported construct: 2-level projection
//!   Field(0, [u64; 5]) -> Index(_27) not yet implemented for assignment
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `<curve25519_dalek::backend::serial::curve_models::EdwardsPoint as
//!   Add<&AffineNielsPoint>>::add` at `field.rs:61:13: 61:35` —
//! `FieldElement51` is `pub struct FieldElement51(pub(crate) [u64; 5])`.
//! Inline arithmetic writes `self.0[i] = ...` against a local
//! `FieldElement51`, lowering to a 2-projection assign-to-place with
//! `[Field(0), Index(_)]`.
//!
//! ## What landed
//!
//! New 2-level-projection assignment arm in
//! `translator/statement.rs`:
//!
//! 1. `mir.field_addr(slot, field_idx)` → pointer to the field (the
//!    inner array)
//! 2. translate the index local
//! 3. `emit_array_element_store(field_ptr, index, value, ...)` —
//!    the same helper the single-level `arr[i] = ...` Index path
//!    uses for the GEP + store.
//!
//! The field's translated type must be a `MirArrayType`; anything
//! else is a structural mismatch and hard-errors with a clear message.
//!
//! ## Build with
//!
//!     cargo oxide build field_index_write

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirrors `FieldElement51`: tuple struct wrapping a fixed-size array.
#[derive(Clone, Copy)]
pub struct Limbs5(pub [u64; 5]);

/// Build a `Limbs5` LOCAL (not a reference) and write into its
/// inner array via `state.0[i] = ...`. The local-ness is the point —
/// the projection list becomes `[Field(0), Index(_)]` (2 elements),
/// matching the failing curve25519-dalek shape. With a `&mut` receiver
/// you'd get `[Deref, Field(0), Index(_)]` (3 elements), which hits
/// the separate "Complex places (3 projections)" arm.
#[inline(never)]
fn fold_local(input: u64) -> Limbs5 {
    let mut state = Limbs5([1, 2, 3, 4, 5]);
    let mut i = 0usize;
    while i < 5 {
        state.0[i] = state.0[i].wrapping_add(input.wrapping_mul((i as u64) + 1));
        i += 1;
    }
    state
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn limbs5_fold(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            let state = super::fold_local(input[i]);
            *slot = state.0[0]
                ^ state.0[1]
                ^ state.0[2]
                ^ state.0[3]
                ^ state.0[4];
        }
    }
}

fn main() {
    println!("=== field_index_write ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u64> = (0..N as u64).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .limbs5_fold(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let mut s: [u64; 5] = [1, 2, 3, 4, 5];
        for k in 0..5 {
            s[k] = s[k].wrapping_add(host[i].wrapping_mul((k as u64) + 1));
        }
        let expected = s[0] ^ s[1] ^ s[2] ^ s[3] ^ s[4];
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: tuple-struct field+index writes codegen'd to PTX");
}
