//! Regression test for the 3-projection writer arm
//! `[Deref, Field, Index(local)]` — `(*ref).field[i] = value`
//! against a tuple/named struct whose field is a fixed-size array.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Compilation error: invalid input program.
//! Unsupported construct: Complex places (3 projections) not yet implemented
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `crypto_bigint::uint::neg_mod` at `uint/neg_mod.rs:15`.
//! `Uint<LIMBS>` is a tuple struct wrapping `[Limb; LIMBS]`; inside
//! the `&mut Self` ADC loop, `(*ret).limbs[i] = ...` lowers to a
//! 3-projection assign-to-place with `[Deref, Field(0), Index(_i)]`.
//!
//! ## What landed
//!
//! New 3-level-projection assignment arm in
//! `translator/statement.rs` for `[Deref, Field, Index(local)]`,
//! composing the existing 2-level building blocks:
//!
//!   1. load the slot — peels the outer pointer to get a `*Self`
//!   2. `mir.field_addr(struct_ptr, field_idx)` → pointer to the
//!      inner array
//!   3. translate the index local
//!   4. `emit_array_element_store(field_ptr, index, value, ...)`
//!      — the same helper the single-level Index path uses
//!
//! Hard-errors clearly if the field's translated type isn't
//! `MirArrayType` (a structural mismatch, not a missing case).
//!
//! Other 3-projection shapes (`[Deref, Field, ConstantIndex]`,
//! `[Deref, Deref, Field]`, etc.) still fall through to the
//! catch-all and can be added when something hits them.
//!
//! ## Build with
//!
//!     cargo oxide build deref_field_index_write

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirrors `crypto_bigint::Uint`: tuple struct wrapping an array of
/// fixed-size primitive limbs.
#[derive(Clone, Copy)]
pub struct UintLike(pub [u64; 5]);

/// Shape of `Uint::neg_mod`'s ADC loop body: takes `&mut Self`,
/// writes `self.0[i] = ...` (which becomes `(*self).0[i] = ...`
/// in MIR — the 3-projection `[Deref, Field, Index]` shape).
/// `#[inline(never)]` keeps the loop in its own MIR function.
#[inline(never)]
fn fold_into(state: &mut UintLike, input: u64) {
    let mut i = 0usize;
    while i < 5 {
        state.0[i] = state.0[i].wrapping_add(input.wrapping_mul((i as u64) + 1));
        i += 1;
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn neg_mod_like(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            let mut state = UintLike([1, 2, 3, 4, 5]);
            super::fold_into(&mut state, input[i]);
            *slot = state.0[0]
                ^ state.0[1]
                ^ state.0[2]
                ^ state.0[3]
                ^ state.0[4];
        }
    }
}

fn main() {
    println!("=== deref_field_index_write ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u64> = (0..N as u64).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .neg_mod_like(
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
    println!("SUCCESS: [Deref, Field, Index] write codegen'd to PTX");
}
