//! Regression test for the 3-projection writer arm
//! `[Field, Index, Field]` — `_local.outer[i].inner = value` against
//! a local struct whose array field holds newtype-wrapped scalars.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Compilation error: invalid input program.
//! Unsupported construct: Complex places (3 projections) not yet implemented
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `crypto_bigint::uint::neg_mod` at `uint/neg_mod.rs:15`:
//!
//! ```ignore
//! ret.limbs[i].0 = z.if_true(ret.limbs[i].0);
//! ```
//!
//! `Uint<LIMBS>(pub [Limb; LIMBS])` and `Limb(pub Word)`; `ret` is a
//! LOCAL Uint, so the LHS projection list is
//! `[Field(0=limbs), Index(_i), Field(0=Limb.0)]` — no Deref.
//! The previous `[Deref, Field, Index]` arm doesn't cover it.
//!
//! ## What landed
//!
//! New 3-level-projection assignment arm in
//! `translator/statement.rs` for `[Field, Index(local), Field]`,
//! composing the existing primitives:
//!
//!   1. `field_addr(slot, outer_field_idx)` → ptr to inner array
//!   2. translate index local
//!   3. `array_element_addr(arr_ptr, index)` → ptr to element
//!   4. `field_addr(elem_ptr, inner_field_idx)` → ptr to inner field
//!   5. `mir.store(inner_ptr, value)`
//!
//! Hard-errors clearly if the outer field's translated type isn't
//! `MirArrayType` (a structural mismatch, not a missing case).
//!
//! ## Build with
//!
//!     cargo oxide build field_index_field_write

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirrors `crypto_bigint::Limb`: tuple struct wrapping a primitive.
#[derive(Clone, Copy)]
pub struct LimbLike(pub u64);

/// Mirrors `crypto_bigint::Uint<LIMBS>`: tuple struct whose inner
/// field is `[LimbLike; N]`.
#[derive(Clone, Copy)]
pub struct UintLike(pub [LimbLike; 5]);

/// `#[inline(never)]` so the 3-projection writer survives to the
/// importer instead of being folded into the caller.
#[inline(never)]
fn select_into(z: bool, mut ret: UintLike) -> UintLike {
    let mut i = 0usize;
    while i < 5 {
        // The LHS projection list:
        //   [Field(0=limbs array), Index(_i), Field(0=Limb.0)]
        ret.0[i].0 = if z { ret.0[i].0 } else { 0 };
        i += 1;
    }
    ret
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn neg_mod_like(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && (i + 1) * 5 <= input.len()
        {
            let base = i * 5;
            let r = UintLike([
                LimbLike(input[base]),
                LimbLike(input[base + 1]),
                LimbLike(input[base + 2]),
                LimbLike(input[base + 3]),
                LimbLike(input[base + 4]),
            ]);
            // Pass z=true so we keep the values; the false branch
            // would zero them.
            let out_ret = super::select_into(input[base] != 0xdead_beef, r);
            *slot = out_ret.0[0].0
                ^ out_ret.0[1].0
                ^ out_ret.0[2].0
                ^ out_ret.0[3].0
                ^ out_ret.0[4].0;
        }
    }
}

fn main() {
    println!("=== field_index_field_write ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u64> = (0..(N * 5) as u64).collect();
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
        let base = i * 5;
        // Same XOR fold as the kernel.
        let expected = host[base]
            ^ host[base + 1]
            ^ host[base + 2]
            ^ host[base + 3]
            ^ host[base + 4];
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: [Field, Index, Field] write codegen'd to PTX");
}
