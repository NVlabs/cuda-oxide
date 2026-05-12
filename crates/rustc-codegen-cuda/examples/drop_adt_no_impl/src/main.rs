//! Regression test for `Drop` terminator elision on ADTs that have
//! no user-defined `impl Drop` and whose fields are all trivial-drop.
//!
//! ## Pre-fix wall
//!
//! Before the fix, `translate_drop` consulted only the cheap
//! shape-recursion (`has_no_drop_glue`), which conservatively
//! returned `false` for every ADT — even ones like
//! `crypto_bigint::Uint<U4>` (a plain wrapper around `[Limb; N]` with
//! no `Drop` impl):
//!
//! ```text
//! Unsupported construct: drop of `RigidTy(Adt(AdtDef(DefId { …,
//!   name: "crypto_bigint::uint::Uint" }), …))` is not supported on
//!   the device; cuda-oxide does not yet emit device-side
//!   `drop_in_place` calls.
//! ```
//!
//! Triggered from `~/vanity-miner-rs/` via
//! `<Uint<U4> as Zero>::is_zero` (and pervasively across `k256` /
//! `elliptic-curve` / `crypto-bigint` — every owned `Uint<_>` going
//! out of scope hit the wall).
//!
//! ## What landed
//!
//! `translate_drop` now consults `Instance::resolve_drop_in_place(ty)
//! .is_empty_shim()` after the shape-recursion fails. rustc has
//! already computed which drop shims are empty (no work to do); we
//! just have to ask. The shape-recursion stays as the fast path
//! for primitives so we don't pay the monomorphization cost in
//! kernel-body-dominant cases.
//!
//! The empty-shim query handles every ADT without a real
//! `Drop` impl whose fields recursively all have trivial drop:
//! `crypto_bigint::Uint`, `core::num::Wrapping`,
//! `#[repr(transparent)]` newtypes over `Copy` types, every
//! user-defined struct/enum-of-primitives, etc.
//!
//! ADTs *with* an `impl Drop` (like `drop_adt_with_impl/`'s
//! `Secret { ... }`) still hard-error because their drop shim is
//! NOT empty.
//!
//! ## Build with
//!
//!     cargo oxide run drop_adt_no_impl
//!
//! Expected: kernel runs, output XOR-folds each input window
//! through the wrapper struct.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Stand-in for `crypto_bigint::Uint<U4>`: plain struct wrapping an
/// array of u64s. No `Drop` impl, all fields `Copy`. Triviality is
/// visible to rustc's drop-shim analysis, just not to the importer's
/// shape recursion.
#[derive(Clone, Copy)]
pub struct UintLike {
    pub limbs: [u64; 4],
}

impl UintLike {
    #[inline(never)]
    pub fn xor_fold(self) -> u64 {
        // Taking `self` by value forces a Drop terminator at the
        // end of the function on the parameter local (which is
        // `UintLike`). `#[inline(never)]` keeps the Drop from being
        // optimized away before reaching cuda-oxide.
        self.limbs[0] ^ self.limbs[1] ^ self.limbs[2] ^ self.limbs[3]
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Build a `UintLike` from input bytes, pass it by value through
    /// `xor_fold`. The owned-by-value receiver is the Drop trigger;
    /// `Uint`-like ADTs without `impl Drop` should now be elided
    /// via the empty-shim query path.
    #[kernel]
    pub fn uint_xor_fold(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i * 4 + 3 < input.len()
        {
            let u = UintLike {
                limbs: [
                    input[i * 4],
                    input[i * 4 + 1],
                    input[i * 4 + 2],
                    input[i * 4 + 3],
                ],
            };
            *slot = u.xor_fold();
        }
    }
}

fn main() {
    println!("=== drop_adt_no_impl ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u64> = (0..(N * 4) as u64).map(|n| n * 0x9E37_79B9_7F4A_7C15).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .uint_xor_fold(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = host[i * 4] ^ host[i * 4 + 1] ^ host[i * 4 + 2] ^ host[i * 4 + 3];
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: ADT-no-impl-Drop codegen'd to PTX");
}
