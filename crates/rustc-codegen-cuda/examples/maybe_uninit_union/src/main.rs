//! Known-failure repro for `Aggregate(Adt(<union>, ...))` — the MIR
//! shape produced by `core::mem::MaybeUninit::uninit()`.
//!
//! ## Wall (current state)
//!
//! ```text
//! Compilation error: invalid input program.
//! Unsupported construct: Union aggregate not yet supported: MaybeUninit
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `<GenericArray<u8, U33> as GenericSequence>::generate`, whose
//! `Default::default` closure builds a `[MaybeUninit<u8>; N]` by
//! calling `MaybeUninit::uninit()` per element. The body of
//! `MaybeUninit::uninit()` is a single rvalue:
//!
//! ```ignore
//! pub const fn uninit() -> MaybeUninit<T> {
//!     MaybeUninit { uninit: () }   // Aggregate(Adt(MaybeUninit, ...))
//! }
//! ```
//!
//! ## Where it bails
//!
//! `crates/mir-importer/src/translator/rvalue.rs` — the
//! `mir::Rvalue::Aggregate` arm dispatches `AggregateKind::Adt` on
//! `adt_def.kind()` and the `AdtKind::Union` branch is a hard
//! `unsupported` bail. Structs and enums have full handlers; unions
//! have nothing.
//!
//! ## What a fix needs to do
//!
//! Unions have one variant with overlapping fields and a single
//! "active" field at construction. The aggregate's `operands` should
//! have exactly one initializing operand for that active field. The
//! lowered representation needs to write that operand into a union
//! storage value of the union's translated type — most likely by
//! treating the union as opaque bag-of-bits storage and storing the
//! initializer at offset 0 (or wherever rustc says the active
//! field lives, but for `repr(Rust)` unions the offset is 0).
//!
//! For the `MaybeUninit { uninit: () }` shape specifically, the
//! initializer is a ZST `()`, so the union storage just needs to
//! be allocated/poisoned to the right size with no actual store.
//!
//! ## Build with
//!
//!     cargo oxide build maybe_uninit_union

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// `#[inline(never)]` shim so the `MaybeUninit::uninit()` body
/// (the union aggregate `MaybeUninit { uninit: () }` at
/// `maybe_uninit.rs:430:9`) cannot be folded away by the inliner —
/// the device codegen pipeline has to actually translate it.
///
/// This mirrors the `vanity-miner-rs` failure path, where
/// `<GenericArray<u8, U33> as GenericSequence>::generate` calls
/// `MaybeUninit::uninit()` per-element through a closure that the
/// inliner can't reach across the trait boundary.
#[inline(never)]
fn make_uninit_u8() -> core::mem::MaybeUninit<u8> {
    core::mem::MaybeUninit::uninit()
}

#[cuda_module]
pub mod kernels {
    use super::*;
    use core::mem::MaybeUninit;

    /// Build an array of `MaybeUninit<u8>` via the `#[inline(never)]`
    /// shim, write each slot, then fold the values out. The shim
    /// forces the union aggregate to surface as a real Aggregate
    /// rvalue inside `make_uninit_u8`'s MIR.
    #[kernel]
    pub fn maybe_uninit_array_roundtrip(input: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && (i + 1) * 8 <= input.len()
        {
            let mut buf: [MaybeUninit<u8>; 8] = [
                super::make_uninit_u8(),
                super::make_uninit_u8(),
                super::make_uninit_u8(),
                super::make_uninit_u8(),
                super::make_uninit_u8(),
                super::make_uninit_u8(),
                super::make_uninit_u8(),
                super::make_uninit_u8(),
            ];
            let base = i * 8;
            let mut acc: u32 = 0;
            let mut k = 0;
            while k < 8 {
                buf[k].write(input[base + k]);
                k += 1;
            }
            let mut k = 0;
            while k < 8 {
                acc = acc.wrapping_mul(31).wrapping_add(unsafe {
                    buf[k].assume_init()
                } as u32);
                k += 1;
            }
            *slot = acc;
        }
    }
}

fn main() {
    println!("=== maybe_uninit_union ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 4;
    let host: Vec<u8> = (0..(N * 8) as u8).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .maybe_uninit_array_roundtrip(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let mut expected: u32 = 0;
        for k in 0..8 {
            expected = expected.wrapping_mul(31).wrapping_add(host[i * 8 + k] as u32);
        }
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: MaybeUninit union aggregate codegen'd to PTX");
}
