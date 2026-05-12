//! Regression test for `Aggregate(Adt(<union>, ...))` translation —
//! the MIR shape produced by `core::mem::MaybeUninit::uninit()`.
//!
//! ## Pre-fix wall
//!
//! Before the fix, the `mir::Rvalue::Aggregate` arm in
//! `crates/mir-importer/src/translator/rvalue.rs` dispatched
//! `AggregateKind::Adt` on `adt_def.kind()` and the `AdtKind::Union`
//! branch was a hard `unsupported` bail:
//!
//! ```text
//! Compilation error: invalid input program.
//! Unsupported construct: Union aggregate not yet supported: MaybeUninit
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `<GenericArray<u8, U33> as GenericSequence>::generate`, whose
//! `Default::default` closure builds a `[MaybeUninit<u8>; N]` by
//! calling `MaybeUninit::uninit()` per element:
//!
//! ```ignore
//! pub const fn uninit() -> MaybeUninit<T> {
//!     MaybeUninit { uninit: () }   // Aggregate(Adt(MaybeUninit, ...))
//! }
//! ```
//!
//! ## What landed
//!
//! Unions diverge from struct/enum aggregates in two ways: they have
//! a single operand initializing the *active* field (whose declaration
//! index is the 5th tuple element of `AggregateKind::Adt`), and the
//! fields all overlap at offset 0. The struct/enum field-walk in
//! `translate_adt_aggregate_field_values` tries to align operands to
//! non-ZST fields in declaration order, which mis-assigns when the
//! active field is itself ZST.
//!
//! `translate_union_aggregate` (new helper) handles unions directly:
//! it pulls the active-field index from the aggregate kind, translates
//! the single operand, and synthesizes `MirUndefOp` of the appropriate
//! type for each inactive field. The resulting `MirConstructStructOp`
//! then lowers the same way the struct path does — ZST fields are
//! skipped, non-ZST fields become a single `insertvalue` chain.
//!
//! Covers both:
//! * `MaybeUninit::uninit()` — `MaybeUninit { uninit: () }`, active
//!   operand is `()` (ZST). Lowers to a bare `undef` of the storage.
//! * `MaybeUninit::new(x)` — `MaybeUninit { value: ManuallyDrop::new(x) }`,
//!   active operand is non-ZST. Lowers to `insertvalue undef, %x, [0]`.
//!
//! Unions with multiple non-ZST overlapping fields would still
//! mis-lower at the type level (`build_struct_with_explicit_padding`
//! lays them out sequentially, not overlapping). Nothing in
//! `vanity-miner-rs` hits that case; a future fix can add
//! union-as-byte-storage type translation when something does.
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
