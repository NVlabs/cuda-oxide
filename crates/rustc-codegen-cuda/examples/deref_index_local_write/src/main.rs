//! Regression test for `(*place)[local_idx] = value` writes —
//! 2-level MIR projection `Deref -> Index(local)` — through a
//! `&mut [T; N]` array reference (and the slice equivalent).
//! Sibling of the existing `Deref -> ConstantIndex` handler, but
//! with a runtime index.
//!
//! ## Pre-fix wall
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Translation failed:
//!        <kernel-symbol>: Compilation error: invalid input program.
//!        Unsupported construct: 2-level projection Deref -> Index(_)
//!        not yet implemented for assignment
//! ```
//!
//! ## What triggers it
//!
//! A device helper takes `&mut [T; N]` (or `&mut [T]`) and writes to
//! `output[i] = val` where `i` is a runtime `usize`. MIR lowers the
//! LHS place as projections `[Deref, Index(_local_holding_i)]` on the
//! local that holds the reference. cuda-oxide's `statement.rs`
//! already handles the cousin shape `Deref -> ConstantIndex` (constant
//! offsets, common from `*slot = chunk[0]` after a pattern match) but
//! the runtime-index sibling falls through to the catch-all
//! "2-level projection … not yet implemented for assignment" error.
//!
//! ## Where the gap lives
//!
//! `crates/mir-importer/src/translator/statement.rs::translate_statement`
//! at the `Assign(place, rvalue)` arm processes `place.projection`
//! pairs:
//!
//! ```rust,ignore
//! match (&place.projection[0], &place.projection[1]) {
//!     (Deref, Field(_, _))              => …,   // *p.field = v
//!     (Deref, ConstantIndex { offset, … }) => …, // *p[K] = v
//!     (Field, Deref)                    => …,
//!     // ↓ missing
//!     // (Deref, Index(local))             => …, // *p[i] = v
//!     _ => input_err!("2-level projection {} -> {} not yet implemented"),
//! }
//! ```
//!
//! The constant-offset arm already does the right LLVM-level
//! sequence: load the slot's fat pointer, extract the data pointer,
//! GEP to the offset, store. The fix mirrors that arm with two
//! differences:
//!
//! 1. **Offset operand**: instead of building a `MirConstantOp` from
//!    the `offset: u64` projection field, look up the
//!    `index_local`'s SSA value in the value map.
//! 2. **Pointee match**: the existing arm assumes
//!    `MirPtrType<MirSliceType<T>>` (i.e. `&mut [T]`). Extend the
//!    match to also accept `MirPtrType<MirArrayType<T, N>>` (i.e.
//!    `&mut [T; N]`) — vanity-miner-rs's `base58_encode_32` takes
//!    `&mut [u8; 64]` for the output buffer, and that's the actual
//!    in-the-wild trigger.
//!
//! ## What landed
//!
//! Added the `Deref -> Index(local)` arm in
//! `crates/mir-importer/src/translator/statement.rs`'s 2-level
//! projection switch. The new arm classifies the slot's inner type
//! once:
//!
//! * `&mut [T; N]` → slot is `MirPtrType<MirPtrType<MirArrayType<T, N>>>`.
//!   Load to a thin pointer to the array, then `mir.array_element_addr`
//!   + `mir.store` — the same shape the single-level `arr[i] = value`
//!   path uses.
//! * `&mut [T]` → slot is `MirPtrType<MirSliceType<T>>`. Load the fat
//!   slice value, extract the data pointer, `mir.ptr_offset`, store —
//!   identical to the existing `Deref -> ConstantIndex` arm but with
//!   the offset coming from the translated index local instead of a
//!   freshly emitted `MirConstantOp`.
//!
//! Anything else (e.g. a thin `*mut T` with an Index projection — a
//! structural mismatch that shouldn't occur in well-formed MIR) hard-
//! errors with the actual slot type in the diagnostic.
//!
//! Originally surfaced from `~/vanity-miner-rs/`:
//! `logic::base58_encode_32` at `logic/src/base58.rs:80`:
//!
//! ```rust,ignore
//! pub fn base58_encode_32(input: &[u8; 32], output: &mut [u8; 64]) -> usize {
//!     // …
//!     for i in 0..DIGITS_PER_LIMB {
//!         let temp = (limb_value / DIVISORS[i]) % 58;
//!         output[output_offset + i] = temp as u8;   // ← Deref -> Index(local)
//!     }
//!     // …
//! }
//! ```
//!
//! ## Build with
//!
//!     cargo oxide run deref_index_local_write
//!
//! Expected: kernel runs, each output slot equals the corresponding
//! `(i & 0xff)`.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Helper that does the offending write. Lives in a separate function
/// so its MIR is forced into a real device-side body (instead of
/// getting inlined into the kernel where the trigger pattern would
/// be obscured by the surrounding code).
///
/// `&mut [u8; 64]` matches vanity-miner-rs's `base58_encode_32`
/// signature exactly.
#[inline(never)]
fn write_at(output: &mut [u8; 64], idx: usize, value: u8) {
    output[idx] = value; // ← Deref -> Index(local). The trigger.
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Each thread writes its own index into the disjoint output slot,
    /// going through `write_at(&mut [u8; 64], idx, value)`. The stack
    /// array forces the `&mut [u8; 64]` reference shape.
    #[kernel]
    pub fn fill_array(mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            let mut buf = [0u8; 64];
            let pos = i % 64;
            write_at(&mut buf, pos, (i & 0xff) as u8);
            *slot = buf[pos];
        }
    }
}

fn main() {
    println!("=== deref_index_local_write ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 16;
    let mut out = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .fill_array(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = (i & 0xff) as u8;
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: (*&mut [u8; 64])[local] = value codegen'd to PTX");
}
