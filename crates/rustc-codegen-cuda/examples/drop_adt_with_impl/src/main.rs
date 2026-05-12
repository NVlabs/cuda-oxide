//! Known-failure reproducer for `TerminatorKind::Drop` on an ADT
//! whose field tree contains a user-defined `Drop` impl
//! (zeroization-style destructor).
//!
//! ## Expected failure
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Translation failed:
//!        <kernel-symbol>: Compilation error: invalid input program.
//!        Unsupported construct: drop of `RigidTy(Adt(AdtDef(DefId {
//!          id: ..., name: "Secret" }), GenericArgs([])))` is not
//!        supported on the device; cuda-oxide does not yet emit
//!        device-side `drop_in_place` calls. Restructure the kernel
//!        to use only `Copy` types, or wrap the value in
//!        `core::mem::ManuallyDrop` to suppress drop glue.
//! ```
//!
//! ## What triggers it
//!
//! The kernel owns a `Secret` value. `Secret` implements `Drop`
//! (the canonical "zero out the bytes" destructor that
//! `elliptic_curve::SecretKey`, `zeroize::Zeroizing`,
//! `secrecy::SecretBox`, etc. all use). When the local goes out of
//! scope, rustc emits a `TerminatorKind::Drop` terminator pointing
//! at the local's slot. The previous trivial-drop fix elides drops
//! for `Copy` primitives and aggregates of primitives, but
//! deliberately keeps `Adt(_, _)` on the hard-error path:
//!
//! ```rust,ignore
//! // crates/mir-importer/src/translator/terminator/mod.rs
//! fn has_no_drop_glue(ty: &Ty) -> bool {
//!     match ty.kind() {
//!         TyKind::RigidTy(rigid) => match rigid {
//!             RigidTy::Int(_) | RigidTy::Uint(_) | … => true,
//!             RigidTy::Array(elem, _) => has_no_drop_glue(&elem),
//!             RigidTy::Tuple(es) => es.iter().all(has_no_drop_glue),
//!             _ => false,     // ← Adt lands here
//!         },
//!         _ => false,
//!     }
//! }
//! ```
//!
//! That's load-bearing: `Secret`'s drop is semantically observable
//! (it writes zeros into the bytes), and an importer that silently
//! lowered the Drop to a goto would produce PTX where the
//! zeroization is missing entirely — a quiet miscompile in the same
//! shape as the pre-`copy_nonoverlapping`-fix silent-drop regression.
//!
//! Same root cause — any user-defined struct/enum with `impl Drop`,
//! any standard-library type with drop glue (`Vec`, `String`, `Box`,
//! `RefCell`, `Rc`, …), and any crypto secret type
//! (`elliptic_curve::SecretKey`, `zeroize::Zeroizing<T>`,
//! `secrecy::SecretBox`, …) trips this.
//!
//! ## What would it take to fix
//!
//! Real device-side `drop_in_place` lowering. Conceptual sketch:
//!
//! 1. **Importer**: rewrite `TerminatorKind::Drop { place, target }`
//!    into a `mir.call` to `<T as Drop>::drop_in_place(addr_of_mut
//!    place)` followed by `mir.goto target`. The callee name is
//!    rustc's synthesized drop shim symbol for `T`.
//! 2. **Collector**: pull the drop shim's MIR body in as a normal
//!    `MonoItem`. rustc's stable MIR exposes drop shims through
//!    `Instance::resolve_drop_in_place(tcx, ty)` (host-side; we'd
//!    need a `rustc_public` equivalent).
//! 3. **Translation**: the shim body calls `<T as Drop>::drop(&mut
//!    self)` (the user's destructor), then drops each field
//!    recursively. Both the user `drop` and the field destructors
//!    translate as ordinary functions.
//!
//! Layer-1 risk: the shim body for `Secret` calls
//! `core::ptr::write_volatile` (zeroize's primitive). Volatile
//! writes are MIR `Rvalue::Use` with the volatile flag — likely
//! another wall, but bounded.
//!
//! Layer-2 risk: `Drop::drop` may pull in
//! `core::ptr::drop_in_place::<u8>` (no-op) and similar generic
//! intrinsics. Each one might surface as a new mangled-symbol
//! collection issue.
//!
//! Layer-3 risk: real-world types like
//! `elliptic_curve::SecretKey<C>` drag in the entire crypto crate's
//! type tree (`NonZeroScalar`, `ScalarPrimitive`, `FieldBytes`, …)
//! and several layers of generic helper functions. Some of those
//! may use unsupported features (atomics on shared refs, `Cell`,
//! TLS access). Each is a separate dig.
//!
//! Alternative: **device-side allowlist of trivially-zeroizing
//! types**. Recognize `Zeroizing<T>` / `SecretKey<C>` / similar by
//! ADT name and elide their drops (with a documented caveat that
//! the bytes don't get zeroed — which is fine on GPU memory that's
//! either ephemeral or freed shortly after). Pragmatic but
//! brittle.
//!
//! Alternative: **`#[device_no_drop]` attribute**. Let the user
//! mark a type as drop-elided for device codegen, on the principle
//! that they know whether the drop matters in their context.
//!
//! Originally surfaced from `~/vanity-miner-rs/`: the
//! `logic::secp256k1_derive_public_key` path constructs an
//! `elliptic_curve::SecretKey<k256::Secp256k1>` from random bytes,
//! and the local's destructor calls into `SecretKey`'s
//! `Drop::drop`.
//!
//! ## Build with
//!
//!     cargo oxide build drop_adt_with_impl
//!
//! Expected: build error from the mir-importer's drop terminator
//! handler with the `Secret` ADT in the diagnostic.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Minimal stand-in for `elliptic_curve::SecretKey<C>` /
/// `zeroize::Zeroizing<T>`. Holds bytes; on drop zeroes them.
/// The `Drop` impl is what makes this hit the wall — without it,
/// `Secret` is just a `[u8; 32]` newtype and would fall through to
/// the trivial-drop path.
pub struct Secret {
    pub bytes: [u8; 32],
}

impl Drop for Secret {
    #[inline(never)]
    fn drop(&mut self) {
        // Zeroize-style write. The exact body doesn't matter for the
        // reproducer — what matters is that `Drop::drop` exists and
        // forces rustc to emit a `TerminatorKind::Drop` for any owned
        // `Secret` going out of scope. `volatile_*` would be more
        // faithful to `zeroize`, but a plain loop is enough.
        for b in self.bytes.iter_mut() {
            *b = 0;
        }
    }
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Trigger: owning a `Secret` (which has `impl Drop`) makes
    /// rustc emit a Drop terminator on the local's slot when it
    /// goes out of scope. cuda-oxide's `translate_drop` hard-errors
    /// on `Adt(_, _)` types (see
    /// `crates/mir-importer/src/translator/terminator/mod.rs`).
    #[kernel]
    pub fn secret_xor(input: &[u8], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            let mut bytes = [0u8; 32];
            // Fill from input so the value can't be constant-folded.
            let k = 0;
            while k < 32 && (i + k) < input.len() {
                bytes[k] = input[i + k];
                #[allow(clippy::assign_op_pattern)]
                let _ = k; // placate iter loops vs. while-let machinery
                break;
            }
            let secret = Secret { bytes };
            // XOR-fold so the compiler can't elide the read.
            let mut acc = 0u8;
            for b in &secret.bytes {
                acc ^= *b;
            }
            *slot = acc;
            // `secret` is dropped here — that's the Drop terminator
            // the importer rejects.
        }
    }
}

fn main() {
    println!("=== drop_adt_with_impl ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u8> = (0..(N + 32) as u32).map(|n| (n & 0xff) as u8).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .secret_xor(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    println!("output: {:?}", result);
    println!("SUCCESS: Drop on ADT with impl Drop codegen'd to PTX");
}
