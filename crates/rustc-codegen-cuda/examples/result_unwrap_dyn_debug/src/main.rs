//! Known-failure reproducer for `dyn Trait` (trait object) types
//! reaching the mir-importer's type translator via
//! `Result::unwrap()` / `Result::expect()`.
//!
//! ## Expected failure
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Translation failed:
//!        <core::result::Result<[u8; 8], core::array::TryFromSliceError>>::unwrap:
//!        Compilation error: invalid input program.
//!        Unsupported construct: Type translation not yet implemented for:
//!        RigidTy(Dynamic([Binder { value: Trait(ExistentialTraitRef {
//!          def_id: TraitDef(DefId { id: ..., name: "std::fmt::Debug" }),
//!          generic_args: GenericArgs([]) }), bound_vars: [] }],
//!          Region { kind: ReErased }))
//! ```
//!
//! Decoded: `dyn std::fmt::Debug`.
//!
//! ## What triggers it
//!
//! The kernel does:
//!
//! ```rust,ignore
//! let chunk: [u8; 8] = input[i * 8..i * 8 + 8].try_into().unwrap();
//! ```
//!
//! `<[u8] as TryInto<[u8; 8]>>::try_into` returns
//! `Result<[u8; 8], core::array::TryFromSliceError>`. `Result::unwrap`
//! is generic in `E: Debug` and its panic path internally calls
//! `<E as Debug>::fmt(&dyn Debug, ...)` to format the unwrapped error
//! into the panic message. Once monomorphized for
//! `E = TryFromSliceError`, the MIR carries a `&dyn Debug` operand
//! and the type translator at
//! `crates/mir-importer/src/translator/types.rs` has no arm for
//! `RigidTy(Dynamic(...))`.
//!
//! Same root cause — any other `.unwrap()` (or `.expect("...")`) on
//! a `Result<_, E>` where `E: Debug` produces the identical failure
//! shape. `Option::unwrap` is fine: it doesn't take an error value,
//! so there's no `Debug` formatting path. `unwrap_or`,
//! `unwrap_or_default`, `unwrap_or_else(|_| ...)` are also fine for
//! the same reason — they don't panic-format the error.
//!
//! ## What it would take to fix
//!
//! `dyn Trait` is fundamentally hard for static-codegen-from-MIR
//! backends because it carries runtime dispatch through a vtable.
//! Three plausible paths, in order of effort:
//!
//! 1. **Reject early with a structured error.** Add a
//!    `TyKind::RigidTy(RigidTy::Dynamic(...))` arm in `translate_type`
//!    that bails with: "cuda-oxide does not support trait objects
//!    (`dyn Trait`). The path here is usually `Result::unwrap()` /
//!    `.expect()` — replace with `unwrap_or_else(|_| {...})` or
//!    `match` on the `Err` variant." Tells the user what to change
//!    without making them decode v0 mangled names.
//!
//! 2. **Synthesize a vtable.** Translate `dyn Trait` as `&{ ptr, vtable }`
//!    where the vtable is a static struct of fn pointers per trait
//!    method. Call sites lower to indirect calls through the vtable.
//!    Works for simple traits, but `Debug` (with its `&mut Formatter`
//!    machinery) drags in the `fmt::Arguments` / `Formatter` graph
//!    that `helper_no_inline`'s sibling `panic_fmt_path/` already
//!    documented as expensive.
//!
//! 3. **Specialize `Result::unwrap` paths.** Detect calls to
//!    `Result::unwrap` and rewrite them to abort-on-Err without the
//!    `Debug` panic message — analogous to what
//!    `convert_rust_bit_intrinsic` does for primitive intrinsics.
//!    Avoids the `dyn Trait` problem at the source. Doesn't help
//!    user code that legitimately uses `&dyn Trait` for other
//!    reasons.
//!
//! Option 1 is the right immediate move (cheap, dramatically better
//! UX). Option 2 is the principled long-term fix. Option 3 is a
//! narrow band-aid for the most common surface — worth doing if 1
//! and 2 stay deferred.
//!
//! Originally surfaced from `~/vanity-miner-rs/`: the `logic` crate
//! uses `<[u8]>::try_into().unwrap()` to convert slice prefixes
//! into fixed-size arrays for cryptographic operations
//! (SHA-256 block extraction, ed25519 / secp256k1 key slicing).
//!
//! ## Build with
//!
//!     cargo oxide build result_unwrap_dyn_debug
//!
//! Expected: build error from the mir-importer's type translator:
//! `Type translation not yet implemented for: RigidTy(Dynamic([...
//! "std::fmt::Debug" ...]))`.
//!
//! ## What this example is NOT
//!
//! - Not about `Result::Err` per se — `unwrap_or_else(|e| { ... })`
//!   works fine (no `dyn Debug` in its monomorphization).
//! - Not about `panic_fmt` (covered by `examples/panic_fmt_path/`).
//!   That bug was about `core::fmt::Arguments`'s `FnPtr` slots.
//!   This one is one layer up: the `&dyn Debug` operand that those
//!   FnPtr slots are *called on*.
//! - Not about cross-crate / `#[device]` / `Alias` types
//!   (`cross_crate_pubfn`, `helper_*`, `iter_zip_chunks_exact`).
//!   The trigger is purely intra-`core`, reproducible in a one-file
//!   kernel.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// FAILS: the `.try_into().unwrap()` line is the trigger.
    /// `TryInto::try_into` returns `Result<[u8; 8], TryFromSliceError>`
    /// and `unwrap`'s monomorphization for that `E` drags
    /// `&dyn Debug` into the MIR.
    #[kernel]
    pub fn unwrap_dyn_debug(input: &[u8], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            // SAFETY: caller is responsible for `input.len() >= (i + 1) * 8`;
            // the launch config matches the input length to one chunk per thread.
            let start = i * 8;
            let chunk: [u8; 8] = input[start..start + 8].try_into().unwrap();
            *slot = chunk[0];
        }
    }
}

fn main() {
    println!("=== result_unwrap_dyn_debug ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const THREADS: usize = 64;
    const INPUT_LEN: usize = THREADS * 8;
    let host: Vec<u8> = (0..INPUT_LEN as u32).map(|n| (n & 0xff) as u8).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u8>::zeroed(&stream, THREADS).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .unwrap_dyn_debug(
            &stream,
            LaunchConfig::for_num_elems(THREADS as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..THREADS {
        assert_eq!(result[i], host[i * 8]);
    }
    println!("SUCCESS: try_into().unwrap() codegen'd to PTX");
}
