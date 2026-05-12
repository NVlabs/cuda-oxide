//! Regression test for `Drop` terminators on `Copy` primitives.
//!
//! ## Pre-fix wall
//!
//! Before the fix, `translate_drop` rejected every drop terminator with:
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Translation failed:
//!        <usize as core::cmp::Ord>::min: Compilation error: invalid
//!        input program.
//!        Unsupported construct: drop of `RigidTy(Uint(Usize))` is not
//!        supported on the device; cuda-oxide does not yet emit
//!        device-side `drop_in_place` calls.
//! ```
//!
//! Decoded mangled name: `<usize as core::cmp::Ord>::min`.
//!
//! ## What triggers it
//!
//! Calling any `Ord` method that takes `self` by-value on a `usize`
//! (or any other `Copy` primitive). `<T as Ord>::min(self, other)` has
//! roughly the body:
//!
//! ```rust,ignore
//! match Ord::cmp(&self, &other) {
//!     Ordering::Less | Ordering::Equal => self,
//!     Ordering::Greater => other,
//! }
//! ```
//!
//! rustc emits `TerminatorKind::Drop` terminators for the *un-returned*
//! local on each branch — that is, on the `usize` value that wasn't
//! picked. For `usize` the drop is semantically a no-op (no
//! `drop_in_place` to run, no destructor anywhere in the call chain),
//! but the MIR carries the drop edge anyway. cuda-oxide's
//! `translate_drop` at
//! `crates/mir-importer/src/translator/terminator/mod.rs` rejects
//! *all* drops uniformly with a hard error.
//!
//! Same root cause — any kernel that returns one of two owned
//! primitives, or has an `if`/`match` that picks between two owned
//! primitives, lowers to MIR with a Drop on the un-picked branch and
//! hits this wall. Common surface area: `.min()`, `.max()`,
//! `.clamp()`, `core::cmp::min(a, b)`, ternary-style `if cond { a }
//! else { b }` where both branches own a primitive.
//!
//! ## What landed
//!
//! `translate_drop` now consults a recursive `has_no_drop_glue` shape
//! check on the dropped place's type. When the type is guaranteed to
//! have no drop glue, the terminator lowers to a plain `mir.goto
//! target` instead of erroring. The covered set:
//!
//! 1. **Primitives**: `Int`, `Uint`, `Float`, `Bool`, `Char`, `Never`.
//! 2. **References, raw pointers, function pointers, fn-defs**: `Ref`,
//!    `RawPtr`, `FnPtr`, `FnDef` — none own anything that needs
//!    running.
//! 3. **Arrays of trivial types**: `[T; N]` where `T` is itself
//!    trivial-drop (recursive).
//! 4. **Tuples of trivial types**: same recursion.
//!
//! ADTs (`Adt(_, _)`) and closures with captures stay on the
//! hard-error path. An `impl Drop` somewhere in the field tree would
//! be exactly the silent miscompile the original error message was
//! written to prevent.
//!
//! `rustc_public` (stable MIR) doesn't expose rustc's `needs_drop`
//! query directly — the recursive shape check is the principled
//! fallback for this layer.
//!
//! Originally surfaced from `~/vanity-miner-rs/`: the SHA-256 inner
//! loop uses `cmp::min` to clamp a digest comparison length, which
//! monomorphizes to `<usize as Ord>::min`.
//!
//! ## Build with
//!
//!     cargo oxide run drop_copy_primitive
//!
//! Expected: kernel runs, each output slot equals the corresponding
//! input.
//!
//! ## What this example is NOT
//!
//! - Not about `Drop` on user-defined types — `impl Drop for MyStruct`
//!   stays on the hard-error path until proper drop glue lands.
//! - Not about closure captures of non-Copy types.
//! - Not about `panic_fmt` or `dyn Debug` (covered by their own
//!   reproducers). The trigger here is purely the Copy-primitive
//!   drop terminator, reproducible in a one-line kernel.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Trigger: `.min(N)` on a `usize`. `<usize as Ord>::min` takes
    /// `self: usize` and `other: usize` by value, and rustc emits a
    /// `Drop(_unused)` terminator on whichever the match didn't
    /// return. Pre-fix, `translate_drop` hard-errored on the drop;
    /// post-fix, `has_no_drop_glue` recognizes `usize` and lowers the
    /// terminator to a plain `mir.goto`.
    #[kernel]
    pub fn min_clamp(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            // The `.min(input.len() - 1)` is the line that drags
            // `<usize as Ord>::min` into the MIR. Without it, the
            // kernel translates fine.
            let clamped = i.min(input.len().saturating_sub(1));
            *slot = input[clamped];
        }
    }
}

fn main() {
    println!("=== drop_copy_primitive ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 16;
    let host: Vec<u32> = (0..N as u32).map(|n| n * 11 + 3).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .min_clamp(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        assert_eq!(result[i], host[i], "thread {} mismatch", i);
    }
    println!("SUCCESS: <usize as Ord>::min codegen'd to PTX");
}
