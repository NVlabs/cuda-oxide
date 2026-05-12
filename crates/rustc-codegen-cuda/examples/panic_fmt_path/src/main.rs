//! Reproducer for the `RigidTy(FnPtr(... std::fmt::Formatter ...))`
//! limitation in mir-importer's type translator.
//!
//! Observed diagnostic on current `main`:
//!
//!   Unsupported construct: Type translation not yet implemented for:
//!   RigidTy(FnPtr(Binder { value: FnSig { inputs_and_output: [
//!     Ty { ... "std::ptr::NonNull" ... },
//!     Ty { ... Ref Mut "std::fmt::Formatter" ... },
//!     Ty { ... "std::result::Result" ... "std::fmt::Error" ... }
//!   ], ... }})
//!
//! That function pointer is the type-erased `<_ as Display>::fmt` slot
//! that `core::fmt::Arguments` uses to format each placeholder. Any
//! kernel-reachable call that goes through `core::panicking::panic_fmt`
//! drags it in. That covers most of the natural Rust panic paths:
//!
//!   - `assert!(cond, "msg")`
//!   - `panic!("msg")`
//!   - `unreachable!("msg")`
//!   - `.expect("msg")` on `Option` / `Result`
//!   - `.unwrap()` on `Option` / `Result` (lowers through panic_fmt)
//!
//! ## Relationship to limitation #1 (RigidTy(Str))
//!
//! The brief framed #1 as "fixing `RigidTy(Str)` unlocks every panic
//! message, every `.unwrap`, every slice/array bounds check." Fixing
//! #1 gets the type translator past `Str`, but `assert!` / `panic!` /
//! `.unwrap` then hit this `FnPtr` arm next — a deeper layer. The
//! `str_panic_path` reproducer for #1 carefully avoids this by using
//! `"abc".len()` rather than a panic path, so #1 can be exercised in
//! isolation.
//!
//! ## Relationship to limitation #4 (helper #[inline])
//!
//! The same `FnPtr` shows up when a helper inside `#[cuda_module]` is
//! marked `#[inline]` and its body wraps `cuda_device::thread::index_1d()`.
//! See `helper_no_inline/src/main.rs` for the second independent
//! trigger. Two independent paths converging on the same FnPtr suggests
//! the issue is the importer's type translator missing a function-pointer
//! arm, not anything specific to panics.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Trigger 1: `assert!` with a static-string message. The false-edge
    /// reaches `core::panicking::panic_fmt(format_args!("msg"))`, which
    /// carries the Formatter FnPtr through the kernel CFG.
    #[kernel]
    pub fn passthrough_with_assert(data: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            // Always true at runtime, but the panic edge is in the MIR.
            assert!(i < usize::MAX, "thread index overflow");
            *slot = data[i];
        }
    }
}

fn main() {
    println!("=== fmt::Arguments / Formatter FnPtr reproducer ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let host: Vec<u32> = (0..N as u32).collect();
    let dev = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .passthrough_with_assert(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &dev,
            &mut out,
        )
        .expect("Kernel launch failed");

    let r = out.to_host_vec(&stream).unwrap();
    assert_eq!(r, host);

    println!("SUCCESS: panic-fmt FnPtr path codegen'd to PTX");
}
