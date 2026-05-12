//! Regression test for the cuda-macros `#[device]` expansion bug
//! where the generated wrapper dropped `unsafe` from the user's
//! signature.
//!
//! ## Pre-fix diagnostic
//!
//! ```text
//! error[E0133]: call to unsafe function `cuda_oxide_device_…_atomic_add_u32`
//!               is unsafe and requires unsafe function or block
//! ```
//!
//! The error was inside the macro's generated wrapper — not at the
//! user's call site. `cuda-macros/src/lib.rs:2090-2102` (the
//! non-generic arm of `generate_device_function`) expanded
//! `#[device] pub unsafe fn foo` into roughly:
//!
//! ```rust,ignore
//! #[unsafe(no_mangle)]
//! pub unsafe fn cuda_oxide_device_HASH_foo(/* ... */) { /* user body */ }
//!
//! #[inline(always)]
//! pub fn foo(/* ... */) {                           // ← not `unsafe`!
//!     cuda_oxide_device_HASH_foo(/* ... */)        // ← E0133
//! }
//! ```
//!
//! The wrapper line `#vis fn #fn_name #generics …` had no
//! `#unsafety` token, so it was always safe regardless of the
//! user's modifier. Calling the unsafe inner from the safe wrapper
//! body then hit E0133. The generic arm at lines 2076-2087 had the
//! same bug.
//!
//! ## Fix
//!
//! Capture `input.sig.unsafety` and splice it into both arms'
//! wrapper signatures: `#vis #unsafety fn #fn_name …`. Rust 2024
//! removed implicit unsafe blocks inside `unsafe fn` bodies, so
//! the wrapper body conditionally wraps the inner call in
//! `unsafe { … }` when the inner is unsafe.
//!
//! ## What this example tests
//!
//! End-to-end: a `pub unsafe fn` annotated `#[device]` compiles
//! cleanly and is callable from a `#[kernel]`. The kernel does
//! one atomic increment on thread 0; main() checks the counter
//! is 1.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{
    atomic::{AtomicOrdering, DeviceAtomicU32},
    cuda_module, device, kernel, thread,
};

#[cuda_module]
mod kernels {
    use super::*;

    /// FAILS today: the macro-generated wrapper drops `unsafe` and
    /// then calls the unsafe inner from a safe context (E0133).
    #[device]
    pub unsafe fn atomic_add_u32(addr: &mut u32, val: u32) -> u32 {
        unsafe { DeviceAtomicU32::from_ptr(addr as *mut u32).fetch_add(val, AtomicOrdering::Relaxed) }
    }

    #[kernel]
    pub unsafe fn bump_counter(counter: &mut [u32]) {
        let idx = thread::index_1d();
        if idx.get() == 0 {
            // The macro generates a safe-fn wrapper for `atomic_add_u32`,
            // so calling it from this `unsafe fn` is fine on our end —
            // but the wrapper's body fails to compile internally.
            unsafe { atomic_add_u32(&mut counter[0], 1) };
        }
    }
}

fn main() {
    println!("=== #[device] unsafe wrapper repro ===\n");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let mut counter_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    unsafe {
        module
            .bump_counter(&stream, LaunchConfig::for_num_elems(32), &mut counter_dev)
            .expect("kernel launch");
    }

    let mut counter = [0u32; 1];
    counter_dev.copy_to_host(&stream, &mut counter).unwrap();
    stream.synchronize().unwrap();

    // With 32 threads but only thread 0 incrementing, expected = 1.
    assert_eq!(counter[0], 1, "atomic_add_u32 should have run once");
    println!("SUCCESS: #[device] unsafe wrapper compiled and ran");
}
