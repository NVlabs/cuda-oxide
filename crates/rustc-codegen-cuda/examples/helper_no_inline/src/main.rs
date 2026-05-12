//! Reproducer for bug B of limitation #4: helpers *inside* `#[cuda_module]`
//! still silently require `#[inline]`.
//!
//! Observed diagnostic on current `main`:
//!
//!   Verification failed for 'llvm module': Symbol
//!   helper_no_inline__kernels__get_thread_idx not found
//!
//! Same shape as bug A — the call site is emitted but the body isn't —
//! except this time the helper is *inside* the `#[cuda_module]` mod, so
//! the workaround for bug A ("move it inside the mod") doesn't help.
//! The second, undocumented requirement is `#[inline]`.
//!
//! ## How to flip this from failing to passing
//!
//! Add `#[inline]` above `get_thread_idx`. Nothing else changes; the
//! attribute alone is load-bearing. This diagnostically isolates
//! `#[inline]` as the load-bearing thing — not visibility, not module
//! placement, not call-site qualification.
//!
//! The in-tree `cross_crate_kernel/kernel-lib/src/lib.rs:45-48` example
//! relies on this exact attribute for its `device_helper` but the
//! `#[cuda_module]` macro README doesn't surface that requirement.
//!
//! ## The `#[inline]` "workaround" doesn't actually work
//!
//! We previously thought adding `#[inline]` here was a clean workaround
//! because the build succeeded. It isn't. With `#[inline]`, this
//! example builds successfully BUT the kernel's PTX body collapses
//! to a single `exit;` — the kernel does nothing at runtime.
//!
//! ### Root cause (current understanding)
//!
//! `cuda_device::thread::index_1d()` is an intrinsic with a host-side
//! `unreachable!("thread::index_1d called outside #[kernel] / #[device]
//! — the macro rewrites real call sites; the public item is a stub")`
//! body. The `#[cuda_module]`/`#[kernel]` macros rewrite intrinsic
//! call sites at expansion time, but **only inside `#[kernel]` items**
//! — non-kernel helpers in the same mod retain the stub body.
//!
//! - Without `#[inline]`: the kernel's call site references a function
//!   whose body is the unreachable stub. The collector skips bodies
//!   that are just panic calls (`is_unreachable_body` in
//!   `collector.rs:1142`), so the symbol is declared but never
//!   defined. LLVM verification fails with `Symbol ... not found`.
//!   This is the loud failure this example reproduces.
//!
//! - With `#[inline]`: rustc inlines the stub into the kernel's body.
//!   The optimizer sees `unreachable!` at the top of the kernel and
//!   collapses everything after it. The kernel body reduces to
//!   `panic_fmt("...stub message...")` and the importer lowers that
//!   to LLVM `unreachable`, which llc renders as `exit;`. Silent
//!   empty PTX.
//!
//! ### Real fix
//!
//! Annotate the intrinsic-wrapping helper with `#[device]`. The
//! `#[device]` proc-macro (already exported from `cuda_device`) runs
//! the same `inject_thread_index_scope` hook `#[kernel]` uses, so the
//! `thread::index_1d()` call inside gets rewritten to its
//! `__internal::*` form and the public stub body never executes.
//!
//! A defensive Layer-1 check in
//! `crates/rustc-codegen-cuda/src/collector.rs` now errors out when a
//! kernel body collapses to `panic_fmt(...)`, so the silent
//! empty-PTX failure mode is detected at build time even when a user
//! forgets `#[device]`. This example reproduces the loud "Symbol not
//! found" form (no `#[inline]`); adding `#[inline]` triggers the new
//! Layer-1 diagnostic instead.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// FAILS: same body as the `#[inline]` device_helper in
    /// cross_crate_kernel/kernel-lib, only without the attribute. The
    /// PTX module ends up with a call site to
    /// `helper_no_inline__kernels__get_thread_idx` but no body.
    ///
    /// **WARNING:** adding `#[inline]` to this function does NOT fix the
    /// bug. With `#[inline]`, rustc's optimizer collapses the inlined
    /// `unreachable!` stub up into the kernel body, the resulting PTX
    /// kernel body becomes a single `exit;`, and the build reports
    /// success — a silent no-op kernel.
    ///
    /// The Layer-1 collector check in `crates/rustc-codegen-cuda/src/collector.rs`
    /// now hard-errors when a kernel body collapses to `panic_fmt(...)`,
    /// converting that silent failure into a build error.
    ///
    /// The real fix is to annotate this helper with `#[device]`. That
    /// attribute runs the same `inject_thread_index_scope` hook
    /// `#[kernel]` uses, which rewrites the `thread::index_1d()` call
    /// site to its `__internal::*` form so the stub body never runs.
    pub fn get_thread_idx() -> usize {
        thread::index_1d().get()
    }

    #[kernel]
    pub fn fill_with_helper(mut out: DisjointSlice<u32>) {
        let i = get_thread_idx();
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            *slot = i as u32;
        }
    }
}

fn main() {
    println!("=== Limitation #4 bug B: helper inside mod, no #[inline] ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 64;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .fill_with_helper(&stream, LaunchConfig::for_num_elems(N as u32), &mut out)
        .expect("Kernel launch failed");

    let r = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        assert_eq!(r[i], i as u32);
    }

    println!("SUCCESS: non-inline helper inside #[cuda_module] codegen'd to PTX");
}
