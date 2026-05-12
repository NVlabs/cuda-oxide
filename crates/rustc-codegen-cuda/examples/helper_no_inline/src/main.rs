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
//! ## Side-effect: adding `#[inline]` exposes a different bug
//!
//! Reproducing locally on current `main`: adding `#[inline]` here flips
//! the diagnostic to:
//!
//!   Unsupported construct: Type translation not yet implemented for:
//!   RigidTy(FnPtr(... std::fmt::Formatter ...))
//!
//! Inlining the helper pulls `thread::index_1d()`'s body into the
//! kernel's own MIR, where the importer trips over a `FnPtr` referencing
//! `<_ as Display>::fmt`. The non-inline path quietly avoids this by
//! keeping the offending body in a separate function the importer
//! never visits. So even when bug B's `#[inline]` workaround is
//! applied, real kernels that wrap intrinsics still hit a deeper
//! limitation. Worth flagging in the upstream issue.

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
    /// (To make this example compile cleanly, add `#[inline]` here.)
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
