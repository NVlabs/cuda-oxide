//! Regression test for closure-body resolution when an outer closure
//! captures another closure by move and invokes it via `FnMut::call_mut`.
//!
//! ## Pre-fix wall (from `~/vanity-miner-rs/`)
//!
//! ```text
//! [Span core/src/iter/traits/iterator.rs:2985:20: 2985:33]
//! Function call has incorrect type:
//!   expected llvm.func <i1(ptr, struct<{ptr}>)>
//!   got      llvm.func <i1(ptr, ptr)>
//! ```
//!
//! Line 2985 of `iterator.rs` is the body of `Iterator::find`'s helper:
//!
//! ```ignore
//! fn check<T>(mut predicate: impl FnMut(&T) -> bool)
//!     -> impl FnMut((), T) -> ControlFlow<T>
//! {
//!     move |(), x| {
//!         if predicate(&x) { ControlFlow::Break(x) } else { ... }
//!         //  ^^^^^^^^^^^^^^^ predicate is captured by move into the
//!         //                  outer closure. Calling it goes through
//!         //                  `<PredicateClosure as FnMut>::call_mut`.
//!     }
//! }
//! ```
//!
//! In MIR the inner call surfaces as:
//!
//! ```ignore
//! _7 = <P as FnMut<(&T,)>>::call_mut(&mut (*_self).0, (move _4,))
//! ```
//!
//! cuda-oxide's `translate_closure_call` unpacks the tuple
//! `(move _4,)` so the actual closure body (which takes unpacked
//! args) gets called correctly. But `extract_closure_body_name`
//! reads the LOCAL'S declared type, not the place-after-projection
//! type. If `args[0]` is `&mut (*self).predicate_field` materialised
//! directly through a Deref+Field projection rather than via an
//! intermediate ref-local, the local's type is `&mut OuterClosure`
//! (not `&mut PredicateClosure`), so body-name resolution either
//! returns the wrong name or falls back to `call_name` (the FnMut
//! trait shim). The trait shim's signature is tuple-wrapped, but the
//! call site is already unpacked → `expected struct{ptr}, got ptr`.
//!
//! ## Build with
//!
//!     cargo oxide build nested_closure_capture_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            // Same shape as curve25519-dalek's `mul_base` (the
            // function where the vanity-miner-rs wall surfaces):
            //
            //   for i in (0..N).filter(|x| x % 2 == 1) { ... }
            //
            // `Filter::next` internally calls
            // `self.iter.find(&mut self.predicate)`. That hands
            // `find` a `&mut Predicate` — a reference to the
            // captured predicate closure, not the closure value.
            // `find::check` then wraps that reference, and inside
            // `check`'s body the call `predicate(&x)` becomes
            // `<&mut P as FnMut<(&u32,)>>::call_mut(...)`. `args[0]`
            // of that call is built from a `Deref+Field` projection
            // of the outer-closure env — the projection shape the
            // bug triggers on.
            let target = input[i] as u32;
            let mut found: u64 = 0;
            for v in (0u32..1000).filter(|x| x % 2 == 1) {
                if v.wrapping_mul(7).wrapping_add(3) == target {
                    found = v as u64;
                    break;
                }
            }
            *slot = found;
        }
    }
}

fn main() {
    println!("=== nested_closure_capture_repro ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 4;
    let host: Vec<u64> = vec![10, 20, 30, 40];
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let threshold = host[i];
        let expected = host.iter().find(|&&x| x > threshold).copied().unwrap_or(0);
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: nested closure capture codegen'd to PTX");
}
