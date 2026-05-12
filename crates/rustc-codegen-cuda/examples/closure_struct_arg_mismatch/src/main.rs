//! Known-failure repro for the entry-prologue / call-site / function-sig
//! disagreement on ZST function args. Surfaced from `~/vanity-miner-rs/`
//! as a closure-struct mismatch where MIR arg slot 1 was the empty
//! tuple `()` and every subsequent struct arg's reconstruction
//! drifted off by one:
//!
//! ```text
//! Entry block arg mismatch: struct arg expects 2 non-ZST fields
//! but only 1 LLVM args remain at idx 3/4. MIR struct type:
//!   mir.struct <DefId{...fold::enumerate::{closure#0}},
//!     [capture_0, capture_1],
//!     [mir.struct <DefId{...for_each::call::{closure#0}}, ...>,
//!      builtin.integer ui64], [], [], 0>
//! ```
//!
//! ## What's happening
//!
//! Three lowering paths classify kernel arg types and must stay in
//! lockstep:
//!
//! * `convert_function_type` (callee LLVM signature) — `FlattenKind::None`
//!   arm: convert the type, push the LLVM type if non-ZST, **skip
//!   silently if ZST**.
//! * `flatten_arguments` (call site) — `FlattenKind::None` arm:
//!   pushes the arg unconditionally (no ZST check).
//! * `classify_argument_type` (callee entry-prologue reconstruction) —
//!   `ReconstructKind::None`: always reads exactly 1 LLVM arg.
//!
//! When a top-level arg is ZST (`()`, `PhantomData`, capture-less
//! closures), the signature emits 0 LLVM args while the call-site
//! and the entry-prologue still consume 1 each. The signature/prologue
//! disagreement trips the entry-prologue's "arg mismatch" verifier;
//! the signature/call-site disagreement trips the call-site's
//! "Function call has incorrect type" verifier.
//!
//! ## Repro shape
//!
//! `with_unit_then_struct(_unit: (), stuff: (u64, u64))` — `()` at a
//! non-final slot followed by a non-trivial tuple. `#[inline(never)]`
//! keeps the function in its own MIR signature so the entry-prologue
//! and call-site both run against it.
//!
//! ## Build with
//!
//!     cargo oxide build closure_struct_arg_mismatch

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Function with a `()` parameter at a non-final slot, followed by a
/// non-trivial struct arg. `convert_function_type` skips the `()`
/// (ZST), but `classify_argument_type` used to consume one LLVM arg
/// for it anyway — pushing every subsequent struct's reconstruction
/// off by one. `#[inline(never)]` keeps the function in its own MIR
/// signature so the entry-prologue actually runs against it.
#[inline(never)]
fn with_unit_then_struct(_unit: (), stuff: (u64, u64)) -> u64 {
    stuff.0.wrapping_add(stuff.1)
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && (i + 1) * 2 <= input.len()
        {
            let base = i * 2;
            *slot = super::with_unit_then_struct((), (input[base], input[base + 1]));
        }
    }
}

fn main() {
    println!("=== closure_struct_arg_mismatch ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 4;
    let host: Vec<u64> = (0..(N * 4) as u64).collect();
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
        let base = i * 2;
        let expected = host[base].wrapping_add(host[base + 1]);
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: ZST kernel-arg slot reconstruction OK");
}
