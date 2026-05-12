//! Regression test for ZST kernel-arg slot handling across the three
//! lowering paths that classify function args. Surfaced from
//! `~/vanity-miner-rs/` via the `<GenericArray<u8, U33> as
//! GenericSequence>::generate` → `Iterator::for_each` → `Iterator::fold`
//! → `Enumerate` chain.
//!
//! ## Pre-fix walls
//!
//! ```text
//! Entry block arg mismatch: struct arg expects 2 non-ZST fields
//! but only 1 LLVM args remain at idx 3/4
//! ```
//!
//! followed (after fixing the entry-prologue side) by:
//!
//! ```text
//! Function call has incorrect type:
//!   expected llvm.func<i64(struct<{i64,i64}>)>,
//!   got      llvm.func<i64(struct<{}>, struct<{i64,i64}>)>
//! ```
//!
//! ## What landed
//!
//! Three lowering paths classify kernel arg types and must stay in
//! lockstep. `convert_function_type` (callee signature) skipped ZST
//! args silently in its `FlattenKind::None` arm. The two consumers
//! didn't:
//!
//! * `classify_argument_type` (entry-prologue) gained a new
//!   `ReconstructKind::Skip` variant. When the converted LLVM type
//!   is ZST, the prologue synthesizes an `undef` of the original
//!   MIR type and **doesn't advance** `llvm_arg_idx`.
//! * `flatten_arguments` (call site) — `FlattenKind::None` arm
//!   `continue`s if the arg's type is ZST, matching the signature's
//!   skip behavior.
//!
//! All three paths now agree: ZST top-level args contribute zero
//! LLVM args. `()` / `PhantomData` / capture-less closure values
//! still flow through MIR-level uses correctly because the entry
//! prologue feeds an `undef` of the right type.
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
