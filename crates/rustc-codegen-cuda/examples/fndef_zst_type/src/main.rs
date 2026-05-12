//! Known-failure repro for the type translator's missing `FnDef` arm.
//!
//! ## Wall (current state)
//!
//! ```text
//! Compilation error: invalid input program.
//! Unsupported construct: Type translation not yet implemented for:
//!   RigidTy(FnDef(FnDef(DefId { ..., name: "std::fmt::Display::fmt" }),
//!     GenericArgs([Type(Ty { ..., kind: RigidTy(Uint(Usize)) })])))
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `curve25519_dalek::scalar::read_le_u64_into`, whose body calls
//! `assert!(src.len() == 8 * dst.len())`. The assert's panic edge
//! materializes `Display::fmt::<usize>` as a `FnDef` value to embed
//! in `core::fmt::Arguments`.
//!
//! `RigidTy::FnDef` is a per-function-definition ZST. Rustc carries
//! it everywhere a "specific function" is named (vs. erased
//! function-pointer / closure types). Layout is zero bytes; semantically
//! it's just a marker that uniquely identifies the function and can
//! coerce to a `fn(...)` pointer. The type translator has arms for
//! `FnPtr` and `Dynamic` (both modelled as opaque generic pointers
//! since they only appear on panic edges) but no arm for `FnDef`.
//!
//! ## What a fix needs to do
//!
//! Add a `RigidTy::FnDef` arm that returns the empty tuple type
//! `MirTupleType::get(ctx, vec![])` — the same shape `is_rust_type_zst`
//! already treats as zero-sized. That lets `FnDef` flow through stores,
//! loads, and aggregate fields the way other ZSTs (`PhantomData`, unit
//! structs) do.
//!
//! ## Build with
//!
//!     cargo oxide build fndef_zst_type

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// `assert!` triggers the same `Display::fmt::<usize>` materialization
/// that curve25519-dalek's `read_le_u64_into` hits — the panic-fmt
/// edge embeds the function as a FnDef ZST in `core::fmt::Arguments`.
/// `#[inline(never)]` keeps the assertion in its own MIR function.
#[inline(never)]
fn checked_div(a: usize, b: usize) -> usize {
    assert!(b != 0, "division by zero: a = {}, b = {}", a, b);
    a / b
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn divide(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            let a = input[i] as usize;
            // b is always non-zero so the assert never fires at runtime.
            let b = (i + 1).max(1);
            *slot = super::checked_div(a, b) as u32;
        }
    }
}

fn main() {
    println!("=== fndef_zst_type ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u32> = (0..N as u32).map(|n| n * 100).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .divide(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = host[i] / ((i + 1) as u32);
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: FnDef ZST type codegen'd to PTX");
}
