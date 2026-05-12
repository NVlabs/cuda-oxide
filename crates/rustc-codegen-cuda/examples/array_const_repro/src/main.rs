//! Regression test for the `MirArrayType` by-value-constant gap in
//! mir-importer's `translate_constant`.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Unsupported construct: Unsupported constant type in translate_constant.
//!   Rust type : Ty { ... RigidTy(Array(Uint(U32), 8)) }
//!   pliron type: MirArrayType { element_ty: ..., size: 8 }
//!   const repr : MirConst { kind: Allocated(...), ty: ..., id: ... }
//!
//!   The type dispatch (ZST -> ptr_to_array -> struct -> enum -> float
//!     -> pointer -> integer) did not match this constant.
//! ```
//!
//! ## Why this fires
//!
//! `const K: [u32; 8]` (note: `const`, not `static`) is treated by
//! rustc as a value substitution rather than a memory location. When
//! the kernel does `K[idx]`, MIR carries the entire array literal as
//! a single operand of `MirArrayType` to the indexing op. The
//! pointer-to-array path (`&K[i]` style) is already handled by
//! `translate_ptr_to_array_constant`, but the by-value path landed
//! in the catch-all "no matching type handler" arm.
//!
//! The shallenge-shaped reproducer hits this on SHA-256's
//! `const K: [u32; 64]` (the round constants).
//!
//! ## What landed
//!
//! `crates/mir-importer/src/translator/rvalue.rs` got a `MirArrayType`
//! dispatch arm in `translate_constant` that routes to a new
//! `translate_array_constant` helper. The helper mirrors the
//! per-element loop from the existing `translate_ptr_to_array_constant`
//! (bytes → typed constants → `MirConstructArrayOp`), minus the
//! trailing `MirRefOp` wrap, since the result is the array value
//! itself rather than a pointer to it.
//!
//! After this, `const TABLE: [u32; 8]` indexed by a runtime index
//! materialises all eight values into PTX (`st.local.b32 [%rd1], 10;
//! ... [%rd1+28], 80;`) instead of collapsing to `exit;` from the
//! upstream error.
//!
//! Unblocks the SHA-256 K-table pattern in `shallenge`-shaped kernels.
//!
//! ## What this exercises
//!
//! Smallest possible trigger: index a `const [u32; N]` by a runtime
//! index inside a kernel. Eight elements is enough; the constant only
//! has to be `[T; N]`-typed.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const TABLE: [u32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn read_const_array(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            // `TABLE[i % 8]` carries the entire 8-element u32 array as a
            // MirArrayType-typed const operand — handled by the
            // `translate_array_constant` helper added in 2e9fbf6.
            *slot = input[i].wrapping_add(TABLE[i % 8]);
        }
    }
}

fn main() {
    println!("=== Array<T, N> by-value constant repro ===\n");

    let ctx = CudaContext::new(0).expect("CudaContext::new");
    let stream = ctx.default_stream();

    const N: usize = 16;
    let host: Vec<u32> = (0..N as u32).collect();
    let dev = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .read_const_array(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &dev,
            &mut out,
        )
        .expect("kernel launch");

    let r = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let expected = host[i].wrapping_add(TABLE[i % 8]);
        assert_eq!(r[i], expected, "idx {}: {} != {}", i, r[i], expected);
    }

    println!("SUCCESS: kernel indexed `const [u32; 8]` correctly");
}
