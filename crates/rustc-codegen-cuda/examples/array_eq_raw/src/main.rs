//! Known-failure reproducer for `core::intrinsics::raw_eq` reaching
//! the mir-importer without an intrinsic handler.
//!
//! ## Expected failure
//!
//! ```text
//! error: [rustc_codegen_cuda] Device codegen failed: PTX generation
//!        failed: Verification failed for 'llvm module':
//!        Compilation error: verification failed.
//!        Symbol _RINvNtCs..._4core10intrinsics6raw_eqAhj10_E... not found
//! ```
//!
//! Decoded: `core::intrinsics::raw_eq::<[u8; 16]>`.
//!
//! ## What triggers it
//!
//! Comparing two fixed-size arrays with `==`:
//!
//! ```rust,ignore
//! let a: [u8; 16] = ...;
//! let b: [u8; 16] = ...;
//! if a == b { ... }
//! ```
//!
//! `<[T; N] as PartialEq>::eq` (in `core/src/array/equality.rs`) takes
//! a memcmp-style fast path for types where `T: BytewiseEq` (any
//! integer / `bool` / `char`). It calls `core::intrinsics::raw_eq`, a
//! compiler-built-in that does an N-byte equality comparison on the
//! raw memory of the two operands. cuda-oxide's MIR importer has no
//! arm for `raw_eq` in
//! `crates/mir-importer/src/translator/terminator/intrinsics/`, so
//! the call survives as an extern reference. PTX assembly later fails
//! because the symbol has no body anywhere in the module.
//!
//! Same root cause — any `==` on `[T; N]` for `T: BytewiseEq` (or any
//! `#[derive(Eq)]` struct made of such fields) produces the identical
//! failure. Comparing slices (`&[T]`) takes a different path that
//! lowers to a length check + element-wise loop, so it doesn't trip.
//! Comparing scalars (`a == b` for `u8`) also doesn't trip — `raw_eq`
//! is only used when the operand is wider than a register-sized
//! primitive.
//!
//! ## What it would take to fix
//!
//! Add a `raw_eq` arm in
//! `crates/mir-importer/src/translator/terminator/intrinsics/`,
//! analogous to the existing `convert_rust_bit_intrinsic` /
//! `ptr_offset_from_unsigned` handlers. `raw_eq::<T>(a, b) -> bool`
//! has a known operand layout — both `a` and `b` are `&T` — and the
//! body is a fixed-size byte comparison whose length is known from
//! `T`'s size. Two reasonable lowerings:
//!
//! 1. **Bitcast + scalar compare** for power-of-two sizes ≤ 16 bytes:
//!    `bitcast &T -> *u128 / *u64 / *u32 / *u16 / *u8`, load both
//!    sides, integer compare. Cheapest, no loop, optimal codegen for
//!    the common case (`[u8; 16]` keys, `[u8; 32]` digests).
//!
//! 2. **Element-wise loop** for arbitrary sizes: emit a `for k in
//!    0..size_of::<T>()` byte-by-byte loop with early exit. Slower
//!    but unconditionally correct.
//!
//! Tier 1 covers the cryptographic-key sizes that motivated this
//! reproducer (16, 20, 32, 64). Tier 2 is the principled fallback.
//!
//! Originally surfaced from `~/vanity-miner-rs/`: SHA-256 / ed25519
//! key matching uses `[u8; 16]` and `[u8; 32]` equality on the
//! device side.
//!
//! ## Build with
//!
//!     cargo oxide build array_eq_raw
//!
//! Expected: build error from llc — undefined symbol
//! `core::intrinsics::raw_eq::<[u8; 16]>`.
//!
//! ## What this example is NOT
//!
//! - Not about `Result::unwrap` / `&dyn Debug`
//!   (covered by `examples/result_unwrap_dyn_debug/`).
//! - Not about slice equality (`&[T] == &[T]`) — that lowers
//!   element-wise and works fine.
//! - Not about cross-crate `pub fn` (covered by `cross_crate_pubfn/`).
//!   The trigger is purely intra-`core`, reproducible in a one-file
//!   kernel.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// FAILS: the `a == b` line is the trigger. `<[u8; 16] as
    /// PartialEq>::eq` calls `core::intrinsics::raw_eq::<[u8; 16]>`,
    /// which has no handler in the mir-importer.
    #[kernel]
    pub fn array_eq_raw(input: &[u8], target: &[u8], mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            // SAFETY: caller ensures `input.len() >= (i + 1) * 16` and
            // `target.len() >= 16`. Build the two arrays bytewise so
            // we don't drag any other intrinsic into the MIR — the
            // only interesting operation is the `==` below.
            let mut a = [0u8; 16];
            let mut b = [0u8; 16];
            let base = i * 16;
            let mut k = 0;
            while k < 16 {
                a[k] = input[base + k];
                b[k] = target[k];
                k += 1;
            }
            *slot = if a == b { 1 } else { 0 };
        }
    }
}

fn main() {
    println!("=== array_eq_raw ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const THREADS: usize = 8;
    const CHUNK: usize = 16;
    const INPUT_LEN: usize = THREADS * CHUNK;

    // Input: bytes 0..128. Target: matches chunk 3 exactly.
    let host: Vec<u8> = (0..INPUT_LEN as u32).map(|n| (n & 0xff) as u8).collect();
    let target: Vec<u8> = host[3 * CHUNK..3 * CHUNK + CHUNK].to_vec();

    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let target_dev = DeviceBuffer::from_host(&stream, &target).unwrap();
    let mut out = DeviceBuffer::<u8>::zeroed(&stream, THREADS).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .array_eq_raw(
            &stream,
            LaunchConfig::for_num_elems(THREADS as u32),
            &input,
            &target_dev,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..THREADS {
        let expected = if i == 3 { 1 } else { 0 };
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: [u8; 16] == [u8; 16] codegen'd to PTX");
}
