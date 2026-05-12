//! Reproducer for the `bitcast [N x i8] to iX` mir-lower limitation.
//!
//! ## Pre-fix diagnostic
//!
//! ```text
//! /nix/store/.../llc: array_to_int_cast.ll:NN:NN: error:
//!   invalid cast opcode for cast from '[4 x i8]' to 'i32'
//!     %vN = bitcast [4 x i8] %vM to i32
//! ```
//!
//! ## Where this comes from
//!
//! `u32::from_be_bytes([b0, b1, b2, b3])` in MIR lowers to:
//!   1. `Rvalue::Aggregate(Array, [b0, b1, b2, b3])` → `[i8; 4]`
//!   2. `Rvalue::Cast(Transmute, _, u32)` → `i32`
//!
//! The Transmute step routes through `emit_pointer_cast` in
//! `crates/mir-lower/src/convert/ops/cast.rs`. That helper handles
//! struct↔ptr, struct↔int (via memory round-trip), and ptr↔int, but
//! has no arm where the source or destination is an LLVM `ArrayType`.
//! It falls through to the catch-all `BitcastOp`, which LLVM rejects
//! for aggregates: `bitcast` requires equal-sized first-class scalar
//! types.
//!
//! ## Fix
//!
//! Mirror the existing struct↔scalar memory round-trip: `alloca {T}`
//! where `T` is the larger of the two types, `store` the source
//! value typed as the source, `load` typed as the destination.
//!
//! ## What this example tests
//!
//! A minimal SHA-256-flavoured byte-shuffle: read four `u8`s from an
//! input slice, concatenate via `u32::from_be_bytes`, store into a
//! `u32` slot. The kernel call generates the same `[4 x i8] → i32`
//! transmute the SHA-256 message-schedule does in shallenge_repro,
//! without the rest of the cryptographic machinery.
//!
//! Surfaces in shallenge_repro at:
//!
//!   shallenge_repro.ll:901:19: invalid cast opcode for cast from
//!   '[4 x i8]' to 'i32'
//!     %v112 = bitcast [4 x i8] %v111 to i32

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// FAILS: `u32::from_be_bytes(...)` lowers to a Transmute that
    /// `emit_pointer_cast` can't model, falling through to a raw
    /// `bitcast [4 x i8] to i32` that llc rejects.
    #[kernel]
    pub fn pack_be_bytes(input: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            let base = i * 4;
            // Same byte-shuffle as SHA-256's message-schedule load step.
            let word = u32::from_be_bytes([
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
            ]);
            *slot = word;
        }
    }
}

fn main() {
    println!("=== [4 x i8] → i32 transmute (u32::from_be_bytes) repro ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 16; // 4 u32 words = 16 input bytes
    let host: Vec<u8> = (0..(N * 4) as u8).collect();
    let dev = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .pack_be_bytes(&stream, LaunchConfig::for_num_elems(N as u32), &dev, &mut out)
        .expect("Kernel launch failed");

    let r = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let base = i * 4;
        let expected = u32::from_be_bytes([
            host[base],
            host[base + 1],
            host[base + 2],
            host[base + 3],
        ]);
        assert_eq!(r[i], expected, "slot {} mismatch", i);
    }

    println!("SUCCESS: u32::from_be_bytes lowered through mir-lower without invalid bitcast");
}
