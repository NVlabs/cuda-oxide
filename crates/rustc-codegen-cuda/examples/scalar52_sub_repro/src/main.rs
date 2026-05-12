//! Known-failure repro for translator structural bug: a `mir.goto`
//! terminator landing not-last in its basic block.
//!
//! ## Wall (current state)
//!
//! ```text
//! Verification failed for '...::Scalar52::sub':
//! Basic block "block2953v1" has a terminator that is not the last
//! operation in the block
//!   Failed operation:
//!     mir.goto () [^block2943v1] []: <() -> ()> !0
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `curve25519_dalek::backend::serial::u64::scalar::Scalar52::sub` at
//! `scalar.rs:190:9: 193:10`. The shape is a borrow-chain subtraction
//! loop:
//!
//! ```ignore
//! let mut borrow: u64 = 0;
//! for i in 0..5 {
//!     borrow = a[i].wrapping_sub(b[i] + (borrow >> 63));
//!     difference[i] = borrow & mask;
//! }
//! ```
//!
//! `a` / `b` are `&Scalar52` (== `&[u64; 5]`); `difference` is
//! `&mut Scalar52`. The loop body has both `a[i]` reads through a
//! reference and `difference[i] = ...` writes through a `&mut` ref —
//! the latter goes through the `Deref → Index(local)` writer arm.
//!
//! ## Where to look
//!
//! `mir.goto` is only emitted in `translator/terminator/mod.rs`
//! (`translate_goto`, `translate_drop`'s trivial-drop arm) and in
//! `translator/terminator/helpers.rs` (`emit_goto`). The verifier
//! complains the goto isn't last, which means *something later in
//! statement translation appended an op to the same block AFTER the
//! terminator was emitted* — the `prev_op` chaining must be off by
//! one somewhere when the loop's back-edge or the borrow-chain
//! iteration emits its sequence.
//!
//! Best approach: build this repro and dump the LLVM/MIR for the
//! failing block to see exactly which op landed after the goto.
//!
//! ## Build with
//!
//!     cargo oxide build scalar52_sub_repro

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirrors `curve25519_dalek::backend::serial::u64::scalar::Scalar52`:
/// a tuple struct wrapping `[u64; 5]` with custom `Index` /
/// `IndexMut` impls. The trait dispatch (instead of direct array
/// indexing) is what changes the MIR shape.
#[derive(Clone, Copy)]
pub struct Scalar52Like(pub [u64; 5]);

impl core::ops::Index<usize> for Scalar52Like {
    type Output = u64;
    fn index(&self, i: usize) -> &u64 {
        &self.0[i]
    }
}

impl core::ops::IndexMut<usize> for Scalar52Like {
    fn index_mut(&mut self, i: usize) -> &mut u64 {
        &mut self.0[i]
    }
}

/// Same skeletal body as the real `Scalar52::sub`: borrow-chain
/// subtraction with `a[i]` reads and `difference[i]` writes going
/// through the trait Index / IndexMut calls. `#[inline(never)]`
/// keeps the loop body (the suspected trigger) in its own MIR
/// function.
#[inline(never)]
fn sub_borrow_chain(a: &Scalar52Like, b: &Scalar52Like) -> Scalar52Like {
    let mut difference = Scalar52Like([0u64; 5]);
    let mask: u64 = (1u64 << 52) - 1;
    let mut borrow: u64 = 0;
    for i in 0..5 {
        borrow = a[i].wrapping_sub(b[i].wrapping_add(borrow >> 63));
        difference[i] = borrow & mask;
    }
    difference
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn scalar52_sub(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && (i + 1) * 10 <= input.len()
        {
            let base = i * 10;
            let a = super::Scalar52Like([
                input[base],
                input[base + 1],
                input[base + 2],
                input[base + 3],
                input[base + 4],
            ]);
            let b = super::Scalar52Like([
                input[base + 5],
                input[base + 6],
                input[base + 7],
                input[base + 8],
                input[base + 9],
            ]);
            let d = super::sub_borrow_chain(&a, &b);
            *slot = d.0[0] ^ d.0[1] ^ d.0[2] ^ d.0[3] ^ d.0[4];
        }
    }
}

fn main() {
    println!("=== scalar52_sub_repro ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 4;
    let host: Vec<u64> = (0..(N * 10) as u64).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .scalar52_sub(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    let mask: u64 = (1u64 << 52) - 1;
    for i in 0..N {
        let base = i * 10;
        let a: [u64; 5] = [
            host[base], host[base + 1], host[base + 2], host[base + 3], host[base + 4],
        ];
        let b: [u64; 5] = [
            host[base + 5], host[base + 6], host[base + 7], host[base + 8], host[base + 9],
        ];
        let mut d = [0u64; 5];
        let mut borrow: u64 = 0;
        for k in 0..5 {
            borrow = a[k].wrapping_sub(b[k].wrapping_add(borrow >> 63));
            d[k] = borrow & mask;
        }
        let expected = d[0] ^ d[1] ^ d[2] ^ d[3] ^ d[4];
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: scalar52_sub-shaped borrow chain codegen'd to PTX");
}
