//! Regression test for an orphan-op leak from `translate_place_addr_from_slot`
//! that produced "terminator not last" verifier failures on tuple-struct
//! `IndexMut` writes inside loops.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Verification failed for '...::Scalar52::sub':
//! Basic block "block2953v1" has a terminator that is not the last
//! operation in the block
//!   Failed operation:
//!     mir.goto () [^block2943v1] []
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via
//! `curve25519_dalek::backend::serial::u64::scalar::Scalar52::sub` at
//! `scalar.rs:190:9: 193:10`. `Scalar52` is a tuple struct wrapping
//! `[u64; 5]` with custom `Index` / `IndexMut` impls; inside the
//! borrow-chain loop, `&mut difference[i] = ...` translated to a
//! reference of a place with projections `[Field(0), Index(_i)]`
//! against the slot.
//!
//! ## What was happening
//!
//! `Rvalue::Ref` Case 4 calls `translate_place_addr_from_slot` to
//! compute the in-memory address. The helper walks projections
//! emitting ops as it goes — `Field(0)` produced a `mir.field_addr`,
//! then it hit `Index(_i)` which the catch-all arm bailed on with
//! `Ok(None)`. The `mir.field_addr` was already in the block.
//! The caller saw `None` and fell back to Case 5 (materialise +
//! `mir.ref`), which doesn't know about the orphan. Subsequent
//! statements chained against the original `prev_op`, so the
//! orphaned `field_addr` floated to the bottom of the block —
//! after the loop's back-edge `mir.goto`.
//!
//! ## What landed
//!
//! `translate_place_addr_from_slot` now pre-validates the projection
//! list before inserting any ops. If anything other than `Field` or
//! `ConstantIndex { from_end: false }` is present, it returns
//! `Ok(None)` immediately — no half-built op chain leaks into the
//! block. The caller's fallback path stays as the only emitter.
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
