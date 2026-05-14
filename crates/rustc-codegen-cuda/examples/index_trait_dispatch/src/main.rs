//! Bug E: `core::ops::Index` / `IndexMut` trait dispatch on a
//! newtype wrapping `[u64; N]` is miscompiled.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. On real hardware (vanity-miner-rs
//! slot 91), `(read == val)` is 0 — the IndexMut write at one offset
//! is not visible to the Index read at the same offset.
//!
//! Surfaced from dalek `Scalar52` (the exact same newtype shape:
//! `pub struct Scalar52(pub [u64; 5])` with hand-written
//! `impl Index<usize>` returning `&self.0[i]`). The bug cascades:
//!
//! * slot 71 — `Scalar::from_bytes_mod_order(...).to_bytes()` round-trip
//! * slot 72 — Scalar1·basepoint = basepoint (depends on 71)
//! * slots 2/3/11/12 — ed25519 / solana derive (cascade of 71)
//!
//! Inspection of `kernels.ptx` for `Scalar52::sub` showed the bug
//! exactly: inside the `for i in 0..5` loop body, the `difference`
//! buffer is re-allocated and zero-init'd on every iteration, and
//! `constants::L[i]` is hardcoded to L[0] regardless of `i`. Both
//! map onto the broken `Index`/`IndexMut` dispatch.
//!
//! ## Pattern that triggers the bug
//!
//! ```rust
//! pub struct Wrap(pub [u64; 5]);
//!
//! impl Index<usize> for Wrap {
//!     type Output = u64;
//!     fn index(&self, i: usize) -> &u64 { &(self.0[i]) }
//! }
//! impl IndexMut<usize> for Wrap {
//!     fn index_mut(&mut self, i: usize) -> &mut u64 { &mut (self.0[i]) }
//! }
//!
//! let mut w = Wrap([0u64; 5]);
//! let idx = black_box(2usize);
//! let val = black_box(0xCAFEBABE_DEADBEEFu64);
//! w[idx] = val;            // IndexMut::index_mut
//! let read = black_box(w[idx]);   // Index::index
//! assert_eq!(read, val);   // FAILs today
//! ```
//!
//! ## Build
//!
//!     cargo oxide build index_trait_dispatch

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

pub struct IdxProbe(pub [u64; 5]);

impl core::ops::Index<usize> for IdxProbe {
    type Output = u64;
    fn index(&self, i: usize) -> &u64 {
        &(self.0[i])
    }
}

impl core::ops::IndexMut<usize> for IdxProbe {
    fn index_mut(&mut self, i: usize) -> &mut u64 {
        &mut (self.0[i])
    }
}

#[inline(never)]
fn check_index_trait_dispatch() -> u32 {
    let mut p = IdxProbe([0u64; 5]);
    let idx = core::hint::black_box(2usize);
    let val = core::hint::black_box(0xCAFE_BABE_DEAD_BEEF_u64);
    p[idx] = val;
    let read = core::hint::black_box(p[idx]);
    (read == val) as u32
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            *slot = check_index_trait_dispatch();
        }
    }
}

fn main() {
    println!("=== index_trait_dispatch ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let n_out = 4usize;
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, n_out).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(n_out as u32), &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for (i, r) in result.iter().enumerate() {
        assert_eq!(*r, 1, "thread {} got {} (expected 1)", i, r);
    }
    println!("SUCCESS: Index/IndexMut trait dispatch round-trips");
}
