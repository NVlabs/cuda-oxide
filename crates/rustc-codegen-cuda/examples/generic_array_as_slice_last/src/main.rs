//! Bug-96 narrowed: `&GenericArray<u8, N>::as_slice().last()` chain.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. vanity-miner-rs slot 101 FAILs on
//! hardware (v1.53.0): a function takes `&GenericArray<u8, U32>`,
//! does `value.as_slice().last()`, returns the last byte. The
//! returned byte is wrong (likely 0 or garbage). Slot 100 (same
//! algorithm with raw `[u8; 33]` instead of GA) PASSes — so the
//! bug is specifically in the GA → slice → `.last()` chain.
//!
//! This is the exact shape `sec1::EncodedPoint::
//! from_affine_coordinates` runs at the very first line:
//!
//! ```ignore
//! let tag = if compress {
//!     Tag::compress_y(y.as_slice())   // ← `y: &GA<u8, N>`,
//!                                     //    `as_slice() -> &[u8]`,
//!                                     //    `compress_y` uses `last()`
//! } else { Tag::Uncompressed };
//! ```
//!
//! Where `Tag::compress_y`:
//! ```ignore
//! fn compress_y(y: &[u8]) -> Self {
//!     if y.last().expect("...") & 1 == 1 { Self::CompressedOddY } else { Self::CompressedEvenY }
//! }
//! ```
//!
//! Two suspect operations:
//! 1. `GA::as_slice(&self) -> &[T]` — calls `Deref::deref` which is
//!    `unsafe { slice::from_raw_parts(self as *const Self as *const T, N) }`.
//! 2. `slice::last(&self) -> Option<&T>` — which is roughly
//!    `if self.is_empty() { None } else { Some(&self[len-1]) }`.
//!
//! Slot 98/99 confirmed GA basic IndexMut + write-side copy_from_slice
//! work. Slot 96/101 say `.as_slice().last()` (read-side over the
//! whole slice including its length) is what miscompiles.
//!
//! ## What this test locks down
//!
//! Two functions exercise the slice path differently:
//!
//! 1. `last_byte_via_as_slice(p: &Probe) -> u8` — `.as_slice().last()`,
//!    matches dalek/k256's exact shape.
//! 2. `last_byte_via_deref(p: &Probe) -> u8` — `(*p)[N-1]` direct
//!    Deref-Index, bypasses `.last()`/`as_slice()`. Probes whether
//!    the bug is in `as_slice()` itself or in `.last()`'s Option handling.
//!
//! ## Build
//!
//!     cargo oxide build generic_array_as_slice_last

use core::ops::{Deref, DerefMut};

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirror of `GenericArray<u8, U32>` — same Deref shape (raw-ptr
/// cast + from_raw_parts), same Default behavior.
pub struct Probe {
    data: [u8; 32],
}

impl Default for Probe {
    fn default() -> Self {
        Probe { data: [0u8; 32] }
    }
}

impl Deref for Probe {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        unsafe { core::slice::from_raw_parts(self as *const Self as *const u8, 32) }
    }
}

impl DerefMut for Probe {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { core::slice::from_raw_parts_mut(self as *mut Self as *mut u8, 32) }
    }
}

impl Probe {
    /// Mirrors `GA::as_slice(&self) -> &[T]` — alias for `Deref::deref`.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self
    }
}

/// Reproduces `Tag::compress_y`'s exact shape: take `&Probe`, call
/// `as_slice()`, call `.last()` on the slice, return the byte.
/// `#[inline(never)]` preserves the function boundary so the codegen
/// matches the cross-crate `sec1::EncodedPoint::from_affine_coordinates`
/// → `Tag::compress_y` call.
#[inline(never)]
fn last_byte_via_as_slice(p: &Probe) -> u8 {
    let s: &[u8] = p.as_slice();
    *s.last().expect("non-empty")
}

/// Bypass `.last()`'s Option machinery — use direct slice indexing
/// via Deref to test whether the bug is in `as_slice()` or in `.last()`.
#[inline(never)]
fn last_byte_via_deref(p: &Probe) -> u8 {
    let s: &[u8] = &**&p; // explicit Deref to &[u8]
    s[s.len() - 1]
}

/// Even more direct — Deref then ConstantIndex from_end. This is
/// what rustc lowers `*self.last().unwrap_or(&0)` to when len is
/// known statically.
#[inline(never)]
fn last_byte_via_const_index_from_end(p: &Probe) -> u8 {
    let s: &[u8] = p.as_slice();
    s[31]
}

#[inline(never)]
fn check_generic_array_as_slice_last() -> u32 {
    // Probe filled with a known pattern so we can verify the
    // returned byte matches what we expect.
    let mut p = Probe::default();
    for i in 0..32 {
        p[i] = core::hint::black_box((i as u8).wrapping_mul(17).wrapping_add(5));
    }
    let expected_last = (31u8).wrapping_mul(17).wrapping_add(5);

    // Shape 1 — `.as_slice().last()` (the dalek/k256 shape).
    {
        let got = last_byte_via_as_slice(&p);
        if got != expected_last {
            return 0;
        }
    }

    // Shape 2 — Deref-then-runtime-index via `len() - 1`.
    {
        let got = last_byte_via_deref(&p);
        if got != expected_last {
            return 0;
        }
    }

    // Shape 3 — Deref-then-const-index.
    {
        let got = last_byte_via_const_index_from_end(&p);
        if got != expected_last {
            return 0;
        }
    }

    1
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            *slot = check_generic_array_as_slice_last();
        }
    }
}

fn main() {
    println!("=== generic_array_as_slice_last ===");

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
    println!("SUCCESS: GA `&p.as_slice().last()` returns the right byte");
}
