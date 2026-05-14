//! Bug E follow-on: Index/IndexMut trait dispatch through a
//! borrowed `&Newtype([T; N])` parameter.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeds. On real hardware (vanity-miner-rs
//! slot 71), `Scalar::from_bytes_mod_order(_).to_bytes()` round-trip
//! still failed *after* the slot-91 Bug E fix landed (cuda-oxide
//! 87548c1). The slot-91 minimal only exercised
//! `IdxProbe(_).idx_via_trait` — Index dispatch on an owned local.
//!
//! Dalek's hot path is one Deref deeper:
//!
//! ```ignore
//! pub fn sub(a: &Scalar52, b: &Scalar52) -> Scalar52 {
//!     let mut difference = Scalar52::ZERO;
//!     for i in 0..5 {
//!         difference[i] = a[i].wrapping_sub(b[i] + ...);   // ← `a[i]` on &Scalar52
//!     }
//!     ...
//! }
//! ```
//!
//! Rustc lowers `a[i]` (where `a: &Scalar52`) via the inlined
//! `Index::index` body to MIR `&(*a).0[_i]` — projection chain
//! `[Deref, Field(0), Index(_i)]`. rvalue.rs Case 2 handles
//! `Deref+Field+` but bailed on the trailing `Index`, falling
//! through to Case 5 (MirRefOp + fresh alloca per access). Same
//! Bug E symptom: writes/reads through a fresh stack copy, original
//! data unchanged.
//!
//! ## What this test locks down
//!
//! Three shapes that exercise the projection chain:
//!
//! 1. `sum_via_trait(a: &Wrap) -> u64` — sums `a[i]` for i in 0..5
//!    via the Index trait. Confirms `Deref→Field→Index(local)` read.
//! 2. `copy_via_trait(dst: &mut Wrap, src: &Wrap)` — `dst[i] = src[i]`.
//!    Confirms `Deref→Field→Index(local)` works for both Read and
//!    Write through borrowed references.
//! 3. `sub_like(a: &Wrap, b: &Wrap)` — the dalek `Scalar52::sub`
//!    shape verbatim (minus the conditional-add): `let mut out;
//!    for i in 0..5 { out[i] = a[i].wrapping_sub(b[i]); }`.
//!
//! ## Build
//!
//!     cargo oxide build deref_index_trait_dispatch

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[derive(Copy, Clone)]
pub struct Wrap(pub [u64; 5]);

impl core::ops::Index<usize> for Wrap {
    type Output = u64;
    #[inline]
    fn index(&self, i: usize) -> &u64 {
        &(self.0[i])
    }
}

impl core::ops::IndexMut<usize> for Wrap {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut u64 {
        &mut (self.0[i])
    }
}

/// Sum elements via the Index trait on a borrowed `&Wrap`.
/// `#[inline(never)]` preserves the function-call boundary so
/// rustc emits the `&(*self).0[i]` projection chain at the call
/// site, not folded into the caller.
#[inline(never)]
fn sum_via_trait(a: &Wrap) -> u64 {
    let mut s: u64 = 0;
    for i in 0..5 {
        s = s.wrapping_add(a[i]);
    }
    s
}

/// Element-wise copy via the Index/IndexMut trait on borrowed
/// references — same shape as dalek's `for i in 0..5 { dst[i] =
/// src[i]; }` loops.
#[inline(never)]
fn copy_via_trait(dst: &mut Wrap, src: &Wrap) {
    for i in 0..5 {
        dst[i] = src[i];
    }
}

/// Faithful copy of dalek's `Scalar52::sub` shape (5-limb borrow
/// chain, no underflow-add). The Index trait dispatches on borrowed
/// `&Wrap` for both reads (`a[i]`, `b[i]`) and writes (`difference[i]`).
#[inline(never)]
fn sub_like(a: &Wrap, b: &Wrap) -> Wrap {
    let mut difference = Wrap([0u64; 5]);
    let mask = (1u64 << 52) - 1;
    let mut borrow: u64 = 0;
    for i in 0..5 {
        borrow = a[i].wrapping_sub(b[i] + (borrow >> 63));
        difference[i] = borrow & mask;
    }
    difference
}

#[inline(never)]
fn check_deref_index_trait_dispatch() -> u32 {
    // Shape 1 — sum_via_trait. a = [10, 20, 30, 40, 50] → sum = 150.
    {
        let a = Wrap([
            core::hint::black_box(10u64),
            core::hint::black_box(20u64),
            core::hint::black_box(30u64),
            core::hint::black_box(40u64),
            core::hint::black_box(50u64),
        ]);
        let s = sum_via_trait(&a);
        if s != 150 {
            return 0;
        }
    }

    // Shape 2 — copy_via_trait. Each dst[i] must equal the
    // matching src[i].
    {
        let src = Wrap([
            core::hint::black_box(0xAAu64),
            core::hint::black_box(0xBBu64),
            core::hint::black_box(0xCCu64),
            core::hint::black_box(0xDDu64),
            core::hint::black_box(0xEEu64),
        ]);
        let mut dst = Wrap([0u64; 5]);
        copy_via_trait(&mut dst, &src);
        for i in 0..5 {
            if dst.0[i] != src.0[i] {
                return 0;
            }
        }
    }

    // Shape 3 — sub_like on equal inputs → all-zero difference.
    {
        let r = Wrap([
            core::hint::black_box(0x000f_48bd_6721_e6edu64),
            core::hint::black_box(0x0003_bab5_ac67_e45au64),
            core::hint::black_box(0x000f_fffe_b35e_51b0u64),
            core::hint::black_box(0x000f_ffff_ffff_ffffu64),
            core::hint::black_box(0x0000_0fff_ffff_ffffu64),
        ]);
        let d = sub_like(&r, &r);
        for i in 0..5 {
            if d.0[i] != 0 {
                return 0;
            }
        }
    }

    // Shape 3b — sub_like(R, 1) → R - 1 limb 0, R limbs 1..5.
    {
        let r = Wrap([
            core::hint::black_box(0x000f_48bd_6721_e6edu64),
            core::hint::black_box(0x0003_bab5_ac67_e45au64),
            core::hint::black_box(0x000f_fffe_b35e_51b0u64),
            core::hint::black_box(0x000f_ffff_ffff_ffffu64),
            core::hint::black_box(0x0000_0fff_ffff_ffffu64),
        ]);
        let one = Wrap([
            core::hint::black_box(1u64),
            core::hint::black_box(0u64),
            core::hint::black_box(0u64),
            core::hint::black_box(0u64),
            core::hint::black_box(0u64),
        ]);
        let d = sub_like(&r, &one);
        let mask = (1u64 << 52) - 1;
        let expected0 = (0x000f_48bd_6721_e6edu64.wrapping_sub(1)) & mask;
        if d.0[0] != expected0
            || d.0[1] != r.0[1]
            || d.0[2] != r.0[2]
            || d.0[3] != r.0[3]
            || d.0[4] != r.0[4]
        {
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
            *slot = check_deref_index_trait_dispatch();
        }
    }
}

fn main() {
    println!("=== deref_index_trait_dispatch ===");

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
    println!("SUCCESS: Index/IndexMut trait dispatch through &Newtype works");
}
