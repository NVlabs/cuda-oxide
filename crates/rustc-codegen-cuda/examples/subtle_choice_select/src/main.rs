//! Bug F probe: `subtle::ConditionallySelectable` trait dispatch.
//!
//! ## Background
//!
//! Vanity-miner-rs slot 93 (`AffinePoint::GENERATOR.to_encoded_point`)
//! FAILs even after the Bug E (Index trait dispatch) fix landed.
//! `AffinePoint::to_encoded_point` ends in:
//!
//! ```ignore
//! EncodedPoint::conditional_select(
//!     &EncodedPoint::from_affine_coordinates(&x, &y, compress),
//!     &EncodedPoint::identity(),
//!     self.is_identity(),    // Choice(0) for generator
//! )
//! ```
//!
//! For the generator, `is_identity()` returns `Choice(0)`, so the
//! select should pick the first arm. Slot 93's symptom — wrong
//! bytes returned — implies the WRONG arm was picked. Hand-rolled
//! `(cond as u64).wrapping_neg()` mask blend PASSes (slots 53/54),
//! so the math itself works; suspect is the trait dispatch through
//! `subtle::Choice(u8)` + `ConditionallySelectable`.
//!
//! ## What this test locks down
//!
//! Three escalating shapes:
//!
//! 1. `u64::conditional_select` with both Choice arms. Tests the
//!    integer primitive — `let mask = -(choice.unwrap_u8() as i64) as u64;`
//!    + `a ^ (mask & (a ^ b))`.
//! 2. `Choice::from(0u8)` / `Choice::from(1u8)` round-trip through
//!    runtime `black_box`'d u8. Tests the newtype field access in
//!    `unwrap_u8()`.
//! 3. `<[u8; 32] as ConditionallySelectable>::conditional_select`
//!    via the generic blanket `for [T; N]` impl that subtle ships:
//!    `output = *a; output.conditional_assign(b, choice)` where the
//!    assign iterates `self.iter_mut().zip(other)`. This is the
//!    EXACT shape `EncodedPoint::conditional_select` runs.
//!
//! ## Build
//!
//!     cargo oxide build subtle_choice_select

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[derive(Copy, Clone, Debug)]
pub struct Choice(u8);

impl Choice {
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        Choice(value)
    }

    #[inline]
    pub fn unwrap_u8(&self) -> u8 {
        self.0
    }
}

pub trait ConditionallySelectable: Copy {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self;
    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        *self = Self::conditional_select(self, other, choice);
    }
}

impl ConditionallySelectable for u8 {
    #[inline]
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mask = -(choice.unwrap_u8() as i8) as u8;
        a ^ (mask & (a ^ b))
    }

    #[inline]
    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        let mask = -(choice.unwrap_u8() as i8) as u8;
        *self ^= mask & (*self ^ *other);
    }
}

impl ConditionallySelectable for u64 {
    #[inline]
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mask = -(choice.unwrap_u8() as i64) as u64;
        a ^ (mask & (a ^ b))
    }

    #[inline]
    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        let mask = -(choice.unwrap_u8() as i64) as u64;
        *self ^= mask & (*self ^ *other);
    }
}

impl<T, const N: usize> ConditionallySelectable for [T; N]
where
    T: ConditionallySelectable,
{
    #[inline]
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut output = *a;
        output.conditional_assign(b, choice);
        output
    }

    fn conditional_assign(&mut self, other: &Self, choice: Choice) {
        for (a_i, b_i) in self.iter_mut().zip(other) {
            a_i.conditional_assign(b_i, choice)
        }
    }
}

#[inline(never)]
fn check_subtle_choice_select() -> u32 {
    // Shape 1 — u64 primitive, choice=0 → pick a.
    {
        let a = core::hint::black_box(0xAAAA_AAAA_AAAA_AAAAu64);
        let b = core::hint::black_box(0xBBBB_BBBB_BBBB_BBBBu64);
        let c = Choice::from_u8(core::hint::black_box(0u8));
        let r = u64::conditional_select(&a, &b, c);
        if r != a {
            return 0;
        }
    }
    // Shape 1 — u64 primitive, choice=1 → pick b.
    {
        let a = core::hint::black_box(0xAAAA_AAAA_AAAA_AAAAu64);
        let b = core::hint::black_box(0xBBBB_BBBB_BBBB_BBBBu64);
        let c = Choice::from_u8(core::hint::black_box(1u8));
        let r = u64::conditional_select(&a, &b, c);
        if r != b {
            return 0;
        }
    }

    // Shape 2 — Choice round-trip via newtype `from_u8`/`unwrap_u8`.
    {
        let raw = core::hint::black_box(0u8);
        let c = Choice::from_u8(raw);
        if c.unwrap_u8() != raw {
            return 0;
        }
        let raw = core::hint::black_box(1u8);
        let c = Choice::from_u8(raw);
        if c.unwrap_u8() != raw {
            return 0;
        }
    }

    // Shape 3 — `[u8; 32]::conditional_select` (the EncodedPoint shape).
    // a = (0x11,0x12,...,0x30); b = (0xA1,...,0xC0). Choice=0 → pick a.
    {
        let mut a_arr = [0u8; 32];
        let mut b_arr = [0u8; 32];
        for i in 0..32 {
            a_arr[i] = 0x10 + (i as u8 + 1);
            b_arr[i] = 0xA0 + (i as u8 + 1);
        }
        let a = core::hint::black_box(a_arr);
        let b = core::hint::black_box(b_arr);
        let c = Choice::from_u8(core::hint::black_box(0u8));
        let r = <[u8; 32]>::conditional_select(&a, &b, c);
        for i in 0..32 {
            if r[i] != a[i] {
                return 0;
            }
        }
    }
    // Shape 3 — Choice=1 → pick b.
    {
        let mut a_arr = [0u8; 32];
        let mut b_arr = [0u8; 32];
        for i in 0..32 {
            a_arr[i] = 0x10 + (i as u8 + 1);
            b_arr[i] = 0xA0 + (i as u8 + 1);
        }
        let a = core::hint::black_box(a_arr);
        let b = core::hint::black_box(b_arr);
        let c = Choice::from_u8(core::hint::black_box(1u8));
        let r = <[u8; 32]>::conditional_select(&a, &b, c);
        for i in 0..32 {
            if r[i] != b[i] {
                return 0;
            }
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
            *slot = check_subtle_choice_select();
        }
    }
}

fn main() {
    println!("=== subtle_choice_select ===");

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
    println!("SUCCESS: subtle Choice / ConditionallySelectable round-trips");
}
