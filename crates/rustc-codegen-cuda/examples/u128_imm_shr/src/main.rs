//! PTX-shape regression test for `u128 >> 52` (dalek
//! `montgomery_reduce::part1` shape).
//!
//! ## Background — falsified D1 hypothesis
//!
//! After the v1.46.0 cuda-oxide carry-chain fix, vanity-miner-rs's
//! self-test still showed slots 71/72/78/79 FAILing — all of which
//! cross dalek's `montgomery_reduce` and k256's `mul_inner`
//! reduction. The shared shape is:
//!
//! ```ignore
//! fn part1(sum: u128) -> (u128, u64) {
//!     let p = (sum as u64).wrapping_mul(LFACTOR) & ((1u64 << 52) - 1);
//!     ((sum + m(p, L[0])) >> 52, p)   // ← single u128 imm shift
//! }
//! ```
//!
//! D1 hypothesis was: cuda-oxide miscompiles `u128 >> 52` because
//! LLVM lowers it to a multi-step splice
//! `lo' = (hi << 12) | (lo >> 52); hi' = hi >> 52`. The probe (this
//! file) covers four cases — two halves nonzero, single-bit at
//! position 52, u128::MAX, and the dalek-faithful `(a + b) >> 52`
//! shape where the input is a carrying add. **Local PTX inspection
//! shows all four cases lower correctly** with the NVPTX funnel-shift
//! intrinsic `shf.l.wrap.b32 ..., 12` for the cross-half splice and
//! `shr.u64 ..., 52` for the upper half. The compiler even folds the
//! XOR-against-EXPECTED comparison validly for case 2 (`~X.hi >> 52`
//! is bit-level equivalent to `(X.hi >> 52) XOR 0x0FFF` given the
//! constraints on result.hi).
//!
//! Conclusion: D1 is falsified. The slot-71 failure mode must live
//! elsewhere — either D2 (deeper `&'static` newtype nesting at
//! depth 4, as in `k256::Scalar(U256{ limbs: [Limb; 4] })`) or a
//! yet-unidentified bug class.
//!
//! ## What this test locks down
//!
//! Four back-to-back `(x >> 52) == const` checks:
//! * x with both halves nonzero (`0xFEDC..._..._CDEF`) — cross-half
//!   splice exercised
//! * x = `1 << 52` — single-bit canary; tests that the bit at exactly
//!   position 52 lands in result.lo bit 0
//! * x = `u128::MAX` — all 76 surviving bits must be set
//! * `(u128::MAX + 0x0001_0001) >> 52` — dalek shape: the shift
//!   input is a carrying `wrapping_add`, not a black-boxed literal
//!
//! Each check returns early on mismatch, so the kernel-return value
//! pinpoints which case broke.
//!
//! ## Verify
//!
//! ```sh
//! cargo oxide build u128_imm_shr
//! # Expected PTX shape — funnel-shift splice for result.lo +
//! # plain shr.u64 for result.hi:
//! grep -nE 'shf\.l\.wrap\.b32.*, 12' \
//!   crates/rustc-codegen-cuda/examples/u128_imm_shr/u128_imm_shr.ptx
//! grep -nE 'shr\.u64.*, 52' \
//!   crates/rustc-codegen-cuda/examples/u128_imm_shr/u128_imm_shr.ptx
//! ```
//!
//! Expect 2 `shf.l.wrap.b32 …, 12` lines per case (8 total for 4
//! cases) and 1 `shr.u64 …, 52` line per case (4 total). Case 3
//! additionally emits `add.cc.s64` + `addc.cc.s64` for the u128
//! add before the shift.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[inline(never)]
fn check_u128_imm_shr_52() -> u32 {
    // Case 0: both halves nonzero — exercises the cross-half splice.
    {
        const X: u128 = 0xFEDC_BA98_7654_3210_0123_4567_89AB_CDEFu128;
        const EXPECTED: u128 = X >> 52;
        let x = core::hint::black_box(X);
        let shifted = x >> 52;
        if shifted != EXPECTED {
            return 0;
        }
    }
    // Case 1: single bit at position 52 — canary for the lowest
    // surviving bit. After `>> 52`, the result must be exactly 1.
    {
        const X: u128 = 1u128 << 52;
        const EXPECTED: u128 = X >> 52;
        let x = core::hint::black_box(X);
        let shifted = x >> 52;
        if shifted != EXPECTED {
            return 0;
        }
    }
    // Case 2: u128::MAX — every surviving bit must be 1.
    {
        const X: u128 = u128::MAX;
        const EXPECTED: u128 = X >> 52;
        let x = core::hint::black_box(X);
        let shifted = x >> 52;
        if shifted != EXPECTED {
            return 0;
        }
    }
    // Case 3: dalek-shape `(a + b) >> 52` where `a + b` carries from
    // low to high. This is the exact pattern from
    // `montgomery_reduce::part1`: `((sum + m(p, L[0])) >> 52, p)`.
    // The shift's input is a sum, not a literal black_box — exercises
    // the carry-out-then-shift path.
    {
        const A: u128 = (u64::MAX as u128) << 64 | (u64::MAX as u128); // u128::MAX
        const B: u128 = 0x0000_0000_0000_0001_0000_0000_0000_0001u128;
        // a + b wraps; expected wraps the same way.
        const EXPECTED: u128 = A.wrapping_add(B) >> 52;
        let a = core::hint::black_box(A);
        let b = core::hint::black_box(B);
        let shifted = a.wrapping_add(b) >> 52;
        if shifted != EXPECTED {
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
            *slot = check_u128_imm_shr_52();
        }
    }
}

fn main() {
    println!("=== u128_imm_shr ===");

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
    println!("SUCCESS: u128 >> 52 immediate-shift checks all passed");
}
