//! PTX-shape regression test for u128 switch-target truncation.
//!
//! ## Pre-fix wall
//!
//! In vanity-miner-rs's v1.44.0 PTX dump, function
//! `kernel_self_test_arith_i128_chain_add` for case 0 emitted:
//!
//! ```text
//!     add.cc.s64 %rd11, %a_lo, %b_lo ;
//!     addc.cc.s64 %rd12, %a_hi, %b_hi ;
//!     or.b64 %rd13, %rd11, %rd12 ;          ; sum.lo | sum.hi
//!     setp.eq.b64 %p1, %rd13, 0 ;           ; checks sum == 0 !!
//! ```
//!
//! The correct comparison for `sum != 2^64` requires an extra
//! `xor.b64 %sum.hi, 1` step before the OR, because the high half
//! of E is 1. That XOR was missing — slot 65 case 0 was effectively
//! checking `sum == 0` and always reporting FAIL. Same shape in
//! case 2: the `xor.b64 ..., 3` for the high half was missing.
//!
//! ## Root cause
//!
//! [crates/mir-importer/src/translator/terminator/mod.rs::translate_switch]
//! built each `SwitchInt` target constant with
//! `APInt::from_u64(val as u64, width_nz)`. `SwitchTargets.branches()`
//! yields `(u128, BasicBlockIdx)`, but the `val as u64` cast dropped
//! the high 64 bits. For a 128-bit discriminant comparing against
//! `2^64`, the target collapsed to `0`. rustc-emitted MIR for
//! `if a + b != E { return 0; }` lowers to a `SwitchInt` over the
//! i128 sum with the target value being `E` itself — so any E with
//! non-zero high half got silently truncated.
//!
//! Fix: replace `APInt::from_u64(val as u64, width_nz)` with
//! `APInt::from_u128(val, width_nz)` at both branch sites in
//! `translate_switch` (single-branch i1-cmp path + multi-branch
//! chain path).
//!
//! ## What this test locks down
//!
//! Three back-to-back `if sum != const_u128 { return 0; }` checks
//! covering:
//! * E with high half = 1 (`(u64::MAX as u128) + 1 = 2^64`)
//! * E with high half = 0 (`u128::MAX + 1 = 0` — sanity case)
//! * E with high half = 3 (`4 * 2^64 - 2`)
//!
//! The emitted PTX must include `xor.b64 %sum.hi, 1` for case 0 and
//! `xor.b64 %sum.hi, 3` for case 2. If a future regression of
//! `translate_switch` (or an algebraically-equivalent shape) drops
//! either XOR, the shape grep below catches it.
//!
//! ## Verify
//!
//! ```sh
//! cargo oxide build u128_ne_early_return
//! grep -nE 'xor\.b64.*, (1|3)\b' \
//!   crates/rustc-codegen-cuda/examples/u128_ne_early_return/u128_ne_early_return.ptx
//! ```
//!
//! Expect both `xor.b64 …, 1` and `xor.b64 …, 3` lines.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[inline(never)]
fn check_i128_chain_add() -> u32 {
    {
        let a = core::hint::black_box(u64::MAX as u128);
        let b = core::hint::black_box(1u128);
        const E: u128 = (u64::MAX as u128).wrapping_add(1);
        if a.wrapping_add(b) != E {
            return 0;
        }
    }
    {
        let a = core::hint::black_box(u128::MAX);
        let b = core::hint::black_box(1u128);
        const E: u128 = u128::MAX.wrapping_add(1);
        if a.wrapping_add(b) != E {
            return 0;
        }
    }
    {
        let a = core::hint::black_box(u64::MAX as u128);
        let b = core::hint::black_box(u64::MAX as u128);
        let c = core::hint::black_box(u64::MAX as u128);
        let d = core::hint::black_box((1u128 << 64) | 1u128);
        let s = a.wrapping_add(b).wrapping_add(c).wrapping_add(d);
        const E: u128 = (u64::MAX as u128)
            .wrapping_add(u64::MAX as u128)
            .wrapping_add(u64::MAX as u128)
            .wrapping_add((1u128 << 64) | 1u128);
        if s != E {
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
            *slot = check_i128_chain_add();
        }
    }
}

fn main() {
    println!("=== u128_ne_early_return ===");

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
    println!("SUCCESS: u128 != const_u128 + early-return chain is correct");
}
