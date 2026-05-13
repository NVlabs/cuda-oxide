//! Candidate repro for u128 + u128 cross-half carry (Bug C hypothesis).
//!
//! Slot 40 (`u128 wrapping_mul`) and slot 49 (`widening mul pair`)
//! both PASS in vanity-miner-rs's self-test — they only test a single
//! u128 mul. dalek's `FieldElement51::mul_internal` and k256's
//! `FieldElement5x52::mul_inner` accumulate ~25 partial-product
//! u128 values via plain `u128 + u128` between widening multiplies.
//! Native LLVM `add i128` lowers on NVPTX to `add.cc.u64` for the
//! low limb + `addc.u64` for the high. If that carry-chain plumbing
//! is broken (and the `overflowing_add` path was an unrelated bug
//! already fixed), every dalek/k256 accumulation drops bits.
//!
//! ## What this kernel does
//!
//! Builds two u128 values from kernel-param u64 halves where the low
//! halves are chosen to wrap (forcing a carry into the high), then
//! checks both halves of the sum against host const-eval.
//!
//! Three test cases per thread group:
//! * `(MAX, 0) + (1, 0)` → `(0, 1)` — pure low-to-high carry
//! * `(MAX, MAX) + (1, 0)` → `(0, 0)` — carry rolls all the way over
//! * `(MAX/2, 1) + (MAX/2 + 1, 2)` → `(0, 4)` — combined low+high adds
//!
//! ## Pre-fix indicators (if Bug C is real)
//!
//! In the emitted PTX look for `add.cc.u64` / `addc.u64` paired with
//! the same register for both halves. If the `addc.u64` reads the
//! wrong carry-input register (or no carry at all — just `add.u64`),
//! that's the bug. The host-side assert fails on the high half.
//!
//! ## Build / run
//!
//!     cargo oxide build i128_add_carry_chain
//!     cargo oxide run   i128_add_carry_chain   # needs GPU

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `args` layout per test case: `[a_lo, a_hi, b_lo, b_hi]`.
    /// Test cases consumed at offsets `i * 4` for i = 0..N.
    /// `out` receives `[sum_lo, sum_hi]` per test case at offsets `i * 2`.
    #[kernel]
    pub fn run(args: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i / 2 < args.len() / 4
        {
            let case = i / 2;
            let base = case * 4;
            let a = ((args[base + 1] as u128) << 64) | (args[base] as u128);
            let b = ((args[base + 3] as u128) << 64) | (args[base + 2] as u128);
            let s = a.wrapping_add(b);
            *slot = if i % 2 == 0 {
                s as u64
            } else {
                (s >> 64) as u64
            };
        }
    }
}

fn main() {
    println!("=== i128_add_carry_chain ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    // Three test cases, each `(a_lo, a_hi, b_lo, b_hi)`.
    let args_host: [u64; 12] = [
        // Case 0: (MAX, 0) + (1, 0) = (0, 1). Pure low→high carry.
        u64::MAX, 0, 1, 0,
        // Case 1: (MAX, MAX) + (1, 0) = (0, 0). Carry rolls fully over.
        u64::MAX, u64::MAX, 1, 0,
        // Case 2: (MAX/2, 1) + (MAX/2+1, 2) = (0, 4). Combined low+high.
        u64::MAX / 2, 1, u64::MAX / 2 + 1, 2,
    ];
    let args = DeviceBuffer::from_host(&stream, &args_host).unwrap();
    let n_cases = args_host.len() / 4;
    let n_out = n_cases * 2;
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, n_out).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(n_out as u32), &args, &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for case in 0..n_cases {
        let base = case * 4;
        let a = ((args_host[base + 1] as u128) << 64) | (args_host[base] as u128);
        let b = ((args_host[base + 3] as u128) << 64) | (args_host[base + 2] as u128);
        let s = a.wrapping_add(b);
        let expected_lo = s as u64;
        let expected_hi = (s >> 64) as u64;
        assert_eq!(
            result[2 * case],
            expected_lo,
            "case {} low mismatch (a={:#x}, b={:#x})",
            case,
            a,
            b
        );
        assert_eq!(
            result[2 * case + 1],
            expected_hi,
            "case {} HIGH mismatch (a={:#x}, b={:#x}) — carry-out dropped?",
            case,
            a,
            b
        );
    }
    println!("SUCCESS: u128 + u128 propagates carry across 64-bit halves");
}
