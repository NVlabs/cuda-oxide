//! Candidate repro for the partial-product accumulation chain used
//! by dalek and k256 multi-limb field multiplication.
//!
//! ## Why
//!
//! Slot 40 (one `u128.wrapping_mul`) and slot 49 (one widening
//! mul-pair) both PASS in vanity-miner-rs's self-test. Slot 48
//! (3-limb carry-chain) PASSes after the recent carry-flag fix.
//! Yet slots 2, 4, 5 (ed25519 / secp256k1 derives) still FAIL,
//! which means there's still a bug in the actual field-math shape
//! these crates emit.
//!
//! The structural difference between the passing slots and the
//! failing crates: dalek's `Scalar52::mul_internal` and k256's
//! `FieldElement5x52::mul_inner` do *sequences* of widening
//! multiplies, summed with `u128 + u128 + u128`. Each individual op
//! works in isolation; the open question is whether composing them
//! in a chain breaks something — register pressure, the
//! widening-mul / add interaction, or LLVM's NVPTX lowering picking
//! a bad schedule.
//!
//! ## What this kernel does
//!
//! Computes a single partial-product accumulator the same way dalek
//! does for `z[k]` at limb index k:
//!
//! ```text
//! z2 = (a0 as u128) * (b2 as u128)
//!    + (a1 as u128) * (b1 as u128)
//!    + (a2 as u128) * (b0 as u128)
//! ```
//!
//! Both halves of the resulting u128 are written to the output and
//! asserted against host const-eval.
//!
//! ## Build / verify
//!
//!     cargo oxide build widening_mul_chain
//!     cargo oxide run   widening_mul_chain   # needs GPU

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `args` layout: `[a0, a1, a2, b0, b1, b2]`.
    /// Two threads write the (lo, hi) halves of the accumulated u128.
    #[kernel]
    pub fn run(args: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < 2
            && args.len() >= 6
        {
            let a0 = args[0] as u128;
            let a1 = args[1] as u128;
            let a2 = args[2] as u128;
            let b0 = args[3] as u128;
            let b1 = args[4] as u128;
            let b2 = args[5] as u128;

            // dalek-shape partial product for limb 2:
            //   z2 = a0*b2 + a1*b1 + a2*b0
            let z2 = (a0.wrapping_mul(b2))
                .wrapping_add(a1.wrapping_mul(b1))
                .wrapping_add(a2.wrapping_mul(b0));

            *slot = if i == 0 { z2 as u64 } else { (z2 >> 64) as u64 };
        }
    }
}

fn main() {
    println!("=== widening_mul_chain ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    // Inputs that force every partial product into the high half of
    // its u128 (each ai, bj is large enough that a*b exceeds 2^64),
    // and force the three u128 sums to carry across the 64-bit
    // boundary between additions. dalek's limbs are 52-bit, so the
    // products span ~104 bits and the sums need careful carry
    // plumbing.
    let a0: u64 = 0x000F_FFFF_FFFF_FFFF; // 52-bit max
    let a1: u64 = 0x000F_FFFF_FFFF_FFFE;
    let a2: u64 = 0x000F_FFFF_FFFF_FFFD;
    let b0: u64 = 0x000F_FFFF_FFFF_FFFC;
    let b1: u64 = 0x000F_FFFF_FFFF_FFFB;
    let b2: u64 = 0x000F_FFFF_FFFF_FFFA;

    let z2 = (a0 as u128).wrapping_mul(b2 as u128)
        .wrapping_add((a1 as u128).wrapping_mul(b1 as u128))
        .wrapping_add((a2 as u128).wrapping_mul(b0 as u128));
    let expected_lo = z2 as u64;
    let expected_hi = (z2 >> 64) as u64;

    let args = DeviceBuffer::from_host(&stream, &[a0, a1, a2, b0, b1, b2]).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 2).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(2), &args, &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    println!("z2 lo  host = {:#x}, GPU = {:#x}", expected_lo, result[0]);
    println!("z2 hi  host = {:#x}, GPU = {:#x}", expected_hi, result[1]);
    assert_eq!(
        result[0], expected_lo,
        "z2 low half mismatch — widening-mul-chain low bits wrong"
    );
    assert_eq!(
        result[1], expected_hi,
        "z2 HIGH half mismatch — partial-product accumulation carry dropped?"
    );
    println!("SUCCESS: 3-term widening-mul + u128-add chain accumulates correctly");
}
