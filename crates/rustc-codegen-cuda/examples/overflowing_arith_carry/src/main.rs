//! Regression test for the dropped carry flag in `u64::overflowing_add`
//! / `u64::overflowing_sub` (and signed counterparts).
//!
//! ## Pre-fix wall
//!
//! [crates/mir-lower/src/convert/ops/arithmetic.rs:244-293] lowers
//! `mir.checked_add` / `mir.checked_sub` / `mir.checked_mul` to a
//! regular add/sub/mul packed with a hardcoded `false` overflow flag.
//! The comment in-source: "GPU kernels don't perform overflow checking
//! for performance." That's load-bearing wrong — Rust's
//! `overflowing_*` returns `(value, carry_out)` and downstream
//! consumers depend on the *real* carry to thread multi-limb add
//! chains (dalek's 5×u52, k256's 4×u64, base58's byte-by-byte limb
//! loop). With the flag stuck at false, every carry-chain silently
//! drops the carry bit between limbs.
//!
//! Surfaced from vanity-miner-rs/logic/src/self_test.rs slots 46, 47,
//! 48 (direct), and is the prime suspect for slots 2, 3, 4, 5 and
//! their downstream (ed25519 / secp256k1 / base58_encode_pub all use
//! multi-limb math).
//!
//! ## What this kernel does
//!
//! Operands come from a kernel-parameter buffer (host-written), so
//! const folding can't kick in. Two operand pairs:
//!
//! * `(u64::MAX, 1)` — `overflowing_add` must return `(0, true)`.
//! * `(0, 1)` — `overflowing_sub` must return `(u64::MAX, true)`.
//!
//! The kernel writes (sum, carry, diff, borrow) into a 4-slot output
//! buffer. The host asserts all four. Pre-fix the two `carry` /
//! `borrow` slots will be 0 instead of 1.
//!
//! ## PTX-text smoke test
//!
//! Even without a GPU, the pre-fix PTX should show `st.b32 [out+X], 0;`
//! (or `mov.b32 %rN, 0; st.b32 ...`) for the carry/borrow stores —
//! i.e. a constant zero, not a `setp.lt.u64` derived flag. After the
//! fix, the same stores must come from a real comparison.
//!
//! ## Build / run
//!
//!     cargo oxide build overflowing_arith_carry
//!     cargo oxide run   overflowing_arith_carry   # needs a GPU; fails pre-fix

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `args` holds `[add_lhs, add_rhs, sub_lhs, sub_rhs]`.
    /// `out` is laid out as `[sum, carry_as_u64, diff, borrow_as_u64]`.
    /// A 4-thread launch lets each thread own one output slot.
    #[kernel]
    pub fn run(args: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < 4
            && args.len() >= 4
        {
            let add_lhs = args[0];
            let add_rhs = args[1];
            let sub_lhs = args[2];
            let sub_rhs = args[3];

            let (sum, add_carry) = add_lhs.overflowing_add(add_rhs);
            let (diff, sub_borrow) = sub_lhs.overflowing_sub(sub_rhs);

            *slot = match i {
                0 => sum,
                1 => add_carry as u64,
                2 => diff,
                3 => sub_borrow as u64,
                _ => 0,
            };
        }
    }
}

fn main() {
    println!("=== overflowing_arith_carry ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    // u64::MAX + 1 = (0, true); 0 - 1 = (u64::MAX, true).
    let args_host: [u64; 4] = [u64::MAX, 1, 0, 1];
    let args = DeviceBuffer::from_host(&stream, &args_host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 4).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(4), &args, &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], 0, "sum: MAX + 1 should wrap to 0");
    assert_eq!(
        result[1], 1,
        "carry: MAX + 1 overflowed, carry should be 1 (pre-fix this is 0)"
    );
    assert_eq!(result[2], u64::MAX, "diff: 0 - 1 should wrap to u64::MAX");
    assert_eq!(
        result[3], 1,
        "borrow: 0 - 1 underflowed, borrow should be 1 (pre-fix this is 0)"
    );

    println!("SUCCESS: overflowing_add / overflowing_sub return the real carry");
}
