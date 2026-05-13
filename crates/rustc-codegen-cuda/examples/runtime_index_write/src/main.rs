//! Candidate repro for the dynamic-limb-growth pattern in
//! `base58_encode_32`. Slot 43 (all-zero input) PASSes because the
//! growth loop never runs (`remaining_carry` is 0 every iteration);
//! slot 3 (real input) FAILs and the only structural difference is
//! that the growth loop *does* run with non-zero values.
//!
//! ## What pattern this exercises
//!
//! ```text
//! while remaining_carry > 0 && limb_count < N {
//!     limbs[limb_count] = (remaining_carry % D) as u32;   // ← runtime-index WRITE
//!     remaining_carry = remaining_carry / D;
//!     limb_count += 1;
//! }
//! ```
//!
//! Two suspect operations:
//! * `limbs[limb_count] = …` — store to a stack array at a runtime
//!   index. PTX: `add.s64 + st.local.b32` against an SP-relative base.
//! * `limb_count += 1` after each iteration — the index variable that
//!   feeds the next iteration's store.
//!
//! Earlier candidates (`divrem_large_const`, `base58_limb_divrem`,
//! `i128_add_carry_chain`) cover the *arithmetic* shapes. None of
//! them exercise this *control-flow + dynamic-write* shape — slot 60
//! (`iter static table lookup`) does runtime-index *reads*, but
//! reading and writing go down different codegen paths in
//! `mir-importer/src/translator/statement.rs` (write side) vs
//! `rvalue.rs` (read side).
//!
//! ## What this kernel does
//!
//! Mimics the base58 growth loop in isolation: starts with a u64
//! `remaining_carry` from the kernel param buffer, divrem by a
//! compile-time-constant divisor, walks the limb count up from 0 in
//! the same shape. Then reads all 10 slots back and writes them to
//! the output buffer. Host computes the expected sequence on CPU
//! and asserts every slot matches.
//!
//! ## Build / verify
//!
//!     cargo oxide build runtime_index_write
//!     cargo oxide run   runtime_index_write   # needs GPU

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const DIVISOR: u64 = 58_u64.pow(5); // 656_356_768

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `args[0]` is the initial `remaining_carry`.
    /// Threads 0..10 each read back `limbs[i]` after the growth loop.
    /// Thread 10 reads back `limb_count`.
    #[kernel]
    pub fn run(args: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < 11
            && !args.is_empty()
        {
            let mut limbs = [0u32; 10];
            let mut limb_count: usize = 0;
            let mut remaining_carry = args[0];

            // The exact base58 growth-loop shape. Runtime-index write
            // into `limbs[limb_count]`, then `limb_count += 1` for the
            // next iteration.
            while remaining_carry > 0 && limb_count < 10 {
                limbs[limb_count] = (remaining_carry % DIVISOR) as u32;
                remaining_carry /= DIVISOR;
                limb_count += 1;
            }

            *slot = if i < 10 {
                limbs[i as usize] as u64
            } else {
                limb_count as u64
            };
        }
    }
}

fn main() {
    println!("=== runtime_index_write ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    // A value large enough to grow the limbs array through 3 iterations.
    let carry_in: u64 = 0xDEAD_BEEF_CAFE_BABE;

    // Compute the expected limb sequence on CPU.
    let mut expected = [0u32; 10];
    let mut expected_count = 0usize;
    {
        let mut c = carry_in;
        while c > 0 && expected_count < 10 {
            expected[expected_count] = (c % DIVISOR) as u32;
            c /= DIVISOR;
            expected_count += 1;
        }
    }

    let args = DeviceBuffer::from_host(&stream, &[carry_in]).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 11).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(11), &args, &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    println!("limb_count host = {}, GPU = {}", expected_count, result[10]);
    for k in 0..10 {
        println!(
            "  limbs[{}] host = {:>10}, GPU = {:>10}{}",
            k,
            expected[k],
            result[k],
            if result[k] != expected[k] as u64 { "  <-- MISMATCH" } else { "" }
        );
    }
    assert_eq!(
        result[10],
        expected_count as u64,
        "limb_count mismatch — growth loop iterated wrong number of times"
    );
    for k in 0..10 {
        assert_eq!(
            result[k],
            expected[k] as u64,
            "limbs[{}] mismatch — runtime-index write didn't land in the right slot",
            k
        );
    }
    println!("SUCCESS: dynamic-growth limb writes land in the right slots");
}
