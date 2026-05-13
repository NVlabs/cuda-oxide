//! Targeted repro for the failing `base58_encode_32` inner loop's
//! exact PTX shape.
//!
//! ## Why a third divrem repro
//!
//! `divrem_large_const` already covers `x / 656_356_768` where `x`
//! flows in from a single u64 kernel param. That repro's PTX feeds
//! the dividend register directly into `mul.hi.u64`. The base58
//! failing path is structurally different: the dividend is
//! *reconstructed* from a u32 limb loaded out of a stack array,
//! shifted into the high half, and added to a u64 carry. The
//! observed failing PTX (from vanity-miner-rs's `kernels.ptx` v1.42,
//! inside `logic__base58_encode_32`):
//!
//! ```text
//! ld.local.b32  %rd195, [%rd194]                  ; limbs[i]  →  64-bit reg (zero-extended)
//! shl.b64       %rd196, %rd195, 32                ; (limbs[i] as u64) << 32
//! add.s64       %rd197, %rd7, %rd196              ; carry + ...
//! mul.hi.u64    %rd198, %rd197, 7544311872078572213
//! shr.u64       %rd7,   %rd198, 28
//! ```
//!
//! If `divrem_large_const` ends up passing on hardware but this one
//! fails, the bug is in how `mul.hi.u64` (or the magic-multiply
//! sequence) handles a multiplicand that was just reconstructed via
//! `shl + add`, vs. one that came straight from a memory load. That
//! would be a real, narrow codegen bug worth filing.
//!
//! ## Forcing the shape
//!
//! `limbs` is a stack array of `[u32; 8]`. A runtime index is used
//! for both the write and the read so mem2reg can't promote the array
//! to an SSA value — the access has to materialise as `st.local.b32`
//! / `ld.local.b32`. The dividend is then `carry + ((limbs[i] as u64)
//! << 32)`, and divrem by the compile-time-constant 58^5 forces the
//! magic-multiply lowering.
//!
//! ## Build / verify
//!
//!     cargo oxide build base58_limb_divrem
//!     # then grep the emitted PTX for the suspect shape:
//!     grep -E 'ld\.local\.b32|shl\.b64.*32|mul\.hi\.u64.*7544311872078572213' \
//!         crates/rustc-codegen-cuda/examples/base58_limb_divrem/base58_limb_divrem.ptx

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const NEXT_LIMB_DIVISOR: u64 = 58_u64.pow(5); // = 656_356_768

#[cuda_module]
pub mod kernels {
    use super::*;

    /// `args` layout:
    ///   args[0] = the u32 limb value (low 32 bits used)
    ///   args[1] = the u64 carry
    ///   args[2] = runtime index into limbs[] (0..7)
    ///
    /// Per-thread output (3 threads):
    ///   thread 0 → quotient  of `(carry + (limbs[idx] << 32)) / 58^5`
    ///   thread 1 → remainder of the same
    ///   thread 2 → reconstructed dividend (for host sanity)
    #[kernel]
    pub fn run(args: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < 3
            && args.len() >= 3
        {
            let mut limbs = [0u32; 8];
            // Runtime index defeats mem2reg — the array stays in
            // local memory, the read becomes ld.local.b32.
            let write_idx = (args[2] as usize) & 7;
            limbs[write_idx] = args[0] as u32;

            let carry = args[1];
            // The exact base58 reconstruction:
            //   dividend = carry + (limbs[i] as u64) << 32
            let dividend = carry.wrapping_add((limbs[write_idx] as u64) << 32);

            *slot = match i {
                0 => dividend / NEXT_LIMB_DIVISOR,
                1 => dividend % NEXT_LIMB_DIVISOR,
                2 => dividend,
                _ => 0,
            };
        }
    }
}

fn main() {
    println!("=== base58_limb_divrem ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    // Pick a limb value whose `<< 32` contribution dominates the
    // dividend's high half — distinguishes "low only" from full divrem.
    let limb_u32: u64 = 0x089A_23FF; // first u32 chunk of slot 3's input
    let carry: u64 = 0xDEAD_BEEF;
    let write_idx: u64 = 3;

    let dividend = carry.wrapping_add(limb_u32 << 32);
    let expected_q = dividend / NEXT_LIMB_DIVISOR;
    let expected_r = dividend % NEXT_LIMB_DIVISOR;

    let args_host = [limb_u32, carry, write_idx];
    let args = DeviceBuffer::from_host(&stream, &args_host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 3).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(3), &args, &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    println!("dividend  host = {:#x}, GPU = {:#x}", dividend, result[2]);
    println!("quotient  host = {}, GPU = {}", expected_q, result[0]);
    println!("remainder host = {}, GPU = {}", expected_r, result[1]);

    assert_eq!(
        result[2], dividend,
        "dividend reconstruction failed — shl+add path wrong"
    );
    assert_eq!(
        result[0], expected_q,
        "quotient mismatch — divrem-by-58^5 broken in this PTX shape"
    );
    assert_eq!(
        result[1], expected_r,
        "remainder mismatch — divrem-by-58^5 broken in this PTX shape"
    );
    println!("SUCCESS: base58 limb divrem returns correct values");
}
