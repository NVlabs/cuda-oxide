//! Candidate repro for divrem-by-a-large-constant (Bug B hypothesis).
//!
//! `slot 59` of vanity-miner-rs's self-test (`x / 58`) passes; `slot 3`
//! (the full `base58_encode_32` divide loop, which does
//! `x / 656_356_768` per limb) fails. Both divisors are small enough
//! to fit in 32 bits but produce very different magic-multiply
//! constants (~7 vs 30 effective bits).
//!
//! ## What this kernel does
//!
//! Operands come from a kernel-param buffer (host-written), so const
//! folding can't fold the divide. Per-thread `i`, write
//! `(args[i] / 656_356_768)` to `out[2*i]` and the corresponding
//! remainder to `out[2*i + 1]`. Host computes the same on CPU and
//! asserts byte equality.
//!
//! ## Pre-fix indicators (if Bug B is real)
//!
//! In the emitted PTX, look for the magic-multiply pattern:
//! `mul.hi.u64 %hi, %x, MAGIC; shr.u64 %q, %hi, SHIFT;` and check
//! whether `MAGIC` and `SHIFT` look right for divisor 656_356_768.
//! Wrong MAGIC or wrong SHIFT = bug. If the PTX uses `div.u64
//! %x, 656356768` directly (hardware divide), it's slow but correct
//! — the bug would have to be elsewhere.
//!
//! ## Build / run
//!
//!     cargo oxide build divrem_large_const
//!     cargo oxide run   divrem_large_const   # needs GPU

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const NEXT_LIMB_DIVISOR: u64 = 58_u64.pow(5); // = 656_356_768

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Writes `(args[i] / D, args[i] % D)` for D = 58^5 into out[2i..2i+2].
    #[kernel]
    pub fn run(args: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i / 2 < args.len()
        {
            let x = args[i / 2];
            // The divisor must be a compile-time constant for rustc to
            // produce the magic-multiply lowering — the same shape
            // base58_encode_32 hits.
            *slot = if i % 2 == 0 {
                x / NEXT_LIMB_DIVISOR
            } else {
                x % NEXT_LIMB_DIVISOR
            };
        }
    }
}

fn main() {
    println!("=== divrem_large_const ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    // Mix of values: a small one, the divisor itself, a multiple,
    // values straddling magic-multiply boundary cases, and the
    // first 4-byte chunk of slot 3's input (the failing case).
    let args_host: [u64; 6] = [
        0x089A23FF,                   // first 4 bytes of slot 3's input
        NEXT_LIMB_DIVISOR,            // exactly the divisor: should give (1, 0)
        NEXT_LIMB_DIVISOR - 1,        // (0, D-1)
        NEXT_LIMB_DIVISOR.wrapping_mul(7).wrapping_add(123), // (7, 123)
        u64::MAX,                     // stress the high end
        0xFFFFFFFF_00000000,
    ];
    let args = DeviceBuffer::from_host(&stream, &args_host).unwrap();
    let n_out = args_host.len() * 2;
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, n_out).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(n_out as u32), &args, &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for (i, &x) in args_host.iter().enumerate() {
        let q = x / NEXT_LIMB_DIVISOR;
        let r = x % NEXT_LIMB_DIVISOR;
        assert_eq!(
            result[2 * i],
            q,
            "thread {} quotient mismatch: input {:#x}, got {}, expected {}",
            2 * i,
            x,
            result[2 * i],
            q,
        );
        assert_eq!(
            result[2 * i + 1],
            r,
            "thread {} remainder mismatch: input {:#x}, got {}, expected {}",
            2 * i + 1,
            x,
            result[2 * i + 1],
            r,
        );
    }
    println!("SUCCESS: divrem by {} returns correct values", NEXT_LIMB_DIVISOR);
}
