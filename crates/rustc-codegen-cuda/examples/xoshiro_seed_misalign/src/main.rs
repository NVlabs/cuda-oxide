//! Known-failure reproducer for `ld.local.v2.b64` being emitted
//! against an 8-byte-aligned local depot.
//!
//! ## Status
//!
//! `cargo oxide build` succeeds today. The emitted PTX contains
//! `ld.local.v2.b64 {%rd, %rd}, [...]` reading from a `.local .align 1`
//! depot (in vanity-miner-rs the equivalent depot is `.align 8`; either
//! way it's < 16, which is what the wide load needs).
//!
//! The runtime fault is **nondeterministic**: it depends on whether the
//! depot's runtime address happens to be 16-byte aligned. Shallow
//! kernels (this repro, `array_eq_raw`) tend to land 16-aligned and run
//! fine; deep kernels (vanity-miner-rs slot 0) reliably get a depot at
//! `0x...8` and fault with `CUDA_ERROR_ILLEGAL_ADDRESS` (700).
//!
//! Verify the bad PTX shape:
//!
//! ```sh
//! cargo oxide build xoshiro_seed_misalign
//! grep -B1 'ld\.local\.v2\.b64' \
//!     crates/rustc-codegen-cuda/examples/xoshiro_seed_misalign/xoshiro_seed_misalign.ptx
//! ```
//!
//! Bug present today: `ld.local.v2.b64` reads from a register that
//! points into `__local_depot? : .local .align N` where `N < 16`.
//! After the fix: either the depot is bumped to `.align 16`, or the
//! wide load is split into two `ld.local.b64`s.
//!
//! ## Pre-fix wall (compute-sanitizer on real GPU)
//!
//! ```text
//! Invalid __local__ read of size 16 bytes
//!     at $kernel_..._Xoroshiro128StarStar9from_seed+0x70c0
//!     by thread (0,0,0) in block (0,0,0)
//!     Access to 0xffee08 is misaligned
//! ```
//!
//! `0xffee08` ends in `0x08` -> 8-byte aligned, not 16. The 16-byte
//! `ld.local.v2.b64` requires natural 16-byte alignment.
//!
//! Surfaced from `~/vanity-miner-rs/`'s self-test slot 0
//! (`kernel_self_test_solana_priv`):
//!   `logic::generate_random_private_key`
//!     -> `Xoroshiro128StarStar::seed_from_u64`
//!     -> `from_rng` (using `SplitMix64`)
//!     -> `Xoroshiro128StarStar::from_seed`        <-- faulting frame
//!         -> `deal_with_zero_seed!(seed, Self)`   <-- the `==` check
//!
//! ## What triggers it
//!
//! `Xoroshiro128StarStar::from_seed` opens with the
//! `deal_with_zero_seed!` macro from `rand_xoshiro`:
//!
//! ```rust,ignore
//! fn from_seed(seed: Self::Seed) -> Self {
//!     deal_with_zero_seed!(seed, Self);   // <-- expands to `if seed == [0; 16] { ... }`
//!     ...
//! }
//! ```
//!
//! The `==` on `[u8; 16]` lowers through `<[u8; 16] as PartialEq>::eq`
//! -> `core::intrinsics::raw_eq` (mir-importer handler added by
//! `array_eq_raw`). The handler emits a wide 16-byte load + XOR + ne
//! check. NVPTX backend turns that into `ld.local.v2.b64`, which
//! requires the source to be 16-byte aligned.
//!
//! The source is a by-value parameter `seed: [u8; 16]` whose callee
//! alloca lives inside `__local_depot? : .local .align 8`. PTX
//! alignment is a contract with the driver: the JIT trusts the
//! depot's declared alignment and places it at an 8-byte boundary
//! that may or may not also be 16-byte. When the runtime address
//! ends in `0x8`, the wide load faults.
//!
//! ## What we expect to land
//!
//! In `crates/mir-lower/src/convert/ops/`:
//!
//! 1. **Either** when lowering a vector / wide load whose source
//!    pointer's underlying alloca has alignment smaller than the
//!    load's natural alignment, split the load into element-sized
//!    loads honouring the alloca's alignment, **or**
//! 2. when lowering an alloca that has any wide load against it,
//!    bump the alloca's alignment up to
//!    `max(declared_align, natural_align_of_widest_load)`.
//!
//! Option 2 is simpler but costs stack space; option 1 is more
//! surgical. Either flips this repro (and `array_eq_raw`) from
//! "PTX may fault depending on stack offset" to "PTX is alignment-
//! safe regardless of runtime address."
//!
//! ## Why a fresh hardware run of this repro may pass
//!
//! Single-kernel, shallow-stack workloads land the depot at a
//! 16-aligned address by luck. Confirming the fix is in place is
//! a PTX-text check, not a hardware run:
//!
//! - Before fix: PTX has `ld.local.v2.b64` against `.align 8` depot.
//! - After fix: PTX has either `.align 16` depot or two
//!   `ld.local.b64`s (no wide local load against under-aligned source).
//!
//! ## Relationship to other repros
//!
//! - `array_eq_raw` -- introduced the `raw_eq` intrinsic handler
//!   that emits the wide load. That fix is correct in isolation;
//!   this repro proves that the wide-load shape it produces is
//!   not alignment-safe.
//! - `nested_struct_const`, `slice_const_idx_write`, etc. -- unrelated
//!   bugs at codegen-failure time. This bug is at runtime.
//!
//! ## Build with
//!
//!     cargo oxide build xoshiro_seed_misalign   # codegen check (passes today)
//!     cargo oxide run   xoshiro_seed_misalign   # passes on shallow stacks; faults on deep

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirrors `Xoroshiro128StarStar::from_seed`'s `deal_with_zero_seed!`
/// macro: take the seed by value, compare against `[0u8; 16]`. The
/// `==` here is what triggers the `ld.local.v2.b64` against the
/// callee's `.align 8` stack alloca. `#[inline(never)]` keeps the
/// by-value parameter from being collapsed into a borrow of the
/// caller's alloca.
#[inline(never)]
fn from_seed_repro(seed: [u8; 16]) -> u64 {
    // Trigger: `==` on `[u8; 16]` lowers via `raw_eq`, emitting a
    // wide 16-byte load + XOR. PTX: `ld.local.v2.b64` against the
    // 8-byte-aligned local depot for `seed`.
    if seed == [0u8; 16] {
        return 0;
    }
    // Use the seed so dead-code elimination doesn't drop the alloca.
    u64::from_le_bytes([
        seed[0], seed[1], seed[2], seed[3], seed[4], seed[5], seed[6], seed[7],
    ])
}

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Stages a `[u8; 16]` onto the kernel's stack, then passes it
    /// by value into `from_seed_repro`. The callee's own stack
    /// alloca is the 8-byte-aligned storage that the wide load
    /// reads from.
    #[kernel]
    pub fn xoshiro_seed_misalign(input: &[u8], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            // SAFETY: caller ensures `input.len() >= (i + 1) * 16`.
            // Bytewise copy defeats any optimizer attempt to read
            // directly from `input`'s global memory.
            let base = i * 16;
            let mut seed = [0u8; 16];
            let mut k = 0;
            while k < 16 {
                seed[k] = input[base + k];
                k += 1;
            }

            // Pass by value: callee gets its own stack alloca for
            // the [u8; 16]. The `==` inside emits the wide load.
            *slot = super::from_seed_repro(seed);
        }
    }
}

fn main() {
    println!("=== xoshiro_seed_misalign ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const THREADS: usize = 4;
    const CHUNK: usize = 16;
    const INPUT_LEN: usize = THREADS * CHUNK;

    // Inputs: bytes 0..64 -- never all-zero, so each thread takes
    // the non-zero branch.
    let host: Vec<u8> = (0..INPUT_LEN as u32).map(|n| ((n & 0xff) as u8).max(1)).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, THREADS).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .xoshiro_seed_misalign(
            &stream,
            LaunchConfig::for_num_elems(THREADS as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..THREADS {
        let base = i * CHUNK;
        let expected = u64::from_le_bytes([
            host[base],
            host[base + 1],
            host[base + 2],
            host[base + 3],
            host[base + 4],
            host[base + 5],
            host[base + 6],
            host[base + 7],
        ]);
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: xoshiro from_seed shape codegen'd to PTX");
}
