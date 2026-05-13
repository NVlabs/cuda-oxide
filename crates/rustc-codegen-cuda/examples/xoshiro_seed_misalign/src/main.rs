//! Regression test for `ld.local.v2.b64` being emitted against a
//! local depot with alignment less than the wide load requires.
//!
//! ## Pre-fix wall
//!
//! `cargo oxide build` succeeded, but the emitted PTX contained
//! `ld.local.v2.b64 {%rd, %rd}, [...]` reading from a `.local .align 1`
//! depot (vanity-miner-rs equivalent: `.align 8`). Wide local loads
//! need 16-byte natural alignment; the depot didn't have it.
//!
//! The fault was **nondeterministic**: it depended on whether the
//! depot's runtime address happened to be 16-byte aligned. Shallow
//! kernels (this repro, `array_eq_raw`) tended to land 16-aligned and
//! ran fine. Deep kernels (vanity-miner-rs slot 0) reliably got a
//! depot at `0x...8` and faulted with `CUDA_ERROR_ILLEGAL_ADDRESS`
//! (700). compute-sanitizer on the surfacing kernel:
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
//! ## What landed
//!
//! `crates/dialect-llvm/src/export.rs::conservative_alloca_align`:
//! the LLVM exporter now emits `alloca <type>, align N` where
//! `N = min(16, next_pow2(byte_size_of(type)))`. NVPTX wide ops cap
//! at 16-byte natural alignment (`b128`, `v2.b64`, `v4.b32`), so 16
//! is sufficient. `next_pow2(size)` caps the cost — a 4-byte alloca
//! stays `.align 4`; only 16-byte-or-larger storage gets bumped to
//! 16.
//!
//! Deliberately a blunt instrument. The more surgical fix is to
//! thread pointer alignment through mir-lower's load/store emission
//! and attach `align N` metadata to each load/store, letting NVPTX
//! backend pick the right lowering instead of forcing storage to
//! satisfy the worst-case consumer. That's a real refactor; the
//! conservative bump unblocks the bug class as a one-locality fix
//! and wastes at most ~7 bytes of stack per under-aligned alloca.
//! See the TODO on `conservative_alloca_align` for the longer plan.
//!
//! Post-fix PTX shape:
//!
//! - `ld.local.v2.b64` here now reads from `.local .align 16`
//!   depots. The wide load itself isn't gone — it's operating on
//!   correctly-aligned storage.
//! - `array_eq_raw`'s identical-shape latent bug fixed by the same
//!   change (depot went `.align 8` -> `.align 16`).
//! - `abi_hmm`'s `st.local.v2.b64` against its `.align 8` depot
//!   fixed by the same change (depot bumped to 16).
//! - `cross_crate_pubfn` no longer emits any wide local ops at all
//!   — the bumped allocas let NVPTX make better lowering choices
//!   end-to-end.
//!
//! ## Relationship to other repros
//!
//! - `array_eq_raw` -- introduced the `raw_eq` intrinsic handler
//!   that emits the wide load. That fix was correct in isolation;
//!   this repro surfaced that the wide-load shape it produces was
//!   not alignment-safe against its source storage.
//! - `nested_struct_const`, `slice_const_idx_write`, etc. -- unrelated
//!   bugs at codegen-failure time. This bug was at runtime.
//!
//! ## Build with
//!
//!     cargo oxide build xoshiro_seed_misalign   # codegen check
//!     cargo oxide run   xoshiro_seed_misalign   # full hardware run

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirrors `Xoroshiro128StarStar::from_seed`'s `deal_with_zero_seed!`
/// macro: take the seed by value, compare against `[0u8; 16]`. The
/// `==` triggers `ld.local.v2.b64` against the callee's stack alloca.
/// Pre-fix that alloca was `.align 8`, mismatching the load's 16-byte
/// requirement; post-fix the exporter bumps it to `.align 16`.
/// `#[inline(never)]` keeps the by-value parameter from being
/// collapsed into a borrow of the caller's alloca.
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
    /// alloca is what the wide load reads from; pre-fix that alloca
    /// was under-aligned for the load width.
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
