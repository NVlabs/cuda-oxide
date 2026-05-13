//! Regression test for indexing a `&'static [u8]` slice constant.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Invalid __global__ read of size 1 bytes
//!     at $kernel_self_test_*+0xNNN
//!     Access to 0xfff_XXXXXXXX_XX is out of bounds
//!     and is N petabytes after the nearest allocation
//! ```
//!
//! Surfaced from `vanity-miner-rs/logic/src/self_test.rs` slots 41
//! (`base58 var-len`), 43 (`base58 all-zeros`), 45 (`bech32
//! p2wpkh`). Each compares `out[i] != EXPECTED[i]` where `EXPECTED`
//! is a `const EXPECTED: &[u8] = b"…";` slice constant. The build
//! succeeds, the PTX looks reasonable, and the kernel runs — until
//! it dereferences the slice base pointer, which was never
//! materialized.
//!
//! ## Root cause
//!
//! [crates/mir-importer/src/translator/rvalue.rs:2501-2519] lowers
//! every slice-typed constant to `mir.undef` on the load-bearing
//! assumption that `&'static [u8]` constants only appear in panic
//! helper code paths (where the unreachable terminator makes the
//! undef's value unobservable). The user's bisection slots actually
//! consume slice constants at runtime — the undef materialises into a
//! register holding whatever garbage was there, and the indexed load
//! reads from a wild address.
//!
//! ## What needs to land
//!
//! Slice constants need real materialisation: emit the underlying
//! bytes as a `MirGlobalAllocOp` (or reuse one if the same bytes have
//! been seen) and produce a fat-pointer `(ptr, len)` value. Same
//! shape as the `static`-item path already in rvalue.rs, just
//! triggered by `ConstantKind::Allocated` with slice type instead of
//! by `static_def_from_constant`.
//!
//! ## Why this repro doesn't use `core::hint::black_box`
//!
//! `black_box` had its own lowering bug (untied asm constraint;
//! fixed in commit 6cd7aab). Driving the index from a kernel
//! parameter buffer instead means the repro stays valid regardless
//! of what's happening to `black_box` at any given point.
//!
//! ## Build / run
//!
//!     cargo oxide build slice_const_indexing
//!     cargo oxide run   slice_const_indexing   # needs a GPU; faults pre-fix

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Indexes a `&'static [u8]` slice constant with an index supplied via a
    /// kernel parameter buffer. The host writes `idx_buf[i] = 0`; the kernel
    /// should write `out[i] = TABLE[0] = b'1'`. Pre-fix the slice's base
    /// pointer is undef, so the load goes to a wild address and faults.
    #[kernel]
    pub fn run_slice(idx_buf: &[u64], mut out: DisjointSlice<u8>) {
        const TABLE: &[u8] = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < idx_buf.len()
        {
            let table_idx = idx_buf[i] as usize;
            *slot = TABLE[table_idx];
        }
    }
}

fn main() {
    println!("=== slice_const_indexing ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 16;
    let idx_host: Vec<u64> = (0..N as u64).collect();
    let idx_buf = DeviceBuffer::from_host(&stream, &idx_host).unwrap();
    let mut out = DeviceBuffer::<u8>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run_slice(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &idx_buf,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    const EXPECTED_TABLE: &[u8] =
        b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for i in 0..N {
        assert_eq!(
            result[i], EXPECTED_TABLE[i],
            "thread {i}: got {:#x}, expected {:#x}",
            result[i], EXPECTED_TABLE[i]
        );
    }
    println!("SUCCESS: slice-constant indexing reads the right bytes");
}
