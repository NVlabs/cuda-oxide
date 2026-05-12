//! Regression test for `core::intrinsics::typed_swap_nonoverlapping`
//! lowering.
//!
//! ## Pre-fix wall
//!
//! ```text
//! Symbol _RINvNtCsbBDxv2Oq2Kj_4core10intrinsics25typed_swap_nonoverlappinghE
//! not found
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/` via any `core::mem::swap` call,
//! which lowers to `typed_swap_nonoverlapping<T>(x: *mut T, y: *mut T)`.
//! The intrinsic has no MIR body, so the collector skipped it; the
//! translator emitted a regular call to a symbol nothing defined.
//!
//! ## What landed
//!
//! New inline handler `emit_typed_swap_nonoverlapping` in
//! `terminator/mod.rs` registered in `try_dispatch_intrinsic`'s
//! match block. Lowers the call as:
//!
//!   - `tmp_x = mir.load(x)` (T)
//!   - `tmp_y = mir.load(y)` (T)
//!   - `mir.store(x, tmp_y)`
//!   - `mir.store(y, tmp_x)`
//!   - unit-store + goto epilogue (same as `emit_unit_noop_intrinsic`)
//!
//! No mir-lower placeholder needed — all ops already exist. T is
//! recovered from the x arg's `MirPtrType` pointee.
//!
//! ## Build with
//!
//!     cargo oxide build typed_swap_intrinsic

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[inline(never)]
fn swap_pair(a: &mut u64, b: &mut u64) {
    core::mem::swap(a, b);
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(input: &[u64], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && (i + 1) * 2 <= input.len()
        {
            let base = i * 2;
            let mut a = input[base];
            let mut b = input[base + 1];
            super::swap_pair(&mut a, &mut b);
            // After swap, slot = a (which was input[base+1]) - input[base]
            *slot = a.wrapping_sub(b);
        }
    }
}

fn main() {
    println!("=== typed_swap_intrinsic ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u64> = (0..(N * 2) as u64).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        let base = i * 2;
        let expected = host[base + 1].wrapping_sub(host[base]);
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: typed_swap_nonoverlapping codegen'd to PTX");
}
