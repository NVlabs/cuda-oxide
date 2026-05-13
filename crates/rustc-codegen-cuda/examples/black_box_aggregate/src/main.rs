//! Codegen-time known-failure: `core::hint::black_box` on aggregate
//! types (arrays, structs, tuples).
//!
//! ## Pre-fix wall
//!
//! ```text
//! Lowering failed: Compilation error: invalid input program.
//! nvvm.black_box of non-integer type not yet supported
//! ```
//!
//! mir-lower's `convert_black_box` only handles scalar integer
//! widths (1/8/16/32/64/128). Any aggregate input — array, struct,
//! tuple — trips the catch-all error.
//!
//! Surfaced from vanity-miner-rs's dalek bisection slots (84/85/86/87),
//! which black-box the input bytes/limbs/Scalar52 wrapper before
//! handing them to `from_bytes`, `montgomery_reduce`, etc.:
//!
//! ```ignore
//! let bytes = core::hint::black_box(bytes);             // [u8; 32]
//! let widened = core::hint::black_box(widened);         // [u128; 9]
//! let one = core::hint::black_box(ONE);                 // Scalar52
//! let r = core::hint::black_box(bisect_scalar52::R);    // Scalar52
//! ```
//!
//! ## Build
//!
//!     cargo oxide build black_box_aggregate   # fails today

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn run(mut out: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            let mut bytes = [0u8; 32];
            bytes[0] = 1;
            let bytes = core::hint::black_box(bytes);
            *slot = bytes[0].wrapping_add(bytes[31]);
        }
    }
}

fn main() {
    println!("=== black_box_aggregate ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let mut out = DeviceBuffer::<u8>::zeroed(&stream, 1).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .run(&stream, LaunchConfig::for_num_elems(1), &mut out)
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], 1);
    println!("SUCCESS: black_box of [u8; 32] preserved value");
}
