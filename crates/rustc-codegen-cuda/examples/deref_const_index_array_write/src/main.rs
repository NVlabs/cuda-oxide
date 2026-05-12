//! Known-failure repro for the `Deref -> ConstantIndex` writer arm
//! when the slot is `&mut [T; N]` (pointer-to-array), not the
//! pointer-to-slice case the existing handler covers.
//!
//! ## Wall (current state)
//!
//! ```text
//! Compilation error: invalid input program.
//! Unsupported construct: Deref->ConstantIndex write expects slot of
//! MirPtrType<MirSliceType<T>>, got mir.ptr <mir.ptr <mir.array <u32,8>>>
//! ```
//!
//! Surfaced from `~/vanity-miner-rs/`'s `logic::sha256::process_block`
//! at `logic/src/sha256.rs:102:5: 102:40` — every line of the SHA-256
//! state update writes `state[k] = value` where `state: &mut [u32; 8]`,
//! lowering to `Place(local, [Deref, ConstantIndex { offset: k, .. }])`.
//!
//! ## Where it bails
//!
//! `crates/mir-importer/src/translator/statement.rs` — the
//! `(Deref, ConstantIndex)` writer arm only handles
//! `MirPtrType<MirSliceType<T>>` slots (`&mut [T]`). The sibling
//! `(Deref, Index(local))` arm already dispatches on both the array
//! and slice pointee shapes (see `examples/deref_index_local_write/`);
//! the constant-index arm just needs the same dispatch.
//!
//! ## Build with
//!
//!     cargo oxide build deref_const_index_array_write

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Mirrors the `process_block` shape from `~/vanity-miner-rs/`'s
/// SHA-256: takes a `&mut [u32; 8]` and writes constant-indexed slots.
/// `#[inline(never)]` keeps the slot's pointee at `[u32; 8]` (a thin
/// pointer to the array) instead of being collapsed.
#[inline(never)]
fn process_block(state: &mut [u32; 8], input: u32) {
    state[0] = state[0].wrapping_add(input);
    state[1] = state[1].wrapping_add(input.wrapping_mul(2));
    state[2] = state[2].wrapping_add(input.wrapping_mul(3));
    state[3] = state[3].wrapping_add(input.wrapping_mul(5));
    state[4] = state[4].wrapping_add(input.wrapping_mul(7));
    state[5] = state[5].wrapping_add(input.wrapping_mul(11));
    state[6] = state[6].wrapping_add(input.wrapping_mul(13));
    state[7] = state[7].wrapping_add(input.wrapping_mul(17));
}

#[cuda_module]
pub mod kernels {
    use super::*;

    #[kernel]
    pub fn sha256_state_update(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            let mut state: [u32; 8] = [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ];
            super::process_block(&mut state, input[i]);
            // XOR-fold so the value depends on every constant-index
            // write inside `process_block`.
            *slot = state[0]
                ^ state[1]
                ^ state[2]
                ^ state[3]
                ^ state[4]
                ^ state[5]
                ^ state[6]
                ^ state[7];
        }
    }
}

fn main() {
    println!("=== deref_const_index_array_write ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 8;
    let host: Vec<u32> = (0..N as u32).collect();
    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .sha256_state_update(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    let mults = [1u32, 2, 3, 5, 7, 11, 13, 17];
    let init = [
        0x6a09_e667u32,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    for i in 0..N {
        let mut state = init;
        for k in 0..8 {
            state[k] = state[k].wrapping_add(host[i].wrapping_mul(mults[k]));
        }
        let expected = state.iter().fold(0u32, |a, b| a ^ b);
        assert_eq!(result[i], expected, "thread {} mismatch", i);
    }
    println!("SUCCESS: deref+constant-index array write codegen'd to PTX");
}
