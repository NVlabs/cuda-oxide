/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `core::ptr::copy_nonoverlapping` codegen.
//!
//! ## Pre-fix wall
//!
//! `copy_nonoverlapping` lowers to a MIR
//! `StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(_))` —
//! a *statement* with `(src, dst, count)` operands, not a
//! `Terminator::Call`. Before the lowering landed, the importer hard-errored
//! to keep the previous catch-all from silently dropping the statement and
//! producing PTX where the memcpy was missing entirely. Build failed with:
//!
//! ```text
//! Unsupported construct: core::ptr::copy_nonoverlapping is not yet
//! supported on the device; until it is lowered, the call would be
//! silently dropped from the PTX
//! ```
//!
//! ## What landed
//!
//! 1. `crates/dialect-mir/src/rust_intrinsics.rs` — `CALLEE_COPY_NONOVERLAPPING`
//!    placeholder string.
//! 2. `crates/mir-importer/src/translator/statement.rs::translate_copy_nonoverlapping`
//!    — reshapes the MIR statement into a void `mir.call` carrying the
//!    placeholder and `(src, dst, count)` operands. Replaces the previous
//!    hard-error in the `Intrinsic(CopyNonOverlapping)` arm.
//! 3. `crates/mir-lower/src/convert/ops/call.rs::convert_rust_copy_nonoverlapping`
//!    — recovers `sizeof(T)` from the `dst` operand's most-recent
//!    `MirPtrType` (same mechanism `convert_rust_raw_eq` /
//!    `convert_rust_ptr_arith_intrinsic` use), multiplies by `count` to get
//!    a byte length, and emits
//!    `@llvm.memcpy.p0.p0.i64(dst, src, byte_len, false)`. NVPTX legalizes
//!    that into byte / vector ld+st sequences.
//!
//! ZST pointee short-circuits to a deleted call (zero-byte copies are
//! always no-ops regardless of `count`).
//!
//! ## What triggers it
//!
//! The kernel calls `core::ptr::copy_nonoverlapping(src, dst, 1)` directly.
//! Originally surfaced from `~/vanity-miner-rs/` via the indirect path —
//! `<[u8]>::copy_from_slice` calls into `copy_nonoverlapping` for buffer
//! marshalling in the SHA-256 / message-prep code.
//!
//! ## Build with
//!
//!     cargo oxide run copy_nonoverlapping_basic
//!
//! Expected: kernel runs, each output slot equals the corresponding input.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
pub mod kernels {
    use super::*;

    /// Trigger: `core::ptr::copy_nonoverlapping(src, dst, 1)` copies one
    /// `u32` (4 bytes) from `input[idx]` into the output slot.
    #[kernel]
    pub fn copy_nonoverlapping_kernel(input: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx)
            && i < input.len()
        {
            unsafe {
                let src = input.as_ptr().add(i);
                let dst = slot as *mut u32;
                core::ptr::copy_nonoverlapping(src, dst, 1);
            }
        }
    }
}

fn main() {
    println!("=== copy_nonoverlapping_basic ===");

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    const N: usize = 32;
    let host: Vec<u32> = (0..N as u32).map(|n| n * 7 + 1).collect();

    let input = DeviceBuffer::from_host(&stream, &host).unwrap();
    let mut out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("kernels::load");
    module
        .copy_nonoverlapping_kernel(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input,
            &mut out,
        )
        .expect("kernel launch");

    let result = out.to_host_vec(&stream).unwrap();
    for i in 0..N {
        assert_eq!(result[i], host[i], "thread {} mismatch", i);
    }
    println!("SUCCESS: copy_nonoverlapping codegen'd to @llvm.memcpy");
}
