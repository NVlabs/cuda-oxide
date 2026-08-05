/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression tests for the runtime row pitch carried inside
//! `DisjointSlice<T, Runtime2DIndex>`.
//!
//! Three properties, each of which failed loudly at some point in the
//! pitched-slice design:
//!
//! 1. **Nonzero pitch readback**: the host binds a pitch via
//!    `cuda_host::Pitched`, and every device thread must read that exact
//!    value back. An entry prologue that dropped the third kernel parameter
//!    compiled and ran while giving every thread pitch 0; only checking a
//!    NONZERO value catches that.
//!
//! 2. **Two-pitch witness mixing stays sound**: a `Runtime2DIndex` witness
//!    carries the thread's `(row, col)` coordinates, and `get_mut` resolves
//!    them against the ADDRESSED slice's own pitch. Safe code that mints a
//!    witness from each of two differently pitched slices and selects one
//!    under a thread-varying condition must still write each thread to its
//!    own cell of the addressed grid. A flat-index witness fails this test:
//!    the minting slice's grid would leak through the selection.
//!
//! 3. **By-value pitched slice across a non-inlined call boundary**: passing
//!    a pitched `DisjointSlice` by value to an `#[inline(never)]` device
//!    helper must marshal all three fields (ptr, len, pitch) through the
//!    internal call ABI, matching the three-parameter callee signature.
//!
//! Run: cargo oxide run pitched_slice

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_host::cuda_module;

const SENTINEL: u32 = 0xDEAD_BEEF;

#[cuda_module]
mod kernels {
    use cuda_device::thread::{Runtime2DIndex, ThreadIndex};
    use cuda_device::{DisjointSlice, kernel, thread};

    /// Property 1: every in-grid thread writes the pitch it read back from
    /// the slice. A dropped or zeroed pitch mints no witness at all (pitch 0
    /// resolves nothing), leaving the sentinel behind for the host to catch.
    #[kernel]
    pub fn write_pitch_readback(mut out: DisjointSlice<u32, Runtime2DIndex>) {
        let pitch = out.row_pitch();
        if let Some(idx) = thread::index_2d_runtime(&out)
            && let Some(cell) = out.get_mut(idx)
        {
            *cell = pitch;
        }
    }

    /// Property 2: mint witnesses from BOTH slices (pitch 5 and pitch 100),
    /// select one under a thread-varying condition, and address `b` through
    /// whichever witness won. Per-slice resolution must land every thread on
    /// `b`'s own cell `(row, col)`, so writing the expected flat index makes
    /// any grid leakage or aliasing a value mismatch the host can see.
    #[kernel]
    pub fn two_pitch_selection(
        a: DisjointSlice<u32, Runtime2DIndex>,
        mut b: DisjointSlice<u32, Runtime2DIndex>,
    ) {
        let row = thread::index_2d_row();
        let col = thread::index_2d_col();
        let b_pitch = b.row_pitch() as usize;
        let wa = thread::index_2d_runtime(&a);
        let wb = thread::index_2d_runtime(&b);
        if let (Some(wa), Some(wb)) = (wa, wb) {
            // Thread-varying selection between two witnesses of one type:
            // exactly the shape that aliased flat-index witnesses.
            let chosen = if (row + col) % 2 == 0 { wa } else { wb };
            if let Some(cell) = b.get_mut(chosen) {
                *cell = (row * b_pitch + col) as u32;
            }
        }
    }

    /// Property 3 callee: takes the pitched slice BY VALUE. `inline(never)`
    /// keeps the call visible to the device pipeline, so the three-field
    /// slice must survive the internal call ABI intact.
    #[inline(never)]
    fn write_pitch_by_value(
        mut c: DisjointSlice<u32, Runtime2DIndex>,
        idx: ThreadIndex<Runtime2DIndex>,
    ) {
        let pitch = c.row_pitch();
        if let Some(cell) = c.get_mut(idx) {
            *cell = pitch;
        }
    }

    /// Property 3 caller: mints the witness, then moves the slice and the
    /// witness across the non-inlined call boundary.
    #[kernel]
    pub fn byvalue_helper_pitch(out: DisjointSlice<u32, Runtime2DIndex>) {
        if let Some(idx) = thread::index_2d_runtime(&out) {
            write_pitch_by_value(out, idx);
        }
    }
}

fn launch_cfg(grid: (u32, u32), block: (u32, u32)) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (grid.0, grid.1, 1),
        block_dim: (block.0, block.1, 1),
        shared_mem_bytes: 0,
    }
}

fn device_buffer_of_sentinels(
    stream: &cuda_core::CudaStream,
    len: usize,
) -> DeviceBuffer<u32> {
    DeviceBuffer::from_host(stream, &vec![SENTINEL; len]).expect("sentinel buffer")
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    // ── Property 1: nonzero pitch readback ─────────────────────────────
    const PITCH: u32 = 37;
    const ROWS: usize = 8;
    let len = PITCH as usize * ROWS;
    let mut out = device_buffer_of_sentinels(&stream, len);
    // SAFETY: 2D launch; the kernel bounds itself by the slice's pitch/len.
    unsafe {
        module.write_pitch_readback(
            &stream,
            launch_cfg((3, 1), (16, 16)),
            cuda_host::Pitched::new(&mut out, PITCH),
        )
    }
    .expect("write_pitch_readback launch");
    let host = out.to_host_vec(&stream).unwrap();
    for (i, &v) in host.iter().enumerate() {
        assert_eq!(
            v, PITCH,
            "out[{i}]: got {v:#x}, want pitch {PITCH}; a zero or dropped pitch reached the device"
        );
    }
    println!("pitch readback: all {len} cells saw pitch {PITCH}");

    // ── Property 2: two-pitch witness mixing ───────────────────────────
    const PITCH_A: u32 = 5;
    const PITCH_B: u32 = 100;
    const ROWS_B: usize = 4;
    let mut a = device_buffer_of_sentinels(&stream, PITCH_A as usize * ROWS_B);
    let mut b = device_buffer_of_sentinels(&stream, PITCH_B as usize * ROWS_B);
    // SAFETY: 2D launch; each thread writes at most one cell of `b`.
    unsafe {
        module.two_pitch_selection(
            &stream,
            launch_cfg((1, 1), (16, 16)),
            cuda_host::Pitched::new(&mut a, PITCH_A),
            cuda_host::Pitched::new(&mut b, PITCH_B),
        )
    }
    .expect("two_pitch_selection launch");
    let host_a = a.to_host_vec(&stream).unwrap();
    let host_b = b.to_host_vec(&stream).unwrap();
    for (i, &v) in host_a.iter().enumerate() {
        assert_eq!(v, SENTINEL, "a[{i}] was written; `a` must stay untouched");
    }
    // Threads with col < 5 hold both witnesses; whichever they select must
    // resolve against b's pitch of 100. Everything else stays sentinel.
    for row in 0..16usize {
        for col in 0..PITCH_B as usize {
            let flat = row * PITCH_B as usize + col;
            if flat >= host_b.len() {
                continue;
            }
            let expected = if col < PITCH_A as usize && row < ROWS_B {
                flat as u32
            } else {
                SENTINEL
            };
            assert_eq!(
                host_b[flat], expected,
                "b[{flat}] (row {row}, col {col}): got {:#x}, want {expected:#x}; \
                 a witness resolved against the wrong slice's pitch",
                host_b[flat]
            );
        }
    }
    println!("two-pitch selection: every thread landed on b's own (row, col) cell");

    // ── Property 3: by-value pitched slice through a helper ────────────
    const PITCH_C: u32 = 13;
    const ROWS_C: usize = 4;
    let len_c = PITCH_C as usize * ROWS_C;
    let mut c = device_buffer_of_sentinels(&stream, len_c);
    // SAFETY: 2D launch; the helper bounds itself by the slice's pitch/len.
    unsafe {
        module.byvalue_helper_pitch(
            &stream,
            launch_cfg((1, 1), (16, 16)),
            cuda_host::Pitched::new(&mut c, PITCH_C),
        )
    }
    .expect("byvalue_helper_pitch launch");
    let host_c = c.to_host_vec(&stream).unwrap();
    for (i, &v) in host_c.iter().enumerate() {
        assert_eq!(
            v, PITCH_C,
            "c[{i}]: got {v:#x}, want pitch {PITCH_C}; the by-value call ABI dropped the pitch"
        );
    }
    println!("by-value helper: pitch {PITCH_C} survived the internal call ABI");

    println!("SUCCESS: runtime pitch bound, resolved per-slice, and marshalled by value");
}
