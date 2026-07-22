/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for issue #138.
//!
//! `for x in arr` over a by-value array `[T; N]` desugars to a loop over
//! `core::array::IntoIter<T, N>`, and rustc places a `Drop` terminator
//! for the iterator at the loop exit because `IntoIter` has an
//! `impl Drop`. For element types without drop glue that destructor is
//! provably a no-op (`IntoIter::drop` is `if needs_drop::<T>() { .. }`,
//! which is statically false), so the importer lowers the `Drop`
//! terminator to a plain branch instead of rejecting the kernel.
//!
//! Before the fix the build failed with
//!
//!   Unsupported construct: drop of `...std::array::IntoIter...` is not
//!   supported on the device; cuda-oxide does not yet emit device-side
//!   `drop_in_place` calls.
//!
//! Two kernels cover the shapes from the issue: a `for` loop over a
//! plain `[u32; 4]` and one over an array of Copy structs. Both sums are
//! verified on the host.
//!
//! Run: cargo oxide run array_for_loop

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// A plain Copy struct; an array of these has no drop glue either.
    #[derive(Clone, Copy)]
    pub struct Point {
        pub x: u32,
        pub y: u32,
    }

    #[derive(Clone, Copy)]
    #[repr(C, align(32))]
    struct ScalarPair {
        values: [f32; 2],
        padding: [u8; 24],
    }

    #[derive(Clone, Copy)]
    struct GridShape {
        counts: [u32; 3],
        periodic: [bool; 3],
    }

    #[inline(always)]
    fn select_grid_field(grid: &GridShape, index: usize) -> u32 {
        let count = grid.counts[index];
        if grid.periodic[index] {
            count + 10
        } else {
            count
        }
    }

    trait LaneCount {
        const LANES: usize;
    }

    struct TwoLanes;

    impl LaneCount for TwoLanes {
        const LANES: usize = 2;
    }

    /// The issue-384 shape: a generic helper switching on an associated
    /// const. Collector dead-edge pruning and importer translation must fold
    /// `L::LANES` to the same `switchInt` edge, or the dead `panic!` arm gets
    /// pulled into device codegen by one of them. The live arm goes through
    /// `get_unchecked`, whose ub-checks precondition adds a
    /// `RuntimeChecks`-guarded edge that both walks must also fold
    /// identically (to the device value, never the host session's).
    fn lane_sum<L: LaneCount>(values: &[u32; 4]) -> u32 {
        match L::LANES {
            2 => {
                // SAFETY: indices 0 and 1 are in bounds of a `[u32; 4]`.
                unsafe { *values.get_unchecked(0) + *values.get_unchecked(1) }
            }
            _ => panic!("unsupported lane count"),
        }
    }

    /// Sum a by-value `[u32; 4]` with a `for` loop (the issue-138 shape).
    #[kernel]
    pub fn sum_u32_array(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let arr: [u32; 4] = [t, t + 1, t + 2, t + 3];
            let mut acc: u32 = 0;
            for x in arr {
                acc += x;
            }
            *out_elem = acc;
        }
    }

    /// Same loop shape over an array of Copy structs.
    #[kernel]
    pub fn sum_point_array(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let pts: [Point; 4] = [
                Point { x: t, y: 1 },
                Point { x: t + 1, y: 2 },
                Point { x: t + 2, y: 3 },
                Point { x: t + 3, y: 4 },
            ];
            let mut acc: u32 = 0;
            for p in pts {
                acc += p.x * p.y;
            }
            *out_elem = acc;
        }
    }

    /// Runtime reads from small owned arrays should remain in SSA.
    #[kernel]
    pub fn runtime_aggregate_array_read(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let index = tid.get();
        let t = index as f32;
        if let Some(out_elem) = out.get_mut(tid) {
            let components = [
                ScalarPair {
                    values: [t, t + 3.0],
                    padding: [0; 24],
                },
                ScalarPair {
                    values: [t + 1.0, t + 4.0],
                    padding: [0; 24],
                },
                ScalarPair {
                    values: [t + 2.0, t + 5.0],
                    padding: [0; 24],
                },
            ];
            let selected = components[index % components.len()];
            *out_elem = selected.values[0] + selected.values[1];
        }
    }

    /// Runtime writes through small owned arrays should use scalarizable
    /// constant element addresses rather than one dynamic GEP.
    #[kernel]
    pub fn runtime_aggregate_array_write(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let index = tid.get();
        let t = index as f32;
        if let Some(out_elem) = out.get_mut(tid) {
            let zero = ScalarPair {
                values: [0.0; 2],
                padding: [0; 24],
            };
            let mut components = [zero; 3];
            components[index % components.len()] = ScalarPair {
                values: [t, t + 3.0],
                padding: [0; 24],
            };
            *out_elem = components[0].values[0]
                + components[0].values[1]
                + components[1].values[0]
                + components[1].values[1]
                + components[2].values[0]
                + components[2].values[1];
        }
    }

    /// `array::map` uses `[MaybeUninit<U>; N]` internally. Keep that union
    /// usable when `U` has stronger alignment than NVPTX's scalar types.
    #[kernel]
    pub fn map_over_aligned_array(mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        let t = tid.get() as f32;
        if let Some(out_elem) = out.get_mut(tid) {
            let components = [0.0_f32, 1.0, 2.0].map(|offset| ScalarPair {
                values: [t + offset, t + offset + 3.0],
                padding: [0; 24],
            });
            *out_elem = components[0].values[0]
                + components[0].values[1]
                + components[1].values[0]
                + components[1].values[1]
                + components[2].values[0]
                + components[2].values[1];
        }
    }

    /// Runtime indexing of small array fields through a shared aggregate
    /// reference should expose constant candidate loads for caller SROA.
    #[kernel]
    pub fn runtime_borrowed_array_field(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let index = tid.get();
        let t = index as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let grid = GridShape {
                counts: [t, t + 1, t + 2],
                periodic: [false, true, false],
            };
            *out_elem = select_grid_field(&grid, index % 3);
        }
    }

    /// Associated-const switch with a dead panic arm (issue-384 shape) must
    /// compile and take the instantiated edge on the device.
    #[kernel]
    pub fn associated_const_lane_sum(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let t = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let values = [t, t + 1, t + 2, t + 3];
            *out_elem = lane_sum::<TwoLanes>(&values);
        }
    }
}

fn kernel_body<'a>(ptx: &'a str, kernel_prefix: &str) -> &'a str {
    let marker = format!(".entry {kernel_prefix}(");
    let entry = ptx
        .find(&marker)
        .unwrap_or_else(|| panic!("missing PTX entry with prefix {kernel_prefix}"));
    let body_start = ptx[entry..]
        .find('{')
        .map(|offset| entry + offset)
        .expect("kernel entry has a body");
    let mut depth = 0_u32;
    for (offset, byte) in ptx.as_bytes()[body_start..].iter().enumerate() {
        match byte {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return &ptx[body_start..=body_start + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated PTX body for {kernel_prefix}");
}

fn main() {
    println!("=== array_for_loop regression (issue #138) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/array_for_loop.ptx");
    let module = ctx
        .load_module_from_file(ptx_path)
        .expect("Failed to load PTX");
    let module = kernels::from_module(module).expect("Failed to initialize typed module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 32;
    const N: usize = BLOCK as usize;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_u32 = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches both kernels' indexing model and
    // the 32-element output allocations.
    unsafe { module.sum_u32_array(stream.as_ref(), cfg, &mut d_u32) }
        .expect("launch sum_u32_array");
    let got_u32 = d_u32.to_host_vec(&stream).unwrap();

    let mut d_pts = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.sum_point_array(stream.as_ref(), cfg, &mut d_pts) }
        .expect("launch sum_point_array");
    let got_pts = d_pts.to_host_vec(&stream).unwrap();

    let mut d_runtime_read = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.runtime_aggregate_array_read(stream.as_ref(), cfg, &mut d_runtime_read) }
        .expect("launch runtime_aggregate_array_read");
    let got_runtime_read = d_runtime_read.to_host_vec(&stream).unwrap();

    let mut d_runtime_write = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.runtime_aggregate_array_write(stream.as_ref(), cfg, &mut d_runtime_write) }
        .expect("launch runtime_aggregate_array_write");
    let got_runtime_write = d_runtime_write.to_host_vec(&stream).unwrap();

    let mut d_aligned_map = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.map_over_aligned_array(stream.as_ref(), cfg, &mut d_aligned_map) }
        .expect("launch map_over_aligned_array");
    let got_aligned_map = d_aligned_map.to_host_vec(&stream).unwrap();

    let mut d_borrowed_field = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.runtime_borrowed_array_field(stream.as_ref(), cfg, &mut d_borrowed_field) }
        .expect("launch runtime_borrowed_array_field");
    let got_borrowed_field = d_borrowed_field.to_host_vec(&stream).unwrap();

    let mut d_lane_sum = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: the 32-thread 1D block matches the kernel's indexing model and
    // the 32-element output allocation.
    unsafe { module.associated_const_lane_sum(stream.as_ref(), cfg, &mut d_lane_sum) }
        .expect("launch associated_const_lane_sum");
    let got_lane_sum = d_lane_sum.to_host_vec(&stream).unwrap();

    let ptx = std::fs::read_to_string(ptx_path).expect("read generated PTX");
    for kernel in [
        "runtime_aggregate_array_read",
        "runtime_aggregate_array_write",
        "map_over_aligned_array",
        "runtime_borrowed_array_field",
    ] {
        let body = kernel_body(&ptx, kernel);
        assert!(
            !body.contains(".local") && !body.contains("ld.local") && !body.contains("st.local"),
            "small runtime aggregate indexing must not use local memory:\n{body}"
        );
    }

    let mut failures = 0usize;
    for tid in 0..N {
        let t = tid as u32;
        // sum of [t, t+1, t+2, t+3]
        let want_u32 = 4 * t + 6;
        // t*1 + (t+1)*2 + (t+2)*3 + (t+3)*4
        let want_pts = t + (t + 1) * 2 + (t + 2) * 3 + (t + 3) * 4;
        if got_u32[tid] != want_u32 {
            println!(
                "FAIL tid={tid}: sum_u32_array={} expected={want_u32}",
                got_u32[tid]
            );
            failures += 1;
        }
        if got_pts[tid] != want_pts {
            println!(
                "FAIL tid={tid}: sum_point_array={} expected={want_pts}",
                got_pts[tid]
            );
            failures += 1;
        }
        let t = t as f32;
        let component = tid % 3;
        let want_runtime_read = 2.0 * t + 3.0 + 2.0 * component as f32;
        if (got_runtime_read[tid] - want_runtime_read).abs() > 1.0e-4 {
            println!(
                "FAIL tid={tid}: runtime_aggregate_array_read={} expected={want_runtime_read}",
                got_runtime_read[tid]
            );
            failures += 1;
        }
        let want_runtime_write = 2.0 * t + 3.0;
        if (got_runtime_write[tid] - want_runtime_write).abs() > 1.0e-4 {
            println!(
                "FAIL tid={tid}: runtime_aggregate_array_write={} expected={want_runtime_write}",
                got_runtime_write[tid]
            );
            failures += 1;
        }
        let want_aligned_map = 6.0 * t + 15.0;
        if (got_aligned_map[tid] - want_aligned_map).abs() > 1.0e-4 {
            println!(
                "FAIL tid={tid}: map_over_aligned_array={} expected={want_aligned_map}",
                got_aligned_map[tid]
            );
            failures += 1;
        }
        let component = tid % 3;
        let want_borrowed_field = tid as u32 + component as u32 + u32::from(component == 1) * 10;
        if got_borrowed_field[tid] != want_borrowed_field {
            println!(
                "FAIL tid={tid}: runtime_borrowed_array_field={} expected={want_borrowed_field}",
                got_borrowed_field[tid]
            );
            failures += 1;
        }
        // lane_sum::<TwoLanes> adds elements 0 and 1: t + (t + 1)
        let want_lane_sum = 2 * tid as u32 + 1;
        if got_lane_sum[tid] != want_lane_sum {
            println!(
                "FAIL tid={tid}: associated_const_lane_sum={} expected={want_lane_sum}",
                got_lane_sum[tid]
            );
            failures += 1;
        }
    }

    if failures == 0 {
        println!("array_for_loop: PASS ({N} threads, array iteration/indexing is correct)");
    } else {
        println!("array_for_loop: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
