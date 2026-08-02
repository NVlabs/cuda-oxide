/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for issue #399.
//!
//! Small local arrays consumed through iterator adapters must remain
//! scalarizable after bounded MIR unrolling.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn iter_copied_take(mut output: DisjointSlice<f32>, limit: usize) {
        let index = thread::index_1d();
        if let Some(output) = output.get_mut(index) {
            let values = [4.0_f32, -3.0, 4.0 / 3.0, -0.25];
            let mut result = 0.0_f32;

            for value in values.iter().copied().take(limit.min(values.len())) {
                result += value;
            }

            *output = result;
        }
    }

    #[kernel]
    pub fn iter_copied_skip_take_enumerate(
        mut output: DisjointSlice<f32>,
        offset: usize,
        limit: usize,
    ) {
        let thread_index = thread::index_1d();
        if let Some(output) = output.get_mut(thread_index) {
            let values = [4.0_f32, -3.0, 4.0 / 3.0, -0.25];
            let mut result = 0.0_f32;

            for (index, value) in values
                .iter()
                .copied()
                .skip(offset.min(values.len()))
                .take(limit.min(values.len()))
                .enumerate()
            {
                result += value * (index + 1) as f32;
            }

            *output = result;
        }
    }

    #[kernel]
    pub fn iterator_before_unrelated_loop(
        mut output: DisjointSlice<f32>,
        offset: usize,
        iterations: usize,
    ) {
        let thread_index = thread::index_1d();
        if let Some(output) = output.get_mut(thread_index) {
            let values = [4.0_f32, -3.0, 4.0 / 3.0, -0.25];
            let mut iterator = values.iter().copied().skip(offset.min(values.len()));
            let mut result = iterator.next().unwrap_or(0.0);
            let mut iteration = 0usize;

            while iteration < iterations {
                result += 1.0;
                iteration += 1;
            }

            *output = result;
        }
    }
}

fn main() {
    let context = CudaContext::new(0).expect("create CUDA context");
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/array_iterator_adapters.ptx");
    let module = context.load_module_from_file(ptx_path).expect("load PTX");
    let module = kernels::from_module(module).expect("initialize typed module");
    let stream = context.default_stream();

    let configuration = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    for limit in 0usize..=6 {
        let mut output = DeviceBuffer::<f32>::zeroed(&stream, 1).expect("allocate output");
        unsafe { module.iter_copied_take(stream.as_ref(), configuration, &mut output, limit) }
            .expect("launch iter_copied_take");

        let actual = output.to_host_vec(&stream).expect("copy output")[0];
        let values = [4.0_f32, -3.0, 4.0 / 3.0, -0.25];
        let expected: f32 = values.iter().copied().take(limit.min(values.len())).sum();
        assert!(
            (actual - expected).abs() <= 1.0e-5,
            "limit={limit}: actual={actual}, expected={expected}"
        );
    }

    for offset in 0usize..=6 {
        for limit in 0usize..=6 {
            let mut output = DeviceBuffer::<f32>::zeroed(&stream, 1).expect("allocate output");
            unsafe {
                module.iter_copied_skip_take_enumerate(
                    stream.as_ref(),
                    configuration,
                    &mut output,
                    offset,
                    limit,
                )
            }
            .expect("launch iter_copied_skip_take_enumerate");

            let actual = output.to_host_vec(&stream).expect("copy output")[0];
            let values = [4.0_f32, -3.0, 4.0 / 3.0, -0.25];
            let expected: f32 = values
                .iter()
                .copied()
                .skip(offset.min(values.len()))
                .take(limit.min(values.len()))
                .enumerate()
                .map(|(index, value)| value * (index + 1) as f32)
                .sum();
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "offset={offset}, limit={limit}: actual={actual}, expected={expected}"
            );
        }
    }

    for offset in [0usize, 2, 6] {
        for iterations in [0usize, 4, 7] {
            let mut output = DeviceBuffer::<f32>::zeroed(&stream, 1).expect("allocate output");
            unsafe {
                module.iterator_before_unrelated_loop(
                    stream.as_ref(),
                    configuration,
                    &mut output,
                    offset,
                    iterations,
                )
            }
            .expect("launch iterator_before_unrelated_loop");

            let actual = output.to_host_vec(&stream).expect("copy output")[0];
            let values = [4.0_f32, -3.0, 4.0 / 3.0, -0.25];
            let seed = values
                .iter()
                .copied()
                .skip(offset.min(values.len()))
                .next()
                .unwrap_or(0.0);
            let expected = seed + iterations as f32;
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "offset={offset}, iterations={iterations}: actual={actual}, expected={expected}"
            );
        }
    }

    println!("array_iterator_adapters: PASS");
}
