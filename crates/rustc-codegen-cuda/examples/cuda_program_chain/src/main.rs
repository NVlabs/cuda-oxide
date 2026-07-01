/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Multi-stage semantic CUDA program graph example.
//!
//! Build and run with:
//!   cargo oxide run cuda_program_chain

use cuda_core::{CudaContext, DeviceBuffer};
use cuda_async::device_operation::DeviceOperation;
use cuda_device::{DisjointSlice, cuda_program, kernel, thread};
use cuda_host::{ProgramArgumentRole, ProgramLowering, ProgramResourceRole};
use std::time::Instant;

#[cuda_program]
mod kernels {
    use super::*;

    #[kernel]
    pub fn affine_mix(
        a: &[f32],
        b: &[f32],
        mut tmp0: DisjointSlice<f32>,
        alpha: f32,
        beta: f32,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = tmp0.get_mut(idx) {
            *out = alpha * a[i] + beta * b[i];
        }
    }

    #[kernel]
    pub fn relu_bias(input: &[f32], mut tmp1: DisjointSlice<f32>, bias: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = tmp1.get_mut(idx) {
            let value = input[i] + bias;
            *out = if value > 0.0 { value } else { 0.0 };
        }
    }

    #[kernel]
    pub fn add_third(input: &[f32], c: &[f32], mut tmp2: DisjointSlice<f32>, gamma: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = tmp2.get_mut(idx) {
            *out = input[i] + gamma * c[i];
        }
    }

    #[kernel]
    pub fn residual(input: &[f32], a: &[f32], mut tmp3: DisjointSlice<f32>, delta: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = tmp3.get_mut(idx) {
            *out = input[i] + delta * a[i];
        }
    }

    #[kernel]
    pub fn clamp(input: &[f32], mut output: DisjointSlice<f32>, lo: f32, hi: f32) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = output.get_mut(idx) {
            let value = input[i];
            let value = if value < lo { lo } else { value };
            *out = if value > hi { hi } else { value };
        }
    }

    #[program]
    pub fn forward(
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        c: &DeviceBuffer<f32>,
        tmp0: &mut DeviceBuffer<f32>,
        tmp1: &mut DeviceBuffer<f32>,
        tmp2: &mut DeviceBuffer<f32>,
        tmp3: &mut DeviceBuffer<f32>,
        output: &mut DeviceBuffer<f32>,
        alpha: f32,
        beta: f32,
        bias: f32,
        gamma: f32,
        delta: f32,
        lo: f32,
        hi: f32,
        n: u32,
    ) {
        affine_mix(a, b, tmp0, alpha, beta).grid_len(n);
        relu_bias(tmp0, tmp1, bias).grid_len(n);
        add_third(tmp1, c, tmp2, gamma).grid_len(n);
        residual(tmp2, a, tmp3, delta).grid_len(n);
        clamp(tmp3, output, lo, hi).grid_len(n);
    }
}

fn main() {
    println!("=== CUDA Program Chain Example ===");

    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 4096;
    let alpha = 1.25f32;
    let beta = -0.5f32;
    let bias = 0.75f32;
    let gamma = 0.2f32;
    let delta = -0.1f32;
    let lo = -8.0f32;
    let hi = 24.0f32;

    let a_host: Vec<f32> = (0..N).map(|i| (i % 97) as f32 * 0.25 - 8.0).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i % 53) as f32 * 0.5 - 10.0).collect();
    let c_host: Vec<f32> = (0..N).map(|i| (i % 31) as f32 * 0.125 + 1.0).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let c_dev = DeviceBuffer::from_host(&stream, &c_host).unwrap();
    let mut tmp0_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut tmp1_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut tmp2_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut tmp3_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("failed to load CUDA program module");
    let graph = kernels::forward_graph(
        &a_dev,
        &b_dev,
        &c_dev,
        &mut tmp0_dev,
        &mut tmp1_dev,
        &mut tmp2_dev,
        &mut tmp3_dev,
        &mut output_dev,
        alpha,
        beta,
        bias,
        gamma,
        delta,
        lo,
        hi,
        N as u32,
    );

    let metadata = graph.metadata();
    assert_eq!(metadata.operations.len(), 5);
    assert_eq!(graph.dependencies().len(), 4);
    assert_eq!(graph.dependencies()[0].resource, "tmp0");
    assert_eq!(graph.dependencies()[1].resource, "tmp1");
    assert_eq!(graph.dependencies()[2].resource, "tmp2");
    assert_eq!(graph.dependencies()[3].resource, "tmp3");

    assert_eq!(graph.resources()[0].role, ProgramResourceRole::Input);
    assert_eq!(graph.resources()[3].role, ProgramResourceRole::Scratch);
    assert_eq!(graph.resources()[7].role, ProgramResourceRole::Output);
    assert_eq!(graph.resources()[8].role, ProgramResourceRole::Scalar);
    assert_eq!(graph.operations()[4].arguments[1].role, ProgramArgumentRole::Write);

    let bound = graph
        .bind(&module, ProgramLowering::SequentialLaunches)
        .expect("failed to bind CUDA program graph");
    bound
        .sync_on(&stream)
        .expect("failed to schedule bound program");

    let output_host = output_dev.to_host_vec(&stream).unwrap();
    let mut errors = 0usize;
    for i in 0..N {
        let tmp0 = alpha * a_host[i] + beta * b_host[i];
        let tmp1 = (tmp0 + bias).max(0.0);
        let tmp2 = tmp1 + gamma * c_host[i];
        let tmp3 = tmp2 + delta * a_host[i];
        let expected = tmp3.clamp(lo, hi);
        let got = output_host[i];
        if (got - expected).abs() > 1.0e-4 {
            if errors < 8 {
                eprintln!("mismatch[{i}]: expected {expected}, got {got}");
            }
            errors += 1;
        }
    }

    println!("operations: {:?}", metadata.operations);
    println!("dependencies: {:?}", metadata.dependencies);
    println!("first outputs: {:?}", &output_host[..8]);

    if errors == 0 {
        println!("SUCCESS: complex CUDA program chain matched host reference");
    } else {
        eprintln!("FAILED: {errors} mismatches");
        std::process::exit(1);
    }

    if let Ok(iterations) = std::env::var("BENCH_ITERS") {
        let iterations: usize = iterations.parse().expect("BENCH_ITERS must be a usize");
        if iterations > 0 {
            run_benchmark(
                &module,
                &stream,
                &a_dev,
                &b_dev,
                &c_dev,
                &mut tmp0_dev,
                &mut tmp1_dev,
                &mut tmp2_dev,
                &mut tmp3_dev,
                &mut output_dev,
                alpha,
                beta,
                bias,
                gamma,
                delta,
                lo,
                hi,
                N as u32,
                iterations,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    a_dev: &DeviceBuffer<f32>,
    b_dev: &DeviceBuffer<f32>,
    c_dev: &DeviceBuffer<f32>,
    tmp0_dev: &mut DeviceBuffer<f32>,
    tmp1_dev: &mut DeviceBuffer<f32>,
    tmp2_dev: &mut DeviceBuffer<f32>,
    tmp3_dev: &mut DeviceBuffer<f32>,
    output_dev: &mut DeviceBuffer<f32>,
    alpha: f32,
    beta: f32,
    bias: f32,
    gamma: f32,
    delta: f32,
    lo: f32,
    hi: f32,
    n: u32,
    iterations: usize,
) {
    stream.synchronize().unwrap();
    let direct_start = Instant::now();
    for _ in 0..iterations {
        module
            .affine_mix(stream, cuda_core::LaunchConfig::for_num_elems(n), a_dev, b_dev, tmp0_dev, alpha, beta)
            .unwrap();
        module
            .relu_bias(stream, cuda_core::LaunchConfig::for_num_elems(n), tmp0_dev, tmp1_dev, bias)
            .unwrap();
        module
            .add_third(stream, cuda_core::LaunchConfig::for_num_elems(n), tmp1_dev, c_dev, tmp2_dev, gamma)
            .unwrap();
        module
            .residual(stream, cuda_core::LaunchConfig::for_num_elems(n), tmp2_dev, a_dev, tmp3_dev, delta)
            .unwrap();
        module
            .clamp(stream, cuda_core::LaunchConfig::for_num_elems(n), tmp3_dev, output_dev, lo, hi)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let direct = direct_start.elapsed();

    stream.synchronize().unwrap();
    let graph_start = Instant::now();
    for _ in 0..iterations {
        let graph = kernels::forward_graph(
            a_dev,
            b_dev,
            c_dev,
            tmp0_dev,
            tmp1_dev,
            tmp2_dev,
            tmp3_dev,
            output_dev,
            alpha,
            beta,
            bias,
            gamma,
            delta,
            lo,
            hi,
            n,
        );
        graph
            .bind(module, ProgramLowering::SequentialLaunches)
            .unwrap()
            .launch(stream)
            .unwrap();
    }
    stream.synchronize().unwrap();
    let graph = graph_start.elapsed();

    let launches = iterations * 5;
    println!(
        "benchmark: {iterations} chains / {launches} launches, direct={direct:?}, graph={graph:?}"
    );
    println!(
        "benchmark per chain: direct={:.3} us, graph={:.3} us, delta={:.3} us",
        direct.as_secs_f64() * 1.0e6 / iterations as f64,
        graph.as_secs_f64() * 1.0e6 / iterations as f64,
        (graph.as_secs_f64() - direct.as_secs_f64()) * 1.0e6 / iterations as f64
    );
}
