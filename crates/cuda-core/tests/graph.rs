/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for CUDA graph capture, instantiation, and launch.
//!
//! These tests share the same CUDA primary context (device 0), so they must
//! run serially. Use `cargo test --test graph -- --test-threads=1`.

use cuda_core::{CaptureMode, CudaContext, CudaGraph, CudaStreamCaptureExt, IntoResult};
use std::sync::Arc;

fn make_ctx() -> Arc<CudaContext> {
    CudaContext::new(0).expect("failed to create CUDA context")
}

// ---------------------------------------------------------------------------
// Graph lifecycle (no kernel capture)
// ---------------------------------------------------------------------------

#[test]
fn create_empty_graph_succeeds() {
    let ctx = make_ctx();
    let graph = CudaGraph::new(&ctx).expect("failed to create empty graph");
    assert!(!graph.cu_graph().is_null());
}

#[test]
fn instantiate_empty_graph_succeeds() {
    let ctx = make_ctx();
    let graph = CudaGraph::new(&ctx).expect("failed to create empty graph");
    let exec = graph
        .instantiate()
        .expect("failed to instantiate empty graph");
    assert!(!exec.cu_graph_exec().is_null());
}

#[test]
fn graph_drop_destroys_handle() {
    let ctx = make_ctx();
    let graph = CudaGraph::new(&ctx).expect("failed to create graph");
    let raw = graph.cu_graph();
    assert!(!raw.is_null());
    drop(graph);
    assert!(ctx.check_err().is_ok());
}

#[test]
fn exec_drop_destroys_handle() {
    let ctx = make_ctx();
    let graph = CudaGraph::new(&ctx).expect("failed to create graph");
    let exec = graph.instantiate().expect("failed to instantiate");
    let raw = exec.cu_graph_exec();
    assert!(!raw.is_null());
    drop(exec);
    drop(graph);
    assert!(ctx.check_err().is_ok());
}

// ---------------------------------------------------------------------------
// Stream capture API
// ---------------------------------------------------------------------------

#[test]
fn stream_begin_capture_rejects_default_stream() {
    let ctx = make_ctx();
    let _err = ctx
        .default_stream()
        .begin_capture(CaptureMode::Global)
        .expect_err("default stream must reject capture");
}

#[test]
fn stream_is_not_capturing_by_default() {
    let ctx = make_ctx();
    let stream = ctx.new_stream().expect("failed to create stream");
    assert!(!stream.is_capturing().expect("is_capturing failed"));
}

#[test]
fn stream_begin_capture_succeeds_on_new_stream() {
    let ctx = make_ctx();
    let stream = ctx.new_stream().expect("failed to create stream");

    stream
        .begin_capture(CaptureMode::Relaxed)
        .expect("begin_capture should succeed");
    assert!(stream.is_capturing().expect("is_capturing failed"));

    let graph = stream.end_capture().expect("end_capture should succeed");
    assert!(!graph.cu_graph().is_null());
}

#[test]
fn stream_end_capture_without_begin_fails() {
    let ctx = make_ctx();
    let _err = ctx
        .new_stream()
        .expect("failed to create stream")
        .end_capture()
        .expect_err("end_capture without begin should fail");
}

// ---------------------------------------------------------------------------
// End-to-end: capture, instantiate, launch
// ---------------------------------------------------------------------------

fn capture_memset_graph(ctx: &Arc<CudaContext>) -> (CudaGraph, cuda_core::sys::CUdeviceptr) {
    let stream = ctx.new_stream().expect("failed to create stream");
    stream
        .begin_capture(CaptureMode::Relaxed)
        .expect("begin_capture failed");

    let mut dev_ptr: cuda_core::sys::CUdeviceptr = 0;
    unsafe {
        cuda_core::sys::cuMemAlloc_v2(&mut dev_ptr, 16)
            .result()
            .expect("cuMemAlloc failed");
        cuda_core::sys::cuMemsetD32_v2(dev_ptr, 0, 4)
            .result()
            .expect("cuMemsetD32 failed");
    }

    let graph = stream.end_capture().expect("end_capture failed");
    (graph, dev_ptr)
}

#[test]
fn captured_graph_launches_and_syncs() {
    let ctx = make_ctx();
    let stream = ctx.new_stream().expect("failed to create stream");
    let (graph, dev_ptr) = capture_memset_graph(&ctx);

    let exec = graph.instantiate().expect("instantiate failed");
    exec.launch(&stream).expect("launch failed");
    stream.synchronize().expect("synchronize failed");

    unsafe {
        cuda_core::sys::cuMemFree_v2(dev_ptr)
            .result()
            .expect("cuMemFree failed");
    }
}

#[test]
fn graph_upload_and_launch() {
    let ctx = make_ctx();
    let stream = ctx.new_stream().expect("failed to create stream");
    let (graph, dev_ptr) = capture_memset_graph(&ctx);

    let exec = graph.instantiate().expect("instantiate failed");
    exec.upload(&stream).expect("upload failed");
    exec.launch(&stream).expect("launch failed");
    stream.synchronize().expect("synchronize failed");

    unsafe {
        cuda_core::sys::cuMemFree_v2(dev_ptr)
            .result()
            .expect("cuMemFree failed");
    }
}

#[test]
fn graph_repeat_launch() {
    let ctx = make_ctx();
    let stream = ctx.new_stream().expect("failed to create stream");
    let (graph, dev_ptr) = capture_memset_graph(&ctx);

    let exec = graph.instantiate().expect("instantiate failed");
    for _ in 0..5 {
        exec.launch(&stream).expect("launch failed");
    }
    stream.synchronize().expect("synchronize failed");

    unsafe {
        cuda_core::sys::cuMemFree_v2(dev_ptr)
            .result()
            .expect("cuMemFree failed");
    }
}
