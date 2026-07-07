/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Integration tests for CUDA graph capture, instantiation, and launch.
//!
//! Each test creates a dedicated CUDA context via [`CudaContext::new_dedicated`]
//! so that stream capture and graph operations are fully isolated. Tests can
//! run in parallel without `--test-threads=1`.

use cuda_core::{
    CaptureMode, CudaContext, CudaGraph, CudaStreamCaptureExt, GraphUpdateResult, IntoResult,
    KernelNodeParams,
};
use std::sync::Arc;

fn make_ctx() -> Arc<CudaContext> {
    CudaContext::new_dedicated(0).expect("failed to create dedicated CUDA context")
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
// Manual node construction
// ---------------------------------------------------------------------------

#[test]
fn add_empty_node_succeeds() {
    let ctx = make_ctx();
    let graph = CudaGraph::new(&ctx).expect("failed to create graph");

    let node = graph.add_empty_node(&[]).expect("failed to add empty node");
    assert!(!node.cu_node().is_null());

    // The graph should still be instantiable with an empty node.
    let exec = graph.instantiate().expect("failed to instantiate");
    drop(exec);
    drop(node);
}

#[test]
fn add_empty_node_with_dependency() {
    let ctx = make_ctx();
    let graph = CudaGraph::new(&ctx).expect("failed to create graph");

    let a = graph.add_empty_node(&[]).expect("node a");
    let b = graph.add_empty_node(&[&a]).expect("node b depends on a");
    assert!(!b.cu_node().is_null());
    drop(b);
    drop(a);
}

#[test]
fn add_dependencies_between_nodes() {
    let ctx = make_ctx();
    let graph = CudaGraph::new(&ctx).expect("failed to create graph");

    let a = graph.add_empty_node(&[]).expect("node a");
    let b = graph.add_empty_node(&[]).expect("node b");

    // Add a dependency edge: b waits for a.
    graph
        .add_dependencies(&[&a], &[&b])
        .expect("add_dependencies failed");

    let exec = graph.instantiate().expect("failed to instantiate");
    drop(exec);
    drop(b);
    drop(a);
}

#[test]
fn add_dependencies_noop_on_empty_input() {
    let ctx = make_ctx();
    let graph = CudaGraph::new(&ctx).expect("failed to create graph");
    // Empty dependencies should silently succeed.
    graph
        .add_dependencies(&[], &[])
        .expect("empty add_dependencies should succeed");
}

#[test]
fn node_drop_removes_from_graph() {
    let ctx = make_ctx();
    let graph = CudaGraph::new(&ctx).expect("failed to create graph");

    let node = graph.add_empty_node(&[]).expect("node");
    let raw = node.cu_node();
    assert!(!raw.is_null());
    drop(node);
    // After drop, graph should still be valid (node was removed).
    assert!(ctx.check_err().is_ok());
}

// ---------------------------------------------------------------------------
// KernelNodeParams
// ---------------------------------------------------------------------------

#[test]
fn kernel_node_params_push_and_to_raw() {
    let mut params = KernelNodeParams::new(std::ptr::null_mut(), (1, 1, 1), (32, 1, 1), 0);

    let a: u32 = 42;
    let b: f32 = 3.14;
    params.push_param(&a);
    params.push_param(&b);

    let (ptrs, buf) = params.to_raw();
    assert_eq!(ptrs.len(), 2, "two params");
    assert_eq!(buf.len(), 8, "u32 + f32 = 8 bytes");
}

// ---------------------------------------------------------------------------
// Graph update (topology-only; no kernel needed)
// ---------------------------------------------------------------------------

#[test]
fn graph_update_succeeds_for_identical_topology() {
    let ctx = make_ctx();

    // Build a graph with one empty node.
    let g1 = CudaGraph::new(&ctx).expect("g1");
    g1.add_empty_node(&[]).expect("node");

    // Build an identical graph.
    let g2 = CudaGraph::new(&ctx).expect("g2");
    g2.add_empty_node(&[]).expect("node");

    let exec = g1.instantiate().expect("instantiate g1");
    let result = exec.update(&g2).expect("update should succeed");
    assert_eq!(result, GraphUpdateResult::Success);
}

#[test]
fn graph_update_accepts_or_rejects_changed_graph() {
    let ctx = make_ctx();

    let g1 = CudaGraph::new(&ctx).expect("g1");
    g1.add_empty_node(&[]).expect("single node");

    let g2 = CudaGraph::new(&ctx).expect("g2");
    let a = g2.add_empty_node(&[]).expect("node a");
    g2.add_empty_node(&[&a]).expect("node b");
    drop(a);

    let exec = g1.instantiate().expect("instantiate g1");
    // The driver may accept or reject this depending on version;
    // we just verify it doesn't crash.
    let _ = exec.update(&g2);
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
