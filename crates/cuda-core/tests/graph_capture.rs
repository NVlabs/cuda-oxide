/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hardware tests for `cuda_core::graph`, written to falsify the claims the
//! module's design rests on rather than to demonstrate that it works.
//!
//! These share the device's *primary* context, so an invalidated capture in one
//! test is observable in another. Run single-threaded:
//!
//! ```text
//! cargo test -p cuda-core --test graph_capture -- --test-threads=1
//! ```

use cuda_core::graph::{CaptureMode, CaptureStatus};
use cuda_core::{CudaContext, DeviceBuffer};

fn context() -> std::sync::Arc<CudaContext> {
    CudaContext::new(0).expect("failed to create CUDA context")
}

/// A device-to-device copy is capturable, so the whole capture → instantiate →
/// replay path can be exercised without a device kernel.
#[test]
fn captures_and_replays_a_device_copy() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");

    let src = DeviceBuffer::from_host(&stream, &[7u32, 8, 9, 10]).expect("src");
    let mut dst = DeviceBuffer::<u32>::zeroed(&stream, 4).expect("dst");
    stream.synchronize().expect("sync");

    let capture = stream.begin_capture(CaptureMode::Global).expect("begin");
    dst.copy_from_device_async(&src, &stream)
        .expect("record copy");
    let graph = capture.end().expect("end");

    assert_eq!(graph.node_count().expect("nodes"), 1, "one copy, one node");

    // The capture recorded the copy instead of performing it.
    let mut host = [0u32; 4];
    dst.copy_to_host(&stream, &mut host).expect("read back");
    stream.synchronize().expect("sync");
    assert_eq!(host, [0, 0, 0, 0], "capture must not execute the work");

    let mut exec = unsafe { graph.instantiate() }.expect("instantiate");
    exec.launch(&stream).expect("replay");
    stream.synchronize().expect("sync");
    dst.copy_to_host(&stream, &mut host).expect("read back");
    stream.synchronize().expect("sync");
    assert_eq!(host, [7, 8, 9, 10], "replay must perform the copy");

    // Replay is repeatable from the same exec.
    dst.copy_from_host(&stream, &[0, 0, 0, 0]).expect("reset");
    stream.synchronize().expect("sync");
    exec.launch(&stream).expect("second replay");
    stream.synchronize().expect("sync");
    dst.copy_to_host(&stream, &mut host).expect("read back");
    stream.synchronize().expect("sync");
    assert_eq!(host, [7, 8, 9, 10], "exec is reusable");
}

#[test]
fn capture_status_reports_the_transitions() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");
    assert_eq!(stream.capture_status().expect("idle"), CaptureStatus::None);

    let capture = stream.begin_capture(CaptureMode::Global).expect("begin");
    assert_eq!(
        stream.capture_status().expect("active"),
        CaptureStatus::Active,
        "a stream must report Active between begin and end"
    );
    let graph = capture.end().expect("end");
    assert_eq!(
        stream.capture_status().expect("ended"),
        CaptureStatus::None,
        "ending must return the stream to the idle state"
    );
    drop(graph);
}

/// The claim the RAII guard is justified by: an unterminated capture leaves the
/// stream unusable, so `Drop` must terminate it.
///
/// This test does not assert the *hazard* (that would require leaking a guard);
/// it asserts the guard's remedy — after dropping a capture without calling
/// `end`, the stream is idle and still works.
#[test]
fn dropping_the_guard_leaves_the_stream_usable() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");
    let mut buf = DeviceBuffer::<u32>::zeroed(&stream, 2).expect("buf");

    {
        let capture = stream.begin_capture(CaptureMode::Global).expect("begin");
        buf.zero_async(&stream).expect("record a zero");
        drop(capture); // no `end` -- the guard must still terminate the capture
    }

    assert_eq!(
        stream.capture_status().expect("status"),
        CaptureStatus::None,
        "Drop must terminate the capture, not leave it Active"
    );

    // And the stream must still be usable for ordinary work.
    buf.copy_from_host(&stream, &[3, 4])
        .expect("post-drop write");
    let mut host = [0u32; 2];
    buf.copy_to_host(&stream, &mut host)
        .expect("post-drop read");
    stream.synchronize().expect("sync");
    assert_eq!(host, [3, 4], "stream must survive an abandoned capture");
}

/// Whether a *second* capture can begin on a stream whose previous capture was
/// abandoned. If `Drop` did not terminate properly this would fail, so it is a
/// second, independent probe of the same property.
#[test]
fn a_stream_can_be_recaptured_after_an_abandoned_capture() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");
    let mut buf = DeviceBuffer::<u32>::zeroed(&stream, 2).expect("buf");

    drop(
        stream
            .begin_capture(CaptureMode::Global)
            .expect("first begin"),
    );

    let capture = stream
        .begin_capture(CaptureMode::Global)
        .expect("second capture must be permitted after an abandoned one");
    buf.zero_async(&stream).expect("record");
    let graph = capture.end().expect("end");
    assert_eq!(graph.node_count().expect("nodes"), 1);
}

/// Probes the hazard the module documents but cannot prevent: a graph outliving
/// a buffer it captured.
///
/// Ignored by default — it is deliberately unsound, and its purpose is to record
/// *how* the unsoundness presents, not to assert a passing property. Run with:
///
/// ```text
/// cargo test -p cuda-core --test graph_capture -- --ignored --nocapture
/// ```
#[test]
#[ignore = "deliberately unsound; run manually to observe the failure mode"]
fn replay_after_freeing_a_captured_buffer_is_not_diagnosed() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");
    let mut dst = DeviceBuffer::<u32>::zeroed(&stream, 4).expect("dst");

    let mut exec = {
        let src = DeviceBuffer::from_host(&stream, &[7u32, 8, 9, 10]).expect("src");
        stream.synchronize().expect("sync");
        let capture = stream.begin_capture(CaptureMode::Global).expect("begin");
        dst.copy_from_device_async(&src, &stream).expect("record");
        let graph = capture.end().expect("end");
        unsafe { graph.instantiate() }.expect("instantiate")
        // `src` is dropped here while `exec` still references its address.
    };

    let launch = exec.launch(&stream);
    let sync = stream.synchronize();
    let mut host = [0u32; 4];
    let read = dst
        .copy_to_host(&stream, &mut host)
        .and_then(|()| stream.synchronize());

    println!("launch  = {launch:?}");
    println!("sync    = {sync:?}");
    println!("read    = {read:?}");
    println!("dst     = {host:?}   (captured source was [7, 8, 9, 10])");
    println!(
        "diagnosed = {}",
        launch.is_err() || sync.is_err() || read.is_err()
    );
}
