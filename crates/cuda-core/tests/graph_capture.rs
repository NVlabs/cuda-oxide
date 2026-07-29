/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hardware tests for `cuda_core::graph`, written against the public API only.
//!
//! These share the device's *primary* context, so a capture left active in one
//! test would be observable in another. Run single-threaded:
//!
//! ```text
//! cargo test -p cuda-core --test graph_capture -- --test-threads=1
//! ```
//!
//! Note what is *absent*: there is no test for replaying a graph whose captured
//! buffer was freed. That program cannot be written through this API -- the exec
//! owns the buffers, so nothing outside it can name them, let alone drop them.

use cuda_core::graph::{CaptureMode, CaptureStatus};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError};

fn context() -> std::sync::Arc<CudaContext> {
    CudaContext::new(0).expect("failed to create CUDA context")
}

struct Bufs {
    src: DeviceBuffer<u32>,
    dst: DeviceBuffer<u32>,
}

fn bufs(stream: &CudaStream) -> Bufs {
    Bufs {
        src: DeviceBuffer::<u32>::zeroed(stream, 4).expect("src"),
        dst: DeviceBuffer::<u32>::zeroed(stream, 4).expect("dst"),
    }
}

fn abandon() -> DriverError {
    DriverError(cuda_core::sys::cudaError_enum_CUDA_ERROR_INVALID_VALUE)
}

/// A device-to-device copy is capturable, so the whole capture -> replay path can
/// be exercised without a device kernel.
#[test]
fn captures_and_replays_a_device_copy() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");
    let mut state = bufs(&stream);
    state
        .src
        .copy_from_host(&stream, &[7u32, 8, 9, 10])
        .expect("fill src");
    stream.synchronize().expect("sync");

    let mut exec = stream
        .capture_owning(CaptureMode::Global, state, |b| {
            b.dst.copy_from_device_async(&b.src, &stream)
        })
        .expect("capture");

    assert_eq!(exec.node_count(), 1, "one copy, one node");

    // Capture records the work rather than performing it, so `dst` is still zero.
    let before = exec.state().dst.to_host_vec(&stream).expect("read");
    assert_eq!(
        before,
        vec![0, 0, 0, 0],
        "capture must not execute the work"
    );

    exec.launch(&stream).expect("replay");
    stream.synchronize().expect("sync");
    let after = exec.state().dst.to_host_vec(&stream).expect("read");
    assert_eq!(after, vec![7, 8, 9, 10], "replay must perform the copy");
}

/// The loop graphs exist for: capture once, feed new input, replay.
///
/// This is what a borrow-based exec cannot express -- holding a shared borrow of
/// the captured buffers makes writing to them `E0502`. Owning them permits it.
/// The closure needs no `unsafe`: `&mut b.dst` and `&b.src` are disjoint fields.
#[test]
fn write_then_replay_observes_the_new_input() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");
    let mut exec = stream
        .capture_owning(CaptureMode::Global, bufs(&stream), |b| {
            b.dst.copy_from_device_async(&b.src, &stream)
        })
        .expect("capture");

    for i in 0..3u32 {
        exec.state_mut()
            .src
            .copy_from_host(&stream, &[i, i + 1, i + 2, i + 3])
            .expect("feed new input");
        exec.launch(&stream).expect("replay");
        stream.synchronize().expect("sync");
        let got = exec.state().dst.to_host_vec(&stream).expect("read back");
        assert_eq!(
            got,
            vec![i, i + 1, i + 2, i + 3],
            "replay {i} must observe the input written before it"
        );
    }

    // The buffers come back out; the graph dies with the exec.
    let recovered = exec.into_state();
    assert_eq!(recovered.src.len(), 4);
}

/// A `record` closure that fails must leave the stream usable.
///
/// This is the property the RAII capture guard exists for, observed through the
/// public API. The driver requires that even an *invalidated* capture be
/// terminated before the stream can be used again, so an early return between
/// begin and end would otherwise strand it -- and, because a capturing blocking
/// stream also makes the legacy null stream unusable, take that down too.
#[test]
fn a_failed_capture_leaves_the_stream_usable() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");

    let result = stream.capture_owning(CaptureMode::Global, bufs(&stream), |b| {
        b.dst.copy_from_device_async(&b.src, &stream)?;
        Err(abandon())
    });
    assert!(result.is_err(), "the closure's error must propagate");

    assert_eq!(
        stream.capture_status().expect("status"),
        CaptureStatus::None,
        "a failed capture must not leave the stream capturing"
    );

    let mut buf = DeviceBuffer::<u32>::zeroed(&stream, 2).expect("buf");
    buf.copy_from_host(&stream, &[3, 4])
        .expect("post-failure write");
    let got = buf.to_host_vec(&stream).expect("post-failure read");
    assert_eq!(got, vec![3, 4], "stream survives an abandoned capture");
}

/// A second capture must be permitted after a failed one -- an independent probe
/// of the same property, since a stranded capture would reject it.
#[test]
fn a_stream_can_be_recaptured_after_a_failed_capture() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");

    let _ = stream.capture_owning(CaptureMode::Global, bufs(&stream), |_| Err(abandon()));

    let exec = stream
        .capture_owning(CaptureMode::Global, bufs(&stream), |b| {
            b.dst.copy_from_device_async(&b.src, &stream)
        })
        .expect("second capture must be permitted");
    assert_eq!(exec.node_count(), 1);
}

#[test]
fn an_idle_stream_reports_no_capture() {
    let ctx = context();
    let stream = ctx.new_stream().expect("stream");
    assert_eq!(stream.capture_status().expect("idle"), CaptureStatus::None);
}
