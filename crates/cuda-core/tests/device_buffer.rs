/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer};

#[test]
fn device_buffer_from_host_roundtrip() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let data = [1_u32, 2, 3, 4, 5];
    let dev_buf =
        DeviceBuffer::from_host(&stream, &data).expect("failed to allocate DeviceBuffer from host");

    assert_eq!(dev_buf.len(), 5);
    assert_eq!(dev_buf.num_bytes(), 20);
    assert!(!dev_buf.is_empty());

    let host_vec = dev_buf
        .to_host_vec(&stream)
        .expect("failed to copy back to host");
    assert_eq!(host_vec, data);
}

#[test]
fn device_buffer_zeroed_initializes_with_zeros() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let dev_buf =
        DeviceBuffer::<f32>::zeroed(&stream, 4).expect("failed to allocate zeroed DeviceBuffer");

    assert_eq!(dev_buf.len(), 4);
    assert_eq!(dev_buf.num_bytes(), 16);

    let host_vec = dev_buf
        .to_host_vec(&stream)
        .expect("failed to copy back to host");
    assert_eq!(host_vec, &[0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn device_buffer_supports_empty_allocations() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let dev_buf =
        DeviceBuffer::<u8>::zeroed(&stream, 0).expect("failed to allocate empty device buffer");
    assert_eq!(dev_buf.len(), 0);
    assert_eq!(dev_buf.num_bytes(), 0);
    assert!(dev_buf.is_empty());

    let dev_buf_host = DeviceBuffer::<u8>::from_host(&stream, &[])
        .expect("failed to allocate empty device buffer from empty slice");
    assert_eq!(dev_buf_host.len(), 0);
    assert_eq!(dev_buf_host.num_bytes(), 0);
    assert!(dev_buf_host.is_empty());
}

#[test]
fn device_buffer_async_compat_methods_roundtrip() {
    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.new_stream().expect("failed to create CUDA stream");

    let data = [7_u32, 11, 13, 17];
    let mut dev = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, data.len()) }
        .expect("failed to allocate uninitialized device buffer");
    dev.copy_from_host_async(&data, &stream)
        .expect("failed to copy host data into device buffer");

    let mut clone = unsafe { DeviceBuffer::<u32>::uninitialized_async(&stream, data.len()) }
        .expect("failed to allocate clone device buffer");
    clone
        .copy_from_device_async(&dev, &stream)
        .expect("failed to copy device buffer");
    assert_eq!(
        clone
            .to_host_vec(&stream)
            .expect("failed to copy clone back to host"),
        data
    );

    clone
        .zero_async(&stream)
        .expect("failed to zero device buffer");
    assert_eq!(
        clone
            .to_host_vec(&stream)
            .expect("failed to copy zeroed buffer back to host"),
        [0, 0, 0, 0]
    );

    clone
        .drop_async(&stream)
        .expect("failed to async free clone");
    dev.drop_async(&stream)
        .expect("failed to async free source");

    let empty = unsafe { DeviceBuffer::<u8>::uninitialized_async(&stream, 0) }
        .expect("failed to allocate empty uninitialized device buffer");
    empty
        .drop_async(&stream)
        .expect("failed to async free empty buffer");
    stream.synchronize().expect("stream sync failed");
}
