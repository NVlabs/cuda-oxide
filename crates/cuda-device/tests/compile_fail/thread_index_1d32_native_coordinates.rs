/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::thread::{KernelScopeRef, __internal};

fn missing_u32_coordinates<'kernel>(
    scope: KernelScopeRef<'kernel, __internal::Domain1, __internal::NativeCoordinates>,
) {
    let _ = __internal::index_1d32(scope);
}

fn main() {}
