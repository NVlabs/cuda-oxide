/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::thread::{KernelScopeRef, __internal};

fn one_dimensional_scope<'kernel>(
    scope: KernelScopeRef<'kernel, __internal::Domain1, __internal::U32Coordinates>,
) {
    let _ = __internal::coord_2d32(scope);
}

fn main() {}
