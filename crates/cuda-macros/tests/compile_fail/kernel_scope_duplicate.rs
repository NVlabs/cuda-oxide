// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::kernel;

#[kernel(scope = first, scope = second)]
pub fn duplicate() {}

fn main() {}
