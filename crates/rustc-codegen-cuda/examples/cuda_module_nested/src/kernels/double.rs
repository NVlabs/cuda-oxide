/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Out-of-line body of `pub mod double;` declared inside the #[cuda_module]
// in main.rs. Resolved by module directory: main.rs owns src/, the inline
// `kernels` module owns src/kernels/, so `mod double;` reads this file.

use cuda_device::{DisjointSlice, kernel, thread};

/// out[i] = a[i] + a[i]
#[kernel]
pub fn double_all(a: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    let idx_raw = idx.get();
    if let Some(elem) = out.get_mut(idx) {
        *elem = a[idx_raw] + a[idx_raw];
    }
}
