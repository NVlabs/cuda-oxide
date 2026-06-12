/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Common utils for tests

/// Initialize the logger for tests
pub fn init_env_logger() {
    let _ = env_logger::builder().is_test(true).try_init();
}
