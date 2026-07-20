/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#[test]
fn thread_index_safety_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/thread_index_*.rs");
    t.compile_fail("tests/compile_fail/thread_coord_*.rs");
}
