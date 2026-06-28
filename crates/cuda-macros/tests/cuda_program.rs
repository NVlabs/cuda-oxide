// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-time coverage for `#[cuda_program]`.

#[test]
fn cuda_program_metadata_expands() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/cuda_program_metadata.rs");
}

#[test]
fn cuda_program_compile_failures() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/cuda_program_arg_count.rs");
    t.compile_fail("tests/compile_fail/cuda_program_non_kernel.rs");
}
