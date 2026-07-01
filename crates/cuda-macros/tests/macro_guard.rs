// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail tests for `#[kernel]`, `#[device]`, and low-level launch API
//! contracts. These keep invalid signatures and reserved names on clear macro
//! diagnostics instead of allowing confusing generated-code failures.

#[test]
fn macro_guards() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/kernel_reserved_name.rs");
    t.compile_fail("tests/compile_fail/device_reserved_name.rs");
    t.compile_fail("tests/compile_fail/device_extern_reserved_name.rs");
    t.compile_fail("tests/compile_fail/device_extern_wrong_abi.rs");
    t.compile_fail("tests/compile_fail/kernel_legacy_const_instantiation.rs");
    t.compile_fail("tests/compile_fail/kernel_legacy_lifetime_instantiation.rs");
    t.compile_fail("tests/compile_fail/kernel_impl_trait_parameter.rs");
    t.compile_fail("tests/compile_fail/cuda_module_impl_trait_parameter.rs");
    t.compile_fail("tests/compile_fail/device_impl_trait_parameter.rs");
    t.compile_fail("tests/compile_fail/kernel_instantiation_on_non_generic.rs");
}

/// `cuda_launch!` is a caller-unsafe API: its expansion calls the unsafe
/// `cuda_core` launch functions without an internal `unsafe { }`, so a bare
/// invocation must fail to compile with an unsafe-required error (E0133).
#[test]
fn cuda_launch_requires_unsafe() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/launch_requires_unsafe.rs");
}
