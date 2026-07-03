// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compile-fail tests for `#[kernel]`, `#[device]`, and low-level launch API
//! contracts. These keep invalid signatures and reserved names on clear macro
//! diagnostics instead of allowing confusing generated-code failures.

#[test]
fn macro_guards() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/const_generic_hygiene.rs");
    t.pass("tests/pass/cuda_module_inline_namespaces.rs");
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
    t.compile_fail("tests/compile_fail/cuda_module_duplicate_nested_kernel.rs");
    t.compile_fail("tests/compile_fail/cuda_module_raw_duplicate_kernel.rs");
    t.compile_fail("tests/compile_fail/cuda_module_raw_loaded_module.rs");
    t.compile_fail("tests/compile_fail/cuda_module_reserved_from_parent.rs");
    t.compile_fail("tests/compile_fail/cuda_module_nested_type_mismatch.rs");
    t.compile_fail("tests/compile_fail/cuda_module_pub_super_scope.rs");
    t.compile_fail("tests/compile_fail/cuda_module_file_kernel_boundary.rs");
    t.compile_fail("tests/compile_fail/cuda_module_include_kernel_boundary.rs");
}

/// `cuda_launch!` is a caller-unsafe API: its expansion calls the unsafe
/// `cuda_core` launch functions without an internal `unsafe { }`, so a bare
/// invocation must fail to compile with an unsafe-required error (E0133).
#[test]
fn cuda_launch_requires_unsafe() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/launch_requires_unsafe.rs");
}
