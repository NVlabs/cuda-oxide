// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// trybuild's generated crate does not mirror feature flags onto the separate
// cuda-host dev dependency. Async expansion is covered by the macro unit tests
// and cuda-host integration tests instead.
#[cfg(not(feature = "async"))]
#[test]
fn launch_contract_types_are_resolved_semantically() {
    let t = trybuild::TestCases::new();
    t.pass("tests/pass/launch_contract_disjoint_aliases.rs");
    t.compile_fail("tests/compile_fail/launch_contract_misleading_index_alias.rs");
    t.compile_fail("tests/compile_fail/launch_contract_fake_disjoint_slice.rs");
    t.compile_fail("tests/compile_fail/launch_contract_untrusted_loaders.rs");
    t.compile_fail("tests/compile_fail/launch_contract_wrong_const_brand.rs");
    t.compile_fail("tests/compile_fail/launch_contract_reordered_disjoint_alias.rs");
}
