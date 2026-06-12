/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Value names, block labels, and symbol normalization.
//!
//! Device-symbol detection and base-name extraction route through
//! `reserved-oxide-symbols`, the workspace-internal source of truth for the
//! `cuda_oxide_*` namespace.
//!
//! Note on FQDN forms: MIR import converts `::` to `__`, so a fully-qualified
//! device symbol can appear as `mycrate__cuda_oxide_device_<hash>_foo`. Because
//! the helpers in `reserved-oxide-symbols` use substring matching (not
//! `starts_with`), they handle both bare and FQDN forms uniformly — no separate
//! `FQDN_DEVICE_PREFIX` constant is needed.

use reserved_oxide_symbols::{device_base_name, is_device_extern_symbol, is_device_symbol};

/// Returns true if `name` is a device function (definition, not extern).
pub(super) fn has_device_prefix(name: &str) -> bool {
    is_device_symbol(name)
}

/// Strip the device-function prefix from `name` if present.
///
/// The reserved prefix is needed internally for MIR-level detection but
/// should not leak into the final LLVM IR / PTX / LTOIR output. Returns
/// `name` unchanged for non-device symbols and for device-extern declarations
/// (those keep their original-name `link_name` attribute).
pub(super) fn strip_device_prefix(name: &str) -> String {
    if is_device_extern_symbol(name) {
        return name.to_string();
    }
    device_base_name(name)
        .map(str::to_string)
        .unwrap_or_else(|| name.to_string())
}
