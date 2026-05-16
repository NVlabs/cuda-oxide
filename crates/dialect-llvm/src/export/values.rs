/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::value::Value;

use super::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_value(
        &self,
        val: Value,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        if let Some(name) = value_names.get(&val) {
            write!(output, "{name}").unwrap();
            Ok(())
        } else {
            write!(output, "undef").unwrap();
            Ok(())
        }
    }
}
