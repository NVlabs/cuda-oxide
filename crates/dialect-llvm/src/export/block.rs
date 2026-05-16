/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{
    basic_block::BasicBlock, context::Ptr, linked_list::ContainsLinkedList, r#type::Typed,
    value::Value,
};

use super::{ModuleExportState, PredecessorMap};

impl<'a> ModuleExportState<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn export_block(
        &mut self,
        block: Ptr<BasicBlock>,
        value_names: &mut HashMap<Value, String>,
        next_value_id: &mut usize,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        pred_map: &PredecessorMap,
        is_entry: bool,
        output: &mut String,
    ) -> Result<(), String> {
        // Always print label to ensure it can be referenced by PHI nodes
        let label = block_labels.get(&block).unwrap();
        writeln!(output, "{label}:").unwrap();

        // Generate PHI nodes for block arguments (except entry block which uses function args)
        let args: Vec<_> = block.deref(self.ctx).arguments().collect();
        if !args.is_empty() && !is_entry {
            let preds = pred_map
                .get(&block)
                .ok_or_else(|| "Block with args has no predecessors".to_string())?;

            for (arg_idx, arg) in args.iter().enumerate() {
                // Use pre-assigned name or generate new one
                let arg_name = if let Some(name) = value_names.get(arg) {
                    name.clone()
                } else {
                    let name = format!("%v{next_value_id}");
                    *next_value_id += 1;
                    value_names.insert(*arg, name.clone());
                    name
                };

                write!(output, "  {arg_name} = phi ").unwrap();
                self.export_type(arg.get_type(self.ctx), output)?;
                write!(output, " ").unwrap();

                for (i, (pred_block, pred_args)) in preds.iter().enumerate() {
                    if i > 0 {
                        write!(output, ", ").unwrap();
                    }

                    if arg_idx < pred_args.len() {
                        let val = pred_args[arg_idx];
                        write!(output, "[ ").unwrap();
                        self.export_value(val, value_names, output)?;
                        let label = block_labels.get(pred_block).unwrap();
                        write!(output, ", %{label} ]").unwrap();
                    } else {
                        write!(
                            output,
                            "[ undef, %{} ]",
                            block_labels.get(pred_block).unwrap()
                        )
                        .unwrap();
                    }
                }
                writeln!(output).unwrap();
            }
        }

        for op in block.deref(self.ctx).iter(self.ctx) {
            self.export_op(op, value_names, next_value_id, block_labels, output)?;
        }
        Ok(())
    }
}
