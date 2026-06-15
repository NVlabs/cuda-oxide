/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Line-table debug metadata emission.
//!
//! Stage 2 deliberately emits only the debug-info pieces needed to map machine
//! instructions back to source lines. Locals, argument variables, lexical
//! blocks, and Rust types need more MIR/type plumbing and belong to later stages.

use std::{
    fmt::Write,
    path::{Path, PathBuf},
};

use combine::stream::position::SourcePosition;
use pliron::{
    location::{Location, Source},
    uniqued_any,
};

use super::state::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn has_debug_metadata(&self) -> bool {
        self.debug_compile_unit.is_some()
    }

    pub(super) fn debug_subprogram_for_function(
        &mut self,
        name: &str,
        loc: &Location,
    ) -> Option<usize> {
        if !self.debug_kind.line_tables_enabled() {
            return None;
        }

        let (path, pos) = self.source_position_from_location(loc)?;
        let cu_id = self.ensure_debug_compile_unit(&path);
        let file_id = self.ensure_debug_file(&path);
        let subroutine_type_id = self.ensure_debug_subroutine_type();
        let name = escape_debug_string(name);
        let line = pos.line;
        let id = self.alloc_metadata_id();

        self.debug_nodes.push((
            id,
            format!(
                "distinct !DISubprogram(name: \"{name}\", scope: !{file_id}, file: !{file_id}, \
                 line: {line}, type: !{subroutine_type_id}, scopeLine: {line}, \
                 spFlags: DISPFlagDefinition, unit: !{cu_id}, retainedNodes: !{{}})"
            ),
        ));
        self.debug_subprogram_files.insert(id, path);
        self.debug_subprogram_fallbacks
            .insert(id, (pos.line, pos.column));

        Some(id)
    }

    pub(super) fn attach_debug_to_last_line(
        &mut self,
        output: &mut String,
        output_before: usize,
        scope: Option<usize>,
        loc: &Location,
        allow_scope_fallback: bool,
    ) {
        if output.len() == output_before {
            return;
        }

        let Some(scope) = scope else {
            return;
        };
        let location_id = self.debug_location_for_scope(scope, loc).or_else(|| {
            if allow_scope_fallback {
                // LLVM rejects inlinable calls inside a debug-scoped function
                // unless the call itself has a location. When rustc/pliron did
                // not give the call one, point it at the function line instead
                // of letting opt discard the whole debug graph.
                self.debug_fallback_location_for_scope(scope)
            } else {
                None
            }
        });
        let Some(location_id) = location_id else {
            return;
        };

        if output.ends_with('\n') {
            output.pop();
            writeln!(output, ", !dbg !{location_id}").unwrap();
        }
    }

    pub(super) fn emit_debug_metadata(&mut self, output: &mut String) {
        let Some(cu_id) = self.debug_compile_unit else {
            return;
        };

        let dwarf_version_id = self.alloc_metadata_id();
        let debug_info_version_id = self.alloc_metadata_id();

        writeln!(output, "!llvm.dbg.cu = !{{!{cu_id}}}").unwrap();
        writeln!(
            output,
            "!llvm.module.flags = !{{!{dwarf_version_id}, !{debug_info_version_id}}}"
        )
        .unwrap();
        writeln!(
            output,
            "!{dwarf_version_id} = !{{i32 2, !\"Dwarf Version\", i32 2}}"
        )
        .unwrap();
        writeln!(
            output,
            "!{debug_info_version_id} = !{{i32 2, !\"Debug Info Version\", i32 3}}"
        )
        .unwrap();

        for (id, node) in &self.debug_nodes {
            writeln!(output, "!{id} = {node}").unwrap();
        }
    }

    fn ensure_debug_compile_unit(&mut self, path: &Path) -> usize {
        if let Some(id) = self.debug_compile_unit {
            return id;
        }

        let file_id = self.ensure_debug_file(path);
        let id = self.alloc_metadata_id();
        self.debug_nodes.push((
            id,
            format!(
                "distinct !DICompileUnit(language: DW_LANG_Rust, file: !{file_id}, \
                 producer: \"cuda-oxide\", isOptimized: true, runtimeVersion: 0, \
                 emissionKind: LineTablesOnly)"
            ),
        ));
        self.debug_compile_unit = Some(id);
        id
    }

    fn ensure_debug_file(&mut self, path: &Path) -> usize {
        if let Some(id) = self.debug_files.get(path).copied() {
            return id;
        }

        let (filename, directory) = split_file_and_directory(path);
        let filename = escape_debug_string(&filename);
        let directory = escape_debug_string(&directory);
        let id = self.alloc_metadata_id();

        self.debug_nodes.push((
            id,
            format!("!DIFile(filename: \"{filename}\", directory: \"{directory}\")"),
        ));
        self.debug_files.insert(path.to_path_buf(), id);

        id
    }

    fn ensure_debug_subroutine_type(&mut self) -> usize {
        if let Some(id) = self.debug_subroutine_type {
            return id;
        }

        let id = self.alloc_metadata_id();
        self.debug_nodes
            .push((id, "!DISubroutineType(types: !{null})".to_string()));
        self.debug_subroutine_type = Some(id);

        id
    }

    fn debug_location_for_scope(&mut self, scope: usize, loc: &Location) -> Option<usize> {
        if !self.debug_kind.line_tables_enabled() {
            return None;
        }

        let (path, pos) = self.source_position_from_location(loc)?;
        if self
            .debug_subprogram_files
            .get(&scope)
            .is_some_and(|scope_path| scope_path.as_path() != path)
        {
            return None;
        }

        let key = (scope, pos.line, pos.column);
        if let Some(id) = self.debug_locations.get(&key).copied() {
            return Some(id);
        }

        let id = self.alloc_metadata_id();
        self.debug_nodes.push((
            id,
            format!(
                "!DILocation(line: {}, column: {}, scope: !{})",
                pos.line, pos.column, scope
            ),
        ));
        self.debug_locations.insert(key, id);

        Some(id)
    }

    fn debug_fallback_location_for_scope(&mut self, scope: usize) -> Option<usize> {
        let (line, column) = self.debug_subprogram_fallbacks.get(&scope).copied()?;
        let key = (scope, line, column);
        if let Some(id) = self.debug_locations.get(&key).copied() {
            return Some(id);
        }

        let id = self.alloc_metadata_id();
        self.debug_nodes.push((
            id,
            format!("!DILocation(line: {line}, column: {column}, scope: !{scope})"),
        ));
        self.debug_locations.insert(key, id);

        Some(id)
    }

    fn source_position_from_location(&self, loc: &Location) -> Option<(PathBuf, SourcePosition)> {
        match loc {
            Location::SrcPos {
                src: Source::File(path_key),
                pos,
            } if pos.line > 0 && pos.column > 0 => Some((
                uniqued_any::get(self.ctx, *path_key).clone(),
                SourcePosition {
                    line: pos.line,
                    column: pos.column,
                },
            )),
            Location::SrcPos { .. } | Location::Unknown => None,
            Location::Named { child_loc, .. } => self.source_position_from_location(child_loc),
            Location::Fused { locations, .. } => locations
                .iter()
                .find_map(|loc| self.source_position_from_location(loc)),
            Location::CallSite { caller, callee } => self
                .source_position_from_location(caller)
                .or_else(|| self.source_position_from_location(callee)),
        }
    }
}

fn split_file_and_directory(path: &Path) -> (String, String) {
    let filename = path
        .file_name()
        .filter(|name| !name.is_empty())
        .unwrap_or(path.as_os_str())
        .to_string_lossy()
        .into_owned();

    let directory = path
        .parent()
        .map(|parent| {
            let dir = parent.to_string_lossy();
            if dir.is_empty() {
                ".".to_string()
            } else {
                dir.into_owned()
            }
        })
        .unwrap_or_else(|| ".".to_string());

    (filename, directory)
}

fn escape_debug_string(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '\\' => out.push_str("\\5C"),
            '"' => out.push_str("\\22"),
            '\n' => out.push_str("\\0A"),
            '\r' => out.push_str("\\0D"),
            '\t' => out.push_str("\\09"),
            _ => out.push(ch),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use combine::stream::position::SourcePosition;
    use pliron::{context::Context, location::Source};

    #[test]
    fn debug_strings_use_llvm_metadata_escapes() {
        assert_eq!(escape_debug_string("a\\b\"c\n\t"), "a\\5Cb\\22c\\0A\\09");
    }

    #[test]
    fn split_file_and_directory_handles_bare_and_nested_paths() {
        assert_eq!(
            split_file_and_directory(Path::new("kernel.rs")),
            ("kernel.rs".to_string(), ".".to_string())
        );
        assert_eq!(
            split_file_and_directory(Path::new("/tmp/cuda-oxide/kernel.rs")),
            ("kernel.rs".to_string(), "/tmp/cuda-oxide".to_string())
        );
    }

    #[test]
    fn source_position_from_location_unwraps_named_locations() {
        let mut ctx = Context::new();
        let loc = Location::Named {
            name: "lowered".to_string(),
            child_loc: Box::new(Location::SrcPos {
                src: Source::new_from_file(&mut ctx, PathBuf::from("/tmp/kernel.rs")),
                pos: SourcePosition {
                    line: 12,
                    column: 4,
                },
            }),
        };
        let state = ModuleExportState::new(
            &ctx,
            false,
            true,
            super::super::config::DebugKind::LineTables,
        );

        let (path, pos) = state
            .source_position_from_location(&loc)
            .expect("location should unwrap");

        assert_eq!(path, PathBuf::from("/tmp/kernel.rs"));
        assert_eq!(pos.line, 12);
        assert_eq!(pos.column, 4);
    }
}
