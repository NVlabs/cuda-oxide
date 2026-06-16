/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Debug metadata emission.
//!
//! Line-table mode emits just enough metadata to map machine instructions back
//! to source lines. Full mode builds on that with the first variable/type slice:
//! simple source locals are described with `llvm.dbg.declare` and compact DWARF
//! type nodes.

use std::{
    fmt::Write,
    path::{Path, PathBuf},
};

use combine::stream::position::SourcePosition;
use pliron::{
    location::{Location, Source},
    uniqued_any,
};

use crate::ops::{DebugLocalTypeKind, DebugLocalVariableInfo};

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

    pub(super) fn emit_debug_intrinsic_declarations(&self, output: &mut String) {
        if self.debug_declare_used {
            writeln!(
                output,
                "declare void @llvm.dbg.declare(metadata, metadata, metadata)"
            )
            .unwrap();
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

    pub(super) fn debug_local_variable_for_scope(
        &mut self,
        scope: usize,
        loc: &Location,
        info: &DebugLocalVariableInfo,
    ) -> Option<(usize, usize)> {
        if !self.debug_kind.variables_enabled() {
            return None;
        }

        let (path, pos) = self.local_variable_position_from_location(loc)?;
        let file_id = self.ensure_debug_file(&path);
        let variable_scope = self.debug_scope_for_file(scope, &path)?;
        let type_id = self.ensure_debug_type(&info.ty);
        let location_id = self.debug_location_for_scope(scope, loc)?;
        let name = escape_debug_string(&info.name);
        let arg = info
            .argument_index
            .map(|idx| format!("arg: {idx}, "))
            .unwrap_or_default();
        let id = self.alloc_metadata_id();

        self.debug_nodes.push((
            id,
            format!(
                "!DILocalVariable(name: \"{name}\", {arg}scope: !{variable_scope}, file: !{file_id}, \
                 line: {}, type: !{type_id})",
                pos.line
            ),
        ));

        Some((id, location_id))
    }

    fn ensure_debug_compile_unit(&mut self, path: &Path) -> usize {
        if let Some(id) = self.debug_compile_unit {
            return id;
        }

        let file_id = self.ensure_debug_file(path);
        let id = self.alloc_metadata_id();
        let is_optimized = if self.debug_kind.variables_enabled() {
            "false"
        } else {
            "true"
        };
        let emission_kind = if self.debug_kind.variables_enabled() {
            "FullDebug"
        } else {
            "LineTablesOnly"
        };
        self.debug_nodes.push((
            id,
            format!(
                "distinct !DICompileUnit(language: DW_LANG_Rust, file: !{file_id}, \
                 producer: \"cuda-oxide\", isOptimized: {is_optimized}, runtimeVersion: 0, \
                 emissionKind: {emission_kind})"
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

    fn ensure_debug_type(&mut self, ty: &DebugLocalTypeKind) -> usize {
        if let Some(id) = self.debug_types.get(ty).copied() {
            return id;
        }

        let node = match ty {
            DebugLocalTypeKind::Basic {
                name,
                size_bits,
                encoding,
            } => {
                let name = escape_debug_string(name);
                format!("!DIBasicType(name: \"{name}\", size: {size_bits}, encoding: {encoding})")
            }
            DebugLocalTypeKind::Pointer { name, size_bits } => {
                let name = escape_debug_string(name);
                format!(
                    "!DIDerivedType(tag: DW_TAG_pointer_type, name: \"{name}\", \
                     baseType: null, size: {size_bits})"
                )
            }
        };

        let id = self.alloc_metadata_id();
        self.debug_nodes.push((id, node));
        self.debug_types.insert(ty.clone(), id);

        id
    }

    fn debug_location_for_scope(&mut self, scope: usize, loc: &Location) -> Option<usize> {
        if !self.debug_kind.line_tables_enabled() {
            return None;
        }

        match loc {
            Location::CallSite { callee, caller } => {
                return self.debug_call_site_location_for_scope(scope, callee, caller);
            }
            Location::Named { child_loc, .. } => {
                return self.debug_location_for_scope(scope, child_loc);
            }
            Location::Fused { locations, .. } => {
                return locations
                    .iter()
                    .find_map(|loc| self.debug_location_for_scope(scope, loc));
            }
            Location::SrcPos { .. } | Location::Unknown => {}
        }

        let (path, pos) = self.source_position_from_location(loc)?;
        let location_scope = self.debug_scope_for_file(scope, &path)?;

        self.ensure_debug_location(location_scope, pos, None)
    }

    fn debug_call_site_location_for_scope(
        &mut self,
        scope: usize,
        callee: &Location,
        caller: &Location,
    ) -> Option<usize> {
        let caller_location = self
            .source_position_from_location(caller)
            .and_then(|(path, pos)| {
                let caller_scope = self.debug_scope_for_file(scope, &path)?;
                self.ensure_debug_location(caller_scope, pos, None)
            });

        let Some((callee_path, callee_pos)) = self.source_position_from_location(callee) else {
            return caller_location;
        };
        let callee_scope = self.debug_scope_for_file(scope, &callee_path)?;

        self.ensure_debug_location(callee_scope, callee_pos, caller_location)
    }

    fn debug_scope_for_file(&mut self, scope: usize, path: &Path) -> Option<usize> {
        let scope_path = self.debug_subprogram_files.get(&scope)?;
        if scope_path.as_path() == path {
            return Some(scope);
        }

        let key = (scope, path.to_path_buf());
        if let Some(id) = self.debug_file_scopes.get(&key).copied() {
            return Some(id);
        }

        let file_id = self.ensure_debug_file(path);
        let id = self.alloc_metadata_id();
        self.debug_nodes.push((
            id,
            format!("!DILexicalBlockFile(scope: !{scope}, file: !{file_id}, discriminator: 0)"),
        ));
        self.debug_file_scopes.insert(key, id);

        Some(id)
    }

    fn ensure_debug_location(
        &mut self,
        scope: usize,
        pos: SourcePosition,
        inlined_at: Option<usize>,
    ) -> Option<usize> {
        if pos.line <= 0 || pos.column <= 0 {
            return None;
        }

        let key = (scope, pos.line, pos.column, inlined_at);
        if let Some(id) = self.debug_locations.get(&key).copied() {
            return Some(id);
        }

        let id = self.alloc_metadata_id();
        let inlined_at = inlined_at
            .map(|location_id| format!(", inlinedAt: !{location_id}"))
            .unwrap_or_default();
        self.debug_nodes.push((
            id,
            format!(
                "!DILocation(line: {}, column: {}, scope: !{}{inlined_at})",
                pos.line, pos.column, scope
            ),
        ));
        self.debug_locations.insert(key, id);

        Some(id)
    }

    fn debug_fallback_location_for_scope(&mut self, scope: usize) -> Option<usize> {
        let (line, column) = self.debug_subprogram_fallbacks.get(&scope).copied()?;
        self.ensure_debug_location(scope, SourcePosition { line, column }, None)
    }

    fn local_variable_position_from_location(
        &self,
        loc: &Location,
    ) -> Option<(PathBuf, SourcePosition)> {
        match loc {
            Location::CallSite { callee, caller } => self
                .source_position_from_location(callee)
                .or_else(|| self.source_position_from_location(caller)),
            Location::Named { child_loc, .. } => {
                self.local_variable_position_from_location(child_loc)
            }
            Location::Fused { locations, .. } => locations
                .iter()
                .find_map(|loc| self.local_variable_position_from_location(loc)),
            Location::SrcPos { .. } | Location::Unknown => self.source_position_from_location(loc),
        }
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
