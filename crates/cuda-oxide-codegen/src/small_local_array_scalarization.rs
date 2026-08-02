/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Opt-in textual LLVM IR scalarization for small compiler-owned local arrays.
//!
//! The pass runs after the first LLVM `default<O2>` pipeline has inlined the
//! iterator adapters. At that point, `slice::Iter::{next,nth}` pointer
//! arithmetic and the kernel-owned array allocation are in the same function.
//! Dynamic scalar loads rooted at a fully initialized small array allocation are
//! rewritten as constant-address candidate loads plus pointer comparisons and
//! value selects. A following `default<O2>` run removes the now-dead dynamic
//! pointer arithmetic and lets SROA promote the allocation to registers.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

const MAX_ELEMENTS: usize = 16;
const MAX_BYTES: usize = 128;

/// Whether the issue #399 small-array iterator optimization is enabled.
pub(crate) fn enabled() -> bool {
    std::env::var("CUDA_OXIDE_MIR_OPTS")
        .ok()
        .is_some_and(|value| {
            value
                .split(',')
                .map(str::trim)
                .any(|name| name == "small-array-iterators")
        })
}

#[derive(Clone, Debug)]
struct SmallArray {
    value: String,
    llvm_type: String,
    element_type: String,
    elements: usize,
    element_bytes: usize,
}

#[derive(Clone, Debug)]
struct PointerInfo {
    root: String,
    dynamic: bool,
    element_index: Option<i64>,
}

/// Scalarize eligible dynamic loads in `input` and write the rewritten module
/// to `output`. The return value is the number of replaced loads.
pub(crate) fn scalarize_file(input: &Path, output: &Path) -> io::Result<usize> {
    let source = fs::read_to_string(input)?;
    let (rewritten, replacements) = scalarize_text(&source);
    fs::write(output, rewritten)?;
    Ok(replacements)
}

fn scalarize_text(source: &str) -> (String, usize) {
    let had_trailing_newline = source.ends_with('\n');
    let lines: Vec<&str> = source.lines().collect();
    let mut output = Vec::with_capacity(lines.len());
    let mut replacements = 0usize;
    let mut cursor = 0usize;

    while cursor < lines.len() {
        if !lines[cursor].trim_start().starts_with("define ") {
            output.push(lines[cursor].to_string());
            cursor += 1;
            continue;
        }

        let start = cursor;
        cursor += 1;
        while cursor < lines.len() && lines[cursor].trim() != "}" {
            cursor += 1;
        }
        if cursor < lines.len() {
            cursor += 1;
        }

        let (function, count) = scalarize_function(&lines[start..cursor]);
        output.extend(function);
        replacements += count;
    }

    let mut rewritten = output.join("\n");
    if had_trailing_newline {
        rewritten.push('\n');
    }
    (rewritten, replacements)
}

fn scalarize_function(lines: &[&str]) -> (Vec<String>, usize) {
    let arrays = discover_small_arrays(lines);
    if arrays.is_empty() {
        return (lines.iter().map(|line| (*line).to_string()).collect(), 0);
    }

    let pointers = discover_pointer_roots(lines, &arrays);
    let values = discover_immutable_array_values(lines, &arrays, &pointers);
    if values.is_empty() {
        return (lines.iter().map(|line| (*line).to_string()).collect(), 0);
    }

    let mut output = Vec::with_capacity(lines.len());
    let mut serial = 0usize;
    let mut replacements = 0usize;

    for line in lines {
        let Some(load) = parse_load(line) else {
            output.push((*line).to_string());
            continue;
        };
        let Some(pointer) = pointers.get(&load.pointer) else {
            output.push((*line).to_string());
            continue;
        };
        let Some(array) = arrays.get(&pointer.root) else {
            output.push((*line).to_string());
            continue;
        };
        let Some(element_values) = values.get(&pointer.root) else {
            output.push((*line).to_string());
            continue;
        };

        if !pointer.dynamic
            || load.volatile_or_atomic
            || normalize_type(&load.value_type) != normalize_type(&array.element_type)
        {
            output.push((*line).to_string());
            continue;
        }

        output.extend(rewrite_load(&load, array, element_values, serial));
        serial += 1;
        replacements += 1;
    }

    (output, replacements)
}

#[derive(Default)]
struct ArrayStoreState {
    values: Vec<Option<String>>,
    invalid: bool,
}

fn discover_immutable_array_values(
    lines: &[&str],
    arrays: &HashMap<String, SmallArray>,
    pointers: &HashMap<String, PointerInfo>,
) -> HashMap<String, Vec<String>> {
    let mut states: HashMap<String, ArrayStoreState> = arrays
        .iter()
        .map(|(root, array)| {
            (
                root.clone(),
                ArrayStoreState {
                    values: vec![None; array.elements],
                    invalid: false,
                },
            )
        })
        .collect();

    for line in lines {
        invalidate_escaping_pointer_uses(line, pointers, &mut states);

        let Some(store) = parse_store(line) else {
            continue;
        };

        // Storing an array-derived pointer anywhere makes subsequent writes
        // through that alias invisible to this textual analysis.
        invalidate_pointer_values(&store.value, pointers, &mut states);

        let Some(pointer) = pointers.get(&store.pointer) else {
            continue;
        };
        let Some(array) = arrays.get(&pointer.root) else {
            continue;
        };
        let state = states
            .get_mut(&pointer.root)
            .expect("every discovered array has store state");

        if store.volatile_or_atomic || pointer.dynamic {
            state.invalid = true;
            continue;
        }

        if normalize_type(&store.stored_type) == normalize_type(&array.element_type) {
            let Some(index) = pointer.element_index else {
                state.invalid = true;
                continue;
            };
            if index < 0 || index as usize >= array.elements || store.value.contains('%') {
                state.invalid = true;
                continue;
            }
            let slot = &mut state.values[index as usize];
            if slot.replace(store.value).is_some() {
                state.invalid = true;
            }
            continue;
        }

        if normalize_type(&store.stored_type) == normalize_type(&array.llvm_type)
            && store.pointer == array.value
        {
            let Some(elements) = parse_constant_array_value(&store.value, array) else {
                state.invalid = true;
                continue;
            };
            if state.values.iter().any(Option::is_some) {
                state.invalid = true;
                continue;
            }
            for (slot, value) in state.values.iter_mut().zip(elements) {
                *slot = Some(value);
            }
            continue;
        }

        state.invalid = true;
    }

    states
        .into_iter()
        .filter_map(|(root, state)| {
            if state.invalid || state.values.iter().any(Option::is_none) {
                return None;
            }
            Some((
                root,
                state
                    .values
                    .into_iter()
                    .map(|value| value.expect("checked above"))
                    .collect(),
            ))
        })
        .collect()
}

fn invalidate_escaping_pointer_uses(
    line: &str,
    pointers: &HashMap<String, PointerInfo>,
    states: &mut HashMap<String, ArrayStoreState>,
) {
    let trimmed = line.trim_start();
    let expression = split_assignment(line)
        .map(|(_, expression)| expression)
        .unwrap_or_else(|| trimmed.to_string());

    if !is_unsafe_pointer_operation(&expression) || is_benign_intrinsic_call(&expression) {
        return;
    }

    invalidate_pointer_values(&expression, pointers, states);
}

fn invalidate_pointer_values(
    text: &str,
    pointers: &HashMap<String, PointerInfo>,
    states: &mut HashMap<String, ArrayStoreState>,
) {
    for value in all_ssa_values(text) {
        let Some(pointer) = pointers.get(&value) else {
            continue;
        };
        if let Some(state) = states.get_mut(&pointer.root) {
            state.invalid = true;
        }
    }
}

fn is_unsafe_pointer_operation(expression: &str) -> bool {
    let expression = expression.trim_start();
    expression.starts_with("call ")
        || expression.starts_with("invoke ")
        || expression.starts_with("callbr ")
        || expression.starts_with("atomicrmw ")
        || expression.starts_with("cmpxchg ")
        || expression.starts_with("ptrtoint ")
        || expression.starts_with("ret ")
}

fn is_benign_intrinsic_call(expression: &str) -> bool {
    expression.contains("@llvm.lifetime.start")
        || expression.contains("@llvm.lifetime.end")
        || expression.contains("@llvm.dbg.")
        || expression.contains("@llvm.invariant.start")
        || expression.contains("@llvm.invariant.end")
        || expression.contains("@llvm.assume")
}

fn parse_constant_array_value(value: &str, array: &SmallArray) -> Option<Vec<String>> {
    let value = value.trim();
    if !value.starts_with('[') || !value.ends_with(']') || value.contains('%') {
        return None;
    }
    let inner = &value[1..value.len() - 1];
    let fields = split_top_level(inner, ',');
    if fields.len() != array.elements {
        return None;
    }

    fields
        .into_iter()
        .map(|field| {
            let field = field.trim();
            let ty = first_type_token(field)?;
            if normalize_type(&ty) != normalize_type(&array.element_type) {
                return None;
            }
            let scalar = field[ty.len()..].trim();
            (!scalar.is_empty() && !scalar.contains('%')).then(|| scalar.to_string())
        })
        .collect()
}

fn discover_small_arrays(lines: &[&str]) -> HashMap<String, SmallArray> {
    let mut arrays = HashMap::new();
    for line in lines {
        let Some((value, llvm_type, elements, element_type)) = parse_array_alloca(line) else {
            continue;
        };
        let Some(element_bytes) = scalar_type_bytes(&element_type) else {
            continue;
        };
        if elements == 0 || elements > MAX_ELEMENTS {
            continue;
        }
        let Some(total_bytes) = elements.checked_mul(element_bytes) else {
            continue;
        };
        if total_bytes > MAX_BYTES {
            continue;
        }
        arrays.insert(
            value.clone(),
            SmallArray {
                value,
                llvm_type,
                element_type,
                elements,
                element_bytes,
            },
        );
    }
    arrays
}

fn discover_pointer_roots(
    lines: &[&str],
    arrays: &HashMap<String, SmallArray>,
) -> HashMap<String, PointerInfo> {
    let mut pointers: HashMap<String, PointerInfo> = arrays
        .keys()
        .map(|value| {
            (
                value.clone(),
                PointerInfo {
                    root: value.clone(),
                    dynamic: false,
                    element_index: Some(0),
                },
            )
        })
        .collect();

    for _ in 0..lines.len().max(1) {
        let mut changed = false;
        for line in lines {
            let Some((result, expression)) = split_assignment(line) else {
                continue;
            };
            if pointers.contains_key(&result) {
                continue;
            }

            let info = if expression.starts_with("getelementptr ")
                || expression.starts_with("getelementptr inbounds ")
            {
                derive_gep_pointer(&expression, &pointers, arrays)
            } else if expression.starts_with("bitcast ")
                || expression.starts_with("addrspacecast ")
                || expression.starts_with("freeze ")
            {
                first_ssa_value(&expression).and_then(|value| pointers.get(&value).cloned())
            } else if expression.starts_with("select ") || expression.starts_with("phi ") {
                merge_pointer_values(&expression, &pointers)
            } else {
                None
            };

            if let Some(info) = info {
                pointers.insert(result, info);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    pointers
}

fn derive_gep_pointer(
    expression: &str,
    pointers: &HashMap<String, PointerInfo>,
    arrays: &HashMap<String, SmallArray>,
) -> Option<PointerInfo> {
    let base = pointer_operand_after_ptr(expression)?;
    let parent = pointers.get(&base)?.clone();
    let array = arrays.get(&parent.root)?;
    let indices = gep_indices(expression)?;

    let mut dynamic = parent.dynamic;
    let mut element_index = parent.element_index;

    let source_type = gep_source_type(expression)?;
    let source_type = normalize_type(&source_type);

    if source_type == normalize_type(&array.llvm_type) && indices.len() >= 2 {
        if indices[0] == IndexValue::Constant(0) {
            match indices[1] {
                IndexValue::Constant(index) => element_index = Some(index),
                IndexValue::Dynamic => {
                    element_index = None;
                    dynamic = true;
                }
            }
        } else {
            element_index = None;
            dynamic = true;
        }
    } else if source_type == normalize_type(&array.element_type) && !indices.is_empty() {
        match indices[0] {
            IndexValue::Constant(offset) => {
                element_index = element_index.and_then(|index| index.checked_add(offset));
            }
            IndexValue::Dynamic => {
                element_index = None;
                dynamic = true;
            }
        }
    } else if source_type == "i8" {
        match indices.first().copied() {
            Some(IndexValue::Constant(bytes))
                if bytes >= 0 && (bytes as usize).is_multiple_of(array.element_bytes) =>
            {
                let offset = bytes / array.element_bytes as i64;
                element_index = element_index.and_then(|index| index.checked_add(offset));
            }
            _ => {
                element_index = None;
                dynamic = true;
            }
        }
    } else {
        element_index = None;
        dynamic = true;
    }

    if indices.contains(&IndexValue::Dynamic) {
        dynamic = true;
        element_index = None;
    }

    Some(PointerInfo {
        root: parent.root,
        dynamic,
        element_index,
    })
}

fn gep_source_type(expression: &str) -> Option<String> {
    let mut rest = expression.strip_prefix("getelementptr ")?.trim_start();

    loop {
        let token = rest.split_whitespace().next()?;
        if matches!(token, "inbounds" | "nuw" | "nusw" | "nsw") {
            rest = rest[token.len()..].trim_start();
        } else {
            break;
        }
    }

    first_type_token(rest)
}

fn merge_pointer_values(
    expression: &str,
    pointers: &HashMap<String, PointerInfo>,
) -> Option<PointerInfo> {
    let values = all_ssa_values(expression);
    let mut candidates = values
        .into_iter()
        .filter_map(|value| pointers.get(&value).cloned());
    let first = candidates.next()?;
    let mut merged = first;
    let mut count = 1usize;
    for candidate in candidates {
        if candidate.root != merged.root {
            return None;
        }
        count += 1;
        merged.dynamic = true;
        if candidate.element_index != merged.element_index {
            merged.element_index = None;
        }
    }
    (count >= 2).then_some(merged)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IndexValue {
    Constant(i64),
    Dynamic,
}

fn gep_indices(expression: &str) -> Option<Vec<IndexValue>> {
    let ptr_pos = expression.find(" ptr ")?;
    let after_ptr = &expression[ptr_pos + 5..];
    let first_comma = after_ptr.find(',')?;
    let index_text = &after_ptr[first_comma + 1..];
    let mut indices = Vec::new();

    for field in split_top_level(index_text, ',') {
        let field = field.trim();
        if field.is_empty() {
            continue;
        }
        let value = field.split_whitespace().last()?;
        if value.starts_with('%') {
            indices.push(IndexValue::Dynamic);
        } else if let Ok(constant) = value.parse::<i64>() {
            indices.push(IndexValue::Constant(constant));
        } else {
            indices.push(IndexValue::Dynamic);
        }
    }
    Some(indices)
}

fn parse_array_alloca(line: &str) -> Option<(String, String, usize, String)> {
    let (result, expression) = split_assignment(line)?;
    let rest = expression.strip_prefix("alloca ")?;
    let llvm_type = first_type_token(rest)?;
    let (elements, element_type) = parse_array_type(&llvm_type)?;
    Some((result, llvm_type, elements, element_type))
}

fn parse_array_type(llvm_type: &str) -> Option<(usize, String)> {
    let trimmed = llvm_type.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return None;
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let split = find_top_level_x(inner)?;
    let elements = inner[..split].trim().parse::<usize>().ok()?;
    let element_type = inner[split + 1..].trim().to_string();
    Some((elements, element_type))
}

fn find_top_level_x(text: &str) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0i32;
    for index in 0..bytes.len() {
        match bytes[index] as char {
            '[' | '{' | '<' | '(' => depth += 1,
            ']' | '}' | '>' | ')' => depth -= 1,
            'x' if depth == 0 => {
                let before_space = index > 0 && bytes[index - 1].is_ascii_whitespace();
                let after_space = index + 1 < bytes.len() && bytes[index + 1].is_ascii_whitespace();
                if before_space && after_space {
                    return Some(index);
                }
            }
            _ => {}
        }
    }
    None
}

fn first_type_token(text: &str) -> Option<String> {
    let text = text.trim_start();
    let first = text.chars().next()?;
    if matches!(first, '[' | '{' | '<' | '(') {
        let closing = match first {
            '[' => ']',
            '{' => '}',
            '<' => '>',
            '(' => ')',
            _ => unreachable!(),
        };
        let mut depth = 0i32;
        for (index, ch) in text.char_indices() {
            if ch == first {
                depth += 1;
            } else if ch == closing {
                depth -= 1;
                if depth == 0 {
                    return Some(text[..=index].to_string());
                }
            }
        }
        None
    } else {
        Some(
            text.split(|ch: char| ch == ',' || ch.is_whitespace())
                .next()?
                .to_string(),
        )
    }
}

fn scalar_type_bytes(ty: &str) -> Option<usize> {
    match normalize_type(ty).as_str() {
        "half" | "bfloat" | "i16" => Some(2),
        "float" | "i32" => Some(4),
        "double" | "i64" | "ptr" => Some(8),
        "i1" | "i8" => Some(1),
        "i128" => Some(16),
        _ => None,
    }
}

fn normalize_type(ty: &str) -> String {
    ty.split_whitespace().collect::<Vec<_>>().join(" ")
}

struct LoadLine {
    indent: String,
    result: String,
    value_type: String,
    pointer: String,
    volatile_or_atomic: bool,
}

fn parse_load(line: &str) -> Option<LoadLine> {
    let indent_len = line.len() - line.trim_start().len();
    let indent = line[..indent_len].to_string();
    let (result, expression) = split_assignment(line)?;
    let rest = expression.strip_prefix("load ")?;
    let volatile_or_atomic = rest.starts_with("volatile ") || rest.starts_with("atomic ");
    if volatile_or_atomic {
        return Some(LoadLine {
            indent,
            result,
            value_type: String::new(),
            pointer: String::new(),
            volatile_or_atomic: true,
        });
    }

    let fields = split_top_level(rest, ',');
    if fields.len() < 2 {
        return None;
    }
    let value_type = fields[0].trim().to_string();
    let pointer = pointer_value_from_typed_operand(fields[1].trim())?;
    Some(LoadLine {
        indent,
        result,
        value_type,
        pointer,
        volatile_or_atomic: false,
    })
}

struct StoreLine {
    stored_type: String,
    value: String,
    pointer: String,
    volatile_or_atomic: bool,
}

fn parse_store(line: &str) -> Option<StoreLine> {
    let mut rest = line.trim_start().strip_prefix("store ")?;
    let mut volatile_or_atomic = false;

    loop {
        if let Some(after) = rest.strip_prefix("volatile ") {
            volatile_or_atomic = true;
            rest = after;
        } else if let Some(after) = rest.strip_prefix("atomic ") {
            volatile_or_atomic = true;
            rest = after;
        } else {
            break;
        }
    }

    let fields = split_top_level(rest, ',');
    if fields.len() < 2 {
        return None;
    }
    let first = fields[0].trim();
    let stored_type = first_type_token(first)?;
    let value = first[stored_type.len()..].trim().to_string();
    if value.is_empty() {
        return None;
    }
    let pointer = pointer_value_from_typed_operand(fields[1].trim())?;
    Some(StoreLine {
        stored_type,
        value,
        pointer,
        volatile_or_atomic,
    })
}

fn rewrite_load(
    load: &LoadLine,
    array: &SmallArray,
    element_values: &[String],
    serial: usize,
) -> Vec<String> {
    let prefix = format!("%__oxide_sa_{serial}");
    let mut lines = Vec::new();

    for index in 0..array.elements {
        lines.push(format!(
            "{}{}_ptr_{} = getelementptr inbounds {}, ptr {}, i64 0, i64 {}",
            load.indent, prefix, index, array.llvm_type, array.value, index
        ));
    }

    if array.elements == 1 {
        lines.push(format!(
            "{}{} = select i1 true, {} {}, {} {}",
            load.indent,
            load.result,
            load.value_type,
            element_values[0],
            load.value_type,
            element_values[0]
        ));
        return lines;
    }

    let mut selected = element_values[array.elements - 1].clone();
    for index in (0..array.elements - 1).rev() {
        let cmp = format!("{}_cmp_{}", prefix, index);
        lines.push(format!(
            "{}{} = icmp eq ptr {}, {}_ptr_{}",
            load.indent, cmp, load.pointer, prefix, index
        ));
        let result = if index == 0 {
            load.result.clone()
        } else {
            format!("{}_sel_{}", prefix, index)
        };
        lines.push(format!(
            "{}{} = select i1 {}, {} {}, {} {}",
            load.indent,
            result,
            cmp,
            load.value_type,
            element_values[index],
            load.value_type,
            selected
        ));
        selected = result;
    }

    lines
}

fn split_assignment(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('%') {
        return None;
    }
    let (result, expression) = trimmed.split_once(" = ")?;
    Some((result.to_string(), expression.to_string()))
}

fn pointer_operand_after_ptr(text: &str) -> Option<String> {
    let position = text.find(" ptr ")?;
    first_ssa_value(&text[position + 5..])
}

fn pointer_value_from_typed_operand(text: &str) -> Option<String> {
    first_ssa_value(text)
}

fn first_ssa_value(text: &str) -> Option<String> {
    all_ssa_values(text).into_iter().next()
}

fn all_ssa_values(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let mut values = Vec::new();
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            index += 1;
            continue;
        }
        let start = index;
        index += 1;
        if index < bytes.len() && bytes[index] == b'"' {
            index += 1;
            while index < bytes.len() {
                if bytes[index] == b'"' && bytes[index - 1] != b'\\' {
                    index += 1;
                    break;
                }
                index += 1;
            }
        } else {
            while index < bytes.len()
                && (bytes[index].is_ascii_alphanumeric()
                    || matches!(bytes[index], b'_' | b'.' | b'$' | b'-'))
            {
                index += 1;
            }
        }
        values.push(text[start..index].to_string());
    }
    values
}

fn split_top_level(text: &str, separator: char) -> Vec<&str> {
    let mut fields = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (index, ch) in text.char_indices() {
        match ch {
            '[' | '{' | '<' | '(' => depth += 1,
            ']' | '}' | '>' | ')' => depth -= 1,
            _ if ch == separator && depth == 0 => {
                fields.push(&text[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    fields.push(&text[start..]);
    fields
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_not_scalarized(input: &str) {
        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 0);
        assert_eq!(output, input);
    }

    #[test]
    fn rewrites_dynamic_load_after_full_initialization() {
        let input = r#"define void @kernel(i64 %index) {
entry:
  %array = alloca [4 x float], align 4
  %p0 = getelementptr inbounds [4 x float], ptr %array, i64 0, i64 0
  %p1 = getelementptr inbounds [4 x float], ptr %array, i64 0, i64 1
  %p2 = getelementptr inbounds [4 x float], ptr %array, i64 0, i64 2
  %p3 = getelementptr inbounds [4 x float], ptr %array, i64 0, i64 3
  store float 1.000000e+00, ptr %p0, align 4
  store float 2.000000e+00, ptr %p1, align 4
  store float 3.000000e+00, ptr %p2, align 4
  store float 4.000000e+00, ptr %p3, align 4
  %dynamic = getelementptr inbounds float, ptr %p0, i64 %index
  %value = load float, ptr %dynamic, align 4
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 1);
        assert!(!output.contains("%value = load float, ptr %dynamic"));
        assert!(output.contains("%value = select i1 %__oxide_sa_0_cmp_0"));
        assert_eq!(
            output.matches("getelementptr inbounds [4 x float]").count(),
            8
        );
        assert!(!output.contains("%__oxide_sa_0_val_"));
    }

    #[test]
    fn rewrites_byte_gep_shape_emitted_for_slice_iterators() {
        let input = r#"define void @kernel(i64 %byte_offset) {
entry:
  %array = alloca [4 x float], align 4
  %p1 = getelementptr inbounds i8, ptr %array, i64 4
  %p2 = getelementptr inbounds i8, ptr %array, i64 8
  %p3 = getelementptr inbounds i8, ptr %array, i64 12
  store float 4.000000e+00, ptr %array, align 4
  store float -1.500000e+00, ptr %p1, align 4
  store float 1.300000e+00, ptr %p2, align 4
  store float -2.000000e-01, ptr %p3, align 4
  %dynamic = getelementptr inbounds i8, ptr %array, i64 %byte_offset
  %value = load float, ptr %dynamic, align 4
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 1);
        assert!(!output.contains("%value = load float, ptr %dynamic"));
        assert!(output.contains("%value = select i1 %__oxide_sa_0_cmp_0"));
        assert!(output.contains("float -1.500000e+00"));
    }

    #[test]
    fn rewrites_llvm22_nuw_byte_gep_shape() {
        let input = r#"define void @kernel(i64 %byte_offset0, i64 %byte_offset1) {
entry:
  %array = alloca [4 x float], align 4
  %p1 = getelementptr inbounds nuw i8, ptr %array, i64 4
  %p2 = getelementptr inbounds nuw i8, ptr %array, i64 8
  %p3 = getelementptr inbounds nuw i8, ptr %array, i64 12
  store float 4.000000e+00, ptr %array, align 4
  store float -3.000000e+00, ptr %p1, align 4
  store float 0x3FF5555560000000, ptr %p2, align 4
  store float -2.500000e-01, ptr %p3, align 4
  %dynamic0 = getelementptr inbounds nuw i8, ptr %array, i64 %byte_offset0
  %value0 = load float, ptr %dynamic0, align 4
  %dynamic1 = getelementptr inbounds nuw i8, ptr %array, i64 %byte_offset1
  %value1 = load float, ptr %dynamic1, align 4
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 2);
        assert!(!output.contains("%value0 = load float, ptr %dynamic0"));
        assert!(!output.contains("%value1 = load float, ptr %dynamic1"));
        assert!(output.contains("float -3.000000e+00"));
        assert!(output.contains("float 0x3FF5555560000000"));
    }

    #[test]
    fn rewrites_one_constant_aggregate_store() {
        let input = r#"define void @kernel(i64 %index) {
entry:
  %array = alloca [2 x i32], align 4
  store [2 x i32] [i32 7, i32 9], ptr %array, align 4
  %base = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 0
  %dynamic = getelementptr inbounds i32, ptr %base, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 1);
        assert!(output.contains("select i1 %__oxide_sa_0_cmp_0, i32 7, i32 9"));
    }

    #[test]
    fn leaves_partially_initialized_array_unchanged() {
        let input = r#"define void @kernel(i64 %index) {
entry:
  %array = alloca [4 x float], align 4
  %p0 = getelementptr inbounds [4 x float], ptr %array, i64 0, i64 0
  store float 1.000000e+00, ptr %p0, align 4
  %dynamic = getelementptr inbounds float, ptr %p0, i64 %index
  %value = load float, ptr %dynamic, align 4
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 0);
        assert!(output.contains("%value = load float, ptr %dynamic"));
    }

    #[test]
    fn follows_select_and_phi_pointer_provenance() {
        let input = r#"define void @kernel(i64 %index, i1 %condition) {
entry:
  %array = alloca [2 x i32], align 4
  %p0 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 0
  %p1 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 1
  store i32 7, ptr %p0, align 4
  store i32 9, ptr %p1, align 4
  %a = getelementptr inbounds i32, ptr %p0, i64 %index
  %b = select i1 %condition, ptr %a, ptr %p1
  %value = load i32, ptr %b, align 4
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 1);
        assert!(output.contains("%value = select i1 %__oxide_sa_0_cmp_0"));
    }

    #[test]
    fn rejects_volatile_store_to_candidate_array() {
        let input = r#"define void @kernel(i64 %index) {
entry:
  %array = alloca [2 x i32], align 4
  %p0 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 0
  %p1 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 1
  store i32 7, ptr %p0, align 4
  store i32 9, ptr %p1, align 4
  store volatile i32 11, ptr %p0, align 4
  %dynamic = getelementptr inbounds i32, ptr %p0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rejects_atomic_store_to_candidate_array() {
        let input = r#"define void @kernel(i64 %index) {
entry:
  %array = alloca [2 x i32], align 4
  %p0 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 0
  %p1 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 1
  store i32 7, ptr %p0, align 4
  store i32 9, ptr %p1, align 4
  store atomic i32 11, ptr %p0 seq_cst, align 4
  %dynamic = getelementptr inbounds i32, ptr %p0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rejects_atomicrmw_on_candidate_array() {
        let input = r#"define void @kernel(i64 %index) {
entry:
  %array = alloca [2 x i32], align 4
  %p0 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 0
  %p1 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 1
  store i32 7, ptr %p0, align 4
  store i32 9, ptr %p1, align 4
  %old = atomicrmw add ptr %p0, i32 1 seq_cst
  %dynamic = getelementptr inbounds i32, ptr %p0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rejects_cmpxchg_on_candidate_array() {
        let input = r#"define void @kernel(i64 %index) {
entry:
  %array = alloca [2 x i32], align 4
  %p0 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 0
  %p1 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 1
  store i32 7, ptr %p0, align 4
  store i32 9, ptr %p1, align 4
  %pair = cmpxchg ptr %p0, i32 7, i32 11 seq_cst seq_cst
  %dynamic = getelementptr inbounds i32, ptr %p0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rejects_pointer_escape_through_call() {
        let input = r#"declare void @escape(ptr)

define void @kernel(i64 %index) {
entry:
  %array = alloca [2 x i32], align 4
  %p0 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 0
  %p1 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 1
  store i32 7, ptr %p0, align 4
  store i32 9, ptr %p1, align 4
  call void @escape(ptr %array)
  %dynamic = getelementptr inbounds i32, ptr %p0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rejects_pointer_escape_through_store() {
        let input = r#"define void @kernel(i64 %index) {
entry:
  %slot = alloca ptr, align 8
  %array = alloca [2 x i32], align 4
  %p0 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 0
  %p1 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 1
  store i32 7, ptr %p0, align 4
  store i32 9, ptr %p1, align 4
  store ptr %array, ptr %slot, align 8
  %dynamic = getelementptr inbounds i32, ptr %p0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn allows_lifetime_intrinsics() {
        let input = r#"declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

define void @kernel(i64 %index) {
entry:
  %array = alloca [2 x i32], align 4
  call void @llvm.lifetime.start.p0(i64 8, ptr %array)
  %p0 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 0
  %p1 = getelementptr inbounds [2 x i32], ptr %array, i64 0, i64 1
  store i32 7, ptr %p0, align 4
  store i32 9, ptr %p1, align 4
  %dynamic = getelementptr inbounds i32, ptr %p0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  call void @llvm.lifetime.end.p0(i64 8, ptr %array)
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 1);
        assert!(!output.contains("%value = load i32, ptr %dynamic"));
    }

    #[test]
    fn leaves_oversized_arrays_unchanged() {
        let input = r#"define void @kernel(i64 %index) {
entry:
  %array = alloca [17 x i32], align 4
  %base = getelementptr inbounds [17 x i32], ptr %array, i64 0, i64 0
  %dynamic = getelementptr inbounds i32, ptr %base, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 0);
        assert_eq!(output, input);
    }
}
