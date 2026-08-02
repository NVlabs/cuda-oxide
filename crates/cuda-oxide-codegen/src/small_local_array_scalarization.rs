/*
 * SPDX-License-Identifier: Apache-2.0
 */

//! Opt-in textual LLVM IR scalarization for small compiler-owned local arrays
//! and read-only array fields borrowed from by-value aggregates.
//!
//! The pass runs after the first LLVM `default<O2>` pipeline has inlined helper
//! methods and iterator adapters. At that point, dynamic pointer arithmetic and
//! the compiler-owned allocation are visible in the same function. Eligible
//! dynamic scalar loads are rewritten as constant-address candidates plus value
//! selects. A following `default<O2>` run removes the dead dynamic pointer
//! arithmetic and lets SROA promote the allocation to registers.

use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;

const MAX_ELEMENTS: usize = 16;
const MAX_BYTES: usize = 128;

#[derive(Clone, Copy, Debug, Default)]
struct ScalarizationModes {
    small_array_iterators: bool,
    copy_aggregate_borrows: bool,
}

impl ScalarizationModes {
    fn from_env() -> Self {
        Self {
            small_array_iterators: mir_option_enabled("small-array-iterators"),
            copy_aggregate_borrows: mir_option_enabled("copy-aggregate-borrows"),
        }
    }

    fn any(self) -> bool {
        self.small_array_iterators || self.copy_aggregate_borrows
    }

    #[cfg(test)]
    const fn all() -> Self {
        Self {
            small_array_iterators: true,
            copy_aggregate_borrows: true,
        }
    }
}

/// Whether a post-inline small-local scalarization mode is enabled.
pub(crate) fn enabled() -> bool {
    ScalarizationModes::from_env().any()
}

fn mir_option_enabled(name: &str) -> bool {
    std::env::var("CUDA_OXIDE_MIR_OPTS")
        .ok()
        .is_some_and(|value| value.split(',').map(str::trim).any(|item| item == name))
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
    let (rewritten, replacements) =
        scalarize_text_with_modes(&source, ScalarizationModes::from_env());
    fs::write(output, rewritten)?;
    Ok(replacements)
}

#[cfg(test)]
fn scalarize_text(source: &str) -> (String, usize) {
    scalarize_text_with_modes(source, ScalarizationModes::all())
}

fn scalarize_text_with_modes(source: &str, modes: ScalarizationModes) -> (String, usize) {
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

        let (function, count) = scalarize_function(&lines[start..cursor], modes);
        output.extend(function);
        replacements += count;
    }

    let mut rewritten = output.join("\n");
    if had_trailing_newline {
        rewritten.push('\n');
    }
    (rewritten, replacements)
}

fn scalarize_function(lines: &[&str], modes: ScalarizationModes) -> (Vec<String>, usize) {
    let arrays = if modes.small_array_iterators {
        discover_small_arrays(lines)
    } else {
        HashMap::new()
    };
    let pointers = discover_pointer_roots(lines, &arrays);
    let values = discover_immutable_array_values(lines, &arrays, &pointers);
    let aggregate_analysis = if modes.copy_aggregate_borrows {
        discover_copy_aggregate_borrows(lines)
    } else {
        AggregateAnalysis::default()
    };

    let mut output = Vec::with_capacity(lines.len());
    let mut serial = 0usize;
    let mut aggregate_serial = 0usize;
    let mut replacements = 0usize;

    for (line_index, line) in lines.iter().enumerate() {
        let Some(load) = parse_load(line) else {
            output.push((*line).to_string());
            continue;
        };

        if let Some(pointer) = pointers.get(&load.pointer)
            && let Some(array) = arrays.get(&pointer.root)
            && let Some(element_values) = values.get(&pointer.root)
            && pointer.dynamic
            && !load.volatile_or_atomic
            && normalize_type(&load.value_type) == normalize_type(&array.element_type)
        {
            output.extend(rewrite_load(&load, array, element_values, serial));
            serial += 1;
            replacements += 1;
            continue;
        }

        if let Some(candidate) = aggregate_load_candidate(&aggregate_analysis, &load, line_index) {
            output.extend(rewrite_copy_aggregate_load(
                &load,
                &candidate,
                aggregate_serial,
            ));
            aggregate_serial += 1;
            replacements += 1;
            continue;
        }

        output.push((*line).to_string());
    }

    (output, replacements)
}

#[derive(Clone, Debug)]
struct AggregateGepIndex {
    ty: String,
    value: String,
}

impl AggregateGepIndex {
    fn constant(&self) -> Option<i64> {
        self.value.parse::<i64>().ok()
    }

    fn is_dynamic(&self) -> bool {
        self.value.starts_with('%')
    }

    fn is_zero(&self) -> bool {
        self.constant() == Some(0)
    }
}

#[derive(Clone, Debug)]
struct AggregateGepStep {
    modifiers: Vec<String>,
    source_type: String,
    indices: Vec<AggregateGepIndex>,
}

#[derive(Clone, Debug)]
struct AggregatePointerInfo {
    root: String,
    path: Vec<AggregateGepStep>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct ScalarStoreKey {
    byte_offset: usize,
    llvm_type: String,
}

#[derive(Clone, Debug)]
struct ScalarStore {
    line: usize,
    bytes: usize,
}

#[derive(Clone, Debug)]
struct AggregateRootState {
    llvm_type: String,
    alloca_line: usize,
    full_init_store_line: Option<usize>,
    scalar_stores: HashMap<ScalarStoreKey, ScalarStore>,
    initialization_lines: Vec<usize>,
    invalid: bool,
}

impl AggregateRootState {
    fn initialization_end_line(&self) -> Option<usize> {
        self.full_init_store_line
            .or_else(|| self.scalar_stores.values().map(|store| store.line).max())
    }

    fn is_initialized_before(&self, line: usize) -> bool {
        !self.invalid
            && self
                .initialization_end_line()
                .is_some_and(|initialization| {
                    self.alloca_line < initialization && initialization < line
                })
    }
}

#[derive(Default)]
struct AggregateAnalysis {
    roots: HashMap<String, AggregateRootState>,
    pointers: HashMap<String, AggregatePointerInfo>,
    index_upper_bounds: HashMap<String, usize>,
}

#[derive(Clone, Debug)]
struct AggregateLoadCandidate {
    root: String,
    path: Vec<AggregateGepStep>,
    dynamic_step: usize,
    dynamic_index: usize,
    elements: usize,
}

fn discover_copy_aggregate_borrows(lines: &[&str]) -> AggregateAnalysis {
    let Some((entry_start, entry_end)) = first_basic_block_range(lines) else {
        return AggregateAnalysis::default();
    };

    let index_upper_bounds = discover_unsigned_index_upper_bounds(lines);
    let mut roots = HashMap::new();
    for (line_index, line) in lines.iter().enumerate() {
        if line_index < entry_start || line_index >= entry_end {
            continue;
        }
        let Some((value, llvm_type)) = parse_aggregate_alloca(line) else {
            continue;
        };
        roots.insert(
            value,
            AggregateRootState {
                llvm_type,
                alloca_line: line_index,
                full_init_store_line: None,
                scalar_stores: HashMap::new(),
                initialization_lines: Vec::new(),
                invalid: false,
            },
        );
    }

    let mut pointers: HashMap<String, AggregatePointerInfo> = roots
        .keys()
        .map(|root| {
            (
                root.clone(),
                AggregatePointerInfo {
                    root: root.clone(),
                    path: Vec::new(),
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
            let Some((base, step)) = parse_aggregate_gep_step(&expression) else {
                continue;
            };
            let Some(parent) = pointers.get(&base) else {
                continue;
            };
            let mut path = parent.path.clone();
            path.push(step);
            pointers.insert(
                result,
                AggregatePointerInfo {
                    root: parent.root.clone(),
                    path,
                },
            );
            changed = true;
        }
        if !changed {
            break;
        }
    }

    for (line_index, line) in lines.iter().enumerate() {
        let Some(store) = parse_store(line) else {
            continue;
        };

        if let Some(pointer) = pointers.get(&store.pointer) {
            let state = roots
                .get_mut(&pointer.root)
                .expect("every aggregate pointer has a root state");
            if !record_aggregate_initialization_store(
                state,
                pointer,
                &store,
                line_index,
                entry_start,
                entry_end,
            ) {
                state.invalid = true;
            }
        }

        for value in all_ssa_values(&store.value) {
            let Some(pointer) = pointers.get(&value) else {
                continue;
            };
            roots
                .get_mut(&pointer.root)
                .expect("every aggregate pointer has a root state")
                .invalid = true;
        }
    }

    for (line_index, line) in lines.iter().enumerate() {
        let used_roots = aggregate_roots_in_text(line, &pointers);
        if used_roots.is_empty() {
            continue;
        }

        let expression = split_assignment(line)
            .map(|(_, expression)| expression)
            .unwrap_or_else(|| line.trim_start().to_string());

        if is_benign_intrinsic_call(&expression) {
            continue;
        }

        let tracked_alloca = split_assignment(line).is_some_and(|(result, expression)| {
            expression.starts_with("alloca ") && roots.contains_key(&result)
        });
        let tracked_gep = split_assignment(line).is_some_and(|(result, expression)| {
            expression.starts_with("getelementptr ") && pointers.contains_key(&result)
        });
        let pointer_compare = expression.starts_with("icmp ");
        let initializing_store = roots
            .values()
            .any(|state| state.initialization_lines.contains(&line_index));

        if tracked_alloca || tracked_gep || pointer_compare || initializing_store {
            continue;
        }

        if let Some(load) = parse_load(line)
            && !load.volatile_or_atomic
            && used_roots.iter().all(|root| {
                roots
                    .get(root)
                    .is_some_and(|state| state.is_initialized_before(line_index))
            })
        {
            continue;
        }

        for root in used_roots {
            roots
                .get_mut(&root)
                .expect("every used aggregate root has state")
                .invalid = true;
        }
    }

    for state in roots.values_mut() {
        if state.initialization_end_line().is_none()
            || (state.full_init_store_line.is_some() && !state.scalar_stores.is_empty())
        {
            state.invalid = true;
        }
    }

    AggregateAnalysis {
        roots,
        pointers,
        index_upper_bounds,
    }
}

fn record_aggregate_initialization_store(
    state: &mut AggregateRootState,
    pointer: &AggregatePointerInfo,
    store: &StoreLine,
    line_index: usize,
    entry_start: usize,
    entry_end: usize,
) -> bool {
    if line_index < entry_start || line_index >= entry_end || store.volatile_or_atomic {
        return false;
    }

    if pointer.path.is_empty()
        && normalize_type(&store.stored_type) == normalize_type(&state.llvm_type)
    {
        if state.full_init_store_line.is_some() || !state.scalar_stores.is_empty() {
            return false;
        }
        state.full_init_store_line = Some(line_index);
        state.initialization_lines.push(line_index);
        return true;
    }

    if state.full_init_store_line.is_some() {
        return false;
    }

    let normalized_type = normalize_type(&store.stored_type);
    let Some(bytes) = scalar_type_bytes(&normalized_type) else {
        return false;
    };
    if normalized_type == "ptr" {
        return false;
    }
    let Some(byte_offset) = constant_pointer_byte_offset(&pointer.path) else {
        return false;
    };
    let Some(end) = byte_offset.checked_add(bytes) else {
        return false;
    };
    if state.scalar_stores.iter().any(|(key, existing)| {
        let existing_end = key.byte_offset + existing.bytes;
        byte_offset < existing_end && key.byte_offset < end
    }) {
        return false;
    }

    let key = ScalarStoreKey {
        byte_offset,
        llvm_type: normalized_type,
    };
    state.scalar_stores.insert(
        key,
        ScalarStore {
            line: line_index,
            bytes,
        },
    );
    state.initialization_lines.push(line_index);
    true
}

fn aggregate_load_candidate(
    analysis: &AggregateAnalysis,
    load: &LoadLine,
    line_index: usize,
) -> Option<AggregateLoadCandidate> {
    if load.volatile_or_atomic {
        return None;
    }

    let pointer = analysis.pointers.get(&load.pointer)?;
    let root = analysis.roots.get(&pointer.root)?;
    if !root.is_initialized_before(line_index) {
        return None;
    }

    if root.full_init_store_line.is_some() {
        return structural_aggregate_load_candidate(pointer, load, &analysis.index_upper_bounds);
    }

    scalar_store_aggregate_load_candidate(
        pointer,
        root,
        load,
        line_index,
        &analysis.index_upper_bounds,
    )
}

fn structural_aggregate_load_candidate(
    pointer: &AggregatePointerInfo,
    load: &LoadLine,
    index_upper_bounds: &HashMap<String, usize>,
) -> Option<AggregateLoadCandidate> {
    let normalized_load_type = normalize_type(&load.value_type);
    let mut dynamic_location = None;
    let mut elements = 0usize;
    let mut element_type = String::new();

    for (step_index, step) in pointer.path.iter().enumerate() {
        let dynamic_indices: Vec<_> = step
            .indices
            .iter()
            .enumerate()
            .filter_map(|(index, operand)| operand.is_dynamic().then_some(index))
            .collect();
        if dynamic_indices.is_empty() {
            continue;
        }
        if dynamic_indices.len() != 1 || dynamic_location.is_some() {
            return None;
        }

        let dynamic_index = dynamic_indices[0];
        if let Some((array_elements, array_element_type)) = parse_array_type(&step.source_type) {
            if step.indices.len() != 2 || dynamic_index != 1 || !step.indices[0].is_zero() {
                return None;
            }
            elements = array_elements;
            element_type = array_element_type;
            dynamic_location = Some((step_index, dynamic_index));
            continue;
        }

        if normalize_type(&step.source_type) == normalized_load_type
            && step.indices.len() == 1
            && dynamic_index == 0
        {
            let previous = step_index
                .checked_sub(1)
                .and_then(|index| pointer.path.get(index))?;
            let (array_elements, array_element_type) = parse_array_type(&previous.source_type)?;
            if previous.indices.len() != 2
                || !previous.indices[0].is_zero()
                || !previous.indices[1].is_zero()
            {
                return None;
            }
            elements = array_elements;
            element_type = array_element_type;
            dynamic_location = Some((step_index, dynamic_index));
            continue;
        }

        return None;
    }

    let (dynamic_step, dynamic_index) = dynamic_location?;
    validate_aggregate_candidate(
        pointer,
        &normalized_load_type,
        &element_type,
        dynamic_step,
        dynamic_index,
        elements,
        index_upper_bounds,
    )
}

fn scalar_store_aggregate_load_candidate(
    pointer: &AggregatePointerInfo,
    root: &AggregateRootState,
    load: &LoadLine,
    line_index: usize,
    index_upper_bounds: &HashMap<String, usize>,
) -> Option<AggregateLoadCandidate> {
    let dynamic_step = pointer.path.len().checked_sub(1)?;
    let step = &pointer.path[dynamic_step];
    let dynamic_indices: Vec<_> = step
        .indices
        .iter()
        .enumerate()
        .filter_map(|(index, operand)| operand.is_dynamic().then_some(index))
        .collect();
    if dynamic_indices.len() != 1
        || dynamic_indices[0] != 0
        || step.indices.len() != 1
        || normalize_type(&step.source_type) != normalize_type(&load.value_type)
    {
        return None;
    }

    let normalized_type = normalize_type(&load.value_type);
    if normalized_type == "ptr" {
        return None;
    }
    let element_bytes = scalar_type_bytes(&normalized_type)?;
    let base_offset = constant_pointer_byte_offset(&pointer.path[..dynamic_step])?;
    let mut elements = 0usize;

    while elements < MAX_ELEMENTS {
        let byte_offset = base_offset.checked_add(elements.checked_mul(element_bytes)?)?;
        let key = ScalarStoreKey {
            byte_offset,
            llvm_type: normalized_type.clone(),
        };
        let Some(store) = root.scalar_stores.get(&key) else {
            break;
        };
        if store.line >= line_index {
            return None;
        }
        elements += 1;
    }

    validate_aggregate_candidate(
        pointer,
        &normalized_type,
        &normalized_type,
        dynamic_step,
        0,
        elements,
        index_upper_bounds,
    )
}

fn validate_aggregate_candidate(
    pointer: &AggregatePointerInfo,
    normalized_load_type: &str,
    element_type: &str,
    dynamic_step: usize,
    dynamic_index: usize,
    elements: usize,
    index_upper_bounds: &HashMap<String, usize>,
) -> Option<AggregateLoadCandidate> {
    if normalize_type(element_type) != normalized_load_type || normalized_load_type == "ptr" {
        return None;
    }
    let element_bytes = scalar_type_bytes(element_type)?;
    if elements == 0 || elements > MAX_ELEMENTS || elements.checked_mul(element_bytes)? > MAX_BYTES
    {
        return None;
    }

    let dynamic_value = &pointer
        .path
        .get(dynamic_step)?
        .indices
        .get(dynamic_index)?
        .value;
    if !index_upper_bounds
        .get(dynamic_value)
        .is_some_and(|upper_bound| *upper_bound > 0 && *upper_bound <= elements)
    {
        return None;
    }

    Some(AggregateLoadCandidate {
        root: pointer.root.clone(),
        path: pointer.path.clone(),
        dynamic_step,
        dynamic_index,
        elements,
    })
}

fn discover_unsigned_index_upper_bounds(lines: &[&str]) -> HashMap<String, usize> {
    lines
        .iter()
        .filter_map(|line| {
            let (result, expression) = split_assignment(line)?;
            let upper_bound = parse_urem_upper_bound(&expression)?;
            Some((result, upper_bound))
        })
        .collect()
}

fn parse_urem_upper_bound(expression: &str) -> Option<usize> {
    let rest = expression.strip_prefix("urem ")?;
    let integer_type = first_type_token(rest)?;
    let width = integer_type.strip_prefix('i')?.parse::<u32>().ok()?;
    if width == 0 {
        return None;
    }

    let operands = rest[integer_type.len()..].trim_start();
    let fields = split_top_level(operands, ',');
    if fields.len() != 2 {
        return None;
    }

    let divisor = fields[1].trim().parse::<usize>().ok()?;
    (divisor > 0).then_some(divisor)
}

fn constant_pointer_byte_offset(path: &[AggregateGepStep]) -> Option<usize> {
    let mut offset = 0usize;
    for step in path {
        if step.indices.iter().any(AggregateGepIndex::is_dynamic) {
            return None;
        }

        let normalized_type = normalize_type(&step.source_type);
        if let Some((_, element_type)) = parse_array_type(&normalized_type) {
            if step.indices.len() != 2 || !step.indices[0].is_zero() {
                return None;
            }
            let index = usize::try_from(step.indices[1].constant()?).ok()?;
            let element_bytes = scalar_type_bytes(&element_type)?;
            offset = offset.checked_add(index.checked_mul(element_bytes)?)?;
            continue;
        }

        let element_bytes = scalar_type_bytes(&normalized_type)?;
        if step.indices.len() != 1 {
            return None;
        }
        let index = usize::try_from(step.indices[0].constant()?).ok()?;
        offset = offset.checked_add(index.checked_mul(element_bytes)?)?;
    }
    Some(offset)
}

fn rewrite_copy_aggregate_load(
    load: &LoadLine,
    candidate: &AggregateLoadCandidate,
    serial: usize,
) -> Vec<String> {
    let prefix = format!("%__oxide_ca_{serial}");
    let mut lines = Vec::new();
    let mut candidate_pointers = Vec::with_capacity(candidate.elements);
    let mut candidate_values = Vec::with_capacity(candidate.elements);

    for element_index in 0..candidate.elements {
        let mut base = candidate.root.clone();
        for (step_index, step) in candidate.path.iter().enumerate() {
            let pointer = format!("{prefix}_ptr_{element_index}_{step_index}");
            let opcode = if step.modifiers.is_empty() {
                "getelementptr".to_string()
            } else {
                format!("getelementptr {}", step.modifiers.join(" "))
            };
            let indices = step
                .indices
                .iter()
                .enumerate()
                .map(|(index, operand)| {
                    let value = if step_index == candidate.dynamic_step
                        && index == candidate.dynamic_index
                    {
                        element_index.to_string()
                    } else {
                        operand.value.clone()
                    };
                    format!("{} {value}", operand.ty)
                })
                .collect::<Vec<_>>()
                .join(", ");
            lines.push(format!(
                "{}{} = {} {}, ptr {}, {}",
                load.indent, pointer, opcode, step.source_type, base, indices
            ));
            base = pointer;
        }

        let value = format!("{prefix}_val_{element_index}");
        lines.push(format!(
            "{}{} = load {}, ptr {}{}",
            load.indent, value, load.value_type, base, load.suffix
        ));
        candidate_pointers.push(base);
        candidate_values.push(value);
    }

    if candidate.elements == 1 {
        lines.push(format!(
            "{}{} = select i1 true, {} {}, {} {}",
            load.indent,
            load.result,
            load.value_type,
            candidate_values[0],
            load.value_type,
            candidate_values[0]
        ));
        return lines;
    }

    let mut selected = candidate_values[candidate.elements - 1].clone();
    for element_index in (0..candidate.elements - 1).rev() {
        let comparison = format!("{prefix}_cmp_{element_index}");
        lines.push(format!(
            "{}{} = icmp eq ptr {}, {}",
            load.indent, comparison, load.pointer, candidate_pointers[element_index]
        ));
        let result = if element_index == 0 {
            load.result.clone()
        } else {
            format!("{prefix}_sel_{element_index}")
        };
        lines.push(format!(
            "{}{} = select i1 {}, {} {}, {} {}",
            load.indent,
            result,
            comparison,
            load.value_type,
            candidate_values[element_index],
            load.value_type,
            selected
        ));
        selected = result;
    }

    lines
}

fn aggregate_roots_in_text(
    text: &str,
    pointers: &HashMap<String, AggregatePointerInfo>,
) -> Vec<String> {
    let mut roots = Vec::new();
    for value in all_ssa_values(text) {
        let Some(pointer) = pointers.get(&value) else {
            continue;
        };
        if !roots.contains(&pointer.root) {
            roots.push(pointer.root.clone());
        }
    }
    roots
}

fn parse_aggregate_alloca(line: &str) -> Option<(String, String)> {
    let (result, expression) = split_assignment(line)?;
    let rest = expression.strip_prefix("alloca ")?;
    let llvm_type = first_type_token(rest)?;
    let normalized = normalize_type(&llvm_type);
    let aggregate = (normalized.starts_with('%')
        || normalized.starts_with('{')
        || normalized.starts_with("<{"))
        && parse_array_type(&normalized).is_none();
    aggregate.then_some((result, llvm_type))
}

fn parse_aggregate_gep_step(expression: &str) -> Option<(String, AggregateGepStep)> {
    let mut rest = expression.strip_prefix("getelementptr ")?.trim_start();
    let mut modifiers = Vec::new();
    loop {
        let token = rest.split_whitespace().next()?;
        if matches!(token, "inbounds" | "nuw" | "nusw" | "nsw") {
            modifiers.push(token.to_string());
            rest = rest[token.len()..].trim_start();
        } else {
            break;
        }
    }

    let source_type = first_type_token(rest)?;
    rest = rest[source_type.len()..].trim_start();
    let operands = rest.strip_prefix(',')?.trim_start();
    let fields = split_top_level(operands, ',');
    if fields.len() < 2 {
        return None;
    }
    let base = first_ssa_value(fields[0])?;
    let mut indices = Vec::new();
    for field in fields.iter().skip(1) {
        let field = field.trim();
        if field.starts_with('!') {
            continue;
        }
        let ty = first_type_token(field)?;
        let value = field[ty.len()..].split_whitespace().next()?;
        if !value.starts_with('%') && value.parse::<i64>().is_err() {
            return None;
        }
        indices.push(AggregateGepIndex {
            ty,
            value: value.to_string(),
        });
    }
    if indices.is_empty() {
        return None;
    }

    Some((
        base,
        AggregateGepStep {
            modifiers,
            source_type,
            indices,
        },
    ))
}

fn first_basic_block_range(lines: &[&str]) -> Option<(usize, usize)> {
    let start = lines.iter().position(|line| is_basic_block_label(line))?;
    let end = lines
        .iter()
        .enumerate()
        .skip(start + 1)
        .find_map(|(index, line)| is_basic_block_label(line).then_some(index))
        .unwrap_or(lines.len().saturating_sub(1));
    Some((start + 1, end))
}

fn is_basic_block_label(line: &str) -> bool {
    let code = line.split(';').next().unwrap_or_default().trim();
    code.ends_with(':') && !code.contains(" = ")
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
    suffix: String,
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
            suffix: String::new(),
            volatile_or_atomic: true,
        });
    }

    let fields = split_top_level(rest, ',');
    if fields.len() < 2 {
        return None;
    }
    let value_type = fields[0].trim().to_string();
    let pointer = pointer_value_from_typed_operand(fields[1].trim())?;
    let suffix = if fields.len() > 2 {
        format!(
            ", {}",
            fields[2..]
                .iter()
                .map(|field| field.trim())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        String::new()
    };
    Some(LoadLine {
        indent,
        result,
        value_type,
        pointer,
        suffix,
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
    fn scalarization_modes_do_not_enable_each_other() {
        let local_array = r#"define void @kernel(i64 %index) {
entry:
  %array = alloca [2 x i32], align 4
  store i32 1, ptr %array, align 4
  %p1 = getelementptr inbounds i8, ptr %array, i64 4
  store i32 2, ptr %p1, align 4
  %dynamic = getelementptr inbounds i32, ptr %array, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;
        let aggregate = r#"%Shape = type { [2 x i32] }

define void @kernel(%Shape %shape, i64 %index) {
entry:
  %borrow = alloca %Shape, align 4
  store %Shape %shape, ptr %borrow, align 4
  %axis = urem i64 %index, 2
  %field = getelementptr inbounds %Shape, ptr %borrow, i64 0, i32 0
  %dynamic = getelementptr inbounds [2 x i32], ptr %field, i64 0, i64 %axis
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        let iterator_only = ScalarizationModes {
            small_array_iterators: true,
            copy_aggregate_borrows: false,
        };
        let aggregate_only = ScalarizationModes {
            small_array_iterators: false,
            copy_aggregate_borrows: true,
        };

        assert_eq!(scalarize_text_with_modes(local_array, aggregate_only).1, 0);
        assert_eq!(scalarize_text_with_modes(aggregate, iterator_only).1, 0);
        assert_eq!(scalarize_text_with_modes(local_array, iterator_only).1, 1);
        assert_eq!(scalarize_text_with_modes(aggregate, aggregate_only).1, 1);
    }

    #[test]
    fn rewrites_scalarized_copy_aggregate_initialization_from_llvm22() {
        let input = r#"define void @kernel(
    ptr %output,
    i64 %index,
    { [3 x i32], [3 x i1], [1 x i8] } %shape
) {
entry:
  %borrow = alloca { [3 x i32], [3 x i1], [1 x i8] }, align 4
  %count0 = extractvalue { [3 x i32], [3 x i1], [1 x i8] } %shape, 0, 0
  store i32 %count0, ptr %borrow, align 4
  %count1 = extractvalue { [3 x i32], [3 x i1], [1 x i8] } %shape, 0, 1
  %count1.ptr = getelementptr inbounds nuw i8, ptr %borrow, i64 4
  store i32 %count1, ptr %count1.ptr, align 4
  %count2 = extractvalue { [3 x i32], [3 x i1], [1 x i8] } %shape, 0, 2
  %count2.ptr = getelementptr inbounds nuw i8, ptr %borrow, i64 8
  store i32 %count2, ptr %count2.ptr, align 4
  %flag0 = extractvalue { [3 x i32], [3 x i1], [1 x i8] } %shape, 1, 0
  %flag0.ptr = getelementptr inbounds nuw i8, ptr %borrow, i64 12
  store i1 %flag0, ptr %flag0.ptr, align 4
  %flag1 = extractvalue { [3 x i32], [3 x i1], [1 x i8] } %shape, 1, 1
  %flag1.ptr = getelementptr inbounds nuw i8, ptr %borrow, i64 13
  store i1 %flag1, ptr %flag1.ptr, align 1
  %flag2 = extractvalue { [3 x i32], [3 x i1], [1 x i8] } %shape, 1, 2
  %flag2.ptr = getelementptr inbounds nuw i8, ptr %borrow, i64 14
  store i1 %flag2, ptr %flag2.ptr, align 2
  %axis = urem i64 %index, 3
  %dynamic.count = getelementptr inbounds nuw i32, ptr %borrow, i64 %axis
  %count = load i32, ptr %dynamic.count, align 4
  %dynamic.flag = getelementptr inbounds nuw i1, ptr %flag0.ptr, i64 %axis
  %flag = load i1, ptr %dynamic.flag, align 1
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 2);
        assert!(!output.contains("%count = load i32, ptr %dynamic.count"));
        assert!(!output.contains("%flag = load i1, ptr %dynamic.flag"));
        assert_eq!(
            output
                .matches(" = load i32, ptr %__oxide_ca_0_ptr_")
                .count(),
            3
        );
        assert_eq!(
            output.matches(" = load i1, ptr %__oxide_ca_1_ptr_").count(),
            3
        );
        assert!(output.contains("%count = select i1 %__oxide_ca_0_cmp_0"));
        assert!(output.contains("%flag = select i1 %__oxide_ca_1_cmp_0"));
    }

    #[test]
    fn rejects_overlapping_scalarized_copy_aggregate_stores() {
        let input = r#"define void @kernel(i64 %index, { [3 x i32] } %shape) {
entry:
  %borrow = alloca { [3 x i32] }, align 4
  %count0 = extractvalue { [3 x i32] } %shape, 0, 0
  store i32 %count0, ptr %borrow, align 4
  %overlap = getelementptr inbounds i8, ptr %borrow, i64 2
  store i32 99, ptr %overlap, align 2
  %dynamic = getelementptr inbounds i32, ptr %borrow, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rejects_unbounded_scalarized_copy_aggregate_index() {
        let input = r#"define void @kernel(i64 %index, { [3 x i32] } %shape) {
entry:
  %borrow = alloca { [3 x i32] }, align 4
  %count0 = extractvalue { [3 x i32] } %shape, 0, 0
  store i32 %count0, ptr %borrow, align 4
  %count1 = extractvalue { [3 x i32] } %shape, 0, 1
  %count1.ptr = getelementptr inbounds i8, ptr %borrow, i64 4
  store i32 %count1, ptr %count1.ptr, align 4
  %count2 = extractvalue { [3 x i32] } %shape, 0, 2
  %count2.ptr = getelementptr inbounds i8, ptr %borrow, i64 8
  store i32 %count2, ptr %count2.ptr, align 4
  %dynamic = getelementptr inbounds i32, ptr %borrow, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rejects_volatile_load_from_copy_aggregate_borrow() {
        let input = r#"%GridShape = type { [3 x i32] }

define void @kernel(%GridShape %shape, i64 %index) {
entry:
  %borrow = alloca %GridShape, align 4
  store %GridShape %shape, ptr %borrow, align 4
  %axis = urem i64 %index, 3
  %counts = getelementptr inbounds %GridShape, ptr %borrow, i64 0, i32 0
  %dynamic = getelementptr inbounds [3 x i32], ptr %counts, i64 0, i64 %axis
  %value = load volatile i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn leaves_external_dynamic_pointer_load_unchanged() {
        let input = r#"define i32 @kernel(ptr %external, i64 %index) {
entry:
  %bounded = urem i64 %index, 3
  %dynamic = getelementptr inbounds i32, ptr %external, i64 %bounded
  %value = load i32, ptr %dynamic, align 4
  ret i32 %value
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rewrites_runtime_indexed_array_field_from_copy_aggregate_borrow() {
        let input = r#"%GridShape = type { [3 x i32], [3 x i8] }

define void @kernel(%GridShape %shape, i64 %index) {
entry:
  %borrow = alloca %GridShape, align 4
  store %GridShape %shape, ptr %borrow, align 4
  %axis = urem i64 %index, 3
  %counts = getelementptr inbounds %GridShape, ptr %borrow, i64 0, i32 0
  %dynamic = getelementptr inbounds [3 x i32], ptr %counts, i64 0, i64 %axis
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 1);
        assert!(!output.contains("%value = load i32, ptr %dynamic"));
        assert_eq!(
            output
                .matches(" = load i32, ptr %__oxide_ca_0_ptr_")
                .count(),
            3
        );
        assert!(output.contains("%value = select i1 %__oxide_ca_0_cmp_0"));
        assert!(output.contains(
            "%__oxide_ca_0_ptr_2_1 = getelementptr inbounds [3 x i32], ptr %__oxide_ca_0_ptr_2_0, i64 0, i64 2"
        ));
    }

    #[test]
    fn rewrites_element_pointer_shape_from_copy_aggregate_borrow() {
        let input = r#"%GridShape = type { [3 x i32] }

define void @kernel(%GridShape %shape, i64 %index) {
entry:
  %borrow = alloca %GridShape, align 4
  store %GridShape %shape, ptr %borrow, align 4
  %axis = urem i64 %index, 3
  %counts = getelementptr inbounds %GridShape, ptr %borrow, i64 0, i32 0
  %base = getelementptr inbounds [3 x i32], ptr %counts, i64 0, i64 0
  %dynamic = getelementptr inbounds i32, ptr %base, i64 %axis
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        let (output, replacements) = scalarize_text(input);
        assert_eq!(replacements, 1);
        assert!(!output.contains("%value = load i32, ptr %dynamic"));
        assert!(output.contains(
            "%__oxide_ca_0_ptr_2_2 = getelementptr inbounds i32, ptr %__oxide_ca_0_ptr_2_1, i64 2"
        ));
    }

    #[test]
    fn rejects_copy_aggregate_with_write_through_derived_pointer() {
        let input = r#"%GridShape = type { [3 x i32] }

define void @kernel(%GridShape %shape, i64 %index) {
entry:
  %borrow = alloca %GridShape, align 4
  store %GridShape %shape, ptr %borrow, align 4
  %counts = getelementptr inbounds %GridShape, ptr %borrow, i64 0, i32 0
  %first = getelementptr inbounds [3 x i32], ptr %counts, i64 0, i64 0
  store i32 99, ptr %first, align 4
  %dynamic = getelementptr inbounds [3 x i32], ptr %counts, i64 0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rejects_copy_aggregate_pointer_escape_after_inlining() {
        let input = r#"%GridShape = type { [3 x i32] }

declare void @escape(ptr)

define void @kernel(%GridShape %shape, i64 %index) {
entry:
  %borrow = alloca %GridShape, align 4
  store %GridShape %shape, ptr %borrow, align 4
  call void @escape(ptr %borrow)
  %counts = getelementptr inbounds %GridShape, ptr %borrow, i64 0, i32 0
  %dynamic = getelementptr inbounds [3 x i32], ptr %counts, i64 0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  ret void
}
"#;

        assert_not_scalarized(input);
    }

    #[test]
    fn rejects_copy_aggregate_initialized_outside_entry() {
        let input = r#"%GridShape = type { [3 x i32] }

define void @kernel(%GridShape %shape, i64 %index, i1 %condition) {
entry:
  br i1 %condition, label %initialize, label %exit

initialize:
  %borrow = alloca %GridShape, align 4
  store %GridShape %shape, ptr %borrow, align 4
  %counts = getelementptr inbounds %GridShape, ptr %borrow, i64 0, i32 0
  %dynamic = getelementptr inbounds [3 x i32], ptr %counts, i64 0, i64 %index
  %value = load i32, ptr %dynamic, align 4
  br label %exit

exit:
  ret void
}
"#;

        assert_not_scalarized(input);
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
