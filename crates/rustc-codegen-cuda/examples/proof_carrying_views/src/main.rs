/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Safe proof-carrying views beside equivalent raw-pointer kernels.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D, LaunchConfig2D};
use cuda_device::{
    DisjointSlice, LinearTiles, RowMajorTiles, cuda_module, kernel, launch_bounds, launch_contract,
    thread,
};

const EPILOGUE_ROWS: u32 = 64;
const EPILOGUE_STRIDE: u32 = 64;
const EPILOGUE_COLS_PER_THREAD: u32 = 2;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    fn checked_raw_tile_start(thread: u32, len: usize) -> u64 {
        const WIDTH: u32 = 4;
        const LAST_OFFSET: u32 = WIDTH - 1;
        if thread > (u32::MAX - LAST_OFFSET) / WIDTH {
            return u64::MAX;
        }
        let base = thread * WIDTH;
        let last = base + LAST_OFFSET;
        if (last as usize) < len {
            u64::from(base)
        } else {
            u64::MAX
        }
    }

    #[inline(always)]
    fn checked_raw_epilogue_start(row: u32, tile_col: u32, len: usize) -> u64 {
        let last_col_offset = EPILOGUE_COLS_PER_THREAD - 1;
        if tile_col > (u32::MAX - last_col_offset) / EPILOGUE_COLS_PER_THREAD {
            return u64::MAX;
        }
        let col = tile_col * EPILOGUE_COLS_PER_THREAD;
        let last_col = col + last_col_offset;
        if last_col >= EPILOGUE_STRIDE || row > (u32::MAX - last_col) / EPILOGUE_STRIDE {
            return u64::MAX;
        }
        let start = row * EPILOGUE_STRIDE + col;
        let last = start + last_col_offset;
        if (last as usize) < len {
            u64::from(start)
        } else {
            u64::MAX
        }
    }

    /// One bounds proof gives this thread read/write access to one element.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub fn safe_element(mut values: DisjointSlice<u32>) {
        let index = thread::index_1d_u32(launch_context);
        if let Some(mut element) = values.element_thread32(index) {
            let value = element.read();
            element.write(value.wrapping_mul(3).wrapping_add(1));
        }
    }

    /// Legacy kernels now fail closed if a caller supplies non-1D Y/Z axes.
    #[kernel]
    pub fn legacy_rank_guard(mut values: DisjointSlice<u32>) {
        let index = thread::index_1d();
        if let Some(value) = values.get_mut(index) {
            *value = 1;
        }
    }

    /// The same operation with a manually checked raw pointer.
    ///
    /// # Safety
    ///
    /// `values` must reference `len` readable and writable device `u32`
    /// elements for the duration of the launch.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub unsafe fn raw_element(values: *mut u32, len: usize) {
        let index = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x());
        if (index as usize) < len {
            // SAFETY: the branch proves `index < len`, the host allocation has
            // `len` elements, and each 1-D thread computes a distinct index.
            unsafe {
                let element = values.add(index as usize);
                let value = element.read();
                element.write(value.wrapping_mul(3).wrapping_add(1));
            }
        }
    }

    /// One range proof gives this thread a four-element static view.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub fn safe_tile(mut values: DisjointSlice<u32, LinearTiles<4>>) {
        let index = thread::index_1d_u32(launch_context);
        if let Some(mut tile) = values.tile_thread32(index) {
            let v0 = tile.at_const::<0>().read();
            let v1 = tile.at_const::<1>().read();
            let v2 = tile.at_const::<2>().read();
            let v3 = tile.at_const::<3>().read();
            tile.at_const::<0>().write(v0.wrapping_add(1));
            tile.at_const::<1>().write(v1.wrapping_add(2));
            tile.at_const::<2>().write(v2.wrapping_add(3));
            tile.at_const::<3>().write(v3.wrapping_add(4));
        }
    }

    /// The same operation with explicit overflow and range checks.
    ///
    /// # Safety
    ///
    /// `values` must reference `len` readable and writable device `u32`
    /// elements for the duration of the launch.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, coordinates = u32, block = (64, 1, 1))]
    pub unsafe fn raw_tile(values: *mut u32, len: usize) {
        let thread = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x());
        let base = checked_raw_tile_start(thread, len);
        if base == u64::MAX {
            return;
        }
        let base = base as u32;
        // SAFETY: the combined guard proves all four offsets are in the
        // allocation; distinct 1-D threads own disjoint four-element ranges.
        unsafe {
            let tile = values.add(base as usize);
            let v0 = tile.add(0).read();
            let v1 = tile.add(1).read();
            let v2 = tile.add(2).read();
            let v3 = tile.add(3).read();
            tile.add(0).write(v0.wrapping_add(1));
            tile.add(1).write(v1.wrapping_add(2));
            tile.add(2).write(v2.wrapping_add(3));
            tile.add(3).write(v3.wrapping_add(4));
        }
    }

    /// A two-column GEMM-style epilogue tile with static row-major layout.
    #[kernel(launch_context = launch_context)]
    #[launch_bounds(64)]
    #[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
    pub fn safe_epilogue(mut values: DisjointSlice<f32, RowMajorTiles<1, 2, 64>>) {
        let coord = thread::coord_2d_u32(launch_context);
        if let Some(mut tile) = values.tile_2d32(coord) {
            let left = tile.at_const::<0, 0>().read();
            let right = tile.at_const::<0, 1>().read();
            tile.at_const::<0, 0>().write(left + 1.0);
            tile.at_const::<0, 1>().write(right + 1.0);
        }
    }

    /// Equivalent manually proved raw-pointer epilogue.
    ///
    /// # Safety
    ///
    /// `values` must reference `len` readable and writable device `f32`
    /// elements for the duration of the launch.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
    pub unsafe fn raw_epilogue(values: *mut f32, len: usize) {
        let row = thread::blockIdx_y()
            .wrapping_mul(thread::blockDim_y())
            .wrapping_add(thread::threadIdx_y());
        let tile_col = thread::blockIdx_x()
            .wrapping_mul(thread::blockDim_x())
            .wrapping_add(thread::threadIdx_x());
        let start = checked_raw_epilogue_start(row, tile_col, len);
        if start == u64::MAX {
            return;
        }
        // SAFETY: the scalar proof above covers both columns, and distinct 2D
        // threads own disjoint row/column tiles.
        unsafe {
            let pair = values.add(start as usize);
            let left = pair.read();
            let right = pair.add(1).read();
            pair.write(left + 1.0);
            pair.add(1).write(right + 1.0);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--verify-ptx") {
        return verify_ptx();
    }

    const N: usize = 4096;
    const BLOCK: u32 = 64;

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    // SAFETY: this standalone example owns the package-named device bundle,
    // whose four entry definitions are generated by the module above.
    let module = unsafe { kernels::load(&context)? };
    let initial: Vec<u32> = (0..N as u32).collect();

    let mut safe_elements = DeviceBuffer::from_host(&stream, &initial)?;
    let raw_elements = DeviceBuffer::from_host(&stream, &initial)?;
    let element_grid = (N as u32).div_ceil(BLOCK);
    let safe_element_launch =
        module.prepare_safe_element(LaunchConfig1D::new(element_grid, BLOCK, 0))?;
    let raw_element_launch =
        module.prepare_raw_element(LaunchConfig1D::new(element_grid, BLOCK, 0))?;
    module.safe_element(&stream, &safe_element_launch, &mut safe_elements)?;
    // SAFETY: `raw_elements` owns exactly N live device u32 elements and
    // remains alive until after stream synchronization in `to_host_vec`.
    unsafe {
        module.raw_element(
            &stream,
            &raw_element_launch,
            raw_elements.cu_deviceptr() as *mut u32,
            N,
        )?;
    }

    let safe_elements = safe_elements.to_host_vec(&stream)?;
    let raw_elements = raw_elements.to_host_vec(&stream)?;
    assert_eq!(safe_elements, raw_elements);
    assert!(
        safe_elements
            .iter()
            .enumerate()
            .all(|(i, &value)| value == (i as u32).wrapping_mul(3).wrapping_add(1))
    );

    let mut safe_tiles = DeviceBuffer::from_host(&stream, &initial)?;
    let raw_tiles = DeviceBuffer::from_host(&stream, &initial)?;
    let tile_threads = (N as u32).div_ceil(4);
    let tile_grid = tile_threads.div_ceil(BLOCK);
    let safe_tile_launch = module.prepare_safe_tile(LaunchConfig1D::new(tile_grid, BLOCK, 0))?;
    let raw_tile_launch = module.prepare_raw_tile(LaunchConfig1D::new(tile_grid, BLOCK, 0))?;
    module.safe_tile(&stream, &safe_tile_launch, &mut safe_tiles)?;
    // SAFETY: `raw_tiles` owns exactly N live device u32 elements and remains
    // alive until after stream synchronization in `to_host_vec`.
    unsafe {
        module.raw_tile(
            &stream,
            &raw_tile_launch,
            raw_tiles.cu_deviceptr() as *mut u32,
            N,
        )?;
    }

    let safe_tiles = safe_tiles.to_host_vec(&stream)?;
    let raw_tiles = raw_tiles.to_host_vec(&stream)?;
    assert_eq!(safe_tiles, raw_tiles);
    assert!(safe_tiles.iter().enumerate().all(|(i, &value)| {
        let lane = (i % 4) as u32;
        value == i as u32 + lane + 1
    }));

    let epilogue_input: Vec<f32> = (0..N)
        .map(|index| index as f32 - (N as f32 / 2.0))
        .collect();
    let mut safe_epilogue = DeviceBuffer::from_host(&stream, &epilogue_input)?;
    let raw_epilogue = DeviceBuffer::from_host(&stream, &epilogue_input)?;
    let epilogue_config = LaunchConfig2D::new(
        (
            EPILOGUE_STRIDE / EPILOGUE_COLS_PER_THREAD / 8,
            EPILOGUE_ROWS / 8,
        ),
        (8, 8),
        0,
    );
    let safe_epilogue_launch = module.prepare_safe_epilogue(epilogue_config)?;
    let raw_epilogue_launch = module.prepare_raw_epilogue(epilogue_config)?;
    module.safe_epilogue(&stream, &safe_epilogue_launch, &mut safe_epilogue)?;
    // SAFETY: `raw_epilogue` owns exactly N live device f32 elements and
    // remains alive until after stream synchronization in `to_host_vec`.
    unsafe {
        module.raw_epilogue(
            &stream,
            &raw_epilogue_launch,
            raw_epilogue.cu_deviceptr() as *mut f32,
            N,
        )?;
    }
    let safe_epilogue = safe_epilogue.to_host_vec(&stream)?;
    let raw_epilogue = raw_epilogue.to_host_vec(&stream)?;
    assert_eq!(safe_epilogue, raw_epilogue);
    assert!(
        safe_epilogue
            .iter()
            .enumerate()
            .all(|(index, &value)| { value == epilogue_input[index] + 1.0 })
    );

    println!("SUCCESS: safe proof-carrying views matched raw kernels");
    Ok(())
}

fn verify_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("proof_carrying_views.ptx");
    let ptx = std::fs::read_to_string(&path)?;

    for marker in [
        "__launch_contract_config",
        "__launch_bounds_config",
        "make_kernel_scope",
    ] {
        if ptx.contains(marker) {
            return Err(
                format!("compile-time marker `{marker}` leaked into the PTX module").into(),
            );
        }
    }

    let safe_element = entry_body(&ptx, "safe_element")?;
    let raw_element = entry_body(&ptx, "raw_element")?;
    let safe_tile = entry_body(&ptx, "safe_tile")?;
    let raw_tile = entry_body(&ptx, "raw_tile")?;
    let legacy_rank_guard = entry_body(&ptx, "legacy_rank_guard")?;
    let safe_epilogue = entry_body(&ptx, "safe_epilogue")?;
    let raw_epilogue = entry_body(&ptx, "raw_epilogue")?;

    for (name, body) in [
        ("safe_element", safe_element),
        ("raw_element", raw_element),
        ("safe_tile", safe_tile),
        ("raw_tile", raw_tile),
    ] {
        if !body.contains(".maxntid 64, 1, 1") {
            return Err(format!("{name} lost its 64-thread launch bound").into());
        }
        verify_u32_coordinates(name, body)?;
        verify_no_calls(name, body)?;
        let branches = conditional_branches(body);
        let expected_branches = 1;
        if branches != expected_branches {
            return Err(format!(
                "{name} has {branches} guard branches; expected {expected_branches}"
            )
            .into());
        }
        verify_no_interior_branches(name, body)?;
    }

    compare_memory_widths("element", safe_element, raw_element)?;
    compare_memory_widths("tile", safe_tile, raw_tile)?;
    compare_memory_operations("element", safe_element, raw_element)?;
    compare_memory_operations("tile", safe_tile, raw_tile)?;
    verify_legacy_rank_guard(legacy_rank_guard)?;

    for (name, body) in [
        ("safe_epilogue", safe_epilogue),
        ("raw_epilogue", raw_epilogue),
    ] {
        if !body.contains(".maxntid 64, 1, 1") {
            return Err(format!("{name} lost its 64-thread launch bound").into());
        }
        verify_u32_coordinates(name, body)?;
        verify_no_calls(name, body)?;
        for register in ["%ctaid.y", "%ntid.y", "%tid.y"] {
            if !body.contains(register) {
                return Err(format!("{name} does not read {register}").into());
            }
        }
        verify_no_interior_branches(name, body)?;
    }
    let safe_epilogue_branches = conditional_branches(safe_epilogue);
    let raw_epilogue_branches = conditional_branches(raw_epilogue);
    if safe_epilogue_branches != raw_epilogue_branches {
        return Err(format!(
            "epilogue guard branches differ: safe={safe_epilogue_branches}, raw={raw_epilogue_branches}"
        )
        .into());
    }
    compare_memory_widths("epilogue", safe_epilogue, raw_epilogue)?;
    compare_memory_operations("epilogue", safe_epilogue, raw_epilogue)?;

    println!("SUCCESS: proof-carrying views match raw PTX structure");
    Ok(())
}

fn verify_legacy_rank_guard(body: &str) -> Result<(), Box<dyn std::error::Error>> {
    for register in ["%ntid.y", "%nctaid.y", "%ntid.z", "%nctaid.z"] {
        if !body.contains(register) {
            return Err(format!("legacy 1D witness does not validate {register}").into());
        }
    }
    let first_store = body
        .lines()
        .position(|line| data_memory_width(line, "st").is_some())
        .ok_or("legacy rank-guard entry has no data store")?;
    if !body.lines().take(first_store).any(|line| {
        let line = line.trim_start();
        line.starts_with("@%p") && line.split_whitespace().any(|word| word == "bra")
    }) {
        return Err("legacy 1D store is not dominated by a rank guard".into());
    }
    Ok(())
}

fn compare_memory_operations(
    pair: &str,
    safe: &str,
    raw: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for operation in ["ld", "st"] {
        let safe_ops = data_memory_operations(safe, operation);
        let raw_ops = data_memory_operations(raw, operation);
        if safe_ops != raw_ops {
            return Err(format!(
                "{pair} {operation} operations differ: safe={safe_ops:?}, raw={raw_ops:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn data_memory_operations(body: &str, operation: &str) -> Vec<String> {
    let mut operations: Vec<String> = body
        .lines()
        .filter_map(|line| data_memory_operation(line, operation))
        .collect();
    operations.sort();
    operations
}

fn data_memory_operation(line: &str, operation: &str) -> Option<String> {
    let prefix = format!("{operation}.");
    let mnemonic = line
        .split_whitespace()
        .find(|word| word.starts_with(&prefix))?
        .trim_end_matches([';', ',']);
    if mnemonic.contains(".param.") || mnemonic.contains(".shared.") || mnemonic.contains(".local.")
    {
        return None;
    }
    Some(mnemonic.to_owned())
}

fn entry_body<'a>(ptx: &'a str, name: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    let start = ptx
        .find(&format!(".visible .entry {name}("))
        .ok_or_else(|| format!("missing PTX entry `{name}`"))?;
    let rest = &ptx[start..];
    let open = rest
        .find('{')
        .ok_or_else(|| format!("PTX entry `{name}` has no body"))?;
    let close = rest[open + 1..]
        .find("\n}")
        .map(|offset| open + 1 + offset + 2)
        .ok_or_else(|| format!("PTX entry `{name}` has no closing brace"))?;
    Ok(&rest[..close])
}

fn verify_u32_coordinates(name: &str, body: &str) -> Result<(), Box<dyn std::error::Error>> {
    for register in ["%ctaid.x", "%ntid.x", "%tid.x"] {
        if !body.contains(register) {
            return Err(format!("{name} does not read {register}").into());
        }
    }
    for forbidden in ["mul.lo.s64", "mul.lo.u64", "mad.lo.s64", "mad.lo.u64"] {
        if body.contains(forbidden) {
            return Err(
                format!("{name} widened coordinate arithmetic through `{forbidden}`").into(),
            );
        }
    }
    if !body.contains("mul.wide.u32") && !body.contains("cvt.u64.u32") {
        return Err(format!("{name} has no final u32-to-address widening operation").into());
    }
    Ok(())
}

fn conditional_branches(body: &str) -> usize {
    body.lines()
        .filter(|line| line.trim_start().starts_with('@') && is_branch_instruction(line))
        .count()
}

fn verify_no_calls(name: &str, body: &str) -> Result<(), Box<dyn std::error::Error>> {
    if body.lines().any(is_call_instruction) {
        return Err(format!("{name} contains an out-of-line device call").into());
    }
    Ok(())
}

fn is_call_instruction(line: &str) -> bool {
    line.split_whitespace()
        .any(|word| word == "call" || word.starts_with("call."))
}

fn is_branch_instruction(line: &str) -> bool {
    line.split_whitespace()
        .any(|word| word == "bra" || word.starts_with("bra."))
}

fn verify_no_interior_branches(name: &str, body: &str) -> Result<(), Box<dyn std::error::Error>> {
    let lines: Vec<&str> = body.lines().collect();
    let first_load = lines
        .iter()
        .position(|line| data_memory_width(line, "ld").is_some())
        .ok_or_else(|| format!("{name} has no data load"))?;
    if lines[first_load..]
        .iter()
        .any(|line| is_branch_instruction(line))
    {
        return Err(format!("{name} repeats a bounds branch inside its proven data range").into());
    }
    Ok(())
}

fn compare_memory_widths(
    pair: &str,
    safe: &str,
    raw: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    for operation in ["ld", "st"] {
        let safe_widths = data_memory_widths(safe, operation);
        let raw_widths = data_memory_widths(raw, operation);
        if safe_widths.is_empty() {
            return Err(format!("safe {pair} entry has no `{operation}` data operation").into());
        }
        if safe_widths != raw_widths {
            return Err(format!(
                "{pair} {operation} widths differ: safe={safe_widths:?}, raw={raw_widths:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn data_memory_widths(body: &str, operation: &str) -> Vec<String> {
    let mut widths: Vec<String> = body
        .lines()
        .filter_map(|line| data_memory_width(line, operation))
        .collect();
    widths.sort();
    widths
}

fn data_memory_width(line: &str, operation: &str) -> Option<String> {
    let prefix = format!("{operation}.");
    let mnemonic = line
        .split_whitespace()
        .find(|word| word.starts_with(&prefix))?
        .trim_end_matches([';', ',']);
    if mnemonic.contains(".param.") || mnemonic.contains(".shared.") || mnemonic.contains(".local.")
    {
        return None;
    }

    let parts: Vec<&str> = mnemonic.split('.').collect();
    let value_type = parts.last()?;
    let vector_width = parts
        .iter()
        .find(|part| part.starts_with('v') && part[1..].chars().all(|ch| ch.is_ascii_digit()));
    Some(match vector_width {
        Some(width) => format!("{width}.{value_type}"),
        None => (*value_type).to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ptx_control_flow_parser_handles_suffixes_and_inverted_predicates() {
        let ptx = "@!%p1 bra L1;\n@%p2 bra.uni L2;\nbra.uni L3;\ncall.uni helper;";
        assert_eq!(conditional_branches(ptx), 2);
        assert!(is_branch_instruction("bra.uni L3;"));
        assert!(is_call_instruction("call.uni helper;"));
    }
}
