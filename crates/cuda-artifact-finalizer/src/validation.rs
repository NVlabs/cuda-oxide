/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const ELF64_HEADER_LENGTH: usize = 64;
const ELF64_PROGRAM_HEADER_LENGTH: u16 = 56;
const ELF64_SECTION_HEADER_LENGTH: u16 = 64;
const ELF_VERSION_CURRENT: u32 = 1;
const CUDA_ABI_RUNTIME_JIT_LINK: u8 = 7;
const CUDA_12_TOOLKIT_VERSIONS: std::ops::RangeInclusive<u32> = 120..=129;
const ELF_SECTION_TYPE_SYMTAB: u32 = 2;
const ELF_SECTION_TYPE_DYNSYM: u32 = 11;
const ELF_SYMBOL_TYPE_FUNCTION: u8 = 2;
const CUDA_SYMBOL_OTHER_ENTRY: u8 = 0x10;
const ELF64_SYMBOL_LENGTH: u64 = 24;

/// Check that bytes are a complete 64-bit little-endian CUDA executable ELF.
///
/// This validates every declared table and file-backed range, so a truncated
/// ELF prefix cannot be accepted as a cubin.
pub fn is_valid_cubin(bytes: &[u8]) -> bool {
    if bytes.len() < ELF64_HEADER_LENGTH
        || !bytes.starts_with(b"\x7fELF")
        || bytes[4] != 2
        || bytes[5] != 1
        || bytes[6] != 1
        || read_u16(bytes, 16) != Some(2)
        || read_u16(bytes, 18) != Some(190)
        || !has_supported_cuda_elf_version(bytes[8], read_u32(bytes, 20))
        || read_u16(bytes, 52) != Some(ELF64_HEADER_LENGTH as u16)
    {
        return false;
    }

    let Some(program_offset) = read_u64(bytes, 32) else {
        return false;
    };
    let Some(section_offset) = read_u64(bytes, 40) else {
        return false;
    };
    let Some(program_entry_size) = read_u16(bytes, 54) else {
        return false;
    };
    let Some(program_count) = read_u16(bytes, 56) else {
        return false;
    };
    let Some(section_entry_size) = read_u16(bytes, 58) else {
        return false;
    };
    let Some(section_count) = read_u16(bytes, 60) else {
        return false;
    };
    let Some(section_name_index) = read_u16(bytes, 62) else {
        return false;
    };

    if program_count == u16::MAX || (program_count == 0 && section_count == 0) {
        return false;
    }
    let program_table = if program_count == 0 {
        if program_offset != 0 || !matches!(program_entry_size, 0 | ELF64_PROGRAM_HEADER_LENGTH) {
            return false;
        }
        None
    } else {
        if program_entry_size != ELF64_PROGRAM_HEADER_LENGTH {
            return false;
        }
        table_bounds(
            program_offset,
            program_entry_size,
            program_count,
            bytes.len(),
        )
    };
    let section_table = if section_count == 0 {
        if section_offset != 0
            || section_name_index != 0
            || !matches!(section_entry_size, 0 | ELF64_SECTION_HEADER_LENGTH)
        {
            return false;
        }
        None
    } else {
        if section_entry_size != ELF64_SECTION_HEADER_LENGTH
            || (section_name_index != 0 && section_name_index >= section_count)
        {
            return false;
        }
        table_bounds(
            section_offset,
            section_entry_size,
            section_count,
            bytes.len(),
        )
    };
    if program_count != 0 && program_table.is_none()
        || section_count != 0 && section_table.is_none()
    {
        return false;
    }

    let mut has_meaningful_contents = false;
    if let Some((start, _)) = program_table {
        for index in 0..usize::from(program_count) {
            let header = start + index * usize::from(program_entry_size);
            let Some(program_type) = read_u32(bytes, header) else {
                return false;
            };
            let Some(file_offset) = read_u64(bytes, header + 8) else {
                return false;
            };
            let Some(file_size) = read_u64(bytes, header + 32) else {
                return false;
            };
            let Some(memory_size) = read_u64(bytes, header + 40) else {
                return false;
            };
            if !file_range_is_valid(file_offset, file_size, bytes.len()) {
                return false;
            }
            if program_type == 1 {
                if file_size > memory_size {
                    return false;
                }
                has_meaningful_contents |= file_size != 0;
            }
        }
    }

    if let Some((start, _)) = section_table {
        for index in 0..usize::from(section_count) {
            let header = start + index * usize::from(section_entry_size);
            let Some(section_type) = read_u32(bytes, header + 4) else {
                return false;
            };
            let Some(file_offset) = read_u64(bytes, header + 24) else {
                return false;
            };
            let Some(file_size) = read_u64(bytes, header + 32) else {
                return false;
            };
            if section_type != 8 && !file_range_is_valid(file_offset, file_size, bytes.len()) {
                return false;
            }
            // Section zero is the mandatory null entry. A cubin needs actual
            // file-backed contents beyond an otherwise-valid ELF shell.
            if index != 0 && section_type != 0 && section_type != 8 && file_size != 0 {
                has_meaningful_contents = true;
            }
        }
    }

    has_meaningful_contents
}

/// Collect kernel entry names from linked PTX.
///
/// This is intentionally a small structural lexer rather than a substring
/// search: comments and quoted strings cannot satisfy the final-link guard.
pub(crate) fn ptx_kernel_entries(bytes: &[u8]) -> Option<Vec<String>> {
    let bytes = bytes.strip_suffix(b"\0").unwrap_or(bytes);
    if bytes.contains(&0) {
        return None;
    }
    let text = std::str::from_utf8(bytes).ok()?;
    let tokens = ptx_tokens(text)?;
    let mut entries = Vec::new();
    let mut index = 0;
    while index < tokens.len() {
        if tokens[index] == ".entry" {
            let name = tokens.get(index + 1)?;
            if name.is_empty() {
                return None;
            }
            entries.push((*name).to_string());
            index += 2;
        } else {
            index += 1;
        }
    }
    entries.sort();
    entries.dedup();
    Some(entries)
}

fn ptx_tokens(text: &str) -> Option<Vec<&str>> {
    let bytes = text.as_bytes();
    let mut tokens = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b if b.is_ascii_whitespace() => index += 1,
            b'/' if bytes.get(index + 1) == Some(&b'/') => {
                index += 2;
                while index < bytes.len() && bytes[index] != b'\n' {
                    index += 1;
                }
            }
            b'/' if bytes.get(index + 1) == Some(&b'*') => {
                index += 2;
                let mut closed = false;
                while index + 1 < bytes.len() {
                    if bytes[index] == b'*' && bytes[index + 1] == b'/' {
                        index += 2;
                        closed = true;
                        break;
                    }
                    index += 1;
                }
                if !closed {
                    return None;
                }
            }
            b'"' => {
                index += 1;
                let mut escaped = false;
                let mut closed = false;
                while index < bytes.len() {
                    let byte = bytes[index];
                    index += 1;
                    if escaped {
                        escaped = false;
                    } else if byte == b'\\' {
                        escaped = true;
                    } else if byte == b'"' {
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return None;
                }
            }
            b'(' | b')' | b'{' | b'}' | b',' | b';' => index += 1,
            _ => {
                let start = index;
                while index < bytes.len() {
                    match bytes[index] {
                        b if b.is_ascii_whitespace() => break,
                        b'(' | b')' | b'{' | b'}' | b',' | b';' | b'"' => break,
                        b'/' if matches!(bytes.get(index + 1), Some(b'/') | Some(b'*')) => {
                            break;
                        }
                        _ => index += 1,
                    }
                }
                if start == index {
                    return None;
                }
                tokens.push(&text[start..index]);
            }
        }
    }
    Some(tokens)
}

/// Collect CUDA kernel entry symbols from a linked cubin.
///
/// nvdisasm represents kernel entries as defined `STT_FUNC` symbols carrying
/// `STO_CUDA_ENTRY` in `st_other`. Walking `.symtab`/`.dynsym` therefore
/// distinguishes real entries from device functions and incidental strings.
pub(crate) fn cubin_kernel_entries(bytes: &[u8]) -> Option<Vec<String>> {
    if !is_valid_cubin(bytes) {
        return None;
    }
    let section_offset = usize::try_from(read_u64(bytes, 40)?).ok()?;
    let section_entry_size = usize::from(read_u16(bytes, 58)?);
    let section_count = usize::from(read_u16(bytes, 60)?);
    if section_count == 0 || section_entry_size != usize::from(ELF64_SECTION_HEADER_LENGTH) {
        return None;
    }

    let mut entries = Vec::new();
    let mut found_symbol_table = false;
    for section_index in 0..section_count {
        let header = section_offset.checked_add(section_index.checked_mul(section_entry_size)?)?;
        let section_type = read_u32(bytes, header + 4)?;
        if !matches!(
            section_type,
            ELF_SECTION_TYPE_SYMTAB | ELF_SECTION_TYPE_DYNSYM
        ) {
            continue;
        }
        found_symbol_table = true;
        let symbols_offset = usize::try_from(read_u64(bytes, header + 24)?).ok()?;
        let symbols_size = usize::try_from(read_u64(bytes, header + 32)?).ok()?;
        let string_section_index = usize::try_from(read_u32(bytes, header + 40)?).ok()?;
        let symbol_entry_size = read_u64(bytes, header + 56)?;
        if symbol_entry_size < ELF64_SYMBOL_LENGTH
            || symbols_size % usize::try_from(symbol_entry_size).ok()? != 0
            || string_section_index >= section_count
        {
            return None;
        }
        let strings_header =
            section_offset.checked_add(string_section_index.checked_mul(section_entry_size)?)?;
        let strings_offset = usize::try_from(read_u64(bytes, strings_header + 24)?).ok()?;
        let strings_size = usize::try_from(read_u64(bytes, strings_header + 32)?).ok()?;
        let strings = bytes.get(strings_offset..strings_offset.checked_add(strings_size)?)?;
        let symbol_entry_size = usize::try_from(symbol_entry_size).ok()?;
        let symbols = bytes.get(symbols_offset..symbols_offset.checked_add(symbols_size)?)?;
        for symbol in symbols.chunks_exact(symbol_entry_size) {
            let name_offset = usize::try_from(read_u32(symbol, 0)?).ok()?;
            let info = *symbol.get(4)?;
            let other = *symbol.get(5)?;
            let section_index = read_u16(symbol, 6)?;
            if info & 0x0f != ELF_SYMBOL_TYPE_FUNCTION
                || other & CUDA_SYMBOL_OTHER_ENTRY == 0
                || section_index == 0
                || name_offset == 0
            {
                continue;
            }
            let name = read_c_string(strings, name_offset)?;
            if !name.is_empty() {
                entries.push(name.to_string());
            }
        }
    }
    if !found_symbol_table {
        return None;
    }
    entries.sort();
    entries.dedup();
    Some(entries)
}

fn read_c_string(bytes: &[u8], offset: usize) -> Option<&str> {
    let tail = bytes.get(offset..)?;
    let length = tail.iter().position(|&byte| byte == 0)?;
    std::str::from_utf8(&tail[..length]).ok()
}

fn has_supported_cuda_elf_version(abi_version: u8, file_version: Option<u32>) -> bool {
    match file_version {
        // Preserve the generic ELF version accepted before CUDA toolkit
        // encodings were recognized, regardless of the CUDA ABI revision.
        Some(ELF_VERSION_CURRENT) => true,
        // CUDA ABI 7 identifies the runtime-JIT-link format. CUDA 12 producers
        // encode their toolkit release as major * 10 + minor in e_version;
        // for example, cuobjdump renders 128 (0x80) as toolkit 12.8.
        Some(version) if abi_version == CUDA_ABI_RUNTIME_JIT_LINK => {
            CUDA_12_TOOLKIT_VERSIONS.contains(&version)
        }
        _ => false,
    }
}

fn table_bounds(
    offset: u64,
    entry_size: u16,
    count: u16,
    file_len: usize,
) -> Option<(usize, usize)> {
    if offset < ELF64_HEADER_LENGTH as u64 {
        return None;
    }
    let length = u64::from(entry_size).checked_mul(u64::from(count))?;
    let end = offset.checked_add(length)?;
    if end > file_len as u64 {
        return None;
    }
    Some((usize::try_from(offset).ok()?, usize::try_from(end).ok()?))
}

fn file_range_is_valid(offset: u64, length: u64, file_len: usize) -> bool {
    offset
        .checked_add(length)
        .is_some_and(|end| end <= file_len as u64)
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_cubin() -> Vec<u8> {
        const PAYLOAD_LENGTH: usize = 4;
        let section_table_length = 2 * usize::from(ELF64_SECTION_HEADER_LENGTH);
        let payload_offset = ELF64_HEADER_LENGTH + section_table_length;
        let mut bytes = vec![0; payload_offset + PAYLOAD_LENGTH];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[16..18].copy_from_slice(&2_u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&190_u16.to_le_bytes());
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
        bytes[40..48].copy_from_slice(&(ELF64_HEADER_LENGTH as u64).to_le_bytes());
        bytes[52..54].copy_from_slice(&(ELF64_HEADER_LENGTH as u16).to_le_bytes());
        bytes[58..60].copy_from_slice(&ELF64_SECTION_HEADER_LENGTH.to_le_bytes());
        bytes[60..62].copy_from_slice(&2_u16.to_le_bytes());
        let section = ELF64_HEADER_LENGTH + usize::from(ELF64_SECTION_HEADER_LENGTH);
        bytes[section + 4..section + 8].copy_from_slice(&1_u32.to_le_bytes());
        bytes[section + 24..section + 32].copy_from_slice(&(payload_offset as u64).to_le_bytes());
        bytes[section + 32..section + 40].copy_from_slice(&(PAYLOAD_LENGTH as u64).to_le_bytes());
        bytes[payload_offset..].copy_from_slice(b"CUDA");
        bytes
    }

    fn program_only_cubin(memory_size: u64) -> Vec<u8> {
        const PAYLOAD_LENGTH: usize = 4;
        let payload_offset = ELF64_HEADER_LENGTH + usize::from(ELF64_PROGRAM_HEADER_LENGTH);
        let mut bytes = vec![0; payload_offset + PAYLOAD_LENGTH];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[16..18].copy_from_slice(&2_u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&190_u16.to_le_bytes());
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
        bytes[32..40].copy_from_slice(&(ELF64_HEADER_LENGTH as u64).to_le_bytes());
        bytes[52..54].copy_from_slice(&(ELF64_HEADER_LENGTH as u16).to_le_bytes());
        bytes[54..56].copy_from_slice(&ELF64_PROGRAM_HEADER_LENGTH.to_le_bytes());
        bytes[56..58].copy_from_slice(&1_u16.to_le_bytes());
        let program = ELF64_HEADER_LENGTH;
        bytes[program..program + 4].copy_from_slice(&1_u32.to_le_bytes());
        bytes[program + 8..program + 16].copy_from_slice(&(payload_offset as u64).to_le_bytes());
        bytes[program + 32..program + 40].copy_from_slice(&(PAYLOAD_LENGTH as u64).to_le_bytes());
        bytes[program + 40..program + 48].copy_from_slice(&memory_size.to_le_bytes());
        bytes[payload_offset..].copy_from_slice(b"CUDA");
        bytes
    }

    #[test]
    fn validates_complete_cuda_elf_and_rejects_truncation() {
        let cubin = minimal_cubin();
        assert!(is_valid_cubin(&cubin));
        assert!(!is_valid_cubin(&cubin[..cubin.len() - 1]));
        let mut shell =
            cubin[..ELF64_HEADER_LENGTH + usize::from(ELF64_SECTION_HEADER_LENGTH)].to_vec();
        shell[60..62].copy_from_slice(&1_u16.to_le_bytes());
        assert!(!is_valid_cubin(&shell));
        assert!(is_valid_cubin(&program_only_cubin(4)));
        assert!(!is_valid_cubin(&program_only_cubin(3)));
        assert!(!is_valid_cubin(b"\x7fELF"));
    }

    #[test]
    fn accepts_cuda_12_runtime_jit_link_versions_without_weakening_other_abis() {
        let mut cubin = minimal_cubin();
        let cases: &[(u8, u32, bool)] = &[
            (0, 1, true),
            (CUDA_ABI_RUNTIME_JIT_LINK, 1, true),
            (8, 1, true),
            (u8::MAX, 1, true),
            (CUDA_ABI_RUNTIME_JIT_LINK, 120, true),
            (CUDA_ABI_RUNTIME_JIT_LINK, 128, true),
            (CUDA_ABI_RUNTIME_JIT_LINK, 129, true),
            (CUDA_ABI_RUNTIME_JIT_LINK, 0, false),
            (CUDA_ABI_RUNTIME_JIT_LINK, 2, false),
            (CUDA_ABI_RUNTIME_JIT_LINK, 119, false),
            (CUDA_ABI_RUNTIME_JIT_LINK, 130, false),
            (CUDA_ABI_RUNTIME_JIT_LINK, u32::MAX, false),
            (6, 128, false),
            (8, 128, false),
            (u8::MAX, 128, false),
        ];
        for &(abi_version, file_version, expected) in cases {
            cubin[8] = abi_version;
            cubin[20..24].copy_from_slice(&file_version.to_le_bytes());
            assert_eq!(
                is_valid_cubin(&cubin),
                expected,
                "ABI version {abi_version}, file version {file_version}"
            );
        }
    }

    fn cubin_with_kernel_symbols(symbols: &[(&str, bool)]) -> Vec<u8> {
        const SECTION_COUNT: usize = 4;
        let section_table_length = SECTION_COUNT * usize::from(ELF64_SECTION_HEADER_LENGTH);
        let strings = {
            let mut strings = vec![0_u8];
            for (name, _) in symbols {
                strings.extend_from_slice(name.as_bytes());
                strings.push(0);
            }
            strings
        };
        let symbol_table_length = (symbols.len() + 1) * ELF64_SYMBOL_LENGTH as usize;
        let strings_offset = ELF64_HEADER_LENGTH + section_table_length;
        let symbols_offset = strings_offset + strings.len();
        let text_offset = symbols_offset + symbol_table_length;
        let mut bytes = vec![0_u8; text_offset + 4];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[6] = 1;
        bytes[16..18].copy_from_slice(&2_u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&190_u16.to_le_bytes());
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
        bytes[40..48].copy_from_slice(&(ELF64_HEADER_LENGTH as u64).to_le_bytes());
        bytes[52..54].copy_from_slice(&(ELF64_HEADER_LENGTH as u16).to_le_bytes());
        bytes[58..60].copy_from_slice(&ELF64_SECTION_HEADER_LENGTH.to_le_bytes());
        bytes[60..62].copy_from_slice(&(SECTION_COUNT as u16).to_le_bytes());

        let strtab = ELF64_HEADER_LENGTH + usize::from(ELF64_SECTION_HEADER_LENGTH);
        bytes[strtab + 4..strtab + 8].copy_from_slice(&3_u32.to_le_bytes());
        bytes[strtab + 24..strtab + 32].copy_from_slice(&(strings_offset as u64).to_le_bytes());
        bytes[strtab + 32..strtab + 40].copy_from_slice(&(strings.len() as u64).to_le_bytes());

        let symtab = strtab + usize::from(ELF64_SECTION_HEADER_LENGTH);
        bytes[symtab + 4..symtab + 8].copy_from_slice(&ELF_SECTION_TYPE_SYMTAB.to_le_bytes());
        bytes[symtab + 24..symtab + 32].copy_from_slice(&(symbols_offset as u64).to_le_bytes());
        bytes[symtab + 32..symtab + 40]
            .copy_from_slice(&(symbol_table_length as u64).to_le_bytes());
        bytes[symtab + 40..symtab + 44].copy_from_slice(&1_u32.to_le_bytes());
        bytes[symtab + 56..symtab + 64].copy_from_slice(&ELF64_SYMBOL_LENGTH.to_le_bytes());

        let text = symtab + usize::from(ELF64_SECTION_HEADER_LENGTH);
        bytes[text + 4..text + 8].copy_from_slice(&1_u32.to_le_bytes());
        bytes[text + 24..text + 32].copy_from_slice(&(text_offset as u64).to_le_bytes());
        bytes[text + 32..text + 40].copy_from_slice(&4_u64.to_le_bytes());
        bytes[text_offset..text_offset + 4].copy_from_slice(b"CUDA");
        bytes[strings_offset..strings_offset + strings.len()].copy_from_slice(&strings);

        let mut name_offset = 1_u32;
        for (index, (name, is_kernel)) in symbols.iter().enumerate() {
            let symbol = symbols_offset + (index + 1) * ELF64_SYMBOL_LENGTH as usize;
            bytes[symbol..symbol + 4].copy_from_slice(&name_offset.to_le_bytes());
            bytes[symbol + 4] = 0x12;
            bytes[symbol + 5] = if *is_kernel {
                CUDA_SYMBOL_OTHER_ENTRY
            } else {
                0
            };
            bytes[symbol + 6..symbol + 8].copy_from_slice(&3_u16.to_le_bytes());
            name_offset += u32::try_from(name.len() + 1).unwrap();
        }
        bytes
    }

    #[test]
    fn ptx_entry_inventory_ignores_comments_and_strings() {
        let ptx = br#"
.version 8.9
// .entry fake_comment() {}
.const .b8 text[] = {0}; // ".entry fake_string"
.visible .entry kernel_a() { ret; }
.entry
kernel_b() { ret; }
"#;
        assert_eq!(
            ptx_kernel_entries(ptx).unwrap(),
            vec!["kernel_a".to_string(), "kernel_b".to_string()]
        );
        assert!(ptx_kernel_entries(b".entry kernel() {}\0").is_some());
        assert!(ptx_kernel_entries(b".entry kernel() {}\0garbage").is_none());
    }

    #[test]
    fn cubin_entry_inventory_uses_cuda_entry_symbol_flag() {
        let cubin = cubin_with_kernel_symbols(&[("kernel_a", true), ("helper", false)]);
        assert!(is_valid_cubin(&cubin));
        assert_eq!(cubin_kernel_entries(&cubin).unwrap(), vec!["kernel_a"]);
    }
}
