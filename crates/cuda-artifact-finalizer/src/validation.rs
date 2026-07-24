/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

const ELF64_HEADER_LENGTH: usize = 64;
const ELF64_PROGRAM_HEADER_LENGTH: u16 = 56;
const ELF64_SECTION_HEADER_LENGTH: u16 = 64;

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
        || read_u32(bytes, 20) != Some(1)
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
}
