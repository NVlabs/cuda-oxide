/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Embedded device artifact bundles.
//!
//! The wire format is intentionally independent of a particular accelerator
//! backend. A bundle names a producer, records the device target it was built
//! for, and carries one or more generated device-code payloads.

use core::fmt;

pub const ARTIFACT_SECTION_NAME: &str = ".oxart";
pub const ARTIFACT_MAGIC: [u8; 8] = *b"OXIDEART";
pub const ARTIFACT_VERSION: u16 = 1;

const HEADER_BYTES: usize = 32;
const PAYLOAD_RECORD_BYTES: usize = 24;
const ENTRY_RECORD_BYTES: usize = 24;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArtifactPayloadKind {
    Ptx,
    NvvmIr,
    Ltoir,
    Cubin,
}

impl ArtifactPayloadKind {
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::Ptx => 0x100,
            Self::NvvmIr => 0x110,
            Self::Ltoir => 0x120,
            Self::Cubin => 0x200,
        }
    }

    pub const fn from_u16(value: u16) -> Option<Self> {
        match value {
            0x100 => Some(Self::Ptx),
            0x110 => Some(Self::NvvmIr),
            0x120 => Some(Self::Ltoir),
            0x200 => Some(Self::Cubin),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArtifactEntryKind {
    Kernel,
    DeviceFunction,
}

impl ArtifactEntryKind {
    pub const fn to_u16(self) -> u16 {
        match self {
            Self::Kernel => 1,
            Self::DeviceFunction => 2,
        }
    }

    pub const fn from_u16(value: u16) -> Option<Self> {
        match value {
            1 => Some(Self::Kernel),
            2 => Some(Self::DeviceFunction),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactPayloadSpec<'a> {
    pub kind: ArtifactPayloadKind,
    pub name: &'a str,
    pub bytes: &'a [u8],
}

impl<'a> ArtifactPayloadSpec<'a> {
    pub const fn new(kind: ArtifactPayloadKind, name: &'a str, bytes: &'a [u8]) -> Self {
        Self { kind, name, bytes }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactEntrySpec<'a> {
    pub symbol: &'a str,
    pub kind: ArtifactEntryKind,
    pub metadata: Option<u64>,
}

impl<'a> ArtifactEntrySpec<'a> {
    pub const fn new(symbol: &'a str, kind: ArtifactEntryKind) -> Self {
        Self {
            symbol,
            kind,
            metadata: None,
        }
    }

    pub const fn with_metadata(mut self, metadata: u64) -> Self {
        self.metadata = Some(metadata);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactBundleSpec<'a> {
    pub name: &'a str,
    pub target: &'a str,
    pub payloads: Vec<ArtifactPayloadSpec<'a>>,
    pub entries: Vec<ArtifactEntrySpec<'a>>,
}

impl<'a> ArtifactBundleSpec<'a> {
    pub fn new(name: &'a str, target: &'a str) -> Self {
        Self {
            name,
            target,
            payloads: Vec::new(),
            entries: Vec::new(),
        }
    }

    pub fn with_payload(mut self, payload: ArtifactPayloadSpec<'a>) -> Self {
        self.payloads.push(payload);
        self
    }

    pub fn with_entry(mut self, entry: ArtifactEntrySpec<'a>) -> Self {
        self.entries.push(entry);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactPayload<'a> {
    pub kind: ArtifactPayloadKind,
    pub name: &'a str,
    pub bytes: &'a [u8],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactEntry<'a> {
    pub symbol: &'a str,
    pub kind: ArtifactEntryKind,
    pub metadata: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactBundle<'a> {
    pub name: &'a str,
    pub target: &'a str,
    pub payloads: Vec<ArtifactPayload<'a>>,
    pub entries: Vec<ArtifactEntry<'a>>,
}

impl<'a> ArtifactBundle<'a> {
    pub fn payload(&self, kind: ArtifactPayloadKind) -> Option<&'a [u8]> {
        self.payloads
            .iter()
            .find(|payload| payload.kind == kind)
            .map(|payload| payload.bytes)
    }

    pub fn entry(&self, symbol: &str) -> Option<&ArtifactEntry<'a>> {
        self.entries.iter().find(|entry| entry.symbol == symbol)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedArtifactPayload {
    pub kind: ArtifactPayloadKind,
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedArtifactEntry {
    pub symbol: String,
    pub kind: ArtifactEntryKind,
    pub metadata: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedArtifactBundle {
    pub name: String,
    pub target: String,
    pub payloads: Vec<OwnedArtifactPayload>,
    pub entries: Vec<OwnedArtifactEntry>,
}

impl OwnedArtifactBundle {
    pub fn payload(&self, kind: ArtifactPayloadKind) -> Option<&[u8]> {
        self.payloads
            .iter()
            .find(|payload| payload.kind == kind)
            .map(|payload| payload.bytes.as_slice())
    }

    pub fn entry(&self, symbol: &str) -> Option<&OwnedArtifactEntry> {
        self.entries.iter().find(|entry| entry.symbol == symbol)
    }
}

impl<'a> From<ArtifactBundle<'a>> for OwnedArtifactBundle {
    fn from(bundle: ArtifactBundle<'a>) -> Self {
        Self {
            name: bundle.name.to_string(),
            target: bundle.target.to_string(),
            payloads: bundle
                .payloads
                .into_iter()
                .map(|payload| OwnedArtifactPayload {
                    kind: payload.kind,
                    name: payload.name.to_string(),
                    bytes: payload.bytes.to_vec(),
                })
                .collect(),
            entries: bundle
                .entries
                .into_iter()
                .map(|entry| OwnedArtifactEntry {
                    symbol: entry.symbol.to_string(),
                    kind: entry.kind,
                    metadata: entry.metadata,
                })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArtifactError {
    TooLarge(&'static str),
    EmptyBundleName,
    EmptyTarget,
    EmptyPayloadName,
    EmptyPayload,
    EmptyEntrySymbol,
    Truncated(&'static str),
    BadMagic,
    UnsupportedVersion(u16),
    UnsupportedPayloadKind(u16),
    UnsupportedEntryKind(u16),
    InvalidUtf8(&'static str),
    UnsupportedHostTarget(String),
    Object(String),
    Malformed(String),
}

impl fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge(field) => write!(f, "embedded artifact {field} is too large"),
            Self::EmptyBundleName => f.write_str("embedded artifact bundle name is empty"),
            Self::EmptyTarget => f.write_str("embedded artifact target is empty"),
            Self::EmptyPayloadName => f.write_str("embedded artifact payload name is empty"),
            Self::EmptyPayload => f.write_str("embedded artifact payload is empty"),
            Self::EmptyEntrySymbol => f.write_str("embedded artifact entry symbol is empty"),
            Self::Truncated(field) => write!(f, "embedded artifact is truncated in {field}"),
            Self::BadMagic => f.write_str("embedded artifact has bad magic"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported embedded artifact version {version}")
            }
            Self::UnsupportedPayloadKind(kind) => {
                write!(f, "unsupported embedded artifact payload kind {kind}")
            }
            Self::UnsupportedEntryKind(kind) => {
                write!(f, "unsupported embedded artifact entry kind {kind}")
            }
            Self::InvalidUtf8(field) => write!(f, "embedded artifact {field} is not utf-8"),
            Self::UnsupportedHostTarget(target) => {
                write!(f, "unsupported host object target '{target}'")
            }
            Self::Object(message) | Self::Malformed(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for ArtifactError {}

pub fn build_artifact_blob(spec: &ArtifactBundleSpec<'_>) -> Result<Vec<u8>, ArtifactError> {
    validate_spec(spec)?;

    let mut out = vec![0; HEADER_BYTES];
    push_bytes(&mut out, spec.name.as_bytes());
    push_bytes(&mut out, spec.target.as_bytes());

    let payload_record_start = out.len();
    out.resize(out.len() + spec.payloads.len() * PAYLOAD_RECORD_BYTES, 0);
    let entry_record_start = out.len();
    out.resize(out.len() + spec.entries.len() * ENTRY_RECORD_BYTES, 0);

    for (index, payload) in spec.payloads.iter().enumerate() {
        let name_offset = checked_u32(out.len(), "payload name offset")?;
        push_bytes(&mut out, payload.name.as_bytes());
        align_vec(&mut out, 8);
        let data_offset = checked_u32(out.len(), "payload data offset")?;
        push_bytes(&mut out, payload.bytes);
        align_vec(&mut out, 8);

        let record = payload_record_start + index * PAYLOAD_RECORD_BYTES;
        write_u16(&mut out, record, payload.kind.to_u16());
        write_u16(&mut out, record + 2, 0);
        write_u32(&mut out, record + 4, data_offset);
        write_u32(
            &mut out,
            record + 8,
            checked_u32(payload.bytes.len(), "payload length")?,
        );
        write_u32(&mut out, record + 12, name_offset);
        write_u16(
            &mut out,
            record + 16,
            checked_u16(payload.name.len(), "payload name length")?,
        );
    }

    for (index, entry) in spec.entries.iter().enumerate() {
        let symbol_offset = checked_u32(out.len(), "entry symbol offset")?;
        push_bytes(&mut out, entry.symbol.as_bytes());
        align_vec(&mut out, 8);

        let record = entry_record_start + index * ENTRY_RECORD_BYTES;
        write_u16(&mut out, record, entry.kind.to_u16());
        write_u16(&mut out, record + 2, u16::from(entry.metadata.is_some()));
        write_u64(&mut out, record + 4, entry.metadata.unwrap_or(0));
        write_u32(&mut out, record + 12, symbol_offset);
        write_u16(
            &mut out,
            record + 16,
            checked_u16(entry.symbol.len(), "entry symbol length")?,
        );
    }

    let total_len = checked_u32(out.len(), "total length")?;
    out[0..8].copy_from_slice(&ARTIFACT_MAGIC);
    write_u16(&mut out, 8, ARTIFACT_VERSION);
    write_u16(&mut out, 10, HEADER_BYTES as u16);
    write_u32(&mut out, 12, total_len);
    write_u16(&mut out, 16, checked_u16(spec.name.len(), "name length")?);
    write_u16(
        &mut out,
        18,
        checked_u16(spec.target.len(), "target length")?,
    );
    write_u16(
        &mut out,
        20,
        checked_u16(spec.payloads.len(), "payload count")?,
    );
    write_u16(
        &mut out,
        22,
        checked_u16(spec.entries.len(), "entry count")?,
    );

    Ok(out)
}

pub fn parse_artifact_section(mut bytes: &[u8]) -> Result<Vec<ArtifactBundle<'_>>, ArtifactError> {
    let mut bundles = Vec::new();
    while !bytes.is_empty() {
        if bytes.iter().all(|byte| *byte == 0) {
            break;
        }
        let total_len = artifact_blob_total_len(bytes)?;
        let (blob, rest) = bytes.split_at(total_len);
        bundles.push(parse_artifact_blob(blob)?);
        bytes = rest;
    }
    Ok(bundles)
}

pub fn parse_artifact_blob(bytes: &[u8]) -> Result<ArtifactBundle<'_>, ArtifactError> {
    require_len(bytes, HEADER_BYTES, "header")?;
    if bytes[0..8] != ARTIFACT_MAGIC {
        return Err(ArtifactError::BadMagic);
    }
    let version = read_u16(bytes, 8)?;
    if version != ARTIFACT_VERSION {
        return Err(ArtifactError::UnsupportedVersion(version));
    }
    let header_len = read_u16(bytes, 10)? as usize;
    if header_len != HEADER_BYTES {
        return Err(ArtifactError::Malformed(format!(
            "unsupported embedded artifact header length {header_len}"
        )));
    }
    let total_len = read_u32(bytes, 12)? as usize;
    if total_len < HEADER_BYTES {
        return Err(ArtifactError::Malformed(format!(
            "embedded artifact length {total_len} is smaller than the header"
        )));
    }
    if total_len > bytes.len() {
        return Err(ArtifactError::Truncated("blob"));
    }
    let bytes = &bytes[..total_len];
    let name_len = read_u16(bytes, 16)? as usize;
    let target_len = read_u16(bytes, 18)? as usize;
    let payload_count = read_u16(bytes, 20)? as usize;
    let entry_count = read_u16(bytes, 22)? as usize;

    let mut cursor = HEADER_BYTES;
    let name = read_str(bytes, cursor, name_len, "bundle name")?;
    cursor += name_len;
    let target = read_str(bytes, cursor, target_len, "target")?;
    cursor += target_len;

    let payload_records = cursor;
    cursor = cursor
        .checked_add(payload_count * PAYLOAD_RECORD_BYTES)
        .ok_or(ArtifactError::TooLarge("payload records"))?;
    require_len(bytes, cursor, "payload records")?;
    let entry_records = cursor;
    cursor = cursor
        .checked_add(entry_count * ENTRY_RECORD_BYTES)
        .ok_or(ArtifactError::TooLarge("entry records"))?;
    require_len(bytes, cursor, "entry records")?;

    let mut payloads = Vec::with_capacity(payload_count);
    for index in 0..payload_count {
        let record = payload_records + index * PAYLOAD_RECORD_BYTES;
        let kind_raw = read_u16(bytes, record)?;
        let kind = ArtifactPayloadKind::from_u16(kind_raw)
            .ok_or(ArtifactError::UnsupportedPayloadKind(kind_raw))?;
        let data_offset = read_u32(bytes, record + 4)? as usize;
        let data_len = read_u32(bytes, record + 8)? as usize;
        let name_offset = read_u32(bytes, record + 12)? as usize;
        let name_len = read_u16(bytes, record + 16)? as usize;
        let name = read_str(bytes, name_offset, name_len, "payload name")?;
        let data = read_slice(bytes, data_offset, data_len, "payload data")?;
        payloads.push(ArtifactPayload {
            kind,
            name,
            bytes: data,
        });
    }

    let mut entries = Vec::with_capacity(entry_count);
    for index in 0..entry_count {
        let record = entry_records + index * ENTRY_RECORD_BYTES;
        let kind_raw = read_u16(bytes, record)?;
        let kind = ArtifactEntryKind::from_u16(kind_raw)
            .ok_or(ArtifactError::UnsupportedEntryKind(kind_raw))?;
        let flags = read_u16(bytes, record + 2)?;
        let metadata = if flags & 1 != 0 {
            Some(read_u64(bytes, record + 4)?)
        } else {
            None
        };
        let symbol_offset = read_u32(bytes, record + 12)? as usize;
        let symbol_len = read_u16(bytes, record + 16)? as usize;
        let symbol = read_str(bytes, symbol_offset, symbol_len, "entry symbol")?;
        entries.push(ArtifactEntry {
            symbol,
            kind,
            metadata,
        });
    }

    Ok(ArtifactBundle {
        name,
        target,
        payloads,
        entries,
    })
}

pub fn artifact_blob_total_len(bytes: &[u8]) -> Result<usize, ArtifactError> {
    require_len(bytes, HEADER_BYTES, "header")?;
    if bytes[0..8] != ARTIFACT_MAGIC {
        return Err(ArtifactError::BadMagic);
    }
    let total_len = read_u32(bytes, 12)? as usize;
    if total_len < HEADER_BYTES {
        return Err(ArtifactError::Malformed(format!(
            "embedded artifact length {total_len} is smaller than the header"
        )));
    }
    if total_len > bytes.len() {
        return Err(ArtifactError::Truncated("blob"));
    }
    Ok(total_len)
}

#[cfg(feature = "object-read")]
pub fn read_artifact_bundles_from_object_bytes(
    bytes: &[u8],
) -> Result<Vec<OwnedArtifactBundle>, ArtifactError> {
    use object::{Object, ObjectSection};

    let file = object::File::parse(bytes).map_err(|e| ArtifactError::Object(e.to_string()))?;
    let mut bundles = Vec::new();
    for section in file.sections() {
        let name = section
            .name()
            .map_err(|e| ArtifactError::Object(e.to_string()))?;
        if name != ARTIFACT_SECTION_NAME {
            continue;
        }
        let data = section
            .data()
            .map_err(|e| ArtifactError::Object(e.to_string()))?;
        bundles.extend(parse_artifact_section(data)?.into_iter().map(Into::into));
    }
    Ok(bundles)
}

/// Wrap an artifact section blob in a relocatable host object file.
///
/// The object contains a single `.oxart` data section. When
/// `anchor_symbol` is given, a global symbol with that name is defined at
/// the start of the section. The anchor matters for *library* crates:
/// their artifact object becomes a member of an `.rlib` archive, and a
/// linker only extracts an archive member when the member defines a
/// symbol that resolves an outstanding undefined reference. Without a
/// defined symbol the member is silently skipped and the bundle never
/// reaches the final binary. Host-side code (the `#[cuda_module]` macro)
/// emits a matching reference to the anchor to force the extraction.
/// `SHF_GNU_RETAIN` on the section additionally protects it from
/// `--gc-sections` once the member has been linked in.
#[cfg(feature = "object-write")]
pub fn build_host_object_for_target(
    section_data: &[u8],
    target: &str,
    anchor_symbol: Option<&str>,
) -> Result<Vec<u8>, ArtifactError> {
    use object::write::{Object, Symbol, SymbolSection};
    use object::{SectionFlags, SectionKind, SymbolFlags, SymbolKind, SymbolScope};

    if section_data.is_empty() {
        return Err(ArtifactError::EmptyPayload);
    }

    let target = HostObjectTarget::parse(target)?;
    let mut object = Object::new(target.format, target.architecture, target.endianness);
    let section_id = object.add_section(
        Vec::new(),
        ARTIFACT_SECTION_NAME.as_bytes().to_vec(),
        SectionKind::Data,
    );
    let section = object.section_mut(section_id);
    section.set_data(section_data.to_vec(), 8);
    section.flags = SectionFlags::Elf {
        sh_flags: elf::SHF_ALLOC | elf::SHF_GNU_RETAIN,
    };

    if let Some(anchor_symbol) = anchor_symbol {
        // Global binding so the symbol can satisfy undefined references
        // from other objects (that is what triggers archive extraction);
        // `Linkage` scope so it stays hidden and never leaks into the
        // dynamic symbol table of the final binary.
        object.add_symbol(Symbol {
            name: anchor_symbol.as_bytes().to_vec(),
            value: 0,
            size: 0,
            kind: SymbolKind::Data,
            scope: SymbolScope::Linkage,
            weak: false,
            section: SymbolSection::Section(section_id),
            flags: SymbolFlags::None,
        });
    }

    object
        .write()
        .map_err(|e| ArtifactError::Object(e.to_string()))
}

#[cfg(feature = "object-write")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HostObjectTarget {
    format: object::BinaryFormat,
    architecture: object::Architecture,
    endianness: object::Endianness,
}

#[cfg(feature = "object-write")]
impl HostObjectTarget {
    fn parse(target: &str) -> Result<Self, ArtifactError> {
        let target = target.to_ascii_lowercase();
        let (architecture, endianness) = if target.starts_with("x86_64")
            || target.starts_with("amd64")
            || target.starts_with("x86-64")
        {
            (object::Architecture::X86_64, object::Endianness::Little)
        } else if target.starts_with("aarch64") || target.starts_with("arm64") {
            (object::Architecture::Aarch64, object::Endianness::Little)
        } else {
            return Err(ArtifactError::UnsupportedHostTarget(target));
        };

        let format = if target.contains("linux") {
            object::BinaryFormat::Elf
        } else {
            return Err(ArtifactError::UnsupportedHostTarget(target));
        };

        Ok(Self {
            format,
            architecture,
            endianness,
        })
    }
}

#[cfg(feature = "object-write")]
mod elf {
    pub const SHF_ALLOC: u64 = 0x2;
    pub const SHF_GNU_RETAIN: u64 = 0x20_0000;
}

fn validate_spec(spec: &ArtifactBundleSpec<'_>) -> Result<(), ArtifactError> {
    if spec.name.is_empty() {
        return Err(ArtifactError::EmptyBundleName);
    }
    if spec.target.is_empty() {
        return Err(ArtifactError::EmptyTarget);
    }
    if spec.payloads.is_empty() {
        return Err(ArtifactError::EmptyPayload);
    }
    for payload in &spec.payloads {
        if payload.name.is_empty() {
            return Err(ArtifactError::EmptyPayloadName);
        }
        if payload.bytes.is_empty() {
            return Err(ArtifactError::EmptyPayload);
        }
    }
    for entry in &spec.entries {
        if entry.symbol.is_empty() {
            return Err(ArtifactError::EmptyEntrySymbol);
        }
    }
    Ok(())
}

fn checked_u16(value: usize, field: &'static str) -> Result<u16, ArtifactError> {
    u16::try_from(value).map_err(|_| ArtifactError::TooLarge(field))
}

fn checked_u32(value: usize, field: &'static str) -> Result<u32, ArtifactError> {
    u32::try_from(value).map_err(|_| ArtifactError::TooLarge(field))
}

fn push_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(bytes);
}

fn align_vec(out: &mut Vec<u8>, alignment: usize) {
    let rem = out.len() % alignment;
    if rem != 0 {
        out.resize(out.len() + alignment - rem, 0);
    }
}

fn read_slice<'a>(
    bytes: &'a [u8],
    offset: usize,
    len: usize,
    field: &'static str,
) -> Result<&'a [u8], ArtifactError> {
    let end = offset
        .checked_add(len)
        .ok_or(ArtifactError::TooLarge(field))?;
    bytes
        .get(offset..end)
        .ok_or(ArtifactError::Truncated(field))
}

fn read_str<'a>(
    bytes: &'a [u8],
    offset: usize,
    len: usize,
    field: &'static str,
) -> Result<&'a str, ArtifactError> {
    let bytes = read_slice(bytes, offset, len, field)?;
    core::str::from_utf8(bytes).map_err(|_| ArtifactError::InvalidUtf8(field))
}

fn require_len(bytes: &[u8], len: usize, field: &'static str) -> Result<(), ArtifactError> {
    if bytes.len() < len {
        Err(ArtifactError::Truncated(field))
    } else {
        Ok(())
    }
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16, ArtifactError> {
    let bytes = read_slice(bytes, offset, 2, "u16")?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, ArtifactError> {
    let bytes = read_slice(bytes, offset, 4, "u32")?;
    Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64, ArtifactError> {
    let bytes = read_slice(bytes, offset, 8, "u64")?;
    Ok(u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]))
}

fn write_u16(out: &mut [u8], offset: usize, value: u16) {
    out[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(out: &mut [u8], offset: usize, value: u32) {
    out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(out: &mut [u8], offset: usize, value: u64) {
    out[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_blob() -> Vec<u8> {
        build_artifact_blob(
            &ArtifactBundleSpec::new("demo", "sm_90")
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Ptx,
                    "demo.ptx",
                    b"ptx",
                ))
                .with_entry(ArtifactEntrySpec::new("hello", ArtifactEntryKind::Kernel)),
        )
        .unwrap()
    }

    fn sample_payload_record_start() -> usize {
        HEADER_BYTES + "demo".len() + "sm_90".len()
    }

    #[test]
    fn artifact_blob_round_trips_ptx_payload() {
        let blob = sample_blob();
        let bundles = parse_artifact_section(&blob).unwrap();

        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].name, "demo");
        assert_eq!(bundles[0].target, "sm_90");
        assert_eq!(
            bundles[0].payload(ArtifactPayloadKind::Ptx),
            Some(&b"ptx"[..])
        );
        assert_eq!(
            bundles[0].entry("hello").unwrap().kind,
            ArtifactEntryKind::Kernel
        );
    }

    #[test]
    fn artifact_blob_round_trips_non_ptx_payload_kinds() {
        let blob = build_artifact_blob(
            &ArtifactBundleSpec::new("demo", "sm_90")
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::NvvmIr,
                    "demo.ll",
                    b"nvvm ir",
                ))
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Ltoir,
                    "demo.ltoir",
                    b"ltoir",
                ))
                .with_payload(ArtifactPayloadSpec::new(
                    ArtifactPayloadKind::Cubin,
                    "demo.cubin",
                    b"cubin",
                )),
        )
        .unwrap();
        let bundles = parse_artifact_section(&blob).unwrap();

        assert_eq!(bundles.len(), 1);
        assert_eq!(
            bundles[0].payload(ArtifactPayloadKind::NvvmIr),
            Some(&b"nvvm ir"[..])
        );
        assert_eq!(
            bundles[0].payload(ArtifactPayloadKind::Ltoir),
            Some(&b"ltoir"[..])
        );
        assert_eq!(
            bundles[0].payload(ArtifactPayloadKind::Cubin),
            Some(&b"cubin"[..])
        );
    }

    #[test]
    fn artifact_section_ignores_trailing_zero_padding() {
        let mut section = sample_blob();
        section.extend_from_slice(&[0; HEADER_BYTES]);

        let bundles = parse_artifact_section(&section).unwrap();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].name, "demo");
    }

    #[test]
    fn artifact_section_parses_concatenated_blobs() {
        let first = build_artifact_blob(&ArtifactBundleSpec::new("a", "sm_80").with_payload(
            ArtifactPayloadSpec::new(ArtifactPayloadKind::Ptx, "a.ptx", b"a"),
        ))
        .unwrap();
        let second = build_artifact_blob(&ArtifactBundleSpec::new("b", "sm_90").with_payload(
            ArtifactPayloadSpec::new(ArtifactPayloadKind::Ptx, "b.ptx", b"b"),
        ))
        .unwrap();

        let mut section = first;
        section.extend_from_slice(&second);

        let bundles = parse_artifact_section(&section).unwrap();
        assert_eq!(
            bundles.iter().map(|bundle| bundle.name).collect::<Vec<_>>(),
            ["a", "b"]
        );
    }

    #[test]
    fn artifact_section_rejects_truncated_blob_without_panicking() {
        let mut blob = sample_blob();
        let oversized_len = (blob.len() + 1) as u32;
        write_u32(&mut blob, 12, oversized_len);

        let error = parse_artifact_section(&blob).unwrap_err();
        assert_eq!(error, ArtifactError::Truncated("blob"));
    }

    #[test]
    fn artifact_blob_rejects_total_len_smaller_than_header() {
        let mut blob = sample_blob();
        write_u32(&mut blob, 12, (HEADER_BYTES - 1) as u32);

        let error = parse_artifact_blob(&blob).unwrap_err();
        assert!(matches!(
            error,
            ArtifactError::Malformed(message) if message.contains("smaller than the header")
        ));
    }

    #[test]
    fn artifact_blob_rejects_invalid_utf8_payload_name() {
        let mut blob = sample_blob();
        let payload_record = sample_payload_record_start();
        let payload_name_offset = read_u32(&blob, payload_record + 12).unwrap() as usize;
        blob[payload_name_offset] = 0xff;

        let error = parse_artifact_blob(&blob).unwrap_err();
        assert_eq!(error, ArtifactError::InvalidUtf8("payload name"));
    }

    #[test]
    fn artifact_blob_rejects_unknown_payload_kind() {
        let mut blob = sample_blob();
        write_u16(&mut blob, sample_payload_record_start(), 0xffff);

        let error = parse_artifact_blob(&blob).unwrap_err();
        assert_eq!(error, ArtifactError::UnsupportedPayloadKind(0xffff));
    }

    #[test]
    fn artifact_section_name_is_portable() {
        assert!(ARTIFACT_SECTION_NAME.len() <= 8);
    }

    #[cfg(all(feature = "object-read", feature = "object-write"))]
    #[test]
    fn host_object_round_trips_section_on_supported_formats() {
        let blob = build_artifact_blob(&ArtifactBundleSpec::new("demo", "sm_90").with_payload(
            ArtifactPayloadSpec::new(ArtifactPayloadKind::Ptx, "demo.ptx", b"ptx"),
        ))
        .unwrap();

        for target in ["x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu"] {
            let object = build_host_object_for_target(&blob, target, None).unwrap();
            let bundles = read_artifact_bundles_from_object_bytes(&object).unwrap();
            assert_eq!(bundles.len(), 1);
            assert_eq!(
                bundles[0].payload(ArtifactPayloadKind::Ptx),
                Some(&b"ptx"[..])
            );
        }
    }

    /// The anchor symbol must be a *defined* global pointing at the
    /// `.oxart` section. A linker only extracts an rlib archive member if
    /// the member defines a symbol someone references, so an undefined or
    /// missing anchor would reintroduce the dropped-bundle bug.
    #[cfg(all(feature = "object-read", feature = "object-write"))]
    #[test]
    fn host_object_defines_requested_anchor_symbol() {
        use object::{Object, ObjectSymbol};

        let blob = sample_blob();
        let bytes =
            build_host_object_for_target(&blob, "x86_64-unknown-linux-gnu", Some("demo_anchor"))
                .unwrap();

        let file = object::File::parse(bytes.as_slice()).unwrap();
        let anchor = file
            .symbols()
            .find(|symbol| symbol.name() == Ok("demo_anchor"))
            .expect("anchor symbol missing from artifact object");
        assert!(anchor.is_definition());
        assert!(anchor.is_global());
        assert_eq!(anchor.address(), 0);

        // The data must still round-trip with the symbol present.
        let bundles = read_artifact_bundles_from_object_bytes(&bytes).unwrap();
        assert_eq!(bundles.len(), 1);
        assert_eq!(bundles[0].name, "demo");
    }

    /// Omitting the anchor must keep producing a symbol-free object (the
    /// shape used by tests and any non-rlib embedding).
    #[cfg(all(feature = "object-read", feature = "object-write"))]
    #[test]
    fn host_object_without_anchor_has_no_symbols() {
        use object::Object;

        let blob = sample_blob();
        let bytes = build_host_object_for_target(&blob, "x86_64-unknown-linux-gnu", None).unwrap();

        let file = object::File::parse(bytes.as_slice()).unwrap();
        assert_eq!(file.symbols().count(), 0);
    }

    #[cfg(feature = "object-write")]
    #[test]
    fn host_object_rejects_non_cuda_host_targets() {
        let blob = sample_blob();

        for target in ["powerpc64le-unknown-linux-gnu", "x86_64-apple-darwin"] {
            assert!(matches!(
                build_host_object_for_target(&blob, target, None),
                Err(ArtifactError::UnsupportedHostTarget(_))
            ));
        }
    }
}
