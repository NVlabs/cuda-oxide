/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{CatalogIntrinsic, CatalogLlvm, EvidenceStageKind, IntrinsicBackend};
use crate::ptx::{InstructionPattern, OperandPattern};
use crate::render::render_probe;
use crate::resolve::resolve;
use crate::util::{pretty_json, sha256_bytes, sha256_file};
use anyhow::{Context, Result, ensure};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProbeMode {
    SelectedEvidence,
    Comparison,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LlcIdentity {
    version: String,
    sha256: String,
}

pub fn run(
    repo_root: &Path,
    intrinsic_id: &str,
    llc: Option<PathBuf>,
    skip_terminal: bool,
) -> Result<()> {
    let catalog = resolve(repo_root)?;
    let record = catalog
        .intrinsics
        .iter()
        .find(|record| record.id == intrinsic_id)
        .with_context(|| format!("unknown catalog intrinsic {intrinsic_id}"))?;
    let (llc, mode) = match llc {
        Some(path) => (path, ProbeMode::Comparison),
        None => (rust_toolchain_llc()?, ProbeMode::SelectedEvidence),
    };
    let identity = llc_identity(&llc)?;
    validate_backend_identity(
        mode,
        &record.backend.version,
        &record.backend.sha256,
        &identity,
    )?;
    let output_dir = repo_root.join("target/intrinsics/probes");
    fs::create_dir_all(&output_dir)?;
    let catalog_json = pretty_json(&catalog)?;
    let catalog_hash = sha256_bytes(catalog_json.as_bytes());
    let input = output_dir.join(format!("{intrinsic_id}.ll"));
    fs::write(&input, render_probe(&catalog, record, &catalog_hash))
        .with_context(|| format!("write in-memory probe {}", input.display()))?;
    if record.llvm.is_some() && mode == ProbeMode::SelectedEvidence {
        assert_intrinsic_declaration_canonicalizes(
            &llc,
            &input,
            &output_dir,
            intrinsic_id,
            record,
        )?;
    }
    let output = output_dir.join(format!("{intrinsic_id}.ptx"));
    let status = Command::new(&llc)
        .arg(&input)
        .arg("-march=nvptx64")
        .arg(format!("-mcpu={}", record.backend.gpu_target))
        .arg(format!("-mattr={}", record.backend.ptx_feature))
        .arg("-o")
        .arg(&output)
        .status()
        .with_context(|| format!("run {}", llc.display()))?;
    ensure!(status.success(), "LLVM probe failed with {status}");
    let ptx = fs::read_to_string(&output)
        .with_context(|| format!("read generated PTX {}", output.display()))?;
    validate_probe_instructions(record, &ptx)?;
    let has_terminal_stage = record.backend_lowerings.iter().any(|lowering| {
        lowering.backend == IntrinsicBackend::LlvmNvptx
            && lowering
                .stages
                .iter()
                .any(|stage| stage.stage == EvidenceStageKind::PtxAssembly)
    });
    if mode == ProbeMode::SelectedEvidence && has_terminal_stage {
        if skip_terminal {
            println!(
                "backend-only probe: `--skip-terminal` was explicit, so recorded ptxas evidence was not revalidated"
            );
        } else {
            assemble_probe_ptx(record, &output, &output_dir, intrinsic_id)?;
        }
    }
    match mode {
        ProbeMode::SelectedEvidence => println!(
            "selected evidence backend {} (SHA-256 {}) lowered {} to `{}` for {} {}",
            identity.version,
            identity.sha256,
            intrinsic_id,
            record.expected_ptx,
            record.backend.gpu_target,
            record.backend.ptx_feature,
        ),
        ProbeMode::Comparison => println!(
            "comparison backend {} (SHA-256 {}) lowered {} to `{}` for {} {}; this does not validate selected evidence {} (SHA-256 {})",
            identity.version,
            identity.sha256,
            intrinsic_id,
            record.expected_ptx,
            record.backend.gpu_target,
            record.backend.ptx_feature,
            record.backend.version,
            record.backend.sha256,
        ),
    }
    println!("PTX: {}", output.display());
    Ok(())
}

fn validate_probe_instructions(record: &CatalogIntrinsic, ptx: &str) -> Result<()> {
    ensure!(
        record.expected_ptx.matches(ptx),
        "probe PTX has no instruction matching `{}`",
        record.expected_ptx
    );
    if record.vote.is_some() {
        validate_register_and_immediate_forms(&record.expected_ptx, 2, "-1", ptx)?;
    }
    if record.warp_match.is_some() {
        validate_two_register_and_immediate_forms(&record.expected_ptx, 1, "7", 2, "-1", ptx)?;
    }
    if record.warp_barrier.is_some() {
        validate_register_and_immediate_forms(&record.expected_ptx, 0, "-1", ptx)?;
    }
    if let Some(shuffle) = &record.warp_shuffle {
        validate_warp_shuffle_forms(&record.expected_ptx, shuffle.clamp, ptx)?;
    }
    Ok(())
}

fn validate_warp_shuffle_forms(expected: &InstructionPattern, clamp: u32, ptx: &str) -> Result<()> {
    let clamp_operand = expected
        .operands
        .get(3)
        .context("shuffle probe clamp operand index is out of range")?;
    ensure!(
        matches!(clamp_operand, OperandPattern::Exact { value } if value == &clamp.to_string()),
        "shuffle probe clamp operand is not the exact catalog clamp {clamp}"
    );
    validate_two_register_and_immediate_forms(expected, 2, "1", 4, "-1", ptx)
}

fn validate_register_and_immediate_forms(
    expected: &InstructionPattern,
    operand_index: usize,
    immediate: &str,
    ptx: &str,
) -> Result<()> {
    let mut register = expected.clone();
    let register_operand = register
        .operands
        .get_mut(operand_index)
        .context("probe register-or-immediate operand index is out of range")?;
    ensure!(
        *register_operand == OperandPattern::RegisterOrImmediate,
        "probe operand {operand_index} is not register-or-immediate"
    );
    *register_operand = OperandPattern::Register;

    let mut immediate_pattern = expected.clone();
    immediate_pattern.operands[operand_index] = OperandPattern::Exact {
        value: immediate.into(),
    };

    ensure!(
        register.matches(ptx),
        "probe PTX has no register form matching `{register}`"
    );
    ensure!(
        immediate_pattern.matches(ptx),
        "probe PTX has no immediate form matching `{immediate_pattern}`"
    );
    Ok(())
}

fn validate_two_register_and_immediate_forms(
    expected: &InstructionPattern,
    first_operand_index: usize,
    first_immediate: &str,
    second_operand_index: usize,
    second_immediate: &str,
    ptx: &str,
) -> Result<()> {
    ensure!(
        first_operand_index != second_operand_index,
        "probe register-or-immediate operand indices must be distinct"
    );
    for operand_index in [first_operand_index, second_operand_index] {
        let operand = expected
            .operands
            .get(operand_index)
            .context("probe register-or-immediate operand index is out of range")?;
        ensure!(
            *operand == OperandPattern::RegisterOrImmediate,
            "probe operand {operand_index} is not register-or-immediate"
        );
    }

    let combinations = [
        ("rr", OperandPattern::Register, OperandPattern::Register),
        (
            "ri",
            OperandPattern::Register,
            OperandPattern::Exact {
                value: second_immediate.into(),
            },
        ),
        (
            "ir",
            OperandPattern::Exact {
                value: first_immediate.into(),
            },
            OperandPattern::Register,
        ),
        (
            "ii",
            OperandPattern::Exact {
                value: first_immediate.into(),
            },
            OperandPattern::Exact {
                value: second_immediate.into(),
            },
        ),
    ];

    for (name, first, second) in combinations {
        let mut pattern = expected.clone();
        pattern.operands[first_operand_index] = first;
        pattern.operands[second_operand_index] = second;
        ensure!(
            pattern.matches(ptx),
            "probe PTX has no {name} form matching `{pattern}`"
        );
    }
    Ok(())
}

fn assert_intrinsic_declaration_canonicalizes(
    llc: &Path,
    input: &Path,
    output_dir: &Path,
    intrinsic_id: &str,
    record: &CatalogIntrinsic,
) -> Result<()> {
    let tool_dir = llc
        .parent()
        .context("selected llc has no containing tool directory")?;
    let llvm_as = tool_dir.join("llvm-as");
    let llvm_dis = tool_dir.join("llvm-dis");
    ensure!(
        llvm_as.is_file() && llvm_dis.is_file(),
        "selected LLVM toolchain omits llvm-as or llvm-dis"
    );
    let bitcode = output_dir.join(format!("{intrinsic_id}.bc"));
    let canonical = output_dir.join(format!("{intrinsic_id}.canonical.ll"));
    let status = Command::new(&llvm_as)
        .arg(input)
        .arg("-o")
        .arg(&bitcode)
        .status()
        .with_context(|| format!("run {}", llvm_as.display()))?;
    ensure!(status.success(), "llvm-as probe failed with {status}");
    let status = Command::new(&llvm_dis)
        .arg(&bitcode)
        .arg("-o")
        .arg(&canonical)
        .status()
        .with_context(|| format!("run {}", llvm_dis.display()))?;
    ensure!(status.success(), "llvm-dis probe failed with {status}");
    let canonical = fs::read_to_string(&canonical)?;
    assert_canonical_intrinsic_declaration(
        &canonical,
        record
            .llvm
            .as_ref()
            .context("LLVM-backed probe has no LLVM facts")?,
    )
}

fn assert_canonical_intrinsic_declaration(canonical: &str, llvm: &CatalogLlvm) -> Result<()> {
    let symbol = llvm.resolved_symbol.as_deref().unwrap_or(&llvm.symbol);
    let symbol_marker = format!("@{symbol}(");
    let declaration = canonical
        .lines()
        .find(|line| {
            let line = line.trim_start();
            line.starts_with("declare ") && line.contains(&symbol_marker)
        })
        .with_context(|| format!("canonical module has no declaration for @{symbol}"))?;
    let declaration_prefix = declaration
        .split_once(&symbol_marker)
        .map(|(prefix, _)| prefix)
        .context("canonical intrinsic declaration has a malformed symbol")?;
    let arguments = declaration_arguments(declaration, &symbol_marker)?;
    let function_attributes = declaration_attribute_group(canonical, declaration)?;

    let mut no_memory = false;
    let mut argument_memory_only = false;
    let mut inaccessible_memory_only = false;
    let mut reads_memory = false;
    let mut writes_memory = false;
    let mut has_side_effects = false;

    for property in &llvm.properties {
        match property.as_str() {
            "IntrConvergent" => {
                require_attribute_token(function_attributes, "convergent", symbol, "function")?
            }
            "IntrNoCallback" => {
                require_attribute_token(function_attributes, "nocallback", symbol, "function")?
            }
            "IntrSpeculatable" => {
                require_attribute_token(function_attributes, "speculatable", symbol, "function")?
            }
            "IntrNoMem" => no_memory = true,
            "IntrArgMemOnly" => argument_memory_only = true,
            "IntrInaccessibleMemOnly" => inaccessible_memory_only = true,
            "IntrReadMem" => reads_memory = true,
            "IntrWriteMem" => writes_memory = true,
            "IntrHasSideEffects" => has_side_effects = true,
            "NoUndef<ret>" => {
                // Return attributes are asserted from the normalized result facts below.
                ensure!(
                    llvm.result_facts.no_undef,
                    "@{symbol} imported NoUndef return property disagrees with its normalized result facts"
                );
            }
            property if property.starts_with("Range<") => {
                // Return attributes are asserted from the normalized result facts below.
                let range = llvm.result_facts.range.as_ref().with_context(|| {
                    format!(
                        "@{symbol} imported range property disagrees with its normalized result facts"
                    )
                })?;
                ensure!(
                    property == format!("Range<ret,{},{}>", range.lower, range.upper_exclusive),
                    "malformed or unsupported imported LLVM range property {property:?} on @{symbol}"
                );
            }
            property if property.starts_with("NoCapture<") => {
                let index = property_argument_index(property, "NoCapture")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_fragment(argument, "captures(none)", symbol, "argument")?;
            }
            property if property.starts_with("ReadOnly<") => {
                let index = property_argument_index(property, "ReadOnly")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_token(argument, "readonly", symbol, "argument")?;
            }
            property if property.starts_with("WriteOnly<") => {
                let index = property_argument_index(property, "WriteOnly")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_token(argument, "writeonly", symbol, "argument")?;
            }
            property if property.starts_with("ImmArg<") => {
                let index = property_argument_index(property, "ImmArg")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_token(argument, "immarg", symbol, "argument")?;
            }
            property if property.starts_with("NoUndef<") => {
                let index = property_argument_index(property, "NoUndef")?;
                let argument = arguments.get(index).with_context(|| {
                    format!("@{symbol} has no argument {index} required by {property}")
                })?;
                require_attribute_token(argument, "noundef", symbol, "argument")?;
            }
            unsupported => anyhow::bail!(
                "cannot verify unsupported imported LLVM property {unsupported:?} on @{symbol}"
            ),
        }
    }

    let memory = canonical_memory_attribute(
        no_memory,
        argument_memory_only,
        inaccessible_memory_only,
        reads_memory,
        writes_memory,
    )?;
    if has_side_effects {
        let memory = memory.as_deref().with_context(|| {
            format!(
                "@{symbol} IntrHasSideEffects requires a concrete non-`memory(none)` canonical memory effect"
            )
        })?;
        ensure!(
            memory != "memory(none)",
            "@{symbol} IntrHasSideEffects requires a concrete non-`memory(none)` canonical memory effect"
        );
    }
    if let Some(memory) = memory {
        require_attribute_fragment(function_attributes, &memory, symbol, "function")?;
    }

    if llvm.result_facts.no_undef {
        require_attribute_token(declaration_prefix, "noundef", symbol, "return")?;
    }
    if let Some(range) = &llvm.result_facts.range {
        ensure!(
            llvm.results.len() == 1,
            "@{symbol} has a return range but not exactly one imported result"
        );
        let width = llvm.results[0]
            .strip_prefix('i')
            .with_context(|| {
                format!(
                    "@{symbol} has a return range on unsupported result type {}",
                    llvm.results[0]
                )
            })?
            .parse::<u32>()
            .with_context(|| {
                format!(
                    "@{symbol} has a return range on malformed result type {}",
                    llvm.results[0]
                )
            })?;
        let lower = canonical_integer_literal(&range.lower, width)?;
        let upper = canonical_integer_literal(&range.upper_exclusive, width)?;
        require_attribute_fragment(
            declaration_prefix,
            &format!("range(i{width} {lower}, {upper})"),
            symbol,
            "return",
        )?;
    }
    Ok(())
}

fn declaration_attribute_group<'a>(canonical: &'a str, declaration: &str) -> Result<&'a str> {
    let Some(group) = declaration
        .split_ascii_whitespace()
        .rev()
        .find(|token| token.starts_with('#'))
    else {
        return Ok("");
    };
    let prefix = format!("attributes {group} = ");
    canonical
        .lines()
        .find_map(|line| line.trim_start().strip_prefix(&prefix))
        .with_context(|| format!("canonical intrinsic declaration references missing {group}"))
}

fn declaration_arguments<'a>(declaration: &'a str, symbol_marker: &str) -> Result<Vec<&'a str>> {
    let start = declaration
        .find(symbol_marker)
        .map(|offset| offset + symbol_marker.len())
        .context("canonical intrinsic declaration has no argument list")?;
    let arguments = &declaration[start..];
    let mut parentheses = 0_u32;
    let mut braces = 0_u32;
    let mut brackets = 0_u32;
    let mut angles = 0_u32;
    let mut argument_start = 0;
    let mut split = Vec::new();

    for (offset, character) in arguments.char_indices() {
        match character {
            '(' => parentheses += 1,
            ')' if parentheses == 0 => {
                let argument = arguments[argument_start..offset].trim();
                if !argument.is_empty() {
                    split.push(argument);
                }
                return Ok(split);
            }
            ')' => parentheses -= 1,
            '{' => braces += 1,
            '}' => braces = braces.saturating_sub(1),
            '[' => brackets += 1,
            ']' => brackets = brackets.saturating_sub(1),
            '<' => angles += 1,
            '>' => angles = angles.saturating_sub(1),
            ',' if parentheses == 0 && braces == 0 && brackets == 0 && angles == 0 => {
                split.push(arguments[argument_start..offset].trim());
                argument_start = offset + character.len_utf8();
            }
            _ => {}
        }
    }
    anyhow::bail!("canonical intrinsic declaration has an unterminated argument list")
}

fn property_argument_index(property: &str, property_name: &str) -> Result<usize> {
    let prefix = format!("{property_name}<arg");
    property
        .strip_prefix(&prefix)
        .and_then(|index| index.strip_suffix('>'))
        .with_context(|| format!("malformed imported LLVM property {property:?}"))?
        .parse::<usize>()
        .with_context(|| format!("malformed imported LLVM property {property:?}"))
}

fn canonical_memory_attribute(
    no_memory: bool,
    argument_memory_only: bool,
    inaccessible_memory_only: bool,
    reads_memory: bool,
    writes_memory: bool,
) -> Result<Option<String>> {
    ensure!(
        !(argument_memory_only && inaccessible_memory_only),
        "imported LLVM properties specify two incompatible memory locations"
    );
    if no_memory {
        ensure!(
            !argument_memory_only && !inaccessible_memory_only && !reads_memory && !writes_memory,
            "IntrNoMem is combined with another imported LLVM memory property"
        );
        return Ok(Some("memory(none)".into()));
    }

    let access = match (reads_memory, writes_memory) {
        (true, false) => "read",
        (false, true) => "write",
        _ => "readwrite",
    };
    let location = if argument_memory_only {
        Some("argmem")
    } else if inaccessible_memory_only {
        Some("inaccessiblemem")
    } else {
        None
    };
    match location {
        Some(location) => Ok(Some(format!("memory({location}: {access})"))),
        None if reads_memory && !writes_memory => Ok(Some("memory(read)".into())),
        None if writes_memory && !reads_memory => Ok(Some("memory(write)".into())),
        // Read-write access to unrestricted memory is LLVM's default and has no
        // canonical attribute to assert.
        None => Ok(None),
    }
}

fn canonical_integer_literal(value: &str, width: u32) -> Result<String> {
    ensure!(
        (1..=64).contains(&width),
        "cannot verify a canonical range for unsupported i{width}"
    );
    if value.starts_with('-') {
        let signed = value
            .parse::<i128>()
            .with_context(|| format!("invalid signed LLVM range bound {value:?}"))?;
        let minimum = -(1_i128 << (width - 1));
        ensure!(
            signed >= minimum,
            "LLVM range bound {value} does not fit in i{width}"
        );
        return Ok(signed.to_string());
    }

    let unsigned = value
        .parse::<u128>()
        .with_context(|| format!("invalid unsigned LLVM range bound {value:?}"))?;
    let modulus = 1_u128 << width;
    ensure!(
        unsigned < modulus,
        "LLVM range bound {value} does not fit in i{width}"
    );
    let sign_bit = 1_u128 << (width - 1);
    if unsigned < sign_bit {
        Ok(unsigned.to_string())
    } else {
        Ok((unsigned as i128 - modulus as i128).to_string())
    }
}

fn require_attribute_token(text: &str, required: &str, symbol: &str, position: &str) -> Result<()> {
    let present = text.split_ascii_whitespace().any(|token| {
        token.trim_matches(|character| matches!(character, '{' | '}' | ',')) == required
    });
    ensure!(
        present,
        "canonicalized @{symbol} {position} attributes are missing {required:?}"
    );
    Ok(())
}

fn require_attribute_fragment(
    text: &str,
    required: &str,
    symbol: &str,
    position: &str,
) -> Result<()> {
    ensure!(
        text.contains(required),
        "canonicalized @{symbol} {position} attributes are missing {required:?}"
    );
    Ok(())
}

fn assemble_probe_ptx(
    record: &crate::model::CatalogIntrinsic,
    ptx: &Path,
    output_dir: &Path,
    intrinsic_id: &str,
) -> Result<()> {
    let stage = record
        .backend_lowerings
        .iter()
        .find(|lowering| lowering.backend == IntrinsicBackend::LlvmNvptx)
        .and_then(|lowering| {
            lowering
                .stages
                .iter()
                .find(|stage| stage.stage == EvidenceStageKind::PtxAssembly)
        })
        .context("selected LLVM evidence has no PTX-assembly stage")?;
    let tool = PathBuf::from(
        stage
            .tool_path
            .as_deref()
            .context("PTX-assembly stage has no tool path")?,
    );
    let expected_sha256 = stage
        .tool_sha256
        .as_deref()
        .context("PTX-assembly stage has no tool SHA-256")?;
    ensure!(
        sha256_file(&tool)? == expected_sha256,
        "ptxas binary does not match selected evidence"
    );
    let cubin = output_dir.join(format!("{intrinsic_id}.cubin"));
    let architecture = stage
        .targets
        .iter()
        .find(|target| target.starts_with("sm_"))
        .context("PTX-assembly evidence has no sm_NN target")?;
    let status = Command::new(&tool)
        .arg(format!("-arch={architecture}"))
        .arg(ptx)
        .arg("-o")
        .arg(&cubin)
        .status()
        .with_context(|| format!("run {}", tool.display()))?;
    ensure!(status.success(), "ptxas probe failed with {status}");
    println!(
        "terminal PTX assembly revalidated with {} for {}",
        tool.display(),
        architecture
    );
    Ok(())
}

fn llc_identity(llc: &Path) -> Result<LlcIdentity> {
    let version = Command::new(llc)
        .arg("--version")
        .output()
        .with_context(|| format!("query {} --version", llc.display()))?;
    ensure!(
        version.status.success(),
        "{} --version failed",
        llc.display()
    );
    let version = String::from_utf8_lossy(&version.stdout)
        .lines()
        .find(|line| line.contains("LLVM version"))
        .context("llc --version did not report an LLVM version")?
        .trim()
        .to_owned();
    Ok(LlcIdentity {
        version,
        sha256: sha256_file(llc)?,
    })
}

fn validate_backend_identity(
    mode: ProbeMode,
    expected_version: &str,
    expected_sha256: &str,
    actual: &LlcIdentity,
) -> Result<()> {
    if mode == ProbeMode::Comparison {
        return Ok(());
    }
    ensure!(
        actual.version == expected_version,
        "rust-toolchain llc version mismatch: selected evidence records {expected_version:?}, found {:?}; use an explicit `--llc` only for a comparison probe",
        actual.version
    );
    ensure!(
        actual.sha256 == expected_sha256,
        "rust-toolchain llc SHA-256 mismatch: selected evidence records {expected_sha256}, found {}; use an explicit `--llc` only for a comparison probe",
        actual.sha256
    );
    Ok(())
}

fn rust_toolchain_llc() -> Result<PathBuf> {
    let sysroot = Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .context("query rustc sysroot")?;
    ensure!(sysroot.status.success(), "rustc --print sysroot failed");
    let verbose = Command::new("rustc")
        .arg("-vV")
        .output()
        .context("query rustc host")?;
    ensure!(verbose.status.success(), "rustc -vV failed");
    let host = String::from_utf8_lossy(&verbose.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .context("rustc -vV did not report a host")?
        .to_owned();
    let path = PathBuf::from(String::from_utf8_lossy(&sysroot.stdout).trim())
        .join("lib/rustlib")
        .join(host)
        .join("bin/llc");
    ensure!(
        path.is_file(),
        "rust toolchain has no llc at {}",
        path.display()
    );
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{CatalogHalfOpenRange, CatalogLlvmResultFacts};

    #[test]
    fn vote_probe_requires_register_and_negative_one_mask_forms() {
        let expected = InstructionPattern::new(
            "vote",
            &["sync", "all", "pred"],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
            ],
        );
        let register = "vote.sync.all.pred %p1, %p2, %r3;";
        let immediate = "vote.sync.all.pred %p4, %p5, -1;";

        validate_register_and_immediate_forms(
            &expected,
            2,
            "-1",
            &format!("{register}\n{immediate}"),
        )
        .unwrap();

        let error =
            validate_register_and_immediate_forms(&expected, 2, "-1", register).unwrap_err();
        assert!(error.to_string().contains("no immediate form"));

        let error =
            validate_register_and_immediate_forms(&expected, 2, "-1", immediate).unwrap_err();
        assert!(error.to_string().contains("no register form"));
    }

    #[test]
    fn warp_barrier_probe_requires_register_and_negative_one_mask_forms() {
        let expected = InstructionPattern::new(
            "bar",
            &["warp", "sync"],
            vec![OperandPattern::RegisterOrImmediate],
        );
        let register = "bar.warp.sync %r1;";
        let immediate = "bar.warp.sync -1;";

        validate_register_and_immediate_forms(
            &expected,
            0,
            "-1",
            &format!("{register}\n{immediate}"),
        )
        .unwrap();

        let error =
            validate_register_and_immediate_forms(&expected, 0, "-1", register).unwrap_err();
        assert!(error.to_string().contains("no immediate form"));

        let error =
            validate_register_and_immediate_forms(&expected, 0, "-1", immediate).unwrap_err();
        assert!(error.to_string().contains("no register form"));
    }

    #[test]
    fn warp_match_probe_requires_every_register_and_immediate_combination() {
        let expected = InstructionPattern::new(
            "match",
            &["any", "sync", "b32"],
            vec![
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
                OperandPattern::RegisterOrImmediate,
            ],
        );
        let forms = [
            ("rr", "match.any.sync.b32 %r1, %r2, %r3;"),
            ("ri", "match.any.sync.b32 %r4, %r5, -1;"),
            ("ir", "match.any.sync.b32 %r6, 7, %r7;"),
            ("ii", "match.any.sync.b32 %r8, 7, -1;"),
        ];
        let complete = forms
            .iter()
            .map(|(_, instruction)| *instruction)
            .collect::<Vec<_>>()
            .join("\n");

        validate_two_register_and_immediate_forms(&expected, 1, "7", 2, "-1", &complete).unwrap();

        for (missing_index, (name, _)) in forms.iter().enumerate() {
            let incomplete = forms
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != missing_index)
                .map(|(_, (_, instruction))| *instruction)
                .collect::<Vec<_>>()
                .join("\n");
            let error =
                validate_two_register_and_immediate_forms(&expected, 1, "7", 2, "-1", &incomplete)
                    .unwrap_err();
            assert!(
                error.to_string().contains(&format!("no {name} form")),
                "{error:#}"
            );
        }
    }

    #[test]
    fn warp_shuffle_probe_requires_lane_mask_forms_and_exact_clamp() {
        let expected = InstructionPattern::new(
            "shfl",
            &["sync", "idx", "b32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
                OperandPattern::Exact { value: "31".into() },
                OperandPattern::RegisterOrImmediate,
            ],
        );
        let forms = [
            ("rr", "shfl.sync.idx.b32 %r1, %r2, %r3, 31, %r4;"),
            ("ri", "shfl.sync.idx.b32 %r5, %r6, %r7, 31, -1;"),
            ("ir", "shfl.sync.idx.b32 %r8, %r9, 1, 31, %r10;"),
            ("ii", "shfl.sync.idx.b32 %r11, %r12, 1, 31, -1;"),
        ];
        let complete = forms
            .iter()
            .map(|(_, instruction)| *instruction)
            .collect::<Vec<_>>()
            .join("\n");

        validate_warp_shuffle_forms(&expected, 31, &complete).unwrap();

        for (missing_index, (name, _)) in forms.iter().enumerate() {
            let incomplete = forms
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != missing_index)
                .map(|(_, (_, instruction))| *instruction)
                .collect::<Vec<_>>()
                .join("\n");
            let error = validate_warp_shuffle_forms(&expected, 31, &incomplete).unwrap_err();
            assert!(
                error.to_string().contains(&format!("no {name} form")),
                "{error:#}"
            );
        }

        let wrong_clamp = InstructionPattern::new(
            "shfl",
            &["sync", "idx", "b32"],
            vec![
                OperandPattern::Register,
                OperandPattern::Register,
                OperandPattern::RegisterOrImmediate,
                OperandPattern::Exact { value: "0".into() },
                OperandPattern::RegisterOrImmediate,
            ],
        );
        let error = validate_warp_shuffle_forms(&wrong_clamp, 31, &complete).unwrap_err();
        assert!(error.to_string().contains("exact catalog clamp 31"));
    }

    fn identity() -> LlcIdentity {
        LlcIdentity {
            version: "LLVM version 22.1.2-test".into(),
            sha256: "abc123".into(),
        }
    }

    #[test]
    fn selected_probe_requires_exact_recorded_backend() {
        validate_backend_identity(
            ProbeMode::SelectedEvidence,
            "LLVM version 22.1.2-test",
            "abc123",
            &identity(),
        )
        .unwrap();

        let version_error = validate_backend_identity(
            ProbeMode::SelectedEvidence,
            "LLVM version 21",
            "abc123",
            &identity(),
        )
        .unwrap_err();
        assert!(version_error.to_string().contains("version mismatch"));

        let hash_error = validate_backend_identity(
            ProbeMode::SelectedEvidence,
            "LLVM version 22.1.2-test",
            "different",
            &identity(),
        )
        .unwrap_err();
        assert!(hash_error.to_string().contains("SHA-256 mismatch"));
    }

    #[test]
    fn explicit_probe_is_always_comparison_only() {
        validate_backend_identity(ProbeMode::Comparison, "different", "different", &identity())
            .unwrap();
    }

    fn llvm_facts(
        symbol: &str,
        resolved_symbol: Option<&str>,
        arguments: &[&str],
        results: &[&str],
        properties: &[&str],
        no_undef: bool,
        range: Option<(&str, &str)>,
    ) -> CatalogLlvm {
        CatalogLlvm {
            symbol: symbol.into(),
            resolved_symbol: resolved_symbol.map(str::to_owned),
            arguments: arguments.iter().map(|value| (*value).into()).collect(),
            results: results.iter().map(|value| (*value).into()).collect(),
            properties: properties.iter().map(|value| (*value).into()).collect(),
            result_facts: CatalogLlvmResultFacts {
                no_undef,
                range: range.map(|(lower, upper_exclusive)| CatalogHalfOpenRange {
                    lower: lower.into(),
                    upper_exclusive: upper_exclusive.into(),
                }),
            },
        }
    }

    #[test]
    fn verifies_lane_id_result_and_function_attributes() {
        let llvm = llvm_facts(
            "llvm.nvvm.read.ptx.sreg.laneid",
            None,
            &[],
            &["i32"],
            &[
                "IntrNoMem",
                "IntrSpeculatable",
                "NoUndef<ret>",
                "Range<ret,0,32>",
            ],
            true,
            Some(("0", "32")),
        );
        let canonical = r#"
declare noundef range(i32 0, 32) i32 @llvm.nvvm.read.ptx.sreg.laneid() #0
attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn verifies_redux_convergence_callback_and_inaccessible_memory() {
        let llvm = llvm_facts(
            "llvm.nvvm.redux.sync.add",
            None,
            &["i32", "i32"],
            &["i32"],
            &[
                "IntrConvergent",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ],
            false,
            None,
        );
        let canonical = r#"
declare i32 @llvm.nvvm.redux.sync.add(i32, i32) #0
attributes #0 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn verifies_side_effects_have_a_concrete_memory_effect() {
        let llvm = llvm_facts(
            "llvm.nvvm.activemask",
            None,
            &[],
            &["i32"],
            &[
                "IntrConvergent",
                "IntrHasSideEffects",
                "IntrInaccessibleMemOnly",
                "IntrNoCallback",
            ],
            false,
            None,
        );
        let canonical = r#"
declare i32 @llvm.nvvm.activemask() #0
attributes #0 = { convergent nocallback nounwind memory(inaccessiblemem: readwrite) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn side_effects_without_a_concrete_memory_effect_fail_closed() {
        let llvm = llvm_facts(
            "llvm.nvvm.activemask",
            None,
            &[],
            &["i32"],
            &["IntrHasSideEffects"],
            false,
            None,
        );
        let error =
            assert_canonical_intrinsic_declaration("declare i32 @llvm.nvvm.activemask()\n", &llvm)
                .unwrap_err();
        assert!(error.to_string().contains("concrete non-`memory(none)`"));

        let no_memory = llvm_facts(
            "llvm.nvvm.activemask",
            None,
            &[],
            &["i32"],
            &["IntrHasSideEffects", "IntrNoMem"],
            false,
            None,
        );
        let canonical = r#"
declare i32 @llvm.nvvm.activemask() #0
attributes #0 = { nounwind memory(none) }
"#;
        let error = assert_canonical_intrinsic_declaration(canonical, &no_memory).unwrap_err();
        assert!(error.to_string().contains("concrete non-`memory(none)`"));
    }

    #[test]
    fn verifies_sync_threads_convergence_and_callback_attributes() {
        let llvm = llvm_facts(
            "llvm.nvvm.barrier.cta.sync.aligned.all",
            None,
            &["i32"],
            &[],
            &["IntrConvergent", "IntrNoCallback"],
            false,
            None,
        );
        let canonical = r#"
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #0
attributes #0 = { convergent nocallback nounwind }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn verifies_dp2a_immediate_argument_attribute() {
        let llvm = llvm_facts(
            "llvm.nvvm.idp2a.s.s",
            None,
            &["i32", "i32", "i1", "i32"],
            &["i32"],
            &["ImmArg<arg2>", "IntrNoMem", "IntrSpeculatable"],
            false,
            None,
        );
        let canonical = r#"
declare i32 @llvm.nvvm.idp2a.s.s(i32, i32, i1 immarg, i32) #0
attributes #0 = { speculatable memory(none) }
"#;
        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();

        let missing_immarg = r#"
declare i32 @llvm.nvvm.idp2a.s.s(i32, i32, i1, i32) #0
attributes #0 = { speculatable memory(none) }
"#;
        let error = assert_canonical_intrinsic_declaration(missing_immarg, &llvm).unwrap_err();
        assert!(error.to_string().contains("missing \"immarg\""));
    }

    #[test]
    fn retains_ldmatrix_function_and_argument_requirements() {
        let llvm = llvm_facts(
            "llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16",
            Some("llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3"),
            &["anyptr"],
            &["i32", "i32", "i32", "i32"],
            &[
                "IntrArgMemOnly",
                "IntrConvergent",
                "IntrNoCallback",
                "IntrReadMem",
                "NoCapture<arg0>",
                "ReadOnly<arg0>",
            ],
            false,
            None,
        );
        let canonical = r#"
declare { i32, i32, i32, i32 } @llvm.nvvm.ldmatrix.sync.aligned.m8n8.x4.b16.p3(ptr addrspace(3) readonly captures(none)) #0
attributes #0 = { convergent nocallback nounwind memory(argmem: read) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn canonicalizes_unsigned_range_bounds_as_llvm_signed_literals() {
        let llvm = llvm_facts(
            "llvm.nvvm.read.ptx.sreg.nctaid.x",
            None,
            &[],
            &["i32"],
            &[
                "IntrNoMem",
                "IntrSpeculatable",
                "NoUndef<ret>",
                "Range<ret,1,2147483648>",
            ],
            true,
            Some(("1", "2147483648")),
        );
        let canonical = r#"
declare noundef range(i32 1, -2147483648) i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
attributes #0 = { speculatable memory(none) }
"#;

        assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap();
    }

    #[test]
    fn fails_when_a_required_attribute_is_only_mentioned_outside_the_declaration() {
        let llvm = llvm_facts(
            "llvm.nvvm.redux.sync.add",
            None,
            &["i32", "i32"],
            &["i32"],
            &["IntrConvergent", "IntrInaccessibleMemOnly"],
            false,
            None,
        );
        let canonical = r#"
; convergent memory(inaccessiblemem: readwrite)
declare i32 @llvm.nvvm.redux.sync.add(i32, i32) #0
attributes #0 = { nounwind }
"#;

        let error = assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap_err();
        assert!(error.to_string().contains("missing \"convergent\""));
    }

    #[test]
    fn unsupported_imported_properties_fail_closed() {
        let llvm = llvm_facts(
            "llvm.nvvm.test",
            None,
            &[],
            &["i32"],
            &["UnmodeledProperty"],
            false,
            None,
        );
        let canonical = "declare i32 @llvm.nvvm.test()\n";

        let error = assert_canonical_intrinsic_declaration(canonical, &llvm).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("unsupported imported LLVM property")
        );
    }
}
