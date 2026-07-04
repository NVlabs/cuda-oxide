/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::model::{EvidenceStageKind, IntrinsicBackend};
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
    if record.ldmatrix.is_some() && mode == ProbeMode::SelectedEvidence {
        assert_intrinsic_declaration_canonicalizes(&llc, &input, &output_dir, intrinsic_id)?;
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
    ensure!(
        record.expected_ptx.matches(&ptx),
        "probe PTX has no instruction matching `{}`",
        record.expected_ptx
    );
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

fn assert_intrinsic_declaration_canonicalizes(
    llc: &Path,
    input: &Path,
    output_dir: &Path,
    intrinsic_id: &str,
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
    for required in [
        "convergent nocallback",
        "memory(argmem: read)",
        "readonly captures(none)",
    ] {
        ensure!(
            canonical.contains(required),
            "canonicalized intrinsic declaration is missing {required:?}"
        );
    }
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
}
