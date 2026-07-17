/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::generated::GeneratedModuleRequirements;
use crate::llvm_tools::LlvmToolchain;
use crate::options::BackendOptions;
use crate::target::{
    detect_module_requirements_in_llvm_file, merge_generated_module_requirements,
    merge_generated_module_requirements_for_target, required_ptx_feature,
    resolve_ptx_target_with_generated, validate_ptx_isa_for_llvm_major,
    validate_target_for_llvm_major,
};
use llvm_export::export::DebugKind;
use std::path::{Path, PathBuf};

/// Runs LLVM's middle-end (`opt -O2`) on the emitted IR before `llc`.
///
/// This is what consumes the per-op ABI alignment we emit: the
/// LoadStoreVectorizer fuses aligned aggregate/element accesses, SROA
/// scalarizes stack aggregates, and InferAddressSpaces promotes generic
/// pointers to `.global` (LDG/STG). Gated on alignment — fusion only fires
/// when loads/stores carry matching `align N` hints.
///
/// The `opt` binary comes from the resolved [`LlvmToolchain`], which
/// guarantees it shares the LLVM major of the `llc` that will consume its
/// output (issue #150: an LLVM 22 `opt` emits sizeless
/// `llvm.lifetime.start/end` intrinsics that an LLVM 21 `llc` rejects).
///
/// Returns the optimized path plus caller-owned diagnostics. Experimental v1
/// is strict; the legacy rustc path retains its warn-and-continue behavior.
fn optimize_ll(
    ll_path: &Path,
    toolchain: &LlvmToolchain,
    opts: &BackendOptions,
    strict: bool,
) -> Result<(Option<PathBuf>, Vec<String>), PipelineError> {
    if opts.no_opt {
        return Ok((None, Vec::new()));
    }
    let Some(opt) = toolchain.opt.as_ref() else {
        if strict {
            return Err(PipelineError::Optimization(
            "optimization was requested, but no `opt` matching the selected `llc` is available; \
             install the matching LLVM tools or explicitly disable optimization"
                .to_string(),
            ));
        }
        return Ok((None, Vec::new()));
    };

    let opt_ll = ll_path.with_extension("opt.ll");
    match std::process::Command::new(&opt.path)
        .arg("-O2")
        .arg(ll_path)
        .arg("-S")
        .arg("-o")
        .arg(&opt_ll)
        .output()
    {
        Ok(output) if output.status.success() => {
            let diagnostics = opts
                .verbose
                .then(|| format!("opt -O2 via {}: {}", opt.path, opt_ll.display()))
                .into_iter()
                .collect();
            Ok((Some(opt_ll), diagnostics))
        }
        Ok(output) => {
            let message = format!(
                "opt ({}) failed with status {}:\n{}",
                opt.path,
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            );
            if strict {
                Err(PipelineError::Optimization(message))
            } else {
                Ok((
                    None,
                    vec![format!(
                        "warning: {message}\nwarning: continuing with unoptimized IR"
                    )],
                ))
            }
        }
        Err(error) => {
            let message = format!("failed to run opt ({}): {error}", opt.path);
            if strict {
                Err(PipelineError::Optimization(message))
            } else {
                Ok((
                    None,
                    vec![format!(
                        "warning: {message}; continuing with unoptimized IR"
                    )],
                ))
            }
        }
    }
}

/// Legacy rustc-pipeline result, including messages the CLI should print.
#[doc(hidden)]
#[allow(missing_docs)]
#[derive(Debug)]
pub struct GeneratedPtx {
    pub target: String,
    pub diagnostics: Vec<String>,
}

struct PtxBackend<'a> {
    options: &'a BackendOptions,
    toolchain: &'a LlvmToolchain,
    generated: &'a GeneratedModuleRequirements,
}

/// Generates PTX from LLVM IR using `llc`.
///
/// LLVM 21+ is the minimum supported version: earlier `llc` releases reject
/// the modern TMA / tcgen05 / WGMMA intrinsic signatures that cuda-oxide emits
/// (e.g. the 10-operand `llvm.nvvm.cp.async.bulk.tensor.g2s.tile.2d` with
/// `addrspace(7)` + CTA group parameter requires LLVM 21). If
/// `opts.llc_override` (historically `CUDA_OXIDE_LLC`) is set, it is used
/// exclusively; power users can point it at an older `llc` at their own risk.
///
/// `opt` and `llc` are resolved together via [`LlvmToolchain`] so the
/// middle-end never runs under a different LLVM major than the backend
/// (issue #150).
///
/// Target arch resolves (highest priority first) to: `opts.target_arch`
/// (historically `CUDA_OXIDE_TARGET`), else the detected-GPU hint
/// `opts.device_arch_hint` (historically `CUDA_OXIDE_DEVICE_ARCH`) when that
/// GPU can run the kernel, else the minimum arch the IR's features require.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub fn generate_ptx(
    ll_path: &Path,
    ptx_path: &Path,
    debug_kind: DebugKind,
    opts: &BackendOptions,
    diagnostic_sink: Option<fn(&str)>,
    generated: &GeneratedModuleRequirements,
) -> Result<GeneratedPtx, PipelineError> {
    let Some(toolchain) = LlvmToolchain::resolve(opts) else {
        return Err(PipelineError::PtxGeneration(
            "No working llc found.\n\
             cuda-oxide tries (in order): opts.llc_override (CUDA_OXIDE_LLC), the \
             Rust toolchain's llvm-tools llc, then llc-22 / llc-21 on PATH. \
             LLVM 21+ is required (earlier versions reject the TMA / tcgen05 / \
             WGMMA intrinsic signatures we emit).\n\
             Easiest fix: `rustup component add llvm-tools` (auto-picked up).\n\
             Alternative: `sudo apt install llvm-21` (or `llvm-22`).\n\
             Or set opts.llc_override (CUDA_OXIDE_LLC) to a specific binary."
                .to_string(),
        ));
    };
    let mut diagnostics = toolchain.diagnostics.clone();
    if !opts.no_opt && toolchain.opt.is_none() {
        diagnostics.push(
            "warning: continuing with unoptimized IR (as with CUDA_OXIDE_NO_OPT=1)".to_string(),
        );
    }
    if opts.verbose {
        diagnostics.push(format!(
            "LLVM toolchain: llc = {}, opt = {}",
            crate::llvm_tools::describe_tool(&toolchain.llc_path, toolchain.llc_major),
            match &toolchain.opt {
                Some(tool) => crate::llvm_tools::describe_tool(&tool.path, tool.major),
                None => "(skipped)".to_string(),
            }
        ));
    }
    emit_diagnostics(diagnostic_sink, &diagnostics);
    let mut generated = generate_ptx_impl(
        ll_path,
        ptx_path,
        debug_kind,
        PtxBackend {
            options: opts,
            toolchain: &toolchain,
            generated,
        },
        false,
        diagnostic_sink,
    )?;
    diagnostics.append(&mut generated.diagnostics);
    generated.diagnostics = diagnostics;
    Ok(generated)
}

/// Generate PTX with an already-resolved toolchain.
///
/// The experimental compiler uses this entry point so discovery is explicit
/// and one [`LlvmToolchain`] can be reused across compilations.
pub(crate) fn generate_ptx_with_toolchain(
    ll_path: &Path,
    ptx_path: &Path,
    debug_kind: DebugKind,
    opts: &BackendOptions,
    toolchain: &LlvmToolchain,
    generated: &GeneratedModuleRequirements,
) -> Result<String, PipelineError> {
    generate_ptx_impl(
        ll_path,
        ptx_path,
        debug_kind,
        PtxBackend {
            options: opts,
            toolchain,
            generated,
        },
        true,
        None,
    )
    .map(|generated| generated.target)
}

fn generate_ptx_impl(
    ll_path: &Path,
    ptx_path: &Path,
    debug_kind: DebugKind,
    backend: PtxBackend<'_>,
    strict_optimization: bool,
    diagnostic_sink: Option<fn(&str)>,
) -> Result<GeneratedPtx, PipelineError> {
    let PtxBackend {
        options: opts,
        toolchain,
        generated,
    } = backend;
    // Explicit, hard override: `--arch` or a caller-set `opts.target_arch`.
    let explicit_override = opts.target_arch.clone();
    // Advisory hint: the arch of the GPU in this machine, forwarded by
    // `cargo oxide run`. Used only when that GPU can actually run the kernel.
    let device_hint = opts.device_arch_hint.clone();

    let requirements = merge_generated_module_requirements(
        detect_module_requirements_in_llvm_file(ll_path)?,
        generated,
    )
    .map_err(PipelineError::PtxGeneration)?;
    let detected = requirements.features;

    // Resolve the final target:
    //   1. explicit override -- accepted only if it can lower the kernel's
    //      features; reject an invalid floor before llc emits unusable PTX.
    //   2. detected-device hint -- used only if that GPU can run the kernel;
    //      otherwise we build for the feature floor. The resulting PTX will not
    //      load on this GPU, but feature-gated examples handle that at load time
    //      (cuModuleLoad reports INVALID_PTX and they skip execution).
    //   3. neither set -- the feature floor.
    let (target, target_source) = resolve_ptx_target_with_generated(
        explicit_override.as_deref(),
        device_hint.as_deref(),
        detected,
        generated,
    )?;
    let requirements =
        merge_generated_module_requirements_for_target(requirements, generated, &target)
            .map_err(PipelineError::PtxGeneration)?;

    let mut diagnostics = Vec::new();
    if opts.verbose {
        record_diagnostic(
            &mut diagnostics,
            diagnostic_sink,
            format!("Target: {target} (from {target_source}; detected {detected:?})"),
        );
    }

    validate_target_for_llvm_major(&target, toolchain.llc_major)
        .map_err(PipelineError::PtxGeneration)?;
    validate_ptx_isa_for_llvm_major(requirements.ptx_isa, toolchain.llc_major)
        .map_err(PipelineError::PtxGeneration)?;

    // Run the LLVM middle-end (opt -O2) before llc. Feature detection above
    // intentionally reads the original (pre-opt) IR so the target is determined
    // by what the source actually needs, not what opt elides.
    //
    // Full-debug is a `-G`-style build: it keeps every local in memory and
    // describes it with `llvm.dbg.declare`. Running `opt -O2` would promote
    // those slots to registers and collapse their live ranges, turning most
    // in-scope locals into `<optimized out>` under cuda-gdb. So we feed the
    // unoptimized IR straight to llc when variable info is requested, matching
    // nvcc `-G`. (llc itself is invoked at `-O0` for the same builds below.)
    let optimized = if debug_kind.variables_enabled() {
        if opts.verbose {
            record_diagnostic(
                &mut diagnostics,
                diagnostic_sink,
                "Skipping opt -O2 (full debug keeps locals inspectable)".to_string(),
            );
        }
        None
    } else {
        let (optimized, mut opt_diagnostics) =
            optimize_ll(ll_path, toolchain, opts, strict_optimization)?;
        for diagnostic in opt_diagnostics.drain(..) {
            record_diagnostic(&mut diagnostics, diagnostic_sink, diagnostic);
        }
        optimized
    };
    let llc_input: &Path = optimized.as_deref().unwrap_or(ll_path);

    let llc_desc = if toolchain.llc_from_env {
        format!("llc_override ({})", toolchain.llc_path)
    } else {
        format!("llc ({})", toolchain.llc_path)
    };
    if opts.verbose {
        let source = if toolchain.llc_from_env {
            "from opts.llc_override"
        } else {
            "auto-detected"
        };
        record_diagnostic(
            &mut diagnostics,
            diagnostic_sink,
            format!(
                "Using llc: {} ({source})",
                crate::llvm_tools::describe_tool(&toolchain.llc_path, toolchain.llc_major)
            ),
        );
    }

    let mut llc_cmd = std::process::Command::new(&toolchain.llc_path);
    llc_cmd
        .arg("-march=nvptx64")
        .arg(format!("-mcpu={}", target));
    if let Some(feature) = required_ptx_feature(&target, requirements.ptx_isa) {
        llc_cmd.arg(format!("-mattr={feature}"));
    }
    // Full-debug (`-G`-style): run llc at -O0 so its own mem2reg/SROA does not
    // promote the stack slots we deliberately kept in memory, which would
    // invalidate the `llvm.dbg.declare` locations cuda-gdb reads.
    if debug_kind.variables_enabled() {
        llc_cmd.arg("-O0");
    }
    // Fuse fmul+fadd/fsub into fma.rn.f32, matching nvcc's default --fmad=true.
    // The IR-side `contract` flag (set during lowering when contraction is
    // allowed) grants permission; this llc flag activates the NVPTX backend's
    // contract mode. `opts.no_fma` (allow_fma_contraction = !no_fma) drives both
    // stages, so IR permission and this backend gate cannot disagree.
    if !opts.no_fma {
        llc_cmd.arg("-fp-contract=fast");
    }
    let result = llc_cmd.arg(llc_input).arg("-o").arg(ptx_path).output();

    match result {
        Ok(output) if output.status.success() => {
            if matches!(debug_kind, DebugKind::LineTables) {
                strip_target_debug_from_ptx(ptx_path)?;
                if opts.verbose {
                    record_diagnostic(
                        &mut diagnostics,
                        diagnostic_sink,
                        "line-table debug: stripped PTX target debug flag; source line tables remain"
                            .to_string(),
                    );
                }
            }
            Ok(GeneratedPtx {
                target: target.to_string(),
                diagnostics,
            })
        }
        Ok(output) => Err(PipelineError::PtxGeneration(format!(
            "{} failed:\n{}",
            llc_desc,
            String::from_utf8_lossy(&output.stderr).trim()
        ))),
        Err(e) => Err(PipelineError::PtxGeneration(format!("{llc_desc}: {e}"))),
    }
}

fn emit_diagnostics(sink: Option<fn(&str)>, diagnostics: &[String]) {
    if let Some(sink) = sink {
        for diagnostic in diagnostics {
            sink(diagnostic);
        }
    }
}

fn record_diagnostic(diagnostics: &mut Vec<String>, sink: Option<fn(&str)>, diagnostic: String) {
    if let Some(sink) = sink {
        sink(&diagnostic);
    }
    diagnostics.push(diagnostic);
}

fn strip_target_debug_from_ptx(ptx_path: &Path) -> Result<(), PipelineError> {
    let ptx = std::fs::read_to_string(ptx_path).map_err(|e| {
        PipelineError::PtxGeneration(format!(
            "failed to read PTX for line-table debug cleanup ({}): {e}",
            ptx_path.display()
        ))
    })?;
    let stripped = strip_target_debug_from_ptx_text(&ptx);
    if stripped != ptx {
        std::fs::write(ptx_path, stripped).map_err(|e| {
            PipelineError::PtxGeneration(format!(
                "failed to write PTX after line-table debug cleanup ({}): {e}",
                ptx_path.display()
            ))
        })?;
    }
    Ok(())
}

fn strip_target_debug_from_ptx_text(ptx: &str) -> String {
    let mut out = String::with_capacity(ptx.len());
    for line in ptx.split_inclusive('\n') {
        let (line_body, newline) = line
            .strip_suffix('\n')
            .map_or((line, ""), |without_newline| (without_newline, "\n"));
        out.push_str(&strip_target_debug_from_ptx_line(line_body));
        out.push_str(newline);
    }
    out
}

fn strip_target_debug_from_ptx_line(line: &str) -> String {
    let indent_len = line.len() - line.trim_start().len();
    let indent = &line[..indent_len];
    let body = &line[indent_len..];
    let Some(rest) = body.strip_prefix(".target") else {
        return line.to_string();
    };

    let mut parts = rest.split(',');
    let Some(arch) = parts.next() else {
        return line.to_string();
    };

    let options: Vec<&str> = parts
        .map(str::trim)
        .filter(|option| *option != "debug")
        .collect();
    if !rest
        .split(',')
        .skip(1)
        .any(|option| option.trim() == "debug")
    {
        return line.to_string();
    }

    let mut stripped = format!("{indent}.target{arch}");
    for option in options {
        stripped.push_str(", ");
        stripped.push_str(option);
    }
    stripped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    static LEGACY_DIAGNOSTICS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

    #[cfg(unix)]
    fn collect_legacy_diagnostic(message: &str) {
        LEGACY_DIAGNOSTICS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(message.to_string());
    }

    #[test]
    #[cfg(unix)]
    fn legacy_opt_failure_warns_but_experimental_mode_fails() {
        let toolchain = LlvmToolchain {
            llc_path: "/bin/true".to_string(),
            llc_major: Some(21),
            llc_from_env: false,
            opt: Some(crate::llvm_tools::OptTool {
                path: "/bin/false".to_string(),
                major: Some(21),
            }),
            diagnostics: Vec::new(),
        };
        let opts = BackendOptions::default();
        let input = Path::new("unused.ll");

        let (optimized, diagnostics) = optimize_ll(input, &toolchain, &opts, false).unwrap();
        assert!(optimized.is_none());
        assert!(diagnostics[0].contains("continuing with unoptimized IR"));

        let error = optimize_ll(input, &toolchain, &opts, true).unwrap_err();
        assert!(matches!(&error, PipelineError::Optimization(_)));
        assert!(error.to_string().contains("opt (/bin/false) failed"));
    }

    #[test]
    #[cfg(unix)]
    fn legacy_tool_warnings_survive_a_later_llc_failure() {
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_legacy_diagnostics_{}",
            std::process::id()
        ));
        std::fs::create_dir(&root).unwrap();
        let ll_path = root.join("module.ll");
        let ptx_path = root.join("module.ptx");
        let llc_path = root.join("llc-999");
        std::fs::write(&ll_path, "define void @kernel() { ret void }\n").unwrap();
        std::fs::write(
            &llc_path,
            "#!/bin/sh\nif [ \"${1:-}\" = \"--version\" ]; then echo 'LLVM version 999.0.0'; exit 0; fi\necho 'deliberate llc failure' >&2\nexit 1\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = std::fs::metadata(&llc_path).unwrap().permissions();
        permissions.set_mode(0o700);
        std::fs::set_permissions(&llc_path, permissions).unwrap();

        LEGACY_DIAGNOSTICS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        let opts = BackendOptions {
            target_arch: Some("sm_80".to_string()),
            no_opt: false,
            llc_override: Some(llc_path),
            ..BackendOptions::default()
        };
        let error = generate_ptx(
            &ll_path,
            &ptx_path,
            DebugKind::Off,
            &opts,
            Some(collect_legacy_diagnostic),
            &GeneratedModuleRequirements::default(),
        )
        .unwrap_err();
        assert!(matches!(error, PipelineError::PtxGeneration(_)));

        let diagnostics = LEGACY_DIAGNOSTICS
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            diagnostics
                .iter()
                .any(|message| message.contains("LLVM optimization is unavailable")),
            "{diagnostics:?}"
        );
        assert!(
            diagnostics
                .iter()
                .any(|message| message.contains("continuing with unoptimized IR")),
            "{diagnostics:?}"
        );
        drop(diagnostics);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn line_table_ptx_cleanup_strips_only_target_debug_flag() {
        let ptx = "\
.version 8.9
.target sm_120a, debug
.address_size 64

.section .debug_info
\t.b8 1;
";

        let stripped = strip_target_debug_from_ptx_text(ptx);

        assert!(
            stripped.contains(".target sm_120a\n"),
            "line-table mode should not ask the driver for debug compilation:\n{stripped}"
        );
        assert!(
            stripped.contains(".section .debug_info"),
            "line-table mode must keep the emitted DWARF sections:\n{stripped}"
        );
    }

    #[test]
    fn line_table_ptx_cleanup_preserves_other_target_options() {
        let ptx = ".target sm_90a, texmode_independent, debug\n";

        let stripped = strip_target_debug_from_ptx_text(ptx);

        assert_eq!(stripped, ".target sm_90a, texmode_independent\n");
    }
}
