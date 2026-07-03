/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::export::render_llvm_ir;
use crate::llvm_tools::LlvmToolchain;
use crate::lower::lower_to_llvm;
use crate::options::{BackendOptions, PtxConfig};
use crate::target::{
    detect_module_requirements_in_llvm_file, required_ptx_feature, resolve_ptx_target,
    validate_target_for_llvm_major,
};
use crate::verify::verify_operation;
use llvm_export::export::DebugKind;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Per-process call counter appended to the scratch directory name so two
/// calls landing in the same nanosecond still get distinct directories.
static CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Compiles a `dialect-mir` module to PTX bytes without rustc.
///
/// This is the front-end-agnostic hook into cuda-oxide's back end: it
/// replicates the post-translation tail of `run_pipeline` (verify the
/// module, lower MIR to the LLVM dialect, render `.ll`, run `opt`/`llc`,
/// return the PTX bytes) on a module the caller assembled by any means, so it
/// depends on neither CubeCL nor rustc. The `module_op` must be a
/// `pliron::builtin::ops::ModuleOp` built against a [`Context`] whose dialects
/// were registered via [`crate::register_backend_dialects`].
///
/// # Thread safety
///
/// This function reads no process-global state other than a private atomic
/// call counter used only to name its own scratch directory: it derives a
/// [`BackendOptions`] from `cfg` (see [`BackendOptions::from_ptx_config`]) and
/// threads it explicitly through every helper, and it stages its `.ll`/`.ptx`
/// files in a per-call temp directory named from the process id, a nanosecond
/// timestamp, and that call counter, so the directory name is unique per call
/// even for two calls landing in the same nanosecond. It is therefore safe to
/// call concurrently from multiple threads of the same process.
///
/// # Linking
///
/// Unlike functions in mir-importer which require `#![feature(rustc_private)]`
/// and `extern crate rustc_driver;`, this function requires neither.
/// cuda-oxide-codegen does not depend on rustc or any compiler internals, so a
/// consumer needs no `rustc_private` feature gate and no toolchain matched to
/// a specific nightly's `rustc_driver`; the only toolchain requirement is
/// whatever this crate's ordinary dependencies need.
pub fn compile_to_ptx(
    ctx: &mut Context,
    module_op: Ptr<Operation>,
    cfg: &PtxConfig,
) -> Result<Vec<u8>, crate::PtxError> {
    use crate::PtxError;

    if cfg.target_arch.trim().is_empty() {
        return Err(PtxError::InvalidConfig("empty target_arch".into()));
    }

    // Build the options this call uses; trims whitespace from target_arch so
    // " sm_120 " cannot reach llc -mcpu. No process-global state is touched.
    let opts = BackendOptions::from_ptx_config(cfg);

    let debug_kind = if cfg.debug {
        DebugKind::Full
    } else {
        DebugKind::Off
    };

    // 1. Verify the module before lowering.
    verify_operation(ctx, module_op, "module")?;

    // 2. Lower dialect-mir -> LLVM dialect (registers mir-lower, runs
    //    `lower_mir_to_llvm_with_options`). `cfg.fma` allows fmul+fadd
    //    contraction at the IR level, matching the llc `-fp-contract` gate.
    lower_to_llvm(ctx, module_op, cfg.fma)?;

    // 3. Render the standard PTX `.ll` (no device externs, no NVVM dialect).
    let llvm_ir = render_llvm_ir(ctx, module_op, &[], false, None, debug_kind)?;

    // 4. Stage the `.ll` in a unique temp dir and run opt (gated on
    //    `opts.no_opt`, derived above) + llc via the existing helper, which
    //    resolves the matched opt/llc toolchain itself.
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let call = CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "cuda_oxide_ptx_{}_{}_{}",
        std::process::id(),
        unique,
        call
    ));
    std::fs::create_dir_all(&dir)
        .map_err(|e| PtxError::Codegen(format!("create temp dir: {e}")))?;
    let ll_path = dir.join("module.ll");
    let ptx_path = dir.join("module.ptx");
    std::fs::write(&ll_path, &llvm_ir).map_err(|e| PtxError::Codegen(format!("write .ll: {e}")))?;

    let result = generate_ptx(&ll_path, &ptx_path, debug_kind, &opts)
        .map_err(PtxError::from)
        .and_then(|_target| {
            std::fs::read(&ptx_path).map_err(|e| PtxError::Codegen(format!("read .ptx: {e}")))
        });

    // Best-effort cleanup of the scratch dir regardless of outcome.
    let _ = std::fs::remove_dir_all(&dir);

    result
}

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
/// Returns the optimised `.ll` path, or `None` when the middle-end is off
/// (`opts.no_opt`, historically `CUDA_OXIDE_NO_OPT=1`), no same-major `opt`
/// exists, or the chosen `opt` fails at runtime; the caller then feeds the
/// unoptimised `ll_path` to `llc`, which is always safe.
fn optimize_ll(
    ll_path: &Path,
    toolchain: &LlvmToolchain,
    opts: &BackendOptions,
) -> Option<PathBuf> {
    let opt = toolchain.opt.as_ref()?;

    let opt_ll = ll_path.with_extension("opt.ll");
    match std::process::Command::new(&opt.path)
        .arg("-O2")
        .arg(ll_path)
        .arg("-S")
        .arg("-o")
        .arg(&opt_ll)
        .output()
    {
        Ok(o) if o.status.success() => {
            if opts.verbose {
                eprintln!("opt -O2 via {}: {}", opt.path, opt_ll.display());
            }
            Some(opt_ll)
        }
        Ok(o) => {
            // The matched opt exists but rejected the input. Warn loudly
            // (there is no second candidate any more) and fall back to
            // unoptimised IR rather than to a different LLVM major.
            eprintln!(
                "warning: opt ({}) failed; continuing with unoptimised IR:\n{}",
                opt.path,
                String::from_utf8_lossy(&o.stderr).trim()
            );
            None
        }
        Err(e) => {
            eprintln!(
                "warning: opt ({}): {e}; continuing with unoptimised IR",
                opt.path
            );
            None
        }
    }
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
) -> Result<String, PipelineError> {
    // Explicit, hard override: `--arch` or a caller-set `opts.target_arch`.
    let explicit_override = opts.target_arch.clone();
    // Advisory hint: the arch of the GPU in this machine, forwarded by
    // `cargo oxide run`. Used only when that GPU can actually run the kernel.
    let device_hint = opts.device_arch_hint.clone();

    let requirements = detect_module_requirements_in_llvm_file(ll_path)?;
    let detected = requirements.features;

    // Resolve the final target:
    //   1. explicit override -- accepted only if it can lower the kernel's
    //      features; reject an invalid floor before llc emits unusable PTX.
    //   2. detected-device hint -- used only if that GPU can run the kernel;
    //      otherwise we build for the feature floor. The resulting PTX will not
    //      load on this GPU, but feature-gated examples handle that at load time
    //      (cuModuleLoad reports INVALID_PTX and they skip execution).
    //   3. neither set -- the feature floor.
    let (target, target_source) =
        resolve_ptx_target(explicit_override.as_deref(), device_hint.as_deref(), detected)?;

    let verbose = opts.verbose;
    if verbose {
        eprintln!("Target: {target} (from {target_source}; detected {detected:?})");
    }

    // Resolve `opt` and `llc` as a matched pair (issue #150): llc first
    // (opts.llc_override, historically CUDA_OXIDE_LLC, then the Rust toolchain's
    // llvm-tools llc, then llc-22 / llc-21 on PATH -- newest first), then an opt
    // of the same LLVM major. LLVM 21 is the floor: older releases reject the
    // modern TMA / tcgen05 / WGMMA intrinsic signatures cuda-oxide emits.
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
    validate_target_for_llvm_major(&target, toolchain.llc_major)
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
        if verbose {
            eprintln!("Skipping opt -O2 (full debug keeps locals inspectable)");
        }
        None
    } else {
        optimize_ll(ll_path, &toolchain, opts)
    };
    let llc_input: &Path = optimized.as_deref().unwrap_or(ll_path);

    if verbose {
        let source = if toolchain.llc_from_env {
            "from opts.llc_override"
        } else {
            "auto-detected"
        };
        eprintln!("Using llc: {} ({source})", toolchain.llc_description());
    }
    let llc_desc = if toolchain.llc_from_env {
        format!("llc_override ({})", toolchain.llc_path)
    } else {
        format!("llc ({})", toolchain.llc_path)
    };

    let mut llc_cmd = std::process::Command::new(&toolchain.llc_path);
    llc_cmd.arg("-march=nvptx64").arg(format!("-mcpu={}", target));
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
                if verbose {
                    eprintln!(
                        "line-table debug: stripped PTX target debug flag; source line tables remain"
                    );
                }
            }
            Ok(target.to_string())
        }
        Ok(output) => Err(PipelineError::PtxGeneration(format!(
            "{} failed:\n{}",
            llc_desc,
            String::from_utf8_lossy(&output.stderr).trim()
        ))),
        Err(e) => Err(PipelineError::PtxGeneration(format!("{llc_desc}: {e}"))),
    }
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

#[cfg(test)]
mod env_independence {
    use super::*;
    use crate::options::PtxConfig;
    use crate::register_backend_dialects;
    use pliron::context::Context;
    use pliron::op::Op;

    /// The backend must neither read nor mutate CUDA_OXIDE_* process env.
    #[test]
    fn compile_to_ptx_ignores_and_preserves_env() {
        // SAFETY: single-threaded test binary section; set_var is process-global.
        unsafe { std::env::set_var("CUDA_OXIDE_TARGET", "sm_80") };
        let mut ctx = Context::new();
        register_backend_dialects(&mut ctx);
        let module = pliron::builtin::ops::ModuleOp::new(&mut ctx, "env_probe".try_into().unwrap());
        let cfg = PtxConfig::new("sm_120");
        let ptx = compile_to_ptx(&mut ctx, module.get_operation(), &cfg).unwrap();
        let text = String::from_utf8_lossy(&ptx);
        assert!(text.contains(".target sm_120"), "config must win: {text}");
        assert_eq!(
            std::env::var("CUDA_OXIDE_TARGET").as_deref(),
            Ok("sm_80"),
            "backend must not mutate process env"
        );
    }
}
