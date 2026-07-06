/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! External post-IR hooks.
//!
//! The IR-stage analogue of the `llc_override` escape hatch:
//! [`BackendOptions::post_ir_hooks`](crate::options::BackendOptions::post_ir_hooks)
//! names programs that are run in order on the exported `.ll` after IR export,
//! before the IR is consumed by PTX generation (or returned as the NVVM-IR
//! artifact). The rustc frontend fills the list from `CUDA_OXIDE_POST_IR` (a
//! PATH-style list) at its own boundary; this module never reads the
//! environment.
//!
//! Each hook is invoked as
//!
//! ```text
//! <hook> <ll_path> <output_dir> <output_name> <target>
//! ```
//!
//! (`<target>` is the explicit target arch, possibly empty) and may rewrite
//! `<ll_path>` in place; the next hook — and ultimately code generation — sees
//! the edits. Hooks are transform-only: stdout is ignored and the pipeline
//! always continues with its own code generation. Exit 0 means success; a
//! non-zero exit aborts the build, surfacing the hook's stderr. Uses include
//! custom LLVM passes/plugins, instrumentation, external-bitcode linking, and
//! automatic differentiation (see the `post_ir_hook` and `enzyme_autodiff`
//! examples).

use std::path::Path;

use crate::error::PipelineError;
use crate::pipeline::PipelineTrace;

/// Runs `hooks` in order on the exported `.ll`; see the module docs for the
/// contract. The hook's `<output_dir>`/`<output_name>` arguments are derived
/// from `ll_path`, whose file name is always `<output_name>.ll`.
pub(crate) fn run_post_ir_hooks(
    hooks: &[impl AsRef<Path>],
    ll_path: &Path,
    target: &str,
    trace: &PipelineTrace,
) -> Result<(), PipelineError> {
    for hook in hooks {
        run_one_post_ir_hook(hook.as_ref(), ll_path, target, trace)?;
    }
    Ok(())
}

/// Runs a single post-IR hook program.
fn run_one_post_ir_hook(
    hook: &Path,
    ll_path: &Path,
    target: &str,
    trace: &PipelineTrace,
) -> Result<(), PipelineError> {
    if trace.verbose {
        trace.emit(format!(
            "\n=== Running post-IR hook: {} ===",
            hook.display()
        ));
    }

    let output_dir = ll_path.parent().unwrap_or(Path::new(""));
    let output_name = ll_path.file_stem().unwrap_or_default();
    let output = std::process::Command::new(hook)
        .arg(ll_path)
        .arg(output_dir)
        .arg(output_name)
        .arg(target)
        .output()
        .map_err(|e| PipelineError::PostIr(format!("failed to spawn '{}': {e}", hook.display())))?;

    if !output.status.success() {
        return Err(PipelineError::PostIr(format!(
            "'{}' exited with {}:\n{}",
            hook.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// A fresh per-test directory holding the `.ll` and the hook scripts.
    fn hook_test_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "cuda_oxide_post_ir_{}_{}",
            std::process::id(),
            name
        ));
        fs::create_dir_all(&dir).expect("create hook test dir");
        dir
    }

    /// Writes an executable shell script for use as a post-IR hook.
    #[cfg(unix)]
    fn write_hook_script(dir: &Path, name: &str, body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let path = dir.join(name);
        fs::write(&path, format!("#!/bin/sh\n{body}\n")).expect("write hook script");
        fs::set_permissions(&path, fs::Permissions::from_mode(0o755)).expect("chmod hook script");
        path
    }

    /// An empty hook list is a no-op: nothing is spawned, the IR is untouched.
    /// (The unset-env case reduces to this: `BackendOptions::from_env` yields
    /// an empty `post_ir_hooks` when `CUDA_OXIDE_POST_IR` is not set.)
    #[test]
    fn empty_hook_list_leaves_ir_untouched() {
        let dir = hook_test_dir("empty");
        let ll_path = dir.join("kernel.ll");
        fs::write(&ll_path, "; original IR\n").unwrap();

        let hooks: &[PathBuf] = &[];
        run_post_ir_hooks(hooks, &ll_path, "sm_70", &PipelineTrace::default())
            .expect("empty list is ok");

        assert_eq!(fs::read_to_string(&ll_path).unwrap(), "; original IR\n");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A hook is invoked as `<hook> <ll_path> <output_dir> <output_name>
    /// <target>` and may rewrite the `.ll` in place.
    #[cfg(unix)]
    #[test]
    fn hook_rewrites_ir_in_place() {
        let dir = hook_test_dir("rewrite");
        let ll_path = dir.join("kernel.ll");
        fs::write(&ll_path, "define void @f() {\n  ret void\n}\n").unwrap();

        // Rewrite the IR and record the argv the hook was given.
        let hook = write_hook_script(
            &dir,
            "rewrite.sh",
            r#"sed -i 's/@f/@g/' "$1"
printf '; argv: %s %s %s\n' "$2" "$3" "$4" >> "$1""#,
        );

        run_post_ir_hooks(&[hook], &ll_path, "sm_70", &PipelineTrace::default())
            .expect("hook succeeds");

        let rewritten = fs::read_to_string(&ll_path).unwrap();
        assert!(
            rewritten.contains("@g"),
            "in-place edit visible: {rewritten}"
        );
        assert!(!rewritten.contains("@f"), "old symbol gone: {rewritten}");
        assert!(
            rewritten.contains(&format!("; argv: {} kernel sm_70", dir.display())),
            "hook argv mismatch: {rewritten}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// Multiple hooks run in list order, each seeing the previous one's edits.
    #[cfg(unix)]
    #[test]
    fn hooks_chain_in_order() {
        let dir = hook_test_dir("chain");
        let ll_path = dir.join("kernel.ll");
        fs::write(&ll_path, "start\n").unwrap();

        let first = write_hook_script(
            &dir,
            "first.sh",
            r#"echo "first saw: $(head -n 1 "$1")" >> "$1""#,
        );
        let second = write_hook_script(
            &dir,
            "second.sh",
            r#"echo "second saw: $(tail -n 1 "$1")" >> "$1""#,
        );

        run_post_ir_hooks(&[first, second], &ll_path, "", &PipelineTrace::default()).unwrap();

        assert_eq!(
            fs::read_to_string(&ll_path).unwrap(),
            "start\nfirst saw: start\nsecond saw: first saw: start\n"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A non-zero exit aborts the build surfacing the hook's stderr, and later
    /// hooks in the list must not run.
    #[cfg(unix)]
    #[test]
    fn hook_failure_aborts_with_stderr() {
        let dir = hook_test_dir("fail");
        let ll_path = dir.join("kernel.ll");
        fs::write(&ll_path, "; original IR\n").unwrap();

        let failing = write_hook_script(
            &dir,
            "failing.sh",
            "echo 'enzyme: no differentiable function found' >&2\nexit 3",
        );
        let never_runs = write_hook_script(&dir, "never.sh", r#"echo TAINTED >> "$1""#);

        let err = run_post_ir_hooks(
            &[failing, never_runs],
            &ll_path,
            "",
            &PipelineTrace::default(),
        )
        .expect_err("non-zero exit must abort");

        match &err {
            PipelineError::PostIr(msg) => {
                assert!(
                    msg.contains("no differentiable function found"),
                    "stderr surfaced: {msg}"
                );
                assert!(msg.contains("exit status: 3"), "exit code surfaced: {msg}");
            }
            other => panic!("expected PostIr error, got {other:?}"),
        }
        assert_eq!(
            fs::read_to_string(&ll_path).unwrap(),
            "; original IR\n",
            "later hooks must not run after a failure"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A hook that cannot be spawned surfaces a `PostIr` error.
    #[test]
    fn missing_hook_fails_to_spawn() {
        let dir = hook_test_dir("missing");
        let ll_path = dir.join("kernel.ll");
        fs::write(&ll_path, "; original IR\n").unwrap();

        let err = run_post_ir_hooks(
            &[PathBuf::from("/nonexistent/cuda-oxide-post-ir-hook")],
            &ll_path,
            "",
            &PipelineTrace::default(),
        )
        .expect_err("missing hook must fail");
        assert!(matches!(err, PipelineError::PostIr(_)), "got {err:?}");
        let _ = fs::remove_dir_all(&dir);
    }
}
