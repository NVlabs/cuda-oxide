/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

mod abi_history;
mod extract;
mod generate;
mod model;
mod probe;
mod ptx;
mod render;
mod resolve;
mod util;

use anyhow::{Context, Result, bail};
use extract::ExtractOptions;
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
enum ProbeInvocation {
    Evidence {
        intrinsic_id: String,
        llc: Option<PathBuf>,
        skip_terminal: bool,
    },
    Candidate(probe::CandidateProbeOptions),
}

fn main() {
    if let Err(error) = try_main() {
        eprintln!("error: {error:#}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<()> {
    let mut arguments: Vec<String> = env::args().skip(1).collect();
    let repo_root = take_option(&mut arguments, "--repo-root")?
        .map(PathBuf::from)
        .unwrap_or(env::current_dir().context("get current directory")?);
    let Some(command) = arguments.first().cloned() else {
        print_usage();
        bail!("missing command");
    };
    arguments.remove(0);
    match command.as_str() {
        "extract" => {
            let options = ExtractOptions {
                intrinsics_json: take_option(&mut arguments, "--intrinsics-json")?
                    .map(PathBuf::from),
                nvptx_json: take_option(&mut arguments, "--nvptx-json")?.map(PathBuf::from),
                llvm_src: take_option(&mut arguments, "--llvm-src")?.map(PathBuf::from),
                llvm_tblgen: take_option(&mut arguments, "--llvm-tblgen")?.map(PathBuf::from),
            };
            reject_extra(arguments)?;
            extract::run(&repo_root, options)
        }
        "generate" => {
            reject_extra(arguments)?;
            generate::run(&repo_root, false)
        }
        "check" => {
            reject_extra(arguments)?;
            generate::run(&repo_root, true)
        }
        "probe" => match parse_probe_invocation(arguments)? {
            ProbeInvocation::Evidence {
                intrinsic_id,
                llc,
                skip_terminal,
            } => probe::run(&repo_root, &intrinsic_id, llc, skip_terminal),
            ProbeInvocation::Candidate(options) => probe::run_candidate(&repo_root, options),
        },
        "check-abi-history" => {
            let base_ref = take_option(&mut arguments, "--base-ref")?
                .context("check-abi-history requires --base-ref REF")?;
            reject_extra(arguments)?;
            abi_history::run(&repo_root, &base_ref)
        }
        "-h" | "--help" | "help" => {
            print_usage();
            Ok(())
        }
        _ => {
            print_usage();
            bail!("unknown command {command:?}")
        }
    }
}

fn parse_probe_invocation(mut arguments: Vec<String>) -> Result<ProbeInvocation> {
    if !take_flag(&mut arguments, "--candidate") {
        let intrinsic_id =
            take_option(&mut arguments, "--intrinsic")?.unwrap_or_else(|| "thread_idx_x".into());
        let llc = take_option(&mut arguments, "--llc")?.map(PathBuf::from);
        let skip_terminal = take_flag(&mut arguments, "--skip-terminal");
        reject_extra(arguments)?;
        return Ok(ProbeInvocation::Evidence {
            intrinsic_id,
            llc,
            skip_terminal,
        });
    }

    let intrinsic_id = take_option(&mut arguments, "--intrinsic")?
        .context("candidate probe requires --intrinsic ID")?;
    let llc = take_option(&mut arguments, "--llc")?
        .map(PathBuf::from)
        .context("candidate probe requires --llc FILE")?;
    let gpu_target = take_option(&mut arguments, "--gpu-target")?
        .context("candidate probe requires --gpu-target TARGET")?;
    let ptx_feature = take_option(&mut arguments, "--ptx-feature")?
        .context("candidate probe requires --ptx-feature FEATURE")?;
    let ptxas = take_option(&mut arguments, "--ptxas")?.map(PathBuf::from);
    let skip_terminal = take_flag(&mut arguments, "--skip-terminal");
    if skip_terminal == ptxas.is_some() {
        bail!("candidate probe requires either --ptxas FILE or explicit --skip-terminal");
    }
    reject_extra(arguments)?;
    Ok(ProbeInvocation::Candidate(probe::CandidateProbeOptions {
        intrinsic_id,
        llc,
        gpu_target,
        ptx_feature,
        ptxas,
        skip_terminal,
    }))
}

fn take_option(arguments: &mut Vec<String>, name: &str) -> Result<Option<String>> {
    let Some(index) = arguments.iter().position(|argument| argument == name) else {
        return Ok(None);
    };
    if index + 1 >= arguments.len() {
        bail!("{name} requires a value");
    }
    arguments.remove(index);
    Ok(Some(arguments.remove(index)))
}

fn take_flag(arguments: &mut Vec<String>, name: &str) -> bool {
    if let Some(index) = arguments.iter().position(|argument| argument == name) {
        arguments.remove(index);
        true
    } else {
        false
    }
}

fn reject_extra(arguments: Vec<String>) -> Result<()> {
    if !arguments.is_empty() {
        bail!("unexpected arguments: {}", arguments.join(" "));
    }
    Ok(())
}

fn print_usage() {
    eprintln!(
        "cuda-intrinsics-gen\n\n\
         Usage:\n  \
         cuda-intrinsics-gen extract --intrinsics-json FILE --nvptx-json FILE [--repo-root DIR]\n  \
         cuda-intrinsics-gen extract --llvm-src DIR --llvm-tblgen FILE [--repo-root DIR]\n  \
         cuda-intrinsics-gen generate [--repo-root DIR]\n  \
         cuda-intrinsics-gen check [--repo-root DIR]\n  \
         cuda-intrinsics-gen check-abi-history --base-ref REF [--repo-root DIR]\n  \
         cuda-intrinsics-gen probe [--intrinsic ID] [--llc FILE] [--skip-terminal] [--repo-root DIR]\n  \
         cuda-intrinsics-gen probe --candidate --intrinsic ID --llc FILE --gpu-target TARGET --ptx-feature FEATURE (--ptxas FILE | --skip-terminal) [--repo-root DIR]"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).into()).collect()
    }

    #[test]
    fn candidate_probe_requires_every_explicit_input() {
        let base = [
            "--candidate",
            "--intrinsic",
            "thread_idx_x",
            "--llc",
            "/tool/llc",
            "--gpu-target",
            "sm_80",
            "--ptx-feature",
            "+ptx70",
        ];
        let error = parse_probe_invocation(strings(&base)).unwrap_err();
        assert!(error.to_string().contains("--ptxas FILE"));

        for missing in ["--intrinsic", "--llc", "--gpu-target", "--ptx-feature"] {
            let mut arguments = base.to_vec();
            let index = arguments
                .iter()
                .position(|value| *value == missing)
                .unwrap();
            arguments.drain(index..=index + 1);
            arguments.extend(["--skip-terminal"]);
            let error = parse_probe_invocation(strings(&arguments)).unwrap_err();
            assert!(error.to_string().contains(missing), "{error:#}");
        }
    }

    #[test]
    fn candidate_terminal_mode_is_explicit_and_has_no_fallback() {
        let arguments = strings(&[
            "--candidate",
            "--intrinsic",
            "thread_idx_x",
            "--llc",
            "/tool/llc",
            "--gpu-target",
            "sm_80",
            "--ptx-feature",
            "+ptx70",
            "--ptxas",
            "/tool/ptxas",
        ]);
        let ProbeInvocation::Candidate(options) = parse_probe_invocation(arguments).unwrap() else {
            panic!("expected candidate probe")
        };
        assert_eq!(options.ptxas, Some(PathBuf::from("/tool/ptxas")));
        assert!(!options.skip_terminal);

        let error = parse_probe_invocation(strings(&[
            "--candidate",
            "--intrinsic",
            "thread_idx_x",
            "--llc",
            "/tool/llc",
            "--gpu-target",
            "sm_80",
            "--ptx-feature",
            "+ptx70",
            "--ptxas",
            "/tool/ptxas",
            "--skip-terminal",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("either --ptxas"));
    }

    #[test]
    fn normal_probe_does_not_accept_candidate_target_inputs() {
        let error = parse_probe_invocation(strings(&[
            "--intrinsic",
            "thread_idx_x",
            "--gpu-target",
            "sm_80",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("unexpected arguments"));
    }
}
