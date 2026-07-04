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
        "probe" => {
            let intrinsic = take_option(&mut arguments, "--intrinsic")?
                .unwrap_or_else(|| "thread_idx_x".into());
            let llc = take_option(&mut arguments, "--llc")?.map(PathBuf::from);
            let skip_terminal = take_flag(&mut arguments, "--skip-terminal");
            reject_extra(arguments)?;
            probe::run(&repo_root, &intrinsic, llc, skip_terminal)
        }
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
         cuda-intrinsics-gen probe [--intrinsic ID] [--llc FILE] [--skip-terminal] [--repo-root DIR]"
    );
}
