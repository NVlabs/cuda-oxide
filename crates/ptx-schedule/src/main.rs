/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use ptx_schedule::{InjectionOptions, analyze_ptx, perturb_ptx};
use std::env;
use std::fs;
use std::path::PathBuf;

fn usage() -> ! {
    eprintln!(
        "usage: ptx-schedule INPUT.ptx [--list-sites] [--seed N] [--intensity F] [--max-sleep-ns N] [--focus TEXT] [-o|--output OUTPUT] [--decisions-json FILE]"
    );
    std::process::exit(2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = PathBuf::from(args.next().unwrap_or_else(|| usage()));
    let mut options = InjectionOptions::default();
    let mut list_sites = false;
    let mut output = None;
    let mut decisions_json = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--list-sites" => list_sites = true,
            "--seed" => options.seed = args.next().unwrap_or_else(|| usage()).parse()?,
            "--intensity" => options.intensity = args.next().unwrap_or_else(|| usage()).parse()?,
            "--max-sleep-ns" => {
                options.max_sleep_ns = args.next().unwrap_or_else(|| usage()).parse()?
            }
            "--focus" => options.focus = Some(args.next().unwrap_or_else(|| usage())),
            "-o" | "--output" => {
                output = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            "--decisions-json" => {
                decisions_json = Some(PathBuf::from(args.next().unwrap_or_else(|| usage())))
            }
            _ => usage(),
        }
    }

    let source = fs::read_to_string(&input)?;
    if list_sites {
        let analysis = analyze_ptx(&source)?;
        println!("{}", serde_json::to_string_pretty(analysis.sites())?);
        return Ok(());
    }
    let output = output.unwrap_or_else(|| usage());
    let rewrite = perturb_ptx(&source, &options)?;
    fs::write(output, &rewrite.ptx)?;
    if let Some(path) = decisions_json {
        fs::write(path, serde_json::to_string_pretty(&rewrite.report)?)?;
    }
    println!(
        "ptx-schedule: seed={} intensity={} sites={} injected={} ns_per_visit={}",
        rewrite.report.seed,
        rewrite.report.intensity,
        rewrite.report.sites_total,
        rewrite.report.sites_injected,
        rewrite.report.injected_ns_per_visit
    );
    Ok(())
}
