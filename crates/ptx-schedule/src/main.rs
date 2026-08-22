/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use ptx_schedule::{InjectionOptions, analyze_ptx, perturb_ptx};
use std::env;
use std::fs;
use std::path::PathBuf;

/// The two invocation forms, one per line.
///
/// A single bracketed list cannot state this CLI's shape: `--output` is
/// required to perturb and meaningless to list, and bracketing it read as
/// optional in both. One line per form is what `cuda-intrinsics-gen`'s usage
/// text does for the same reason.
const USAGE: &str = "usage:
  ptx-schedule INPUT.ptx --list-sites
        print the schedule-sensitive sites as JSON; the options below are ignored
  ptx-schedule INPUT.ptx -o|--output OUTPUT [--seed N] [--intensity F]
        [--max-sleep-ns N] [--focus TEXT] [--decisions-json FILE]
        write a perturbed copy of INPUT.ptx to OUTPUT";

fn usage() -> ! {
    eprintln!("{USAGE}");
    std::process::exit(2);
}

/// Exit with the usage text, having first said what was wrong with the command
/// line. Without the reason, a missing `--output` and an unknown flag print the
/// same wall of text and leave the reader to spot the difference.
fn usage_because(reason: &str) -> ! {
    eprintln!("error: {reason}");
    usage();
}

/// The value that must follow `flag`, or the usage text saying it is missing.
fn next_value(args: &mut impl Iterator<Item = String>, flag: &str) -> String {
    args.next()
        .unwrap_or_else(|| usage_because(&format!("{flag} needs a value")))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let input = PathBuf::from(
        args.next()
            .unwrap_or_else(|| usage_because("an input PTX file is required")),
    );
    let mut options = InjectionOptions::default();
    let mut list_sites = false;
    let mut output = None;
    let mut decisions_json = None;

    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--list-sites" => list_sites = true,
            "--seed" => options.seed = next_value(&mut args, "--seed").parse()?,
            "--intensity" => options.intensity = next_value(&mut args, "--intensity").parse()?,
            "--max-sleep-ns" => {
                options.max_sleep_ns = next_value(&mut args, "--max-sleep-ns").parse()?
            }
            "--focus" => options.focus = Some(next_value(&mut args, "--focus")),
            "-o" | "--output" => output = Some(PathBuf::from(next_value(&mut args, "--output"))),
            "--decisions-json" => {
                decisions_json = Some(PathBuf::from(next_value(&mut args, "--decisions-json")))
            }
            other => usage_because(&format!("unrecognized argument {other:?}")),
        }
    }

    let source = fs::read_to_string(&input)?;
    if list_sites {
        let analysis = analyze_ptx(&source)?;
        println!("{}", serde_json::to_string_pretty(analysis.sites())?);
        return Ok(());
    }
    let output = output
        .unwrap_or_else(|| usage_because("-o|--output is required unless --list-sites is given"));
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

#[cfg(test)]
mod tests {
    use super::USAGE;

    /// The bidirectional parity #1109 established, now asserted rather than
    /// checked by hand: the usage text names exactly the long flags the parser
    /// accepts.
    #[test]
    fn the_usage_text_names_every_flag_the_parser_accepts() {
        let source = include_str!("main.rs");
        // Match arms, minus the `other =>` catch-all.
        // An arm can name several spellings -- `"-o" | "--output" =>` -- so take
        // every quoted flag on the line, not just the first.
        let mut parsed: Vec<&str> = source
            .lines()
            .map(str::trim_start)
            .filter(|line| line.starts_with('"') && line.contains("=>"))
            .flat_map(|line| {
                line.split('"')
                    .skip(1)
                    .step_by(2)
                    .filter(|flag| flag.starts_with('-'))
            })
            .collect();
        parsed.sort_unstable();
        parsed.dedup();
        assert!(parsed.len() >= 7, "{parsed:?}");
        for flag in &parsed {
            assert!(
                USAGE.contains(flag),
                "{flag} is accepted but not in the usage text"
            );
        }
        for word in USAGE.split_whitespace() {
            let word = word.trim_start_matches('[').trim_end_matches(']');
            for flag in word.split('|') {
                if flag.starts_with('-') {
                    assert!(
                        parsed.contains(&flag),
                        "{flag} is in the usage text but not parsed"
                    );
                }
            }
        }
    }

    /// `--output` is required to perturb and ignored when listing, so it must
    /// not be shown bracketed the way the genuinely optional flags are. That
    /// bracketing is what #1109's review flagged.
    #[test]
    fn output_is_shown_as_required_on_the_form_that_requires_it() {
        let write_form = USAGE
            .lines()
            .find(|line| line.contains("--output"))
            .expect("a form naming --output");
        assert!(
            !write_form.contains("[-o|--output"),
            "--output is bracketed as optional: {write_form}"
        );
        assert!(write_form.contains("-o|--output OUTPUT"), "{write_form}");

        let list_form = USAGE
            .lines()
            .find(|line| line.contains("--list-sites"))
            .expect("a form naming --list-sites");
        assert!(
            !list_form.contains("--output"),
            "the listing form should not name --output: {list_form}"
        );
    }
}
