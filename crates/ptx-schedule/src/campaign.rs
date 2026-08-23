/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end schedule campaigns for existing cuda-oxide examples.
//!
//! A campaign builds an example once, mutates the generated PTX in memory,
//! patches a copy of the executable's embedded `.oxart` section for each
//! mutation, and runs that copy without changing the production loader.

use crate::{InjectionOptions, RewriteReport};
use oxide_artifacts::{
    ArtifactBundleSpec, ArtifactEntrySpec, ArtifactPayloadKind, ArtifactPayloadSpec,
    build_artifact_blob, read_artifact_bundles_from_object_bytes,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CampaignError {
    #[error("invalid seed range '{0}', expected START..END with END > START")]
    InvalidSeedRange(String),
    #[error("confirmation runs must be at least 1, got {0}")]
    InvalidConfirmRuns(u32),
    #[error("example '{0}' was not found at {1}")]
    ExampleNotFound(String, PathBuf),
    #[error("campaign I/O failed: {0}")]
    Io(#[from] io::Error),
    #[error("could not parse Cargo metadata: {0}")]
    Metadata(String),
    #[error("cuda-oxide build failed with {0}")]
    BuildFailed(ExitStatus),
    #[error("generated PTX was not found at {0}")]
    MissingPtx(PathBuf),
    #[error("example executable was not found at {0}")]
    MissingExecutable(PathBuf),
    #[error("PTX schedule rewrite failed: {0}")]
    Schedule(#[from] crate::ScheduleError),
    #[error("could not patch embedded PTX: {0}")]
    ArtifactPatch(String),
    #[error("could not serialize campaign report: {0}")]
    Json(#[from] serde_json::Error),
}

#[derive(Clone, Debug)]
pub struct CampaignOptions {
    pub workspace_root: PathBuf,
    pub oxide_binary: PathBuf,
    pub example: String,
    /// Half-open seed interval: `0..100` runs seeds 0 through 99.
    pub seed_start: u64,
    pub seed_end: u64,
    pub intensity: f64,
    pub max_sleep_ns: u32,
    pub timeout: Duration,
    pub arch: Option<String>,
    pub focus: Option<String>,
    pub output_dir: Option<PathBuf>,
    pub keep_going: bool,
    /// Total executions for a finding, including the initial execution.
    pub confirm_runs: u32,
    /// Treat a changed stdout stream as a finding when the example has no
    /// explicit failure marker.
    pub compare_output: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum RunKind {
    Pass,
    Skipped,
    Hang,
    Crash,
    Mismatch,
    OutputChanged,
    GpuWedged,
    /// The harness failed to perturb or patch this seed, so the variant
    /// never ran. Not a schedule finding.
    HarnessError,
}

impl RunKind {
    fn finding_label(&self) -> Option<&'static str> {
        match self {
            Self::Mismatch => Some("SCHEDULE-SENSITIVE CORRECTNESS FAILURE"),
            Self::OutputChanged => Some("SCHEDULE-SENSITIVE OUTPUT CHANGE"),
            Self::Hang => Some("TIMEOUT CANDIDATE"),
            Self::Crash => Some("CRASH CANDIDATE"),
            Self::GpuWedged => Some("GPU WEDGE CANDIDATE"),
            Self::Pass | Self::Skipped | Self::HarnessError => None,
        }
    }

    fn is_finding(&self) -> bool {
        self.finding_label().is_some()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RunResult {
    pub kind: RunKind,
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SeedResult {
    pub seed: u64,
    pub artifact_dir: PathBuf,
    pub report: RewriteReport,
    pub run: RunResult,
    pub confirmation: Option<ConfirmationSummary>,
    pub replay: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConfirmationSummary {
    pub attempts: u32,
    pub findings: u32,
    pub confirmed: bool,
    pub outcomes: Vec<RunKind>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StaticSiteReport {
    pub sites_total: usize,
    pub sites_by_kind: BTreeMap<String, usize>,
    pub sites: Vec<crate::ScheduleSite>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CampaignSettings {
    pub seed_start: u64,
    pub seed_end: u64,
    pub intensity: f64,
    pub max_sleep_ns: u32,
    pub timeout_secs: u64,
    pub arch: Option<String>,
    pub focus: Option<String>,
    pub confirm_runs: u32,
    pub compare_output: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CampaignSummary {
    pub example: String,
    pub ptx: PathBuf,
    pub executable: PathBuf,
    pub settings: CampaignSettings,
    pub static_sites: StaticSiteReport,
    pub baseline: RunResult,
    pub seeds: Vec<SeedResult>,
}

impl CampaignSummary {
    pub fn finding_count(&self) -> usize {
        self.seeds
            .iter()
            .filter(|seed| seed.run.kind.is_finding())
            .count()
    }
}

pub fn parse_seed_range(value: &str) -> Result<(u64, u64), CampaignError> {
    let Some((start, end)) = value.split_once("..") else {
        return Err(CampaignError::InvalidSeedRange(value.to_string()));
    };
    let start = start
        .parse::<u64>()
        .map_err(|_| CampaignError::InvalidSeedRange(value.to_string()))?;
    let end = end
        .parse::<u64>()
        .map_err(|_| CampaignError::InvalidSeedRange(value.to_string()))?;
    if start >= end {
        return Err(CampaignError::InvalidSeedRange(value.to_string()));
    }
    Ok((start, end))
}

pub fn run_campaign(options: &CampaignOptions) -> Result<CampaignSummary, CampaignError> {
    if options.seed_start >= options.seed_end {
        return Err(CampaignError::InvalidSeedRange(format!(
            "{}..{}",
            options.seed_start, options.seed_end
        )));
    }
    if options.confirm_runs == 0 {
        return Err(CampaignError::InvalidConfirmRuns(options.confirm_runs));
    }

    let example_dir = options
        .workspace_root
        .join("crates/rustc-codegen-cuda/examples")
        .join(&options.example);
    if !example_dir.join("Cargo.toml").is_file() {
        return Err(CampaignError::ExampleNotFound(
            options.example.clone(),
            example_dir,
        ));
    }

    let build_status = build_example(options)?;
    if !build_status.success() {
        return Err(CampaignError::BuildFailed(build_status));
    }

    let stem = options.example.replace('-', "_");
    let ptx_path = example_dir.join(format!("{stem}.ptx"));
    if !ptx_path.is_file() {
        return Err(CampaignError::MissingPtx(ptx_path));
    }
    let executable = find_executable(&example_dir, &options.example)?;
    let pristine = fs::read_to_string(&ptx_path)?;
    let analysis = crate::analyze_ptx(&pristine)?;
    let static_sites = static_site_report(&analysis);
    let output_dir = options.output_dir.clone().unwrap_or_else(|| {
        options
            .workspace_root
            .join("crates/fuzzer/artifacts/schedule")
            .join(&options.example)
    });
    fs::create_dir_all(&output_dir)?;
    fs::write(
        output_dir.join("sites.json"),
        serde_json::to_vec_pretty(&static_sites)?,
    )?;
    println!(
        "schedule-fuzz: static sites={} kinds={}",
        static_sites.sites_total,
        format_site_kinds(&static_sites.sites_by_kind)
    );

    // Captured once, so the baseline, every variant and every replay script
    // describe the same environment.
    let environment = RunEnvironment::from_process();
    println!(
        "schedule-fuzz: environment captured={} unset={} required-external={}",
        environment.replay.captured.len(),
        environment.replay.unset.len(),
        environment.replay.required_external.len()
    );

    println!("schedule-fuzz: baseline {}", executable.display());
    let baseline = run_binary(&executable, &example_dir, options.timeout, &environment);
    println!("schedule-fuzz: baseline {:?}", baseline.kind);
    if !matches!(baseline.kind, RunKind::Pass) {
        let summary = CampaignSummary {
            example: options.example.clone(),
            ptx: ptx_path.clone(),
            executable: executable.clone(),
            settings: campaign_settings(options),
            static_sites,
            baseline,
            seeds: Vec::new(),
        };
        println!(
            "schedule-fuzz: BASELINE FAILURE: {:?}; no schedule variants were run",
            summary.baseline.kind
        );
        fs::write(
            output_dir.join("summary.json"),
            serde_json::to_vec_pretty(&summary)?,
        )?;
        return Ok(summary);
    }

    let context = SeedContext {
        options,
        pristine: &pristine,
        executable: &executable,
        example_dir: &example_dir,
        baseline: &baseline,
        environment: &environment,
    };
    let mut seeds = Vec::new();
    for seed in options.seed_start..options.seed_end {
        let artifact_dir = output_dir.join(format!("seed-{seed}"));
        // A perturbation or patching failure is scoped to this seed: record
        // it and keep going so the campaign still covers the remaining seeds
        // and still writes summary.json.
        let result = run_seed(&context, seed, &artifact_dir).unwrap_or_else(|error| {
            harness_error_result(seed, options.intensity, artifact_dir, &error)
        });
        print_seed_result(&result);
        let stop = matches!(result.run.kind, RunKind::GpuWedged) && !options.keep_going;
        seeds.push(result);
        if stop {
            break;
        }
    }

    let summary = CampaignSummary {
        example: options.example.clone(),
        ptx: ptx_path,
        executable,
        settings: campaign_settings(options),
        static_sites,
        baseline,
        seeds,
    };
    fs::write(
        output_dir.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    print_campaign_result(&summary, &output_dir);
    Ok(summary)
}

/// Everything a seed needs that does not change between seeds.
///
/// These were threaded through `run_seed` one parameter at a time, which grew
/// past what a signature carries legibly once the environment joined them. All
/// six are constant for the whole campaign, so they belong together.
struct SeedContext<'a> {
    options: &'a CampaignOptions,
    pristine: &'a str,
    executable: &'a Path,
    example_dir: &'a Path,
    baseline: &'a RunResult,
    environment: &'a RunEnvironment,
}

fn run_seed(
    context: &SeedContext<'_>,
    seed: u64,
    artifact_dir: &Path,
) -> Result<SeedResult, CampaignError> {
    let SeedContext {
        options,
        pristine,
        executable,
        example_dir,
        baseline,
        environment,
    } = context;
    let rewrite = crate::perturb_ptx(
        pristine,
        &InjectionOptions {
            seed,
            intensity: options.intensity,
            max_sleep_ns: options.max_sleep_ns,
            focus: options.focus.clone(),
        },
    )?;
    create_private_dir(artifact_dir)?;
    let mutated_ptx = artifact_dir.join("module.ptx");
    let report_path = artifact_dir.join("report.json");
    let stdout_path = artifact_dir.join("stdout.log");
    let stderr_path = artifact_dir.join("stderr.log");
    let replay_path = artifact_dir.join("replay.sh");
    fs::write(&mutated_ptx, &rewrite.ptx)?;
    fs::write(&report_path, serde_json::to_vec_pretty(&rewrite.report)?)?;

    let variant_executable = artifact_dir.join(
        executable
            .file_name()
            .ok_or_else(|| CampaignError::ArtifactPatch("executable has no file name".into()))?,
    );
    patch_executable(
        executable,
        &variant_executable,
        pristine.as_bytes(),
        rewrite.ptx.as_bytes(),
    )?;
    write_replay_script(
        &replay_path,
        example_dir,
        &variant_executable,
        &environment.replay,
    )?;

    let mut run = run_variant(
        &variant_executable,
        executable,
        example_dir,
        options.timeout,
        environment,
    );
    classify_output_change(baseline, &mut run, options.compare_output);

    fs::write(&stdout_path, &run.stdout)?;
    fs::write(&stderr_path, &run.stderr)?;
    let confirmation = if run.kind.is_finding() && options.confirm_runs > 1 {
        let mut outcomes = vec![run.kind.clone()];
        let mut findings = 1;
        for attempt in 1..options.confirm_runs {
            let mut confirmed_run = run_variant(
                &variant_executable,
                executable,
                example_dir,
                options.timeout,
                environment,
            );
            classify_output_change(baseline, &mut confirmed_run, options.compare_output);
            if confirmed_run.kind.is_finding() {
                findings += 1;
            }
            fs::write(
                artifact_dir.join(format!("confirm-{attempt}-stdout.log")),
                &confirmed_run.stdout,
            )?;
            fs::write(
                artifact_dir.join(format!("confirm-{attempt}-stderr.log")),
                &confirmed_run.stderr,
            )?;
            outcomes.push(confirmed_run.kind);
        }
        Some(ConfirmationSummary {
            attempts: options.confirm_runs,
            findings,
            confirmed: findings == options.confirm_runs,
            outcomes,
        })
    } else {
        None
    };
    Ok(SeedResult {
        seed,
        artifact_dir: artifact_dir.to_path_buf(),
        report: rewrite.report,
        run,
        confirmation,
        replay: replay_path,
    })
}

/// The seed's variant never ran. The error text lands in the run's stderr,
/// a zeroed rewrite report keeps summary.json uniform, and the replay path
/// names where the script would have been written.
fn harness_error_result(
    seed: u64,
    intensity: f64,
    artifact_dir: PathBuf,
    error: &CampaignError,
) -> SeedResult {
    let replay = artifact_dir.join("replay.sh");
    SeedResult {
        seed,
        artifact_dir,
        report: RewriteReport {
            seed,
            intensity,
            sites_total: 0,
            sites_injected: 0,
            injected_ns_per_visit: 0,
            decisions: Vec::new(),
        },
        run: RunResult {
            kind: RunKind::HarnessError,
            exit_code: None,
            timed_out: false,
            stdout: String::new(),
            stderr: error.to_string(),
        },
        confirmation: None,
        replay,
    }
}

fn print_seed_result(result: &SeedResult) {
    if matches!(result.run.kind, RunKind::HarnessError) {
        println!(
            "schedule-fuzz: seed={} HARNESS ERROR: {}",
            result.seed,
            result.run.stderr.trim()
        );
    } else if let Some(label) = result.run.kind.finding_label() {
        println!("schedule-fuzz: FINDING seed={}: {}", result.seed, label);
        if let Some(confirmation) = &result.confirmation {
            println!(
                "  confirmation: {}/{} reproductions{}",
                confirmation.findings,
                confirmation.attempts,
                if confirmation.confirmed {
                    " (CONFIRMED)"
                } else {
                    ""
                }
            );
        } else {
            println!("  confirmation: not requested");
        }
        println!("  replay: {}", result.artifact_dir.display());
        println!(
            "  logs:   {}/stdout.log and stderr.log",
            result.artifact_dir.display()
        );
    } else {
        println!(
            "schedule-fuzz: seed={} PASS sites={}/{}",
            result.seed, result.report.sites_injected, result.report.sites_total
        );
    }
}

fn print_campaign_result(summary: &CampaignSummary, output_dir: &Path) {
    let findings: Vec<&SeedResult> = summary
        .seeds
        .iter()
        .filter(|result| result.run.kind.finding_label().is_some())
        .collect();

    let harness_errors = summary
        .seeds
        .iter()
        .filter(|result| matches!(result.run.kind, RunKind::HarnessError))
        .count();

    println!();
    println!("=== schedule-fuzz result ===");
    println!("example: {}", summary.example);
    println!("baseline: PASS");
    println!("variants: {}", summary.seeds.len());
    if harness_errors > 0 {
        println!("harness errors: {harness_errors} (seeds not run, not schedule findings)");
    }
    if findings.is_empty() {
        println!("RESULT: no schedule-sensitive failures found");
    } else {
        println!("RESULT: FOUND {} CANDIDATE(S)", findings.len());
        for result in findings {
            let confirmation = result
                .confirmation
                .as_ref()
                .map(|confirmation| {
                    format!(
                        ", reproduced {}/{}{}",
                        confirmation.findings,
                        confirmation.attempts,
                        if confirmation.confirmed {
                            " CONFIRMED"
                        } else {
                            ""
                        }
                    )
                })
                .unwrap_or_default();
            println!(
                "  seed {}: {}{} [{}]",
                result.seed,
                result.run.kind.finding_label().unwrap_or("failure"),
                confirmation,
                result.artifact_dir.display()
            );
        }
    }
    println!("report: {}/summary.json", output_dir.display());
    println!("sites:  {}/sites.json", output_dir.display());
}

fn campaign_settings(options: &CampaignOptions) -> CampaignSettings {
    CampaignSettings {
        seed_start: options.seed_start,
        seed_end: options.seed_end,
        intensity: options.intensity,
        max_sleep_ns: options.max_sleep_ns,
        timeout_secs: options.timeout.as_secs(),
        arch: options.arch.clone(),
        focus: options.focus.clone(),
        confirm_runs: options.confirm_runs,
        compare_output: options.compare_output,
    }
}

fn static_site_report(analysis: &crate::ScheduleAnalysis) -> StaticSiteReport {
    let mut sites_by_kind = BTreeMap::new();
    for site in analysis.sites() {
        *sites_by_kind.entry(format!("{:?}", site.kind)).or_insert(0) += 1;
    }
    StaticSiteReport {
        sites_total: analysis.sites().len(),
        sites_by_kind,
        sites: analysis.sites().to_vec(),
    }
}

fn format_site_kinds(sites_by_kind: &BTreeMap<String, usize>) -> String {
    sites_by_kind
        .iter()
        .map(|(kind, count)| format!("{kind}={count}"))
        .collect::<Vec<_>>()
        .join(",")
}

/// The exact environment a variant runs under, and that its replay script
/// reproduces.
///
/// A variant used to inherit the campaign's whole environment -- `run_binary`
/// built a `Command` and never called `env_clear` -- while the replay script
/// re-exported a prefix-matched subset. Two different environments, so the
/// script could not reproduce the run, and it also persisted anything whose
/// name happened to start with a captured prefix: `CUDA_API_TOKEN` and
/// `CUDA_OXIDE_LICENSE_KEY` were both written into a file on disk.
///
/// One value now decides both. Every name is explicit, and each falls in
/// exactly one of three fields:
///
/// * `captured` -- run-affecting knobs with a recordable value. Set on the
///   child, and written to the script as `export`.
/// * `unset` -- captured names the campaign did *not* have. Written as `unset`,
///   so an ambient value in the replay shell cannot silently change the run.
/// * `required_external` -- names the child needs but that are not recorded,
///   either because they are machine-specific (`PATH`, `LD_LIBRARY_PATH`) or
///   because the value is not valid UTF-8 and cannot be written into a POSIX
///   script. The script guards each with `${NAME:?}`, so a replay in the wrong
///   environment *refuses* instead of reproducing something else.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayEnvironment {
    captured: Vec<(String, String)>,
    unset: Vec<String>,
    required_external: Vec<String>,
}

/// Names whose values change what a run does, and are safe to write down.
///
/// Explicit names, never prefixes. A prefix cannot distinguish
/// `CUDA_VISIBLE_DEVICES` from `CUDA_API_TOKEN`, and this list is written to a
/// file, so the only defensible rule is one an author states deliberately.
/// Every entry is read at run time by something in this tree:
///
/// * `CUDA_VISIBLE_DEVICES`, `CUDA_LAUNCH_BLOCKING` -- driver knobs that pick
///   the device and serialize launches.
/// * `CUDA_OXIDE_TARGET` -- read by `device_ffi_test`, `mathdx_ffi_test` and
///   `small_type_ffi_test`.
/// * `GEMM_SOL_PHASE` -- read by `gemm_sol`; `GEMM_SOL_MODE` and
///   `GEMM_SOL_VARIANT` by `gemm_sol_final`, where they select the mode and the
///   kernel variant.
/// * `MATHDX_ROOT` -- read by `mathdx_ffi_test`.
const CAPTURED_ENV: [&str; 7] = [
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_OXIDE_TARGET",
    "CUDA_VISIBLE_DEVICES",
    "GEMM_SOL_MODE",
    "GEMM_SOL_PHASE",
    "GEMM_SOL_VARIANT",
    "MATHDX_ROOT",
];

/// Names passed through to the child but never recorded.
///
/// The child is a dynamically linked binary that has to find libcuda, and
/// `mathdx_ffi_test` reads `HOME`. These are machine-specific and can carry
/// paths a reader has no business seeing, so the script requires them from the
/// replaying environment rather than pinning this machine's values.
const REQUIRED_EXTERNAL_ENV: [&str; 3] = ["HOME", "LD_LIBRARY_PATH", "PATH"];

/// The captured environment plus the process environment it came from.
///
/// The two are always used together -- one says what to set, the other holds
/// the values for names that are required but not recorded -- so they travel as
/// one value instead of as a pair threaded through every call.
pub struct RunEnvironment {
    replay: ReplayEnvironment,
    source: BTreeMap<OsString, OsString>,
}

impl RunEnvironment {
    /// Capture from this process.
    fn from_process() -> Self {
        let source: BTreeMap<OsString, OsString> = std::env::vars_os().collect();
        Self {
            replay: ReplayEnvironment::capture(source.clone()),
            source,
        }
    }

    /// Give `command` exactly this environment and nothing else.
    ///
    /// `env_clear` first: without it the child inherits every variable the
    /// campaign happened to hold, which is the half of the reproducibility gap
    /// no replay script can fix.
    fn apply_to(&self, command: &mut Command) {
        command.env_clear();
        for (name, value) in &self.replay.captured {
            command.env(name, value);
        }
        for name in &self.replay.required_external {
            if let Some(value) = self.source.get(OsStr::new(name.as_str())) {
                command.env(name, value);
            }
        }
    }
}

impl ReplayEnvironment {
    /// Classify one process environment into the three fields.
    ///
    /// Takes `OsString` pairs so a value that is not valid UTF-8 is
    /// representable: the child still receives it, but it moves to
    /// `required_external` because no POSIX script can carry it.
    ///
    /// Pure, so every rule below is testable without mutating the process
    /// environment -- which `set_var` is `unsafe` for in edition 2024, and which
    /// races across test threads.
    pub fn capture<I>(vars: I) -> Self
    where
        I: IntoIterator<Item = (OsString, OsString)>,
    {
        let present: BTreeMap<OsString, OsString> = vars.into_iter().collect();
        let mut captured = Vec::new();
        let mut unset = Vec::new();
        let mut required_external = Vec::new();

        for name in CAPTURED_ENV {
            match present.get(OsStr::new(name)) {
                None => unset.push(name.to_owned()),
                Some(value) => match value.to_str() {
                    Some(value) => captured.push((name.to_owned(), value.to_owned())),
                    // Recordable only as a requirement, not as a value.
                    None => required_external.push(name.to_owned()),
                },
            }
        }
        for name in REQUIRED_EXTERNAL_ENV {
            if present.contains_key(OsStr::new(name)) {
                required_external.push(name.to_owned());
            }
        }

        captured.sort();
        unset.sort();
        required_external.sort();
        Self {
            captured,
            unset,
            required_external,
        }
    }

    /// The environment prelude of a replay script.
    fn script_prelude(&self) -> String {
        let mut prelude = String::new();
        for name in &self.required_external {
            prelude.push_str(&format!(
                ": \"${{{name}:?required by this run and not recorded here}}\"\n"
            ));
        }
        for name in &self.unset {
            prelude.push_str(&format!("unset {name}\n"));
        }
        for (name, value) in &self.captured {
            prelude.push_str(&format!("export {name}={}\n", shell_quote_value(value)));
        }
        prelude
    }
}

fn write_replay_script(
    path: &Path,
    cwd: &Path,
    executable: &Path,
    environment: &ReplayEnvironment,
) -> Result<(), CampaignError> {
    let script = format!(
        "#!/bin/sh\nset -eu\n{}cd {}\nexec {}\n",
        environment.script_prelude(),
        shell_quote(cwd),
        shell_quote(executable)
    );
    write_private_file(path, script.as_bytes())
}

/// Create `path` with owner-only permissions from the start.
///
/// `fs::write` followed by `set_permissions` created the file at the process
/// umask -- world-readable on a default configuration -- and only narrowed it
/// afterwards, so a file holding captured values was readable by other users
/// for a window. `mode` on `OpenOptions` applies at creation, which closes it;
/// `create_new` after an explicit remove keeps a stale file from surviving with
/// its old mode, since `mode` would not touch it.
fn write_private_file(path: &Path, contents: &[u8]) -> Result<(), CampaignError> {
    if path.exists() {
        fs::remove_file(path)?;
    }
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o700);
    }
    let mut file = options.open(path)?;
    file.write_all(contents)?;
    Ok(())
}

/// Create `path` as an owner-only directory.
///
/// The seed directory holds the replay script and both output logs, so its mode
/// matters for the same reason the script's does.
fn create_private_dir(path: &Path) -> Result<(), CampaignError> {
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path)?;
    Ok(())
}

fn shell_quote(path: &Path) -> String {
    shell_quote_value(&path.display().to_string())
}

fn shell_quote_value(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}

fn build_example(options: &CampaignOptions) -> Result<ExitStatus, CampaignError> {
    let mut command = Command::new(&options.oxide_binary);
    command.args(["build", &options.example]);
    if let Some(arch) = &options.arch {
        command.args(["--arch", arch]);
    }
    command.current_dir(&options.workspace_root);
    Ok(command.status()?)
}

fn find_executable(example_dir: &Path, example: &str) -> Result<PathBuf, CampaignError> {
    let metadata = Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1"])
        .current_dir(example_dir)
        .output()?;
    if !metadata.status.success() {
        return Err(CampaignError::Metadata(
            String::from_utf8_lossy(&metadata.stderr).trim().to_string(),
        ));
    }
    let document: Value = serde_json::from_slice(&metadata.stdout)
        .map_err(|error| CampaignError::Metadata(error.to_string()))?;
    let target_dir = document
        .get("target_directory")
        .and_then(Value::as_str)
        .ok_or_else(|| CampaignError::Metadata("target_directory is missing".to_string()))?;
    let packages = document
        .get("packages")
        .and_then(Value::as_array)
        .ok_or_else(|| CampaignError::Metadata("package is missing".to_string()))?;
    // Examples with a nested kernel-lib crate are multi-package workspaces,
    // and cargo does not put the binary package first, so every package's bin
    // targets are candidates. A bin named after the example or picked by its
    // package's default_run wins; otherwise the first bin found is used.
    let normalized_example = example.replace('-', "_");
    let mut bins: Vec<(&str, bool)> = Vec::new();
    for package in packages {
        let default_run = package.get("default_run").and_then(Value::as_str);
        let Some(targets) = package.get("targets").and_then(Value::as_array) else {
            continue;
        };
        for target in targets {
            let is_bin = target
                .get("kind")
                .and_then(Value::as_array)
                .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
            let Some(name) = target.get("name").and_then(Value::as_str) else {
                continue;
            };
            if is_bin {
                bins.push((name, Some(name) == default_run));
            }
        }
    }
    let target_name = bins
        .iter()
        .find(|(name, is_default_run)| *is_default_run || *name == normalized_example)
        .or_else(|| bins.first())
        .map(|(name, _)| *name)
        .ok_or_else(|| CampaignError::Metadata("no binary target found".to_string()))?;
    let executable = PathBuf::from(target_dir).join("release").join(target_name);
    if executable.is_file() {
        Ok(executable)
    } else {
        Err(CampaignError::MissingExecutable(executable))
    }
}

fn run_variant(
    variant: &Path,
    pristine: &Path,
    cwd: &Path,
    timeout: Duration,
    environment: &RunEnvironment,
) -> RunResult {
    let mut run = run_binary(variant, cwd, timeout, environment);

    // A CUDA watchdog timeout can leave the device unusable. Re-run the
    // pristine binary before continuing so a real device wedge is not
    // misreported as a collection of independent schedule failures.
    if matches!(run.kind, RunKind::Hang) {
        let health = run_binary(pristine, cwd, timeout, environment);
        if !matches!(health.kind, RunKind::Pass) {
            run.kind = RunKind::GpuWedged;
        }
    }
    run
}

fn classify_output_change(baseline: &RunResult, run: &mut RunResult, compare_output: bool) {
    if compare_output
        && matches!(run.kind, RunKind::Pass)
        && run.stdout.trim() != baseline.stdout.trim()
    {
        run.kind = RunKind::OutputChanged;
    }
}

fn run_binary(
    executable: &Path,
    cwd: &Path,
    timeout: Duration,
    environment: &RunEnvironment,
) -> RunResult {
    let mut command = Command::new(executable);
    command
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    environment.apply_to(&mut command);

    let Ok(mut child) = command.spawn() else {
        return RunResult {
            kind: RunKind::Crash,
            exit_code: None,
            timed_out: false,
            stdout: String::new(),
            stderr: "could not start example executable".to_string(),
        };
    };
    // Drain both pipes on their own threads while the watchdog polls. A child
    // that writes more than a pipe buffer would otherwise block on the full
    // pipe until the watchdog kills it, turning a large failure dump into a
    // spurious hang.
    let stdout_reader = drain_pipe(child.stdout.take());
    let stderr_reader = drain_pipe(child.stderr.take());
    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                let _ = child.kill();
                break;
            }
            Ok(None) => thread::sleep(Duration::from_millis(20)),
            Err(_) => break,
        }
    }

    let status = child.wait();
    let stdout = stdout_reader.join().unwrap_or_default();
    let stderr = stderr_reader.join().unwrap_or_default();
    let (exit_code, success, stderr) = match status {
        Ok(status) => (status.code(), status.success(), stderr),
        Err(error) => (None, false, error.to_string()),
    };
    let combined = format!("{stdout}\n{stderr}").to_ascii_lowercase();
    let kind = if timed_out {
        RunKind::Hang
    } else if has_skip_marker(&combined) {
        RunKind::Skipped
    } else if has_mismatch_marker(&combined) {
        RunKind::Mismatch
    } else if success {
        RunKind::Pass
    } else {
        RunKind::Crash
    };
    RunResult {
        kind,
        exit_code,
        timed_out,
        stdout,
        stderr,
    }
}

fn drain_pipe<R: io::Read + Send + 'static>(pipe: Option<R>) -> thread::JoinHandle<String> {
    thread::spawn(move || {
        let mut buffer = Vec::new();
        if let Some(mut pipe) = pipe {
            let _ = pipe.read_to_end(&mut buffer);
        }
        String::from_utf8_lossy(&buffer).into_owned()
    })
}

fn patch_executable(
    original: &Path,
    variant: &Path,
    pristine_ptx: &[u8],
    mutated_ptx: &[u8],
) -> Result<(), CampaignError> {
    fs::copy(original, variant)?;
    let original_bytes = fs::read(original)?;
    let bundles = read_artifact_bundles_from_object_bytes(&original_bytes)
        .map_err(|error| CampaignError::ArtifactPatch(error.to_string()))?;
    let section = rebuild_artifact_section(bundles, pristine_ptx, mutated_ptx)?;
    let section_path = variant.with_extension("oxart");
    fs::write(&section_path, section)?;

    let objcopy = std::env::var_os("OBJCOPY").unwrap_or_else(|| "objcopy".into());
    let section_arg = format!(".oxart={}", section_path.display());
    let status = Command::new(objcopy)
        .arg("--update-section")
        .arg(&section_arg)
        .arg(variant)
        .status()
        .map_err(|error| CampaignError::ArtifactPatch(format!("could not run objcopy: {error}")))?;
    if !status.success() {
        return Err(CampaignError::ArtifactPatch(format!(
            "objcopy failed with {status}"
        )));
    }
    Ok(())
}

fn rebuild_artifact_section(
    mut bundles: Vec<oxide_artifacts::OwnedArtifactBundle>,
    pristine_ptx: &[u8],
    mutated_ptx: &[u8],
) -> Result<Vec<u8>, CampaignError> {
    let ptx_count = bundles
        .iter()
        .flat_map(|bundle| bundle.payloads.iter())
        .filter(|payload| payload.kind == ArtifactPayloadKind::Ptx)
        .count();
    let mut replaced = false;
    let mut section = Vec::new();

    for bundle in &mut bundles {
        for payload in &mut bundle.payloads {
            let exact_match = payload.kind == ArtifactPayloadKind::Ptx
                && payload.bytes.as_slice() == pristine_ptx;
            let only_ptx_fallback =
                payload.kind == ArtifactPayloadKind::Ptx && ptx_count == 1 && !replaced;
            if exact_match || only_ptx_fallback {
                payload.bytes = mutated_ptx.to_vec();
                replaced = true;
            }
        }

        let payloads = bundle
            .payloads
            .iter()
            .map(|payload| ArtifactPayloadSpec::new(payload.kind, &payload.name, &payload.bytes))
            .collect();
        let entries = bundle
            .entries
            .iter()
            .map(|entry| {
                let spec = ArtifactEntrySpec::new(&entry.symbol, entry.kind);
                match entry.metadata {
                    Some(metadata) => spec.with_metadata(metadata),
                    None => spec,
                }
            })
            .collect();
        let spec = ArtifactBundleSpec {
            name: &bundle.name,
            target: &bundle.target,
            compile_options: bundle.compile_options,
            payloads,
            entries,
        };
        let blob = build_artifact_blob(&spec)
            .map_err(|error| CampaignError::ArtifactPatch(error.to_string()))?;
        section.extend(blob);
    }

    if !replaced {
        return Err(CampaignError::ArtifactPatch(
            "the generated PTX was not found in the executable's .oxart section".into(),
        ));
    }
    Ok(section)
}

/// Both spellings an example uses to decline a run.
///
/// The convention belongs to `scripts/smoketest.sh`, whose `verdict_standard`
/// greps `^[[:space:]]*(skipping:|pass \(skipped\))` case-insensitively and
/// whose own comment names the second form: "`PASS (skipped): ...` form below
/// sm_75". `generated_ldmatrix` prints exactly that when the device is under
/// sm_75.
///
/// Only the first form used to be accepted here. An example that declined with
/// the second one therefore exited 0 with no mismatch marker, so `run_binary`
/// called it [`RunKind::Pass`], the baseline gate let the campaign through, and
/// every seed "passed" a kernel that never ran -- a clean report from a
/// campaign that measured nothing.
const SKIP_MARKERS: [&str; 2] = ["skipping:", "pass (skipped)"];

/// `output` is already lowercased by `run_binary`, which is what makes the
/// comparison case-insensitive the way the smoketest's `grep -i` is.
fn has_skip_marker(output: &str) -> bool {
    output.lines().any(|line| {
        let line = line.trim_start();
        SKIP_MARKERS.iter().any(|marker| line.starts_with(marker))
    })
}

fn has_mismatch_marker(output: &str) -> bool {
    [
        "mismatch",
        "max error too large",
        "barrier sync failed",
        "fail:",
        "failed:",
        "failed!",
        "not unique",
        "incorrect",
        "wrong",
        "wrong answer",
        "does not match",
        "did not match",
        "validation failed",
        "deadlock",
    ]
    .iter()
    .any(|marker| output.contains(marker))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two spellings `scripts/smoketest.sh` accepts, and the near-misses
    /// that must not be mistaken for either.
    #[test]
    fn both_smoketest_skip_spellings_are_recognised() {
        // `run_binary` lowercases before classifying, so these arrive lowered.
        for declined in [
            "skipping: cluster launch requires sm_90",
            "  skipping: needs two devices",
            "pass (skipped): ldmatrix.m8n8.x4.b16 requires sm_75+; device is sm_70",
            "    pass (skipped): no peer access",
        ] {
            assert!(has_skip_marker(declined), "{declined}");
        }

        for ran in [
            "pass",
            "pass: 1024 elements verified",
            "success",
            "no skipping: here",
            "result was skipped by the host",
        ] {
            assert!(!has_skip_marker(ran), "{ran}");
        }
    }

    /// A declined run must not be reported as a passing baseline: the campaign
    /// gates on `RunKind::Pass` and would otherwise sweep every seed against a
    /// kernel that never launched.
    #[test]
    fn a_declined_run_is_skipped_not_passed() {
        for declined in [
            "skipping: needs sm_90\n",
            "pass (skipped): ldmatrix requires sm_75+\n",
        ] {
            assert!(has_skip_marker(declined), "{declined}");
        }
    }

    #[test]
    fn seed_ranges_are_half_open() {
        assert_eq!(parse_seed_range("3..8").unwrap(), (3, 8));
        assert!(parse_seed_range("8..8").is_err());
        assert!(parse_seed_range("8").is_err());
    }

    #[test]
    fn findings_use_schedule_sensitive_labels() {
        assert_eq!(
            RunKind::Mismatch.finding_label(),
            Some("SCHEDULE-SENSITIVE CORRECTNESS FAILURE")
        );
        assert_eq!(RunKind::Hang.finding_label(), Some("TIMEOUT CANDIDATE"));
        assert!(!RunKind::Pass.is_finding());
    }

    #[test]
    fn output_comparison_is_opt_in() {
        let baseline = RunResult {
            kind: RunKind::Pass,
            exit_code: Some(0),
            timed_out: false,
            stdout: "ok\n".to_string(),
            stderr: String::new(),
        };
        let mut unchanged = baseline.clone();
        classify_output_change(&baseline, &mut unchanged, false);
        assert!(matches!(unchanged.kind, RunKind::Pass));

        let mut changed = RunResult {
            stdout: "different\n".to_string(),
            ..baseline.clone()
        };
        classify_output_change(&baseline, &mut changed, true);
        assert!(matches!(changed.kind, RunKind::OutputChanged));
    }

    fn env(pairs: &[(&str, &str)]) -> Vec<(OsString, OsString)> {
        pairs
            .iter()
            .map(|(key, value)| (OsString::from(*key), OsString::from(*value)))
            .collect()
    }

    fn run_environment(pairs: &[(&str, &str)]) -> RunEnvironment {
        let source: BTreeMap<OsString, OsString> = env(pairs).into_iter().collect();
        RunEnvironment {
            replay: ReplayEnvironment::capture(source.clone()),
            source,
        }
    }

    /// A prefix rule cannot tell `CUDA_VISIBLE_DEVICES` from `CUDA_API_TOKEN`,
    /// and this text is written to a file. Nothing outside the explicit list
    /// reaches any field or the script, whatever its name looks like.
    #[test]
    fn a_secret_is_never_captured_or_written() {
        let secrets = [
            "CUDA_API_TOKEN",
            "CUDA_OXIDE_LICENSE_KEY",
            "GEMM_SOL_API_KEY",
            "MATHDX_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
        ];
        let mut pairs = vec![("CUDA_VISIBLE_DEVICES", "0")];
        for secret in secrets {
            pairs.push((secret, "sk-live-do-not-persist"));
        }
        let environment = ReplayEnvironment::capture(env(&pairs));
        let script = environment.script_prelude();

        assert_eq!(
            environment.captured,
            vec![("CUDA_VISIBLE_DEVICES".to_owned(), "0".to_owned())]
        );
        for secret in secrets {
            assert!(
                !environment.captured.iter().any(|(name, _)| name == secret),
                "{secret} captured"
            );
            assert!(
                !environment.unset.contains(&secret.to_owned()),
                "{secret} named"
            );
            assert!(
                !environment.required_external.contains(&secret.to_owned()),
                "{secret} named"
            );
            assert!(!script.contains(secret), "{secret} reached the script");
        }
        assert!(!script.contains("sk-live-do-not-persist"));
    }

    /// A captured name the campaign did not have must be *unset* by the script,
    /// not merely absent from it. Left absent, an ambient value in the replay
    /// shell silently changes the run -- which for `GEMM_SOL_MODE` picks a
    /// different mode than the one that produced the finding.
    #[test]
    fn an_absent_variable_is_unset_rather_than_left_to_the_replay_shell() {
        let environment = ReplayEnvironment::capture(env(&[("GEMM_SOL_MODE", "bench")]));
        assert!(environment.unset.contains(&"GEMM_SOL_VARIANT".to_owned()));
        assert!(
            environment
                .unset
                .contains(&"CUDA_VISIBLE_DEVICES".to_owned())
        );

        let script = environment.script_prelude();
        assert!(script.contains("unset GEMM_SOL_VARIANT\n"), "{script}");
        assert!(
            script.contains("export GEMM_SOL_MODE='bench'\n"),
            "{script}"
        );

        // Every captured name is accounted for exactly once.
        assert_eq!(
            environment.captured.len() + environment.unset.len(),
            CAPTURED_ENV.len()
        );
    }

    /// A value that is not valid UTF-8 cannot go into a POSIX script. The child
    /// still receives it, so the run is faithful, but the script must say the
    /// value is required from outside instead of inventing one.
    #[cfg(unix)]
    #[test]
    fn a_non_unicode_value_becomes_a_requirement_not_a_guess() {
        use std::os::unix::ffi::OsStringExt;
        let invalid = OsString::from_vec(vec![b'/', 0x80, 0xff, b'x']);
        let environment = ReplayEnvironment::capture(vec![
            (OsString::from("MATHDX_ROOT"), invalid),
            (OsString::from("CUDA_VISIBLE_DEVICES"), OsString::from("0")),
        ]);

        assert!(!environment.captured.iter().any(|(n, _)| n == "MATHDX_ROOT"));
        assert!(!environment.unset.contains(&"MATHDX_ROOT".to_owned()));
        assert!(
            environment
                .required_external
                .contains(&"MATHDX_ROOT".to_owned())
        );
        let script = environment.script_prelude();
        assert!(script.contains(": \"${MATHDX_ROOT:?"), "{script}");
        assert!(script.is_ascii(), "a non-UTF-8 byte reached the script");
    }

    /// The run must see the captured environment and nothing else. This is the
    /// half a replay script cannot fix: without `env_clear` the child inherits
    /// whatever the campaign held, so the script and the run describe different
    /// environments no matter how careful the script is.
    #[cfg(unix)]
    #[test]
    fn the_child_inherits_nothing_that_was_not_captured() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("ptx-schedule-env-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("dump-env.sh");
        fs::write(&script, "#!/bin/sh\nenv\n").unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).unwrap();

        let pairs = [
            ("CUDA_VISIBLE_DEVICES", "3"),
            ("CUDA_API_TOKEN", "sk-live-do-not-inherit"),
            ("UNRELATED_CARRIER", "should-not-appear"),
            ("PATH", "/usr/bin:/bin"),
        ];
        let environment = run_environment(&pairs);
        let result = run_binary(&script, &dir, Duration::from_secs(30), &environment);
        fs::remove_dir_all(&dir).ok();

        assert!(matches!(result.kind, RunKind::Pass), "{result:?}");
        let seen: Vec<&str> = result
            .stdout
            .lines()
            .filter_map(|line| line.split('=').next())
            .collect();
        assert!(seen.contains(&"CUDA_VISIBLE_DEVICES"), "{seen:?}");
        assert!(seen.contains(&"PATH"), "{seen:?}");
        assert!(!seen.contains(&"CUDA_API_TOKEN"), "{seen:?}");
        assert!(!seen.contains(&"UNRELATED_CARRIER"), "{seen:?}");
        assert!(!result.stdout.contains("sk-live-do-not-inherit"));

        // The assertions above are not enough on their own: a name that exists
        // only in the synthetic map is absent from the child whether
        // `env_clear` ran or not, so they pass vacuously. What proves the clear
        // is that nothing from *this* process's real environment reached the
        // child. `sh` adds `PWD`, `SHLVL` and `_` itself, so those are not
        // inheritance.
        let intended: Vec<&str> = environment
            .replay
            .captured
            .iter()
            .map(|(name, _)| name.as_str())
            .chain(
                environment
                    .replay
                    .required_external
                    .iter()
                    .map(String::as_str),
            )
            .collect();
        let leaked: Vec<&&str> = seen
            .iter()
            .filter(|name| !intended.contains(*name))
            .filter(|name| !["PWD", "SHLVL", "_"].contains(*name))
            .filter(|name| std::env::var_os(*name).is_some())
            .collect();
        assert!(
            leaked.is_empty(),
            "{} variables reached the child from the parent environment: {leaked:?}",
            leaked.len()
        );
    }

    /// The script and the directory holding it carry captured values, so both
    /// are owner-only, and the script is created that way rather than narrowed
    /// afterwards.
    #[cfg(unix)]
    #[test]
    fn the_script_and_its_directory_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let root = std::env::temp_dir().join(format!("ptx-schedule-mode-{}", std::process::id()));
        fs::remove_dir_all(&root).ok();
        let seed_dir = root.join("seed-0");
        create_private_dir(&seed_dir).unwrap();
        let path = seed_dir.join("replay.sh");
        let environment = ReplayEnvironment::capture(env(&[("CUDA_VISIBLE_DEVICES", "0")]));
        write_replay_script(&path, &seed_dir, Path::new("/bin/true"), &environment).unwrap();

        let dir_mode = fs::metadata(&seed_dir).unwrap().permissions().mode() & 0o777;
        let file_mode = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(dir_mode, 0o700, "directory mode {dir_mode:o}");
        assert_eq!(file_mode, 0o700, "script mode {file_mode:o}");

        // Rewriting must not leave a wider mode behind either.
        write_replay_script(&path, &seed_dir, Path::new("/bin/true"), &environment).unwrap();
        let again = fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(again, 0o700, "script mode after rewrite {again:o}");
        fs::remove_dir_all(&root).ok();
    }

    /// The `${NAME:?}` guard has to make the script refuse, not warn. A replay
    /// in an environment missing something the run needed must stop.
    #[cfg(unix)]
    #[test]
    fn a_replay_missing_a_required_variable_refuses_to_run() {
        let dir = std::env::temp_dir().join(format!("ptx-schedule-guard-{}", std::process::id()));
        fs::remove_dir_all(&dir).ok();
        create_private_dir(&dir).unwrap();
        let path = dir.join("replay.sh");
        let environment = ReplayEnvironment::capture(env(&[
            ("CUDA_VISIBLE_DEVICES", "0"),
            ("LD_LIBRARY_PATH", "/opt/cuda/lib64"),
        ]));
        assert!(
            environment
                .required_external
                .contains(&"LD_LIBRARY_PATH".to_owned())
        );
        write_replay_script(&path, &dir, Path::new("/bin/true"), &environment).unwrap();

        let refused = Command::new("/bin/sh")
            .arg(&path)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .output()
            .unwrap();
        assert!(!refused.status.success(), "the guard did not refuse");
        assert!(
            String::from_utf8_lossy(&refused.stderr).contains("LD_LIBRARY_PATH"),
            "{}",
            String::from_utf8_lossy(&refused.stderr)
        );

        let accepted = Command::new("/bin/sh")
            .arg(&path)
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LD_LIBRARY_PATH", "/opt/cuda/lib64")
            .output()
            .unwrap();
        assert!(
            accepted.status.success(),
            "{}",
            String::from_utf8_lossy(&accepted.stderr)
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// A value with a quote in it still has to survive the round trip.
    #[test]
    fn a_value_containing_a_quote_is_escaped_for_the_shell() {
        assert_eq!(shell_quote_value("it's"), "'it'\\''s'");
        assert_eq!(shell_quote_value("plain"), "'plain'");
    }

    #[test]
    fn shell_quotes_replay_values() {
        assert_eq!(shell_quote_value("a'b"), "'a'\\''b'");
    }

    #[cfg(unix)]
    #[test]
    fn run_binary_drains_output_larger_than_a_pipe_buffer() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("ptx-schedule-drain-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let script = dir.join("spam.sh");
        // 1 MiB of stdout, far beyond the pipe buffer the watchdog loop used
        // to deadlock against before the reader threads were added.
        fs::write(
            &script,
            "#!/bin/sh\ndd if=/dev/zero bs=65536 count=16 2>/dev/null | tr '\\0' 'a'\n",
        )
        .unwrap();
        let mut permissions = fs::metadata(&script).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).unwrap();
        let result = run_binary(
            &script,
            &dir,
            Duration::from_secs(30),
            &run_environment(&[]),
        );
        fs::remove_dir_all(&dir).ok();
        assert!(matches!(result.kind, RunKind::Pass), "{:?}", result.kind);
        assert!(!result.timed_out);
        assert_eq!(result.stdout.len(), 16 * 65536);
    }

    #[test]
    fn harness_errors_are_recorded_without_becoming_findings() {
        let error = CampaignError::ArtifactPatch("objcopy failed".into());
        let result = harness_error_result(7, 1.5, PathBuf::from("seed-7"), &error);
        assert!(matches!(result.run.kind, RunKind::HarnessError));
        assert!(!result.run.kind.is_finding());
        assert!(result.run.stderr.contains("objcopy failed"));
        assert_eq!(result.seed, 7);
        assert_eq!(result.report.seed, 7);
        assert_eq!(result.report.sites_injected, 0);
    }
}
