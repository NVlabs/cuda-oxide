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
use std::fs;
use std::io;
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
}

impl RunKind {
    fn finding_label(&self) -> Option<&'static str> {
        match self {
            Self::Mismatch => Some("SCHEDULE-SENSITIVE CORRECTNESS FAILURE"),
            Self::OutputChanged => Some("SCHEDULE-SENSITIVE OUTPUT CHANGE"),
            Self::Hang => Some("TIMEOUT CANDIDATE"),
            Self::Crash => Some("CRASH CANDIDATE"),
            Self::GpuWedged => Some("GPU WEDGE CANDIDATE"),
            Self::Pass | Self::Skipped => None,
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

    println!("schedule-fuzz: baseline {}", executable.display());
    let baseline = run_binary(&executable, &example_dir, options.timeout);
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

    let mut seeds = Vec::new();
    for seed in options.seed_start..options.seed_end {
        let rewrite = crate::perturb_ptx(
            &pristine,
            &InjectionOptions {
                seed,
                intensity: options.intensity,
                max_sleep_ns: options.max_sleep_ns,
                focus: options.focus.clone(),
            },
        )?;
        let artifact_dir = output_dir.join(format!("seed-{seed}"));
        fs::create_dir_all(&artifact_dir)?;
        let mutated_ptx = artifact_dir.join("module.ptx");
        let report_path = artifact_dir.join("report.json");
        let stdout_path = artifact_dir.join("stdout.log");
        let stderr_path = artifact_dir.join("stderr.log");
        let replay_path = artifact_dir.join("replay.sh");
        fs::write(&mutated_ptx, &rewrite.ptx)?;
        fs::write(&report_path, serde_json::to_vec_pretty(&rewrite.report)?)?;

        let variant_executable =
            artifact_dir.join(executable.file_name().ok_or_else(|| {
                CampaignError::ArtifactPatch("executable has no file name".into())
            })?);
        patch_executable(
            &executable,
            &variant_executable,
            pristine.as_bytes(),
            rewrite.ptx.as_bytes(),
        )?;
        write_replay_script(&replay_path, &example_dir, &variant_executable)?;

        let mut run = run_variant(
            &variant_executable,
            &executable,
            &example_dir,
            options.timeout,
        );
        classify_output_change(&baseline, &mut run, options.compare_output);

        fs::write(&stdout_path, &run.stdout)?;
        fs::write(&stderr_path, &run.stderr)?;
        let confirmation = if run.kind.is_finding() && options.confirm_runs > 1 {
            let mut outcomes = vec![run.kind.clone()];
            let mut findings = 1;
            for attempt in 1..options.confirm_runs {
                let mut confirmed_run = run_variant(
                    &variant_executable,
                    &executable,
                    &example_dir,
                    options.timeout,
                );
                classify_output_change(&baseline, &mut confirmed_run, options.compare_output);
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
        let result = SeedResult {
            seed,
            artifact_dir,
            report: rewrite.report,
            run,
            confirmation,
            replay: replay_path,
        };
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

fn print_seed_result(result: &SeedResult) {
    if let Some(label) = result.run.kind.finding_label() {
        println!("schedule-fuzz: FINDING seed={} — {}", result.seed, label);
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

    println!();
    println!("=== schedule-fuzz result ===");
    println!("example: {}", summary.example);
    println!("baseline: PASS");
    println!("variants: {}", summary.seeds.len());
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

fn write_replay_script(path: &Path, cwd: &Path, executable: &Path) -> Result<(), CampaignError> {
    let mut script = format!("#!/bin/sh\nset -eu\n");
    for key in [
        "CUDA_VISIBLE_DEVICES",
        "CUDA_LAUNCH_BLOCKING",
        "GEMM_SOL_PHASE",
    ] {
        if let Ok(value) = std::env::var(key) {
            script.push_str(&format!("export {key}={}\n", shell_quote_value(&value)));
        }
    }
    script.push_str(&format!(
        "cd {}\nexec {}\n",
        shell_quote(cwd),
        shell_quote(executable)
    ));
    fs::write(path, script)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut permissions = fs::metadata(path)?.permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(path, permissions)?;
    }
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
    let package = document
        .get("packages")
        .and_then(Value::as_array)
        .and_then(|packages| packages.first())
        .ok_or_else(|| CampaignError::Metadata("package is missing".to_string()))?;
    let default_run = package.get("default_run").and_then(Value::as_str);
    let normalized_example = example.replace('-', "_");
    let target_name = package
        .get("targets")
        .and_then(Value::as_array)
        .and_then(|targets| {
            targets.iter().find_map(|target| {
                let is_bin = target
                    .get("kind")
                    .and_then(Value::as_array)
                    .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
                let name = target.get("name").and_then(Value::as_str)?;
                if is_bin && (Some(name) == default_run || name == normalized_example) {
                    Some(name)
                } else {
                    None
                }
            })
        })
        .or_else(|| {
            package
                .get("targets")
                .and_then(Value::as_array)
                .and_then(|targets| {
                    targets.iter().find_map(|target| {
                        let is_bin =
                            target
                                .get("kind")
                                .and_then(Value::as_array)
                                .is_some_and(|kinds| {
                                    kinds.iter().any(|kind| kind.as_str() == Some("bin"))
                                });
                        is_bin
                            .then(|| target.get("name").and_then(Value::as_str))
                            .flatten()
                    })
                })
        })
        .ok_or_else(|| CampaignError::Metadata("no binary target found".to_string()))?;
    let executable = PathBuf::from(target_dir).join("release").join(target_name);
    if executable.is_file() {
        Ok(executable)
    } else {
        Err(CampaignError::MissingExecutable(executable))
    }
}

fn run_variant(variant: &Path, pristine: &Path, cwd: &Path, timeout: Duration) -> RunResult {
    let mut run = run_binary(variant, cwd, timeout);

    // A CUDA watchdog timeout can leave the device unusable. Re-run the
    // pristine binary before continuing so a real device wedge is not
    // misreported as a collection of independent schedule failures.
    if matches!(run.kind, RunKind::Hang) {
        let health = run_binary(pristine, cwd, timeout);
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

fn run_binary(executable: &Path, cwd: &Path, timeout: Duration) -> RunResult {
    let mut command = Command::new(executable);
    command
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let Ok(mut child) = command.spawn() else {
        return RunResult {
            kind: RunKind::Crash,
            exit_code: None,
            timed_out: false,
            stdout: String::new(),
            stderr: "could not start example executable".to_string(),
        };
    };
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

    let output = child.wait_with_output();
    let (exit_code, success, stdout, stderr) = match output {
        Ok(output) => (
            output.status.code(),
            output.status.success(),
            String::from_utf8_lossy(&output.stdout).into_owned(),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ),
        Err(error) => (None, false, String::new(), error.to_string()),
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

fn has_skip_marker(output: &str) -> bool {
    output
        .lines()
        .any(|line| line.trim_start().starts_with("skipping:"))
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

    #[test]
    fn shell_quotes_replay_values() {
        assert_eq!(shell_quote_value("a'b"), "'a'\\''b'");
    }
}
