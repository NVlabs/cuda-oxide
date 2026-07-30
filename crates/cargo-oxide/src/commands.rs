/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Command implementations for cargo-oxide.
//!
//! These port the xtask commands with improvements:
//! - Backend path resolved via discovery chain instead of hardcoded relative path
//! - Workspace root resolved by walking up from CWD instead of assuming CWD

use crate::backend;
use sha2::Digest as _;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const MATERIALIZE_ENV: &str = reserved_oxide_symbols::MATERIALIZE_CUBIN_ENV;
const EXPECTED_PROVENANCE_ENV: &str = reserved_oxide_symbols::MATERIALIZER_PROVENANCE_ENV;
const CODEGEN_FINGERPRINT_ENV: &str = reserved_oxide_symbols::CODEGEN_FINGERPRINT_ENV;
const DEVICE_CODEGEN_CRATE_ENV: &str = reserved_oxide_symbols::DEVICE_CODEGEN_CRATE_ENV;
const BACKEND_IDENTITY_CFG: &str = "cuda_oxide_internal_backend_identity";
const LEGACY_CODEGEN_FINGERPRINT_CFG: &str = "cuda_oxide_internal_codegen_env";
const LEGACY_MATERIALIZER_PROVENANCE_CFG: &str = "cuda_oxide_internal_materializer_provenance";

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct MaterializationMode {
    provenance: Option<String>,
}

impl MaterializationMode {
    fn enabled(&self) -> bool {
        self.provenance.is_some()
    }

    fn apply(&self, cmd: &mut Command) {
        if let Some(provenance) = &self.provenance {
            // These override inherited/project values: they are a single
            // wrapper-generated handshake tied to this Cargo invocation.
            cmd.env(MATERIALIZE_ENV, "1")
                .env(EXPECTED_PROVENANCE_ENV, provenance)
                .env("CUDA_OXIDE_EMIT_NVVM_IR", "1");
        }
    }
}

fn prepare_materialization(
    ctx: &Context,
    cli_requested: bool,
    cli_arch: Option<&str>,
    emit_nvvm_ir: bool,
) -> MaterializationMode {
    prepare_materialization_result(ctx, cli_requested, cli_arch, emit_nvvm_ir).unwrap_or_else(
        |error| {
            eprintln!("Error: {error}");
            std::process::exit(2);
        },
    )
}

/// `prepare_materialization` with the ambient `CUDA_OXIDE_MATERIALIZE_CUBIN`
/// injected, so `cargo_passthrough_command_with_env` can reach
/// `materialization_requested_with_env`.
///
/// Note this still exits the process on error, which inside a unit test aborts
/// the whole test binary rather than failing one case -- a further reason tests
/// must not reach the ambient read.
fn prepare_materialization_with_env(
    ctx: &Context,
    cli_requested: bool,
    cli_arch: Option<&str>,
    emit_nvvm_ir: bool,
    materialize_env: Option<std::ffi::OsString>,
) -> MaterializationMode {
    prepare_materialization_result_with_env(
        ctx,
        cli_requested,
        cli_arch,
        emit_nvvm_ir,
        materialize_env,
    )
    .unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(2);
    })
}

const EMIT_NVVM_IR_ENV: &str = "CUDA_OXIDE_EMIT_NVVM_IR";

fn nvvm_ir_requested(ctx: &Context) -> Result<bool, String> {
    nvvm_ir_requested_with_env(ctx, std::env::var_os(EMIT_NVVM_IR_ENV))
}

/// `nvvm_ir_requested` with the ambient `CUDA_OXIDE_EMIT_NVVM_IR` injected.
///
/// The process value outranks project config, so resolution has to be
/// injectable for unit tests: an exported `CUDA_OXIDE_EMIT_NVVM_IR` would
/// otherwise decide the answer before the configured value is consulted.
fn nvvm_ir_requested_with_env(
    ctx: &Context,
    env_value: Option<std::ffi::OsString>,
) -> Result<bool, String> {
    if let Some(value) = env_value {
        let value = value
            .into_string()
            .map_err(|_| format!("{EMIT_NVVM_IR_ENV} is not valid Unicode"))?;
        return parse_strict_bool(EMIT_NVVM_IR_ENV, &value);
    }

    if let Some(value) = project_config_env(ctx, EMIT_NVVM_IR_ENV) {
        return parse_strict_bool(EMIT_NVVM_IR_ENV, value);
    }

    Ok(false)
}

fn materialization_requested(ctx: &Context, cli_requested: bool) -> Result<bool, String> {
    materialization_requested_with_env(ctx, cli_requested, std::env::var_os(MATERIALIZE_ENV))
}

/// `materialization_requested` with the ambient `CUDA_OXIDE_MATERIALIZE_CUBIN`
/// injected.
///
/// The process value outranks project config, so resolution has to be
/// injectable for unit tests: an exported value would otherwise turn
/// materialization on for tests that pass `materialize_cubin: false`, sending
/// them into `discover_materializer_provenance`. Same rationale as
/// `nvvm_ir_requested_with_env`.
fn materialization_requested_with_env(
    ctx: &Context,
    cli_requested: bool,
    env_value: Option<std::ffi::OsString>,
) -> Result<bool, String> {
    if cli_requested {
        return Ok(true);
    }

    if let Some(value) = env_value {
        let value = value
            .into_string()
            .map_err(|_| format!("{MATERIALIZE_ENV} is not valid Unicode"))?;
        return parse_strict_bool(MATERIALIZE_ENV, &value);
    }

    if let Some(value) = project_config_env(ctx, MATERIALIZE_ENV) {
        return parse_strict_bool(MATERIALIZE_ENV, value);
    }

    Ok(false)
}

fn prepare_materialization_result(
    ctx: &Context,
    cli_requested: bool,
    cli_arch: Option<&str>,
    emit_nvvm_ir: bool,
) -> Result<MaterializationMode, String> {
    prepare_materialization_result_with_env(
        ctx,
        cli_requested,
        cli_arch,
        emit_nvvm_ir,
        std::env::var_os(MATERIALIZE_ENV),
    )
}

/// `prepare_materialization_result` with the ambient
/// `CUDA_OXIDE_MATERIALIZE_CUBIN` injected, forwarded to
/// `materialization_requested_with_env`.
fn prepare_materialization_result_with_env(
    ctx: &Context,
    cli_requested: bool,
    cli_arch: Option<&str>,
    emit_nvvm_ir: bool,
    materialize_env: Option<std::ffi::OsString>,
) -> Result<MaterializationMode, String> {
    let enabled = materialization_requested_with_env(ctx, cli_requested, materialize_env)?;
    if !enabled {
        return Ok(MaterializationMode::default());
    }
    if emit_nvvm_ir {
        return Err(
            "--materialize-cubin cannot be combined with --emit-nvvm-ir; one requests a final cubin and the other requests NVVM IR"
                .to_string(),
        );
    }

    let arch = configured_arch_label(ctx, cli_arch).ok_or_else(|| {
        "--materialize-cubin requires --arch, CUDA_OXIDE_TARGET, or a configured default-arch"
            .to_string()
    })?;
    let _: cuda_artifact_finalizer::CudaArch = arch
        .parse()
        .map_err(|error| format!("invalid materialization target {arch:?}: {error}"))?;

    Ok(MaterializationMode {
        provenance: Some(discover_materializer_provenance(ctx)?),
    })
}

fn discover_materializer_provenance(ctx: &Context) -> Result<String, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("could not locate cargo-oxide executable: {error}"))?;
    let mut command = materializer_discovery_command(ctx, &executable);
    let output = command
        .output()
        .map_err(|error| format!("could not start CUDA materializer discovery: {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "CUDA materializer discovery failed: {}",
            stderr.trim()
        ));
    }
    let provenance = String::from_utf8(output.stdout)
        .map_err(|_| "CUDA materializer discovery returned non-UTF-8 output".to_string())?;
    let provenance = provenance.trim();
    if provenance.len() != 64
        || !provenance
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    {
        return Err(format!(
            "CUDA materializer discovery returned an invalid provenance digest: {provenance:?}"
        ));
    }
    Ok(provenance.to_string())
}

fn materializer_discovery_command(ctx: &Context, executable: &Path) -> Command {
    let mut command = Command::new(executable);
    command.arg("__materializer-provenance");
    apply_config_env(&mut command, ctx);
    apply_ld_library_path(&mut command, ctx);
    command
}

pub fn print_materializer_provenance() {
    let finalizer = cuda_artifact_finalizer::Finalizer::discover().unwrap_or_else(|error| {
        eprintln!("could not discover CUDA artifact finalizer: {error}");
        std::process::exit(1);
    });
    let provenance = finalizer.provenance_digest().unwrap_or_else(|| {
        eprintln!(
            "the loaded libNVVM or nvJitLink library cannot be tied to an exact file; refusing materialization because Cargo could not fingerprint the compiler inputs"
        );
        std::process::exit(1);
    });
    println!("{}", digest_hex(&provenance));
}

fn parse_strict_bool(name: &str, value: &str) -> Result<bool, String> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(format!(
            "{name} must be a boolean (accepted true values: 1, true, yes, on; false values: 0, false, no, off), got {value:?}"
        )),
    }
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use std::fmt::Write;
    let mut hex = String::with_capacity(64);
    for byte in digest {
        write!(&mut hex, "{byte:02x}").expect("writing to String cannot fail");
    }
    hex
}

/// Project-local cuda-oxide defaults loaded from `.cargo/cuda-oxide.toml`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OxideConfig {
    /// Explicit backend shared object path.
    pub backend: Option<PathBuf>,
    /// Default CUDA architecture for codegen commands.
    pub default_arch: Option<String>,
    /// Additional rustflags appended after cuda-oxide's required flags.
    pub extra_rustflags: Vec<String>,
    /// Environment variables applied to child Cargo invocations.
    pub env: Vec<(String, String)>,
}

/// Pre-resolved context shared across all commands.
///
/// Built once at startup by [`resolve_context`] and passed by reference to
/// every command handler. Avoids repeated filesystem walks and backend builds.
pub struct Context {
    /// Absolute path to the workspace root (contains top-level `Cargo.toml`).
    pub workspace_root: PathBuf,
    /// Path to `crates/rustc-codegen-cuda` (backend source tree).
    pub codegen_crate: PathBuf,
    /// Path to `crates/rustc-codegen-cuda/examples/`.
    pub examples_dir: PathBuf,
    /// Path to the built `librustc_codegen_cuda.so` shared object.
    pub backend_so: PathBuf,
    /// True when running from inside the cuda-oxide workspace; false for
    /// standalone projects scaffolded by `cargo oxide new`.
    pub is_workspace: bool,
    /// Project-local cuda-oxide defaults.
    pub config: OxideConfig,
}

/// Resolve the workspace root and backend, or exit with a helpful error.
///
/// Supports two modes:
/// - **Workspace mode**: CWD is inside the cuda-oxide repo (detected by
///   `crates/rustc-codegen-cuda` directory). Examples are resolved from the
///   workspace examples directory.
/// - **Standalone mode**: CWD has a `Cargo.toml` but is not inside the
///   workspace. The backend is located via cache or auto-fetch. Commands
///   like `run` operate on the current directory directly.
pub fn resolve_context() -> Context {
    if let Some(workspace_root) = backend::find_workspace_root() {
        let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
        let examples_dir = codegen_crate.join("examples");
        let config = load_oxide_config(&workspace_root);
        let backend_so = backend::find_or_build_backend(&workspace_root, config.backend.as_deref());
        return Context {
            workspace_root,
            codegen_crate,
            examples_dir,
            backend_so,
            is_workspace: true,
            config,
        };
    }

    let cwd = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("Error: cannot determine current directory: {}", e);
        std::process::exit(1);
    });

    if cwd.join("Cargo.toml").is_file() {
        let config = load_oxide_config(&cwd);
        let backend_so = backend::find_or_build_backend(&cwd, config.backend.as_deref());
        return Context {
            workspace_root: cwd.clone(),
            codegen_crate: cwd.clone(),
            examples_dir: cwd.clone(),
            backend_so,
            is_workspace: false,
            config,
        };
    }

    eprintln!("Error: Could not find cuda-oxide workspace or a standalone Cargo.toml.");
    eprintln!();
    eprintln!("Run from inside the cuda-oxide repository, or from a project created");
    eprintln!("with `cargo oxide new <name>`.");
    std::process::exit(1);
}

/// Resolve a context for commands that must not build or fetch the backend.
///
/// Identical discovery to [`resolve_context`], except the backend `.so` is
/// only located via [`backend::backend_so_candidate`], never built and never
/// cloned, and an invalid `.cargo/cuda-oxide.toml` degrades to defaults with
/// a warning instead of exiting (so `doctor` can report it as a failed
/// check). Passive commands such as `doctor` and `clean` must remain usable
/// without triggering backend setup or network access.
/// `run`/`build`/`pipeline`/`setup` still build the backend on demand.
pub fn resolve_passive_context() -> Context {
    if let Some(workspace_root) = backend::find_workspace_root() {
        let codegen_crate = workspace_root.join("crates/rustc-codegen-cuda");
        let examples_dir = codegen_crate.join("examples");
        let config = load_oxide_config_lenient(&workspace_root);
        let backend_so = backend::backend_so_candidate(&workspace_root, config.backend.as_deref());
        return Context {
            workspace_root,
            codegen_crate,
            examples_dir,
            backend_so,
            is_workspace: true,
            config,
        };
    }

    let cwd = std::env::current_dir().unwrap_or_else(|e| {
        eprintln!("Error: cannot determine current directory: {}", e);
        std::process::exit(1);
    });

    if cwd.join("Cargo.toml").is_file() {
        let config = load_oxide_config_lenient(&cwd);
        let backend_so = backend::backend_so_candidate(&cwd, config.backend.as_deref());
        return Context {
            workspace_root: cwd.clone(),
            codegen_crate: cwd.clone(),
            examples_dir: cwd.clone(),
            backend_so,
            is_workspace: false,
            config,
        };
    }

    eprintln!("Error: Could not find cuda-oxide workspace or a standalone Cargo.toml.");
    eprintln!();
    eprintln!("Run from inside the cuda-oxide repository, or from a project created");
    eprintln!("with `cargo oxide new <name>`.");
    std::process::exit(1);
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ExampleInfo {
    name: String,
    title: String,
    description: String,
    requirements: Vec<String>,
}

#[derive(Debug, Default, Eq, PartialEq)]
struct ParsedReadme {
    title: Option<String>,
    description: Option<String>,
    requirements: Vec<String>,
}

#[derive(Debug, Default, Eq, PartialEq)]
struct ManifestInfo {
    description: Option<String>,
}

pub fn list_examples(ctx: &Context, json: bool) {
    if !ctx.is_workspace {
        eprintln!("Error: `cargo oxide list` must be run from inside a cuda-oxide checkout.");
        eprintln!();
        eprintln!("The command lists examples under crates/rustc-codegen-cuda/examples/.");
        std::process::exit(1);
    }

    let examples = discover_examples(&ctx.examples_dir).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });

    let output = if json {
        format_examples_json(&examples).unwrap_or_else(|error| {
            eprintln!("Error: could not serialize example list: {error}");
            std::process::exit(1);
        })
    } else {
        format_examples_human(&examples)
    };

    print!("{output}");
}

fn discover_examples(examples_dir: &Path) -> Result<Vec<ExampleInfo>, String> {
    let entries = fs::read_dir(examples_dir).map_err(|error| {
        format!(
            "could not read examples directory {}: {error}",
            examples_dir.display()
        )
    })?;

    let mut examples = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "could not read an entry under {}: {error}",
                examples_dir.display()
            )
        })?;

        let file_type = entry
            .file_type()
            .map_err(|error| format!("could not inspect {}: {error}", entry.path().display()))?;

        if !file_type.is_dir() {
            continue;
        }

        let example_dir = entry.path();
        let name = entry.file_name().into_string().map_err(|name| {
            format!(
                "example directory name is not valid UTF-8: {}",
                name.to_string_lossy()
            )
        })?;

        // A directory without a manifest is not an example (scratch dirs,
        // checked-out tooling, ...). Skip it instead of failing the listing.
        let manifest_path = example_dir.join("Cargo.toml");
        if !manifest_path.is_file() {
            eprintln!(
                "Warning: skipping {}: no top-level Cargo.toml",
                example_dir.display()
            );
            continue;
        }

        let manifest = parse_example_manifest(&manifest_path)?;

        let readme_path = example_dir.join("README.md");
        let parsed_readme = if readme_path.is_file() {
            let contents = fs::read_to_string(&readme_path)
                .map_err(|error| format!("could not read {}: {error}", readme_path.display()))?;
            parse_example_readme(&name, &contents)
        } else {
            ParsedReadme::default()
        };

        let ParsedReadme {
            title,
            description,
            requirements,
        } = parsed_readme;

        let title = title.unwrap_or_else(|| name.clone());

        let description = description.or(manifest.description).unwrap_or_else(|| {
            if title != name {
                title.clone()
            } else {
                "No description documented.".to_string()
            }
        });

        examples.push(ExampleInfo {
            name,
            title,
            description,
            requirements,
        });
    }

    examples.sort_by(|left, right| left.name.cmp(&right.name));

    if examples.is_empty() {
        return Err(format!(
            "no examples found under {}",
            examples_dir.display()
        ));
    }

    Ok(examples)
}

fn parse_example_manifest(path: &Path) -> Result<ManifestInfo, String> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;

    let manifest: toml::Value = toml::from_str(&contents)
        .map_err(|error| format!("could not parse {}: {error}", path.display()))?;

    let package = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| format!("{} has no [package] table", path.display()))?;

    let description = package
        .get("description")
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|description| !description.is_empty())
        .map(str::to_owned);

    Ok(ManifestInfo { description })
}

fn parse_example_readme(crate_name: &str, contents: &str) -> ParsedReadme {
    let lines: Vec<&str> = contents.lines().collect();
    let mut headings = Vec::new();
    let mut in_code_fence = false;

    for (index, line) in lines.iter().enumerate() {
        if is_code_fence(line) {
            in_code_fence = !in_code_fence;
            continue;
        }

        if in_code_fence {
            continue;
        }

        if let Some((level, title)) = parse_markdown_heading(line) {
            headings.push((index, level, title));
        }
    }

    let crate_heading = normalize_heading(crate_name);

    let selected_heading = match headings.first() {
        Some(first) if normalize_heading(&first.2) == crate_heading => headings
            .get(1)
            .filter(|heading| {
                heading.1 == 2
                    && first_prose_paragraph_in_range(&lines, first.0 + 1, heading.0).is_none()
                    && !is_generic_section_heading(&normalize_heading(&heading.2))
            })
            .or(Some(first)),
        Some(first) => Some(first),
        None => None,
    };

    let title = selected_heading
        .map(|(_, _, title)| strip_inline_markdown(title))
        .filter(|title| !title.is_empty());

    let description_start = selected_heading.map(|(index, _, _)| index + 1).unwrap_or(0);

    let description_end = headings
        .iter()
        .find(|(index, _, _)| *index >= description_start)
        .map(|(index, _, _)| *index)
        .unwrap_or(lines.len());

    let description = first_prose_paragraph_in_range(&lines, description_start, description_end);
    let requirements = extract_requirements(&lines);

    ParsedReadme {
        title,
        description,
        requirements,
    }
}

fn parse_markdown_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    let level = trimmed.bytes().take_while(|byte| *byte == b'#').count();

    if !(1..=6).contains(&level) {
        return None;
    }

    let remainder = &trimmed[level..];
    if !remainder.chars().next().is_some_and(char::is_whitespace) {
        return None;
    }

    let title = remainder.trim().trim_end_matches('#').trim();

    if title.is_empty() {
        None
    } else {
        Some((level, title.to_string()))
    }
}

fn is_code_fence(line: &str) -> bool {
    let trimmed = line.trim_start();
    trimmed.starts_with("```") || trimmed.starts_with("~~~")
}

fn strip_inline_markdown(value: &str) -> String {
    value
        .replace("**", "")
        .replace("__", "")
        .replace('`', "")
        .trim()
        .to_string()
}

fn normalize_heading(value: &str) -> String {
    strip_inline_markdown(value)
        .trim_matches(|character: char| {
            character == ':' || character == '-' || character.is_whitespace()
        })
        .to_ascii_lowercase()
}

fn is_generic_section_heading(heading: &str) -> bool {
    matches!(
        heading,
        "overview"
            | "what this example does"
            | "key concepts"
            | "key concepts demonstrated"
            | "build"
            | "build and run"
            | "usage"
            | "expected output"
            | "requirements"
            | "hardware requirements"
            | "prerequisites"
            | "potential errors"
            | "how it works"
            | "how it works under the hood"
            | "generated ptx"
            | "run"
            | "test"
            | "tests"
            | "correctness"
            | "trigger"
            | "kernels"
            | "features tested"
            | "what this tests"
            | "what it tests"
            | "what this demonstrates"
            | "why this exists"
            | "the bug"
            | "final design"
    )
}

fn first_prose_paragraph_in_range(lines: &[&str], start: usize, end: usize) -> Option<String> {
    let end = end.min(lines.len());
    let start = start.min(end);
    let mut paragraph = Vec::new();
    let mut in_code_fence = false;

    for line in &lines[start..end] {
        if is_code_fence(line) {
            if !paragraph.is_empty() {
                break;
            }
            in_code_fence = !in_code_fence;
            continue;
        }

        if in_code_fence {
            continue;
        }

        let trimmed = line.trim();

        if parse_markdown_heading(trimmed).is_some() {
            break;
        }

        if trimmed.is_empty() {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }

        if is_non_prose_markdown(trimmed) {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }

        paragraph.push(trimmed);
    }

    if paragraph.is_empty() {
        None
    } else {
        Some(paragraph.join(" "))
    }
}

fn is_non_prose_markdown(line: &str) -> bool {
    line.starts_with("- ")
        || line.starts_with("* ")
        || line.starts_with("+ ")
        || line.starts_with('>')
        || line.starts_with('|')
        || line.starts_with("![")
        || line.starts_with("<!--")
        || is_ordered_list_item(line)
}

fn is_ordered_list_item(line: &str) -> bool {
    strip_ordered_list_marker(line).is_some()
}

/// Strip a `1. ` / `42. ` ordered-list marker, returning the item text.
fn strip_ordered_list_marker(line: &str) -> Option<&str> {
    let (marker, item) = line.split_once(". ")?;
    if !marker.is_empty() && marker.bytes().all(|byte| byte.is_ascii_digit()) {
        Some(item.trim_start())
    } else {
        None
    }
}

/// Collect the requirement entries documented under a requirements-style
/// heading ([`is_requirements_heading`]).
///
/// Recognized forms:
/// - unordered list items (`- ` / `* ` / `+ `), with indented
///   wrap-continuation lines joined onto the item;
/// - ordered list items (`1. `), same continuation rule;
/// - two-column markdown tables, emitted as `name: value` per data row.
///
/// Tables with any other column count are skipped whole: without knowing
/// which columns carry the requirement, half-parsing them would produce
/// garbage entries.
fn extract_requirements(lines: &[&str]) -> Vec<String> {
    let mut requirements = Vec::new();
    let mut current_requirement: Option<String> = None;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut requirement_level = None;
    let mut in_code_fence = false;

    for line in lines {
        if is_code_fence(line) {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }
            flush_requirement_table(&mut table_rows, &mut requirements);
            in_code_fence = !in_code_fence;
            continue;
        }

        if in_code_fence {
            continue;
        }

        if let Some((level, heading)) = parse_markdown_heading(line) {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }
            flush_requirement_table(&mut table_rows, &mut requirements);

            let normalized = normalize_heading(&heading);

            if is_requirements_heading(&normalized) {
                requirement_level = Some(level);
            } else if requirement_level.is_some_and(|active| level <= active) {
                requirement_level = None;
            }

            continue;
        }

        if requirement_level.is_none() {
            continue;
        }

        let trimmed = line.trim();

        // A blank line terminates the current list item or table. Whatever
        // follows is a new paragraph (prose, a code fence, ...), not a
        // wrapped continuation of the bullet above it.
        if trimmed.is_empty() {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }
            flush_requirement_table(&mut table_rows, &mut requirements);
            continue;
        }

        if trimmed.starts_with('|') {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }
            table_rows.push(split_table_row(trimmed));
            continue;
        }
        flush_requirement_table(&mut table_rows, &mut requirements);

        let item = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
            .or_else(|| trimmed.strip_prefix("+ "))
            .or_else(|| strip_ordered_list_marker(trimmed));

        if let Some(item) = item {
            if let Some(requirement) = current_requirement.take() {
                requirements.push(requirement);
            }

            let item = strip_inline_markdown(item);
            if !item.is_empty() {
                current_requirement = Some(item);
            }
        } else if let Some(requirement) = &mut current_requirement {
            requirement.push(' ');
            requirement.push_str(&strip_inline_markdown(trimmed));
        }
    }

    if let Some(requirement) = current_requirement {
        requirements.push(requirement);
    }
    flush_requirement_table(&mut table_rows, &mut requirements);

    requirements.dedup();
    requirements
}

/// Split a markdown table row into trimmed cells, honoring `\|` escapes and
/// dropping the empty leading/trailing cells produced by the outer pipes.
fn split_table_row(row: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut characters = row.chars().peekable();

    while let Some(character) = characters.next() {
        match character {
            '\\' if characters.peek() == Some(&'|') => {
                cell.push('|');
                characters.next();
            }
            '|' => {
                cells.push(cell.trim().to_string());
                cell.clear();
            }
            _ => cell.push(character),
        }
    }
    cells.push(cell.trim().to_string());

    if cells.first().is_some_and(|first| first.is_empty()) {
        cells.remove(0);
    }
    if cells.last().is_some_and(|last| last.is_empty()) {
        cells.pop();
    }

    cells
}

/// The `|---|:---:|` row separating a table header from its data rows.
fn is_table_separator_row(cells: &[String]) -> bool {
    !cells.is_empty()
        && cells.iter().all(|cell| {
            !cell.is_empty()
                && cell
                    .chars()
                    .all(|character| character == '-' || character == ':')
        })
}

/// Convert a buffered `| name | value |` requirements table into one
/// `name: value` entry per data row. Tables whose header or data rows are
/// not exactly two columns are dropped whole rather than half-parsed.
fn flush_requirement_table(table_rows: &mut Vec<Vec<String>>, requirements: &mut Vec<String>) {
    let rows = std::mem::take(table_rows);

    // Header, separator, and at least one data row.
    if rows.len() < 3 || !is_table_separator_row(&rows[1]) {
        return;
    }

    if !rows.iter().all(|row| row.len() == 2) {
        return;
    }

    for row in &rows[2..] {
        let name = strip_inline_markdown(&row[0]);
        let value = strip_inline_markdown(&row[1]);
        if !name.is_empty() && !value.is_empty() {
            requirements.push(format!("{name}: {value}"));
        }
    }
}

fn is_requirements_heading(heading: &str) -> bool {
    matches!(
        heading,
        "requirements"
            | "hardware requirements"
            | "software requirements"
            | "system requirements"
            | "toolkit requirements"
            | "build requirements"
            | "prerequisites"
    )
}

fn format_examples_human(examples: &[ExampleInfo]) -> String {
    let mut output = String::new();

    for (index, example) in examples.iter().enumerate() {
        if index != 0 {
            output.push('\n');
        }

        output.push_str(&example.name);
        output.push('\n');

        if example.title != example.name {
            output.push_str("  ");
            output.push_str(&example.title);
            output.push('\n');
        }

        output.push_str("  ");
        output.push_str(&example.description);
        output.push('\n');

        if !example.requirements.is_empty() {
            output.push_str("  Requirements:\n");
            for requirement in &example.requirements {
                output.push_str("    - ");
                output.push_str(requirement);
                output.push('\n');
            }
        }
    }

    output
}

fn format_examples_json(examples: &[ExampleInfo]) -> Result<String, serde_json::Error> {
    let examples = examples
        .iter()
        .map(|example| {
            serde_json::json!({
                "name": example.name,
                "title": example.title,
                "description": example.description,
                "requirements": example.requirements,
            })
        })
        .collect::<Vec<_>>();

    let document = serde_json::json!({
        "schema_version": 1,
        "examples": examples,
    });

    let mut output = serde_json::to_string_pretty(&document)?;
    output.push('\n');
    Ok(output)
}

// =============================================================================
// Run command
// =============================================================================

/// Build and run an example with the custom codegen backend.
///
/// Cleans stale artifacts, sets encoded rustc flags to point at the backend `.so`,
/// and invokes `cargo run --release` from the example directory. Environment
/// variables control output format (PTX / NVVM IR) and verbosity. Trailing
/// `app_args` are forwarded to the example binary after `--`.
#[allow(clippy::too_many_arguments)]
pub fn codegen_run(
    ctx: &Context,
    example: &str,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialize_cubin: bool,
    app_args: &[String],
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    let interop = load_interop_config(&example_dir);

    let output_format = format_label(emit_nvvm_ir);
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, emit_nvvm_ir);
    // Target precedence for `cargo oxide run` (highest first):
    //   1. --arch <sm_XX>            explicit user override   -> CUDA_OXIDE_TARGET
    //   2. CUDA_OXIDE_TARGET=<sm_XX> explicit env override (from the parent)
    //   3. detected GPU arch (via nvidia-smi) -> CUDA_OXIDE_DEVICE_ARCH (a hint)
    //   4. backend feature-based default (`select_target` in mir-importer)
    //
    // Slot 3 is a HINT, not an override: the backend builds for the detected
    // GPU only when that GPU can run the kernel. If the kernel needs a newer
    // arch (tcgen05 needs sm_100a even on a consumer sm_120 GPU), the backend
    // builds for the required arch and the module simply skips at load time.
    // We only detect for `run`, not `build`/`pipeline`: `run` loads the cubin
    // on the local GPU, whereas those may legitimately cross-compile for
    // another machine.
    let detected_device_arch =
        detect_run_target_arch(target_arch, emit_nvvm_ir || materialization.enabled());

    if let Some(interop) = interop.filter(|config| !config.device_crates.is_empty()) {
        codegen_run_interop(
            ctx,
            example,
            &example_dir,
            &interop,
            verbose,
            emit_nvvm_ir,
            target_arch,
            detected_device_arch.as_deref(),
            features,
            bin,
            no_fmad,
            unchecked_indexing,
            &materialization,
            app_args,
        );
        return;
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA: {}", example);
    println!("=========================================");
    println!();
    if materialization.enabled() {
        println!("Output format: materialized cubin");
        println!(
            "Target arch: {}",
            configured_arch_label(ctx, arch)
                .expect("materialization requires a configured architecture")
        );
        println!();
    } else if emit_nvvm_ir {
        println!("Output format: {}", output_format);
        println!(
            "Target arch: {}",
            configured_arch_label(ctx, arch)
                .expect("--emit-nvvm-ir requires a configured architecture")
        );
        println!();
    } else if let Some(dev) = detected_device_arch.as_deref() {
        // Surface the detected GPU so it isn't silent magic. It is a hint, not
        // a hard target: the backend builds for it unless a kernel needs a
        // newer arch (e.g. tcgen05 forces sm_100a even on a consumer sm_120
        // GPU), so the final PTX target may differ.
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
        println!();
    }
    println!("This is the proper cargo workflow:");
    println!("  CARGO_ENCODED_RUSTFLAGS=<cuda-oxide flags> cargo run");
    println!();

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["run", "--release"]).current_dir(&example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }
    if !app_args.is_empty() {
        cmd.arg("--").args(app_args);
    }

    apply_common_codegen_env(
        &mut cmd,
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
    );
    let fingerprint = standard_codegen_fingerprint(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        emit_nvvm_ir,
        target_arch,
        detected_device_arch.as_deref(),
        &materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    );
    apply_output_mode(&mut cmd, emit_nvvm_ir, target_arch, &materialization);
    apply_device_arch_hint(&mut cmd, target_arch, detected_device_arch.as_deref());

    if let Some(bin) = bin {
        println!("Building and running {} (bin: {})...", example, bin);
    } else {
        println!("Building and running {}...", example);
    }
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!("\nFailed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }
}

// =============================================================================
// Sanitize command
// =============================================================================

/// Build an example and run the produced host binary under NVIDIA Compute
/// Sanitizer.
#[allow(clippy::too_many_arguments)]
pub fn codegen_sanitize(
    ctx: &Context,
    example: &str,
    tool: &str,
    sanitizer_args: &[String],
    application_args: &[String],
    verbose: bool,
    arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialize_cubin: bool,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    let interop = load_interop_config(&example_dir);
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, false);
    let detected_device_arch = detect_run_target_arch(target_arch, materialization.enabled());

    if let Some(interop) = interop.filter(|config| !config.device_crates.is_empty()) {
        reject_interop_output_mode(false, &materialization);
        println!("=========================================");
        println!("RUSTC-CODEGEN-CUDA SANITIZE INTEROP: {}", example);
        println!("=========================================");
        if let Some(kind) = &interop.kind {
            println!("Interop kind: {}", kind);
        }
        if let Some(dev) = detected_device_arch.as_deref() {
            println!("Detected GPU arch: {dev} (via nvidia-smi)");
        }
        println!("Compute Sanitizer tool: {tool}");
        println!();

        build_interop_device_crates(
            ctx,
            &example_dir,
            &interop,
            verbose,
            target_arch,
            detected_device_arch.as_deref(),
            InteropDeviceBuildOptions {
                no_fmad,
                unchecked_indexing,
                sanitizer_line_tables: true,
            },
            &materialization,
        );
        let binary = build_host_cargo(ctx, example, &example_dir, features, bin, verbose);
        run_compute_sanitizer(
            ctx,
            &example_dir,
            tool,
            sanitizer_args,
            application_args,
            &binary,
        );
        return;
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA SANITIZE: {}", example);
    println!("=========================================");
    if let Some(dev) = detected_device_arch.as_deref() {
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
    }
    println!("Compute Sanitizer tool: {tool}");
    println!();

    touch_main_rs(&example_dir);
    let binary = codegen_build_host_binary(
        ctx,
        example,
        &example_dir,
        verbose,
        target_arch,
        detected_device_arch.as_deref(),
        features,
        bin,
        no_fmad,
        unchecked_indexing,
        device_debug,
        &materialization,
    );
    run_compute_sanitizer(
        ctx,
        &example_dir,
        tool,
        sanitizer_args,
        application_args,
        &binary,
    );
}

// =============================================================================
// Interop host/device workflow
// =============================================================================

#[derive(Debug, Clone)]
struct InteropConfig {
    kind: Option<String>,
    device_crates: Vec<DeviceCrateConfig>,
}

#[derive(Debug, Clone)]
struct DeviceCrateConfig {
    manifest_path: PathBuf,
    ptx_dir: PathBuf,
    artifact_name: Option<String>,
}

#[derive(Clone, Copy, Debug, Default)]
struct InteropDeviceBuildOptions {
    no_fmad: bool,
    unchecked_indexing: bool,
    sanitizer_line_tables: bool,
}

impl InteropDeviceBuildOptions {
    fn standard(no_fmad: bool, unchecked_indexing: bool) -> Self {
        Self {
            no_fmad,
            unchecked_indexing,
            sanitizer_line_tables: false,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn codegen_run_interop(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    interop: &InteropConfig,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    materialization: &MaterializationMode,
    app_args: &[String],
) {
    reject_interop_output_mode(emit_nvvm_ir, materialization);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA INTEROP: {}", example);
    println!("=========================================");
    if let Some(kind) = &interop.kind {
        println!("Interop kind: {}", kind);
    }
    if let Some(dev) = detected_device_arch {
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
    }
    println!();

    build_interop_device_crates(
        ctx,
        example_dir,
        interop,
        verbose,
        arch,
        detected_device_arch,
        InteropDeviceBuildOptions::standard(no_fmad, unchecked_indexing),
        materialization,
    );
    run_host_cargo(
        ctx,
        example,
        example_dir,
        "run",
        features,
        bin,
        verbose,
        app_args,
    );
}

#[allow(clippy::too_many_arguments)]
fn codegen_build_interop(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    interop: &InteropConfig,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    materialization: &MaterializationMode,
) {
    reject_interop_output_mode(emit_nvvm_ir, materialization);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA INTEROP BUILD: {}", example);
    println!("=========================================");
    if let Some(kind) = &interop.kind {
        println!("Interop kind: {}", kind);
    }
    println!();

    // `build` may cross-compile for another machine, so no device-arch hint:
    // only an explicit `--arch` pins the target here.
    build_interop_device_crates(
        ctx,
        example_dir,
        interop,
        verbose,
        arch,
        None,
        InteropDeviceBuildOptions::standard(no_fmad, unchecked_indexing),
        materialization,
    );
    run_host_cargo(
        ctx,
        example,
        example_dir,
        "build",
        features,
        None,
        verbose,
        &[],
    );
}

fn reject_interop_output_mode(emit_nvvm_ir: bool, materialization: &MaterializationMode) {
    if materialization.enabled() {
        eprintln!("Error: --materialize-cubin is not supported for metadata interop examples yet.");
        eprintln!("Interop host crates currently consume PTX files from nested device crates.");
        std::process::exit(2);
    }
    if emit_nvvm_ir {
        eprintln!("Error: --emit-nvvm-ir is not supported for metadata interop examples yet.");
        eprintln!("Interop host crates embed PTX artifacts produced by nested device crates.");
        std::process::exit(2);
    }
}

#[allow(clippy::too_many_arguments)]
fn build_interop_device_crates(
    ctx: &Context,
    example_dir: &Path,
    interop: &InteropConfig,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    options: InteropDeviceBuildOptions,
    materialization: &MaterializationMode,
) {
    for device_crate in &interop.device_crates {
        build_interop_device_crate(
            ctx,
            example_dir,
            device_crate,
            verbose,
            arch,
            detected_device_arch,
            options,
            materialization,
        );
    }
}

fn interop_device_artifact_name(manifest_path: &Path, device_crate: &DeviceCrateConfig) -> String {
    device_crate
        .artifact_name
        .clone()
        .unwrap_or_else(|| normalize_crate_name(&package_name_from_manifest(manifest_path)))
}

fn interop_device_ptx_path(
    example_dir: &Path,
    device_crate: &DeviceCrateConfig,
    artifact_name: &str,
) -> PathBuf {
    example_dir
        .join(&device_crate.ptx_dir)
        .join(format!("{}.ptx", artifact_stem(artifact_name)))
}

#[allow(clippy::too_many_arguments)]
fn build_interop_device_crate(
    ctx: &Context,
    example_dir: &Path,
    device_crate: &DeviceCrateConfig,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    options: InteropDeviceBuildOptions,
    materialization: &MaterializationMode,
) {
    let manifest_path = example_dir.join(&device_crate.manifest_path);
    let manifest_path = manifest_path.canonicalize().unwrap_or_else(|e| {
        eprintln!(
            "Error: could not resolve device crate manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });
    let device_dir = manifest_path.parent().unwrap_or(example_dir);
    let ptx_dir = example_dir.join(&device_crate.ptx_dir);
    std::fs::create_dir_all(&ptx_dir).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not create device artifact directory {}: {}",
            ptx_dir.display(),
            e
        );
        std::process::exit(1);
    });

    let artifact_name = interop_device_artifact_name(&manifest_path, device_crate);
    clean_generated_files(&ptx_dir, &artifact_name);
    touch_main_rs(device_dir);

    println!("Building device crate {}...", manifest_path.display());

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release", "--manifest-path"])
        .arg(&manifest_path)
        .current_dir(device_dir);

    apply_interop_device_codegen_options(&mut cmd, ctx, verbose, options);
    let fingerprint = interop_codegen_fingerprint(
        ctx,
        verbose,
        options.no_fmad,
        options.unchecked_indexing,
        DeviceDebug::Off,
        arch,
        detected_device_arch,
        &ptx_dir,
        options.sanitizer_line_tables,
        materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    );
    // This is an internal artifact contract, so it must override a project
    // `[env]` default for the same variable.
    cmd.env("CUDA_OXIDE_PTX_DIR", &ptx_dir);
    apply_output_mode(&mut cmd, false, arch, materialization);
    apply_device_arch_hint(&mut cmd, arch, detected_device_arch);

    let status = cmd.status().expect("Failed to build interop device crate");
    if !status.success() {
        eprintln!(
            "\nDevice crate build failed with exit code: {:?}",
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    let ptx_path = interop_device_ptx_path(example_dir, device_crate, &artifact_name);
    if !ptx_path.exists() {
        eprintln!(
            "Error: device crate build succeeded but did not produce {}",
            ptx_path.display()
        );
        std::process::exit(1);
    }
    println!("PTX written: {}", ptx_path.display());
}

#[allow(clippy::too_many_arguments)]
fn run_host_cargo(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    cargo_subcommand: &str,
    features: Option<&str>,
    bin: Option<&str>,
    verbose: bool,
    app_args: &[String],
) {
    let mut cmd = Command::new("cargo");
    cmd.arg(cargo_subcommand)
        .arg("--release")
        .current_dir(example_dir);

    if cargo_subcommand == "run"
        && let Some(bin) = bin
    {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }
    if cargo_subcommand == "run" && !app_args.is_empty() {
        cmd.arg("--").args(app_args);
    }

    apply_config_env(&mut cmd, ctx);
    apply_ld_library_path(&mut cmd, ctx);

    if cargo_subcommand == "run" {
        if let Some(bin) = bin {
            println!("Building and running {} (bin: {})...", example, bin);
        } else {
            println!("Building and running {}...", example);
        }
    } else {
        println!("Building host crate {}...", example);
    }
    println!();

    if verbose {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    }

    let status = cmd.status().expect("Failed to run host cargo command");
    if !status.success() {
        eprintln!(
            "\nHost cargo command failed with exit code: {:?}",
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }
}

#[allow(clippy::too_many_arguments)]
fn codegen_build_host_binary(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    verbose: bool,
    arch: Option<&str>,
    detected_device_arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialization: &MaterializationMode,
) -> PathBuf {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_common_codegen_env(
        &mut cmd,
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
    );
    apply_default_sanitizer_line_tables(&mut cmd, ctx, device_debug);
    let fingerprint = sanitize_codegen_fingerprint(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        arch,
        detected_device_arch,
        None,
        materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    );
    apply_output_mode(&mut cmd, false, arch, materialization);
    apply_device_arch_hint(&mut cmd, arch, detected_device_arch);

    if let Some(bin) = bin {
        println!("Building {} (bin: {})...", example, bin);
    } else {
        println!("Building {}...", example);
    }
    println!();

    run_cargo_build_for_executable(&mut cmd, example_dir, bin).unwrap_or_else(|message| {
        eprintln!("\nBuild failed: {message}");
        std::process::exit(1);
    })
}

fn build_host_cargo(
    ctx: &Context,
    example: &str,
    example_dir: &Path,
    features: Option<&str>,
    bin: Option<&str>,
    verbose: bool,
) -> PathBuf {
    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_config_env(&mut cmd, ctx);
    apply_ld_library_path(&mut cmd, ctx);

    if let Some(bin) = bin {
        println!("Building host crate {} (bin: {})...", example, bin);
    } else {
        println!("Building host crate {}...", example);
    }
    println!();

    if verbose {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    }

    run_cargo_build_for_executable(&mut cmd, example_dir, bin).unwrap_or_else(|message| {
        eprintln!("\nHost cargo build failed: {message}");
        std::process::exit(1);
    })
}

fn run_cargo_build_for_executable(
    cmd: &mut Command,
    manifest_dir: &Path,
    explicit_bin: Option<&str>,
) -> Result<PathBuf, String> {
    let selection = cargo_executable_selection(manifest_dir, explicit_bin)?;

    cmd.arg("--message-format=json-render-diagnostics");
    let output = cmd
        .output()
        .map_err(|error| format!("could not start Cargo: {error}"))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }

    let mut executables = Vec::<CargoExecutableArtifact>::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let message: serde_json::Value = match serde_json::from_str(line) {
            Ok(message) => message,
            Err(_) => {
                if !line.is_empty() {
                    println!("{line}");
                }
                continue;
            }
        };

        if let Some(rendered) = message
            .get("message")
            .and_then(|message| message.get("rendered"))
            .and_then(|rendered| rendered.as_str())
        {
            eprint!("{rendered}");
        }

        if message.get("reason").and_then(|reason| reason.as_str()) != Some("compiler-artifact") {
            continue;
        }
        let is_binary = message
            .get("target")
            .and_then(|target| target.get("kind"))
            .and_then(|kind| kind.as_array())
            .is_some_and(|kinds| kinds.iter().any(|kind| kind.as_str() == Some("bin")));
        if !is_binary {
            continue;
        }
        let Some(path) = message.get("executable").and_then(|path| path.as_str()) else {
            continue;
        };
        let Some(package_id) = message
            .get("package_id")
            .and_then(|package_id| package_id.as_str())
        else {
            continue;
        };
        let Some(name) = message
            .get("target")
            .and_then(|target| target.get("name"))
            .and_then(|name| name.as_str())
        else {
            continue;
        };
        executables.push(CargoExecutableArtifact {
            package_id: package_id.to_string(),
            target_name: name.to_string(),
            path: PathBuf::from(path),
        });
    }

    if !output.status.success() {
        return Err(format!("Cargo exited with status {}", output.status));
    }

    select_cargo_executable_artifact(&selection, &executables)
}

#[derive(Debug, PartialEq, Eq)]
struct CargoExecutableSelection {
    packages: Vec<CargoSelectedPackage>,
    explicit_bin: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
struct CargoSelectedPackage {
    package_id: String,
    package_name: String,
    default_run: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
struct CargoExecutableArtifact {
    package_id: String,
    target_name: String,
    path: PathBuf,
}

fn cargo_executable_selection(
    manifest_dir: &Path,
    explicit_bin: Option<&str>,
) -> Result<CargoExecutableSelection, String> {
    let metadata = cargo_metadata(manifest_dir)?;
    let manifest_path = manifest_dir.join("Cargo.toml");
    let manifest_path = manifest_path
        .canonicalize()
        .map_err(|error| format!("could not resolve {}: {error}", manifest_path.display()))?;

    let packages = metadata
        .get("packages")
        .and_then(|packages| packages.as_array())
        .ok_or_else(|| "Cargo metadata did not include packages".to_string())?;

    let selected_packages = cargo_selected_packages(&metadata, packages, &manifest_path)?;
    let packages = selected_packages
        .into_iter()
        .map(cargo_selected_package)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CargoExecutableSelection {
        packages,
        explicit_bin: explicit_bin.map(str::to_owned),
    })
}

/// Return the packages Cargo selects for a command launched from
/// `manifest_path`.
///
/// At a workspace root, Cargo uses `workspace.default-members` even when the
/// root manifest also contains a `[package]`. Inside a member directory, Cargo
/// instead selects that member. `cargo metadata` has already resolved the
/// workspace defaults for us, so mirror that distinction here.
fn cargo_selected_packages<'a>(
    metadata: &serde_json::Value,
    packages: &'a [serde_json::Value],
    manifest_path: &Path,
) -> Result<Vec<&'a serde_json::Value>, String> {
    let workspace_root = metadata
        .get("workspace_root")
        .and_then(|path| path.as_str())
        .ok_or_else(|| "Cargo metadata did not include workspace_root".to_string())?;
    let workspace_manifest = PathBuf::from(workspace_root).join("Cargo.toml");
    let workspace_manifest = workspace_manifest.canonicalize().map_err(|error| {
        format!(
            "could not resolve workspace manifest {}: {error}",
            workspace_manifest.display()
        )
    })?;

    if manifest_path != workspace_manifest {
        let package = packages
            .iter()
            .find(|package| cargo_package_manifest_matches(package, manifest_path))
            .ok_or_else(|| {
                format!(
                    "could not determine the Cargo package for {}",
                    manifest_path.display()
                )
            })?;
        return Ok(vec![package]);
    }

    let default_members = metadata
        .get("workspace_default_members")
        .and_then(|members| members.as_array())
        .ok_or_else(|| "Cargo metadata did not include workspace_default_members".to_string())?;
    if default_members.is_empty() {
        return Err("Cargo selected no workspace default members".to_string());
    }

    default_members
        .iter()
        .map(|member| {
            let package_id = member.as_str().ok_or_else(|| {
                "Cargo metadata contained a non-string workspace default member".to_string()
            })?;
            packages
                .iter()
                .find(|package| cargo_package_id(package).ok() == Some(package_id))
                .ok_or_else(|| {
                    format!(
                        "Cargo workspace default member `{package_id}` was missing from metadata packages"
                    )
                })
        })
        .collect()
}

fn cargo_package_manifest_matches(package: &serde_json::Value, manifest_path: &Path) -> bool {
    package
        .get("manifest_path")
        .and_then(|path| path.as_str())
        .and_then(|path| PathBuf::from(path).canonicalize().ok())
        .is_some_and(|path| path == manifest_path)
}

fn cargo_metadata(manifest_dir: &Path) -> Result<serde_json::Value, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version=1", "--no-deps"])
        .current_dir(manifest_dir)
        .output()
        .map_err(|error| format!("could not start cargo metadata: {error}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "cargo metadata failed with status {}{}{}",
            output.status,
            if stderr.is_empty() { "" } else { ": " },
            stderr.trim()
        ));
    }

    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("could not parse cargo metadata JSON: {error}"))
}

fn cargo_package_id(package: &serde_json::Value) -> Result<&str, String> {
    package
        .get("id")
        .and_then(|id| id.as_str())
        .ok_or_else(|| "Cargo metadata package is missing id".to_string())
}

fn cargo_package_name(package: &serde_json::Value) -> Result<&str, String> {
    package
        .get("name")
        .and_then(|name| name.as_str())
        .ok_or_else(|| "Cargo metadata package is missing name".to_string())
}

fn cargo_selected_package(package: &serde_json::Value) -> Result<CargoSelectedPackage, String> {
    Ok(CargoSelectedPackage {
        package_id: cargo_package_id(package)?.to_string(),
        package_name: cargo_package_name(package)?.to_string(),
        default_run: package
            .get("default_run")
            .and_then(|name| name.as_str())
            .map(str::to_owned),
    })
}

fn select_cargo_executable_artifact(
    selection: &CargoExecutableSelection,
    executables: &[CargoExecutableArtifact],
) -> Result<PathBuf, String> {
    if let Some(explicit_bin) = selection.explicit_bin.as_deref() {
        let matches = selection
            .packages
            .iter()
            .flat_map(|package| {
                executables
                    .iter()
                    .filter(move |artifact| {
                        artifact.package_id == package.package_id
                            && artifact.target_name == explicit_bin
                    })
                    .map(move |artifact| (package, artifact))
            })
            .collect::<Vec<_>>();
        return match matches.as_slice() {
            [(_, artifact)] => Ok(artifact.path.clone()),
            [] => Err(format!(
                "Cargo produced no executable artifact for target `{explicit_bin}` in selected packages {}",
                selected_package_names(selection)
            )),
            matches => Err(format!(
                "Cargo produced executable target `{explicit_bin}` for multiple selected packages: {}; run from a package directory",
                matches
                    .iter()
                    .map(|(package, _)| package.package_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        };
    }

    let mut candidates = Vec::new();
    for package in &selection.packages {
        let artifacts = executables
            .iter()
            .filter(|artifact| artifact.package_id == package.package_id)
            .collect::<Vec<_>>();

        if let Some(default_run) = package.default_run.as_deref() {
            let matches = artifacts
                .iter()
                .copied()
                .filter(|artifact| artifact.target_name == default_run)
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [artifact] => candidates.push((package, *artifact)),
                [] => {
                    return Err(format!(
                        "Cargo produced no executable artifact for package `{}` default-run target `{default_run}`",
                        package.package_name
                    ));
                }
                _ => {
                    return Err(format!(
                        "Cargo produced multiple executable artifacts for package `{}` default-run `{default_run}`",
                        package.package_name
                    ));
                }
            }
            continue;
        }

        // A selected package without an emitted binary may simply be a
        // library-only workspace member. A package with `default-run` is
        // handled above: silently skipping its missing target could launch a
        // different default member's program instead.
        if artifacts.is_empty() {
            continue;
        }

        match artifacts.as_slice() {
            [artifact] => candidates.push((package, *artifact)),
            artifacts => {
                let choices = artifacts
                    .iter()
                    .map(|artifact| artifact.target_name.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(format!(
                    "Cargo produced multiple executable targets for package `{}`: {choices}; pass --bin <name>",
                    package.package_name
                ));
            }
        }
    }

    match candidates.as_slice() {
        [(_, artifact)] => Ok(artifact.path.clone()),
        [] => Err(format!(
            "Cargo produced no executable artifact for selected packages {}",
            selected_package_names(selection)
        )),
        candidates => Err(format!(
            "Cargo produced executables for multiple selected packages: {}; pass --bin <name> that is unique among them",
            candidates
                .iter()
                .map(|(package, _)| package.package_name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

fn selected_package_names(selection: &CargoExecutableSelection) -> String {
    selection
        .packages
        .iter()
        .map(|package| package.package_name.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

const DEFAULT_SANITIZER_ERROR_EXITCODE: &str = "86";

#[derive(Debug, PartialEq, Eq)]
struct SanitizerInvocationArgs {
    args: Vec<String>,
    uses_default_error_exitcode: bool,
    status_checks_weakened: bool,
}

fn sanitizer_invocation_args(sanitizer_args: &[String]) -> SanitizerInvocationArgs {
    let has_explicit_error_exitcode = sanitizer_args
        .iter()
        .any(|arg| arg == "--error-exitcode" || arg.starts_with("--error-exitcode="));
    if has_explicit_error_exitcode {
        return SanitizerInvocationArgs {
            args: sanitizer_args.to_vec(),
            uses_default_error_exitcode: false,
            status_checks_weakened: sanitizer_option_is_no(sanitizer_args, "check-exit-code")
                || sanitizer_option_is_no(sanitizer_args, "require-cuda-init"),
        };
    }

    let mut args = Vec::with_capacity(sanitizer_args.len() + 2);
    args.extend([
        "--error-exitcode".to_string(),
        DEFAULT_SANITIZER_ERROR_EXITCODE.to_string(),
    ]);
    args.extend_from_slice(sanitizer_args);
    SanitizerInvocationArgs {
        args,
        uses_default_error_exitcode: true,
        status_checks_weakened: sanitizer_option_is_no(sanitizer_args, "check-exit-code")
            || sanitizer_option_is_no(sanitizer_args, "require-cuda-init"),
    }
}

fn sanitizer_option_is_no(args: &[String], name: &str) -> bool {
    let option = format!("--{name}");
    let equals_prefix = format!("{option}=");
    args.iter().enumerate().any(|(index, arg)| {
        arg.strip_prefix(&equals_prefix)
            .is_some_and(|value| value.eq_ignore_ascii_case("no"))
            || (arg == &option
                && args
                    .get(index + 1)
                    .is_some_and(|value| value.eq_ignore_ascii_case("no")))
    })
}

/// Fallback locations probed for `compute-sanitizer` when it is neither on
/// PATH nor under the configured CUDA toolkit root. Shared by `sanitize`
/// (`run_compute_sanitizer`) and `doctor` so both use the same discovery
/// order by construction.
const COMPUTE_SANITIZER_FALLBACK_PATHS: &[&str] = &[
    "/usr/local/cuda/bin/compute-sanitizer",
    "/opt/cuda/bin/compute-sanitizer",
    "/usr/bin/compute-sanitizer",
];

fn run_compute_sanitizer(
    ctx: &Context,
    example_dir: &Path,
    tool: &str,
    sanitizer_args: &[String],
    application_args: &[String],
    binary: &Path,
) {
    let compute_sanitizer = find_cuda_toolkit_executable(
        ctx,
        "compute-sanitizer",
        COMPUTE_SANITIZER_FALLBACK_PATHS,
    )
    .unwrap_or_else(|| {
        eprintln!("Error: compute-sanitizer not found.");
        eprintln!(
            "It is installed with the CUDA Toolkit; run `cargo oxide doctor` to check CUDA setup."
        );
        std::process::exit(1);
    });

    let invocation_args = sanitizer_invocation_args(sanitizer_args);
    let mut cmd = Command::new(compute_sanitizer);
    cmd.args(["--tool", tool])
        .args(&invocation_args.args)
        .arg(binary)
        .args(application_args)
        .current_dir(example_dir);
    apply_config_env(&mut cmd, ctx);
    apply_ld_library_path(&mut cmd, ctx);

    let forwarded_args = if invocation_args.args.is_empty() {
        String::new()
    } else {
        format!(" {}", invocation_args.args.join(" "))
    };
    let displayed_application_args = if application_args.is_empty() {
        String::new()
    } else {
        format!(" {}", application_args.join(" "))
    };
    println!(
        "Running compute-sanitizer --tool {tool}{forwarded_args} {}{displayed_application_args}...",
        binary.display()
    );
    println!();

    let status = cmd.status().expect("Failed to run compute-sanitizer");
    if !status.success() {
        eprintln!(
            "\nCompute Sanitizer failed with exit code: {:?}",
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    println!();
    println!("Compute Sanitizer completed with exit code 0.");
    if !invocation_args.uses_default_error_exitcode {
        println!(
            "An explicit --error-exitcode was supplied, so it controls whether findings fail the command."
        );
    }
    if invocation_args.status_checks_weakened {
        println!(
            "The supplied sanitizer options can allow target or CUDA-initialization failures to exit 0."
        );
    }
    println!(
        "Inspect the sanitizer report above; exit status alone is not a clean-report assertion."
    );
}

// =============================================================================
// Build command (compile only, don't run)
// =============================================================================

/// Compile an example without running it.
///
/// Same as [`codegen_run`] but uses `cargo build --release` instead of
/// `cargo run`. Useful for cross-compilation or when the target hardware
/// (e.g., Blackwell tensor cores) isn't available on the build machine.
#[allow(clippy::too_many_arguments)]
pub fn codegen_build(
    ctx: &Context,
    example: &str,
    verbose: bool,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    features: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialize_cubin: bool,
) {
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, emit_nvvm_ir);
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    if let Some(interop) =
        load_interop_config(&example_dir).filter(|config| !config.device_crates.is_empty())
    {
        codegen_build_interop(
            ctx,
            example,
            &example_dir,
            &interop,
            verbose,
            emit_nvvm_ir,
            target_arch,
            features,
            no_fmad,
            unchecked_indexing,
            &materialization,
        );
        return;
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA BUILD: {}", example);
    println!("=========================================");
    println!();

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&example_dir);

    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_common_codegen_env(
        &mut cmd,
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
    );
    let fingerprint = standard_codegen_fingerprint(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        emit_nvvm_ir,
        target_arch,
        None,
        &materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    );
    apply_output_mode(&mut cmd, emit_nvvm_ir, target_arch, &materialization);

    println!("Building {}...", example);
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!("\nBuild failed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }
}

// =============================================================================
// Inspect command
// =============================================================================

/// Build an example as PTX and print the generated artifact.
#[allow(clippy::too_many_arguments)]
pub fn codegen_inspect_ptx(
    ctx: &Context,
    example: &str,
    arch: Option<&str>,
    features: Option<&str>,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
) {
    let materialization_enabled = materialization_requested(ctx, false).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(2);
    });

    if materialization_enabled {
        eprintln!("Error: inspect requires PTX output, but {MATERIALIZE_ENV} is enabled");
        std::process::exit(2);
    }

    let nvvm_ir_enabled = nvvm_ir_requested(ctx).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(2);
    });

    if nvvm_ir_enabled {
        eprintln!("Error: inspect requires PTX output, but CUDA_OXIDE_EMIT_NVVM_IR is enabled");
        std::process::exit(2);
    }

    codegen_build(
        ctx,
        example,
        verbose,
        false,
        arch,
        features,
        no_fmad,
        unchecked_indexing,
        device_debug,
        false,
    );

    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    for path in ptx_artifact_paths(&example_dir, example) {
        print_ptx_artifact(&path).unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        });
    }
}

// =============================================================================
// emit-ltoir command
// =============================================================================

/// Compile a crate's device code to a binary LTOIR artifact in one step.
///
/// `cargo oxide build --emit-nvvm-ir` produces NVVM IR, which a consumer then
/// has to run through libNVVM separately to get linkable LTOIR. This folds both
/// halves into one command for the Tile-to-SIMT interop workflow (#96): it
/// builds the crate in NVVM IR mode, then compiles the emitted `<crate>.ll`
/// with libNVVM `-gen-lto` and writes `<crate>.ltoir` (or `output`) plus the
/// matching `.target` and `.options` files used for runtime loading and final
/// nvJitLink policy.
///
/// `arch` is required because LTOIR is architecture-specific. It accepts
/// `sm_XX`, `compute_XX`, or a bare `XX`, all mapped to libNVVM's
/// `-arch=compute_XX`.
#[allow(clippy::too_many_arguments)]
pub fn emit_ltoir(
    ctx: &Context,
    example: &str,
    arch: &str,
    features: Option<&str>,
    output: Option<&Path>,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    if load_interop_config(&example_dir).is_some_and(|config| !config.device_crates.is_empty()) {
        eprintln!("Error: emit-ltoir does not support metadata interop examples.");
        eprintln!("Point it at a single SIMT device crate instead.");
        std::process::exit(1);
    }

    // Normalize once: libNVVM consumes compute_XX, while the compiler records
    // and nvJitLink consumes the equivalent sm_XX spelling.
    let parsed_arch = parse_nvvm_arch(arch).unwrap_or_else(|error| {
        eprintln!("Error: {error}");
        std::process::exit(1);
    });
    let sm_arch = parsed_arch.sm();

    // Step 1: build in NVVM IR mode so the backend writes `<crate>.ll` as
    // libNVVM-ready NVVM IR. codegen_build exits on build failure. Pass
    // quiet=true so the intermediate "✓ Build succeeded" line is suppressed;
    // emit_ltoir prints its own unified summary at the end.
    codegen_build(
        ctx,
        example,
        verbose,
        true,
        Some(&sm_arch),
        features,
        no_fmad,
        unchecked_indexing,
        device_debug,
        false,
    );

    // Step 2: compile that NVVM IR to LTOIR via libNVVM -gen-lto.
    let ll_path = emitted_ll_path(&example_dir, example);
    let ir = std::fs::read(&ll_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read emitted NVVM IR at {}: {e}",
            ll_path.display()
        );
        std::process::exit(1);
    });
    let source_options_path = ll_path.with_extension("options");
    let source_options = std::fs::read_to_string(&source_options_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read emitted compile options at {}: {e}",
            source_options_path.display()
        );
        std::process::exit(1);
    });
    let compile_options = oxide_artifacts::ArtifactCompileOptions::from_sidecar_text(
        &source_options,
    )
    .unwrap_or_else(|e| {
        eprintln!(
            "Error: invalid emitted compile options at {}: {e}",
            source_options_path.display()
        );
        std::process::exit(1);
    });

    let compute_arch = parsed_arch.compute();
    let ltoir = compile_nvvm_to_ltoir(&ir, example, &parsed_arch, compile_options);

    // Step 3: write the artifact.
    let out_path = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| default_ltoir_path(&example_dir, example));
    for metadata_path in [
        out_path.with_extension("target"),
        out_path.with_extension("options"),
    ] {
        match std::fs::remove_file(&metadata_path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                eprintln!(
                    "Error: could not clear stale LTOIR metadata {}: {error}",
                    metadata_path.display()
                );
                std::process::exit(1);
            }
        }
    }
    std::fs::write(&out_path, &ltoir).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not write LTOIR to {}: {e}",
            out_path.display()
        );
        std::process::exit(1);
    });
    let options_path = out_path.with_extension("options");
    std::fs::write(&options_path, compile_options.sidecar_text()).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not write LTOIR compile options to {}: {e}",
            options_path.display()
        );
        std::process::exit(1);
    });
    let target_path = out_path.with_extension("target");
    std::fs::write(
        &target_path,
        format!(
            "{sm_arch}\n{}\n",
            oxide_artifacts::COMPILE_OPTIONS_TARGET_MARKER
        ),
    )
    .unwrap_or_else(|e| {
        eprintln!(
            "Error: could not write LTOIR target metadata to {}: {e}",
            target_path.display()
        );
        std::process::exit(1);
    });

    println!();
    println!(
        "✓ LTOIR written to {} ({} bytes, {compute_arch})",
        out_path.display(),
        ltoir.len()
    );
}

/// Normalize a target architecture to libNVVM's `compute_XX` form.
///
/// Accepts `sm_XX` (the form `--arch` and the rest of cargo-oxide use),
/// `compute_XX` (passed through), or a bare `XX`.
fn parse_nvvm_arch(
    arch: &str,
) -> Result<cuda_artifact_finalizer::CudaArch, cuda_artifact_finalizer::CudaArchParseError> {
    let normalized = if arch.starts_with("sm_") || arch.starts_with("compute_") {
        arch.to_string()
    } else {
        format!("compute_{arch}")
    };
    normalized.parse()
}

/// Compile NVVM IR text to binary LTOIR with libNVVM `-gen-lto`. Exits with a
/// diagnostic on any libNVVM failure (the program log is attached to the error).
///
fn compile_nvvm_to_ltoir(
    ir: &[u8],
    name: &str,
    arch: &cuda_artifact_finalizer::CudaArch,
    compile_options: oxide_artifacts::ArtifactCompileOptions,
) -> Vec<u8> {
    let compiler = cuda_artifact_finalizer::NvvmCompiler::discover().unwrap_or_else(|e| {
        eprintln!("Error: could not initialize the CUDA artifact compiler: {e}");
        eprintln!("libNVVM ships with the CUDA Toolkit at <CUDA>/nvvm/lib64/libnvvm.so.");
        eprintln!("Run `cargo oxide doctor` to check your toolkit setup.");
        std::process::exit(1);
    });
    let options = finalization_options_from_artifact(arch, compile_options);
    compiler
        .compile_nvvm_ir_to_ltoir(name, ir, &options)
        .unwrap_or_else(|e| {
            eprintln!("Error: libNVVM -gen-lto compilation failed: {e}");
            std::process::exit(1);
        })
}

fn finalization_options_from_artifact(
    arch: &cuda_artifact_finalizer::CudaArch,
    compile_options: oxide_artifacts::ArtifactCompileOptions,
) -> cuda_artifact_finalizer::FinalizationOptions {
    let debug = match compile_options.debug_policy() {
        oxide_artifacts::ArtifactDebugPolicy::None => cuda_artifact_finalizer::DebugPolicy::None,
        oxide_artifacts::ArtifactDebugPolicy::LineTables => {
            cuda_artifact_finalizer::DebugPolicy::LineTables
        }
        oxide_artifacts::ArtifactDebugPolicy::Full => cuda_artifact_finalizer::DebugPolicy::Full,
    };
    cuda_artifact_finalizer::FinalizationOptions::new(arch.clone())
        .with_fma_contraction(compile_options.fma_contraction_enabled())
        .with_debug_policy(debug)
}

/// Device debug-information policy requested on the command line.
///
/// Mirrors nvcc: `--lineinfo` is `-lineinfo` (line tables, optimization intact)
/// and `--device-debug` is `-G` (full debug, libNVVM optimization disabled).
/// The two are ordered, not exclusive: asking for both yields [`Self::Full`],
/// because full debug already carries line tables.
///
/// This is the CLI surface for a policy that already exists end to end --
/// `CUDA_OXIDE_DEBUG`, `ArtifactCompileOptions`'s debug bits, and
/// `FinalizationOptions::with_debug_policy` all predate it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceDebug {
    /// Request no device debug information (the default).
    #[default]
    Off,
    /// Preserve source line mappings without disabling optimization.
    LineTables,
    /// Emit full debug information; libNVVM finalization runs unoptimized.
    Full,
}

impl DeviceDebug {
    /// Resolve the two independent CLI booleans into one ordered policy.
    #[must_use]
    pub fn from_flags(lineinfo: bool, device_debug: bool) -> Self {
        match (device_debug, lineinfo) {
            (true, _) => Self::Full,
            (false, true) => Self::LineTables,
            (false, false) => Self::Off,
        }
    }

    /// Value for `CUDA_OXIDE_DEBUG`, or `None` when nothing must be exported.
    ///
    /// `Off` deliberately returns `None` rather than `"off"`: exporting `off`
    /// would override a debug level the surrounding environment had already
    /// asked for, turning an absent flag into an active opt-out.
    #[must_use]
    pub fn env_value(self) -> Option<&'static str> {
        match self {
            Self::Off => None,
            Self::LineTables => Some("line"),
            Self::Full => Some("full"),
        }
    }
}

/// Options for `cargo oxide build -- ...` / `cargo oxide test -- ...`.
#[derive(Clone, Copy)]
pub struct CargoPassthroughOptions<'a> {
    pub verbose: bool,
    pub emit_nvvm_ir: bool,
    pub arch: Option<&'a str>,
    pub features: Option<&'a str>,
    pub cargo_target_dir: Option<&'a Path>,
    pub device_codegen_crate: Option<&'a str>,
    pub device_cfgs: &'a [String],
    pub no_fmad: bool,
    pub unchecked_indexing: bool,
    pub materialize_cubin: bool,
    pub device_debug: DeviceDebug,
}

/// Cargo operations supported by the passthrough path.
///
/// The subcommand determines who owns profile-related rustc flags: regular
/// builds retain cuda-oxide's release-like defaults, while tests leave the
/// selected Cargo profile intact (including `--release` and `--profile`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CargoPassthroughSubcommand {
    Build,
    Test,
}

impl CargoPassthroughSubcommand {
    fn as_str(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Test => "test",
        }
    }

    fn codegen_profile(self) -> CodegenProfilePolicy {
        match self {
            Self::Build => CodegenProfilePolicy::ReleaseLike,
            Self::Test => CodegenProfilePolicy::CargoSelected,
        }
    }
}

fn normalize_device_codegen_crates(raw: &str) -> Result<String, String> {
    let mut normalized = Vec::new();
    for item in raw.split(',') {
        let name = item.trim().replace('-', "_");
        if name.is_empty() {
            return Err(
                "--device-codegen-crate requires a comma-separated list without empty entries"
                    .to_string(),
            );
        }
        if !name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
        {
            return Err(format!(
                "invalid device-codegen crate name `{item}`; use Cargo crate names separated by commas"
            ));
        }
        if !normalized.contains(&name) {
            normalized.push(name);
        }
    }
    Ok(normalized.join(","))
}

fn project_config_env<'a>(ctx: &'a Context, key: &str) -> Option<&'a str> {
    ctx.config
        .env
        .iter()
        .find(|(configured_key, _)| configured_key == key)
        .map(|(_, value)| value.as_str())
}

fn configured_device_codegen_crates(
    ctx: &Context,
    explicit: Option<&str>,
) -> Result<Option<String>, String> {
    let inherited = std::env::var(DEVICE_CODEGEN_CRATE_ENV).ok();
    resolve_device_codegen_crates(
        explicit,
        inherited.as_deref(),
        project_config_env(ctx, DEVICE_CODEGEN_CRATE_ENV),
    )
}

fn resolve_device_codegen_crates(
    explicit: Option<&str>,
    inherited: Option<&str>,
    configured: Option<&str>,
) -> Result<Option<String>, String> {
    if let Some(explicit) = explicit {
        return normalize_device_codegen_crates(explicit).map(Some);
    }

    inherited
        .or(configured)
        .filter(|value| !value.trim().is_empty())
        .map(normalize_device_codegen_crates)
        .transpose()
}

/// The ambient environment, in the shape the fingerprint helpers consume.
///
/// Split out so both fingerprint wrappers share one collection, and so their
/// `_with_env` counterparts stay the only entry points a unit test needs.
fn inherited_process_env() -> BTreeMap<String, Vec<u8>> {
    std::env::vars_os()
        .filter_map(|(key, value)| {
            key.into_string()
                .ok()
                .map(|key| (key, value.as_encoded_bytes().to_vec()))
        })
        .collect()
}

fn passthrough_codegen_fingerprint(
    ctx: &Context,
    opts: &CargoPassthroughOptions<'_>,
    owner_filter: Option<&str>,
    target_arch: Option<&str>,
    materialization: &MaterializationMode,
) -> String {
    passthrough_codegen_fingerprint_with_env(
        ctx,
        opts,
        owner_filter,
        target_arch,
        materialization,
        &inherited_process_env(),
    )
}

fn passthrough_codegen_fingerprint_with_env(
    ctx: &Context,
    opts: &CargoPassthroughOptions<'_>,
    owner_filter: Option<&str>,
    target_arch: Option<&str>,
    materialization: &MaterializationMode,
    inherited_env: &BTreeMap<String, Vec<u8>>,
) -> String {
    let mut effective_env = BTreeMap::new();

    // Project-configured CUDA_OXIDE_* variables are defaults. Mirror the same
    // parent override rule as `apply_config_env` so changes that can affect
    // codegen also change Cargo's rustflags fingerprint.
    for (key, configured_value) in &ctx.config.env {
        if !key.starts_with("CUDA_OXIDE_") {
            continue;
        }
        if let Some(value) = inherited_env.get(key) {
            // Keep the platform encoding. Presence-only backend switches such
            // as CUDA_OXIDE_NO_FMA remain effective even when their value is
            // not Unicode, so dropping those bytes could reuse stale code.
            effective_env.insert(key.clone(), value.clone());
        } else {
            effective_env.insert(key.clone(), configured_value.as_bytes().to_vec());
        }
    }
    // Capture backend settings inherited outside project config, including
    // current and future CUDA_OXIDE_* switches.
    for (key, value) in inherited_env.iter().filter(|(key, _)| {
        key.starts_with("CUDA_OXIDE_") && key.as_str() != CODEGEN_FINGERPRINT_ENV
    }) {
        effective_env.insert(key.clone(), value.clone());
    }

    // These are wrapper-owned semantic values. Normalize away inherited
    // false/stale handshakes before inserting the effective materialization
    // state below, so no-op values do not create distinct Cargo identities.
    effective_env.remove(CODEGEN_FINGERPRINT_ENV);
    effective_env.remove(MATERIALIZE_ENV);
    effective_env.remove(EXPECTED_PROVENANCE_ENV);

    if opts.verbose {
        effective_env.insert("CUDA_OXIDE_VERBOSE".to_string(), b"1".to_vec());
    }
    if opts.no_fmad {
        effective_env.insert("CUDA_OXIDE_NO_FMA".to_string(), b"1".to_vec());
    }
    if opts.unchecked_indexing {
        effective_env.insert("CUDA_OXIDE_UNCHECKED_INDEXING".to_string(), b"1".to_vec());
    }
    if let Some(level) = opts.device_debug.env_value() {
        effective_env.insert("CUDA_OXIDE_DEBUG".to_string(), level.as_bytes().to_vec());
    }
    if opts.emit_nvvm_ir || materialization.enabled() {
        effective_env.insert("CUDA_OXIDE_EMIT_NVVM_IR".to_string(), b"1".to_vec());
    }
    if let Some(provenance) = &materialization.provenance {
        effective_env.insert(MATERIALIZE_ENV.to_string(), b"1".to_vec());
        effective_env.insert(
            EXPECTED_PROVENANCE_ENV.to_string(),
            provenance.as_bytes().to_vec(),
        );
    }
    if let Some(target_arch) = target_arch {
        effective_env.insert(
            "CUDA_OXIDE_TARGET".to_string(),
            target_arch.as_bytes().to_vec(),
        );
    }
    if let Some(owner_filter) = owner_filter {
        effective_env.insert(
            DEVICE_CODEGEN_CRATE_ENV.to_string(),
            owner_filter.as_bytes().to_vec(),
        );
    }

    // SHA-256 over length-delimited key/value pairs. The complete digest is
    // tracked by device-owning procedural macros, so settings are neither
    // exposed verbatim in diagnostics nor reduced to a small collision space.
    let mut hash = sha2::Sha256::new();
    for (key, value) in effective_env {
        update_codegen_fingerprint_hash(&mut hash, key.as_bytes());
        update_codegen_fingerprint_hash(&mut hash, &value);
    }
    finish_codegen_fingerprint(hash)
}

fn update_codegen_fingerprint_hash(hash: &mut sha2::Sha256, bytes: &[u8]) {
    use sha2::Digest as _;

    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

fn finish_codegen_fingerprint(hash: sha2::Sha256) -> String {
    use sha2::Digest as _;

    let digest: [u8; 32] = hash.finalize().into();
    digest_hex(&digest)
}

/// Track sanitizer-only device output settings in crates that declare device
/// code, without invalidating their host-only dependency graph.
#[allow(clippy::too_many_arguments)]
fn sanitize_codegen_fingerprint(
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    ptx_dir: Option<&Path>,
    materialization: &MaterializationMode,
) -> String {
    sanitize_codegen_fingerprint_with_env(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        target_arch,
        detected_device_arch,
        ptx_dir,
        materialization,
        &inherited_process_env(),
    )
}

/// `sanitize_codegen_fingerprint` with the inherited environment injected, the
/// counterpart to `passthrough_codegen_fingerprint_with_env`.
#[allow(clippy::too_many_arguments)]
fn sanitize_codegen_fingerprint_with_env(
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    ptx_dir: Option<&Path>,
    materialization: &MaterializationMode,
    inherited_env: &BTreeMap<String, Vec<u8>>,
) -> String {
    let opts = CargoPassthroughOptions {
        verbose,
        emit_nvvm_ir: false,
        arch: target_arch,
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad,
        unchecked_indexing,
        materialize_cubin: materialization.enabled(),
        device_debug,
    };
    let base = passthrough_codegen_fingerprint_with_env(
        ctx,
        &opts,
        None,
        target_arch,
        materialization,
        inherited_env,
    );
    let mut hash = sha2::Sha256::new();
    for bytes in [
        "sanitize-line-tables-v1".as_bytes(),
        base.as_bytes(),
        detected_device_arch.unwrap_or("").as_bytes(),
    ] {
        update_codegen_fingerprint_hash(&mut hash, bytes);
    }
    if let Some(ptx_dir) = ptx_dir {
        update_codegen_fingerprint_hash(&mut hash, ptx_dir.as_os_str().as_encoded_bytes());
    }
    finish_codegen_fingerprint(hash)
}

#[allow(clippy::too_many_arguments)]
fn standard_codegen_fingerprint(
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    emit_nvvm_ir: bool,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    materialization: &MaterializationMode,
) -> String {
    let opts = CargoPassthroughOptions {
        verbose,
        emit_nvvm_ir,
        arch: target_arch,
        features: None,
        cargo_target_dir: None,
        device_codegen_crate: None,
        device_cfgs: &[],
        no_fmad,
        unchecked_indexing,
        materialize_cubin: materialization.enabled(),
        device_debug,
    };
    let base = passthrough_codegen_fingerprint(ctx, &opts, None, target_arch, materialization);
    let mut hash = sha2::Sha256::new();
    for bytes in [
        "standard-codegen-v1".as_bytes(),
        base.as_bytes(),
        detected_device_arch.unwrap_or("").as_bytes(),
    ] {
        update_codegen_fingerprint_hash(&mut hash, bytes);
    }
    finish_codegen_fingerprint(hash)
}

fn pipeline_codegen_fingerprint(
    ctx: &Context,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    emit_nvvm_ir: bool,
    target_arch: Option<&str>,
    materialization: &MaterializationMode,
) -> String {
    let base = standard_codegen_fingerprint(
        ctx,
        true,
        no_fmad,
        unchecked_indexing,
        device_debug,
        emit_nvvm_ir,
        target_arch,
        None,
        materialization,
    );
    let mut hash = sha2::Sha256::new();
    for value in [
        base.as_str(),
        "CUDA_OXIDE_SHOW_RUSTC_MIR=1",
        "CUDA_OXIDE_DUMP_MIR=1",
        "CUDA_OXIDE_DUMP_LLVM=1",
    ] {
        update_codegen_fingerprint_hash(&mut hash, value.as_bytes());
    }
    finish_codegen_fingerprint(hash)
}

#[allow(clippy::too_many_arguments)]
fn interop_codegen_fingerprint(
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    target_arch: Option<&str>,
    detected_device_arch: Option<&str>,
    ptx_dir: &Path,
    sanitizer_line_tables: bool,
    materialization: &MaterializationMode,
) -> String {
    let base = standard_codegen_fingerprint(
        ctx,
        verbose,
        no_fmad,
        unchecked_indexing,
        device_debug,
        false,
        target_arch,
        detected_device_arch,
        materialization,
    );
    let mut hash = sha2::Sha256::new();
    for bytes in [
        "interop-codegen-v1".as_bytes(),
        base.as_bytes(),
        if sanitizer_line_tables {
            b"line-tables"
        } else {
            b"default-debug"
        },
        ptx_dir.as_os_str().as_encoded_bytes(),
    ] {
        update_codegen_fingerprint_hash(&mut hash, bytes);
    }
    finish_codegen_fingerprint(hash)
}

fn backend_artifact_digest(path: &Path) -> Result<String, String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    let mut hasher = Sha256::new();
    if path == Path::new("llvm") {
        hasher.update(b"rustc built-in LLVM backend");
        let digest: [u8; 32] = hasher.finalize().into();
        return Ok(digest_hex(&digest));
    }

    let canonical = path
        .canonicalize()
        .map_err(|error| format!("could not resolve backend {}: {error}", path.display()))?;
    let mut file = std::fs::File::open(&canonical).map_err(|error| {
        format!(
            "could not open backend {} for fingerprinting: {error}",
            canonical.display()
        )
    })?;
    let mut hasher = Sha256::new();
    let mut chunk = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut chunk).map_err(|error| {
            format!(
                "could not read backend {} for fingerprinting: {error}",
                canonical.display()
            )
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&chunk[..read]);
    }
    let digest: [u8; 32] = hasher.finalize().into();
    Ok(digest_hex(&digest))
}

fn cargo_passthrough_command(
    ctx: &Context,
    cargo_subcommand: CargoPassthroughSubcommand,
    opts: &CargoPassthroughOptions<'_>,
    cargo_args: &[String],
) -> Result<Command, String> {
    cargo_passthrough_command_with_env(
        ctx,
        cargo_subcommand,
        opts,
        cargo_args,
        std::env::var_os(MATERIALIZE_ENV),
    )
}

/// `cargo_passthrough_command` with the ambient
/// `CUDA_OXIDE_MATERIALIZE_CUBIN` injected.
///
/// Unit tests must call this with `None`: the ambient value outranks
/// `opts.materialize_cubin`, so an exported one turns materialization on and
/// sends the test into `discover_materializer_provenance`, which re-executes
/// `current_exe` -- the libtest binary under `cargo test` -- and then exits the
/// process over the unusable digest, taking the whole suite with it.
fn cargo_passthrough_command_with_env(
    ctx: &Context,
    cargo_subcommand: CargoPassthroughSubcommand,
    opts: &CargoPassthroughOptions<'_>,
    cargo_args: &[String],
    materialize_env: Option<std::ffi::OsString>,
) -> Result<Command, String> {
    let target_arch = configured_arch(ctx, opts.arch);
    let materialization = prepare_materialization_with_env(
        ctx,
        opts.materialize_cubin,
        opts.arch,
        opts.emit_nvvm_ir,
        materialize_env,
    );
    let owner_filter = configured_device_codegen_crates(ctx, opts.device_codegen_crate)?;
    // Device-owning macros track this identity in their crate dep-info. Keep it
    // out of global rustflags so host-only dependencies retain one cache key.
    let fingerprint = passthrough_codegen_fingerprint(
        ctx,
        opts,
        owner_filter.as_deref(),
        target_arch,
        &materialization,
    );
    let mut cmd = Command::new("cargo");
    cmd.arg(cargo_subcommand.as_str());
    if let Some(features) = opts.features {
        cmd.args(["--features", features]);
    }
    cmd.args(cargo_args).current_dir(&ctx.workspace_root);

    // Project configuration provides defaults. Explicit wrapper flags and
    // internal compiler requirements are applied afterward and therefore win.
    apply_common_codegen_env(
        &mut cmd,
        ctx,
        opts.verbose,
        opts.no_fmad,
        opts.unchecked_indexing,
        opts.device_debug,
    );
    apply_codegen_configuration(
        &mut cmd,
        ctx,
        cargo_subcommand.codegen_profile(),
        opts.device_cfgs,
        &fingerprint,
    )?;

    if let Some(cargo_target_dir) = opts.cargo_target_dir {
        cmd.env("CARGO_TARGET_DIR", cargo_target_dir);
    }
    if let Some(owner_filter) = owner_filter {
        cmd.env(DEVICE_CODEGEN_CRATE_ENV, owner_filter);
    }
    apply_output_mode(&mut cmd, opts.emit_nvvm_ir, target_arch, &materialization);
    Ok(cmd)
}

/// Run an arbitrary Cargo build-like subcommand through the cuda-oxide backend.
///
/// Unlike example mode, this does not touch source files or clean generated
/// artifacts. It is intended for final-target workspace builds where Cargo's
/// incremental behavior should remain intact.
pub fn codegen_cargo_passthrough(
    ctx: &Context,
    cargo_subcommand: CargoPassthroughSubcommand,
    opts: CargoPassthroughOptions<'_>,
    cargo_args: &[String],
) {
    let cargo_subcommand_name = cargo_subcommand.as_str();
    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA CARGO {}", cargo_subcommand_name);
    println!("=========================================");
    println!();

    let mut cmd = cargo_passthrough_command(ctx, cargo_subcommand, &opts, cargo_args)
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(2);
        });

    let displayed_args: Vec<_> = cmd
        .get_args()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    if displayed_args.is_empty() {
        println!("Running cargo {}...", cargo_subcommand_name);
    } else {
        println!(
            "Running cargo {} {}...",
            cargo_subcommand_name,
            displayed_args.join(" ")
        );
    }
    println!();

    let status = cmd.status().expect("Failed to run cargo");
    if !status.success() {
        eprintln!(
            "\nCargo {} failed with exit code: {:?}",
            cargo_subcommand_name,
            status.code()
        );
        std::process::exit(status.code().unwrap_or(1));
    }

    println!();
    println!("✓ Cargo {} succeeded", cargo_subcommand_name);
}

// =============================================================================
// Pipeline command
// =============================================================================

/// Show verbose pipeline progress and the available intermediate artifacts.
///
/// Enables all diagnostic env vars (`CUDA_OXIDE_VERBOSE`, `SHOW_RUSTC_MIR`,
/// `DUMP_MIR`, `DUMP_LLVM`) so the user can see MIR collection, the
/// `dialect-mir` module (pre- and post-`mem2reg`), the LLVM dialect
/// module, textual LLVM IR, and the final PTX or NVVM IR. After the build,
/// generated artifacts are printed to stdout.
#[allow(clippy::too_many_arguments)]
pub fn codegen_show_pipeline(
    ctx: &Context,
    example: &str,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
    materialize_cubin: bool,
) {
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, emit_nvvm_ir);
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };

    if load_interop_config(&example_dir).is_some_and(|config| !config.device_crates.is_empty()) {
        reject_interop_output_mode(emit_nvvm_ir, &materialization);
    }

    clean_generated_files(&example_dir, example);

    println!("=========================================");
    println!("RUSTC-CODEGEN-CUDA PIPELINE: {}", example);
    println!("=========================================");
    println!();
    let target_arch_label = configured_arch_label(ctx, arch);
    match (
        materialization.enabled(),
        emit_nvvm_ir,
        target_arch_label.as_deref(),
    ) {
        (true, _, Some(target_arch)) => {
            println!("Output format: materialized cubin (arch: {target_arch})")
        }
        (false, true, Some(target_arch)) => {
            println!("Output format: NVVM IR (arch: {})", target_arch)
        }
        (false, false, Some(target_arch)) => {
            println!("Output format: PTX (arch override: {})", target_arch)
        }
        (false, false, None) => println!("Output format: PTX (auto-detected arch)"),
        (true, _, None) | (false, true, None) => {
            unreachable!("IR/final materialization requires a configured architecture")
        }
    }
    println!();
    println!("Required flags (applied via CARGO_ENCODED_RUSTFLAGS):");
    println!("  -C opt-level=3              MIR optimization");
    println!("  -C debug-assertions=off     Remove debug checks");
    println!("  -Z mir-enable-passes=-JumpThreading");
    println!("                              Prevent barrier duplication");
    println!("  -Z always-encode-mir        Emit MIR for all reachable device deps");
    println!();
    println!("Note: panic=abort is NOT required - the codegen backend treats");
    println!("      unwind paths as unreachable (CUDA toolchain limitation, not HW).");
    println!();

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&example_dir);

    apply_config_env(&mut cmd, ctx);
    let fingerprint = pipeline_codegen_fingerprint(
        ctx,
        no_fmad,
        unchecked_indexing,
        device_debug,
        emit_nvvm_ir,
        target_arch,
        &materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::ReleaseLike,
        &[],
        &fingerprint,
    );
    cmd.env("CUDA_OXIDE_VERBOSE", "1");
    cmd.env("CUDA_OXIDE_SHOW_RUSTC_MIR", "1");
    cmd.env("CUDA_OXIDE_DUMP_MIR", "1");
    cmd.env("CUDA_OXIDE_DUMP_LLVM", "1");
    if no_fmad {
        cmd.env("CUDA_OXIDE_NO_FMA", "1");
    }
    if unchecked_indexing {
        cmd.env("CUDA_OXIDE_UNCHECKED_INDEXING", "1");
    }
    if let Some(level) = device_debug.env_value() {
        cmd.env("CUDA_OXIDE_DEBUG", level);
    }

    apply_output_mode(&mut cmd, emit_nvvm_ir, target_arch, &materialization);
    apply_ld_library_path(&mut cmd, ctx);

    println!("Building {}...", example);
    println!();

    let status = cmd.status().expect("Failed to run cargo");

    if !status.success() {
        eprintln!("\nBuild failed with exit code: {:?}", status.code());
        std::process::exit(status.code().unwrap_or(1));
    }

    show_generated_artifacts(&example_dir, example);
}

// =============================================================================
// Debug command
// =============================================================================

/// Build with debug info and launch cuda-gdb (or cgdb).
///
/// Compiles the example with `-C debuginfo=2` on top of the normal release
/// flags, then launches the debugger on the resulting binary. Prints a
/// quick-reference cheat sheet for common cuda-gdb commands before handing
/// control to the debugger.
#[allow(clippy::too_many_arguments)]
pub fn codegen_debug(
    ctx: &Context,
    example: &str,
    arch: Option<&str>,
    features: Option<&str>,
    bin: Option<&str>,
    use_cgdb: bool,
    use_tui: bool,
    materialize_cubin: bool,
) {
    let example_dir = if ctx.is_workspace {
        resolve_example_dir(ctx, example)
    } else {
        ctx.workspace_root.clone()
    };
    let target_arch = configured_arch(ctx, arch);
    let materialization = prepare_materialization(ctx, materialize_cubin, arch, false);
    if load_interop_config(&example_dir).is_some_and(|config| !config.device_crates.is_empty()) {
        reject_interop_output_mode(false, &materialization);
    }

    let cuda_gdb = find_cuda_toolkit_executable(
        ctx,
        "cuda-gdb",
        &[
            "/usr/local/cuda/bin/cuda-gdb",
            "/opt/cuda/bin/cuda-gdb",
            "/usr/bin/cuda-gdb",
        ],
    )
    .unwrap_or_else(|| {
        eprintln!("Error: cuda-gdb not found!");
        eprintln!();
        eprintln!("Make sure CUDA toolkit is installed and cuda-gdb is in your PATH");
        eprintln!("or configured CUDA toolkit root:");
        eprintln!("  export PATH=\"/usr/local/cuda/bin:$PATH\"");
        eprintln!("  export CUDA_TOOLKIT_PATH=/usr/local/cuda");
        std::process::exit(1);
    });

    let cgdb_path = if use_cgdb {
        Some(find_executable("cgdb", &[]).unwrap_or_else(|| {
            eprintln!("Error: cgdb not found!");
            eprintln!("Install with: sudo apt install cgdb");
            std::process::exit(1);
        }))
    } else {
        None
    };

    let detected_device_arch = detect_run_target_arch(target_arch, materialization.enabled());

    if let Some(bin) = bin {
        println!("Building {} (bin: {}) with debug info...", example, bin);
    } else {
        println!("Building {} with debug info...", example);
    }
    if let Some(dev) = detected_device_arch.as_deref() {
        println!("Detected GPU arch: {dev} (via nvidia-smi)");
    }

    clean_generated_files(&example_dir, example);

    touch_main_rs(&example_dir);

    let mut cmd = Command::new("cargo");
    cmd.args(["build", "--release"]).current_dir(&example_dir);

    if let Some(bin) = bin {
        cmd.args(["--bin", bin]);
    }
    if let Some(features) = features {
        cmd.args(["--features", features]);
    }

    apply_config_env(&mut cmd, ctx);
    let fingerprint = standard_codegen_fingerprint(
        ctx,
        false,
        false,
        false,
        DeviceDebug::Off,
        false,
        target_arch,
        detected_device_arch.as_deref(),
        &materialization,
    );
    apply_codegen_configuration_or_exit(
        &mut cmd,
        ctx,
        CodegenProfilePolicy::ReleaseLikeWithDebugInfo,
        &[],
        &fingerprint,
    );
    cmd.env("CARGO_PROFILE_RELEASE_DEBUG", "2");
    apply_output_mode(&mut cmd, false, target_arch, &materialization);
    apply_device_arch_hint(&mut cmd, target_arch, detected_device_arch.as_deref());
    apply_ld_library_path(&mut cmd, ctx);

    let binary =
        run_cargo_build_for_executable(&mut cmd, &example_dir, bin).unwrap_or_else(|message| {
            eprintln!("Failed to build {example}: {message}");
            std::process::exit(1);
        });
    if !binary.exists() {
        eprintln!(
            "Error: Cargo reported executable artifact {}, but it does not exist",
            binary.display()
        );
        std::process::exit(1);
    }

    if cgdb_path.is_some() {
        println!("Launching cgdb (cuda-gdb frontend)...");
    } else {
        println!(
            "Launching cuda-gdb{}...",
            if use_tui { " (TUI mode)" } else { "" }
        );
    }
    println!();
    println!("Quick reference:");
    println!("  set cuda break_on_launch application");
    println!("                           - Break at start of any kernel");
    println!("  run                      - Start the program");
    println!("  info cuda kernels        - List active kernels");
    println!("  info cuda threads        - List GPU threads");
    println!("  cuda thread (0,0,0)      - Switch to thread");
    println!("  cuda block (0,0,0)       - Switch to block");
    println!("  print <var>              - Print variable");
    println!("  next / step / continue   - Execution control");
    println!("  quit                     - Exit debugger");
    if cgdb_path.is_some() {
        println!();
        println!("cgdb shortcuts:");
        println!("  Esc                      - Focus source window (vim keys work)");
        println!("  i                        - Focus command window");
        println!("  space                    - Set breakpoint on current line");
        println!("  o                        - Open file dialog");
    } else if use_tui {
        println!();
        println!("TUI shortcuts:");
        println!("  Ctrl+x a                 - Toggle TUI mode");
        println!("  Ctrl+x 2                 - Split view (source + asm)");
        println!("  Ctrl+l                   - Refresh screen");
    }
    println!();

    let status = if let Some(cgdb) = cgdb_path {
        Command::new(cgdb)
            .arg("-d")
            .arg(&cuda_gdb)
            .arg(&binary)
            .current_dir(&example_dir)
            .status()
            .expect("Failed to launch cgdb")
    } else {
        let mut gdb_cmd = Command::new(&cuda_gdb);
        if use_tui {
            gdb_cmd.arg("--tui");
        }
        gdb_cmd.arg(&binary);
        gdb_cmd.current_dir(&example_dir);
        gdb_cmd.status().expect("Failed to launch cuda-gdb")
    };

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
}

// =============================================================================
// Fmt command
// =============================================================================

/// Format (or check formatting of) all crates in the workspace.
///
/// Runs `cargo fmt --all` in three scopes: root workspace, codegen backend
/// crate, and every example that has a `Cargo.toml`. In `check` mode,
/// reports which files need formatting without modifying them.
pub fn format_all(ctx: &Context, check: bool) {
    let mode = if check { "Checking" } else { "Formatting" };
    let mut failed = false;

    println!("📦 {} root workspace...", mode);
    if !run_cargo_fmt(&ctx.workspace_root, check) {
        failed = true;
    }

    println!("📦 {} rustc-codegen-cuda...", mode);
    if !run_cargo_fmt(&ctx.codegen_crate, check) {
        failed = true;
    }

    if let Ok(entries) = std::fs::read_dir(&ctx.examples_dir) {
        let mut examples: Vec<_> = entries.flatten().filter(|e| e.path().is_dir()).collect();
        examples.sort_by_key(|e| e.file_name());

        for entry in examples {
            let example_name = entry.file_name();
            let example_path = entry.path();

            if !example_path.join("Cargo.toml").exists() {
                continue;
            }

            println!("📦 {} example: {}...", mode, example_name.to_string_lossy());
            if !run_cargo_fmt(&example_path, check) {
                failed = true;
            }
        }
    }

    if failed {
        if check {
            eprintln!();
            eprintln!("❌ Some files need formatting. Run: cargo oxide fmt");
        } else {
            eprintln!();
            eprintln!("⚠️  Some formatting commands failed (see above)");
        }
        std::process::exit(1);
    } else {
        println!();
        if check {
            println!("✅ All files are properly formatted");
        } else {
            println!("✅ All crates formatted");
        }
    }
}

/// Run `cargo fmt --all` in a single directory. Returns `true` on success.
fn run_cargo_fmt(dir: &Path, check: bool) -> bool {
    let mut cmd = Command::new("cargo");
    cmd.arg("fmt").arg("--all").current_dir(dir);

    if check {
        cmd.arg("--check");
    }

    match cmd.status() {
        Ok(status) => status.success(),
        Err(e) => {
            eprintln!("  Failed to run cargo fmt: {}", e);
            false
        }
    }
}

// =============================================================================
// Doctor command
// =============================================================================

/// Parsed contents of a `rust-toolchain.toml` pin.
#[derive(Clone, Debug, Eq, PartialEq)]
struct RustToolchainPin {
    channel: String,
    components: Vec<String>,
}

/// Components that doctor treats as hard requirements for the cuda-oxide
/// pipeline even if `rust-toolchain.toml` stops listing them: `rust-src`
/// (device-side core sources), `rustc-dev` (rustc_private, required to build
/// the codegen backend), and `llvm-tools`.
const DOCTOR_REQUIRED_COMPONENTS: &[&str] = &["rust-src", "rustc-dev", "llvm-tools"];

/// The components doctor verifies for a pin: everything the pin itself lists,
/// plus the [`DOCTOR_REQUIRED_COMPONENTS`] floor.
///
/// rustup auto-installs every component named in `rust-toolchain.toml` when it
/// installs the pinned toolchain, so a pinned component that is absent from
/// `rustup component list --installed` means a broken or manually trimmed
/// install and is worth failing doctor over. The floor guards against a future
/// edit of the pin file dropping a component the pipeline genuinely needs.
fn doctor_verified_components(pin: &RustToolchainPin) -> Vec<String> {
    let mut required: Vec<String> = pin.components.clone();
    for component in DOCTOR_REQUIRED_COMPONENTS {
        if !required.iter().any(|existing| existing == component) {
            required.push((*component).to_string());
        }
    }
    required
}

/// Parse a `rust-toolchain.toml` document for channel and components.
fn parse_rust_toolchain_toml(contents: &str) -> Result<RustToolchainPin, String> {
    let value: toml::Value =
        toml::from_str(contents).map_err(|error| format!("invalid TOML: {error}"))?;
    let toolchain = value
        .get("toolchain")
        .ok_or_else(|| "missing [toolchain] table".to_string())?;
    let channel = toolchain
        .get("channel")
        .and_then(toml::Value::as_str)
        .map(str::trim)
        .filter(|channel| !channel.is_empty())
        .ok_or_else(|| "missing toolchain.channel".to_string())?
        .to_string();
    let components = match toolchain.get("components") {
        None => Vec::new(),
        Some(toml::Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(|name| name.trim().to_string())
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| {
                        "toolchain.components entries must be non-empty strings".to_string()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => {
            return Err("toolchain.components must be an array of strings".to_string());
        }
    };
    Ok(RustToolchainPin {
        channel,
        components,
    })
}

/// True when `rustup show active-toolchain` output matches the pinned channel.
///
/// The toolchain name is the first whitespace-delimited token of the first
/// line in every rustup output format seen so far:
///
/// - pre-1.28 and 1.29+: `nightly-2026-04-03-<triple> (default)` or
///   `nightly-2026-04-03-<triple> (overridden by '<path>')` on one line
///   (verified against rustup 1.29.0);
/// - 1.28.x: the bare name on the first line with the reason on a second
///   `active because: ...` line.
fn active_toolchain_matches_channel(active_toolchain: &str, channel: &str) -> bool {
    let active = active_toolchain
        .lines()
        .next()
        .unwrap_or("")
        .split_whitespace()
        .next()
        .unwrap_or("");
    if active.is_empty() || channel.is_empty() {
        return false;
    }
    active == channel || active.starts_with(&format!("{channel}-"))
}

/// Return required components that are absent from `rustup component list --installed`.
fn missing_rustup_components<S: AsRef<str>>(installed_list: &str, required: &[S]) -> Vec<String> {
    required
        .iter()
        .map(AsRef::as_ref)
        .filter(|component| !rustup_component_installed(installed_list, component))
        .map(str::to_string)
        .collect()
}

fn rustup_component_installed(installed_list: &str, component: &str) -> bool {
    installed_list.lines().any(|line| {
        let name = line.split_whitespace().next().unwrap_or("");
        name == component || name.starts_with(&format!("{component}-"))
    })
}

fn doctor_report_toolchain_pin(ctx: &Context, ok: &mut bool) {
    let toolchain_file = ctx.workspace_root.join("rust-toolchain.toml");
    print!("rust-toolchain.toml... ");
    if !toolchain_file.exists() {
        println!("✗ not found at {}", toolchain_file.display());
        *ok = false;
        return;
    }

    let contents = match std::fs::read_to_string(&toolchain_file) {
        Ok(contents) => contents,
        Err(error) => {
            println!("✗ present but unreadable ({error})");
            *ok = false;
            return;
        }
    };

    let pin = match parse_rust_toolchain_toml(&contents) {
        Ok(pin) => pin,
        Err(error) => {
            println!("✗ present but invalid ({error})");
            *ok = false;
            return;
        }
    };
    println!("✓ channel {}", pin.channel);

    print!("Pinned toolchain active... ");
    match Command::new("rustup")
        .args(["show", "active-toolchain"])
        .output()
    {
        Ok(output) if output.status.success() => {
            let active = String::from_utf8_lossy(&output.stdout);
            let active = active.trim();
            if active_toolchain_matches_channel(active, &pin.channel) {
                println!("✓ {active}");
            } else {
                println!(
                    "✗ active `{active}`, expected `{pin_channel}`",
                    pin_channel = pin.channel
                );
                eprintln!(
                    "  Install/select the pin with `rustup toolchain install {}` and reopen the shell",
                    pin.channel
                );
                eprintln!("  in this workspace so rust-toolchain.toml can select it.");
                *ok = false;
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("✗ rustup show active-toolchain failed");
            if !stderr.trim().is_empty() {
                eprintln!("  {}", stderr.trim());
            }
            *ok = false;
        }
        Err(_) => {
            println!("✗ rustup not found");
            eprintln!("  Install rustup from https://rustup.rs/ so doctor can verify the pin.");
            *ok = false;
        }
    }

    let required = doctor_verified_components(&pin);

    print!("Required rustup components... ");
    match Command::new("rustup")
        .args([
            "component",
            "list",
            "--installed",
            "--toolchain",
            &pin.channel,
        ])
        .output()
    {
        Ok(output) if output.status.success() => {
            let installed = String::from_utf8_lossy(&output.stdout);
            let missing = missing_rustup_components(&installed, &required);
            if missing.is_empty() {
                println!("✓ {}", required.join(", "));
            } else {
                println!("✗ missing {}", missing.join(", "));
                eprintln!(
                    "  Install with `rustup component add --toolchain {} {}`",
                    pin.channel,
                    missing.join(" ")
                );
                *ok = false;
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            println!("✗ could not list components for {}", pin.channel);
            if !stderr.trim().is_empty() {
                eprintln!("  {}", stderr.trim());
            }
            eprintln!(
                "  Try `rustup toolchain install {channel} -c {components}`",
                channel = pin.channel,
                components = required.join(" -c ")
            );
            *ok = false;
        }
        Err(_) => {
            println!("✗ rustup not found");
            *ok = false;
        }
    }
}

/// Validate the development environment.
///
/// Checks for: Rust nightly toolchain, `rust-toolchain.toml`, the codegen
/// backend `.so` (informational), CUDA headers (`cuda.h`), CUDA toolkit
/// (`nvcc`, libNVVM, nvJitLink, libdevice), LLVM (`llc`), clang/libclang,
/// the NVIDIA driver / GPU (informational), and optionally `cuda-gdb` /
/// `compute-sanitizer`.
/// Exits non-zero if any required check fails.
///
/// Doctor itself needs neither the CUDA toolkit nor a driver: every check
/// is a subprocess, a filesystem probe, or a runtime `dlopen`, and the
/// caller resolves the context via [`resolve_passive_context`] so nothing is
/// built first. This is what lets it diagnose a bare machine (issue #87).
pub fn doctor(ctx: &Context) {
    println!("cargo-oxide environment check");
    println!("==============================");
    println!();

    let mut ok = true;

    // 1. Rust toolchain
    print!("Rust nightly toolchain... ");
    match Command::new("rustc").args(["--version"]).output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            let version = version.trim();
            if version.contains("nightly") {
                println!("✓ {}", version);
            } else {
                println!("✗ expected nightly, got: {}", version);
                ok = false;
            }
        }
        _ => {
            println!("✗ rustc not found");
            ok = false;
        }
    }

    // 2. rust-toolchain.toml pin + active channel + required components
    doctor_report_toolchain_pin(ctx, &mut ok);

    // 3. Backend .so. Informational, not fatal: `run`/`build`/`pipeline`
    // build the backend on demand, so "not built yet" is a healthy state
    // for a fresh clone.
    print!("Codegen backend... ");
    if ctx.backend_so.exists() {
        println!("✓ {}", ctx.backend_so.display());
    } else {
        println!("- not built yet (run `cargo oxide setup`)");
    }

    // 3a. Project config (`.cargo/cuda-oxide.toml`)
    doctor_report_oxide_config(ctx, &mut ok);

    // 3b. Shared cache. The check above reports the backend this context
    // resolves to, which inside the repository is the local build. A project
    // outside the repository resolves to the cache instead, so the two can
    // disagree while every other check passes.
    print!("Shared cache (external projects)... ");
    match backend::cached_backend_path() {
        Some(cached) => match backend::compare_cache_to_local(&cached, &ctx.backend_so) {
            backend::CacheReport::Absent => {
                println!("- empty; external projects build on first use");
            }
            backend::CacheReport::UpToDate => {
                println!("✓ {}", cached.display());
            }
            backend::CacheReport::OlderThanLocal => {
                println!("⚠ {}", cached.display());
                println!("  Older than the backend built here, so projects outside this");
                println!("  repository would load a different one. Run `cargo oxide setup`");
                println!("  to publish, or set CUDA_OXIDE_BACKEND to pin an explicit path.");
            }
        },
        None => println!("- cache directory unknown (set CARGO_HOME or HOME)"),
    }

    // 4. CUDA headers (cuda.h). The host `cuda-bindings` crate cannot build
    // without them; cargo-oxide itself deliberately can, which is what makes
    // this check reachable on a toolkit-less machine instead of dying inside
    // cuda-bindings' build script (issue #87).
    print!("CUDA headers (cuda.h)... ");
    let toolkit = cuda_toolkit_root(|var| std::env::var(var).ok());
    let header_candidates = cuda_header_candidates(&toolkit, std::env::consts::ARCH);
    match header_candidates.iter().find(|path| path.is_file()) {
        Some(found) => println!("✓ {}", found.display()),
        None => {
            println!("✗ not found in the CUDA toolkit at `{}`", toolkit);
            eprintln!("  Probed:");
            for candidate in &header_candidates {
                eprintln!("    {}", candidate.display());
            }
            eprintln!("  Host crates (cuda-bindings) cannot build without cuda.h. Set");
            eprintln!("  CUDA_TOOLKIT_PATH or CUDA_HOME to a CUDA Toolkit install root;");
            eprintln!("  when neither is set, /usr/local/cuda is used.");
            ok = false;
        }
    }

    // 5. CUDA toolkit
    print!("CUDA toolkit (nvcc)... ");
    match Command::new("nvcc").arg("--version").output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = version.lines().find(|l| l.contains("release")) {
                println!("✓ {}", line.trim());
            } else {
                println!("✓ (version unknown)");
            }
        }
        _ => {
            println!("✗ nvcc not found");
            ok = false;
        }
    }

    // 5b. libNVVM + nvJitLink + libdevice (only required when a kernel uses
    // CUDA libdevice math, e.g. sin/cos/exp/pow). All three ship with the
    // CUDA Toolkit; checking them here surfaces missing or split packagings
    // before a runtime failure inside `cuda_host::ltoir::load_kernel_module`.
    print!("libNVVM (libnvvm.so)... ");
    match libnvvm_sys::LibNvvm::load() {
        Ok(nvvm) => match nvvm.version() {
            Ok((major, minor)) => println!("✓ libNVVM {}.{}", major, minor),
            Err(_) => println!("✓ (version query failed but library loaded)"),
        },
        Err(e) => {
            println!("✗ {}", e);
            eprintln!("  Required only when kernels call CUDA libdevice math");
            eprintln!("  (sin/cos/exp/pow/...). Ships with the CUDA Toolkit at");
            eprintln!("  <CUDA>/nvvm/lib64/libnvvm.so. No separate download.");
            ok = false;
        }
    }

    print!("nvJitLink (libnvJitLink.so)... ");
    match nvjitlink_sys::LibNvJitLink::load() {
        Ok(nvj) => match nvj.version() {
            Some((major, minor)) => println!("✓ nvJitLink {}.{}", major, minor),
            None => println!("✓ (version symbol not exported on this CTK)"),
        },
        Err(e) => {
            println!("✗ {}", e);
            eprintln!("  Required only when kernels call CUDA libdevice math.");
            eprintln!("  Ships with the CUDA Toolkit at <CUDA>/lib64/libnvJitLink.so.");
            ok = false;
        }
    }

    print!("libdevice (libdevice.10.bc)... ");
    match libnvvm_sys::find_libdevice() {
        Ok(path) => println!("✓ {}", path.display()),
        Err(e) => {
            println!("✗ {}", e);
            eprintln!("  Required only when kernels call CUDA libdevice math.");
            eprintln!("  Ships with the CUDA Toolkit at");
            eprintln!("  <CUDA>/nvvm/libdevice/libdevice.10.bc. Override the search");
            eprintln!("  with `CUDA_OXIDE_LIBDEVICE=<path>` if you have it elsewhere.");
            ok = false;
        }
    }

    // 6. llc (LLVM static compiler for PTX)
    //
    // cuda-oxide requires LLVM 21+: earlier releases reject modern TMA /
    // tcgen05 / WGMMA intrinsic signatures. Probe in the same order as the
    // pipeline:
    //   1. `CUDA_OXIDE_LLC` (caller-supplied override)
    //   2. Rust toolchain's `llvm-tools` component (auto-installed via rustup)
    //   3. `llc-22`, `llc-21`, `llc` on `PATH`
    // Whatever we pick, reject if the major version is < 21.
    print!("llc (LLVM)... ");

    // The pipeline's primary entry: the `llc` bundled with the pinned Rust
    // toolchain's `llvm-tools` component. Built with the NVPTX backend
    // enabled, so the typical novice path is `rustup component add llvm-tools`
    // and that's it. Surface the absolute path so doctor's output matches
    // what the pipeline actually invokes.
    let rustup_llc_path: Option<String> = Command::new("rustc")
        .args(["--print", "sysroot", "--print", "host-tuple"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|output| {
            let stdout = String::from_utf8(output.stdout).ok()?;
            let mut lines = stdout.lines();
            let sysroot = lines.next()?;
            let host = lines.next()?;
            let path: std::path::PathBuf = [sysroot, "lib", "rustlib", host, "bin", "llc"]
                .iter()
                .collect();
            path.is_file()
                .then(|| path.to_str().map(str::to_string))
                .flatten()
        });

    let mut candidates: Vec<String> = Vec::new();
    if let Ok(env_llc) = std::env::var("CUDA_OXIDE_LLC") {
        candidates.push(env_llc);
    }
    if let Some(rustup) = rustup_llc_path.clone() {
        candidates.push(rustup);
    }
    for name in ["llc-22", "llc-21", "llc"] {
        candidates.push(name.to_string());
    }

    let llc_pick = candidates.iter().find_map(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                (
                    candidate.clone(),
                    String::from_utf8_lossy(&o.stdout).into_owned(),
                )
            })
    });
    match llc_pick {
        Some((binary, stdout)) => {
            let banner = stdout
                .lines()
                .find(|l| l.contains("LLVM version"))
                .unwrap_or("(version unknown)")
                .trim()
                .to_string();
            let major = banner
                .split("LLVM version")
                .nth(1)
                .and_then(|rest| rest.trim().split('.').next())
                .and_then(|s| s.parse::<u32>().ok());
            match major {
                Some(v) if v >= 21 => println!("✓ {} ({})", banner, binary),
                Some(v) => {
                    println!("✗ {} ({}) — need LLVM 21+", banner, binary);
                    eprintln!(
                        "  Your `{}` reports LLVM {}, which rejects the TMA / tcgen05 /",
                        binary, v
                    );
                    eprintln!("  WGMMA intrinsic signatures cuda-oxide emits. Install a newer");
                    eprintln!("  toolchain (`rustup component add llvm-tools` is usually enough,");
                    eprintln!("  or `sudo apt install llvm-21`) and either add it to PATH or set");
                    eprintln!("  `CUDA_OXIDE_LLC=/path/to/llc`.");
                    ok = false;
                }
                None => println!("✓ {} ({}, version could not be parsed)", banner, binary),
            }
        }
        None => {
            println!("✗ llc not found");
            eprintln!("  cuda-oxide probes (in order): $CUDA_OXIDE_LLC, the Rust toolchain's");
            eprintln!("  llvm-tools llc, then llc-22/llc-21/llc on PATH. Easiest fix:");
            eprintln!("    rustup component add llvm-tools");
            eprintln!("  Alternative: `sudo apt install llvm-21` (older versions reject");
            eprintln!("  modern TMA / tcgen05 / WGMMA intrinsics).");
            ok = false;
        }
    }

    // 7. clang / libclang resource dir (host `cuda-bindings` / bindgen)
    //
    // The host `cuda-bindings` crate's build.rs runs bindgen, which loads
    // libclang at runtime to parse `wrapper.h`. That parse pulls in
    // `<stddef.h>`, which must be served from clang's own resource
    // directory — the system/GCC copy is not compatible. Fresh installs of
    // bare `libclang1-*` (without the matching `libclang-common-*-dev`)
    // leave `/usr/lib/clang/*/include` empty and bindgen explodes with a
    // mysterious "'stddef.h' file not found". Catch that up front.
    print!("clang / libclang resource dir... ");
    let clang_resource_dir = Command::new("clang")
        .arg("-print-resource-dir")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    match clang_resource_dir {
        Some(ref dir) if std::path::Path::new(&format!("{}/include/stddef.h", dir)).exists() => {
            println!("✓ {}", dir);
        }
        Some(ref dir) => {
            println!(
                "✗ resource dir present but `include/stddef.h` missing: {}",
                dir
            );
            eprintln!("  Host `cuda-bindings` uses bindgen, which needs clang's own stddef.h.");
            eprintln!("  Install the matching dev headers: sudo apt install clang-21");
            eprintln!("  (or libclang-common-21-dev)");
            ok = false;
        }
        None => {
            println!("✗ clang not found");
            eprintln!(
                "  Host `cuda-bindings` uses bindgen, which needs clang + its resource headers."
            );
            eprintln!("  Install with: sudo apt install clang-21");
            eprintln!("  (or at minimum `libclang-common-21-dev` alongside your libclang)");
            ok = false;
        }
    }

    // 8. NVIDIA driver / GPU. Informational, not fatal: only `cargo oxide
    // run` (kernel execution) needs a driver. Cross-compiling and GPU-less
    // CI boxes are supported workflows (`build`/`pipeline` work fine), and
    // the examples-compile CI job is exactly that.
    print!("NVIDIA driver / GPU... ");
    match query_gpu_name_and_compute_cap() {
        Some((name, (major, minor))) => {
            println!("✓ {} (compute capability {}.{})", name, major, minor);
        }
        None => {
            // Some containers mount the kernel driver without shipping
            // nvidia-smi; /proc distinguishes "driver loaded, tool broken"
            // from "no driver at all".
            if Path::new("/proc/driver/nvidia/version").exists() {
                println!("- driver loaded, but nvidia-smi is missing or not reporting a GPU");
                eprintln!("  A kernel-mode NVIDIA driver is present (/proc/driver/nvidia/");
                eprintln!("  version), but `nvidia-smi` did not report a usable GPU.");
                eprintln!("  `cargo oxide run` may still work; arch auto-detection will fall");
                eprintln!("  back to the backend default (override with --arch=<sm_XX>).");
            } else {
                println!("- no NVIDIA driver detected");
                eprintln!("  Only `cargo oxide run` (kernel execution) needs the driver;");
                eprintln!("  `cargo oxide build` and `pipeline` work without one.");
            }
        }
    }

    // 9. cuda-gdb (optional)
    print!("cuda-gdb (optional)... ");
    match Command::new("cuda-gdb").arg("--version").output() {
        Ok(output) if output.status.success() => {
            let version = String::from_utf8_lossy(&output.stdout);
            if let Some(line) = version.lines().next() {
                println!("✓ {}", line.trim());
            } else {
                println!("✓");
            }
        }
        _ => {
            println!("- not found (only needed for `cargo oxide debug`)");
        }
    }

    // 10. compute-sanitizer (optional) — same discovery order as `sanitize`
    print!("compute-sanitizer (optional)... ");
    match find_cuda_toolkit_executable(ctx, "compute-sanitizer", COMPUTE_SANITIZER_FALLBACK_PATHS) {
        Some(path) => match Command::new(&path).arg("--version").output() {
            Ok(output) if output.status.success() => {
                let version = String::from_utf8_lossy(&output.stdout);
                // `compute-sanitizer --version` prints a banner and a
                // copyright line before the actual "Version ..." line.
                let line = version
                    .lines()
                    .map(str::trim)
                    .find(|line| line.starts_with("Version"))
                    .or_else(|| version.lines().next().map(str::trim));
                if let Some(line) = line {
                    println!("✓ {} ({})", line, path.display());
                } else {
                    println!("✓ {}", path.display());
                }
            }
            _ => println!("✓ {}", path.display()),
        },
        None => {
            println!("- not found (only needed for `cargo oxide sanitize`)");
        }
    }

    println!();
    if ok {
        println!("✅ Environment looks good!");
    } else {
        println!("❌ Some checks failed. Fix the issues above and re-run `cargo oxide doctor`.");
        std::process::exit(1);
    }
}

/// CUDA toolkit install root for doctor's `cuda.h` probe: the first set
/// variable among `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, else `/usr/local/cuda`.
///
/// Kept in lockstep BY HAND with `crates/cuda-bindings/build.rs`
/// (`cuda_toolkit_dir` / `find_cuda_include_dir` / `toolkit_target_dir`):
/// doctor cannot import that probe because build.rs logic is not a library.
/// If the build.rs discovery changes, mirror it here.
fn cuda_toolkit_root(mut get_env: impl FnMut(&str) -> Option<String>) -> String {
    ["CUDA_TOOLKIT_PATH", "CUDA_HOME"]
        .iter()
        .find_map(|var| get_env(var).filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| "/usr/local/cuda".to_string())
}

/// Candidate `cuda.h` paths under `toolkit`, in probe order: the standard
/// `include/` layout first, then the redistributable `targets/<dir>/include`
/// layout. CUDA names the target dirs after the GPU platform, not the Rust
/// triple: x86_64 hosts use `x86_64-linux`, aarch64 servers use `sbsa-linux`.
///
/// `arch` is the host CPU architecture; the caller passes
/// `std::env::consts::ARCH` (doctor runs at runtime, so there is no cargo
/// `TARGET` to consult). Injected as a parameter for unit tests.
fn cuda_header_candidates(toolkit: &str, arch: &str) -> Vec<PathBuf> {
    let base = Path::new(toolkit);
    let mut candidates = vec![base.join("include/cuda.h")];
    let target_dir = match arch {
        "x86_64" => Some("x86_64-linux"),
        "aarch64" => Some("sbsa-linux"),
        _ => None,
    };
    if let Some(dir) = target_dir {
        candidates.push(base.join("targets").join(dir).join("include/cuda.h"));
    }
    candidates
}

// =============================================================================
// Clean command
// =============================================================================

pub fn clean(ctx: &Context) {
    match clean_context(ctx) {
        Ok(summary) if summary.removed_directories == 0 && summary.removed_files == 0 => {
            println!("Nothing to clean.");
        }
        Ok(summary) => {
            println!(
                "Removed {} directories and {} generated artifacts.",
                summary.removed_directories, summary.removed_files
            );
        }
        Err(error) => {
            eprintln!("Error: {error}");
            std::process::exit(1);
        }
    }
}

#[derive(Debug, Default, Eq, PartialEq)]
struct CleanSummary {
    removed_directories: usize,
    removed_files: usize,
}

fn clean_context(ctx: &Context) -> Result<CleanSummary, String> {
    let mut summary = CleanSummary::default();

    if ctx.is_workspace {
        clean_workspace(ctx, &mut summary)?;
    } else {
        clean_standalone_project(&ctx.workspace_root, &mut summary)?;
    }

    Ok(summary)
}

fn clean_standalone_project(project_dir: &Path, summary: &mut CleanSummary) -> Result<(), String> {
    let manifest_path = project_dir.join("Cargo.toml");
    let package_name = package_name_for_clean(&manifest_path)?;

    if remove_local_target(project_dir)? {
        summary.removed_directories += 1;
    }

    summary.removed_files += remove_generated_artifacts(project_dir, &package_name)?;

    Ok(())
}

fn clean_workspace(ctx: &Context, summary: &mut CleanSummary) -> Result<(), String> {
    if remove_local_target(&ctx.workspace_root)? {
        summary.removed_directories += 1;
    }

    if remove_local_target(&ctx.codegen_crate)? {
        summary.removed_directories += 1;
    }

    let entries = std::fs::read_dir(&ctx.examples_dir).map_err(|error| {
        format!(
            "could not read examples directory {}: {error}",
            ctx.examples_dir.display()
        )
    })?;

    let mut example_dirs = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|error| {
            format!(
                "could not read an entry in {}: {error}",
                ctx.examples_dir.display()
            )
        })?;

        let file_type = entry.file_type().map_err(|error| {
            format!(
                "could not inspect example entry {}: {error}",
                entry.path().display()
            )
        })?;

        if !file_type.is_dir() {
            continue;
        }

        let example_dir = entry.path();
        if example_dir.join("Cargo.toml").is_file() {
            example_dirs.push(example_dir);
        }
    }

    example_dirs.sort();

    for example_dir in example_dirs {
        clean_example(&example_dir, summary)?;
    }

    Ok(())
}

fn clean_example(example_dir: &Path, summary: &mut CleanSummary) -> Result<(), String> {
    let manifest_path = example_dir.join("Cargo.toml");
    let package_name = package_name_for_clean(&manifest_path)?;

    if remove_local_target(example_dir)? {
        summary.removed_directories += 1;
    }

    summary.removed_files += remove_generated_artifacts(example_dir, &package_name)?;

    Ok(())
}

fn package_name_for_clean(manifest_path: &Path) -> Result<String, String> {
    let source = std::fs::read_to_string(manifest_path).map_err(|error| {
        format!(
            "could not read manifest {}: {error}",
            manifest_path.display()
        )
    })?;

    let document: toml::Value = toml::from_str(&source).map_err(|error| {
        format!(
            "could not parse manifest {}: {error}",
            manifest_path.display()
        )
    })?;

    document
        .get("package")
        .and_then(|value| value.get("name"))
        .and_then(|value| value.as_str())
        .map(str::to_owned)
        .ok_or_else(|| {
            format!(
                "manifest {} is missing package.name",
                manifest_path.display()
            )
        })
}

fn remove_local_target(project_dir: &Path) -> Result<bool, String> {
    let target_dir = project_dir.join("target");

    let metadata = match std::fs::symlink_metadata(&target_dir) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(false);
        }
        Err(error) => {
            return Err(format!(
                "could not inspect {}: {error}",
                target_dir.display()
            ));
        }
    };

    if metadata.file_type().is_symlink() {
        return Err(format!(
            "refusing to remove symlinked target directory {}",
            target_dir.display()
        ));
    }

    if !metadata.is_dir() {
        return Err(format!(
            "expected {} to be a directory",
            target_dir.display()
        ));
    }

    std::fs::remove_dir_all(&target_dir).map_err(|error| {
        format!(
            "could not remove target directory {}: {error}",
            target_dir.display()
        )
    })?;

    println!("Removed {}", target_dir.display());

    Ok(true)
}

fn remove_generated_artifacts(project_dir: &Path, package_name: &str) -> Result<usize, String> {
    let mut removed = 0;

    for path in generated_artifact_paths(project_dir, package_name) {
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                continue;
            }
            Err(error) => {
                return Err(format!("could not inspect {}: {error}", path.display()));
            }
        };

        if metadata.file_type().is_symlink() {
            return Err(format!(
                "refusing to remove symlinked generated artifact {}",
                path.display()
            ));
        }

        if !metadata.is_file() {
            return Err(format!(
                "expected generated artifact {} to be a file",
                path.display()
            ));
        }

        std::fs::remove_file(&path)
            .map_err(|error| format!("could not remove {}: {error}", path.display()))?;

        println!("Removed {}", path.display());
        removed += 1;
    }

    Ok(removed)
}

// =============================================================================
// Setup command
// =============================================================================

/// Explicitly build (or rebuild) the codegen backend.
///
/// Normally the backend is built automatically on every `run`/`build`/`pipeline`
/// invocation. `setup` exists for first-time setup, CI, or after pulling new
/// changes when you want to rebuild without running an example.
pub fn setup(ctx: &Context) {
    println!("Building cuda-oxide codegen backend...");
    println!();

    let built_so = backend::build_backend_from_source(&ctx.codegen_crate);

    println!();
    println!("✓ Backend is ready. You can now use:");
    println!("  cargo oxide run <example>");
    println!("  cargo oxide build <example>");

    // A project outside this repository resolves the backend through the
    // shared cache, since `find_workspace_root` finds no
    // `crates/rustc-codegen-cuda` above it. Publishing the build there keeps
    // those projects on the backend that was just built instead of on whatever
    // the cache last held.
    match backend::publish_to_cache(&built_so) {
        Some(path) => {
            println!();
            println!("✓ Published to {}", path.display());
            println!("  Projects outside this repo will now use this build.");
        }
        None => {
            eprintln!();
            eprintln!("Warning: could not publish the backend to the shared cache.");
            eprintln!("Projects outside this repo may keep using an older build.");
            eprintln!("Set CUDA_OXIDE_BACKEND to this build to override.");
        }
    }
}

/// How `cargo oxide update` should behave for the current project mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdatePlan {
    /// Inside the monorepo: tell the user to run `setup` (non-destructive).
    AdviseSetup,
    /// Inside the monorepo with `--force`: rebuild via `setup`.
    RunSetup,
    /// Outside the monorepo: clear and rebuild the shared cache.
    RefreshCache,
}

pub fn plan_update(is_workspace: bool, force: bool) -> UpdatePlan {
    match (is_workspace, force) {
        (true, false) => UpdatePlan::AdviseSetup,
        (true, true) => UpdatePlan::RunSetup,
        (false, _) => UpdatePlan::RefreshCache,
    }
}

/// Refresh the codegen backend used by this project.
///
/// Inside the cuda-oxide workspace the authoritative backend is the local
/// source tree, so the default path points at `cargo oxide setup`. Outside
/// the workspace, the shared `~/.cargo/cuda-oxide/` cache is cleared and
/// rebuilt via the auto-fetch path.
/// The backend pin that outranks the shared cache `update` refreshes, if any.
///
/// Both the `CUDA_OXIDE_BACKEND` env var and a `.cargo/cuda-oxide.toml`
/// `backend` entry sit above the cache in backend discovery, so a refreshed
/// cache would never be consulted while either is set. `update` refuses
/// rather than mislead.
fn update_pin_refusal(ctx: &Context) -> Option<String> {
    update_pin_refusal_with_env(ctx, std::env::var_os("CUDA_OXIDE_BACKEND"))
}

/// `update_pin_refusal` with the ambient `CUDA_OXIDE_BACKEND` injected.
///
/// The env var is checked before the project pin, so resolution has to be
/// injectable for unit tests: a developer with `CUDA_OXIDE_BACKEND` exported
/// would otherwise get the env refusal for every input, including the
/// unpinned case that must return `None`. Same rationale as
/// `nvvm_ir_requested_with_env`.
fn update_pin_refusal_with_env(
    ctx: &Context,
    backend_env: Option<std::ffi::OsString>,
) -> Option<String> {
    if backend_env.is_some() {
        return Some(
            "CUDA_OXIDE_BACKEND is set, so `cargo oxide update` will not\n\
             modify the shared cache. Unset CUDA_OXIDE_BACKEND and re-run, or\n\
             rebuild the pinned backend path yourself."
                .to_string(),
        );
    }
    ctx.config.backend.as_deref().map(|pinned| {
        format!(
            "`.cargo/cuda-oxide.toml` pins the backend to {}, so\n\
             `cargo oxide update` will not modify the shared cache. Remove the\n\
             `backend` entry and re-run, or rebuild the pinned path yourself.",
            pinned.display()
        )
    })
}

pub fn update(ctx: &Context, force: bool) {
    if let Some(refusal) = update_pin_refusal(ctx) {
        eprintln!("Error: {refusal}");
        std::process::exit(1);
    }

    match plan_update(ctx.is_workspace, force) {
        UpdatePlan::AdviseSetup => {
            println!("Inside the cuda-oxide workspace the codegen backend is built from");
            println!("local source (`crates/rustc-codegen-cuda`).");
            println!();
            println!("Run `cargo oxide setup` to rebuild and publish to the shared cache,");
            println!("or pass `--force` to run setup from this command.");
        }
        UpdatePlan::RunSetup => {
            println!("`--force` requested inside the workspace; running setup...");
            println!();
            setup(ctx);
        }
        UpdatePlan::RefreshCache => {
            println!("Refreshing the shared codegen backend cache for external projects...");
            println!();
            let so = backend::refresh_cached_backend();
            println!();
            println!("✓ Cached backend ready at {}", so.display());
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Load `.cargo/cuda-oxide.toml`, exiting on an invalid config.
///
/// Build commands ([`resolve_context`]) stay strict: they must not run with
/// a config the user wrote but cargo-oxide cannot honor.
fn load_oxide_config(workspace_root: &Path) -> OxideConfig {
    match inspect_oxide_config(workspace_root) {
        OxideConfigInspection::Missing => OxideConfig::default(),
        OxideConfigInspection::Valid { config, warnings } => {
            for warning in warnings {
                eprintln!("Warning: {warning}");
            }
            config
        }
        OxideConfigInspection::Invalid { errors, warnings } => {
            for warning in warnings {
                eprintln!("Warning: {warning}");
            }
            for error in errors {
                eprintln!("Error: {error}");
            }
            std::process::exit(1);
        }
    }
}

/// Load `.cargo/cuda-oxide.toml`, falling back to defaults on an invalid
/// config instead of exiting.
///
/// Passive commands ([`resolve_passive_context`]: `doctor`, `clean`, ...)
/// must stay usable with a broken config. `doctor` in particular re-inspects
/// the file and reports the failure as a regular failed check, which it can
/// only do if context resolution survives long enough for the scan to start.
fn load_oxide_config_lenient(workspace_root: &Path) -> OxideConfig {
    match inspect_oxide_config(workspace_root) {
        OxideConfigInspection::Missing => OxideConfig::default(),
        OxideConfigInspection::Valid { config, warnings } => {
            for warning in warnings {
                eprintln!("Warning: {warning}");
            }
            config
        }
        OxideConfigInspection::Invalid { errors, warnings } => {
            for warning in warnings {
                eprintln!("Warning: {warning}");
            }
            for error in errors {
                eprintln!("Warning: {error}");
            }
            eprintln!("Warning: ignoring invalid cuda-oxide config and continuing with defaults");
            OxideConfig::default()
        }
    }
}

/// Result of reading `.cargo/cuda-oxide.toml` without exiting the process.
///
/// `doctor` uses this so a bad config is reported alongside other checks
/// instead of aborting before the rest of the environment scan runs.
#[derive(Debug)]
enum OxideConfigInspection {
    Missing,
    Valid {
        config: OxideConfig,
        warnings: Vec<String>,
    },
    Invalid {
        errors: Vec<String>,
        warnings: Vec<String>,
    },
}

fn inspect_oxide_config(workspace_root: &Path) -> OxideConfigInspection {
    let config_path = workspace_root.join(".cargo/cuda-oxide.toml");
    if !config_path.exists() {
        return OxideConfigInspection::Missing;
    }

    let source = match std::fs::read_to_string(&config_path) {
        Ok(source) => source,
        Err(error) => {
            return OxideConfigInspection::Invalid {
                errors: vec![format!(
                    "could not read cuda-oxide config {}: {error}",
                    config_path.display()
                )],
                warnings: Vec::new(),
            };
        }
    };

    let document: toml::Value = match toml::from_str(&source) {
        Ok(document) => document,
        Err(error) => {
            return OxideConfigInspection::Invalid {
                errors: vec![format!(
                    "could not parse cuda-oxide config {}: {error}",
                    config_path.display()
                )],
                warnings: Vec::new(),
            };
        }
    };

    let Some(table) = document.as_table() else {
        return OxideConfigInspection::Invalid {
            errors: vec![format!(
                "cuda-oxide config {} must be a TOML table",
                config_path.display()
            )],
            warnings: Vec::new(),
        };
    };

    let mut errors = Vec::new();
    let mut warnings = Vec::new();

    let backend = match optional_config_string(table, "backend", &config_path) {
        Ok(value) => value
            .map(PathBuf::from)
            .map(|path| absolutize_config_path(path, &config_path)),
        Err(error) => {
            errors.push(error);
            None
        }
    };
    let default_arch = match optional_config_string(table, "default-arch", &config_path) {
        Ok(value) => {
            if let Some(ref arch) = value {
                // Validate with the same parser the consumers use
                // (`parse_nvvm_arch` normalizes `sm_XX` / `compute_XX` / bare
                // `XX` into a `CudaArch`), so load-time validation is exactly
                // as permissive as what a build would accept. Non-`sm_XX`
                // spellings work but are advisory-warned: `sm_XX` is the form
                // `--arch` and the rest of cargo-oxide document.
                match parse_nvvm_arch(arch) {
                    Ok(parsed) => {
                        if !arch.starts_with("sm_") {
                            warnings.push(format!(
                                "cuda-oxide config {} spells `default-arch` as `{arch}`; \
                                 prefer the `{}` form used by `--arch`",
                                config_path.display(),
                                parsed.sm()
                            ));
                        }
                    }
                    Err(error) => {
                        errors.push(format!(
                            "cuda-oxide config {} field `default-arch`: {error}",
                            config_path.display()
                        ));
                    }
                }
            }
            value
        }
        Err(error) => {
            errors.push(error);
            None
        }
    };
    let extra_rustflags = match optional_config_string_array(table, "extra-rustflags", &config_path)
    {
        Ok(value) => value,
        Err(error) => {
            errors.push(error);
            Vec::new()
        }
    };
    let env = match table.get("env") {
        None => Vec::new(),
        Some(value) => match parse_config_env(value, &config_path) {
            Ok(env) => {
                for (key, _) in &env {
                    if matches!(key.as_str(), "RUSTFLAGS" | "CARGO_ENCODED_RUSTFLAGS") {
                        warnings.push(format!(
                            "cuda-oxide config {} `[env]` key `{key}` is ignored; \
                             use `extra-rustflags` for project rustc defaults",
                            config_path.display()
                        ));
                    }
                }
                env
            }
            Err(error) => {
                errors.push(error);
                Vec::new()
            }
        },
    };

    if !errors.is_empty() {
        return OxideConfigInspection::Invalid { errors, warnings };
    }

    OxideConfigInspection::Valid {
        config: OxideConfig {
            backend,
            default_arch,
            extra_rustflags,
            env,
        },
        warnings,
    }
}

/// Outcome of doctor's project-config check, separated from printing so
/// tests can assert the doctor-level behavior (headline, detail lines,
/// pass/fail) directly.
struct OxideConfigCheck {
    /// Line printed after the check label.
    headline: String,
    /// Indented detail lines (warnings, then errors).
    details: Vec<String>,
    /// Whether the check failed (flips doctor to a nonzero exit).
    failed: bool,
}

fn check_oxide_config(workspace_root: &Path) -> OxideConfigCheck {
    let config_path = workspace_root.join(".cargo/cuda-oxide.toml");
    match inspect_oxide_config(workspace_root) {
        OxideConfigInspection::Missing => OxideConfigCheck {
            headline: "- not present (using defaults)".to_string(),
            details: Vec::new(),
            failed: false,
        },
        OxideConfigInspection::Valid { config, warnings } => OxideConfigCheck {
            headline: match &config.default_arch {
                Some(arch) => format!("✓ {} (default-arch = {arch})", config_path.display()),
                None => format!("✓ {}", config_path.display()),
            },
            details: warnings
                .into_iter()
                .map(|warning| format!("⚠ {warning}"))
                .collect(),
            failed: false,
        },
        OxideConfigInspection::Invalid { errors, warnings } => OxideConfigCheck {
            headline: format!("✗ {}", config_path.display()),
            details: warnings
                .into_iter()
                .map(|warning| format!("⚠ {warning}"))
                .chain(errors.into_iter().map(|error| format!("✗ {error}")))
                .collect(),
            failed: true,
        },
    }
}

fn doctor_report_oxide_config(ctx: &Context, ok: &mut bool) {
    print!("Project config (.cargo/cuda-oxide.toml)... ");
    let check = check_oxide_config(&ctx.workspace_root);
    println!("{}", check.headline);
    for line in check.details {
        println!("  {line}");
    }
    if check.failed {
        *ok = false;
    }
}

fn absolutize_config_path(path: PathBuf, config_path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    config_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(path)
}

fn optional_config_string(
    table: &toml::Table,
    key: &str,
    config_path: &Path,
) -> Result<Option<String>, String> {
    match table.get(key) {
        None => Ok(None),
        Some(value) => value.as_str().map(|s| Some(s.to_string())).ok_or_else(|| {
            format!(
                "cuda-oxide config {} field `{key}` must be a string",
                config_path.display()
            )
        }),
    }
}

fn optional_config_string_array(
    table: &toml::Table,
    key: &str,
    config_path: &Path,
) -> Result<Vec<String>, String> {
    match table.get(key) {
        None => Ok(Vec::new()),
        Some(value) => {
            let array = value.as_array().ok_or_else(|| {
                format!(
                    "cuda-oxide config {} field `{key}` must be an array of strings",
                    config_path.display()
                )
            })?;
            array
                .iter()
                .map(|item| {
                    item.as_str().map(str::to_string).ok_or_else(|| {
                        format!(
                            "cuda-oxide config {} field `{key}` must be an array of strings",
                            config_path.display()
                        )
                    })
                })
                .collect()
        }
    }
}

fn parse_config_env(
    value: &toml::Value,
    config_path: &Path,
) -> Result<Vec<(String, String)>, String> {
    let table = value.as_table().ok_or_else(|| {
        format!(
            "cuda-oxide config {} field `env` must be a table of strings",
            config_path.display()
        )
    })?;
    let mut env: Vec<_> = table
        .iter()
        .map(|(key, value)| {
            let value = value.as_str().ok_or_else(|| {
                format!(
                    "cuda-oxide config {} env value `{key}` must be a string",
                    config_path.display()
                )
            })?;
            Ok((key.clone(), value.to_string()))
        })
        .collect::<Result<Vec<_>, String>>()?;
    env.sort_by(|left, right| left.0.cmp(&right.0));
    Ok(env)
}

fn load_interop_config(example_dir: &Path) -> Option<InteropConfig> {
    let manifest_path = example_dir.join("Cargo.toml");
    let source = std::fs::read_to_string(&manifest_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });
    let document: toml::Value = toml::from_str(&source).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not parse manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });

    let oxide = document
        .get("package")
        .and_then(|value| value.get("metadata"))
        .and_then(|value| value.get("cuda-oxide"))?;

    let kind = oxide.get("interop").and_then(|value| {
        value.as_str().map(str::to_string).or_else(|| {
            value
                .get("kind")
                .and_then(|kind| kind.as_str())
                .map(str::to_string)
        })
    });

    let device_crates = oxide
        .get("device-crates")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .map(|item| parse_device_crate_config(item, &manifest_path))
                .collect()
        })
        .unwrap_or_default();

    Some(InteropConfig {
        kind,
        device_crates,
    })
}

fn parse_device_crate_config(value: &toml::Value, manifest_path: &Path) -> DeviceCrateConfig {
    let table = value.as_table().unwrap_or_else(|| {
        eprintln!(
            "Error: each package.metadata.cuda-oxide.device-crates entry in {} must be a table",
            manifest_path.display()
        );
        std::process::exit(1);
    });

    let device_manifest = required_metadata_string(table, "manifest-path", manifest_path);
    let ptx_dir = optional_metadata_string(table, "ptx-dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(&device_manifest)
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        });
    let artifact_name = optional_metadata_string(table, "artifact-name");

    DeviceCrateConfig {
        manifest_path: PathBuf::from(device_manifest),
        ptx_dir,
        artifact_name,
    }
}

fn required_metadata_string(table: &toml::Table, key: &str, manifest_path: &Path) -> String {
    optional_metadata_string(table, key).unwrap_or_else(|| {
        eprintln!(
            "Error: package.metadata.cuda-oxide.device-crates entry in {} is missing string field `{}`",
            manifest_path.display(),
            key
        );
        std::process::exit(1);
    })
}

fn optional_metadata_string(table: &toml::Table, key: &str) -> Option<String> {
    table
        .get(key)
        .and_then(|value| value.as_str())
        .map(str::to_string)
}

fn package_name_from_manifest(manifest_path: &Path) -> String {
    let source = std::fs::read_to_string(manifest_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not read device manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });
    let document: toml::Value = toml::from_str(&source).unwrap_or_else(|e| {
        eprintln!(
            "Error: could not parse device manifest {}: {}",
            manifest_path.display(),
            e
        );
        std::process::exit(1);
    });

    document
        .get("package")
        .and_then(|value| value.get("name"))
        .and_then(|value| value.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| {
            eprintln!(
                "Error: device manifest {} is missing package.name",
                manifest_path.display()
            );
            std::process::exit(1);
        })
}

fn normalize_crate_name(package_name: &str) -> String {
    package_name.replace('-', "_")
}

/// Resolve an example name to its directory path, or exit with a list of
/// available examples if not found.
fn resolve_example_dir(ctx: &Context, example: &str) -> PathBuf {
    let example_dir = ctx.examples_dir.join(example);
    if !example_dir.exists() {
        eprintln!("Error: Example not found: {}", example_dir.display());
        eprintln!();
        eprintln!("Available examples:");
        if let Ok(entries) = std::fs::read_dir(&ctx.examples_dir) {
            let mut names: Vec<_> = entries
                .flatten()
                .filter(|e| e.path().is_dir())
                .map(|e| e.file_name().to_string_lossy().to_string())
                .collect();
            names.sort();
            for name in names {
                eprintln!("  - {}", name);
            }
        }
        std::process::exit(1);
    }
    example_dir
}

const ENCODED_RUSTFLAGS_SEPARATOR: char = '\u{1f}';

/// Profile-related rustc flags owned by cuda-oxide.
///
/// Backend selection and MIR/symbol invariants are always applied separately.
/// `CargoSelected` deliberately adds no optimization, assertion, or debug-info
/// flags so Cargo's chosen profile remains authoritative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CodegenProfilePolicy {
    CargoSelected,
    ReleaseLike,
    ReleaseLikeWithDebugInfo,
}

/// Construct boundary-preserving rustc flags for Cargo.
///
/// `RUSTFLAGS` is whitespace-split by Cargo, which corrupts a single flag
/// containing spaces. `CARGO_ENCODED_RUSTFLAGS` uses unit separators and keeps
/// every configured array element and `--device-cfg` value intact.
fn build_encoded_rustflags(
    ctx: &Context,
    profile: CodegenProfilePolicy,
    device_cfgs: &[String],
) -> String {
    let existing_encoded = std::env::var("CARGO_ENCODED_RUSTFLAGS").ok();
    let existing = std::env::var("RUSTFLAGS").ok();
    let mut explicit_rustflags = Vec::new();
    for cfg in device_cfgs {
        explicit_rustflags.push("--cfg".to_string());
        explicit_rustflags.push(cfg.clone());
    }
    build_encoded_rustflags_with_existing(
        &ctx.backend_so,
        profile,
        &ctx.config.extra_rustflags,
        &explicit_rustflags,
        existing_encoded.as_deref(),
        existing.as_deref(),
    )
}

fn build_encoded_rustflags_with_existing(
    backend_so: &Path,
    profile: CodegenProfilePolicy,
    configured_rustflags: &[String],
    explicit_rustflags: &[String],
    existing_encoded_rustflags: Option<&str>,
    existing_rustflags: Option<&str>,
) -> String {
    // Project flags are defaults, inherited flags are user overrides, and
    // explicit wrapper flags are stronger. cuda-oxide's compiler invariants
    // come last because rustc resolves repeated -C/-Z options last-one-wins.
    let mut flags = configured_rustflags.to_vec();

    if let Some(existing) = existing_encoded_rustflags {
        flags.extend(
            existing
                .split(ENCODED_RUSTFLAGS_SEPARATOR)
                .filter(|flag| !flag.is_empty())
                .map(str::to_string),
        );
    } else if let Some(existing) = existing_rustflags {
        // Match Cargo's legacy RUSTFLAGS behavior when converting it to the
        // encoded representation.
        flags.extend(existing.split_whitespace().map(str::to_string));
    }
    flags.extend(explicit_rustflags.iter().cloned());
    strip_wrapper_owned_codegen_cfgs(&mut flags);
    flags.push(format!("-Zcodegen-backend={}", backend_so.display()));
    if matches!(
        profile,
        CodegenProfilePolicy::ReleaseLike | CodegenProfilePolicy::ReleaseLikeWithDebugInfo
    ) {
        flags.extend([
            "-Copt-level=3".to_string(),
            "-Cdebug-assertions=off".to_string(),
        ]);
    }
    flags.extend([
        "-Zmir-enable-passes=-JumpThreading".to_string(),
        // Device codegen is whole-program: `collector` walks the call graph from
        // each `#[kernel]` and must emit every reachable dependency function into
        // one module. rustc encodes cross-crate MIR only for `#[inline]`/generic
        // items, so a non-`#[inline]`, non-generic dependency function that cannot
        // be inlined away (canonically: a recursive one) would be *called* but
        // never *defined* -> LLVM verification fails with "Symbol <crate>__<fn>
        // not found". Encode all MIR so any reachable dependency function is
        // device-compilable. This applies build-wide (like the other required
        // flags), so it also encodes MIR for host-only deps — an intentional,
        // interim trade (rmeta size) until a surgical device-dep-scoped or
        // per-crate device-link path lands. It matches the established approach
        // for whole-program-MIR tools (e.g. Miri).
        "-Zalways-encode-mir".to_string(),
        "-Csymbol-mangling-version=v0".to_string(),
    ]);
    if profile == CodegenProfilePolicy::ReleaseLikeWithDebugInfo {
        flags.push("-Cdebuginfo=2".to_string());
    }
    flags.join(&ENCODED_RUSTFLAGS_SEPARATOR.to_string())
}

fn strip_wrapper_owned_codegen_cfgs(flags: &mut Vec<String>) {
    fn is_wrapper_owned_cfg(value: &str) -> bool {
        [
            LEGACY_CODEGEN_FINGERPRINT_CFG,
            LEGACY_MATERIALIZER_PROVENANCE_CFG,
        ]
        .iter()
        .any(|name| {
            value
                .strip_prefix(name)
                .is_some_and(|suffix| suffix.is_empty() || suffix.starts_with('='))
        })
    }

    let mut retained = Vec::with_capacity(flags.len());
    let mut index = 0;
    while index < flags.len() {
        let flag = &flags[index];
        if flag == "--cfg"
            && flags
                .get(index + 1)
                .is_some_and(|value| is_wrapper_owned_cfg(value))
        {
            index += 2;
            continue;
        }
        if flag
            .strip_prefix("--cfg=")
            .is_some_and(is_wrapper_owned_cfg)
        {
            index += 1;
            continue;
        }
        retained.push(flag.clone());
        index += 1;
    }
    *flags = retained;
}

fn apply_codegen_rustflags(
    cmd: &mut Command,
    ctx: &Context,
    profile: CodegenProfilePolicy,
    device_cfgs: &[String],
) {
    cmd.env(
        "CARGO_ENCODED_RUSTFLAGS",
        build_encoded_rustflags(ctx, profile, device_cfgs),
    )
    .env_remove("RUSTFLAGS");
}

/// Apply the two deliberately different Cargo cache boundaries:
///
/// - the exact backend binary is global because it compiles every crate;
/// - mode/architecture/tool settings are an env dependency recorded only by
///   CUDA macros in crates that can own or instantiate device code.
fn apply_codegen_configuration(
    cmd: &mut Command,
    ctx: &Context,
    profile: CodegenProfilePolicy,
    user_device_cfgs: &[String],
    codegen_fingerprint: &str,
) -> Result<(), String> {
    let backend_digest = backend_artifact_digest(&ctx.backend_so)?;
    let mut global_cfgs = Vec::with_capacity(user_device_cfgs.len() + 1);
    global_cfgs.push(format!("{BACKEND_IDENTITY_CFG}=\"{backend_digest}\""));
    global_cfgs.extend(user_device_cfgs.iter().cloned());

    apply_codegen_rustflags(cmd, ctx, profile, &global_cfgs);
    cmd.env(CODEGEN_FINGERPRINT_ENV, codegen_fingerprint);
    Ok(())
}

fn apply_codegen_configuration_or_exit(
    cmd: &mut Command,
    ctx: &Context,
    profile: CodegenProfilePolicy,
    user_device_cfgs: &[String],
    codegen_fingerprint: &str,
) {
    apply_codegen_configuration(cmd, ctx, profile, user_device_cfgs, codegen_fingerprint)
        .unwrap_or_else(|error| {
            eprintln!("Error: {error}");
            std::process::exit(1);
        });
}

/// Set environment variables for the codegen backend.
///
/// `arch` is an explicit pin (`--arch`); it becomes `CUDA_OXIDE_TARGET`, the
/// hard override the backend honors as-is. The auto-detected GPU arch is *not*
/// routed here -- see [`apply_device_arch_hint`].
fn apply_output_mode(
    cmd: &mut Command,
    emit_nvvm_ir: bool,
    arch: Option<&str>,
    materialization: &MaterializationMode,
) {
    if let Some(target_arch) = arch {
        cmd.env("CUDA_OXIDE_TARGET", target_arch);
    }
    if emit_nvvm_ir || materialization.enabled() {
        cmd.env("CUDA_OXIDE_EMIT_NVVM_IR", "1");
    }
    materialization.apply(cmd);
}

fn configured_arch<'a>(ctx: &'a Context, cli_arch: Option<&'a str>) -> Option<&'a str> {
    if cli_arch.is_some() || std::env::var("CUDA_OXIDE_TARGET").is_ok() {
        cli_arch
    } else {
        ctx.config
            .default_arch
            .as_deref()
            .or_else(|| project_config_env(ctx, "CUDA_OXIDE_TARGET"))
    }
}

fn configured_arch_label(ctx: &Context, cli_arch: Option<&str>) -> Option<String> {
    cli_arch
        .map(str::to_string)
        .or_else(|| std::env::var("CUDA_OXIDE_TARGET").ok())
        .or_else(|| ctx.config.default_arch.clone())
        .or_else(|| project_config_env(ctx, "CUDA_OXIDE_TARGET").map(str::to_string))
}

pub fn has_configured_arch(ctx: &Context, cli_arch: Option<&str>) -> bool {
    cli_arch.is_some()
        || std::env::var("CUDA_OXIDE_TARGET").is_ok()
        || ctx.config.default_arch.is_some()
        || project_config_env(ctx, "CUDA_OXIDE_TARGET").is_some()
}

fn apply_config_env(cmd: &mut Command, ctx: &Context) {
    for (key, value) in &ctx.config.env {
        if matches!(key.as_str(), "RUSTFLAGS" | "CARGO_ENCODED_RUSTFLAGS") {
            continue;
        }
        // Project values are defaults. An explicitly inherited environment is
        // stronger, and command-specific CLI/internal settings are applied
        // after this helper and are stronger still.
        if std::env::var_os(key).is_none() {
            cmd.env(key, value);
        }
    }
}

fn apply_common_codegen_env(
    cmd: &mut Command,
    ctx: &Context,
    verbose: bool,
    no_fmad: bool,
    unchecked_indexing: bool,
    device_debug: DeviceDebug,
) {
    apply_config_env(cmd, ctx);
    if verbose {
        cmd.env("CUDA_OXIDE_VERBOSE", "1");
    }
    if no_fmad {
        cmd.env("CUDA_OXIDE_NO_FMA", "1");
    }
    if unchecked_indexing {
        cmd.env("CUDA_OXIDE_UNCHECKED_INDEXING", "1");
    }
    // An explicit flag outranks an ambient `CUDA_OXIDE_DEBUG`, matching how
    // `--no-fmad` outranks `CUDA_OXIDE_NO_FMA`. `DeviceDebug::Off` exports
    // nothing rather than `off`, so omitting the flag cannot silently cancel a
    // debug level the environment or project config already asked for.
    if let Some(level) = device_debug.env_value() {
        cmd.env("CUDA_OXIDE_DEBUG", level);
    }
    apply_ld_library_path(cmd, ctx);
}

/// Give Compute Sanitizer source line attribution without disabling normal
/// device optimization. An explicit process or project setting remains
/// authoritative, including an intentional `CUDA_OXIDE_DEBUG=off`. So does an
/// explicit `--lineinfo` / `--device-debug` flag: `apply_common_codegen_env`
/// has already exported its level onto `cmd`, and the default must not
/// overwrite it.
fn apply_default_sanitizer_line_tables(
    cmd: &mut Command,
    ctx: &Context,
    device_debug: DeviceDebug,
) {
    apply_default_sanitizer_line_tables_with_env(
        cmd,
        ctx,
        std::env::var_os("CUDA_OXIDE_DEBUG").is_some(),
        device_debug,
    );
}

/// `apply_default_sanitizer_line_tables` with the `CUDA_OXIDE_DEBUG` probe
/// injected.
///
/// `env_debug_set` is presence-only, matching the `var_os` check it replaces.
/// Injected so a unit test can assert the defaulting without an exported
/// `CUDA_OXIDE_DEBUG` suppressing it. `device_debug` carries the CLI flag:
/// any level other than [`DeviceDebug::Off`] is an explicit request that
/// outranks the line-tables default.
fn apply_default_sanitizer_line_tables_with_env(
    cmd: &mut Command,
    ctx: &Context,
    env_debug_set: bool,
    device_debug: DeviceDebug,
) {
    if device_debug == DeviceDebug::Off
        && !env_debug_set
        && project_config_env(ctx, "CUDA_OXIDE_DEBUG").is_none()
    {
        cmd.env("CUDA_OXIDE_DEBUG", "line-tables");
    }
}

fn apply_interop_device_codegen_options(
    cmd: &mut Command,
    ctx: &Context,
    verbose: bool,
    options: InteropDeviceBuildOptions,
) {
    apply_interop_device_codegen_options_with_env(
        cmd,
        ctx,
        verbose,
        options,
        std::env::var_os("CUDA_OXIDE_DEBUG").is_some(),
    );
}

/// `apply_interop_device_codegen_options` with the `CUDA_OXIDE_DEBUG` probe
/// injected, forwarded to `apply_default_sanitizer_line_tables_with_env`.
fn apply_interop_device_codegen_options_with_env(
    cmd: &mut Command,
    ctx: &Context,
    verbose: bool,
    options: InteropDeviceBuildOptions,
    env_debug_set: bool,
) {
    apply_common_codegen_env(
        cmd,
        ctx,
        verbose,
        options.no_fmad,
        options.unchecked_indexing,
        DeviceDebug::Off,
    );
    if options.sanitizer_line_tables {
        apply_default_sanitizer_line_tables_with_env(cmd, ctx, env_debug_set, DeviceDebug::Off);
    }
}

/// Forward the auto-detected GPU arch as a *hint* via `CUDA_OXIDE_DEVICE_ARCH`.
///
/// Unlike `CUDA_OXIDE_TARGET` (a hard override), this is advisory: the backend
/// builds for the detected GPU only when that GPU can actually run the kernel.
/// If the kernel needs a newer arch (e.g. tcgen05 / cta_group TMA multicast
/// need sm_100a, which a consumer sm_120 GPU lacks), the backend builds for the
/// required arch instead. Skipped when the user pinned `--arch` (that explicit
/// choice already went to `CUDA_OXIDE_TARGET`).
fn apply_device_arch_hint(
    cmd: &mut Command,
    explicit_arch: Option<&str>,
    detected_device_arch: Option<&str>,
) {
    if let (None, Some(dev)) = (explicit_arch, detected_device_arch) {
        cmd.env("CUDA_OXIDE_DEVICE_ARCH", dev);
    }
}

/// Pick a runnable target for `cargo oxide run` when the user has not pinned
/// one explicitly.
///
/// # Precedence
///
/// `cargo oxide run` resolves the target architecture in this order, highest
/// priority first:
///
/// 1. `--arch <sm_XX>`            (explicit user override)
/// 2. `CUDA_OXIDE_TARGET=<sm_XX>` (explicit env override, set in the parent
///    process before invoking `cargo oxide run`)
/// 3. **This function**: the compute capability of the first GPU reported by
///    `nvidia-smi`, forwarded as the `CUDA_OXIDE_DEVICE_ARCH` *hint*. Emits
///    the arch-specific `sm_XYa` form for cc >= 9.0 (so the backend can lower
///    WGMMA / tcgen05 / TMA-multicast when the GPU supports them) and the
///    plain `sm_XY` form for cc < 9.0.
/// 4. Backend feature-based default (`select_target` in
///    `mir-importer::pipeline`), which picks the minimum `sm_XX` required by
///    the IR shape (e.g. `Basic -> sm_80`, `Cluster -> sm_90`, `Tma -> sm_100`).
///
/// Slot 3 is advisory: the backend builds for the detected GPU only when that
/// GPU can run the kernel, otherwise it falls back to slot 4 (the arch the
/// kernel requires). This function returns `Some(sm_XY[a])` to fill slot 3, or
/// `None` (falling through to slot 4) when the machine has no usable GPU.
///
/// # Why only `run`
///
/// `run` immediately loads the generated module on the local GPU and launches
/// the kernel, so a target older than the local GPU's compute capability is
/// the only safe default. `build` and `pipeline` may legitimately
/// cross-compile to a different machine, so they keep the backend's
/// feature-based default untouched.
///
/// # Why this is needed even with the backend default
///
/// The backend's `select_target` picks the minimum `sm_XX` the IR requires.
/// `Basic → sm_80` is a fine *compilation* baseline, but PTX for `sm_80` will
/// not load on a Turing (`sm_75`) GPU because the JIT refuses
/// forward-incompatible PTX. Detecting the device CC in `run` keeps the
/// generated module loadable on the actual hardware that will execute it.
///
/// # When this returns `None`
///
/// - The user passed `--arch` (slot 1 wins).
/// - `CUDA_OXIDE_TARGET` is set in the environment (slot 2 wins).
/// - `--emit-nvvm-ir` is in effect (NVVM IR mode requires explicit `--arch`,
///   enforced by the CLI parser).
/// - No CUDA driver / GPU is available on the machine (CI runners without
///   GPUs, headless build boxes), or `nvidia-smi` is missing or broken. The
///   caller falls through to slot 4 and the backend's feature-based default
///   applies.
fn detect_run_target_arch(arch: Option<&str>, emit_nvvm_ir: bool) -> Option<String> {
    detect_run_target_arch_with_env(
        arch,
        emit_nvvm_ir,
        std::env::var_os("CUDA_OXIDE_TARGET").is_some(),
    )
}

/// `detect_run_target_arch` with the `CUDA_OXIDE_TARGET` probe injected.
///
/// `env_target_set` is presence-only, matching the `var_os` check it replaces.
/// Injected so a unit test can exercise the slot-2 skip without exporting the
/// variable: `set_var` would be a data race against the `vars_os` reads the
/// fingerprint helpers perform on other test threads.
fn detect_run_target_arch_with_env(
    arch: Option<&str>,
    emit_nvvm_ir: bool,
    env_target_set: bool,
) -> Option<String> {
    if arch.is_some() || emit_nvvm_ir || env_target_set {
        return None;
    }

    query_device_compute_cap().map(format_sm_arch)
}

/// Query the compute capability of the first GPU via `nvidia-smi`.
///
/// Runs `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and parses
/// the first output line. A subprocess probe (rather than the CUDA driver
/// API) keeps cargo-oxide free of any link-time or dlopen dependency on
/// `libcuda`, so the subcommand builds and runs on machines with no CUDA
/// toolkit and no driver; `scripts/smoketest.sh` derives `sm_XX` from
/// `nvidia-smi` the same way.
///
/// Caveat: `nvidia-smi` enumerates GPUs in PCI bus order, while CUDA's
/// default device order is fastest-first, so on heterogeneous multi-GPU
/// machines this may describe a different GPU than CUDA device 0. That is
/// safe because `CUDA_OXIDE_DEVICE_ARCH` is advisory (the backend only
/// honors a compatible hint) and `--arch` / `CUDA_OXIDE_TARGET` remain hard
/// overrides.
fn query_device_compute_cap() -> Option<(u32, u32)> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    parse_compute_cap(&String::from_utf8_lossy(&output.stdout))
}

/// Parse the first line of `nvidia-smi --query-gpu=compute_cap` output as a
/// `(major, minor)` compute-capability pair. Returns `None` for anything
/// that is not shaped `<digits>.<digits>`.
fn parse_compute_cap(stdout: &str) -> Option<(u32, u32)> {
    parse_compute_cap_field(stdout.lines().next()?)
}

/// Parse a single `compute_cap` CSV field (e.g. `"12.0"`).
///
/// Only the `<digits>.<digits>` shape is accepted: `nvidia-smi` prints its
/// failure banners ("NVIDIA-SMI has failed ...") to *stdout*, sometimes with
/// exit status 0, so this shape check is the real gate, not the exit status.
fn parse_compute_cap_field(field: &str) -> Option<(u32, u32)> {
    let (major, minor) = field.trim().split_once('.')?;
    let all_digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    if !all_digits(major) || !all_digits(minor) {
        return None;
    }
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Query the name and compute capability of the first GPU via `nvidia-smi`,
/// for doctor's driver / GPU report. Same trust rules as
/// [`query_device_compute_cap`].
fn query_gpu_name_and_compute_cap() -> Option<(String, (u32, u32))> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name,compute_cap", "--format=csv,noheader"])
        .output()
        .ok()
        .filter(|o| o.status.success())?;
    parse_gpu_name_and_compute_cap(&String::from_utf8_lossy(&output.stdout))
}

/// Parse the first line of `nvidia-smi --query-gpu=name,compute_cap` output
/// into the GPU name and `(major, minor)` pair. Splits on the LAST comma:
/// GPU names may contain commas in principle, `compute_cap` never does.
fn parse_gpu_name_and_compute_cap(stdout: &str) -> Option<(String, (u32, u32))> {
    let line = stdout.lines().next()?;
    let (name, cap) = line.rsplit_once(',')?;
    Some((name.trim().to_string(), parse_compute_cap_field(cap)?))
}

/// Format a `(major, minor)` compute-capability tuple as the `sm_XX` /
/// `sm_XXX[a]` string the codegen backend expects on `CUDA_OXIDE_TARGET`.
///
/// Concatenates without a separator, matching CUDA conventions:
/// `(7, 5)` → `"sm_75"`, `(12, 0)` → `"sm_120a"`.
///
/// # Arch-specific (`a`) suffix
///
/// Compute capability ≥ 9.0 always has an arch-specific PTX target (`sm_90a`,
/// `sm_100a`, `sm_103a`, `sm_120a`, …) that is a strict superset of the plain
/// target on that chip. The `a` form is what unlocks WGMMA on Hopper and
/// `tcgen05` / TMA multicast / `cta_group::*` on Blackwell datacenter — and
/// every chip that reports cc ≥ 9.0 *is* the `a`-variant chip in NVIDIA's
/// lineup (there is no consumer Hopper, no non-`a` sm_100, and so on).
///
/// This helper is only used by [`detect_run_target_arch`] in `cargo oxide
/// run`, where the local GPU is known exactly and no cross-compile is in
/// flight. Emitting the `a` form there:
///
/// - **No false negatives:** kernels that need `tcgen05` / WGMMA compile and
///   load on that GPU (was: silent fallback to `sm_100` / `sm_90` and a
///   `ptxas: 'tcgen05.alloc' not supported on .target 'sm_100'` failure).
/// - **No false positives:** cc < 9.0 keeps the plain `sm_XY` form, since
///   there is no `sm_80a` / `sm_86a` / `sm_89a` target in the PTX ISA.
/// - **Strict superset:** PTX targeting `sm_XYa` accepts every kernel that
///   would have compiled for plain `sm_XY`; the `a` form only permits
///   *additional* arch-specific intrinsics.
fn format_sm_arch((major, minor): (u32, u32)) -> String {
    if major >= 9 {
        format!("sm_{}{}a", major, minor)
    } else {
        format!("sm_{}{}", major, minor)
    }
}

fn inherited_or_configured_env(ctx: &Context, key: &str) -> Option<String> {
    std::env::var(key).ok().or_else(|| {
        ctx.config
            .env
            .iter()
            .find(|(configured_key, _)| configured_key == key)
            .map(|(_, value)| value.clone())
    })
}

/// Build `LD_LIBRARY_PATH` for the child cargo process.
///
/// Includes the rustc sysroot lib (for `librustc_driver.so` etc.), the
/// libmathdx lib (when `LIBMATHDX_PATH` is set), and any existing
/// `LD_LIBRARY_PATH` from the parent environment.
fn apply_ld_library_path(cmd: &mut Command, ctx: &Context) {
    let mut ld_paths: Vec<String> = Vec::new();
    if let Some(sysroot) = backend::get_rustc_sysroot() {
        ld_paths.push(format!("{}/lib", sysroot));
    }
    if let Some(libmathdx_path) = inherited_or_configured_env(ctx, "LIBMATHDX_PATH") {
        ld_paths.push(format!("{}/lib", libmathdx_path));
    }
    if let Some(existing) = inherited_or_configured_env(ctx, "LD_LIBRARY_PATH") {
        ld_paths.push(existing);
    }
    if !ld_paths.is_empty() {
        cmd.env("LD_LIBRARY_PATH", ld_paths.join(":"));
    }
}

/// Touch main.rs to force recompilation (faster than cargo clean).
fn touch_main_rs(example_dir: &Path) {
    // Force a rebuild so the codegen backend re-runs and emits a fresh
    // .ptx alongside the example. Touch every source file that might
    // host `#[kernel]` items so multi-bin layouts (kernels in `lib.rs`,
    // tests in `main.rs`, perf bench in `bin/<name>.rs`, etc.) all
    // re-codegen on every `cargo oxide run/build` invocation.
    for rel in ["src/main.rs", "src/lib.rs"] {
        let path = example_dir.join(rel);
        if path.exists()
            && let Ok(content) = std::fs::read(&path)
        {
            let _ = std::fs::write(&path, content);
        }
    }
}

/// Artifacts are named after the crate, and cargo normalizes hyphens in
/// package names to underscores (`rustlantis-smoke` emits
/// `rustlantis_smoke.ptx`). Always go through this when deriving an
/// artifact filename from an example name, or hyphenated examples keep
/// stale artifacts forever.
fn artifact_stem(example: &str) -> String {
    example.replace('-', "_")
}

/// Return the PTX artifacts generated for a regular or metadata-interop project.
fn ptx_artifact_paths(example_dir: &Path, example: &str) -> Vec<PathBuf> {
    if let Some(interop) =
        load_interop_config(example_dir).filter(|config| !config.device_crates.is_empty())
    {
        return interop
            .device_crates
            .iter()
            .map(|device_crate| {
                let manifest_path = example_dir.join(&device_crate.manifest_path);
                let artifact_name = interop_device_artifact_name(&manifest_path, device_crate);

                interop_device_ptx_path(example_dir, device_crate, &artifact_name)
            })
            .collect();
    }

    let stem = artifact_stem(example);
    vec![example_dir.join(format!("{stem}.ptx"))]
}

fn read_ptx_artifact(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path)
        .map_err(|error| format!("could not read generated PTX {}: {error}", path.display()))
}

/// Print one generated PTX artifact.
fn print_ptx_artifact(path: &Path) -> Result<(), String> {
    let content = read_ptx_artifact(path)?;

    let name = path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string());

    println!();
    println!("=========================================");
    println!("PTX ({name})");
    println!("=========================================");
    print!("{content}");

    if !content.ends_with('\n') {
        println!();
    }

    Ok(())
}

/// Path to the NVVM IR (`.ll`) the backend emits for `example`. Named after the
/// Cargo-normalized crate stem, so a hyphenated example resolves to the
/// underscore-spelled file the build actually wrote. Route `emit-ltoir` reads
/// through here rather than deriving the name from the raw example.
fn emitted_ll_path(example_dir: &Path, example: &str) -> PathBuf {
    example_dir.join(format!("{}.ll", artifact_stem(example)))
}

/// Default LTOIR output path for `example` when no explicit `--output` is given.
/// Uses the same Cargo-normalized crate stem as [`emitted_ll_path`] so reads and
/// writes agree on hyphenated examples.
fn default_ltoir_path(example_dir: &Path, example: &str) -> PathBuf {
    example_dir.join(format!("{}.ltoir", artifact_stem(example)))
}

const GENERATED_ARTIFACT_SUFFIXES: &[&str] = &[
    "ptx",
    "ll",
    "opt.ll",
    "ltoir",
    "cubin",
    "target",
    "options",
    "cubin.target",
];

fn generated_artifact_paths(project_dir: &Path, package_name: &str) -> Vec<PathBuf> {
    let stem = artifact_stem(package_name);

    GENERATED_ARTIFACT_SUFFIXES
        .iter()
        .map(|suffix| project_dir.join(format!("{stem}.{suffix}")))
        .collect()
}

/// Remove stale generated artifacts (`.ptx`, `.ll`, `.ltoir`, `.cubin`) from a
/// previous run so we can verify the build produces fresh output.
fn clean_generated_files(example_dir: &Path, example: &str) {
    for file in generated_artifact_paths(example_dir, example) {
        if file.exists() {
            let _ = std::fs::remove_file(file);
        }
    }
}

/// Human-readable label for the selected output format.
fn format_label(emit_nvvm_ir: bool) -> &'static str {
    if emit_nvvm_ir { "NVVM IR" } else { "PTX" }
}

/// Print generated artifacts (LLVM IR or PTX) to stdout after a pipeline build.
fn show_generated_artifacts(example_dir: &Path, example: &str) {
    let stem = artifact_stem(example);
    let ll_file = example_dir.join(format!("{}.ll", stem));
    let ptx_file = example_dir.join(format!("{}.ptx", stem));

    if ll_file.exists() {
        println!();
        println!("=========================================");
        println!("LLVM IR ({}.ll)", stem);
        println!("=========================================");
        if let Ok(content) = std::fs::read_to_string(&ll_file) {
            println!("{}", content);
        }
    }

    if ptx_file.exists() {
        println!();
        println!("=========================================");
        println!("PTX ({}.ptx)", stem);
        println!("=========================================");
        if let Ok(content) = std::fs::read_to_string(&ptx_file) {
            println!("{}", content);
        }
    }
}

// =========================================================================
// cargo oxide new -- standalone project scaffolding
// =========================================================================

const GIT_REPO: &str = "https://github.com/NVlabs/cuda-oxide.git";

const RUST_TOOLCHAIN_TOML: &str = r#"[toolchain]
channel = "nightly-2026-04-03"
components = ["rust-src", "rustc-dev", "rust-analyzer", "clippy", "llvm-tools"]
"#;

const SCAFFOLD_GITIGNORE_EXTRA: &[&str] = &[
    "/target/",
    "**/*.bc", // bitcode leftovers not in the clean suffix list
    ".DS_Store",
];

fn scaffold_gitignore() -> String {
    let mut lines: Vec<String> = SCAFFOLD_GITIGNORE_EXTRA
        .iter()
        .map(|line| (*line).to_string())
        .collect();
    // Keep in lockstep with `GENERATED_ARTIFACT_SUFFIXES` so `cargo oxide new`
    // ignores every artifact `cargo oxide clean` knows how to delete.
    for suffix in GENERATED_ARTIFACT_SUFFIXES {
        let pattern = format!("**/*.{suffix}");
        if !lines.iter().any(|line| line == &pattern) {
            lines.push(pattern);
        }
    }
    // Stable order for readable diffs: keep the three fixed entries first,
    // then sort generated patterns.
    let (fixed, rest) = lines.split_at(SCAFFOLD_GITIGNORE_EXTRA.len());
    let mut rest = rest.to_vec();
    rest.sort();
    let mut out = fixed.to_vec();
    out.append(&mut rest);
    out.push(String::new());
    out.join("\n")
}

/// File contents produced by `cargo oxide new`.
#[derive(Clone, Debug, Eq, PartialEq)]
struct ScaffoldFiles {
    cargo_toml: String,
    rust_toolchain_toml: String,
    gitignore: String,
    readme: String,
    main_rs: String,
}

fn scaffold_readme(name: &str, async_mode: bool) -> String {
    let mode = if async_mode {
        "async cuda-oxide"
    } else {
        "cuda-oxide"
    };
    let template_notes = if async_mode {
        "The template is a vector-add kernel launched through `cuda-async`:\n\
         `vecadd_async` returns a lazy `DeviceOperation` scheduled on the\n\
         stream pool. See the cuda-oxide book getting-started chapter for the\n\
         next steps."
    } else {
        "The template is a vector-add kernel. It uses `#[launch_contract]` and\n\
         `PreparedLaunch` so geometry is checked before launch. See the\n\
         cuda-oxide book getting-started chapter for the next steps."
    };
    format!(
        r#"# {name}

Scaffolded {mode} project.

## Setup

```bash
cargo oxide doctor
```

Fix anything doctor reports before building.

## Run

```bash
cargo oxide run
```

{template_notes}
"#
    )
}

fn scaffold_cargo_toml(name: &str, async_mode: bool) -> String {
    if async_mode {
        format!(
            r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"

[workspace]

[dependencies]
cuda-device = {{ git = "{GIT_REPO}" }}
cuda-host = {{ git = "{GIT_REPO}", features = ["async"] }}
cuda-core = {{ git = "{GIT_REPO}" }}
cuda-async = {{ git = "{GIT_REPO}" }}
cuda-bindings = {{ git = "{GIT_REPO}" }}
tokio = {{ version = "1", features = ["rt", "rt-multi-thread", "macros"] }}
"#
        )
    } else {
        format!(
            r#"[package]
name = "{name}"
version = "0.1.0"
edition = "2024"

[workspace]

[dependencies]
cuda-device = {{ git = "{GIT_REPO}" }}
cuda-host = {{ git = "{GIT_REPO}" }}
cuda-core = {{ git = "{GIT_REPO}" }}
"#
        )
    }
}

fn scaffold_main_rs(async_mode: bool) -> String {
    if async_mode {
        r#"use cuda_device::{kernel, thread, DisjointSlice};
use cuda_host::cuda_module;
use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::DeviceOperation;
use cuda_core::LaunchConfig;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use cuda_async::device_box::DeviceBox;
    use cuda_core::memory::{malloc_async, memcpy_dtoh_async, memcpy_htod_async};
    use std::mem;

    init_device_contexts(0, 1)?;
    let module = kernels::load_async(0)?;

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let (a_dev, b_dev, mut c_dev) = cuda_async::device_context::with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        let num_bytes = N * mem::size_of::<f32>();
        unsafe {
            let a = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            let b = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            let c = malloc_async(stream.cu_stream(), num_bytes).unwrap();
            memcpy_htod_async(a, a_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            memcpy_htod_async(b, b_host.as_ptr(), num_bytes, stream.cu_stream()).unwrap();
            stream.synchronize().unwrap();
            (
                DeviceBox::<[f32]>::from_raw_parts(a, N, 0),
                DeviceBox::<[f32]>::from_raw_parts(b, N, 0),
                DeviceBox::<[f32]>::from_raw_parts(c, N, 0),
            )
        }
    })?;

    // SAFETY: this is a 1D launch and `vecadd` guards its index against the
    // output length before writing.
    unsafe {
        module.vecadd_async(
            LaunchConfig::for_num_elems(N as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )
    }?
    .sync()?;

    let mut c_host = vec![0.0f32; N];
    cuda_async::device_context::with_cuda_context(0, |ctx| {
        let stream = ctx.default_stream();
        unsafe {
            memcpy_dtoh_async(
                c_host.as_mut_ptr(),
                c_dev.cu_deviceptr(),
                N * mem::size_of::<f32>(),
                stream.cu_stream(),
            )
            .unwrap();
            stream.synchronize().unwrap();
        }
    })?;

    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }

    Ok(())
}
"#
        .to_string()
    } else {
        r#"use cuda_device::{kernel, launch_bounds, launch_contract, thread, DisjointSlice};
use cuda_host::cuda_module;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
    let b_dev = DeviceBuffer::from_host(&stream, &b_host)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // kernels module above.
    let module = unsafe { kernels::load(&ctx)? };
    let prepared = module.prepare_vecadd(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;
    module.vecadd(&stream, &prepared, &a_dev, &b_dev, &mut c_dev)?;

    let c_host = c_dev.to_host_vec(&stream)?;
    let errors = (0..N)
        .filter(|&i| (c_host[i] - (a_host[i] + b_host[i])).abs() > 1e-5)
        .count();

    if errors == 0 {
        println!("PASSED: all {} elements correct", N);
    } else {
        eprintln!("FAILED: {} errors", errors);
        std::process::exit(1);
    }
    Ok(())
}
"#
        .to_string()
    }
}

fn scaffold_files(name: &str, async_mode: bool) -> ScaffoldFiles {
    ScaffoldFiles {
        cargo_toml: scaffold_cargo_toml(name, async_mode),
        rust_toolchain_toml: RUST_TOOLCHAIN_TOML.to_string(),
        gitignore: scaffold_gitignore(),
        readme: scaffold_readme(name, async_mode),
        main_rs: scaffold_main_rs(async_mode),
    }
}

/// Scaffold a new standalone cuda-oxide project.
pub fn scaffold_new(name: &str, async_mode: bool) {
    let project_dir = PathBuf::from(name);
    if project_dir.exists() {
        eprintln!("Error: directory '{}' already exists.", name);
        std::process::exit(1);
    }

    let src_dir = project_dir.join("src");
    std::fs::create_dir_all(&src_dir).unwrap_or_else(|e| {
        eprintln!("Error creating directory: {}", e);
        std::process::exit(1);
    });

    let files = scaffold_files(name, async_mode);
    std::fs::write(project_dir.join("Cargo.toml"), files.cargo_toml)
        .expect("Failed to write Cargo.toml");
    std::fs::write(
        project_dir.join("rust-toolchain.toml"),
        files.rust_toolchain_toml,
    )
    .expect("Failed to write rust-toolchain.toml");
    std::fs::write(project_dir.join(".gitignore"), files.gitignore)
        .expect("Failed to write .gitignore");
    std::fs::write(project_dir.join("README.md"), files.readme).expect("Failed to write README.md");
    std::fs::write(src_dir.join("main.rs"), files.main_rs).expect("Failed to write src/main.rs");

    let mode = if async_mode { " (async)" } else { "" };
    println!("✓ Created cuda-oxide project '{}'{}", name, mode);
    println!();
    println!("  cd {}", name);
    println!("  cargo oxide doctor");
    println!("  cargo oxide run");
}

/// Locate an executable by name, first via `which` (PATH lookup), then by
/// checking a list of common fallback absolute paths.
fn find_executable(name: &str, fallback_paths: &[&str]) -> Option<PathBuf> {
    if let Ok(output) = Command::new("which").arg(name).output()
        && output.status.success()
    {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    for path in fallback_paths {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Locate a CUDA Toolkit executable using the same configured toolkit roots as
/// `doctor`, after the user's PATH and before generic system fallbacks.
fn find_cuda_toolkit_executable(
    ctx: &Context,
    name: &str,
    fallback_paths: &[&str],
) -> Option<PathBuf> {
    find_cuda_toolkit_executable_with_env(ctx, name, fallback_paths, |key| std::env::var(key).ok())
}

/// `find_cuda_toolkit_executable` with the ambient environment injected.
///
/// The process environment takes precedence over `cuda-oxide.toml`'s `env`, so
/// resolution has to be injectable for unit tests: a developer with a real
/// `CUDA_TOOLKIT_PATH` (or `CUDA_HOME`) exported would otherwise shadow the
/// configured root a test is trying to assert on. Same rationale as
/// `cuda_toolkit_root` and `cuda_header_candidates`.
fn find_cuda_toolkit_executable_with_env(
    ctx: &Context,
    name: &str,
    fallback_paths: &[&str],
    mut get_env: impl FnMut(&str) -> Option<String>,
) -> Option<PathBuf> {
    if let Some(path) = find_executable(name, &[]) {
        return Some(path);
    }

    let toolkit = cuda_toolkit_root(|key| {
        get_env(key)
            .filter(|value| !value.trim().is_empty())
            .or_else(|| project_config_env(ctx, key).map(str::to_owned))
    });
    let configured = PathBuf::from(toolkit).join("bin").join(name);
    if configured.exists() {
        return Some(configured);
    }

    for path in fallback_paths {
        let path = PathBuf::from(path);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    fn command_env(cmd: &Command, key: &str) -> Option<String> {
        cmd.get_envs()
            .find(|(name, _)| *name == OsStr::new(key))
            .and_then(|(_, value)| value.map(|v| v.to_string_lossy().into_owned()))
    }

    fn decoded_rustflags(encoded: &str) -> Vec<&str> {
        encoded.split(ENCODED_RUSTFLAGS_SEPARATOR).collect()
    }

    fn has_backend_identity_cfg(flags: &[&str]) -> bool {
        flags.windows(2).any(|pair| {
            pair[0] == "--cfg"
                && pair[1].starts_with("cuda_oxide_internal_backend_identity=\"")
                && pair[1].ends_with('"')
        })
    }

    fn is_sha256(value: &str) -> bool {
        value.len() == 64
            && value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    }

    /// `cargo_passthrough_command` with an empty ambient
    /// `CUDA_OXIDE_MATERIALIZE_CUBIN`.
    ///
    /// Every test builds the command through this. Reading the real variable
    /// would let an exported value override `opts.materialize_cubin` and drive
    /// the test into materializer discovery, which re-executes the libtest
    /// binary and then exits the process -- aborting the whole suite instead of
    /// failing one case.
    fn passthrough_command_for_test(
        ctx: &Context,
        cargo_subcommand: CargoPassthroughSubcommand,
        opts: &CargoPassthroughOptions<'_>,
        cargo_args: &[String],
    ) -> Result<Command, String> {
        cargo_passthrough_command_with_env(ctx, cargo_subcommand, opts, cargo_args, None)
    }

    fn cargo_artifact_freshness(
        ctx: &Context,
        opts: &CargoPassthroughOptions<'_>,
        materializer_provenance: Option<&str>,
    ) -> BTreeMap<String, bool> {
        let mut cmd = passthrough_command_for_test(
            ctx,
            CargoPassthroughSubcommand::Build,
            opts,
            &["--message-format=json-render-diagnostics".to_string()],
        )
        .unwrap();
        if let Some(provenance) = materializer_provenance {
            // Exercise a non-canonical spelling accepted by the backend. The
            // macro must still track exact provenance rather than keying that
            // dependency on the wrapper's canonical `1` spelling.
            cmd.env(MATERIALIZE_ENV, "true");
            cmd.env(EXPECTED_PROVENANCE_ENV, provenance);
        }
        let output = cmd.output().expect("failed to run Cargo cache probe");
        assert!(
            output.status.success(),
            "Cargo cache probe failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        );

        String::from_utf8(output.stdout)
            .expect("Cargo JSON must be UTF-8")
            .lines()
            .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
            .filter(|message| message["reason"] == "compiler-artifact")
            .filter_map(|message| {
                Some((
                    message["target"]["name"].as_str()?.to_string(),
                    message["fresh"].as_bool()?,
                ))
            })
            .collect()
    }

    fn test_context(config: OxideConfig) -> Context {
        Context {
            workspace_root: PathBuf::from("/tmp/cargo-oxide-test-workspace"),
            codegen_crate: PathBuf::from("/tmp/cargo-oxide-test-codegen"),
            examples_dir: PathBuf::from("/tmp/cargo-oxide-test-examples"),
            backend_so: PathBuf::from("llvm"),
            is_workspace: false,
            config,
        }
    }

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), unique))
    }

    #[test]
    fn strict_materialization_boolean_rejects_presence_only_values() {
        for value in ["1", "true", " YES ", "on"] {
            assert!(parse_strict_bool(MATERIALIZE_ENV, value).unwrap());
        }
        for value in ["0", "false", " NO ", "off"] {
            assert!(!parse_strict_bool(MATERIALIZE_ENV, value).unwrap());
        }
        for value in ["", "enabled", "2"] {
            let error = parse_strict_bool(MATERIALIZE_ENV, value).unwrap_err();
            assert!(error.contains("must be a boolean"), "{error}");
        }
    }

    #[test]
    fn materialization_rejects_nvvm_ir_as_a_competing_final_output() {
        let error = prepare_materialization_result(
            &test_context(OxideConfig::default()),
            true,
            Some("sm_90"),
            true,
        )
        .expect_err("the two user-facing final output modes must conflict");

        assert!(error.contains("cannot be combined with --emit-nvvm-ir"));
    }

    #[test]
    fn materializer_discovery_uses_the_same_project_tool_environment_as_rustc() {
        let configured_libdevice = "/configured/cuda/nvvm/libdevice/libdevice.10.bc";
        let ctx = test_context(OxideConfig {
            env: vec![
                (
                    "CUDA_OXIDE_LIBDEVICE".to_string(),
                    configured_libdevice.to_string(),
                ),
                (
                    "CUDA_TOOLKIT_PATH".to_string(),
                    "/configured/cuda".to_string(),
                ),
                (
                    "LD_LIBRARY_PATH".to_string(),
                    "/configured/cuda/lib64".to_string(),
                ),
            ],
            ..OxideConfig::default()
        });
        let discovery = materializer_discovery_command(&ctx, Path::new("/fake/cargo-oxide"));
        let mut rustc_child = Command::new("cargo");
        apply_common_codegen_env(
            &mut rustc_child,
            &ctx,
            false,
            false,
            false,
            DeviceDebug::Off,
        );

        for key in [
            "CUDA_OXIDE_LIBDEVICE",
            "CUDA_TOOLKIT_PATH",
            "LD_LIBRARY_PATH",
        ] {
            assert_eq!(
                command_env(&discovery, key),
                command_env(&rustc_child, key),
                "discovery and rustc must see the same {key}"
            );
        }
        if std::env::var_os("CUDA_OXIDE_LIBDEVICE").is_none() {
            assert_eq!(
                command_env(&discovery, "CUDA_OXIDE_LIBDEVICE").as_deref(),
                Some(configured_libdevice)
            );
        }
    }

    #[test]
    fn artifact_stem_normalizes_hyphens_like_cargo() {
        assert_eq!(artifact_stem("rustlantis-smoke"), "rustlantis_smoke");
        assert_eq!(artifact_stem("vecadd"), "vecadd");
    }

    #[test]
    fn emit_ltoir_paths_use_normalized_crate_stem() {
        // Regression for the emit-ltoir read/write mismatch on hyphenated
        // crates: the backend writes `rustlantis_smoke.{ll,ltoir}`, so both the
        // NVVM IR read and the default LTOIR write must resolve to the
        // underscore stem rather than the raw example name.
        let dir = Path::new("/tmp/cargo-oxide-emit-ltoir");
        assert_eq!(
            emitted_ll_path(dir, "rustlantis-smoke"),
            dir.join("rustlantis_smoke.ll")
        );
        assert_eq!(
            default_ltoir_path(dir, "rustlantis-smoke"),
            dir.join("rustlantis_smoke.ltoir")
        );
        // A non-hyphenated example is unaffected.
        assert_eq!(emitted_ll_path(dir, "vecadd"), dir.join("vecadd.ll"));
        assert_eq!(default_ltoir_path(dir, "vecadd"), dir.join("vecadd.ltoir"));
    }

    #[test]
    fn generated_file_cleanup_preserves_ltoir_cubin_cache() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_clean_cache_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        for extension in ["ptx", "ll", "ltoir", "cubin", "target"] {
            std::fs::write(root.join(format!("my_kernel.{extension}")), b"stale").unwrap();
        }
        let cached_cubin =
            root.join(".oxide-artifacts/ltoir-cubin-cache/v1/entries/key/image.cubin");
        std::fs::create_dir_all(cached_cubin.parent().unwrap()).unwrap();
        std::fs::write(&cached_cubin, b"persistent cache entry").unwrap();

        clean_generated_files(&root, "my-kernel");

        for extension in ["ptx", "ll", "ltoir", "cubin", "target"] {
            assert!(!root.join(format!("my_kernel.{extension}")).exists());
        }
        assert_eq!(
            std::fs::read(&cached_cubin).unwrap(),
            b"persistent cache entry"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn clean_removes_only_local_target_and_matching_artifacts() {
        let root = unique_temp_dir("cargo_oxide_clean_standalone");
        std::fs::create_dir_all(root.join("target/debug")).unwrap();

        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "my-kernel"
version = "0.1.0"
edition = "2024"
"#,
        )
        .unwrap();

        for suffix in GENERATED_ARTIFACT_SUFFIXES {
            std::fs::write(root.join(format!("my_kernel.{suffix}")), b"generated").unwrap();
        }

        let unrelated_artifact = root.join("other_kernel.ptx");
        std::fs::write(&unrelated_artifact, b"preserve").unwrap();

        let cached_cubin =
            root.join(".oxide-artifacts/ltoir-cubin-cache/v1/entries/key/image.cubin");
        std::fs::create_dir_all(cached_cubin.parent().unwrap()).unwrap();
        std::fs::write(&cached_cubin, b"persistent cache").unwrap();

        let ctx = Context {
            workspace_root: root.clone(),
            codegen_crate: root.clone(),
            examples_dir: root.clone(),
            backend_so: root.join("unused-backend.so"),
            is_workspace: false,
            config: OxideConfig::default(),
        };

        let summary = clean_context(&ctx).unwrap();

        assert_eq!(summary.removed_directories, 1);
        assert_eq!(summary.removed_files, GENERATED_ARTIFACT_SUFFIXES.len());
        assert!(!root.join("target").exists());

        for suffix in GENERATED_ARTIFACT_SUFFIXES {
            assert!(!root.join(format!("my_kernel.{suffix}")).exists());
        }

        assert_eq!(std::fs::read(&unrelated_artifact).unwrap(), b"preserve");
        assert_eq!(std::fs::read(&cached_cubin).unwrap(), b"persistent cache");

        let second_summary = clean_context(&ctx).unwrap();

        assert_eq!(second_summary, CleanSummary::default());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn workspace_clean_removes_root_backend_and_example_targets() {
        let root = unique_temp_dir("cargo_oxide_clean_workspace");
        let codegen_crate = root.join("crates/rustc-codegen-cuda");
        let examples_dir = codegen_crate.join("examples");
        let example_dir = examples_dir.join("demo");

        std::fs::create_dir_all(root.join("target/debug")).unwrap();
        std::fs::create_dir_all(codegen_crate.join("target/debug")).unwrap();
        std::fs::create_dir_all(example_dir.join("target/debug")).unwrap();

        std::fs::write(
            example_dir.join("Cargo.toml"),
            r#"
[package]
name = "demo"
version = "0.1.0"
edition = "2024"
"#,
        )
        .unwrap();

        std::fs::write(example_dir.join("demo.ptx"), b"generated").unwrap();

        let ctx = Context {
            workspace_root: root.clone(),
            codegen_crate: codegen_crate.clone(),
            examples_dir,
            backend_so: root.join("unused-backend.so"),
            is_workspace: true,
            config: OxideConfig::default(),
        };

        let summary = clean_context(&ctx).unwrap();

        assert_eq!(summary.removed_directories, 3);
        assert_eq!(summary.removed_files, 1);
        assert!(!root.join("target").exists());
        assert!(!codegen_crate.join("target").exists());
        assert!(!example_dir.join("target").exists());
        assert!(!example_dir.join("demo.ptx").exists());

        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn clean_refuses_symlinked_target_directory() {
        use std::os::unix::fs::symlink;

        let root = unique_temp_dir("cargo_oxide_clean_symlink");
        let external = unique_temp_dir("cargo_oxide_clean_external");

        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(&external).unwrap();
        std::fs::write(external.join("sentinel"), b"preserve").unwrap();

        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "symlink-test"
version = "0.1.0"
edition = "2024"
"#,
        )
        .unwrap();

        symlink(&external, root.join("target")).unwrap();

        let ctx = Context {
            workspace_root: root.clone(),
            codegen_crate: root.clone(),
            examples_dir: root.clone(),
            backend_so: root.join("unused-backend.so"),
            is_workspace: false,
            config: OxideConfig::default(),
        };

        let error = clean_context(&ctx).unwrap_err();

        assert!(error.contains("symlinked target directory"), "{error}");
        assert_eq!(
            std::fs::read(external.join("sentinel")).unwrap(),
            b"preserve"
        );

        std::fs::remove_dir_all(root).unwrap();
        std::fs::remove_dir_all(external).unwrap();
    }

    #[test]
    fn cargo_metadata_selection_prefers_default_run() {
        let root = unique_temp_dir("cargo_oxide_default_run");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "multi-bin-package"
default-run = "main_bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "main_bin"
path = "src/main.rs"

[[bin]]
name = "other_bin"
path = "src/other.rs"
"#,
        )
        .unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(root.join("src/other.rs"), "fn main() {}\n").unwrap();

        let selection = cargo_executable_selection(&root, None).unwrap();
        assert_eq!(selection.packages.len(), 1);
        let package = &selection.packages[0];
        assert!(package.package_id.starts_with("path+file://"));
        assert!(package.package_id.contains("multi-bin-package@0.1.0"));
        assert_eq!(package.package_name, "multi-bin-package");
        assert_eq!(package.default_run.as_deref(), Some("main_bin"));
        assert_eq!(selection.explicit_bin, None);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cargo_json_ignores_bins_disabled_by_required_features() {
        let root = unique_temp_dir("cargo_oxide_artifact_required_features");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "feature-gated-bins"
version = "0.1.0"
edition = "2024"

[features]
extra = []

[[bin]]
name = "always"
path = "src/always.rs"

[[bin]]
name = "gated"
path = "src/gated.rs"
required-features = ["extra"]
"#,
        )
        .unwrap();
        std::fs::write(root.join("src/always.rs"), "fn main() {}\n").unwrap();
        std::fs::write(root.join("src/gated.rs"), "fn main() {}\n").unwrap();

        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release"]).current_dir(&root);
        let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

        let expected_name = format!("always{}", std::env::consts::EXE_SUFFIX);
        assert_eq!(
            binary.file_name().and_then(OsStr::to_str),
            Some(expected_name.as_str())
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cargo_json_selects_custom_bin_in_configured_target_dir() {
        let root = unique_temp_dir("cargo_oxide_artifact_binary");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(root.join(".cargo")).unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "package-bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "actual-bin"
path = "src/main.rs"
"#,
        )
        .unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(
            root.join(".cargo/config.toml"),
            "[build]\ntarget-dir = \"configured-target\"\n",
        )
        .unwrap();

        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release"]).current_dir(&root);
        let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

        assert!(binary.exists());
        let expected_name = format!("actual-bin{}", std::env::consts::EXE_SUFFIX);
        assert_eq!(
            binary.file_name().and_then(OsStr::to_str),
            Some(expected_name.as_str())
        );
        assert!(
            binary
                .components()
                .any(|part| part.as_os_str() == "configured-target")
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cargo_json_selects_single_binary_from_virtual_workspace() {
        let root = unique_temp_dir("cargo_oxide_artifact_workspace");
        let member = root.join("member");
        std::fs::create_dir_all(member.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"member\"]\nresolver = \"2\"\n",
        )
        .unwrap();
        std::fs::write(
            member.join("Cargo.toml"),
            r#"
[package]
name = "workspace-package"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "workspace-bin"
path = "src/main.rs"
"#,
        )
        .unwrap();
        std::fs::write(member.join("src/main.rs"), "fn main() {}\n").unwrap();

        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release"]).current_dir(&root);
        let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

        let expected_name = format!("workspace-bin{}", std::env::consts::EXE_SUFFIX);
        assert_eq!(
            binary.file_name().and_then(OsStr::to_str),
            Some(expected_name.as_str())
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cargo_json_honors_virtual_workspace_default_member_default_run() {
        let root = unique_temp_dir("cargo_oxide_artifact_default_member");
        let app = root.join("app");
        let ignored = root.join("ignored");
        std::fs::create_dir_all(app.join("src")).unwrap();
        std::fs::create_dir_all(ignored.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"app\", \"ignored\"]\ndefault-members = [\"app\"]\nresolver = \"2\"\n",
        )
        .unwrap();
        std::fs::write(
            app.join("Cargo.toml"),
            r#"
[package]
name = "selected-package"
default-run = "chosen-bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "chosen-bin"
path = "src/chosen.rs"

[[bin]]
name = "other-bin"
path = "src/other.rs"
"#,
        )
        .unwrap();
        std::fs::write(app.join("src/chosen.rs"), "fn main() {}\n").unwrap();
        std::fs::write(app.join("src/other.rs"), "fn main() {}\n").unwrap();
        std::fs::write(
            ignored.join("Cargo.toml"),
            r#"
[package]
name = "ignored-package"
version = "0.1.0"
edition = "2024"
"#,
        )
        .unwrap();
        std::fs::write(ignored.join("src/main.rs"), "fn main() {}\n").unwrap();

        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release"]).current_dir(&root);
        let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

        let expected_name = format!("chosen-bin{}", std::env::consts::EXE_SUFFIX);
        assert_eq!(
            binary.file_name().and_then(OsStr::to_str),
            Some(expected_name.as_str())
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cargo_json_honors_nonvirtual_workspace_default_member() {
        let root = unique_temp_dir("cargo_oxide_artifact_nonvirtual_default_member");
        let member = root.join("member");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::create_dir_all(member.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "workspace-root-package"
version = "0.1.0"
edition = "2024"

[workspace]
members = ["member"]
default-members = ["member"]
resolver = "2"

[[bin]]
name = "root-bin"
path = "src/main.rs"
"#,
        )
        .unwrap();
        std::fs::write(root.join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(
            member.join("Cargo.toml"),
            r#"
[package]
name = "selected-member"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "member-bin"
path = "src/main.rs"
"#,
        )
        .unwrap();
        std::fs::write(member.join("src/main.rs"), "fn main() {}\n").unwrap();

        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release"]).current_dir(&root);
        let binary = run_cargo_build_for_executable(&mut cmd, &root, None).unwrap();

        let expected_name = format!("member-bin{}", std::env::consts::EXE_SUFFIX);
        assert_eq!(
            binary.file_name().and_then(OsStr::to_str),
            Some(expected_name.as_str())
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cargo_json_explicit_bin_selects_one_of_multiple_default_members() {
        let root = unique_temp_dir("cargo_oxide_artifact_multiple_default_members");
        let first = root.join("first");
        let second = root.join("second");
        std::fs::create_dir_all(first.join("src")).unwrap();
        std::fs::create_dir_all(second.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"first\", \"second\"]\ndefault-members = [\"first\", \"second\"]\nresolver = \"2\"\n",
        )
        .unwrap();
        std::fs::write(
            first.join("Cargo.toml"),
            r#"
[package]
name = "first-package"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "first-bin"
path = "src/main.rs"
"#,
        )
        .unwrap();
        std::fs::write(first.join("src/main.rs"), "fn main() {}\n").unwrap();
        std::fs::write(
            second.join("Cargo.toml"),
            r#"
[package]
name = "second-package"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "chosen-bin"
path = "src/main.rs"
"#,
        )
        .unwrap();
        std::fs::write(second.join("src/main.rs"), "fn main() {}\n").unwrap();

        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release", "--bin", "chosen-bin"])
            .current_dir(&root);
        let binary = run_cargo_build_for_executable(&mut cmd, &root, Some("chosen-bin")).unwrap();

        let expected_name = format!("chosen-bin{}", std::env::consts::EXE_SUFFIX);
        assert_eq!(
            binary.file_name().and_then(OsStr::to_str),
            Some(expected_name.as_str())
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn explicit_bin_must_be_unique_across_selected_packages() {
        let selection = CargoExecutableSelection {
            packages: vec![
                CargoSelectedPackage {
                    package_id: "first-package 0.1.0".to_string(),
                    package_name: "first-package".to_string(),
                    default_run: None,
                },
                CargoSelectedPackage {
                    package_id: "second-package 0.1.0".to_string(),
                    package_name: "second-package".to_string(),
                    default_run: None,
                },
            ],
            explicit_bin: Some("shared-bin".to_string()),
        };
        let artifacts = vec![
            CargoExecutableArtifact {
                package_id: "first-package 0.1.0".to_string(),
                target_name: "shared-bin".to_string(),
                path: PathBuf::from("/tmp/first/shared-bin"),
            },
            CargoExecutableArtifact {
                package_id: "second-package 0.1.0".to_string(),
                target_name: "shared-bin".to_string(),
                path: PathBuf::from("/tmp/second/shared-bin"),
            },
        ];

        let error = select_cargo_executable_artifact(&selection, &artifacts)
            .expect_err("the binary name does not uniquely identify an artifact");

        assert!(error.contains("multiple selected packages"), "{error}");
        assert!(error.contains("first-package"), "{error}");
        assert!(error.contains("second-package"), "{error}");
    }

    #[test]
    fn one_executable_package_is_selected_alongside_library_only_defaults() {
        let selection = CargoExecutableSelection {
            packages: vec![
                CargoSelectedPackage {
                    package_id: "library-package 0.1.0".to_string(),
                    package_name: "library-package".to_string(),
                    default_run: None,
                },
                CargoSelectedPackage {
                    package_id: "application-package 0.1.0".to_string(),
                    package_name: "application-package".to_string(),
                    default_run: None,
                },
            ],
            explicit_bin: None,
        };
        let artifact = CargoExecutableArtifact {
            package_id: "application-package 0.1.0".to_string(),
            target_name: "application-bin".to_string(),
            path: PathBuf::from("/tmp/application/application-bin"),
        };

        assert_eq!(
            select_cargo_executable_artifact(&selection, &[artifact]).unwrap(),
            PathBuf::from("/tmp/application/application-bin")
        );
    }

    #[test]
    fn unbuilt_default_run_is_not_skipped_for_another_selected_package() {
        let selection = CargoExecutableSelection {
            packages: vec![
                CargoSelectedPackage {
                    package_id: "first-package 0.1.0".to_string(),
                    package_name: "first-package".to_string(),
                    default_run: Some("gated-bin".to_string()),
                },
                CargoSelectedPackage {
                    package_id: "second-package 0.1.0".to_string(),
                    package_name: "second-package".to_string(),
                    default_run: None,
                },
            ],
            explicit_bin: None,
        };
        let artifacts = [CargoExecutableArtifact {
            package_id: "second-package 0.1.0".to_string(),
            target_name: "other-bin".to_string(),
            path: PathBuf::from("/tmp/second/other-bin"),
        }];

        let error = select_cargo_executable_artifact(&selection, &artifacts)
            .expect_err("a missing default-run must not fall back to another package");

        assert!(error.contains("first-package"), "{error}");
        assert!(error.contains("target `gated-bin`"), "{error}");
    }

    #[test]
    fn cargo_json_errors_when_requested_bin_was_not_built() {
        let root = unique_temp_dir("cargo_oxide_artifact_missing_bin");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "package-bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "actual-bin"
path = "src/actual.rs"

[[bin]]
name = "other-bin"
path = "src/other.rs"
"#,
        )
        .unwrap();
        std::fs::write(root.join("src/actual.rs"), "fn main() {}\n").unwrap();
        std::fs::write(root.join("src/other.rs"), "fn main() {}\n").unwrap();

        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release", "--bin", "actual-bin"])
            .current_dir(&root);
        let error = run_cargo_build_for_executable(&mut cmd, &root, Some("other-bin"))
            .expect_err("requested but unbuilt binary should be rejected");

        assert!(error.contains("target `other-bin`"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn cargo_json_errors_when_default_run_was_not_built() {
        let root = unique_temp_dir("cargo_oxide_artifact_missing_default_run");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(
            root.join("Cargo.toml"),
            r#"
[package]
name = "package-bin"
default-run = "default-bin"
version = "0.1.0"
edition = "2024"

[[bin]]
name = "default-bin"
path = "src/default.rs"

[[bin]]
name = "other-bin"
path = "src/other.rs"
"#,
        )
        .unwrap();
        std::fs::write(root.join("src/default.rs"), "fn main() {}\n").unwrap();
        std::fs::write(root.join("src/other.rs"), "fn main() {}\n").unwrap();

        let mut cmd = Command::new("cargo");
        cmd.args(["build", "--release", "--bin", "other-bin"])
            .current_dir(&root);
        let error = run_cargo_build_for_executable(&mut cmd, &root, None)
            .expect_err("unbuilt default-run binary should be rejected");

        assert!(error.contains("target `default-bin`"), "{error}");
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn artifact_selection_ignores_executable_artifacts_from_other_packages() {
        let selection = CargoExecutableSelection {
            packages: vec![CargoSelectedPackage {
                package_id: "app 0.1.0".to_string(),
                package_name: "app".to_string(),
                default_run: None,
            }],
            explicit_bin: Some("app-bin".to_string()),
        };
        let artifacts = vec![
            CargoExecutableArtifact {
                package_id: "build-tool 0.1.0".to_string(),
                target_name: "app-bin".to_string(),
                path: PathBuf::from("/tmp/build-tool/app-bin"),
            },
            CargoExecutableArtifact {
                package_id: "app 0.1.0".to_string(),
                target_name: "helper-bin".to_string(),
                path: PathBuf::from("/tmp/app/helper-bin"),
            },
        ];

        let error = select_cargo_executable_artifact(&selection, &artifacts)
            .expect_err("foreign package artifacts must not be selected");
        assert!(error.contains("target `app-bin`"), "{error}");
        assert!(error.contains("selected packages app"), "{error}");
    }

    #[test]
    fn sanitizer_adds_nonzero_error_exitcode_by_default() {
        let invocation =
            sanitizer_invocation_args(&["--leak-check".to_string(), "full".to_string()]);

        assert_eq!(
            invocation.args,
            ["--error-exitcode", "86", "--leak-check", "full"]
        );
        assert!(invocation.uses_default_error_exitcode);
        assert!(!invocation.status_checks_weakened);
    }

    #[test]
    fn sanitizer_preserves_explicit_zero_error_exitcode_without_claiming_detection() {
        let separated = sanitizer_invocation_args(&[
            "--error-exitcode".to_string(),
            "0".to_string(),
            "--leak-check".to_string(),
        ]);
        let equals = sanitizer_invocation_args(&["--error-exitcode=0".to_string()]);
        let repeated = sanitizer_invocation_args(&[
            "--error-exitcode=86".to_string(),
            "--error-exitcode=0".to_string(),
        ]);

        assert_eq!(separated.args, ["--error-exitcode", "0", "--leak-check"]);
        assert!(!separated.uses_default_error_exitcode);
        assert!(!separated.status_checks_weakened);
        assert_eq!(equals.args, ["--error-exitcode=0"]);
        assert!(!equals.uses_default_error_exitcode);
        assert_eq!(repeated.args, ["--error-exitcode=86", "--error-exitcode=0"]);
        assert!(!repeated.uses_default_error_exitcode);
    }

    #[test]
    fn sanitizer_detects_options_that_weaken_success_status() {
        for args in [
            vec!["--check-exit-code=no".to_string()],
            vec!["--check-exit-code".to_string(), "no".to_string()],
            vec!["--require-cuda-init=no".to_string()],
            vec!["--require-cuda-init".to_string(), "NO".to_string()],
        ] {
            let invocation = sanitizer_invocation_args(&args);
            assert!(invocation.status_checks_weakened, "{args:?}");
        }
    }

    #[test]
    fn sanitize_interop_codegen_defaults_to_line_tables_and_forwards_no_fmad() {
        let ctx = test_context(OxideConfig::default());
        let mut cmd = Command::new("cargo");

        apply_interop_device_codegen_options_with_env(
            &mut cmd,
            &ctx,
            false,
            InteropDeviceBuildOptions {
                no_fmad: true,
                unchecked_indexing: false,
                sanitizer_line_tables: true,
            },
            false,
        );

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_NO_FMA").as_deref(), Some("1"));
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEBUG").as_deref(),
            Some("line-tables")
        );

        let fingerprint = sanitize_codegen_fingerprint(
            &ctx,
            false,
            true,
            false,
            DeviceDebug::Off,
            Some("sm_80"),
            None,
            Some(Path::new("/tmp/generated-ptx")),
            &MaterializationMode::default(),
        );
        apply_codegen_configuration(
            &mut cmd,
            &ctx,
            CodegenProfilePolicy::ReleaseLike,
            &[],
            &fingerprint,
        )
        .unwrap();
        let encoded = command_env(&cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();
        assert!(has_backend_identity_cfg(&decoded_rustflags(&encoded)));
        assert_eq!(
            command_env(&cmd, CODEGEN_FINGERPRINT_ENV).as_deref(),
            Some(fingerprint.as_str())
        );
    }

    #[test]
    fn sanitize_device_debug_flag_overrides_the_line_tables_default() {
        let ctx = test_context(OxideConfig::default());
        let mut cmd = Command::new("cargo");

        // Mirror codegen_build_host_binary's ordering: the flag's level lands
        // on `cmd` first, then the sanitizer default runs. `env_debug_set` is
        // injected as false, so with no ambient CUDA_OXIDE_DEBUG the explicit
        // flag alone must suppress the line-tables default.
        apply_common_codegen_env(&mut cmd, &ctx, false, false, false, DeviceDebug::Full);
        apply_default_sanitizer_line_tables_with_env(&mut cmd, &ctx, false, DeviceDebug::Full);

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEBUG").as_deref(),
            Some("full")
        );
    }

    #[test]
    fn standard_interop_codegen_forwards_no_fmad_without_debug_override() {
        let ctx = test_context(OxideConfig::default());
        let mut cmd = Command::new("cargo");

        apply_interop_device_codegen_options_with_env(
            &mut cmd,
            &ctx,
            false,
            InteropDeviceBuildOptions::standard(true, false),
            false,
        );

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_NO_FMA").as_deref(), Some("1"));
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEBUG"), None);
    }

    #[test]
    fn interop_codegen_forwards_unchecked_indexing() {
        let ctx = test_context(OxideConfig::default());
        let mut cmd = Command::new("cargo");

        apply_interop_device_codegen_options_with_env(
            &mut cmd,
            &ctx,
            false,
            InteropDeviceBuildOptions::standard(false, true),
            false,
        );

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_UNCHECKED_INDEXING").as_deref(),
            Some("1")
        );
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_NO_FMA"), None);
    }

    #[test]
    fn sanitize_fingerprint_tracks_output_affecting_settings() {
        let ctx = test_context(OxideConfig::default());
        // Empty inherited environment, matching
        // `passthrough_fingerprint_tracks_output_affecting_settings`. An
        // ambient CUDA_OXIDE_NO_FMA / CUDA_OXIDE_UNCHECKED_INDEXING is folded
        // into the digest on its own, so reading the real environment would
        // make toggling the corresponding argument a no-op and collapse these
        // fingerprints onto the base.
        let inherited_env = BTreeMap::new();
        let fingerprint = |no_fmad: bool,
                           unchecked_indexing: bool,
                           target_arch: Option<&str>,
                           detected_device_arch: Option<&str>,
                           ptx_dir: Option<&Path>| {
            sanitize_codegen_fingerprint_with_env(
                &ctx,
                false,
                no_fmad,
                unchecked_indexing,
                DeviceDebug::Off,
                target_arch,
                detected_device_arch,
                ptx_dir,
                &MaterializationMode::default(),
                &inherited_env,
            )
        };

        let base = fingerprint(false, false, None, Some("sm_80"), None);

        for changed in [
            fingerprint(true, false, None, Some("sm_80"), None),
            fingerprint(false, true, None, Some("sm_80"), None),
            fingerprint(false, false, None, Some("sm_90"), None),
            fingerprint(false, false, Some("sm_80"), None, None),
            fingerprint(
                false,
                false,
                None,
                Some("sm_80"),
                Some(Path::new("/tmp/generated-ptx")),
            ),
        ] {
            assert_ne!(base, changed);
        }
    }

    #[test]
    fn pipeline_diagnostics_have_a_distinct_device_fingerprint() {
        let ctx = test_context(OxideConfig::default());
        let materialization = MaterializationMode::default();
        let standard = standard_codegen_fingerprint(
            &ctx,
            true,
            false,
            false,
            DeviceDebug::Off,
            false,
            Some("sm_86"),
            None,
            &materialization,
        );
        let pipeline = pipeline_codegen_fingerprint(
            &ctx,
            false,
            false,
            DeviceDebug::Off,
            false,
            Some("sm_86"),
            &materialization,
        );

        assert_ne!(standard, pipeline);
    }

    /// A `Context` whose `cuda-oxide.toml` points `CUDA_TOOLKIT_PATH` at
    /// `root`, alongside a fake executable named `name` under `root/bin`.
    fn toolkit_context_with_tool(root: &Path, name: &str) -> (Context, PathBuf) {
        let tool = root.join("bin").join(name);
        std::fs::create_dir_all(tool.parent().unwrap()).unwrap();
        std::fs::write(&tool, b"fake tool").unwrap();
        let ctx = test_context(OxideConfig {
            env: vec![(
                "CUDA_TOOLKIT_PATH".to_string(),
                root.to_string_lossy().into_owned(),
            )],
            ..OxideConfig::default()
        });
        (ctx, tool)
    }

    #[test]
    fn sanitizer_tool_lookup_uses_project_cuda_toolkit_root() {
        let root = unique_temp_dir("cargo_oxide_sanitizer_tool");
        let (ctx, tool) = toolkit_context_with_tool(&root, "cuda-oxide-test-sanitizer");

        // `|_| None` stands in for an empty ambient environment. Reading the
        // real one would let an exported CUDA_TOOLKIT_PATH/CUDA_HOME shadow the
        // configured root this test asserts on.
        assert_eq!(
            find_cuda_toolkit_executable_with_env(&ctx, "cuda-oxide-test-sanitizer", &[], |_| None),
            Some(tool)
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn doctor_compute_sanitizer_lookup_matches_sanitize_discovery() {
        // Hermetic: a fake tool name keeps the user's real PATH (and any
        // installed compute-sanitizer) out of the lookup, the injected empty
        // environment keeps an exported toolkit root out of it, and the shared
        // fallback const exercises the exact argument both `doctor` and
        // `sanitize` pass. The configured toolkit root wins before any
        // fallback path is consulted.
        let root = unique_temp_dir("cargo_oxide_doctor_sanitizer");
        let (ctx, tool) = toolkit_context_with_tool(&root, "cuda-oxide-test-doctor-sanitizer");

        assert_eq!(
            find_cuda_toolkit_executable_with_env(
                &ctx,
                "cuda-oxide-test-doctor-sanitizer",
                COMPUTE_SANITIZER_FALLBACK_PATHS,
                |_| None,
            ),
            Some(tool)
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ambient_cuda_toolkit_path_shadows_the_project_configured_root() {
        // The precedence the two lookups above have to be insulated from: an
        // exported CUDA_TOOLKIT_PATH outranks `cuda-oxide.toml`, so a tool
        // present only under the configured root is not found.
        let root = unique_temp_dir("cargo_oxide_ambient_shadow");
        let (ctx, _tool) = toolkit_context_with_tool(&root, "cuda-oxide-test-shadowed-sanitizer");
        let ambient = unique_temp_dir("cargo_oxide_ambient_root");

        assert_eq!(
            find_cuda_toolkit_executable_with_env(
                &ctx,
                "cuda-oxide-test-shadowed-sanitizer",
                &[],
                |key| (key == "CUDA_TOOLKIT_PATH").then(|| ambient.to_string_lossy().into_owned()),
            ),
            None
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn test_passthrough_defers_profile_flags_to_cargo_and_keeps_invariants() {
        let rustflags = build_encoded_rustflags_with_existing(
            Path::new("/tmp/librustc_codegen_cuda.so"),
            CargoPassthroughSubcommand::Test.codegen_profile(),
            &[],
            &["--cfg".to_string(), "device_test".to_string()],
            None,
            None,
        );
        let flags = decoded_rustflags(&rustflags);

        assert_eq!(
            flags,
            [
                "--cfg",
                "device_test",
                "-Zcodegen-backend=/tmp/librustc_codegen_cuda.so",
                "-Zmir-enable-passes=-JumpThreading",
                "-Zalways-encode-mir",
                "-Csymbol-mangling-version=v0",
            ]
        );
        assert!(!flags.iter().any(|flag| flag.starts_with("-Copt-level")));
        assert!(
            !flags
                .iter()
                .any(|flag| flag.starts_with("-Cdebug-assertions"))
        );
        assert!(!flags.iter().any(|flag| flag.starts_with("-Cdebuginfo")));

        let ctx = test_context(OxideConfig::default());
        let opts = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: None,
            features: None,
            cargo_target_dir: None,
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
            unchecked_indexing: false,
            materialize_cubin: false,
            device_debug: DeviceDebug::Off,
        };
        for cargo_args in [
            vec!["--release".to_string()],
            vec!["--profile".to_string(), "ci".to_string()],
        ] {
            let cmd = passthrough_command_for_test(
                &ctx,
                CargoPassthroughSubcommand::Test,
                &opts,
                &cargo_args,
            )
            .unwrap();
            let mut expected = vec!["test".to_string()];
            expected.extend(cargo_args);
            assert_eq!(
                cmd.get_args()
                    .map(|arg| arg.to_string_lossy().into_owned())
                    .collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn build_passthrough_retains_release_profile_and_required_flags() {
        let rustflags = build_encoded_rustflags_with_existing(
            Path::new("/tmp/librustc_codegen_cuda.so"),
            CargoPassthroughSubcommand::Build.codegen_profile(),
            &[],
            &[],
            Some(
                "-Lnative=/nix/store/cuda-cudart/lib\u{1f}-Copt-level=0\u{1f}-Zcodegen-backend=llvm",
            ),
            Some("-L native=/nix/store/cuda-cudart/lib"),
        );
        let flags = decoded_rustflags(&rustflags);

        assert_eq!(flags[0], "-Lnative=/nix/store/cuda-cudart/lib");
        assert!(flags.contains(&"-Copt-level=0"));
        assert!(flags.contains(&"-Zcodegen-backend=llvm"));
        assert_eq!(
            &flags[flags.len() - 6..],
            [
                "-Zcodegen-backend=/tmp/librustc_codegen_cuda.so",
                "-Copt-level=3",
                "-Cdebug-assertions=off",
                "-Zmir-enable-passes=-JumpThreading",
                "-Zalways-encode-mir",
                "-Csymbol-mangling-version=v0",
            ]
        );
        assert!(!flags.contains(&"native=/nix/store/cuda-cudart/lib"));
    }

    #[test]
    fn encoded_rustflags_preserve_configured_flag_boundaries_and_spaces() {
        let rustflags = build_encoded_rustflags_with_existing(
            Path::new("/tmp/backend path/librustc_codegen_cuda.so"),
            CodegenProfilePolicy::ReleaseLike,
            &["--cfg".to_string(), "model=\"alpha beta\"".to_string()],
            &[],
            None,
            Some("-L native=/nix/store/cuda-cudart/lib"),
        );
        let flags = decoded_rustflags(&rustflags);

        assert!(
            flags
                .windows(2)
                .any(|pair| pair == ["--cfg", "model=\"alpha beta\""])
        );
        assert_eq!(&flags[2..4], ["-L", "native=/nix/store/cuda-cudart/lib"]);
        assert_eq!(
            flags[flags.len() - 6],
            "-Zcodegen-backend=/tmp/backend path/librustc_codegen_cuda.so"
        );
    }

    #[test]
    fn encoded_rustflags_remove_legacy_global_codegen_fingerprints() {
        let encoded = [
            "--cfg",
            "cuda_oxide_internal_codegen_env=\"inherited\"",
            "--cfg=cuda_oxide_internal_materializer_provenance=\"inherited\"",
            "--cfg",
            "keep_inherited",
        ]
        .join(&ENCODED_RUSTFLAGS_SEPARATOR.to_string());
        let rustflags = build_encoded_rustflags_with_existing(
            Path::new("/tmp/librustc_codegen_cuda.so"),
            CodegenProfilePolicy::ReleaseLike,
            &[
                "--cfg".to_string(),
                "cuda_oxide_internal_codegen_env=\"configured\"".to_string(),
                "--cfg".to_string(),
                "keep_configured".to_string(),
            ],
            &[
                "--cfg".to_string(),
                "cuda_oxide_internal_materializer_provenance=\"explicit\"".to_string(),
                "--cfg".to_string(),
                "keep_explicit".to_string(),
            ],
            Some(&encoded),
            None,
        );
        let flags = decoded_rustflags(&rustflags);

        assert!(!flags.iter().any(|flag| {
            flag.contains(LEGACY_CODEGEN_FINGERPRINT_CFG)
                || flag.contains(LEGACY_MATERIALIZER_PROVENANCE_CFG)
        }));
        for retained in ["keep_configured", "keep_inherited", "keep_explicit"] {
            assert!(flags.contains(&retained));
        }
    }

    #[test]
    fn debug_profile_retains_release_defaults_and_adds_debuginfo() {
        let rustflags = build_encoded_rustflags_with_existing(
            Path::new("/tmp/librustc_codegen_cuda.so"),
            CodegenProfilePolicy::ReleaseLikeWithDebugInfo,
            &[],
            &[],
            None,
            Some(""),
        );
        let flags = decoded_rustflags(&rustflags);

        assert!(flags.contains(&"-Copt-level=3"));
        assert!(flags.contains(&"-Cdebug-assertions=off"));
        assert!(flags.contains(&"-Cdebuginfo=2"));
        assert!(flags.contains(&"-Zmir-enable-passes=-JumpThreading"));
        assert!(flags.contains(&"-Zalways-encode-mir"));
        assert!(flags.contains(&"-Csymbol-mangling-version=v0"));
        assert!(!flags.contains(&""));
    }

    #[test]
    fn project_config_parser_loads_backend_arch_flags_and_env() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_config_test_{}_{}",
            std::process::id(),
            unique
        ));
        let cargo_dir = root.join(".cargo");
        std::fs::create_dir_all(&cargo_dir).unwrap();
        std::fs::write(
            cargo_dir.join("cuda-oxide.toml"),
            r#"
backend = "../backend/librustc_codegen_cuda.so"
default-arch = "sm_90"
extra-rustflags = ["--cfg", "model=\"alpha beta\""]

[env]
MY_BUILD_FLAG = "configured"
"#,
        )
        .unwrap();

        let config = load_oxide_config(&root);
        assert_eq!(
            config.backend,
            Some(cargo_dir.join("../backend/librustc_codegen_cuda.so"))
        );
        assert_eq!(config.default_arch.as_deref(), Some("sm_90"));
        assert_eq!(config.extra_rustflags, ["--cfg", "model=\"alpha beta\""]);
        assert_eq!(
            config.env,
            vec![("MY_BUILD_FLAG".to_string(), "configured".to_string())]
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inspect_oxide_config_missing_is_informational() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_config_missing_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        assert!(matches!(
            inspect_oxide_config(&root),
            OxideConfigInspection::Missing
        ));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inspect_oxide_config_rejects_bad_toml_and_arch() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_config_bad_{}_{}",
            std::process::id(),
            unique
        ));
        let cargo_dir = root.join(".cargo");
        std::fs::create_dir_all(&cargo_dir).unwrap();
        std::fs::write(cargo_dir.join("cuda-oxide.toml"), "default-arch = [\n").unwrap();
        match inspect_oxide_config(&root) {
            OxideConfigInspection::Invalid { errors, .. } => {
                assert!(errors.iter().any(|e| e.contains("could not parse")));
            }
            other => panic!("expected Invalid, got {other:?}"),
        }

        std::fs::write(
            cargo_dir.join("cuda-oxide.toml"),
            "default-arch = \"sm_9x\"\n",
        )
        .unwrap();
        match inspect_oxide_config(&root) {
            OxideConfigInspection::Invalid { errors, .. } => {
                assert!(errors.iter().any(|e| e.contains("default-arch")));
            }
            other => panic!("expected Invalid, got {other:?}"),
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    /// `default-arch` load-time validation must be exactly as permissive as
    /// the consumers: `parse_nvvm_arch` (NVVM path) accepts `sm_XX`,
    /// `compute_XX`, and bare `XX`, so none of those may fail the load.
    /// Non-`sm_XX` spellings only earn an advisory warning.
    #[test]
    fn default_arch_validation_matches_the_real_arch_parser() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_config_arch_{}_{}",
            std::process::id(),
            unique
        ));
        let cargo_dir = root.join(".cargo");
        std::fs::create_dir_all(&cargo_dir).unwrap();
        let config_path = cargo_dir.join("cuda-oxide.toml");

        for accepted in ["sm_80", "sm_90a", "sm_100f", "sm_120"] {
            std::fs::write(&config_path, format!("default-arch = \"{accepted}\"\n")).unwrap();
            match inspect_oxide_config(&root) {
                OxideConfigInspection::Valid { warnings, .. } => {
                    assert!(warnings.is_empty(), "unexpected warnings for {accepted}");
                }
                other => panic!("expected {accepted} to be Valid, got {other:?}"),
            }
        }

        // Spellings that genuinely work today (the NVVM path normalizes
        // them) load fine but get the preferred-spelling advice.
        for (works_with_warning, preferred) in [("compute_90", "sm_90"), ("90", "sm_90")] {
            std::fs::write(
                &config_path,
                format!("default-arch = \"{works_with_warning}\"\n"),
            )
            .unwrap();
            match inspect_oxide_config(&root) {
                OxideConfigInspection::Valid { config, warnings } => {
                    assert_eq!(config.default_arch.as_deref(), Some(works_with_warning));
                    assert!(
                        warnings.iter().any(|w| w.contains(preferred)),
                        "expected a `{preferred}` spelling advisory for \
                         {works_with_warning}, got {warnings:?}"
                    );
                }
                other => panic!("expected {works_with_warning} to be Valid, got {other:?}"),
            }
        }

        for rejected in ["sm_9", "sm_90x", "hopper"] {
            std::fs::write(&config_path, format!("default-arch = \"{rejected}\"\n")).unwrap();
            match inspect_oxide_config(&root) {
                OxideConfigInspection::Invalid { errors, .. } => {
                    assert!(
                        errors.iter().any(|e| e.contains("default-arch")),
                        "expected a default-arch error for {rejected}, got {errors:?}"
                    );
                }
                other => panic!("expected {rejected} to be Invalid, got {other:?}"),
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn inspect_oxide_config_warns_on_forbidden_env_keys() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_config_warn_{}_{}",
            std::process::id(),
            unique
        ));
        let cargo_dir = root.join(".cargo");
        std::fs::create_dir_all(&cargo_dir).unwrap();
        std::fs::write(
            cargo_dir.join("cuda-oxide.toml"),
            r#"
default-arch = "sm_90a"

[env]
RUSTFLAGS = "-C opt-level=3"
CARGO_ENCODED_RUSTFLAGS = "legacy"
MY_OK = "1"
"#,
        )
        .unwrap();

        match inspect_oxide_config(&root) {
            OxideConfigInspection::Valid { config, warnings } => {
                assert_eq!(config.default_arch.as_deref(), Some("sm_90a"));
                assert!(
                    warnings
                        .iter()
                        .any(|w| w.contains("RUSTFLAGS") && w.contains("ignored"))
                );
                assert!(
                    warnings
                        .iter()
                        .any(|w| w.contains("CARGO_ENCODED_RUSTFLAGS") && w.contains("ignored"))
                );
            }
            other => panic!("expected Valid with warnings, got {other:?}"),
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn doctor_survives_malformed_config_and_reports_the_failed_check() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_doctor_config_bad_{}_{}",
            std::process::id(),
            unique
        ));
        let cargo_dir = root.join(".cargo");
        std::fs::create_dir_all(&cargo_dir).unwrap();
        std::fs::write(cargo_dir.join("cuda-oxide.toml"), "default-arch = [\n").unwrap();

        // Passive context resolution must not exit: it degrades to defaults
        // so the doctor scan can start at all.
        assert_eq!(load_oxide_config_lenient(&root), OxideConfig::default());

        // Doctor's own check re-inspects the file and fails.
        let check = check_oxide_config(&root);
        assert!(check.failed);
        assert!(check.headline.starts_with('✗'), "{}", check.headline);
        assert!(
            check
                .details
                .iter()
                .any(|line| line.contains("could not parse")),
            "{:?}",
            check.details
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn doctor_reports_env_rustflags_warning_without_failing_the_check() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_doctor_config_warn_{}_{}",
            std::process::id(),
            unique
        ));
        let cargo_dir = root.join(".cargo");
        std::fs::create_dir_all(&cargo_dir).unwrap();
        std::fs::write(
            cargo_dir.join("cuda-oxide.toml"),
            "default-arch = \"sm_90a\"\n\n[env]\nRUSTFLAGS = \"-C opt-level=3\"\n",
        )
        .unwrap();

        let check = check_oxide_config(&root);
        assert!(!check.failed);
        assert!(check.headline.contains("default-arch = sm_90a"));
        assert!(
            check
                .details
                .iter()
                .any(|line| line.contains("RUSTFLAGS") && line.contains("ignored")),
            "{:?}",
            check.details
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn doctor_reports_missing_config_as_informational() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_doctor_config_missing_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();

        let check = check_oxide_config(&root);
        assert!(!check.failed);
        assert_eq!(check.headline, "- not present (using defaults)");
        assert!(check.details.is_empty());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn passthrough_command_preserves_argv_and_cli_overrides_config_defaults() {
        let config = OxideConfig {
            extra_rustflags: vec!["--cfg".to_string(), "from_config".to_string()],
            env: vec![
                ("CARGO_TARGET_DIR".to_string(), "config-target".to_string()),
                (
                    "CUDA_OXIDE_DEVICE_CODEGEN_CRATE".to_string(),
                    "config_owner".to_string(),
                ),
                ("CUDA_OXIDE_VERBOSE".to_string(), "configured".to_string()),
            ],
            ..OxideConfig::default()
        };
        let ctx = test_context(config);
        let device_cfgs = vec!["model=\"alpha beta\"".to_string()];
        let opts = CargoPassthroughOptions {
            verbose: true,
            emit_nvvm_ir: false,
            arch: Some("sm_90"),
            features: Some("wrapper_feature"),
            cargo_target_dir: Some(Path::new("cli-target")),
            device_codegen_crate: Some("gpu-kernels, math_gpu"),
            device_cfgs: &device_cfgs,
            no_fmad: false,
            unchecked_indexing: false,
            materialize_cubin: false,
            device_debug: DeviceDebug::Off,
        };
        let cargo_args = vec![
            "-p".to_string(),
            "gpu-app".to_string(),
            "--".to_string(),
            "--nocapture".to_string(),
        ];

        let cmd = passthrough_command_for_test(
            &ctx,
            CargoPassthroughSubcommand::Test,
            &opts,
            &cargo_args,
        )
        .unwrap();
        assert_eq!(
            cmd.get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            [
                "test",
                "--features",
                "wrapper_feature",
                "-p",
                "gpu-app",
                "--",
                "--nocapture",
            ]
        );
        assert_eq!(
            command_env(&cmd, "CARGO_TARGET_DIR").as_deref(),
            Some("cli-target")
        );
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEVICE_CODEGEN_CRATE").as_deref(),
            Some("gpu_kernels,math_gpu")
        );
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
            Some("sm_90")
        );
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_VERBOSE").as_deref(),
            Some("1")
        );

        let encoded = command_env(&cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();
        let flags = decoded_rustflags(&encoded);
        assert!(
            flags
                .windows(2)
                .any(|pair| pair == ["--cfg", "from_config"])
        );
        assert!(
            flags
                .windows(2)
                .any(|pair| pair == ["--cfg", "model=\"alpha beta\""])
        );
        assert!(has_backend_identity_cfg(&flags));
        assert!(!flags.iter().any(|flag| {
            flag.contains("cuda_oxide_internal_codegen_env")
                || flag.contains("cuda_oxide_internal_materializer_provenance")
        }));
        assert!(is_sha256(
            &command_env(&cmd, CODEGEN_FINGERPRINT_ENV).unwrap()
        ));
        assert!(
            cmd.get_envs()
                .any(|(key, value)| key == OsStr::new("RUSTFLAGS") && value.is_none())
        );
    }

    #[test]
    fn passthrough_command_accepts_empty_cargo_args() {
        let ctx = test_context(OxideConfig::default());
        let opts = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: None,
            features: None,
            cargo_target_dir: None,
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
            unchecked_indexing: false,
            materialize_cubin: false,
            device_debug: DeviceDebug::Off,
        };

        let cmd = passthrough_command_for_test(&ctx, CargoPassthroughSubcommand::Test, &opts, &[])
            .unwrap();
        assert_eq!(
            cmd.get_args()
                .map(|arg| arg.to_string_lossy().into_owned())
                .collect::<Vec<_>>(),
            ["test"]
        );
    }

    #[test]
    fn architecture_and_output_mode_do_not_change_global_rustflags() {
        let ctx = test_context(OxideConfig::default());
        let base = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: Some("sm_80"),
            features: None,
            cargo_target_dir: None,
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
            unchecked_indexing: false,
            materialize_cubin: false,
            device_debug: DeviceDebug::Off,
        };
        let base_cmd =
            passthrough_command_for_test(&ctx, CargoPassthroughSubcommand::Build, &base, &[])
                .unwrap();
        let different_mode = CargoPassthroughOptions {
            emit_nvvm_ir: true,
            arch: Some("sm_90"),
            ..base
        };
        let different_cmd = passthrough_command_for_test(
            &ctx,
            CargoPassthroughSubcommand::Build,
            &different_mode,
            &[],
        )
        .unwrap();

        assert_eq!(
            command_env(&base_cmd, "CARGO_ENCODED_RUSTFLAGS"),
            command_env(&different_cmd, "CARGO_ENCODED_RUSTFLAGS"),
            "architecture/output switches must not invalidate every dependency"
        );
        assert_ne!(
            command_env(&base_cmd, CODEGEN_FINGERPRINT_ENV),
            command_env(&different_cmd, CODEGEN_FINGERPRINT_ENV),
            "device owners still need a distinct Cargo identity"
        );
    }

    #[test]
    fn codegen_mode_changes_rebuild_only_the_tracked_device_owner() {
        let root = unique_temp_dir("cargo_oxide_scoped_codegen_fingerprint");
        let target = root.join("target");
        for path in [
            root.join("shared-dep/src"),
            root.join("tracked-macro/src"),
            root.join("device-owner/src"),
            root.join("device-consumer/src"),
        ] {
            std::fs::create_dir_all(path).unwrap();
        }
        std::fs::write(
            root.join("Cargo.toml"),
            r#"[workspace]
resolver = "3"
members = ["shared-dep", "tracked-macro", "device-owner", "device-consumer"]
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("shared-dep/Cargo.toml"),
            r#"[package]
name = "shared-dep"
version = "0.0.0"
edition = "2024"
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("shared-dep/src/lib.rs"),
            "pub fn shared_value() -> u32 { 42 }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("tracked-macro/Cargo.toml"),
            r#"[package]
name = "tracked-macro"
version = "0.0.0"
edition = "2024"

[lib]
proc-macro = true
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("tracked-macro/src/lib.rs"),
            format!(
                r#"#![feature(proc_macro_tracked_env)]
extern crate proc_macro;

#[proc_macro]
pub fn track_codegen(_input: proc_macro::TokenStream) -> proc_macro::TokenStream {{
    let _ = proc_macro::tracked::env_var({CODEGEN_FINGERPRINT_ENV:?});
    let _ = proc_macro::tracked::env_var({MATERIALIZE_ENV:?});
    let _ = proc_macro::tracked::env_var({EXPECTED_PROVENANCE_ENV:?});
    "()".parse().unwrap()
}}
"#
            ),
        )
        .unwrap();
        std::fs::write(
            root.join("device-owner/Cargo.toml"),
            r#"[package]
name = "device-owner"
version = "0.0.0"
edition = "2024"

[dependencies]
shared-dep = { path = "../shared-dep" }
tracked-macro = { path = "../tracked-macro" }
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("device-owner/src/lib.rs"),
            "const _: () = tracked_macro::track_codegen!();\npub fn device_value() -> u32 { shared_dep::shared_value() }\n",
        )
        .unwrap();
        std::fs::write(
            root.join("device-consumer/Cargo.toml"),
            r#"[package]
name = "device-consumer"
version = "0.0.0"
edition = "2024"

[dependencies]
device-owner = { path = "../device-owner" }
"#,
        )
        .unwrap();
        std::fs::write(
            root.join("device-consumer/src/main.rs"),
            "fn main() { assert_eq!(device_owner::device_value(), 42); }\n",
        )
        .unwrap();

        let ctx = Context {
            workspace_root: root.clone(),
            codegen_crate: root.join("unused-codegen-source"),
            examples_dir: root.join("unused-examples"),
            backend_so: PathBuf::from("llvm"),
            is_workspace: false,
            config: OxideConfig::default(),
        };
        let base = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: Some("sm_80"),
            features: None,
            cargo_target_dir: Some(&target),
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
            unchecked_indexing: false,
            materialize_cubin: false,
            device_debug: DeviceDebug::Off,
        };

        let cold = cargo_artifact_freshness(&ctx, &base, None);
        assert_eq!(cold.get("shared_dep"), Some(&false));
        assert_eq!(cold.get("tracked_macro"), Some(&false));
        assert_eq!(cold.get("device_owner"), Some(&false));
        assert_eq!(cold.get("device-consumer"), Some(&false));

        let warm = cargo_artifact_freshness(&ctx, &base, None);
        assert_eq!(warm.get("shared_dep"), Some(&true));
        assert_eq!(warm.get("tracked_macro"), Some(&true));
        assert_eq!(warm.get("device_owner"), Some(&true));
        assert_eq!(warm.get("device-consumer"), Some(&true));

        let different_arch = CargoPassthroughOptions {
            arch: Some("sm_90"),
            ..base
        };
        let arch_switch = cargo_artifact_freshness(&ctx, &different_arch, None);
        assert_eq!(arch_switch.get("shared_dep"), Some(&true));
        assert_eq!(arch_switch.get("tracked_macro"), Some(&true));
        assert_eq!(arch_switch.get("device_owner"), Some(&false));
        assert_eq!(arch_switch.get("device-consumer"), Some(&false));

        let different_output = CargoPassthroughOptions {
            emit_nvvm_ir: true,
            ..different_arch
        };
        let output_switch = cargo_artifact_freshness(&ctx, &different_output, None);
        assert_eq!(output_switch.get("shared_dep"), Some(&true));
        assert_eq!(output_switch.get("tracked_macro"), Some(&true));
        assert_eq!(output_switch.get("device_owner"), Some(&false));
        assert_eq!(output_switch.get("device-consumer"), Some(&false));

        let repeated_output = cargo_artifact_freshness(&ctx, &different_output, None);
        assert_eq!(repeated_output.get("shared_dep"), Some(&true));
        assert_eq!(repeated_output.get("tracked_macro"), Some(&true));
        assert_eq!(repeated_output.get("device_owner"), Some(&true));
        assert_eq!(repeated_output.get("device-consumer"), Some(&true));

        let provenance_switch = cargo_artifact_freshness(
            &ctx,
            &different_output,
            Some("11d91fbe164094f6242d44103d0fb01968b96c6d8f48f124eac8fa73a307a657"),
        );
        assert_eq!(provenance_switch.get("shared_dep"), Some(&true));
        assert_eq!(provenance_switch.get("tracked_macro"), Some(&true));
        assert_eq!(provenance_switch.get("device_owner"), Some(&false));
        assert_eq!(provenance_switch.get("device-consumer"), Some(&false));

        let changed_provenance = cargo_artifact_freshness(
            &ctx,
            &different_output,
            Some("5b11618c2e44027877d0cd4d0cfd10afed5ef262876791e483ec58f4c5569139"),
        );
        assert_eq!(changed_provenance.get("shared_dep"), Some(&true));
        assert_eq!(changed_provenance.get("tracked_macro"), Some(&true));
        assert_eq!(changed_provenance.get("device_owner"), Some(&false));
        assert_eq!(changed_provenance.get("device-consumer"), Some(&false));

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn owner_filter_resolution_is_normalized_and_has_explicit_precedence() {
        assert_eq!(
            resolve_device_codegen_crates(None, None, Some("gpu-kernels, math_gpu"))
                .unwrap()
                .as_deref(),
            Some("gpu_kernels,math_gpu"),
        );
        assert_eq!(
            resolve_device_codegen_crates(None, Some("parent-owner"), Some("config-owner"))
                .unwrap()
                .as_deref(),
            Some("parent_owner"),
        );
        assert!(
            resolve_device_codegen_crates(Some(""), Some("parent-owner"), Some("config-owner"))
                .is_err()
        );
    }

    #[test]
    fn passthrough_fingerprint_tracks_output_affecting_settings() {
        let ctx = test_context(OxideConfig::default());
        let base = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: Some("sm_80"),
            features: None,
            cargo_target_dir: None,
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
            unchecked_indexing: false,
            materialize_cubin: false,
            device_debug: DeviceDebug::Off,
        };
        let inherited_env = BTreeMap::new();
        let base_hash = passthrough_codegen_fingerprint_with_env(
            &ctx,
            &base,
            None,
            Some("sm_80"),
            &MaterializationMode::default(),
            &inherited_env,
        );

        let arch = CargoPassthroughOptions {
            arch: Some("sm_90"),
            ..base
        };
        let emit = CargoPassthroughOptions {
            emit_nvvm_ir: true,
            ..base
        };
        let no_fmad = CargoPassthroughOptions {
            no_fmad: true,
            ..base
        };
        let unchecked_indexing = CargoPassthroughOptions {
            unchecked_indexing: true,
            ..base
        };
        let configured_ptx = test_context(OxideConfig {
            env: vec![(
                "CUDA_OXIDE_PTX_DIR".to_string(),
                "configured-ptx".to_string(),
            )],
            ..OxideConfig::default()
        });

        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &arch,
                None,
                Some("sm_90"),
                &MaterializationMode::default(),
                &inherited_env,
            )
        );
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &emit,
                None,
                Some("sm_80"),
                &MaterializationMode::default(),
                &inherited_env,
            )
        );
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &no_fmad,
                None,
                Some("sm_80"),
                &MaterializationMode::default(),
                &inherited_env,
            )
        );
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &unchecked_indexing,
                None,
                Some("sm_80"),
                &MaterializationMode::default(),
                &inherited_env,
            )
        );
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &base,
                Some("gpu_kernel"),
                Some("sm_80"),
                &MaterializationMode::default(),
                &inherited_env,
            )
        );
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &configured_ptx,
                &base,
                None,
                Some("sm_80"),
                &MaterializationMode::default(),
                &inherited_env,
            )
        );
        let materialized = MaterializationMode {
            provenance: Some("ab".repeat(32)),
        };
        assert_ne!(
            base_hash,
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &base,
                None,
                Some("sm_80"),
                &materialized,
                &inherited_env,
            ),
            "exact CUDA-tool provenance must change Cargo's rustc fingerprint"
        );
    }

    #[test]
    fn passthrough_fingerprint_tracks_non_unicode_presence_switch_bytes() {
        let ctx = test_context(OxideConfig::default());
        let opts = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: Some("sm_80"),
            features: None,
            cargo_target_dir: None,
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
            unchecked_indexing: false,
            materialize_cubin: false,
            device_debug: DeviceDebug::Off,
        };
        let fingerprint = |inherited_env: &BTreeMap<String, Vec<u8>>| {
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                &opts,
                None,
                Some("sm_80"),
                &MaterializationMode::default(),
                inherited_env,
            )
        };
        let absent = BTreeMap::new();
        let first = BTreeMap::from([("CUDA_OXIDE_NO_FMA".to_string(), vec![0xff])]);
        let second = BTreeMap::from([("CUDA_OXIDE_NO_FMA".to_string(), vec![0xfe])]);

        assert_ne!(fingerprint(&absent), fingerprint(&first));
        assert_ne!(fingerprint(&first), fingerprint(&second));
    }

    #[test]
    fn global_backend_identity_tracks_rebuild_at_same_path() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cargo_oxide_backend_fingerprint_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let backend = root.join("librustc_codegen_cuda.so");
        std::fs::write(&backend, b"first").unwrap();
        let original = std::fs::metadata(&backend).unwrap();
        let original_modified = original.modified().unwrap();

        let mut ctx = test_context(OxideConfig::default());
        ctx.backend_so = backend.clone();
        let fingerprint = "42".repeat(32);
        let mut before_cmd = Command::new("cargo");
        apply_codegen_configuration(
            &mut before_cmd,
            &ctx,
            CodegenProfilePolicy::ReleaseLike,
            &[],
            &fingerprint,
        )
        .unwrap();
        let before = command_env(&before_cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();
        // Preserve the weak metadata identity that used to be fingerprinted:
        // only the bytes differ.
        std::fs::write(&backend, b"other").unwrap();
        std::fs::File::options()
            .write(true)
            .open(&backend)
            .unwrap()
            .set_times(std::fs::FileTimes::new().set_modified(original_modified))
            .unwrap();
        let replacement = std::fs::metadata(&backend).unwrap();
        assert_eq!(replacement.len(), original.len());
        assert_eq!(replacement.modified().unwrap(), original_modified);
        let mut after_cmd = Command::new("cargo");
        apply_codegen_configuration(
            &mut after_cmd,
            &ctx,
            CodegenProfilePolicy::ReleaseLike,
            &[],
            &fingerprint,
        )
        .unwrap();
        let after = command_env(&after_cmd, "CARGO_ENCODED_RUSTFLAGS").unwrap();

        assert_ne!(before, after);
        assert_eq!(
            command_env(&before_cmd, CODEGEN_FINGERPRINT_ENV),
            command_env(&after_cmd, CODEGEN_FINGERPRINT_ENV)
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn owner_filter_rejects_empty_or_invalid_entries() {
        assert_eq!(
            normalize_device_codegen_crates("gpu-kernels, math_gpu").unwrap(),
            "gpu_kernels,math_gpu"
        );
        assert!(normalize_device_codegen_crates("").is_err());
        assert!(normalize_device_codegen_crates("   ").is_err());
        assert!(normalize_device_codegen_crates("gpu,").is_err());
        assert!(normalize_device_codegen_crates("gpu,not a crate").is_err());
    }

    #[test]
    fn internal_ptx_directory_overrides_project_env_default() {
        let ctx = test_context(OxideConfig {
            env: vec![(
                "CUDA_OXIDE_PTX_DIR".to_string(),
                "configured-ptx".to_string(),
            )],
            ..OxideConfig::default()
        });
        let mut cmd = Command::new("cargo");
        apply_common_codegen_env(&mut cmd, &ctx, false, false, false, DeviceDebug::Off);
        cmd.env("CUDA_OXIDE_PTX_DIR", "internal-ptx");
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_PTX_DIR").as_deref(),
            Some("internal-ptx")
        );
    }

    #[test]
    fn nvvm_arch_normalizes_all_accepted_forms() {
        // `sm_XX` is the form `--arch` and the rest of cargo-oxide use.
        assert_eq!(parse_nvvm_arch("sm_120").unwrap().compute(), "compute_120");
        assert_eq!(parse_nvvm_arch("sm_90").unwrap().compute(), "compute_90");
        // `compute_XX` passes through unchanged.
        assert_eq!(
            parse_nvvm_arch("compute_100").unwrap().compute(),
            "compute_100"
        );
        // A bare capability is accepted too.
        assert_eq!(parse_nvvm_arch("120").unwrap().compute(), "compute_120");
        assert!(parse_nvvm_arch("sm_90x").is_err());
    }

    #[test]
    fn emit_ltoir_preserves_fma_and_debug_policy_for_libnvvm() {
        let arch = parse_nvvm_arch("sm_90").unwrap();
        for (artifact_debug, finalizer_debug) in [
            (
                oxide_artifacts::ArtifactDebugPolicy::None,
                cuda_artifact_finalizer::DebugPolicy::None,
            ),
            (
                oxide_artifacts::ArtifactDebugPolicy::LineTables,
                cuda_artifact_finalizer::DebugPolicy::LineTables,
            ),
            (
                oxide_artifacts::ArtifactDebugPolicy::Full,
                cuda_artifact_finalizer::DebugPolicy::Full,
            ),
        ] {
            let artifact_options = oxide_artifacts::ArtifactCompileOptions::new()
                .with_fma_contraction(false)
                .with_debug_policy(artifact_debug);
            let finalizer_options = finalization_options_from_artifact(&arch, artifact_options);

            assert_eq!(finalizer_options.target(), &arch);
            assert!(!finalizer_options.allow_fma_contraction());
            assert_eq!(finalizer_options.debug_policy(), finalizer_debug);
        }
    }

    #[test]
    fn apply_output_mode_sets_target_for_arch_override() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(
            &mut cmd,
            false,
            Some("sm_120"),
            &MaterializationMode::default(),
        );

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
            Some("sm_120")
        );
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
    }

    #[test]
    fn apply_output_mode_sets_nvvm_ir_flag_and_target() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(
            &mut cmd,
            true,
            Some("sm_100a"),
            &MaterializationMode::default(),
        );

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
            Some("sm_100a")
        );
        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR").as_deref(),
            Some("1")
        );
    }

    #[test]
    fn materialization_forces_nvvm_ir_and_exact_provenance_handshake() {
        let mut cmd = Command::new("cargo");
        let materialization = MaterializationMode {
            provenance: Some("42".repeat(32)),
        };

        apply_output_mode(&mut cmd, false, Some("sm_90"), &materialization);

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR").as_deref(),
            Some("1")
        );
        assert_eq!(command_env(&cmd, MATERIALIZE_ENV).as_deref(), Some("1"));
        assert_eq!(
            command_env(&cmd, EXPECTED_PROVENANCE_ENV).as_deref(),
            Some("4242424242424242424242424242424242424242424242424242424242424242")
        );
    }

    #[test]
    fn apply_output_mode_leaves_auto_detect_ptx_unset() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(&mut cmd, false, None, &MaterializationMode::default());

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_TARGET"), None);
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
    }

    #[test]
    fn apply_device_arch_hint_sets_hint_when_no_explicit_arch() {
        let mut cmd = Command::new("cargo");

        apply_device_arch_hint(&mut cmd, None, Some("sm_120a"));

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH").as_deref(),
            Some("sm_120a")
        );
        // The hint must never masquerade as the hard override.
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_TARGET"), None);
    }

    #[test]
    fn apply_device_arch_hint_skipped_when_arch_explicit() {
        // An explicit --arch already went to CUDA_OXIDE_TARGET; don't also
        // emit a competing device hint.
        let mut cmd = Command::new("cargo");

        apply_device_arch_hint(&mut cmd, Some("sm_90"), Some("sm_120a"));

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH"), None);
    }

    #[test]
    fn apply_device_arch_hint_noop_without_detection() {
        let mut cmd = Command::new("cargo");

        apply_device_arch_hint(&mut cmd, None, None);

        assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH"), None);
    }

    #[test]
    fn debug_output_mode_forwards_detected_gpu_hint() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(&mut cmd, false, None, &MaterializationMode::default());
        apply_device_arch_hint(&mut cmd, None, Some("sm_120a"));

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH").as_deref(),
            Some("sm_120a")
        );
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_TARGET"), None);
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
    }

    #[test]
    fn debug_output_mode_honors_explicit_arch_override() {
        let mut cmd = Command::new("cargo");

        apply_output_mode(
            &mut cmd,
            false,
            Some("sm_90"),
            &MaterializationMode::default(),
        );
        apply_device_arch_hint(&mut cmd, Some("sm_90"), Some("sm_120a"));

        assert_eq!(
            command_env(&cmd, "CUDA_OXIDE_TARGET").as_deref(),
            Some("sm_90")
        );
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_DEVICE_ARCH"), None);
        assert_eq!(command_env(&cmd, "CUDA_OXIDE_EMIT_NVVM_IR"), None);
    }

    #[test]
    fn format_sm_arch_uses_cuda_target_spelling() {
        // cc < 9.0 — no arch-specific target exists in the PTX ISA, so we
        // emit the plain `sm_XY` form. Confirms we do not produce false
        // positives like `sm_75a` / `sm_80a` / `sm_89a`.
        assert_eq!(format_sm_arch((7, 0)), "sm_70");
        assert_eq!(format_sm_arch((7, 5)), "sm_75");
        assert_eq!(format_sm_arch((8, 0)), "sm_80");
        assert_eq!(format_sm_arch((8, 6)), "sm_86");
        assert_eq!(format_sm_arch((8, 9)), "sm_89");

        // cc ≥ 9.0 — every chip that reports this CC is an arch-specific
        // (`a`) variant. Auto-detect emits the `a` form so the codegen
        // backend can lower WGMMA / tcgen05 / TMA-multicast / cta_group
        // intrinsics without falling through to a plain target that ptxas
        // would reject. Confirms we do not produce false negatives.
        assert_eq!(format_sm_arch((9, 0)), "sm_90a"); // Hopper (H100/H200)
        assert_eq!(format_sm_arch((10, 0)), "sm_100a"); // Blackwell DC
        assert_eq!(format_sm_arch((10, 1)), "sm_101a");
        assert_eq!(format_sm_arch((10, 3)), "sm_103a");
        assert_eq!(format_sm_arch((12, 0)), "sm_120a"); // consumer Blackwell
    }

    #[test]
    fn parse_compute_cap_accepts_real_nvidia_smi_output() {
        assert_eq!(parse_compute_cap("12.0\n"), Some((12, 0)));
        assert_eq!(parse_compute_cap("7.5\n"), Some((7, 5)));
        assert_eq!(parse_compute_cap("10.3"), Some((10, 3)));
        // End-to-end with format_sm_arch: the values the backend sees.
        assert_eq!(
            format_sm_arch(parse_compute_cap("12.0\n").unwrap()),
            "sm_120a"
        );
        assert_eq!(format_sm_arch(parse_compute_cap("7.5\n").unwrap()), "sm_75");
    }

    #[test]
    fn parse_compute_cap_takes_first_gpu_on_multi_gpu_machines() {
        assert_eq!(parse_compute_cap("9.0\n12.0\n"), Some((9, 0)));
    }

    #[test]
    fn parse_gpu_name_and_compute_cap_splits_on_last_comma() {
        assert_eq!(
            parse_gpu_name_and_compute_cap("NVIDIA GeForce RTX 5090, 12.0\n"),
            Some(("NVIDIA GeForce RTX 5090".to_string(), (12, 0)))
        );
        // Failure banner: no comma-separated cc field.
        assert_eq!(
            parse_gpu_name_and_compute_cap("NVIDIA-SMI has failed.\n"),
            None
        );
        assert_eq!(parse_gpu_name_and_compute_cap(""), None);
    }

    #[test]
    fn cuda_toolkit_root_prefers_toolkit_path_then_home_then_default() {
        let toolkit_and_home = cuda_toolkit_root(|var| match var {
            "CUDA_TOOLKIT_PATH" => Some("/cuda/toolkit".to_string()),
            "CUDA_HOME" => Some("/cuda/home".to_string()),
            _ => None,
        });
        assert_eq!(toolkit_and_home, "/cuda/toolkit");

        let home_only =
            cuda_toolkit_root(|var| (var == "CUDA_HOME").then(|| "/cuda/home".to_string()));
        assert_eq!(home_only, "/cuda/home");

        let empty_toolkit_path = cuda_toolkit_root(|var| match var {
            "CUDA_TOOLKIT_PATH" => Some("  ".to_string()),
            "CUDA_HOME" => Some("/cuda/home".to_string()),
            _ => None,
        });
        assert_eq!(empty_toolkit_path, "/cuda/home");

        assert_eq!(cuda_toolkit_root(|_| None), "/usr/local/cuda");
    }

    #[test]
    fn cuda_header_candidates_cover_standard_and_redistributable_layouts() {
        // Standard install layout first, then the matching targets/ layout.
        assert_eq!(
            cuda_header_candidates("/usr/local/cuda", "x86_64"),
            vec![
                PathBuf::from("/usr/local/cuda/include/cuda.h"),
                PathBuf::from("/usr/local/cuda/targets/x86_64-linux/include/cuda.h"),
            ]
        );
        // aarch64 servers use the sbsa-linux target dir.
        assert_eq!(
            cuda_header_candidates("/opt/ctk", "aarch64"),
            vec![
                PathBuf::from("/opt/ctk/include/cuda.h"),
                PathBuf::from("/opt/ctk/targets/sbsa-linux/include/cuda.h"),
            ]
        );
        // Unknown host arch: only the standard layout is probed.
        assert_eq!(
            cuda_header_candidates("/opt/ctk", "riscv64"),
            vec![PathBuf::from("/opt/ctk/include/cuda.h")]
        );
    }

    #[test]
    fn parse_rust_toolchain_toml_reads_channel_and_components() {
        let pin = parse_rust_toolchain_toml(
            r#"[toolchain]
channel = "nightly-2026-04-03"
components = ["rust-src", "rustc-dev", "llvm-tools"]
"#,
        )
        .expect("pin should parse");
        assert_eq!(pin.channel, "nightly-2026-04-03");
        assert_eq!(
            pin.components,
            vec![
                "rust-src".to_string(),
                "rustc-dev".to_string(),
                "llvm-tools".to_string()
            ]
        );
    }

    #[test]
    fn parse_rust_toolchain_toml_rejects_missing_channel() {
        let error = parse_rust_toolchain_toml("[toolchain]\ncomponents = [\"rust-src\"]\n")
            .expect_err("channel is required");
        assert!(error.contains("channel"), "{error}");
    }

    #[test]
    fn active_toolchain_matches_channel_accepts_target_triple_suffix() {
        assert!(active_toolchain_matches_channel(
            "nightly-2026-04-03-aarch64-apple-darwin (default)",
            "nightly-2026-04-03"
        ));
        assert!(active_toolchain_matches_channel(
            "nightly-2026-04-03",
            "nightly-2026-04-03"
        ));
        assert!(!active_toolchain_matches_channel(
            "nightly-2026-01-01-x86_64-unknown-linux-gnu (default)",
            "nightly-2026-04-03"
        ));
    }

    #[test]
    fn active_toolchain_matches_channel_accepts_rustup_128_and_129_formats() {
        // rustup 1.29 single-line form with an override annotation, as
        // observed verbatim on a workspace with rust-toolchain.toml.
        assert!(active_toolchain_matches_channel(
            "nightly-2026-04-03-x86_64-unknown-linux-gnu (overridden by \
             '/home/user/cuda-oxide/rust-toolchain.toml')",
            "nightly-2026-04-03"
        ));
        // rustup 1.28 two-line form: bare name, then the reason line.
        assert!(active_toolchain_matches_channel(
            "nightly-2026-04-03-x86_64-unknown-linux-gnu\nactive because: \
             overridden by '/home/user/cuda-oxide/rust-toolchain.toml'",
            "nightly-2026-04-03"
        ));
        // A mismatched pin must not be rescued by later lines.
        assert!(!active_toolchain_matches_channel(
            "stable-x86_64-unknown-linux-gnu\nactive because: default",
            "nightly-2026-04-03"
        ));
    }

    #[test]
    fn plan_update_selects_advise_setup_or_cache_refresh() {
        assert_eq!(plan_update(true, false), UpdatePlan::AdviseSetup);
        assert_eq!(plan_update(true, true), UpdatePlan::RunSetup);
        assert_eq!(plan_update(false, false), UpdatePlan::RefreshCache);
        assert_eq!(plan_update(false, true), UpdatePlan::RefreshCache);
    }

    /// A `.cargo/cuda-oxide.toml` backend pin outranks the shared cache, so
    /// `update` must refuse just like it does for `CUDA_OXIDE_BACKEND`.
    #[test]
    fn update_refuses_when_the_config_pins_a_backend() {
        let pinned = test_context(OxideConfig {
            backend: Some(PathBuf::from("/tmp/pinned-backend.so")),
            ..OxideConfig::default()
        });
        // `None` stands in for an unset ambient `CUDA_OXIDE_BACKEND`. Reading
        // the real one would let an exported value produce the env refusal for
        // both inputs, including the unpinned case asserted to be `None`.
        let refusal =
            update_pin_refusal_with_env(&pinned, None).expect("config pin must refuse update");
        assert!(refusal.contains("pins the backend"), "{refusal}");
        assert!(refusal.contains("/tmp/pinned-backend.so"), "{refusal}");

        let unpinned = test_context(OxideConfig::default());
        assert_eq!(update_pin_refusal_with_env(&unpinned, None), None);

        // The env var outranks the project pin: set, it refuses even unpinned.
        let from_env = update_pin_refusal_with_env(&unpinned, Some("/tmp/env-backend.so".into()))
            .expect("exported CUDA_OXIDE_BACKEND must refuse update");
        assert!(from_env.contains("CUDA_OXIDE_BACKEND is set"), "{from_env}");
    }

    #[test]
    fn doctor_verified_components_unions_pin_list_with_required_floor() {
        // Pin lists everything: order preserved, no duplicates appended.
        let pin = RustToolchainPin {
            channel: "nightly-2026-04-03".to_string(),
            components: vec![
                "rust-src".to_string(),
                "rustc-dev".to_string(),
                "rust-analyzer".to_string(),
                "clippy".to_string(),
                "llvm-tools".to_string(),
            ],
        };
        assert_eq!(
            doctor_verified_components(&pin),
            vec![
                "rust-src",
                "rustc-dev",
                "rust-analyzer",
                "clippy",
                "llvm-tools"
            ]
        );

        // A trimmed pin still gets the hard floor appended.
        let trimmed = RustToolchainPin {
            channel: "nightly-2026-04-03".to_string(),
            components: vec!["clippy".to_string()],
        };
        assert_eq!(
            doctor_verified_components(&trimmed),
            vec!["clippy", "rust-src", "rustc-dev", "llvm-tools"]
        );
    }

    #[test]
    fn missing_rustup_components_detects_host_triple_suffixes() {
        let installed = "\
rust-src-aarch64-apple-darwin
clippy-aarch64-apple-darwin
";
        assert_eq!(
            missing_rustup_components(installed, &["rust-src", "llvm-tools"]),
            vec!["llvm-tools".to_string()]
        );
        assert!(missing_rustup_components(installed, &["rust-src"]).is_empty());
    }

    #[test]
    fn parse_compute_cap_rejects_failure_banners_and_garbage() {
        // nvidia-smi prints failure text to STDOUT, not stderr.
        assert_eq!(
            parse_compute_cap(
                "NVIDIA-SMI has failed because it couldn't communicate \
                 with the NVIDIA driver.\n"
            ),
            None
        );
        assert_eq!(parse_compute_cap(""), None);
        assert_eq!(parse_compute_cap("\n"), None);
        assert_eq!(parse_compute_cap("N/A\n"), None);
        assert_eq!(parse_compute_cap("12\n"), None);
        assert_eq!(parse_compute_cap("12.\n"), None);
        assert_eq!(parse_compute_cap(".5\n"), None);
        assert_eq!(parse_compute_cap("12.0.1\n"), None);
    }

    // All three skip cases inject the `CUDA_OXIDE_TARGET` probe rather than
    // reading the ambient one, so each asserts the slot it names instead of
    // passing because the developer happens to have the variable exported.

    #[test]
    fn detect_run_target_arch_skips_when_arch_explicit() {
        // --arch wins; never query the GPU.
        assert_eq!(
            detect_run_target_arch_with_env(Some("sm_120"), false, false),
            None
        );
    }

    #[test]
    fn detect_run_target_arch_skips_when_emit_nvvm_ir() {
        // NVVM IR mode requires explicit --arch; auto-detect must not run.
        assert_eq!(detect_run_target_arch_with_env(None, true, false), None);
    }

    #[test]
    fn detect_run_target_arch_skips_when_env_target_set() {
        // Slot 2 wins; never query the GPU. Injected rather than exported:
        // `set_var` is a data race against the `vars_os` reads the fingerprint
        // helpers perform on other test threads, which the cargo test harness
        // runs concurrently by default.
        assert_eq!(detect_run_target_arch_with_env(None, false, true), None);
    }

    fn write_list_example(
        examples_dir: &Path,
        name: &str,
        manifest_description: Option<&str>,
        readme: Option<&str>,
    ) {
        let example_dir = examples_dir.join(name);
        std::fs::create_dir_all(&example_dir).unwrap();

        let description = manifest_description
            .map(|value| format!("description = {value:?}\n"))
            .unwrap_or_default();

        std::fs::write(
            example_dir.join("Cargo.toml"),
            format!(
                "[package]\nname = {name:?}\nversion = \"0.1.0\"\nedition = \"2024\"\n{description}"
            ),
        )
        .unwrap();

        if let Some(readme) = readme {
            std::fs::write(example_dir.join("README.md"), readme).unwrap();
        }
    }

    #[test]
    fn readme_parser_extracts_title_description_and_requirements() {
        let parsed = parse_example_readme(
            "vecadd",
            r#"
# vecadd

## Vector Addition

Adds two vectors using one CUDA thread per element.

## Hardware Requirements

- **Minimum GPU**: sm_70+
- **CUDA Toolkit**: 12.x+
"#,
        );

        assert_eq!(parsed.title.as_deref(), Some("Vector Addition"));
        assert_eq!(
            parsed.description.as_deref(),
            Some("Adds two vectors using one CUDA thread per element.")
        );
        assert_eq!(
            parsed.requirements,
            ["Minimum GPU: sm_70+", "CUDA Toolkit: 12.x+"]
        );
    }

    #[test]
    fn readme_parser_does_not_use_run_as_title() {
        let parsed = parse_example_readme(
            "cuda_module_nested",
            r#"
# cuda_module_nested

## Run

Expected output:

```text
PASS
```

"#,
        );

        assert_eq!(parsed.title.as_deref(), Some("cuda_module_nested"));
        assert_eq!(parsed.description, None);
    }

    #[test]
    fn readme_parser_does_not_scan_later_headings_for_title() {
        let parsed = parse_example_readme(
            "example",
            r#"

# example

Introductory description.

## Build

Build instructions.

## Advanced Implementation Details

Internal details.
"#,
        );

        assert_eq!(parsed.title.as_deref(), Some("example"));
        assert_eq!(
            parsed.description.as_deref(),
            Some("Introductory description.")
        );
    }

    #[test]
    fn readme_parser_stops_description_at_next_heading() {
        let parsed = parse_example_readme(
            "vecadd",
            r#"

# vecadd

## Vector Addition

Adds two vectors on the GPU.

## Run

Run the example with cargo oxide.
"#,
        );

        assert_eq!(parsed.title.as_deref(), Some("Vector Addition"));
        assert_eq!(
            parsed.description.as_deref(),
            Some("Adds two vectors on the GPU.")
        );
    }

    #[test]
    fn requirement_parser_joins_wrapped_list_items() {
        let parsed = parse_example_readme(
            "example",
            r#"

# example

## Requirements

* CUDA Toolkit 13.1+ with nvcc and tileiras available. This example
  also requires the CUDA development libraries.
* Blackwell GPU with sm_100+ support.
  "#,
        );

        assert_eq!(
            parsed.requirements,
            [
                "CUDA Toolkit 13.1+ with nvcc and tileiras available. This example also requires the CUDA development libraries.",
                "Blackwell GPU with sm_100+ support.",
            ]
        );
    }

    #[test]
    fn requirement_parser_does_not_absorb_paragraph_after_blank_line() {
        // Modeled on the cpp_consumes_rust_device README: a bullet list under
        // the requirements heading, then a blank line, then a follow-up
        // paragraph and a code fence. The paragraph is a new paragraph, not a
        // wrapped continuation of the last bullet.
        let parsed = parse_example_readme(
            "cpp_consumes_rust_device",
            r#"
# cpp_consumes_rust_device

## Prerequisites

- CUDA Toolkit (nvcc, libNVVM, nvJitLink)
- Blackwell+ GPU (sm_100+) — LTOIR requires NVVM 20 dialect

If your default host compiler is newer than the CUDA Toolkit supports, set
`NVCC_CCBIN` or `CUDAHOSTCXX` before running the example:

```bash
NVCC_CCBIN=/usr/bin/g++-15 cargo oxide run cpp_consumes_rust_device
```
"#,
        );

        assert_eq!(
            parsed.requirements,
            [
                "CUDA Toolkit (nvcc, libNVVM, nvJitLink)",
                "Blackwell+ GPU (sm_100+) — LTOIR requires NVVM 20 dialect",
            ]
        );
    }

    #[test]
    fn requirement_parser_joins_wrapped_items_but_not_following_paragraphs() {
        // Modeled on the cutile_inter_kernel README: the last bullet wraps
        // across indented lines (joined), and the paragraph after the blank
        // line must not be glued onto it.
        let parsed = parse_example_readme(
            "cutile_inter_kernel",
            r#"
# cutile_inter_kernel

## Requirements

- cuda-oxide from this repository.
- CUDA Toolkit 13.1+ with `nvcc` and `tileiras` available. This example
  defaults `CUDA_TOOLKIT_PATH` to `/usr/local/cuda` through its local Cargo
  config; set `CUDA_TOOLKIT_PATH` yourself if your toolkit lives elsewhere.

`cargo oxide run` targets explicit `--arch` first, then `CUDA_OXIDE_TARGET`,
then auto-detects the local GPU.

## Run

Run instructions.
"#,
        );

        assert_eq!(
            parsed.requirements,
            [
                "cuda-oxide from this repository.",
                "CUDA Toolkit 13.1+ with nvcc and tileiras available. This example \
                 defaults CUDA_TOOLKIT_PATH to /usr/local/cuda through its local Cargo \
                 config; set CUDA_TOOLKIT_PATH yourself if your toolkit lives elsewhere.",
            ]
        );
    }

    #[test]
    fn requirement_parser_captures_ordered_list_items() {
        // Modeled on the mathdx_ffi_test README: prerequisites written as an
        // ordered list, followed by a paragraph that is not part of the list.
        let parsed = parse_example_readme(
            "mathdx_ffi_test",
            r#"
# mathdx_ffi_test

## Prerequisites

1. **CUDA Toolkit 12.x+** with nvcc
2. **MathDx Library** - Download from: https://developer.nvidia.com/cublasdx-downloads
3. **cuda-oxide compiler** toolchain

If your default host compiler is newer than the CUDA Toolkit supports, set
`NVCC_CCBIN` or `CUDAHOSTCXX` before running the example.
"#,
        );

        assert_eq!(
            parsed.requirements,
            [
                "CUDA Toolkit 12.x+ with nvcc",
                "MathDx Library - Download from: https://developer.nvidia.com/cublasdx-downloads",
                "cuda-oxide compiler toolchain",
            ]
        );
    }

    #[test]
    fn requirement_parser_recognizes_build_requirements_heading() {
        let parsed = parse_example_readme(
            "example",
            r#"
# example

## Build Requirements

- nvcc with `--expt-relaxed-constexpr`
"#,
        );

        assert_eq!(parsed.requirements, ["nvcc with --expt-relaxed-constexpr"]);
    }

    #[test]
    fn requirement_parser_parses_two_column_requirement_tables() {
        // Modeled on the abi_hmm README: requirements in a two-column table,
        // including an escaped pipe inside a cell.
        let parsed = parse_example_readme(
            "abi_hmm",
            r#"
# abi_hmm

## Requirements

| Requirement   | Minimum                                           |
|---------------|---------------------------------------------------|
| GPU           | Turing or newer (RTX 20xx+)                       |
| Linux Kernel  | 6.1.24+                                           |
| HMM Support   | `nvidia-smi -q \| grep Addressing` shows "HMM"    |

## Build and Run

Instructions.
"#,
        );

        assert_eq!(
            parsed.requirements,
            [
                "GPU: Turing or newer (RTX 20xx+)",
                "Linux Kernel: 6.1.24+",
                "HMM Support: nvidia-smi -q | grep Addressing shows \"HMM\"",
            ]
        );
    }

    #[test]
    fn requirement_parser_skips_tables_that_are_not_two_columns() {
        // A three-column table has no unambiguous name/value mapping, so it
        // must be skipped whole instead of half-parsed.
        let parsed = parse_example_readme(
            "example",
            r#"
# example

## Requirements

| Test  | Status | Description |
|-------|--------|-------------|
| alpha | Pass   | First test  |
| beta  | Pass   | Second test |
"#,
        );

        assert_eq!(parsed.requirements, Vec::<String>::new());
    }

    #[test]
    fn example_discovery_is_sorted_and_uses_manifest_fallback() {
        let root = unique_temp_dir("cargo_oxide_list_examples");
        std::fs::create_dir_all(&root).unwrap();

        write_list_example(&root, "zeta", Some("Manifest fallback description"), None);

        write_list_example(
            &root,
            "alpha",
            None,
            Some("# alpha\n\n## Alpha Example\n\nREADME description.\n"),
        );

        let examples = discover_examples(&root).unwrap();

        assert_eq!(
            examples
                .iter()
                .map(|example| example.name.as_str())
                .collect::<Vec<_>>(),
            ["alpha", "zeta"]
        );
        assert_eq!(examples[0].description, "README description.");
        assert_eq!(examples[1].description, "Manifest fallback description");

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn example_discovery_keeps_examples_without_readmes() {
        let root = unique_temp_dir("cargo_oxide_list_missing_readme");
        std::fs::create_dir_all(&root).unwrap();

        write_list_example(&root, "minimal", None, None);

        let examples = discover_examples(&root).unwrap();

        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].name, "minimal");
        assert_eq!(examples[0].description, "No description documented.");

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn example_discovery_skips_directory_without_manifest() {
        let root = unique_temp_dir("cargo_oxide_list_missing_manifest");
        std::fs::create_dir_all(root.join("scratch")).unwrap();

        write_list_example(&root, "real", Some("A real example"), None);

        let examples =
            discover_examples(&root).expect("manifest-less directories must not abort listing");

        assert_eq!(
            examples
                .iter()
                .map(|example| example.name.as_str())
                .collect::<Vec<_>>(),
            ["real"]
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ptx_artifact_paths_normalize_hyphenated_example_names() {
        let root = unique_temp_dir("cargo_oxide_inspect_regular");
        std::fs::create_dir_all(&root).unwrap();

        std::fs::write(
            root.join("Cargo.toml"),
            r#"[package]
name = "demo-app"
version = "0.1.0"
edition = "2024"
"#,
        )
        .unwrap();

        assert_eq!(
            ptx_artifact_paths(&root, "demo-app"),
            vec![root.join("demo_app.ptx")]
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn ptx_artifact_paths_resolve_interop_device_artifacts() {
        let root = unique_temp_dir("cargo_oxide_inspect_interop");
        let device_dir = root.join("device");
        std::fs::create_dir_all(&device_dir).unwrap();

        std::fs::write(
            root.join("Cargo.toml"),
            r#"[package]
name = "host-app"
version = "0.1.0"
edition = "2024"

[package.metadata.cuda-oxide]
interop = "device"

[[package.metadata.cuda-oxide.device-crates]]
manifest-path = "device/Cargo.toml"
ptx-dir = "generated"
artifact-name = "custom-device"
"#,
        )
        .unwrap();

        std::fs::write(
            device_dir.join("Cargo.toml"),
            r#"[package]
name = "device-app"
version = "0.1.0"
edition = "2024"
"#,
        )
        .unwrap();

        assert_eq!(
            ptx_artifact_paths(&root, "host-app"),
            vec![root.join("generated/custom_device.ptx")]
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn read_ptx_artifact_returns_exact_contents() {
        let root = unique_temp_dir("cargo_oxide_read_ptx");
        std::fs::create_dir_all(&root).unwrap();

        let path = root.join("demo.ptx");
        std::fs::write(&path, ".version 8.0\n.target sm_90\n").unwrap();

        assert_eq!(
            read_ptx_artifact(&path).unwrap(),
            ".version 8.0\n.target sm_90\n"
        );

        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn list_json_has_versioned_stable_shape() {
        let examples = vec![ExampleInfo {
            name: "vecadd".to_string(),
            title: "Vector Addition".to_string(),
            description: "Adds two vectors.".to_string(),
            requirements: vec!["Minimum GPU: sm_70+".to_string()],
        }];

        let output = format_examples_json(&examples).unwrap();
        let value: serde_json::Value = serde_json::from_str(&output).unwrap();

        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["examples"][0]["name"], "vecadd");
        assert_eq!(
            value["examples"][0]["requirements"][0],
            "Minimum GPU: sm_70+"
        );
        assert!(output.ends_with('\n'));
    }

    #[test]
    fn read_ptx_artifact_reports_missing_file() {
        let root = unique_temp_dir("cargo_oxide_missing_ptx");
        let path = root.join("missing.ptx");

        let error = read_ptx_artifact(&path).unwrap_err();

        assert!(error.contains("could not read generated PTX"));
        assert!(error.contains("missing.ptx"));
    }

    #[test]
    fn nvvm_ir_requested_reads_project_configuration() {
        let ctx = Context {
            workspace_root: PathBuf::from("/tmp/project"),
            codegen_crate: PathBuf::from("/tmp/project"),
            examples_dir: PathBuf::from("/tmp/project"),
            backend_so: PathBuf::from("/tmp/backend.so"),
            is_workspace: false,
            config: OxideConfig {
                env: vec![("CUDA_OXIDE_EMIT_NVVM_IR".to_string(), "true".to_string())],
                ..OxideConfig::default()
            },
        };

        assert_eq!(nvvm_ir_requested_with_env(&ctx, None), Ok(true));
    }

    #[test]
    fn nvvm_ir_requested_accepts_disabled_project_configuration() {
        let ctx = Context {
            workspace_root: PathBuf::from("/tmp/project"),
            codegen_crate: PathBuf::from("/tmp/project"),
            examples_dir: PathBuf::from("/tmp/project"),
            backend_so: PathBuf::from("/tmp/backend.so"),
            is_workspace: false,
            config: OxideConfig {
                env: vec![("CUDA_OXIDE_EMIT_NVVM_IR".to_string(), "false".to_string())],
                ..OxideConfig::default()
            },
        };

        assert_eq!(nvvm_ir_requested_with_env(&ctx, None), Ok(false));
    }

    #[test]
    fn nvvm_ir_requested_env_disable_overrides_enabled_project_configuration() {
        let ctx = Context {
            workspace_root: PathBuf::from("/tmp/project"),
            codegen_crate: PathBuf::from("/tmp/project"),
            examples_dir: PathBuf::from("/tmp/project"),
            backend_so: PathBuf::from("/tmp/backend.so"),
            is_workspace: false,
            config: OxideConfig {
                env: vec![("CUDA_OXIDE_EMIT_NVVM_IR".to_string(), "true".to_string())],
                ..OxideConfig::default()
            },
        };

        // The process environment outranks `cuda-oxide.toml`: an explicit
        // false in the environment wins over the project's `true`, in either
        // accepted spelling.
        for disabled in ["false", "0"] {
            assert_eq!(
                nvvm_ir_requested_with_env(&ctx, Some(disabled.into())),
                Ok(false)
            );
        }
    }

    #[test]
    fn scaffold_sync_template_uses_launch_contract_and_docs() {
        let files = scaffold_files("demo_kernel", false);
        assert!(files.cargo_toml.contains("name = \"demo_kernel\""));
        assert!(files.readme.contains("cargo oxide doctor"));
        assert!(files.readme.contains("cargo oxide run"));
        assert!(files.gitignore.contains("/target/"));
        // The template uses the launch_bounds / launch_contract attribute
        // macros, so the cuda_device import must bring them in; a scaffolded
        // project fails to compile without this exact line.
        assert!(files.main_rs.starts_with(
            "use cuda_device::{kernel, launch_bounds, launch_contract, thread, DisjointSlice};"
        ));
        assert!(
            files
                .main_rs
                .contains("#[launch_contract(domain = 1, block = (256, 1, 1))]")
        );
        assert!(files.main_rs.contains("prepare_vecadd"));
        assert!(files.main_rs.contains("LaunchConfig1D"));
        assert!(!files.main_rs.contains("LaunchConfig::for_num_elems"));
    }

    #[test]
    fn scaffold_async_template_keeps_async_deps_and_docs() {
        let files = scaffold_files("async_demo", true);
        assert!(files.cargo_toml.contains("cuda-async"));
        assert!(files.cargo_toml.contains("tokio"));
        assert!(files.readme.contains("async cuda-oxide"));
        assert!(files.readme.contains("cargo oxide doctor"));
        // The async README must stand alone: it describes the async launch
        // path and never talks about "the sync template".
        assert!(files.readme.contains("DeviceOperation"));
        assert!(!files.readme.contains("sync template"));
        assert!(files.gitignore.contains("**/*.ptx"));
        assert!(files.main_rs.contains("vecadd_async"));
        assert!(files.main_rs.contains("use cuda_host::cuda_module;"));
        assert!(!files.main_rs.contains("use cuda_device::{cuda_module"));
    }

    #[test]
    fn scaffold_gitignore_covers_every_clean_artifact_suffix() {
        let gitignore = scaffold_gitignore();
        assert!(gitignore.contains("/target/"));
        assert!(gitignore.contains("**/*.bc"));
        for suffix in GENERATED_ARTIFACT_SUFFIXES {
            // Match whole lines, not substrings: `**/*.cubin.target` contains
            // `**/*.cubin` as a substring, so `contains()` would keep passing
            // even if the `cubin` pattern itself were dropped.
            let pattern = format!("**/*.{suffix}");
            assert!(
                gitignore.lines().any(|line| line == pattern),
                "scaffold .gitignore must ignore clean suffix `{suffix}`"
            );
        }
    }

    #[test]
    fn device_debug_env_value_matches_the_backend_parser() {
        // These must be strings `device_debug_kind_with_override` accepts; a typo
        // would silently fall through to the profile-derived default instead of
        // failing, so pin them.
        assert_eq!(DeviceDebug::Off.env_value(), None);
        assert_eq!(DeviceDebug::LineTables.env_value(), Some("line"));
        assert_eq!(DeviceDebug::Full.env_value(), Some("full"));
    }

    #[test]
    fn passthrough_fingerprint_separates_the_device_debug_policies() {
        let ctx = test_context(OxideConfig::default());
        let base = CargoPassthroughOptions {
            verbose: false,
            emit_nvvm_ir: false,
            arch: None,
            features: None,
            cargo_target_dir: None,
            device_codegen_crate: None,
            device_cfgs: &[],
            no_fmad: false,
            unchecked_indexing: false,
            materialize_cubin: false,
            device_debug: DeviceDebug::Off,
        };
        let line_tables = CargoPassthroughOptions {
            device_debug: DeviceDebug::LineTables,
            ..base
        };
        let full = CargoPassthroughOptions {
            device_debug: DeviceDebug::Full,
            ..base
        };
        let materialization = MaterializationMode::default();
        // Empty inherited env, for the same reason as the sibling fingerprint
        // tests: an ambient CUDA_OXIDE_DEBUG is folded in on its own, which would
        // collapse these onto the base.
        let inherited_env = BTreeMap::new();
        let fp = |opts: &CargoPassthroughOptions<'_>| {
            passthrough_codegen_fingerprint_with_env(
                &ctx,
                opts,
                None,
                None,
                &materialization,
                &inherited_env,
            )
        };
        // The policy changes what libNVVM and nvJitLink are asked to do (`-g`,
        // `-opt=0`, `-lineinfo`), so it must not share a fingerprint with the
        // default -- otherwise Cargo reuses artifacts built without it.
        let off = fp(&base);
        assert_ne!(off, fp(&line_tables));
        assert_ne!(off, fp(&full));
        assert_ne!(fp(&line_tables), fp(&full));
    }
}
