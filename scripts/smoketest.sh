#!/usr/bin/env bash
#
# scripts/smoketest.sh -- run every cuda-oxide example and report pass/fail
# per GPU-aware gating rules.
#
# Default behavior (no arguments): run every example under
# crates/rustc-codegen-cuda/examples/. Examples are categorized and a
# category-specific verdict rule is applied to the cargo output:
#
#   standard     -- execution must succeed (SUCCESS/PASS/Complete marker).
#   error        -- compilation must fail with a recognized diagnostic.
#                   Covers both intentional diagnostic fixtures and known
#                   support gaps (see STATUS.md). Signal termination is
#                   never accepted.
#   tcgen05      -- 5th-gen tensor cores; sm_100 datacenter only. On sm_100
#                   require full execution; elsewhere PTX compilation is
#                   sufficient.
#   wgmma        -- Hopper only (sm_90a). On Hopper require execution;
#                   elsewhere PTX compilation is sufficient.
#   ltoir        -- runs with `--emit-nvvm-ir --arch=<host>`; the host
#                   compute capability is detected via `nvidia-smi` so the
#                   resulting cubin actually loads. Execution must succeed
#                   — or the example may opt to `println!("skipping: ...")`
#                   with exit 0 (e.g. mathdx_ffi_test when MATHDX_ROOT is
#                   unset), in which case we require the cuda-oxide
#                   NVVM IR (`.ll`) to have been generated.
#   auto-nvvm    -- runs without NVVM or architecture flags to check automatic
#                   libdevice and target selection. Compile-only CI supplies a
#                   target because no GPU is available.
#   NVVM_VERIFY_EXAMPLES are compiled through the real libNVVM verifier and
#                   compiler in compile-only mode.
#
# Categories are bash arrays at the top of this file. When adding an
# error* example, also update STATUS.md and run
# scripts/check-error-example-status.sh to verify both are in sync.
#
# See --help for runtime flags.

set -uo pipefail

# ---- Example categorization ---------------------------------------------

TCGEN05_EXAMPLES=(gemm_sol tcgen05 tcgen05_matmul)
WGMMA_EXAMPLES=(wgmma)
LTOIR_EXAMPLES=(addressof_sharedarray cpp_consumes_rust_device device_ffi_test legacy_nvvm_pointer_shapes manual_launch_libdevice mathdx_ffi_test primitive_stress)
AUTO_NVVM_EXAMPLES=(libdevice_math)
NVVM_VERIFY_EXAMPLES=(libdevice_math legacy_nvvm_pointer_shapes primitive_stress)
ERROR_EXAMPLES=(error error_wgmma_mma_unimplemented error_set_discriminant_unhandled error_drop_glue error_heap_alloc error_missing_device_attr)

classify() {
    local ex="$1" cat
    for cat in "${TCGEN05_EXAMPLES[@]}";     do [[ "$ex" == "$cat" ]] && { echo tcgen05;     return; }; done
    for cat in "${WGMMA_EXAMPLES[@]}";       do [[ "$ex" == "$cat" ]] && { echo wgmma;       return; }; done
    for cat in "${LTOIR_EXAMPLES[@]}";       do [[ "$ex" == "$cat" ]] && { echo ltoir;       return; }; done
    for cat in "${AUTO_NVVM_EXAMPLES[@]}";   do [[ "$ex" == "$cat" ]] && { echo auto-nvvm;   return; }; done
    for cat in "${ERROR_EXAMPLES[@]}";       do [[ "$ex" == "$cat" ]] && { echo error;       return; }; done
    echo standard
}

verify_nvvm_in_compile_only() {
    local ex="$1" candidate
    for candidate in "${NVVM_VERIFY_EXAMPLES[@]}"; do
        [[ "$ex" == "$candidate" ]] && return 0
    done
    return 1
}

# ---- CLI -----------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: scripts/smoketest.sh [OPTIONS]

Run every cuda-oxide example and report PASS/FAIL per GPU-aware gating
rules. With no options, runs all examples.

OPTIONS
  -o, --only PATTERN   Run only examples whose name matches the bash regex
                       PATTERN (e.g. -o 'tcgen05|wgmma').
  -s, --skip PATTERN   Skip examples whose name matches PATTERN.
  -c, --compile-only   Compile each example without running it. Most use
                       `cargo oxide build`; designated NVVM regressions use
                       `emit-ltoir` to include real libNVVM verification.
                       Non-error categories must leave a fresh artifact;
                       error examples must still fail. Works on GPU-less CI.
  -x, --fail-fast      Stop at the first failure.
  -v, --verbose        Stream cargo output live (instead of capturing to
                       a per-example log file). Verdict is printed at the
                       end of each example.
      --keep-logs      Retain per-example logs on success as well as
                       failure. Logs for failures are always kept.
      --no-color       Disable ANSI color. Also honours the NO_COLOR env
                       var (https://no-color.org/).
  -h, --help           Show this help and exit.

POSITIONALS
  Any bare arguments are treated as additive --only patterns and joined
  with `|`. If --only is also supplied, positionals extend it (OR).

EXAMPLES
  scripts/smoketest.sh                 # run all examples
  scripts/smoketest.sh vecadd          # examples matching 'vecadd'
  scripts/smoketest.sh vecadd gemm     # matching 'vecadd' OR 'gemm'
  scripts/smoketest.sh -o '^vecadd$'   # exact-match form
  scripts/smoketest.sh -s 'wgmma|tma'  # skip wgmma and tma examples
  scripts/smoketest.sh -x -v vecadd    # stop on first fail, stream output
  scripts/smoketest.sh --compile-only  # GPU-less compile gate (used by CI)

Per-example logs live under .smoketest-logs/ by default. Set
SMOKETEST_LOG_DIR to override this path.
EOF
}

ONLY=""
SKIP=""
FAIL_FAST=0
VERBOSE=0
KEEP_LOGS=0
FORCE_NO_COLOR=0
COMPILE_ONLY=0
declare -a POSITIONAL=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|--only)      [[ $# -lt 2 ]] && { echo "error: $1 requires a pattern" >&2; exit 2; }; ONLY="$2"; shift 2;;
        -s|--skip)      [[ $# -lt 2 ]] && { echo "error: $1 requires a pattern" >&2; exit 2; }; SKIP="$2"; shift 2;;
        -c|--compile-only) COMPILE_ONLY=1; shift;;
        -x|--fail-fast) FAIL_FAST=1; shift;;
        -v|--verbose)   VERBOSE=1; shift;;
        --keep-logs)    KEEP_LOGS=1; shift;;
        --no-color)     FORCE_NO_COLOR=1; shift;;
        -h|--help)      usage; exit 0;;
        --)             shift; POSITIONAL+=("$@"); break;;
        -*)             echo "error: unknown option: $1" >&2; usage >&2; exit 2;;
        *)              POSITIONAL+=("$1"); shift;;
    esac
done

# Bare positionals act as additive --only patterns joined with `|`.
# Combine them with any explicit --only (OR, not replace) so that
# `--only foo bar` and `-o foo bar` produce `foo|bar`.
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
    joined="$(IFS='|'; echo "${POSITIONAL[*]}")"
    if [[ -n "${ONLY}" ]]; then
        ONLY="${ONLY}|${joined}"
    else
        ONLY="${joined}"
    fi
fi

# ---- Preflight -----------------------------------------------------------

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

if [[ ! -f "Cargo.toml" ]] || [[ ! -d "crates/rustc-codegen-cuda/examples" ]]; then
    echo "error: must be run from inside the cuda-oxide repo (got ${PWD})" >&2
    exit 2
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "error: cargo not found in PATH" >&2
    exit 2
fi

if ! cargo oxide --help >/dev/null 2>&1; then
    echo "error: 'cargo oxide' subcommand missing; build it with:" >&2
    echo "         cargo build -p cargo-oxide --release" >&2
    exit 2
fi

# ---- Colors --------------------------------------------------------------

if [[ -t 1 ]] && [[ -z "${NO_COLOR:-}" ]] && [[ ${FORCE_NO_COLOR} -eq 0 ]]; then
    C_PASS=$'\e[32m'; C_FAIL=$'\e[31m'; C_SKIP=$'\e[33m'
    C_DIM=$'\e[2m'; C_BOLD=$'\e[1m'; C_RESET=$'\e[0m'
else
    C_PASS=""; C_FAIL=""; C_SKIP=""; C_DIM=""; C_BOLD=""; C_RESET=""
fi

# ---- Banner --------------------------------------------------------------

git_head="$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
git_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
# nvidia-smi can be present yet broken (driver mismatch, sandboxes,
# containers) and it prints its failure text to STDOUT, so trust it only
# when it exits 0 AND the compute capability parses. One probe feeds both
# the banner and the LTOIR arch so they can never disagree.
host_cc=""
if gpu_query="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null)"; then
    gpu_info="$(head -1 <<<"${gpu_query}")"
    host_cc="$(awk -F', *' '{print $2}' <<<"${gpu_info}" | tr -d '[:space:]')"
else
    gpu_info='no GPU detected'
fi

# Detect host compute capability for the ltoir category: the cubin produced
# by the LTOIR linker must match the GPU's arch or it refuses to load.
# nvidia-smi reports something like "12.0" → sm_120, "10.0" → sm_100.
if [[ "${host_cc}" =~ ^[0-9]+\.[0-9]+$ ]]; then
    # Strip the dot: 12.0 -> 120
    LTOIR_ARCH="sm_${host_cc//./}"
else
    # No working GPU detected. ltoir examples will likely fail to execute,
    # but pick a sensible floor so the cuda-oxide side still compiles.
    LTOIR_ARCH="sm_90"
fi

printf "%scuda-oxide smoketest%s @ %s%s%s (%s)\n" "${C_BOLD}" "${C_RESET}" "${C_BOLD}" "${git_head}" "${C_RESET}" "${git_branch}"
printf "GPU: %s\n" "${gpu_info}"
printf "LTOIR arch: %s\n" "${LTOIR_ARCH}"
if [[ ${COMPILE_ONLY} -eq 1 ]]; then
    printf "Mode: compile-only (device artifacts only; nothing is executed)\n"
fi
if [[ -n "${ONLY}" ]]; then printf "Filter --only: %s\n" "${ONLY}"; fi
if [[ -n "${SKIP}" ]]; then printf "Filter --skip: %s\n" "${SKIP}"; fi
echo ""

# ---- Example selection ---------------------------------------------------

mapfile -t ALL_EXAMPLES < <(
    cd crates/rustc-codegen-cuda/examples
    for manifest in */Cargo.toml; do
        [[ -e "${manifest}" ]] || continue
        echo "${manifest%/Cargo.toml}"
    done | sort
)

# An example dir without a top-level Cargo.toml would be skipped by the
# glob above and silently shrink coverage (e.g. a restructure that nests
# the manifest one level down). Fail loudly instead.
for dir in crates/rustc-codegen-cuda/examples/*/; do
    if [[ ! -f "${dir}Cargo.toml" ]]; then
        echo "error: ${dir} has no top-level Cargo.toml; every directory under" >&2
        echo "       crates/rustc-codegen-cuda/examples/ must be an example crate" >&2
        exit 2
    fi
done

selected=()
for ex in "${ALL_EXAMPLES[@]}"; do
    if [[ -n "${ONLY}" ]] && ! [[ "${ex}" =~ ${ONLY} ]]; then continue; fi
    if [[ -n "${SKIP}" ]] &&   [[ "${ex}" =~ ${SKIP} ]]; then continue; fi
    selected+=("${ex}")
done

total=${#selected[@]}
if [[ ${total} -eq 0 ]]; then
    echo "error: no examples matched the given filters" >&2
    exit 1
fi

# ---- Verdict functions ---------------------------------------------------
#
# Each verdict_* function consumes a log file + exit code, prints the
# classification string to stdout, and returns 0 (pass) or 1 (fail).
# They never run cargo themselves; that is the caller's job.

verdict_standard() {
    local log="$1" ec="$2"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if [[ ${ec} -ne 0 ]]; then   echo "FAIL (exit=${ec})";                    return 1; fi
    if grep_failure_markers "${log}"; then
        echo "FAIL (failure marker in output)"
        return 1
    fi
    # `skipping:` is an explicit, graceful opt-out (e.g. cluster on
    # pre-Hopper, mathdx_ffi_test with no MathDx SDK). Accept it as PASS so
    # standard-category examples can gate themselves on hardware/SDK presence
    # without having to fake a success marker.
    if grep -qE '^[[:space:]]*skipping:' "${log}"; then
        echo "PASS (skipped)"
        return 0
    fi
    if grep -qE 'SUCCESS|PASS|Complete' "${log}"; then echo "PASS"; return 0; fi
    echo "FAIL (no success marker)"
    return 1
}

# Returns 0 iff the log contains any of our known failure signals. Designed
# to be aggressive about false negatives — we'd rather flag a borderline
# example for human review than miss a regression. Each pattern is anchored
# enough not to match incidental prose ("This will fail if...", or the
# success line "(no assertion failed)" some examples print).
grep_failure_markers() {
    local log="$1"
    grep -qE \
        -e '(^|[[:space:]])FAIL(ED)?($|[[:space:]:!.])' \
        -e '(^|[[:space:]])✗($|[[:space:]])' \
        -e 'panicked at |thread .* panicked' \
        -e 'assertion failed:|assertion `[^`]*` failed' \
        -e 'illegal memory access|invalid argument|misaligned address' \
        -e 'CUDA(_ERROR)?[ _][A-Z_]*(FAIL|ERROR)' \
        -e 'fatal( error)?:' \
        "${log}"
}

verdict_error() {
    local log="$1" ec="$2"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if grep -qE 'Device codegen failed|Translation failed|Compilation error|Unsupported construct' "${log}"; then
        echo "PASS (expected compile failure)"
        return 0
    fi
    # Non-zero exit + a real rustc/cargo error line is still a legit
    # "example refused to compile" signal. Crucially, we require the error
    # line: just exit=42 with no diagnostic is NOT accepted (unlike the old
    # CLAUDE.md blob).
    if [[ ${ec} -ne 0 ]] && grep -qE '^error(\[|:)|error: could not compile|error aborting due to' "${log}"; then
        echo "PASS (expected compile failure, exit=${ec})"
        return 0
    fi
    if [[ ${ec} -eq 0 ]]; then
        echo "FAIL (compilation succeeded, expected failure)"
    else
        echo "FAIL (exit=${ec} but no compile-error marker)"
    fi
    return 1
}

verdict_tcgen05() {
    local log="$1" ec="$2"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if grep -qE 'WARNING: tcgen05 requires|Skipping GPU test: requires sm_100|Skipping benchmark: requires sm_100|tcgen05 \(5th gen tensor cores\) requires sm_100|PTX was generated successfully' "${log}"; then
        if grep -qE 'PTX written|PTX Verification|PTX file generated' "${log}"; then
            echo "PASS (tcgen05, PTX compiled)"
            return 0
        fi
        echo "FAIL (tcgen05, PTX not generated)"
        return 1
    fi
    if [[ ${ec} -ne 0 ]]; then echo "FAIL (tcgen05, exit=${ec})"; return 1; fi
    if grep_failure_markers "${log}"; then
        echo "FAIL (tcgen05, failure marker in output)"
        return 1
    fi
    if grep -qE 'SUCCESS|PASS|Complete' "${log}"; then echo "PASS (tcgen05, executed)"; return 0; fi
    echo "FAIL (tcgen05, no success marker)"
    return 1
}

verdict_wgmma() {
    local log="$1" ec="$2"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if grep -qE 'WARNING: WGMMA requires|WGMMA is Hopper-only|PTX load failed \(expected on non-Hopper\)|PTX module loaded' "${log}"; then
        if grep -qE 'PTX written|PTX Verification|PTX file generated|inspect generated PTX|\.ptx' "${log}"; then
            echo "PASS (wgmma, PTX compiled)"
            return 0
        fi
        echo "FAIL (wgmma, PTX not generated)"
        return 1
    fi
    if [[ ${ec} -ne 0 ]]; then echo "FAIL (wgmma, exit=${ec})"; return 1; fi
    if grep_failure_markers "${log}"; then
        echo "FAIL (wgmma, failure marker in output)"
        return 1
    fi
    if grep -qE 'SUCCESS|PASS|Complete' "${log}"; then echo "PASS (wgmma, executed)"; return 0; fi
    echo "FAIL (wgmma, no success marker)"
    return 1
}

verdict_ltoir() {
    local ex="$1" log="$2" ec="$3"
    # Hyphens in example names become underscores in the crate-named
    # artifact (see verdict_compile).
    local ll_file="crates/rustc-codegen-cuda/examples/${ex}/${ex//-/_}.ll"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    # `skipping:` marker -- the example opted out (e.g. mathdx_ffi_test with
    # MATHDX_ROOT unset). Accept as pass so long as the cuda-oxide side
    # still produced NVVM IR.
    if grep -qE '^[[:space:]]*skipping:' "${log}"; then
        if [[ -f "${ll_file}" ]]; then
            echo "PASS (LTOIR, skipped: NVVM IR generated)"
            return 0
        fi
        echo "FAIL (LTOIR, skipped but no NVVM IR)"
        return 1
    fi
    if [[ ${ec} -ne 0 ]]; then   echo "FAIL (LTOIR, exit=${ec})";             return 1; fi
    if grep_failure_markers "${log}"; then
        echo "FAIL (LTOIR, failure marker in output)"
        return 1
    fi
    if grep -qE 'SUCCESS|PASS|Complete|NVVM IR is ready' "${log}"; then
        echo "PASS (LTOIR)"
        return 0
    fi
    echo "FAIL (LTOIR, no success marker)"
    return 1
}

# Compile-only verdict, used for every non-error category when
# --compile-only is set. Two requirements:
#   1. `cargo oxide build` exited 0. Device codegen failures are rustc
#      fatals (see rustc-codegen-cuda/src/lib.rs join on device results),
#      so a broken device pipeline cannot exit 0.
#   2. A fresh device artifact exists: {ex}.ptx, or {ex}.ll plus a concrete
#      {ex}.target for the NVVM-IR path. cargo-oxide deletes stale ones before
#      building (clean_generated_files), so presence proves this build emitted
#      them. This catches collector regressions where the build "succeeds"
#      because no #[kernel] was found and device codegen never ran, and prevents
#      target-specific NVVM IR from being published without its source target.
# Interop examples write PTX into their configured ptx_dir instead, and
# cargo-oxide itself verifies that file exists (exits non-zero if not),
# so the exit code alone is trusted for them.
verdict_compile() {
    local ex="$1" log="$2" ec="$3"
    local ex_dir="crates/rustc-codegen-cuda/examples/${ex}"
    # Artifacts are named after the crate, and cargo normalizes hyphens
    # to underscores (rustlantis-smoke emits rustlantis_smoke.ptx). This
    # assumes dir name == package name, which holds for every example and
    # is equally assumed by cargo-oxide's clean_generated_files; a renamed
    # package fails here loudly rather than passing on a stale artifact.
    local artifact="${ex//-/_}"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi
    if [[ ${ec} -ne 0 ]]; then   echo "FAIL (exit=${ec})";                    return 1; fi
    if verify_nvvm_in_compile_only "${ex}"; then
        if [[ -s "${ex_dir}/${artifact}.ll" && -s "${ex_dir}/${artifact}.ltoir" ]]; then
            echo "PASS (verified and compiled by libNVVM)"
            return 0
        fi
        echo "FAIL (libNVVM compile produced no fresh LTOIR)"
        return 1
    fi
    if [[ -s "${ex_dir}/${artifact}.ptx" ]]; then
        echo "PASS (compiled)"
        return 0
    fi
    if [[ -s "${ex_dir}/${artifact}.ll" ]]; then
        local target_file="${ex_dir}/${artifact}.target"
        if [[ -s "${target_file}" ]] && grep -qE '^sm_[0-9]+[af]?$' "${target_file}"; then
            echo "PASS (compiled NVVM IR for $(tr -d '[:space:]' < "${target_file}"))"
            return 0
        fi
        echo "FAIL (NVVM IR emitted without a concrete .target sidecar)"
        return 1
    fi
    # Match only the real interop config shapes ([[package.metadata.
    # cuda-oxide.device-crates]] tables or a device-crates = [...] key),
    # not the substring anywhere: a stray comment must not silently
    # exempt an example from the artifact check.
    if grep -qE '^[[:space:]]*(\[\[package\.metadata\.cuda-oxide\.device-crates\]\]|device-crates[[:space:]]*=)' \
        "${ex_dir}/Cargo.toml" 2>/dev/null; then
        echo "PASS (compiled, interop)"
        return 0
    fi
    echo "FAIL (built, but no device artifact emitted)"
    return 1
}

# ---- Runner --------------------------------------------------------------

# Run cargo oxide for ${ex} in category ${cat}. Writes to ${log}. Returns
# the cargo process exit code via the global ${CARGO_EC}.
run_cargo() {
    local ex="$1" log="$2" cat="$3"
    # Designated NVVM examples use `emit-ltoir` in compile-only mode so CI
    # checks both textual export and real libNVVM compilation. Other examples
    # use `build`.
    if [[ ${COMPILE_ONLY} -eq 1 ]] && verify_nvvm_in_compile_only "${ex}"; then
        local -a args=("emit-ltoir" "${ex}" "--arch=${LTOIR_ARCH}")
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${args[@]}" 2>&1 | tee "${log}"
            CARGO_EC=${PIPESTATUS[0]}
        else
            cargo oxide "${args[@]}" >"${log}" 2>&1
            CARGO_EC=$?
        fi
        return
    fi

    local verb="run"
    if [[ ${COMPILE_ONLY} -eq 1 ]]; then verb="build"; fi
    local -a args=("${verb}" "${ex}")
    if [[ "${cat}" == "ltoir" || ( "${cat}" == "auto-nvvm" && ${COMPILE_ONLY} -eq 1 ) ]]; then
        args+=("--emit-nvvm-ir" "--arch=${LTOIR_ARCH}")
    fi
    if [[ ${VERBOSE} -eq 1 ]]; then
        cargo oxide "${args[@]}" 2>&1 | tee "${log}"
        CARGO_EC=${PIPESTATUS[0]}
    else
        cargo oxide "${args[@]}" >"${log}" 2>&1
        CARGO_EC=$?
    fi
}

# ---- Shared example target dir -------------------------------------------
#
# Each example under crates/rustc-codegen-cuda/examples/ is its own standalone
# cargo workspace (the codegen backend is swapped in via RUSTFLAGS, so they
# can't live in the root [workspace]). By default every `cargo oxide run`
# materializes its own target/ and recompiles the whole shared dependency tree
# (cuda-device, cuda-host, proc-macros, bindgen, ...) from scratch — the
# dominant cost of this script. Point all example builds at one shared
# CARGO_TARGET_DIR: cargo fingerprints each unit by package + features +
# workspace_root + toolchain, so identical deps built with the same pinned
# nightly + backend RUSTFLAGS compile exactly once and are reused across every
# example. (Same trick the clippy CI job uses for these workspaces.)
#
# The codegen backend .so is built FIRST, with CARGO_TARGET_DIR explicitly
# cleared, so it lands at its fixed path (crates/rustc-codegen-cuda/target/
# debug) where cargo-oxide looks for it. A CARGO_TARGET_DIR in scope during
# that build would redirect the .so into the shared dir and break backend
# discovery. `cargo oxide setup` is a fast no-op when the backend is current.
printf "%sBuilding codegen backend (one-time; fast if current)...%s\n" "${C_DIM}" "${C_RESET}"
if ! env -u CARGO_TARGET_DIR cargo oxide setup >/dev/null 2>&1; then
    echo "error: failed to build the codegen backend; run 'cargo oxide setup' to see why" >&2
    exit 2
fi
# Honor an externally-set CARGO_TARGET_DIR (e.g. CI); otherwise share one under
# the repo's target/ so it is gitignored and cleaned by `cargo clean`.
: "${CARGO_TARGET_DIR:=${repo_root}/target/oxide-examples}"
export CARGO_TARGET_DIR
printf "Examples share CARGO_TARGET_DIR=%s\n\n" "${CARGO_TARGET_DIR}"

# ---- Main loop -----------------------------------------------------------

log_dir="${SMOKETEST_LOG_DIR:-.smoketest-logs}"
mkdir -p "${log_dir}"

pass=0
failures=()
started=${SECONDS}
i=0

for ex in "${selected[@]}"; do
    i=$((i + 1))
    cat="$(classify "${ex}")"
    log="${log_dir}/${ex}.log"
    : > "${log}"

    if [[ ${VERBOSE} -eq 1 ]]; then
        printf "%s[%2d/%2d] %s (%s)%s\n" "${C_BOLD}" "${i}" "${total}" "${ex}" "${cat}" "${C_RESET}"
    else
        printf "[%2d/%2d] %-32s ... " "${i}" "${total}" "${ex}"
    fi

    t0=${SECONDS}
    run_cargo "${ex}" "${log}" "${cat}"
    ec=${CARGO_EC}
    dt=$((SECONDS - t0))

    if [[ ! -f "${log}" ]]; then
        verdict="FAIL (log missing: ${log})"
        status=1
    elif [[ ${COMPILE_ONLY} -eq 1 && "${cat}" != "error" ]]; then
        # Compile-only collapses the GPU-gated categories: with nothing
        # executed, "PTX (or NVVM IR) compiled" is the bar for everything
        # except error examples, which must still fail with a diagnostic.
        verdict="$(verdict_compile "${ex}" "${log}" "${ec}")" && status=0 || status=$?
    else
        case "${cat}" in
            error)       verdict="$(verdict_error       "${log}" "${ec}")"        && status=0 || status=$? ;;
            tcgen05)     verdict="$(verdict_tcgen05     "${log}" "${ec}")"        && status=0 || status=$? ;;
            wgmma)       verdict="$(verdict_wgmma       "${log}" "${ec}")"        && status=0 || status=$? ;;
            ltoir)       verdict="$(verdict_ltoir       "${ex}" "${log}" "${ec}")" && status=0 || status=$? ;;
            auto-nvvm)   verdict="$(verdict_ltoir       "${ex}" "${log}" "${ec}")" && status=0 || status=$? ;;
            standard)    verdict="$(verdict_standard    "${log}" "${ec}")"        && status=0 || status=$? ;;
            *)           verdict="FAIL (unknown category: ${cat})"; status=1 ;;
        esac
    fi

    if [[ ${status} -eq 0 ]]; then
        color="${C_PASS}"
    else
        color="${C_FAIL}"
    fi

    if [[ ${VERBOSE} -eq 1 ]]; then
        printf "  => %s%s%s %s[%ds]%s\n" "${color}" "${verdict}" "${C_RESET}" "${C_DIM}" "${dt}" "${C_RESET}"
    else
        printf "%s%s%s %s[%ds]%s\n" "${color}" "${verdict}" "${C_RESET}" "${C_DIM}" "${dt}" "${C_RESET}"
    fi

    if [[ ${status} -eq 0 ]]; then
        pass=$((pass + 1))
        if [[ ${KEEP_LOGS} -eq 0 ]]; then
            rm -f "${log}"
        fi
    else
        failures+=("${ex}|${verdict}|${log}")
        if [[ ${FAIL_FAST} -eq 1 ]]; then
            break
        fi
    fi
done

elapsed=$((SECONDS - started))
ran=${i}
fail=$((ran - pass))

# ---- Summary -------------------------------------------------------------

echo ""
printf "%s===== SMOKETEST SUMMARY =====%s\n" "${C_BOLD}" "${C_RESET}"
printf "Pass:    %s%d%s / %d\n" "${C_PASS}" "${pass}" "${C_RESET}" "${ran}"
printf "Fail:    %s%d%s / %d\n" "${C_FAIL}" "${fail}" "${C_RESET}" "${ran}"
if [[ ${ran} -lt ${total} ]]; then
    printf "Skipped: %s%d%s (fail-fast stopped early)\n" "${C_SKIP}" "$((total - ran))" "${C_RESET}"
fi
printf "Elapsed: %ds\n" "${elapsed}"

if [[ ${#failures[@]} -gt 0 ]]; then
    echo ""
    printf "%sFailures:%s\n" "${C_BOLD}" "${C_RESET}"
    for f in "${failures[@]}"; do
        IFS='|' read -r fex fverdict flog <<<"${f}"
        printf "  %s%s%s  %s\n  %s(log: %s)%s\n" "${C_FAIL}" "${fex}" "${C_RESET}" "${fverdict}" "${C_DIM}" "${flog}" "${C_RESET}"
    done
    exit 1
fi

exit 0
