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
#   blackwell-compile -- compile-only coverage pinned to exact sm_120a. These
#                   kernels are never launched.
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

TCGEN05_EXAMPLES=(gemm_sol gemm_sol_final tcgen05 tcgen05_matmul)
WGMMA_EXAMPLES=(wgmma)
LTOIR_EXAMPLES=(addressof_sharedarray cpp_consumes_rust_device device_ffi_test legacy_nvvm_pointer_shapes manual_launch_libdevice mathdx_ffi_test primitive_stress)
AUTO_NVVM_EXAMPLES=(libdevice_math)
BLACKWELL_COMPILE_EXAMPLES=(generated_intrinsics_blackwell)
NVVM_VERIFY_EXAMPLES=(cp_async_small device_global generated_intrinsics generated_intrinsics_blackwell generated_ldmatrix libdevice_math legacy_nvvm_pointer_shapes packed_atomic_add primitive_stress shuffle_64 tcgen05)
ERROR_EXAMPLES=(error error_wgmma_mma_unimplemented error_set_discriminant_niche error_set_discriminant_uninhabited error_static_initializer_provenance error_drop_glue error_heap_alloc error_missing_device_attr error_generated_intrinsic_abi error_generated_intrinsic_unknown_id error_generated_intrinsic_fn_pointer error_generated_intrinsic_callable)

classify() {
    local ex="$1" cat
    for cat in "${TCGEN05_EXAMPLES[@]}";     do [[ "$ex" == "$cat" ]] && { echo tcgen05;     return; }; done
    for cat in "${WGMMA_EXAMPLES[@]}";       do [[ "$ex" == "$cat" ]] && { echo wgmma;       return; }; done
    for cat in "${LTOIR_EXAMPLES[@]}";       do [[ "$ex" == "$cat" ]] && { echo ltoir;       return; }; done
    for cat in "${AUTO_NVVM_EXAMPLES[@]}";   do [[ "$ex" == "$cat" ]] && { echo auto-nvvm;   return; }; done
    for cat in "${BLACKWELL_COMPILE_EXAMPLES[@]}"; do [[ "$ex" == "$cat" ]] && { echo blackwell-compile; return; }; done
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

# Return a concrete libNVVM target that satisfies both the detected/default
# target and an example's generated-intrinsic floor. Compile-only mode never
# executes the artifact, so raising an older host target here is intentional:
# this lane proves that libNVVM accepts the selected lowering.
nvvm_verify_arch() {
    local ex="$1" arch="${LTOIR_ARCH}" floor=0 number
    case "${ex}" in
        cp_async_small) floor=80 ;;
        generated_intrinsics) floor=80 ;;
        generated_ldmatrix) floor=75 ;;
        packed_atomic_add) floor=90 ;;
        shuffle_64) floor=75 ;;
    esac
    if [[ ${floor} -ne 0 && "${arch}" =~ ^sm_([0-9]+)[af]?$ ]]; then
        number=$((10#${BASH_REMATCH[1]}))
        if [[ ${number} -lt ${floor} ]]; then
            arch="sm_${floor}"
        fi
    fi
    printf '%s\n' "${arch}"
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
    local log="$1" ec="$2" ex="$3"
    if [[ ${ec} -gt 128 ]]; then echo "FAIL (crashed, signal $((ec - 128)))"; return 1; fi

    # The generated-intrinsic fixtures protect fail-closed compiler contracts,
    # so merely observing an unrelated compile error is not enough.
    case "${ex}" in
        error_generated_intrinsic_abi)
            if ! grep -Fq 'cuda-intrinsics ABI mismatch' "${log}" \
                || ! grep -Fq '__cuda_oxide_intrinsic_abi_v2::i0001' "${log}"; then
                echo "FAIL (missing intrinsic ABI-v2 diagnostic)"
                return 1
            fi
            ;;
        error_generated_intrinsic_unknown_id)
            if ! grep -Fq 'cuda-intrinsics ABI mismatch' "${log}" \
                || ! grep -Fq '__cuda_oxide_intrinsic_abi_v1::i9999' "${log}"; then
                echo "FAIL (missing unknown intrinsic-ID diagnostic)"
                return 1
            fi
            ;;
        error_generated_intrinsic_fn_pointer)
            if ! grep -Fq 'must be called directly and cannot be converted to a function pointer' "${log}"; then
                echo "FAIL (missing function-pointer intrinsic diagnostic)"
                return 1
            fi
            ;;
        error_generated_intrinsic_callable)
            if ! grep -Fq 'generated CUDA intrinsics require direct call-site lowering' "${log}"; then
                echo "FAIL (missing direct-call-only intrinsic diagnostic)"
                return 1
            fi
            ;;
    esac

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
        local target
        target="$(sed -n '1p' "${target_file}" 2>/dev/null)"
        if [[ "${target}" =~ ^sm_[0-9]+[af]?$ ]]; then
            if grep -qx 'compile-options=v1' "${target_file}" && \
               [[ ! -s "${ex_dir}/${artifact}.options" ]]; then
                echo "FAIL (versioned NVVM IR target is missing its .options sidecar)"
                return 1
            fi
            echo "PASS (compiled NVVM IR for ${target})"
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
    # This exact-target batch must pass both compiler routes. The second build
    # may replace the first artifact, so preserve both exit codes in one gate.
    if [[ "${cat}" == "blackwell-compile" ]]; then
        local -a llvm_args=("build" "${ex}" "--arch=sm_120a")
        local -a nvvm_args=("emit-ltoir" "${ex}" "--arch=sm_120a")
        local llvm_ec
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${llvm_args[@]}" 2>&1 | tee "${log}"
            llvm_ec=${PIPESTATUS[0]}
        else
            cargo oxide "${llvm_args[@]}" >"${log}" 2>&1
            llvm_ec=$?
        fi
        if [[ ${llvm_ec} -ne 0 ]]; then
            CARGO_EC=${llvm_ec}
            return
        fi
        local llvm_ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local instruction_re='mma\.sp::ordered_metadata\.sync\.aligned\.m16n8k64\.row\.col\.kind::f8f6f4\.f32\.[[:alnum:]]+\.[[:alnum:]]+\.f32'
        local sparse_f16_instruction_re='mma\.sp::ordered_metadata\.sync\.aligned\.m16n8k64\.row\.col\.kind::f8f6f4\.f16\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.f16'
        local dense_f32_instruction_re='mma\.sync\.aligned\.m16n8k32\.row\.col\.kind::f8f6f4\.f32\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.f32'
        local dense_f16_instruction_re='mma\.sync\.aligned\.m16n8k32\.row\.col\.kind::f8f6f4\.f16\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.(e2m1|e2m3|e3m2|e4m3|e5m2)\.f16'
        local standard_fp8_f32_instruction_re='mma\.sync\.aligned\.m16n8k(16|32)\.row\.col\.f32\.(e4m3|e5m2)\.(e4m3|e5m2)\.f32'
        local standard_fp8_f16_instruction_re='mma\.sync\.aligned\.m16n8k(16|32)\.row\.col\.f16\.(e4m3|e5m2)\.(e4m3|e5m2)\.f16'
        local unresolved_dense_f32_re='llvm\.nvvm\.mma\.m16n8k32\.row\.col\.kind\.f8f6f4\.f32\.'
        local unresolved_dense_f16_re='llvm\.nvvm\.mma\.m16n8k32\.row\.col\.kind\.f8f6f4\.f16\.'
        local unresolved_sparse_f16_re='llvm\.nvvm\.mma\.sp\.ordered\.metadata\.m16n8k64\.row\.col\.kind\.f8f6f4\.f16\.'
        local unresolved_standard_fp8_re='llvm\.nvvm\.mma\.m16n8k(16|32)\.row\.col\.f(16|32)\.(e4m3|e5m2)\.'
        local fp8_re='cvt\.rn\.satfinite(\.relu)?\.(e4m3x2|e5m2x2)\.f32'
        local tf32_re='cvt\.(rna|rn|rz)(\.relu)?(\.satfinite)?\.tf32\.f32'
        local ldmatrix_re='ldmatrix\.sync\.aligned\.(m16n16\.x(1|2)\.trans\.shared\.(b8|b8x16\.(b4x16_p64|b6x16_p32))|m8n16\.x(1|2|4)\.shared\.b8x16\.(b4x16_p64|b6x16_p32))'
        local instruction_count unique_instruction_count
        local sparse_f16_instruction_count unique_sparse_f16_instruction_count
        local dense_f32_instruction_count unique_dense_f32_instruction_count
        local dense_f16_instruction_count unique_dense_f16_instruction_count
        local standard_fp8_f32_instruction_count unique_standard_fp8_f32_instruction_count
        local standard_fp8_f16_instruction_count unique_standard_fp8_f16_instruction_count
        local fp8_count unique_fp8_count
        local tf32_count unique_tf32_count
        local ldmatrix_count unique_ldmatrix_count
        instruction_count="$(grep -oE "${instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_instruction_count="$(grep -oE "${instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        sparse_f16_instruction_count="$(grep -oE "${sparse_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_sparse_f16_instruction_count="$(grep -oE "${sparse_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        dense_f32_instruction_count="$(grep -oE "${dense_f32_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_dense_f32_instruction_count="$(grep -oE "${dense_f32_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        dense_f16_instruction_count="$(grep -oE "${dense_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_dense_f16_instruction_count="$(grep -oE "${dense_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        standard_fp8_f32_instruction_count="$(grep -oE "${standard_fp8_f32_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_standard_fp8_f32_instruction_count="$(grep -oE "${standard_fp8_f32_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        standard_fp8_f16_instruction_count="$(grep -oE "${standard_fp8_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_standard_fp8_f16_instruction_count="$(grep -oE "${standard_fp8_f16_instruction_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        fp8_count="$(grep -oE "${fp8_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_fp8_count="$(grep -oE "${fp8_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        tf32_count="$(grep -oE "${tf32_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_tf32_count="$(grep -oE "${tf32_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        ldmatrix_count="$(grep -oE "${ldmatrix_re}" "${llvm_ptx}" 2>/dev/null | wc -l)"
        unique_ldmatrix_count="$(grep -oE "${ldmatrix_re}" "${llvm_ptx}" 2>/dev/null | sort -u | wc -l)"
        if [[ ! -s "${llvm_ptx}" ]] \
            || ! grep -qx '\.version 8\.7' "${llvm_ptx}" \
            || ! grep -qx '\.target sm_120a' "${llvm_ptx}" \
            || [[ ${instruction_count} -ne 25 || ${unique_instruction_count} -ne 25 ]] \
            || [[ ${sparse_f16_instruction_count} -ne 25 || ${unique_sparse_f16_instruction_count} -ne 25 ]] \
            || [[ ${dense_f32_instruction_count} -ne 25 || ${unique_dense_f32_instruction_count} -ne 25 ]] \
            || [[ ${dense_f16_instruction_count} -ne 25 || ${unique_dense_f16_instruction_count} -ne 25 ]] \
            || [[ ${standard_fp8_f32_instruction_count} -ne 8 || ${unique_standard_fp8_f32_instruction_count} -ne 8 ]] \
            || [[ ${standard_fp8_f16_instruction_count} -ne 8 || ${unique_standard_fp8_f16_instruction_count} -ne 8 ]] \
            || [[ ${fp8_count} -ne 4 || ${unique_fp8_count} -ne 4 ]] \
            || [[ ${tf32_count} -ne 10 || ${unique_tf32_count} -ne 10 ]] \
            || [[ ${ldmatrix_count} -ne 12 || ${unique_ldmatrix_count} -ne 12 ]] \
            || grep -qE "${unresolved_sparse_f16_re}|${unresolved_dense_f32_re}|${unresolved_dense_f16_re}|${unresolved_standard_fp8_re}" "${llvm_ptx}"; then
            printf 'direct LLVM route did not emit the expected sparse, dense, standard FP8, conversion, and ldmatrix instructions\n' >>"${log}"
            if [[ ${VERBOSE} -eq 1 ]]; then
                printf 'direct LLVM route did not emit the expected sparse, dense, standard FP8, conversion, and ldmatrix instructions\n'
            fi
            CARGO_EC=1
            return
        fi
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${nvvm_args[@]}" 2>&1 | tee -a "${log}"
            CARGO_EC=${PIPESTATUS[0]}
        else
            cargo oxide "${nvvm_args[@]}" >>"${log}" 2>&1
            CARGO_EC=$?
        fi
        if [[ ${CARGO_EC} -ne 0 ]]; then
            return
        fi
        local nvvm_ll="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ll"
        local inline_fp8_re='call i16 asm "cvt\.rn\.satfinite(\.relu)?\.(e4m3x2|e5m2x2)\.f32 \$0, \$2, \$1;", "=h,f,f"'
        local inline_tf32_re='call i32 asm "cvt\.(rna|rn|rz)(\.relu)?(\.satfinite)?\.tf32\.f32 \$0, \$1;", "=r,f"'
        local inline_fp8_count unique_inline_fp8_count
        local inline_tf32_count unique_inline_tf32_count
        local inline_ldmatrix_count unique_inline_ldmatrix_count
        local inline_sparse_f16_count unique_inline_sparse_f16_count
        local inline_dense_f32_count unique_inline_dense_f32_count
        local inline_dense_f16_count unique_inline_dense_f16_count
        local inline_standard_fp8_f32_count unique_inline_standard_fp8_f32_count
        local inline_standard_fp8_f16_count unique_inline_standard_fp8_f16_count
        inline_fp8_count="$(grep -oE "${inline_fp8_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_fp8_count="$(grep -oE "${inline_fp8_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_tf32_count="$(grep -oE "${inline_tf32_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_tf32_count="$(grep -oE "${inline_tf32_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_ldmatrix_count="$(grep -oE "${ldmatrix_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_ldmatrix_count="$(grep -oE "${ldmatrix_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_sparse_f16_count="$(grep -oE "${sparse_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_sparse_f16_count="$(grep -oE "${sparse_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_dense_f32_count="$(grep -oE "${dense_f32_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_dense_f32_count="$(grep -oE "${dense_f32_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_dense_f16_count="$(grep -oE "${dense_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_dense_f16_count="$(grep -oE "${dense_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_standard_fp8_f32_count="$(grep -oE "${standard_fp8_f32_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_standard_fp8_f32_count="$(grep -oE "${standard_fp8_f32_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        inline_standard_fp8_f16_count="$(grep -oE "${standard_fp8_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | wc -l)"
        unique_inline_standard_fp8_f16_count="$(grep -oE "${standard_fp8_f16_instruction_re}" "${nvvm_ll}" 2>/dev/null | sort -u | wc -l)"
        if [[ ! -s "${nvvm_ll}" ]] \
            || [[ ${inline_fp8_count} -ne 4 || ${unique_inline_fp8_count} -ne 4 ]] \
            || [[ ${inline_tf32_count} -ne 10 || ${unique_inline_tf32_count} -ne 10 ]] \
            || [[ ${inline_ldmatrix_count} -ne 12 || ${unique_inline_ldmatrix_count} -ne 12 ]] \
            || [[ ${inline_sparse_f16_count} -ne 25 || ${unique_inline_sparse_f16_count} -ne 25 ]] \
            || [[ ${inline_dense_f32_count} -ne 25 || ${unique_inline_dense_f32_count} -ne 25 ]] \
            || [[ ${inline_dense_f16_count} -ne 25 || ${unique_inline_dense_f16_count} -ne 25 ]] \
            || [[ ${inline_standard_fp8_f32_count} -ne 8 || ${unique_inline_standard_fp8_f32_count} -ne 8 ]] \
            || [[ ${inline_standard_fp8_f16_count} -ne 8 || ${unique_inline_standard_fp8_f16_count} -ne 8 ]] \
            || grep -qE 'llvm\.nvvm\.(ff\.to\.|f2tf32|ldmatrix\.)' "${nvvm_ll}" \
            || grep -qE "${unresolved_sparse_f16_re}|${unresolved_dense_f32_re}|${unresolved_dense_f16_re}|${unresolved_standard_fp8_re}" "${nvvm_ll}"; then
            printf 'libNVVM route did not emit the expected sparse, dense, standard FP8, conversion, and ldmatrix inline-PTX calls\n' >>"${log}"
            if [[ ${VERBOSE} -eq 1 ]]; then
                printf 'libNVVM route did not emit the expected sparse, dense, standard FP8, conversion, and ldmatrix inline-PTX calls\n'
            fi
            CARGO_EC=1
        fi
        return
    fi

    # The generated tcgen05 families must pass both compiler routes.
    if [[ ${COMPILE_ONLY} -eq 1 && "${ex}" == "tcgen05" ]]; then
        local cp_re='tcgen05\.cp\.cta_group::[12]\.(128x128b|128x256b|32x128b\.warpx4|4x256b|64x128b\.warpx2::(01_23|02_13))(\.b8x16\.(b4x16_p64|b6x16_p32))?[[:space:]]'
        local mma_base_re='tcgen05\.mma(\.sp)?\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::a::(discard|lastuse|fill|use)(\.ashift)?'
        local mma_base_plain_re='tcgen05\.mma\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::a::(discard|lastuse|fill|use)(\.ashift)?'
        local mma_base_sp_re='tcgen05\.mma\.sp\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::a::(discard|lastuse|fill|use)(\.ashift)?'
        local mma_ws_re='tcgen05\.mma\.ws(\.sp)?\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::b[0-3]::(discard|lastuse|fill|use)'
        local mma_ws_plain_re='tcgen05\.mma\.ws\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::b[0-3]::(discard|lastuse|fill|use)'
        local mma_ws_sp_re='tcgen05\.mma\.ws\.sp\.cta_group::[12]\.kind::(f16|tf32|f8f6f4|i8)\.collector::b[0-3]::(discard|lastuse|fill|use)'
        local ptx_shared_a_re='\[[^]]+\],[[:space:]]+%rd[0-9]+'
        local ptx_tensor_a_re='\[[^]]+\],[[:space:]]+\[%r[0-9]+\]'
        local ptx_zero_mask_re='%enable_pred,[[:space:]]+%rd[0-9]+;'
        local nvvm_shared_a_re='\[\$0\],[[:space:]]+\$1'
        local nvvm_tensor_a_re='\[\$0\],[[:space:]]+\[\$1\]'
        local nvvm_zero_mask_re='%enable_pred,[[:space:]]+\$[56];'
        local unresolved_mma_re='llvm\.nvvm\.tcgen05\.mma'
        local ld_re='tcgen05\.ld\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)(\.pack::16b)?\.b32'
        local ld_pack16_re='tcgen05\.ld\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)\.pack::16b\.b32'
        local ld_raw_re='tcgen05\.ld\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)\.b32'
        local st_re='tcgen05\.st\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)(\.unpack::16b)?\.b32'
        local st_unpack16_re='tcgen05\.st\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)\.unpack::16b\.b32'
        local st_raw_re='tcgen05\.st\.sync\.aligned\.(16x64b|16x128b|16x256b|32x32b)\.x(1|2|4|8|16|32|64|128)\.b32'
        local ld_offset_re='tcgen05\.ld\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)(\.pack::16b)?\.b32'
        local ld_offset_pack16_re='tcgen05\.ld\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)\.pack::16b\.b32'
        local ld_offset_raw_re='tcgen05\.ld\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)\.b32'
        local st_offset_re='tcgen05\.st\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)(\.unpack::16b)?\.b32'
        local st_offset_unpack16_re='tcgen05\.st\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)\.unpack::16b\.b32'
        local st_offset_raw_re='tcgen05\.st\.sync\.aligned\.16x32bx2\.x(1|2|4|8|16|32|64|128)\.b32'
        local commit_multicast_cg1_re='tcgen05\.commit\.cta_group::1\.mbarrier::arrive::one\.shared::cluster\.multicast::cluster\.b64'
        local shift_down_cg1_re='tcgen05\.shift\.cta_group::1\.down'
        local shift_down_cg2_re='tcgen05\.shift\.cta_group::2\.down'
        local unresolved_control_re='llvm\.nvvm\.tcgen05\.(commit\.mc\.shared\.cg1|shift\.down\.cg[12])'
        local llvm_ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local -a llvm_args=("build" "${ex}" "--arch=sm_100a")
        local llvm_ec
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${llvm_args[@]}" 2>&1 | tee "${log}"
            llvm_ec=${PIPESTATUS[0]}
        else
            cargo oxide "${llvm_args[@]}" >"${log}" 2>&1
            llvm_ec=$?
        fi
        if [[ ${llvm_ec} -ne 0 ]]; then
            CARGO_EC=${llvm_ec}
            return
        fi

        local llvm_cg1 llvm_cg2 llvm_cp_count llvm_cp_unique
        llvm_cg1="$(awk '/^\.visible \.entry compile_tcgen05_cp_cg1\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_cg2="$(awk '/^\.visible \.entry compile_tcgen05_cp_cg2\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_cp_count="$(grep -oE "${cp_re}" <<<"${llvm_cg1}"$'\n'"${llvm_cg2}" | wc -l)"
        llvm_cp_unique="$(grep -oE "${cp_re}" <<<"${llvm_cg1}"$'\n'"${llvm_cg2}" | sort -u | wc -l)"
        if [[ -z "${llvm_cg1}" || -z "${llvm_cg2}" ]] \
            || ! grep -qx '\.target sm_100a' "${llvm_ptx}" \
            || [[ $(grep -oE "${cp_re}" <<<"${llvm_cg1}" | wc -l) -ne 18 ]] \
            || [[ $(grep -oE "${cp_re}" <<<"${llvm_cg2}" | wc -l) -ne 18 ]] \
            || grep -q 'tcgen05\.cp\.cta_group::2\.' <<<"${llvm_cg1}" \
            || grep -q 'tcgen05\.cp\.cta_group::1\.' <<<"${llvm_cg2}" \
            || [[ ${llvm_cp_count} -ne 36 || ${llvm_cp_unique} -ne 36 ]]; then
            printf 'tcgen05 LLVM route expected 18 copy forms per CTA group and 36 unique total; got %s/%s total/unique\n' \
                "${llvm_cp_count}" "${llvm_cp_unique}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_control_cg1 llvm_control_cg2
        llvm_control_cg1="$(awk '/^\.visible \.entry compile_tcgen05_control_cg1\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_control_cg2="$(awk '/^\.visible \.entry compile_tcgen05_control_cg2\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        if [[ -z "${llvm_control_cg1}" || -z "${llvm_control_cg2}" ]] \
            || [[ $(grep -oE "${commit_multicast_cg1_re}" <<<"${llvm_control_cg1}" | wc -l) -ne 1 ]] \
            || [[ $(grep -oE "${shift_down_cg1_re}" <<<"${llvm_control_cg1}" | wc -l) -ne 1 ]] \
            || [[ $(grep -oE "${shift_down_cg2_re}" <<<"${llvm_control_cg2}" | wc -l) -ne 1 ]] \
            || grep -qE 'tcgen05\.(commit|shift)\.cta_group::2' <<<"${llvm_control_cg1}" \
            || grep -qE 'tcgen05\.(commit|shift)\.cta_group::1' <<<"${llvm_control_cg2}" \
            || grep -qE "${unresolved_control_re}" "${llvm_ptx}"; then
            printf 'tcgen05 LLVM route did not emit the exact cg1 multicast commit and cg1/cg2 shift-down forms\n' >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_mma_base llvm_mma_ws llvm_mma_base_count llvm_mma_base_unique
        local llvm_mma_ws_count llvm_mma_ws_unique
        llvm_mma_base="$(awk '/^\.visible \.entry compile_tcgen05_mma_base\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_mma_ws="$(awk '/^\.visible \.entry compile_tcgen05_mma_ws\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_mma_base_count="$(grep -oE "${mma_base_re}" <<<"${llvm_mma_base}" | wc -l)"
        llvm_mma_base_unique="$(grep -oE "${mma_base_re}" <<<"${llvm_mma_base}" | sort -u | wc -l)"
        llvm_mma_ws_count="$(grep -oE "${mma_ws_re}" <<<"${llvm_mma_ws}" | wc -l)"
        llvm_mma_ws_unique="$(grep -oE "${mma_ws_re}" <<<"${llvm_mma_ws}" | sort -u | wc -l)"
        if [[ -z "${llvm_mma_base}" || -z "${llvm_mma_ws}" ]] \
            || [[ ${llvm_mma_base_count} -ne 9 || ${llvm_mma_base_unique} -ne 8 ]] \
            || [[ ${llvm_mma_ws_count} -ne 16 || ${llvm_mma_ws_unique} -ne 10 ]] \
            || [[ $(grep -oE "${mma_base_plain_re}" <<<"${llvm_mma_base}" | wc -l) -ne 6 ]] \
            || [[ $(grep -oE "${mma_base_sp_re}" <<<"${llvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE "${mma_ws_plain_re}" <<<"${llvm_mma_ws}" | wc -l) -ne 12 ]] \
            || [[ $(grep -oE "${mma_ws_sp_re}" <<<"${llvm_mma_ws}" | wc -l) -ne 4 ]] \
            || [[ $(grep -cE "${mma_base_plain_re}[[:space:]]+${ptx_shared_a_re}" <<<"${llvm_mma_base}") -ne 4 ]] \
            || [[ $(grep -cE "${mma_base_plain_re}[[:space:]]+${ptx_tensor_a_re}" <<<"${llvm_mma_base}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_base_sp_re}[[:space:]]+${ptx_shared_a_re}" <<<"${llvm_mma_base}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_base_sp_re}[[:space:]]+${ptx_tensor_a_re}" <<<"${llvm_mma_base}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${ptx_shared_a_re}" <<<"${llvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${ptx_tensor_a_re}" <<<"${llvm_mma_ws}") -ne 10 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${ptx_shared_a_re}" <<<"${llvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${ptx_tensor_a_re}" <<<"${llvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -oE '\.cta_group::1\.' <<<"${llvm_mma_base}" | wc -l) -ne 5 ]] \
            || [[ $(grep -oE '\.cta_group::2\.' <<<"${llvm_mma_base}" | wc -l) -ne 4 ]] \
            || [[ $(grep -oE '\.cta_group::1\.' <<<"${llvm_mma_ws}" | wc -l) -ne 16 ]] \
            || grep -qE '\.cta_group::2\.' <<<"${llvm_mma_ws}" \
            || [[ $(grep -oE '\.kind::f16\.' <<<"${llvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::tf32\.' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::f8f6f4\.' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::i8\.' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::f16\.' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::tf32\.' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::f8f6f4\.' <<<"${llvm_mma_ws}" | wc -l) -ne 8 ]] \
            || [[ $(grep -oE '\.kind::i8\.' <<<"${llvm_mma_ws}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::discard' <<<"${llvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::a::lastuse' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::fill' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::use' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::b0::discard' <<<"${llvm_mma_ws}" | wc -l) -ne 7 ]] \
            || [[ $(grep -oE '\.collector::b1::(lastuse|fill)' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::b2::(fill|use)' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::b3::(use|discard)' <<<"${llvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.ashift' <<<"${llvm_mma_base}" | wc -l) -ne 2 ]] \
            || grep -qE '\.ashift' <<<"${llvm_mma_ws}" \
            || [[ $(grep -cE "${mma_ws_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 4 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${ptx_shared_a_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${ptx_tensor_a_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${ptx_shared_a_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${ptx_tensor_a_re}.*${ptx_zero_mask_re}" <<<"${llvm_mma_ws}") -ne 1 ]] \
            || grep -qE "${unresolved_mma_re}" "${llvm_ptx}"; then
            printf 'tcgen05 LLVM route expected 9 base and 16 warp-specialized MMA calls across all forms; got %s/%s and %s/%s total/unique\n' \
                "${llvm_mma_base_count}" "${llvm_mma_base_unique}" "${llvm_mma_ws_count}" "${llvm_mma_ws_unique}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_ld llvm_ld_count llvm_ld_unique llvm_ld_pack16 llvm_ld_raw
        llvm_ld="$(awk '/^\.visible \.entry compile_tcgen05_ld\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_ld_count="$(grep -oE "${ld_re}" <<<"${llvm_ld}" | wc -l)"
        llvm_ld_unique="$(grep -oE "${ld_re}" <<<"${llvm_ld}" | sort -u | wc -l)"
        llvm_ld_pack16="$(grep -oE "${ld_pack16_re}" <<<"${llvm_ld}" | wc -l)"
        llvm_ld_raw="$(grep -oE "${ld_raw_re}" <<<"${llvm_ld}" | wc -l)"
        if [[ -z "${llvm_ld}" ]] \
            || [[ ${llvm_ld_count} -ne 58 || ${llvm_ld_unique} -ne 58 ]] \
            || [[ ${llvm_ld_pack16} -ne 29 || ${llvm_ld_raw} -ne 29 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.ld' "${llvm_ptx}"; then
            printf 'tcgen05 LLVM route expected 58 unique loads (29 raw and 29 pack16); got %s/%s total/unique and %s/%s raw/pack16\n' \
                "${llvm_ld_count}" "${llvm_ld_unique}" "${llvm_ld_raw}" "${llvm_ld_pack16}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_st llvm_st_count llvm_st_unique llvm_st_unpack16 llvm_st_raw
        llvm_st="$(awk '/^\.visible \.entry compile_tcgen05_st\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_st_count="$(grep -oE "${st_re}" <<<"${llvm_st}" | wc -l)"
        llvm_st_unique="$(grep -oE "${st_re}" <<<"${llvm_st}" | sort -u | wc -l)"
        llvm_st_unpack16="$(grep -oE "${st_unpack16_re}" <<<"${llvm_st}" | wc -l)"
        llvm_st_raw="$(grep -oE "${st_raw_re}" <<<"${llvm_st}" | wc -l)"
        if [[ -z "${llvm_st}" ]] \
            || [[ ${llvm_st_count} -ne 58 || ${llvm_st_unique} -ne 58 ]] \
            || [[ ${llvm_st_unpack16} -ne 29 || ${llvm_st_raw} -ne 29 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.st' "${llvm_ptx}"; then
            printf 'tcgen05 LLVM route expected 58 unique stores (29 raw and 29 unpack16); got %s/%s total/unique and %s/%s raw/unpack16\n' \
                "${llvm_st_count}" "${llvm_st_unique}" "${llvm_st_raw}" "${llvm_st_unpack16}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_ld_offset llvm_ld_offset_count llvm_ld_offset_unique llvm_ld_offset_pack16 llvm_ld_offset_raw
        llvm_ld_offset="$(awk '/^\.visible \.entry compile_tcgen05_ld_offset\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_ld_offset_count="$(grep -oE "${ld_offset_re}" <<<"${llvm_ld_offset}" | wc -l)"
        llvm_ld_offset_unique="$(grep -oE "${ld_offset_re}" <<<"${llvm_ld_offset}" | sort -u | wc -l)"
        llvm_ld_offset_pack16="$(grep -oE "${ld_offset_pack16_re}" <<<"${llvm_ld_offset}" | wc -l)"
        llvm_ld_offset_raw="$(grep -oE "${ld_offset_raw_re}" <<<"${llvm_ld_offset}" | wc -l)"
        if [[ -z "${llvm_ld_offset}" ]] \
            || [[ ${llvm_ld_offset_count} -ne 16 || ${llvm_ld_offset_unique} -ne 16 ]] \
            || [[ ${llvm_ld_offset_pack16} -ne 8 || ${llvm_ld_offset_raw} -ne 8 ]] \
            || [[ $(grep -c ', 16;' <<<"${llvm_ld_offset}") -ne 16 ]]; then
            printf 'tcgen05 LLVM route expected 16 unique offset loads with immediate 16; got %s/%s total/unique and %s/%s raw/pack16\n' \
                "${llvm_ld_offset_count}" "${llvm_ld_offset_unique}" "${llvm_ld_offset_raw}" "${llvm_ld_offset_pack16}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local llvm_st_offset llvm_st_offset_count llvm_st_offset_unique llvm_st_offset_unpack16 llvm_st_offset_raw
        llvm_st_offset="$(awk '/^\.visible \.entry compile_tcgen05_st_offset\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        llvm_st_offset_count="$(grep -oE "${st_offset_re}" <<<"${llvm_st_offset}" | wc -l)"
        llvm_st_offset_unique="$(grep -oE "${st_offset_re}" <<<"${llvm_st_offset}" | sort -u | wc -l)"
        llvm_st_offset_unpack16="$(grep -oE "${st_offset_unpack16_re}" <<<"${llvm_st_offset}" | wc -l)"
        llvm_st_offset_raw="$(grep -oE "${st_offset_raw_re}" <<<"${llvm_st_offset}" | wc -l)"
        if [[ -z "${llvm_st_offset}" ]] \
            || [[ ${llvm_st_offset_count} -ne 16 || ${llvm_st_offset_unique} -ne 16 ]] \
            || [[ ${llvm_st_offset_unpack16} -ne 8 || ${llvm_st_offset_raw} -ne 8 ]] \
            || [[ $(grep -c ', 16,' <<<"${llvm_st_offset}") -ne 16 ]]; then
            printf 'tcgen05 LLVM route expected 16 unique offset stores with immediate 16; got %s/%s total/unique and %s/%s raw/unpack16\n' \
                "${llvm_st_offset_count}" "${llvm_st_offset_unique}" "${llvm_st_offset_raw}" "${llvm_st_offset_unpack16}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local -a nvvm_args=("emit-ltoir" "${ex}" "--arch=sm_100a")
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${nvvm_args[@]}" 2>&1 | tee -a "${log}"
            CARGO_EC=${PIPESTATUS[0]}
        else
            cargo oxide "${nvvm_args[@]}" >>"${log}" 2>&1
            CARGO_EC=$?
        fi
        if [[ ${CARGO_EC} -ne 0 ]]; then
            return
        fi

        local nvvm_ll="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ll"
        local nvvm_target="crates/rustc-codegen-cuda/examples/${ex}/${ex}.target"
        local nvvm_cg1 nvvm_cg2 nvvm_cp_count nvvm_cp_unique
        nvvm_cg1="$(awk '/^define .*@compile_tcgen05_cp_cg1\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_cg2="$(awk '/^define .*@compile_tcgen05_cp_cg2\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_cp_count="$(grep -oE "${cp_re}" <<<"${nvvm_cg1}"$'\n'"${nvvm_cg2}" | wc -l)"
        nvvm_cp_unique="$(grep -oE "${cp_re}" <<<"${nvvm_cg1}"$'\n'"${nvvm_cg2}" | sort -u | wc -l)"
        if [[ -z "${nvvm_cg1}" || -z "${nvvm_cg2}" ]] \
            || [[ $(grep -oE "${cp_re}" <<<"${nvvm_cg1}" | wc -l) -ne 18 ]] \
            || [[ $(grep -oE "${cp_re}" <<<"${nvvm_cg2}" | wc -l) -ne 18 ]] \
            || grep -q 'tcgen05\.cp\.cta_group::2\.' <<<"${nvvm_cg1}" \
            || grep -q 'tcgen05\.cp\.cta_group::1\.' <<<"${nvvm_cg2}" \
            || [[ ${nvvm_cp_count} -ne 36 || ${nvvm_cp_unique} -ne 36 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.cp' "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route expected 18 inline copy forms per CTA group and 36 unique total; got %s/%s total/unique\n' \
                "${nvvm_cp_count}" "${nvvm_cp_unique}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_control_cg1 nvvm_control_cg2
        nvvm_control_cg1="$(awk '/^define .*@compile_tcgen05_control_cg1\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_control_cg2="$(awk '/^define .*@compile_tcgen05_control_cg2\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        local -a nvvm_control_attrs=()
        mapfile -t nvvm_control_attrs < <(
            sed -nE '/call void asm sideeffect "tcgen05\.(commit|shift)\.cta_group::[12]/s/.* (#[0-9]+)$/\1/p' \
                <<<"${nvvm_control_cg1}"$'\n'"${nvvm_control_cg2}"
        )
        local nvvm_control_convergent=1 control_attr control_attr_definition
        if [[ ${#nvvm_control_attrs[@]} -ne 4 ]]; then
            nvvm_control_convergent=0
        else
            for control_attr in "${nvvm_control_attrs[@]}"; do
                control_attr_definition="$(grep -E "^attributes ${control_attr} = \\{[^}]*\\}$" "${nvvm_ll}")"
                if [[ $(wc -l <<<"${control_attr_definition}") -ne 1 ]] \
                    || ! grep -qw convergent <<<"${control_attr_definition}"; then
                    nvvm_control_convergent=0
                    break
                fi
            done
        fi
        if [[ -z "${nvvm_control_cg1}" || -z "${nvvm_control_cg2}" ]] \
            || [[ $(grep -oE "${commit_multicast_cg1_re}" <<<"${nvvm_control_cg1}" | wc -l) -ne 1 ]] \
            || [[ $(grep -oE "${shift_down_cg1_re}" <<<"${nvvm_control_cg1}" | wc -l) -ne 1 ]] \
            || [[ $(grep -oE "${shift_down_cg2_re}" <<<"${nvvm_control_cg2}" | wc -l) -ne 1 ]] \
            || ! grep -q 'asm sideeffect' <<<"${nvvm_control_cg1}" \
            || ! grep -q 'asm sideeffect' <<<"${nvvm_control_cg2}" \
            || [[ ${nvvm_control_convergent} -ne 1 ]] \
            || grep -qE 'tcgen05\.(commit|shift)\.cta_group::2' <<<"${nvvm_control_cg1}" \
            || grep -qE 'tcgen05\.(commit|shift)\.cta_group::1' <<<"${nvvm_control_cg2}" \
            || grep -qE "${unresolved_control_re}" "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route did not emit the exact cg1 multicast commit and cg1/cg2 shift-down inline assembly\n' >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_mma_base nvvm_mma_ws nvvm_mma_base_count nvvm_mma_base_unique
        local nvvm_mma_ws_count nvvm_mma_ws_unique nvvm_mma_inline_count nvvm_mma_memory_count
        nvvm_mma_base="$(awk '/^define .*@compile_tcgen05_mma_base\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_mma_ws="$(awk '/^define .*@compile_tcgen05_mma_ws\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_mma_base_count="$(grep -oE "${mma_base_re}" <<<"${nvvm_mma_base}" | wc -l)"
        nvvm_mma_base_unique="$(grep -oE "${mma_base_re}" <<<"${nvvm_mma_base}" | sort -u | wc -l)"
        nvvm_mma_ws_count="$(grep -oE "${mma_ws_re}" <<<"${nvvm_mma_ws}" | wc -l)"
        nvvm_mma_ws_unique="$(grep -oE "${mma_ws_re}" <<<"${nvvm_mma_ws}" | sort -u | wc -l)"
        nvvm_mma_inline_count="$(grep -cE 'call void asm sideeffect ".*tcgen05\.mma' <<<"${nvvm_mma_base}"$'\n'"${nvvm_mma_ws}")"
        nvvm_mma_memory_count="$(grep -E 'call void asm sideeffect ".*tcgen05\.mma' <<<"${nvvm_mma_base}"$'\n'"${nvvm_mma_ws}" | grep -cF '~{memory}')"
        local -a nvvm_mma_attrs=()
        mapfile -t nvvm_mma_attrs < <(
            sed -nE '/call void asm sideeffect ".*tcgen05\.mma/s/.* (#[0-9]+)$/\1/p' \
                <<<"${nvvm_mma_base}"$'\n'"${nvvm_mma_ws}"
        )
        local nvvm_mma_convergent=1 mma_attr mma_attr_definition
        if [[ ${#nvvm_mma_attrs[@]} -ne 25 ]]; then
            nvvm_mma_convergent=0
        else
            for mma_attr in "${nvvm_mma_attrs[@]}"; do
                mma_attr_definition="$(grep -E "^attributes ${mma_attr} = \\{[^}]*\\}$" "${nvvm_ll}")"
                if [[ $(wc -l <<<"${mma_attr_definition}") -ne 1 ]] \
                    || ! grep -qw convergent <<<"${mma_attr_definition}"; then
                    nvvm_mma_convergent=0
                    break
                fi
            done
        fi
        if [[ -z "${nvvm_mma_base}" || -z "${nvvm_mma_ws}" ]] \
            || [[ ${nvvm_mma_base_count} -ne 9 || ${nvvm_mma_base_unique} -ne 8 ]] \
            || [[ ${nvvm_mma_ws_count} -ne 16 || ${nvvm_mma_ws_unique} -ne 10 ]] \
            || [[ ${nvvm_mma_inline_count} -ne 25 || ${nvvm_mma_memory_count} -ne 25 ]] \
            || [[ ${nvvm_mma_convergent} -ne 1 ]] \
            || [[ $(grep -oE "${mma_base_plain_re}" <<<"${nvvm_mma_base}" | wc -l) -ne 6 ]] \
            || [[ $(grep -oE "${mma_base_sp_re}" <<<"${nvvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE "${mma_ws_plain_re}" <<<"${nvvm_mma_ws}" | wc -l) -ne 12 ]] \
            || [[ $(grep -oE "${mma_ws_sp_re}" <<<"${nvvm_mma_ws}" | wc -l) -ne 4 ]] \
            || [[ $(grep -cE "${mma_base_plain_re}[[:space:]]+${nvvm_shared_a_re}" <<<"${nvvm_mma_base}") -ne 4 ]] \
            || [[ $(grep -cE "${mma_base_plain_re}[[:space:]]+${nvvm_tensor_a_re}" <<<"${nvvm_mma_base}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_base_sp_re}[[:space:]]+${nvvm_shared_a_re}" <<<"${nvvm_mma_base}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_base_sp_re}[[:space:]]+${nvvm_tensor_a_re}" <<<"${nvvm_mma_base}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${nvvm_shared_a_re}" <<<"${nvvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${nvvm_tensor_a_re}" <<<"${nvvm_mma_ws}") -ne 10 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${nvvm_shared_a_re}" <<<"${nvvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${nvvm_tensor_a_re}" <<<"${nvvm_mma_ws}") -ne 2 ]] \
            || [[ $(grep -oE '\.cta_group::1\.' <<<"${nvvm_mma_base}" | wc -l) -ne 5 ]] \
            || [[ $(grep -oE '\.cta_group::2\.' <<<"${nvvm_mma_base}" | wc -l) -ne 4 ]] \
            || [[ $(grep -oE '\.cta_group::1\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 16 ]] \
            || grep -qE '\.cta_group::2\.' <<<"${nvvm_mma_ws}" \
            || [[ $(grep -oE '\.kind::f16\.' <<<"${nvvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::tf32\.' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::f8f6f4\.' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::i8\.' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.kind::f16\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::tf32\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.kind::f8f6f4\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 8 ]] \
            || [[ $(grep -oE '\.kind::i8\.' <<<"${nvvm_mma_ws}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::discard' <<<"${nvvm_mma_base}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::a::lastuse' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::fill' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::a::use' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || [[ $(grep -oE '\.collector::b0::discard' <<<"${nvvm_mma_ws}" | wc -l) -ne 7 ]] \
            || [[ $(grep -oE '\.collector::b1::(lastuse|fill)' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::b2::(fill|use)' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.collector::b3::(use|discard)' <<<"${nvvm_mma_ws}" | wc -l) -ne 3 ]] \
            || [[ $(grep -oE '\.ashift' <<<"${nvvm_mma_base}" | wc -l) -ne 2 ]] \
            || grep -qE '\.ashift' <<<"${nvvm_mma_ws}" \
            || [[ $(grep -cE "${mma_ws_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 4 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${nvvm_shared_a_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_plain_re}[[:space:]]+${nvvm_tensor_a_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${nvvm_shared_a_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 1 ]] \
            || [[ $(grep -cE "${mma_ws_sp_re}[[:space:]]+${nvvm_tensor_a_re}.*${nvvm_zero_mask_re}" <<<"${nvvm_mma_ws}") -ne 1 ]] \
            || grep -qE "${unresolved_mma_re}" "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route expected 9 base and 16 convergent, side-effecting MMA calls across all forms; got %s/%s and %s/%s total/unique\n' \
                "${nvvm_mma_base_count}" "${nvvm_mma_base_unique}" "${nvvm_mma_ws_count}" "${nvvm_mma_ws_unique}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_ld nvvm_ld_count nvvm_ld_unique nvvm_ld_pack16 nvvm_ld_raw
        nvvm_ld="$(awk '/^define .*@compile_tcgen05_ld\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_ld_count="$(grep -oE "${ld_re}" <<<"${nvvm_ld}" | wc -l)"
        nvvm_ld_unique="$(grep -oE "${ld_re}" <<<"${nvvm_ld}" | sort -u | wc -l)"
        nvvm_ld_pack16="$(grep -oE "${ld_pack16_re}" <<<"${nvvm_ld}" | wc -l)"
        nvvm_ld_raw="$(grep -oE "${ld_raw_re}" <<<"${nvvm_ld}" | wc -l)"
        if [[ -z "${nvvm_ld}" ]] \
            || [[ "$(head -n 1 "${nvvm_target}" 2>/dev/null)" != "sm_100a" ]] \
            || [[ ${nvvm_ld_count} -ne 58 || ${nvvm_ld_unique} -ne 58 ]] \
            || [[ ${nvvm_ld_pack16} -ne 29 || ${nvvm_ld_raw} -ne 29 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.ld' "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route expected sm_100a and 58 unique inline loads (29 raw and 29 pack16); got %s/%s total/unique and %s/%s raw/pack16\n' \
                "${nvvm_ld_count}" "${nvvm_ld_unique}" "${nvvm_ld_raw}" "${nvvm_ld_pack16}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_st nvvm_st_count nvvm_st_unique nvvm_st_unpack16 nvvm_st_raw
        nvvm_st="$(awk '/^define .*@compile_tcgen05_st\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_st_count="$(grep -oE "${st_re}" <<<"${nvvm_st}" | wc -l)"
        nvvm_st_unique="$(grep -oE "${st_re}" <<<"${nvvm_st}" | sort -u | wc -l)"
        nvvm_st_unpack16="$(grep -oE "${st_unpack16_re}" <<<"${nvvm_st}" | wc -l)"
        nvvm_st_raw="$(grep -oE "${st_raw_re}" <<<"${nvvm_st}" | wc -l)"
        if [[ -z "${nvvm_st}" ]] \
            || [[ ${nvvm_st_count} -ne 58 || ${nvvm_st_unique} -ne 58 ]] \
            || [[ ${nvvm_st_unpack16} -ne 29 || ${nvvm_st_raw} -ne 29 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.st' "${nvvm_ll}"; then
            printf 'tcgen05 libNVVM route expected 58 unique inline stores (29 raw and 29 unpack16); got %s/%s total/unique and %s/%s raw/unpack16\n' \
                "${nvvm_st_count}" "${nvvm_st_unique}" "${nvvm_st_raw}" "${nvvm_st_unpack16}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_ld_offset nvvm_ld_offset_count nvvm_ld_offset_unique nvvm_ld_offset_pack16 nvvm_ld_offset_raw
        nvvm_ld_offset="$(awk '/^define .*@compile_tcgen05_ld_offset\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_ld_offset_count="$(grep -oE "${ld_offset_re}" <<<"${nvvm_ld_offset}" | wc -l)"
        nvvm_ld_offset_unique="$(grep -oE "${ld_offset_re}" <<<"${nvvm_ld_offset}" | sort -u | wc -l)"
        nvvm_ld_offset_pack16="$(grep -oE "${ld_offset_pack16_re}" <<<"${nvvm_ld_offset}" | wc -l)"
        nvvm_ld_offset_raw="$(grep -oE "${ld_offset_raw_re}" <<<"${nvvm_ld_offset}" | wc -l)"
        if [[ -z "${nvvm_ld_offset}" ]] \
            || [[ ${nvvm_ld_offset_count} -ne 16 || ${nvvm_ld_offset_unique} -ne 16 ]] \
            || [[ ${nvvm_ld_offset_pack16} -ne 8 || ${nvvm_ld_offset_raw} -ne 8 ]] \
            || [[ $(grep 'asm sideeffect' <<<"${nvvm_ld_offset}" | grep -o 'i64 16' | wc -l) -ne 16 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.ld' <<<"${nvvm_ld_offset}"; then
            printf 'tcgen05 libNVVM route expected 16 unique inline offset loads with immediate 16; got %s/%s total/unique and %s/%s raw/pack16\n' \
                "${nvvm_ld_offset_count}" "${nvvm_ld_offset_unique}" "${nvvm_ld_offset_raw}" "${nvvm_ld_offset_pack16}" >>"${log}"
            CARGO_EC=1
        fi

        local nvvm_st_offset nvvm_st_offset_count nvvm_st_offset_unique nvvm_st_offset_unpack16 nvvm_st_offset_raw
        nvvm_st_offset="$(awk '/^define .*@compile_tcgen05_st_offset\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        nvvm_st_offset_count="$(grep -oE "${st_offset_re}" <<<"${nvvm_st_offset}" | wc -l)"
        nvvm_st_offset_unique="$(grep -oE "${st_offset_re}" <<<"${nvvm_st_offset}" | sort -u | wc -l)"
        nvvm_st_offset_unpack16="$(grep -oE "${st_offset_unpack16_re}" <<<"${nvvm_st_offset}" | wc -l)"
        nvvm_st_offset_raw="$(grep -oE "${st_offset_raw_re}" <<<"${nvvm_st_offset}" | wc -l)"
        if [[ -z "${nvvm_st_offset}" ]] \
            || [[ ${nvvm_st_offset_count} -ne 16 || ${nvvm_st_offset_unique} -ne 16 ]] \
            || [[ ${nvvm_st_offset_unpack16} -ne 8 || ${nvvm_st_offset_raw} -ne 8 ]] \
            || [[ $(grep 'asm sideeffect' <<<"${nvvm_st_offset}" | grep -o 'i64 16' | wc -l) -ne 16 ]] \
            || grep -q 'llvm\.nvvm\.tcgen05\.st' <<<"${nvvm_st_offset}"; then
            printf 'tcgen05 libNVVM route expected 16 unique inline offset stores with immediate 16; got %s/%s total/unique and %s/%s raw/unpack16\n' \
                "${nvvm_st_offset_count}" "${nvvm_st_offset_unique}" "${nvvm_st_offset_raw}" "${nvvm_st_offset_unpack16}" >>"${log}"
            CARGO_EC=1
        fi
        return
    fi

    # This example gates scalar arithmetic, Ampere MMA, and extended min/max
    # on both backends.
    if [[ ${COMPILE_ONLY} -eq 1 && "${ex}" == "generated_intrinsics" ]]; then
        local llvm_ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local -a llvm_args=("build" "${ex}" "--arch=sm_86")
        local llvm_ec
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${llvm_args[@]}" 2>&1 | tee "${log}"
            llvm_ec=${PIPESTATUS[0]}
        else
            cargo oxide "${llvm_args[@]}" >"${log}" 2>&1
            llvm_ec=$?
        fi
        if [[ ${llvm_ec} -ne 0 ]]; then
            CARGO_EC=${llvm_ec}
            return
        fi

        local scalar_ptx_re='((mul|div)\.(rn|rz|rm|rp)(\.ftz)?\.f32|fma\.(rn|rz|rm|rp)(\.ftz)?(\.sat)?\.f32|add\.(rn|rz|rm|rp)(\.sat)?(\.ftz)?\.f32|(mul|div|fma|add)\.(rn|rz|rm|rp)\.f64)'
        local ampere_float_mma_re='mma\.sync\.aligned\.(m16n8k4\.row\.col\.f32\.tf32\.tf32\.f32|m16n8k8\.row\.col\.f16\.f16\.f16\.f16|m16n8k8\.row\.col\.f32\.bf16\.bf16\.f32|m16n8k8\.row\.col\.f32\.f16\.f16\.f32|m16n8k16\.row\.col\.f16\.f16\.f16\.f16)'
        local extended_minmax_re='(min|max)(\.ftz)?(\.NaN)?(\.xorsign\.abs)?\.(f32|f16x2|bf16x2)'
        local scalar_ptx_body scalar_ptx_count scalar_ptx_unique
        local mma_ptx_body mma_ptx_count mma_ptx_unique
        local minmax_ptx_body minmax_ptx_count minmax_ptx_unique
        scalar_ptx_body="$(awk '/^\.visible \.entry compile_scalar_explicit_rounding\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        scalar_ptx_count="$(grep -oE "${scalar_ptx_re}" <<<"${scalar_ptx_body}" | wc -l)"
        scalar_ptx_unique="$(grep -oE "${scalar_ptx_re}" <<<"${scalar_ptx_body}" | sort -u | wc -l)"
        mma_ptx_body="$(awk '/^\.visible \.entry compile_register_mma\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        mma_ptx_count="$(grep -oE "${ampere_float_mma_re}" <<<"${mma_ptx_body}" | wc -l)"
        mma_ptx_unique="$(grep -oE "${ampere_float_mma_re}" <<<"${mma_ptx_body}" | sort -u | wc -l)"
        minmax_ptx_body="$(awk '/^\.visible \.entry compile_extended_minmax\(/,/^}/' "${llvm_ptx}" 2>/dev/null)"
        minmax_ptx_count="$(grep -oE "${extended_minmax_re}" <<<"${minmax_ptx_body}" | wc -l)"
        minmax_ptx_unique="$(grep -oE "${extended_minmax_re}" <<<"${minmax_ptx_body}" | sort -u | wc -l)"
        if [[ ! -s "${llvm_ptx}" || -z "${scalar_ptx_body}" || -z "${mma_ptx_body}" || -z "${minmax_ptx_body}" ]] \
            || ! grep -qx '\.target sm_86' "${llvm_ptx}" \
            || [[ ${scalar_ptx_count} -ne 64 || ${scalar_ptx_unique} -ne 64 ]] \
            || [[ ${mma_ptx_count} -ne 5 || ${mma_ptx_unique} -ne 5 ]] \
            || [[ ${minmax_ptx_count} -ne 28 || ${minmax_ptx_unique} -ne 28 ]] \
            || grep -qE '(\.extern|call)[^;]*llvm[.$]nvvm[.$](mul|div|fma|add)[.$](rn|rz|rm|rp)' "${llvm_ptx}"; then
            printf 'generated_intrinsics LLVM route expected 64 scalar, 5 Ampere MMA, and 28 extended min/max forms; got %s/%s, %s/%s, and %s/%s total/unique\n' \
                "${scalar_ptx_count}" "${scalar_ptx_unique}" \
                "${mma_ptx_count}" "${mma_ptx_unique}" \
                "${minmax_ptx_count}" "${minmax_ptx_unique}" >>"${log}"
            CARGO_EC=1
            return
        fi

        local -a nvvm_args=("emit-ltoir" "${ex}" "--arch=sm_86")
        if [[ ${VERBOSE} -eq 1 ]]; then
            cargo oxide "${nvvm_args[@]}" 2>&1 | tee -a "${log}"
            CARGO_EC=${PIPESTATUS[0]}
        else
            cargo oxide "${nvvm_args[@]}" >>"${log}" 2>&1
            CARGO_EC=$?
        fi
        if [[ ${CARGO_EC} -ne 0 ]]; then
            return
        fi

        local nvvm_ll="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ll"
        local scalar_ir_body mma_ir_body minmax_ir_body mma_ir_count mma_ir_unique
        scalar_ir_body="$(awk '/^define .*@compile_scalar_explicit_rounding\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        mma_ir_body="$(awk '/^define .*@compile_register_mma\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        minmax_ir_body="$(awk '/^define .*@compile_extended_minmax\(/,/^}/' "${nvvm_ll}" 2>/dev/null)"
        mma_ir_count="$(grep -oE "${ampere_float_mma_re}" <<<"${mma_ir_body}" | wc -l)"
        mma_ir_unique="$(grep -oE "${ampere_float_mma_re}" <<<"${mma_ir_body}" | sort -u | wc -l)"
        local f32_binary_re='call float asm "((mul|div)\.(rn|rz|rm|rp)(\.ftz)?|add\.(rn|rz|rm|rp)(\.sat)?(\.ftz)?)\.f32 \$0, \$1, \$2;", "=f,f,f"'
        local f32_fma_re='call float asm "fma\.(rn|rz|rm|rp)(\.ftz)?(\.sat)?\.f32 \$0, \$1, \$2, \$3;", "=f,f,f,f"'
        local f64_binary_re='call double asm "(mul|div|add)\.(rn|rz|rm|rp)\.f64 \$0, \$1, \$2;", "=d,d,d"'
        local f64_fma_re='call double asm "fma\.(rn|rz|rm|rp)\.f64 \$0, \$1, \$2, \$3;", "=d,d,d,d"'
        local f32_minmax_re='call float asm "(min|max)(\.ftz)?(\.NaN)?(\.xorsign\.abs)?\.f32 \$0, \$1, \$2;", "=f,f,f"'
        local packed_minmax_re='call i32 asm "(min|max)(\.ftz)?(\.NaN)?(\.xorsign\.abs)?\.(f16x2|bf16x2) \$0, \$1, \$2;", "=r,r,r"'
        local f32_binary_count f32_binary_unique f32_fma_count f32_fma_unique
        local f64_binary_count f64_binary_unique f64_fma_count f64_fma_unique
        local f32_minmax_count f32_minmax_unique packed_minmax_count packed_minmax_unique
        f32_binary_count="$(grep -oE "${f32_binary_re}" <<<"${scalar_ir_body}" | wc -l)"
        f32_binary_unique="$(grep -oE "${f32_binary_re}" <<<"${scalar_ir_body}" | sort -u | wc -l)"
        f32_fma_count="$(grep -oE "${f32_fma_re}" <<<"${scalar_ir_body}" | wc -l)"
        f32_fma_unique="$(grep -oE "${f32_fma_re}" <<<"${scalar_ir_body}" | sort -u | wc -l)"
        f64_binary_count="$(grep -oE "${f64_binary_re}" <<<"${scalar_ir_body}" | wc -l)"
        f64_binary_unique="$(grep -oE "${f64_binary_re}" <<<"${scalar_ir_body}" | sort -u | wc -l)"
        f64_fma_count="$(grep -oE "${f64_fma_re}" <<<"${scalar_ir_body}" | wc -l)"
        f64_fma_unique="$(grep -oE "${f64_fma_re}" <<<"${scalar_ir_body}" | sort -u | wc -l)"
        f32_minmax_count="$(grep -oE "${f32_minmax_re}" <<<"${minmax_ir_body}" | wc -l)"
        f32_minmax_unique="$(grep -oE "${f32_minmax_re}" <<<"${minmax_ir_body}" | sort -u | wc -l)"
        packed_minmax_count="$(grep -oE "${packed_minmax_re}" <<<"${minmax_ir_body}" | wc -l)"
        packed_minmax_unique="$(grep -oE "${packed_minmax_re}" <<<"${minmax_ir_body}" | sort -u | wc -l)"
        if [[ ! -s "${nvvm_ll}" || -z "${scalar_ir_body}" || -z "${mma_ir_body}" || -z "${minmax_ir_body}" ]] \
            || [[ ${f32_binary_count} -ne 32 || ${f32_binary_unique} -ne 32 ]] \
            || [[ ${f32_fma_count} -ne 16 || ${f32_fma_unique} -ne 16 ]] \
            || [[ ${f64_binary_count} -ne 12 || ${f64_binary_unique} -ne 12 ]] \
            || [[ ${f64_fma_count} -ne 4 || ${f64_fma_unique} -ne 4 ]] \
            || [[ ${mma_ir_count} -ne 5 || ${mma_ir_unique} -ne 5 ]] \
            || [[ ${f32_minmax_count} -ne 8 || ${f32_minmax_unique} -ne 8 ]] \
            || [[ ${packed_minmax_count} -ne 20 || ${packed_minmax_unique} -ne 20 ]] \
            || grep -qE 'llvm\.nvvm\.(mul|div|fma|add)\.(rn|rz|rm|rp)|llvm\.nvvm\.f(min|max)' "${nvvm_ll}"; then
            printf 'generated_intrinsics libNVVM route expected exact scalar forms plus 5 Ampere MMA and 28 extended min/max forms; got scalar %s/%s, %s/%s, %s/%s, %s/%s, MMA %s/%s, and min/max %s/%s + %s/%s total/unique\n' \
                "${f32_binary_count}" "${f32_binary_unique}" \
                "${f32_fma_count}" "${f32_fma_unique}" \
                "${f64_binary_count}" "${f64_binary_unique}" \
                "${f64_fma_count}" "${f64_fma_unique}" \
                "${mma_ir_count}" "${mma_ir_unique}" \
                "${f32_minmax_count}" "${f32_minmax_unique}" \
                "${packed_minmax_count}" "${packed_minmax_unique}" >>"${log}"
            CARGO_EC=1
        fi
        return
    fi

    # Designated NVVM examples use `emit-ltoir` in compile-only mode so CI
    # checks both textual export and real libNVVM compilation. Other examples
    # use `build`.
    if [[ ${COMPILE_ONLY} -eq 1 ]] && verify_nvvm_in_compile_only "${ex}"; then
        local nvvm_arch
        nvvm_arch="$(nvvm_verify_arch "${ex}")"
        local -a args=("emit-ltoir" "${ex}" "--arch=${nvvm_arch}")
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
    if [[ ${COMPILE_ONLY} -eq 1 && "${ex}" == "cluster" ]]; then
        args+=("--arch=sm_90")
    fi
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
    if [[ ${CARGO_EC} -eq 0 && ${COMPILE_ONLY} -eq 1 && "${ex}" == "standalone_device_fn" ]]; then
        local ptx="crates/rustc-codegen-cuda/examples/${ex}/${ex}.ptx"
        local tf32_count
        tf32_count="$(grep -oF 'cvt.rna.tf32.f32' "${ptx}" 2>/dev/null | wc -l)"
        if [[ ! -s "${ptx}" || ${tf32_count} -ne 6 ]] \
            || grep -qE '(\.extern.*f2tf32|call[^;]*f2tf32)' "${ptx}"; then
            printf 'standalone_device_fn did not emit exactly 6 direct generated TF32 conversions\n' >>"${log}"
            CARGO_EC=1
        fi
    fi
    if [[ ${CARGO_EC} -eq 0 && ${COMPILE_ONLY} -eq 1 && "${ex}" == "cluster" ]]; then
        local ptx="crates/rustc-codegen-cuda/examples/cluster/cluster.ptx"
        local body cluster_count cluster_unique ncluster_count ncluster_unique mad_count mul_count store_count
        body="$(awk '/^\.visible \.entry compile_cluster_grid_helpers\(/,/^}/' "${ptx}" 2>/dev/null)"
        cluster_count="$(grep -oE '%clusterid\.[xyz]' <<<"${body}" | wc -l)"
        cluster_unique="$(grep -oE '%clusterid\.[xyz]' <<<"${body}" | sort -u | wc -l)"
        ncluster_count="$(grep -oE '%nclusterid\.[xyz]' <<<"${body}" | wc -l)"
        ncluster_unique="$(grep -oE '%nclusterid\.[xyz]' <<<"${body}" | sort -u | wc -l)"
        mad_count="$(grep -c 'mad\.lo\.s32' <<<"${body}")"
        mul_count="$(grep -c 'mul\.lo\.s32' <<<"${body}")"
        store_count="$(grep -c 'st\.global\.b32' <<<"${body}")"
        if [[ ! -s "${ptx}" || -z "${body}" ]] \
            || ! grep -qx '\.target sm_90' "${ptx}" \
            || [[ ${cluster_count} -ne 3 || ${cluster_unique} -ne 3 ]] \
            || [[ ${ncluster_count} -ne 3 || ${ncluster_unique} -ne 3 ]] \
            || [[ ${mad_count} -ne 2 || ${mul_count} -ne 2 || ${store_count} -ne 2 ]]; then
            printf 'cluster helper did not emit the exact 6-read, 2-mad, 2-mul formula\n' >>"${log}"
            if [[ ${VERBOSE} -eq 1 ]]; then
                printf 'cluster helper did not emit the exact 6-read, 2-mad, 2-mul formula\n'
            fi
            CARGO_EC=1
        fi
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
    elif [[ ( ${COMPILE_ONLY} -eq 1 || "${cat}" == "blackwell-compile" ) && "${cat}" != "error" ]]; then
        # Compile-only collapses the GPU-gated categories: with nothing
        # executed, "PTX (or NVVM IR) compiled" is the bar for everything
        # except error examples, which must still fail with a diagnostic.
        verdict="$(verdict_compile "${ex}" "${log}" "${ec}")" && status=0 || status=$?
    else
        case "${cat}" in
            error)       verdict="$(verdict_error       "${log}" "${ec}" "${ex}")" && status=0 || status=$? ;;
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
