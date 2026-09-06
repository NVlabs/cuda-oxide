#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Reject a prose reference to a source file that is not in the repository.
#
# #1190 split large files into module directories and left three prose
# references pointing at paths it had removed -- a test's module doc at
# `crates/mir-lower/src/convert/types.rs`, two example READMEs at
# `crates/mir-importer/src/translator/rvalue.rs`. #1197 repointed those, and 14
# more in other spellings. Nothing had noticed: a path in prose is compiled by
# nothing, `check-host-api-paths.sh` checks Rust *API* paths rather than files,
# and the book gate only builds. The reference just goes quiet, and a reader
# follows it into nothing.
#
# "In the repository" means **tracked by git**, not present on disk. Those are
# different questions and only the first one is reproducible: a path that
# exists solely as a local build artifact, or an untracked file someone has in
# their tree, resolves for them and for nobody else. Testing the filesystem
# would make this guard pass or fail depending on whose checkout it ran in.
#
# Scope, deliberately narrow, because this is the kind of guard that goes soft
# the moment it starts inferring:
#
#   * A path is only checked when it is anchored at a repo root -- crates/,
#     cuda-oxide-book/, scripts/ -- and carries a file extension. That
#     is the spelling a new reference normally takes, and the only one that is
#     unambiguous on its own. Each root is verified to be a real tracked
#     directory before the sweep, so a renamed root fails here rather than
#     silently matching nothing.
#   * Prose only: every line of a tracked *.md, and `///` or `//!` lines in a
#     tracked *.rs. This is what removes the need for a general exemption
#     list. The non-existent paths that live in Rust *code* are all deliberate
#     -- synthetic fixture names (`intrinsics/probes/removed.ll`,
#     `intrinsics/overlay/test.toml`), two paths that cuda-intrinsics-gen
#     render tests assert are *absent*, and the mktemp canary in
#     check-reserved-prefixes.sh -- and none of them is prose.
#
# Generated outputs are declared, never inferred. An earlier revision skipped a
# path whose parent directory held no tracked file, reasoning that such a
# directory must be an output location. That is exactly backwards: deleting or
# misspelling a directory produces the same signal as generating into one, so
# the broken reference this guard exists to catch was the case it let through.
# The list below is the whole accommodation, and it is verified rather than
# trusted -- an entry nothing refers to any more is an error, so it cannot rot
# into a silent exemption.
#
# Two things stay out of scope on purpose:
#
#   * `intrinsics/` is not a root. It is both a real top-level directory and a
#     common crate-relative fragment: examples/atomics/README.md names
#     `intrinsics/atomic.rs` in a pipeline diagram, meaning
#     `mir-importer/src/translator/terminator/intrinsics/atomic.rs`, which is
#     correct as shorthand. Rooting there would fail that line.
#   * Crate-relative (`mir-lower/src/convert/types.rs`) and bare-basename
#     ("the walker in `rvalue.rs`") spellings are not checked. #1197's
#     follow-up fixed 14 references written that way, so this is a real gap and
#     is stated rather than papered over -- telling `rvalue.rs` in prose from
#     any other mention of it is guesswork, and a guard that guesses is worse
#     than one that is narrow.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

ROOTS='crates|cuda-oxide-book|scripts'
EXTS='rs|md|sh|toml|jsonl|json|ll|py|yaml|yml'
# The trailing \b matters: without it `.json` matches inside `.jsonl` and the
# guard reports a path nobody wrote. Longer alternatives lead for the same
# reason.
PATTERN="(${ROOTS})/[A-Za-z0-9._/-]+\.(${EXTS})\b"

# Paths that prose may name even though the repository does not contain them,
# because something writes them at run time. Each one is an explicit decision,
# not a shape the guard infers.
#
#   crates/fuzzer/artifacts/summary.jsonl -- run_seed.py writes it and clears
#   the directory on every invocation; crates/fuzzer/README.md documents it.
GENERATED_OUTPUT_PATHS=(
    crates/fuzzer/artifacts/summary.jsonl
)

tracked_list="$(mktemp)"
trap 'rm -f "${tracked_list}"' EXIT
git ls-files >"${tracked_list}"

is_tracked() { grep -qxF -- "$1" "${tracked_list}"; }

is_generated_output() {
    local declared
    for declared in "${GENERATED_OUTPUT_PATHS[@]}"; do
        [[ "$1" == "${declared}" ]] && return 0
    done
    return 1
}

# Roots are verified rather than assumed: if one is renamed, the pattern would
# quietly stop matching anything under it and the guard would report a clean
# tree for the wrong reason.
for root in crates cuda-oxide-book scripts; do
    if ! grep -qE "^${root}/" "${tracked_list}"; then
        echo "error: source-reference guard: '${root}/' holds no tracked file," >&2
        echo "       so the pattern anchored there can no longer match; update" >&2
        echo "       ROOTS in $0" >&2
        exit 1
    fi
done

# `grep -n` keeps the line number; the *.rs arm narrows to doc comments first
# so a path mentioned in code is never read as a claim about the tree.
prose_lines() {
    case "$1" in
    *.md) grep -nE "${PATTERN}" -- "$1" 2>/dev/null || true ;;
    *.rs) grep -nE '^[[:space:]]*(///|//!)' -- "$1" 2>/dev/null |
        grep -E "${PATTERN}" || true ;;
    esac
}

# The one scanner. The self-test below runs *this*, not a paraphrase of it, so
# disabling any step -- the pattern, the tracked test, the generated-output
# accommodation -- fails the self-test instead of quietly reporting a clean
# tree.
broken_in_file() {
    local file="$1" hit lineno path
    while IFS= read -r hit; do
        [ -n "${hit}" ] || continue
        lineno="${hit%%:*}"
        while IFS= read -r path; do
            [ -n "${path}" ] || continue
            # A traversal segment cannot be resolved against the repository
            # root, so it is never a valid claim about a tracked file.
            case "${path}" in
            */../* | ../* | */..) printf '%s:%s: names %s, which escapes the repository root\n' \
                "${file}" "${lineno}" "${path}"; continue ;;
            esac
            is_generated_output "${path}" && continue
            if ! is_tracked "${path}"; then
                printf '%s:%s: names %s, which is not tracked in this repository\n' \
                    "${file}" "${lineno}" "${path}"
            fi
        done < <(printf '%s\n' "${hit#*:}" | grep -oE "${PATTERN}" | sort -u)
    done < <(prose_lines "${file}")
}

# Declared accommodations are verified, not trusted: an entry nothing names any
# more is a silent exemption waiting to hide a real break.
for declared in "${GENERATED_OUTPUT_PATHS[@]}"; do
    if is_tracked "${declared}"; then
        echo "error: source-reference guard: '${declared}' is listed as a" >&2
        echo "       generated output but is tracked; drop it from" >&2
        echo "       GENERATED_OUTPUT_PATHS in $0" >&2
        exit 1
    fi
    if ! git grep -qF -- "${declared}" -- '*.md' '*.rs' 2>/dev/null; then
        echo "error: source-reference guard: nothing refers to '${declared}'" >&2
        echo "       any more; drop it from GENERATED_OUTPUT_PATHS in $0" >&2
        exit 1
    fi
done

# Self-test. The failure mode this guard has to survive is "silently stops
# reporting", so run the real scanner over prose naming, in turn: a missing
# file whose directory the repo does track, a missing file whose directory it
# does *not* (the case an earlier revision let through), an untracked file that
# exists on disk right now, and a tracked file. The first three must be
# reported and the fourth must not.
canary="$(mktemp -d)"
canary_untracked="crates/zz-source-reference-canary-$$.rs"
trap 'rm -f "${tracked_list}" "${canary_untracked}"; rm -rf "${canary}"' EXIT
printf 'prose naming crates/cuda-device/src/no-such-file-%s.rs inline\n' "$$" >"${canary}/dead_in_live_dir.md"
printf 'prose naming crates/no-such-crate-%s/src/lib.rs inline\n' "$$" >"${canary}/dead_dir.md"
printf 'prose naming %s inline\n' "${canary_untracked}" >"${canary}/untracked.md"
# Any tracked path the pattern matches, chosen from the index rather than
# named: hard-coding this script's own path made the self-test depend on the
# guard already being committed, which is false on the branch that adds it and
# in any historical worktree.
canary_live="$(grep -m1 -E "^(${ROOTS})/[A-Za-z0-9._/-]+\.(${EXTS})$" "${tracked_list}")"
if [ -z "${canary_live}" ]; then
    echo "error: source-reference guard: the index holds no path the pattern" >&2
    echo "       matches, so the self-test cannot prove the tracked case" >&2
    exit 1
fi
printf 'prose naming %s inline\n' "${canary_live}" >"${canary}/live.md"
: >"${canary_untracked}"

canary_hits() { broken_in_file "${canary}/$1" | wc -l; }
if [ "$(canary_hits dead_in_live_dir.md)" -ne 1 ] ||
    [ "$(canary_hits dead_dir.md)" -ne 1 ] ||
    [ "$(canary_hits untracked.md)" -ne 1 ] ||
    [ "$(canary_hits live.md)" -ne 0 ]; then
    echo "error: source-reference guard self-test failed: the scanner no longer" >&2
    echo "       separates a tracked path from a missing one, a missing" >&2
    echo "       directory, or a file that exists only in this checkout, so a" >&2
    echo "       clean result on the tree means nothing" >&2
    exit 1
fi
rm -f "${canary_untracked}"

checked=0
broken=0
while IFS= read -r file; do
    while IFS= read -r problem; do
        [ -n "${problem}" ] || continue
        printf '%s\n' "${problem}" >&2
        broken=$((broken + 1))
    done < <(broken_in_file "${file}")
    while IFS= read -r _path; do
        [ -n "${_path}" ] || continue
        checked=$((checked + 1))
    done < <(prose_lines "${file}" | grep -oE "${PATTERN}" | sort -u)
done < <(git ls-files -- '*.md' '*.rs')

if [ "${broken}" -ne 0 ]; then
    echo "" >&2
    echo "error: ${broken} prose reference(s) name a path this repository does" >&2
    echo "       not track. Repoint each one at where the code lives now; if" >&2
    echo "       the surrounding sentence describes the old shape, it needs" >&2
    echo "       rewriting too, not just a new path. A path written at run" >&2
    echo "       time belongs in GENERATED_OUTPUT_PATHS in $0, with the reason." >&2
    exit 1
fi

echo "source references ok: ${checked} repo-anchored paths in prose, all tracked"
