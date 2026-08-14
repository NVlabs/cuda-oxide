#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify dependency-licenses.csv still records every crate the workspace
# declares: each root-workspace member, and each directly declared third-party
# dependency (normal, dev, or build).  Run this after adding or removing a
# dependency or a workspace member.
#
# Scope, and why it stops where it does:
#
#   * Presence only, never versions.  The CSV records a snapshot while
#     Cargo.lock moves on its own, so comparing versions would fail on every
#     routine bump while saying nothing about licensing.
#   * Direct dependencies only, not the whole resolved graph.  cargo-deny
#     already enforces the license *policy* over every transitive crate
#     (deny.toml `[licenses]`).  This guard covers the other half: that the
#     human-readable inventory does not silently fall behind what the
#     workspace declares.  Transitive rows in the CSV are welcome, just not
#     required.
set -euo pipefail

# Pin the collation locale for every sort/comm in this script.  Both comm
# inputs are produced with byte-wise C ordering; without this, an ambient
# UTF-8 locale (e.g. en_US.UTF-8) makes GNU comm re-check the order under
# dictionary collation, reject the pair rustc-hash/rustc_apfloat with
# "input is not in sorted order", and abort the run via set -e.
export LC_ALL=C

cd "$(dirname "$0")/.."

CSV=dependency-licenses.csv

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "error: $1 is required to verify ${CSV}" >&2
        echo "       refusing to report success from a check that cannot run" >&2
        exit 1
    fi
}
require_tool cargo
require_tool python3

test -s "${CSV}"

# Column 1 is the package name: always a bare crate name, never quoted and
# never containing a comma, so a plain field split is safe.  Later columns do
# use quoting (descriptions contain commas), which is why only column 1 is
# read this way.  The file uses CRLF, hence the `tr -d '\r'`.
recorded="$(tail -n +2 "${CSV}" | cut -d, -f1 | tr -d '\r' | LC_ALL=C sort -u)"

# Self-test.  The failure mode a guard like this has to survive is "quietly
# stops seeing anything", so prove the CSV still parses into a plausible set
# of names before believing a clean result.
recorded_count="$(printf '%s\n' "${recorded}" | grep -c . || true)"
data_rows=$(($(wc -l <"${CSV}") - 1))
: "${recorded_count:=0}"
if [[ ${recorded_count} -lt 20 || ${data_rows} -lt 20 ]]; then
    echo "error: ${CSV} parse self-test failed: read ${recorded_count} package" \
        "names from ${data_rows} data rows" >&2
    echo "       the file layout changed; fix this script before trusting it" >&2
    exit 1
fi

# Members and directly declared third-party dependencies of one workspace.
#
# `--locked` so the guard reads the committed resolution rather than silently
# updating Cargo.lock to satisfy itself: a check that can rewrite its own input
# is not a check.
declared_crates() {
    cargo metadata --locked --format-version 1 --manifest-path "$1" | python3 -c '
import json, sys

metadata = json.load(sys.stdin)
workspace = set(metadata["workspace_members"])
members = [p for p in metadata["packages"] if p["id"] in workspace]
member_names = {p["name"] for p in members}

names = set(member_names)
for package in members:
    for dependency in package["dependencies"]:
        # Sibling path dependencies are already covered as workspace members.
        if dependency["name"] not in member_names:
            names.add(dependency["name"])

print("\n".join(sorted(names)))
'
}

# Both first-party workspaces, not just the root one.  crates/rustc-codegen-cuda
# carries its own `[workspace]` for the rustc-private dylibs, so `-p` from the
# root cannot reach it and the root `cargo metadata` above stops at that
# boundary -- which is how the backend crate itself, the largest first-party
# crate in the tree, sat unrecorded while every other member had a row.  This is
# the second pass asked for in the #662 review.
required="$(
    {
        declared_crates Cargo.toml
        declared_crates crates/rustc-codegen-cuda/Cargo.toml
    } | LC_ALL=C sort -u
)"

missing="$(comm -23 <(printf '%s\n' "${required}") <(printf '%s\n' "${recorded}"))"

if [[ -n "${missing}" ]]; then
    echo "error: ${CSV} is missing a row for:" >&2
    printf '%s\n' "${missing}" | sed 's/^/  /' >&2
    echo >&2
    echo "Every workspace member and every directly declared third-party" >&2
    echo "dependency needs a row.  See CONTRIBUTING.md ('If adding a new" >&2
    echo "dependency, update dependency-licenses.csv accordingly') and copy" >&2
    echo "the column layout from an existing row of the same kind." >&2
    exit 1
fi

echo "OK: ${CSV} records all $(printf '%s\n' "${required}" | grep -c .) declared crates."

# ---------------------------------------------------------------------------
# Second half: the example workspaces.
#
# Every example under crates/rustc-codegen-cuda/examples/ sets its own
# [workspace], so neither `cargo deny check` nor the check above resolves any
# of them -- both stop at the root workspace boundary.  Most examples declare
# only path dependencies on first-party crates and so bring nothing new, but a
# few link third-party code (tokio, rayon, libm, the cutile-rs git dependency),
# and that code is compiled by `cargo oxide run <example>` and by
# scripts/smoketest.sh without any license gate seeing it.
#
# Presence only, as above.  Lock files are parsed directly rather than through
# `cargo metadata`: resolving every example workspace separately would be slow
# and would need the network, and the lock files already record the resolved
# graph.  The search is recursive so a lockfile in a nested sub-workspace
# (e.g. cutile_inter_kernel/simt) is inventoried under its top-level example
# instead of escaping the guard.
#
# A package counts as covered when it is in the root graph, has a CSV row, or
# carries no `source` field.  That last case is a path dependency, which is
# first-party by construction.  Name matching is deliberately avoided -- some
# first-party crates are pulled by git rather than by path (cuda-core and
# friends in cutile_inter_kernel), so a heuristic over names would misfile
# them.
# Examples whose third-party dependencies are deliberately out of inventory
# scope.  cutile_inter_kernel links cutile-rs by git, which resolves a further
# ~60 crates (wasm-bindgen, wit-bindgen, wasmparser, windows-targets) that exist
# in this tree only to build one interop example.  Whether those belong in the
# inventory is the open question in #663; until it is settled the example is
# listed here rather than left silently uncovered.  Delete the entry to require
# the rows.
#
# Every name here is checked against the examples on disk below, so a typo or a
# rename fails the run instead of quietly exempting nothing -- or everything.
INVENTORY_EXEMPT_EXAMPLES=(cutile_inter_kernel)

examples_missing="$(python3 -c '
import glob, os, re, sys

def packages(path):
    """(name, has_source) for every [[package]] in a Cargo.lock."""
    out = []
    for block in open(path).read().split("[[package]]")[1:]:
        name = re.search(r"^name = \"([^\"]+)\"", block, re.M)
        if name:
            out.append((name.group(1), re.search(r"^source = ", block, re.M) is not None))
    return out

covered = set()
for lock in ("Cargo.lock", "crates/rustc-codegen-cuda/Cargo.lock"):
    covered |= {name for name, _ in packages(lock)}

with open("dependency-licenses.csv", newline="") as handle:
    next(handle, None)
    covered |= {line.split(",")[0].strip().strip("\r") for line in handle if line.strip()}

examples_root = "crates/rustc-codegen-cuda/examples"
locks = sorted(glob.glob(os.path.join(examples_root, "**", "Cargo.lock"), recursive=True))
if len(locks) < 20:
    sys.exit("parse self-test failed: found %d example lock files" % len(locks))

def example_of(lock):
    """Top-level example directory a lockfile belongs to, however deep it sits."""
    return os.path.relpath(lock, examples_root).split(os.sep)[0]

present = {example_of(lock) for lock in locks}
exempt = set(sys.argv[1:])
unknown = sorted(exempt - present)
if unknown:
    sys.exit("INVENTORY_EXEMPT_EXAMPLES names no such example: " + ", ".join(unknown))

seen = 0
findings = []
for lock in locks:
    example = example_of(lock)
    entries = packages(lock)
    seen += len(entries)
    if example in exempt:
        continue
    extra = sorted({n for n, sourced in entries if sourced and n not in covered})
    if extra:
        findings.append((example, extra))

if seen < 100:
    sys.exit("parse self-test failed: read %d packages from %d lock files" % (seen, len(locks)))

for example, extra in findings:
    print("%s: %s" % (example, " ".join(extra)))
' "${INVENTORY_EXEMPT_EXAMPLES[@]}")"

if [[ -n "${examples_missing}" ]]; then
    echo "error: ${CSV} is missing rows for third-party crates that example" >&2
    echo "       workspaces compile (neither cargo-deny nor the check above" >&2
    echo "       resolves these -- both stop at the root workspace):" >&2
    printf '%s\n' "${examples_missing}" | sed 's/^/  /' >&2
    exit 1
fi

echo "OK: ${CSV} also covers every third-party crate the example workspaces pull."
