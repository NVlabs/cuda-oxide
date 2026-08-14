#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Enforce deny.toml over the example workspaces, which `cargo deny check` does
# not reach.
#
# `cargo deny check` resolves the root workspace.  Every example under
# crates/rustc-codegen-cuda/examples/ sets its own `[workspace]`, so the root
# run stops at that boundary and the license, source and ban policies never see
# any crate an example pulls on its own.  #664 closed the *inventory* half of
# this (dependency-licenses.csv now records those crates); this closes the
# *policy* half, so the allow-list in deny.toml actually governs them.
#
# Measured when this guard was written: 186 of the 187 example lock files
# already satisfy the policy unchanged, so this is a guard against future drift
# rather than a fix for a present violation.  The one exception is exempted
# below, with its reason.
#
# Why one run per distinct dependency set, not one per workspace:
#
#   Every example depends on cuda-core/cuda-device/cuda-host by path, so each
#   lock file re-lists the root workspace's own transitive crates.  Grouping the
#   lock files by their exact set of third-party (name, version, source) triples
#   collapses 187 lock files (186 example workspaces plus one nested
#   sub-workspace) to 26 distinct sets and one cargo-deny run each.  That is an
#   equivalence, not a sample: license, source and advisory verdicts are
#   per-crate properties, so two workspaces resolving the identical crate set
#   get the identical verdict.  `bans.multiple-versions` is a graph property,
#   but deny.toml sets it to "warn", so it cannot change the exit status.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

EXAMPLES_ROOT=crates/rustc-codegen-cuda/examples

# Examples whose dependencies deliberately cannot satisfy deny.toml today.
#
# cutile_inter_kernel links NVlabs/cutile-rs by git.  Measured against the
# current policy it fails two ways, neither of which this guard should paper
# over:
#
#   error[source-not-allowed]: detected 'git' source not explicitly allowed  (x7)
#   error[unlicensed]: cuda-bindings = 0.1.0 is unlicensed
#
# The first needs cutile-rs added to `[sources] allow-git`, the second needs a
# license field on a crate this repository does not own.  Both are policy calls
# for a maintainer, and they are the open question in #663 -- the same reason
# the example is already exempt from the inventory guard.  Delete the entry once
# that is settled.
#
# Every name here is checked against the examples on disk below, so a typo or a
# rename fails the run instead of quietly exempting nothing -- or everything.
# An exemption covers every lock file under the example, so
# cutile_inter_kernel/simt (a nested sub-workspace) is exempt with its parent.
POLICY_EXEMPT_EXAMPLES=(cutile_inter_kernel)

command -v cargo-deny >/dev/null 2>&1 || {
    echo "error: cargo-deny not found; install it with 'cargo install cargo-deny --locked'" >&2
    exit 1
}

# One representative manifest per distinct third-party dependency set.  Lock
# files, not example directories, are the unit: a nested sub-workspace resolves
# its own graph, so it must be checked (or exempted) as a workspace of its own,
# not merely folded into its parent's grouping key.
representatives="$(python3 -c '
import glob, os, re, sys

examples_root, *exempt = sys.argv[1:]

def third_party(lock):
    """(name, version, source) for every locked package that has a source.

    A package with no `source` is a path dependency, i.e. first-party by
    construction, and carries no policy question of its own.  The source is
    part of the key because the sources policy judges where a crate comes
    from: the same (name, version) from crates.io and from a git fork are
    different policy questions.
    """
    found = set()
    seen = 0
    for block in open(lock).read().split("[[package]]")[1:]:
        seen += 1
        name = re.search(r"^name = \"([^\"]+)\"", block, re.M)
        version = re.search(r"^version = \"([^\"]+)\"", block, re.M)
        source = re.search(r"^source = \"([^\"]+)\"", block, re.M)
        if name and version and source:
            found.add((name.group(1), version.group(1), source.group(1)))
    return found, seen

locks = sorted(glob.glob(os.path.join(examples_root, "**", "Cargo.lock"), recursive=True))
if len(locks) < 20:
    sys.exit("parse self-test failed: found %d example lock files" % len(locks))

on_disk = {os.path.relpath(lock, examples_root).split(os.sep)[0] for lock in locks}
unknown = sorted(set(exempt) - on_disk)
if unknown:
    sys.exit("POLICY_EXEMPT_EXAMPLES names no such example: " + ", ".join(unknown))

groups = {}
total_seen = 0
for lock in locks:
    example = os.path.relpath(lock, examples_root).split(os.sep)[0]
    crates, seen = third_party(lock)
    total_seen += seen
    if example in exempt:
        continue
    groups.setdefault(frozenset(crates), os.path.join(os.path.dirname(lock), "Cargo.toml"))

# Mirrors the inventory guard: if the lock-file regexes silently rotted, the
# groups would quietly collapse and a single run would vouch for everything.
if total_seen < 100:
    sys.exit("parse self-test failed: read %d packages from %d lock files" % (total_seen, len(locks)))

for manifest in sorted(groups.values()):
    print(manifest)
' "${EXAMPLES_ROOT}" "${POLICY_EXEMPT_EXAMPLES[@]}")"

total="$(printf '%s\n' "${representatives}" | grep -c .)"
echo "Checking deny.toml over ${total} representative example workspaces."

# No --config: cargo-deny resolves the config by walking up from the manifest
# directory, so every example workspace finds the repository-root deny.toml.
# (Verified on cargo-deny 0.19 and 0.20; a top-level --config flag only exists
# on 0.20.)  --locked so a stale lock file fails the run instead of being
# silently re-resolved: the committed lock is what grouped the workspace, so it
# must be what the policy judges.
failed=()
for manifest in ${representatives}; do
    if ! cargo deny --manifest-path "${manifest}" --locked check 2>&1 |
        sed "s|^|  [$(dirname "${manifest#"${EXAMPLES_ROOT}/"}")] |"; then
        failed+=("${manifest}")
    fi
done

if ((${#failed[@]})); then
    echo "error: deny.toml is not satisfied by these example workspaces:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    echo "       Each is the representative for a group of workspaces resolving the" >&2
    echo "       same third-party crates, so the cause is shared by its whole group." >&2
    exit 1
fi

echo "OK: deny.toml holds over every example workspace outside POLICY_EXEMPT_EXAMPLES."
