#!/usr/bin/env bash
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
# Measured when this guard was written: 186 of the 187 example workspaces
# already satisfy the policy unchanged, so this is a guard against future drift
# rather than a fix for a present violation.  The one exception is exempted
# below, with its reason.
#
# Why one run per distinct dependency set, not one per example:
#
#   Every example depends on cuda-core/cuda-device/cuda-host by path, so each
#   lock file re-lists the root workspace's own transitive crates.  Grouping the
#   examples by their exact set of third-party (name, version) pairs collapses
#   187 workspaces to 27 distinct sets and one cargo-deny run each.  That is an
#   equivalence, not a sample: licenses and sources are per-crate properties, so
#   two workspaces resolving the identical crate set get the identical verdict.
#   `bans.multiple-versions` is a graph property, but deny.toml sets it to
#   "warn", so it cannot change the exit status.
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
POLICY_EXEMPT_EXAMPLES=(cutile_inter_kernel)

command -v cargo-deny >/dev/null 2>&1 || {
    echo "error: cargo-deny not found; install it with 'cargo install cargo-deny --locked'" >&2
    exit 1
}

# One representative example per distinct third-party dependency set.
representatives="$(python3 -c '
import glob, os, re, sys

examples_root, *exempt = sys.argv[1:]

def third_party(lock):
    """(name, version) for every locked package that has a source.

    A package with no `source` is a path dependency, i.e. first-party by
    construction, and carries no policy question of its own.
    """
    found = set()
    for block in open(lock).read().split("[[package]]")[1:]:
        name = re.search(r"^name = \"([^\"]+)\"", block, re.M)
        version = re.search(r"^version = \"([^\"]+)\"", block, re.M)
        source = re.search(r"^source = \"([^\"]+)\"", block, re.M)
        if name and version and source:
            found.add((name.group(1), version.group(1)))
    return found

# Recursive, so a lock file in a nested sub-workspace (cutile_inter_kernel/simt)
# is attributed to its top-level example instead of escaping the guard.
sets = {}
for lock in sorted(glob.glob(os.path.join(examples_root, "**", "Cargo.lock"), recursive=True)):
    example = os.path.relpath(lock, examples_root).split(os.sep)[0]
    sets.setdefault(example, set()).update(third_party(lock))

on_disk = set(sets)
unknown = sorted(set(exempt) - on_disk)
if unknown:
    sys.exit("POLICY_EXEMPT_EXAMPLES names no such example: " + ", ".join(unknown))

groups = {}
for example, crates in sorted(sets.items()):
    if example in exempt:
        continue
    groups.setdefault(frozenset(crates), example)

for example in sorted(groups.values()):
    print(example)
' "${EXAMPLES_ROOT}" "${POLICY_EXEMPT_EXAMPLES[@]}")"

total="$(printf '%s\n' "${representatives}" | grep -c .)"
echo "Checking deny.toml over ${total} representative example workspaces."

failed=()
for example in ${representatives}; do
    manifest="${EXAMPLES_ROOT}/${example}/Cargo.toml"
    if ! cargo deny --manifest-path "${manifest}" --config deny.toml check 2>&1 |
        sed "s/^/  [${example}] /"; then
        failed+=("${example}")
    fi
done

if ((${#failed[@]})); then
    echo "error: deny.toml is not satisfied by these example workspaces:" >&2
    printf '  %s\n' "${failed[@]}" >&2
    echo "       Each is the representative for a group of examples resolving the" >&2
    echo "       same third-party crates, so the cause is shared by its whole group." >&2
    exit 1
fi

echo "OK: deny.toml holds over every example workspace outside POLICY_EXEMPT_EXAMPLES."
