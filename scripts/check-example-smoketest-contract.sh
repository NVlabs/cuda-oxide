#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify the three things scripts/smoketest.sh assumes about the examples, all
# of which are otherwise only discoverable by running the suite on a GPU:
#
#   1. Every name in a *_EXAMPLES array is a real example directory.  These
#      arrays drive classify(), so a typo or a renamed example does not fail
#      loudly -- the entry simply matches nothing and the example silently
#      falls through to the `standard` category and the wrong verdict rules.
#
#   2. No example is claimed by two category arrays.  classify() returns the
#      first category that matches, so a second claim is not a conflict it can
#      report -- it is silently ignored, taking the example's arch gate and
#      verdict rules with it.  Listing `wgmma` in TCGEN05_EXAMPLES, say, gates
#      a Hopper example on Blackwell and both existing guards still pass.
#
#   3. Every `standard` example can actually report success.  That category's
#      verdict requires a SUCCESS/PASS/Complete marker in the output, and an
#      example that verifies its results but never prints one is reported as
#      `FAIL (no success marker)`.  That is a false failure, and it has landed
#      before: 981b9eb0 had to add a marker to disjoint_from_raw_parts after
#      #670 (the new example) and #665 (stricter verdicts) merged in one batch.
#
# None of these need a GPU, and none is reachable from the compile-only CI
# lane, which collapses every category into verdict_compile.
#
# Run this after adding an example or editing a *_EXAMPLES array.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

SMOKETEST=scripts/smoketest.sh
EXAMPLES_ROOT=crates/rustc-codegen-cuda/examples

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify the smoketest example contract" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -s "${SMOKETEST}"

# The marker and skip patterns below duplicate verdict_standard's greps.  Pin
# that coupling: if the verdict changes its patterns, this guard is stale and
# must say so rather than keep checking the old contract.
if ! grep -Fq "'SUCCESS|PASS|Complete'" "${SMOKETEST}"; then
    echo "error: ${SMOKETEST} no longer greps 'SUCCESS|PASS|Complete' for success" >&2
    echo "       verdict_standard's contract changed; update this guard" >&2
    exit 1
fi
if ! grep -Fq "'^[[:space:]]*(skipping:|pass \\(skipped\\))'" "${SMOKETEST}"; then
    echo "error: ${SMOKETEST} no longer greps the skip declaration this guard expects" >&2
    echo "       verdict_standard's contract changed; update this guard" >&2
    exit 1
fi

python3 - "${SMOKETEST}" "${EXAMPLES_ROOT}" <<'PY'
import glob
import os
import re
import sys

smoketest, examples_root = sys.argv[1], sys.argv[2]
source = open(smoketest, encoding="utf-8").read()

# Same two patterns verdict_standard applies to the run log.
MARKER = re.compile(r"SUCCESS|PASS|Complete")
SKIP = re.compile(r"^[ \t]*(skipping:|pass \(skipped\))", re.IGNORECASE)

lists = re.findall(r"^([A-Z0-9_]+_EXAMPLES)=\(([^)]*)\)", source, re.M)
on_disk = sorted(
    os.path.basename(os.path.dirname(path))
    for path in glob.glob(os.path.join(examples_root, "*", "Cargo.toml"))
)

# Parse self-tests: a guard whose failure mode is "matched nothing" has to
# prove it still reads both inputs before a clean result is believed.
if len(lists) < 9:
    sys.exit(f"parse self-test failed: found {len(lists)} *_EXAMPLES arrays in {smoketest}")
if len(on_disk) < 100:
    sys.exit(f"parse self-test failed: found {len(on_disk)} examples under {examples_root}")

failures = []

known = set(on_disk)
for name, body in lists:
    phantom = [entry for entry in body.split() if entry not in known]
    if phantom:
        failures.append(
            f"{name} names {len(phantom)} example(s) that do not exist: " + " ".join(phantom)
        )

# classify() returns `standard` for anything not claimed by a category array.
# NVVM_VERIFY / NOINLINE_MIR / NO_LAUNCH / NO_OPT_SHAPE are modifiers, not
# categories, so their members stay `standard` and still need a marker.
CATEGORY_LISTS = (
    "TCGEN05_EXAMPLES",
    "WGMMA_EXAMPLES",
    "BLACKWELL_MMA_EXAMPLES",
    "LTOIR_EXAMPLES",
    "LTOIR_MODERN_EXAMPLES",
    "AUTO_NVVM_EXAMPLES",
    "IKET_EXAMPLES",
    "BLACKWELL_COMPILE_EXAMPLES",
    "SM100_COMPILE_EXAMPLES",
    "ERROR_EXAMPLES",
)
by_name = dict(lists)
missing_arrays = [name for name in CATEGORY_LISTS if name not in by_name]
if missing_arrays:
    sys.exit(
        "parse self-test failed: classify() arrays not found in "
        f"{smoketest}: " + " ".join(missing_arrays)
    )

categorised = {
    entry for name in CATEGORY_LISTS for entry in by_name[name].split()
}

# classify() returns the *first* category that claims an example, so an entry
# in two category arrays silently takes whichever loop runs first, and with it
# the wrong compute-capability gate and the wrong verdict rules. Nothing else
# reports that: the phantom check above passes because both names are real
# examples, and the set comprehension here discards the duplicate by
# construction. Uniqueness has to be asserted against the raw lists.
claimed_by = {}
for name in CATEGORY_LISTS:
    for entry in by_name[name].split():
        claimed_by.setdefault(entry, []).append(name)
for entry, names in sorted(claimed_by.items()):
    if len(names) > 1:
        failures.append(
            f"{entry} is claimed by {len(names)} category arrays "
            f"({', '.join(names)}); classify() checks {names[0]} first "
            "and ignores the rest"
        )

unmarked = []
for example in on_disk:
    if example in categorised:
        continue
    pattern = os.path.join(examples_root, example, "src", "**", "*.rs")
    has_marker = False
    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lstrip().startswith("//"):
                    continue
                if MARKER.search(line) and not SKIP.search(line):
                    has_marker = True
                    break
        if has_marker:
            break
    if not has_marker:
        unmarked.append(example)

if unmarked:
    failures.append(
        "these `standard` examples print no SUCCESS/PASS/Complete marker, so "
        "smoketest reports FAIL (no success marker):\n  "
        + "\n  ".join(unmarked)
    )

if failures:
    print("error: smoketest example contract violated", file=sys.stderr)
    for failure in failures:
        print(f"  {failure}", file=sys.stderr)
    sys.exit(1)

standard = len(on_disk) - len(categorised)
print(
    f"OK: {len(lists)} *_EXAMPLES arrays name only real examples, no example "
    f"is claimed by two categories, and all {standard} standard examples "
    "print a success marker."
)
PY
