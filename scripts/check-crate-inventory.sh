#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify README.md's Crate Overview still lists every workspace member.
#
# The overview is the only map of the tree a newcomer gets, and a crate absent
# from it is invisible: nothing else in the repo enumerates the members for a
# human, and no build fails when one goes unlisted.  `dialect-iket` and
# `iket-lower` were both missing when this guard was written -- 1,700 lines of
# compiler across two crates, with no row between them.
#
# Direction matters.  Every member needs a row; extra rows are fine and
# expected.  `rustc-codegen-cuda` is deliberately not a workspace member (it
# needs its own [workspace] for the rustc_private dylibs) and is listed anyway,
# which is right -- readers care about the crate, not about which workspace
# resolves it.  So this checks for members with no row, never for rows with no
# member.
#
# Members are read from the root Cargo.toml rather than from `cargo metadata`:
# the question is which crates the workspace declares, which is exactly what
# that list says, and reading it needs no cargo, no lockfile and no network.
#
# Run this after adding or removing a workspace member.
set -euo pipefail

export LC_ALL=C

cd "$(dirname "$0")/.."

MANIFEST=Cargo.toml
README=README.md

if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to verify the crate inventory" >&2
    echo "       refusing to report success from a check that cannot run" >&2
    exit 1
fi

test -s "${MANIFEST}"
test -s "${README}"

python3 - "${MANIFEST}" "${README}" <<'PY'
import re
import sys

manifest_path, readme_path = sys.argv[1], sys.argv[2]


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


manifest = read(manifest_path)
readme = read(readme_path)

members_block = re.search(r"^members\s*=\s*\[(.*?)\]", manifest, re.M | re.S)
if not members_block:
    sys.exit(
        f"parse self-test failed: no `members = [...]` list in {manifest_path}; "
        "fix this script before trusting it"
    )

paths = re.findall(r'"([^"]+)"', members_block.group(1))

# A glob entry would silently name crates this list cannot resolve, so the
# guard has to say it went blind rather than pass over them.
globbed = [path for path in paths if "*" in path]
if globbed:
    sys.exit(
        "unsupported glob in workspace members: "
        + " ".join(globbed)
        + "; this guard resolves explicit paths only, so extend it before adding globs"
    )

# The directory name is the crate name everywhere in this tree, and the README
# tables key on the crate name.
members = sorted({path.rstrip("/").rsplit("/", 1)[-1] for path in paths})

if len(members) < 20:
    sys.exit(
        f"parse self-test failed: read {len(members)} members from {manifest_path}"
    )

# Only the Crate Overview section counts.  A crate named in passing elsewhere in
# the README -- a command line, the pipeline diagram, a prose aside -- is not an
# inventory row, and accepting one would let a crate stay unlisted forever.
overview = re.search(r"^## Crate Overview$(.*?)^## ", readme, re.M | re.S)
if not overview:
    sys.exit(
        f"parse self-test failed: no `## Crate Overview` section in {readme_path}; "
        "it was renamed or removed, so fix this script"
    )

# First column of every table row in that section.
listed = set(re.findall(r"^\|\s*`([^`]+)`\s*\|", overview.group(1), re.M))
if len(listed) < 20:
    sys.exit(
        f"parse self-test failed: read {len(listed)} crate rows from the "
        f"Crate Overview in {readme_path}"
    )

missing = [member for member in members if member not in listed]

if missing:
    print(
        f"error: {readme_path}'s Crate Overview has no row for:",
        file=sys.stderr,
    )
    for member in missing:
        print(f"  {member}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        "Every workspace member needs a row. Add it to the table that fits "
        "(User-Facing,\nCompiler, or Build Tooling) and copy the column layout "
        "from a neighbouring row.",
        file=sys.stderr,
    )
    sys.exit(1)

print(
    f"OK: {readme_path}'s Crate Overview lists all {len(members)} workspace members."
)
PY
