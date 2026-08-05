#!/usr/bin/env bash
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

required="$(cargo metadata --format-version 1 | python3 -c '
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
' | LC_ALL=C sort -u)"

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
