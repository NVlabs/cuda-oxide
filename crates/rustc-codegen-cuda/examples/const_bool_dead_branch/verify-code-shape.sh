#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
artifacts=(
    "${root}/const_bool_dead_branch.ll"
    "${root}/const_bool_dead_branch.ptx"
)
if [[ -s "${root}/const_bool_dead_branch.opt.ll" ]]; then
    artifacts+=("${root}/const_bool_dead_branch.opt.ll")
fi

for artifact in "${artifacts[@]}"; do
    test -s "${artifact}"
    grep -q 'ExplicitOn.*ExplicitMode.*hook' "${artifact}"
    grep -q 'DefaultOn.*DefaultMode.*hook' "${artifact}"
    if grep -q 'ExplicitOff.*ExplicitMode.*hook' "${artifact}"; then
        echo "dead explicit hook found in ${artifact}" >&2
        exit 1
    fi
    if grep -q 'DefaultOff.*DefaultMode.*hook' "${artifact}"; then
        echo "dead default hook found in ${artifact}" >&2
        exit 1
    fi
done

echo "const_bool_dead_branch code shape: PASS"
