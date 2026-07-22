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
    grep -q 'Dynamic.*DynamicMode.*hook' "${artifact}"
    if grep -q 'ExplicitOff.*ExplicitMode.*hook' "${artifact}"; then
        echo "dead explicit hook found in ${artifact}" >&2
        exit 1
    fi
    if grep -q 'DefaultOff.*DefaultMode.*hook' "${artifact}"; then
        echo "dead default hook found in ${artifact}" >&2
        exit 1
    fi
done

extract_llvm_function() {
    local pattern="$1"
    awk -v pattern="${pattern}" '
        /^define / && $0 ~ pattern { in_function = 1 }
        in_function { print }
        in_function && /^}/ { exit }
    ' "${root}/const_bool_dead_branch.ll"
}

for pattern in \
    'select_default.*DefaultOn' \
    'select_default.*DefaultOff' \
    'select_explicit.*ExplicitOn' \
    'select_explicit.*ExplicitOff'; do
    body="$(extract_llvm_function "${pattern}")"
    test -n "${body}"
    if grep -q 'br i1' <<<"${body}"; then
        echo "const-selected function still contains a conditional branch: ${pattern}" >&2
        exit 1
    fi
done

dynamic_body="$(extract_llvm_function 'select_dynamic.*Dynamic')"
test -n "${dynamic_body}"
if [[ "$(grep -c 'br i1' <<<"${dynamic_body}")" -ne 1 ]]; then
    echo "dynamic control must retain exactly one conditional branch" >&2
    exit 1
fi

echo "const_bool_dead_branch code shape: PASS"
