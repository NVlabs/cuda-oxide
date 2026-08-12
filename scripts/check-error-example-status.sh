#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Verify every error* example is documented in STATUS.md and listed in
# the ERROR_EXAMPLES array in smoketest.sh.  Run this after adding or
# removing an error* example.
set -euo pipefail

cd "$(dirname "$0")/.."

fail=0

# `mapfile` is a bash 4 builtin and `grep -P` is a GNU extension; macOS ships
# neither (bash 3.2, BSD grep), so this guard aborted there before it checked
# anything. Collect with a read loop and extract with `sed` instead -- both
# portable, and the extracted values are unchanged.
collect() {
    COLLECTED=()
    local line
    while IFS= read -r line; do
        COLLECTED+=("$line")
    done
}

# Examples that exist on disk.
collect < <(
    find crates/rustc-codegen-cuda/examples -mindepth 1 -maxdepth 1 \
        -type d -name 'error*' -exec basename {} \; | sort
)
on_disk=("${COLLECTED[@]+"${COLLECTED[@]}"}")

# Examples listed in STATUS.md (backtick-quoted names in the table).
collect < <(
    sed -n 's/^|[[:space:]]*`\([^`]*\)`.*/\1/p' \
        crates/rustc-codegen-cuda/STATUS.md | sort
)
in_status=("${COLLECTED[@]+"${COLLECTED[@]}"}")

# Examples listed in ERROR_EXAMPLES in smoketest.sh.
collect < <(
    sed -n 's/.*ERROR_EXAMPLES=(\([^)]*\)).*/\1/p' scripts/smoketest.sh \
        | tr ' ' '\n' | grep -v '^$' | sort
)
in_smoketest=("${COLLECTED[@]+"${COLLECTED[@]}"}")

contains() {
    local needle="$1"; shift
    printf '%s\n' "$@" | grep -qx "$needle"
}

for ex in "${on_disk[@]}"; do
    if ! contains "$ex" "${in_status[@]+"${in_status[@]}"}"; then
        echo "error: $ex is not in STATUS.md" >&2; fail=1
    fi
    if ! contains "$ex" "${in_smoketest[@]+"${in_smoketest[@]}"}"; then
        echo "error: $ex is not in ERROR_EXAMPLES in smoketest.sh" >&2; fail=1
    fi
done

for ex in "${in_status[@]}"; do
    if [[ ! -d "crates/rustc-codegen-cuda/examples/$ex" ]]; then
        echo "error: STATUS.md lists '$ex' but no such directory exists" >&2; fail=1
    fi
done

[[ $fail -eq 0 ]] && echo "OK: all error* examples are documented and classified."
exit $fail
