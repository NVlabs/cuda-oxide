#!/usr/bin/env bash
# Verify every error* example is documented in STATUS.md and listed in
# the ERROR_EXAMPLES array in smoketest.sh.  Run this after adding or
# removing an error* example.
set -euo pipefail

cd "$(dirname "$0")/.."

fail=0

# Examples that exist on disk.
mapfile -t on_disk < <(
    find crates/rustc-codegen-cuda/examples -mindepth 1 -maxdepth 1 \
        -type d -name 'error*' -exec basename {} \; | sort
)

# Examples listed in STATUS.md (backtick-quoted names in the table).
mapfile -t in_status < <(
    grep -oP '^\|\s*`\K[^`]+' \
        crates/rustc-codegen-cuda/STATUS.md | sort
)

# Examples listed in ERROR_EXAMPLES in smoketest.sh.
mapfile -t in_smoketest < <(
    grep -oP 'ERROR_EXAMPLES=\(\K[^)]+' scripts/smoketest.sh \
        | tr ' ' '\n' | grep -v '^$' | sort
)

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
