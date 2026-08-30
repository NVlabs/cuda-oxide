#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build first with full device debug information:
#
#   cargo oxide build debug --device-debug
#
# This pins the kernel-entry line-table contract in both emitted LLVM IR and
# PTX. Source lines come from a same-line marker, never an assumed offset from
# `#[kernel]`; the fixture deliberately has documentation plus multiple
# attributes before the function.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="${root}/src/main.rs"
llvm_ir="${root}/debug.ll"
ptx="${root}/debug.ptx"

test -s "${llvm_ir}"
test -s "${ptx}"

entry_line="$(grep -nF 'CUDA_OXIDE_DEBUG_KERNEL_ENTRY_LINE' "${source_file}" | cut -d: -f1)"
kernel_attr_line="$(grep -nF 'CUDA_OXIDE_DEBUG_KERNEL_ATTRIBUTE_LINE' "${source_file}" | cut -d: -f1)"
launch_bounds_attr_line="$(grep -nF 'CUDA_OXIDE_DEBUG_LAUNCH_BOUNDS_ATTRIBUTE_LINE' "${source_file}" | cut -d: -f1)"
test -n "${entry_line}"
test -n "${kernel_attr_line}"
test -n "${launch_bounds_attr_line}"

llvm_body="$(awk '
    /^define ptx_kernel void @clock_test\(/ { in_function = 1 }
    in_function { print }
    in_function && /^}/ { exit }
' "${llvm_ir}")"
test -n "${llvm_body}"

scope_debug_id="$(
    grep -m1 'call void @.*make_kernel_scope' <<<"${llvm_body}" |
        sed -nE 's/.*!dbg !([0-9]+).*/\1/p'
)"
index_debug_id="$(
    grep -m1 'call i64 @.*index_1d' <<<"${llvm_body}" |
        sed -nE 's/.*!dbg !([0-9]+).*/\1/p'
)"
test -n "${scope_debug_id}"
test -n "${index_debug_id}"

if ! grep -Eq "^!${scope_debug_id} = !DILocation\\(line: 0, column: 0," "${llvm_ir}"; then
    echo "error: generated kernel-scope call does not use an artificial LLVM location" >&2
    exit 1
fi
if ! grep -Eq "^!${index_debug_id} = !DILocation\\(line: ${entry_line}, column:" "${llvm_ir}"; then
    echo "error: rewritten index call lost source line ${entry_line} in LLVM IR" >&2
    exit 1
fi

ptx_body="$(awk '
    !emit && !candidate && /[.]entry[[:space:]]+clock_test\(/ { candidate = 1 }
    candidate && /^[[:space:]]*;[[:space:]]*$/ {
        candidate = 0
        next
    }
    candidate && /^[[:space:]]*\{[[:space:]]*$/ {
        emit = 1
        candidate = 0
    }
    emit { print }
    emit && /End function/ { exit }
' "${ptx}")"
test -n "${ptx_body}"

ptx_prologue="$(awk '
    { print }
    /call[.]uni .*index_1d/ { exit }
' <<<"${ptx_body}")"
if grep -Eq "^[[:space:]]*[.]loc[[:space:]]+[0-9]+[[:space:]]+(${kernel_attr_line}|${launch_bounds_attr_line})([[:space:]]|$)" <<<"${ptx_prologue}"; then
    echo "error: synthetic kernel prologue still maps to a macro attribute" >&2
    exit 1
fi

if ! awk -v expected_index_line="${entry_line}" '
    $1 == ".loc" { current_line = $3 }
    /call[.]uni .*make_kernel_scope/ {
        saw_scope = 1
        scope_is_artificial = (current_line == 0)
    }
    /call[.]uni .*index_1d/ {
        saw_index = 1
        index_is_user_line = (current_line == expected_index_line)
    }
    END {
        exit !(saw_scope && scope_is_artificial && saw_index && index_is_user_line)
    }
' <<<"${ptx_body}"; then
    echo "error: PTX kernel prologue/index .loc state does not match the source contract" >&2
    exit 1
fi

echo "debug kernel-entry lineinfo shape: PASS"
