#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_file="${root}/src/main.rs"
llvm_ir="${1:-${root}/shared_debug.ll}"

test -s "${llvm_ir}"

definition_line() {
    grep -nF "$1" "${source_file}" | cut -d: -f1
}

die_at_marker() {
    local name="$1" marker="$2" line
    line="$(definition_line "${marker}")"
    grep -F "!DIGlobalVariable(name: \"${name}\"" "${llvm_ir}" | grep -F "line: ${line},"
}

linkage_name() {
    sed -E 's/.*linkageName: "([^"]+)".*/\1/'
}

require_retained_class8() {
    local die="$1" die_id expression expression_id cu globals_id tuple
    die_id="$(printf '%s\n' "${die}" | sed -E 's/^!([0-9]+) = .*/\1/')"
    expression="$(grep -F "!DIGlobalVariableExpression(var: !${die_id}, expr: !DIExpression(DW_OP_constu, 8, DW_OP_swap, DW_OP_xderef))" "${llvm_ir}")"
    [[ "$(printf '%s\n' "${expression}" | wc -l)" -eq 1 ]]
    expression_id="$(printf '%s\n' "${expression}" | sed -E 's/^!([0-9]+) = .*/\1/')"
    cu="$(grep -F '!DICompileUnit(' "${llvm_ir}")"
    globals_id="$(printf '%s\n' "${cu}" | sed -E 's/.*globals: !([0-9]+)\).*/\1/')"
    tuple="$(grep -E "^!${globals_id} = !\\{" "${llvm_ir}")"
    grep -Fq "!${expression_id}" <<<"${tuple}"
}

require_physical() {
    local die="$1" type="$2" align="$3" linkage
    linkage="$(printf '%s\n' "${die}" | linkage_name)"
    [[ "$(grep -Ec "^@${linkage} = addrspace\\(3\\) global ${type} .*align ${align}, !dbg !" "${llvm_ir}")" -eq 1 ]]
}

tile_die="$(die_at_marker TILE DEBUG_SHARED_TILE)"
other_die="$(die_at_marker OTHER DEBUG_SHARED_OTHER)"
barrier_die="$(die_at_marker BAR DEBUG_SHARED_BARRIER)"
same_left_die="$(die_at_marker SAME DEBUG_SHARED_SAME_LEFT)"
same_right_die="$(die_at_marker SAME DEBUG_SHARED_SAME_RIGHT)"
other_scope_die="$(die_at_marker TILE DEBUG_SHARED_OTHER_SCOPE)"

for die in "${tile_die}" "${other_die}" "${barrier_die}" "${same_left_die}" "${same_right_die}" "${other_scope_die}"; do
    require_retained_class8 "${die}"
done

require_physical "${tile_die}" '\[32 x i32\]' 4
require_physical "${other_die}" '\[8 x i16\]' 2
require_physical "${barrier_die}" '\[1 x i64\]' 8
require_physical "${same_left_die}" '\[2 x i16\]' 2
require_physical "${same_right_die}" '\[2 x i16\]' 2
require_physical "${other_scope_die}" '\[4 x i64\]' 8

tile_type="$(printf '%s\n' "${tile_die}" | sed -E 's/.*type: !([0-9]+).*/\1/')"
grep -Eq "^!${tile_type} = !DICompositeType\\(tag: DW_TAG_array_type, baseType: ![0-9]+, size: 1024, elements: ![0-9]+\\)" "${llvm_ir}"
grep -Fq '!DISubrange(count: 32)' "${llvm_ir}"

barrier_type="$(printf '%s\n' "${barrier_die}" | sed -E 's/.*type: !([0-9]+).*/\1/')"
grep -Eq "^!${barrier_type} = !DICompositeType\\(tag: DW_TAG_structure_type, name: \"Barrier\", size: 64," "${llvm_ir}"

tile_scope="$(printf '%s\n' "${tile_die}" | sed -E 's/.*scope: !([0-9]+).*/\1/')"
other_scope="$(printf '%s\n' "${other_scope_die}" | sed -E 's/.*scope: !([0-9]+).*/\1/')"
same_left_scope="$(printf '%s\n' "${same_left_die}" | sed -E 's/.*scope: !([0-9]+).*/\1/')"
same_right_scope="$(printf '%s\n' "${same_right_die}" | sed -E 's/.*scope: !([0-9]+).*/\1/')"
grep -Eq "^!${tile_scope} = distinct !DISubprogram\\(name: \"shared_debug\", linkageName: \"[^\"]+\", scope: ![0-9]+," "${llvm_ir}"
grep -Eq "^!${other_scope} = distinct !DISubprogram\\(name: \"other_scope\", linkageName: \"[^\"]+\", scope: ![0-9]+," "${llvm_ir}"
grep -Eq "^!${same_left_scope} = distinct !DISubprogram\\(name: \"same_leaf\", linkageName: \"[^\"]+\", scope: ![0-9]+," "${llvm_ir}"
grep -Fq '!DINamespace(name: "shared_debug", scope: null)' "${llvm_ir}"
grep -Fq '!DINamespace(name: "kernels", scope: !' "${llvm_ir}"
[[ "$(grep -Ec 'distinct !DISubprogram\(name: "shared_debug"' "${llvm_ir}")" -eq 1 ]]
[[ "$(grep -Ec 'distinct !DISubprogram\(name: "other_scope"' "${llvm_ir}")" -eq 1 ]]
[[ "$(grep -Ec 'distinct !DISubprogram\(name: "same_leaf"' "${llvm_ir}")" -eq 1 ]]
[[ "${other_scope}" != "${tile_scope}" ]]
[[ "${same_left_scope}" == "${same_right_scope}" ]]
[[ "${same_left_scope}" != "${tile_scope}" ]]
grep -Eq "^define .*@shared_debug\\(.* !dbg !${tile_scope} \\{$" "${llvm_ir}"
grep -Eq "^define .*@other_scope\\(.* !dbg !${other_scope} \\{$" "${llvm_ir}"
grep -Eq "^define .*@same_leaf\\(.* !dbg !${same_left_scope} \\{$" "${llvm_ir}"

left_linkage="$(printf '%s\n' "${same_left_die}" | linkage_name)"
right_linkage="$(printf '%s\n' "${same_right_die}" | linkage_name)"
[[ "${left_linkage}" != "${right_linkage}" ]]
[[ "$(grep -Fc '!DIGlobalVariableExpression(var:' "${llvm_ir}")" -eq 6 ]]

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

if command -v llvm-as >/dev/null 2>&1; then
    llvm-as -o "${tmpdir}/shared-debug.bc" "${llvm_ir}"
    if command -v opt >/dev/null 2>&1; then
        opt -passes=verify -disable-output "${tmpdir}/shared-debug.bc"
    fi
fi

# Validate the supported LLVM floor and current LLVM independently. PTXAS's
# final cubin is authoritative for the NVPTX address-class translation.
for version in 21 22; do
    llvm_as="llvm-as-${version}"
    opt="opt-${version}"
    llc="llc-${version}"
    dwarfdump="llvm-dwarfdump-${version}"
    if ! command -v "${llvm_as}" >/dev/null 2>&1 \
        || ! command -v "${opt}" >/dev/null 2>&1 \
        || ! command -v "${llc}" >/dev/null 2>&1 \
        || ! command -v "${dwarfdump}" >/dev/null 2>&1 \
        || ! command -v ptxas >/dev/null 2>&1; then
        continue
    fi

    bitcode="${tmpdir}/shared-debug-${version}.bc"
    ptx="${tmpdir}/shared-debug-${version}.ptx"
    cubin="${tmpdir}/shared-debug-${version}.cubin"
    dwarf="${tmpdir}/shared-debug-${version}.dwarf"
    "${llvm_as}" -o "${bitcode}" "${llvm_ir}"
    "${opt}" -passes=verify -disable-output "${bitcode}"
    "${llc}" -march=nvptx64 -mcpu=sm_90 -mattr=+ptx80 -O0 \
        -filetype=asm "${bitcode}" -o "${ptx}"
    ptxas -arch=sm_90 -g "${ptx}" -o "${cubin}"
    "${dwarfdump}" --debug-info "${cubin}" >"${dwarf}"
    [[ "$(grep -c 'DW_AT_address_class.*0x08' "${dwarf}")" -eq 6 ]]
done

echo "shared-static AS3 debug-info shape verified"
