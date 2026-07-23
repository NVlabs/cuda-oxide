#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llvm_ir="${root}/array_constants.ll"
optimized_llvm_ir="${root}/array_constants.opt.ll"
ptx="${root}/array_constants.ptx"

test -s "${llvm_ir}"
test -s "${optimized_llvm_ir}"
test -s "${ptx}"

require_shape() {
    local description="$1"
    local pattern="$2"
    if ! grep -Eq "${pattern}" "${llvm_ir}"; then
        echo "error: missing ${description} in ${llvm_ir}" >&2
        exit 1
    fi
}

symbol_body() {
    local artifact="$1"
    local format="$2"
    local symbol="$3"

    if [[ "${format}" == "llvm" ]]; then
        awk -v marker="${symbol}(" '
            !emit && index($0, marker) && $1 == "define" { emit = 1 }
            emit { print }
            emit && $0 == "}" { exit }
        ' "${artifact}"
    else
        # Unoptimized PTX includes forward declarations. Wait for the matching
        # header to reach `{`; a prototype reaches `;` and is skipped.
        awk -v marker="${symbol}(" '
            !emit && !candidate && index($0, marker) &&
                (index($0, ".func") || index($0, ".entry")) {
                candidate = 1
            }
            candidate && $0 ~ /^[[:space:]]*;[[:space:]]*$/ {
                candidate = 0
                next
            }
            candidate && $0 ~ /^[[:space:]]*\{[[:space:]]*$/ {
                emit = 1
                candidate = 0
            }
            emit { print }
            emit && index($0, "End function") != 0 { exit }
        ' "${artifact}"
    fi
}

require_symbol_shape() {
    local artifact="$1"
    local format="$2"
    local symbol="$3"
    local description="$4"
    local pattern="$5"

    if ! symbol_body "${artifact}" "${format}" "${symbol}" |
        grep -E "${pattern}" >/dev/null; then
        echo "error: missing ${description} in ${artifact}:${symbol}" >&2
        exit 1
    fi
}

reject_symbol_shape() {
    local artifact="$1"
    local format="$2"
    local symbol="$3"
    local description="$4"
    local pattern="$5"

    if symbol_body "${artifact}" "${format}" "${symbol}" |
        grep -E "${pattern}" >/dev/null; then
        echo "error: found ${description} in ${artifact}:${symbol}" >&2
        exit 1
    fi
}

# Explicit padding slots come from rustc layout metadata added to mir.tuple.
# These assertions deliberately name both the constant and its physical LLVM
# slot so a declaration-order or packed-byte regression cannot pass silently.

# Direct padded tuple: the u32 follows an explicit three-byte padding slot.
require_shape \
    "direct padded tuple value in LLVM slot 2" \
    'insertvalue \{ i8, \[3 x i8\], i32 \} .* i32 41, 2'

# Nested tuple with a zero-sized field: the ZST is stripped, but padding and
# the outer u32's physical slot remain layout-exact.
require_shape \
    "nested tuple value after explicit padding" \
    'insertvalue \{ \{ i8 \}, \[3 x i8\], i32 \} .* i32 17, 2'

# Padded tuple array: the repr(u32) enum follows a bool and three pad bytes.
require_shape \
    "padded tuple-array enum value in LLVM slot 2" \
    'insertvalue \{ i1, \[3 x i8\], \{ i32 \} \} .* \{ i32 \} .* 2'

# A non-empty tuple made entirely of ZST fields must still be decoded by the
# tuple path. Its stripped LLVM representation leaves the outer u32 intact.
require_shape \
    "all-ZST nested tuple's following value" \
    'insertvalue \{ i32 \} undef, i32 59, 0'
require_shape \
    "all-ZST tuple array" \
    'insertvalue \[2 x \{ i32 \}\] .* \{ i32 \} .* 1'

# rustc lays `(u8, u32, u64)` out at byte offsets 4, 0, and 8. The lowered
# LLVM tuple is therefore `{ i32, i8, [3 x i8], i64 }`; each declaration-order
# constant must land in its mapped physical slot.
require_shape \
    "reordered tuple u8 in LLVM slot 1" \
    'insertvalue \{ i32, i8, \[3 x i8\], i64 \} undef, i8 165, 1'
require_shape \
    "reordered tuple u32 in LLVM slot 0" \
    'insertvalue \{ i32, i8, \[3 x i8\], i64 \} .* i32 287454020, 0'
require_shape \
    "reordered tuple u64 in LLVM slot 3" \
    'insertvalue \{ i32, i8, \[3 x i8\], i64 \} .* i64 72623859790382856, 3'
require_shape \
    "reordered tuple array stride" \
    'insertvalue \[2 x \{ i32, i8, \[3 x i8\], i64 \}\] .* 1'

# `(Align32, u8)` has Rust ABI alignment 32 even though its lowered LLVM
# struct contains only an i8 plus byte padding and therefore looks align-1 to
# LLVM. Pin every memory operation in the unoptimized pipeline: `%pair` is a
# surviving MirAllocaOp, while the array alloca/store/element load are the
# synthetic spill used for a dynamic array index.
overaligned_symbol='array_constants__kernels__overaligned_zst_tuple_array_value'
overaligned_tuple='\{ i8, \[31 x i8\] \}'
overaligned_array="\\[2 x ${overaligned_tuple}\\]"

require_symbol_shape "${llvm_ir}" llvm "${overaligned_symbol}" \
    "align-32 tuple local alloca in unoptimized LLVM" \
    "alloca ${overaligned_tuple}, align 32"
require_symbol_shape "${llvm_ir}" llvm "${overaligned_symbol}" \
    "align-32 dynamic array spill alloca in unoptimized LLVM" \
    "alloca ${overaligned_array}, align 32"
require_symbol_shape "${llvm_ir}" llvm "${overaligned_symbol}" \
    "align-32 dynamic array spill store in unoptimized LLVM" \
    "store ${overaligned_array} .* align 32"
require_symbol_shape "${llvm_ir}" llvm "${overaligned_symbol}" \
    "align-32 dynamic array element load in unoptimized LLVM" \
    "load ${overaligned_tuple}, .* align 32"
require_symbol_shape "${llvm_ir}" llvm "${overaligned_symbol}" \
    "align-32 tuple local store in unoptimized LLVM" \
    "store ${overaligned_tuple} .* align 32"
reject_symbol_shape "${llvm_ir}" llvm "${overaligned_symbol}" \
    "under-aligned memory operation in unoptimized LLVM" \
    '(^|, )align 1($|[^0-9])'

# Optimization may scalarize the aggregate stores and removes the address
# low-bit computation once the alloca is provably aligned, but the surviving
# dynamic spill must remain align 32 throughout. Where it survives depends on
# the middle-end: with plain -O2 the external helper is kept and owns the
# spill, while internalizing non-root helpers lets `opt` inline it into the
# kernel and delete the definition. Assert on whichever function actually
# carries the code in each artifact.
surviving_symbol() {
    local artifact="$1"
    if grep -q "${overaligned_symbol}" "${artifact}"; then
        echo "${overaligned_symbol}"
    else
        echo 'check_array_constants'
    fi
}
optimized_symbol="$(surviving_symbol "${optimized_llvm_ir}")"
ptx_symbol="$(surviving_symbol "${ptx}")"

require_symbol_shape "${optimized_llvm_ir}" llvm "${optimized_symbol}" \
    "align-32 dynamic array spill alloca in optimized LLVM" \
    "alloca ${overaligned_array}, align 32"
require_symbol_shape "${optimized_llvm_ir}" llvm "${optimized_symbol}" \
    "align-32 scalarized store in optimized LLVM" \
    'store i8 18, .* align 32'
require_symbol_shape "${optimized_llvm_ir}" llvm "${optimized_symbol}" \
    "align-32 scalarized load in optimized LLVM" \
    'load i8, .* align 32'
reject_symbol_shape "${optimized_llvm_ir}" llvm "${optimized_symbol}" \
    "under-aligned memory operation in optimized LLVM" \
    '(^|, )align 1($|[^0-9])'

require_symbol_shape "${ptx}" ptx "${ptx_symbol}" \
    "32-byte-aligned PTX local depot" \
    '\.local \.align 32 \.b8'
reject_symbol_shape "${ptx}" ptx "${ptx_symbol}" \
    "align-1 PTX local depot" \
    '\.local \.align 1 \.b8'

echo "array_constants code shape: PASS"
