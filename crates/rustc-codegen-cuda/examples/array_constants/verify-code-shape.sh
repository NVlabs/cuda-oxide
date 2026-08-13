#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Bare struct arrays must use the same rustc-recorded field offsets as direct
# struct constants. `PaddedStruct` is `#[repr(C)] { u8, u32 }`, so the lowered
# element contains an explicit three-byte padding slot before the u32.
padded_struct_symbol='array_constants__kernels__padded_struct_array_value'
require_symbol_shape "${llvm_ir}" llvm "${padded_struct_symbol}" \
    "padded bare-struct array storage" \
    'alloca \[2 x \{ i8, \[3 x i8\], i32 \}\]'

# Recursive aggregate decoding must preserve the inner struct's padding while
# placing the following u64 at its `#[repr(C)]` offset.
nested_struct_symbol='array_constants__kernels__nested_struct_array_value'
require_symbol_shape "${llvm_ir}" llvm "${nested_struct_symbol}" \
    "nested bare-struct array storage" \
    'alloca \[2 x \{ \{ i8, \[3 x i8\], i32 \}, i64 \}\]'

# The standalone over-aligned ZST struct array has no data bytes to pin in LLVM
# IR. Runtime coverage in `array_constants` checks that indexing it still
# materializes a correctly aligned value.

# Nested tuple with a zero-sized field: the ZST is stripped, but padding and
# the outer u32's physical slot remain layout-exact. The array form is now
# materialized from rustc's evaluated allocation, so the values are pinned as the
# byte image -- `((3, ()), 17), ((5, ()), 29)` is the u8 at offset 0, three pad
# bytes, then the u32 at offset 4 -- alongside the lowered element type that says
# where each of those bytes is read from.
require_shape \
    "nested tuple array byte image" \
    'addrspace\(1\) constant \[16 x i8\] c"\\03\\00\\00\\00\\11\\00\\00\\00\\05\\00\\00\\00\\1D\\00\\00\\00"'
require_shape \
    "nested tuple array element type after explicit padding" \
    'alloca \[2 x \{ \{ i8 \}, \[3 x i8\], i32 \}\]'

# Padded tuple array: the repr(u32) enum follows a bool and three pad bytes.
# `(false, LowX), (true, HighX), ...` is a zero/one byte, three pad bytes, then
# the discriminant 1..6 at offset 4 of each eight-byte element.
require_shape \
    "padded tuple-array byte image" \
    'addrspace\(1\) constant \[48 x i8\] c"\\00\\00\\00\\00\\01\\00\\00\\00\\01\\00\\00\\00\\02\\00\\00\\00'
require_shape \
    "padded tuple-array element type" \
    'alloca \[6 x \{ i1, \[3 x i8\], \{ i32 \} \}\]'

bare_enum_symbol='array_constants__kernels__bare_enum_array_value'

# A bare enum array must be materialized by the importer and indexed
# dynamically. This distinguishes it from the already-covered enum nested in a
# tuple array and prevents optimization-only success from hiding importer
# coverage.
#
# It is materialized as one immutable device global holding rustc's evaluated
# allocation, copied into the array's own storage, so the discriminants are
# asserted in that initializer rather than as a chain of `insertvalue`. Pinning
# the whole image in a single pattern also fixes their *order*, which the
# previous per-discriminant search did not: `BARE_ENUM_TABLE` is `LowX, HighZ,
# HighY, HighX, LowZ, LowY`, so 1, 6, 4, 2, 5, 3 little-endian at four bytes
# each. A permuted or truncated table now fails where before it passed.
require_shape \
    "bare enum-array discriminants in a read-only global initializer" \
    'addrspace\(1\) constant \[24 x i8\] c"\\01\\00\\00\\00\\06\\00\\00\\00\\04\\00\\00\\00\\02\\00\\00\\00\\05\\00\\00\\00\\03\\00\\00\\00"'
require_symbol_shape "${llvm_ir}" llvm "${bare_enum_symbol}" \
    "six-element direct-tag enum array storage" \
    'alloca \[6 x \{ i32 \}\]'
require_symbol_shape "${llvm_ir}" llvm "${bare_enum_symbol}" \
    "enum array filled from a read-only global" \
    'llvm\.memcpy\.p0\.p1\.i64\(ptr %[A-Za-z0-9_.]+, ptr addrspace\(1\) @'
require_symbol_shape "${llvm_ir}" llvm "${bare_enum_symbol}" \
    "runtime enum-array index" \
    'urem i64 .*, 6'
require_symbol_shape "${llvm_ir}" llvm "${bare_enum_symbol}" \
    "runtime enum-array element load" \
    'load \{ i32 \},'

union_array_symbol='array_constants__kernels__union_array_value'
direct_union_symbol='array_constants__kernels__direct_union_value'
union_tuple_symbol='array_constants__kernels__union_tuple_value'
union_struct_symbol='array_constants__kernels__union_struct_value'
partial_union_symbol='array_constants__kernels__partial_union_value'
maybe_uninit_symbol='array_constants__kernels__maybe_uninit_array_value'
union_storage='\{ \[0 x i32\], i32 \}'

# Initialized union constants are deliberately materialized element-wise instead
# of taking the promoted-global fast path. rustc gives the importer a byte image
# plus an initialization mask, and inactive union bytes must remain `undef`
# rather than being zero-filled by a global initializer.
require_symbol_shape "${llvm_ir}" llvm "${union_array_symbol}" \
    "four-element initialized-union array storage" \
    "alloca \\[4 x ${union_storage}\\]"
require_symbol_shape "${llvm_ir}" llvm "${union_array_symbol}" \
    "runtime initialized-union array index" \
    'and i64 .*, 3'
require_symbol_shape "${llvm_ir}" llvm "${union_array_symbol}" \
    "runtime initialized-union element load" \
    "load ${union_storage},"

# Direct, tuple-nested, struct-nested, and partially initialized constants all
# cross the same byte-faithful transmute boundary. A four-byte union therefore
# needs a temporary `[4 x i8]` storage image before it becomes union storage.
for symbol in \
    "${direct_union_symbol}" \
    "${union_tuple_symbol}" \
    "${union_struct_symbol}" \
    "${partial_union_symbol}"; do
    require_symbol_shape "${llvm_ir}" llvm "${symbol}" \
        "byte-faithful initialized-union materialization" \
        'alloca \[4 x i8\]'
done

# `MaybeUninit<u32>` is itself a union. Runtime indexing prevents rustc from
# folding the table to one scalar and exercises the same initialized-union
# constant path through the core-library type.
require_symbol_shape "${llvm_ir}" llvm "${maybe_uninit_symbol}" \
    "runtime MaybeUninit constant index" \
    'and i64 .*, 1'
require_symbol_shape "${llvm_ir}" llvm "${maybe_uninit_symbol}" \
    "two-element MaybeUninit array storage" \
    'alloca \[2 x \{'

pointer_tuple_symbol='array_constants__kernels__pointer_tuple_array_value'

# Device globals use backend-generated internal symbols, so source-level static
# names are not stable in LLVM IR. Runtime coverage checks the zero-addend
# whole-static case. This assertion pins the non-zero interior-static
# projection and ensures provenance is not reconstructed from placeholder bytes.
require_symbol_shape "${llvm_ir}" llvm "${pointer_tuple_symbol}" \
    "eight-byte device-static subobject projection" \
    'getelementptr( inbounds)? i8,.*i64 8([^0-9]|$)'

reject_symbol_shape "${llvm_ir}" llvm "${pointer_tuple_symbol}" \
    "placeholder-byte inttoptr reconstruction" \
    'inttoptr'

# A non-empty tuple made entirely of ZST fields must still be decoded by the
# tuple path. Its stripped LLVM representation leaves the outer u32 intact.
require_shape \
    "all-ZST tuple array byte image" \
    'addrspace\(1\) constant \[8 x i8\] c"\\3B\\00\\00\\00\\3D\\00\\00\\00"'
require_shape \
    "all-ZST tuple array element type" \
    'alloca \[2 x \{ i32 \}\]'

# rustc lays `(u8, u32, u64)` out at byte offsets 4, 0, and 8. The lowered
# LLVM tuple is therefore `{ i32, i8, [3 x i8], i64 }`; each declaration-order
# constant must land in its mapped physical slot.
# One pattern now pins all three fields, their byte offsets, the padding between
# them and both elements' stride at once, which the four separate slot searches
# did not: element 0 is u32 0x11223344 at offset 0, u8 0xa5 at 4, three pad
# bytes, then u64 0x0102030405060708 at 8. A field that moved, a padding byte
# that carried data, or a swapped element all fail here.
require_shape \
    "reordered tuple array byte image" \
    'addrspace\(1\) constant \[32 x i8\] c"\\44\\33\\22\\11\\A5\\00\\00\\00\\08\\07\\06\\05\\04\\03\\02\\01\\CC\\BB\\AA\\99\\5A\\00\\00\\00\\11\\22\\33\\44\\55\\66\\77\\88"'
require_shape \
    "reordered tuple array element type" \
    'alloca \[2 x \{ i32, i8, \[3 x i8\], i64 \}\]'

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
