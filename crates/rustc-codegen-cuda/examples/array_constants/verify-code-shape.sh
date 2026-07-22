#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llvm_ir="${root}/array_constants.ll"

test -s "${llvm_ir}"

require_shape() {
    local description="$1"
    local pattern="$2"
    if ! grep -Eq "${pattern}" "${llvm_ir}"; then
        echo "error: missing ${description} in ${llvm_ir}" >&2
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

echo "array_constants code shape: PASS"
