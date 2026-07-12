#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
llvm_ir="${root}/array_constants.ll"

test -s "${llvm_ir}"

# Direct padded tuple: the u32 value must populate field 1, not padding after
# the leading u8.
grep -Eq 'insertvalue \{ i8, i32 \} .* i32 41, 1' "${llvm_ir}"

# Nested tuple with a zero-sized field: the u32 remains outer field 1.
grep -Eq 'insertvalue \{ \{ i8 \}, i32 \} .* i32 17, 1' "${llvm_ir}"

# Padded tuple array: the repr(u32) enum occupies tuple field 1 after a bool.
grep -Eq 'insertvalue \{ i1, \{ i32 \} \} .* \{ i32 \} .* 1' "${llvm_ir}"

echo "array_constants code shape: PASS"
