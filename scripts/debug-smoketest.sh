#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# scripts/debug-smoketest.sh -- end-to-end cuda-gdb validation of device
# debug info (CUDA_OXIDE_DEBUG=full).
#
# Builds an example with full device debug info and drives cuda-gdb in batch
# mode to prove that source debugging actually works on a real GPU: a source
# breakpoint binds, the backtrace is correct, and `info args` / `info locals`
# show real values (scalars, pointers, structs, and Rust enums), not just
# metadata in the emitted IR.
#
# For compiler_features, dedicated debugger passes additionally validate
# optimized-MIR fragments, closure environments, Rust enums, static projections,
# enum projections, and dereference projections. The fragment pass requires a
# scalarized tuple to reach LLVM as multiple value-based `DW_OP_LLVM_fragment`
# records, rejects fragment `dbg.declare`, and then checks that cuda-gdb
# reconstructs both fields. The closure pass separately requires optimized
# direct-field captures that mix constant and local backing to use one
# whole-variable debug mirror before verifying both captures in cuda-gdb.
# The remaining passes cover direct/niche enums, static fields/constant indices,
# Downcast -> Field payloads, and thin-reference dereferences.
#
# This complements scripts/smoketest.sh (which validates the compile pipeline)
# by validating debugger *consumption* of the DWARF we emit.
#
# Gating: requires both cuda-gdb and a working NVIDIA GPU. When either is
# missing the script prints a skip notice and exits 0, so CI without a GPU is
# unaffected.
#
# Usage:
#   scripts/debug-smoketest.sh            # default example (compiler_features)
#   scripts/debug-smoketest.sh vecadd     # a specific example
#   CUDA_OXIDE_TARGET=sm_90 scripts/debug-smoketest.sh   # pin the arch

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXAMPLE="${1:-compiler_features}"
EXAMPLE_DIR="$REPO_ROOT/crates/rustc-codegen-cuda/examples/$EXAMPLE"

CUDA_GDB="${CUDA_OXIDE_CUDA_GDB:-$(command -v cuda-gdb || echo /usr/local/cuda/bin/cuda-gdb)}"

skip() {
    echo "debug-smoketest: SKIP ($1)"
    exit 0
}

[ -x "$CUDA_GDB" ] || skip "cuda-gdb not found (set CUDA_OXIDE_CUDA_GDB)"
command -v nvidia-smi >/dev/null 2>&1 || skip "nvidia-smi not found"
nvidia-smi -L >/dev/null 2>&1 || skip "no usable NVIDIA GPU / driver"
[ -d "$EXAMPLE_DIR" ] || { echo "debug-smoketest: FAIL (no example '$EXAMPLE')"; exit 1; }

# Resolve the device arch: explicit override wins, else the local GPU's cc.
if [ -n "${CUDA_OXIDE_TARGET:-}" ]; then
    ARCH="$CUDA_OXIDE_TARGET"
else
    CC="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
    [ -n "$CC" ] || skip "could not read compute capability"
    ARCH="sm_${CC}"
fi

echo "debug-smoketest: example=$EXAMPLE arch=$ARCH"

# Build with full device debug info.
( cd "$REPO_ROOT" && CUDA_OXIDE_DEBUG=full CUDA_OXIDE_TARGET="$ARCH" \
    cargo oxide build "$EXAMPLE" ) || { echo "debug-smoketest: FAIL (build)"; exit 1; }

BIN="$EXAMPLE_DIR/target/release/$EXAMPLE"
[ -x "$BIN" ] || { echo "debug-smoketest: FAIL (no binary at $BIN)"; exit 1; }

# Full debug deliberately leaves rustc MIR optimization at its normal level.
# `debug_fragment_pair` lives in a standalone `#[inline(never)]` helper whose
# body mirrors the importer SROA regression. It must reach LLVM as at least two
# fragments of one source variable before cuda-gdb is asked to consume it.
if [ "$EXAMPLE" = "compiler_features" ]; then
    LLVM_IR="$EXAMPLE_DIR/compiler_features.ll"
    [ -f "$LLVM_IR" ] || { echo "debug-smoketest: FAIL (no LLVM IR at $LLVM_IR)"; exit 1; }

    FRAGMENT_VAR_ID="$(sed -n 's/^!\([0-9][0-9]*\) = !DILocalVariable(name: "debug_fragment_pair".*/\1/p' "$LLVM_IR" | head -1)"
    [ -n "$FRAGMENT_VAR_ID" ] || { echo "debug-smoketest: FAIL (debug_fragment_pair DILocalVariable not found)"; exit 1; }

    FRAGMENT_VALUE_COUNT="$(grep -Ec "@llvm.dbg.value.*metadata !$FRAGMENT_VAR_ID, metadata !DIExpression\\(DW_OP_LLVM_fragment" "$LLVM_IR" || true)"
    [ "$FRAGMENT_VALUE_COUNT" -ge 2 ] || {
        echo "debug-smoketest: FAIL (debug_fragment_pair retained $FRAGMENT_VALUE_COUNT value-based scalarized fragments; expected at least 2)"
        grep -E "@llvm.dbg.(value|declare).*metadata !$FRAGMENT_VAR_ID|^!$FRAGMENT_VAR_ID = " "$LLVM_IR" | head -20 || true
        exit 1
    }
    if grep -Eq "@llvm.dbg.declare.*metadata !$FRAGMENT_VAR_ID, metadata !DIExpression\\(DW_OP_LLVM_fragment" "$LLVM_IR"; then
        echo "debug-smoketest: FAIL (debug_fragment_pair still uses stack-address fragment dbg.declare records)"
        grep -E "@llvm.dbg.declare.*metadata !$FRAGMENT_VAR_ID" "$LLVM_IR" | head -20 || true
        exit 1
    fi

    CLOSURE_VAR_ID="$(sed -n 's/^!\([0-9][0-9]*\) = !DILocalVariable(name: "closure".*/\1/p' "$LLVM_IR" | head -1)"
    [ -n "$CLOSURE_VAR_ID" ] || { echo "debug-smoketest: FAIL (closure DILocalVariable not found)"; exit 1; }

    CLOSURE_DECLARE_COUNT="$(grep -Ec "@llvm.dbg.declare\(metadata ptr %[^,]+, metadata !$CLOSURE_VAR_ID, metadata !DIExpression\(\)" "$LLVM_IR" || true)"
    [ "$CLOSURE_DECLARE_COUNT" -ge 1 ] || {
        echo "debug-smoketest: FAIL (closure whole-variable debug mirror dbg.declare not found)"
        grep -E "@llvm.dbg.(value|declare).*metadata !$CLOSURE_VAR_ID|^!$CLOSURE_VAR_ID = " "$LLVM_IR" | head -20 || true
        exit 1
    }
    if grep -Eq "@llvm.dbg.(value|declare).*metadata !$CLOSURE_VAR_ID, metadata !DIExpression\(DW_OP_LLVM_fragment" "$LLVM_IR"; then
        echo "debug-smoketest: FAIL (closure still uses fragmented debug locations instead of one mirror)"
        grep -E "@llvm.dbg.(value|declare).*metadata !$CLOSURE_VAR_ID" "$LLVM_IR" | head -20 || true
        exit 1
    fi
    grep -Eq "store volatile i64 4294967328" "$LLVM_IR" || {
        echo "debug-smoketest: FAIL (closure constant capture was not written to the debug mirror)"
        exit 1
    }
fi
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Drive cuda-gdb: stop at a kernel, walk to the kernel frame, dump args/locals.
GDB_LOG="$(mktemp)"
FRAGMENT_GDB_LOG="$(mktemp)"
CLOSURE_GDB_LOG="$(mktemp)"
ENUM_GDB_LOG="$(mktemp)"
PROJECTION_GDB_LOG="$(mktemp)"
ENUM_PROJECTION_GDB_LOG="$(mktemp)"
DEREF_GDB_LOG="$(mktemp)"
trap 'rm -f "$GDB_LOG" "$FRAGMENT_GDB_LOG" "$CLOSURE_GDB_LOG" "$ENUM_GDB_LOG" "$PROJECTION_GDB_LOG" "$ENUM_PROJECTION_GDB_LOG" "$DEREF_GDB_LOG"' EXIT

# Run from the example dir: the host binary resolves its embedded device
# artifact relative to the working directory.
( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
    -ex 'set pagination off' \
    -ex 'set breakpoint pending on' \
    -ex "break ${BREAK_AT:-test_option}" \
    -ex 'run' \
    -ex 'frame 1' \
    -ex 'info args' \
    -ex 'info locals' \
    -ex 'backtrace' \
    -ex 'kill' \
    "./target/release/$EXAMPLE" ) >"$GDB_LOG" 2>&1

echo "----- cuda-gdb output (tail) -----"
tail -25 "$GDB_LOG"
echo "----------------------------------"

# Verdict: a device breakpoint must have bound and fired, and at least one
# concrete value (scalar, pointer, struct field, or enum payload) must be visible.
fail=0
grep -qiE "CUDA thread hit .*Breakpoint" "$GDB_LOG" || { echo "debug-smoketest: FAIL (no device breakpoint hit)"; fail=1; }
grep -qE "= [0-9]|0x[0-9a-f]|\{.*:" "$GDB_LOG"        || { echo "debug-smoketest: FAIL (no inspectable args/locals)"; fail=1; }
grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$GDB_LOG" && { echo "debug-smoketest: FAIL (PTX did not load under cuda-gdb)"; fail=1; }

# compiler_features contains dedicated full-debug fixtures. Break on the source
# line immediately after each value is initialized, then require cuda-gdb to
# consume the generated DWARF rather than merely accepting the LLVM metadata.
if [ "$EXAMPLE" = "compiler_features" ]; then
    DEBUG_SOURCE="$EXAMPLE_DIR/src/main.rs"

    FRAGMENT_MARKER="CUDA_OXIDE_DEBUG_FRAGMENT_BREAKPOINT"
    FRAGMENT_LINE="$(grep -nF "$FRAGMENT_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$FRAGMENT_LINE" ]; then
        echo "debug-smoketest: FAIL (fragment debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$FRAGMENT_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype debug_fragment_pair' \
            -ex 'print debug_fragment_pair' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$FRAGMENT_GDB_LOG" 2>&1

        echo "----- cuda-gdb fragment output (tail) -----"
        tail -25 "$FRAGMENT_GDB_LOG"
        echo "-----------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$FRAGMENT_GDB_LOG" || { echo "debug-smoketest: FAIL (fragment source breakpoint did not hit)"; fail=1; }
        if grep -qE '\$[0-9]+ = \(17,[[:space:]]*4294967328\)' "$FRAGMENT_GDB_LOG"; then
            : # cuda-gdb rendered the Rust tuple directly.
        else
            grep -qE "__0[^0-9]*17([[:space:],}]|$)" "$FRAGMENT_GDB_LOG" || { echo "debug-smoketest: FAIL (debug_fragment_pair.__0 is not inspectable)"; fail=1; }
            grep -qE "__1[^0-9]*4294967328([[:space:],}]|$)" "$FRAGMENT_GDB_LOG" || { echo "debug-smoketest: FAIL (debug_fragment_pair.__1 is not inspectable)"; fail=1; }
        fi
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$FRAGMENT_GDB_LOG" && { echo "debug-smoketest: FAIL (fragment-debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    # Break between two observable closure calls. The second call keeps both
    # scalarized captures semantically live at the breakpoint even under normal
    # MIR optimization. The LLVM gate above still requires both source-variable
    # fragments, so this does not weaken reconstruction coverage.
    CLOSURE_MARKER="CUDA_OXIDE_DEBUG_CLOSURE_BREAKPOINT"
    CLOSURE_LINE="$(grep -nF "$CLOSURE_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$CLOSURE_LINE" ]; then
        echo "debug-smoketest: FAIL (closure debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$CLOSURE_LINE" \
            -ex 'run' \
            -ex 'frame 1' \
            -ex 'ptype closure' \
            -ex 'print closure' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$CLOSURE_GDB_LOG" 2>&1

        echo "----- cuda-gdb closure output (tail) -----"
        tail -25 "$CLOSURE_GDB_LOG"
        echo "------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$CLOSURE_GDB_LOG" || { echo "debug-smoketest: FAIL (closure source breakpoint did not hit)"; fail=1; }
        grep -qE "capture_0[[:space:]]*:[[:space:]]*(0x[[:xdigit:]]+|[0-9]+)" "$CLOSURE_GDB_LOG" || { echo "debug-smoketest: FAIL (closure capture_0 is not inspectable)"; fail=1; }
        grep -qE "capture_1[[:space:]]*:[[:space:]]*(0x[[:xdigit:]]+|[0-9]+)" "$CLOSURE_GDB_LOG" || { echo "debug-smoketest: FAIL (closure capture_1 is not inspectable)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$CLOSURE_GDB_LOG" && { echo "debug-smoketest: FAIL (closure-debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    ENUM_MARKER="CUDA_OXIDE_DEBUG_ENUM_BREAKPOINT"
    ENUM_LINE="$(grep -nF "$ENUM_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$ENUM_LINE" ]; then
        echo "debug-smoketest: FAIL (enum debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$ENUM_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype option_value' \
            -ex 'print option_value' \
            -ex 'ptype result_value' \
            -ex 'print result_value' \
            -ex 'ptype direct_value' \
            -ex 'print direct_value' \
            -ex 'ptype niche_value' \
            -ex 'print niche_value' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$ENUM_GDB_LOG" 2>&1

        echo "----- cuda-gdb enum output (tail) -----"
        tail -45 "$ENUM_GDB_LOG"
        echo "---------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (enum source breakpoint did not hit)"; fail=1; }
        grep -qE '\$[0-9]+ = ([^[:space:]]+::)?Some \(8\)' "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (Option<u32> active variant/payload is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = ([^[:space:]]+::)?Err \(4294967305\)' "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (Result<u32,u64> active Err variant/payload is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = ([^[:space:]]+::)?Wide \(8589934603\)' "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (direct-tag custom enum active variant/payload is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = ([^[:space:]]+::)?Some \(0x[[:xdigit:]]+\)' "$ENUM_GDB_LOG" || { echo "debug-smoketest: FAIL (niche Option<&u32> active variant/payload is not inspectable)"; fail=1; }
        grep -qF '<No data fields>' "$ENUM_GDB_LOG" && { echo "debug-smoketest: FAIL (enum value resolved without an active variant payload)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$ENUM_GDB_LOG" && { echo "debug-smoketest: FAIL (enum-debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    ENUM_PROJECTION_MARKER="CUDA_OXIDE_DEBUG_ENUM_PROJECTION_BREAKPOINT"
    ENUM_PROJECTION_LINE="$(grep -nF "$ENUM_PROJECTION_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$ENUM_PROJECTION_LINE" ]; then
        echo "debug-smoketest: FAIL (enum projection debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$ENUM_PROJECTION_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype projected_enum_payload' \
            -ex 'print projected_enum_payload' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$ENUM_PROJECTION_GDB_LOG" 2>&1

        echo "----- cuda-gdb enum projection output (tail) -----"
        tail -25 "$ENUM_PROJECTION_GDB_LOG"
        echo "--------------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$ENUM_PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (enum projection source breakpoint did not hit)"; fail=1; }
        grep -qE '\$[0-9]+ = 8589934603([[:space:]]|$)' "$ENUM_PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (enum Downcast -> Field payload binding is not inspectable)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$ENUM_PROJECTION_GDB_LOG" && { echo "debug-smoketest: FAIL (enum-projection debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    PROJECTION_MARKER="CUDA_OXIDE_DEBUG_PROJECTION_BREAKPOINT"
    PROJECTION_LINE="$(grep -nF "$PROJECTION_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$PROJECTION_LINE" ]; then
        echo "debug-smoketest: FAIL (projection debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$PROJECTION_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype projected_field' \
            -ex 'print projected_field' \
            -ex 'ptype projected_tuple' \
            -ex 'print projected_tuple' \
            -ex 'ptype projected_array' \
            -ex 'print projected_array' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$PROJECTION_GDB_LOG" 2>&1

        echo "----- cuda-gdb projection output (tail) -----"
        tail -35 "$PROJECTION_GDB_LOG"
        echo "---------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (projection source breakpoint did not hit)"; fail=1; }
        grep -qE '\$[0-9]+ = 18([[:space:]]|$)' "$PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (struct.field projection is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = 4294967329([[:space:]]|$)' "$PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (tuple.1 projection is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = 37([[:space:]]|$)' "$PROJECTION_GDB_LOG" || { echo "debug-smoketest: FAIL (array constant-index projection is not inspectable)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$PROJECTION_GDB_LOG" && { echo "debug-smoketest: FAIL (projection-debug PTX did not load under cuda-gdb)"; fail=1; }
    fi

    DEREF_MARKER="CUDA_OXIDE_DEBUG_DEREF_BREAKPOINT"
    DEREF_LINE="$(grep -nF "$DEREF_MARKER" "$DEBUG_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$DEREF_LINE" ]; then
        echo "debug-smoketest: FAIL (dereference debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $DEBUG_SOURCE:$DEREF_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
            -ex 'ptype deref_field' \
            -ex 'print deref_field' \
            -ex 'ptype deref_value' \
            -ex 'print deref_value' \
            -ex 'backtrace' \
            -ex 'kill' \
            "./target/release/$EXAMPLE" ) >"$DEREF_GDB_LOG" 2>&1

        echo "----- cuda-gdb dereference projection output (tail) -----"
        tail -30 "$DEREF_GDB_LOG"
        echo "----------------------------------------------------------"

        grep -qiE "CUDA thread hit .*Breakpoint" "$DEREF_GDB_LOG" || { echo "debug-smoketest: FAIL (dereference projection source breakpoint did not hit)"; fail=1; }
        grep -qE '\$[0-9]+ = 18([[:space:]]|$)' "$DEREF_GDB_LOG" || { echo "debug-smoketest: FAIL (dereference field projection is not inspectable)"; fail=1; }
        grep -qE '\$[0-9]+ = 41([[:space:]]|$)' "$DEREF_GDB_LOG" || { echo "debug-smoketest: FAIL (dereference projection is not inspectable)"; fail=1; }
        grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$DEREF_GDB_LOG" && { echo "debug-smoketest: FAIL (dereference-debug PTX did not load under cuda-gdb)"; fail=1; }
    fi
fi

if [ "$fail" -eq 0 ]; then
    if [ "$EXAMPLE" = "compiler_features" ]; then
        echo "debug-smoketest: PASS (optimized-MIR fragments + closure environments + source debugging + Rust enums + static, enum payload, and dereference projections verified on $ARCH)"
    else
        echo "debug-smoketest: PASS (source debugging + info args/locals verified on $ARCH)"
    fi
    exit 0
fi
exit 1
