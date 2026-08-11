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
# show real values (scalars, pointers, and struct fields), not just metadata
# in the emitted IR.
#
# For compiler_features, a second debugger pass validates closure-environment
# DWARF specifically: the source breakpoint stops after a `move` closure has
# been initialized, and cuda-gdb must expose `capture_0` and `capture_1` as
# inspectable members of the closure local.
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

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# Drive cuda-gdb: stop at a kernel, walk to the kernel frame, dump args/locals.
GDB_LOG="$(mktemp)"
CLOSURE_GDB_LOG="$(mktemp)"
trap 'rm -f "$GDB_LOG" "$CLOSURE_GDB_LOG"' EXIT

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
# concrete value (scalar, pointer, or struct field) must be visible.
fail=0
grep -qiE "CUDA thread hit .*Breakpoint" "$GDB_LOG" || { echo "debug-smoketest: FAIL (no device breakpoint hit)"; fail=1; }
grep -qE "= [0-9]|0x[0-9a-f]|\{.*:" "$GDB_LOG"        || { echo "debug-smoketest: FAIL (no inspectable args/locals)"; fail=1; }
grep -qiE "INVALID_PTX|JIT compilation failed|No device code" "$GDB_LOG" && { echo "debug-smoketest: FAIL (PTX did not load under cuda-gdb)"; fail=1; }

# compiler_features contains a dedicated closure-environment fixture. Break on
# the source line immediately after the closure is created, then require
# cuda-gdb to describe and print both captured fields.
if [ "$EXAMPLE" = "compiler_features" ]; then
    CLOSURE_SOURCE="$EXAMPLE_DIR/src/main.rs"
    CLOSURE_MARKER="CUDA_OXIDE_DEBUG_CLOSURE_BREAKPOINT"
    CLOSURE_LINE="$(grep -nF "$CLOSURE_MARKER" "$CLOSURE_SOURCE" | head -1 | cut -d: -f1)"

    if [ -z "$CLOSURE_LINE" ]; then
        echo "debug-smoketest: FAIL (closure debug breakpoint marker not found)"
        fail=1
    else
        ( cd "$EXAMPLE_DIR" && timeout 300 "$CUDA_GDB" --batch \
            -ex 'set pagination off' \
            -ex 'set breakpoint pending on' \
            -ex "break $CLOSURE_SOURCE:$CLOSURE_LINE" \
            -ex 'run' \
            -ex 'frame 0' \
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
fi

if [ "$fail" -eq 0 ]; then
    if [ "$EXAMPLE" = "compiler_features" ]; then
        echo "debug-smoketest: PASS (source debugging + closure environments verified on $ARCH)"
    else
        echo "debug-smoketest: PASS (source debugging + info args/locals verified on $ARCH)"
    fi
    exit 0
fi
exit 1
