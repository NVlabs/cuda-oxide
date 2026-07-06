#!/bin/sh
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# `CUDA_OXIDE_POST_IR` hook: differentiate `poly` with Enzyme and rewire the
# `poly_dx` stub to the generated derivative. Invoked by the pipeline as
#
#   <hook> <ll_path> <output_dir> <output_name> <target>
#
# The contract is transform-only: rewrite <ll_path> in place and exit 0 to let
# the pipeline continue PTX generation on the edited IR; non-zero aborts the
# build, surfacing this script's stderr.
#
# Configuration (all optional):
#   LLVMENZYME  path to the LLVMEnzyme plugin (default: probe a few common
#               install locations; if none exists the hook leaves the IR
#               untouched and exits 0 — src/main.rs then prints `skipping:`)
#   ENZYME_OPT  the `opt` matching the plugin's LLVM major (default: opt-21,
#               /usr/lib/llvm-21/bin/opt, then plain `opt`)
set -eu

ll_path="$1"
say() { echo "enzyme_autodiff/enzyme.sh: $*" >&2; }

# --- 1. Locate the Enzyme plugin; missing plugin = graceful no-op. ---------
plugin="${LLVMENZYME:-}"
if [ -z "${plugin}" ]; then
    for cand in /usr/local/lib/LLVMEnzyme-21.so /usr/lib/LLVMEnzyme-21.so; do
        if [ -f "${cand}" ]; then plugin="${cand}"; break; fi
    done
fi
if [ -z "${plugin}" ] || [ ! -f "${plugin}" ]; then
    say "LLVMEnzyme plugin not found; leaving the IR untouched"
    say "(build Enzyme against LLVM 21 and set LLVMENZYME=/path/to/LLVMEnzyme-21.so)"
    exit 0
fi

opt_bin="${ENZYME_OPT:-}"
if [ -z "${opt_bin}" ]; then
    for cand in opt-21 /usr/lib/llvm-21/bin/opt opt; do
        if command -v "${cand}" >/dev/null 2>&1; then opt_bin="${cand}"; break; fi
    done
fi
if [ -z "${opt_bin}" ]; then
    say "no opt binary found (set ENZYME_OPT=/path/to/opt matching the plugin's LLVM)"
    exit 1
fi

# --- 2. Find the primal and stub symbols in the exported IR. ---------------
# cuda-oxide exports device functions under their Rust path with `::` lowered
# to `__`, so match on the path suffix rather than hard-coding the mangling.
find_sym() {
    sed -n "s/^define double @\([A-Za-z0-9_]*$1\)(double[^)]*).*{\$/\1/p" "${ll_path}" | head -n 1
}
poly_sym="$(find_sym poly)"
stub_sym="$(find_sym poly_dx)"
if [ -z "${poly_sym}" ] || [ -z "${stub_sym}" ]; then
    say "could not find @…poly / @…poly_dx defines in ${ll_path}"
    exit 1
fi
say "differentiating @${poly_sym}; rewiring @${stub_sym} to the Enzyme derivative"

# --- 3. Retarget the stub. --------------------------------------------------
# Rename the stub's `define` line only — call sites keep referring to the
# original name — then append a replacement definition whose body asks Enzyme
# for the forward-mode derivative (input tangent seeded with 1.0).
tmp="${ll_path}.enzyme.tmp"
sed "s/^define double @${stub_sym}(/define double @${stub_sym}__sentinel_stub(/" \
    "${ll_path}" > "${tmp}"
cat >> "${tmp}" <<EOF

declare double @__enzyme_fwddiff(...)

define double @${stub_sym}(double %x) {
  %r = call double (...) @__enzyme_fwddiff(ptr @${poly_sym}, double %x, double 1.0)
  ret double %r
}
EOF

# --- 4. Run Enzyme (+ O2) and put the result back in place. -----------------
"${opt_bin}" -load-pass-plugin="${plugin}" -passes='enzyme,default<O2>' \
    "${tmp}" -S -o "${ll_path}"
rm -f "${tmp}"
say "done: @${stub_sym} now computes d ${poly_sym}/dx"
