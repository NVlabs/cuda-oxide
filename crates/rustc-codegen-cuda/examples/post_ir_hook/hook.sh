#!/bin/sh
#
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Trivial `CUDA_OXIDE_POST_IR` hook. The pipeline invokes each hook as
#
#   <hook> <ll_path> <output_dir> <output_name> <target>
#
# and the contract is transform-only: rewrite <ll_path> in place, then exit 0
# to let the pipeline continue PTX generation on the edited IR (non-zero
# aborts the build, surfacing this script's stderr).
#
# Here the "transform" is the smallest observable one: rewrite the MARKER
# constant that `src/main.rs` bakes into the kernel, so the GPU result proves
# the PTX really came from the edited IR.
set -eu

ll_path="$1"

echo "post_ir_hook/hook.sh: rewriting MARKER 1010101 -> 2020202 in ${ll_path}" >&2

tmp="${ll_path}.hook.tmp"
sed 's/1010101/2020202/g' "${ll_path}" > "${tmp}"
mv "${tmp}" "${ll_path}"
