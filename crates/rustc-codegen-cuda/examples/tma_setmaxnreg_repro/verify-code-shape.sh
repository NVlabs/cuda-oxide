#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-ptx}"
target="${2:-sm_120a}"
cta='cp\.async\.bulk\.tensor\.[1-5]d\.shared::cta\.global\.tile\.mbarrier::complete_tx::bytes'
cluster='cp\.async\.bulk\.tensor\.[1-5]d\.shared::cluster\.global\.tile\.mbarrier::complete_tx::bytes'

case "${mode}" in
    ptx)
        artifact="${root}/tma_setmaxnreg_repro.ptx"
        test -s "${artifact}"
        grep -qx "\.target ${target}" "${artifact}"
        [[ "$(grep -Ec "${cta}" "${artifact}")" -eq 6 ]]
        [[ "$(grep -Eo "${cta}" "${artifact}" | sort -u | wc -l)" -eq 5 ]]
        ! grep -Eq "${cluster}" "${artifact}"

        ptxas_bin=""
        for candidate in "${CUDA_OXIDE_PTXAS:-}" \
                         "$(command -v ptxas 2>/dev/null)" \
                         "${CUDA_HOME:+${CUDA_HOME}/bin/ptxas}" \
                         /usr/local/cuda/bin/ptxas \
                         /usr/local/cuda-*/bin/ptxas; do
            if [[ -n "${candidate}" && -x "${candidate}" ]]; then
                ptxas_bin="${candidate}"
                break
            fi
        done
        if [[ -z "${ptxas_bin}" ]]; then
            echo "error: ptxas is required for the setmaxnreg SASS regression" >&2
            exit 1
        fi
        nvdisasm_bin="${CUDA_OXIDE_NVDISASM:-${ptxas_bin%/ptxas}/nvdisasm}"
        if [[ ! -x "${nvdisasm_bin}" ]]; then
            nvdisasm_bin="$(command -v nvdisasm 2>/dev/null)"
        fi
        if [[ -z "${nvdisasm_bin}" || ! -x "${nvdisasm_bin}" ]]; then
            echo "error: nvdisasm is required for the setmaxnreg SASS regression" >&2
            exit 1
        fi

        scratch="$(mktemp -d /tmp/cuda-oxide-tma-setmaxnreg.XXXXXX)"
        trap 'rm -rf "${scratch}"' EXIT
        "${ptxas_bin}" --gpu-name="${target}" --verbose "${artifact}" \
            --output-file="${scratch}/repro.cubin" 2>"${scratch}/ptxas.log"
        ! grep -q 'C7506' "${scratch}/ptxas.log"
        "${nvdisasm_bin}" --print-code "${scratch}/repro.cubin" >"${scratch}/sass.txt"
        [[ "$(grep -c 'USETMAXREG.DEALLOC' "${scratch}/sass.txt")" -eq 2 ]]
        [[ "$(grep -c 'USETMAXREG.TRY_ALLOC' "${scratch}/sass.txt")" -eq 2 ]]
        grep -q 'UTMALDG.2D' "${scratch}/sass.txt"
        ! grep -q 'CALL.ABS' "${scratch}/sass.txt"
        ;;
    nvvm)
        artifact="${root}/tma_setmaxnreg_repro.ll"
        test -s "${artifact}"
        [[ "$(grep -Ec "${cta}" "${artifact}")" -eq 6 ]]
        [[ "$(grep -Eo "${cta}" "${artifact}" | sort -u | wc -l)" -eq 5 ]]
        ! grep -Eq "${cluster}" "${artifact}"
        ! grep -q 'llvm\.nvvm\.cp\.async\.bulk\.tensor\.g2s\.cta' "${artifact}"
        ;;
    *)
        echo "usage: $0 [ptx|nvvm] [sm_target]" >&2
        exit 2
        ;;
esac

echo "tma_setmaxnreg_repro ${mode} code shape: PASS"
