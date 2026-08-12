# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Use PowerShell on Windows
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

# Use Bash on Unix
set shell := ["bash", "-c"]

# Format all Rust code (root, codegen, examples)
fmt:
    cargo oxide fmt

# Check formatting without modifying files
fmt-check:
    cargo oxide fmt --check

# Lint every scope CI's clippy job lints, minus the per-example pass.
#
# CI runs clippy three times: over the root workspace, over
# crates/rustc-codegen-cuda (its own [workspace] for the rustc_private dylibs,
# which `--workspace` from the root cannot reach), and once per example. The
# first two are here, the root invocation spelled the way CI spells it.
# (Cargo's target-selection flags are additive and the virtual root workspace
# has no default-members, so the earlier `--all-targets --lib --tests` covered
# the same members and targets; the spelling was the only difference.)
#
# The per-example pass stays CI-only: it is one clippy run per example across
# 200-odd separate workspaces, which is a CI job rather than something to wait
# on locally.
# Run clippy with warnings as errors (root + codegen workspaces)
clippy:
    cargo clippy --workspace --all-targets -- -D warnings
    # Its own [workspace], so `--workspace` above stops at that boundary.
    cd crates/rustc-codegen-cuda && cargo clippy --all-targets -- -D warnings

# Run clippy and auto-fix warnings
clippy-fix:
    cargo clippy --all-targets --lib --tests --fix --allow-dirty --allow-staged

# Run unit tests for every package CI covers, so `just check` predicts CI.
# Mirrors .github/workflows/unit-tests.yml; keep the two in step.
#
# CI splits this across a matrix and marks some entries `needs_cuda`, meaning
# cuda-bindings' bindgen needs cuda.h at build time. Those packages live in
# `test-cuda` below, so this recipe runs on a machine with no CUDA at all.
# `--all-targets` matches the matrix default; the two exceptions below carry
# CI's own overrides.
# Run unit tests for every package CI covers that needs no CUDA
test:
    cargo test --all-targets \
        -p cuda-intrinsics-gen -p cuda-intrinsics -p llvm-export \
        -p dialect-mir -p dialect-nvvm -p mir-importer -p mir-lower \
        -p mir-transforms -p nvvm-transforms -p reserved-oxide-symbols \
        -p cuda-device -p libnvvm-sys -p nvjitlink-sys \
        -p cuda-artifact-finalizer -p cargo-oxide
    # `default = []`, but every consumer turns the object features on, and the
    # default set alone skips the eight ELF emit/extract tests.
    cargo test -p oxide-artifacts --all-targets --features object
    # Its own [workspace] (rustc-private dylibs), so a separate CI job covers
    # it and `-p` from the root cannot reach it.
    cd crates/rustc-codegen-cuda && cargo test --lib

# The five packages CI's matrix marks `needs_cuda`, kept separate so `test`
# runs on a machine with no CUDA at all.
#
# These need cuda.h for cuda-bindings' bindgen at build time, and their test
# binaries keep a DT_NEEDED on `libcuda.so.1`, so without a driver they fail to
# *load* -- `error while loading shared libraries: libcuda.so.1` -- which is a
# loader failure, not a test failure. CI runs them on driverless runners by
# shadowing the toolkit's link-time stub under that name; see unit-tests.yml.
#
# `--lib` for cuda-core skips the GPU-only VMM/P2P test in tests/vmm_p2p.rs.
# Run the five CUDA-linked packages (shadows the libcuda stub if no driver)
test-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    # The toolkit ships only the link-time stub `libcuda.so`, while the linker
    # stamps `libcuda.so.1` into the binary, so a machine with a toolkit but no
    # driver cannot even load these tests. CI shadows the stub under that name
    # (unit-tests.yml) and CONTRIBUTING documents the same recipe by hand; do it
    # here so `just check` works on the driverless machine CONTRIBUTING calls
    # the common case. A real driver already provides the name, so this is a
    # no-op there.
    if ! ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; then
        for root in "${CUDA_TOOLKIT_PATH:-}" "${CUDA_HOME:-}" /usr/local/cuda; do
            if [ -n "${root}" ] && [ -f "${root}/lib64/stubs/libcuda.so" ]; then
                shadow="$(mktemp -d)"
                trap 'rm -rf "${shadow}"' EXIT
                ln -sf "${root}/lib64/stubs/libcuda.so" "${shadow}/libcuda.so.1"
                # `:+` keeps the joining colon out when the variable is unset:
                # a trailing colon would put the cwd on the loader search path.
                export LD_LIBRARY_PATH="${shadow}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
                break
            fi
        done
    fi
    cargo test --all-targets \
        -p cuda-oxide-codegen -p cuda-macros -p cuda-host -p cuda-async
    cargo test -p cuda-core --lib

# Build docs warning-free + run doctests (mirrors the docs CI gate). The `test`
# recipe uses `--all-targets`, which skips doctests, so this covers them.
# cuda-bindings is excluded from doctests (its generated C doc comments are not
# valid Rust); its docs still build under the rustdoc allows in its lib.rs.
# Build docs warning-free and run doctests
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace
    cargo test --doc --workspace --exclude cuda-bindings
    # The docs gate builds this workspace's rustdoc separately too (#725); the
    # root `--workspace` cannot reach it.
    cd crates/rustc-codegen-cuda && RUSTDOCFLAGS="-D warnings" cargo doc --no-deps

# Run all checks (fmt + clippy + tests + guards + docs). Includes `test-cuda`:
# `clippy` and `doc-check` already build cuda-bindings, so this recipe needs a
# CUDA toolkit either way. Machines without even a toolkit get `test`. A driver
# is no longer required: `test-cuda` shadows the toolkit's libcuda stub itself.
# `check-guards` covers the status-guard and cargo-deny workflows in full; see
# its comment for prerequisites. Still CI-only: clippy's per-example pass (one
# run per example workspace), naming-guard (its grep pipeline
# lives inline in the workflow, with no script to invoke), examples-compile
# (needs the CUDA codegen backend), the book build, and CodeQL.
# Run CI's gates minus naming-guard, examples-compile, book, CodeQL
check: fmt-check clippy test test-cuda check-guards doc-check

# Clean project-local Cargo outputs and known cuda-oxide artifacts
clean-artifacts:
    cargo oxide clean

# Build an example (compile only)
build example:
    cargo oxide build {{example}}

# Build and run an example
run example:
    cargo oxide run {{example}}

# Show full compilation pipeline with verbose output
pipeline example:
    cargo oxide pipeline {{example}}

# Run every example with GPU-aware gating (see scripts/smoketest.sh --help)
smoketest *args:
    scripts/smoketest.sh {{args}}

# Verify every error* example is in STATUS.md and smoketest.sh ERROR_EXAMPLES
check-errors:
    scripts/check-error-example-status.sh

# Every status-guard job (error* examples in STATUS.md, the smoketest example
# contract, the README crate inventory, toolchain-pin parity, and the book's CLI
# command reference) plus all three cargo-deny jobs: `cargo deny check` enforces
# deny.toml over the root workspace's resolved graph and again over
# crates/rustc-codegen-cuda, which resolves its own because it has its own
# `[workspace]`; the license inventory covers what both declare; and deny.toml
# holds over the example workspaces. These were only reachable by reading the
# workflows, so `just
# check` could pass while status-guard or cargo-deny failed. Keep this list in
# step when a guard is added to either workflow -- the crate-inventory and
# toolchain-parity jobs were added to CI without being added here, which is the
# same drift this recipe exists to prevent. Prerequisites: `cargo-deny` on PATH
# (`cargo install cargo-deny --locked`) and `python3` (most of the scripts drive
# it). The scripts are invoked via `bash` as CI does, since not all of them
# carry an exec bit.
# Run the status-guard and cargo-deny CI jobs (needs cargo-deny, python3)
check-guards:
    bash scripts/check-error-example-status.sh
    bash scripts/check-example-smoketest-contract.sh
    bash scripts/check-crate-inventory.sh
    bash scripts/check-toolchain-parity.sh
    bash scripts/check-cli-doc-coverage.sh
    bash scripts/check-book-api-names.sh
    cargo deny --locked check
    cargo deny --manifest-path crates/rustc-codegen-cuda/Cargo.toml --locked check
    bash scripts/check-dependency-licenses.sh
    bash scripts/check-example-license-policy.sh
