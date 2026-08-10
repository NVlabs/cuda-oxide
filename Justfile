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

# Run clippy with warnings as errors
clippy:
    cargo clippy --all-targets --lib --tests -- -D warnings

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
test-cuda:
    cargo test --all-targets \
        -p cuda-oxide-codegen -p cuda-macros -p cuda-host -p cuda-async
    cargo test -p cuda-core --lib

# Build docs warning-free + run doctests (mirrors the docs CI gate). The `test`
# recipe uses `--all-targets`, which skips doctests, so this covers them.
# cuda-bindings is excluded from doctests (its generated C doc comments are not
# valid Rust); its docs still build under the rustdoc allows in its lib.rs.
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace
    cargo test --doc --workspace --exclude cuda-bindings

# Run all checks (fmt + clippy + tests + docs). Includes `test-cuda`: `clippy`
# and `doc-check` already build cuda-bindings, so this recipe needs a CUDA
# toolkit either way, and the pre-split `test` already ran driver-linked
# cuda-host/cuda-macros binaries. Machines without even a toolkit get `test`.
check: fmt-check clippy test test-cuda doc-check

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
