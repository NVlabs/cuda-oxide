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

# Run unit tests (excludes rustc_private and device-only crates)
test:
    cargo test -p cuda-host -p cuda-macros -p llvm-export -p dialect-mir -p dialect-nvvm --lib --tests

# Build docs warning-free + run doctests (mirrors the docs CI gate). The `test`
# recipe uses `--lib --tests`, which skips doctests, so this covers them.
# cuda-bindings is excluded from doctests (its generated C doc comments are not
# valid Rust); its docs still build under the rustdoc allows in its lib.rs.
doc-check:
    RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --workspace
    cargo test --doc --workspace --exclude cuda-bindings

# Run all checks (fmt + clippy + test + docs)
check: fmt-check clippy test doc-check

# Clean local Rust build outputs and generated example IR/PTX artifacts
clean-artifacts:
    cargo clean
    (cd crates/rustc-codegen-cuda && cargo clean)
    for manifest in crates/rustc-codegen-cuda/examples/*/Cargo.toml; do example_dir="$(dirname "$manifest")"; (cd "$example_dir" && cargo clean); rm -f "$example_dir"/*.ll "$example_dir"/*.ptx; done

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
