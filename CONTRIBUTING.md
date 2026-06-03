# Contributing to cuda-oxide

Thank you for your interest in contributing to cuda-oxide! This document
explains the contribution process and requirements.

cuda-oxide is licensed under the [Apache License, Version 2.0](LICENSE).

## Table of Contents

- [Developer Certificate of Origin](#developer-certificate-of-origin)
- [Signing Your Commits](#signing-your-commits)
- [Contribution Process](#contribution-process)
- [Code Requirements](#code-requirements)
- [IP Review Process](#ip-review-process)

## Developer Certificate of Origin

cuda-oxide requires the Developer Certificate of Origin (DCO) process for all
contributions. The DCO is a lightweight mechanism to certify that you wrote or
otherwise have the right to submit the code you are contributing.

By making a contribution to this project, you agree to the following:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Signing Your Commits

Every commit in your pull request must include a `Signed-off-by` line.
This certifies that you agree to the DCO above.

To sign off on a commit, use the `-s` flag:

```bash
git commit -s -m "Description of change"
```

This adds a line to your commit message:

```text
Signed-off-by: Your Name <your.email@example.com>
```

If you have already made commits without sign-off, you can amend or rebase
to add it:

```bash
# Amend the most recent commit
git commit --amend -s --no-edit

# Rebase and sign all commits in a branch
git rebase --signoff main
```

Your `Signed-off-by` name and email must match your Git configuration
(`user.name` and `user.email`).

## Contribution Process

1. **Open an issue** describing the bug or feature you want to work on.
2. **Fork the repository** and create a feature branch from `main`.
3. **Implement your changes** following the code requirements below.
4. **Sign all commits** using `git commit -s` (see above).
5. **Submit a pull request** against the `main` branch with a clear
   description of the changes and their motivation.
6. **Respond to review feedback.** All submissions require review before
   merging. Maintainers may request changes or ask questions.

Pull requests that do not meet the requirements below or lack proper DCO
sign-off will not be merged.

## Code Requirements

### Toolchain

cuda-oxide requires the Rust nightly toolchain with `rustc_private` support.
See the [README](README.md) for setup instructions.

The repository includes a `flake.nix` that provides a fully reproducible development
environment (CUDA 13, LLVM 22, Clang, pinned Rust nightly). If you have Nix with
flakes enabled, `nix develop` is the quickest way to get everything in place.

### Formatting and Style

- Run `cargo fmt` before submitting. All code must be formatted with
  `rustfmt`.
- Run `cargo clippy` and address any warnings where reasonable.
- Follow existing code patterns and conventions in the crate you are
  modifying.

### License Headers

All new source files must include the NVIDIA copyright and SPDX header as the
first two lines:

```rust
// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
```

### Testing

- Compiler pipeline changes should be validated against the existing examples
  in `crates/rustc-codegen-cuda/examples/`.
- New GPU intrinsics should include a corresponding example demonstrating
  correct behavior.
- Dialect changes should include appropriate tests in the crate's `tests/`
  directory.

### Dependencies

- New dependencies must use permissive licenses (MIT, Apache-2.0, BSD, ISC,
  Zlib, or similar).
- No GPL, AGPL, SSPL, or other copyleft-licensed dependencies.
- If adding a new dependency, update `THIRD_PARTY_NOTICES` accordingly.

## IP Review Process

All contributions to cuda-oxide are subject to NVIDIA's IP review process.
Maintainers will ensure that contributions are reviewed in accordance with
NVIDIA's open source policies before merging.

For questions about the contribution process, please open an issue or contact
the maintainers.
