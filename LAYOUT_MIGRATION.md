# Layout migration: current tree → `cuda-rust/` sketch

This inventory maps the current working tree onto the layout in
[REPOSITORY_MERGE.md](REPOSITORY_MERGE.md) (Repo layout). It lists **moves**
first, then every class of **path rewrite** those moves force.

> **Current migration batch:** SIMT crates, book, scripts, Justfile, and
> toolchain live under `cuda-oxide/`. The Cargo **workspace** (`Cargo.toml` /
> `Cargo.lock`) stays at the repository root so `cuda-core` and friends can
> inherit `[workspace.package]` (they are siblings of `cuda-oxide/`, not
> children). Oxide copies of the three host crates are at the root; cutile-rs
> still has its own copies. The “today” paths below are the historical source
> side of the plan.

Assumed target (repository root = `cuda-rust/`):

| Location | Role |
| --- | --- |
| `Cargo.toml` | Stable workspace: `cuda-bindings`, `cuda-core`, `cuda-async` |
| `cuda-bindings/`, `cuda-core/`, `cuda-async/` | Shared published host crates |
| `cutile-rs/` | Tile model (no copy of the three shared crates) |
| `cuda-oxide/` | SIMT model: its own `Cargo.toml`, `rust-toolchain.toml`, `Justfile`, `crates/` |
| `.github/workflows/` | Path-filtered lanes at the repository root |

The sketch omits several current crates and root files; those are called out
under [Not in the sketch](#not-in-the-sketch).

## Current vs target (high level)

```
# today                                              # target
./Cargo.toml                                         ./Cargo.toml          (new: 3 shared crates)
./crates/cuda-bindings  ─┐                           ./cuda-bindings/
./crates/cuda-core      ─┼─ oxide copies  ──merge──► ./cuda-core/
./crates/cuda-async     ─┘                           ./cuda-async/
./cutile-rs/cuda-bindings  ─┐
./cutile-rs/cuda-core      ─┼─ cutile copies ─merge─┘
./cutile-rs/cuda-async     ─┘
./crates/<rest>                                      ./cuda-oxide/crates/<rest>
./rust-toolchain.toml, Justfile, Cargo.toml          ./cuda-oxide/{rust-toolchain.toml,Justfile,Cargo.toml}
./intrinsics, scripts, assets, cuda-oxide-book       ./cuda-oxide/{intrinsics,scripts,assets,cuda-oxide-book}
./cutile-rs/{cutile*, cuda-tile-rs, book, …}         ./cutile-rs/ (unchanged names, minus shared crates)
./.github                                            ./github (stay; add path filters)
```

Nesting the SIMT tree under `cuda-oxide/` **does not** by itself break
intra-`crates/` relative paths (`../mir-importer`, `../../../cuda-device`,
etc.). Lifting the three host crates **out** of `crates/` **does**.

## Moves

Use `git mv` so history follows. Do the shared-crate merge **before** bulk
path rewrites.

### 1. Nest the SIMT tree under `cuda-oxide/`

These live at the repository root today and belong under `cuda-oxide/` in the
sketch.

| From (today) | To |
| --- | --- |
| `Cargo.toml` | `cuda-oxide/Cargo.toml` |
| `Cargo.lock` | `cuda-oxide/Cargo.lock` |
| `Justfile` | `cuda-oxide/Justfile` |
| `rust-toolchain.toml` | `cuda-oxide/rust-toolchain.toml` |
| `crates/` | `cuda-oxide/crates/` |
| `cuda-oxide-book/` | `cuda-oxide/cuda-oxide-book/` |
| `intrinsics/` | `cuda-oxide/intrinsics/` |
| `assets/` | `cuda-oxide/assets/` |
| `scripts/` | `cuda-oxide/scripts/` |
| `.cargo/` | `cuda-oxide/.cargo/` (alias `oxide` is SIMT-specific) |
| `deny.toml` | `cuda-oxide/deny.toml` (or keep a root deny that covers all workspaces) |
| `dependency-licenses.csv` | `cuda-oxide/dependency-licenses.csv` |

`crates/` still contains `cuda-bindings`, `cuda-core`, and `cuda-async` at this
step. Lift them next.

### 2. Lift shared host crates to the repository root

**One directory each at the root**, not two copies.

| From | To | Notes |
| --- | --- | --- |
| `cuda-oxide/crates/cuda-bindings` **or** `cutile-rs/cuda-bindings` | `cuda-bindings/` | Two existing trees; see [Shared crate merge](#shared-crate-merge) |
| `cuda-oxide/crates/cuda-core` **or** `cutile-rs/cuda-core` | `cuda-core/` | Same |
| `cuda-oxide/crates/cuda-async` **or** `cutile-rs/cuda-async` | `cuda-async/` | Same |

After the lift, delete the leftover copy under `cutile-rs/` and under
`cuda-oxide/crates/`.

### 3. `cutile-rs/` — keep in place, drop shared crates

Already imported at `cutile-rs/`. Do **not** re-prefix. After the lift, this
directory should match the sketch:

- Keep: `cutile/`, `cutile-macro/`, `cutile-compiler/`, `cutile-ir/`,
  `cutile-kernels/`, `cuda-tile-rs/`, `cutile-benchmarks/`, `cutile-examples/`,
  `cutile-book/`, `assets/`, `scripts/`
- Remove (after merge): `cuda-bindings/`, `cuda-core/`, `cuda-async/`
- Leave nested `cutile-rs/.github/` inert (workflows already live at root)

Submodule path is already `cutile-rs/cuda-tile-rs/cuda-tile`. No `.gitmodules`
change for this layout.

### 4. Stay at the repository root

| Path | Why |
| --- | --- |
| `.github/` | Path-filtered lanes; do not nest under `cuda-oxide/` |
| `.gitmodules` | Git only reads the root file |
| `.gitignore`, `.gitattributes` | Whole-repo |
| `LICENSE`, `SECURITY.md`, `CONTRIBUTING.md`, `THIRD_PARTY_NOTICES` | Repo-level (fold cutile-rs copies later if needed) |
| `README.md` | New top-level readme; current SIMT readme can move to `cuda-oxide/README.md` |
| `REPOSITORY_MERGE.md`, this file | Planning |

## Shared crate merge

This is not a pure `git mv`. The two copies are **not** the same crate.

| | Oxide (`crates/cuda-*`, 0.2.1) | cutile-rs (`cutile-rs/cuda-*`, 0.3.0) |
| --- | --- | --- |
| `cuda-bindings` | bindgen 0.71, no `libloading` | bindgen 0.69, `libloading`, generated API helpers |
| `cuda-core` | depends on `cuda-macros`, `oxide-artifacts` | thin RAII over bindings only |
| `cuda-async` | oxide completion reactor | cutile reactor (loom/miri/TSan jobs in `cutile-rs.yml`) |

The stable workspace cannot publish two packages named `cuda-core`. Before
path rewrites, pick a policy:

1. **Unify** into one tree at the root (API/version/MSRV/bindgen work; oxide
   `cuda-core` must not keep a hard `path = "../cuda-macros"` into the SIMT
   workspace unless that dependency is optional or split into another crate).
2. **Rename** the SIMT-specific surface (`cuda-oxide-core` or similar) and
   keep cutile’s `cuda-core` as the shared published crate.

Until that is decided, do not delete either copy.

Oxide `cuda-core` today:

```
cuda-bindings = { path = "../cuda-bindings" }
cuda-macros = { path = "../cuda-macros" }          # SIMT-only
oxide-artifacts = { workspace = true, ... }        # SIMT-only
```

After the lift, those SIMT deps become
`path = "../cuda-oxide/crates/cuda-macros"` (or workspace path deps from the
root `Cargo.toml`).

## Not in the sketch

Still move under `cuda-oxide/crates/` with the rest of the SIMT crates:

- `dialect-iket`, `dialect-ptx`, `iket-lower`, `ptx-parse`, `ptx-schedule`

Decide separately (sketch is silent):

| Path | Suggestion |
| --- | --- |
| `flake.nix`, `flake.lock` | Root (devshell for the whole repo) **or** `cuda-oxide/` |
| `.devcontainer/`, `.dockerignore` | Root (context is still the repo) |
| `cutile-rs/flake.nix`, `cutile-rs/deny.toml` | Fold into root / `cuda-oxide` deny, then drop |
| Root `README.md` vs `cuda-oxide` vs `cutile-rs` | Three READMEs: root index + two product trees |

## New / rewritten manifests

Create after the moves:

### Root `Cargo.toml` (stable workspace)

Members only:

```toml
members = [
    "cuda-bindings",
    "cuda-core",
    "cuda-async",
]
```

`[workspace.dependencies]` path entries:

```toml
cuda-bindings = { path = "cuda-bindings" }
cuda-core = { path = "cuda-core" }
cuda-async = { path = "cuda-async" }
```

If oxide crates keep depending on these by workspace inheritance, **either**
the SIMT workspace uses `path = "../cuda-core"` (not `workspace = true` from a
different workspace) **or** the root workspace also lists
`cuda-oxide/crates/...` (that contradicts “stable workspace = three crates”).
Prefer explicit `path = "../cuda-*"` from `cuda-oxide/Cargo.toml` and from
`cutile-rs/Cargo.toml`.

### `cuda-oxide/Cargo.toml`

- Drop members `crates/cuda-bindings`, `crates/cuda-core`, `crates/cuda-async`.
- Change remaining members from `"crates/foo"` to the same relative paths
  (still valid once this file lives in `cuda-oxide/`).
- Replace

  ```toml
  cuda-bindings = { path = "crates/cuda-bindings" }
  cuda-core = { path = "crates/cuda-core" }
  cuda-async = { path = "crates/cuda-async" }
  ```

  with

  ```toml
  cuda-bindings = { path = "../cuda-bindings" }
  cuda-core = { path = "../cuda-core" }
  cuda-async = { path = "../cuda-async" }
  ```

- Comment that currently says `crates/rustc-codegen-cuda` stays correct
  relative to `cuda-oxide/`.

### `cutile-rs/Cargo.toml`

Remove `cuda-bindings`, `cuda-core`, `cuda-async` from `members` and
`default-members`. Point workspace deps at the parent:

```toml
cuda-bindings = { path = "../cuda-bindings", version = "…" }
cuda-core = { path = "../cuda-core", version = "…" }
cuda-async = { path = "../cuda-async", version = "…" }
```

Keep `cutile-compiler = { path = "cutile-compiler", … }` etc. as they are.

`cutile/Cargo.toml` `readme = "../README.md"` stays valid.

## Path rewrites (by class)

Relative paths that only go through `cuda-oxide/crates/<sibling>` do **not**
need an extra `../` after nesting. Paths that pointed at the three host crates
**do**.

### A. SIMT example `Cargo.toml` path deps (largest set)

~200 files under `crates/rustc-codegen-cuda/examples/**/Cargo.toml`.

Today, from `crates/rustc-codegen-cuda/examples/<ex>/`:

| Dep | Today | After lift to repo root |
| --- | --- | --- |
| `cuda-core` / `cuda-async` / `cuda-bindings` | `../../../cuda-core` | `../../../../../cuda-core` |
| nested `kernel-lib/` etc. | `../../../../cuda-core` | `../../../../../../cuda-core` |
| `cuda-device`, `cuda-host`, `ptx-parse`, `cuda-macros`, `cuda-intrinsics` | `../../../cuda-device` (or `../../../../crates/cuda-macros`) | **unchanged** if those crates stay under `cuda-oxide/crates/` |

The oddballs that already use `../../../../crates/cuda-macros` (e.g.
`printf`, `cluster`, `clc`, `error`, `debug`, `compiler_features`,
`future_apis`) stay valid after nesting; they must still be rewritten if any
of those deps become the root host crates.

Also:

- `crates/cuda-core/Cargo.toml` — `path = "../cuda-bindings"` →
  `path = "../cuda-bindings"` at repo root is the same *if both crates sit
  next to each other at root*; `cuda-macros` / `oxide-artifacts` must point
  into `cuda-oxide/crates/…`.
- `crates/cuda-async/Cargo.toml` — `../cuda-core`, `../cuda-bindings` stay
  sibling paths at the root.

Mechanical check after the rewrite:

```bash
# from a typical example dir, the host crate must exist
test -f cuda-oxide/crates/rustc-codegen-cuda/examples/vecadd/Cargo.toml
# cuda-core at repo root
test -f cuda-core/Cargo.toml
```

### B. `cargo-oxide` hard-coded repo layout

`find_workspace_root` walks **up** from CWD and requires
`Cargo.toml` + `crates/rustc-codegen-cuda`. After the move that pair lives
under `cuda-oxide/`, so a command run at the **repository root** no longer
finds the backend.

| File | What to change |
| --- | --- |
| `crates/cargo-oxide/src/backend.rs` | Accept `cuda-oxide/crates/rustc-codegen-cuda` **or** `crates/rustc-codegen-cuda`; do not require the stable-workspace `Cargo.toml` to sit beside `crates/` |
| `crates/cargo-oxide/src/commands.rs` | Same `join("crates/rustc-codegen-cuda")` sites; `rust-toolchain.toml` is `cuda-oxide/rust-toolchain.toml`; `.cargo/cuda-oxide.toml` is `cuda-oxide/.cargo/cuda-oxide.toml` |
| `crates/cargo-oxide/src/commands.rs` | Comments that lockstep with `crates/cuda-bindings/build.rs` → `cuda-bindings/build.rs` |
| Tests in `commands.rs` / `backend.rs` that mkdir `crates/rustc-codegen-cuda` | Fixture layout must match discovery |

### C. Intrinsics generator (repo-root = SIMT root)

`cuda-intrinsics-gen` joins `intrinsics/overlay.toml`, `intrinsics/probes`,
`intrinsics/evidence`, `crates/dialect-nvvm/...`. Today `repo_root` is the
git/workspace root. After the move it must be `cuda-oxide/` (the directory
that contains both `intrinsics/` and `crates/`).

| File | Strings |
| --- | --- |
| `crates/cuda-intrinsics-gen/src/{extract,resolve,generate,coverage,probe,render,abi_history}.rs` | `intrinsics/...` relative to SIMT root |
| Call sites that pass git top-level as `repo_root` | Pass `cuda-oxide/` or detect the nested root |

### D. Other Rust hard-coded `crates/` paths

| File | Update |
| --- | --- |
| `crates/ptx-schedule/src/campaign.rs` | `crates/rustc-codegen-cuda/examples`, `crates/fuzzer/artifacts/schedule` → under `cuda-oxide/` |
| `crates/cuda-oxide-codegen/tests/spine_kernel_ptx.rs` | comment pointing at `crates/cuda-bindings/build.rs` |
| `crates/cuda-core/build.rs` | comment pointing at `crates/cuda-bindings/toolkit_target.rs` → `cuda-bindings/toolkit_target.rs` |

### E. GitHub Actions (root `.github/` stays)

Add `defaults.run.working-directory: cuda-oxide` **or** prefix every path.
Path-filter lanes should key off `cuda-oxide/**`, `cutile-rs/**`,
`cuda-bindings/**`, `cuda-core/**`, `cuda-async/**`.

| Workflow | Today | After |
| --- | --- | --- |
| `fmt.yml`, `clippy.yml`, `unit-tests.yml`, `docs.yml` | `working-directory: crates/rustc-codegen-cuda` | `cuda-oxide/crates/rustc-codegen-cuda` |
| `fmt.yml` | `crates/cuda-macros/tests/device-only` | `cuda-oxide/crates/cuda-macros/tests/device-only` |
| `fmt.yml` / `clippy.yml` | `crates/rustc-codegen-cuda/examples/**` | `cuda-oxide/crates/rustc-codegen-cuda/examples/**` |
| `cargo-deny.yml` | `--manifest-path crates/...` | `--manifest-path cuda-oxide/crates/...`; add root workspace deny for the three host crates |
| `status-guard.yml`, `naming-guard.yml`, `cargo-deny.yml` | `bash scripts/...` | `bash cuda-oxide/scripts/...` **or** `working-directory: cuda-oxide` |
| `examples-compile.yml` | `scripts/smoketest.sh` | same as scripts |
| `book.yml` | `cuda-oxide-book/...` | `cuda-oxide/cuda-oxide-book/...` |
| `book.yml` | `cutile-rs/scripts/build_versioned_book.sh` | unchanged |
| `cutile-rs.yml` | `working-directory: cutile-rs` | still valid; `cargo test -p cuda-async` now resolves via `path = "../cuda-async"` |
| `docs.yml` | comment `crates/cuda-bindings/src/lib.rs` | `cuda-bindings/src/lib.rs` |
| `CODEOWNERS` | `/crates/cuda-bindings/` | `/cuda-bindings/` |

Root `cargo` invocations (`cargo clippy --workspace`, `cargo test -p cuda-host`)
must run with `--manifest-path cuda-oxide/Cargo.toml` (or `working-directory:
cuda-oxide`). Host-crate tests (`-p cuda-core`) need the **root**
manifest.

### F. `cuda-oxide/scripts/` (move with the tree, then fix “parent is git root”)

Most scripts `cd "$(dirname "$0")/.."`. After the move that parent is
`cuda-oxide/`, which is what they want for `Cargo.toml`, `crates/`,
`intrinsics/`. **Do not** change those `../` hops.

Update strings that still mean **git** root or the old host-crate path:

| Script | Issue |
| --- | --- |
| `check-crate-inventory.sh` | `MANIFEST=Cargo.toml` is `cuda-oxide/Cargo.toml` (OK). `README.md` if the crate table moves with the SIMT readme. Members will no longer include `crates/cuda-*`. |
| `check-dependency-licenses.sh` | `crates/rustc-codegen-cuda/Cargo.toml`, `Cargo.lock` paths; add root `cuda-bindings` lock if the stable workspace has its own lock |
| `check-toolchain-parity.sh` | `crates/rustc-codegen-cuda/rust-toolchain.toml` still relative to SIMT root (OK); root `rust-toolchain.toml` is now `cuda-oxide/rust-toolchain.toml` (OK if script cds to SIMT root) |
| `smoketest.sh` | Requires `./Cargo.toml` and `crates/rustc-codegen-cuda/examples` — OK if run from `cuda-oxide/` |
| `check-device-only-build.sh` | Host crate names unchanged; fixture path `crates/cuda-macros/...` OK |
| `check-book-api-names.sh` | `DEVICE=crates/cuda-device/src` OK |
| `check-book-catalog-stamp.sh` | `crates/dialect-nvvm/...` OK |

CI must invoke them from `cuda-oxide/` (see E).

`cutile-rs/scripts/*` already treat `cutile-rs/` as `REPO_ROOT`. Keep that.
`run_cpu_tests.sh` / reactor jobs that `cargo test -p cuda-async` keep working
if the cutile workspace path dep points at `../cuda-async`.

### G. Justfile

Moves to `cuda-oxide/Justfile`. `cd crates/rustc-codegen-cuda` stays valid.
`-p cuda-core` / `-p cuda-async` / `-p cuda-bindings` recipes must either
call `cargo test --manifest-path ../Cargo.toml -p cuda-core` or move those
recipes to a root Justfile.

### H. Nix / devcontainer / cargo config

| File | Update |
| --- | --- |
| `flake.nix` | Crane/workspace source dir; comments `crates/rustc-codegen-cuda/examples/mathdx_ffi_test`; `rust-toolchain.toml` path |
| `.devcontainer/devcontainer.json` | `context: ".."` still repo root if the folder stays at `.devcontainer/`; rust-analyzer `linkedProjects` if any |
| `cuda-oxide/.cargo/config.toml` | Comment path to mathdx example; `[alias] oxide` must run against `cuda-oxide/Cargo.toml` |

### I. Interop example

`crates/rustc-codegen-cuda/examples/cutile_inter_kernel/Cargo.toml` depends on
cutile crates **by git**. After the layout lands, switch to path deps:

```toml
cutile = { path = "../../../../../../cutile-rs/cutile" }
# plus cutile-compiler, and shared cuda-async / cuda-core at repo root
```

(Exact `../` count: example dir is
`cuda-oxide/crates/rustc-codegen-cuda/examples/cutile_inter_kernel` → six
levels to repo root.)

### J. Documentation (string paths, not Cargo)

Prefix SIMT paths with `cuda-oxide/` in github blob URLs and in commands meant
to be run from the git root.

Highest-signal files:

- Root `README.md` (today) → become `cuda-oxide/README.md`; crate overview table
- `CONTRIBUTING.md` — `cd crates/rustc-codegen-cuda`, example paths
- `cuda-oxide-book/**` — `crates/...` links in
  `appendix/building-from-source.md`,
  `gpu-programming/virtual-memory-and-peer-access.md`,
  `gpu-programming/kernel-families.md`,
  `gpu-programming/launching-kernels.md`,
  `compiler/*`, `projects/async-mlp-pipeline.md`, book `README.md`
- Example READMEs that show
  `./crates/rustc-codegen-cuda/examples/...` as shell commands
- `crates/cuda-intrinsics-gen/README.md` generated-output paths

`check-crate-inventory.sh` and `check-book-api-names.sh` will fail until the
book/README tables match the new member list.

### K. rust-analyzer / toolchain

`rust-toolchain.toml` under `cuda-oxide/` applies when CWD is inside that
tree. Opening the **repository** root in the IDE will not pick the nightly pin
unless you add a root `rust-toolchain.toml` that forwards, or document
“open `cuda-oxide/` as the workspace”.

`crates/rustc-codegen-cuda/rust-toolchain.toml` stays next to that crate.

## Suggested order

1. Decide the shared-crate merge policy (unify vs rename).
2. `git mv` SIMT root files into `cuda-oxide/` (section 1). Confirm
   `cargo metadata --manifest-path cuda-oxide/Cargo.toml` still works
   (relative `crates/` paths unchanged).
3. Lift/merge `cuda-bindings`, `cuda-core`, `cuda-async` to the repo root;
   write the stable workspace `Cargo.toml`.
4. Point `cuda-oxide/Cargo.toml` and `cutile-rs/Cargo.toml` at `../cuda-*`.
5. Bulk-rewrite example `path = "../../../cuda-core"` (class A).
6. Fix `cargo-oxide` discovery and `cuda-intrinsics-gen` repo root (B, C).
7. Retarget GitHub workflows and CODEOWNERS (E).
8. Justfile host-crate recipes, flake, docs, cutile interop example (G–J).
9. `cargo metadata` on all three workspaces, then fmt/clippy/unit-tests/book
   equivalent of CI.

## What should keep working without a path edit

- Sibling `path = "../mir-importer"` inside `cuda-oxide/crates/*`
- `cuda-macros/tests/device-only` → `../../../cuda-device`
- `rustc-codegen-cuda/Cargo.toml` → `../mir-importer` (and other SIMT crates)
- `cutile-rs/cutile*/` internal paths
- `.gitmodules` submodule path
- `cutile-rs/scripts` that resolve `REPO_ROOT` as `cutile-rs/`
