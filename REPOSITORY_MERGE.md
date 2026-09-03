# Repository merge notes

This document records the agreed? procedure for consolidating Rust repositories
into this repository. The first source is
[`NVlabs/cutile-rs`](https://github.com/NVlabs/cutile-rs).

Set these paths once before running any commands. Replace the example values
with the locations of your own checkouts:

```bash
REPO_ROOT=/path/to/cuda-oxide
CUTILE_RS_ROOT=/path/to/cutile-rs
REPO_ROOT=$(pwd)
CUTILE_RS_ROOT=$(realpath $(pwd)/../cutile-rs)
```

## External planning material

- [Repository-merge planning document](https://docs.google.com/document/d/1SMKq7F3oDCfSgq1TTN11CJ2GYs-p5Meip82Psh1l54E/edit?usp=sharing)
- [Repository-merge tracking spreadsheet](https://docs.google.com/spreadsheets/d/19_fQLnhXxoPwpouQJiRxOnu8izbW6VjJgQ1dZvG2k90/edit?usp=sharing)

## Repo layout

```
cuda-rust/
├── README.md
├── Cargo.toml             # stable workspace
├── cuda-bindings/         # shared published crates
├── cuda-core/
├── cuda-async/
├── cutile-rs/ # Tile model, everything except shared host crates
│   ├── cutile/ # published DSL crate
│   ├── cutile-macro/ # module macro
│   ├── cutile-compiler/ # kernel compilation
│   ├── cutile-ir/ # Tile IR representation + bytecode writer
│   ├── cutile-kernels/ # reusable kernels
│   ├── cuda-tile-rs/ # wrapper around the cuda-tile C++ library
│   ├── cutile-benchmarks/ # unpublished
│   ├── cutile-examples/ # unpublished
│   ├── cutile-book/
│   ├── assets/
│   └── scripts/
├── cuda-oxide/ # SIMT model
│   └── rust-toolchain.toml
│   ├── Cargo.toml
│   ├── Justfile
│   ├── crates/
│   │   ├── cuda-host/
│   │   ├── cuda-device/
│   │   ├── cuda-macros/
│   │   ├── cuda-intrinsics/
│   │   ├── cuda-intrinsics-gen/
│   │   ├── rustc-codegen-cuda/
│   │   ├── mir-importer/
│   │   ├── mir-lower/
│   │   ├── mir-transforms/
│   │   ├── dialect-mir/
│   │   ├── dialect-nvvm/
│   │   ├── nvvm-transforms/
│   │   ├── llvm-export/
│   │   ├── cuda-oxide-codegen/
│   │   ├── libnvvm-sys/
│   │   ├── nvjitlink-sys/
│   │   ├── cargo-oxide/
│   │   ├── cuda-artifact-finalizer/
│   │   ├── oxide-artifacts/
│   │   ├── reserved-oxide-symbols/
│   │   └── fuzzer/
│   ├── cuda-oxide-book/
│   ├── intrinsics/
│   ├── assets/
│   └── scripts/
└── .github/workflows/ # path-filtered lanes
```

## Current decisions

- `cuda-oxide` is the target repository.
- Import `cutile-rs` beneath `cutile-rs` while preserving its commit history.
- Preserve source tags using the `cutile-rs/` namespace. This avoids conflicts
  with target tags such as `v0.1.0` and `v0.2.0`.
- Preserve source branches as `cutile-rs/*` remote-tracking refs during the
  local integration. Decide which should become permanent archival refs before
  the final merge.
- Reconcile `CODEOWNERS` and GitHub Actions only after the history import.
  `filter-repo` places those files under `cutile-rs/.github`; they do
  not automatically affect the target repository's root `.github` directory.
- After the initial import, bring later `cutile-rs` commits in with
  `format-patch` / `git am --directory=cutile-rs` (see below). Do not
  repeat the unrelated-histories `filter-repo` merge.
- Landing order onto **NVlabs/cuda-oxide `main`** (draft, needs review):
  [MERGE_TO_MAIN.md](MERGE_TO_MAIN.md). Preserve cutile-rs history; land the
  rest as a **series of stacked PRs**. Private is not the destination.

## Prerequisite

Install `git-filter-repo` on macOS:

```bash
brew install git-filter-repo rust
git filter-repo --version
```

## Local integration import

These commands perform a real local history rewrite and a real local merge. The
target's `main` branch remains untouched only because the merge happens on a
dedicated integration branch. Nothing is pushed unless a later publication
command is run.

> Warning: `git filter-repo --force` rewrites the local `cutile-rs` clone in
> place, including its branches and tags. Re-clone that repository afterwards
> if an unmodified checkout is needed.

First verify both checkouts are clean:

```bash
git -C "$REPO_ROOT" status --short
git -C "$CUTILE_RS_ROOT" status --short
```

Force-reset the source clone to GitHub `main` so the rewrite starts from the
current upstream tip. `git filter-repo` removes remotes, so `main` in
`$CUTILE_RS_ROOT` has no upstream until `origin` is restored. It also leaves
local tags (`v0.0.1`, `v0.1.0`, …) pointing at rewritten commits. A plain
`fetch --tags` then rejects those names (`would clobber existing tag`).
`--force` overwrites them with the GitHub tags. This discards uncommitted and
unpushed work in that clone, including local tag targets. Skip `remote add` if
`origin` already exists.

```bash
git -C "$CUTILE_RS_ROOT" remote add origin \
  https://github.com/NVlabs/cutile-rs.git

git -C "$CUTILE_RS_ROOT" fetch origin --prune --tags --force
git -C "$CUTILE_RS_ROOT" switch main
git -C "$CUTILE_RS_ROOT" reset --hard origin/main
```

Rewrite the source history into its destination directory.

If `filter-repo` asks whether to treat this as a continuation of a previous
run (`already_ran` older than a day), answer **N**. That marker is leftover
from an earlier rewrite; continuing would reuse the old commit/ref maps on
history that was just reset to GitHub.

```bash
git -C "$CUTILE_RS_ROOT" filter-repo \
  --force \
  --to-subdirectory-filter cutile-rs
```

Create the dedicated integration branch, fetch the rewritten history, and
merge it:

```bash
git -C "$REPO_ROOT" switch -c dry-run/cutile-rs-2

git -C "$REPO_ROOT" remote add cutile-rs "$CUTILE_RS_ROOT"

git -C "$REPO_ROOT" fetch cutile-rs \
  '+refs/heads/*:refs/remotes/cutile-rs/*' \
  '+refs/tags/*:refs/tags/cutile-rs/*'

git -C "$REPO_ROOT" merge --no-ff --allow-unrelated-histories \
  -m 'Import cutile-rs history under cutile-rs' \
  cutile-rs/main
```

Inspect the result before changing the Cargo workspace or root GitHub files:

```bash
git -C "$REPO_ROOT" diff --stat main...dry-run/cutile-rs-2
git -C "$REPO_ROOT" log --graph --oneline --decorate --all
git -C "$REPO_ROOT" log --graph --decorate --oneline --date-order --all
git -C "$REPO_ROOT" tag --list 'cutile-rs/*'
git -C "$REPO_ROOT" fsck --full

cd "$REPO_ROOT"
? cargo metadata --no-deps --format-version 1
```

To discard the local integration branch:

```bash
git -C "$REPO_ROOT" switch main
git -C "$REPO_ROOT" branch -D dry-run/cutile-rs-2
git -C "$REPO_ROOT" remote remove cutile-rs
```

## Pulling later changes from `cutile-rs`

After the initial import, do **not** re-run `filter-repo` and merge with
`--allow-unrelated-histories` again. A second rewrite produces new commit IDs
for the same history, so Git treats it as a second unrelated root and will
conflict across `cutile-rs`.

Instead, keep a **pristine** (unfiltered) clone of
[`NVlabs/cutile-rs`](https://github.com/NVlabs/cutile-rs) and replay only the
new upstream commits into `cuda-oxide` with path rewriting.

Set an extra path for that pristine checkout (distinct from any filter-repo
mirror used during the initial import):

```bash
CUTILE_RS_UPSTREAM_ROOT=/path/to/cutile-rs-pristine
```

### Record what was already imported

At import time the filtered tip was merged as `cutile-rs/main`. Map that tip
back to the matching commit on the **unfiltered** upstream `main` (same subject
line / PR number). Record it:

```bash
# Example: after the first import, the filtered tip subject was
#   build(deps): bump the all-actions group with 2 updates (#198)
# On pristine upstream that corresponds to a commit such as:
CUTILE_RS_LAST_IMPORTED=<upstream-sha-for-last-imported-commit>
```

Re-check the mapping whenever upstream force-pushes `main` (subject and PR
number usually survive; the SHA may not).

### Apply new commits onto an integration branch

```bash
git -C "$CUTILE_RS_UPSTREAM_ROOT" fetch origin --prune --tags
git -C "$CUTILE_RS_UPSTREAM_ROOT" switch main
git -C "$CUTILE_RS_UPSTREAM_ROOT" merge --ff-only origin/main

# Inspect what would be imported.
git -C "$CUTILE_RS_UPSTREAM_ROOT" log --oneline \
  "$CUTILE_RS_LAST_IMPORTED"..main

git -C "$REPO_ROOT" switch -c sync/cutile-rs-$(date +%Y%m%d) dry-run/cutile-rs-2

git -C "$CUTILE_RS_UPSTREAM_ROOT" format-patch --stdout \
  "$CUTILE_RS_LAST_IMPORTED"..main \
  | git -C "$REPO_ROOT" am --3way --directory=cutile-rs
```

If `git am` stops on a conflict, resolve files under `cutile-rs`, then
`git am --continue`. To abort: `git am --abort`.

If a patch touches `.gitmodules`, it lands as `cutile-rs/.gitmodules`.
Git only reads `.gitmodules` at the repository root, so fold the change into the
root file with re-prefixed paths and drop the nested copy — see
[Submodules](#submodules).

Fetch any new upstream tags into the namespaced tag space (same convention as
the initial import):

```bash
git -C "$REPO_ROOT" fetch "$CUTILE_RS_UPSTREAM_ROOT" \
  '+refs/tags/*:refs/tags/cutile-rs/*'
```

Only tags that do not already exist locally are added; existing `cutile-rs/*`
tags are left alone unless you deliberately retarget them.

### Verify, then advance the bookmark

```bash
git -C "$REPO_ROOT" diff --stat dry-run/cutile-rs-2...HEAD
git -C "$REPO_ROOT" log --oneline dry-run/cutile-rs-2..HEAD
git -C "$REPO_ROOT" fsck --full

cd "$REPO_ROOT"
cargo metadata --no-deps --format-version 1
```

When the sync branch looks good, update the recorded tip to the upstream commit
you just imported (usually `origin/main` at the time of the sync), merge or
publish the sync branch through the normal review path, and reuse that new SHA
as `CUTILE_RS_LAST_IMPORTED` next time.

### Why this shape

| Approach | Use? |
| --- | --- |
| `format-patch` / `am --directory=cutile-rs` | Yes — incremental updates after the first import |
| Re-`filter-repo` + `merge --allow-unrelated-histories` | No — duplicates rewritten history |
| Copy working tree files without commits | No — drops attribution and bisectability |

## Submodules

`filter-repo --to-subdirectory-filter` moves `.gitmodules` along with every
other file, but it does not rewrite the `path` values inside it, and Git only
reads `.gitmodules` from the repository root. An imported submodule therefore
arrives with a tracked gitlink at `cutile-rs/<path>` and no mapping Git
can see, which breaks any recursive checkout:

```text
fatal: No url found for submodule path 'cutile-rs/cuda-tile-rs/cuda-tile' in .gitmodules
```

The root `.gitmodules` is the single source of truth. Each imported entry needs
both its section name and its `path` prefixed with the destination directory,
while `url` stays as upstream had it:

```ini
[submodule "cutile-rs/cuda-tile-rs/cuda-tile"]
	path = cutile-rs/cuda-tile-rs/cuda-tile
	url = https://github.com/NVIDIA/cuda-tile.git
```

Verify with the same commands the `actions/checkout` step runs:

```bash
git -C "$REPO_ROOT" submodule sync --recursive
git -C "$REPO_ROOT" -c protocol.version=2 submodule update --init --force --recursive
git -C "$REPO_ROOT" submodule status
```

## Follow-up work before a final merge

1. Add the imported crate to the root Cargo workspace and resolve package-name,
   feature, dependency, lockfile, MSRV, and CUDA-toolchain conflicts.
2. Merge the source and target `CODEOWNERS` rules, translating source paths to
   `cutile-rs/...` and checking owner-team access.
3. Compare and consolidate GitHub workflows, including triggers, permissions,
   runners, secrets, cache keys, required check names, and release automation.
4. Retain all licensing, NOTICE, third-party attribution, Git LFS, submodule
   (see [Submodules](#submodules)), generated-file, and build-environment
   requirements.
5. Recreate GitHub Releases separately: release records, assets, checksums,
   signatures, and notes are not transferred by Git history or tags.
6. Validate the final integration with formatting, clippy, tests, documentation,
   CI-equivalent jobs, `git fsck`, history inspection, and tag/asset checksums.

## Private target repository

`nvidia-dev/cuda-rust-private` cannot be an official GitHub fork of the public
`NVlabs/cuda-oxide` repository while remaining private. Use it as a standalone
private repository with `NVlabs/cuda-oxide` configured as an `upstream` remote.
Pushing to it is a separate, deliberate step and is not part of the local
integration procedure.

### Add the private remote

Add the private repository under the unambiguous name `private` (do not replace
the existing `origin` or `upstream` remotes during the migration):

```bash
git -C "$REPO_ROOT" remote add private \
  git@github.com:nvidia-dev/cuda-rust-private.git

git -C "$REPO_ROOT" fetch private --prune
git -C "$REPO_ROOT" remote -v
```

### Initial publication: original `main` and the import branch

For the initial private-repository setup, publish the original `main` without
changing it, then publish the in-progress import branch separately. At the time
of writing, the current import branch is `dry-run/cutile-rs-2`.

Fetch first so that a non-empty private repository is detected instead of being
overwritten. Each push below is non-forced and will fail safely if the remote
branch has unrelated commits.

```bash
git -C "$REPO_ROOT" fetch private --prune

# Publish the existing target history exactly as main.
git -C "$REPO_ROOT" push private main:main

# Publish only the target repository's pre-import release tags.
git -C "$REPO_ROOT" push private \
  v0.1.0 v0.2.0 v0.2.1

# Publish the current integration result without making it private/main.
git -C "$REPO_ROOT" push --set-upstream private \
  HEAD:dry-run/cutile-rs-2
```

Do not use `git push private --tags` for this initial publication: it would
also publish the namespaced `cutile-rs/*` tags before their release migration
has been reviewed.

### Synchronize safely

Before starting work, incorporate any changes that were made directly in the
private repository:

```bash
git -C "$REPO_ROOT" fetch private --prune
git -C "$REPO_ROOT" switch main
git -C "$REPO_ROOT" merge --ff-only private/main
```

After a reviewed local change is merged into `main`, publish it:

```bash
git -C "$REPO_ROOT" push private main
git -C "$REPO_ROOT" push private --follow-tags
```

To bring in new changes from the public upstream, fetch first, review the
range, then merge it through the normal review process. Do not push directly to
the private repository until the merge has been validated.

```bash
git -C "$REPO_ROOT" fetch upstream --prune --tags
git -C "$REPO_ROOT" log --oneline main..upstream/main
# Create a review branch, merge or rebase upstream/main there, validate, then
# merge the reviewed result into main and push it to private.
```
