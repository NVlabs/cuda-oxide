# Merging onto NVlabs/cuda-oxide `main`

> **Draft — needs review.** This playbook is not agreed procedure yet. Do not
> treat it as the source of truth for landing until someone has reviewed it
> against [LAYOUT_MIGRATION.md](LAYOUT_MIGRATION.md),
> [REPOSITORY_MERGE.md](REPOSITORY_MERGE.md), and how NVlabs PRs will actually
> be opened.

High-level landing order for the cutile-rs + layout work. **Destination is
[`NVlabs/cuda-oxide`](https://github.com/NVlabs/cuda-oxide) `main`.** Path
inventory and `git mv` detail stay in [LAYOUT_MIGRATION.md](LAYOUT_MIGRATION.md).
History import commands stay in [REPOSITORY_MERGE.md](REPOSITORY_MERGE.md). Do
not treat this file as a second layout.

**Private vs NVlabs:** `nvidia-dev/cuda-rust-private` is a rehearsal copy and
the place to try NVIDIA self-hosted jobs first. Compare Oxide behavior to
NVlabs `main`, not private `main`. **Publish the same product tree to NVlabs**,
including `.github/copy-pr-bot.yaml`.

`cutile-rs.yml` lands on NVlabs `main` in step 3. Cutile CI **must** run on
NVIDIA **private/self-hosted** runners (`linux-amd64-cpu16` + copy-pr-bot
`pull-request/<n>`). Do not replace it with GitHub-hosted `ubuntu-latest`.
The copy-pr-bot **GitHub App** must be installed on `NVlabs/cuda-oxide` or
the yaml is inert.

**Stacked PRs onto NVlabs:** land this work as a **series of stacked PRs**, not
one mega-PR and not a pile of unrelated branches off `main`. Each numbered
step in this document is one PR. PR *n+1* is based on PR *n*’s branch (GitHub
stacked-PR / “base this PR on that branch”). Merge in order: 1, then 2, then
3, … After a PR merges, retarget the rest onto `main` (or the new stack
base) so the series stays linear.

`dry-run/cutile-rs-2` may already contain several steps on **one** branch.
That is rehearsal only. Slice it into the stacked series before opening
NVlabs PRs. Mixing two steps in one PR makes a red CI run hard to blame.

**Do not** `git rebase` `dry-run/cutile-rs-2` onto `upstream/main`. That
replays the cutile import and the SIMT nest onto a tree that still has
`crates/` and rewrites hundreds of SHAs. How to replay instead:
`.cursor/rules/cuda-rust-upstream.mdc` (`--rebase-merges`, backup ref). If a
replay drops `copy-pr-bot.yaml` or `cutile-rs.yml`, restore them — they belong
on NVlabs `main` once those steps have landed.

After every step: merge that stacked PR only when its checks are green
(Oxide jobs on GitHub-hosted runners; **cutile jobs on private hosted
runners**). Do not merge PR *n+1* while PR *n* is red or unmerged.

## History and PR shape (hard rules)

**Preserve cutile-rs history on NVlabs `main`.** The tile tree is not a
one-shot copy. NVlabs must keep:

- The `Import cutile-rs` **merge** (filtered history as second parent). Do
  not squash, flatten, or rebase that merge away.
- Later NVlabs/cutile-rs commits as **their own first-parent commits** under
  `cutile-rs/` (`git am --directory=cutile-rs`), in upstream order, including
  patches whose files are later deleted. Land those **below** `add merge doc`,
  not onto current `HEAD`.
- `cutile-rs/v*` tags as ancestors of `main` (retarget after rewrites; never
  delete).

Any rewrite onto `upstream/main` uses `--rebase-merges` so that import merge
survives. A squash merge of the import PR, or a linear “add `cutile-rs/`”
commit, is a failed landing.

**Everything else is atomic and logically organized.** Each stacked PR after
the import is one concern: copy-bot yaml first; then workflows; then a
`git mv` nest; then path retargets; then a green fix (disk, path depth);
then bindings unify; then drop nested `cutile-rs/cuda-*`. Do not combine
unrelated trees (Oxide nest + cutile layout + CI glue) in one PR. Squash is
fine **for those PRs**; it is not fine for the cutile import.

---

## 0. Preconditions (before step 1)

- [ ] `upstream` remote is [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide);
      `private/main` is a rehearsal copy, not the source of truth.
- [ ] Integration work is sliced into the stacked PR series (each numbered
      step its own branch, PR *n+1* based on PR *n*). The rehearsal branch
      (`dry-run/cutile-rs-2`) is not opened as a single NVlabs PR.
- [ ] `cutile-rs/v*` tags exist and are ancestors of the integration branch
      (never delete; never leave orphaned). Push those tags to NVlabs when the
      import lands there (`cutile-rs/*` namespace).
- [ ] NVlabs merge settings for the **import** PR: merge commit (or rebase
      with merge preservation). **Disable squash** for that PR.
- [ ] No `git push` to `main`/`master` without an explicit request; no force
      on those branches unless force-push was named.

---

## 1. Copy-PR bot on NVlabs `main`

NVIDIA self-hosted runners do not run `pull_request` events. copy-pr-bot
mirrors a trusted PR to `pull-request/<n>`; later `cutile-rs.yml` listens for
that `push`.

Land **only** `.github/copy-pr-bot.yaml` on NVlabs `main`
(`enabled: true`, `auto_sync_draft: false`, `auto_sync_ready: true` unless
policy changes). Do not add `cutile-rs.yml` in the same PR if `cutile-rs/` is
not on `main` yet. Do not add Oxide layout moves.

Install the copy-pr-bot GitHub App on **NVlabs/cuda-oxide** (same config
shape as private). Rehearse on private first if useful; the NVlabs PR is the
one that counts.

### Checks after step 1

- [ ] `.github/copy-pr-bot.yaml` is on **NVlabs** `main` (`enabled: true`).
- [ ] App is installed on `NVlabs/cuda-oxide` (yaml alone does not mirror).
- [ ] PR is yaml-only (no `cutile-rs/`, no SIMT `git mv`).
- [ ] A ready PR against NVlabs produces `pull-request/<n>`. Drafts wait
      for `/ok to test`.

**Status:** yaml already on private; still needs the NVlabs PR + app install.

---

## 2. Merge cutile-rs (history + tree) onto NVlabs

This is the history-preserving landing. Product files may still be the
un-unified tree. Do not sneak layout or workflow rewrites into this PR.

- Merge commit: `Import cutile-rs` with filtered history as second parent.
  Keep `--rebase-merges` on any later rewrite so that merge is not flattened.
- First-parent after import: NVlabs/cutile-rs syncs (`format-patch` /
  `git am --directory=cutile-rs`) **below** `add merge doc`. CI and Oxide
  layout are **later PRs** (steps 3–6), not this merge.
- Nested `cutile-rs/cuda-{bindings,core,async}` stay until step 6.
- Do not 3-way cutile `.github` files into root Oxide workflows.
- Push `cutile-rs/v*` tags to NVlabs with the import (do not clobber Oxide
  `v0.*` tags).

Rehearse the same import on private first if you want; the NVlabs PR is the
one that counts.

### Checks after step 2

- [ ] `cutile-rs/` exists on **NVlabs** `main`; `cutile-rs/.github/` is inert
      (root `.github` still owns Actions).
- [ ] `git merge-base --is-ancestor <cutile-rs/v*> HEAD` for every
      `cutile-rs/*` tag on that `main`.
- [ ] `git cat-file -p HEAD` (or the import commit) shows **two parents**.
      `git log --first-parent` still shows Oxide `main` as the left side;
      `git log <import>^2` is the filtered cutile history.
- [ ] First-parent log: `Import cutile-rs` → cutile syncs → `add merge doc`
      (match by **subject** if SHAs moved). Each sync is its own commit, not
      folded into the merge.
- [ ] NVlabs Oxide jobs that do not need the tile tree still pass (fmt,
      clippy, unit-tests, cargo-deny, examples-compile, CodeQL if required).
      If examples-compile hits disk (`ENOSPC`), that is a step 3/5 workflow
      fix, not a reason to delete the import.
- [ ] Import PR diff does not nest SIMT, retarget Oxide workflows, or delete
      `cutile-rs/cuda-*`.
- [ ] CodeQL “new” alerts on `#[cuda_module]` after a later `git mv` are
      path-key noise; do not “fix” kernels for that unless asked.

---

## 3. Workflows and related repo glue

One (or few) PRs whose **only** subject is CI/CODEOWNERS/book/deny. No
`git mv` of SIMT, no crate unify.

Root `.github/` stays at the git root. After cutile is on NVlabs `main`, ship
CI glue there (copy-pr-bot yaml already landed in step 1):

| Piece | On NVlabs |
| --- | --- |
| `book.yml` cutile book | Yes — versioned book uses `cutile-rs/v*` |
| `CODEOWNERS` | Yes — paths after nest/lift |
| cargo-deny / SPDX | Yes — include `cutile-rs/` workspace |
| examples-compile | Yes — drop `cutile-rs/` on the hosted disk |
| status-guard / naming-guard / scripts | Yes — `cuda-oxide/` once nested (step 4) |
| `cutile-rs.yml` (private hosted + `pull-request/<n>`) | Yes — NVIDIA self-hosted only; pairs with copy-pr-bot from step 1 |

Also: submodule checkout (`cutile-rs/cuda-tile-rs/cuda-tile`), rust-toolchain
at git root. Nested `cutile-rs/.github/workflows` stay unused.

`cutile-rs.yml` stays `push` to `pull-request/[0-9]+` until cutile is on
`main`; then you may add `main`+`paths`. copy-pr-bot pushes often have empty
path diffs — do not path-filter the mirror trigger.

### Checks after step 3

- [ ] PR diff is workflows / CODEOWNERS / deny / book only (no SIMT `git mv`,
      no deleting `cutile-rs/cuda-*`).
- [ ] NVlabs `ci.yml` (fmt, clippy, unit-tests, cargo-deny) green.
- [ ] NVlabs examples-compile green **with** `cutile-rs/` in the default
      checkout (job drops that tree if disk is tight).
- [ ] NVlabs book job still publishes Oxide; cutile book succeeds once
      `cutile-rs/v*` tags are on NVlabs.
- [ ] `cargo deny` / license scripts cover every workspace claimed in NVlabs
      CI (root, SIMT, cutile, rustc-codegen-cuda, device-only fixture).
- [ ] **NVlabs `main` still has** `copy-pr-bot.yaml` (step 1); this PR adds
      `cutile-rs.yml` (private hosted), not a second copy-bot change.
- [ ] Ready PR / `/ok to test` on NVlabs mirrors to `pull-request/<n>` and
      **cutile-rs-pr** runs green (build, fmt, clippy, test --no-run, CPU
      tests, reactor).

---

## 4. Move Oxide crates according to the layout plan

Atomic `git mv` PRs (nest SIMT, then lift host crates), not mixed with cutile
unify or unrelated CI. Canonical moves:
[LAYOUT_MIGRATION.md](LAYOUT_MIGRATION.md). Target sketch:
[REPOSITORY_MERGE.md](REPOSITORY_MERGE.md) (Repo layout).

**Today on the dry-run (not yet the final sketch):** git-root workspace
`Cargo.toml`; host crates `cuda-bindings/`, `cuda-core/`, `cuda-async/`
**beside** `cuda-oxide/`; SIMT under `cuda-oxide/crates/`. Do **not** move
the workspace `Cargo.toml` under `cuda-oxide/` while those three are still
siblings.

Remaining toward the sketch (after unify/rename in step 6, or in parallel
once policy is fixed):

1. Split workspaces: SIMT `Cargo.toml` / `Cargo.lock` / Justfile /
   `rust-toolchain.toml` live under `cuda-oxide/` only.
2. Root workspace members are **only** the three shared host crates.
3. `cutile-rs/Cargo.toml` path-deps `../cuda-*` (step 6 deletes nested copies).
4. Path-filter CI lanes: `cuda-oxide/**`, `cutile-rs/**`, `cuda-bindings/**`,
   `cuda-core/**`, `cuda-async/**`.

Unify vs rename for `cuda-core` is decided **before** deleting either copy
(spike: bindings `load`/`curand` on the root crate; then peel SIMT deps from
Oxide `cuda-core`).

### Checks after step 4

- [ ] `git mv` only (history follows). No second `crates/` at repo root.
- [ ] `Import cutile-rs` is still a two-parent merge on NVlabs `main` after
      the nest (replay used `--rebase-merges`; no squash of the import).
- [ ] `cargo metadata` for the workspace(s) you actually have:
      - while host crates are siblings: root `Cargo.toml`
      - after split: `--manifest-path cuda-oxide/Cargo.toml` **and** root
        stable workspace
- [ ] Oxide intra-`crates/` `path = "../mir-importer"` still resolves
      (nesting SIMT does not break those; lifting host crates out **does** —
      retarget examples, see layout doc class A).
- [ ] `cuda-intrinsics-gen` and `cargo-oxide` treat `cuda-oxide/` as SIMT
      root (`../..` from that crate, not `../../..`).
- [ ] Justfile `cd crates/rustc-codegen-cuda` is relative to `cuda-oxide/`.
- [ ] Workflows use `cuda-oxide/crates/...` and `cuda-oxide/scripts/...`.
- [ ] One `rust-toolchain.toml` at git root is enough until the split;
      then pin lives next to the SIMT workspace.

---

## 5. Fixes to keep workflows green

Separate PRs, one failure class each (disk, path depth, SPDX, …). Do not
bundle them into the nest or the import.

Anything that is not a layout move but unblocks CI after steps 2–4:

- Example `cuda-core` path depth after nest (`../../../` vs
  `../../../../../`).
- examples-compile disk (`rm -rf cutile-rs`, `CARGO_INCREMENTAL=0`).
- Root `cuda-bindings` default **link** path vs cutile `load`/`curand`
  (Linux toolkit only; do not validate bindgen on macOS).
- cargo-deny SPDX / `dependency-licenses.csv` for new members.
- Naming/status-guard scripts and book inventory tables after member list
  changes.
- Private only: draft PRs need `/ok to test` after history rewrites
  (copy-pr-bot mirror is not an ancestor of the new tip).

Do not use CodeQL path-key churn as a kernel cleanup task.

### Checks after step 5

- [ ] Full **NVlabs** gate green: GitHub-hosted Oxide jobs (`ci.yml`
      children, examples-compile, book, CodeQL if required) **and** private
      hosted **cutile-rs-pr**.
- [ ] `cargo test -p cuda-bindings` (toolkit_target tests) on Linux CI.
- [ ] `cargo check -p cuda-bindings` (default features, no `load`) and
      `cargo check --manifest-path cutile-rs/Cargo.toml -p cuda-core`
      (with `load`+`curand`) on Linux + CUDA toolkit.
- [ ] No `cutile-rs/` left on the examples-compile workspace after the drop
      step (job log).
- [ ] Inventory/license/SPDX scripts match the tree you shipped.

---

## 6. cutile-rs layout and remaining tile fixes

Own PR(s) after steps 4–5. Do not fold “delete nested `cuda-core`” into the
Oxide nest.

After shared crates are one tree at the repo root (unify or rename):

- Remove `cutile-rs/cuda-bindings`, `cutile-rs/cuda-core`,
  `cutile-rs/cuda-async`.
- Point `cutile-rs/Cargo.toml` at `../cuda-*` (features `load`/`curand` on
  bindings as needed).
- Fold `cutile-rs/deny.toml` / flake if that is still open; leave nested
  `.github` inert.
- Switch Oxide cutile interop example from git deps to path deps
  (`LAYOUT_MIGRATION.md` §I).
- Incremental NVlabs cutile commits: `am --directory=cutile-rs` **below**
  `add merge doc`, not onto current `HEAD` (`.cursor/rules/cutile-rs-sync.mdc`).
- Retarget `cutile-rs/v*` after any rewrite of those commits.

### Checks after step 6

- [ ] `cutile-rs/Cargo.toml` members do **not** include the three host
      crates; only one package name `cuda-core` / `cuda-bindings` /
      `cuda-async` in the combined workspaces.
- [ ] `ls cutile-rs/cuda-bindings` (and core/async) fails — copies gone.
- [ ] Cutile workspace green on NVlabs **private hosted** `cutile-rs.yml`:
      `working-directory: cutile-rs`, `cuda-async` / `cuda-core` resolve
      via `../`.
- [ ] Reactor jobs (`loom_`, miri `slot_table`, TSan) still `-p cuda-async`
      on those same private hosted runners.
- [ ] Orphan tag loop clean:

      ```bash
      for t in $(git tag -l 'cutile-rs/*'); do
        git merge-base --is-ancestor "$t" HEAD || echo "ORPHAN $t"
      done
      ```

- [ ] Interop example builds against in-tree cutile (path deps).
- [ ] First-parent vs local NVlabs clone (`../cutile-rs`) still matches
      for the imported range (vimdiff recipes in the cutile-sync Cursor rule).

---

## Done when

**[NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) `main`** matches
the sketch in `REPOSITORY_MERGE.md`, **cutile-rs history is still a merge plus
per-commit syncs**, copy-pr-bot + cutile self-hosted lane are on that `main`,
and NVlabs-hosted Oxide CI is green.
