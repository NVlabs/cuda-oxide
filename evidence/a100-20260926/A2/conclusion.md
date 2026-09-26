# A2 · #811 / #1305 — native A100 validation

**Status: `#1305` head validated on real A100/SM80. Correctness, sanitizers,
target gating and a paired measurement are all green. No new implementation
was written; the vendored implementation and the #1304 dependency are unchanged
and remain with their author.**

| item | value |
|---|---|
| PR | [#1305](https://github.com/NVlabs/cuda-oxide/pull/1305) `feat(cuda-device): derive an SM floor cfg and use redux.sync in warp_reduce` |
| validated HEAD | `cf4c55c38615d973cebf059211ab0d4624881bda` |
| dependency | [#1304](https://github.com/NVlabs/cuda-oxide/pull/1304) `6b4ac7d0147129c8a9c7ab86185e61b6b616c3fd` |
| merge base | `b9847e9515ed3a23096f22567d3eaf0a6e3e440c` |
| stack | 5 commits, `b9847e95..cf4c55c3` |
| hardware | 1× `NVIDIA A100-SXM4-40GB`, cc 8.0, driver `595.91.07`, no MIG |
| toolchain | rustc `nightly-2026-08-28`, `llc` LLVM `23.1.0-rust-1.100.0-nightly`, CUDA `13.0.88`, `libNVVM 2.0` |

## Scope audit — what was already covered

The author's body reports 1,395 CPU tests at `96ea0b56`, strict Clippy and fmt,
and B200 GPU runs at `db5c23f8` for `sm_80` and `sm_75`, including memcheck and
synccheck on the `sm_80` run. It also states plainly: *"GPU tests were not
repeated after the scanner rebase; no new speedup measurement was made."*

| area | author | this run |
|---|---|---|
| CPU selector / scanner / cfg regressions | covered | not repeated |
| device oracle, 9 integer pairs + f32 + sub-warp controls | A10G/B200 | **A100, this report** |
| memcheck / synccheck | B200 | **A100, this report** |
| `sm_75` compile + PTX fallback | B200 | **A100 host, this report** |
| paired measurement of the two forms | not made | **A100, this report** |
| malformed-hint regression | covered | not repeated |

Nothing here re-implements the architecture gating. The one artefact added is a
measurement harness (below); it does not change device behaviour.

## Correctness on A100

`crates/rustc-codegen-cuda/examples/warp_reduce_redux` (the PR's own oracle,
unmodified):

```
cargo oxide run      warp_reduce_redux --arch sm_80   -> SUCCESS
cargo oxide sanitize warp_reduce_redux --arch sm_80 --tool memcheck    -> 0 errors
cargo oxide sanitize warp_reduce_redux --arch sm_80 --tool synccheck   -> 0 errors
cargo oxide build    warp_reduce_redux --arch sm_75   -> ok
```

| case | expected | observed |
|---|---|---|
| `u32` full warp: sum / min / max / and / or / xor | `[496, 0, 31, 0, 0xffffffff, 0xffffffff]` | equal |
| `i32` full warp: sum / min / max over `-16..=15` | `[-16, -16, 15]` | equal |
| control `f32` full warp: sum / min / max | `[496.0, 0.0, 31.0]` | equal |
| control `WarpTile<16>`: per-tile sums | `[120, 120]` | equal |

The `i32` case is the one that matters for a single-instruction form: unsigned
min/max over the same bits would read `0 / 4294967295`, and the oracle rejects
that. The `f32` case is the negative control — there is no float `redux` below
`sm_100`, so a hook that claimed one would name an instruction that cannot
assemble. `WarpTile<16>` is the sub-warp control: a broken `N == 32` gate would
make both tiles answer 496.

## Emitted code, per target

`cargo oxide inspect warp_reduce_redux --arch sm_80` and `--arch sm_75`:

| kernel | sm_80 | sm_75 |
|---|---|---|
| `u32` / `i32` full warp | `redux.sync.*` — the reduction is one instruction | 5× `shfl.sync.bfly.b32` + 5× `add.s32` |
| `f32` full warp | `shfl.sync.bfly.b32` + `add.f32` (fallback) | same |
| `WarpTile<16>` | `shfl.sync.bfly.b32` pairs (fallback) | same |

`sfl`/`shfl` counts in the two module dumps are 19 vs 64, i.e. the `sm_80`
module keeps shuffles only where the fast path does not apply. This is the
"expected path vs observed path" check: the fast path is not merely requested by
a flag, it is what the emitted module contains.

For the measurement harness the same check was done per kernel, which is the
sharper form:

```text
.visible .entry shuffle_chain(...)   # 5.599 ms arm
    shfl.sync.bfly.b32  %r10, %r39, 16, 31, -1;
    add.s32             %r11, %r10, %r39;
    ... five rounds, no redux in this kernel ...
.visible .entry api_chain(...)       # 1.731 ms arm
    redux.sync.add.s32  %r10, %r20, %r9;
    redux.sync.add.s32  %r11, %r10, %r9;
    ... no shfl in this kernel ...
```

## Paired measurement

Harness: `crates/rustc-codegen-cuda/examples/warp_reduce_redux_bench` (new,
A100-only evidence; not part of the PR's device code). One source, two builds —
the fast path is chosen by the compile-time cfg, so there is no run-time switch
to hold constant.

* `api_chain` — `warp_reduce::<u32, Sum>` fed its own result: a dependency
  chain, so it exposes reduction latency.
* `shuffle_chain` — the same butterfly, written out with
  `WarpCollective::shfl_xor`. At `sm_80` this isolates the reduction from the
  rest of the build: same target, same compiler, same binary.
* `*_independent` — four unrelated reductions per iteration, so issue slots and
  the shuffle port have something to overlap.

128 threads (4 warps), 40,000 iterations, 15 samples per arm, 3 warm-up rounds,
all four arms sampled **round-robin within one pass** so clock drift lands on
every arm. Values are amortised per reduction over a multi-millisecond kernel,
so launch and host overhead are not in the number. The first iterations are
compared against a host simulation.

Each cell is `median ms / ns per reduction`; three independent batches:

| build | batch | api_chain | shuffle_chain | api_independent | shuffle_independent |
|---|---|---|---|---|---|
| `sm_80` (redux) | 1 | 1.731 / 10.8 | 5.599 / 35.0 | 2.384 / 3.7 | 7.718 / 12.1 |
| `sm_80` (redux) | 2 | 1.346 / 8.4 | 4.349 / 27.2 | 1.853 / 2.9 | 5.995 / 9.4 |
| `sm_80` (redux) | 3 | 1.732 / 10.8 | 5.599 / 35.0 | 2.384 / 3.7 | 7.718 / 12.1 |
| `sm_75` (butterfly) | 1 | 4.350 / 27.2 | 4.350 / 27.2 | 5.995 / 9.4 | 5.996 / 9.4 |
| `sm_75` (butterfly) | 2 | 4.349 / 27.2 | 4.349 / 27.2 | 5.995 / 9.4 | 5.995 / 9.4 |
| `sm_75` (butterfly) | 3 | 4.350 / 27.2 | 4.350 / 27.2 | 5.996 / 9.4 | 5.996 / 9.4 |

**Speedup, `sm_80`, paired inside one binary and one interleaved pass:**

| workload | redux (ns/reduction) | shuffle (ns/reduction) | ratio |
|---|---|---|---|
| dependent chain | 8.4 – 10.8 | 27.2 – 35.0 | **3.24×** (all three batches) |
| four independent reductions | 2.9 – 3.7 | 9.4 – 12.1 | **3.24 – 3.27×** |

**Cross-build, the PR's own framing** (`--arch sm_80` against `--arch sm_75`):
`api_chain` 8.4–10.8 ns against 27.2 ns, i.e. **2.5×–3.2×**.

The control is exact: at `sm_75` the API path and the hand-written butterfly
agree to the printed digit on all six cells (27.2 = 27.2, 9.4 = 9.4), which is
what a build where `HAS_FULL_WARP_FORM` is false must do. That is the evidence
that the harness measures the reduction and not something else.

This is a primitive measurement. The reduction is 1 instruction against 10, so
the ratio is bounded by Amdahl at the call site: a kernel that reduces once
after a memory-bound pass sees far less, and nothing here is an end-to-end
model or serving number.

## Residual issues with #1304 that this run did not close

The PR body lists two known scanner gaps: valid whitespace/comment-separated
opcode modifiers are still missed, and a quoted `.file` path is still treated as
an instruction. These are false negatives on a *float* `redux` in hand-written
module text, so the failure direction is "an sm_100a-only float redux could be
mistaken for an integer one" rather than a false positive on integer redux. They
live in the #1304 dependency and are not re-fixed here; the integer
false-positive case the PR was filed for is covered by the PR's own regression
and was not re-tested by this run.

## Evidence index

| file | content |
|---|---|
| `a2.log` | PR oracle: run `sm_80`, build `sm_75`, memcheck, synccheck |
| `a2bench2-full.log` | three interleaved batches, both builds |
| `bench-ptx-sm80.txt` | per-kernel PTX for the harness at `sm_80` |
| `a2-ptx-sm80.txt`, `a2-ptx-sm75.txt` | module dumps for the PR oracle |

## Limitations

- One A100 SKU, one MIG mode (`none`), one driver. Not A10G, RTX 5090 or B200.
- The measurement is a microbenchmark of the primitive. Registers, occupancy and
  SASS were **not** captured: `cargo oxide inspect` stops at PTX, the driver JITs
  the PTX at load, and no `cuobjdump`-visible cubin was extracted. This run
  therefore reports an executed-PTX path, not executed SASS.
- The `sm_75` build is compile/PTX coverage on an A100 host. It is not a Turing
  GPU run and is not reported as one.
- No CI claim: `just check`, the full example matrix and the libNVVM route were
  not re-run for this head.
