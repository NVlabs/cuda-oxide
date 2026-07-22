# cuda-oxide fuzzer support

`crates/fuzzer` contains the reusable pieces for rustlantis-based differential
codegen testing:

- `src/trace.rs`: the `no_std` trace API used by both CPU and GPU runs.
- `rustlantis/`: vendored upstream rustlantis, used as a MIR program generator.
- `tools/mir_generator.py`: adapts one rustlantis seed into a cuda-oxide smoke case.
- `tools/run_seed.py`: generates a seed, injects it into `rustlantis-smoke`, and runs it.

The execution harness is still the example at
`crates/rustc-codegen-cuda/examples/rustlantis-smoke`. The fuzzer tools rewrite
only `src/generated_case.rs`; `src/main.rs` remains the stable CPU/GPU harness.

## Basic usage

Run one seed:

```bash
python3 crates/fuzzer/tools/run_seed.py --seed 192
```

Run a range:

```bash
python3 crates/fuzzer/tools/run_seed.py --start 0 --count 20 --keep-going --keep-logs
```

The seed controls rustlantis' pseudo-random generator. Same seed plus same
rustlantis config produces the same custom-MIR program, which makes failures
reproducible.

## What gets compared

For each accepted seed:

1. rustlantis generates a Rust/custom-MIR function.
2. `mir_generator.py` rewrites rustlantis' `dump_var(...)` calls into the
   generic `fuzzer::dump_var(...)` trace API.
3. `rustlantis-smoke` runs the same generated case on the CPU and GPU.
4. The CPU and GPU traces are compared as `u64` hashes.

`dump_var` hashes intermediate values, not just the final return value. A seed
can have one dump site or several dump sites. Seed `162` is the current checked
in example, because its device code calls libdevice (`fmaf64`) and so covers the
artifact path that a PTX-only loader cannot serve. Seed `192` is a smaller case
with two dump sites:

```rust
__rl_dump0 = (Move(_1), Move(_2), Move(_3), Move(_4));
Call(_9 = dump_var(Move(__rl_dump0)), ReturnTo(bb4), UnwindUnreachable())

__rl_dump1 = (Move(_6),);
Call(_9 = dump_var(Move(__rl_dump1)), ReturnTo(bb5), UnwindUnreachable())
```

## Result statuses

- `PASS`: The adapter produced a case, both CPU and GPU runs completed, and the
  trace hashes matched.
- `MISMATCH`: Both CPU and GPU runs completed, but the trace hashes differed.
  This is the highest-priority result because it can indicate a backend
  correctness bug.
- `COMPILE_FAIL [backend]`: The adapter produced a case, but cuda-oxide failed
  while compiling or running it. The log records the backend reason and includes
  the generated `generated_case.rs` snapshot.
- `UNSUPPORTED [adapter]`: rustlantis generated a MIR program, but our Python
  adapter refused to turn it into a cuda-oxide smoke case.

For example, seed `0` dumps a `u128`, which the adapter once refused; the
trace API has since widened and the seed currently reports:

```text
seed 0: PASS

results:
  seed 0: PASS [run] CPU/GPU traces matched
summary: PASS=1
```

The typical `UNSUPPORTED [adapter]` cause is a generated `dump_var(...)` call
or function signature that uses a type the adapter cannot rewrite. The trace
API hashes:

```text
bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, char
```

It does not hash `f32` or `f64`. In many `UNSUPPORTED [adapter]` cases, the MIR
can probably be patched by widening the adapter and trace API. The adapter stops
because it does not yet know how to rewrite/hash that dumped type safely.

## Floating point and libdevice seeds

The comparison is exact `u64` hash equality, so it assumes the CPU and the GPU
agree bit for bit. Floats are never hashed directly, since the trace API has no
`f32` or `f64` arm and the adapter refuses a bare float dump. A float can still
reach the hash indirectly, through an `as` cast to an integer, through a
comparison that yields a `bool`, or through rustlantis' `transmute_place`.

Where that happens on a seed whose device code calls libdevice, a mismatch is
not on its own evidence of a backend bug. Only a few libdevice entry points are
specified as single correctly-rounded operations, `fma` among them. The
transcendentals (`sin`, `cos`, `exp`, `log`, `pow`, `atan2` and the rest) are not
required to be bit-identical to the host's libm, and the repository compares them
within a tolerance elsewhere: see the 2-ULP comparison in
`examples/math_atan/src/main.rs` and `ulp_distance` in
`examples/libdevice_math/src/main.rs`.

So triage a `MISMATCH` on a float-influenced seed by hand before filing it. Check
whether the differing value derives from a transcendental, and compare the two
results in ULPs before treating the difference as a miscompile.

## Artifacts

`run_seed.py` writes artifacts under `crates/fuzzer/artifacts/`, which is
ignored by git.

Per-seed logs:

```text
crates/fuzzer/artifacts/seed-<N>-<status>.log
```

Failure logs include:

- seed
- status
- stage (`adapter`, `backend`, or `run`)
- reason
- return code
- command
- full command output
- generated case snapshot, when the adapter produced one

The run summary is also written as:

```text
crates/fuzzer/artifacts/summary.jsonl
```

`run_seed.py` clears `crates/fuzzer/artifacts/` at the start of every
invocation, so the logs and `summary.jsonl` always describe only the latest run.

The terminal also prints a full per-seed summary; entries that wrote a log
append its path. For example, `--start 0 --count 2` currently prints:

```text
results:
  seed 0: PASS [run] CPU/GPU traces matched
  seed 1: PASS [run] CPU/GPU traces matched
summary: PASS=2
```
