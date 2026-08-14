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
python3 crates/fuzzer/tools/run_seed.py --seed 33
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
can have one dump site or several dump sites. Seed `33` is a small case with
two dump sites:

```rust
__rl_dump0 = (Move(_1), Move(_2), Move(_3), Move(_4));
Call(_9 = dump_var(Move(__rl_dump0)), ReturnTo(bb4), UnwindUnreachable())

__rl_dump1 = (Move(_6),);
Call(_9 = dump_var(Move(__rl_dump1)), ReturnTo(bb5), UnwindUnreachable())
```

The checked-in `generated_case.rs` is kept because its device code calls
libdevice (`fmaf64`) and so covers the artifact path that a PTX-only loader
cannot serve. It was generated from seed `162` under the adapter's earlier
scalar-only rustlantis config; enabling composites changed what every seed
generates, so regenerating seed `162` today produces a different program
rather than that file. Its header records this.

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

For example, seed `1436` returns a `*const i8`, which the trace API does not
hash, so `--start 1436 --count 2 --keep-going` currently reports:

```text
results:
  seed 1436: UNSUPPORTED [adapter] unsupported return type for return-value tracing: *const i8 (crates/fuzzer/artifacts/seed-1436-unsupported.log)
  seed 1437: PASS [run] CPU/GPU traces matched
summary: PASS=1, UNSUPPORTED=1
```

A pointer is a permanent refusal rather than a gap to widen later. The CPU
oracle and the device hold different addresses for the same object by
construction, so folding one into the trace would report a MISMATCH on every
seed that dumped it.

The typical `UNSUPPORTED [adapter]` cause is a generated `dump_var(...)` call
or function signature that uses a type the adapter cannot rewrite. The trace
API hashes these scalars:

```text
bool, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, char,
f32, f64
```

It also hashes an array or a tuple of anything in that list, to any nesting
depth, by folding the leaves. A tuple is hashed up to arity 5, matching the
`TraceDump` implementations. An aggregate's padding is never read, so a dumped
`(u8, u32)` hashes as the two fields and nothing else.

What remains refused at a dump site is a shape with no leaf reading, such as a
reference or a slice. An aggregate in an *argument* position is refused
elsewhere, by `literal_for_type`, which has no literal to construct for one.

In many `UNSUPPORTED [adapter]` cases, the MIR can probably be patched by
widening the adapter and trace API. The adapter stops because it does not yet
know how to rewrite/hash that dumped type safely.

## Floating point and libdevice seeds

The comparison is exact `u64` hash equality, so it assumes the CPU and the GPU
agree bit for bit. Floats are folded as their `to_bits()` pattern, which keeps
that assumption intact for every non-NaN value: a tolerance in the trace would
compare something other than the value a backend produced, and would hide the
differences the fuzzer exists to find. NaN payload bits are the exception,
because Rust does not pin them down, so the trace canonicalizes every NaN to
the quiet-NaN bit pattern at the hash boundary. A payload divergence therefore
cannot produce a `MISMATCH`; a NaN on one backend against a non-NaN on the
other still hashes differently, and that difference is a real signal. Floats
also still reach the hash indirectly, through an `as` cast to an integer,
through a comparison that yields a `bool`, or through rustlantis'
`transmute_place`.

Two sources of difference are therefore expected, and neither is a backend bug.

The first is FMA contraction. Device codegen contracts an `fmul` feeding an
`fadd` into a single `fma.rn` by default, matching nvcc's `--fmad=true`, while
the CPU oracle rounds the multiply and the add separately. `run_seed.py` passes
`--no-fmad` so the two agree. Contraction is worth fuzzing on its own terms,
against a contracted reference.

The second is libdevice. Only a few libdevice entry points are
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
append its path, relative to the repo root. Without `--keep-going` a run
stops at the first non-PASS seed, so `--start 1436 --count 2` alone would end
at seed 1436's `UNSUPPORTED`. `--start 1436 --count 2 --keep-going` currently
prints:

```text
results:
  seed 1436: UNSUPPORTED [adapter] unsupported return type for return-value tracing: *const i8 (crates/fuzzer/artifacts/seed-1436-unsupported.log)
  seed 1437: PASS [run] CPU/GPU traces matched
summary: PASS=1, UNSUPPORTED=1
```
