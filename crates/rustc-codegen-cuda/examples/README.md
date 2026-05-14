# Examples

This directory holds **two different kinds of crates** that share a build
harness and a directory but serve different purposes. Both kinds matter
for regression coverage, but a future agent looking around shouldn't
assume one shape is "the right shape."

**1. Regression tests for codegen bugs** (added on the
`reproduce-errors` branch). Each one has a `//!` doc block at the top
of `src/main.rs` naming the pre-fix wall, the fix that landed (or "no
fix yet — documents an unsupported construct"), and what triggers the
case. The doc block is load-bearing: it's the contract for what the
example proves.

**2. Demos and feature showcases** (pre-date the `reproduce-errors`
branch). These exercise specific cuda-oxide features (intrinsics,
dialect ops, launch APIs, subsystem demos like wgmma/tcgen05/tma)
rather than locking down a specific bug. Most don't carry a `//!`-
block in the regression-test format, and that's fine — they're
working sample code, not regression-test contracts. They still build
and break the same way the regression tests do, so they participate
in the codegen-regression sweep.

The contract that holds for both: **anything passing today must keep
passing.** The sweep loop below treats every crate uniformly.

## Building

The codegen backend is a `.so` plugin that `cargo oxide` loads. **After
any change to `crates/mir-importer`, `crates/mir-lower`, or
`crates/rustc-codegen-cuda` itself, the backend `.so` must be rebuilt:**

```sh
cargo oxide setup
```

Without this, `cargo oxide build` still runs but uses the previously
cached backend — your code changes have no effect, and the failure modes
are confusing (apparent "no-ops" that are really stale plugins).

Then build individual examples:

```sh
cargo oxide build <example_name>     # codegen-only — no GPU needed
cargo oxide run   <example_name>     # codegen + launch — needs a GPU
```

`build` is the regression gate. Codegen success means well-formed PTX
came out, not that the math is right. Always sanity-check on real
hardware with `run` before declaring a fix complete.

To sweep all examples and check for regressions:

```sh
for d in crates/rustc-codegen-cuda/examples/*/; do
  name=$(basename "$d")
  cargo oxide build "$name" 2>&1 | tail -3 | grep -q "Build succeeded" \
    && echo "PASS $name" || echo "FAIL $name"
done
```

## TDD flow for new codegen bugs

When a new wall comes in from a downstream consumer (`cuda-oxide`'s job
is to compile arbitrary Rust to PTX, so "new wall" means: a Rust feature
or pattern the codegen hasn't been taught yet):

1. **Write a minimal repro example crate** in this directory. Shrink
   from the downstream consumer's code until you have the smallest
   shape that still triggers the bug. Defeat over-eager inliners by
   adding `#[inline(never)]` or by making the shape large enough that
   the optimizer can't fold it — otherwise your "repro" silently
   succeeds and the real bug stays hidden.

2. **Watch it fail.** Run `cargo oxide setup` (only needed if backend
   changed) then `cargo oxide build <name>`. The error you see should
   match the downstream consumer's error. If it doesn't reproduce,
   the repro is wrong — go back to step 1.

3. **Commit the failing repro by itself.** The commit message names
   the bug and the surfaced-from context. The example's `//!` block
   uses past tense ("pre-fix wall: ...") so the next change-author has
   the evidence. Title pattern:
   `Add <name> — known-failure for <one-line bug description>`.

4. **Implement the fix.** Touch only the codegen path the doc block
   called out. Rebuild the backend with `cargo oxide setup` so the new
   `.so` is what `cargo oxide build` loads.

5. **Confirm the same example now builds.** No diff to the example
   crate is allowed between steps 3 and 5 — only the codegen fix may
   flip the repro from fail to pass. If you had to change the example
   to make it pass, the fix is incomplete (or the repro was wrong).

6. **Sweep for regressions.** Rebuild a handful of unrelated examples
   (e.g. `vecadd`, `hashmap`, plus anything near the path you touched).
   For the full sweep, see the loop above — every codegen-time known-
   failure should still fail with its documented diagnostic, and every
   other crate should still build.

7. **Commit the fix.** Reference the example name in the commit message
   so future bisects connect the dots. Title pattern:
   `<verb> <thing> (fixes <name>)` or similar.

If a wall is a genuinely unsupported construct (e.g. `wgmma.mma_async`
lowering, `SetDiscriminant`), the repro stays as a permanent
known-failure documenting the contract — the doc block says "no fix
yet — documents an unsupported construct" and the build is expected to
fail with exactly that diagnostic. Don't delete or work around these:
they're what a future implementer needs to know exists.

## A note on bug categories

Repros land in one of three buckets:

1. **Codegen-time known-failures** — `cargo oxide build` itself fails.
   The walls are diagnostics like "unsupported construct", "symbol not
   found", or LLC verification errors. Easy to spot in CI: build exit
   code is non-zero.
2. **PTX-shape known-failures (build passes, emitted PTX is wrong)**
   — `cargo oxide build` succeeds, but the emitted PTX has the wrong
   instruction sequence, operand wiring, or address space. The fix
   loop is entirely local: rebuild, grep the new PTX for the expected
   shape, iterate. No hardware run is in the contract. The doc block
   names the expected PTX shape (e.g. `add.cc.u64 ... ; addc.u64 ...`),
   the observed wrong shape, and the grep that verifies the fix.
3. **Runtime known-failures** — `cargo oxide build` succeeds, the PTX
   looks plausible on inspection, but the kernel faults under
   compute-sanitizer or returns wrong answers on real hardware. The
   contract requires a hardware run to verify a fix. Symptom classes
   include misaligned wide loads, ABI mismatches at kernel-call
   boundaries, undefined-behavior PTX that happens to text-grep as
   correct.
4. **Passing regression tests (fix landed)** — built and (where
   possible) hardware-verified. Locks in a fix.

When writing a new repro, pick the one that matches the surfaced
behaviour and tag the doc block accordingly. Symptoms drive the
category — a misaligned-load fault is a runtime known-failure even if
the bug class is technically also a codegen bug. A wrong-shape PTX
that *also* faults on hardware is a PTX-shape known-failure if the
text-grep alone catches the bug.

## Codegen-time known-failures

These build-fail by design. Each one's `//!` block describes an
unsupported construct and asserts the exact diagnostic the build
emits. If any of these *starts* building, the doc block is stale
(either a fix landed silently elsewhere, or the optimizer is now
hiding the path).

* `array_of_tuple_const` — `translate_array_constant: unsupported element type: MirTupleType { … }`
* `drop_adt_with_impl` — drop of `'...Secret'` is not supported
* `error_drop_glue` — drop glue diagnostic
* `error_set_discriminant_unhandled` — `SetDiscriminant` statements are not yet supported
* `error_wgmma_mma_unimplemented` — `wgmma.mma_async` lowering is not yet implemented
* `helper_no_inline` — `Symbol helper_no_inline__kernels__get_thread_idx not found`
* `helper_outside_module` — `Symbol helper_outside_module__get_thread_idx not found`
* `tuple_const_array_field` — `Tuple constant field 0 has unsupported type MirArrayType { … }`

## PTX-shape known-failures (build passes, emitted PTX is wrong)

These build cleanly but the emitted PTX has the wrong shape — a
specific instruction sequence is missing, an operand register is
unwired, or the address space is wrong. The fix loop is entirely
local: `cargo oxide build <name>`, inspect the PTX, grep for the
expected shape. No hardware run is part of the contract.

Each one's `//!` block describes:

* the **expected** PTX shape (e.g. `add.cc.u64 ... ; addc.u64 ...`)
* the **observed** wrong shape (e.g. plain `add.u64` with no carry)
* the **grep** that verifies a fix

(None currently. `u128_ne_early_return` was here pre-fix and has
moved to the passing list.)

## Runtime known-failures (hardware required to verify)

These build cleanly, the emitted PTX text-greps as correct, but the
kernel still misbehaves on real hardware — faults under
compute-sanitizer, returns wrong answers, or exhibits UB-driven
miscompilation that only the hardware exposes. Verification of a fix
requires a hardware run; text inspection alone is insufficient.

(None currently. `static_ref_relocation` and `xoshiro_seed_misalign`
were here pre-fix and have moved to the passing list.)

## Passing regression tests

Each one's `//!` block describes the bug it locked down and the fix
that landed. If any of these *stops* building, a recent change has
regressed a previously-fixed bug — bisect the codegen crates.

* array_const_repro
* array_eq_raw
* array_to_int_cast
* assert_inhabited_intrinsic
* black_box_aggregate
* black_box_intrinsic
* closure_struct_arg_mismatch
* closure_zero_captures_repro
* copy_nonoverlapping_basic
* cross_crate_pubfn
* deref_const_index_array_write
* deref_field_index_write
* deref_index_local_write
* deref_index_trait_dispatch
* device_unsafe_repro
* drop_adt_no_impl
* drop_copy_primitive
* field_addr_tuple
* field_index_field_write
* field_index_write
* fndef_zst_type
* generic_array_deref
* generic_array_to_slice
* generic_sequence_alias
* index_trait_dispatch
* inherent_impl_call
* iter_zip_chunks_exact
* maybe_uninit_union
* mul_output_adt
* mul_output_mismatched
* nested_closure_capture_repro
* nested_struct_const
* newtype_const_index_assign
* overflowing_arith_carry
* panic_fmt_path
* ptr_offset_intrinsic
* result_unwrap_dyn_debug
* scalar52_sub_repro
* slice_const_idx_write
* slice_const_indexing
* slice_last_from_end
* slice_range
* static_ref_relocation
* static_u64_array_load
* str_panic_path
* subtle_choice_select
* typed_swap_intrinsic
* u128_imm_shr
* u128_ne_early_return
* volatile_load_intrinsic
* xoshiro_seed_misalign

## 49 pre-existing demos and feature showcases

These crates pre-date the `reproduce-errors` branch and were not
written as regression tests. Most don't have a `//!` block in the
"pre-fix wall / what landed" format — they're demos of specific
cuda-oxide features (warp/block/cluster primitives, shared memory,
tensor cores, dialect ops, host APIs). All 49 build today; if any
of them break, a recent change has regressed previously-shipping
functionality.

A future agent extending this directory should:

- **Treat them as covered by the regression sweep** (they all build
  today, alongside the regression tests). Don't delete or move them
  just because they don't match the regression-test docstring
  convention.
- **Not retrofit them into the `//!` format** unless you're also
  reframing one as a regression test for a specific bug it locks
  down. They're working samples; turning a sample into a contract is
  a separate intentional act.
- **Read them before writing new regression tests in adjacent areas.**
  If you're testing a warp intrinsic, look at `warp_reduce` first;
  for tcgen05, look at `tcgen05_matmul`; etc. They show the canonical
  pattern.

Alphabetical:

* abi_hmm
* array_index
* async_mlp
* async_vecadd
* atomics
* barrier
* cast_tests
* clc
* cluster
* compiler_features
* coop_groups_demo
* cpp_consumes_rust_device
* cross_crate_kernel
* cuda_module_contract
* debug
* device_closures
* device_ffi_test
* device_global
* dynamic_smem
* error
* f16_stress
* future_apis
* gemm
* gemm_sol
* generic
* hashmap
* hashmap_v2
* hashmap_v3
* hello_constant
* helper_fn
* host_closure
* index2d_const
* manual_launch_generic
* mathdx_ffi_test
* mcast_barrier_test
* numeric_stress
* primitive_stress
* printf
* rustlantis-smoke
* sharedmem
* standalone_device_fn
* tcgen05
* tcgen05_matmul
* tiled_gemm
* tma_copy
* tma_multicast
* vecadd
* warp_reduce
* wgmma
