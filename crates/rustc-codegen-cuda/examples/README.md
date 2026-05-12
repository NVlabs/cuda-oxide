# Examples

Each crate in this directory is a regression test for one specific codegen
bug. The `//!` doc block at the top of `src/main.rs` names the pre-fix
wall, the fix that landed (or notes "no fix yet"), and what triggers the
case. Anything that passes today is expected to keep passing — the crate
list doubles as a contract.

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
   For the full sweep, see the loop above — 86/92 examples pass on a
   clean `reproduce-errors` branch, and that ratio must hold.

7. **Commit the fix.** Reference the example name in the commit message
   so future bisects connect the dots. Title pattern:
   `<verb> <thing> (fixes <name>)` or similar.

If a wall is a genuinely unsupported construct (e.g. `wgmma.mma_async`
lowering, `SetDiscriminant`), the repro stays as a permanent
known-failure documenting the contract — the doc block says "no fix
yet — documents an unsupported construct" and the build is expected to
fail with exactly that diagnostic. Don't delete or work around these:
they're what a future implementer needs to know exists.

## 6 documented known-failures

These build-fail by design. Each one's `//!` block describes an
unsupported construct and asserts the exact diagnostic the build
emits. If any of these *starts* building, the doc block is stale
(either a fix landed silently elsewhere, or the optimizer is now
hiding the path).

* `drop_adt_with_impl` — drop of `'...Secret'` is not supported
* `error_drop_glue` — drop glue diagnostic
* `error_set_discriminant_unhandled` — `SetDiscriminant` statements are not yet supported
* `error_wgmma_mma_unimplemented` — `wgmma.mma_async` lowering is not yet implemented
* `helper_no_inline` — `Symbol helper_no_inline__kernels__get_thread_idx not found`
* `helper_outside_module` — `Symbol helper_outside_module__get_thread_idx not found`

## 37 passing regression tests

Each one's `//!` block describes the bug it locked down and the fix
that landed. If any of these *stops* building, a recent change has
regressed a previously-fixed bug — bisect the codegen crates.

* array_const_repro
* array_eq_raw
* array_to_int_cast
* assert_inhabited_intrinsic
* closure_struct_arg_mismatch
* closure_zero_captures_repro
* copy_nonoverlapping_basic
* cross_crate_pubfn
* deref_const_index_array_write
* deref_field_index_write
* deref_index_local_write
* device_unsafe_repro
* drop_adt_no_impl
* drop_copy_primitive
* field_addr_tuple
* field_index_field_write
* field_index_write
* fndef_zst_type
* generic_array_to_slice
* generic_sequence_alias
* inherent_impl_call
* iter_zip_chunks_exact
* maybe_uninit_union
* mul_output_adt
* mul_output_mismatched
* nested_closure_capture_repro
* nested_struct_const
* panic_fmt_path
* ptr_offset_intrinsic
* result_unwrap_dyn_debug
* scalar52_sub_repro
* slice_const_idx_write
* slice_last_from_end
* slice_range
* str_panic_path
* typed_swap_intrinsic
* volatile_load_intrinsic
