# disjoint_slice_len

Regression test for [issue #343](https://github.com/NVlabs/cuda-oxide/issues/343):
calling `DisjointSlice::len()` (and `is_empty()`) inside a kernel must compile and return the launch-time length.

## The bug

`DisjointSlice::len` is intercepted by the mir-importer as an intrinsic (`emit_len`). 
Because `len(&self)` receives the slice behind a reference, 
the translated operand is a thin `mir.ptr<mir.disjoint_slice<T>>`, 
not the fat `(ptr, len)` value. 
`emit_len` fed that pointer straight into `mir.extract_field`, 
which only accepts the fat value, so device codegen died in dialect verification:

```text
MirExtractFieldOp operand must be tuple, slice, struct, array, or scalar (newtype)
```

The fix loads the fat value through the pointer first. 
Both interceptors now share that logic (`resolve_disjoint_slice_value`).

## Reproducing the original failure

The interceptor only sees the call when rustc's MIR inliner leaves it intact.
The default release pipeline inlines `len()` into the kernel.
To reproduce the problem, disable MIR inlining:

```bash
RUSTFLAGS="-Zinline-mir=no" cargo oxide run disjoint_slice_len
```

Before the fix, this failed with the verification error above.
After the fix, it compiles and runs identically to the default build.

`scripts/smoketest.sh` pins this flag for this example (see `NOINLINE_MIR_EXAMPLES`)

## What the kernel checks

Every in-bounds thread writes `len() + is_empty() as u32` to its slot, 
so for a 256-element launch every slot must read 256. 
The host asserts all elements.

## Run

```bash
cargo oxide run disjoint_slice_len
```

Expected output:

```text
SUCCESS: DisjointSlice::len/is_empty return the launch-time length
```
