# drop_glue

Positive test: verifies that device-side drop glue compiles and
executes correctly.

## What this tests

rustc emits `TerminatorKind::Drop` for places whose type has drop glue
(non-`Copy` types with an `impl Drop`, recursively through fields and
parameters). cuda-oxide now translates these drops into device-side
`drop_in_place` calls so destructors run on the GPU.

This example owns a `DropMarker` whose `Drop::drop` writes `0xDEADBEEF`
through a captured pointer. The host verifies the sentinel was written,
confirming that the destructor executed on the device.

## Usage

```bash
cargo oxide run drop_glue
```

## Expected output

The build succeeds and the kernel writes `0xDEADBEEF` into every output
element via drop glue:

```
SUCCESS: drop glue wrote sentinel in all 256 elements
```
