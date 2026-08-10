# unaligned_memory

Regression coverage for device-side `core::ptr::read_unaligned` and
`core::ptr::write_unaligned`.

## What this tests

The example deliberately forms a `*const u32` / `*mut u32` from a byte address
at `base + 1`, so the pointer is not naturally aligned for `u32`.

It covers two independent paths:

1. `read_unaligned` reads four bytes starting at an unaligned address and
   returns the expected `u32`.
2. `write_unaligned` writes exactly four bytes starting at an unaligned
   address while preserving guard bytes immediately before and after the
   written range.

With cuda-oxide's pinned Rust toolchain, these operations are implemented by
libcore through byte-oriented `copy_nonoverlapping`, so this example is intended
to pin the end-to-end conformance path rather than introduce a new compiler
intrinsic.

## Usage

```bash
cargo oxide run unaligned_memory
CUDA_OXIDE_NO_OPT=1 cargo oxide run unaligned_memory
```

## Expected output

```text
=== unaligned_memory ===
PASS: read_unaligned from base + 1
PASS: write_unaligned to base + 1
PASS: guard bytes preserved
PASS: unaligned_memory
```
