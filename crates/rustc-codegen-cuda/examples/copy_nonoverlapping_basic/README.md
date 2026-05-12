# error_copy_nonoverlapping_unhandled

Negative test: confirms that the mir-importer rejects calls to
`core::ptr::copy_nonoverlapping` with a clear "not yet supported"
diagnostic, rather than silently dropping the memcpy from the lowered
PTX.

## What this tests

`core::ptr::copy_nonoverlapping` lowers to a MIR
`StatementKind::Intrinsic(NonDivergingIntrinsic::CopyNonOverlapping(_))`.
Until the importer implements that lowering, calling it from a `#[kernel]`
must produce a hard build error; the previous catch-all in
`crates/mir-importer/src/translator/statement.rs` silently returned
`Ok(prev_op)`, so the memcpy disappeared from the PTX.

## Usage

```bash
cargo oxide run error_copy_nonoverlapping_unhandled
```

## Expected output

The build **must fail** with a diagnostic similar to:

```
error: [rustc_codegen_cuda] Device codegen failed: PTX generation failed:
       Translation failed: copy_nonoverlapping_kernel: ... Compilation
       error: invalid input program.
       Unsupported construct: core::ptr::copy_nonoverlapping is not yet
       supported on the device; until it is lowered, the call would be
       silently dropped from the PTX
```

If the build succeeds, the silent-miscompile regression has returned —
the importer is once again routing `Intrinsic(CopyNonOverlapping)`
through the catch-all `Ok(prev_op)` arm.

## Categorisation

`scripts/smoketest.sh` classifies this example as the `error` category,
so its expected verdict is "compilation must fail with a recognised
diagnostic" — the same convention as the existing `error/` example.
