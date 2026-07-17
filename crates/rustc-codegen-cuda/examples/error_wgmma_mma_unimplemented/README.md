# error_wgmma_mma_unimplemented

Negative test: confirms that the codegen backend rejects calls to
`cuda_device::wgmma::wgmma_mma_*` with a clear "not yet implemented"
diagnostic, rather than silently emitting a comment placeholder and
producing PTX that multiplies-accumulates to zero.

## What this tests

Until full WGMMA MMA lowering can preserve delayed 32-register accumulator
state across commit and wait, the importer must reject these calls. The
dialect lowering remains fail-closed as a second guard.

## Usage

```bash
cargo oxide run error_wgmma_mma_unimplemented
```

## Expected output

The build **must fail** with a diagnostic similar to:

```
Unsupported construct: WGMMA MMA is not yet supported: lowering must
preserve delayed 32-register accumulator state across commit_group and
wait_group
```

If the build succeeds, the unsupported call escaped both fail-closed guards.

## Categorisation

`scripts/smoketest.sh` classifies this example as the `error` category,
so its expected verdict is "compilation must fail with a recognised
diagnostic" — the same convention as the existing `error/` example.
