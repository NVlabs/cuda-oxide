# rustc-codegen-cuda: error example status

Examples named `error*` fall into two kinds:

- **diagnostics-fixture**: intentional negative test. The compiler is supposed
  to reject this. It is not a gap.
- **support-gap**: a real Rust feature that the compiler does not yet handle.
  Kept as an expected-failure regression test until it is implemented.

When adding a new `error*` example, update this table and the
`ERROR_EXAMPLES` array in `scripts/smoketest.sh` in the same commit.
Run `scripts/check-error-example-status.sh` to verify both are in sync.

| Example                               | Kind                | Fails at                            |
| :------------------------------------ | :------------------ | :---------------------------------- |
| `error`                               | diagnostics-fixture | `core::fmt` reachable from device   |
| `error_copy_nonoverlapping_unhandled` | support-gap         | `StatementKind::CopyNonOverlapping` |
| `error_drop_glue`                     | support-gap         | `TerminatorKind::Drop` (effectful)  |
| `error_heap_alloc`                    | diagnostics-fixture | `__rust_alloc` reachable (#108)     |
| `error_missing_device_attr`           | diagnostics-fixture | `thread::index_*` stub (#76)        |
| `error_set_discriminant_unhandled`    | support-gap         | `StatementKind::SetDiscriminant`    |
| `error_wgmma_mma_unimplemented`       | support-gap         | WGMMA MMA lowering                  |

`error_drop_glue` only fails for destructors that do observable work.
Drops whose monomorphized glue is provably a no-op (e.g. the
`core::array::IntoIter` behind `for x in arr` with Copy elements) lower
to a plain branch since issue #138.
