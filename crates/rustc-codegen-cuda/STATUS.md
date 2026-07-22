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
| `error_enum_constant_provenance`      | support-gap         | Enum constant pointer relocation    |
| `error_enum_pointer_overlap`          | support-gap         | Overlaid pointer/integer payload    |
| `error_enum_shared_pointer_layout`    | support-gap         | Mode-dependent AS3 pointer width    |
| `error_generated_intrinsic_abi`       | diagnostics-fixture | Unsupported raw intrinsic ABI       |
| `error_generated_intrinsic_callable`  | diagnostics-fixture | Raw intrinsic passed through `Fn`   |
| `error_generated_intrinsic_fn_pointer`| diagnostics-fixture | Raw intrinsic made into `fn` pointer|
| `error_generated_intrinsic_unknown_id`| diagnostics-fixture | Unknown ID in a supported ABI       |
| `error_heap_alloc`                    | diagnostics-fixture | `__rust_alloc` reachable (#108)     |
| `error_missing_device_attr`           | diagnostics-fixture | `thread::index_*` stub (#76)        |
| `error_set_discriminant_uninhabited`  | diagnostics-fixture | Invalid enum variant selection      |
| `error_static_initializer_provenance` | support-gap         | Device-global pointer relocation    |
| `error_tuple_array_provenance`        | support-gap         | Tuple-array pointer relocation      |
| `error_tuple_constant_provenance`     | support-gap         | Direct tuple pointer relocation     |
| `error_wgmma_mma_unimplemented`       | support-gap         | WGMMA MMA lowering                  |

Drops whose monomorphized glue is provably a no-op (e.g. the
`core::array::IntoIter` behind `for x in arr` with Copy elements) lower
to a plain branch since issue #138. When the no-op proof fails, the drop
lowers to a device-side call to the monomorphized `drop_in_place::<T>`
shim, which the collector gathers and the pipeline compiles like any
other device function; see the `drop_glue` example. This covers direct
`impl Drop` types reachable from kernels. The shim and every
`Drop::drop` body it reaches go through the same device pipeline, so
drop glue whose MIR uses constructs the pipeline cannot yet translate
(e.g. slice/`Vec` element drops, panic formatting) still fails
compilation with a diagnostic. Destructors are never silently skipped:
a drop is either proven unobservable or its call is emitted.

The enum storage fixtures deliberately fail closed. Enum constants with
relocations cannot be flattened to bytes without replacing an address with
the pointee's contents. Pointer and integer payloads cannot share one lowered
slot without erasing LLVM pointer provenance. Shared-memory pointers are 64
bits in PTX and legacy NVVM output but 32 bits in modern NVVM output, while MIR
lowering does not yet know which exporter mode will be selected. Accepting
any of these cases would risk generating wrong code, so they remain rejected
until their information can be preserved through lowering. (Bools nested in
aggregate payloads are no longer in this list: enum storage claims each
payload's byte-faithful twin and construction zero-extends every `i1` leaf
into its canonical memory byte, so `Option<struct { u32, bool }>` and
friends lower exactly like rustc lays them out.)
