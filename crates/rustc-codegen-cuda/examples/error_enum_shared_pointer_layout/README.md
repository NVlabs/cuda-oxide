# `error_enum_shared_pointer_layout`

Negative test for `Option<SharedPointerArrayWrapper>`, where the wrapper contains
an array of two `&SharedArray<...>` values.

Direct shared-pointer enum fields and shared pointers nested through ordinary
structs/tuples use target-stable CUDA generic physical storage. Arrays remain a
separate boundary because value conversion would emit one extraction, address-
space cast, and insertion sequence per element. That expansion needs an explicit
bound and code-shape contract before it can be accepted safely.

The compiler must reject the array shape instead of retaining target-dependent
address-space-3 pointer storage:

```bash
cargo oxide build error_enum_shared_pointer_layout
cargo oxide build error_enum_shared_pointer_layout --emit-nvvm-ir --arch sm_90
cargo oxide build error_enum_shared_pointer_layout --emit-nvvm-ir --arch sm_100
```

Expected diagnostic:

```text
arrays containing shared-memory pointers are not supported
```
