# `error_enum_shared_pointer_layout`

Negative test for `Option<&SharedArray<...>>`.

Shared-memory pointers use different LLVM storage widths depending on output
mode: 64 bits in the ordinary PTX and legacy NVVM layouts, but 32 bits in the
modern NVVM layout. MIR-to-LLVM lowering currently runs before that mode is
known, so it cannot choose one byte-faithful enum representation.

The compiler must reject this enum instead of silently leaving half of its
carrier undefined:

```bash
cargo oxide build error_enum_shared_pointer_layout
cargo oxide build error_enum_shared_pointer_layout --emit-nvvm-ir --arch sm_90
cargo oxide build error_enum_shared_pointer_layout --emit-nvvm-ir --arch sm_100
```

Expected diagnostic:

```text
contains a shared-memory pointer whose size is target-mode dependent
```
