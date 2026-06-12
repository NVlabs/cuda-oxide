# reserved-oxide-symbols — INTERNAL

> **Not a public API.** This crate is `publish = false` and exists only to
> keep the macro side and the consumer side of the cuda-oxide naming
> contract in lockstep. The constants, builders, and predicates exposed
> here may change without notice between commits. External consumers
> should depend on `cuda-host`, `cuda-device`, or `cuda-macros` instead.

## What this crate owns

The `cuda_oxide_*` symbol namespace that `#[kernel]` and `#[device]`
mangle user functions into, and that the codegen backend, MIR-lowering,
and LLVM-export passes look for.

Every prefix here ends with the magic suffix `246e25db_`, which is
`sha256("cuda_oxide_ + rust")` truncated to 8 hex chars. The hash
exists purely to make accidental collisions impossible — a user is
never going to write `fn cuda_oxide_kernel_246e25db_foo()` by accident.

| Constant                | Value                                  |
|-------------------------|----------------------------------------|
| `KERNEL_PREFIX`         | `cuda_oxide_kernel_246e25db_`          |
| `DEVICE_PREFIX`         | `cuda_oxide_device_246e25db_`          |
| `DEVICE_EXTERN_PREFIX`  | `cuda_oxide_device_extern_246e25db_`   |
| `INSTANTIATE_PREFIX`    | `cuda_oxide_instantiate_246e25db_`     |

## Layered API

Three concentric layers, pick the one that fits the call site:

1. **Constants** — the raw prefix strings, for code that needs the literal.
2. **Builders** — `kernel_symbol(base) -> String`, etc., for the macro side.
3. **Predicates and extractors** — `is_kernel_symbol(name) -> bool`,
   `kernel_base_name(name) -> Option<&str>`, etc., for the consumer side.

The Layer-3 helpers hide the substring matching that used to be
duplicated across `rustc-codegen-cuda`, `llvm-export`, and `mir-lower`.

## Why "reserved"

The `cuda_oxide_*` namespace is **reserved**: user code may not define
functions whose name starts with it. The `#[kernel]` and `#[device]`
proc macros enforce this at the source-code level via a compile-error
guard, and the hash suffix defends against the macro-bypass case (a
plain function literally named `cuda_oxide_kernel_foo`, no macro). Both
defenses are needed: the macro guard catches honest mistakes early at
the source-code level; the hash suffix makes the actually-mangled name
collision-resistant against any code path that bypasses the macro.
