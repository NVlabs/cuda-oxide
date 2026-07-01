# cuda_module_nested

Demonstrates `#[cuda_module]` collecting kernels from nested modules. One
kernel per collection mechanism:

| Kernel | Location | Mechanism |
|---|---|---|
| `fill_index` | module root | flat layout (pre-existing behavior) |
| `scale::scale_by` | `mod scale { ... }` | inline nested module |
| `offset::offset_by` | `src/stages/offset.rs` | `include!("stages/offset.rs")` |
| `double::double_all` | `src/kernels/double.rs` | out-of-line `pub mod double;` |

All four surface as flat methods on `kernels::LoadedModule` — nesting
organizes source, not the launcher API — which is also why kernel names must
be unique across the module tree (the macro rejects collisions with an error
naming both modules).

Files are resolved the way rustc resolves the emitted tokens: `include!`
literals relative to the containing file, out-of-line modules by module
directory. Computed include paths (e.g. `concat!(env!("OUT_DIR"), ...)`) and
`#[cfg]`-gated files are skipped.

Note: the out-of-line form (`pub mod double;`) requires
`#![feature(proc_macro_hygiene)]` because rustc gates non-inline modules in
proc-macro input (rust-lang/rust#54727). Inline nested modules and `include!`
work without any feature gate.

## Run

```
cargo oxide run cuda_module_nested
```

Expected output:

```
✓ SUCCESS: root, inline-mod, include!, and out-of-line kernels all ran
```
