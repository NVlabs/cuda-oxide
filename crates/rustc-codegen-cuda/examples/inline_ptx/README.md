# inline_ptx

Exercises `cuda_device::ptx_asm!` inside a cuda-oxide kernel. The kernel does
Rust arithmetic, uses a register-only PTX instruction, reads the lane-id
register (`%%laneid` in the macro string), emits a memory-clobbering
`membar.gl`, then uses the PTX results in Rust.

A second kernel exercises multi-output `ptx_asm!`: a single asm block with
two `=r` outputs computes both the sum and the product of two
thread-dependent values, written to separate buffers the host verifies
element-wise (the asymmetric results catch swapped output bindings).

Run with:

```bash
cargo oxide run inline_ptx
```

## `ptx_asm!` Reference

This is a quick reference. The canonical description of the macro lives in
the [`ptx_asm!` section of the cuda-macros README](../../../cuda-macros/README.md),
and the template syntax follows NVIDIA's
[Inline PTX Assembly](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
document. If this page ever disagrees with those, they win.

### Syntax

```rust
unsafe {
    ptx_asm!(
        "<template>",
        out("<constraint>") <place>,   // output operands (up to 16)
        in("<constraint>") <expr>,     // input operands (up to 16)
        clobber("<name>"),             // side-effect declarations
        options(<option>, ...),        // assembly options
    );
}
```

Template placeholders use `%N` (zero-based) for operands and `%%reg` for
literal PTX registers (e.g., `%%laneid`, `%%tid.x`).

### Operand Constraints

| Constraint | Direction | PTX register | Rust type     |
|------------|-----------|--------------|---------------|
| `"h"`      | in        | `.b16`       | `u16` / `i16` |
| `"r"`      | in        | `.b32`       | `u32` / `i32` |
| `"l"`      | in        | `.b64`       | `u64` / `i64` / `*T` |
| `"q"`      | in        | `.b128`      | 128-bit value |
| `"f"`      | in        | `.f32`       | `f32`         |
| `"d"`      | in        | `.f64`       | `f64`         |
| `"n"`      | in        | immediate    | integer const |
| `"=h"`     | out       | `.b16`       | `u16` / `i16` |
| `"=r"`     | out       | `.b32`       | `u32` / `i32` |
| `"=l"`     | out       | `.b64`       | `u64` / `i64` |
| `"=q"`     | out       | `.b128`      | 128-bit value |
| `"=f"`     | out       | `.f32`       | `f32`         |
| `"=d"`     | out       | `.f64`       | `f64`         |

Output operands must appear before input operands.

### Options

| Option          | Effect |
|-----------------|--------|
| `register_only` | No memory side-effects; requires at least one `out` operand, incompatible with `clobber` |
| `may_diverge`   | Promises the snippet is safe to move across divergent control flow (drops the LLVM `convergent` guarantee); requires `register_only` |

**Never** use `may_diverge` for `.sync` instructions, collectives, or any
snippet whose participating lanes matter; the optimizer may move those
snippets out of branch control flow.

### Clobbers

| Clobber      | Meaning |
|--------------|---------|
| `"memory"`   | Assembly reads or writes memory not captured by operands |

As an escape hatch, raw LLVM clobber strings of the form `clobber("~{...}")`
are also accepted and passed through verbatim (e.g. `clobber("~{memory}")`).

### Examples

Integer add with lane ID:

```rust
let doubled: u32;
let lane: u32;
unsafe {
    ptx_asm!(
        "add.u32 %0, %1, %1;",
        out("=r") doubled,
        in("r") value,
        options(register_only),
    );
    ptx_asm!("mov.u32 %0, %%laneid;", out("=r") lane);
}
```

Float approximate math (`tanh.approx.f32` requires `sm_75` or newer):

```rust
let result: f32;
unsafe {
    ptx_asm!(
        "tanh.approx.f32 %0, %1;",
        out("=f") result,
        in("f") x,
        options(register_only),
    );
}
```

Multi-output (sum and product in one block):

```rust
let sum: u32;
let prod: u32;
unsafe {
    ptx_asm!(
        "add.u32 %0, %2, %3; mul.lo.u32 %1, %2, %3;",
        out("=r") sum,
        out("=r") prod,
        in("r") x,
        in("r") y,
        options(register_only),
    );
}
```

Memory barrier (void, no outputs):

```rust
unsafe {
    ptx_asm!("membar.gl;", clobber("memory"));
}
```
