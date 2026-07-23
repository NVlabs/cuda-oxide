# approx_math

## Approximate Math via `ptx_asm!`

Demonstrates single-instruction PTX approximate math operations through
`ptx_asm!`, bypassing libdevice for maximum throughput at reduced precision.

## What This Example Does

Exercises four approximate math instructions and composes them into a fast
sigmoid, verifying each against a host reference within tolerance:

| Wrapper            | PTX instruction        | Precision    | Availability |
|--------------------|------------------------|--------------|--------------|
| `tanh_approx`      | `tanh.approx.f32`      | ~2^-8 ULP   | sm_75+       |
| `ex2_approx`       | `ex2.approx.ftz.f32`   | ~2^-8 ULP   | all          |
| `rcp_approx`       | `rcp.approx.ftz.f32`   | ~2^-23 ULP  | all          |
| `lg2_approx`       | `lg2.approx.ftz.f32`   | ~2^-8 ULP   | all          |
| `fast_sigmoid`     | via `tanh.approx.f32`  | ~2^-8 ULP   | sm_75+       |

## Key Concepts

### Why Approximate Instructions?

Libdevice math functions (`__nv_tanhf`, `__nv_expf`, etc.) provide full IEEE
precision but compile to multi-instruction sequences. The hardware approximate
instructions are single-cycle operations that trade precision for throughput:

```rust
// Libdevice tanh: ~20 instructions via function call
let y = x.tanh();

// Hardware approximate: 1 instruction
let y: f32;
unsafe {
    ptx_asm!(
        "tanh.approx.f32 %0, %1;",
        out("=f") y,
        in("f") x,
        options(register_only),
    );
}
```

### Fast Sigmoid Pattern

Sigmoid can be computed from tanh with no `exp` or `rcp`:

```
sigmoid(x) = 0.5 * tanh(x * 0.5) + 0.5
```

This reduces sigmoid to one `tanh.approx.f32` plus two FMAs.

## Build and Run

```bash
cargo oxide run approx_math
```

## Expected Output

```text
=== Approximate Math via ptx_asm! ===

GPU Compute Capability: sm_86
--- tanh.approx.f32 --- max |err| = 3.28e-04
--- fast sigmoid     --- max |err| = 1.64e-04
--- ex2.approx.f32   --- max |err| = 1.23e-03
--- rcp.approx.f32   --- max |err| = 5.96e-08
--- lg2.approx.f32   --- max |err| = 9.77e-04

SUCCESS: all approximate math results within tolerance
```

Exact error magnitudes vary by GPU architecture.

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU for `ex2`, `rcp`, `lg2`
- **tanh / sigmoid**: Turing or later (sm_75+)
- **CUDA Driver**: 11.0+
