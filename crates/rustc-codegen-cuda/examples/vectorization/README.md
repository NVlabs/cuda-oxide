# vectorization

## Alignment-driven vectorized memory access, across the CUDA vector types

This example covers the full set of CUDA built-in vector types (`char1` …
`double4_32a`) as their Rust equivalents (`i8x1` … `f64x4_a32`), each with its
exact CUDA size and alignment. A per-type copy kernel (`output[i] = input[i]`)
shows how **alignment** governs whether the whole-element load/store fuses into
a vectorized `ld/st.global.v*` or stays scalar.

## What This Example Does

For every type, `main`:

- asserts the Rust layout matches CUDA (`size_of` / `align_of`),
- runs the copy on the GPU and checks the round-trip is bit-correct,
- reads the emitted PTX and reports the load it lowered to, asserting the
  alignment-gated invariant: **align ≥ 16 ⇒ vectorized**.

```text
rust        cuda           size align  ptx load            vectorized
i32x3       int3             12     4   ld.global.b32       no
i32x4       int4             16    16   ld.global.v2.b64    yes
f32x4       float4           16    16   ld.global.v2.b64    yes
f64x2       double2          16    16   ld.global.v2.b64    yes
i64x4_a32   longlong4_32a    32    32   ld.global.v4.b64    yes
double4_32a ...                          (256-bit, wider)
```

## Key Concepts Demonstrated

### Alignment is what unlocks vectorization

In LLVM, over-alignment is an *operation* property (`align N` on a load/store),
not a type property. The codegen threads each type's real ABI alignment onto its
loads/stores, and the backend's load/store vectorizer fuses them only when the
alignment permits. So:

- `float4` / `int4` / `longlong2` / `double2` (align 16) → one 128-bit
  `ld.global.v2.b64`.
- `float3` / `int3` (align 4) → scalar — a vector access would need 16-byte
  alignment the type does not guarantee, so emitting one would be unsound.

### Alignment width sets the vector width

The deprecated `double4` (align 16) and its `double4_32a` variant (align 32)
have the *same* 32-byte size but different alignment — and that alone changes
the codegen:

- `f64x4` (align 16) → `ld.global.v2.b64` (two 128-bit transactions),
- `f64x4_a32` (align 32) → `ld.global.v4.b64` (a single 256-bit transaction).

## Running

```bash
cargo oxide run vectorization
# ✓ SUCCESS: 26 CUDA vector types -- layout, copy, and codegen all correct
```
