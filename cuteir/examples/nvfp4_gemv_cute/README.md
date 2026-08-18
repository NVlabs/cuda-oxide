# Packed FP4 GEMV with typed scales

GEMV means “matrix times vector”:

```text
A[M,K] × x[K] = y[M]

y[row] = A[row,0] × x[0]
       + A[row,1] × x[1]
       + ...
```

This example stores the matrix and vector as packed E2M1 values. E2M1 is a
four-bit floating-point format, so two values fit in one byte. Every group of
16 values has an eight-bit UE8M0 scale. UE8M0 stores a power-of-two multiplier.

```text
16 packed values: v0 v1 v2 ... v15
                         │
                         └── one scale applies to the whole group

value used by the dot product = E2M1 value × UE8M0 scale
```

The exact storage contract matters more than the short format name: this
example uses E2M1 data, UE8M0 scales, and groups of 16.

## The Tensor flow

One GPU thread computes one output row. `BlockScaledTensor` keeps the packed
values, their scales, and their layout together.

```text
packed A bytes + A scales ──► BlockScaledTensor<Mkl>
                                      │
                                   row view
                                      │
                                  K=64 tile ──► load + convert ──┐
                                                                │
packed x bytes + x scales ──► BlockScaledTensor<Nkl>             │
                                      │                         │
                                   row view                     │
                                      │                         │
                                  K=64 tile ──► load + convert ──┤
                                                                ▼
                                                        scaled dot product
                                                                │
                                                     repeat across K tiles
                                                                │
                                                                ▼
                                                           one f16 y value
```

The important types are visible at the point where the views are made:

```rust
type MatrixA<'a> =
    BlockScaledTensor<'a, E2M1, UE8M0, 16, KMajor<Mkl>>;
type VectorB<'a> =
    BlockScaledTensor<'a, E2M1, UE8M0, 16, KMajor<Nkl>>;
```

They record:

- the packed value format (`E2M1`);
- the scale format (`UE8M0`);
- the 16-value scale group;
- the K-contiguous layout;
- whether a view plays the matrix or vector role.

`dot_accumulate` accepts the matching matrix tile and vector tile, so a role
or format mismatch is rejected before the kernel runs.

## Shape and device requirements

The packed FP4 operations in this example target `sm_120a`. Use a compatible
Blackwell GPU to execute it.

The command-line shapes must follow these rules:

```text
M > 0 and M is a multiple of 128
K > 0 and K is a multiple of 64
L > 0
```

`L` is the batch count. The default problem is `M=512`, `K=256`, `L=1`.

## Build and run it

From the repository root, build or run this example through the native CuTe
path:

```bash
cargo oxide build nvfp4_gemv_cute --arch sm_120a

cargo oxide run nvfp4_gemv_cute --arch sm_120a -- \
  --m 512 --k 256 --l 1
```

The native path expands the high-level `cute.*` operations through the
in-tree MIR/NVVM/LLVM continuation and emits PTX for the selected target.

Use `--help` to list the shape and device options. Every run performs all
correctness checks.

A successful run compares every GEMV output `f16` bit pattern with the host
calculation. This example is the semantic checkpoint for block-scaled views,
K tiles, packed loads, and a reduction.

## Single-kernel comparison

Measured on 2026-08-18 at commit `de449613a942` on an RTX 5090 (`sm_120`),
Nsight Systems 2026.1.3 reported these kernel-active medians for `M=512`,
`K=256`, `L=1`:

| Implementation | Median |
| :--- | ---: |
| Native CuTe proto | **2.304 µs** |
| CuTeDSL 4.6.2 | **2.240 µs** |

The native proto is 0.064 µs, or 2.86%, longer. All 102 native runs passed
bit-exact validation with output hash `f1c627726a82c6bd`, matching the separate
CuTeDSL correctness run.

Each observation is one main-kernel launch; setup and CuTeDSL conversion
kernels are excluded. The native samples came from 102 independently
validated processes because this host harness launches once per process,
whereas CuTeDSL reused one process. That cache/context difference means this
GEMV comparison is not a perfectly matched hot-reuse experiment.
