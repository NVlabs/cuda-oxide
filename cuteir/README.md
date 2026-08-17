# CuTe-style GPU kernels in Rust

`cuteir` lets a Rust kernel describe GPU data by its meaning, not only by its
address.

The two basic ideas are small:

- a **Tensor** is storage plus a layout;
- a **Layout** says where a logical coordinate lives in that storage.

That is enough to express operations such as “split this vector into
four-value tiles” or “copy this matrix tile into shared memory.” The compiler
can then choose how to lower those operations without asking the kernel author
to rewrite the pointer arithmetic by hand.

```text
plain storage + layout
          │
          ▼
        Tensor ──► tile ──► slice ──► copy / add / matrix multiply
```

## Semantic frontend

An **IR** is the compiler's working form of a program. A **dialect** is a named
set of operations in that IR. A **backend** turns the IR into code another
compiler or the GPU can consume.

The frontend keeps CuTe operations in `dialect-cute` while they are still
recognizable:

```text
Rust kernel using cute-rs
Tensor → zipped_divide → slice → copy / add / matrix multiply
                         │
                         ▼
              high-level dialect-cute
                         │
                         ▼
             preparation + verification
                         │
                         ▼
             selected backend continuation
```

The whole-module verifier checks that tensor provenance, scheduler and
pipeline state, TMA transactions, shared-memory MMA, and epilogue operations
form a complete semantic story. It does not choose or execute a backend.

## What lives here

| Path | What it does |
| :--- | :--- |
| [`cute-layout`](cute-layout) | Layout math shared by the Rust API and compiler. |
| [`cute-rs`](cute-rs) | Device-facing Tensor, Layout, copy, pipeline, and matrix multiply-and-accumulate (MMA) types. |
| [`dialect-cute`](dialect-cute) | CuTe operations and backend-neutral whole-module verification in Pliron. |
| [`elementwise_cute`](examples/elementwise_cute) | Adds two vectors through per-thread Tensor tiles. |
| [`nvfp4_gemv_cute`](examples/nvfp4_gemv_cute) | Multiplies a packed FP4 matrix by a packed FP4 vector. |
| [`blockscale_gemm_cute`](examples/blockscale_gemm_cute) | Multiplies two block-scaled FP4 matrices with the Tensor Memory Accelerator (TMA) and tensor cores. |

The examples are deliberately ordered from smallest to largest:

```text
elementwise       GEMV                    GEMM
vector tiles  →   scaled row tiles   →    global/shared matrix tiles
load + add         load + dot             TMA + pipeline + MMA + epilogue
```

## Run the examples

Run these commands from the repository root after completing the normal
CUDA Oxide setup:

```bash
cargo oxide run elementwise_cute

cargo oxide run nvfp4_gemv_cute --arch sm_120a -- \
  --m 512 --k 256 --l 1

cargo oxide run blockscale_gemm_cute --arch sm_120a
```

The elementwise example uses ordinary `f32` and `f16` operations. The two FP4
examples use Blackwell instructions and should be built for `sm_120a`; their
own READMEs give the exact device and shape requirements.

Each program checks every GPU output bit pattern against a deterministic host
result before reporting success. Backend-specific artifact and code-shape
checks live with the backend implementation rather than in this shared layer.

On the translation branch, the [official CUTLASS backend](docs/translation.md)
maps these semantic operations directly to the CUTLASS 4.7 MLIR profile and
embeds the compiler-produced cubin. It does not pass through native CuTe
expansion.
