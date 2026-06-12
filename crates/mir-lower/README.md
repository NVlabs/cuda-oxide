# mir-lower

`dialect-mir` → LLVM dialect lowering pass for cuda-oxide.

Converts [`dialect-mir`](../dialect-mir/) operations into LLVM dialect
operations (the LLVM dialect is provided by `pliron-llvm`), with GPU-specific
operations lowered to NVVM intrinsics or inline PTX assembly. This is the
bridge between Rust semantics and LLVM's target-agnostic IR.

## Pipeline Position

```text
Rust Source Code
       │
       ▼
┌──────────────┐
│   rustc      │  (extracts Stable MIR)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ mir-importer │  (Stable MIR → dialect-mir, then mem2reg)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  mir-lower   │  ◄── THIS CRATE (dialect-mir → LLVM dialect)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ llvm-export  │  (exports to LLVM IR)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     llc      │  (LLVM IR → PTX)
└──────────────┘
```

## How It Works

The crate uses pliron's `DialectConversion` framework. Each
`dialect-mir` / `dialect-nvvm` op declares its own lowering via the
`MirToLlvmConversion` op interface. The framework handles IR walking,
def-before-use ordering, type conversion, and block argument patching
automatically.

For each `MirFuncOp`, `convert_func` (in `lowering.rs`):

1. Creates an LLVM dialect function with a flattened type signature
2. Propagates GPU kernel attributes (`gpu_kernel`, `maxntid`, etc.)
3. Uses `inline_region` to move the `dialect-mir` blocks into the new function
4. Builds an entry prologue that reconstructs aggregates (slices, structs)
   from the flattened LLVM dialect arguments via `insertvalue`
5. Branches to the original entry block with the reconstructed values

## Module Structure

### Core Modules

| Module                    | Purpose                                                    |
|---------------------------|------------------------------------------------------------|
| `lowering`                | `convert_func` — per-function lowering via `inline_region` |
| `conversion_interface`    | `MirToLlvmConversion` op interface trait                   |
| `convert/interface_impls` | Op interface impls dispatching to converter functions      |
| `context`                 | CUDA-specific state maps (shared globals, dynamic smem)    |
| `helpers`                 | Constants, intrinsic declarations, utilities               |

### Operation Converters (`convert/ops/`)

| Module         | `dialect-mir` Operations Handled                                                                               |
|----------------|----------------------------------------------------------------------------------------------------------------|
| `arithmetic`   | `mir.add`, `mir.sub`, `mir.mul`, `mir.div`, `mir.rem`, checked variants, shifts, bitwise, `mir.neg`, `mir.not` |
| `memory`       | `mir.alloca`, `mir.load`, `mir.store`, `mir.ref`, `mir.assign`, `mir.ptr_offset`                               |
| `control_flow` | `mir.return`, `mir.goto`, `mir.cond_br`, `mir.assert`, `mir.unreachable`, `mir.storage_live`/`dead` (erased)   |
| `constants`    | `mir.constant`, `mir.float_constant`, `mir.undef`                                                              |
| `cast`         | `mir.cast` (widening, narrowing, int↔float, ptr)                                                               |
| `aggregate`    | Struct/tuple/array/enum extract, insert, construct, field/element addr                                         |
| `call`         | `mir.call` (function calls with arg flattening)                                                                |

### Type Converter (`convert/types.rs`)

| `dialect-mir` Type   | LLVM dialect Type                                   |
|----------------------|-----------------------------------------------------|
| `mir.tuple`          | `llvm.struct` (anonymous, ZST fields dropped)       |
| `mir.ptr`            | `llvm.ptr` with address space                       |
| `mir.array`          | `llvm.array`                                        |
| `mir.slice`          | `llvm.struct {ptr, i64}`                            |
| `mir.disjoint_slice` | `llvm.struct {ptr, i64}` (same as slice)            |
| `mir.struct`         | `llvm.struct` (padded if layout known, else flat)   |
| `mir.enum`           | `llvm.struct` matching rustc's byte layout          |

### GPU Intrinsic Converters (`convert/intrinsics/`)

| Module     | Intrinsics                              | Strategy        | GPU       |
|------------|-----------------------------------------|-----------------|-----------|
| `basic`    | Thread/block IDs, `barrier0`            | LLVM intrinsics | All       |
| `warp`     | Shuffle, vote, lane operations          | LLVM intrinsics | All       |
| `debug`    | `vprintf`, clock, trap                  | LLVM intrinsics | All       |
| `atomic`   | Scoped GPU + `core::sync` atomics       | LLVM intrinsics | sm_70+    |
| `mbarrier` | Async barriers                          | LLVM intrinsics | sm_90+    |
| `cluster`  | Block clusters, DSMEM                   | LLVM intrinsics | sm_90+    |
| `tma`      | Tensor Memory Accelerator               | LLVM intrinsics | sm_90+    |
| `stmatrix` | Shared memory matrix store              | Inline PTX      | sm_90+    |
| `wgmma`    | Warpgroup MMA                           | Inline PTX      | sm_90     |
| `tcgen05`  | 5th-gen Tensor Cores, TMEM              | Inline PTX      | sm_100+   |
| `clc`      | Cluster Launch Control                  | LLVM intrinsics | sm_100+   |
| `common`   | Shared helpers across intrinsic modules | —               | —         |

## DialectConversion Framework

The lowering uses pliron's `DialectConversion` + `DialectConversionRewriter`
rather than manual walk-and-replace. The framework manages:

- **Value mapping**: source (`dialect-mir`) → target (LLVM dialect) value tracking
- **Type conversion**: registered via `can_convert_type` / `convert_type`
- **Block argument patching**: automatic type conversion of block args
- **Def-before-use ordering**: operations are visited in correct order

Each converter function receives `(ctx, rewriter, op, operands_info)` and
uses `rewriter.insert_operation()` / `rewriter.replace_operation_with_values()`
to emit LLVM dialect operations.

## Lowering Strategies

### LLVM Intrinsic Calls

For operations with direct NVVM equivalents (thread IDs, barriers,
atomics, TMA):

```text
dialect-mir/dialect-nvvm: nvvm.read_ptx_sreg_tid_x
LLVM dialect:             call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
```

### Inline PTX Assembly

For complex operations or when LLVM intrinsics don't exist (WGMMA,
tcgen05, stmatrix). Uses `convergent` attribute to prevent LLVM from
moving warp-synchronous ops across control flow:

```text
dialect-nvvm:  nvvm.tcgen05_mma_ws_f16
LLVM dialect:  call void asm "tcgen05.mma.cta_group::1.kind::f16...", "..." #convergent
```

## Shared Memory Handling

- **Static** (`SharedArray<T, N>`): Lowered to `@__shared_*` globals
  in address space 3 with deduplication via `SharedGlobalsMap`.
- **Dynamic** (`DynamicSharedArray<T>`): Lowered to `@__dynamic_smem_*`
  extern globals. `DynamicSmemAlignmentMap` tracks max alignment per
  kernel for correct PTX metadata.

## Dependencies

- [pliron](https://github.com/vaivaswatha/pliron) — Pliron IR (MLIR-like) framework
- [dialect-mir](../dialect-mir/) — Source dialect (pliron dialect modelling Rust MIR)
- [llvm-export](../llvm-export/) — pliron-llvm shim + textual `.ll` exporter
- [dialect-nvvm](../dialect-nvvm/) — NVVM intrinsic ops

## Further Reading

- [mir-importer](../mir-importer/) — produces `dialect-mir` from rustc
- [llvm-export](../llvm-export/) — exports textual LLVM IR from an LLVM dialect module
