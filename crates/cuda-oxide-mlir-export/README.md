# CUDA Oxide MLIR export

This crate contains CUDA Oxide's target-specific mappings for
`pliron-mlir-export`. It is the CUTLASS translation backend. The generic
crate builds typed, deterministic MLIR text; this crate explains what each
CUDA Oxide operation means to one pinned MLIR consumer.

```text
pliron-mlir-export                 this crate
------------------                ----------
syntax tree + renderer      +     builtin / MIR / NVVM / CuTe mappings
registry + diagnostics            pinned consumer profiles
```

The split is deliberate. The generic crate can translate any Pliron dialect.
This crate decides what CUDA Oxide operations mean in one exact MLIR consumer.

The first profile targets the official CUTLASS 4.7 compiler library. CUDA
Oxide installs the exact NVIDIA release archive, verifies both the archive and
library digests, and loads that versioned library explicitly:

```text
shared high-level Pliron module
                │
                ▼
    shared CuTe safety checks
                │
                ▼
 builtin + MIR + NVVM + CuTe recipes
                │
                ▼
      generic MLIR renderer
                │
                ▼
 official CUTLASS 4.7 compiler
```

The elementwise, NVFP4 GEMV, and block-scaled GEMM mapping packs translate the
live post-preparation module directly. The pinned library compiles that MLIR to
a validated cubin, which is embedded and launched through CUDA Oxide's ordinary
artifact loader. Set `CUDA_OXIDE_MLIR_OUTPUT=<file>` to retain the exact textual
module for inspection; select `CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir` to compile
it.

## The first mapping pack

The first pack covers the ordinary scalar code around a CuTe kernel:

```text
mir.func / mir.return        → func.func / func.return
mir.goto / mir.cond_br       → cf.br / cf.cond_br
mir.constant                 → arith.constant
mir.add / sub / mul          → arith.add* / sub* / mul*
mir.div / rem                → signed or unsigned arith operation
mir.lt / le / gt / ge        → arith.cmpi or arith.cmpf
mir.eq / ne                  → arith.cmpi or arith.cmpf
mir.cast                     → the matching arith conversion
```

Rust's `usize` is a 64-bit unsigned integer on the CUDA target. MLIR integer
types do not write signedness in the type, so it becomes `i64`:

```text
Pliron                           MLIR
builtin.integer ui64            i64
mir.div on that ui64            arith.divui
mir.lt  on that ui64            arith.cmpi ult
```

The operation name keeps the part that matters. A signed `i64` division uses
`arith.divsi`; an unsigned `u64` division uses `arith.divui`. The translator
looks at the Pliron type before both types become MLIR `i64`.

Control flow maps to `cf`, not `scf`. Rust MIR is already a graph of basic
blocks and can jump in shapes that are not a neat `if` or `for` loop:

```text
        mir.cond_br
          /      \
    block A     block B
          \      /
            join

                ↓

         cf.cond_br
          /      \
    block A     block B
          \      /
            join
```

Later MLIR passes may rebuild structured loops. The exporter does not guess
that structure while translating the graph.

## Pointers and small Rust values

The elementwise kernel still contains ordinary Rust values around its CuTe
views. They keep their physical shape when they cross into MLIR:

```text
mir.ptr<T>                 → !llvm.ptr
mir.slice<T>               → !llvm.struct<(!llvm.ptr, i64)>
mir.array<T, 4>            → !llvm.array<4 x T>
mir.struct<A, B>           → !llvm.struct<(A, B)>

mir.alloca / load / store  → llvm.alloca / load / store
mir.field_addr             → llvm.getelementptr
mir.extract_field          → llvm.extractvalue
```

Rust can reorder fields and insert padding. The translator reads the MIR
layout and rebuilds that exact physical order. It refuses packed or otherwise
different by-value layouts instead of silently changing an address.

The CUDA kernel marker also has an exact mapping:

```text
gpu_kernel = "true"        → cute.kernel
anything else              → translation error
```

## The first CuTe slice

The elementwise path keeps the useful layout explanation instead of reducing
everything to pointer arithmetic first:

```text
make tensor
    │
    ▼
divide it into tiles
    │
    ▼
pick this thread's tile
    │
    ├── full tile ──► cute.copy
    │
    └── edge tile ──► checked scalar load/store
```

The profile recognizes the corresponding `cute.tensor_*` operations and maps
them to CUTLASS's `cute` and `cute_nvgpu` operations. Before translation, it
runs the shared backend-neutral whole-module verifier directly. The exporter
does not invoke native CuTe expansion or use native generated-intrinsic markers
as an intermediate representation.

Thread, block, and grid coordinates are direct NVVM mappings. For example:

```text
nvvm.read_ptx_sreg_tid_x    → nvvm.read.ptx.sreg.tid.x
nvvm.read_ptx_sreg_ntid_y   → nvvm.read.ptx.sreg.ntid.y
nvvm.read_ptx_sreg_nctaid_z → nvvm.read.ptx.sreg.nctaid.z
```

Only the reviewed coordinate reads are registered. Other NVVM operations fail
with a missing-mapping error until they get their own recipe.
