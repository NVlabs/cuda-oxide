# Official CUTLASS translation backend

The translation backend preserves the high-level `dialect-cute` program after
common preparation, verifies the complete semantic story, and maps it directly
to the MLIR dialects accepted by NVIDIA's official CUTLASS 4.7 compiler.

```text
cute-rs kernel
      │
      ▼
high-level dialect-cute
      │
      ├── common preparation
      └── backend-neutral whole-module verification
      │
      ▼
CUTLASS full-CuTe MLIR profile
      │
      ▼
official libCutlassCompiler 4.7
      │
      ▼
validated cubin → embedded host artifact → CUDA launch
```

The backend contains direct mapping packs for all three shared examples:

- tensor tiling and copying for `elementwise_cute`;
- scaled tensor views and NVFP4 GEMV for `nvfp4_gemv_cute`;
- scheduler, work tiles, TMA load pipelines, shared-memory MMA, and epilogue
  stores for `blockscale_gemm_cute`.

It does not run a native CuTe expansion pass or lower through backend-specific
MIR/NVVM leaf intrinsics. In the GEMM epilogue,
`ReadyForTma` directly emits the async-shared proxy publication fence followed
by the counted CTA hand-off barrier. `Reusable` emits only the counted barrier.

## Install and select the compiler

Install the pinned compiler archive once:

```bash
cargo oxide toolchain install cutlass
```

The installer verifies the official archive and `libCutlassCompiler.so`
digests. With no explicit compiler path, `cargo oxide` resolves this managed
installation and includes its library digest in the build fingerprint.

Select the backend for a build:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide build elementwise_cute --arch sm_120a

CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide build nvfp4_gemv_cute --arch sm_120a

CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
  cargo oxide build blockscale_gemm_cute --arch sm_120a
```

An explicit absolute library path overrides the managed installation:

```bash
CUDA_OXIDE_DEVICE_BACKEND=cutlass-mlir \
CUDA_OXIDE_CUTLASS_COMPILER=/absolute/path/to/libCutlassCompiler.so \
  cargo oxide build elementwise_cute --arch sm_120a
```

Use `CUDA_OXIDE_MLIR_OUTPUT=/absolute/path/module.mlir` for an optional textual
observation. MLIR dumps and generated cubins are build artifacts and must not be
committed.

The ordinary CUDA Oxide MIR/NVVM/LLVM-to-PTX backend remains the default for
non-CuTe kernels. Selecting the CUTLASS backend changes only the continuation
after shared preparation and semantic verification.
