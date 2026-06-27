# issue98_repro

End-to-end regression for [issue #98](https://github.com/NVlabs/cuda-oxide/issues/98)
and the matching community reports from pre-Blackwell users.

The example covers the original SwiGLU/`exp` kernel and the complete reported
math set: `sin`, `exp`, `sqrt`, `atan`, `atan2`, `acos`, and `tan`. Kernels use
the same `&[f32]` plus `DisjointSlice` argument shape as the reports.

On a pre-Blackwell GPU, the ordinary command is the important regression:

```bash
cargo oxide run issue98_repro
```

The math calls automatically select the libdevice/NVVM-IR route, and the
detected device architecture must automatically select legacy LLVM 7 typed
pointer output. No explicit NVVM or architecture flag should be necessary.

To verify and compile the same legacy dialect without a pre-Blackwell GPU:

```bash
cargo oxide emit-ltoir issue98_repro --arch sm_86 \
  --output /tmp/issue98_repro-sm86.ltoir
```

`emit-ltoir` passes the generated module through the real libNVVM verifier and
compiler. Numerical execution can then be checked either on a compatible
pre-Blackwell GPU or, for an unsuffixed legacy target, through cuda-oxide's
forward PTX bridge on Blackwell:

```bash
cargo oxide run issue98_repro --arch sm_86
```

The bridge does not guess across suffixed targets such as `sm_90a`.
