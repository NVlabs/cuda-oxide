# legacy_nvvm_pointer_shapes

End-to-end regression coverage for pointer flows that are easy to miss when
lowering cuda-oxide's opaque internal pointers to the typed pointers required
by the legacy LLVM 7 NVVM dialect.

The kernel keeps these shapes live across non-inlined device calls:

- a pointer selected by control flow;
- a pointer carried around a runtime loop;
- pointers stored in nested aggregates;
- an array of pointers loaded through a pointer-to-pointer;
- one address read through both `float*` and `i32*`-equivalent views.

The final `exp` call deliberately pulls in libdevice, so the ordinary loading
path also exercises NVVM IR and nvJitLink. Run it normally on any supported
GPU; target detection selects the matching NVVM dialect automatically:

```bash
cargo oxide run legacy_nvvm_pointer_shapes
```

The program exits nonzero on any numerical or pointer-selection mismatch. To
exercise the legacy parser and compiler on a machine whose GPU is newer than
`sm_86`, generate architecture-specific LTOIR without trying to load it on the
local GPU:

```bash
cargo oxide emit-ltoir legacy_nvvm_pointer_shapes --arch sm_86 \
  --output /tmp/legacy_nvvm_pointer_shapes-sm86.ltoir
```

This command invokes the real libNVVM verifier and compiler. Numerical
execution can be checked on a compatible pre-Blackwell GPU or, for an
unsuffixed legacy target, through cuda-oxide's forward PTX bridge on
Blackwell:

```bash
cargo oxide run legacy_nvvm_pointer_shapes --arch sm_86
```

The bridge does not guess across suffixed targets such as `sm_90a`.
