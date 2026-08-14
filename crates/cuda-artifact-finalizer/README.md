# cuda-artifact-finalizer

Driver-independent finalization of NVVM IR, LTOIR, and PTX into loadable device
code.

This crate is the single owner of cuda-oxide's libNVVM, nvJitLink, and
nvPTXCompiler compilation policy. It deliberately does **not** link the CUDA
Driver, so the same typed target, FMA, debug, input-order, validation, and
provenance rules apply whether an artifact is materialized at build time
(`cargo oxide build --materialize-cubin`) or finalized at run time as a
fallback.

`PtxAssembler` is discovered separately from the ordinary `Finalizer`. This
keeps nvPTXCompiler optional for consumers that only use the NVVM IR pipeline,
while providing a direct PTX-to-cubin boundary when PTX has already been
linked.

Keeping that policy in one driverless crate is what lets the two paths agree.
A rule that lived in the runtime loader alone could not be applied during a
build, and one duplicated across both would drift.

Consumers:

- `cargo-oxide` and `rustc-codegen-cuda`, for build-time materialization;
- `cuda-host`, for the runtime path.
