# mir-transforms

Optimization passes over the `dialect-mir` IR.

These run in the middle of cuda-oxide's pipeline: after `mem2reg` has promoted
memory slots to plain SSA values, and before the IR is lowered to the LLVM
dialect on its way to PTX. Running here means a pass sees Rust-level structure
— typed aggregates, slices, checked arithmetic — that is gone by the time LLVM
IR exists.

The first pass is loop unrolling, requested by the `#[unroll]` /
`#[unroll(N)]` annotation, which the importer records as a `mir.unroll_hint`
operation inside the annotated loop. Further loop passes belong here too.
