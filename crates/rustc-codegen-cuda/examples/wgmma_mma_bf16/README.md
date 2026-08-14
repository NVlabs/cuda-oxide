# wgmma_mma_bf16

Compile-only integration example for BF16 Hopper WGMMA lowering.

The example covers the public `cuda_device::wgmma` API from Rust source through
MIR import, WGMMA region selection, LLVM lowering, and PTX generation.

## What this tests

The crate contains two compile-only kernels.

### Full drain

```text
wgmma_fence
mma acc0
commit_group
wait_group<0>
```

For the canonical `[[f32; 8]; 4]` accumulator, the compiler selects the
value-threaded path and exposes the 32 accumulator values to LLVM only outside
the complete asynchronous WGMMA lifetime.

### Partial wait

```text
wgmma_fence

mma acc0
commit_group

mma acc1
commit_group

wait_group<1>
wait_group<0>
```

The two independent accumulator objects form two accumulator slots. The
compiler can therefore keep two committed groups in flight, lower the static
`wait_group<1>`, and still require a final `wait_group<0>` before either
accumulator is observed.

The example deliberately stops after two groups. Round-robin slot reuse and
the associated legality checks are covered by `mir-lower` integration tests.

## Usage

```bash
cargo oxide build wgmma_mma_bf16 --arch sm_90a
```

The command must complete successfully and generate PTX containing the BF16
WGMMA instruction and the partial wait.

To run the repository smoketest:

```bash
scripts/smoketest.sh -x -v '^wgmma_mma_bf16$'
```

## Expected smoketest marker

```text
SUCCESS: BF16 WGMMA value-threaded and partial-wait lowering compiled.
```

## Important

This is a compile-only example.

Both kernels use zero-valued WGMMA descriptors so compilation and PTX
generation can be tested without allocating Hopper shared-memory tiles. The
kernels must not be launched with those descriptors.

Functional execution requires an `sm_90a` Hopper GPU, valid shared-memory
descriptors, and warpgroup-uniform participation.

## Supported lowering shapes

The current BF16 `m64n64k16.f32.bf16.bf16` lowering recognizes three
conservative shapes:

- linear full-drain regions ending in `wait_group<0>`;
- a canonical counted K-loop with affine descriptor recurrences;
- straight-line static partial-wait pipelines using `N + 1` independent
  accumulator slots for `wait_group<N>`.

Every accepted asynchronous lifetime ends in `wait_group<0>` before
accumulator values escape.

Unsupported pointer shapes retain the deferred pointer-form fallback where the
full-drain sequence can still be proven safe. Dynamic waits, unsupported
control flow, malformed pipeline schedules, and the F16/TF32 compatibility
entry points fail closed.
