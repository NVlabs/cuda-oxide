# wgmma_mma_bf16

Compile-only example for BF16 Hopper WGMMA MMA lowering.

This example validates the deferred 32-register accumulator adapter for:

```text
wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16
```

## What this tests

The compiler recognizes the following statically linear sequence:

```text
wgmma_fence
one or more BF16 WGMMA MMA calls using the same accumulator
wgmma_commit_group
wgmma_wait_group::<0>
```

The sequence is fused into one convergent inline-PTX scope. The scope:

  1. loads the 32 per-thread accumulator values;
  2. executes wgmma.fence;
  3. issues the BF16 WGMMA instructions;
  4. commits the group;
  5. waits for all pending groups;
  6. stores the accumulator values only after wait_group<0>.

## Usage

```bash
cargo oxide build wgmma_mma_bf16 --arch sm_90a
```

The command must complete successfully and generate PTX containing the BF16
WGMMA instruction.

To run the repository smoketest:

```bash
scripts/smoketest.sh -x -v '^wgmma_mma_bf16$'
```

## Expected smoketest marker:

```text
SUCCESS: BF16 WGMMA deferred accumulator lowering compiled.
```

## Important

This is a compile-only example.

The kernel uses zero-valued WGMMA descriptors so that compilation and PTX
generation can be tested without allocating Hopper shared-memory tiles. The
kernel must not be launched with those descriptors.

Functional execution requires a sm_90a Hopper GPU and valid shared-memory
descriptors.

## Current limitations

The initial lowering supports only:

```text
m64n64k16.f32.bf16.bf16
```

It rejects:

  - F16 and TF32 variants;
  - partial waits;
  - multiple accumulator objects;
  - multiple commit operations;
  - branches and control-flow joins;
  - sequences that span a loop boundary (a complete fence-to-wait
    sequence inside a loop body fuses, paying the accumulator memory
    round-trip each iteration);
  - incomplete fence/MMA/commit/wait sequences.
