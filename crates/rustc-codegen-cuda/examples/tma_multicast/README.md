# tma_multicast

## TMA Multicast Cluster Broadcast

Demonstrates TMA multicast: a single `cp.async.bulk.tensor` load broadcasts a
tile from global memory into the shared memory of **every CTA** in a thread
block cluster. One instruction, N copies — no extra bandwidth or thread work.

For basic TMA copies (sm_90+), see the [`tma_copy`](../tma_copy/) example.

## What This Example Does

Launches a cluster of 4 CTAs. CTA-0 thread-0 issues one multicast TMA copy.
The hardware delivers an identical 64x64 tile to all 4 CTAs' shared memory.
Each CTA then writes its tile to global memory for host-side verification.

## Key Concepts

### Multicast TMA Instruction

The multicast variant adds a `cta_mask` bitmask that selects which CTAs in
the cluster receive the tile:

```rust
#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn tma_multicast_test(tensor_map: *const TmaDescriptor, ...) {
    // ...init barriers, cluster_sync, arrive...

    if cluster::block_rank() == 0 && thread::threadIdx_x() == 0 {
        let cta_mask: u16 = 0b1111; // all 4 CTAs
        cp_async_bulk_tensor_2d_g2s_multicast(
            &raw mut TILE as *mut u8,
            tensor_map,
            tile_x, tile_y,  // element offsets, not tile indices!
            &raw mut BAR,
            cta_mask,
        );
    }

    // ALL CTAs wait on their local barrier — each gets the tile
    while !mbarrier_try_wait(&raw const BAR, token) {}
}
```

### Cluster Synchronization

Every CTA must have its mbarrier initialized before the multicast fires,
because the TMA writes to all CTAs' shared memory and signals all their
barriers:

```text
CTA-0: mbarrier_init → fence ─┐
CTA-1: mbarrier_init → fence ─┤
CTA-2: mbarrier_init → fence ─┼→ cluster_sync() → CTA-0 issues multicast
CTA-3: mbarrier_init → fence ─┘
```

### CTA Mask

The `cta_mask` is a bitmask over cluster ranks. With a `(4,1,1)` cluster:

| Mask     | Effect                             |
|----------|------------------------------------|
| `0b1111` | All 4 CTAs receive the tile        |
| `0b0101` | Only CTAs 0 and 2                  |
| `0b0001` | Only CTA 0 (equivalent to unicast) |

## Generated PTX

```ptx
.target sm_100a
.explicitcluster
.reqnctapercluster 4, 1, 1

// Multicast TMA: broadcasts tile to all CTAs matching cta_mask
cp.async.bulk.tensor.2d.shared::cluster.global.tile
    .mbarrier::complete_tx::bytes.multicast::cluster
    [%rd_smem], [%rd_tensor_map, {%r_x, %r_y}], [%rd_mbar], %rs_mask;
```

Key PTX differences from unicast TMA:
- `.multicast::cluster` qualifier on the instruction
- Extra `%rs_mask` operand (16-bit CTA bitmask)
- `.explicitcluster` and `.reqnctapercluster` directives on the entry point

## Build and Run

```bash
cargo oxide run tma_multicast
```

## Expected Output

### On Blackwell Datacenter (sm_100a)

```text
=== TMA Multicast Example (sm_100a) ===

GPU Compute Capability: sm_100
✓ embedded module loaded successfully

--- TMA Multicast (tma_multicast_test) ---

1. Setup: 4 CTAs in cluster, tile 64x64 (4096 floats)
2. Launching tma_multicast_test (cluster=(4,1,1), block=256)...
3. Verifying all 4 CTAs received the same tile...
   ✓ All 4 CTAs have identical tile data (4096 values each)!

🎉 TMA multicast successful — one load, 4 CTAs served!

=== TMA Multicast Test Complete ===
```

### On Consumer Blackwell (sm_120)

Runs on consumer Blackwell: the sm_100a PTX JIT-loads and passes (first
verified on an RTX 5090 in #668). Transcript from an RTX 5090:

```text
=== TMA Multicast Example (sm_100a) ===

GPU Compute Capability: sm_120
✓ embedded module loaded successfully

--- TMA Multicast (tma_multicast_test) ---

1. Setup: 4 CTAs in cluster, tile 64x64 (4096 floats)
2. Launching tma_multicast_test (cluster=(4,1,1), block=256)...
3. Verifying all 4 CTAs received the same tile...
   ✓ All 4 CTAs have identical tile data (4096 values each)!

🎉 TMA multicast successful — one load, 4 CTAs served!

=== TMA Multicast Test Complete ===
```

### On Hopper (sm_90/sm_90a)

Legal per the ISA, hardware run pending. Multicast is sm_90+ in the PTX ISA,
and `--arch=sm_90a` builds and assembles. What the default build cannot do is
*JIT* there, because it ships `.target sm_100a`, so the driver declines it and
the host reports its own clean skip:

Nobody here has run this on Hopper, so the block below is the shape of the
message the host emits on that path, not a captured transcript:

```text
GPU Compute Capability: sm_90

skipping: this build targets sm_100a and the driver declined it
  driver reported: ...
  The sm_100a build runs on B100/B200/GB200 and on sm_120.
  Multicast itself is sm_90+; for Hopper, rebuild with --arch=sm_90a.
  For basic TMA tests, use: cargo oxide run tma_copy
```

Whether an `--arch=sm_90a` build passes on real Hopper hardware is untested;
that run is what closes #966.

## Hardware Requirements

- **Runs on**: Blackwell, both datacenter sm_100a (B100/B200/GB200) and
  consumer sm_120 (RTX 5090, verified in #668)
- **Hopper (sm_90/sm_90a)**: legal per the ISA, hardware run pending. The
  default sm_100a PTX will not JIT there; an `--arch=sm_90a` build assembles
  but has not been run on the hardware (#966)
- **Not supported**: anything below sm_90
- **CUDA Driver**: 12.0+
- **Cluster launch**: Required (`cuLaunchKernelEx` with cluster dimensions)

## Multicast vs Unicast TMA

| Aspect              | Unicast (`tma_copy`)             | Multicast (`tma_multicast`)         |
|---------------------|----------------------------------|-------------------------------------|
| Destination         | One CTA's shared memory          | All CTAs in cluster                 |
| Bandwidth           | 1x tile transfer                 | 1x transfer, N copies               |
| Architecture        | sm_90+ (Hopper+)                 | Blackwell (sm_100a, sm_120)         |
| Use case            | Single-CTA tile loads            | GEMM/convolution with shared tiles  |
| `cluster_launch`    | Optional                         | Required                            |

## Pitfalls

**TMA coordinates are element offsets, not tile indices.** Passing `{1, 0}`
instead of `{64, 0}` for tile (1,0) causes `CUDA_EXCEPTION_27: Warp Illegal
Instruction Parameter` — the hardware requires coordinates aligned to the tile
(box) dimensions. This is easy to miss because `{0, 0}` is trivially aligned
and always works.

**`cluster_sync()` before the multicast is mandatory.** The multicast TMA
writes to every CTA's shared memory and signals every CTA's mbarrier. If any
CTA hasn't finished `mbarrier_init` + `fence_proxy_async_shared_cta` before
the multicast fires, the barrier tracking will be silently corrupt.

## Which GPUs Run This?

The kernel ships as `.target sm_100a` PTX. What matters at run time is
whether the driver JIT accepts that PTX for your GPU:

```text
sm_100a PTX ──JIT──► sm_100a (B100/B200/GB200)  ✓ runs
            ──JIT──► sm_120  (RTX 5090)         ✓ runs (verified in #668)
            ──JIT──► sm_90   (Hopper)           ✗ rejected; host reports a skip

sm_90a PTX  ──JIT──► sm_90   (Hopper)           ? assembles; not yet run (#966)
```

The multicast instruction itself has an sm_90+ floor in the intrinsic catalog,
which matches the PTX ISA: `sm_90a` is advised there for performance, not
required for legality. The host gate is therefore sm_90, not sm_100 — what
keeps the default build off Hopper is its target, not the instruction.
