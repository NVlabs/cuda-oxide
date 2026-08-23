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

### On Blackwell Datacenter (sm_100a):

```text
=== TMA Multicast Example (sm_100a) ===

GPU Compute Capability: sm_100
Loading PTX from: tma_multicast.ptx
✓ PTX loaded successfully

--- TMA Multicast (tma_multicast_test) ---

1. Setup: 4 CTAs in cluster, tile 64x64 (4096 floats)
2. Launching tma_multicast_test (cluster=(4,1,1), block=256)...
3. Verifying all 4 CTAs received the same tile...
   ✓ All 4 CTAs have identical tile data (4096 values each)!

🎉 TMA multicast successful — one load, 4 CTAs served!

=== TMA Multicast Test Complete ===
```

### On Consumer Blackwell (sm_120)

Consumer Blackwell is supported by observed runtime evidence. PR #668
confirmed that the example's PTX JIT-loads and runs successfully on an
RTX 5090 (sm_120).

The previous documentation showing a PTX load failure on sm_120 was stale.

### On Hopper (sm_90/sm_90a)

Hopper runtime behavior has not yet been validated on physical Hopper hardware.
The current host-side example still skips devices with compute capability
below 10, so this documentation change does not claim Hopper runtime support.

## Hardware Requirements

- **Blackwell**: sm_100a datacenter GPUs; consumer sm_120 reported working in #668
- **Hopper (sm_90/sm_90a)**: runtime validation pending; currently skipped by the host example
- **CUDA Driver**: 12.0+
- **Cluster launch**: Required (`cuLaunchKernelEx` with cluster dimensions)

## Multicast vs Unicast TMA

| Aspect              | Unicast (`tma_copy`)             | Multicast (`tma_multicast`)                  |
|---------------------|----------------------------------|----------------------------------------------|
| Destination         | One CTA's shared memory          | All CTAs in cluster                          |
| Bandwidth           | 1x tile transfer                 | 1x transfer, N copies                        |
| Architecture        | sm_90+ (Hopper+)                 | Blackwell; sm_120 reported working           |
| Use case            | Single-CTA tile loads            | GEMM/convolution with shared tiles           |
| `cluster_launch`    | Optional                         | Required                                     |

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

## Consumer Blackwell Compatibility

PR #668 established through runtime measurement that this example's PTX
JIT-loads and executes successfully on an RTX 5090 (sm_120). Therefore,
consumer Blackwell must not be documented as unsupported.

This result does not establish Hopper runtime support. Hopper remains a
separate validation question, and the current host-side example continues
to skip devices with compute capability below 10.
