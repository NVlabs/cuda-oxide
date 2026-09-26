# FP8 WGMMA

This runnable Hopper example checks the complete cuda-oxide path for:

```text
wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3
```

It also measures the existing BF16 `m64n64k16` path under the same effective
matrix workload.

## Run

For the complete H100/H200 evidence collection, follow [the short handoff](HOPPER_VALIDATION.md).
The runner's failure handling can be tested without a GPU from the repo root:
`python3 scripts/test-fp8-wgmma-runner.py`.

```bash
cargo oxide run wgmma_mma_fp8 --arch sm_90a
```

Execution requires an H100 or H200. On another architecture the binary checks
the generated PTX for the FP8 and BF16 instructions and exits without printing
the runtime `SUCCESS` marker.

Pass `-- --check-only` to run the all-output checks and one launch of each
benchmark kernel, without warmups or timing. This mode fails on a non-Hopper GPU.

## Numeric checks

The host creates small integers in `[-3, 3]`, encodes them directly as E4M3 or
BF16, and copies their raw bytes to the device. Both formats represent these
values exactly. Each kernel uses one 4096-byte, 256-byte-aligned shared buffer:

- A occupies bytes `0..2048`; B occupies bytes `2048..4096`.
- Both operands are K-major with a 32-byte K span.
- Logical byte offset `row * 32 + byte` is stored at
  `Swizzle::<1, 4, 7>::apply(offset)`, matching the descriptor's 32-byte
  swizzle.
- Every writer issues `fence.proxy.async.shared::cta`, then the CTA executes
  `sync_threads`, before WGMMA reads shared memory through the async proxy.

The accumulator starts at `1.0`, so the check covers the input accumulator as
well as the matrix product. After `commit_group` and `wait_group<0>`, each of
the 128 threads stores its 32 accumulator registers contiguously. The host
scatters those fragments into the 64x64 result layout and compares every value
exactly with an F32 reference decoded from the raw input bytes. A one-MMA BF16
result is checked the same way.

Exact comparison is intentional for this dataset: products are integers with
absolute value at most 6, and even the effective-K=64 benchmark's partial sums
are bounded by `1 + 64 * 6 = 385`. These stay within the exact-integer range
of half precision, below the documented FP8 WGMMA accumulation precision.
Failures report the mismatch count, maximum absolute error, and first mismatch;
non-finite mismatches report an infinite error.

## Benchmark

The benchmark is an end-to-end tile microbenchmark, not a large GEMM or a peak
throughput claim. Every launch includes global-to-shared staging, proxy and CTA
synchronization, WGMMA, and one checksum store per CTA.

Both variants compute the same effective `64x64x64` workload from the same
logical values. The FP8 K=32 tile contains two copies of the base K=16
pattern, with a common cyclic permutation of A and B in the second half.
This preserves the dot product while making missing SW32 half-swaps visible
to the numeric check. The FP8 kernel reuses that shared tile for two MMAs; the BF16 kernel
reuses its K=16 tile for four MMAs. Their decoded F32 references and device
checksums must agree before timing begins.

Timing uses 8192 CTAs, 10 alternating warmups, and 11 alternating samples of
100 CUDA-event-timed launches. The report uses the median average launch time
and counts `2 * 64 * 64 * 64` operations per CTA for both variants.
Raw sample times are printed, and every CTA's checksum is rechecked after timing.

Expected final marker:

```text
SUCCESS: FP8 WGMMA numeric check and BF16 comparison passed
```
