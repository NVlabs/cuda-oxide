# H100/H200 run for PR #1306

1. Use a Linux H100/H200 machine with CUDA 13+ (including Compute Sanitizer), driver 580+, and the
   [repo prerequisites](https://nvlabs.github.io/cuda-oxide/getting-started/installation.html).
   Keep other GPU jobs stopped. Allow 20–40 minutes for the first build and checks.

2. Clone the branch containing the runner:

   ```bash
   git clone -b feat/fp8-wgmma https://github.com/Vishalkulkarni45/cuda-oxide.git
   cd cuda-oxide
   ```

3. Select the CUDA installation and GPU, then run:

   ```bash
   export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0
   export CUDA_VISIBLE_DEVICES=0
   bash scripts/validate-fp8-wgmma.sh
   ```

4. Send back the printed `fp8-wgmma.*.tar.gz` file, **even if a check fails**.
   A successful archive contains `result.txt` starting with `PASS` and
   `status.txt` with `exit_code=0`. Do not use sanitizer timings as benchmarks.

The archive covers [Nihal's review](https://github.com/NVlabs/cuda-oxide/pull/1306#pullrequestreview-5304974674):

| Required evidence | Captured result |
|---|---|
| All-output correctness | All 4,096 FP8 and 4,096 BF16 values checked against decoded F32 references; maximum error and `SUCCESS` |
| Compute Sanitizer | memcheck, racecheck, initcheck, synccheck; WGMMA checks enabled for memcheck/synccheck |
| Comparable timings | Three runs with raw CUDA-event samples, median launch ms, and BF16/FP8 TFLOPS |
| Reproducibility | Commit, local diff, source, GPU/driver/toolkit versions, GPU state before/after, PTX and assembler report |

Both paths use 8,192 CTAs of 128 threads, the same 64×64×64 logical work,
and 4 KiB shared memory: two FP8 MMAs versus four BF16 MMAs. Each run uses
10 warmups and 11 alternating samples of 100 launches. This measures a tile
microbenchmark including staging and synchronization, not peak GEMM throughput.
