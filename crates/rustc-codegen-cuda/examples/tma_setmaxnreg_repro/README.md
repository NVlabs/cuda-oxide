# CTA-local TMA + `setmaxnreg` regression

This compile-only example contains two 384-thread kernels. Both split one
producer warpgroup from two consumer warpgroups and request 40/232 registers.
The second kernel adds only one CTA-local TMA G2S operation to a static shared
tile. It guards the lowering that avoids ptxas's extern compatibility path on
SM120 while leaving the existing cluster-shared and multicast operations
unchanged.

Build the PTX and assemble it for SM120a:

```sh
cargo oxide build tma_setmaxnreg_repro --arch sm_120a
ptxas --gpu-name=sm_120a --verbose \
  crates/rustc-codegen-cuda/examples/tma_setmaxnreg_repro/tma_setmaxnreg_repro.ptx \
  --output-file=/tmp/tma_setmaxnreg_repro.cubin
nvdisasm --print-code /tmp/tma_setmaxnreg_repro.cubin \
  | rg 'text\.setmaxnreg|USETMAXREG|CALL|UTMALDG'
```

CUDA 13.3 ptxas must not report C7506. Both kernels must retain
`USETMAXREG.DEALLOC` and `USETMAXREG.TRY_ALLOC`; the TMA kernel must contain
`UTMALDG.2D` and no `CALL.ABS`.
