# TMA G2S with `setmaxnreg`

This compile-only regression contains a 384-thread kernel that combines the
existing `cp_async_bulk_tensor_2d_g2s` API with a 40/232-register warpgroup
split. On SM120, the local shared destination must lower to `shared::cta` so
ptxas can retain both `USETMAXREG` instructions without an address-compatibility
call. A second kernel covers the same lowering for all five TMA dimensions.

The lowering rule covers capabilities 120 and 121 independently of the target
suffix, including `sm_120f` and `sm_121a`. The smoketest builds both compiler
routes at `sm_120a`. Its code-shape check
requires CTA-local PTX on the LLVM path, inline CTA-local assembly in libNVVM
IR, and SASS containing `UTMALDG.2D` plus the 40/232 register reallocation with
no `CALL.ABS`.

```sh
bash scripts/smoketest.sh --only '^tma_setmaxnreg_repro$'
```
