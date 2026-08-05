# pitched_slice

Regression tests for the runtime row pitch carried inside
`DisjointSlice<T, Runtime2DIndex>`:

1. **Nonzero pitch readback**: the pitch bound on the host via
   `cuda_host::Pitched` must reach every device thread. An entry prologue
   that drops the third kernel parameter compiles and runs while giving
   every thread pitch 0; checking a nonzero value catches that.
2. **Two-pitch witness mixing**: witnesses minted from two differently
   pitched slices and selected under a thread-varying condition must still
   resolve against the addressed slice's own pitch, keeping every thread on
   its own cell.
3. **By-value pitched slice across a non-inlined call**: the internal call
   ABI must marshal all three fields (ptr, len, pitch) to match the
   three-parameter callee signature.

Run:

```bash
cargo oxide run pitched_slice
```
