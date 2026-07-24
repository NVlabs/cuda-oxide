# inline_ptx

Exercises `cuda_device::ptx_asm!` inside a cuda-oxide kernel. The kernel does
Rust arithmetic, uses a register-only PTX instruction, reads the lane-id
register (`%%laneid` in the macro string), emits a memory-clobbering
`membar.gl`, then uses the PTX results in Rust.

A second kernel exercises multi-output `ptx_asm!`: a single asm block with
two `=r` outputs computes both the sum and the product of two
thread-dependent values, written to separate buffers the host verifies
element-wise (the asymmetric results catch swapped output bindings).

Run with:

```bash
cargo oxide run inline_ptx
```
