# inline_ptx

Exercises `cuda_device::ptx_asm!` inside a cuda-oxide kernel. The kernel does
Rust arithmetic, uses a register-only PTX instruction, reads the lane-id
register (`%%laneid` in the macro string), emits a memory-clobbering
`membar.gl`, then uses the PTX results in Rust.

Run with:

```bash
cargo oxide run inline_ptx
```
