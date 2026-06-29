# error_static_initializer_provenance

Negative test for a device static whose initializer contains a pointer:

```rust
static TARGET: u32 = 0x1234_5678;
static REFERENCE: &u32 = &TARGET;
```

References stored in statics implicitly have the `'static` lifetime. Rust
stores this pointer as bytes plus a relocation (called *provenance*). The
current LLVM exporter can preserve literal bytes, but it cannot emit that
relocation yet. Treating the placeholder bytes as the value would produce a
null pointer and silently miscompile the kernel, so cuda-oxide must stop with a
clear error.

Run:

```bash
cargo oxide build error_static_initializer_provenance
```

The build must fail with a message similar to:

```text
device static REFERENCE contains 1 pointer relocation(s); cuda-oxide cannot yet
emit pointer provenance in device global initializers
```
