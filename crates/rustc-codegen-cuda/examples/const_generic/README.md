# Const-generic kernel entries

This example proves that const values participate in a kernel entry's compiled
identity.

```text
write_value::<4> -> write_value_TID_<hash A>
write_value::<8> -> write_value_TID_<hash B>
```

Both specializations have the same runtime parameter types. The PTX must still
contain two distinct `.entry` symbols, and each body must use its own folded
constant. The kernel also calls a const-generic `#[device]` helper, covering the
same forwarding rule for device functions. The raw-pointer kernels and device
helper are `unsafe`; the example also verifies that macro expansion preserves
that caller-visible contract.

A second kernel deliberately does not read its const parameter. Its `<4>` and
`<8>` specializations must still remain two exact, host-addressable PTX entries
even though their optimized instruction bodies are identical.

```bash
cargo oxide pipeline const_generic
cargo oxide run const_generic
```

The host lookup names and generated PTX can be checked without initializing
CUDA:

```bash
cargo oxide build const_generic
./crates/rustc-codegen-cuda/examples/const_generic/target/release/const_generic \
  --verify-ptx
```

That check proves that the host names resolve to both PTX entries, both entries
retain `#[launch_bounds(64)]`, and each body contains its own folded constant.
