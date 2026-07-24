# `error_tuple_array_provenance`

Negative regression for an array value constant whose tuple elements contain
real pointer relocations:

```rust
static FIRST: u32 = 11;
const POINTERS: [(&u32, bool); 1] = [(&FIRST, false)];
```

The allocation stores placeholder bytes plus a relocation identifying
`FIRST`. Treating those placeholder bytes as an exposed-provenance pointer
would silently lose the pointer's identity. Until aggregate constant lowering
can preserve each relocation, cuda-oxide must stop before decoding any array
element.

```bash
cargo oxide build error_tuple_array_provenance
```

Expected diagnostic:

```text
Array value constant contains 2 pointer relocation(s); cuda-oxide cannot yet preserve array pointer provenance
```

There are two relocations because the fixture has two tuple elements, each
containing one reference.
