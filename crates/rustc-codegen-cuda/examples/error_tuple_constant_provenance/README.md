# `error_tuple_constant_provenance`

Negative regression for a direct tuple constant containing a pointer
relocation:

```rust
static FIRST: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
const DIRECT: (&[u8; 16], bool) = (&FIRST, true);
```

The allocation stores placeholder bytes plus a relocation identifying
`FIRST`. Treating those bytes as an exposed-provenance integer would fabricate
a pointer unrelated to `FIRST`, so cuda-oxide must reject the constant before
decoding its tuple fields.

```bash
cargo oxide build error_tuple_constant_provenance
```

Expected diagnostic:

```text
Tuple constant contains 1 pointer relocation(s); cuda-oxide cannot yet preserve tuple pointer provenance
```
