# `error_enum_constant_provenance`

Negative test for an enum constant containing a real pointer:

```rust
static TARGET: u64 = 7;
const POINTER_ENUM: Option<&'static u64> = Some(&TARGET);
```

Rust represents the pointer as bytes plus a relocation that identifies
`TARGET`. Reading only the bytes loses that identity. The old importer went
further and accidentally followed the relocation, then used `TARGET`'s value
as if it were the pointer address.

Until enum constant construction can preserve relocations, cuda-oxide must
stop instead of emitting a wrong pointer.

```bash
cargo oxide build error_enum_constant_provenance
```

Expected diagnostic:

```text
Enum constant contains 1 pointer relocation(s); cuda-oxide cannot yet preserve
enum pointer provenance
```
