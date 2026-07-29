# `struct_constant_provenance`

Positive smoke test: a struct constant whose field is a thin reference to a
device static. The importer materializes that field via `MirGlobalAllocOp`
(addend from the relocation's stored bytes) and keeps the sibling scalar field.

This covers aggregate **const** values only. Device-global *initializer*
relocations (pointers stored inside a `static`'s own initializer) remain
unsupported.

```bash
cargo oxide run struct_constant_provenance
```
