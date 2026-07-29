# `tuple_constant_provenance`

Positive smoke test: a tuple constant whose first field is a thin reference to
a device static. The importer materializes that field via `MirGlobalAllocOp`.

This covers aggregate **const** values only. Device-global *initializer*
relocations remain unsupported.

```bash
cargo oxide run tuple_constant_provenance
```
