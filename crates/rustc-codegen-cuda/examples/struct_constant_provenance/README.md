# struct_constant_provenance

Positive regression for pointer provenance in a direct struct constant.

The example defines a constant struct containing a reference to a Rust static.
The MIR importer must resolve the pointer relocation through the allocation's
provenance table and materialize a pointer to the corresponding device global.

The pointer field must not be reconstructed from its placeholder bytes because
those bytes contain only the addend into the target allocation. The relocation
side table identifies the static allocation that provides the pointer's
provenance.

Run with:

```bash
cargo oxide run struct_constant_provenance
```

Expected output:

```text
PASS: struct constant pointer provenance preserved at runtime
```
