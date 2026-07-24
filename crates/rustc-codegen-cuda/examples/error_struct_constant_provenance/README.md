# error_struct_constant_provenance

Negative regression: a struct constant holding a reference must fail device
compilation with a pointer-relocation diagnostic instead of silently decoding
the relocation's placeholder bytes as an address.

In rustc's constant representation, a pointer field's bytes hold only the
offset into the target allocation; the actual target lives in a provenance
side table the per-field byte decoders do not carry. Reading the bytes as a
value would fabricate a null or garbage pointer on the device with no error
anywhere. Until aggregate relocations are materialized as device globals, the
importer rejects the constant loudly.

Expected failure:

```text
Struct constant contains 1 pointer relocation(s); cuda-oxide cannot yet preserve struct pointer provenance
```

Run with:

```bash
cargo oxide build error_struct_constant_provenance   # must fail
```
